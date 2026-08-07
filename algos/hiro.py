import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import cv2
import os
import wandb

SEED = 31
np.random.seed(SEED)
# 1. Set seeds for PyTorch (CPU and GPU)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups

# 2. Ensure deterministic behavior for CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Neural Networks (TD3 Base) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

# --- Replay Buffers ---
class WorkerBuffer:
    def __init__(self, max_size, device):
        self.storage = []
        self.ptr = 0
        self.max_size = max_size
        self.device = device

    def add(self, state, goal, action, reward, next_state, next_goal, done):
        transition = (state, goal, action, reward, next_state, next_goal, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        s, g, a, r, ns, ng, d = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(s).to(self.device),
            torch.FloatTensor(g).to(self.device),
            torch.FloatTensor(a).to(self.device),
            torch.FloatTensor(r).unsqueeze(1).to(self.device),
            torch.FloatTensor(ns).to(self.device),
            torch.FloatTensor(ng).to(self.device),
            torch.FloatTensor(d).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.storage)

class ManagerBuffer:
    def __init__(self, max_size, device):
        self.storage = []
        self.ptr = 0
        self.max_size = max_size
        self.device = device

    def add(self, state, goal, reward, next_state, done, states_seq, actions_seq):
        # states_seq and actions_seq are needed for off-policy correction
        transition = (state, goal, reward, next_state, done, states_seq, actions_seq)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        s, g, r, ns, d, s_seq, a_seq = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(s).to(self.device),
            torch.FloatTensor(g).to(self.device),
            torch.FloatTensor(r).unsqueeze(1).to(self.device),
            torch.FloatTensor(ns).to(self.device),
            torch.FloatTensor(d).unsqueeze(1).to(self.device),
            torch.FloatTensor(s_seq).to(self.device),
            torch.FloatTensor(a_seq).to(self.device)
        )

    def __len__(self):
        return len(self.storage)

# --- TD3 Agent (Base for both Worker and Manager) ---
class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, hyperparams={}):
        super(TD3, self).__init__()

        # General TD3 / RL
        self.LR_ACTOR = hyperparams.get('LR_ACTOR', 1e-4)
        self.LR_CRITIC = hyperparams.get('LR_CRITIC', 3e-4)
        self.GAMMA_MANAGER = hyperparams.get('GAMMA_MANAGER', 0.99)
        self.GAMMA_WORKER = hyperparams.get('GAMMA_WORKER', 0.95)
        self.TAU = hyperparams.get('TAU', 0.005)
        self.BATCH_SIZE = hyperparams.get('BATCH_SIZE', 64)
        self.BUFFER_SIZE = hyperparams.get('BUFFER_SIZE', int(1e5))
        self.EXPL_NOISE_MANAGER = hyperparams.get('EXPL_NOISE_MANAGER', 1.0)
        self.EXPL_NOISE_WORKER = hyperparams.get('EXPL_NOISE_WORKER', 0.1)
        self.POLICY_NOISE = hyperparams.get('POLICY_NOISE', 0.2)
        self.NOISE_CLIP = hyperparams.get('NOISE_CLIP', 0.5)
        self.POLICY_FREQ = hyperparams.get('POLICY_FREQ', 4)
        self.MAX_EPISODES = hyperparams.get('MAX_EPISODES', 2000)
        self.MAX_TIMESTEPS = hyperparams.get('MAX_TIMESTEPS', 1000)

        self.device = device
        self.MAX_ACTION = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, gamma, is_manager=False):
        if len(replay_buffer) < self.BATCH_SIZE:
            return

        self.total_it += 1

        if is_manager:
            # Manager uses standard TD3 update
            state, action, reward, next_state, done, _, _ = replay_buffer.sample(self.BATCH_SIZE)
            next_input = next_state
            curr_input = state
        else:
            # Worker concatenates state and goal
            state, goal, action, reward, next_state, next_goal, done = replay_buffer.sample(self.BATCH_SIZE)
            next_input = torch.cat([next_state, next_goal], dim=1)
            curr_input = torch.cat([state, goal], dim=1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.POLICY_NOISE).clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
            next_action = (self.actor_target(next_input) + noise).clamp(-self.MAX_ACTION, self.MAX_ACTION)
            target_Q1, target_Q2 = self.critic_target(next_input, next_action)
            target_Q = reward + (1 - done) * gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(curr_input, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.POLICY_FREQ == 0:
            actor_loss = -self.critic.Q1(curr_input, self.actor(curr_input)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

# --- HIRO Algorithm Wrapper ---
class HIROAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hyperparams={}, device="cpu"):
        super(HIROAgent, self).__init__()

        # HIRO specific
        self.C = hyperparams.get("C", 50)                      # Manager action frequency
        self.GOAL_BOUND = hyperparams.get("GOAL_BOUND", 2.0)   # Max magnitude for goals (state differences)

        # General TD3 / RL
        self.LR_ACTOR = hyperparams.get('LR_ACTOR', 1e-4)
        self.LR_CRITIC = hyperparams.get('LR_CRITIC', 3e-4)
        self.GAMMA_MANAGER = hyperparams.get('GAMMA_M', 0.99)
        self.GAMMA_WORKER = hyperparams.get('GAMMA_W', 0.95)
        self.TAU = hyperparams.get('TAU', 0.005)
        self.BATCH_SIZE = hyperparams.get('BATCH_SIZE', 64)
        self.BUFFER_SIZE = hyperparams.get('BUFFER_SIZE', int(1e5))
        self.EXPL_NOISE_MANAGER = hyperparams.get('EXPL_NOISE_MANAGER', 1.0)
        self.EXPL_NOISE_WORKER = hyperparams.get('EXPL_NOISE_WORKER', 0.1)
        self.POLICY_NOISE = hyperparams.get('POLICY_NOISE', 0.2)
        self.NOISE_CLIP = hyperparams.get('NOISE_CLIP', 0.5)
        self.POLICY_FREQ = hyperparams.get('POLICY_FREQ', 4)
        self.MAX_EPISODES = hyperparams.get('MAX_EPISODES', 2000)
        self.MAX_TIMESTEPS = hyperparams.get('MAX_TIMESTEPS', 1000)
        self.MAX_ACTION = hyperparams.get('MAX_ACTION', 1.0)

        self.wandb_upload = hyperparams.get('WANDB_UPLOAD', False)
        self.save_model = hyperparams.get('SAVE_MODEL', False)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = hyperparams.get('GOAL_DIM', state_dim)  # Goal is a desired change in state
        self.device = device
        
        # Low-level: sees (state + goal), outputs primitive action
        self.worker = TD3(state_dim + self.goal_dim, action_dim, self.MAX_ACTION, self.device, hyperparams)
        # High-level: sees state, outputs goal
        self.manager = TD3(state_dim, self.goal_dim, self.GOAL_BOUND, self.device, hyperparams)
        
        self.worker_buffer = WorkerBuffer(self.BUFFER_SIZE, self.device)
        self.manager_buffer = ManagerBuffer(self.BUFFER_SIZE, self.device)

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")

    def off_policy_correction(self):
        if len(self.manager_buffer) < self.BATCH_SIZE:
            return

        # 1. Sample from manager buffer
        states, goals, rewards, next_states, dones, states_seq, actions_seq = self.manager_buffer.sample(self.BATCH_SIZE)
        
        # 2. Generate candidate goals
        # 10 candidates: original goal, diff (s_{t+c} - s_t), and 8 random goals based on diff
        batch_size = states.shape[0]
        diff = next_states - states
        
        candidates = torch.zeros((batch_size, 10, self.goal_dim)).to(self.device)
        candidates[:, 0, :] = goals
        candidates[:, 1, :] = diff
        
        # 8 random candidates (Gaussian around diff)
        for i in range(2, 10):
            noise = torch.randn_like(diff) * (self.GOAL_BOUND / 2.0)
            candidates[:, i, :] = torch.clamp(diff + noise, -self.GOAL_BOUND, self.GOAL_BOUND)

        # 3. Evaluate candidates by calculating log probability (or negative MSE) of taken actions
        best_goals = torch.zeros((batch_size, self.goal_dim)).to(self.device)
        
        for b in range(batch_size):
            best_idx = 0
            max_log_prob = -float('inf')
            
            for c_idx in range(10):
                candidate_goal = candidates[b, c_idx]
                log_prob = 0.0
                curr_goal = candidate_goal.clone()
                
                # Simulate goal transition over the sequences
                for t in range(self.C):
                    s_t = states_seq[b, t]
                    a_t = actions_seq[b, t]
                    
                    if torch.all(s_t == 0) and torch.all(a_t == 0):
                        break # Padded dummy values if episode ended early
                        
                    worker_input = torch.cat([s_t, curr_goal]).unsqueeze(0)
                    # Negative MSE between deterministic action and actual action (surrogate for log prob)
                    with torch.no_grad():
                        pred_action = self.worker.actor(worker_input).squeeze(0)
                    log_prob += -F.mse_loss(pred_action, a_t).item()
                    
                    if t < self.C - 1:
                        s_next = states_seq[b, t+1]
                        curr_goal = s_t + curr_goal - s_next
                
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_idx = c_idx
            
            best_goals[b] = candidates[b, best_idx]

        # 4. Train Manager with the corrected goals
        # Replace the sampled buffer goals with corrected ones for training
        with torch.no_grad():
            next_actions = self.manager.actor_target(next_states)
            noise = (torch.randn_like(next_actions) * self.POLICY_NOISE).clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
            next_actions = (next_actions + noise).clamp(-self.GOAL_BOUND, self.GOAL_BOUND)
            target_Q1, target_Q2 = self.manager.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.GAMMA_MANAGER * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.manager.critic(states, best_goals) # Use best_goals here
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.manager.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.manager.critic_optimizer.step()

        if self.manager.total_it % self.POLICY_FREQ == 0:
            actor_loss = -self.manager.critic.Q1(states, self.manager.actor(states)).mean()
            self.manager.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.manager.actor_optimizer.step()

            for param, target_param in zip(self.manager.critic.parameters(), self.manager.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            for param, target_param in zip(self.manager.actor.parameters(), self.manager.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def train_worker(self):
        self.worker.train(self.worker_buffer, self.GAMMA_WORKER, is_manager=False)

    def intrinsic_reward(self, state, goal, next_state):
        # Reward worker for getting next_state close to state + goal
        # i.e., minimizing distance between (s_t + g_t) and s_{t+1}
        return -np.linalg.norm(state + goal - next_state)

    def train_agent(self, env):

        env_name = env.spec.id

        model_path = f"Weights/{env_name}_HIRO_weights/HIRO_{env_name}_{self.MAX_EPISODES}.pt"
        os.makedirs(os.path.dirname(f"Weights/{env_name}_HIRO_weights"), exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="HIRO-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "architecture": "HIRO",
                    "actor_lr": self.LR_ACTOR,
                    "critic_lr": self.LR_CRITIC,
                    "tau": self.TAU,
                    "batch_size": self.BATCH_SIZE,
                    "buffer_size": self.BUFFER_SIZE,
                    "policy_noise": self.POLICY_NOISE,
                    "noise_clip": self.NOISE_CLIP,
                    "policy_freq": self.POLICY_FREQ,
                    "max_episodes": self.MAX_EPISODES,
                    "max_timesteps": self.MAX_TIMESTEPS,
                    "expl_noise_manager": self.EXPL_NOISE_MANAGER,
                    "expl_noise_worker": self.EXPL_NOISE_WORKER,
                    "goal_bound": self.GOAL_BOUND,
                    "subgoal_horizon": self.C,
                    "manager_gamma": self.GAMMA_MANAGER,
                    "worker_gamma": self.GAMMA_WORKER,
                    "model_path": model_path,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params
                }
            )
            run.name = f"{env_name}_{self.MAX_EPISODES}_{self.LR_ACTOR:.4f}_{self.LR_CRITIC:.4f}"

        reward_history = []
        env_steps = 0
        
        for episode in tqdm(range(1, self.MAX_EPISODES + 1)):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            t = 0
            
            while not done and t < self.MAX_TIMESTEPS:
                # High-level goal selection every C steps
                if t % self.C == 0:
                    goal = self.manager.select_action(state)
                    # Add exploration noise to goal
                    goal += np.random.normal(0, self.EXPL_NOISE_MANAGER, size=self.state_dim)
                    goal = np.clip(goal, -self.GOAL_BOUND, self.GOAL_BOUND)
                    
                    manager_state_start = state.copy()
                    manager_reward = 0
                    
                    # Storage for off-policy correction
                    states_seq = np.zeros((self.C, self.state_dim))
                    actions_seq = np.zeros((self.C, self.action_dim))
                    seq_step = 0

                # Low-level action selection
                worker_input = np.concatenate([state, goal])
                action = self.worker.select_action(worker_input)
                action += np.random.normal(0, self.EXPL_NOISE_WORKER, size=self.action_dim)
                action = np.clip(action, -self.MAX_ACTION, self.MAX_ACTION)

                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                manager_reward += reward
                env_steps += 1
                if self.wandb_upload:
                    wandb.log({
                        "env_steps": env_steps
                    })
                    
                # Worker Intrinsic Reward & Goal Transition
                worker_reward = self.intrinsic_reward(state, goal, next_state)
                next_goal = state + goal - next_state
                
                # Store Worker Transition
                self.worker_buffer.add(state, goal, action, worker_reward, next_state, next_goal, float(done))
                
                # Track sequence for Manager
                states_seq[seq_step] = state
                actions_seq[seq_step] = action
                seq_step += 1

                # Train Worker
                self.train_worker()

                state = next_state
                t += 1

                # Manager Transition Storage and Training (at end of C steps or end of episode)
                if t % self.C == 0 or done:
                    self.manager_buffer.add(
                        manager_state_start, goal, manager_reward, state, float(done),
                        states_seq, actions_seq
                    )
                    self.off_policy_correction()

            if self.wandb_upload:
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_length": t,
                })

        if self.save_model:
            torch.save(self.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        env.close()

    def test_agent(self, env, n_episodes=1):

        env_name = env.spec.id

        model_path = f"Weights/{env_name}_HIRO_weights/HIRO_{env_name}_{self.MAX_EPISODES}.pt"

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Weights not found. Please train the agent first.")
            return

        for episode in range(n_episodes):

            obs, info = env.reset()
            self.done = False
            self.episode_reward = 0
            self.episode_length = 0
            renders = []  # Store frames for video logging
            
            while not self.done:
                desired_gstate = info.get('desired_goal', env.observation_space.sample())  # Use the desired goal from the info dictionary if available, otherwise sample a random goal
                achieved_gstate = info.get('achieved_goal', obs)  # Use the achieved goal from the info dictionary if available, otherwise use the current observation
                goal = desired_gstate - achieved_gstate  # Compute the goal as the difference between the desired goal and the current observation

                self.highest_subgoal_achieved_level = -1  # Reset the highest subgoal achieved level for this episode
                next_obs = self.test_level(env, self.N_LEVELS-1, obs, info, [goal], subgoal_test=True)  # Start testing from the highest level
                obs = next_obs

            if self.wandb_upload:
                 if renders:
                     renders = [
                         cv2.resize(render, (500, 500), interpolation=cv2.INTER_AREA)
                         for render in renders
                     ]
                     video = np.stack(renders, axis=0)
                     video = np.transpose(video, (0, 3, 1, 2))
                     wandb.log({"showcase_video": wandb.Video(video, fps=8, format="mp4")})
                 wandb.finish()
            
            print(f"Episode {episode + 1}: Total Reward: {self.episode_reward}, Length: {self.episode_length}")
        env.close()

    def ddpg_update(self):

        for k in range(self.N_LEVELS):

            if len(self.replay_buffers[k]) > self.BATCH_SIZE:    
    
                # Sample a batch of transitions
                states, actions, rewards, next_states, dones, goals, gammas = self.replay_buffer[k].sample(self.BATCH_SIZE)
                states = torch.FloatTensor(np.array(states)).to(self.device)
                actions = torch.FloatTensor(np.array(actions)).to(self.device)
                rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
                next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
                dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
                goals = torch.FloatTensor(np.array(goals)).to(self.device)
                gammas = torch.FloatTensor(np.array(gammas)).unsqueeze(1).to(self.device)
                
                # ------------------- Optimize Critic ------------------- #
                # Get predicted next-state actions and Q values from target models
                with torch.no_grad():
                    next_actions = self.actor[k](next_states, goals)
                    target_Q = self.critic[k](next_states, goals, next_actions)
                    # Compute the target Q value
                    target_Q = rewards + (1 - dones) * gammas * target_Q

                # Get current Q estimate
                current_Q = self.critic[k](states, goals, actions)

                # Compute critic loss (MSE)
                critic_loss = F.mse_loss(current_Q, target_Q)

                # Optimize the critic
                self.critic_optimizer[k].zero_grad()
                critic_loss.backward()
                self.critic_optimizer[k].step()

                # ------------------- Optimize Actor ------------------- #
                # Compute actor loss (Negative of the critic's output for the actor's actions)
                actor_loss = -self.critic[k](states, goals, self.actor[k](states, goals)).mean()

                # Optimize the actor
                self.actor_optimizer[k].zero_grad()
                actor_loss.backward()
                self.actor_optimizer[k].step()

                wandb_upload = True  # Whether to upload logs to Weights & Biases

                if wandb_upload:
                    wandb.log({
                        f"critic_loss_{k}": critic_loss.item(),
                        f"actor_loss_{k}": actor_loss.item(),
                    })

                # ------------------- Soft Updates ------------------- #
                # Update target networks
                # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                #     target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                #     target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

