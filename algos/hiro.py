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
        s, g, a, r, ns, ng, d = map(np.asarray, zip(*batch))
        return (
            torch.as_tensor(s, dtype=torch.float32, device=self.device),
            torch.as_tensor(g, dtype=torch.float32, device=self.device),
            torch.as_tensor(a, dtype=torch.float32, device=self.device),
            torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(ns, dtype=torch.float32, device=self.device),
            torch.as_tensor(ng, dtype=torch.float32, device=self.device),
            torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
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
        s, g, r, ns, d, s_seq, a_seq = map(np.asarray, zip(*batch))
        return (
            torch.as_tensor(s, dtype=torch.float32, device=self.device),
            torch.as_tensor(g, dtype=torch.float32, device=self.device),
            torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(ns, dtype=torch.float32, device=self.device),
            torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(s_seq, dtype=torch.float32, device=self.device),
            torch.as_tensor(a_seq, dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return len(self.storage)

# --- TD3 Agent (Base for both Worker and Manager) ---
class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, hyperparams={}):
        super(TD3, self).__init__()

        # General TD3 / RL
        self.LR_ACTOR = hyperparams.get('LR_ACTOR', 1e-4)
        self.LR_CRITIC = hyperparams.get('LR_CRITIC', 5e-4)
        self.TAU = hyperparams.get('TAU', 0.0095)
        self.BATCH_SIZE = hyperparams.get('BATCH_SIZE', 64)
        self.POLICY_NOISE = hyperparams.get('POLICY_NOISE', 0.2)
        self.NOISE_CLIP = hyperparams.get('NOISE_CLIP', 0.5)
        self.POLICY_FREQ = hyperparams.get('UPDATE_DELAY', 2)

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

        self.critic_target.eval()  # Set target critic to evaluation mode
        self.actor_target.eval()   # Set target actor to evaluation mode
        for param in self.critic_target.parameters():
            param.requires_grad = False  # Freeze target critic parameters
        for param in self.actor_target.parameters():
            param.requires_grad = False  # Freeze target actor parameters

        self.total_it = 0

    def select_action(self, state):
        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, gamma, is_manager=False, max_action=None):
        if len(replay_buffer) < self.BATCH_SIZE:
            return

        if max_action is not None:
            self.MAX_ACTION = max_action

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
    def __init__(self, state_dim, action_dim, percept=False, device="cpu", hyperparams={}):
        super(HIROAgent, self).__init__()

        # HIRO specific
        self.C = hyperparams.get("SUBGOAL_HORIZON", 10)                      # Manager action frequency
        self.GOAL_BOUND = hyperparams.get("GOAL_BOUND", 2.0)   # Max magnitude for goals (state differences)
        self.EXPL_NOISE_MANAGER = hyperparams.get('EXPL_NOISE_MANAGER', 1.0)
        self.EXPL_NOISE_WORKER = hyperparams.get('EXPL_NOISE_WORKER', 0.1)
        self.BUFFER_SIZE = hyperparams.get('BUFFER_SIZE', int(1e5))
        self.GAMMA_MANAGER = hyperparams.get('GAMMA_M', 0.99)
        self.GAMMA_WORKER = hyperparams.get('GAMMA_W', 0.95)
        # self.MAX_EPISODES = hyperparams.get('MAX_EPISODES', 2000)
        self.MAX_ENV_STEPS = hyperparams.get('MAX_ENV_STEPS', 1e6)

        # TD3 Hyperparameters
        self.LR_ACTOR = hyperparams.get('LR_ACTOR', 1e-4)
        self.LR_CRITIC = hyperparams.get('LR_CRITIC', 5e-4)
        self.TAU = hyperparams.get('TAU', 0.0095)
        self.BATCH_SIZE = hyperparams.get('BATCH_SIZE', 64)
        self.POLICY_NOISE = hyperparams.get('POLICY_NOISE', 0.2)
        self.NOISE_CLIP = hyperparams.get('NOISE_CLIP', 0.5)
        self.POLICY_FREQ = hyperparams.get('UPDATE_DELAY', 2)

        self.MAX_ACTION = hyperparams.get('MAX_ACTION', 3.0)
        self.SEED = hyperparams.get('SEED', 31)

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

        self.manager.total_it += 1

        # 1. Sample from manager buffer
        states, goals, rewards, next_states, dones, states_seq, actions_seq = self.manager_buffer.sample(self.BATCH_SIZE)
        
        # 2. Generate candidate goals
        # 10 candidates: original goal, diff (s_{t+c} - s_t), and 8 random goals based on diff
        batch_size = states.shape[0]
        diff = next_states - states
        
        num_candidates = 10
        candidates = torch.empty((batch_size, num_candidates, self.goal_dim), device=self.device)
        candidates[:, 0, :] = goals
        candidates[:, 1, :] = diff
        
        # 8 random candidates (Gaussian around diff)
        for i in range(2, num_candidates):
            noise = torch.randn_like(diff) * (self.GOAL_BOUND / 2.0)
            candidates[:, i, :] = torch.clamp(diff + noise, -self.GOAL_BOUND, self.GOAL_BOUND)

        # 3. Evaluate candidates by calculating log probability (or negative MSE) of taken actions
        log_probs = torch.zeros((batch_size, num_candidates), device=self.device)
        curr_goals = candidates.clone()

        with torch.no_grad():
            for t in range(self.C):
                s_t = states_seq[:, t]
                a_t = actions_seq[:, t]

                # Padded dummy values mark early episode termination.
                valid = ~((s_t == 0).all(dim=1) & (a_t == 0).all(dim=1))
                if not valid.any():
                    break

                s_t_expanded = s_t.unsqueeze(1).expand(-1, num_candidates, -1)
                worker_input = torch.cat([s_t_expanded, curr_goals], dim=2)
                pred_action = self.worker.actor(
                    worker_input.reshape(batch_size * num_candidates, -1)
                ).reshape(batch_size, num_candidates, self.action_dim)

                action_error = (pred_action - a_t.unsqueeze(1)).pow(2).mean(dim=2)
                log_probs -= action_error * valid.unsqueeze(1)

                if t < self.C - 1:
                    s_next = states_seq[:, t + 1]
                    curr_goals = s_t_expanded + curr_goals - s_next.unsqueeze(1)

        best_idxs = log_probs.argmax(dim=1)
        best_goals = candidates[torch.arange(batch_size, device=self.device), best_idxs]

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

    def intrinsic_reward(self, state, goal, next_state):
        # Reward worker for getting next_state close to state + goal
        # i.e., minimizing distance between (s_t + g_t) and s_{t+1}
        return -np.linalg.norm(state + goal - next_state)

    def ddpg_update(self):
        # HIRO uses TD3 for both manager and worker, but if you want to implement DDPG updates, you can do so here.

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

                if self.wandb_upload:
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

    def train_agent(self, env, store_episode_rewards=False):

        env_name = env.spec.id
        model_path = None

        if self.save_model:
            model_path = f"Weights/{env_name}_HIRO_weights/HIRO_{env_name}_C_{self.C}_gb_{self.GOAL_BOUND}_xx"
            os.makedirs(model_path, exist_ok=True)

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
                    "seed": self.SEED,
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "actor_lr": self.LR_ACTOR,
                    "critic_lr": self.LR_CRITIC,
                    "tau": self.TAU,
                    "batch_size": self.BATCH_SIZE,
                    "buffer_size": self.BUFFER_SIZE,
                    "policy_noise": self.POLICY_NOISE,
                    "noise_clip": self.NOISE_CLIP,
                    "policy_freq": self.POLICY_FREQ,
                    "max_env_steps": self.MAX_ENV_STEPS,
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
            run.name = f"{env_name}_{self.MAX_ENV_STEPS}_{self.LR_ACTOR:.4f}_{self.LR_CRITIC:.4f}"

        load_model = False
        if load_model:
            load_model_path = "Weights/Ant-v5-GC_TD3/alr_5.346635183767254e-05_clr_0.0002923700418487162_xxx/model.pt"
            print(f"Loading model from {load_model_path}")
            # print(self.worker.state_dict().keys())
            self.worker.load_state_dict(torch.load(f"{load_model_path}", map_location=self.device))

        reward_history = []
        env_steps = 0
        self.MAX_ACTION = env.action_space.high[0] if isinstance(env.action_space, gym.spaces.Box) else 1.0
        print(f"Training HIRO on {env_name} with max action {self.MAX_ACTION}")

        with tqdm(total=self.MAX_ENV_STEPS, desc="Training Progress") as pbar:
            while env_steps < self.MAX_ENV_STEPS:

                state, _ = env.reset(seed=self.SEED+env_steps)  # Seed environment for reproducibility
                episode_reward = 0
                done = False
                episode_length = 0
                
                while not done:
                    # High-level goal selection every C steps
                    if episode_length % self.C == 0:
                        goal = self.manager.select_action(state)
                        # Add exploration noise to goal
                        goal += np.random.normal(0, self.EXPL_NOISE_MANAGER, size=self.goal_dim)
                        goal = np.clip(goal, -self.GOAL_BOUND, self.GOAL_BOUND)
                        original_goal = goal.copy()
                        
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
                    manager_reward += reward * (self.GAMMA_MANAGER ** (episode_length % self.C))
                    env_steps += 1
                        
                    # Worker Intrinsic Reward & Goal Transition
                    worker_reward = self.intrinsic_reward(state, goal, next_state)
                    next_goal = state + goal- next_state

                    if self.wandb_upload:
                        wandb.log({
                            "env_steps": env_steps,
                            "worker_reward": worker_reward,
                            "manager_reward": manager_reward,
                        })
                    
                    # Store Worker Transition
                    self.worker_buffer.add(state, goal, action, worker_reward, next_state, next_goal, float(done))
                    
                    # Track sequence for Manager
                    states_seq[seq_step] = state
                    actions_seq[seq_step] = action
                    seq_step += 1

                    # Train Worker
                    self.worker.train(self.worker_buffer, self.GAMMA_WORKER, is_manager=False, max_action=self.MAX_ACTION)

                    state = next_state
                    goal = next_goal
                    episode_length += 1

                    # Manager Transition Storage and Training (at end of C steps or end of episode)
                    if episode_length % self.C == 0 or done:
                        self.manager_buffer.add(
                            manager_state_start, original_goal, manager_reward, state, float(done),
                            states_seq, actions_seq
                        )
                        self.off_policy_correction()

                    if self.save_model and env_steps % 100000 == 0:
                        torch.save(self.state_dict(), f"{model_path}/model.pt")
                        # print(f"Model saved to {model_path}")

                if self.wandb_upload:
                    wandb.log({
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                    })

                if store_episode_rewards:
                    reward_history.append(episode_reward)
                    if len(reward_history) > 10:
                        reward_history.pop(0)
                    avg_reward = np.mean(reward_history)
                    if self.wandb_upload:
                        wandb.log({
                            "mean_episode_return": avg_reward
                        })

                pbar.update(episode_length)

        env.close()

    def test_agent(self, env, n_episodes=1, record_stats=False, model_path=None):

        env_name = env.spec.id

        if model_path is None:
            model_path = f"Weights/{env_name}_HIRO_weights/HIRO_{env_name}_C_{self.C}_gb_{self.GOAL_BOUND}"

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=self.device))
        else:
            print("Weights not found. Please train the agent first.")
            return

        if record_stats:
            episode_rewards = []
            episode_lengths = []

        for episode in range(n_episodes):

            if self.wandb_upload:
                run = wandb.init(
                    entity="arkapatra31-ust",
                    project="HIRO-Test",
                    config={
                        "env": env_name,
                        "architecture": "HIRO",
                        "model_path": model_path,
                        "seed": self.SEED,
                        "action_dim": self.action_dim,
                        "state_dim": self.state_dim
                    }
                )
                run.name = f"{env_name}_test_{n_episodes}_episodes"

            obs, info = env.reset(seed=self.SEED + episode)  # Seed environment for reproducibility
            done = False
            episode_reward = 0
            episode_length = 0
            renders = []  # Store frames for video logging
            
            while not done:
                if episode_length % self.C == 0:
                    goal = self.manager.select_action(obs)
                    goal = np.clip(goal, -self.GOAL_BOUND, self.GOAL_BOUND)
                    manager_state_start = obs.copy()
                    manager_reward = 0

                worker_input = np.concatenate([obs, goal])
                action = self.worker.select_action(worker_input)
                action = np.clip(action, -self.MAX_ACTION, self.MAX_ACTION)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                manager_reward += reward
                episode_length += 1

                if self.wandb_upload:
                    # Capture the render for logging
                    frame = env.render()
                    if frame is not None:
                        # W&B handles videos best when frames are uint8 CHW tensors.
                        if frame.shape[-1] == 4:
                            frame = frame[:, :, :3]
                        frame = np.asarray(frame, dtype=np.uint8)
                        renders.append(frame)

                worker_reward = self.intrinsic_reward(obs, goal, next_obs)
                next_goal = obs + goal - next_obs

                obs = next_obs
                goal = next_goal

                if self.wandb_upload:
                    wandb.log({
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "manager_reward": manager_reward,
                        "worker_reward": worker_reward,
                    })

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

            if record_stats:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Length: {episode_length}")
        env.close()
        if record_stats:
            return episode_rewards, episode_lengths
