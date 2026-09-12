import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections

from tqdm import tqdm
import wandb
import os
import cv2


# --- Neural Networks ---
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
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
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

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size, device):
        self.storage = []
        self.ptr = 0
        self.max_size = max_size
        self.device = device
        self._shapes = None

    def add(self, state, action, reward, next_state, done, gamma):

        # state = np.asarray(state, dtype=np.float32)
        # action = np.asarray(action, dtype=np.float32)
        # next_state = np.asarray(next_state, dtype=np.float32)
        # reward = float(reward)
        # done = float(done)
        # gamma = float(gamma)

        shapes = (state.shape, action.shape, next_state.shape)
        if self._shapes is None:
            self._shapes = shapes
        elif shapes != self._shapes:
            raise ValueError(
                "ReplayBuffer transition shape mismatch: "
                f"expected state/action/next_state shapes {self._shapes}, got {shapes}"
            )

        transition = (state, action, reward, next_state, done, gamma)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        s, a, r, ns, d, g = map(np.asarray, zip(*batch))
        return (
            torch.as_tensor(s, dtype=torch.float32, device=self.device),
            torch.as_tensor(a, dtype=torch.float32, device=self.device),
            torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(ns, dtype=torch.float32, device=self.device),
            torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(g, dtype=torch.float32, device=self.device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.storage)

# --- Base TD3 Agent ---
class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, hyperparams={}):
        super(TD3, self).__init__()
        # --- Hyperparameters ---
        self.LR_ACTOR = hyperparams.get("LR_ACTOR", 1e-4)
        self.LR_CRITIC = hyperparams.get("LR_CRITIC", 5e-4)
        self.TAU = hyperparams.get("TAU", 0.0095)             # Soft update parameter for target networks
        self.BATCH_SIZE = hyperparams.get("BATCH_SIZE", 64)         # Mini-batch size
        self.POLICY_NOISE = hyperparams.get("POLICY_NOISE", 0.2)      # Noise added to target policy during critic update
        self.NOISE_CLIP = hyperparams.get("NOISE_CLIP", 0.5)        # Range to clip target policy noise
        self.POLICY_FREQ = hyperparams.get("UPDATE_DELAY", 2)        # Frequency of delayed policy updates


        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)#, weight_decay=1e-2)

        self.critic_target.eval()  # Set target critic to evaluation mode
        self.actor_target.eval()   # Set target actor to evaluation mode
        for param in self.critic_target.parameters():
            param.requires_grad = False  # Freeze target critic parameters
        for param in self.actor_target.parameters():
            param.requires_grad = False  # Freeze target actor parameters

        self.max_action = max_action
        self.total_it = 0
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, max_action=None):
        if len(replay_buffer) < self.BATCH_SIZE:
            return

        if max_action is not None:
            self.max_action = max_action

        self.total_it += 1
        state, action, reward, next_state, done, gamma = replay_buffer.sample(self.BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.POLICY_NOISE).clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.POLICY_FREQ == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)



# --- HAC Agent ---
class HACAgent(nn.Module):
    def __init__(self, state_dim, action_dim, percept=False, device="cpu", hyperparams={}):
        super(HACAgent, self).__init__()

        # --- Hyperparameters ---
        # self.MAX_EPISODES = hyperparams.get("MAX_EPISODES", 2000)     # Maximum number of episodes to train
        
        
        self.GOAL_THRESHOLD = hyperparams.get("GOAL_THRESHOLD", 0.1)      # Threshold for considering the goal achieved
        self.NUM_UPDATES = hyperparams.get("NUM_UPDATES", 10)          # Number of DDPG updates per episode
        self.ALPHA = hyperparams.get("ALPHA", 0.01)   # Weight for episode reward
        self.N_LEVELS = hyperparams.get("N_LEVELS", 2) # Number of levels in the hierarchy
        self.SUBGOAL_HORIZON = hyperparams.get("SUBGOAL_HORIZON", 10)    # Number of steps to achieve a subgoal
        self.SUBGOAL_TEST_FREQ = hyperparams.get("SUBGOAL_TEST_FREQ", 0.3)  # Frequency of testing subgoal achievement
        self.EXPL_NOISE_WORKER = hyperparams.get("EXPL_NOISE_WORKER", 0.2)  # Exploration noise for worker level
        self.EXPL_NOISE_MANAGER = hyperparams.get("EXPL_NOISE_MANAGER", 0.5)  # Exploration noise for manager level
        self.SUBGOAL_BOUND = hyperparams.get("GOAL_BOUND", 1.0)  # Bound for subgoal values
        self.MAX_ENV_STEPS = hyperparams.get("MAX_ENV_STEPS", 1e6)  # Maximum number of environment steps to train

        self.wandb_upload = hyperparams.get("WANDB_UPLOAD", False)  # Whether to upload logs to Weights & Biases
        self.save_model = hyperparams.get("SAVE_MODEL", False)  # Whether to save the model weights after training

        # --- TD3 Hyperparameters ---
        self.LR_ACTOR = hyperparams.get('LR_ACTOR', 1e-4)
        self.LR_CRITIC = hyperparams.get('LR_CRITIC', 5e-4)
        self.TAU = hyperparams.get('TAU', 0.0095)
        self.BATCH_SIZE = hyperparams.get('BATCH_SIZE', 64)
        self.POLICY_NOISE = hyperparams.get('POLICY_NOISE', 0.2)
        self.NOISE_CLIP = hyperparams.get('NOISE_CLIP', 0.5)
        self.POLICY_FREQ = hyperparams.get('UPDATE_DELAY', 2)

        self.MAX_ACTION = hyperparams.get("MAX_ACTION", 1.0)          # Maximum action value 
        self.RANDOM_START_STEPS = hyperparams.get("RANDOM_START_STEPS", 1000)    # Number of random steps to populate the replay buffer before training
        self.BUFFER_SIZE = hyperparams.get("BUFFER_SIZE", int(1e5))  # Replay buffer size
        self.GAMMA = hyperparams.get("GAMMA", 0.99)            # Discount factor
        self.SEED = hyperparams.get("SEED", 31)  # Random seed for reproducibility


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = hyperparams.get("GOAL_DIM", state_dim)  # Goal dimension (default to state dimension)



        # Level 0 outputs primitive actions. Higher levels output state-difference subgoals.
        self.levels = nn.ModuleList()
        self.replay_buffers = []
        self.highest_subgoal_achieved_level = -1  # Track the highest level at which a subgoal was achieved
        for level in range(self.N_LEVELS):
            if level == 0:
                self.levels.append(TD3(self.state_dim + self.goal_dim, self.action_dim, self.MAX_ACTION, device=device, hyperparams=hyperparams))
            else:
                self.levels.append(TD3(self.state_dim + self.goal_dim, self.goal_dim, self.SUBGOAL_BOUND, device=device, hyperparams=hyperparams))
            self.replay_buffers.append(ReplayBuffer(self.BUFFER_SIZE, device))


        self.device = device

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")

    def select_level_action(self, level, state, goal, subgoal_test=False):
        if level == 0:
            level_input = np.concatenate([state, goal])
            action = self.levels[level].select_action(level_input)
            if not subgoal_test:
                action += np.random.normal(0, self.EXPL_NOISE_WORKER, size=self.action_dim)
            return np.clip(action, -self.MAX_ACTION, self.MAX_ACTION)

        level_input = np.concatenate([state, goal])
        subgoal = self.levels[level].select_action(level_input)
        if not subgoal_test:
            subgoal += np.random.normal(0, self.EXPL_NOISE_MANAGER, size=self.goal_dim)
        return np.clip(subgoal, -self.SUBGOAL_BOUND, self.SUBGOAL_BOUND)

    
    def compute_her_reward(self, current_state, desired_goal, next_state, dense=False):
        # Compute the reward based on the achieved goal and desired goal
        # HER paper section 4.4 reward function: r(s, a, g) = -1 if ||f(s') - g|| > epsilon else 0
        # reward shaping is detrimental to performance, so we use a sparse reward function
        achieved_goal = next_state - current_state  # Compute the achieved goal as the state difference
        distance = np.linalg.norm(desired_goal - achieved_goal)  # Euclidean distance between desired and achieved goals
        if dense:
            return -distance  # Negative reward proportional to the distance to the goal
        if distance > self.GOAL_THRESHOLD:  # Threshold for considering the goal achieved
            return -1.0  # Penalty for not achieving the goal
        else:
            # return -distance  # Negative reward proportional to the distance to the goal
            return 0.0

    # def goal_achieved(self, achieved_goal, desired_goal):
    #     for i in range(len(achieved_goal)):
    #         if abs(achieved_goal[i] - desired_goal[i]) > self.GOAL_THRESHOLD:
    #             return False
    #     return True

    def goal_transition(self, current_state, goal, next_state):
        # Subgoal remains fixed in absolute coordinates, so in relative coordinates it shifts
        return current_state + goal - next_state
        
    def sample_goal_states(self, next_obs_history, t, type="future", k=4):
        # Sample goals for HER
        # type: "future" or "final"
        # k: number of goals to sample
        if type == "final" or len(next_obs_history)-t <= k:
            # Sample the final state of the episode as the goal
            final_goal = next_obs_history[-1]
            return [final_goal] * k
        elif type == "future":
            # Sample k future states from the same episode
            return random.sample(next_obs_history[t:], k)
        elif type == "episode":
            # Sample from all states from the same episode
            return random.sample(next_obs_history, k)
        elif type == "random":
            # Sample k random goals from the replay buffer
            random_goals = random.sample(self.replay_buffers[0].storage, k)
            return [next_state[-self.goal_dim:] for _, _, _, next_state, _ in random_goals]
        else:
            raise ValueError("Invalid goal sampling type.")
        
    def train_level(self, env, level, state, info, goals=[], subgoal_test=False):

        hindsight_goal_transitions = []
        goal = goals[-1] if goals else env.observation_space.sample()  # Use the last subgoal as the goal for this level, or sample a random goal if no subgoals are provided

        for attempt in range(self.SUBGOAL_HORIZON):

            if np.random.rand() < self.SUBGOAL_TEST_FREQ:
                subgoal_test = True  # Test subgoal achievement with a certain probability

            if level > 0:
                action = self.select_level_action(level, state, goal, subgoal_test=subgoal_test)
                next_state = self.train_level(env, level-1, state, info, goals + [action], subgoal_test=subgoal_test)
            else:
                self.previous_gstate = info.get('achieved_goal', state)  # Store the previous achieved goal 
                action = self.select_level_action(0, state, goal, subgoal_test=subgoal_test)
                next_state, reward, terminated, truncated, info = env.step(action)
                self.done = terminated or truncated
                self.episode_reward += reward * (1-self.done)
                self.env_steps += 1 * (1-self.done)
                self.episode_length += 1 * (1-self.done)
                self.achieved_gstate = info.get('achieved_goal', next_state)  # Use the achieved goal from the info dictionary if available, otherwise use the next state
                for l in range(self.N_LEVELS):
                    if self.compute_her_reward(self.previous_gstate, goals[-1-l], self.achieved_gstate)==0:  # Check if any subgoals was achieved
                        self.highest_subgoal_achieved_level = max(self.highest_subgoal_achieved_level, l)

                if self.wandb_upload:
                    wandb.log({
                        "env_steps": self.env_steps,
                        "highest_subgoal_achieved_level": self.highest_subgoal_achieved_level,
                    })

            new_goal = self.goal_transition(self.previous_gstate, goal, self.achieved_gstate)  # Compute the new goal for the next step based on the achieved goal
            lvl_input = np.concatenate([state, goal])
            nxt_lvl_input = np.concatenate([next_state, new_goal])



            if level>0 and self.compute_her_reward(self.previous_gstate, goal, self.achieved_gstate)==-1:  # Check if goal was missed
                if subgoal_test:
                    penalty = -self.SUBGOAL_HORIZON  # HAC sec 4.5 penalty is -H
                    self.replay_buffers[level].add(lvl_input, action, penalty, nxt_lvl_input, float(self.done), 0) # gamma=0 ?
                action = self.achieved_gstate - self.previous_gstate  # Replace with the subgoal actually achieved.
                    
            # HAC 4.3 Hindsight Action Transition
            r_g = self.compute_her_reward(self.previous_gstate, goal, self.achieved_gstate) + pow(self.ALPHA, self.N_LEVELS-1-level) * self.episode_reward
            self.replay_buffers[level].add(lvl_input, action, r_g, nxt_lvl_input, float(self.done), self.GAMMA)
            # Store hindsight goal transition
            hindsight_goal_transitions.append((state, action, r_g, next_state, float(self.done), goal, self.previous_gstate, self.achieved_gstate))
            state = next_state

            if self.wandb_upload:
                wandb.log(
                    {
                        f"level_{level}_reward": r_g,
                    }
                )
            
            if self.done or self.highest_subgoal_achieved_level >= level:  # If the episode is done or any subgoal at this level or above is achieved, break
                break
        
        # HAC 4.4 Hindsight Goal Transition
        # Push hindsight goal transitions to the replay buffer (fill up TBD HAC sec 4.4)
        achieved_gstates = [transition[7] for transition in hindsight_goal_transitions]  # Extract achieved states from the transitions
        for t in range(len(hindsight_goal_transitions)):
            gstates = self.sample_goal_states(achieved_gstates, t, type="future", k=2)  # Sample future goals for HER
            for gstate in gstates:
                state, action, _, next_state, done, _, previous_gstate, achieved_gstate = hindsight_goal_transitions[t]
                goal = gstate - previous_gstate # Compute the new goal based on the achieved state and the sampled future goal
                r_g = self.compute_her_reward(previous_gstate, goal, achieved_gstate) + pow(self.ALPHA, self.N_LEVELS-1-level) * self.episode_reward
                lvl_input = np.concatenate([state, goal])
                nxt_lvl_input = np.concatenate([next_state, goal])
                self.replay_buffers[level].add(lvl_input, action, r_g, nxt_lvl_input, float(done), self.GAMMA)  

        return next_state

    def test_level(self, env, level, state, info, goals=[], subgoal_test=False):

        goal = goals[-1] if goals else env.observation_space.sample()  # Use the last subgoal as the goal for this level, or sample a random goal if no subgoals are provided

        for attempt in range(self.SUBGOAL_HORIZON):

            if level > 0:
                action = self.select_level_action(level, state, goal, subgoal_test=subgoal_test)
                next_state = self.test_level(env, level-1, state, info, goals + [action], subgoal_test=subgoal_test)
            else:
                self.previous_gstate = info.get('achieved_goal', state)  # Store the previous achieved goal
                action = self.select_level_action(0, state, goal, subgoal_test=subgoal_test)
                next_state, reward, terminated, truncated, info = env.step(action)
                self.done = terminated or truncated
                self.episode_reward += reward * (1-self.done)
                self.episode_length += 1 * (1-self.done)
                self.achieved_gstate = info.get('achieved_goal', next_state)  # Use the achieved goal from the info dictionary if available, otherwise use the next state
                for l in range(self.N_LEVELS):
                    if self.compute_her_reward(self.previous_gstate, goals[-1-l], self.achieved_gstate)==0:  # Check if any subgoals was achieved
                        self.highest_subgoal_achieved_level = max(self.highest_subgoal_achieved_level, l)
                

            goal = self.goal_transition(self.previous_gstate, goal, self.achieved_gstate)  # Compute the new goal for the next step based on the achieved goal
            state = next_state

            if self.done or self.highest_subgoal_achieved_level >= level:  
                break
        
        return next_state
    
    def train_agent(self, env, store_episode_rewards=False):

        env_name = env.spec.id
        model_path = None

        if env.action_space.__class__.__name__ == "Discrete":
            print("Discrete action space detected. DDPG is not suitable for discrete action spaces.")
            # return

        if self.save_model:
            model_path = f"Weights/{env_name}_HAC_weights/HAC_{env_name}_{self.MAX_ENV_STEPS}_{self.LR_ACTOR:.4f}_{self.LR_CRITIC:.4f}_{self.SUBGOAL_HORIZON}"
            os.makedirs(model_path, exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="HAC-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "architecture": "HAC",
                    "seed": self.SEED,
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "actor_lr": self.LR_ACTOR,
                    "critic_lr": self.LR_CRITIC,
                    "gamma": self.GAMMA,
                    "tau": self.TAU,
                    "batch_size": self.BATCH_SIZE,
                    "max_env_steps": self.MAX_ENV_STEPS,
                    "policy_noise": self.POLICY_NOISE,
                    "noise_clip": self.NOISE_CLIP,
                    "policy_freq": self.POLICY_FREQ,
                    "expl_noise_worker": self.EXPL_NOISE_WORKER,
                    "expl_noise_manager": self.EXPL_NOISE_MANAGER,
                    "random_start_steps": self.RANDOM_START_STEPS,
                    "max_action": self.MAX_ACTION,
                    "buffer_size": self.BUFFER_SIZE,
                    "goal_threshold": self.GOAL_THRESHOLD,
                    "num_updates": self.NUM_UPDATES,
                    "alpha": self.ALPHA,
                    "n_levels": self.N_LEVELS,
                    "subgoal_horizon": self.SUBGOAL_HORIZON,
                    "subgoal_test_freq": self.SUBGOAL_TEST_FREQ,
                    "subgoal_bound": self.SUBGOAL_BOUND,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params,
                    "model_path": model_path
                },
            )
            run.name = f"{env_name}_{self.MAX_ENV_STEPS}_{self.LR_ACTOR:.4f}_{self.LR_CRITIC:.4f}_{self.SUBGOAL_HORIZON}"

        # create a wrapper environment to save episode returns and episode lengths
        # create a wrapper environment to save episode returns and episode lengths
        # envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
        #     envs, buffer_length=n_envs * n_updates
        # )

        # obs, _ = env.reset()
        # for i in tqdm(range(self.RANDOM_START_STEPS)):
        #     action = env.action_space.sample()
        #     next_obs, reward, terminated, truncated, info = env.step(action)
        #     done = terminated or truncated
        #     self.replay_buffers[level].add(obs, action, reward, next_obs, float(done), env.observation_space.sample())  # Sample a random goal from the observation space
        #     obs = next_obs
        #     if done:
        #         obs, _ = env.reset()

        self.env_steps = 0
        episode_rewards = []
        self.MAX_ACTION = env.action_space.high[0] if isinstance(env.action_space, gym.spaces.Box) else 1.0

        with tqdm(total=self.MAX_ENV_STEPS, desc="Training Progress") as pbar:
            while self.env_steps < self.MAX_ENV_STEPS:
                obs, info = env.reset(seed = self.SEED + self.env_steps)
                self.episode_reward = 0
                self.episode_length = 0
                self.done = False
                while not self.done:
                    desired_gstate = info.get('desired_goal', env.observation_space.sample())  # Use the desired goal from the info dictionary if available, otherwise sample a random goal
                    achieved_gstate = info.get('achieved_goal', obs)  # Use the achieved goal from the info dictionary if available, otherwise use the current observation
                    goal = desired_gstate - achieved_gstate  # Compute the goal as the difference between the desired goal and the current observation
                    self.highest_subgoal_achieved_level = -1  # Reset the highest subgoal achieved level for this episode
                    next_obs = self.train_level(env, self.N_LEVELS-1, obs, info, [goal])  # Start training from the highest level
                    obs = next_obs

                    # Update the agent
                    for level in range(self.N_LEVELS):
                        self.levels[level].train(self.replay_buffers[level], max_action=self.MAX_ACTION)

                if self.wandb_upload:
                    wandb.log({
                        "episode_reward": self.episode_reward,
                        "episode_length": self.episode_length,
                    })

                if store_episode_rewards:
                    episode_rewards.append(self.episode_reward)
                    if len(episode_rewards) > 10:
                        episode_rewards.pop(0)
                    avg_reward = np.mean(episode_rewards)
                    if self.wandb_upload:
                        wandb.log({
                            "mean_episode_return": avg_reward,
                        })

                # Update the agent after each episode
                # for i in range(self.NUM_UPDATES):
                #     # self.ddpg_update()
                #     for level in range(self.N_LEVELS):
                #         self.levels[level].train(self.replay_buffers[level], max_action=self.MAX_ACTION)

                    # print(f"Saved model weights at episode {episode + 1} to {model_path}")
                
                pbar.update(self.episode_length)

        if self.save_model:
            torch.save(self.state_dict(), f"{model_path}/model.pt")

        if self.wandb_upload:
            run.finish()

    def test_agent(self, env, n_episodes=1, record_stats=False, model_path=None):

        env_name = env.spec.id

        if model_path is None:
            model_path = f"Weights/{env_name}_HAC_weights/HAC_{env_name}_{self.MAX_ENV_STEPS}_{self.LR_ACTOR:.4f}_{self.LR_CRITIC:.4f}_{self.SUBGOAL_HORIZON}"

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=self.device))
        else:
            print("Weights not found. Please train the agent first.")
            return

        if self.wandb_upload:
            run = wandb.init(
                entity="arkapatra31-ust",
                project="HAC-Test",
                config={
                    "env": env_name,
                    "architecture": "HAC",
                    "model_path": model_path,
                    "seed": self.SEED,
                    "action_dim": self.action_dim,
                    "state_dim": self.state_dim
                }
            )
            run.name = f"{env_name}_test_{n_episodes}_episodes"

        if record_stats:
            episode_rewards = []
            episode_lengths = []

        for episode in range(n_episodes):

            obs, info = env.reset(seed=self.SEED + episode)
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
                wandb.log({
                    "episode_reward": self.episode_reward,
                    "episode_length": self.episode_length,
                })
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
                episode_rewards.append(self.episode_reward)
                episode_lengths.append(self.episode_length)
            
            print(f"Episode {episode + 1}: Total Reward: {self.episode_reward}, Length: {self.episode_length}")
        wandb.finish()

        if record_stats:
            return episode_rewards, episode_lengths

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

