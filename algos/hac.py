from turtle import done

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
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # BipedalWalker actions are bounded between -1 and 1
        x = self.max_action * torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # State and action are concatenated before passing into the first hidden layer
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, goal, action):
        # print(f"State shape: {state.shape}, Goal shape: {goal.shape}, Action shape: {action.shape}")
        x = torch.cat([state, goal, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done, goal, gamma):
        self.buffer.append((state, action, reward, next_state, done, goal, gamma))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals, gammas = zip(*batch)
        
        return (states, actions, rewards, next_states, dones, goals, gammas)

    def __len__(self):
        return len(self.buffer)

# --- HAC Agent ---
class HACAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hyperparams={}, device="cpu"):
        super(HACAgent, self).__init__()

        # --- Hyperparameters ---
        self.LR_ACTOR = hyperparams.get("LR_ACTOR", 1e-4)
        self.LR_CRITIC = hyperparams.get("LR_CRITIC", 3e-4)
        self.GAMMA = hyperparams.get("GAMMA", 0.99)            # Discount factor
        self.TAU = hyperparams.get("TAU", 0.005)             # Soft update parameter for target networks
        self.BATCH_SIZE = hyperparams.get("BATCH_SIZE", 64)         # Mini-batch size
        self.MAX_EPISODES = hyperparams.get("MAX_EPISODES", 2000)     # Maximum number of episodes to train
        self.EXPLORATION_NOISE = hyperparams.get("EXPLORATION_NOISE", 0) # Standard deviation of Gaussian noise for exploration
        self.RANDOM_START_STEPS = hyperparams.get("RANDOM_START_STEPS", 1000)    # Number of random steps to populate the replay buffer before training
        self.MAX_ACTION = hyperparams.get("MAX_ACTION", 1.0)          # Maximum action value (for BipedalWalker, actions are bounded between -1 and 1)
        self.BUFFER_SIZE = hyperparams.get("BUFFER_SIZE", int(1e6))  # Replay buffer size
        self.GOAL_THRESHOLD = hyperparams.get("GOAL_THRESHOLD", 0.1)      # Threshold for considering the goal achieved
        self.NUM_UPDATES = hyperparams.get("NUM_UPDATES", 100)          # Number of DDPG updates per episode
        self.ALPHA = hyperparams.get("ALPHA", 1)   # Weight for the HER loss term
        self.N_LEVELS = hyperparams.get("N_LEVELS", 2) # Number of levels in the hierarchy
        self.SUBGOAL_HORIZON = hyperparams.get("SUBGOAL_HORIZON", 500)    # Number of steps to achieve a subgoal
        self.SUBGOAL_TEST_FREQ = hyperparams.get("SUBGOAL_TEST_FREQ", 0.3)  # Frequency of testing subgoal achievement

        goal_dim = state_dim  # Assuming the goal is represented in the same space as the state

        # Higher level actors output subgoals, lowest level actor outputs actions
        self.actor = nn.ModuleList([Actor(state_dim+goal_dim, goal_dim, self.MAX_ACTION).to(device) for k in range(self.N_LEVELS)])
        self.actor[0] = Actor(state_dim+goal_dim, action_dim, self.MAX_ACTION).to(device)  # Lowest level actor

        self.critic = nn.ModuleList([Critic(state_dim+goal_dim, goal_dim).to(device) for k in range(self.N_LEVELS)])
        self.critic[0] = Critic(state_dim+goal_dim, action_dim).to(device)  # Lowest level critic

        self.replay_buffer = [ReplayBuffer(int(self.BUFFER_SIZE)) for k in range(self.N_LEVELS)]

        self.actor_optimizer = []
        self.critic_optimizer = []
        for k in range(self.N_LEVELS):
            self.actor_optimizer.append(optim.Adam(self.actor[k].parameters(), lr=self.LR_ACTOR))
            self.critic_optimizer.append(optim.Adam(self.critic[k].parameters(), lr=self.LR_CRITIC, weight_decay=1e-2))

        self.max_action = self.MAX_ACTION
        self.device = device

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")

    def select_action(self, state, goal, noise_std=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(self.device)
        self.actor[0].eval() # Set to evaluation mode
        with torch.no_grad():
            action = self.actor[0](state, goal).cpu().data.numpy().flatten() # actor[0] is the lowest level actor
        self.actor[0].train() # Set back to train mode
        
        # Add exploration noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
    
    def compute_her_reward(self, achieved_goal, desired_goal):
        # Compute the reward based on the achieved goal and desired goal
        # HER paper section 4.4 reward function: r(s, a, g) = -1 if ||f(s') - g|| > epsilon else 0
        # reward shaping is detrimental to performance, so we use a sparse reward function
        distance = np.linalg.norm(achieved_goal - desired_goal)
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
        
    def sample_goals(self, next_obs_history, t, type="future", k=4):
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
            random_goals = random.sample(self.replay_buffer.buffer, k)
            return [goal for _, _, _, _, _, goal in random_goals]
        else:
            raise ValueError("Invalid goal sampling type.")
        
    def train_level(self, env, level, state, goal, subgoal_test=False):

        hindsight_goal_transitions = []

        for attempt in range(self.SUBGOAL_HORIZON):

            if np.random.rand() < self.SUBGOAL_TEST_FREQ:
                subgoal_test = True  # Test subgoal achievement with a certain probability

            if level > 0:
                if subgoal_test:
                    action = goal  # No exploration noise during subgoal testing
                else:
                    action = goal + np.random.normal(0, self.EXPLORATION_NOISE, size=goal.shape)  # Add exploration noise to the subgoal
                next_state = self.train_level(env, level-1, state, action, subgoal_test=subgoal_test)
            else:
                if subgoal_test:
                    action = self.select_action(state, goal, noise_std=0.0)  # No exploration noise during subgoal testing
                else:
                    action = self.select_action(state, goal, noise_std=self.EXPLORATION_NOISE)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.done = terminated or truncated
                self.episode_reward += reward * (1-self.done)
                self.episode_length += 1 * (1-self.done)

            if level>0 and self.compute_her_reward(next_state, goal)==-1:  # Check if the subgoal was missed
                if subgoal_test:
                    penalty = -self.SUBGOAL_HORIZON  # HAC sec 4.5 penalty is -H
                    self.replay_buffer[level].add(state, action, penalty, next_state, float(self.done), goal, 0) # gamma=0 for subgoal testing
                action = next_state  # Replace original action with action executed in hindsight
                    
            # HAC 4.3 Hindsight Action Transition
            r_g = self.compute_her_reward(next_state, goal) + self.ALPHA * self.episode_reward
            self.replay_buffer[level].add(state, action, r_g, next_state, float(self.done), goal, self.GAMMA)
            # Store hindsight goal transition
            hindsight_goal_transitions.append((state, action, r_g, next_state, float(self.done), goal))
            state = next_state
            
            if self.done:
                break
        
        # HAC 4.4 Hindsight Goal Transition
        # Push hindsight goal transitions to the replay buffer (fill up TBD HAC sec 4.4)
        for t in range(len(hindsight_goal_transitions)-1):
            self.replay_buffer[level].add(
                hindsight_goal_transitions[t][0], # state
                hindsight_goal_transitions[t][1], # action
                -1, # reward = -1
                hindsight_goal_transitions[t][3], # next_state
                hindsight_goal_transitions[t][4], # done
                hindsight_goal_transitions[-1][3],  # goal = last next_state in the trajectory
                self.GAMMA  # gamma
            )
        self.replay_buffer[level].add(
            hindsight_goal_transitions[-1][0], # state
            hindsight_goal_transitions[-1][1], # action
            0, # reward=0
            hindsight_goal_transitions[-1][3], # next_state
            hindsight_goal_transitions[-1][4], # done
            hindsight_goal_transitions[-1][3],  # goal
            0  # gamma = 0 for hindsight goal transitions
        )

        return next_state
    
    def train_agent(self, env):

        wandb_upload = True      # Whether to upload logs to Weights & Biases
        save_weights = True

        env_name = env.spec.id

        if env.action_space.__class__.__name__ == "Discrete":
            print("Discrete action space detected. DDPG is not suitable for discrete action spaces.")
            # return

        model_path = f"Weights/{env_name}_HAC_weights/HAC_{env_name}_{self.MAX_EPISODES}.pt"

        os.makedirs(f"Weights/{env_name}_HAC_weights", exist_ok=True)

        if wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="HAC-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "actor_lr": self.LR_ACTOR,
                    "critic_lr": self.LR_CRITIC,
                    "gamma": self.GAMMA,
                    "tau": self.TAU,
                    "batch_size": self.BATCH_SIZE,
                    "max_episodes": self.MAX_EPISODES,
                    "exploration_noise": self.EXPLORATION_NOISE,
                    "random_start_steps": self.RANDOM_START_STEPS,
                    "max_action": self.MAX_ACTION,
                    "buffer_size": self.BUFFER_SIZE,
                    "goal_threshold": self.GOAL_THRESHOLD,
                    "num_updates": self.NUM_UPDATES,
                    "alpha": self.ALPHA,
                    "n_levels": self.N_LEVELS,
                    "subgoal_horizon": self.SUBGOAL_HORIZON,
                    "subgoal_test_freq": self.SUBGOAL_TEST_FREQ,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params,
                    "model_path": model_path
                },
            )

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
        #     self.replay_buffer.add(obs, action, reward, next_obs, float(done), env.observation_space.sample())  # Sample a random goal from the observation space
        #     obs = next_obs
        #     if done:
        #         obs, _ = env.reset()

        for episode in tqdm(range(self.MAX_EPISODES)):
            obs, info = env.reset()
            self.episode_reward = 0
            self.episode_length = 0
            self.done = False
            goal = env.observation_space.sample()  # Sample a random goal from the observation space (could be desired state space to reach)
            
            next_obs = self.train_level(env, self.N_LEVELS-1, obs, goal)  # Start training from the highest level

            if wandb_upload:
                wandb.log({
                    "episode_reward": self.episode_reward,
                    "episode_length": self.episode_length,
                })

            # Have to check done logic

            # Update the agent after each episode
            for i in range(self.NUM_UPDATES):
                self.ddpg_update()

        if wandb_upload:
            run.finish()

        """ save network weights """
        if save_weights:
            torch.save(self.state_dict(), model_path)

    def test_agent(self, env, n_episodes=1):

        env_name = env.spec.id

        model_path = f"Weights/{env_name}_HAC_weights/HAC_{env_name}_{self.MAX_EPISODES}.pt"

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Weights not found. Please train the agent first.")
            return

        for episode in range(n_episodes):

            obs, info = env.reset()
            done = False
            episode_reward = 0
            goal = env.observation_space.sample()  # Sample a random goal from the observation space
            episode_length = 0
            renders = []  # Store frames for video logging
            
            while not done:
                action = self.select_action(obs, goal, noise_std=0.0)  # No exploration noise during testing
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                obs = next_obs

                if self.wandb_upload:
                     # Capture the render for logging
                     frame = env.render()
                     if frame is not None:
                         # W&B handles videos best when frames are uint8 CHW tensors.
                         if frame.shape[-1] == 4:
                             frame = frame[:, :, :3]
                         frame = np.asarray(frame, dtype=np.uint8)
                         renders.append(frame)

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
            
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Length: {episode_length}")

    def ddpg_update(self):

        for k in range(self.N_LEVELS):

            if len(self.replay_buffer[k]) > self.BATCH_SIZE:    
    
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

def hyperparameter_tuning(env, state_dim, action_dim, percept=False, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'episode_reward', 'goal': 'maximize'},
        'parameters': {
            'LR_ACTOR': {'values': [1e-4, 3e-4, 1e-3]},
            'LR_CRITIC': {'values': [1e-4, 3e-4, 1e-3]},
        },
        'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project="HAC-Hyperparameter-Tuning", entity="arkapatra31-ust")

    # Define the training function for the sweep
    def train_sweep():
        with wandb.init() as run:
            config = wandb.config
            hyperparams = {
                "LR_ACTOR": config.LR_ACTOR,
                "LR_CRITIC": config.LR_CRITIC,
                "wandb_upload": True,
            }
            wandb.run.name = f"HAC_{config.LR_ACTOR:.4f}_{config.LR_CRITIC:.4f}_{env.spec.id}"
            print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
            hac_agent = HACAgent(state_dim, action_dim, hyperparams=hyperparams, device=device)
            hac_agent.train_agent(env=env)
            # hac_agent.test_agent(env=env, n_episodes=3)

    # Start the sweep agent to run the training function for each configuration
    wandb.agent(sweep_id, function=train_sweep, count=5)  # Adjust count as needed for the number of runs

def delay_injection(env, delay_steps=5, delay_type='observation'):

    if getattr(env, '_delay_wrapped', False):
        return env

    if delay_type == 'observation':
        """Injects a delay in the environment's observations."""
        original_step = env.step

        def delayed_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            # Introduce a delay by returning the previous observation for 'delay_steps' steps
            if not hasattr(env, 'delay_buffer'):
                env.delay_buffer = [obs] * delay_steps
            else:
                env.delay_buffer.append(obs)
                if len(env.delay_buffer) > delay_steps:
                    env.delay_buffer.pop(0)
            delayed_obs = env.delay_buffer[0]
            # print(f"Delayed observation: {delayed_obs}")
            return delayed_obs, reward, terminated, truncated, info

    elif delay_type == 'action':
        """Injects a delay in the environment's actions."""
        original_step = env.step

        def delayed_step(action):
            if not hasattr(env, 'action_buffer'):
                env.action_buffer = [action] * delay_steps
            else:
                env.action_buffer.append(action)
                if len(env.action_buffer) > delay_steps:
                    env.action_buffer.pop(0)
            delayed_action = env.action_buffer[0]
            return original_step(delayed_action)
    else:
        raise ValueError(f"Unsupported delay_type: {delay_type}")

    env.step = delayed_step
    env._delay_wrapped = True
    return env
    
def noise_injection(env, noise_std=0.1, noise_type='observation'):

    if getattr(env, '_noise_wrapped', False):
        return env

    if noise_type == 'observation':
        """Injects Gaussian noise into the environment's observations."""
        original_step = env.step

        def noisy_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            noisy_obs = obs + np.random.normal(0, noise_std, size=obs.shape)
            return noisy_obs, reward, terminated, truncated, info

    elif noise_type == 'action':
        if env.action_space.__class__.__name__ != 'Discrete':
            """Chooses random actions if with nois_std probability, otherwise takes the original action."""
            original_step = env.step

            def noisy_step(action):
                if np.random.rand() < noise_std:
                    action = env.action_space.sample()
                return original_step(action)
        else:
            """Injects Gaussian noise into the environment's actions."""
            original_step = env.step

            def noisy_step(action):
                noisy_action = action + np.random.normal(0, noise_std, size=action.shape)
                return original_step(noisy_action)
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    env.step = noisy_step
    env._noise_wrapped = True
    return env
