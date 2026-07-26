import gymnasium as gym
import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections

from tqdm import tqdm
import wandb
import os
import cv2

# --- Neural Networks ---
class Actor(torch_nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = torch_nn.Linear(state_dim, 400)
        self.fc2 = torch_nn.Linear(400, 300)
        self.fc3 = torch_nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # BipedalWalker actions are bounded between -1 and 1
        x = self.max_action * torch.tanh(self.fc3(x))
        return x

class Critic(torch_nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # State and action are concatenated before passing into the first hidden layer
        self.fc1 = torch_nn.Linear(state_dim + action_dim, 400)
        self.fc2 = torch_nn.Linear(400, 300)
        self.fc3 = torch_nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done, goal=None):
        self.buffer.append((state, action, reward, next_state, done, goal))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        
        return (states, actions, rewards, next_states, dones, goals)

    def __len__(self):
        return len(self.buffer)

# --- HER Agent ---
class HERAgent(torch_nn.Module):
    def __init__(self, state_dim, action_dim, hyperparams={}, device="cpu"):
        super(HERAgent, self).__init__()

        self.LR_ACTOR = hyperparams.get("LR_ACTOR", 1e-3)
        self.LR_CRITIC = hyperparams.get("LR_CRITIC", 1e-3)
        self.GAMMA = hyperparams.get("GAMMA", 0.98)            # Discount factor
        self.TAU = hyperparams.get("TAU", 0.05)             # Soft update parameter for target networks
        self.BATCH_SIZE = hyperparams.get("BATCH_SIZE", 128)         # Mini-batch size
        self.MAX_EPISODES = hyperparams.get("MAX_EPISODES", 100000)     # Maximum number of episodes to train
        self.EXPLORATION_NOISE = hyperparams.get("EXPLORATION_NOISE", 0.1) # Standard deviation of Gaussian noise for exploration
        self.RANDOM_START_STEPS = hyperparams.get("RANDOM_START_STEPS", 1000)    # Number of random steps to populate the replay buffer before training
        self.MAX_ACTION = hyperparams.get("MAX_ACTION", 5.0)          # Maximum action value (-x, x) for continuous action spaces
        self.BUFFER_SIZE = hyperparams.get("BUFFER_SIZE", int(1e6))  # Replay buffer size
        self.GOAL_THRESHOLD = hyperparams.get("GOAL_THRESHOLD", 0.5)      # Threshold for considering the goal achieved
        self.NUM_UPDATES = hyperparams.get("NUM_UPDATES", 4)          # Number of DDPG updates per episode
        self.ALPHA = hyperparams.get("ALPHA", 0)                   # Weight for the HER loss term

        self.wandb_upload = hyperparams.get("wandb_upload", False)
        self.save_weights = hyperparams.get("save_weights", False)

        goal_dim = 3  # Assuming the goal is represented in the same space as the state

        self.actor = Actor(state_dim+goal_dim, action_dim, self.MAX_ACTION).to(device)
        self.actor_target = Actor(state_dim+goal_dim, action_dim, self.MAX_ACTION).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

        self.critic = Critic(state_dim+goal_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim+goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC, weight_decay=1e-2)

        self.max_action = self.MAX_ACTION

        self.replay_buffer = ReplayBuffer(int(self.BUFFER_SIZE))
        self.device = device

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")


    def select_action(self, state, goal, noise_std=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(self.device)
        self.actor.eval() # Set to evaluation mode
        with torch.no_grad():
            action = self.actor(torch.cat([state, goal], dim=1)).cpu().data.numpy().flatten()
        self.actor.train() # Set back to train mode
        
        # Add exploration noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
    
    def compute_her_reward(self, desired_goal, achieved_goal):
        # Compute the reward based on the achieved goal and desired goal
        # HER paper section 4.4 reward function: r(s, a, g) = -1 if ||f(s') - g|| > epsilon else 0
        # reward shaping is detrimental to performance, so we use a sparse reward function
        # achieved_goal = achieved_goal[:-10]
        distance = np.linalg.norm(achieved_goal - desired_goal)
        if distance > self.GOAL_THRESHOLD:  # Threshold for considering the goal achieved
            return -1.0 * self.ALPHA # Penalty for not achieving the goal
        else:
            # return -distance  # Negative reward proportional to the distance to the goal
            return 0.0
        
    def sample_goals(self, achieved_goal_history, t, type="future", k=4):
        # Sample goals for HER
        # type: "future" or "final"
        # k: number of goals to sample
        if type == "final" or len(achieved_goal_history)-t <= k:
            # Sample the final state of the episode as the goal
            final_goal = achieved_goal_history[-1]
            return [final_goal] * k
        elif type == "future":
            # Sample k future states from the same episode
            return random.sample(achieved_goal_history[t:], k)
        elif type == "episode":
            # Sample from all states from the same episode
            return random.sample(achieved_goal_history, k)
        elif type == "random":
            # Sample k random goals from the replay buffer
            random_goals = random.sample(self.replay_buffer.buffer, k)
            return [goal for _, _, _, _, _, goal in random_goals]
        else:
            raise ValueError("Invalid goal sampling type.")

    def train_agent(self, env, hyperparams=None):

        env_name = env.spec.id

        if env.action_space.__class__.__name__ == "Discrete":
            print("Discrete action space detected. DDPG is not suitable for discrete action spaces.")
            # return

        actor_weights_path = f"Weights/{env_name}_HER_weights/actor_weights_{self.MAX_EPISODES}.pt"
        critic_weights_path = f"Weights/{env_name}_HER_weights/critic_weights_{self.MAX_EPISODES}.pt"

        os.makedirs(f"Weights/{env_name}_HER_weights", exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="HER-Train",
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
                    "buffer_size": self.BUFFER_SIZE,
                    "goal_threshold": self.GOAL_THRESHOLD,
                    "num_updates": self.NUM_UPDATES,
                    "alpha": self.ALPHA,
                    "actor_weights_path": actor_weights_path,
                    "critic_weights_path": critic_weights_path,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params
                },
            )
            run.name=f"{env_name}_actor_lr_{self.LR_ACTOR}_critic_lr_{self.LR_CRITIC}_episodes_{self.MAX_EPISODES}_alpha_{self.ALPHA}"

        # create a wrapper environment to save episode returns and episode lengths
        # create a wrapper environment to save episode returns and episode lengths
        # envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
        #     envs, buffer_length=n_envs * n_updates
        # )

        obs, desired_goal, achieved_goal, _ = env.reset()
        # print(f"Initial Observation: {obs}, Desired Goal: {desired_goal}, Achieved Goal: {achieved_goal}")
        for i in tqdm(range(self.RANDOM_START_STEPS)):
            action = env.action_space.sample()
            next_obs, desired_goal, achieved_goal, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            goal = (desired_goal, achieved_goal)  # Use the desired goal from the environment as the goal
            self.replay_buffer.add(obs, action, reward, next_obs, float(done), desired_goal)
            obs = next_obs
            if done:
                obs, desired_goal, achieved_goal, _ = env.reset()

        for episode in tqdm(range(self.MAX_EPISODES)):
            obs, desired_goal, achieved_goal, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            # goal = env.observation_space.sample()[:-10]  # Sample a random goal from the observation space
            goal = (desired_goal, achieved_goal)  # Use the desired goal from the environment as the goal
            current_episode_transitions = []  # Store transitions for the current episode
            
            while not done:
                # Select action with noise for exploration
                action = self.select_action(obs, goal[0], noise_std=self.EXPLORATION_NOISE)
                
                # Step environment
                next_obs, desired_goal, achieved_goal, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                goal = (desired_goal, achieved_goal)  # Use the desired goal from the environment as the goal

                # r_g = self.compute_her_reward(next_obs, goal)  # Compute HER reward based on the achieved goal and desired goal
                # r = reward + r_g  # Combine the original reward and HER reward with a weight alpha
                current_episode_transitions.append((obs, action, reward, next_obs, float(done), achieved_goal))  # Store transition for HER
                
                # Store experience
                self.replay_buffer.add(obs, action, reward, next_obs, float(done), desired_goal)

                obs = next_obs
                episode_reward += reward
                episode_length += 1

                self.ddpg_update()  # Update the agent after each step

            if self.wandb_upload:
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                })

            # next_obs_history = [trans[3] for trans in current_episode_transitions]  # Extract next observations from the transitions
            achieved_goal_history = [trans[-1] for trans in current_episode_transitions]  # Extract achieved goals from the transitions
            for t in range(len(current_episode_transitions)):
                goals = self.sample_goals(achieved_goal_history, t, type="future", k=4)
                # print("sampled goals:", goals)
                for goal in goals:
                    # print("achieved_goal_history[t]:", achieved_goal_history[t], "goal:", goal)
                    r_g = self.compute_her_reward(achieved_goal_history[t], goal)  # Compute HER reward based on the achieved goal and sampled goal
                    r = current_episode_transitions[t][2] + r_g  # Combine the original reward and HER reward with a weight alpha
                    # print(f"done: {current_episode_transitions[t][4]}, r: {r}, goal: {goal}")
                    self.replay_buffer.add(current_episode_transitions[t][0], current_episode_transitions[t][1], r, current_episode_transitions[t][3], current_episode_transitions[t][4], goal)

            # Update the agent after each episode
            # for i in range(self.NUM_UPDATES):
            #     self.ddpg_update()

        if self.wandb_upload:
            run.finish()

        """ save network weights """
        if self.save_weights:
            torch.save(self.actor.state_dict(), actor_weights_path)
            torch.save(self.critic.state_dict(), critic_weights_path)

    def test_agent(self, env, n_episodes=1):

        env_name = env.spec.id

        actor_weights_path = f"Weights/{env_name}_HER_weights/actor_weights_{self.MAX_EPISODES}.pt"
        critic_weights_path = f"Weights/{env_name}_HER_weights/critic_weights_{self.MAX_EPISODES}.pt"

        if os.path.exists(actor_weights_path) and os.path.exists(critic_weights_path):
            self.actor.load_state_dict(torch.load(actor_weights_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_weights_path, map_location=self.device))
        else:
            print("Weights not found. Please train the agent first.")
            return

        for episode in range(n_episodes):

            if self.wandb_upload:
                run = wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="HER-Test",
                    # group=f"DDPG-Obs-Delay-10",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "actor_lr": self.LR_ACTOR,
                        "critic_lr": self.LR_CRITIC,
                        "gamma": self.GAMMA,
                        "tau": self.TAU,
                        "batch_size": self.BATCH_SIZE,
                        "max_episodes": self.MAX_EPISODES,
                        "exploration_noise": self.EXPLORATION_NOISE
                    },
                )
                run.name = f"HER_Test_Episode_{episode + 1}"

            obs, info = env.reset()
            done = False
            episode_reward = 0
            goal = env.observation_space.sample()  # Sample a random goal from the observation space
            episode_length = 0
            renders = [] 
            
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

                    wandb.log({
                        "episode_reward": episode_reward,
                        "action": action,
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
            
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Length: {episode_length}")

    def ddpg_update(self):

        if len(self.replay_buffer) > self.BATCH_SIZE:
    
            # Sample a batch of transitions
            states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(self.BATCH_SIZE)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
            goals = torch.FloatTensor(np.array(goals)).to(self.device)

            # ------------------- Optimize Critic ------------------- #
            # Get predicted next-state actions and Q values from target models
            with torch.no_grad():
                next_actions = self.actor_target(torch.cat([next_states, goals], dim=1))
                target_Q = self.critic_target(torch.cat([next_states, goals], dim=1), next_actions)
                # Compute the target Q value
                target_Q = rewards + (1 - dones) * self.GAMMA * target_Q

            # Get current Q estimate
            current_Q = self.critic(torch.cat([states, goals], dim=1), actions)

            # Compute critic loss (MSE)
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ------------------- Optimize Actor ------------------- #
            # Compute actor loss (Negative of the critic's output for the actor's actions)
            actor_loss = -self.critic(torch.cat([states, goals], dim=1), self.actor(torch.cat([states, goals], dim=1))).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.wandb_upload:
                wandb.log({
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                })

            # ------------------- Soft Updates ------------------- #
            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

def hyperparameter_tuning(env, state_dim, action_dim, percept=False, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'episode_reward', 'goal': 'maximize'},
        'parameters': {
            # 'ALPHA': {'distribution': 'uniform', 'min': 1e-3, 'max': 1.0},
            'ALPHA': {'values': [0.01, 0.2, 0.5, 0.8, 1.0]},
        },
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project="HER-Hyperparameter-Tuning", entity="arkapatra31-ust")

    # Define the training function for the sweep
    def train_sweep():
        with wandb.init() as run:
            config = wandb.config
            hyperparams = {
                "ALPHA": config.ALPHA,
                "wandb_upload": True,
            }
            wandb.run.name = f"HER_alpha_{config.ALPHA}_{env.spec.id}"
            print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
            her_agent = HERAgent(state_dim, action_dim, hyperparams=hyperparams, device=device)
            her_agent.train_agent(env=env)
            # her_agent.test_agent(env=env, n_episodes=3)

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
