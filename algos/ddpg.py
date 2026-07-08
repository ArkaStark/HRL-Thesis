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

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

# --- DDPG Agent ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, BUFFER_SIZE=1e6, LR_ACTOR=1e-4, LR_CRITIC=1e-3, device="cpu"):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=1e-2)

        self.max_action = max_action

        self.replay_buffer = ReplayBuffer(int(BUFFER_SIZE))
        self.device = device

    def select_action(self, state, noise_std=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.actor.eval() # Set to evaluation mode
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train() # Set back to train mode
        
        # Add exploration noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action

    def train(self, env, hyperparams=None):

        if hyperparams is None:
            # --- Hyperparameters ---
            LR_ACTOR = 1e-4
            LR_CRITIC = 1e-3
            GAMMA = 0.99            # Discount factor
            TAU = 0.005             # Soft update parameter for target networks
            BATCH_SIZE = 64         # Mini-batch size
            MAX_EPISODES = 2000     # Maximum number of episodes to train
            EXPLORATION_NOISE = 0.1 # Standard deviation of Gaussian noise for exploration
            RANDOM_START_STEPS = 1000    # Number of random steps to populate the replay buffer before training

        wandb_upload = True      # Whether to upload logs to Weights & Biases
        save_weights = True

        env_name = env.spec.id

        if env.action_space.__class__.__name__ == "Discrete":
            print("Discrete action space detected. DDPG is not suitable for discrete action spaces.")
            # return

        actor_weights_path = f"Weights/{env_name}_DDPG_weights/actor_weights_{MAX_EPISODES}.pt"
        critic_weights_path = f"Weights/{env_name}_DDPG_weights/critic_weights_{MAX_EPISODES}.pt"

        os.makedirs(f"Weights/{env_name}_DDPG_weights", exist_ok=True)

        if wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="DDPG-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "actor_lr": LR_ACTOR,
                    "critic_lr": LR_CRITIC,
                    "gamma": GAMMA,
                    "tau": TAU,
                    "batch_size": BATCH_SIZE,
                    "max_episodes": MAX_EPISODES,
                    "exploration_noise": EXPLORATION_NOISE,
                },
            )

            # create a wrapper environment to save episode returns and episode lengths
            # create a wrapper environment to save episode returns and episode lengths
            # envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
            #     envs, buffer_length=n_envs * n_updates
            # )

            critic_losses = []
            actor_losses = []
            entropies = []

        obs, _ = env.reset()
        for i in tqdm(range(RANDOM_START_STEPS)):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            if done:
                obs, _ = env.reset()

        for episode in tqdm(range(MAX_EPISODES)):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action with noise for exploration
                action = self.select_action(obs, noise_std=EXPLORATION_NOISE)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.replay_buffer.add(obs, action, reward, next_obs, float(done))

                obs = next_obs
                episode_reward += reward

                if len(self.replay_buffer) > BATCH_SIZE:
            
                    # Sample a batch of transitions
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
                    states = torch.FloatTensor(np.array(states)).to(self.device)
                    actions = torch.FloatTensor(np.array(actions)).to(self.device)
                    rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
                    dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

                    # ------------------- Optimize Critic ------------------- #
                    # Get predicted next-state actions and Q values from target models
                    with torch.no_grad():
                        next_actions = self.actor_target(next_states)
                        target_Q = self.critic_target(next_states, next_actions)
                        # Compute the target Q value
                        target_Q = rewards + (1 - dones) * GAMMA * target_Q

                    # Get current Q estimate
                    current_Q = self.critic(states, actions)

                    # Compute critic loss (MSE)
                    critic_loss = F.mse_loss(current_Q, target_Q)

                    # Optimize the critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # ------------------- Optimize Actor ------------------- #
                    # Compute actor loss (Negative of the critic's output for the actor's actions)
                    actor_loss = -self.critic(states, self.actor(states)).mean()

                    # Optimize the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ------------------- Soft Updates ------------------- #
                    # Update target networks
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    if wandb_upload:
                        # Log the losses and entropy to wandb
                        wandb.log({
                            "critic_loss": critic_loss.detach().cpu().numpy(),
                            "actor_loss": actor_loss.detach().cpu().numpy()
                        })

            if wandb_upload:
                wandb.log({
                    "episode_reward": episode_reward
                })

        if wandb_upload:
            run.finish()

        """ save network weights """
        if save_weights:
            torch.save(self.actor.state_dict(), actor_weights_path)
            torch.save(self.critic.state_dict(), critic_weights_path)

    def test(self, env, hyperparams=None, n_episodes=1):
        if hyperparams is None:
            # --- Hyperparameters ---
            LR_ACTOR = 1e-4
            LR_CRITIC = 1e-3
            GAMMA = 0.99            # Discount factor
            TAU = 0.005             # Soft update parameter for target networks
            BUFFER_SIZE = int(1e6)  # Replay buffer size
            BATCH_SIZE = 64         # Mini-batch size
            MAX_EPISODES = 2000     # Maximum number of episodes to train
            EXPLORATION_NOISE = 0.1 # Standard deviation of Gaussian noise for exploration
            wandb_upload = True      # Whether to upload logs to Weights & Biases

        env_name = env.spec.id

        actor_weights_path = f"Weights/{env_name}_DDPG_weights/actor_weights_{MAX_EPISODES}.pt"
        critic_weights_path = f"Weights/{env_name}_DDPG_weights/critic_weights_{MAX_EPISODES}.pt"

        if os.path.exists(actor_weights_path) and os.path.exists(critic_weights_path):
            self.actor.load_state_dict(torch.load(actor_weights_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_weights_path, map_location=self.device))
        else:
            print("Weights not found. Please train the agent first.")
            return

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.select_action(obs, noise_std=0.0)  # No exploration noise during testing
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs
            
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")