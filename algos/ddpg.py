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

SEED = 31
np.random.seed(SEED)
# 1. Set seeds for PyTorch (CPU and GPU)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups

# 2. Ensure deterministic behavior for CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
class DDPGAgent(torch_nn.Module):
    def __init__(self, state_dim, action_dim, hyperparams={}, device="cpu"):
        super(DDPGAgent, self).__init__()

        self.LR_ACTOR = hyperparams.get("LR_ACTOR", 1e-4)
        self.LR_CRITIC = hyperparams.get("LR_CRITIC", 3e-4)
        self.GAMMA = hyperparams.get("GAMMA", 0.99)
        self.TAU = hyperparams.get("TAU", 0.005)
        self.BATCH_SIZE = hyperparams.get("BATCH_SIZE", 64)
        self.MAX_EPISODES = hyperparams.get("MAX_EPISODES", 2000)
        self.EXPLORATION_NOISE = hyperparams.get("EXPLORATION_NOISE", 0.1)
        self.RANDOM_START_STEPS = hyperparams.get("RANDOM_START_STEPS", 1000)
        self.MAX_ACTION = hyperparams.get("MAX_ACTION", 1.0)
        self.BUFFER_SIZE = hyperparams.get("BUFFER_SIZE", 1000000)

        self.wandb_upload = hyperparams.get("WANDB_UPLOAD", False)
        self.save_weights = hyperparams.get("SAVE_WEIGHTS", False)
        self.load_weights = hyperparams.get("LOAD_WEIGHTS", True)

        self.actor = Actor(state_dim, action_dim, self.MAX_ACTION).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.MAX_ACTION).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
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

    def train_agent(self, env):

        env_name = env.spec.id

        if env.action_space.__class__.__name__ == "Discrete":
            print("Discrete action space detected. DDPG is not suitable for discrete action spaces.")
            # return

        actor_weights_path = f"Weights/{env_name}_DDPG_weights/actor_weights_{self.MAX_EPISODES}.pt"
        critic_weights_path = f"Weights/{env_name}_DDPG_weights/critic_weights_{self.MAX_EPISODES}.pt"

        os.makedirs(f"Weights/{env_name}_DDPG_weights", exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="DDPG-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "algo": "DDPG",
                    "actor_lr": self.LR_ACTOR,
                    "critic_lr": self.LR_CRITIC,
                    "gamma": self.GAMMA,
                    "tau": self.TAU,
                    "batch_size": self.BATCH_SIZE,
                    "max_episodes": self.MAX_EPISODES,
                    "exploration_noise": self.EXPLORATION_NOISE,
                    "actor_weights_path": actor_weights_path,
                    "critic_weights_path": critic_weights_path,
                    "buffer_size": self.BUFFER_SIZE,
                    "random_start_steps": self.RANDOM_START_STEPS,
                    "max_action": self.MAX_ACTION,
                    "num_updates": self.NUM_UPDATES,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params
                },
            )

        env_steps = 0

        obs, _ = env.reset(seed=SEED)
        for i in tqdm(range(self.RANDOM_START_STEPS)):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            env_steps += 1
            if done:
                obs, _ = env.reset(seed=SEED)

        for episode in tqdm(range(self.MAX_EPISODES)):
            obs, info = env.reset(seed=SEED)
            episode_reward = 0
            done = False
            
            while not done:
                # Select action with noise for exploration
                action = self.select_action(obs, noise_std=self.EXPLORATION_NOISE)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.replay_buffer.add(obs, action, reward, next_obs, float(done))

                obs = next_obs
                env_steps += 1
                episode_reward += reward

                if self.wandb_upload:
                    wandb.log({"env_steps": env_steps})

                if len(self.replay_buffer) > self.BATCH_SIZE:
            
                    # Sample a batch of transitions
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.BATCH_SIZE)
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
                        target_Q = rewards + (1 - dones) * self.GAMMA * target_Q

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
                        target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                    if self.wandb_upload:
                        # Log the losses and entropy to wandb
                        wandb.log({
                            "critic_loss": critic_loss.detach().cpu().numpy(),
                            "actor_loss": actor_loss.detach().cpu().numpy()
                        })

            if self.wandb_upload:
                wandb.log({
                    "episode_reward": episode_reward
                })

        if self.wandb_upload:
            run.finish()

        """ save network weights """
        if self.save_weights:
            torch.save(self.actor.state_dict(), actor_weights_path)
            torch.save(self.critic.state_dict(), critic_weights_path)

    def test_agent(self, env, n_episodes=1, delay_steps=0, delay_type=None, noise_std=0.0, noise_type=None):

        env_name = env.spec.id

        actor_weights_path = f"Weights/{env_name}_DDPG_weights/actor_weights_{self.MAX_EPISODES}.pt"
        critic_weights_path = f"Weights/{env_name}_DDPG_weights/critic_weights_{self.MAX_EPISODES}.pt"

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
                    project="DDPG-Test",
                    group=f"{env_name}-{n_episodes}-Obs-Delay-10",
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
                        "delay_steps": delay_steps,
                        "delay_type": delay_type,
                        "noise_std": noise_std,
                        "noise_type": noise_type,
                        "actor_weights_path": actor_weights_path,
                        "critic_weights_path": critic_weights_path,
                    },
                )
                run.name = f"DDPG_Test_Episode_{episode + 1}"
        
            obs, info = env.reset()
            if hasattr(env, 'delay_buffer'):
                del env.delay_buffer
            if hasattr(env, 'action_buffer'):
                del env.action_buffer
            done = False
            episode_reward = 0
            renders = [] 
            
            while not done:

                action = self.select_action(obs, noise_std=0.0)  # No exploration noise during testing
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
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
            
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

