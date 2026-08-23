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

# --- TD3 Agent ---
class TD3Agent(torch_nn.Module):
    def __init__(self, state_dim, action_dim, percept=False, device="cpu", hyperparams={}):
        super(TD3Agent, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        
        self.LR_ACTOR = hyperparams.get("LR_ACTOR", 1e-4)
        self.LR_CRITIC = hyperparams.get("LR_CRITIC", 3e-4)
        self.GAMMA = hyperparams.get("GAMMA", 0.99)            # Discount factor
        self.TAU = hyperparams.get("TAU", 0.005)             # Soft update parameter for target networks
        self.BATCH_SIZE = hyperparams.get("BATCH_SIZE", 64)         # Mini-batch size
        # self.MAX_EPISODES = hyperparams.get("MAX_EPISODES", 5000)     # Maximum number of episodes to train
        self.EXPLORATION_NOISE = hyperparams.get("EXPLORATION_NOISE", 0.1) # Standard deviation of Gaussian noise for exploration
        self.RANDOM_START_STEPS = hyperparams.get("RANDOM_START_STEPS", 1000)    # Number of random steps to populate the replay buffer before training
        self.UPDATE_DELAY = hyperparams.get("UPDATE_DELAY", 2)          # Number of critic updates per actor update
        self.BUFFER_SIZE = hyperparams.get("BUFFER_SIZE", int(1e6))  # Replay buffer size
        self.MAX_ACTION = hyperparams.get("MAX_ACTION", 1.0)            # Maximum action value (for BipedalWalker, actions are bounded between -1 and 1)
        self.MAX_ENV_STEPS = hyperparams.get("MAX_ENV_STEPS", 1e6)     # Maximum number of environment steps to train

        self.wandb_upload = hyperparams.get("WANDB_UPLOAD", False)
        self.save_model = hyperparams.get("SAVE_MODEL", False)
            

        self.actor = Actor(state_dim, action_dim, self.MAX_ACTION).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.MAX_ACTION).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.LR_CRITIC, weight_decay=1e-2)

        self.replay_buffer = ReplayBuffer(int(self.BUFFER_SIZE))

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")

    def select_action(self, state, noise_std=0.0, target=False):
        if not target:
            # print("Selecting action using actor network.")
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            self.actor.eval() # Set to evaluation mode
            with torch.no_grad():
                action = self.actor(state).cpu().data.numpy().flatten()
            self.actor.train() # Set back to train mode

            # Add exploration noise
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action + noise, -self.MAX_ACTION, self.MAX_ACTION)
        else:
            # print("Selecting action using target actor network.")
            self.actor_target.eval() # Set to evaluation mode
            with torch.no_grad():
                action = self.actor_target(state)
            self.actor_target.train() # Set back to train mode

            # Add exploration noise
            if noise_std > 0:
                noise = torch.clamp(torch.normal(mean=0, std=2*noise_std, size=action.shape), -5*noise_std, 5*noise_std).to(self.device)
                action = torch.clamp(action + noise, -self.MAX_ACTION, self.MAX_ACTION)
            
        return action

    def train_agent(self, env, store_episode_stats=False):

        env_name = env.spec.id

        if env.action_space.__class__.__name__ == "Discrete":
            print("Discrete action space detected. TD3 is not suitable for discrete action spaces.")
            # return

        model_path = f"Weights/{env_name}_TD3_alr_{self.LR_ACTOR}_clr_{self.LR_CRITIC}_xx"

        os.makedirs(model_path, exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="TD3-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "algo": "TD3",
                    "actor_lr": self.LR_ACTOR,
                    "critic_lr": self.LR_CRITIC,
                    "gamma": self.GAMMA,
                    "tau": self.TAU,
                    "batch_size": self.BATCH_SIZE,
                    "max_env_steps": self.MAX_ENV_STEPS,
                    "exploration_noise": self.EXPLORATION_NOISE,
                    "random_start_steps": self.RANDOM_START_STEPS,
                    "update_delay": self.UPDATE_DELAY,
                    "buffer_size": self.BUFFER_SIZE,
                    "max_action": self.MAX_ACTION,
                    "model_path": model_path,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params,
                },
            )
            run.name = f"{env_name}_alr{self.LR_ACTOR}_clr{self.LR_CRITIC}"

            # create a wrapper environment to save episode returns and episode lengths
            # create a wrapper environment to save episode returns and episode lengths
            # envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
            #     envs, buffer_length=n_envs * n_updates
            # )

            critic_losses = []
            actor_losses = []
            entropies = []

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

        episode_returns = []
        episode_lengths = []

        with tqdm(total=self.MAX_ENV_STEPS, desc="Training Progress") as pbar:

            while env_steps < self.MAX_ENV_STEPS:
                
                obs, info = env.reset(seed=SEED+env_steps)
                episode_reward = 0
                episode_length = 0
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
                    episode_reward += reward
                    episode_length += 1
                    env_steps += 1

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
                            target_actions = self.select_action(next_states, noise_std=self.EXPLORATION_NOISE, target=True)
                            target_Q1 = self.critic1_target(next_states, target_actions)
                            target_Q2 = self.critic2_target(next_states, target_actions)
                            # Compute the target Q value
                            target_Q = rewards + (1 - dones) * self.GAMMA * torch.min(target_Q1, target_Q2)

                        # Get current Q estimate
                        current_Q1 = self.critic1(states, actions)
                        current_Q2 = self.critic2(states, actions)

                        # Compute critic loss (MSE)
                        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                        # Optimize the critic
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()

                        if env_steps % self.UPDATE_DELAY == 0:

                            # ------------------- Optimize Actor ------------------- #
                            # Compute actor loss (Negative of the critic's output for the actor's actions)
                            actor_loss = -self.critic1(states, self.actor(states)).mean()

                            # Optimize the actor
                            self.actor_optimizer.zero_grad()
                            actor_loss.backward()
                            self.actor_optimizer.step()

                            # ------------------- Soft Updates ------------------- #
                            # Update target networks
                            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                            if self.wandb_upload:
                                # Log the losses to wandb
                                wandb.log({
                                    "critic_loss": critic_loss.detach().cpu().numpy(),
                                    "actor_loss": actor_loss.detach().cpu().numpy()
                                })

                        """ save network weights """
                        if self.save_model and env_steps%10000 == 0:
                            torch.save(self.state_dict(), f"{model_path}/model.pt")

                if self.wandb_upload:
                    wandb.log({
                        "episode_reward": episode_reward,
                        "episode_length": episode_length
                    })

                if store_episode_stats:
                    episode_returns.append(episode_reward)
                    episode_lengths.append(episode_length)
                    if len(episode_returns) > 10 and self.wandb_upload:
                        wandb.log(
                            {
                                "mean_episode_return_10": np.mean(episode_returns[-10:])
                            }
                        )

                pbar.update(episode_length)

        if self.wandb_upload:
            run.finish()


    def test_agent(self, env, n_episodes=1, record_stats=False, model_path=None):
 
        env_name = env.spec.id

        if model_path is None:
            model_path = f"Weights/{env_name}_TD3_{self.MAX_EPISODES}"

        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
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
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="TD3-Test",
                    # group=f"{env_name}-{n_episodes}-Act-Delay-5",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "model_path": model_path,
                    },
                )
                run.name = f"TD3_Test_Episode_{episode + 1}"
        
            done = False
            episode_reward = 0
            episode_length = 0
            renders = [] 
            obs, _ = env.reset(seed=SEED + episode)
            
            while not done:

                action = self.select_action(obs, noise_std=0.0)  # No exploration noise during testing
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

            if record_stats:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Length: {episode_length}")

        env.close()
        if record_stats:
            return episode_rewards, episode_lengths


 
def hyperparameter_tuning(env, state_dim, action_dim, percept=False, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'mean_episode_return', 'goal': 'maximize'},
        'parameters': {
            'UPDATE_DELAY': {'values': [2, 10]},
            'TAU': {'values': [0.005, 0.01]},
            'LR_ACTOR': {'values': [5e-5, 1e-4]},
            'LR_CRITIC': {'values': [1e-4, 3e-4, 1e-3]},
            'EXPLORATION_NOISE': {'values': [0.1, 0.2, 0.5]},
        },
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project="TD3-Hyperparameter-Tuning", entity="arkapatra31-ust")

    # Define the training function for the sweep
    def train_sweep():
        with wandb.init() as run:
            config = wandb.config
            hyperparams = {
                "LR_ACTOR": config.LR_ACTOR,
                "LR_CRITIC": config.LR_CRITIC,
                "EXPLORATION_NOISE": config.EXPLORATION_NOISE,
                "TAU": config.TAU,
                "UPDATE_DELAY": config.UPDATE_DELAY,
                "wandb_upload": True,
            }
            wandb.run.name = f"{env.spec.id}_{config.LR_ACTOR:.4f}_{config.LR_CRITIC:.4f}_{config.EXPLORATION_NOISE:.2f}_{config.TAU:.3f}_{config.UPDATE_DELAY}"
            print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
            td3_agent = TD3Agent(state_dim, action_dim, hyperparams=hyperparams, device=device)
            td3_agent.train_agent(env=env, store_episode_stats=True)
            # td3_agent.test_agent(env=env, n_episodes=3)

    # Start the sweep agent to run the training function for each configuration
    wandb.agent(sweep_id, function=train_sweep, count=50)  # Adjust count as needed for the number of runs
