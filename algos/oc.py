import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
import wandb
import os
from tqdm import tqdm

SEED = 31
np.random.seed(SEED)
# 1. Set seeds for PyTorch (CPU and GPU)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups

# 2. Ensure deterministic behavior for CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Reshape(nn.Module):
    """Utility module to reshape tensors in Sequential models."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class Perception(nn.Module):
    """Perception Network for processing visual inputs."""

    def __init__(self, obs_shape, d, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        if len(obs_shape) == 3:
            channels, height, width = obs_shape
        elif len(obs_shape) == 2:
            height, width = obs_shape
            channels = 1  # Assume single channel for grayscale images
        else:
            raise ValueError("Unsupported observation shape for perception network.")
        # print(f"Observation shape: {obs_shape}")
        percept_linear_input_dim = 32 * int((int((height - 4) / 4) - 2) / 2) * int((int((width - 4) / 4) - 2) / 2)
        # print(f"Perception linear input dimension: {percept_linear_input_dim}")
        self.f_percept = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4, device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, device=self.device),
            nn.ReLU(),
            Reshape((percept_linear_input_dim)),
            nn.Linear(percept_linear_input_dim, d, device=self.device),
            nn.ReLU(),
            Reshape((-1, d)),
        )
        

    def forward(self, x):
        return self.f_percept(x)

# --- Option-Critic Network ---
class OCAgent(nn.Module):
    def __init__(self, state_dim, action_dim, percept=False, device="cpu", hyperparams={}):
        super(OCAgent, self).__init__()
        self.action_dim = action_dim
        self.device = device

        self.LR = hyperparams.get("LR", 1e-3)
        self.GAMMA = hyperparams.get("GAMMA", 0.99)
        self.NUM_OPTIONS = hyperparams.get("NUM_OPTIONS", 4)
        self.ENTROPY_WEIGHT = hyperparams.get("ENT_COEF", 0.01)
        self.EPSILON_START = hyperparams.get("EPSILON_START", 1.0)
        self.EPSILON_MIN = hyperparams.get("EPSILON_MIN", 0.1)
        self.EPSILON_DECAY = hyperparams.get("EPSILON_DECAY", 0.995)
        self.MAX_EPISODES = hyperparams.get("MAX_EPISODES", 1000)
        self.TERM_REG = hyperparams.get("TERM_REG", 0.001)
        
        self.wandb_upload = hyperparams.get("WANDB_UPLOAD", False)
        self.save_model = hyperparams.get("SAVE_MODEL", False)

        self.percept = percept

        if self.percept:
            self.perception = Perception(state_dim, d=128, device=self.device)
            state_dim = 128  # Update state_dim after perception network
            self.features = nn.Sequential(
                self.perception,
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        else:
            # Shared feature extractor
            self.features = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        
        # Q-values for options: Q_Omega(s, omega)
        self.Q_options = nn.Linear(128, self.NUM_OPTIONS)
        
        # Intra-option policies: pi(a | s, omega)
        # Outputs a flattened tensor of size (num_options * action_dim)
        self.policy_options = nn.Linear(128, self.NUM_OPTIONS * action_dim)
        
        # Termination function: beta(s, omega)
        self.terminations = nn.Linear(128, self.NUM_OPTIONS)

        self.optimizer = optim.Adam(self.parameters(), lr=self.LR)

        self.to(self.device)

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")

    def forward(self, state):
        phi = self.features(state)
        
        q_vals = self.Q_options(phi)
        
        # Reshape policy logits to (batch_size, num_options, action_dim)
        policy_logits = self.policy_options(phi).view(-1, self.NUM_OPTIONS, self.action_dim)
        
        # Sigmoid for termination probabilities (between 0 and 1)
        term_probs = torch.sigmoid(self.terminations(phi))
        
        return q_vals, policy_logits, term_probs

    def get_action_and_log_prob(self, state, current_option):
        _, policy_logits, _ = self.forward(state)
        
        # Extract the policy for the currently active option
        logits = policy_logits[0, current_option, :]
        dist = Categorical(logits=logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy

    def train_agent(self, env, store_episode_rewards=False):

        env_name = env.spec.id

        model_path = f"Weights/{env_name}_OC/model_{self.MAX_EPISODES}_nOpt_{self.NUM_OPTIONS}_LR_{self.LR:.4f}_TermReg_{self.TERM_REG:.3f}_x.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="OC-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "algo": "OC",
                    "num_options": self.NUM_OPTIONS,
                    "learning_rate": self.LR,
                    "gamma": self.GAMMA,
                    "entropy_weight": self.ENTROPY_WEIGHT,
                    "epsilon_start": self.EPSILON_START,
                    "epsilon_min": self.EPSILON_MIN,
                    "epsilon_decay": self.EPSILON_DECAY,
                    "max_episodes": self.MAX_EPISODES,
                    "termination_regularization": self.TERM_REG,
                    "model_path": model_path,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params
                },
            )
            run.name = f"{env_name}_nOpt_{self.NUM_OPTIONS}_LR_{self.LR:.4f}_TermReg_{self.TERM_REG:.4f}"

        epsilon = self.EPSILON_START
        reward_history = []
        env_steps = 0

        for episode in tqdm(range(self.MAX_EPISODES)):

            state, _ = env.reset(seed=SEED + episode)  # Ensure reproducibility by varying the seed per episode
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            episode_reward = 0
            episode_len = 0
            curr_op_len = 0
            done = False
            
            # Initial option selection (epsilon-greedy)
            q_vals, _, _ = self.forward(state)
            if np.random.rand() < epsilon:
                current_option = np.random.randint(self.NUM_OPTIONS)
            else:
                current_option = q_vals.argmax(dim=1).item()

            while not done:
                # 1. Select action from the current option's intra-option policy
                action, log_prob, entropy = self.get_action_and_log_prob(state, current_option)
                
                # 2. Step the environment
                next_state_np, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(self.device)
                episode_reward += reward
                episode_len += 1
                env_steps += 1
                curr_op_len += 1

                if self.wandb_upload:
                    wandb.log({"env_steps": env_steps})
                
                # 3. Calculate Option-Critic Targets and Losses
                next_q_vals, _, next_term_probs = self.forward(next_state)
                
                # Did the current option terminate in the next state?
                term_prob_next = next_term_probs[0, current_option]
                
                # Value of the next state: U(s', omega)
                # If it terminates, we can pick the best next option. If not, we stick with the current.
                best_next_q = next_q_vals.max(dim=1)[0]
                current_next_q = next_q_vals[0, current_option]
                
                # Target for Q-value (TD Target)
                if done:
                    td_target = torch.tensor([reward], dtype=torch.float32).to(self.device)
                else:
                    U_next = (1 - term_prob_next) * current_next_q + term_prob_next * best_next_q
                    td_target = reward + self.GAMMA * U_next

                # Critic Loss (MSE)
                q_vals, policy_logits, term_probs = self.forward(state)
                q_val = q_vals[0, current_option]
                critic_loss = (td_target.detach() - q_val) ** 2
                
                # Actor Loss (Policy Gradient with Advantage)
                advantage = td_target.detach() - q_val.detach()
                actor_loss = -log_prob * advantage - (self.ENTROPY_WEIGHT * entropy)
                
                # Termination Loss
                # We want to increase termination probability if the current option's Q-value 
                # is worse than the best available option's Q-value.
                termination_loss = term_probs[0, current_option] * (q_val.detach() - next_q_vals.max(dim=1)[0].detach() + self.TERM_REG)

                # 4. Backpropagation
                total_loss = critic_loss + actor_loss + termination_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # 5. Roll over state and check for option termination
                state = next_state

                if self.wandb_upload:
                    wandb.log({
                        "epsilon": epsilon,
                        "current_option": current_option,
                        "critic_loss": critic_loss.item(),
                        "actor_loss": actor_loss.item(),
                        "termination_loss": termination_loss.item(),
                        "term_prob_next": term_prob_next.item(),
                        "curr_op_len": curr_op_len
                    })
                
                if not done:
                    # Sample termination to decide if we need to switch options
                    term_dist = Bernoulli(probs=term_prob_next.detach())
                    if term_dist.sample().item() == 1.0:
                        # Option terminated, pick a new one (epsilon-greedy)
                        curr_op_len = 0
                        if np.random.rand() < epsilon:
                            current_option = np.random.randint(self.NUM_OPTIONS)
                        else:
                            current_option = next_q_vals.argmax(dim=1).item()

            # Decay epsilon
            epsilon = max(self.EPSILON_MIN, epsilon * self.EPSILON_DECAY)
            if self.wandb_upload:
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_len": episode_len,
                    "epsilon": epsilon
                })

            if store_episode_rewards:
                reward_history.append(episode_reward)
                if self.wandb_upload and len(reward_history) > 10:
                    wandb.log({"average_reward_last_10": np.mean(reward_history[-10:])})

        if self.save_model:  
            torch.save(self.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        if self.wandb_upload:
            wandb.finish()

        env.close()

    def test_agent(self, env, n_episodes=1, record_stats=False):

        env_name = env.spec.id

        # model_path = f"Weights/{env_name}_OC/model_{self.MAX_EPISODES}_nOpt_{self.NUM_OPTIONS}_LR_{self.LR:.4f}_TermReg_{self.TERM_REG:.3f}_x.pt"
        model_path = "Weights/MiniGrid-FourRooms-v0_OC/model_2000_nOpt_4_LR_0.0003_TermReg_0.005_x.pt"
        """ load network weights """
        if self.load_model:
            print(f"Loading model from {model_path}")
            self.load_state_dict(torch.load(model_path))

        if record_stats:
            episode_rewards = []
            episode_lengths = []


        """ play a couple of showcase episodes """
        for episode in range(n_episodes):

            if self.wandb_upload:
                run = wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="OC-Test",
                    group=f"{env_name}-{n_episodes}-Obs-Noise-0.1",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "num_options": self.NUM_OPTIONS,
                        "learning_rate": self.LR,
                        "gamma": self.GAMMA,
                        "entropy_weight": self.ENTROPY_WEIGHT,
                        "epsilon_start": self.EPSILON_START,
                        "epsilon_min": self.EPSILON_MIN,
                        "epsilon_decay": self.EPSILON_DECAY,
                        "max_episodes": self.MAX_EPISODES,
                        "termination_regularization": self.TERM_REG,
                        "model_path": model_path,
                    },
                )
                # run.name = f"{env_name}_{episode + 1}_nOpt_{self.NUM_OPTIONS}_LR_{self.LR:.6f}_TermReg_{self.TERM_REG:.3f}"

            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            episode_reward = 0
            episode_len = 0
            done = False
            renders = []
            
            # Initial option selection (greedy)
            q_vals, _, _ = self.forward(state)
            current_option = q_vals.argmax(dim=1).item()

            while not done:
                # Select action from the current option's intra-option policy
                action, _, _ = self.get_action_and_log_prob(state, current_option)
                
                # Step the environment
                next_state_np, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(self.device)
                episode_reward += reward
                episode_len += 1
                # Check for option termination
                _, _, next_term_probs = self.forward(next_state)
                term_prob_next = next_term_probs[0, current_option]
                
                if not done:
                    term_dist = Bernoulli(probs=term_prob_next.detach())
                    if term_dist.sample().item() == 1.0:
                        # Option terminated, pick a new one (greedy)
                        q_vals_next, _, _ = self.forward(next_state)
                        current_option = q_vals_next.argmax(dim=1).item()

                state = next_state

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
                        "current_option": current_option,
                        "term_prob_next": term_prob_next.item(),
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
                episode_lengths.append(episode_len)

            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Length: {episode_len}")

        env.close()

        if record_stats:
            return episode_rewards, episode_lengths

# def hyperparameter_tuning(env, state_dim, action_dim, percept=False, device="cpu"):
#     # Define a grid of hyperparameters to sweep over
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'average_reward_last_10', 'goal': 'maximize'},
#         'parameters': {
#             'LR': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
#             # 'NUM_OPTIONS': {'values': [2, 3, 4]},
#             'TERM_REG': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 5e-3},
#             'ENTROPY_WEIGHT': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
#             'EPSILON_DECAY': {'distribution': 'uniform', 'min': 0.99, 'max': 0.999},

#         },
#         # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
#     }

#     # Initialize a W&B sweep
#     sweep_id = wandb.sweep(sweep_config, project="OC-HT-LL-bayes", entity="arkapatra31-ust")

#     # Define the training function for the sweep
#     def train_sweep():
#         with wandb.init() as run:
#             config = wandb.config
#             hyperparams = {
#                 "LR": config.LR,
#                 # "NUM_OPTIONS": config.NUM_OPTIONS,
#                 "TERM_REG": config.TERM_REG,
#                 "ENTROPY_WEIGHT": config.ENTROPY_WEIGHT,
#                 "EPSILON_DECAY": config.EPSILON_DECAY,
#                 "wandb_upload": True,
#             }
#             wandb.run.name = f"LR_{hyperparams['LR']:.4f}_TermReg_{hyperparams['TERM_REG']:.4f}_Entropy_{hyperparams['ENTROPY_WEIGHT']:.4f}_EpsilonDecay_{hyperparams['EPSILON_DECAY']:.4f}_{env.spec.id}"
#             print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
#             oc_agent = OCAgent(state_dim, action_dim, percept=percept, device=device, hyperparams=hyperparams)
#             oc_agent.train_agent(env=env, store_episode_rewards=True)
#             # oc_agent.test_agent(env=env, n_episodes=3)

#     # Start the sweep agent to run the training function for each configuration
#     wandb.agent(sweep_id, function=train_sweep, count=20)  # Adjust count as needed for the number of runs

