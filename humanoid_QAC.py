import gymnasium as gym

import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.distributions.normal import Normal
    
class Actor_Critic(nn.Module):
    """Actor-Critic algorithm."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),
            nn.Sigmoid(),  # output action probabilities for each action dimension
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the value estimate and action logits for a given observation.

        Args:
            x: Observation from the environment

        Returns:
            value: Value estimate for the observation
            logits: Action logits for the observation
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        values = self.critic(x)
        logits = self.actor(x)
        return values, logits
    
    def select_action(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects an action by sampling from the policy distribution.

        Args:
            x: Observation from the environment

        Returns:
            action: Selected action
        """
        values, actions = self.forward(x)
        print(f"Action logits: {actions.cpu().detach().numpy()}")
        force = 0.4*(2.0 * actions - 1.0) # scale to [-0.4, 0.4]
        # action_dist = torch.distributions.Categorical(logits=actions)
        # action = action_dist.sample()
        # action_log_prob = action_dist.log_prob(action)
        return force, values
    
    def get_losses(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_log_probs: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the actor and critic losses.

        Args:
            rewards: Rewards obtained from the environment
            values: Value estimates from the critic
            action_log_probs: Log probabilities of the selected actions
            gamma: Discount factor
        Returns:
            actor_loss: Loss for the actor network
            critic_loss: Loss for the critic network
        """
        returns = torch.zeros_like(rewards).to(self.device)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        # advantages = returns - values.squeeze()
        actor_loss = -torch.mean(action_log_probs * returns.detach())
        critic_loss = torch.mean(returns.pow(2))
        return actor_loss, critic_loss
    
    def update(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor) -> None:
        """Performs a single update step for both actor and critic networks.

        Args:
            actor_loss: Loss for the actor network
            critic_loss: Loss for the critic network
        """        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

def main():

    # Hyperparameters
    num_episodes = 1000
    gamma = 0.99
    critic_lr = 0.005 # critic learns faster than actor to provide better value estimates for policy updates
    actor_lr = 0.001

    # Initialise the environment
    env = gym.make("Humanoid-v5")

    # Get dimensions of observation and action spaces
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]

    print(f"Observation space dimensions: {obs_space_dims}")
    print(f"Action space dimensions: {action_space_dims}")

    # Initialize the Actor-Critic agent
    device = "cpu"
    agent = Actor_Critic(
        n_features=obs_space_dims,
        n_actions=action_space_dims,
        device=device,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
    )

    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset(seed=random.randint(0, 1000))
        done = False

        # Initialize lists to store rewards, values, and action log probabilities for the episode
        rewards = []
        values = []
        action_log_probs = []

        while not done:
            action, action_log_prob, value = agent.select_action(obs)
            print(f"Selected action: {action.cpu().numpy()} with log probability {action_log_prob.item()} and value estimate {value.item()}")
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated

            rewards.append(reward)
            values.append(value)
            action_log_probs.append(action_log_prob)

        # Convert lists to tensors
        rewards = torch.tensor(rewards).to(agent.device)
        values = torch.cat(values).to(agent.device)
        action_log_probs = torch.cat(action_log_probs).to(agent.device)

        # Compute losses and update networks
        actor_loss, critic_loss = agent.get_losses(rewards, values, action_log_probs, gamma)
        agent.update(actor_loss, critic_loss)

    env.close()

    # Plotting rewards over episodes
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Actor-Critic on Humanoid-v5")
    plt.show()

    test_env = gym.make("Humanoid-v5", render_mode="human")
    obs, info = test_env.reset(seed=42)
    done = False
    while not done:
        action, _, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())
        done = terminated or truncated
    test_env.close()



if __name__ == "__main__":
    main()