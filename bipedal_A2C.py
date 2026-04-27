import gymnasium as gym

import random
from tqdm import tqdm
import os

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
        n_envs: int = 1,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.eps = 1e-6

        critic_layers = [
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        self.actor_mean = nn.Linear(32, n_actions).to(self.device)
        self.actor_std = nn.Linear(32, n_actions).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(
            list(self.actor.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_std.parameters()),
            lr=actor_lr,
        )

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the value estimate and action logits for a given observation.

        Args:
            x: Observation from the environment

        Returns:
            value: Value estimate for the observation
            logits: Action logits for the observation
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        # print(f"Input shape: {x.shape}")
        values = self.critic(x) # shape: [n_envs,]
        actor = self.actor(x) # shape: [n_envs, n_actions]
        # print(f"Value Shape: {values.shape} and Actor shape: {actor.shape}") 

        action_means = self.actor_mean(actor)
        action_stds = torch.log(
            1 + torch.exp(self.actor_std(actor))
        )

        # print(f"Action means shape: {action_means.shape} and action stds shape: {action_stds.shape}") 


        
        return values, action_means, action_stds
    
    def select_action(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects an action by sampling from the policy distribution.

        Args:
            x: Observation from the environment

        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            values: Value estimate for the observation
            entropy: Entropy of the action distribution (for exploration bonus)
        """
        values, action_means, action_stds = self.forward(x)
         # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        # print(f"Action means: {action_means.cpu().detach().numpy()} and stddevs: {action_stds.cpu().detach().numpy()}")
        dist = Normal(action_means + self.eps, action_stds + self.eps)
        actions = dist.sample()
        actions = torch.tanh(actions)  # clip actions to be within the action space bounds
        log_prob = dist.log_prob(actions) - torch.log(1 - actions.pow(2) + self.eps)  # adjust log-prob for tanh transformation
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # sum log-probs for multi-dimensional action space

        # compute the entropy of the action distribution
        entropy = dist.entropy()
        entropy = entropy.sum(dim=-1, keepdim=True)  # sum entropies for multi-dimensional action space

        return actions, log_prob, values, entropy

    def get_losses(
        self,
        rewards: torch.Tensor,
        value_preds: torch.Tensor,
        action_log_probs: torch.Tensor,
        entropy: torch.Tensor,
        gamma: float,
        lam: float,
        device: torch.device = torch.device,
        ent_coef: float = 0.01,
        masks: torch.Tensor = torch.Tensor,
        n_envs: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the actor and critic losses.

        Args:
            rewards: Rewards obtained from the environment
            values: Value estimates from the critic
            action_log_probs: Log probabilities of the selected actions
            entropy: Entropy of the action distribution
            gamma: Discount factor
        Returns:
            actor_loss: Loss for the actor network
            critic_loss: Loss for the critic network
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        # print(f"Advantages: {advantages.cpu().detach().numpy().shape} and action log-probs: {action_log_probs.cpu().detach().numpy().shape}")
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)
    
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

    # environment hyperparameters
    n_envs = 10
    n_updates = 10000
    n_steps_per_update = 64
    # randomize_domain = True

    # agent hyperparameters
    gamma = 0.99
    ent_coef = 0.01  # entropy coefficient for exploration bonus
    critic_lr = 0.005 # critic learns faster than actor to provide better value estimates for policy updates
    actor_lr = 0.001
    lam = 0.95  # GAE lambda parameter

    train_agent = True
    save_weights = True
    load_weights = True
    model_path = "bipedalwalker_weights/bipedal_A2C_10env_10000steps"

    # Initialise the environment
    envs = gym.make_vec("BipedalWalker-v3", hardcore=False, num_envs=n_envs, max_episode_steps=1600)

    # Get dimensions of observation and action spaces
    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.shape[0]

    print(f"Observation space dimensions: {obs_shape}")
    print(f"Action space dimensions: {action_shape}")
    
    # set the device
    use_cuda = True
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        else:
            print("CUDA is not available. Using CPU instead.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Initialize the Actor-Critic agent

    agent = Actor_Critic(
        n_features=obs_shape,
        n_actions=action_shape,
        device=device,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        n_envs=n_envs,
    )

    #train_agent = False --- IGNORE ---
    if train_agent:

        # create a wrapper environment to save episode returns and episode lengths
        envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
            envs, buffer_length=n_envs * n_updates
        )

        episode_lengths = []
        episode_returns = []
        actor_losses = []
        critic_losses = []
        entropies = []

        for sample_phase in tqdm(range(n_updates)):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
            ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
            ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
            masks = torch.zeros(n_steps_per_update, n_envs, device=device)

            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = envs_wrapper.reset(seed=42)

            # play n steps in our parallel environments to collect data
            for step in range(n_steps_per_update):
                # print(f"States shape: {states.shape}")
                actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                    states
                )

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                states, rewards, terminated, truncated, infos = envs_wrapper.step(
                    actions.cpu().numpy()
                )

                ep_value_preds[step] = state_value_preds.squeeze()
                ep_rewards[step] = torch.tensor(rewards, device=device)
                ep_action_log_probs[step] = action_log_probs.squeeze()

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor([not term for term in terminated])



            # Convert lists to tensors
            # print(f"ep_rewards.shape: {ep_rewards.shape}") => [n_steps_per_update, n_envs]
            episode_returns.append(sum(ep_rewards.cpu().numpy()))
            episode_lengths.append(len(ep_rewards.cpu().numpy()))

            # Compute losses and update networks
            actor_loss, critic_loss = agent.get_losses(ep_rewards, ep_value_preds, ep_action_log_probs, entropy, gamma, lam, device, ent_coef, masks, n_envs)
            agent.update(actor_loss, critic_loss)
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.mean().item())

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        """ save network weights """
        if save_weights:
            torch.save(agent.actor.state_dict(), f"{model_path}/actor_weights.pth")
            torch.save(agent.critic.state_dict(), f"{model_path}/critic_weights.pth")
            torch.save(actor_losses, f"{model_path}/actor_losses.pt")
            torch.save(critic_losses, f"{model_path}/critic_losses.pt")
            torch.save(entropies, f"{model_path}/entropies.pt")
            torch.save(episode_returns, f"{model_path}/episode_returns.pt")
            torch.save(episode_lengths, f"{model_path}/episode_lengths.pt")
            torch.save(ep_rewards, f"{model_path}/rewards.pt")
            torch.save(ep_action_log_probs, f"{model_path}/action_log_probs.pt")
            torch.save(ep_value_preds, f"{model_path}/value_preds.pt")
            
    

        # PLOTTING
        rolling_length = 20
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(
            f"Training plots for {agent.__class__.__name__} in the BipedalWalker-v3 environment \n \
                    (n_envs={n_envs}, n_steps_per_update={n_steps_per_update})"
        )

        # episode return
        axs[0][0].set_title("Episode Returns")
        avg_episode_returns = np.mean(np.array(episode_returns), axis=1)
        # print(f"Average episode returns shape: {avg_episode_returns.shape}")
        episode_returns_moving_average = (
            np.convolve(
                avg_episode_returns.flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
        )
        # print(f"Episode returns shape: {np.array(episode_returns).shape} and moving average shape: {episode_returns_moving_average.shape}")
        axs[0][0].plot(episode_returns_moving_average)
        axs[0][0].set_xlabel("Number of updates")

        # entropy
        axs[1][0].set_title("Entropy")
        entropy_moving_average = (
            np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
            / rolling_length
        )
        axs[1][0].plot(entropy_moving_average)
        axs[1][0].set_xlabel("Number of updates")


        # critic loss
        axs[0][1].set_title("Critic Loss")
        critic_losses_moving_average = (
            np.convolve(
                np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
        )
        axs[0][1].plot(critic_losses_moving_average)
        axs[0][1].set_xlabel("Number of updates")


        # actor loss
        axs[1][1].set_title("Actor Loss")
        actor_losses_moving_average = (
            np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
            / rolling_length
        )
        axs[1][1].plot(actor_losses_moving_average)
        axs[1][1].set_xlabel("Number of updates")

        plt.tight_layout()
        plt.show()


    """ load network weights """
    if load_weights:
        agent = Actor_Critic(obs_shape, action_shape, device, critic_lr, actor_lr)

        agent.actor.load_state_dict(torch.load(f"{model_path}/actor_weights.pth"))
        agent.critic.load_state_dict(torch.load(f"{model_path}/critic_weights.pth"))
        agent.actor.eval()
        agent.critic.eval()

    # Test the trained agent

    for i in range(5):
        test_env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human", max_episode_steps=500)
        obs, info = test_env.reset(seed=i)
        done = False
        while not done:
            action, _, _, _ = agent.select_action(obs)
            # action = test_env.action_space.sample()
            obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())
            done = terminated or truncated
        test_env.close()



if __name__ == "__main__":
    main()
