import gymnasium as gym

from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.normal import Normal
    
class PPO(nn.Module):
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

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the value estimate and action logits for a given observation.

        Args:
            x: Observation from the environment

        Returns:
            value: Value estimate for the observation
            logits: Action logits for the observation
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        # print(f"Input shape: {x.shape}")
        values = self.critic(x).squeeze(-1)
        actor = self.actor(x) # shape: [n_envs, n_actions]
        # print(f"Value Shape: {values.shape} and Actor shape: {actor.shape}") 

        action_means = self.actor_mean(actor)
        action_stds = F.softplus(self.actor_std(actor)).clamp_min(self.eps)

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
        dist = Normal(action_means, action_stds)
        pre_tanh_actions = dist.rsample()
        actions = torch.tanh(pre_tanh_actions)  # clip actions to be within the action space bounds

        # print(f"Selected actions: {actions.shape} with dist: {dist}")

        log_prob = dist.log_prob(pre_tanh_actions).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - actions.pow(2) + self.eps).sum(dim=-1)

        # compute the entropy of the action distribution
        entropy = dist.entropy()
        entropy = entropy.sum(dim=-1)  # sum entropies for multi-dimensional action space
        # print(f"Entropy shape: {entropy.shape}")

        return actions, log_prob, values, entropy
    
    def evaluate_actions(self, x: np.ndarray, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the log probabilities and value estimates for given observations and actions.

        Args:
            x: Observations from the environment
            actions: Actions to evaluate
        Returns:
            log_prob: Log probabilities of the given actions
            values: Value estimates for the given observations
            entropy: Entropy of the action distribution (for exploration bonus)
        """
        values, action_means, action_stds = self.forward(x)
        dist = Normal(action_means, action_stds)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        clipped_actions = actions.clamp(-1 + self.eps, 1 - self.eps)
        pre_tanh_actions = torch.atanh(clipped_actions)
        # print(f"Evaluating actions: {actions.shape} with dist: {dist}")
        log_prob = dist.log_prob(pre_tanh_actions).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - clipped_actions.pow(2) + self.eps).sum(dim=-1)

        # compute the entropy of the action distribution
        entropy = dist.entropy()
        entropy = entropy.sum(dim=-1)  # sum entropies for multi-dimensional action space

        return log_prob, values, entropy

    def get_losses(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        entropy: torch.Tensor,
        ent_coef: float = 0.01,

    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the actor and critic losses.

        Args:
            advantages: Advantage estimates
            returns: Target returns for value function
            new_values: Value estimates from the critic for the current policy
            old_values: Value estimates from the critic for the old policy (used for clipping)
            new_log_probs: Log probabilities of the actions under the current policy
            old_log_probs: Log probabilities of the actions under the old policy (used for clipping)
            entropy: Entropy of the action distribution (for exploration bonus)
            ent_coef: Coefficient for the entropy bonus
        Returns:
            actor_loss: Loss for the actor network
            critic_loss: Loss for the critic network
        """
        loss_clip = 0.2 # PPO clip parameter 
        clipped_value_preds = old_values.detach() + torch.clamp(
            new_values - old_values.detach(), -loss_clip, loss_clip
        )
        critic_loss_unclipped = (returns.detach() - new_values).pow(2)
        critic_loss_clipped = (returns.detach() - clipped_value_preds).pow(2)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

        # give a bonus for higher entropy to encourage exploration
        # print(f"Advantages: {advantages.cpu().detach().numpy().shape} and action log-probs: {action_log_probs.cpu().detach().numpy().shape}")

        # compute the ratio of new and old action probabilities (in the first iteration, when we compute the loss for the first time, 
        # the "old" action log-probs are actually the same as the new ones
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        # compute the clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - loss_clip, 1 + loss_clip)
        surrogate_objective = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())
        actor_loss = -surrogate_objective.mean() - ent_coef * entropy.mean()

        return (actor_loss, critic_loss)
    
    def update(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor) -> None:
        """Performs a single update step for both actor and critic networks.

        Args:
            actor_loss: Loss for the actor network
            critic_loss: Loss for the critic network
        """        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_std.parameters()),
            max_norm=0.5,
        )
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()

def main():

    # environment hyperparameters
    n_envs = 10
    n_updates = 1000
    n_steps_per_update = 128
    ppo_epochs = 4
    # randomize_domain = True

    # agent hyperparameters
    gamma = 0.99
    ent_coef = 0.01
    critic_lr = 0.001
    actor_lr = 0.0003
    lam = 0.95  # GAE lambda parameter

    train_agent = True
    save_weights = True
    load_weights = True
    model_path = "invpend_mujoco/invpend_PPO_10env_1000_steps"

    # Initialise the environment
    envs = gym.make_vec("InvertedPendulum-v5", num_envs=n_envs, max_episode_steps=1000)

    # Get dimensions of observation and action spaces
    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.shape[0]
    action_high = envs.single_action_space.high
    action_low = envs.single_action_space.low
    action_scale = ((action_high - action_low) / 2.0).astype(np.float32)
    action_bias = ((action_high + action_low) / 2.0).astype(np.float32)

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

    # Initialize the PPO agent

    agent = PPO(
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


        actor_losses = []
        critic_losses = []
        entropies = []
        for sample_phase in tqdm(range(n_updates)):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            old_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
            old_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
            old_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
            old_states = torch.zeros(n_steps_per_update, n_envs, obs_shape, device=device)
            old_actions = torch.zeros(n_steps_per_update, n_envs, action_shape, device=device)
            masks = torch.zeros(n_steps_per_update, n_envs, device=device)

            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = envs_wrapper.reset(seed=42)

            # play n steps in our parallel environments to collect data
            for step in range(n_steps_per_update):
                # print(f"States shape: {states.shape}")
                current_states = torch.as_tensor(states, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions, action_log_probs, state_value_preds, _ = agent.select_action(states)

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                scaled_actions = actions.cpu().numpy() * action_scale + action_bias
                states, rewards, terminated, truncated, infos = envs_wrapper.step(
                    scaled_actions
                )

                old_value_preds[step] = state_value_preds
                old_rewards[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                old_log_probs[step] = action_log_probs
                old_states[step] = current_states
                old_actions[step] = actions

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it has ended
                dones = np.logical_or(terminated, truncated)
                masks[step] = torch.as_tensor(1.0 - dones, dtype=torch.float32, device=device)

            # Compute the advantages and run ppo epochs to update the policy and value networks
            with torch.no_grad():
                last_values, _, _ = agent.forward(states)
            T = n_steps_per_update
            advantages = torch.zeros(T, n_envs, device=device)
            gae = torch.zeros(n_envs, device=device)
            for t in reversed(range(T)):
                next_values = last_values if t == T - 1 else old_value_preds[t + 1]
                td_error = old_rewards[t] + gamma * masks[t] * next_values - old_value_preds[t]
                gae = td_error + gamma * lam * masks[t] * gae
                advantages[t] = gae
            returns = advantages + old_value_preds
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for ppo_epoch in range(ppo_epochs):
                # print(f"old states shape: {old_states.shape}, old actions shape: {old_actions.shape}")
                new_log_probs, new_value_preds, entropy = agent.evaluate_actions(old_states, old_actions)
                actor_loss, critic_loss = agent.get_losses(
                    advantages,
                    returns,
                    new_value_preds,
                    old_value_preds,
                    new_log_probs,
                    old_log_probs,
                    entropy,
                    ent_coef,
                )
                agent.update(actor_loss, critic_loss)
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())


        if not os.path.exists(model_path):
            os.makedirs(model_path)

        """ save network weights """
        if save_weights:
            torch.save(agent.actor.state_dict(), f"{model_path}/actor_weights.pth")
            torch.save(agent.actor_mean.state_dict(), f"{model_path}/actor_mean_weights.pth")
            torch.save(agent.actor_std.state_dict(), f"{model_path}/actor_std_weights.pth")
            torch.save(agent.critic.state_dict(), f"{model_path}/critic_weights.pth")
            torch.save(actor_losses, f"{model_path}/actor_losses.pt")
            torch.save(critic_losses, f"{model_path}/critic_losses.pt")
            torch.save(entropies, f"{model_path}/entropies.pt")
            # torch.save(episode_returns, f"{model_path}/episode_returns.pt")
            # torch.save(episode_lengths, f"{model_path}/episode_lengths.pt")
            # torch.save(ep_rewards, f"{model_path}/rewards.pt")
            # torch.save(ep_action_log_probs, f"{model_path}/action_log_probs.pt")
            # torch.save(ep_value_preds, f"{model_path}/value_preds.pt")
            
    

        # PLOTTING
        rolling_length = 20
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(
            f"Training plots for {agent.__class__.__name__} in the InvertedPendulum-v5 environment \n \
                    (n_envs={n_envs}, n_steps_per_update={n_steps_per_update})"
        )

        # episode return
        axs[0][0].set_title("Episode Returns")
        episode_returns_moving_average = (
            np.convolve(
                np.array(envs_wrapper.return_queue).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
        )
        axs[0][0].plot(
            np.arange(len(episode_returns_moving_average)) / n_envs,
            episode_returns_moving_average,
        )
        axs[0][0].set_xlabel("Number of episodes")

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
        agent = PPO(obs_shape, action_shape, device, critic_lr, actor_lr)

        agent.actor.load_state_dict(torch.load(f"{model_path}/actor_weights.pth"))
        agent.actor_mean.load_state_dict(torch.load(f"{model_path}/actor_mean_weights.pth"))
        agent.actor_std.load_state_dict(torch.load(f"{model_path}/actor_std_weights.pth"))
        agent.critic.load_state_dict(torch.load(f"{model_path}/critic_weights.pth"))
        agent.actor.eval()
        agent.actor_mean.eval()
        agent.actor_std.eval()
        agent.critic.eval()

    # Test the trained agent

    for i in range(5):
        test_env = gym.make("InvertedPendulum-v5", render_mode="human", max_episode_steps=1000)
        obs, info = test_env.reset(seed=i)
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.select_action(obs)
            scaled_action = action.cpu().numpy() * action_scale + action_bias
            obs, reward, terminated, truncated, info = test_env.step(scaled_action)
            done = terminated or truncated
        test_env.close()



if __name__ == "__main__":
    main()
