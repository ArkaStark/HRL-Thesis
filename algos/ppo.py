import gymnasium as gym

from tqdm import tqdm
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device, optim
from torch.distributions.normal import Normal

import wandb
import cv2
    
class PPOAgent(nn.Module):
    """Proximal Policy Optimization with Advantage Actor-Critic algorithm."""

    def __init__(
        self, n_features, n_actions, percept=False, device="cpu", hyperparams={}, n_envs=1
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs # for parallel environments

        # hyperparameters
        self.max_env_steps = hyperparams.get("MAX_ENV_STEPS", 1e6)
        self.n_steps_per_update = hyperparams.get("N_STEPS_PER_UPDATE", 400)
        self.ppo_epochs = hyperparams.get("PPO_EPOCHS", 4)
        self.gamma = hyperparams.get("GAMMA", 0.99)
        self.ent_coef = hyperparams.get("ENT_COEF", 0.001)
        self.critic_lr = hyperparams.get("LR_CRITIC", 0.0001)
        self.actor_lr = hyperparams.get("LR_ACTOR", 0.00001)
        self.lam = hyperparams.get("LAM", 0.95)
        self.eps = hyperparams.get("EPS", 1e-6)
        self.SEED = hyperparams.get("SEED", 31)
        self.MAX_ACTION = hyperparams.get("MAX_ACTION", 1.0)

        self.wandb_upload = hyperparams.get("WANDB_UPLOAD", False)
        self.save_model = hyperparams.get("SAVE_MODEL", False)

        self.n_updates = int(self.max_env_steps / (self.n_steps_per_update * self.n_envs))

        self.obs_shape = n_features
        self.action_shape = n_actions

        critic_layers = [
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        self.actor_mean = nn.Linear(128, n_actions).to(self.device)
        self.actor_std = nn.Linear(128, n_actions).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optim = optim.Adam(
            list(self.actor.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_std.parameters()),
            lr=self.actor_lr,
        )

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the value estimate and action logits for a given observation.

        Args:
            x: Observation from the environment

        Returns:
            values: Value estimate for the observation
            action_means: Means of the action distribution
            action_stds: Standard deviations of the action distribution
        """
        x = torch.as_tensor(x, dtype=torch.float32).to(self.device)

        values = self.critic(x).squeeze(-1) 
        actor = self.actor(x) 

        action_means = self.actor_mean(actor)
        action_stds = F.softplus(self.actor_std(actor)).clamp_min(self.eps)  #TODO

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
        pre_tanh_actions = dist.sample()
        actions = self.MAX_ACTION * torch.tanh(pre_tanh_actions)  # clip actions to be within the action space bounds
        log_prob = dist.log_prob(pre_tanh_actions) - torch.log(1 - actions.pow(2) + self.eps)  # adjust log-prob for tanh transformation
        log_prob = log_prob.sum(dim=-1)  # sum log-probs for multi-dimensional action space

        # compute the entropy of the action distribution
        entropy = dist.entropy()
        entropy = entropy.sum(dim=-1)  # sum entropies for multi-dimensional action space

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
        actions = actions / self.MAX_ACTION  # scale actions back to the range of the tanh function
        actions = torch.clamp(actions, -1+self.eps, 1-self.eps)  # ensure actions are within the valid range for atanh
        pre_tanh_actions = torch.atanh(actions)  # inverse of tanh to get the original action values before clipping
        log_prob = dist.log_prob(pre_tanh_actions) - torch.log(1 - actions.pow(2) + self.eps)  # adjust log-prob for tanh transformation
        log_prob = log_prob.sum(dim=-1)  # sum log-probs for multi-dimensional action space
        entropy = dist.entropy()
        entropy = entropy.sum(dim=-1)  # sum entropies for multi-dimensional action space
        return log_prob, values, entropy

    def get_losses(
        self,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        entropy: torch.Tensor,
        ent_coef: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the actor and critic losses.

        Args:
            advantages: Computed advantages
            new_values: New value estimates from the critic
            old_values: Old value estimates from the critic
            new_log_probs: New log probabilities of the selected actions
            old_log_probs: Old log probabilities of the selected actions
            entropy: Entropy of the action distribution
        Returns:
            actor_loss: Loss for the actor network
            critic_loss: Loss for the critic network
        """
        loss_clip = 0.2  # PPO clip parameter to limit the policy update step size
        # calculate the loss of the minibatch for actor and critic
        clipped_value_preds = old_values.detach() + torch.clamp(
            new_values - old_values.detach(), -loss_clip, loss_clip
        )
        critic_loss_unclipped = (returns.detach() - new_values).pow(2) #TODO
        critic_loss_clipped = (returns.detach() - clipped_value_preds).pow(2)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

        # give a bonus for higher entropy to encourage exploration
        # print(f"Advantages: {advantages.cpu().detach().numpy().shape} and action log-probs: {new_log_probs.cpu().detach().numpy().shape}")

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
        actor_loss.backward()  # retain graph to allow backprop for multiple ppo epochs
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_std.parameters()),
            max_norm=0.5,
        )  # clip gradients to prevent exploding gradients
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)  # clip gradients to prevent exploding gradients
        self.critic_optim.step()

    def train_agent(self, env, store_episode_rewards=False):

        env_name = env.spec.id
        model_path = None

        if self.save_model:
            model_path = f"Weights/{env_name}_PPOWeights/{self.n_updates}_{self.actor_lr}_{self.critic_lr}_{self.ent_coef}"
            os.makedirs(model_path, exist_ok=True)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="PPO-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "algo": "PPO",
                    "seed": self.SEED,
                    "state_dim": self.obs_shape,
                    "action_dim": self.action_shape,
                    "n_updates": self.n_updates,
                    "n_steps_per_update": self.n_steps_per_update,
                    "ppo_epochs": self.ppo_epochs,
                    "gamma": self.gamma,
                    "ent_coef": self.ent_coef,
                    "critic_lr": self.critic_lr,
                    "actor_lr": self.actor_lr,
                    "lam": self.lam,
                    "eps": self.eps,
                    "model_path": model_path,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params,
                },
            )
            run.name = f"{env_name}_{self.actor_lr}_{self.critic_lr}_{self.ent_coef}"

        # # create a wrapper environment to save episode returns and episode lengths
        # envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
        #     envs, buffer_length=n_envs * n_updates
        # )

        actor_losses = []
        critic_losses = []
        entropies = []

        ep_rewards = []
        ep_reward = 0
        ep_lengths = []
        ep_length = 0
        env_steps = 0

        for epoch in tqdm(range(self.n_updates)):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            old_value_preds = torch.zeros(self.n_steps_per_update, self.n_envs, device=self.device)
            old_rewards = torch.zeros(self.n_steps_per_update, self.n_envs, device=self.device)
            old_log_probs = torch.zeros(self.n_steps_per_update, self.n_envs, device=self.device)
            old_states = torch.zeros(self.n_steps_per_update, self.n_envs, self.obs_shape, device=self.device)
            old_actions = torch.zeros(self.n_steps_per_update, self.n_envs, self.action_shape, device=self.device)
            masks = torch.zeros(self.n_steps_per_update, self.n_envs, device=self.device)

            # at the start of training reset all envs to get an initial state
            if epoch == 0:
                states, info = env.reset(seed=self.SEED)

            # play n steps in our parallel environments to collect data
            for step in range(self.n_steps_per_update):
                old_states[step] = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                # print(f"States shape: {states.shape}")
                with torch.no_grad():
                    actions, action_log_probs, state_value_preds, _ = self.select_action(states)

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                states, rewards, terminated, truncated, infos = env.step(
                    actions.cpu().numpy()
                )
                done = terminated or truncated
                ep_reward += rewards
                ep_length += 1
                env_steps += self.n_envs

                if self.wandb_upload:
                    wandb.log({"env_steps": env_steps})

                old_value_preds[step] = state_value_preds.squeeze()
                old_rewards[step] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                old_log_probs[step] = action_log_probs.squeeze()
                old_actions[step] = actions

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                if self.n_envs > 1:
                    masks[step] = torch.as_tensor(
                        [not term for term in terminated], dtype=torch.float32, device=self.device
                    )
                else:
                    masks[step] = torch.as_tensor(
                        [not done], dtype=torch.float32, device=self.device
                    )

                if done:
                    env.reset(seed=self.SEED+epoch)
                    if self.wandb_upload:
                        wandb.log({"episode_rewards": ep_reward, #.mean().item(),
                                   "episode_length": ep_length
                        })
                    if store_episode_rewards:
                        ep_rewards.append(ep_reward)
                        ep_lengths.append(ep_length)
                    ep_reward = 0
                    ep_length = 0

            T = self.n_steps_per_update
            advantages = torch.zeros(T, self.n_envs, device=self.device)
            # compute the advantages using GAE
            gae = torch.zeros(self.n_envs, device=self.device)

            with torch.no_grad():
                last_value_preds, _ , _ = self.forward(states)
            for t in reversed(range(T)):
                next_value_preds = last_value_preds if t == T - 1 else old_value_preds[t + 1]
                td_error = (
                    old_rewards[t] + self.gamma * masks[t] * next_value_preds - old_value_preds[t]
                )
                gae = td_error + self.gamma * self.lam * masks[t] * gae
                advantages[t] = gae
            returns = advantages + old_value_preds
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize advantages

            # Compute losses and update networks
            for ppo_update in range(self.ppo_epochs):
                new_log_probs, new_value_preds, entropy = self.evaluate_actions(old_states, old_actions)
                actor_loss, critic_loss = self.get_losses(
                    returns,
                    advantages, 
                    new_value_preds, 
                    old_value_preds, 
                    new_log_probs, 
                    old_log_probs, 
                    entropy, 
                    self.ent_coef
                )
                self.update(actor_loss, critic_loss)
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

                if self.wandb_upload:
                    wandb.log({
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "entropy": entropy.mean().item(),
                    })

            if self.save_model and (epoch + 1) % 100 == 0:
                torch.save(self.state_dict(), f"{model_path}/model.pt")

            if len(ep_rewards)>50:
                mean_reward = np.mean(ep_rewards[-50:])
                if self.wandb_upload:
                    run.log({"mean_episode_return": mean_reward})

        if self.wandb_upload:
            run.finish()

        env.close()


    def test_agent(self, env, n_episodes=1, record_stats=False, model_path=None):
        
        env_name = env.spec.id

        if model_path is None:
            model_path = f"Weights/{env_name}_PPOWeights/{self.n_updates}_{self.actor_lr}_{self.critic_lr}_{self.ent_coef}"
        self.load_state_dict(torch.load(f"{model_path}/model.pt"))

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
                    project="PPO-Test",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "algo": "PPO",
                        "model_path": model_path,
                        "seed": self.SEED,
                        "state_dim": self.obs_shape,
                        "action_dim": self.action_shape,
                    },
                )
                run.name = f"{env_name}_{episode + 1}_PPO"

            # get an initial state
            state, info = env.reset(seed=self.SEED+episode)
            ep_reward = 0
            ep_length = 0

            # play one episode
            done = False
            renders = []
            while not done:
                # select an action A_{t} using S_{t} as input for the agent
                with torch.no_grad():
                    action, _, _, _ = self.select_action(state)

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                state, reward, terminated, truncated, info = env.step(action.cpu().numpy())

                # update if the environment is done
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1

                if self.wandb_upload:
                    frame = env.render()
                    if frame is not None:
                        # W&B handles videos best when frames are uint8 CHW tensors.
                        if frame.shape[-1] == 4:
                            frame = frame[:, :, :3]
                        frame = np.asarray(frame, dtype=np.uint8)
                        renders.append(frame)
                # cv2.imshow("LunarLander", env.render())
                # cv2.waitKey(1)
                if self.wandb_upload:
                    wandb.log({"reward": reward})
                    for a in range(self.action_shape):
                        wandb.log({f"action_{a}": action[a].item()})
            if self.wandb_upload:
                if renders:
                    renders = [
                        cv2.resize(render, (500, 500), interpolation=cv2.INTER_AREA)
                        for render in renders
                    ]
                    video = np.stack(renders, axis=0)
                    video = np.transpose(video, (0, 3, 1, 2))
                    wandb.log({"showcase_video": wandb.Video(video, fps=8, format="mp4")})

            if record_stats:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)

            print(f"Episode {episode + 1}/{n_episodes}: Reward = {ep_reward}, Length = {ep_length}")


            wandb.finish()

        if record_stats:
            return episode_rewards, episode_lengths