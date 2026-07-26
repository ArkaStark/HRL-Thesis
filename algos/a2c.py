import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import wandb
import cv2

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

class A2CAgent(nn.Module):

    def __init__(self, n_features, n_actions, hyperparams={}, device="cpu"):
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()

        self.n_updates = hyperparams.get("n_updates", 2000)
        self.n_steps_per_update = hyperparams.get("n_steps_per_update", 100)
        self.randomize_domain = hyperparams.get("randomize_domain", False)

        # agent hyperparams
        self.gamma = hyperparams.get("gamma", 0.99)
        self.lam = hyperparams.get("lam", 0.95)  # hyperparameter for GAE
        self.ent_coef = hyperparams.get("ent_coef", 0.01)  # coefficient for the entropy bonus (to encourage exploration)
        self.actor_lr = hyperparams.get("actor_lr", 0.003)
        self.critic_lr = hyperparams.get("critic_lr", 0.003)

        self.wandb_upload = hyperparams.get("wandb_upload", False)
        self.save_model = hyperparams.get("save_model", False)
        self.load_model = hyperparams.get("load_model", True)

        # Note: the actor has a slower learning rate so that the value targets become
        # more stationary and are theirfore easier to estimate for the critic
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
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr)

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        self.total_params = sum(p.numel() for p in self.parameters())   

        unique_params = {p.data_ptr(): p for p in self.parameters()}.values()
        self.total_unique_params = sum(p.numel() for p in unique_params)   

        print(f"Trainable parameters: {self.trainable_params}, Total parameters: {self.total_params}, Total unique parameters: {self.total_unique_params}")

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:

        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x) 
        action_logits_vec = self.actor(x)  
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        state_values, action_logits = self.forward(x)
        # print(f"Action logits: {action_logits.cpu().detach().numpy()}")
        # print(f"State values: {state_values.cpu().detach().numpy()}")
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        # print(f"Action probabilities: {action_pd}")
        actions = action_pd.sample()
        # print(f"Selected actions: {actions.cpu().detach().numpy()}")
        action_log_probs = action_pd.log_prob(actions)
        # print(f"Action log-probs: {action_log_probs.cpu().detach().numpy()}")
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float
    ) -> tuple[torch.Tensor, torch.Tensor]:

        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)

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
            -(advantages.detach() * action_log_probs).mean() - self.ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def train_agent(self, env, store_episode_stats=False):
        """Sets the actor and critic networks to training mode."""
        self.actor.train()
        self.critic.train()


        env_name = env.spec.id

        if env.action_space.__class__.__name__ == "Box":
            print("Continuous action space detected. A2C is not suitable for continuous action spaces.")
            # return

        actor_weights_path = f"Weights/{env_name}_A2C_weights/actor_weights_{self.n_updates}_alr{self.actor_lr}_ent{self.ent_coef}.pt"
        critic_weights_path = f"Weights/{env_name}_A2C_weights/critic_weights_{self.n_updates}_clr{self.critic_lr}_ent{self.ent_coef}.pt"

        os.makedirs(f"Weights/{env_name}_A2C_weights", exist_ok=True)

        """ save network weights """
        if self.save_model:
            torch.save(self.actor.state_dict(), actor_weights_path)
            torch.save(self.critic.state_dict(), critic_weights_path)

        if self.wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="A2C-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "architecture": "A2C",
                    "n_updates": self.n_updates,
                    "n_steps_per_update": self.n_steps_per_update,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "ent_coef": self.ent_coef,
                    "actor_lr": self.actor_lr,
                    "critic_lr": self.critic_lr,
                    "actor_weights_path": actor_weights_path,
                    "critic_weights_path": critic_weights_path,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "total_unique_params": self.total_unique_params
                },
            )
            run.name = f"{env_name}_alr-{self.actor_lr:.4f}_clr-{self.critic_lr:.4f}_entcf-{self.ent_coef:.3f}"

        # critic_losses = []
        # actor_losses = []
        # entropies = []
        ep_step = 0
        episode_returns = []
        episode_lengths = []
        episode_return = 0
        episode_length = 0

        # use tqdm to get a progress bar for training
        for sample_phase in tqdm(range(self.n_updates)):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(self.n_steps_per_update, device=self.device)
            ep_rewards = torch.zeros(self.n_steps_per_update, device=self.device)
            ep_action_log_probs = torch.zeros(self.n_steps_per_update, device=self.device)
            ep_entropies = torch.zeros(self.n_steps_per_update, device=self.device)
            masks = torch.zeros(self.n_steps_per_update, device=self.device)

            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = env.reset(seed=42)

            # play n steps in our parallel environments to collect data
            for step in range(self.n_steps_per_update):

                # select an action A_{t} using S_{t} as input for the agent
                actions, action_log_probs, state_value_preds, entropy = self.select_action(
                    states
                )

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                states, rewards, terminated, truncated, infos = env.step(
                    actions.cpu().numpy()
                )

                ep_value_preds[step] = torch.squeeze(state_value_preds)
                ep_rewards[step] = torch.tensor(rewards, device=self.device)
                ep_action_log_probs[step] = action_log_probs
                ep_entropies[step] = entropy


                episode_return += rewards
                episode_length += 1
                done = terminated or truncated
                ep_step += 1

                if self.wandb_upload:
                    wandb.log(
                        {
                            "ep_step": ep_step
                        }
                    )



                 # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor([not done], device=self.device, dtype=torch.float32)

                if done:
                    if self.wandb_upload:
                        wandb.log(
                            {
                                "episode_return": episode_return,
                                "episode_length": episode_length,
                            }
                        )
                    if store_episode_stats:
                        episode_returns.append(episode_return)
                        episode_lengths.append(episode_length)
                        if len(episode_returns) > 10 and self.wandb_upload:
                            wandb.log(
                                {
                                    "mean_episode_return": np.mean(episode_returns[-10:])
                                }
                            )
                    states, info = env.reset()
                    episode_return = 0
                    episode_length = 0



            # calculate the losses for actor and critic
            critic_loss, actor_loss = self.get_losses(
                ep_rewards,
                ep_action_log_probs,
                ep_value_preds,
                ep_entropies,
                masks,
                self.gamma,
                self.lam,
                self.ent_coef
            )

            # update the actor and critic networks
            self.update(critic_loss, actor_loss)

            if self.wandb_upload:
                # Log the losses and entropy to wandb
                wandb.log(
                    {
                        "critic_loss": critic_loss.detach().cpu().numpy(),
                        "actor_loss": actor_loss.detach().cpu().numpy(),
                        "entropy": ep_entropies.detach().mean().cpu().numpy(),
                    }
                )

            # log the losses and entropy
            # critic_losses.append(critic_loss.detach().cpu().numpy())
            # actor_losses.append(actor_loss.detach().cpu().numpy())
            # entropies.append(ep_entropies.detach().mean().cpu().numpy())

        if self.wandb_upload:
            run.finish()

    def test_agent(self, env, n_episodes=1):

        env_name = env.spec.id

        actor_weights_path = f"Weights/{env_name}_A2C_weights/actor_weights_{self.n_updates}_alr{self.actor_lr}_ent{self.ent_coef}.pt"
        critic_weights_path = f"Weights/{env_name}_A2C_weights/critic_weights_{self.n_updates}_clr{self.critic_lr}_ent{self.ent_coef}.pt"

        """ load network weights """
        if self.load_model:
            # agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

            self.actor.load_state_dict(torch.load(actor_weights_path))
            self.critic.load_state_dict(torch.load(critic_weights_path))
            self.actor.eval()
            self.critic.eval()

        """ play a couple of showcase episodes """

        for episode in range(n_episodes):
            print(f"starting episode {episode}...")

            if self.wandb_upload:
                run = wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="A2C-Test",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "architecture": "A2C",
                        "actor_lr": self.actor_lr,
                        "critic_lr": self.critic_lr,
                        "n_updates": self.n_updates,
                        "n_steps_per_update": self.n_steps_per_update,
                        "gamma": self.gamma,
                        "lam": self.lam,
                        "ent_coef": self.ent_coef,
                    },
                )

            # get an initial state
            state, info = env.reset()

            # play one episode
            done = False
            renders = []
            while not done:
                # select an action A_{t} using S_{t} as input for the agent
                with torch.no_grad():
                    action, _, _, _ = self.select_action(state)

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                state, reward, terminated, truncated, info = env.step(action.item())

                # update if the environment is done
                done = terminated or truncated

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
                    wandb.log({"reward": reward,
                               "action": action.item(),
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

        env.close()

def hyperparameter_tuning(env, state_dim, action_dim, percept=False, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'mean_episode_return', 'goal': 'maximize'},
        'parameters': {
            'LR_ACTOR': {'values': [1e-3, 3e-3, 5e-3]},
            'LR_CRITIC': {'values': [1e-3, 3e-3, 5e-3]},
            'ent_coef': {'values': [0.01, 0.1]},
        },
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project="A2C-Hyperparameter-Tuning", entity="arkapatra31-ust")

    # Define the training function for the sweep
    def train_sweep():
        with wandb.init() as run:
            config = wandb.config
            hyperparams = {
                "LR_ACTOR": config.LR_ACTOR,
                "LR_CRITIC": config.LR_CRITIC,
                "ent_coef": config.ent_coef,
                "wandb_upload": True,
            }
            wandb.run.name = f"{env.spec.id}_alr-{config.LR_ACTOR:.4f}_clr-{config.LR_CRITIC:.4f}_entcf-{config.ent_coef:.3f}"
            print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
            a2c_agent = A2CAgent(state_dim, action_dim, hyperparams=hyperparams, device=device)
            a2c_agent.train_agent(env=env, store_episode_stats=True)
            # a2c_agent.test_agent(env=env, n_episodes=3)

    # Start the sweep agent to run the training function for each configuration
    wandb.agent(sweep_id, function=train_sweep, count=10)  # Adjust count as needed for the number of runs

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