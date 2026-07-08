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
    """
    (Synchronous) Advantage Actor-Critic agent class
    On-Policy, Model-Free, Policy Gradient RL algorithm, Discrete Action Space

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

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
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
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
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=self.device)

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

    def train(self, envs, hyperparams=None):
        """Sets the actor and critic networks to training mode."""
        self.actor.train()
        self.critic.train()

        if hyperparams==None:
            # environment hyperparams
            n_envs = 1
            n_updates = 100
            n_steps_per_update = 128
            randomize_domain = False

            # agent hyperparams
            gamma = 0.999
            lam = 0.95  # hyperparameter for GAE
            ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
            actor_lr = 0.001
            critic_lr = 0.005

            # Note: the actor has a slower learning rate so that the value targets become
            # more stationary and are theirfore easier to estimate for the critic

        env_name = envs.spec.id
        # environment setup
        # if randomize_domain:
        #     envs = gym.vector.AsyncVectorEnv(
        #         [
        #             lambda: gym.make(
        #                 env_name,
        #                 gravity=np.clip(
        #                     np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
        #                 ),
        #                 enable_wind=np.random.choice([True, False]),
        #                 wind_power=np.clip(
        #                     np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
        #                 ),
        #                 turbulence_power=np.clip(
        #                     np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
        #                 ),
        #                 max_episode_steps=600,
        #             )
        #             for i in range(n_envs)
        #         ]
        #     )

        # else:
        #     envs = gym.make_vec(env_name, num_envs=n_envs, max_episode_steps=600)

        # set the device
        # use_cuda = True
        # if use_cuda:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #     device = torch.device("cpu")

        # init the agent
        # agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

        # TRAINING LOOP
        train_agent = True
        save_weights = True
        load_weights = True
        wandb_upload = False

        if envs.action_space.__class__.__name__ == "Box":
            print("Continuous action space detected. A2C is not suitable for continuous action spaces.")
            # return

        actor_weights_path = f"Weights/{env_name}_A2C_weights/actor_weights_{n_updates}.h5"
        critic_weights_path = f"Weights/{env_name}_A2C_weights/critic_weights_{n_updates}.h5"

        os.makedirs(f"Weights/{env_name}_A2C_weights", exist_ok=True)

        """ save network weights """
        if train_agent and save_weights:
            torch.save(self.actor.state_dict(), actor_weights_path)
            torch.save(self.critic.state_dict(), critic_weights_path)

        if train_agent:
            if wandb_upload:
                # Start a new wandb run to track this script.
                run = wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="A2C-Train",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "n_envs": n_envs,
                        "n_updates": n_updates,
                        "n_steps_per_update": n_steps_per_update,
                        "gamma": gamma,
                        "lam": lam,
                        "ent_coef": ent_coef,
                        "actor_lr": actor_lr,
                        "critic_lr": critic_lr,
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

            # use tqdm to get a progress bar for training
            for sample_phase in tqdm(range(n_updates)):
                # we don't have to reset the envs, they just continue playing
                # until the episode is over and then reset automatically

                # reset lists that collect experiences of an episode (sample phase)
                ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=self.device)
                ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=self.device)
                ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=self.device)
                ep_entropies = torch.zeros(n_steps_per_update, n_envs, device=self.device)
                masks = torch.zeros(n_steps_per_update, n_envs, device=self.device)

                # at the start of training reset all envs to get an initial state
                if sample_phase == 0:
                    states, info = envs.reset(seed=42)

                # play n steps in our parallel environments to collect data
                for step in range(n_steps_per_update):
                    # select an action A_{t} using S_{t} as input for the agent
                    actions, action_log_probs, state_value_preds, entropy = self.select_action(
                        states
                    )

                    # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                    states, rewards, terminated, truncated, infos = envs.step(
                        actions.cpu().numpy()
                    )

                    ep_value_preds[step] = torch.squeeze(state_value_preds)
                    ep_rewards[step] = torch.tensor(rewards, device=self.device)
                    ep_action_log_probs[step] = action_log_probs
                    ep_entropies[step] = entropy

                    # add a mask (for the return calculation later);
                    # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                    if n_envs == 1:
                        masks[step] = torch.tensor([not terminated])
                    else:
                        masks[step] = torch.tensor([not term for term in terminated])

                # calculate the losses for actor and critic
                critic_loss, actor_loss = self.get_losses(
                    ep_rewards,
                    ep_action_log_probs,
                    ep_value_preds,
                    ep_entropies,
                    masks,
                    gamma,
                    lam,
                    ent_coef
                )

                # update the actor and critic networks
                self.update(critic_loss, actor_loss)

                if wandb_upload:
                    # Log the losses and entropy to wandb
                    wandb.log(
                        {
                            "critic_loss": critic_loss.detach().cpu().numpy(),
                            "actor_loss": actor_loss.detach().cpu().numpy(),
                            "entropy": ep_entropies.detach().mean().cpu().numpy(),
                            "episode_return": envs.return_queue[-1],
                        }
                    )

                # log the losses and entropy
                critic_losses.append(critic_loss.detach().cpu().numpy())
                actor_losses.append(actor_loss.detach().cpu().numpy())
                entropies.append(ep_entropies.detach().mean().cpu().numpy())

            if wandb_upload:
                run.finish()

    def test(self, env, hyperparams=None, n_episodes=1):

        if hyperparams==None:
            # environment hyperparams
            n_envs = 1
            n_updates = 100
            n_steps_per_update = 128
            randomize_domain = False

            # agent hyperparams
            gamma = 0.999
            lam = 0.95  # hyperparameter for GAE
            ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
            actor_lr = 0.001
            critic_lr = 0.005

            # Note: the actor has a slower learning rate so that the value targets become
            # more stationary and are theirfore easier to estimate for the critic

        load_weights = True
        wandb_upload = False

        env_name = env.spec.id

        actor_weights_path = f"Weights/{env_name}_A2C_weights/actor_weights_{n_updates}.h5"
        critic_weights_path = f"Weights/{env_name}_A2C_weights/critic_weights_{n_updates}.h5"

        """ load network weights """
        if load_weights:
            # agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

            self.actor.load_state_dict(torch.load(actor_weights_path))
            self.critic.load_state_dict(torch.load(critic_weights_path))
            self.actor.eval()
            self.critic.eval()

        """ play a couple of showcase episodes """

        for episode in range(n_episodes):
            print(f"starting episode {episode}...")

            if wandb_upload:
                run = wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="A2C-Test",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "architecture": "A2C",
                        "actor_lr": actor_lr,
                        "critic_lr": critic_lr,
                        "n_updates": n_updates,
                        "n_steps_per_update": n_steps_per_update,
                        "gamma": gamma,
                        "lam": lam,
                        "ent_coef": ent_coef,
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
                if wandb_upload:
                    wandb.log({"reward": reward,
                               "action": action.item(),
                            })
            if wandb_upload:
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
