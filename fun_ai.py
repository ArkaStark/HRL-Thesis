"""
Feudal Network (FuN) agent for Hierarchical Reinforcement Learning on Atari Pong.

This script implements a FuN agent (manager + worker) using PyTorch and OpenAI Gym
for training on the Atari PongNoFrameskip-v4 environment. It is a single-file
implementation including environment wrappers, networks, training loop, and
utility functions.

Note: This is an educational implementation and omits many engineering
optimizations used in production research code (e.g., vectorized envs,
distributed training, extensive hyperparameter tuning).
"""
from __future__ import annotations

import collections
import random
from typing import Deque, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ale_py  # Ensure ALE is registered with Gym
gym.register_envs(ale_py)  # Register ALE environments with Gym

# --------- Utilities and wrappers ---------


class AtariPreprocessing(gym.Wrapper):
    """Minimal Atari preprocessing: grayscale, resize to 84x84, frame skip.

    This is a simplified single-environment wrapper for brevity.
    """

    def __init__(self, env, noop_max=30, frame_skip=4, screen_size=84):
        super().__init__(env)
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = screen_size

    def reset(self):
        obs = self.env.reset()
        return self._process(obs)

    def step(self, action):
        total_reward = 0.0
        done = False
        last_obs = None
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            last_obs = obs
            if done:
                break
        return self._process(last_obs), total_reward, done, info

    def _process(self, obs):
        # Convert to grayscale and resize using numpy (fast but simple)
        import cv2

        img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.uint8)
        return img


def make_env(env_id: str = "PongNoFrameskip-v4"):
    env = gym.make(env_id)
    # env = AtariPreprocessing(env)
    return env


# --------- Feudal Networks models ---------


def weights_init(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.zero_()


class Perception(nn.Module):
    """Shared perception module that maps frames to embeddings."""

    def __init__(self, in_channels=1, embedding_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(7 * 7 * 64, embedding_dim), nn.ReLU())
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape (B, H, W) or (B,1,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.fc(self.conv(x / 255.0))


class Manager(nn.Module):
    """Manager (high-level): outputs goals in latent space and intrinsic reward.

    The manager is trained with an auxiliary predictive objective.
    """

    def __init__(self, embedding_dim=256, goal_dim=64, hidden=256):
        super().__init__()
        self.rnn = nn.GRUCell(embedding_dim, hidden)
        self.goal_proj = nn.Linear(hidden, goal_dim)
        self.apply(weights_init)

    def forward(self, embed: torch.Tensor, hx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hx = self.rnn(embed, hx)
        goal = self.goal_proj(hx)
        return goal, hx


class Worker(nn.Module):
    """Worker (low-level): acts conditioned on goals.

    The worker learns a policy pi(a|s, g) and a value function.
    """

    def __init__(self, embedding_dim=256, goal_dim=64, n_actions=6, hidden=256):
        super().__init__()
        self.fc = nn.Linear(embedding_dim + goal_dim, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        self.apply(weights_init)

    def forward(self, embed: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([embed, goal], dim=-1)
        x = F.relu(self.fc(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


# --------- Agent and training logic ---------


class FuNAgent:
    """Feudal Network agent encapsulating manager, worker, optimizers and training.

    Basic training loop uses actor-critic updates for worker and supervised/predictive
    objective for manager. Intrinsic reward is computed as alignment between
    state-change embedding and manager goal.
    """

    def __init__(self, obs_shape=(84, 84), n_actions=6, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.n_actions = n_actions
        # networks
        self.perception = Perception(in_channels=1, embedding_dim=256).to(self.device)
        self.manager = Manager(embedding_dim=256, goal_dim=64, hidden=256).to(self.device)
        self.worker = Worker(embedding_dim=256, goal_dim=64, n_actions=n_actions, hidden=256).to(self.device)

        # optimizers
        params = list(self.perception.parameters()) + list(self.manager.parameters()) + list(self.worker.parameters())
        self.optimizer = optim.RMSprop(params, lr=7e-4, eps=1e-5)

        # training hyperparams
        self.gamma = 0.99
        self.intrinsic_scale = 0.5
        self.manager_goal_k = 10  # manager sets goal every k steps

    def select_action(self, obs: np.ndarray, manager_hx: torch.Tensor, goal: torch.Tensor):
        obs_t = torch.tensor(obs, dtype=torch.uint8, device=self.device).unsqueeze(0)
        embed = self.perception(obs_t.float())
        # manager outputs next goal only when requested by caller, pass-headed goal
        logits, value = self.worker(embed, goal)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        return action, value.item(), embed.squeeze(0).detach(), m.log_prob(torch.tensor(action, device=self.device)).detach()

    def compute_intrinsic(self, prev_embed: torch.Tensor, cur_embed: torch.Tensor, goal: torch.Tensor) -> float:
        # Cosine similarity between state-change vector and goal as intrinsic reward
        delta = (cur_embed - prev_embed)
        goal = goal.squeeze(0)
        num = torch.dot(delta, goal)
        den = (torch.norm(delta) * torch.norm(goal)).clamp(min=1e-6)
        sim = (num / den).item()
        return sim

    def update(self, batch: dict):
        # Batch expects lists/arrays of transitions collected in one rollout
        obs = torch.tensor(np.array(batch["obs"]), dtype=torch.uint8, device=self.device).float()
        actions = torch.tensor(batch["actions"], device=self.device)
        rewards = torch.tensor(batch["rewards"], device=self.device)
        intrinsic = torch.tensor(batch["intrinsic"], device=self.device)
        old_values = torch.tensor(batch["values"], device=self.device)
        log_probs = torch.stack(batch["log_probs"]).to(self.device)

        # Compute embeddings
        embeds = self.perception(obs)

        # For simplicity use zero goals for updates (manager trained with aux loss)
        goals = torch.zeros((len(embeds), 64), device=self.device)

        # Worker forward to get values for bootstrap and policy
        logits, values = self.worker(embeds, goals)
        probs = F.softmax(logits, dim=-1)

        # Compute returns (extrinsic + intrinsic)
        total_rewards = rewards + self.intrinsic_scale * intrinsic
        returns = torch.zeros_like(total_rewards)
        running = 0.0
        for t in reversed(range(len(total_rewards))):
            running = total_rewards[t] + self.gamma * running
            returns[t] = running

        advantages = returns - values.detach()

        # Policy loss
        new_log_probs = torch.log_softmax(logits, dim=-1)
        new_lp = new_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(advantages.detach() * new_lp).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Manager auxiliary loss: predict future embedding direction
        # simple predictive loss: next-embedding cosine distance with goal vector from manager
        manager_loss = torch.tensor(0.0, device=self.device)
        if len(embeds) > self.manager_goal_k:
            pred_hx = torch.zeros((len(embeds), 256), device=self.device)
            # compute manager goals for each time step
            # Use recurrent rollout (deterministic) for simplicity
            hx = torch.zeros(1, 256, device=self.device)
            mgr_goals = []
            for e in embeds:
                g, hx = self.manager(e.unsqueeze(0), hx.squeeze(0))
                mgr_goals.append(g.squeeze(0))
            mgr_goals = torch.stack(mgr_goals)
            # Predict delta after k steps and compare alignment
            deltas = embeds[self.manager_goal_k:] - embeds[:-self.manager_goal_k]
            goals_slice = mgr_goals[:-self.manager_goal_k]
            # Cosine loss (1 - cosine similarity)
            dnorm = F.normalize(deltas, dim=-1)
            gnorm = F.normalize(goals_slice, dim=-1)
            cos = (dnorm * gnorm).sum(dim=-1)
            manager_loss = (1.0 - cos).mean()

        total_loss = policy_loss + 0.5 * value_loss + manager_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(self.perception.parameters()) + list(self.manager.parameters()) + list(self.worker.parameters()), 40.0)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "manager_loss": manager_loss.item() if isinstance(manager_loss, torch.Tensor) else float(manager_loss),
        }


def main():
    env = make_env("LunarLander-v3")
    n_actions = env.action_space.n
    agent = FuNAgent(n_actions=n_actions)

    max_episodes = 1000
    update_interval = 64
    for ep in range(max_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        # manager state
        manager_hx = torch.zeros(1, 256, device=agent.device)
        # initialize goal
        goal = torch.zeros(1, 64, device=agent.device)

        batch = {"obs": [], "actions": [], "rewards": [], "values": [], "log_probs": [], "intrinsic": []}

        prev_embed = None
        step = 0
        while not done:
            # manager updates goal every k steps
            obs_proc = obs
            action, value, cur_embed, logp = agent.select_action(obs_proc, manager_hx, goal)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if prev_embed is None:
                prev_embed = cur_embed

            # manager sets new goal periodically
            if step % agent.manager_goal_k == 0:
                with torch.no_grad():
                    g, manager_hx = agent.manager(cur_embed.unsqueeze(0), manager_hx.squeeze(0))
                    goal = g.unsqueeze(0)

            intrinsic = agent.compute_intrinsic(prev_embed, cur_embed, goal)

            batch["obs"].append(obs_proc)
            batch["actions"].append(action)
            batch["rewards"].append(float(reward))
            batch["values"].append(float(value))
            batch["log_probs"].append(logp)
            batch["intrinsic"].append(intrinsic)

            prev_embed = cur_embed
            obs = next_obs
            step += 1

            if len(batch["obs"]) >= update_interval or done:
                stats = agent.update(batch)
                batch = {"obs": [], "actions": [], "rewards": [], "values": [], "log_probs": [], "intrinsic": []}

        print(f"Episode {ep} reward={episode_reward:.1f}")


if __name__ == "__main__":
    main()
