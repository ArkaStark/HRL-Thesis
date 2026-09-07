"""Continuous-control PPO Option-Critic (PPOC).

Options own tanh-squashed diagonal-Gaussian policies.  PPO optimises the
intra-option policies from rollouts; option values and terminations use the
Option-Critic continuation target U(s, o).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Normal
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

SEED = 31


class Perception(nn.Module):
    """Compact encoder for channel-first image observations."""

    def __init__(self, obs_shape, output_dim=128):
        super().__init__()
        if len(obs_shape) == 2:
            channels, height, width = 1, *obs_shape
        elif len(obs_shape) == 3:
            channels, height, width = obs_shape
        else:
            raise ValueError("Visual observations need shape (H,W) or (C,H,W).")
        self.add_channel = len(obs_shape) == 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, 8, 4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            n_features = self.conv(torch.zeros(1, channels, height, width)).shape[-1]
        self.head = nn.Sequential(nn.Linear(n_features, output_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1) if self.add_channel and x.ndim == 3 else x))


@dataclass
class Rollout:
    states: list
    next_states: list
    actions: list
    options: list
    log_probs: list
    values: list
    rewards: list
    bootstrap_masks: list
    continuation_masks: list


class PPOCAgent(nn.Module):
    """PPO Option-Critic agent for a single Gymnasium Box-action environment.

    ``N_STEPS_PER_UPDATE`` is the on-policy rollout length.  ``LAM`` is GAE
    lambda, and ``TERM_REG`` discourages premature option termination.
    """

    def __init__(self, state_dim, action_dim, percept=False, device="cpu", hyperparams=None):
        super().__init__()
        h = hyperparams or {}
        if not isinstance(action_dim, (int, np.integer)) or action_dim <= 0:
            raise ValueError("PPOCAgent requires a positive continuous action dimension.")
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.obs_shape = tuple(state_dim) if isinstance(state_dim, (tuple, list)) else (int(state_dim),)
        self.percept = percept

        self.lr = h.get("LR", 3e-4)
        self.gamma = h.get("GAMMA", 0.99)
        self.gae_lambda = h.get("LAM", 0.95)
        self.num_options = int(h.get("NUM_OPTIONS", 4))
        self.max_env_steps = int(h.get("MAX_ENV_STEPS", 1_000_000))
        self.n_steps_per_update = int(h.get("N_STEPS_PER_UPDATE", 2_048))
        self.ppo_epochs = int(h.get("PPO_EPOCHS", 10))
        self.minibatch_size = int(h.get("MINIBATCH_SIZE", 64))
        self.clip_coef = h.get("CLIP_COEF", 0.2)
        self.value_clip_coef = h.get("VALUE_CLIP_COEF", self.clip_coef)
        self.vf_coef = h.get("VF_COEF", 0.5)
        self.ent_coef = h.get("ENT_COEF", 0.0)
        self.term_coef = h.get("TERM_COEF", 1.0)
        self.term_reg = h.get("TERM_REG", 1e-2)
        self.max_grad_norm = h.get("MAX_GRAD_NORM", 0.5)
        self.log_std_min = h.get("LOG_STD_MIN", -5.0)
        self.log_std_max = h.get("LOG_STD_MAX", 2.0)
        self.eps = h.get("EPS", 1e-6)
        self.wandb_upload = h.get("WANDB_UPLOAD", False)
        self.save_model = h.get("SAVE_MODEL", False)
        if self.wandb_upload and wandb is None:
            raise ImportError("WANDB_UPLOAD=True requires wandb.")

        if percept:
            self.encoder, feature_dim = Perception(self.obs_shape), 128
        else:
            if len(self.obs_shape) != 1:
                raise ValueError("Use percept=True for non-vector observations.")
            self.encoder, feature_dim = nn.Identity(), self.obs_shape[0]
        self.features = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.q_options = nn.Linear(128, self.num_options)
        self.policy_means = nn.Linear(128, self.num_options * self.action_dim)
        self.policy_log_stds = nn.Parameter(torch.zeros(self.num_options, self.action_dim))
        self.terminations = nn.Linear(128, self.num_options)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Kept as buffers so model/device moves preserve the Box transformation.
        self.register_buffer("action_low", -torch.ones(self.action_dim))
        self.register_buffer("action_high", torch.ones(self.action_dim))
        self.to(self.device)

    def _state_tensor(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0) if tensor.ndim == len(self.obs_shape) else tensor

    def _configure_action_space(self, env):
        space = env.action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("PPOCAgent only supports gymnasium.spaces.Box action spaces.")
        if space.shape != (self.action_dim,):
            raise ValueError(f"Expected Box shape ({self.action_dim},), received {space.shape}.")
        if not (np.isfinite(space.low).all() and np.isfinite(space.high).all() and np.all(space.high > space.low)):
            raise ValueError("PPOCAgent requires finite Box bounds where high > low.")
        self.action_low.copy_(torch.as_tensor(space.low, dtype=torch.float32, device=self.device))
        self.action_high.copy_(torch.as_tensor(space.high, dtype=torch.float32, device=self.device))

    def _to_env_action(self, normalized_action):
        action = self.action_low + 0.5 * (normalized_action + 1) * (self.action_high - self.action_low)
        return action.clamp(self.action_low, self.action_high).detach().cpu().numpy()

    def forward(self, state):
        """Return Q(s,o), option means/log-stds, and beta(s,o)."""
        state = self._state_tensor(state)
        phi = self.features(self.encoder(state))
        q_values = self.q_options(phi)
        means = self.policy_means(phi).view(-1, self.num_options, self.action_dim)
        log_stds = self.policy_log_stds.clamp(self.log_std_min, self.log_std_max).unsqueeze(0).expand_as(means)
        return q_values, means, log_stds, torch.sigmoid(self.terminations(phi))

    def _distribution(self, state, options):
        _, means, log_stds, _ = self.forward(state)
        rows = torch.arange(means.shape[0], device=self.device)
        return Normal(means[rows, options], log_stds.exp()[rows, options])

    def _log_prob_entropy(self, states, actions, options):
        options = torch.as_tensor(options, dtype=torch.long, device=self.device).reshape(-1)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.clamp(-1 + self.eps, 1 - self.eps)
        dist = self._distribution(states, options)
        pre_tanh = torch.atanh(actions)
        log_prob = (dist.log_prob(pre_tanh) - torch.log(1 - actions.square() + self.eps)).sum(-1)
        return log_prob, dist.entropy().sum(-1)

    @torch.no_grad()
    def select_action(self, state, option, deterministic=False):
        """Return a normalized action, log probability, and Q(s, option)."""
        option_tensor = torch.tensor([option], dtype=torch.long, device=self.device)
        q_values, means, _, _ = self.forward(state)
        dist = self._distribution(state, option_tensor)
        pre_tanh = means[0, option] if deterministic else dist.sample()
        action = torch.tanh(pre_tanh)
        log_prob = (dist.log_prob(pre_tanh) - torch.log(1 - action.square() + self.eps)).sum(-1)
        return action.squeeze(0), log_prob.squeeze(0), q_values[0, option]

    @torch.no_grad()
    def get_action_and_log_prob(self, state, current_option, deterministic=False):
        """Compatibility helper: returns an action already scaled to the Box space."""
        action, log_prob, value = self.select_action(state, current_option, deterministic)
        return self._to_env_action(action), log_prob, value

    @torch.no_grad()
    def _sample_termination(self, next_state, option, bootstrap_mask):
        """Sample beta(s',o); terminal states never continue an option."""
        if not bootstrap_mask:
            return 0.0
        beta = self.forward(next_state)[3][0, option]
        return float(Bernoulli(beta).sample().item() == 0.0)

    def _make_batch(self, rollout):
        states, next_states = torch.cat(rollout.states), torch.cat(rollout.next_states)
        actions = torch.stack(rollout.actions)
        options = torch.tensor(rollout.options, dtype=torch.long, device=self.device)
        old_log_probs, old_values = torch.stack(rollout.log_probs), torch.stack(rollout.values)
        rewards = torch.tensor(rollout.rewards, dtype=torch.float32, device=self.device)
        boot = torch.tensor(rollout.bootstrap_masks, dtype=torch.float32, device=self.device)
        continue_mask = torch.tensor(rollout.continuation_masks, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_q, _, _, next_beta = self.forward(next_states)
            q_same = next_q.gather(1, options[:, None]).squeeze(1)
            q_best = next_q.max(dim=1).values
            beta = next_beta.gather(1, options[:, None]).squeeze(1)
            u_next = (1 - beta) * q_same + beta * q_best
            deltas = rewards + self.gamma * boot * u_next - old_values
            advantages, gae = torch.zeros_like(deltas), torch.zeros((), device=self.device)
            for t in reversed(range(len(deltas))):
                # After beta samples termination, U already bootstraps into the next option.
                gae = deltas[t] + self.gamma * self.gae_lambda * continue_mask[t] * gae
                advantages[t] = gae
            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + self.eps)
        return states, next_states, actions, options, old_log_probs, old_values, returns, advantages, boot

    def _update(self, batch):
        states, next_states, actions, options, old_log_probs, old_values, returns, advantages, boot = batch
        metrics = {key: [] for key in ("policy_loss", "value_loss", "termination_loss", "entropy", "approx_kl")}
        for _ in range(self.ppo_epochs):
            for ids in torch.randperm(len(options), device=self.device).split(self.minibatch_size):
                log_probs, entropy = self._log_prob_entropy(states[ids], actions[ids], options[ids])
                ratio = (log_probs - old_log_probs[ids]).exp()
                policy_loss = -torch.minimum(
                    ratio * advantages[ids], ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef) * advantages[ids]
                ).mean()

                q = self.forward(states[ids])[0].gather(1, options[ids, None]).squeeze(1)
                clipped_q = old_values[ids] + (q - old_values[ids]).clamp(-self.value_clip_coef, self.value_clip_coef)
                value_loss = 0.5 * torch.maximum((returns[ids] - q).square(), (returns[ids] - clipped_q).square()).mean()

                # Detaching option values limits this objective's gradient to beta.
                next_q, _, _, next_beta = self.forward(next_states[ids])
                same_q = next_q.gather(1, options[ids, None]).squeeze(1).detach()
                best_q = next_q.max(dim=1).values.detach()
                beta = next_beta.gather(1, options[ids, None]).squeeze(1)
                termination_loss = (beta * (same_q - best_q + self.term_reg) * boot[ids]).mean()

                loss = policy_loss + self.vf_coef * value_loss + self.term_coef * termination_loss - self.ent_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                for name, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                                    ("termination_loss", termination_loss), ("entropy", entropy.mean()),
                                    ("approx_kl", (old_log_probs[ids] - log_probs).mean())):
                    metrics[name].append(value.item())
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def train_agent(self, env, store_episode_rewards=False):
        self._configure_action_space(env)
        env_name = env.spec.id if env.spec else type(env.unwrapped).__name__
        model_path = os.path.join("Weights", f"{env_name}_PPOC", "model.pt")
        if self.save_model:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        run = wandb.init(project="PPOC-Train", config={"env": env_name, "algo": "PPOC"}) if self.wandb_upload else None
        state, _ = env.reset(seed=SEED)
        current_option, episode_reward, episode_length = None, 0.0, 0
        reward_history, env_steps, updates = [], 0, 0
        progress = tqdm(total=self.max_env_steps, desc="PPOC training")
        try:
            while env_steps < self.max_env_steps:
                rollout = Rollout([], [], [], [], [], [], [], [], [])
                for _ in range(min(self.n_steps_per_update, self.max_env_steps - env_steps)):
                    state_tensor = self._state_tensor(state)
                    with torch.no_grad():
                        if current_option is None:
                            current_option = self.forward(state_tensor)[0].argmax(1).item()
                        action, log_prob, value = self.select_action(state_tensor, current_option)
                    next_state, reward, terminated, truncated, _ = env.step(self._to_env_action(action))
                    done, boot = bool(terminated or truncated), float(not terminated)
                    next_tensor = self._state_tensor(next_state)
                    with torch.no_grad():
                        continue_option = self._sample_termination(next_tensor, current_option, boot)
                    rollout.states.append(state_tensor); rollout.next_states.append(next_tensor)
                    rollout.actions.append(action); rollout.options.append(current_option)
                    rollout.log_probs.append(log_prob); rollout.values.append(value)
                    rollout.rewards.append(float(reward)); rollout.bootstrap_masks.append(boot)
                    rollout.continuation_masks.append(continue_option)
                    state, episode_reward, episode_length = next_state, episode_reward + float(reward), episode_length + 1
                    env_steps += 1; progress.update(1)
                    if done:
                        reward_history.append(episode_reward)
                        if self.wandb_upload:
                            wandb.log({"episode_reward": episode_reward, "episode_length": episode_length, "env_steps": env_steps})
                        state, _ = env.reset()
                        current_option, episode_reward, episode_length = None, 0.0, 0
                    elif not continue_option:
                        with torch.no_grad():
                            current_option = self.forward(next_tensor)[0].argmax(1).item()
                metrics = self._update(self._make_batch(rollout)); updates += 1
                if self.wandb_upload:
                    metrics.update({"env_steps": env_steps, "updates": updates, "mean_episode_return": float(np.mean(reward_history[-10:])) if reward_history else 0.0})
                    wandb.log(metrics)
                if self.save_model:
                    torch.save(self.state_dict(), model_path)
        finally:
            progress.close()
            if run is not None:
                run.finish()
            env.close()
        return reward_history if store_episode_rewards else None

    @torch.no_grad()
    def test_agent(self, env, n_episodes=1, record_stats=False, model_path: Optional[str] = None):
        self._configure_action_space(env)
        if model_path:
            checkpoint = model_path if model_path.endswith(".pt") else os.path.join(model_path, "model.pt")
            self.load_state_dict(torch.load(checkpoint, map_location=self.device))
        rewards, lengths = [], []
        for episode in range(n_episodes):
            state, _ = env.reset(seed=SEED + episode)
            option, total_reward, length, done = None, 0.0, 0, False
            while not done:
                state_tensor = self._state_tensor(state)
                if option is None:
                    option = self.forward(state_tensor)[0].argmax(1).item()
                action, _, _ = self.select_action(state_tensor, option, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(self._to_env_action(action))
                done, total_reward, length = bool(terminated or truncated), total_reward + float(reward), length + 1
                if not done and not self._sample_termination(self._state_tensor(next_state), option, 1.0):
                    option = self.forward(self._state_tensor(next_state))[0].argmax(1).item()
                state = next_state
            rewards.append(total_reward); lengths.append(length)
            print(f"Episode {episode + 1}/{n_episodes}: Reward = {total_reward:.2f}, Length = {length}")
        env.close()
        return (rewards, lengths) if record_stats else None

    @torch.no_grad()
    def get_values(self, state):
        return self.forward(state)[0].squeeze(0).cpu().numpy()


# Allows older external imports to keep working while benchmark.py uses PPOCAgent.
OCAgent = PPOCAgent
