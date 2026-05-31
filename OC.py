import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import wandb

gym.register_envs(ale_py)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, option, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, option, reward, next_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, options, rewards, next_obs, dones = zip(*[self.buffer[idx] for idx in batch])
        return obs, options, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)

class OptCrit(nn.Module):
    def __init__(self, 
                 in_features, 
                 n_actions,
                 n_options,
                 percept=True,
                 lr = 1e-3,
                 device = torch.device('cpu')):
        super(OptCrit, self).__init__()
        
        self.in_features = in_features
        self.n_actions = n_actions
        self.n_options = n_options
        self.device = device
        self.percept = percept


        if percept:
            self.output_dim = 512
            channels, height, width = in_features

            percept_out_dim = 32 * int((int((height - 4) / 4) - 2) / 2) * int((int((width - 4) / 4) - 2) / 2)
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(percept_out_dim, 512),
                nn.ReLU()
            )
        else:
            self.output_dim = 64
            self.features = nn.Sequential(
                nn.Linear(in_features, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            )

        self.critic = nn.Linear(self.output_dim, n_options)
        self.terminator = nn.Linear(self.output_dim, n_options)
        self.options_W = nn.Parameter(torch.zeros(n_options, self.output_dim, n_actions))
        self.options_b = nn.Parameter(torch.zeros(n_options, n_actions))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(device)

    def get_state(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        squeeze_output = False

        if self.percept and obs.dim() == 3:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        elif (not self.percept) and obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True

        state = self.features(obs)

        if squeeze_output:
            state = state.squeeze(0)

        return state

    def get_Q(self, state):
        return self.critic(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminator(state)[current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_termination(self, state):
        return self.terminator(state).sigmoid()
    
    def get_action(self, state, option):
        logits = torch.matmul(state, self.options_W[option]) + self.options_b[option]

        action_dist = Categorical(logits=logits.softmax(dim=-1))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # dist = Normal(logits, torch.ones_like(logits) * 0.1)
        # pre_tanh_actions = dist.sample()
        # actions = torch.tanh(pre_tanh_actions)  # clip actions to be within the action space bounds
        # log_prob = dist.log_prob(pre_tanh_actions) - torch.log(1 - actions.pow(2) + 0.01)  # adjust log-prob for tanh transformation
        # log_prob = log_prob.sum(dim=-1)  # sum log-probs for multi-dimensional action space
        # # compute the entropy of the action distribution
        # entropy = dist.entropy()
        # entropy = entropy.sum(dim=-1)  # sum entropies for multi-dimensional action space

        return action.item(), log_prob, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
    
def critic_loss(model, model_prime, obs, options, rewards, next_obs, dones, gamma=0.99):

    obs = torch.tensor(np.array(obs), dtype=torch.float32).to(model.device)
    options = torch.tensor(options, dtype=torch.long).to(model.device)
    batch_idx = torch.arange(len(options)).to(model.device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(model.device)
    masks = 1 - torch.tensor(dones, dtype=torch.float32).to(model.device)
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(model.device)

    # print(f"Critic loss - obs shape: {obs.shape}, options shape: {options.shape}, rewards shape: {rewards.shape}, next_obs shape: {next_obs.shape}")
    states = model.get_state(obs)
    Q = model.get_Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(next_obs)
    next_Q_prime = model_prime.get_Q(next_states_prime).detach()

    # Additionally, we need the beta probabilities of the next state
    next_states = model.get_state(next_obs)
    next_termination_probs = model.get_termination(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]
    next_Q_option = next_Q_prime[batch_idx, options]
    next_Q_max = next_Q_prime.max(dim=-1)[0]

    # Now we can calculate the update target gt
    gt = rewards + masks * gamma * \
        ((1 - next_options_term_prob) * next_Q_option \
         + next_options_term_prob  * next_Q_max)
    
    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err

def actor_loss(model, model_prime, obs, option, logp, entropy, reward, done, next_obs, gamma=0.99, term_reg=0.01, ent_reg=0.01):

    obs = torch.tensor(np.array(obs), dtype=torch.float32).to(model.device)
    option = torch.tensor(option, dtype=torch.long).to(model.device)
    logp = logp.to(model.device) if torch.is_tensor(logp) else torch.tensor(logp, dtype=torch.float32, device=model.device)
    entropy = entropy.to(model.device) if torch.is_tensor(entropy) else torch.tensor(entropy, dtype=torch.float32, device=model.device)
    reward = torch.tensor(reward, dtype=torch.float32).to(model.device)
    done = torch.tensor(done, dtype=torch.float32).to(model.device)
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(model.device)

    state = model.get_state(obs)
    next_state = model.get_state(next_obs)
    next_state_prime = model_prime.get_state(next_obs)

    # print(f"opt_term_prob shape: {model.get_termination(state).shape}, option: {option}, next_option_term_prob shape: {model.get_termination(next_state).shape}")
    option_term_prob = model.get_termination(state)[option]
    next_option_term_prob = model.get_termination(next_state)[option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + \
         next_option_term_prob * next_Q_prime.max(dim=-1)[0])
    
    # Termination loss
    term_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + term_reg) * (1 - done)

    # Actor-Critic Policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - ent_reg * entropy
    return policy_loss + term_loss
    
    
def main():
    
    gamma = 0.99
    term_reg = 0.01
    ent_reg = 0.01
    lr = 5e-4
    eps = 0.1
    
    n_options = 3
    max_steps = 1e5
    batch_size = 64
    update_freq = 4
    freeze_interval = 200

    train = True
    save_model = True
    load_model = False
    model_path = "Weights/Cartpole-v1/OC_1e5_steps_3_options"
    wandb_upload = True

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    else:        device = torch.device("cpu")

    if train:
        if wandb_upload:
        # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="HRL-OC",
                # Track hyperparameters and run metadata.
                config={
                    "env": "LunarLander-v3",
                    "max_steps": max_steps,
                    "n_options": n_options,
                    "batch_size": batch_size,
                    "update_freq": update_freq,
                    "freeze_interval": freeze_interval,
                    "learning_rate": lr,
                    "gamma": gamma,
                    "term_reg": term_reg,
                    "ent_reg": ent_reg,
                },
            )
    
        env = gym.make('CartPole-v1')
        # env = gym.make('ALE/Pong-v5', obs_type='grayscale')
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.ResizeObservation(env, (42, 42))
        # env = gym.wrappers.ResizeObservation(env, (21, 21))
        # env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
        # env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

        agent = OptCrit(state_dim, action_dim, n_options, device=device, percept=False)
        agent_prime = OptCrit(state_dim, action_dim, n_options, device=device, percept=False)
        agent_prime.load_state_dict(agent.state_dict())

        replay_buffer = ReplayBuffer(capacity=10000)

        steps = 0
        losses = []
        actor_losses = []
        critic_losses = []
        ep_rewards = []
        ep_lengths = []
        ep_option_lengths = []
        while steps < max_steps:

            rewards = 0 
            option_lengths = {opt:[] for opt in range(n_options)}

            obs, _ = env.reset()
            state = agent.get_state(obs)
            greedy_option = agent.greedy_option(state)
            current_option = 0

            done = False
            curr_op_len = 0
            op_term = True


            while not done:

                if op_term:
                    option_lengths[current_option].append(curr_op_len)
                    curr_op_len = 0
                    current_option = np.random.choice(n_options) if np.random.rand() < eps  else greedy_option

                action, logp, entropy = agent.get_action(state, current_option)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                replay_buffer.push(obs, current_option, reward, next_obs, done)
                rewards += reward




                loss_a, loss_c, loss = None, None, None
                if len(replay_buffer) > batch_size:
                    loss_a = actor_loss(
                        agent,
                        agent_prime,
                        obs,
                        current_option,
                        logp,
                        entropy,
                        reward,
                        done,
                        next_obs,
                        gamma,
                        term_reg,
                        ent_reg,
                    )
                    loss = loss_a

                    if steps % update_freq == 0:
                        batch_obs, batch_options, batch_rewards, batch_next_obs, batch_dones = replay_buffer.sample(batch_size)
                        loss_c = critic_loss(
                            agent,
                            agent_prime,
                            batch_obs,
                            batch_options,
                            batch_rewards,
                            batch_next_obs,
                            batch_dones,
                            gamma,
                        )
                        loss += loss_c

                    agent.update(loss)

                    if steps % freeze_interval == 0:
                        agent_prime.load_state_dict(agent.state_dict())
                        print(f"Updated prime network at step {steps}")

                obs = next_obs
                state = agent.get_state(obs)
                op_term, greedy_option = agent.predict_option_termination(state, current_option)
                curr_op_len += 1
                steps += 1

                if wandb_upload:
                    wandb.log({
                        "actor_loss": loss_a,
                        "critic_loss": loss_c,
                        "current_option": current_option,
                        "greedy_option": greedy_option,
                        "current_option_length": curr_op_len,
                        "option_termination": op_term,
                        "option_termination_probability": agent.get_termination(state)[current_option].item(),

                    })

            if wandb_upload:
                wandb.log({
                    "reward": rewards,
                })


            losses.append(loss.item() if loss is not None else None)
            actor_losses.append(loss_a.item() if loss_a is not None else None)
            critic_losses.append(loss_c.item() if loss_c is not None else None)
            ep_rewards.append(rewards)
            ep_lengths.append(steps)
            ep_option_lengths.append(option_lengths)

        if save_model:
            os.makedirs(model_path, exist_ok=True)
            torch.save(agent.state_dict(), f"{model_path}/model.pt")

        if wandb_upload:
            run.finish()

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(ep_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(2, 2, 2)
        plt.plot(critic_losses)
        plt.title("Critic Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        plt.subplot(2, 2, 3)
        plt.plot(actor_losses)
        plt.title("Actor Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        plt.subplot(2, 2, 4)
        plt.plot(losses)
        plt.title("Total Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.show()

        env.close()



    env = gym.make('CartPole-v1', render_mode='human')
    # env = gym.make('ALE/Pong-v5', render_mode='human', obs_type='grayscale')
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.ResizeObservation(env, (42, 42))
    # env = gym.wrappers.ResizeObservation(env, (21, 21))
    # env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
    # env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    if load_model:
        agent = OptCrit(state_dim, action_dim, n_options, device=device, percept=False)
        agent.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
    
    agent.eval()

    obs, _ = env.reset(seed = 42)
    done = False
    while not done:
        with torch.no_grad():
            state = agent.get_state(obs)
            greedy_option = agent.greedy_option(state)
            action, _, _ = agent.get_action(state, greedy_option)
        # print(f"Test action taken: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"Reward received: {reward}")
        # print(f"Terminated: {terminated}, Truncated: {truncated}")
        done = terminated or truncated
    env.close()





if __name__ == "__main__":
    main()
