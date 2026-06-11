import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

import gymnasium as gym

class lower_DQN(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super(lower_DQN, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.goal_dim = goal_dim
        self.device = device

        # define the network architecture for the worker DQN
        self.lower_critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # a linear layer that takes the goal as input and outputs a vector of size action_dim 
        # to be added to the Q-values from the lower critic
        self.lower_policy = nn.Linear(1, action_dim) 

    def forward(self, x, g):
        Q_l = self.lower_critic(x)
        logits = self.lower_policy(g) + Q_l
        return Q_l, logits

class higher_DQN(nn.Module):
    def __init__(self, obs_dim, goal_dim, device):
        super(higher_DQN, self).__init__()

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.device = device

        # define the network architecture for the manager DQN
        self.higher_critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, goal_dim),
        )

    def forward(self, x):
        Q_h = self.higher_critic(x)
        return Q_h

class feudal_DQN(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, device, T=10, eps=0.1, alpha=1e-3, beta=1e-3):
        super(feudal_DQN, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.T = T # the manager selects a new goal every T steps
        self.device = device
        self.alpha = alpha # learning rate for the worker DQN
        self.beta = beta # learning rate for the manager DQN

        # initialize the worker and manager DQNs
        self.lower = lower_DQN(obs_dim, action_dim, device).to(device)
        self.higher = higher_DQN(obs_dim, goal_dim, device).to(device)

        self.lower_optimizer = optim.Adam(self.lower.parameters(), lr=self.alpha)
        self.higher_optimizer = optim.Adam(self.higher.parameters(), lr=self.beta)


    def select_goal(self, state):
        s = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        Q_h = self.higher(s)
        goal = torch.argmax(Q_h)
        return goal, Q_h
    
    def select_action(self, state, goal):
        s = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        g = torch.as_tensor([goal], dtype=torch.float32).to(self.device)

        Q_l, logits = self.lower(s, g)

        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action, log_prob, entropy, Q_l

    def get_losses(self, states, actions, log_probs, entropies, goals, Q_hs, Q_ls, rewards, dones, gamma_h=0.99, gamma_l=0.95, ent_coef=0.01):

        r_l = torch.zeros(self.T, device=self.device)
        r_h = torch.zeros(len(rewards) // self.T, device=self.device)

        loss_l = 0
        loss_h = 0
        policy_loss = 0
        goal_loss = 0

        N = rewards.shape[0] - 1

        for t in reversed(range( N // self.T )):
            for k in reversed(range(self.T)):
                r_l[k] = rewards[t*self.T + k] # free design reward - lower level
                r_h[t] = r_h[t] + rewards[t*self.T + k] * (gamma_h ** k) # discounted sum of rewards (higher lvl)

                # calculate the target Q-values for the lower DQN
                target_l = r_l[k] + gamma_l * Q_ls[t*self.T + k + 1].max() * (1 - dones[t*self.T + k + 1])
                error_l = target_l - Q_ls[t*self.T + k][actions[t*self.T + k]]
                loss_l += 0.5 * error_l.pow(2)

                policy_loss += -log_probs[t*self.T + k] * error_l.detach() - ent_coef * entropies[t*self.T + k]

            # calculate the target Q-values for the higher DQN
            target_h = r_h[t] + (gamma_h ** self.T) * Q_hs[t+1].max() * (1 - dones[(t+1)*self.T])
            error_h = target_h - Q_hs[t][goals[t]]
            loss_h += 0.5 * error_h.pow(2)

            # directional vector from state at time tT to state at time (t+1)T for x, y 
            state_dir = states[(t+1)*self.T, :2] - states[t*self.T, :2] 
            # the goal loss encourages the manager to select goals that point in the direction of the state change that occurs over the next T steps
            # it goes to 1 when goals aligns
            theta = (goals[t] * 3.142 / self.goal_dim).detach().clone().requires_grad_(True)
            goal_dir = torch.stack([torch.cos(theta), torch.sin(theta)])
            dir_cos = F.cosine_similarity(state_dir.unsqueeze(0), goal_dir.unsqueeze(0))
            # when goals aligns with state change, error_h has higher effect on goal_loss
            goal_loss += -dir_cos * error_h.detach() 

        return loss_l, loss_h, policy_loss, goal_loss
    
    def update(self, loss_l, loss_h, policy_loss, goal_loss):

        self.lower_optimizer.zero_grad()
        lower_loss = loss_l #+ policy_loss
        lower_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower.parameters(), max_norm=0.5)
        self.lower_optimizer.step()

        self.higher_optimizer.zero_grad()
        higher_loss = loss_h + goal_loss
        higher_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.higher.parameters(), max_norm=0.5)
        self.higher_optimizer.step()


def main():

    # environment hyperparams
    n_envs = 1
    randomize_domain = False

    lower_lr = 1e-3
    higher_lr = 1e-3
    gamma_h = 0.99
    gamma_l = 0.95
    T = 10
    goal_dim = 8
    n_updates = 1000
    n_steps_per_update = 400


    train_agent = True
    save_weights = True
    load_weights = False
    wandb_upload = True
    use_cuda = False

    model_dir = f"Weights/lunarlander_FeudalDQN_weights/1k_updates_T_{T}_goalDim_{goal_dim}_lowerLR_{lower_lr}_higherLR_{higher_lr}"



    # environment setup
    if randomize_domain:
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "LunarLander-v3",
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=600,
                )
                for i in range(n_envs)
            ]
        )

    else:
        env = gym.make("LunarLander-v3")


    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n

    # set the device
    if use_cuda and torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # init the agent
    agent = feudal_DQN(obs_shape, action_shape, goal_dim=goal_dim, device=device, T=T, alpha=lower_lr, beta=higher_lr)

    if train_agent:
        torch.autograd.set_detect_anomaly(True)
        if wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="Feudal-DQN",
                # Track hyperparameters and run metadata.
                config={
                    "env": "LunarLander-v3",
                    "architecture": "Feudal DQN",
                    "observation_dim": obs_shape,
                    "action_dim": action_shape,
                    "goal_dim": goal_dim,
                    "T": T,
                    "n_envs": n_envs,
                    "n_updates": n_updates,
                    "n_steps_per_update": n_steps_per_update,
                    "randomize_domain": randomize_domain,
                    "lower_lr": lower_lr,
                    "higher_lr": higher_lr,
                    "gamma_h": gamma_h,
                    "gamma_l": gamma_l,
                },
            )

        loss_l_list = []
        loss_h_list = []
        policy_loss_list = []
        goal_loss_list = []

        ep_rewards = [0]
        ep_lengths = [0]

        ep_reward = 0
        ep_len = 0

        for update in tqdm(range(n_updates)):
            obs, infos = env.reset()
            step = 0

            states = torch.zeros((n_steps_per_update, obs_shape), device=device)
            actions = torch.zeros((n_steps_per_update), dtype=torch.long, device=device)
            log_probs = torch.zeros((n_steps_per_update), device=device)
            entropies = torch.zeros((n_steps_per_update), device=device)
            goals = torch.zeros((n_steps_per_update // T), dtype=torch.long, device=device)
            Q_hs = torch.zeros((n_steps_per_update // T, goal_dim), device=device)
            Q_ls = torch.zeros((n_steps_per_update, action_shape), device=device)

            rewards = np.zeros((n_steps_per_update))
            dones = np.zeros((n_steps_per_update))

            while step < n_steps_per_update:
                # select goals and actions for each environment
                goal, Q_h = agent.select_goal(obs)
                goals[step // T] = goal
                Q_hs[step // T, :] = Q_h
                # print(f"Goal selected {step // T}: {goal}")

                for t in range(agent.T):
                    # How to select actions policy based on goals?
                    # Now goal->action_dim (1 NN) + Q_l -> logits for action selection
                    # Directional goal fixed for T steps.
                    action, log_prob, entropy, Q_l = agent.select_action(obs, goal)
                    obs, reward, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated

                    states[step, :] = torch.as_tensor(obs, dtype=torch.float32).to(device)
                    actions[step] = action
                    log_probs[step] = log_prob
                    entropies[step] = entropy
                    rewards[step] = reward/200 # normalize rewards
                    dones[step] = done
                    Q_ls[step, :] = Q_l

                    step += 1
                    ep_reward += reward
                    ep_len += 1

                    if done:
                        obs, infos = env.reset()
                        ep_rewards.append(ep_reward)
                        ep_lengths.append(ep_len)
                        ep_reward = 0
                        ep_len = 0

                    # wandb.log({"goal": goal.item(),
                    #            "action": action.item(),
                    #            })


            # store the transitions and other relevant information for training
            loss_l, loss_h, policy_loss, goal_loss = agent.get_losses(
                states,
                actions,
                log_probs,
                entropies,
                goals,
                Q_hs,
                Q_ls,
                rewards,
                dones,
                gamma_h=gamma_h,
                gamma_l=gamma_l,
            )

            loss_l_list.append(loss_l)
            loss_h_list.append(loss_h)
            policy_loss_list.append(policy_loss)
            goal_loss_list.append(goal_loss)

            agent.update(loss_l, loss_h, policy_loss, goal_loss)

            # log training progress to wandb
            if wandb_upload:
                wandb.log(
                    {
                        "loss_l": loss_l,
                        "loss_h": loss_h,
                        "policy_loss": policy_loss,
                        "goal_loss": goal_loss,
                        "episode_rewards": ep_rewards[-1],
                        "episode_lengths": ep_lengths[-1],
                    }
                )

        if wandb_upload:
            run.finish()


    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    """ save network weights """
    if train_agent and save_weights:
        torch.save(agent.state_dict(), os.path.join(model_dir, "model.pt"))


    """ load network weights """
    if load_weights:
        agent = feudal_DQN(obs_shape, action_shape, goal_dim=goal_dim, device=device, T=T, alpha=lower_lr, beta=higher_lr)

        agent.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        agent.eval()

    """ play a couple of showcase episodes """

    n_showcase_episodes = 3

    for episode in range(n_showcase_episodes):
        print(f"starting episode {episode}...")

        # create a new sample environment to get new random parameters
        if randomize_domain:
            env = gym.make(
                "LunarLander-v3",
                render_mode="human",
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=500,
            )
        else:
            env = gym.make("LunarLander-v3", render_mode="human", max_episode_steps=500)

        # get an initial state
        state, info = env.reset()

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                goal, _ = agent.select_goal(state)            

                for t in range(agent.T):
                    action, _, _, _ = agent.select_action(state, goal)
                    state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    done = terminated or truncated

    env.close()


if __name__ == "__main__":    
    main()