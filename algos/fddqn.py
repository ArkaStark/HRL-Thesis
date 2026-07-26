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
import cv2

class lower_DQN(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, device):
        super(lower_DQN, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.device = device

        # define the network architecture for the worker DQN
        self.lower_critic = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, g):
        goal_one_hot = torch.zeros((self.goal_dim), device=self.device)
        goal_one_hot[g] = 1.0
        x = torch.cat((x, goal_one_hot), dim=-1)
        Q_l = self.lower_critic(x)
        return Q_l

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
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, goal_dim),
        )

    def forward(self, x):
        Q_h = self.higher_critic(x)
        return Q_h

class FDDQNAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hyperparams={}, device="cpu"):
        super(FDDQNAgent, self).__init__()

        self.lower_lr = hyperparams.get("lower_lr", 1e-3)
        self.higher_lr = hyperparams.get("higher_lr", 1e-3)
        self.gamma_h = hyperparams.get("gamma_h", 0.999)
        self.gamma_l = hyperparams.get("gamma_l", 0.99)
        self.T = hyperparams.get("T", 10)
        self.goal_dim = hyperparams.get("goal_dim", 4)
        self.n_updates = hyperparams.get("n_updates", 10000)
        self.n_steps_per_update = hyperparams.get("n_steps_per_update", 40)
        self.rl_wt = hyperparams.get("rl_wt", 0.4)
        self.gl_wt = hyperparams.get("gl_wt", 0.8)
        self.action_eps = hyperparams.get("action_eps", 0.05)
        self.goal_eps = hyperparams.get("goal_eps", 0.1)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = self.goal_dim
        self.T = self.T # the manager selects a new goal every T steps
        self.device = device
        self.alpha = self.lower_lr # learning rate for the worker DQN
        self.beta = self.higher_lr # learning rate for the manager DQN
        self.rl_wt = self.rl_wt # weight for the reinforcement learning loss
        self.gl_wt = self.gl_wt # weight for the goal learning loss
        self.action_eps = self.action_eps # epsilon for action selection
        self.goal_eps = self.goal_eps # epsilon for goal selection
        # initialize the worker and manager DQNs
        self.lower_A = lower_DQN(obs_dim, action_dim, self.goal_dim, device).to(device)
        self.lower_B = lower_DQN(obs_dim, action_dim, self.goal_dim, device).to(device)
        self.higher_A = higher_DQN(obs_dim, self.goal_dim, device).to(device)
        self.higher_B = higher_DQN(obs_dim, self.goal_dim, device).to(device)
        self.goal_loss_cosine = nn.CosineEmbeddingLoss()

        self.lower_optimizer = optim.Adam(list(self.lower_A.parameters()) + list(self.lower_B.parameters()), lr=self.alpha)
        self.higher_optimizer = optim.Adam(list(self.higher_A.parameters()) + list(self.higher_B.parameters()), lr=self.beta)


    def select_goal(self, state):
        s = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        Q_h = self.higher_A(s)

        if np.random.rand() < self.goal_eps:
            goal = torch.randint(0, self.goal_dim, (1,), device=self.device)
        else:
            goal = torch.argmax(Q_h)
        # goal = torch.tensor(0, device=self.device) 
        return goal, Q_h
    
    def select_action(self, state, goal):
        s = torch.as_tensor(state, dtype=torch.float32).to(self.device)

        Q_l = self.lower_A(s, goal)

        # choose action based on epsilon-greedy policy
        if np.random.rand() < self.action_eps:
            action = torch.randint(0, self.action_dim, (1,), device=self.device)
        else:
            action = torch.argmax(Q_l)

        # action_dist = torch.distributions.Categorical(logits=logits)
        # action = action_dist.sample()
        # log_prob = action_dist.log_prob(action)
        # entropy = action_dist.entropy()

        return action, Q_l

    def get_losses(self, states, actions, goals, Q_hs, Q_ls, rewards, dones, gamma_h=0.99, gamma_l=0.95):

        r_l = torch.zeros(self.T, device=self.device)
        r_h = torch.zeros(len(rewards) // self.T, device=self.device)
        dir_cos = torch.zeros(len(rewards) // self.T, device=self.device)

        loss_l = 0
        loss_h = 0
        policy_loss = 0
        goal_loss = 0

        N = rewards.shape[0] - 1

        for t in range( N // self.T ):

            theta = (goals[t] * 6.283185307179586 / self.goal_dim).detach().clone().requires_grad_(True)
            goal_dir = torch.stack([torch.cos(theta), torch.sin(theta)])

            for k in range(self.T):
                state_dir_l = states[t*self.T + k + 1, :2] - states[t*self.T + k, :2]
                dir_cos_l = self.goal_loss_cosine(state_dir_l.unsqueeze(0), goal_dir.unsqueeze(0), torch.tensor([-1.0], device=self.device))
                r_l[k] = self.rl_wt * rewards[t*self.T + k] + dir_cos_l.detach() # free design reward - lower level
                r_h[t] = r_h[t] + rewards[t*self.T + k] * (gamma_h ** k) # discounted sum of rewards (higher lvl)

                # calculate the target Q-values for the lower DQN
                # double DQN style target calculation to mitigate overestimation bias
                if np.random.rand() < 0.5:
                    a_star = self.lower_A(states[t*self.T + k + 1], goals[t]).argmax()
                    Q_l_target = self.lower_B(states[t*self.T + k + 1], goals[t])[a_star].detach() # detach?
                    target_l = r_l[k] + gamma_l * Q_l_target * (1 - dones[t*self.T + k + 1])
                    error_l = target_l - Q_ls[t*self.T + k][actions[t*self.T + k]]
                else:
                    a_star = self.lower_B(states[t*self.T + k + 1], goals[t]).argmax()
                    Q_l_target = self.lower_A(states[t*self.T + k + 1], goals[t])[a_star].detach() # detach?
                    target_l = r_l[k] + gamma_l * Q_l_target * (1 - dones[t*self.T + k + 1])
                    error_l = target_l - self.lower_B(states[t*self.T + k], goals[t])[actions[t*self.T + k]]
                
                loss_l += 0.5 * error_l.pow(2)

                # policy_loss += -log_probs[t*self.T + k] * error_l.detach() - ent_coef * entropies[t*self.T + k].detach() # remove policy

            # calculate the target Q-values for the higher DQN
            if np.random.rand() < 0.5:
                g_star = self.higher_A(states[(t+1)*self.T]).argmax()
                Q_h_target = self.higher_B(states[(t+1)*self.T])[g_star].detach() # detach?
                target_h = r_h[t] + (gamma_h ** self.T) * Q_h_target * (1 - dones[(t+1)*self.T])
                error_h = target_h - Q_hs[t][goals[t]]
            else:
                g_star = self.higher_B(states[(t+1)*self.T]).argmax()
                Q_h_target = self.higher_A(states[(t+1)*self.T])[g_star].detach() # detach?
                target_h = r_h[t] + (gamma_h ** self.T) * Q_h_target * (1 - dones[(t+1)*self.T])
                error_h = target_h - self.higher_B(states[t*self.T])[goals[t]]
            loss_h += 0.5 * error_h.pow(2)

            # directional vector from state at time tT to state at time (t+1)T for x, y 
            state_dir = states[(t+1)*self.T, :2] - states[t*self.T, :2] 
            # the goal loss encourages the manager to select goals that point in the direction of the state change that occurs over the next T steps
            # it goes to 1 when goals aligns
            dir_cos[t] = self.goal_loss_cosine(state_dir.unsqueeze(0), goal_dir.unsqueeze(0), torch.tensor([-1.0], device=self.device))
            # when goals aligns with state change, error_h has higher effect on goal_loss
            goal_loss += -dir_cos[t] * error_h.detach() 

        return loss_l, loss_h, goal_loss, dir_cos[t]
    
    def update(self, loss_l, loss_h, goal_loss):

        self.lower_optimizer.zero_grad()
        lower_loss = loss_l #+ policy_loss
        lower_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.lower_A.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.lower_B.parameters(), max_norm=0.5)
        self.lower_optimizer.step()

        self.higher_optimizer.zero_grad()
        higher_loss = loss_h + self.gl_wt * goal_loss
        higher_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.higher_A.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.higher_B.parameters(), max_norm=0.5)
        self.higher_optimizer.step()

    def train_agent(self, env):

        wandb_upload = True
        save_weights = True
        model_dir = f"Weights/lunarlander_FeudalDQN_weights/1k_updates_T_{self.T}_goalDim_{self.goal_dim}_lowerLR_{self.lower_lr}_higherLR_{self.higher_lr}"

        torch.autograd.set_detect_anomaly(True)

        env_name = env.spec.id
        obs_shape = env.observation_space.shape[0]
        action_shape = env.action_space.n

        if wandb_upload:
            # Start a new wandb run to track this script.
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="arkapatra31-ust",
                # Set the wandb project where this run will be logged.
                project="FDDQN-Train",
                # Track hyperparameters and run metadata.
                config={
                    "env": env_name,
                    "architecture": "Feudal DDQN",
                    "observation_dim": obs_shape,
                    "action_dim": action_shape,
                    "goal_dim": self.goal_dim,
                    "T": self.T,
                    "n_updates": self.n_updates,
                    "n_steps_per_update": self.n_steps_per_update,
                    "lower_lr": self.lower_lr,
                    "higher_lr": self.higher_lr,
                    "gamma_h": self.gamma_h,
                    "gamma_l": self.gamma_l,
                    "rl_wt": self.rl_wt,
                    "gl_wt": self.gl_wt,
                    "action_eps": self.action_eps,
                    "goal_eps": self.goal_eps,
                },
            )


        ep_rewards = [0]
        ep_lengths = [0]

        ep_reward = 0
        ep_len = 0
                    
        obs, infos = env.reset()
        
        for update in tqdm(range(self.n_updates)):

            step = 0

            states = torch.zeros((self.n_steps_per_update, obs_shape), device=self.device)
            actions = torch.zeros((self.n_steps_per_update), dtype=torch.long, device=self.device)
            log_probs = torch.zeros((self.n_steps_per_update), device=self.device)
            entropies = torch.zeros((self.n_steps_per_update), device=self.device)
            goals = torch.zeros((self.n_steps_per_update // self.T), dtype=torch.long, device=self.device)
            Q_hs = torch.zeros((self.n_steps_per_update // self.T, self.goal_dim), device=self.device)
            Q_ls = torch.zeros((self.n_steps_per_update, action_shape), device=self.device)

            rewards = torch.zeros((self.n_steps_per_update), device=self.device)
            dones = torch.zeros((self.n_steps_per_update), device=self.device)

            while step < self.n_steps_per_update:
                # select goals and actions for each environment
                goal, Q_h = self.select_goal(obs)
                goals[step // self.T] = goal
                Q_hs[step // self.T, :] = Q_h
                # print(f"Goal selected {step // self.T}: {goal}")

                for t in range(self.T):
                    # How to select actions policy based on goals?
                    # Now goal->action_dim (1 NN) + Q_l -> logits for action selection
                    # Directional goal fixed for T steps.
                    action, Q_l = self.select_action(obs, goal.item())
                    obs, reward, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated

                    states[step, :] = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                    actions[step] = action
                    # log_probs[step] = log_prob
                    # entropies[step] = entropy
                    rewards[step] = reward/200
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
                        #break? gives error for states[step, :]

                    # wandb.log({"goal": goal.item(),
                    #            "action": action.item(),
                    #            })


            # store the transitions and other relevant information for training
            loss_l, loss_h, goal_loss, dir_cos = self.get_losses(
                states,
                actions,
                goals,
                Q_hs,
                Q_ls,
                rewards,
                dones,
                gamma_h=self.gamma_h,
                gamma_l=self.gamma_l,
            )

            # loss_l_list.append(loss_l)
            # loss_h_list.append(loss_h)
            # policy_loss_list.append(policy_loss)
            # goal_loss_list.append(goal_loss)

            self.update(loss_l, loss_h, goal_loss)

            # log training progress to wandb
            if wandb_upload:
                wandb.log(
                    {
                        "loss_l": loss_l,
                        "loss_h": loss_h,
                        "goal_loss": goal_loss,
                        "episode_reward": ep_rewards[-1],
                        "episode_length": ep_lengths[-1],
                        "dir_cos": dir_cos.mean().item(),
                    }
                )

            if (update+1) % 1000 == 0:
                if sum(ep_rewards[-10:]) <= sum(ep_rewards[-30:-20]):
                    print(f"Episode reward {sum(ep_rewards[-10:])} has not improved over the last 10 episodes{sum(ep_rewards[-30:-20])}.")
                    break

        if wandb_upload:
            run.finish()


        if save_weights:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))
            print(f"Saved model weights to {os.path.join(model_dir, 'model.pt')}")


    def test_agent(self, env, n_episodes=1):

        wandb_upload = True

        env_name = env.spec.id
        obs_shape = env.observation_space.shape[0]
        action_shape = env.action_space.n
        

        model_dir = f"Weights/lunarlander_FeudalDQN_weights/1k_updates_T_{self.T}_goalDim_{self.goal_dim}_lowerLR_{self.lower_lr}_higherLR_{self.higher_lr}"
        self.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        self.eval()

        """ play a couple of showcase episodes """

        for episode in range(n_episodes):
            if wandb_upload:
                run = wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="arkapatra31-ust",
                    # Set the wandb project where this run will be logged.
                    project="FDDQN-Test",
                    # Track hyperparameters and run metadata.
                    config={
                        "env": env_name,
                        "architecture": "Feudal DDQN",
                        "observation_dim": obs_shape,
                        "action_dim": action_shape,
                        "goal_dim": self.goal_dim,
                        "T": self.T,
                        "n_updates": self.n_updates,
                        "n_steps_per_update": self.n_steps_per_update,
                        "lower_lr": self.lower_lr,
                        "higher_lr": self.higher_lr,
                        "gamma_h": self.gamma_h,
                        "gamma_l": self.gamma_l,
                        "rl_wt": self.rl_wt,
                        "gl_wt": self.gl_wt,
                        "action_eps": self.action_eps,
                        "goal_eps": self.goal_eps,
                    },
                )

            print(f"starting episode {episode}...")

            # get an initial state
            state, info = env.reset()

            # play one episode
            done = False
            renders = []
            while not done:
                # select an action A_{t} using S_{t} as input for the agent
                with torch.no_grad():
                    goal, _ = self.select_goal(state)     
                    # goal = torch.tensor(0, device=self.device)   

                    for t in range(self.T):
                        action, _ = self.select_action(state, goal.item())
                        state, reward, terminated, truncated, info = env.step(action.item())
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
                            wandb.log({"showcase_goal": goal.item(),
                                    "showcase_action": action.item(),
                                    "showcase_reward": reward,
                                    "showcase_x_position": state[0],
                                    "showcase_y_position": state[1],
                                    "showcase_velocity_x": state[2],
                                    "showcase_velocity_y": state[3],
                                    "showcase_angle": state[4],
                                    "showcase_angular_velocity": state[5],
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

def hyperparameter_tuning(env, state_dim, action_dim, percept=False, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'episode_reward', 'goal': 'maximize'},
        'parameters': {
            'lower_lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
            'higher_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-4},
            # 'goal_dim': {'values': [4]},
            # 'T': {'values': [5, 10, 20]},
            'rl_wt': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            'gl_wt': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            # 'action_eps': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
            # 'goal_eps': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},

        },
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 500}
    }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project="FDDQN-Hyperparameter-Tuning", entity="arkapatra31-ust")

    # Define the training function for the sweep
    def train_sweep():
        with wandb.init() as run:
            config = wandb.config
            hyperparams = {
                "lower_lr": config.lower_lr,
                "higher_lr": config.higher_lr,
                # "goal_dim": config.goal_dim,
                # "T": config.T,
                "rl_wt": config.rl_wt,
                "gl_wt": config.gl_wt,
                # "action_eps": config.action_eps,
                # "goal_eps": config.goal_eps,
                "wandb_upload": True,
            }
            wandb.run.name = f"FDDQN_lowerLR_{hyperparams['lower_lr']:.4f}_higherLR_{hyperparams['higher_lr']:.5f}_rl_wt_{hyperparams['rl_wt']:.2f}_gl_wt_{hyperparams['gl_wt']:.2f}"
            print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
            fddqn_agent = FDDQNAgent(state_dim, action_dim, device=device, hyperparams=hyperparams)
            fddqn_agent.train_agent(env=env)
            # fddqn_agent.test_agent(env=env, n_episodes=3)

    # Start the sweep agent to run the training function for each configuration
    wandb.agent(sweep_id, function=train_sweep, count=10)  # Adjust count as needed for the number of runs
