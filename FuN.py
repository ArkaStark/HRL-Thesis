import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import wandb

gym.register_envs(ale_py)

class Reshape(nn.Module):
    """Utility module to reshape tensors in Sequential models."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class dLSTM(nn.Module):
    """Dilated LSTM for Feudal Networks."""

    def __init__(self, r, input_size, hidden_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, device=self.device)
        self.r = r

    def init_state(self, batch_size):
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history

        h0 = [torch.zeros(1, batch_size, self.lstm.hidden_size, device=self.device) for _ in range(self.r)]
        c0 = [torch.zeros(1, batch_size, self.lstm.hidden_size, device=self.device) for _ in range(self.r)]
        tick = 0
        return tick, h0, c0

    def forward(self, inputs, state):
        """returns g_t, (tick, hidden)"""
        tick, hx, cx = state
        # print(f"Input shape to dLSTM: {inputs.shape}, state: {state[1][0].shape}")
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # [batch, 1, feature]
        # print(f"Input shape after unsqueeze: {inputs.shape}")
        output_seq, (hx[tick], cx[tick]) = self.lstm(inputs, (hx[tick], cx[tick]))
        tick = (tick + 1) % self.r
        output = sum(hx) / self.r  # [1, batch, hidden]
        output = output.squeeze(0)  # [batch, hidden]
        return output, (tick, hx, cx)

# class dLSTM(nn.Module):
#     def __init__(self, r, input_size, hidden_size, device: torch.device = torch.device("cpu")):
#         super().__init__()
#         self.device = device
#         self.radius = r
#         self.hidden_size = hidden_size
#         self.rnn = nn.LSTMCell(input_size, hidden_size, batch_first=True, device=device)
#         self.index = torch.arange(0, r * hidden_size, r)
#         self.dilation = 0

#     def forward(self, state, hidden):
#         """At each time step only the corresponding part of the state is updated
#         and the output is pooled across the previous c out- puts."""
#         d_idx = self.dilation_idx
#         hx, cx = hidden
#         print(f"Input shape to dLSTM: {state.shape}, hidden: hx, cx: {hx.shape}, {cx.shape}, d_idx: {d_idx}")
#         hx[:, d_idx], cx[:, d_idx] = self.rnn(state, (hx[:, d_idx], cx[:, d_idx]))
#         detached_hx = hx[:, self.masked_idx(d_idx)].detach()
#         detached_hx = detached_hx.view(detached_hx.shape[0], self.hidden_size, self.radius-1)
#         detached_hx = detached_hx.sum(-1)

#         y = (hx[:, d_idx] + detached_hx) / self.radius
#         return y, (hx, cx)
    
#     def init_state(self, batch_size):
#         h0 = torch.zeros(batch_size, self.hidden_size * self.radius, device=self.device)
#         c0 = torch.zeros(batch_size, self.hidden_size * self.radius, device=self.device)
#         return (h0, c0)

#     def masked_idx(self, dilated_idx):
#         """Because we do not want to have gradients flowing through all
#         parameters but only at the dilation index, this function creates a
#         'negated' version of dilated_index, everything EXCEPT these indices."""
#         masked_idx = torch.arange(1, self.radius * self.hidden_size + 1)
#         masked_idx[dilated_idx] = 0
#         masked_idx = masked_idx.nonzero()
#         masked_idx = masked_idx - 1
#         return masked_idx

#     @property
#     def dilation_idx(self):
#         """Keep track at which dilation we currently we are."""
#         dilation_idx = self.dilation + self.index
#         self.dilation = (self.dilation + 1) % self.radius
#         return dilation_idx

class Perception(nn.Module):
    """Perception Network for processing visual inputs."""

    def __init__(self, obs_shape, d, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        channels, height, width = obs_shape
        # print(f"Observation shape: {obs_shape}")
        percept_linear_input_dim = 32 * int((int((height - 4) / 4) - 2) / 2) * int((int((width - 4) / 4) - 2) / 2)
        # print(f"Perception linear input dimension: {percept_linear_input_dim}")
        self.f_percept = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4, device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, device=self.device),
            nn.ReLU(),
            Reshape((percept_linear_input_dim)),
            nn.Linear(percept_linear_input_dim, d, device=self.device),
            nn.ReLU(),
            Reshape((-1, d)),
        )
        

    def forward(self, x):
        return self.f_percept(x)
    
class Manager(nn.Module):
    """Manager Network for high-level goal generation."""

    def __init__(self, d, c, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.c = c  # Prediction horizon
        self.device = device

        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d, device=self.device),
            nn.ReLU(),
            Reshape((-1, d))  
        )

        self.f_Mrnn = dLSTM(c, d, d, device=self.device)

        self.value = nn.Linear(d, 1, device=self.device)
        self.eps = 0.01  # Exploration probability for the manager


    def forward(self, z, hidden_M, mask):
        s = self.f_Mspace(z) # Latent state representation [batch x d]
        # hidden_M = (hidden_M[0], mask * hidden_M[1][0], mask * hidden_M[2][0])  # Apply mask to hidden state
        # print(hidden_M[1].shape, hidden_M[2].shape)
        g_hat, hidden_M = self.f_Mrnn(s, hidden_M)
        g = F.normalize(g_hat, dim=-1)  # goal vector [batch x d]
        value = self.value(g_hat)  # value estimate for the current state [batch x 1]

        if (self.eps > torch.rand(1)[0]):
            # To encourage exploration in transition policy,
            # at every step with a small probability ε
            # we emit a random goal sampled from a uni-variate Gaussian.
            g = torch.randn_like(g, requires_grad=False)
        return value, g, s, hidden_M 
    
    def init_state(self, batch_size):
        return self.f_Mrnn.init_state(batch_size)
    
class Worker(nn.Module):
    """Worker Network for low-level action generation."""

    def __init__(self, action_dim, d, k, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.k = k  # Number of goals

        self.f_Wrnn = nn.LSTM(d, int(action_dim*k), batch_first=True, device=self.device)
        self.phi = nn.Sequential(
            nn.Linear(d, k, bias=False, device=self.device),
            Reshape((-1, 1, k)),
        )
        self.value = nn.Sequential(
            nn.Linear(action_dim * k, 50, device=self.device),
            nn.ReLU(),
            nn.Linear(50, 1, device=self.device)
        )
        self.action_view = Reshape((k, action_dim))

    def forward(self, z, goals, hidden_W, mask):
        sum_g_W = torch.stack(goals).detach().sum(dim=0) # Detaching is vital, no end to end training
        w = self.phi(sum_g_W)  # projection [batch x 1 x k]
        if z.dim() == 2:
            z = z.unsqueeze(1)  # [batch, 1, d]
        hidden_W = (mask*hidden_W[0], mask*hidden_W[1])  # Apply mask to hidden state
        U_flat, hidden_W = self.f_Wrnn(z, hidden_W)  
        U_flat = U_flat.squeeze(1)  # [batch, k*a]
        U = self.action_view(U_flat)  # [batch x k x a]
        a = (w @ U).squeeze(1)  # [batch x a]
        probs = F.softmax(a, dim=-1)  # action probabilities
        value = self.value(U_flat)  # value estimate for the current state [batch x 1]
        return value, probs, hidden_W
    
    def init_state(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.f_Wrnn.hidden_size, device=self.device),
            torch.zeros(1, batch_size, self.f_Wrnn.hidden_size, device=self.device)
        )
    
class FuN(nn.Module):
    """Feudal Networks for Hierarchical Reinforcement Learning."""

    def __init__(
            self,
            state_dim,
            action_dim,
            d = 256,
            k = 16,
            c = 10,
            device: torch.device = torch.device("cpu"),
            percept = True,
            lr = 1e-4,
            b = 1
    ) -> None:
        """Initialize the Feudal Networks."""
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d = d
        self.k = k
        self.c = c
        self.b = b
        self.percept = percept

        if self.percept:
            self.perception = Perception(state_dim, self.d, device=self.device)

        # Manager feedforward network
        self.manager = Manager(d, c, device=self.device)
        self.worker = Worker(action_dim, d, k, device=self.device)
        # self.manager_part_loss = nn.CosineEmbeddingLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr)  
    
    def forward(self, x, goals, states, masks, save = True):
        """Forward pass through the Feudal Networks."""

        x = torch.as_tensor(x, dtype=torch.float32).to(self.device)

        if self.percept:
            z = self.perception(x)  # Get latent representation from perception module
        else:
            z = x  # If no perception, use raw input as latent representation
        
        # s_prev = ss[tick_dlstm]
        # g_prev = F.normalize(hx_M[tick_dlstm].squeeze(0), dim=-1)  # previous goal vector

        value_M, g, s, hidden_M = self.manager(z, self.hidden_M, masks[-1])  # Manager forward pass

        # Ensure that we only have a list of size 2*c + 1, and we use FiFo
        if len(goals) >= (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)

        goals.append(g)
        states.append(s.detach())  # states never have gradients active


        # ss[tick_dlstm] = s.detach()  # Detach s for the next time step
        # target = -torch.ones(g_prev.size(0), device=g_prev.device)

        nabla_dcos = F.cosine_similarity(states[2*self.c] - states[self.c], goals[self.c]).unsqueeze(-1) 
        
        # sum_g = sum(F.normalize(h.squeeze(0), dim=-1) for h in states_M[1]).detach() # detach to prevent gradients flowing back to the manager from the worker

        # The manager is ahead at least c steps, so we feed
        # only the first c+1 states to worker
        value_W, action_probs, hidden_W = self.worker(z, goals[:self.c + 1], self.hidden_W, masks[-1])  # Worker forward pass

        if save:
            # Optional, dont do this for the next_v
            self.hidden_M = hidden_M
            self.hidden_W = hidden_W

        return action_probs, (value_M, value_W), goals, states, nabla_dcos
    
    def init_state(self):
        template = torch.zeros(self.b, self.d)
        goals = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        states = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2*self.c+1)]
        self.hidden_M = self.manager.init_state(self.b)
        self.hidden_W = self.worker.init_state(self.b)
        return goals, states, masks

    def detach_LSTMs(self, goals):
        hidden_M = self.hidden_M
        hidden_W = self.hidden_W
        tick, hx, cx = hidden_M
        hx = [h.detach() for h in hx]
        cx = [c.detach() for c in cx]
        h_w, c_w = hidden_W
        hidden_W = (h_w.detach(), c_w.detach())
        self.hidden_M = (tick, hx, cx)
        self.hidden_W = hidden_W
        goals = [g.detach() for g in goals]
        return goals

    def intrinsic_reward(self, goals, states, masks):
        # states_M, states_W, ss = states
        # tick_dlstm, hx_M, cx_M = states_M
        # t = (tick_dlstm - 1) % self.c  # tick is always ahead
        # s_t = ss[t]
        t = self.c
        rI = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)
        for i in range(1, self.c + 1):
            # t_i = (t - i) % self.c
            # s_t_i = ss[t_i]
            # g_t_i = F.normalize(hx_M[t_i].squeeze(0), dim=-1)
            rI += mask * F.cosine_similarity(states[t] - states[t-i], goals[t-i]).unsqueeze(-1)
            mask = mask * masks[t-i] #TODO

        return rI.detach()/self.c
    
    def select_action(self, x, goals, states, masks) -> int:
        """Sample an action from the policy."""
        action_probs, values, goals, states, nabla_dcos = self.forward(x, goals, states, masks)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), log_prob, entropy, values, goals, states, nabla_dcos

    def get_loss(
            self, 
            values, 
            rewards, 
            log_probs,
            nabla_dcoss,
            rI,  
            masks, 
            entropy, 
            ent_coef=0.01, 
            gamma_M=0.99, 
            gamma_W=0.99, 
            alpha=0.1
        ):
        """Compute the loss for training the Feudal Networks."""
        lam = 0.95
        value_M, value_W = values
        value_M = value_M.view(-1)
        value_W = value_W.view(-1)
        log_probs = log_probs.view(-1)
        rI = rI.view(-1)
        masks = masks.view(-1)
        entropy = entropy.view(-1)
        T = rewards.size(0)

        gae_M = 0
        gae_W = 0
        adv_M = torch.zeros_like(rewards).to(self.device)
        adv_W = torch.zeros_like(rewards).to(self.device)

        for t in reversed(range(T)):
            delta_M = rewards[t] + gamma_M * masks[t] * value_M[t + 1] - value_M[t]
            gae_M = delta_M + gamma_M * lam * masks[t] * gae_M
            adv_M[t] = gae_M
            delta_W = rewards[t] + alpha*rI[t] + gamma_W * masks[t] * value_W[t + 1] - value_W[t]
            gae_W = delta_W + gamma_W * lam * masks[t] * gae_W
            adv_W[t] = gae_W
        critic_loss_M = adv_M.pow(2).mean()
        critic_loss_W = adv_W.pow(2).mean()

        # print(f"nabla_dcoss: {nabla_dcoss.shape}")
        # padded_nabla_dcos = torch.cat([nabla_dcoss[:-self.c], torch.zeros(T-self.c, device=self.device)], dim=0)  # Pad the last T-c values with zeros
        # print(F"padded_nabla_dcos: {padded_nabla_dcos.shape}")
        loss_M = -(adv_M*nabla_dcoss).mean()  # Manager loss based on the cosine embedding loss
        loss_W = -(log_probs * adv_W.detach()).mean() - ent_coef * entropy.mean()
        return critic_loss_M, critic_loss_W, loss_M, loss_W
    
    def update(self, loss):
        """Update the model parameters."""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

def main():

    #Hyperparameters
    d = 256
    k = 16
    c = 10
    gamma_M = 0.99
    gamma_W = 0.95
    alpha = 0.8
    ent_coef = 0.01
    lr = 1e-4

    n_updates = 1000
    batch_size = 400

    train = True
    save_model = True
    load_model = False
    model_path = "Pong-v5/1k_updates_new"
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
                project="HRL-FuN",
                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": lr,
                    "architecture": "FuN",
                    "env": model_path,
                    "epochs": n_updates,
                    "batch_size": batch_size,
                    "d": d,
                    "k": k,
                    "c": c,
                    "gamma_M": gamma_M,
                    "gamma_W": gamma_W,
                    "alpha": alpha,
                    "ent_coef": ent_coef,
                },
            )
    
        env = gym.make('ALE/Pong-v5', obs_type='grayscale')
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.ResizeObservation(env, (42, 42))
        # env = gym.wrappers.ResizeObservation(env, (21, 21))
        env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
        env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n

        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

        agent = FuN(state_dim, action_dim, d, k, c, device, percept='grayscale', lr=lr).to(device)
        if load_model:
            agent.load_state_dict(torch.load("Pong-v5/1k_updates/model.pt", map_location=device))

        total_loss_history = []
        critic_loss_history = []
        worker_loss_history = []
        manager_loss_history = []
        entropy_history = []
        episodes_rewards = [0]
        ep_reward = 0

        obs, _ = env.reset(seed = 0)
        goals, states, masks = agent.init_state()
        for epoch in tqdm(range(n_updates)):

            # Detaching LSTMs and goals
            goals = agent.detach_LSTMs(goals)

            rewards = torch.zeros(batch_size, device=agent.device)
            obss = torch.zeros(batch_size, *state_dim, device=agent.device)
            actions = torch.zeros(batch_size, device=agent.device)
            log_probs = torch.zeros(batch_size, device=agent.device)
            entropies = torch.zeros(batch_size, device=agent.device)
            r_masks = torch.zeros(batch_size, device=agent.device)
            rI = torch.zeros(batch_size, device=agent.device)
            values_M = torch.zeros(batch_size+1, device=agent.device)
            values_W = torch.zeros(batch_size+1, device=agent.device)
            nabla_dcoss = torch.zeros(batch_size, device=agent.device)

            for step in range(batch_size):
                obs = obs.reshape(1, *obs.shape)  # Add batch 
                action, log_prob, entropy, values, goals, states, nabla_dcos = agent.select_action(obs, goals, states, masks) 
                # print(f"Action taken: {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                mask = 1 - float(terminated or truncated)
                # reward = reward + info['lives']  # Add a bonus for keeping lives
                # print(f"Info: {info}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

                rewards[step] = reward
                ep_reward += reward
                obss[step] = torch.as_tensor(obs, dtype=torch.float32).to(agent.device)
                actions[step] = action
                log_probs[step] = log_prob.squeeze(0)
                entropies[step] = entropy.squeeze(0)
                rI[step] = agent.intrinsic_reward(goals, states, masks)  # Compute intrinsic reward for the current state
                values_M[step] = values[0].squeeze()
                values_W[step] = values[1].squeeze()
                nabla_dcoss[step] = nabla_dcos
                r_masks[step] = mask
                masks.pop(0)
                masks.append(mask)

                if terminated or truncated:
                    episodes_rewards.append(ep_reward)
                    ep_reward = 0
                    obs, _ = env.reset(seed = epoch)
                    goals, states, masks = agent.init_state()

            with torch.no_grad():
                _, next_values, *_ = agent.forward(obs, goals, states, masks, save=False)
                values_M[step+1] = next_values[0].squeeze().detach()
                values_W[step+1] = next_values[1].squeeze().detach() 

            critic_loss_M, critic_loss_W, loss_M, loss_W = agent.get_loss(
                (values_M, values_W), 
                rewards, 
                log_probs, 
                nabla_dcoss,
                rI,
                r_masks, 
                entropies, 
                ent_coef, 
                gamma_M, 
                gamma_W, 
                alpha
            )
            agent.update((critic_loss_M + critic_loss_W + loss_M + loss_W))

            if wandb_upload:
                run.log({
                    "total_loss": (critic_loss_M + critic_loss_W + loss_M + loss_W).item(),
                    "critic_loss": (critic_loss_M + critic_loss_W).item(),
                    "critic_loss_M": critic_loss_M.item(),
                    "critic_loss_W": critic_loss_W.item(),
                    "loss_M": loss_M.item(),
                    "loss_W": loss_W.item(),
                    "episodes_rewards": episodes_rewards[-1],
                    "intrinsic_reward": rI.mean().item(),
                    "entropy": entropies.mean().item(),
                })

            # Append losses to history
            total_loss_history.append((critic_loss_M + critic_loss_W + loss_M + loss_W).item())
            critic_loss_history.append((critic_loss_M + critic_loss_W).item())
            worker_loss_history.append(loss_W.item())
            manager_loss_history.append(loss_M.item())
            entropy_history.append(entropies.mean().item())

        if save_model:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(agent.state_dict(), f"{model_path}/model.pt")

        if wandb_upload:
            run.finish()

        # Plot training curves
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(total_loss_history, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()    

        plt.subplot(2, 2, 2)
        plt.plot(critic_loss_history, label='Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(worker_loss_history, label='Worker Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Worker Loss per Step')
        plt.legend()    

        plt.subplot(2, 2, 4)
        plt.plot(manager_loss_history, label='Manager Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Manager Loss per Step')
        plt.legend()  

        plt.tight_layout()
        plt.show()

    env = gym.make('ALE/Pong-v5', render_mode='human', obs_type='grayscale')
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.ResizeObservation(env, (42, 42))
    # env = gym.wrappers.ResizeObservation(env, (21, 21))
    env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
    env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    agent = FuN(state_dim, action_dim, d, k, c, device, percept='grayscale').to(device)
    if load_model:
        agent.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
    
    agent.eval()

    obs, _ = env.reset(seed = 42)
    goals, states, masks = agent.init_state()
    done = False
    while not done:
        with torch.no_grad():
            obs = obs.reshape(1, *obs.shape)
            action, _, _, _, _, _, _ = agent.select_action(obs, goals, states, masks)
        # print(f"Test action taken: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"Reward received: {reward}")
        # print(f"Terminated: {terminated}, Truncated: {truncated}")
        done = terminated or truncated
    env.close()

if __name__ == "__main__":
    main()
