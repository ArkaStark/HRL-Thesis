import torch
import numpy as np
import wandb
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Environment imports
import ogbench
import gymnasium as gym
import ale_py
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
gym.register_envs(ale_py)
gym.register_envs(minigrid)

SEED = 31

from algos import a2c, ddpg, oc, rainbow, td3, her, hac, fddqn, hiro, fun, ppo, rainbow


def main():

    # DECLARE ENVIRONMENT

    # env = gym.make("Pusher-v5", max_episode_steps=500)
    env = gym.make("LunarLander-v3")

    # env = gym.make("MiniGrid-FourRooms-v0")
    # env = FullyObsWrapper(env)  # Use fully observable state instead of partial observations
    # env = ImgObsWrapper(env)  # Use image observations instead of dictionary observations
    # env.action_space = gym.spaces.Discrete(3)  # Update the action space to have 3 discrete actions


    # dataset_name = 'pointmaze-medium-navigate-v0'  # Choose from 'cube-single-object-v0', 'cube-multi-object-v0', 'drawer-single-object-v0', 'drawer-multi-object-v0'.
    # env = ogbench.make_env_and_datasets(dataset_name, env_only=True)

    # obs_shape = env.observation_space
    # action_shape = env.action_space

    # env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # # env = gym.wrappers.ResizeObservation(env, (42, 42))
    # env = gym.wrappers.ResizeObservation(env, (21, 21))
    # env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
    # env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information

    # env = env_wrapper(env)  # Wrap the environment if needed


    # run = wandb.init(
    #     entity = "arkapatra31-ust",
    #     project = "Test-Env-Wrappers",
    # )
    # wandb.run.name = f"{env.spec.id}"

    # obs = env.reset(seed=SEED)
    # env.action_space.seed(SEED)
    # env_step = 0
    # print(f"Initial Observation: {obs}")
    # done = False
    # while not done:
    #     action = env.action_space.sample()  # Sample a random action
    #     obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
    #     done = terminated or truncated  # Check if the episode is done
    #     env_step += 1
    #     print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}, Action: {action}")
    #     wandb.log({"Env_Step": env_step})

    #     for i in range(len(obs)):
    #         wandb.log({f"Observation_{i}": obs[i]})
    #     # for i in range(len(action)):
    #     wandb.log({f"Action": action})
    #     wandb.log({"Reward": reward})

    # wandb.finish()

    obs_shape = env.observation_space.shape[0]
    # action_shape = env.action_space.shape[0]
    action_shape = env.action_space.n

    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    print(f"Environment: {env.spec.id}")
    print(f"Observation type: {env.observation_space}, Action type: {env.action_space}")
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")

    # SET THE DEVICE
    use_cuda = True
    if use_cuda:
        print("Using CUDA")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # SET HYPERPARAMETERS as desired (check the agent's class for available hyperparameters, default values and definitions)

    hyperparameters = {
        # "WANDB_UPLOAD": True,
        # "SAVE_MODEL": True,

        # "LR": 45e-5,                       # OC, FuN, A2C
        # "LR_ACTOR": 0.00001725897285277334,                 # DDPG, TD3, HIRO, HAC
        # "LR_CRITIC": 0.00039843834524731623,                # DDPG, TD3, HIRO, HAC
        # "GAMMA": 0.99,                    # OC, DDPG
        # "GAMMA_M": 0.99,                  # FuN, HIRO
        # "GAMMA_W": 0.95,                  # FuN, HIRO
        # "LAM": 0.977,

        # "NUM_OPTIONS": 4,                 # OC
        # "TERM_REG": 1e-4,                 # OC
        # "EPSILON_DECAY": 0.9952,           # OC
        
        # "MAX_ACTION": 0.4,                # DDPG, TD3
        # "EXPLORATION_NOISE": 0.05,        # DDPG, TD3
        # "TAU": 0.01,                      # DDPG, TD3 
        # "ENT_COEF": 1e-4,                 # OC, A2C

        # "GOAL_DIM": 8,                  # FuN, HIRO
        # "N_GOALS": 1,                    # FuN
        # "SUBGOAL_HORIZON": 5,            # FuN, HIRO, HAC

        # "MAX_EPISODES": 2000,               # OC, DDPG, TD3, HIRO, HAC
        # "N_UPDATES": 1000,                # FuN

        # "ALPHA": 0.028868703061623705,                     # FuN, HAC, HER
        # 'EXPL_NOISE_MANAGER': 1.231936699447596,          # HIRO, HAC
        # 'EXPL_NOISE_WORKER': 0.14683428004294033,           # HIRO, HAC
        # 'GOAL_BOUND': 0.7126979441007939,                  # HIRO, HAC
        # 'GOAL_THRESHOLD': 0.6933242578871506,              # HAC
        # 'N_LEVELS': 3,                      # HAC
        # "MAX_ENV_STEPS": 1e6,              # OC, DDPG, TD3, HIRO, HAC
        # 'SUBGOAL_TEST_FREQ': 0.4,                  # HAC

    }

    # INITIALIZE AGENTS

    agent = rainbow.RainbowAgent(obs_shape, action_shape, percept=False, device=device, hyperparams=hyperparameters)


    # PERFORM HYPERPARAMETER TUNING

    # hyperparameter_tuning(env=env, state_dim=obs_shape, action_dim=action_shape, device=device)


    # TRAIN AGENTS

    # agent.train_agent(env=env)

    # TEST AGENTS

    # env = gym.make("Pusher-v5", max_episode_steps=500, render_mode="human")
    # model_path = "Weights/Pusher-v5_PPOWeights/1250_0.0001_0.001_0.0001"
    # agent.test_agent(env=env, n_episodes=3, model_path=model_path)


    # CREATE TEST ENVIRONMENT FOR EVALUATION

    test_agent_stats = True # Set to True if you want to test the agent and record statistics

    if test_agent_stats:

        # Call the evaluate method to evaluate the agent
        # env = gym.make("FetchSlide-v4", render_mode="human")
        # env = gym.make("BipedalWalker-v3", max_episode_steps=600, render_mode="rgb_array")
        # env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array", max_episode_steps=100)
        # env = FullyObsWrapper(env)  # Use fully observable state instead of partial observations
        # env = ImgObsWrapper(env)  # Use image observations instead of dictionary observations
        # env.action_space = gym.spaces.Discrete(3)  # Update the action space to have 3 discrete actions

        # env = gym.make("ALE/Pong-v5", obs_type="grayscale")
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # # env = gym.wrappers.ResizeObservation(env, (42, 42))
        # env = gym.wrappers.ResizeObservation(env, (21, 21))
        # env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
        # env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information

        # env = env_wrapper(env)  # Wrap the environment if needed
        # delays = [1, 2, 4, 8, 16, 32]  # List of delay steps to test
        noises = [0.01, 0.05, 0.1, 0.2, 0.5]  # List of noise standard deviations to test
        for noise in noises:
        # for delay in delays:
            env = gym.make("LunarLander-v3")
            # env = delay_injection(env, delay_steps=delay, delay_type='obs')  # Inject a delay in the environment obs/act
            env = noise_injection(env, noise_std=noise, noise_type='obs')  # Inject Gaussian noise into the environment obs/act

            # run = wandb.init(
            #     entity = "arkapatra31-ust",
            #     project = "Test-Env-Wrappers",
            # )
            # wandb.run.name = f"{env.spec.id}"

            # obs = env.reset(seed=SEED)
            # env.action_space.seed(SEED)
            # print(f"Initial Observation: {obs}")
            # done = False
            # step_count = 0
            # while not done:
            #     action = env.action_space.sample()  # Sample a random action
            #     obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
            #     done = terminated or truncated  # Check if the episode is done
            #     step_count += 1
            #     print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}, Action: {action}")
            #     wandb.log({"Env_Step": step_count})

            #     for i in range(len(obs)):
            #         wandb.log({f"Observation_{i}": obs[i]})
            #     # for i in range(len(action)):
            #     #     wandb.log({f"Action_{i}": action[i]})
            #     wandb.log({f"Action": action})
            #     wandb.log({"Reward": reward})

            # wandb.finish()

            seed_everything()  # Seed everything before testing the agent
            n_episodes = 1000
            # model_path = f"Weights/LunarLander-v3_fun/4kupdates_d256_k16_c10_lr_0.0003_ent_0.0010"
            # model_path = "Weights/LunarLander-v3_A2C_weights/10000_alr0.001_ent0.01"
            # model_path = "Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_6_gb_2.0_xx"
            # model_path = "Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_5_gb_1.5261559756099234_xx"
            # model_path = "Weights/LunarLander-v3_OC/model_800000.0steps__nOpt_4_LR_0.0004_TermReg_0.000_xx.pt"
            # model_path = "Weights/LunarLanderContinuous-v3_TD3_alr_0.0001_clr_0.0003_xx"
            # model_path = "Weights/LunarLanderContinuous-v3_PPOWeights/1250_0.0001_0.001_0.0001"
            # model_path = "Weights/LunarLanderContinuous-v3_HAC_weights/HAC_LunarLanderContinuous-v3_1000000.0_0.0000_0.0004_5"
            model_path = "Weights/LunarLander-v3_RAINBOW/1000000.0_0.0001_51_3"
            episode_rewards, episode_lengths = agent.test_agent(env=env, n_episodes=n_episodes, record_stats=True, model_path=model_path)
            # print(f"Test Results: Episode Rewards = {episode_rewards}, Episode Lengths = {episode_lengths}")
            # print(f"Test Results: Average Reward = {np.mean(episode_rewards)}, Average Length = {np.mean(episode_lengths)}")
            sorted_rewards = sorted(episode_rewards)
            sorted_lengths = sorted(episode_lengths)
            reward_5th_percentile = np.percentile(sorted_rewards, 5)
            reward_95th_percentile = np.percentile(sorted_rewards, 95)
            length_5th_percentile = np.percentile(sorted_lengths, 5)
            length_95th_percentile = np.percentile(sorted_lengths, 95)
            reward_range = reward_95th_percentile - reward_5th_percentile
            length_range = length_95th_percentile - length_5th_percentile

            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            std_reward = np.std(episode_rewards)
            std_length = np.std(episode_lengths)
            cv_reward = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
            cv_length = std_length / abs(mean_length) if mean_length != 0 else float('inf')
            std_error_reward = std_reward / np.sqrt(len(episode_rewards))
            std_error_length = std_length / np.sqrt(len(episode_lengths))
            confidence_interval_95_reward = 1.96 * std_error_reward
            confidence_interval_95_length = 1.96 * std_error_length

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(episode_rewards, bins=20, kde=True, color='blue')
            plt.title(f'Distribution of Episode Rewards, Agent={agent.__class__.__name__}, Episodes={n_episodes}\nenv={env.spec.id}\nMean Reward={mean_reward:.2f}, Std Dev={std_reward:.2f}\n5th Percentile={reward_5th_percentile:.2f}, 95th Percentile={reward_95th_percentile:.2f}')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            sns.histplot(episode_lengths, bins=20, kde=True, color='green')
            plt.title(f'Distribution of Episode Lengths, Agent={agent.__class__.__name__}, Episodes={n_episodes}\nenv={env.spec.id}\nMean Length={mean_length:.2f}, Std Dev={std_length:.2f}\n5th Percentile={length_5th_percentile:.2f}, 95th Percentile={length_95th_percentile:.2f}')
            plt.xlabel('Episode Length')
            plt.ylabel('Frequency')

            plt.tight_layout()
            os.makedirs(f'stats_figs_{n_episodes}/{agent.__class__.__name__}', exist_ok=True)  # Create the directory if it doesn't exist
            plt.savefig(f'stats_figs_{n_episodes}/{agent.__class__.__name__}/{env.spec.id}_{agent.__class__.__name__}.png')  # Save the figure
            # plt.show()  # Show the figure

            write_stats = True  # Set to True if you want to write stats to a file
            if write_stats:
                # Append mode: adds to end of existing file
                with open(f"stats_figs_{n_episodes}/RAINBOW-LL-stats-{n_episodes}.txt", "a", encoding="utf-8") as file:
                    file.write(f"Environment: {env.spec.id}\tAgent: {agent.__class__.__name__}\tn_episodes: {n_episodes}\n")
                    file.write("-" * 80 + "\n")  # Separator line for clarity
                    # file.write("Episode Rewards: " + str(episode_rewards) + "\n")
                    # file.write("Episode Lengths: " + str(episode_lengths) + "\n")
                    file.write(f"Average Reward: {mean_reward}\tAverage Length: {mean_length}\n")
                    file.write(f"Standard Deviation Reward: {std_reward}\tStandard Deviation Length: {std_length}\n")
                    file.write(f"Coefficient of Variation Reward: {cv_reward}\tCoefficient of Variation Length: {cv_length}\n")
                    file.write(f"Std Error Reward: {std_error_reward}\tStd Error Length: {std_error_length}\n")
                    file.write(f"95% Confidence Interval Reward: {confidence_interval_95_reward}\t95% Confidence Interval Length: {confidence_interval_95_length}\n")
                    file.write(f"95% confidence interval as % of mean reward: {100 * confidence_interval_95_reward / abs(mean_reward) if mean_reward != 0 else float('inf'):.2f}%\n")
                    file.write(f"95% confidence interval as % of mean length: {100 * confidence_interval_95_length / abs(mean_length) if mean_length != 0 else float('inf'):.2f}%\n")
                    file.write(f"5th Percentile Reward: {reward_5th_percentile}\t95th Percentile Reward: {reward_95th_percentile}\n")
                    file.write(f"5th Percentile Length: {length_5th_percentile}\t95th Percentile Length: {length_95th_percentile}\n")
                    file.write(f"Range of Rewards (5th-95th Percentile): {reward_range}\n")
                    file.write(f"Range of Lengths (5th-95th Percentile): {length_range}\n")
                    file.write("-" * 80 + "\n")  # Separator line for clarity

        env.close()  # Close the environment after testing

    

def env_wrapper(env):
    # Wrap the environment with any necessary wrappers
    original_step = env.step
    original_reset = env.reset

    def preprocess_obs(obs):
        # obs = obs['image']  # Extract the image observation
        # obs = obs.transpose(2, 0, 1)  # Convert to channel-first format (C, H, W)
        # obs = obs / 255.0  # Normalize pixel values to [0, 1]
        # obs = obs.sum(axis=0)/3  # Convert the image to a single channel (grayscale)
        # observation = obs['observation']  # Extract the observation from the dictionary
        # desired_goal = obs['desired_goal']  # Extract the desired goal from the dictionary
        # achieved_goal = obs['achieved_goal']  # Extract the achieved goal from the dictionary
        # return observation, desired_goal, achieved_goal
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = original_step(action)
        obs = preprocess_obs(obs)
        # info['desired_goal'] = desired_goal  # Add the desired goal to the info dictionary
        # info['achieved_goal'] = achieved_goal  # Add the achieved goal to the info
        return obs, reward, terminated, truncated, info
    env.step = step.__get__(env)  # Bind the new step method to the environment instance

    def reset(self, seed=None, options=None):
        obs, info = original_reset(seed=seed, options=options)
        obs = preprocess_obs(obs)
        # info['desired_goal'] = desired_goal  # Add the desired goal to the info dictionary
        # info['achieved_goal'] = achieved_goal  # Add the achieved goal to the info
        return obs, info

    env.reset = reset.__get__(env)  # Bind the new reset method to the environment instance
    # env.observation_space = env.observation_space['observation']  # Update the observation space to match the new observation format
    env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 21, 21), dtype=np.float32)  # Update the observation space to match the new observation format
    # env.action_space = gym.spaces.Box(-1.0, 1.0, (4,))  # Update the action space 

    return env

# def delay_injection(env, delay_steps=5, delay_type='obs'):
#     if delay_steps == 0:
#         return env  # No delay to inject

#     if delay_type == 'obs':
#         """Injects a delay in the environment's observations."""
#         original_step = env.step

#         def delayed_step(action):
#             obs, reward, terminated, truncated, info = original_step(action)
#             # Introduce a delay by returning the previous observation for 'delay_steps' steps
#             if not hasattr(env, 'delay_buffer'):
#                 env.delay_buffer = [obs] * delay_steps
#             else:
#                 env.delay_buffer.append(obs)
#                 if len(env.delay_buffer) > delay_steps:
#                     env.delay_buffer.pop(0)
#             delayed_obs = env.delay_buffer[0]
#             # print(f"Delayed observation: {delayed_obs}")
#             return delayed_obs, reward, terminated, truncated, info

#     elif delay_type == 'act':
#         """Injects a delay in the environment's actions."""
#         original_step = env.step

#         def delayed_step(action):
#             if not hasattr(env, 'action_buffer'):
#                 env.action_buffer = [action] * delay_steps
#             else:
#                 env.action_buffer.append(action)
#                 if len(env.action_buffer) > delay_steps:
#                     env.action_buffer.pop(0)
#             delayed_action = env.action_buffer[0]
#             # wandb.log({"Action": delayed_action})
#             return original_step(delayed_action)

#     # elif delay_type == 'both':
#     #     """Injects a delay in both the environment's observations and actions."""
#     #     original_step = env.step

#     #     def delayed_step(action):
#     #         if not hasattr(env, 'action_buffer'):
#     #             env.action_buffer = [action] * delay_steps
#     #         else:
#     #             env.action_buffer.append(action)
#     #             if len(env.action_buffer) > delay_steps:
#     #                 env.action_buffer.pop(0)
#     #         delayed_action = env.action_buffer[0]

#     #         obs, reward, terminated, truncated, info = original_step(delayed_action)

#     #         if not hasattr(env, 'delay_buffer'):
#     #             env.delay_buffer = [obs] * delay_steps
#     #         else:
#     #             env.delay_buffer.append(obs)
#     #             if len(env.delay_buffer) > delay_steps:
#     #                 env.delay_buffer.pop(0)
#     #         delayed_obs = env.delay_buffer[0]
#     #         return delayed_obs, reward, terminated, truncated, info

#     elif delay_type == 'reward':
#         """Injects a delay in the environment's rewards."""
#         original_step = env.step

#         def delayed_step(action):
#             obs, reward, terminated, truncated, info = original_step(action)
#             if not hasattr(env, 'reward_buffer'):
#                 env.reward_buffer = [reward] * delay_steps
#             else:
#                 env.reward_buffer.append(reward)
#                 if len(env.reward_buffer) > delay_steps:
#                     env.reward_buffer.pop(0)
#             delayed_reward = env.reward_buffer[0]
#             return obs, delayed_reward, terminated, truncated, info

#     env.step = delayed_step
#     env.spec.id = env.spec.id + f"-delay-{delay_type}-{delay_steps}"
#     return env

from collections import deque
import numpy as np


def delay_injection(env, delay_steps=5, delay_type="obs"):

    if delay_steps == 0:
        return env

    original_reset = env.reset
    original_step = env.step

    if delay_type == "obs":

        env.obs_buffer = deque(maxlen=delay_steps + 1)

        def reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)

            env.obs_buffer.clear()

            # Repeat initial observation
            for _ in range(delay_steps + 1):
                env.obs_buffer.append(obs)

            return obs, info

        def step(action):

            obs, reward, terminated, truncated, info = \
                original_step(action)

            env.obs_buffer.append(obs)

            delayed_obs = env.obs_buffer[0]

            return (
                delayed_obs,
                reward,
                terminated,
                truncated,
                info
            )

    elif delay_type == "act":

        env.action_buffer = deque(
            maxlen=delay_steps + 1
        )

        def reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)

            env.action_buffer.clear()

            # Initially repeat a neutral action
            zero_action = np.zeros(
                env.action_space.shape,
                dtype=env.action_space.dtype
            )

            for _ in range(delay_steps + 1):
                env.action_buffer.append(zero_action.copy())

            return obs, info

        def step(action):

            env.action_buffer.append(
                np.asarray(action).copy()
            )

            delayed_action = env.action_buffer[0]

            return original_step(
                delayed_action
            )

    env.reset = reset
    env.step = step
    env.spec.id = env.spec.id + f"-delay-{delay_type}-{delay_steps}"

    return env
    
def noise_injection(env, noise_std=0.1, noise_type='obs'):
    if noise_std == 0:
        return env  # No noise to inject

    if noise_type == 'obs':
        """Injects Gaussian noise into the environment's observations as a percentage of their range."""
        original_step = env.step

        def noisy_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            o_range = env.observation_space.high - env.observation_space.low
            noisy_obs = np.clip(obs + np.random.normal(0, noise_std, size=obs.shape) * o_range, env.observation_space.low, env.observation_space.high)
            return noisy_obs, reward, terminated, truncated, info

    elif noise_type == 'act':
        if env.action_space.__class__.__name__ == 'Discrete':
            """Chooses random actions if with nois_std probability, otherwise takes the original action."""
            original_step = env.step

            def noisy_step(action):
                if np.random.rand() < noise_std:
                    action = env.action_space.sample()
                # wandb.log({"Action": action})
                return original_step(action)
        else:
            """Injects Gaussian noise into the environment's actions as a percentage of their range."""
            original_step = env.step

            def noisy_step(action):
                a_range = env.action_space.high - env.action_space.low
                noisy_action = np.clip(action + np.random.normal(0, noise_std, size=action.shape) * a_range, env.action_space.low, env.action_space.high)
                # wandb.log({"Action": noisy_action})
                return original_step(noisy_action)

    # elif noise_type == 'both':
    #     """Injects Gaussian noise into both the environment's observations and actions as a percentage of their ranges."""
    #     original_step = env.step
    #     if env.action_space.__class__.__name__ == 'Discrete':
    #         def noisy_step(action):
    #             if np.random.rand() < noise_std:
    #                 action = env.action_space.sample()
    #             obs, reward, terminated, truncated, info = original_step(action)
    #             o_range = env.observation_space.high - env.observation_space.low
    #             noisy_obs = np.clip(obs + np.random.normal(0, noise_std, size=obs.shape) * o_range, env.observation_space.low, env.observation_space.high)
    #             return noisy_obs, reward, terminated, truncated, info
    #     else:

    #         def noisy_step(action):
    #             a_range = env.action_space.high - env.action_space.low
    #             noisy_action = np.clip(action + np.random.normal(0, noise_std, size=action.shape) * a_range, env.action_space.low, env.action_space.high)
    #             obs, reward, terminated, truncated, info = original_step(noisy_action)
    #             o_range = env.observation_space.high - env.observation_space.low
    #             noisy_obs = np.clip(obs + np.random.normal(0, noise_std, size=obs.shape) * o_range, env.observation_space.low, env.observation_space.high)
    #             return noisy_obs, reward, terminated, truncated, info

    elif noise_type == 'reward':
        """Injects Gaussian noise into the environment's rewards"""
        original_step = env.step

        def noisy_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            noisy_reward = reward + np.random.normal(0, noise_std)
            return obs, noisy_reward, terminated, truncated, info


    env.step = noisy_step
    env.spec.id = env.spec.id + f"-noise-{noise_type}-{noise_std}"
    return env

def hyperparameter_tuning(env, state_dim, action_dim, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'average_reward_last_50', 'goal': 'maximize'},
        'parameters': {
            # 'LR': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            # 'NUM_OPTIONS': {'values': [2, 3, 4]},
            # 'TERM_REG': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 5e-3},
            # 'ENT_COEF': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            # 'EPSILON_DECAY': {'distribution': 'uniform', 'min': 0.99, 'max': 0.999},
            # 'GOAL_DIM': {'values': [2, 4]},
            # 'N_GOALS': {'values': [1, 2, 4]},
            # 'LR_ACTOR': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 3e-4},
            # 'LR_CRITIC': {'distribution': 'log_uniform_values', 'min': 3e-4, 'max': 1e-3},
            # 'EXPL_NOISE_MANAGER': {'distribution': 'uniform', 'min': 0.3, 'max': 0.6},
            # 'EXPL_NOISE_WORKER': {'distribution': 'uniform', 'min': 0.08, 'max': 0.13},
            # 'SUBGOAL_HORIZON': {'values': [5, 10]},
            # 'GOAL_BOUND': {'distribution': 'uniform', 'min': 1.5, 'max': 1.8},
            # 'GOAL_THRESHOLD': {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
            # 'N_LEVELS': {'values': [2, 3]},
            'SUBGOAL_TEST_FREQ': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
            'ALPHA': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1},
            # 'LAM': {'distribution': 'uniform', 'min': 0.9, 'max': 0.99},
        },
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project="HAC-HT-LL-b3", entity="arkapatra31-ust")

    # Define the training function for the sweep
    def train_sweep():
        seed_everything()
        with wandb.init() as run:
            config = wandb.config
            hyperparams = {
                # "LR": config.LR,
                # "NUM_OPTIONS": config.NUM_OPTIONS,
                # "TERM_REG": config.TERM_REG,
                # "ENT_COEF": config.ENT_COEF,
                # "EPSILON_DECAY": config.EPSILON_DECAY,
                # "ALPHA": config.ALPHA,
                # "GOAL_DIM": 8,
                # "N_GOALS": config.N_GOALS,
                "GOAL_BOUND": 0.7,
                "LR_ACTOR": 2e-5,
                "LR_CRITIC": 4e-4,
                "EXPL_NOISE_MANAGER": 1.23,
                "EXPL_NOISE_WORKER": 0.15,
                "SUBGOAL_HORIZON": 5,
                # "N_LEVELS": config.N_LEVELS,
                # "ALPHA": config.ALPHA,
                "GOAL_THRESHOLD": 0.7,
                # "LAM": config.LAM,
                "WANDB_UPLOAD": True,
                # "N_UPDATES": 100,
                "MAX_ENV_STEPS": 4e5,
                # "MAX_ACTION": 0.4,
            }
            # wandb.run.name = f"{hyperparams['LR']:.0e}_{hyperparams['ENT_COEF']:.0e}_{env.spec.id}"
            print(f"Starting a new sweep run with hyperparameters:\n {hyperparams}")
            agent = hac.HACAgent(state_dim, action_dim, percept=False, device=device, hyperparams=hyperparams)
            agent.train_agent(env=env, store_episode_rewards=True)
            # agent.test_agent(env=env, n_episodes=3)

    # Start the sweep agent to run the training function for each configuration
    wandb.agent(sweep_id, function=train_sweep, count=10)  # Adjust count as needed for the number of runs

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    seed_everything()
    main()
