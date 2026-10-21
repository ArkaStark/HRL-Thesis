import argparse
import argcomplete
import shtab
import json
import yaml
import importlib

import torch
import numpy as np
import wandb
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import deque

# Environment imports
import ogbench
import gymnasium as gym
import ale_py
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import gymnasium_robotics
from yaml import parser

gym.register_envs(gymnasium_robotics)
gym.register_envs(ale_py)
gym.register_envs(minigrid)

SEED = 31


def main():

    # DECLARE ENVIRONMENT

    env = gym.make(args.env, render_mode=args.render_mode, max_episode_steps=args.max_episode_steps)
    # env = gym.make("LunarLanderContinuous-v3", enable_wind=False, wind_power=1.0, turbulence_power=0.1, render_mode="human")


    # env = gym.make("InvertedPendulum-v5", max_episode_steps=500)
    # env = gym.make("AntMaze_UMaze-v5", max_episode_steps=1000, render_mode="human")#, exclude_current_positions_from_observation=False)

    # env = gym.make("MiniGrid-FourRooms-v0")
    # env = FullyObsWrapper(env)  # Use fully observable state instead of partial observations
    # env = ImgObsWrapper(env)  # Use image observations instead of dictionary observations
    # env.action_space = gym.spaces.Discrete(3)  # Update the action space to have 3 discrete actions


    # dataset_name = 'antmaze-medium-navigate-v0'  # Choose from 'cube-single-object-v0', 'cube-multi-object-v0', 'drawer-single-object-v0', 'drawer-multi-object-v0'.
    # env = ogbench.make_env_and_datasets(dataset_name, env_only=True)

    # obs_shape = env.observation_space
    # action_shape = env.action_space

    # env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # # env = gym.wrappers.ResizeObservation(env, (42, 42))
    # env = gym.wrappers.ResizeObservation(env, (21, 21))
    # env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
    # env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information

    if args.env_wrapper:
        print(f"Applying environment wrapper !!")
        env = env_wrapper(env)  # Wrap the environment if needed

    # env = noise_injection(env, noise_std=0.2, noise_type='obs')  # Inject Gaussian noise into the environment obs/act


    # run = wandb.init(
    #     entity = "arkapatra31-ust",
    #     project = "Test-Env-Wrappers",
    # )
    # wandb.run.name = f"{env.spec.id}"

    # obs = env.reset(seed=args.seed)
    # env.action_space.seed(args.seed)
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

    if len(env.observation_space.shape) == 1:
        obs_shape = env.observation_space.shape[0]
    else:
        obs_shape = env.observation_space.shape

    if env.action_space.__class__.__name__ == 'Discrete':
        action_shape = env.action_space.n
    else:
        action_shape = env.action_space.shape[0]

    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    print(f"Environment: {env.spec.id}")
    print(f"Observation type: {env.observation_space}, Action type: {env.action_space}")
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")

    # SET THE DEVICE
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # SET HYPERPARAMETERS as desired (check the agent's class for available hyperparameters, default values and definitions)

    if args.hyperparams_json:
        with open(args.hyperparams_json, "r") as f:
            hyperparameters = json.load(f)
    elif args.hyperparams_yaml:
        with open(args.hyperparams_yaml, "r") as f:
            hyperparameters = yaml.safe_load(f)
    else:

        hyperparameters = {
            # "WANDB_UPLOAD": True,
            # "SAVE_MODEL": True,

            # "LR": 6e-4,                       # OC, FuN, A2C
            # "LR_ACTOR": 0.001021589217665543,                 # DDPG, TD3, HIRO, HAC
            # "LR_CRITIC": 0.009676741439909976,                # DDPG, TD3, HIRO, HAC
            # "GAMMA": 0.9856086048981776,                    # OC, DDPG
            # "GAMMA_M": 0.99,                  # FuN, HIRO
            # "GAMMA_W": 0.95,                  # FuN, HIRO
            # "LAM": 0.9647539603130756,
            # "PPO_EPOCHS": 8,                    # PPO

            # "NUM_OPTIONS": 4,                 # OC
            # "TERM_REG": 0.05701939380921457,                 # OC
            # "EPSILON_DECAY": 0.997939454548163,           # OC
            # "ENT_COEF": 0.007626425978885584,                 # OC, A2C, PPO
            
            # "MAX_ACTION": 3.0,                # DDPG, TD3
            # "TAU": 0.0095,                      # DDPG, TD3 
            # 'BATCH_SIZE': 256,                     # DDPG, TD3, HIRO, HAC
            # 'UPDATE_DELAY': 1,                    # TD3
            

            # "GOAL_DIM": 2,                  # FuN, HIRO
            # "N_GOALS": 1,                    # FuN
            # "SUBGOAL_HORIZON": 10,            # FuN, HIRO, HAC

            # "MAX_EPISODES": 2000,               # OC, DDPG, TD3, HIRO, HAC
            # "N_UPDATES": 1000,                # FuN

            # "ALPHA": 0.7532654926423844,                     # FuN, HAC, HER, RAINBOW
            # 'EXPL_NOISE_MANAGER': 1.994184625139012,          # HIRO, HAC
            # 'EXPL_NOISE_WORKER': 1.3807217004016186,           # HIRO, HAC, TD3
            # 'GOAL_BOUND': 1.174062547500722,                  # HIRO, HAC
            # 'GOAL_THRESHOLD': 1e-3,              # HAC
            # 'N_LEVELS': 3,                      # HAC
            # "MAX_ENV_STEPS": 7e4,              # OC, DDPG, TD3, HIRO, HAC
            # 'SUBGOAL_TEST_FREQ': 0.6,                  # HAC
            # 'N_STEP': 5,                        # RAINBOW
            # 'ATOM_SIZE': 8,                    # RAINBOW
            # 'V_MAX': 300,                     # RAINBOW            
        }

        hyperparameters['SEED'] = args.seed


    # INITIALIZE AGENTS

    agent_name = args.agent.upper()+"Agent"
    print(f"Initializing agent: {agent_name}")
    agent_module = importlib.import_module("algos."+args.agent)
    agent_class = getattr(agent_module, agent_name)
    if agent_class:
        agent = agent_class(obs_shape, action_shape, percept=args.percept, device=device, hyperparams=hyperparameters)
    else:
        raise ValueError(f"Agent class {agent_name} not found in module {args.agent}")

    # agent = ppo.PPOAgent(obs_shape, action_shape, percept=False, device=device, hyperparams=hyperparameters)


    # PERFORM HYPERPARAMETER TUNING

    if args.hyperparameter_tuning:
        print(f"Performing hyperparameter tuning for agent: {agent_name} on environment: {env.spec.id}")
        if args.hyperparams_sweep_json:
            with open(args.hyperparams_sweep_json, "r") as f:
                hyperparameters_sweep = json.load(f)
        elif args.hyperparams_sweep_yaml:
            with open(args.hyperparams_sweep_yaml, "r") as f:
                hyperparameters_sweep = yaml.safe_load(f)
        else:
            hyperparameters_sweep = None
        hyperparameter_tuning(env=env, args=args, state_dim=obs_shape, action_dim=action_shape, hyp_sweep=hyperparameters_sweep, device=device)


    # TRAIN AGENTS

    if args.train_agent:
        print(f"Training agent: {agent_name} on environment: {env.spec.id}")
        agent.train_agent(env=env)

    # VISUALIZE VALUE FUNCTION

    # model_path = "Weights/LunarLander-v3_fun/2500updates_d256_k16_c10_lr_0.0003_ent_0.0010/model.pt"
    # visualize_values(agent=agent, env=env, model=model_path, axis1=0, axis2=1)

    # TEST AGENTS

    if args.test_agent:
        print(f"Testing agent: {agent_name} on environment: {env.spec.id}")

        for dist in args.dist_levels:
            env = gym.make(args.env, render_mode=args.render_mode, max_episode_steps=args.max_episode_steps)
            if args.env_wrapper:
                print(f"Applying environment wrapper !!")
                env = env_wrapper(env)  # Wrap the environment if needed

            dist_type = args.dist_type.split('-')[1]
            dist_space = args.dist_type.split('-')[0]
            if dist_type == 'delay':
                env = delay_injection(env, delay_steps=int(dist), delay_type=dist_space)  # Inject a delay in the environment obs/act
            elif dist_type == 'noise':
                env = noise_injection(env, noise_std=float(dist), noise_type=dist_space)  # Inject Gaussian noise into the environment obs/act
            agent.test_agent(env=env, n_episodes=2, model_path=args.model_path)


    # CREATE TEST ENVIRONMENT FOR EVALUATION

    if args.test_agent_stats:
        print(f"Testing agent: {agent_name} on environment: {env.spec.id} for statistics collection")

        # delays = [1, 2, 4, 8, 16, 32]  # List of delay steps to test
        # noises = [0.01, 0.05, 0.1, 0.2, 0.5]  # List of noise standard deviations to test
        for dist in args.dist_levels:
            env = gym.make(args.env, render_mode=args.render_mode, max_episode_steps=args.max_episode_steps)

            if args.env_wrapper:
                print(f"Applying environment wrapper !!")
                env = env_wrapper(env)  # Wrap the environment if needed
            
            dist_type = args.dist_type.split('-')[1]
            dist_space = args.dist_type.split('-')[0]
            if dist_type == 'delay':
                env = delay_injection(env, delay_steps=int(dist), delay_type=dist_space)  # Inject a delay in the environment obs/act
            elif dist_type == 'noise':
                env = noise_injection(env, noise_std=float(dist), noise_type=dist_space)  # Inject Gaussian noise into the environment obs/act

            # run = wandb.init(
            #     entity = "arkapatra31-ust",
            #     project = "Test-Env-Wrappers",
            # )
            # wandb.run.name = f"{env.spec.id}"

            # obs = env.reset(seed=args.seed)
            # env.action_space.seed(args.seed)
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

            seed_everything(args.seed)  # Seed everything before testing the agent
            n_episodes = args.test_episodes
            # model_path = f"Weights/LunarLander-v3_fun/2kupdates_d256_k16_c10_lr_0.0003_ent_0.0010"
            # model_path = "Weights/LunarLander-v3_A2C_weights/10000_alr0.001_ent0.01"
            # model_path = "Weights/CartPole-v1_A2C_weights/1000_alr0.0005_clr_0.0045_ent0.0007-a"
            # model_path = "Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_5_gb_1.5261559756099234_xx"
            # model_path = "Weights/LunarLander-v3_OC/model_800000.0steps__nOpt_4_LR_0.0004_TermReg_0.000_xx.pt"
            # model_path = "Weights/LunarLanderContinuous-v3_TD3_alr_0.0001_clr_0.0003_xx"
            # model_path = "Weights/LunarLanderContinuous-v3_PPOWeights/1250_0.0001_0.001_0.0001"
            # model_path = "Weights/InvertedPendulum-v5_PPOWeights/250_0.0002_0.0002_0.0007"
            # model_path = "Weights/LunarLanderContinuous-v3_HAC_weights/HAC_LunarLanderContinuous-v3_1000000.0_0.0000_0.0004_5"
            # model_path = "Weights/LunarLander-v3_RAINBOW/1000000.0_0.0001_51_3"
            model_path = args.model_path
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
            os.makedirs(f'stats/IP_stats_figs_{n_episodes}/{agent.__class__.__name__}', exist_ok=True)  # Create the directory if it doesn't exist
            plt.savefig(f'stats/IP_stats_figs_{n_episodes}/{agent.__class__.__name__}/{env.spec.id}_{agent.__class__.__name__}.png')  # Save the figure
            # plt.show()  # Show the figure

            write_stats = True  # Set to True if you want to write stats to a file
            if write_stats:
                # Append mode: adds to end of existing file
                with open(f"stats/IP_stats_figs_{n_episodes}/{args.agent.upper()}-LL-stats-{n_episodes}.txt", "a", encoding="utf-8") as file:
                    file.write(f"Environment: {env.spec.id}\tAgent: {agent.__class__.__name__}\tn_episodes: {n_episodes}\n")
                    file.write(f"Model Path: {model_path}\n")
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


def visualize_values(agent, env, model, axis1=0, axis2=1, num_points=20, avg_over=100):
    agent.load_state_dict(torch.load(model, map_location=agent.device))
    obs1_high = 2#env.observation_space.high[axis1]
    obs1_low = -2#env.observation_space.low[axis1]
    obs2_high = env.observation_space.high[axis2]
    obs2_low = 0#env.observation_space.low[axis2]
    obs1 = np.linspace(obs1_low, obs1_high, num_points)
    obs2 = np.linspace(obs2_low, obs2_high, num_points)
    # sum_values_W = np.zeros((num_points, num_points))
    sum_values = np.zeros((num_points, num_points))
    for k in range(avg_over):
        print(f"Evaluating value function: {k+1}/{avg_over}")
        for j in range(num_points):
            for i in range(num_points):
                obs = env.observation_space.sample()  # Sample a random observation
                obs[axis1] = obs1[i]
                obs[axis2] = obs2[j]
                value = agent.get_values(obs)
                # print(f"Observation: {obs}, Value: {value}")
                sum_values[j, i] += value[1] # visualize j,i is y,x in the plot
                # sum_values_W[j, i] += value_W


    avg_values = sum_values / avg_over
    # avg_values_W = sum_values_W / avg_over
    plt.imshow(avg_values, extent=(obs1_low, obs1_high, obs2_low, obs2_high), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Value Function')
    plt.title(f'Value Function over {avg_over} runs')
    plt.xlabel(f'Observation {axis1}')
    plt.ylabel(f'Observation {axis2}')
    plt.show()


def env_wrapper(env):
    # Wrap the environment with any necessary wrappers
    # env = FullyObsWrapper(env)  # Use fully observable state instead of partial observations
    # env = ImgObsWrapper(env)  # Use image observations instead of dictionary observations
    # env = gym.wrappers.ResizeObservation(env, (21, 21))
    # env = gym.wrappers.NormalizeObservation(env)  # Normalize pixel values to [0, 1]
    # env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 frames to capture motion information
    original_step = env.step
    original_reset = env.reset

    def preprocess_obs(observation):
        # obs = observation  # Use the observation as is
        # obs = observation['image']  # Extract the image observation
        # obs = obs.transpose(2, 0, 1)  # Convert to channel-first format (C, H, W)
        # observation = obs / 255.0  # Normalize pixel values to [0, 1]
        # obs = obs.sum(axis=0)/3  # Convert the image to a single channel (grayscale)
        obs = observation['observation']  # Extract the observation from the dictionary
        desired_goal = observation['desired_goal']  # Extract the desired goal from the dictionary
        achieved_goal = observation['achieved_goal']  # Extract the achieved goal from the dictionary
        # obs = np.concatenate([achieved_goal,obs])  # Concatenate the observation and desired goal
        # obs = obs[:29]
        # obs = np.concatenate([obs, np.zeros(obs.shape)])
        return obs, desired_goal, achieved_goal
        # return obs, None, None  # Return None for desired_goal and achieved_goal since they are not used

    def step(self, action):
        obs, reward, terminated, truncated, info = original_step(action)
        obs, desired_goal, achieved_goal = preprocess_obs(obs)
        info['desired_goal'] = desired_goal  # Add the desired goal to the info dictionary
        info['achieved_goal'] = achieved_goal  # Add the achieved goal to the info
        return obs, reward, terminated, truncated, info
    
    env.step = step.__get__(env)  # Bind the new step method to the environment instance

    def reset(self, seed=None, options=None):
        obs, info = original_reset(seed=seed, options=options)
        obs, desired_goal, achieved_goal = preprocess_obs(obs)
        info['desired_goal'] = desired_goal  # Add the desired goal to the info dictionary
        info['achieved_goal'] = achieved_goal  # Add the achieved goal to the info
        return obs, info

    env.reset = reset.__get__(env)  # Bind the new reset method to the environment instance
    # env.observation_space = env.observation_space['observation']  # Update the observation space to match the new observation format
    # env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 21, 21), dtype=np.float32)  # Update the observation space to match the new observation format
    # env.action_space = gym.spaces.Box(-1.0, 1.0, (4,))  # Update the action space 
    # env.action_space = gym.spaces.Discrete(3)  # Update the action space to have 3 discrete actions
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.reset()[0].shape, dtype=np.float32)  # Update the observation space to match the new observation format
    env.spec.id = env.spec.id + "-wrapped"  # Update the environment ID to indicate that it has been wrapped

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


def delay_injection(env, delay_steps=5, delay_type="obs", fixed_delay=True):

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

            if fixed_delay:
                delayed_obs = env.obs_buffer[0]
            else:
                # Randomly select a delayed observation from the buffer
                delayed_obs = env.obs_buffer[np.random.randint(0, len(env.obs_buffer))]

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

            if fixed_delay:
                delayed_action = env.action_buffer[0]
            else:
                # Randomly select a delayed action from the buffer
                delayed_action = env.action_buffer[np.random.randint(0, len(env.action_buffer))]

            return original_step(
                delayed_action
            )

    env.reset = reset
    env.step = step
    env.spec.id = env.spec.id + f"-delay-{delay_type}-{delay_steps}"
    if fixed_delay:
        env.spec.id += "-fixed"

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
            o_range = np.nan_to_num(o_range, nan=0.0, posinf=10.0, neginf=-10.0)  # Replace NaN values with 1.0 to avoid multiplication issues
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
                a_range = np.nan_to_num(a_range, nan=0.0, posinf=10.0, neginf=-10.0)  # Replace NaN values with 1.0 to avoid multiplication issues
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

def hyperparameter_tuning(env, args, state_dim, action_dim, hyp_sweep, device="cpu"):
    # Define a grid of hyperparameters to sweep over
    if hyp_sweep:
        sweep_config = hyp_sweep
    else:
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'mean_episode_return', 'goal': 'maximize'},
            'parameters': {
                # 'LR': {'distribution': 'uniform', 'min': 1e-3, 'max': 4e-3},
                # 'GAMMA': {'distribution': 'uniform', 'min': 0.95, 'max': 0.99},
                'GAMMA_W': {'distribution': 'uniform', 'min': 0.9, 'max': 0.95},
                'GAMMA_M': {'distribution': 'uniform', 'min': 0.95, 'max': 0.99},
                # 'NUM_OPTIONS': {'values': [2, 3, 4]},
                # 'TERM_REG': {'distribution': 'log_uniform_values', 'min': 3e-4, 'max': 3e-1},
                # 'ENT_COEF': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
                # 'EPSILON_DECAY': {'distribution': 'uniform', 'min': 0.99, 'max': 0.999},
                # 'GOAL_DIM': {'values': [2, 4]},
                # 'N_GOALS': {'values': [1, 2, 4]},
                # 'LR_ACTOR': {'distribution': 'log_uniform_values', 'min': 3e-5, 'max': 3e-4},
                # 'LR_CRITIC': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
                # 'EXPL_NOISE_MANAGER': {'distribution': 'uniform', 'min': 0.3, 'max': 0.6},
                # 'EXPL_NOISE_WORKER': {'distribution': 'uniform', 'min': 0.05, 'max': 0.25},
                # 'SUBGOAL_HORIZON': {'values': [2, 5, 10]},
                # 'GOAL_BOUND': {'distribution': 'uniform', 'min': 1.5, 'max': 1.8},
                # 'GOAL_THRESHOLD': {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
                # 'N_LEVELS': {'values': [2, 3]},
                # 'SUBGOAL_TEST_FREQ': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
                # 'ALPHA': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                # 'LAM': {'distribution': 'uniform', 'min': 0.95, 'max': 0.99},
                # "BATCH_SIZE": {"values": [64, 128, 256]},
                # 'UPDATE_DELAY': {'values': [1, 2, 4]},
                # 'TAU': {'distribution': 'uniform', 'min': 1e-3, 'max': 1e-2},
                # 'PPO_EPOCHS': {'values': [2, 4, 8]},
            },
            # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
        }

    # Initialize a W&B sweep
    sweep_id = wandb.sweep(sweep_config, project=f"{args.agent}-HT-{args.env}", entity=args.wandb_entity)

    # Define the training function for the sweep
    def train_sweep():

        seed_everything(args.seed)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        with wandb.init() as run:
            config = wandb.config
            print(f"Starting a new sweep run with hyperparameter config:\n {config}")

            hyperparams = {
                'MAX_ENV_STEPS': 1e5, #args.max_env_steps,
                'GOAL_DIM': 2,
                'WANDB_UPLOAD': True,
            }
            for key in config.keys():
                hyperparams[key] = config[key]
            # wandb.run.name = f"{hyperparams['LR']:.0e}_{hyperparams['ENT_COEF']:.0e}_{env.spec.id}"

            agent_name = args.agent.upper()+"Agent"
            print(f"Initializing agent: {agent_name}")
            agent_module = importlib.import_module("algos."+args.agent)
            agent_class = getattr(agent_module, agent_name)
            if agent_class:
                agent = agent_class(state_dim, action_dim, percept=args.percept, device=device, hyperparams=hyperparams)
            else:
                raise ValueError(f"Agent class {agent_name} not found in module {args.agent}")
            
            # agent = fun.FUNAgent(state_dim, action_dim, percept=False, device=device, hyperparams=hyperparams)
            agent.train_agent(env=env, store_episode_rewards=True)
            # agent.test_agent(env=env, n_episodes=3)

    # Start the sweep agent to run the training function for each configuration
    wandb.agent(sweep_id, function=train_sweep, count=args.sweep_count)  # Adjust count as needed for the number of runs

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.use_deterministic_algorithms(True)  # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking RL Agents")

    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment ID")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array"], help="Render mode for the environment")
    parser.add_argument("--agent", type=str, default="a2c", choices=["a2c", "ppo", "oc", "rainbow", "td3", "fun", "hiro", "hac", "ppoc"], help="RL agent to use")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA and use CPU instead")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--env_wrapper", action="store_true", help="Use environment wrapper for preprocessing observations")
    parser.add_argument("--percept", action="store_true", help="Use perceptual module in the agent")
    

    parser.add_argument("--max_env_steps", type=int, default=1e5, help="Maximum number of steps per episode for the environment")
    parser.add_argument("--train_agent", action="store_true", help="Train the agent")
    parser.add_argument("--hyperparams_json", type=str, default=None, help="Path to a JSON file containing hyperparameters for training")
    parser.add_argument("--hyperparams_yaml", type=str, default=None, help="Path to a YAML file containing hyperparameters for training")
    parser.add_argument("--test_agent", action="store_true", help="Test the agent")

    parser.add_argument("--test_agent_stats", action="store_true", help="Test the agent and record statistics")  
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model for testing")
    parser.add_argument("--test_episodes", type=int, default=1000, help="Number of episodes for testing")
    parser.add_argument("--dist_levels", type=json.loads, default="[0]", help="Disturbance levels to inject into the environment")
    parser.add_argument("--dist_type", type=str, default="obs-delay", choices=["obs-delay", "act-delay", "obs-noise", "act-noise"], help="Type of disturbance to inject into the environment")
    # parser.add_argument("--record_stats", action="store_true", help="Record statistics during testing")
    parser.add_argument("--save_stats_figs", action="store_true", help="Save statistics figures during testing")
    parser.add_argument("--write_stats", action="store_true", help="Write statistics to a file during testing")
    parser.add_argument("--hyperparameter_tuning", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--sweep_count", type=int, default=20, help="Number of hyperparameter sweep runs")
    parser.add_argument("--hyperparams_sweep_json", type=str, default=None, help="Path to a JSON file containing hyperparameters sweep dictionary for hyperparameter tuning")
    parser.add_argument("--hyperparams_sweep_yaml", type=str, default=None, help="Path to a YAML file containing hyperparameters sweep dictionary for hyperparameter tuning")

    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name for logging")
    parser.add_argument("--wandb_project", type=str, default="RL-Benchmark", help="W&B project name for logging")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main()
