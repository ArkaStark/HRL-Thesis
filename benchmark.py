from algos import a2c, ddpg, oc, td3, her, hac, fddqn

import ogbench
import gymnasium as gym
import torch
import ale_py
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
gym.register_envs(ale_py)
gym.register_envs(minigrid)

import numpy as np


def main():

    # DECLARE ENVIRONMENT

    env = gym.make("FetchSlide-v4")
    # envs = gym.make_vec("LunarLander-v3", num_envs=1, max_episode_steps=600)

    # env = gym.make("MiniGrid-FourRooms-v0")
    # env = FullyObsWrapper(env)  # Use fully observable state instead of partial observations
    # env = ImgObsWrapper(env)  # Use image observations instead of dictionary observations
    # env.action_space = gym.spaces.Discrete(3)  # Update the action space to have 3 discrete actions
    
    # obs = env.reset()
    # print(f"Initial Observation: {obs}")
    # done = False
    # while not done:
    #     action = env.action_space.sample()  # Sample a random action
    #     obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
    #     done = terminated or truncated  # Check if the episode is done
    #     print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}, Action: {action}")
        # print(f"Observation shape: {obs.shape}, action shape: {action.shape}")


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

    env = env_wrapper(env)  # Wrap the environment if needed

    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    # action_shape = env.action_space.n

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

    hyperparameters = {
        "wandb_upload": True,
        "save_model": True,
    }

    # INITIALIZE AGENTS

    # a2c_agent = a2c.A2CAgent(obs_shape, action_shape, hyperparams=hyperparameters, device=device)
    # ddpg_agent = ddpg.DDPGAgent(obs_shape, action_shape, hyperparams=hyperparameters, device=device)
    # oc_agent = oc.OCAgent(obs_shape, action_shape, percept=False, device=device, hyperparams=hyperparameters)
    # td3_agent = td3.TD3Agent(obs_shape, action_shape, hyperparams=hyperparameters, device=device)
    her_agent = her.HERAgent(obs_shape, action_shape, hyperparams=hyperparameters, device=device)
    # hac_agent = hac.HACAgent(obs_shape, action_shape, device=device)
    # fddqn_agent = fddqn.FDDQNAgent(obs_shape, action_shape, device=device)



    # TRAIN AGENTS

    # a2c_agent.train_agent(env=env)
    # ddpg_agent.train_agent(env=env)
    # td3_agent.train_agent(env=env)
    # oc_agent.train_agent(env=env) #, noise_std=0.1, noise_type="observation")
    # hac_agent.train_agent(env=env)
    her_agent.train_agent(env=env)
    # fddqn_agent.train_agent(env=env)

    # PERFORM HYPERPARAMETER TUNING

    # a2c.hyperparameter_tuning(env=env, state_dim=obs_shape, action_dim=action_shape, device=device)
    # oc.hyperparameter_tuning(env=env, state_dim=obs_shape, action_dim=action_shape, percept=True, device=device)
    # fddqn.hyperparameter_tuning(env=env, state_dim=obs_shape, action_dim=action_shape, device=device)
    # her.hyperparameter_tuning(env=env, state_dim=obs_shape, action_dim=action_shape, device=device)

    # CREATE TEST ENVIRONMENT FOR EVALUATION


    # Call the evaluate method to evaluate the agent
    # env = gym.make("LunarLanderContinuous-v3", max_episode_steps=600, render_mode="human")
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

    # TEST AGENTS

    # oc_agent.test_agent(env=env, n_episodes=5, noise_std=0.1, noise_type="observation")
    # ddpg_agent.test_agent(env=env, n_episodes=5, delay_type="observation", delay_steps=10)
    # td3_agent.test_agent(env=env, n_episodes=5, delay_type="action", delay_steps=5)
    # her_agent.test_agent(env=env, n_episodes=5)
    # hac_agent.test_agent(env=env, n_episodes=5)

def env_wrapper(env):
    # Wrap the environment with any necessary wrappers
    original_step = env.step
    original_reset = env.reset

    def preprocess_obs(obs):
        # obs = obs['image']  # Extract the image observation
        # obs = obs.transpose(2, 0, 1)  # Convert to channel-first format (C, H, W)
        # obs = obs.sum(axis=0)/(3*255.0)  # Convert the image to a single channel (grayscale)
        observation = obs['observation']  # Extract the observation from the dictionary
        desired_goal = obs['desired_goal']  # Extract the desired goal from the dictionary
        achieved_goal = obs['achieved_goal']  # Extract the achieved goal from the dictionary
        return observation, desired_goal, achieved_goal

    def step(self, action):
        obs, reward, terminated, truncated, info = original_step(action)
        obs, desired_goal, achieved_goal = preprocess_obs(obs)
        return obs, desired_goal, achieved_goal, reward, terminated, truncated, info
    env.step = step.__get__(env)  # Bind the new step method to the environment instance

    def reset(self, seed=None, options=None):
        obs, info = original_reset(seed=seed, options=options)
        obs, desired_goal, achieved_goal = preprocess_obs(obs)
        return obs, desired_goal, achieved_goal, info
    env.reset = reset.__get__(env)  # Bind the new reset method to the environment instance
    env.observation_space = env.observation_space['observation']  # Update the observation space to match the new observation format

    env.action_space = gym.spaces.Box(-1.0, 1.0, (4,))  # Update the action space 

    return env


if __name__ == "__main__":
    main()
