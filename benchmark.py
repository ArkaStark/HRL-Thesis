from algos import a2c, ddpg, ppo, oc

import ogbench
import gymnasium as gym
import torch


def main():
    # Create an instance of the A2C class
    env = gym.make("LunarLander-v3")
    # envs = gym.make_vec("LunarLander-v3", num_envs=1, max_episode_steps=600)
    # dataset_name = 'pointmaze-medium-navigate-v0'  # Choose from 'cube-single-object-v0', 'cube-multi-object-v0', 'drawer-single-object-v0', 'drawer-multi-object-v0'.
    # env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    print(f"Environment: {env.spec.id}")
    # print(f"Action type: {env.action_space}, Observation type: {env.observation_space}")
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")

    # set the device
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # ddpg_agent = ddpg.DDPGAgent(obs_shape, action_shape, device=device)
    # oc_agent = oc.OCAgent(obs_shape, action_shape, num_options=3, lr=1e-4, device=device)
    
    # Call the train method to train the agent
    # ddpg_agent.train(env=env, hyperparams=None)
    # oc_agent.train(env=env, hyperparams=None)

    # Call the evaluate method to evaluate the agent
    # env = gym.make("LunarLander-v3", max_episode_steps=600, render_mode="rgb_array")
    # oc_agent.test(env=env, hyperparams=None)

    oc.hyperparameter_tuning(env, obs_shape, action_shape, device=device)


if __name__ == "__main__":
    main()