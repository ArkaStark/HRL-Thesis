#!/bin/bash

source .hrl_env/bin/activate

# echo "HAC Hyperparameter Tuning on InvertedPendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hac --hyperparameter_tuning --env="InvertedPendulum-v5" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json"
# echo "HIRO Hyperparameter Tuning on InvertedPendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hiro --hyperparameter_tuning --env="InvertedPendulum-v5" --hyperparams_sweep_json="hiro_hyperparams_sweep_config.json"
# echo "A2C Hyperparameter Tuning"
# python3 benchmark.py --max_episode_steps=500 --agent=a2c --hyperparameter_tuning --hyperparams_sweep_json="a2c_hyperparams_sweep_config.json"


# echo "PPO Hyperparameter Tuning on PointMaze_UMazeDense-v3"
# python3 benchmark.py --agent=ppo --hyperparameter_tuning --hyperparams_sweep_json="ppo_hyperparams_sweep_config.json" --env="PointMaze_UMazeDense-v3" --env_wrapper --max_episode_steps=500
# echo "TD3 Hyperparameter Tuning on PointMaze_UMazeDense-v3"
# python3 benchmark.py --agent=td3 --hyperparameter_tuning --hyperparams_sweep_json="td3_hyperparams_sweep_config.json" --env="PointMaze_UMazeDense-v3" --env_wrapper --max_episode_steps=500
# echo "HIRO Hyperparameter Tuning on PointMaze_UMaze-v3"
# python3 benchmark.py --agent=hiro --hyperparameter_tuning --hyperparams_sweep_json="hiro_hyperparams_sweep_config.json" --env="PointMaze_UMaze-v3" --env_wrapper --max_episode_steps=500
# echo "HAC Hyperparameter Tuning on PointMaze_UMaze-v3"
# python3 benchmark.py --agent=hac --hyperparameter_tuning --hyperparams_sweep_json="hac_hyperparams_sweep_config.json" --env="PointMaze_UMaze-v3" --env_wrapper --max_episode_steps=500
python3 benchmark.py --agent=a2c --test_agent_stats --env="CartPole-v1" --model_path="Weights/CartPole-v1_A2C_weights/700_alr0.001021589217665543_clr_0.009676741439909976_ent0.007626425978885584-a" --test_episodes=1000 --max_episode_steps=500 --dist_levels="[0, 1, 2, 4, 8, 16, 32]" --dist_type="obs-delay"
python3 benchmark.py --agent=a2c --test_agent_stats --env="CartPole-v1" --model_path="Weights/CartPole-v1_A2C_weights/700_alr0.001021589217665543_clr_0.009676741439909976_ent0.007626425978885584-a" --test_episodes=1000 --max_episode_steps=500 --dist_levels="[1, 2, 4, 8, 16, 32]" --dist_type="act-delay"
python3 benchmark.py --agent=a2c --test_agent_stats --env="CartPole-v1" --model_path="Weights/CartPole-v1_A2C_weights/700_alr0.001021589217665543_clr_0.009676741439909976_ent0.007626425978885584-a" --test_episodes=1000 --max_episode_steps=500 --dist_levels="[0.01, 0.05, 0.1, 0.2, 0.5]" --dist_type="obs-noise"
python3 benchmark.py --agent=a2c --test_agent_stats --env="CartPole-v1" --model_path="Weights/CartPole-v1_A2C_weights/700_alr0.001021589217665543_clr_0.009676741439909976_ent0.007626425978885584-a" --test_episodes=1000 --max_episode_steps=500 --dist_levels="[0.01, 0.05, 0.1, 0.2, 0.5]" --dist_type="act-noise"
# python3 benchmark.py --agent=hiro --test_agent_stats --env="LunarLanderContinuous-v3" --model_path="Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_5_gb_0.6510136637683205_xx" --dist_levels="[0, 1, 2, 4, 8, 16, 32]" --dist_type="obs-delay"
# python3 benchmark.py --agent=hiro --test_agent_stats --env="LunarLanderContinuous-v3" --model_path="Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_5_gb_0.6510136637683205_xx" --dist_levels="[1, 2, 4, 8, 16, 32]" --dist_type="act-delay"
# python3 benchmark.py --agent=hiro --test_agent_stats --env="LunarLanderContinuous-v3" --model_path="Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_5_gb_0.6510136637683205_xx" --dist_levels="[0.01, 0.05, 0.1, 0.2, 0.5]" --dist_type="obs-noise"
# python3 benchmark.py --agent=hiro --test_agent_stats --env="LunarLanderContinuous-v3" --model_path="Weights/LunarLanderContinuous-v3_HIRO_weights/HIRO_LunarLanderContinuous-v3_C_5_gb_0.6510136637683205_xx" --dist_levels="[0.01, 0.05, 0.1, 0.2, 0.5]" --dist_type="act-noise"