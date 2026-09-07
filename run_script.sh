#!/bin/bash

source .hrl_env/bin/activate

# echo "HAC Hyperparameter Tuning on InvertedPendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hac --hyperparameter_tuning --env="InvertedPendulum-v5" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json"
echo "HIRO Hyperparameter Tuning on InvertedPendulum-v5"
python3 benchmark.py --max_episode_steps=500 --agent=hiro --hyperparameter_tuning --env="InvertedPendulum-v5" --hyperparams_sweep_json="hiro_hyperparams_sweep_config.json"
# echo "RAINBOW Hyperparameter Tuning"
# python3 benchmark.py --max_episode_steps=500 --agent=rainbow --hyperparameter_tuning --hyperparams_sweep_json="rainbow_hyperparams_sweep_config.json"

# echo "A2C Hyperparameter Tuning on Acrobot-v1"
# python3 benchmark.py --max_episode_steps=500 --agent=a2c --hyperparameter_tuning --env="Acrobot-v1" --hyperparams_sweep_json="a2c_hyperparams_sweep_config.json"
# echo "OC Hyperparameter Tuning on Acrobot-v1"
# python3 benchmark.py --max_episode_steps=500 --agent=oc --hyperparameter_tuning --env="Acrobot-v1" --hyperparams_sweep_json="oc_hyperparams_sweep_config.json"
# echo "FUN Hyperparameter Tuning on Acrobot-v1"
# python3 benchmark.py --max_episode_steps=500 --agent=fun --hyperparameter_tuning --env="Acrobot-v1" --hyperparams_sweep_json="fun_hyperparams_sweep_config.json"
# echo "TD3 Hyperparameter Tuning on InvertedDoublePendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=td3 --hyperparameter_tuning --env="InvertedDoublePendulum-v5" --hyperparams_sweep_json="td3_hyperparams_sweep_config.json"
# echo "PPO Hyperparameter Tuning on InvertedDoublePendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=ppo --hyperparameter_tuning --env="InvertedDoublePendulum-v5" --hyperparams_sweep_json="ppo_hyperparams_sweep_config.json"
# echo "HIRO Hyperparameter Tuning on InvertedDoublePendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hiro --hyperparameter_tuning --env="InvertedDoublePendulum-v5" --hyperparams_sweep_json="hiro_hyperparams_sweep_config.json"
# echo "HAC Hyperparameter Tuning on InvertedDoublePendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hac --hyperparameter_tuning --env="InvertedDoublePendulum-v5" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json"

# echo "HAC Hyperparameter Tuning on PointMaze_UMaze-v5"
# python3 benchmark.py --agent=hac --hyperparameter_tuning --env="PointMaze_UMaze-v3" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json" --env_wrapper --sweep_count=10
# echo "HAC Hyperparameter Tuning on AntMaze_UMazeDense-v5"
# python3 benchmark.py --agent=hac --hyperparameter_tuning --env="AntMaze_UMazeDense-v5" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json" --env_wrapper --sweep_count=10
# echo "HAC Hyperparameter Tuning on AntMaze_UMaze-v5"
# python3 benchmark.py --agent=hac --hyperparameter_tuning --env="AntMaze_UMaze-v5" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json" --env_wrapper --sweep_count=10

# echo "PPO Hyperparameter Tuning on Humanoid-v5"
# python3 benchmark.py --max_episode_steps=1000 --agent=ppo --hyperparameter_tuning --env="Humanoid-v5" --hyperparams_sweep_json="ppo_hyperparams_sweep_config.json" --sweep_count=10

