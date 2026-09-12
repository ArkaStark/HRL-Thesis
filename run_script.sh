#!/bin/bash

source .hrl_env/bin/activate

# echo "HAC Hyperparameter Tuning on InvertedPendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hac --hyperparameter_tuning --env="InvertedPendulum-v5" --hyperparams_sweep_json="hac_hyperparams_sweep_config.json"
# echo "HIRO Hyperparameter Tuning on InvertedPendulum-v5"
# python3 benchmark.py --max_episode_steps=500 --agent=hiro --hyperparameter_tuning --env="InvertedPendulum-v5" --hyperparams_sweep_json="hiro_hyperparams_sweep_config.json"
# echo "RAINBOW Hyperparameter Tuning"
# python3 benchmark.py --max_episode_steps=500 --agent=rainbow --hyperparameter_tuning --hyperparams_sweep_json="rainbow_hyperparams_sweep_config.json"




echo "PPO Hyperparameter Tuning on FetchSlideDense-v4"
python3 benchmark.py --agent=ppo --hyperparameter_tuning --hyperparams_sweep_json="ppo_hyperparams_sweep_config.json" --env="FetchSlideDense-v4" --env_wrapper --max_episode_steps=50
echo "HIRO Hyperparameter Tuning on FetchSlideDense-v4"
python3 benchmark.py --agent=hiro --hyperparameter_tuning --hyperparams_sweep_json="hiro_hyperparams_sweep_config.json" --env="FetchSlideDense-v4" --env_wrapper --max_episode_steps=50
echo "HAC Hyperparameter Tuning on FetchSlideDense-v4"
python3 benchmark.py --agent=hac --hyperparameter_tuning --hyperparams_sweep_json="hac_hyperparams_sweep_config.json" --env="FetchSlideDense-v4" --env_wrapper --max_episode_steps=50