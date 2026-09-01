import json
import yaml

def main():

    hyperparameters = {
        "WANDB_UPLOAD": True,
        "SAVE_MODEL": True,

        "MAX_ENV_STEPS": 1.5e5,                 

        "LR": 35e-4,                          # OC, FuN, A2C
        # "LR_ACTOR": 2e-4,                   # DDPG, TD3, HIRO, HAC
        # "LR_CRITIC": 2e-4,                  # DDPG, TD3, HIRO, HAC

        # "GAMMA": 0.99,                      # OC, DDPG
        # "GAMMA_M": 0.99,                    # FuN, HIRO
        # "GAMMA_W": 0.95,                    # FuN, HIRO

        "ENT_COEF": 36e-4,                   # OC, A2C

        # "LAM": 0.97,
        # "PPO_EPOCHS": 8,                    # PPO

        # "NUM_OPTIONS": 1,                   # OC
        # "TERM_REG": 2e-4,                   # OC
        # "EPSILON_DECAY": 0.9952,            # OC
        
        # "MAX_ACTION": 3.0,                  # DDPG, TD3
        # "TAU": 0.0042039333644093325,       # DDPG, TD3 
        # 'BATCH_SIZE': 256,                  # DDPG, TD3, HIRO, HAC


        # 'EXPL_NOISE_MANAGER': 0.935,          # HIRO, HAC
        # 'EXPL_NOISE_WORKER': 0.17866545336681583,           # HIRO, HAC
        # 'GOAL_BOUND': 0.8,                                  # HIRO, HAC
        # 'GOAL_THRESHOLD': 0.69332425706,               # HAC
        # 'N_LEVELS': 3,                                      # HAC
        # 'SUBGOAL_TEST_FREQ': 0.4,                           # HAC

        # "GOAL_DIM": 76,                     # FuN, HIRO
        # "N_GOALS": 1,                       # FuN
        # "SUBGOAL_HORIZON": 2,               # FuN, HIRO, HAC

        "ALPHA": 0.3,                     # FuN, HAC, HER

    }

    hyperparameters_sweep = {
            'method': 'bayes',
            'metric': {'name': 'mean_episode_return', 'goal': 'maximize'},

            'parameters': {
            'LR': {'distribution': 'uniform', 'min': 1e-3, 'max': 4e-3},
            'NUM_OPTIONS': {'values': [2, 3, 4]},
            'TERM_REG': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 2e-4},
            'ENT_COEF': {'distribution': 'log_uniform_values', 'min': 3e-4, 'max': 6e-3},
            'EPSILON_DECAY': {'distribution': 'uniform', 'min': 0.99, 'max': 0.999},
            'GOAL_DIM': {'values': [2, 4]},
            'N_GOALS': {'values': [1, 2, 4]},
            'LR_ACTOR': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 3e-4},
            'LR_CRITIC': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
            'EXPL_NOISE_MANAGER': {'distribution': 'uniform', 'min': 0.3, 'max': 0.6},
            'EXPL_NOISE_WORKER': {'distribution': 'uniform', 'min': 0.05, 'max': 0.25},
            'SUBGOAL_HORIZON': {'values': [2, 5, 10]},
            'GOAL_BOUND': {'distribution': 'uniform', 'min': 1.5, 'max': 1.8},
            'GOAL_THRESHOLD': {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
            'N_LEVELS': {'values': [2, 3]},
            'SUBGOAL_TEST_FREQ': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
            'ALPHA': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
            'LAM': {'distribution': 'uniform', 'min': 0.95, 'max': 0.99},
            "BATCH_SIZE": {"values": [64, 128, 256]},
            'UPDATE_DELAY': {'values': [1, 2, 4]},
            'TAU': {'distribution': 'uniform', 'min': 1e-3, 'max': 1e-2},
            'PPO_EPOCHS': {'values': [2, 4, 8]},
        },
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }
    # Write data to a JSON file
    with open("hyperparams_config.json", "w") as json_file:
        json.dump(hyperparameters, json_file, indent=4)

    # with open("hyperparams_sweep_config.json", "w") as json_file:
    #     json.dump(hyperparameters_sweep, json_file, indent=4)

    # # Write data to a YAML file
    # with open("hyperparams_config.yaml", "w") as yaml_file:
    #     yaml.dump(hyperparameters, yaml_file)

    # with open("hyperparams_sweep_config.yaml", "w") as yaml_file:
    #     yaml.dump(hyperparameters_sweep, yaml_file)

if __name__ == "__main__":
    main()