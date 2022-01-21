from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
import numpy as np



normalizations = ['non_normalization', 'LayerNorm']

learning_rates = [0.01, 0.0001]

batch_sizes = [32, 64, 128, 256, 512]

epsilons = [0.03125, 0.0003125, 0.000003125]

widths = [128, 256]

depths = [1, 2, 3, 4]

convs = [1, 2, 3]

update_periods = [1, 4]

target_update_periods = [100, 800]

gammas = [0.99, 0.999]

min_replay_historys = [750, 1000, 1500]

num_atoms = [21, 31, 51, 61, 81]

update_horizon = [3, 5, 10]

noisy_net = [True, False]

clip_rewards = ["True", "False"]

weight_decays = [0, 0.01]

experiments = {
    "epsilon": epsilons,
    "learning_rate": learning_rates,
    "width": widths,
    "depth": depths,
    "normalization": normalizations,
    "update_period": update_periods,
    "target_update_period": target_update_periods,
    "gamma": gammas,
    "min_replay_history": min_replay_historys,
    "num_atoms": num_atoms,
    "update_horizon": update_horizon,
    "clip_rewards": clip_rewards,
    "weight_decay": weight_decays,
    "conv": convs,
    "batch_size": batch_sizes
    }

groups = { "effective_horizon" : ["update_horizon", "gamma"],
                "network_architecture" : ["depth", "width", "conv", "normalization"],
                "bellman_updates" : ["min_replay_history", "update_period", "target_update_period"],
                "distribution_parameterization" : ["num_atoms"],
                "optimizer_parameters" : ["learning_rate", "epsilon", "batch_size", "weight_decay"],
                }