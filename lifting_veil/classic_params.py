from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
import numpy as np



trivial_inits = ['zeros', 'ones', 'variance_baseline']
nontrivial_inits = ['orthogonal', 'xavier_uni', 'xavier_nor', 'he_uni']

inits = {
    'orthogonal': {
        'function': jax.nn.initializers.orthogonal
    },
    'xavier_uni': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 1,
        'mode': 'fan_avg',
        'distribution': 'uniform'
    },
    'xavier_nor': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 1,
        'mode': 'fan_avg',
        'distribution': 'truncated_normal'
    },
    'he_uni': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 2,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },

    'zeros': {
        'function': jax.nn.initializers.zeros
    },
    'ones': {
        'function': jax.nn.initializers.ones
    },
    'variance_baseline': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 1.0 / np.sqrt(3.0),
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
}

activations = ['relu', 'relu6', 'silu', 'selu', 'gelu']

normalizations = ['non_normalization', 'BatchNorm', 'LayerNorm']

learning_rates = [0.01, 0.001, 0.0001]

batch_sizes = [32, 64, 128, 256, 512]

epsilons = [0.03125, 0.003125, 0.0003125, 0.00003125, 0.000003125]

widths = [32, 64, 128, 256, 512, 1024]

depths = [1, 2, 3, 4]

convs = [1, 2, 3]

update_periods = [1, 2, 3, 4, 8]

target_update_periods = [10, 25, 50, 100, 200, 400, 800, 1600]

gammas = [0.99, 0.995, 0.999]

min_replay_historys = [750, 875, 1000, 1500]

num_atoms = [21, 31, 51, 61, 81]

update_horizon = [3, 4, 5, 8, 10]

noisy_net = [True, False]

clip_rewards = ["True", "False"]

weight_decays = [0, 0.01, 0.1, 1]

experiments = {
    "epsilon": epsilons,
    "learning_rate": learning_rates,
    "width": widths,
    "depth": depths,
    "normalization": normalizations,
    "init": inits,
    "trivial_init": trivial_inits,
    "nontrivial_init": nontrivial_inits,
    "activation": activations,
    "update_period": update_periods,
    "target_update_period": target_update_periods,
    "gamma": gammas,
    "min_replay_history": min_replay_historys,
    "num_atoms": num_atoms,
    "update_horizon": update_horizon,
    "clip_rewards": clip_rewards,
    "noisy_net": noisy_net,
    "weight_decay": weight_decays,
    "conv": convs,
    "batch_size": batch_sizes
    }

groups = { "effective_horizon" : ["update_horizon", "gamma"],
                "constancy_of_parameters" : ["trivial_init", "update_horizon", "noisy_net"],
                "network_starting_point" : ["nontrivial_init", "activation", "depth", "normalization"],
                "network_architecture" : ["depth", "width", "normalization"],
                "bellman_updates" : ["min_replay_history", "update_period", "target_update_period"],
                "distribution_parameterization" : ["clip_rewards", "num_atoms"],
                "optimizer_parameters" : ["learning_rate", "epsilon", "batch_size", "weight_decay"],
                "default" : [ ]
                }