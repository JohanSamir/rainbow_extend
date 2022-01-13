from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from absl import logging
import numpy as np
import itertools
import bisect


agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'c51': JaxRainbowAgentNew,
    'rainbow_without': JaxDQNAgentNew,
}

inits = {
    'orthogonal': {
        'function': jax.nn.initializers.orthogonal
    },
    'zeros': {
        'function': jax.nn.initializers.zeros
    },
    'ones': {
        'function': jax.nn.initializers.ones
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
    'lecun_uni': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 1,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'lecun_nor': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 1,
        'mode': 'fan_in',
        'distribution': 'truncated_normal'
    },
    'he_uni': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 2,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'he_nor': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 2,
        'mode': 'fan_in',
        'distribution': 'truncated_normal'
    },
    'variance_baseline': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 1.0 / np.sqrt(3.0),
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'variance_0.1': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 0.1,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'variance_0.3': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 0.3,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'variance_0.8': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 0.8,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'variance_3': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 3,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'variance_5': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 5,
        'mode': 'fan_in',
        'distribution': 'uniform'
    },
    'variance_10': {
        'function': jax.nn.initializers.variance_scaling,
        'scale': 10,
        'mode': 'fan_in',
        'distribution': 'uniform'
    }
}

activations = ['non_activation', 'relu', 'relu6', 'sigmoid', 'softplus', 'soft_sign', 'silu', 'log_sigmoid', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'hard_tanh', 'elu', 'celu', 'selu', 'gelu', 'glu']

normalizations = ['non_normalization', 'BatchNorm', 'LayerNorm']

learning_rates = [10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

batch_sizes = [32, 64, 128, 256, 512]

epsilons = [1, 0.5, 0.3125, 0.03125, 0.003125, 0.0003125, 0.00003125, 0.000003125]

widths = [32, 64, 128, 256, 512, 1024]

depths = [1, 2, 3, 4]

convs = [1, 2, 3]

update_periods = [1 ,2 ,3, 4, 8, 10, 12]

target_update_periods = [10, 25, 50, 100, 200, 400, 800, 1600]

gammas = [0.1, 0.5, 0.9, 0.99, 0.995, 0.999]

min_replay_historys = [125, 250, 375, 500, 625, 750, 875, 1000]

num_atoms = [11, 21, 31, 41, 51, 61]

update_horizon = [1, 2, 3, 4, 5, 8, 10]

noisy_net = [True, False]

clip_rewards = ["True", "False"]

weight_decays = [0.01, 0.1, 1]

experiments = {
        "epsilon": epsilons,
        "learning_rate": learning_rates,
        "width": widths,
        "depth": depths,
        "normalization": normalizations,
        "init": inits,
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
}

groups = { "effective_horizon" : ["update_period", "gamma"],
                "constancy_of_parameters" : ["init", "update_period", "noisy_net"],
                "network_starting_point" : ["init", "activation", "depth", "normalization"],
                "network_architecture" : ["depth", "width", "normalization"],
                "bellman_updates" : ["min_replay_history", "update_period", "target_update_period"],
                "distribution_parameterization" : ["clip_rewards", "num_atoms"],
                "optimizer_parameters" : ["learning_rate", "epsilon", "batch_size", "weight_decay"],
                }


def get_init_bidings(agent_name, init, seed=None):
    initializer = inits[init]['function'].__name__
    if init == 'zeros' or init == 'ones':
        gin_bindings = [f"{agent_name}.seed={seed}",
                        f"{agent_name}.initzer = @{initializer}"]

    elif init == "orthogonal":
        gin_bindings = [f"{agent_name}.seed={seed}",
                        f"{agent_name}.initzer = @{initializer}()",
                        f"{initializer}.scale = 1"]
    else:
        mode = '"'+inits[init]['mode']+'"'
        scale = inits[init]['scale']
        distribution = '"'+inits[init]['distribution']+'"'
        gin_bindings = [f"{agent_name}.seed={seed}",
                        f"{agent_name}.initzer = @{initializer}()",
                        f"{initializer}.scale = {scale}",
                        f"{initializer}.mode = {mode}",
                        f"{initializer}.distribution = {distribution}"
                        ]
    return gin_bindings

def get_gin_bindings(exp, agent_name, initial_seed, value, test):
     gin_bindings = [f"{agent_name}.seed={initial_seed}"]
    if exp == "epsilon":
        gin_bindings += [f"create_opt.eps = {value}"]

    elif exp == "learning_rate":
        gin_bindings += [f"create_opt.learning_rate = {value}"]
    
    elif exp == "weight_decay":
        gin_bindings += [f"create_opt.weight_decay = {value}"]

    elif exp == "width":
        gin_bindings += [f"{agent_name}.neurons = {value}"]

    elif exp == "depth":
        gin_bindings += [f"{agent_name}.hidden_layer = {value}"]

    elif exp == "conv":
        gin_bindings += [f"{agent_name}.hidden_conv = {value}"]

    elif exp == "normalization":
        gin_bindings += [f"{agent_name}.normalization = '{value}'"]

    elif exp == "init":
        gin_bindings = get_init_bidings(agent_name, value, initial_seed)

    elif exp == "activation":
        gin_bindings += [f"{agent_name}.layer_funct = '{value}'"]

    elif exp == "update_period":
        gin_bindings += [f"{agent_name}.update_period = {value}"]

    elif exp == "target_update_period":
        gin_bindings += [f"{agent_name}.target_update_period = {value}"]
    
    elif exp == "gamma":
        gin_bindings += [f"{agent_name}.gamma = {value}"]
    
    elif exp == "min_replay_history":
        gin_bindings += [f"{agent_name}.min_replay_history = {value}"]

    elif exp == "num_atoms":
        gin_bindings += [f"{agent_name}.num_atoms = {value}"]

    elif exp == "update_horizon":
        gin_bindings += [f"{agent_name}.update_horizon = {value}"]

    elif exp == "clip_rewards":
        gin_bindings += [f"Runner.clip_rewards = {value}"]
    
    elif exp == "batch_size":   
        gin_bindings += [f"OutOfGraphPrioritizedReplayBuffer.batch_size = {value}"]
        
    elif exp == "noisy_net":
        gin_bindings += [f"{agent_name}.noisy = {value}"]
    
    else:
        logging.error("Error! Check the kind of experiment")

    if test:
        gin_bindings.extend(["Runner.num_iterations=4", "Runner.training_steps=200"])
        
    return gin_bindings


def repr_values(values):
    cat = "_".join(str(val) for val in values)
    return cat.replace(".", "p")

def get(container, idx):
    if type(container) is dict:
        return list(container.keys())[idx]
    return container[idx]

def sample_group(grp, seed, num=1):
    rng = np.random.default_rng(seed)
    total = list(itertools.product(*[experiments[exp] for exp in groups[grp]]))
    total = np.array(total)
    indices = rng.choice(len(total), num, replace=False)
    sample = total[indices][0]
    cs = np.cumsum([0] + [len(experiments[exp]) for exp in groups[grp]])
    seed %= cs[-1]
    idx = bisect.bisect(cs, seed)
    sample[idx - 1] = get(experiments[groups[grp][idx-1]], seed - cs[idx-1])
    return sample
