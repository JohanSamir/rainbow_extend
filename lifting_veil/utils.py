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

    elif "init" in exp:
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
        raise ValueError("Experiment not recognized")

    if test:
        gin_bindings.extend(["Runner.num_iterations=4", "Runner.training_steps=200"])
        
    return gin_bindings


def repr_values(values):
    cat = "_".join(str(val) for val in values)
    return cat.replace(".", "p")

def cast_to_int(lst):
    for idx, el in enumerate(lst):
        if type(el) == np.float64 and el.is_integer():
            lst[idx] = int(el)
    return lst

def sample_group(grp, seed, num=1): 
    rng = np.random.default_rng(seed)
    total = list(itertools.product(*[experiments[exp] for exp in groups[grp]]))
    total = np.array(total)
    indices = rng.choice(len(total), num, replace=False)
    sample = cast_to_int(list(total[indices][0]))        
    cs = np.cumsum([0] + [len(experiments[exp]) for exp in groups[grp]])
    seed %= cs[-1]
    idx = bisect.bisect(cs, seed) - 1
    sample[idx] = experiments[groups[grp][idx]][seed - cs[idx]]
    logging.info(f"Sample Seed Index = {idx}")
    logging.info(f"Changed {groups[grp][idx]} of group {grp} to {sample[idx]}")

    return sample

def print_groups():
    print('groups = {}')
    for grp in groups:
      print(f"groups['{grp}'] = " + "{")
      cs = np.cumsum([0] + [len(experiments[exp]) for exp in groups[grp]])
      for seed in range(cs[-1]):
        idx = bisect.bisect(cs, seed) - 1
        sample = experiments[groups[grp][idx]][seed - cs[idx]]
        print(f"  '{seed}': '{groups[grp][idx]}={sample}',")
      print('}')

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
