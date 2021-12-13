from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
import dopamine.replay_memory.prioritized_replay_buffer
import numpy as np


agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    # 'quantile': JaxQuantileAgentNew,
    # 'implicit': JaxImplicitQuantileAgentNew,
}

inits = {
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
    }
}

activations = ['relu', 'relu6', 'silu', 'selu', 'gelu']

normalizations = ['non_normalization', 'BatchNorm', 'LayerNorm']

learning_rates = [ 0.001, 0.0001, 0.00001]

batch_sizes = [32, 64, 128, 256, 512]

epsilons = [0.03125, 0.003125, 0.0003125, 0.00003125]

widths = [32, 64, 128, 256, 512, 1024]

depths = [0, 1, 2, 3, 4]

update_periods = [3, 4, 8]

target_update_periods = [25, 50, 100, 200]

gammas = [0.9, 0.99, 0.995]

min_replay_historys = [750, 875, 1000]

num_atoms = [21, 31, 51, 81]

update_horizon = [3, 4, 5, 8, 10]

clip_rewards = ["True", "False"]
noisy_net = ["True", "False"]


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
        "batch_sizes":batch_sizes
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
    if exp == "effective_horizon":

        gin_bindings = [f"{agent_name}.seed={initial_seed}", f"{agent_name}.gamma = {value}",
                        f"{agent_name}.seed={initial_seed}", f"{agent_name}.update_period = {value}"
                        ]     
    else:
        print("Error! Check the kind of experiment")

    if test:
        gin_bindings.extend(["Runner.num_iterations=4", "Runner.training_steps=200"])
        
    return gin_bindings
