from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *

agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

normalizations = ["non_normalization", 'BatchNorm', 'LayerNorm']

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
        'scale': 1.0 / jnp.sqrt(3.0),
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

activations = {
    'conf_0_non_activation': {
        'layer_fun': 'non_activation'
    },
    'conf_1_relu': {
        'layer_fun': 'relu'
    },
    'conf_2_relu6': {
        'layer_fun': 'relu6'
    },
    'conf_3_sigmoid': {
        'layer_fun': 'sigmoid'
    },
    'conf_4_softplus': {
        'layer_fun': 'softplus'
    },
    'conf_5_soft_sign': {
        'layer_fun': 'soft_sign'
    },
    'conf_6_silu': {
        'layer_fun': 'silu'
    },
    'conf_7_swish': {
        'layer_fun': 'swish'
    },
    'conf_8_log_sigmoid': {
        'layer_fun': 'log_sigmoid'
    },
    'conf_9_hard_sigmoid': {
        'layer_fun': 'hard_sigmoid'
    },
    'conf_10_hard_silu': {
        'layer_fun': 'hard_silu'
    },
    'conf_11_hard_swish': {
        'layer_fun': 'hard_swish'
    },
    'conf_12_hard_tanh': {
        'layer_fun': 'hard_tanh'
    },
    'conf_13_elu': {
        'layer_fun': 'elu'
    },
    'conf_14_celu': {
        'layer_fun': 'celu'
    },
    'conf_15_selu': {
        'layer_fun': 'selu'
    },
    'conf_16_gelu': {
        'layer_fun': 'gelu'
    },
    'conf_17_glu': {
        'layer_fun': 'glu'
    }
}

learning_rates = [10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

epsilons = [ 10, 5, 2, 1, 0.5, 0.3125, 0.03125, 0.003125, 0.0003125, 0.00003125]

widths = [32, 64, 128, 256, 512, 1024]

depths = [1, 2, 3, 4]

update_period = [1 ,2 ,3, 4, 8, 10, 12]

def get_gin_bindings(exp, agent_name, initial_seed, eps):
    if exp == "epsilon":
        gin_bindings = [f"{agent_name}.seed={initial_seed}", f"create_optimizer.eps = {eps}"]

    elif exp == "learning_rate":
        gin_bindings = [f"{agent_name}.seed={i}", f"create_optimizer.learning_rate = {eps}"]

    elif exp == "width":
        gin_bindings = [f"{agent_name}.seed={initial_seed}", f"{agent_name}.neurons = {eps}"]

    elif exp == "depth":
        gin_bindings = [f"{agent_name}.seed={initial_seed}", f"{agent_name}.hidden_layer = {eps}"]

    elif exp == "normalization":
        gin_bindings = [f"{agent_name}.seed={initial_seed}", f"{agent_name}.normalization = '{eps}'"]

    elif exp == "init":
        gin_bindings = get_init_bidings(agent_name, eps, initial_seed)

    elif exp == "activation":
        gin_bindings = [f"{agent_name}.seed={i}", f"{agent_name}.layer_funct = {eps}"]

    elif exp == "update_period":
        gin_bindings = [f"{agent_name}.seed={i}", f"{agent_name}.update_period = {eps}"]

    else:
        print("Error! Check the kind of experiment")
    return gin_bindings