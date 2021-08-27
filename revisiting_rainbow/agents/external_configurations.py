"""External configuration .gin"""

from gin import config
from flax import linen as nn

#config.external_configurable(jax.nn.initializers.zeros, 'jax.nn.initializers.zeros')
#config.external_configurable(jax.nn.initializers.ones, 'jax.nn.initializers.ones')
#config.external_configurable(jax.nn.initializers.variance_scaling, 'jax.nn.initializers.variance_scaling')

config.external_configurable(nn.initializers.zeros, 'nn.initializers.zeros')
config.external_configurable(nn.initializers.ones, 'nn.initializers.ones')
config.external_configurable(nn.initializers.orthogonal, 'nn.initializers.orthogonal')
config.external_configurable(nn.initializers.variance_scaling, 'nn.initializers.variance_scaling')
