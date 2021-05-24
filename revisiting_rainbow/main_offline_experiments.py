import numpy as np
import os
import dopamine
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import sys

import matplotlib
from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *
import agents.networks_new
import agents.external_configurations

agents = {
    # 'dqn'd: JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

inits = {
    'orthogonal': {
        'function': jax.nn.initializers.orthogonal
    }
    # 'zeros': {
    #     'function': jax.nn.initializers.zeros
    # },
    # 'ones': {
    #     'function': jax.nn.initializers.ones
    # },
    # 'xavier_uni': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 1,
    #     'mode': 'fan_avg',
    #     'distribution': 'uniform'
    # },
    # 'xavier_nor': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 1,
    #     'mode': 'fan_avg',
    #     'distribution': 'truncated_normal'
    # },
    # 'lecun_uni': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 1,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'lecun_nor': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 1,
    #     'mode': 'fan_in',
    #     'distribution': 'truncated_normal'
    # },
    # 'he_uni': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 2,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'he_nor': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 2,
    #     'mode': 'fan_in',
    #     'distribution': 'truncated_normal'
    # },
    # 'variance_baseline': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 1.0 / jnp.sqrt(3.0),
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'variance_0.1': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 0.1,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'variance_0.3': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 0.3,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'variance_0.8': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 0.8,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'variance_3': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 3,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'variance_5': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 5,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # },
    # 'variance_10': {
    #     'function': jax.nn.initializers.variance_scaling,
    #     'scale': 10,
    #     'mode': 'fan_in',
    #     'distribution': 'uniform'
    # }
}

num_runs = 10  #7
path = "../../tests_joao/orthogonal/"
#environments = ['cartpole', 'acrobot','lunarlander','mountaincar']
environments = ['cartpole', 'acrobot']
#seeds = [True, False]
seeds = [True]

for seed in seeds:
    for agent in agents:
        for env in environments:
            for init in inits:
                for i in range(1, num_runs + 1):

                    def create_agent(sess, environment, summary_writer=None):
                        return agents[agent](
                            num_actions=environment.action_space.n)

                    agent_name = agents[agent].__name__
                    initializer = inits[init]['function'].__name__

                    LOG_PATH = os.path.join(
                        f'{path}{agent}_{env}',
                        f'test{i}')
                    sys.path.append(path)
                    gin_file = f'Configs/{agent}_{env}.gin'

                    if init == 'zeros' or init == 'ones':
                        gin_bindings = [
                            f"{agent_name}.seed=None"
                        ] if seed is False else [
                            f"{agent_name}.seed={i}",
                            f"{agent_name}.initzer = @{initializer}"
                        ]
                    elif init == "orthogonal":
                        gin_bindings = [f"{agent_name}.seed={i}",
                        f"{agent_name}.initzer = @{initializer}()",
                        f"{initializer}.scale = 1"]
                    else:
                        mode = '"' + inits[init]['mode'] + '"'
                        distribution = '"' + inits[init]['distribution'] + '"'
                        gin_bindings = [
                            f"{agent_name}.seed=None"
                        ] if seed is False else [
                            f"{agent_name}.seed={i}",
                            f"{agent_name}.initzer = @{initializer}()",
                            f"{initializer}.scale = 1",
                            f"{initializer}.mode = {mode}",
                            f"{initializer}.distribution = {distribution}"
                        ]

                    gin.clear_config()
                    gin.parse_config_files_and_bindings([gin_file],
                                                        gin_bindings,
                                                        skip_unknown=False)
                    agent_runner = run_experiment.TrainRunner(
                        LOG_PATH, create_agent)

                    print(
                        f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...'
                    )
                    agent_runner.run_experiment()
                    print('Done training!')
print('Finished!')
