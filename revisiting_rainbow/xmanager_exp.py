import numpy as np
import os
import jax

jax.config.update('jax_platform_name', 'cpu')

import dopamine
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
from absl import flags, app
import gin.tf
import sys

sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *
from replay_runner import FixedReplayRunner

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

inits = {
    'orthogonal': {
        'function': jax.nn.initializers.orthogonal
    },
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

num_runs = 2  #7
path = os.environ['AIP_TENSORBOARD_LOG_DIR']

# path = "../../tests_joao/offline_last_pct/" #TODO point to cloud bucket
seeds = [True]
activations = {
    # 'conf_0_non_activation': {'layer_fun': 'non_activation'},
    # 'conf_1_relu': {'layer_fun':'relu'},
    # 'conf_2_relu6': {'layer_fun':'relu6'},
    # 'conf_3_sigmoid': {'layer_fun':'sigmoid'},
    # 'conf_4_softplus': {'layer_fun':'softplus'},
    # 'conf_5_soft_sign': {'layer_fun':'soft_sign'},
    # 'conf_6_silu': {'layer_fun':'silu'},
    # 'conf_7_swish': {'layer_fun':'swish'},
    # 'conf_8_log_sigmoid': {'layer_fun':'log_sigmoid'},
    # 'conf_9_hard_sigmoid': {'layer_fun':'hard_sigmoid'},
    # 'conf_10_hard_silu': {'layer_fun':'hard_silu'},
    # 'conf_11_hard_swish': {'layer_fun':'hard_swish'},
    # 'conf_12_hard_tanh': {'layer_fun':'hard_tanh'},
    # 'conf_13_elu': {'layer_fun':'elu'},
    # 'conf_14_celu': {'layer_fun':'celu'},
    # 'conf_15_selu': {'layer_fun':'selu'},
    'conf_16_gelu': {
        'layer_fun': 'gelu'
    },
    # 'conf_17_glu': {'layer_fun':'glu'}
}


def main(_):

    def create_agent(sess, environment, summary_writer=None, memory=None):
        ag = agents[FLAGS.agent](num_actions=environment.action_space.n)
        if memory is not None:
            ag._replay = memory
            ag._replay.replay_capacity = (50000 * 0.2)
            # ag._replay.add_count = 0 # only for first x%
        return ag

    for act in activations:
        for i in range(1, num_runs + 1):
            agent_name = agents[FLAGS.agent].__name__
            # initializer = inits[init]['function'].__name__

            layer_fun = "'" + activations[act]['layer_fun'] + "'"
            LOG_PATH = os.path.join(f'{path}{FLAGS.agent}/{FLAGS.env}_{activations[act]["layer_fun"]}_online',
                                    f'test{i}')
            sys.path.append(path)
            gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'

            print('layer_fun:', layer_fun)

            gin_bindings = [f"{agent_name}.seed={i}", f"{agent_name}.layer_funct = {layer_fun}"]

            gin_bindings.append(f"OutOfGraphPrioritizedReplayBuffer.replay_capacity = 50000")
            gin.clear_config()
            gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
            agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

            print(
                f'Will train agent {FLAGS.agent} with activation {layer_fun} in {FLAGS.env}, run {i}, please be patient, may be a while...'
            )
            agent_runner.run_experiment()
            print('Done normal training!')

            LOG_PATH = os.path.join(f'{path}{FLAGS.agent}/{FLAGS.env}_{activations[act]["layer_fun"]}_fixed_20',
                                    f'test{i}')

            offline_runner = FixedReplayRunner(base_dir=LOG_PATH,
                                               create_agent_fn=functools.partial(
                                                   create_agent,
                                                   memory=agent_runner._agent._replay,
                                               ),
                                               num_iterations=30,
                                               training_steps=1000,
                                               evaluation_steps=200,
                                               create_environment_fn=gym_lib.create_gym_environment)
            print(f'Training fixed agent {i+1}, please be patient, may be a while...')
            offline_runner.run_experiment()
            print('Done fixed training!')
        print('Finished!')


if __name__ == "__main__":
    app.run(main)