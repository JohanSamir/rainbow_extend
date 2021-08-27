import numpy as np
import os
import functools

import dopamine
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment, checkpointer
from absl import app
from absl import flags
import gin.tf
import sys

path = "."
sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *
from replay_runner import FixedReplayRunner

ags = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

FLAGS = flags.FLAGS

flags.DEFINE_integer('replay_capacity', 50000, 'Size of the replay buffer', lower_bound=5000)
flags.DEFINE_integer('num_runs', 1, 'Number of runs with different random seeds', lower_bound=1)
flags.DEFINE_string('log_root', "../../test_joao/offline/", "the root of your loggin dir")


def main(argv):
    for agent in ags:

        def create_agent(sess, environment, summary_writer=None, memory=None):
            ag = ags[agent](num_actions=environment.action_space.n)
            if memory is not None:
                ag._replay = memory
            return ag

        for i in range(FLAGS.num_runs):
            LOG_PATH = os.path.join(path, FLAGS.log_root + f'{agent}/online_{i+1}')

            gin_file = f'./Configs/{agent}_cartpole.gin'
            gin.clear_config()
            gin.parse_config_file(gin_file)
            gin.bind_parameter(f"OutOfGraphPrioritizedReplayBuffer.replay_capacity", FLAGS.replay_capacity)

            agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)
            print(f'Training agent {i+1}, please be patient, may be a while...')
            agent_runner.run_experiment()
            print('Done normal training!')

            LOG_PATH = os.path.join(path, FLAGS.log_root + f'{agent}/fixed_{i+1}')

            gin.parse_config_file(gin_file)
            offline_runner = FixedReplayRunner(LOG_PATH,
                                               functools.partial(create_agent,
                                                                 memory=agent_runner._agent._replay,
                                                                 30,
                                                                 1000,
                                                                 200),
                                               create_environment_fn=gym_lib.create_gym_environment)
            print(f'Training fixed agent {i+1}, please be patient, may be a while...')
            offline_runner.run_experiment()
            print('Done fixed training!')


if __name__ == '__main__':
    app.run(main)
