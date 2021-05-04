import numpy as np
import os
import functools

import dopamine
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment, checkpointer
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

names = {
    'dqn': "JaxDQNAgentNew",
    'rainbow': "JaxRainbowAgentNew",
    'quantile': "JaxQuantileAgentNew",
    'implicit': "JaxImplicitQuantileAgentNew",
}

num_runs = 5

for agent in ags:
    for i in range(num_runs):

        def create_agent(sess, environment, summary_writer=None, memory=None):
            ag = ags[agent](num_actions=environment.action_space.n)
            if memory is not None:
                ag._replay = memory
            return ag

        LOG_PATH = os.path.join(
            path, f'../../test_joao/offline/{agent}/online_{i+1}')
        gin_file = f'./Configs/{agent}_cartpole.gin'
        gin.parse_config_file(gin_file)
        gin.bind_parameter(
            f"OutOfGraphPrioritizedReplayBuffer.replay_capacity", 500000)
        agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)
        print(f'Training agent {i+1}, please be patient, may be a while...')
        agent_runner.run_experiment()
        print('Done normal training!')

        LOG_PATH = os.path.join(
            path, f'../../test_joao/offline/{agent}/fixed_{i+1}')
        gin_file = f'./Configs/{agent}_cartpole.gin'
        gin.parse_config_file(gin_file)
        offline_runner = FixedReplayRunner(
            LOG_PATH,
            functools.partial(create_agent,
                              memory=agent_runner._agent._replay),
            create_environment_fn=gym_lib.create_gym_environment)
        print(
            f'Training fixed agent {i+1}, please be patient, may be a while...'
        )
        offline_runner.run_experiment()
        print('Done fixed training!')
