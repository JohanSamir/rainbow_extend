import numpy as np
import os

import dopamine
from dopamine.discrete_domains import run_experiment
from absl import flags
import gin.tf
import sys
path = "."
sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *

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


num_runs = 10

env = 'cartpole'


for agent in ags:
  for width in [64, 128, 256, 512, 1024]:
    for i in range (num_runs):
        def create_agent(sess, environment, summary_writer=None):
            return ags[agent](num_actions=environment.action_space.n)
        
        LOG_PATH = os.path.join(path, f'../../test_joao/{agent}/{width}_test10')
        sys.path.append(path)
        
        gin_file = f'./Configs/{agent}_{env}.gin'
        gin.parse_config_file(gin_file)
        gin.bind_parameter(f"{names[agent]}.neurons", width)

        agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

        print(f'Training agent {i+1}, please be patient, may be a while...')
        agent_runner.run_experiment()
        print('Done training!')