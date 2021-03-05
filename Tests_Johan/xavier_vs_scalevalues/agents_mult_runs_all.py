import os
import dopamine
from dopamine.discrete_domains import run_experiment
import gin.tf

import sys
from dqn_agent_new import *
from rainbow_agent_new import *
from quantile_agent_new import *
from implicit_quantile_agent_new import *


agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

num_runs = 7

environments = ['cartpole', 'acrobot']

for agent in agents:
  for env in environments:
    for i in range (1, num_runs + 1):

      LOG_PATH = os.path.join(agent, f'dqn_test{i}')
      sys.path.append(path)
          
      def create_agent(sess, environment, summary_writer=None):
        return agents[agent](num_actions=environment.action_space.n)

      gin_file = f'{agent}_{env}.gin'
      gin.parse_config_file(gin_file)
      # gin_bindings = ["JaxDQNAgentNew.initzer='variance_scaling_5'", ...]
      # gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)

      agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

      print(f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...')
      agent_runner.run_experiment()
      print('Done training!')
