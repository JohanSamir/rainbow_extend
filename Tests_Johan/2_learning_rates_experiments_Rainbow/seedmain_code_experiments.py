import numpy as np
import os
import dopamine
import dopamine.jax.agents.dqn.dqn_agent

from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils

from absl import flags
import gin.tf
import sys

import matplotlib
#from dqn_agent_new import *
from rainbow_agent_new import *
#from quantile_agent_new import *
#from implicit_quantile_agent_new import *
import networks_new
import external_configurations

agents = {
    #'dqn': JaxDQNAgentNew,
    #rainbow': JaxRainbowAgentNew,
    #'quantile': JaxQuantileAgentNew,
    #'implicit': JaxImplicitQuantileAgentNew,
}

agents = {'rainbow': JaxRainbowAgentNew}
inits = {
    '10': 10,
    '5':5,
    '2':2,
    '1':1 ,
    '0.1':0.1,
    '0.01':0.01,
    '0.001': 0.001,
    '0.0001':0.0001,
    '0.00001': 0.00001}

num_runs = 7
environments = ['cartpole', 'acrobot', 'lunarlander', 'mountaincar']
seeds = [True]
path= '/home/johan/ExperimentsInitializer/2_learning_rates_experiments_Rainbow/'

for seed in seeds:
  for agent in agents:
    for env in environments:
      for init in inits:
        for i in range (1, num_runs + 1):  
          
          def create_agent(sess, environment, summary_writer=None):
            return agents[agent](num_actions=environment.action_space.n)

          agent_name =  agents[agent].__name__
          optimizer = dqn_agent.create_optimizer.__name__

          LOG_PATH = os.path.join(f'{path}{seed}{i}_{agent}_{env}_{init}', f'dqn_test{i}')
          sys.path.append(path)    
          gin_file = f'{path}{agent}_{env}.gin'

          print('init:',init)
          gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                          f"{optimizer}.learning_rate = {init}"]

          gin.clear_config()
          gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
          agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

          print(f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...')
          agent_runner.run_experiment()
          print('Done training!')
print('Finished!')