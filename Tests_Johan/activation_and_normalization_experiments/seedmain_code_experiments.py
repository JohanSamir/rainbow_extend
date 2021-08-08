import numpy as np
import os
import dopamine
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
    'conf_0_non_activation': {'layer_fun': 'non_activation'},
    'conf_1_relu': {'layer_fun':'relu'},
    'conf_2_relu6': {'layer_fun':'relu6'},
    'conf_3_sigmoid': {'layer_fun':'sigmoid'},
    'conf_4_softplus': {'layer_fun':'softplus'},
    'conf_5_soft_sign': {'layer_fun':'soft_sign'},
    'conf_6_silu': {'layer_fun':'silu'},
    'conf_7_swish': {'layer_fun':'swish'},
    'conf_8_log_sigmoid': {'layer_fun':'log_sigmoid'},
    'conf_9_hard_sigmoid': {'layer_fun':'hard_sigmoid'},
    'conf_10_hard_silu': {'layer_fun':'hard_silu'},
    'conf_11_hard_swish': {'layer_fun':'hard_swish'},
    'conf_12_hard_tanh': {'layer_fun':'hard_tanh'},
    'conf_13_elu': {'layer_fun':'elu'},
    'conf_14_celu': {'layer_fun':'celu'},
    'conf_15_selu': {'layer_fun':'selu'},
    'conf_16_gelu': {'layer_fun':'gelu'},
    'conf_17_glu': {'layer_fun':'glu'}
    }

num_runs = 7
environments = ['cartpole', 'acrobot', 'lunarlander', 'mountaincar']
seeds = [True]
path= '/home/johan/ExperimentsInitializer/2_activation_experiments_Rainbow/'

for seed in seeds:
  for agent in agents:
    for env in environments:
      for init in inits:
        for i in range (1, num_runs + 1):  
          
          def create_agent(sess, environment, summary_writer=None):
            return agents[agent](num_actions=environment.action_space.n)

          agent_name = agents[agent].__name__

          LOG_PATH = os.path.join(f'{path}{seed}{i}_{agent}_{env}_{init}', f'dqn_test{i}')
          sys.path.append(path)    
          gin_file = f'{path}{agent}_{env}.gin'

          layer_fun = "'"+inits[init]['layer_fun']+"'"
          print('layer_fun:',layer_fun)

          gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                          f"{agent_name}.layer_funct = {layer_fun}"]

          gin.clear_config()
          gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
          agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

          print(f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...')
          agent_runner.run_experiment()
          print('Done training!')
print('Finished!')