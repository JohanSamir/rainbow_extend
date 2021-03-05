import os
import dopamine
from dopamine.discrete_domains import run_experiment
import gin.tf

import sys
from dqn_agent_new import*
from rainbow_agent_new import*
from quantile_agent_new import*
from implicit_quantile_agent_new import*


#.gin files-> xavier or variance_scalin
inf = {1:{'gin':'dqn_acrobot.gin',
          'path': 'DQN'},
      2:{'gin':'rainbow_acrobot.gin',
          'path': 'Rainbow'},
      3: {'gin':'quantile_acrobot.gin',
          'path': 'QR'},
      4:{'gin':'implicit_acrobot.gin',
          'path': 'IQ'}
          }

for agnt in range (1,5):
  path = inf[agnt]['path']

  for i in range (1,8):

    LOG_PATH = os.path.join(path, 'dqn_test'+str(i))
    sys.path.append(path)

    if agnt == 1:
        def create_random_dqn_agent(sess, environment, summary_writer=None):
           return JaxDQNAgentNew(num_actions=environment.action_space.n)
    elif agnt == 2:
    	def create_random_dqn_agent(sess, environment, summary_writer=None):
    		return JaxRainbowAgentNew(num_actions=environment.action_space.n)
    elif agnt == 3:
    	def create_random_dqn_agent(sess, environment, summary_writer=None):
    	  	return JaxQuantileAgentNew(num_actions=environment.action_space.n)
    elif agnt == 4:
    	def create_random_dqn_agent(sess, environment, summary_writer=None):
    		return JaxImplicitQuantileAgentNew(num_actions=environment.action_space.n)
    else:
    	print('error!')

    gin.parse_config_file(inf[agnt]['gin'])

    random_dqn_runner = run_experiment.TrainRunner(LOG_PATH, create_random_dqn_agent)

    print('Will train agent, please be patient, may be a while...')
    random_dqn_runner.run_experiment()
    print('Done training!')