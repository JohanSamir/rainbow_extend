# %%
import numpy as np
import os

import dopamine
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
# sys.path.append(path_or)
# %%
import sys
path = "."
sys.path.append(".")

from agents.quantile_agent_new import*
# %%
def create_random_dqn_agent(sess, environment, summary_writer=None):
    """The Runner class will expect a function of this type to create an agent."""  
    return JaxQuantileAgentNew(num_actions=environment.action_space.n)

for i in range (1):
    LOG_PATH = os.path.join(path, 'dqn_test'+str(i))
    sys.path.append(path)
    gin.parse_config_file('./Configs/quantile_cartpole.gin')

    random_dqn_runner = run_experiment.TrainRunner(LOG_PATH, create_random_dqn_agent) 

    print(f'Train agent {i}, please be patient, may be a while...')
    random_dqn_runner.run_experiment()
    print('Done training!')