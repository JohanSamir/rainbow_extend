# -*- coding: utf-8 -*-
import os
import dopamine
from dopamine.discrete_domains import run_experiment
import gin.tf

import sys
from dqn_agent_new import *
#from rainbow_agent_new import*
#from quantile_agent_new import*
#from implicit_quantile_agent_new import*

inf = {
    1: 'dqn_cartpole_1.gin',
    2: 'dqn_cartpole_2.gin',
    3: 'dqn_cartpole_3.gin',
    4: 'dqn_cartpole_4.gin',
    5: 'dqn_cartpole_5.gin',
    6: 'dqn_cartpole_6.gin'
}
path_ = {
    1: 'DQN_1',
    2: 'DQN_2',
    3: 'DQN_3',
    4: 'DQN_4',
    5: 'DQN_5',
    6: 'DQN_6'
}

for j in range(1, 7):

    path = path_[j]

    for i in range(1, 8):

        LOG_PATH = os.path.join(path, 'dqn_test' + str(i))
        sys.path.append(path)

        def create_random_dqn_agent(sess, environment, summary_writer=None):
            """The Runner class will expect a function of this type to create an agent."""
            return JaxDQNAgentNew(num_actions=environment.action_space.n)
            #return JaxRainbowAgentNew(num_actions=environment.action_space.n)
            #return JaxQuantileAgentNew(num_actions=environment.action_space.n)
            #return JaxImplicitQuantileAgentNew(num_actions=environment.action_space.n)

        gin.parse_config_file(inf[j])

        random_dqn_runner = run_experiment.TrainRunner(
            LOG_PATH, create_random_dqn_agent)

        print('Will train agent, please be patient, may be a while...')
        random_dqn_runner.run_experiment()
        print('Done training!')
