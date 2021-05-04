import numpy as np
import os

import dopamine
from dopamine.jax.agents.rainbow.rainbow_agent import JaxRainbowAgent
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
from offrunner import OffRunner

ags = {
    # 'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    # 'quantile': JaxQuantileAgentNew,
    # 'implicit': JaxImplicitQuantileAgentNew,
}

names = {
    # 'dqn': "JaxDQNAgentNew",
    'rainbow': "JaxRainbowAgentNew",
    # 'quantile': "JaxQuantileAgentNew",
    # 'implicit': "JaxImplicitQuantileAgentNew",
}


num_runs = 5
ckpt_path = "../../results/rainbow/512_test10/checkpoints/ckpt.29"
env = gym_lib.create_gym_environment("CartPole")


for agent in ags:
    for i in range (num_runs):
        def create_agent(sess, environment, summary_writer=None):
            return ags[agent](num_actions=environment.action_space.n)
        
        LOG_PATH = os.path.join(path, f'../../test_joao/{agent}/offline_{i+1}')
        gin_file = f'./Configs/{agent}_cartpole.gin'
        gin.parse_config_file(gin_file)

        exp_data = checkpointer.Checkpointer(path)._load_data_from_file(ckpt_path)
        trained_agent = JaxRainbowAgentNew(num_actions=env.action_space.n, eval_mode=True)
        trained_agent.unbundle(ckpt_path, 29, exp_data)
        agent_runner = OffRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment, trained_agent)
        print(f'Training agent {i+1}, please be patient, may be a while...')
        agent_runner.run_experiment()
        print('Done training!')