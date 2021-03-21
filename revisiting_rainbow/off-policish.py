import numpy as np
import os

import dopamine
from dopamine.discrete_domains import gym_lib
from dopamine import discrete_domains as dd
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


num_runs = 1
training_steps = 10
ckpt_path = "../../results/rainbow/128_test10/checkpoints/ckpt.29"
# env = 'cartpole'


for agent in ags:
    for i in range (num_runs):
        def create_agent(sess, environment, summary_writer=None):
            return ags[agent](num_actions=environment.action_space.n)
        
        LOG_PATH = os.path.join(agent, f'../../results/{agent}/1024_test10/checkpoints')
        sys.path.append(path)
        gin_file = f'./Configs/{agent}_cartpole.gin'
        gin.parse_config_file(gin_file)
        env = gym_lib.create_gym_environment("CartPole")
        exp_data = dd.checkpointer.Checkpointer(path)._load_data_from_file(ckpt_path)
        trained_agent = JaxRainbowAgentNew(num_actions=env.action_space.n)
        print(trained_agent.unbundle(ckpt_path, 29, exp_data))

        obs = env.reset()
        # for _ in range(training_steps):
        #     a = trained_agent.step(obs)
            # obs, r, terminal, _ = env.step(a)
            # new_agent.train(obs, a, r, terminal)


        # gin.bind_parameter(f"{names[agent]}.neurons", width)

        
        # print(f'Training agent {i+1}, please be patient, may be a while...')
        # agent_runner.run_experiment()
        # print('Done training!')