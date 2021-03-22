import numpy as np
import os

import dopamine
from dopamine.jax.agents.rainbow.rainbow_agent import JaxRainbowAgent
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
ckpt_path = "../../results/rainbow/512_test10/checkpoints/ckpt.29"
env = gym_lib.create_gym_environment("CartPole")


for agent in ags:
    for i in range (num_runs):
        def create_agent(sess, environment, summary_writer=None):
            return ags[agent](num_actions=environment.action_space.n)
        
        gin_file = f'./Configs/{agent}_cartpole.gin'
        gin.parse_config_file(gin_file)

        new_agent = JaxRainbowAgentNew(num_actions=env.action_space.n)

        exp_data = dd.checkpointer.Checkpointer(path)._load_data_from_file(ckpt_path)
        trained_agent = JaxRainbowAgentNew(num_actions=env.action_space.n, eval_mode=True)
        trained_agent.unbundle(ckpt_path, 29, exp_data)
        
        r = 0
        obs = env.reset()
        for _ in range(training_steps):
            a = trained_agent.step(r, obs)
            obs, r, terminal, _ = env.step(a)
            new_agent._store_transition(jnp.reshape(obs, (4,1)), a, r, terminal)
            new_agent._train_step()