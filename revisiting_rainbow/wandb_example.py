import numpy as np
import os

import dopamine
from dopamine.jax.agents.rainbow.rainbow_agent import JaxRainbowAgent
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment, checkpointer
from absl import flags, app
import sys
import wandb

sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *
from wandb_runner import WandBRunner

from constants import agents, inits, activations, learning_rates

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("initial_seed", "1", "the program will run seeds [initial_seed, initial_seed + 5)")

num_runs = 1
agent_name = agents[FLAGS.agent].__name__

for lr in learning_rates:
    for i in range(FLAGS.initial_seed, FLAGS.initial_seed + num_runs):

        def create_agent(sess, environment, summary_writer=None):
            return agents[FLAGS.agent](num_actions=environment.action_space.n)

        run = wandb.init(project="extending-rainbow",
                         entity="ext-rain",
                         config={
                             "random seed": i,
                             "learning rate": lr,
                             "agent": FLAGS.agent,
                             "environment": FLAGS.env
                         },
                         reinit=True)
        with run:
            LOG_PATH = os.path.join(".", f'../../test_joao/{FLAGS.agent}/env/baseline')
            gin_file = f'./Configs/{FLAGS.agent}_{FLAGS.env}.gin'
            gin_bindings = [f"{agent_name}.seed={i}", f"create_optimizer.learning_rate = {lr}"]
            gin.clear_config()
            gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)

            agent_runner = WandBRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)
            print(f'Training agent {i+1}, please be patient, may be a while...')
            agent_runner.run_experiment()
            print('Done training!')
