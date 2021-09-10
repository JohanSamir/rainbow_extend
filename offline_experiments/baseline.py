import numpy as np
import os
import jax

jax.config.update('jax_platform_name', 'cpu')

import dopamine
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
def main(_):
    agent_name = agents[FLAGS.agent].__name__

    for i in range(FLAGS.initial_seed, FLAGS.initial_seed + num_runs):

        def create_agent(sess, environment, summary_writer=None):
            return agents[FLAGS.agent](num_actions=environment.action_space.n)

        run = wandb.init(project="extending-rainbow",
                            entity="ext-rain",
                            config={
                                "random seed": i,
                                "agent": FLAGS.agent,
                                "environment": FLAGS.env
                            },
                            reinit=True)
        with run:
            LOG_PATH = os.path.join(".", f'../../test_joao/baseline/{FLAGS.agent}/{FLAGS.env}')
            gin_file = f'./Configs/{FLAGS.agent}_{FLAGS.env}.gin'
            gin_bindings = [f"{agent_name}.seed={i}"]
            gin.clear_config()
            gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)

            agent_runner = WandBRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)
            print(f'Training agent {i+1}, please be patient, may be a while...')
            agent_runner.run_experiment()
            print('Done training!')

if __name__ == "__main__":
    app.run(main)