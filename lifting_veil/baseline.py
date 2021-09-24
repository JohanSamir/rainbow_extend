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

import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_string("base_path", "../../extending_rainbow_exps/", "The base path for saving runs")

num_runs = 1
def main(_):
    agent_name = utils.agents[FLAGS.agent].__name__

    def create_agent(sess, environment, summary_writer=None):
            return utils.agents[FLAGS.agent](num_actions=environment.action_space.n)

    LOG_PATH = os.path.join(FLAGS.base_path, "baselines", f'{FLAGS.agent}/{FLAGS.env}')
    gin_file = f'./Configs/{FLAGS.agent}_{FLAGS.env}.gin'
    gin_bindings = [f"{agent_name}.seed=1729"]
    gin.clear_config()
    gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)

    agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)
    print(f'Training agent, please be patient, may be a while...')
    agent_runner.run_experiment()
    print('Done training!')

if __name__ == "__main__":
    app.run(main)
