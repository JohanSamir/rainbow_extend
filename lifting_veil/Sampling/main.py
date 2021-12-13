import numpy as np
import os
import jax

jax.config.update('jax_platform_name', 'cpu')

import dopamine
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
from absl import flags, app
import sys

sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents import minatar_env
from replay_runner import FixedReplayRunner
import itertools

from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("seed", "1", "the program PRNG seed")

flags.DEFINE_string("experiment", "normalization=non_normalization", "the experiment will be run in")

flags.DEFINE_boolean("test", "False", "wether it's a test run")

flags.DEFINE_string("base_path", "../../extending_rainbow_exps/", "The base path for saving runs")


# `path=os.environ['AIP_TENSORBOARD_LOG_DIR']`

def main(_):

    def create_agent(sess, environment, summary_writer=None, memory=None):
        ag = agents[FLAGS.agent](num_actions=environment.action_space.n)
        if memory is not None:
            ag._replay = memory
            ag._replay.replay_capacity = (50000 * 0.2)
        return ag
    
    exp, value = FLAGS.experiment.split('=')
    #Generate the combination 
        
    agent_name = agents[FLAGS.agent].__name__

    gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'

    gin.clear_config()
    gin_bindings = get_gin_bindings(exp, agent_name, FLAGS.seed, value, FLAGS.test)
    gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
    LOG_PATH = os.path.join(f'{FLAGS.base_path}/{FLAGS.agent}/{FLAGS.env}/{exp}_{value}_online', f'test{FLAGS.seed}')
    print(f"Saving data at {LOG_PATH}")
    agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

    print(f'Training agent {FLAGS.seed}, please be patient, may be a while...')
    agent_runner.run_experiment()
    print('Done training!')
    if FLAGS.test:
        from datetime import datetime
        dt = datetime.now().strftime("%H-%M-%d-%m")
        os.makedirs(f"test_logs/{dt}", exist_ok=True)
        open(f"test_logs/{dt}/{FLAGS.env}_{FLAGS.agent}_{exp}_{FLAGS.type}", 'x').close()

if __name__ == "__main__":
    app.run(main)
