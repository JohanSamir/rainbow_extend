import os
import jax

jax.config.update('jax_platform_name', 'cpu')

from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
from absl import flags, app
import gin.tf
import sys

sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents import minatar_env

import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("seed", "1", "the program PRNG seed")

flags.DEFINE_string("experiment", "normalization=non_normalization", "the experiment will be run in")

flags.DEFINE_string("base_path", "gs://joao-experiments", "The base path for saving runs")

  

# path = 'gs://joao-experiments'


def main(_):
    def create_agent(sess, environment, summary_writer=None, memory=None):
        ag = utils.agents[FLAGS.agent](num_actions=environment.action_space.n)
        return ag
    
    path = FLAGS.base_path
    grp, values = FLAGS.experiment.split('=') 
    if grp[0] == "'": #deal with quotes
        grp = grp[1:]
        values[-1] = values[-1][:-1]


    agent_name = utils.agents[FLAGS.agent].__name__

    gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'
    
    gin.clear_config()
    gin_bindings = []
    for exp, value in zip(utils.groups[grp], values):
        gin_bindings.extend(utils.get_gin_bindings(exp, agent_name, FLAGS.seed, value, False))
    gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
    LOG_PATH = os.path.join(f'{path}/{FLAGS.agent}/{FLAGS.env}/{grp}_{values}', f'test{FLAGS.seed}')
    print(f"Saving data at {LOG_PATH}")
    agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

    print(f'Training agent {FLAGS.seed}, please be patient, may be a while...')
    agent_runner.run_experiment()
    print('Done training!')

if __name__ == "__main__":
    app.run(main)
