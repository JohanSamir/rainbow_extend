import os
import jax

jax.config.update('jax_platform_name', 'cpu')

from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib

from absl import flags, app, logging
import gin.tf
import sys

sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
# from agents import minatar_env

import utils


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("category", "classic", "the kind of environment will be ")

flags.DEFINE_string("agent", "rainbow", "the agent used in the experiment")

flags.DEFINE_integer("rl_seed", "1", "the agent-environment PRNG seed")

flags.DEFINE_integer("sample_seed", "1", "the sampling PRNG seed")

flags.DEFINE_string("experiment", "effective_horizon", "the experiment will be run in")

flags.DEFINE_string("base_path", "gs://joao-experiments", "The base path for saving runs")


# path = 'gs://joao-experiments'


def main(_):
    def create_agent(sess, environment, summary_writer=None, memory=None):
        ag = utils.agents[FLAGS.agent](num_actions=environment.action_space.n)
        return ag
    
    path = FLAGS.base_path
    grp = FLAGS.experiment
    values = utils.sample_group(FLAGS.category, grp, FLAGS.sample_seed)    

    agent_name = utils.agents[FLAGS.agent].__name__

    if FLAGS.category == "atari_100k":
        gin_file = f'Configs/{FLAGS.agent}_atari_100k.gin'
        gin.clear_config()
        gin_bindings = [f'create_atari_environment.game_name={FLAGS.env}']
    else:
        gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'
        gin.clear_config()
        gin_bindings = []

    for exp, value in zip(utils.suites[FLAGS.category].groups[grp], values):
        gin_bindings.extend(utils.get_gin_bindings(exp, agent_name, FLAGS.rl_seed, value, False))
    gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
    LOG_PATH = os.path.join(f'{path}/{FLAGS.agent}/{FLAGS.env}/{FLAGS.sample_seed}_{grp}_{utils.repr_values(values)}', f'{FLAGS.rl_seed}')
    logging.info(f"Saving data at {LOG_PATH}")
    agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

    logging.info(f'Training agent {FLAGS.rl_seed}, please be patient, may be a while...')
    agent_runner.run_experiment()
    logging.info('Done training!')

if __name__ == "__main__":
    app.run(main)