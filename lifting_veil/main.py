import numpy as np
import os
import jax

jax.config.update('jax_platform_name', 'cpu')

import dopamine
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
from absl import flags, app
import sys
import wandb

sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *
from replay_runner import FixedReplayRunner

from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("seed", "1", "the program PRNG seed")

flags.DEFINE_string("experiment", "normalization=non_normalization", "the experiment will be run in")

flags.DEFINE_boolean("wb", "False", "the program won't use weights&biases")

flags.DEFINE_string("type", "online", "Whether the experiment is online or offline")

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
    
    if FLAGS.wb:
        if exp == "learning_rate":
            rep = "learning rate"
        else:
            rep = exp
        run = wandb.init(project="extending-rainbow",
                        entity="ext-rain",
                        config={
                            "random seed": i,
                            "agent": FLAGS.agent,
                            "environment": FLAGS.env,
                            rep: value, 
                            "varying": rep,
                            "online": False,
                        },
                        reinit=True)
        
    agent_name = agents[FLAGS.agent].__name__

    gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'

    gin_bindings = get_gin_bindings(exp, agent_name, FLAGS.seed, value, FLAGS.type)
    gin.clear_config()
    gin.parse_config_file(gin_file, skip_unknown=False)
    
    if FLAGS.type == "offline":
        BASELINE_PATH = os.path.join(FLAGS.base_path, "baselines/", f'{FLAGS.agent}/{FLAGS.env}')
        trained_agent = run_experiment.TrainRunner(BASELINE_PATH, create_agent)
        trained_agent.run_experiment() #make sure the agent is trained
        print(f'Loaded trained {FLAGS.agent} in {FLAGS.env}')
            
        gin.parse_config(gin_bindings)
        LOG_PATH = os.path.join(f'{FLAGS.base_path}/{FLAGS.agent}/{FLAGS.env}/{exp}_{value}_offline', f'test{FLAGS.seed}')
        agent_runner = FixedReplayRunner(base_dir=LOG_PATH,
                                            create_agent_fn=functools.partial(
                                                create_agent,
                                                memory=trained_agent._agent._replay,
                                            ),
                                            use_wb=FLAGS.wb,
                                            create_environment_fn=gym_lib.create_gym_environment)
    else:
        gin.parse_config(gin_bindings)
        LOG_PATH = os.path.join(f'{FLAGS.base_path}/{FLAGS.agent}/{FLAGS.env}/{exp}_{value}_online', f'test{FLAGS.seed}')
        print(f"Saving data at {LOG_PATH}")
        if FLAGS.wb:
            agent_runner = WandBRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)
        else:
            agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)

    print(f'Training agent {FLAGS.seed}, please be patient, may be a while...')
    agent_runner.run_experiment()
    print('Done training!')
    if FLAGS.wb:
        run.finish()
    print('Finished!')


if __name__ == "__main__":
    app.run(main)
