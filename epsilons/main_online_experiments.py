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

from constants import agents, epsilons

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("initial_seed", "1", "the program will run seeds [initial_seed, initial_seed + 5)")

num_runs = 1  #7
# `path=os.environ['AIP_TENSORBOARD_LOG_DIR']`

path = "../../ExperimentsInitializer/epsilons"  #TODO point to cloud bucket

def main(_):

    def create_agent(sess, environment, summary_writer=None): #memory=None
        ag = agents[FLAGS.agent](num_actions=environment.action_space.n)
        return ag

    for eps in epsilons:
        # layer_fun = "'" + activations[act]['layer_fun'] + "'"
        for i in range(FLAGS.initial_seed, FLAGS.initial_seed + num_runs):
            name = "online"
            run = wandb.init(project="extending-rainbow",
                            entity="ext-rain",
                            config={
                                "random seed": i,
                                "agent": FLAGS.agent,
                                "environment": FLAGS.env,
                                "epsilon": eps, 
                                "varying": "epsilon",

                            },
                            name=name,
                            reinit=True)
            with run:
                agent_name = agents[FLAGS.agent].__name__

                LOG_PATH = os.path.join(".", f'../../ExperimentsInitializer/epsilons/{FLAGS.agent}/{FLAGS.env}')
                sys.path.append(path)
                gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'

                gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"create_optimizer.eps = {eps}"]

                gin.clear_config()
                gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
                agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

                print(f'Training fixed agent {i}, please be patient, may be a while...')
                agent_runner.run_experiment()
                print('Done training!')
            print('Finished!')


if __name__ == "__main__":
    app.run(main)