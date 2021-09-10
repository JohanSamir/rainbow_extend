'''
How to test:
python3 main_online_experiments.py --env="acrobot" --agent="rainbow" --initial_seed=1 --exp="epsilons"
'''

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

from wandb_runner import WandBRunner
from constants import agents, epsilons, learning_rates, widths, depths, normalizations, inits, update_period, activations, get_init_bidings

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("initial_seed", "1", "the program will run seeds [initial_seed, initial_seed + 5)")

flags.DEFINE_string("exp", "epsilons", "the experiment will be run in")

flags.DEFINE_boolean("wb", False, "the program won't use weights&biases")

experiments = {
        "epsilon": epsilons,
        "learning_rate":learning_rates,
        "widths":widths,
        "depths":depths,
        "normalization":normalizations,
        "init":inits,
        "activation":activations,
        "update_period":update_period, 
}
    

num_runs = 1  #7
# `path=os.environ['AIP_TENSORBOARD_LOG_DIR']`

path = "../../ExperimentsInitializer/epsilons_git/"  #TODO

def main(_):

    def create_agent(sess, environment, summary_writer=None): #memory=None
        ag = agents[FLAGS.agent](num_actions=environment.action_space.n)
        return ag

    for eps in experiments[FLAGS.exp]:
        if FLAGS.exp=="activations":
            eps = "'" + activations[eps]['layer_fun'] + "'"

        for i in range(FLAGS.initial_seed, FLAGS.initial_seed + num_runs):
            if FLAGS.wb:
                run = wandb.init(project="extending-rainbow",
                                entity="ext-rain",
                                config={
                                    "random seed": i,
                                    "agent": FLAGS.agent,
                                    "environment": FLAGS.env,
                                    f'{FLAGS.exp}': eps, 
                                    "varying": FLAGS.exp,
                                    "online": True,

                                },
                                reinit=True)

            agent_name = agents[FLAGS.agent].__name__

            LOG_PATH = os.path.join(os.path.join(f'{path}/{FLAGS.agent}/{FLAGS.env}/{FLAGS.exp}_{eps}_online', f'test{i}'))
            sys.path.append(path)
            gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'

            if FLAGS.exp == "epsilons":
              gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"create_optimizer.eps = {eps}"]

            elif FLAGS.exp == "learning_rates":
              gin_bindings = [f"{agent_name}.seed={i}", f"create_optimizer.learning_rate = {eps}"]

            elif FLAGS.exp == "widths":
              gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"{agent_name}.neurons = {eps}"]

            elif FLAGS.exp == "depths":
              gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"{agent_name}.hidden_layer = {eps}"]

            elif FLAGS.exp == "normalizations":
              gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"{agent_name}.normalization = '{eps}'"]

            elif FLAGS.exp == "inits":
              gin_bindings = get_init_bidings(agent_name, eps, FLAGS.initial_seed)

            elif FLAGS.exp == "activations":
              gin_bindings = [f"{agent_name}.seed={i}", f"{agent_name}.layer_funct = {eps}"]

            elif FLAGS.exp == "update_period":
              gin_bindings = [f"{agent_name}.seed={i}", f"{agent_name}.update_period = {eps}"]

            else:
              print("Error! Check the kind of experiment")

            gin.clear_config()
            gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)

            if FLAGS.wb:
              agent_runner = WandBRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)
            else:
              agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment)

            print(f'Training fixed agent {i}, please be patient, may be a while...')
            agent_runner.run_experiment()
            print('Done training!')
            if FLAGS.wb:
              run.finish()
    print('Finished!')


if __name__ == "__main__":
    app.run(main)
