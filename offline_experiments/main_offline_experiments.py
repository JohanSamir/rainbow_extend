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

from constants import agents, epsilons, learning_rates, widths, depths, normalizations, inits, activations, get_init_bidings

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "cartpole", "the environment the experiment will be run in")

flags.DEFINE_string("agent", "dqn", "the agent used in the experiment")

flags.DEFINE_integer("initial_seed", "1", "the program will run seeds [initial_seed, initial_seed + 5)")

flags.DEFINE_string("exp", "normalizations", "the experiment will be run in")

flags.DEFINE_boolean("wb", "False", "the program won't use weights&biases")

experiments = {
        "epsilon": epsilons,
        "learning_rate":learning_rates,
        "width":widths,
        "depth":depths,
        "normalization":normalizations,
        "init":inits,
        "activation":activations,
}
num_runs = 1  #7
# `path=os.environ['AIP_TENSORBOARD_LOG_DIR']`

path = "../../extending_rainbow_exps/"  #TODO point to cloud bucket


def main(_):

    def create_agent(sess, environment, summary_writer=None, memory=None):
        ag = agents[FLAGS.agent](num_actions=environment.action_space.n)
        if memory is not None:
            ag._replay = memory
            ag._replay.replay_capacity = (50000 * 0.2)
        return ag

    for eps in experiments[FLAGS.exp]:
        if FLAGS.exp=="activation":
            eps = "'" + activations[eps]['layer_fun'] + "'"
        for i in range(FLAGS.initial_seed, FLAGS.initial_seed + num_runs):
            if FLAGS.wb:
                if FLAGS.exp == "learning_rate":
                    rep = "learning rate"
                else:
                    rep = FLAGS.exp
                run = wandb.init(project="extending-rainbow",
                                entity="ext-rain",
                                config={
                                    "random seed": i,
                                    "agent": FLAGS.agent,
                                    "environment": FLAGS.env,
                                    rep: eps, 
                                    "varying": rep,
                                    "online": False,
                                },
                                reinit=True)
            agent_name = agents[FLAGS.agent].__name__

            LOG_PATH = os.path.join(os.path.join(f'{path}/{FLAGS.agent}/{FLAGS.env}/{FLAGS.exp}_{eps}_online', f'test{i}'))
            sys.path.append(path)
            gin_file = f'Configs/{FLAGS.agent}_{FLAGS.env}.gin'

            if FLAGS.exp == "epsilon":
                gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"create_optimizer.eps = {eps}"]

            elif FLAGS.exp == "learning_rate":
                gin_bindings = [f"{agent_name}.seed={i}", f"create_optimizer.learning_rate = {eps}"]

            elif FLAGS.exp == "width":
                gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"{agent_name}.neurons = {eps}"]

            elif FLAGS.exp == "depth":
                gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"{agent_name}.hidden_layer = {eps}"]

            elif FLAGS.exp == "normalization":
                gin_bindings = [f"{agent_name}.seed={FLAGS.initial_seed}", f"{agent_name}.normalization = '{eps}'"]

            elif FLAGS.exp == "init":
                gin_bindings = get_init_bidings(agent_name, eps, FLAGS.initial_seed)

            elif FLAGS.exp == "activation":
                gin_bindings = [f"{agent_name}.seed={i}", f"{agent_name}.layer_funct = {eps}"]

            else:
                print("Error! Check the kind of experiment")

            gin.clear_config()
            gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
            trained_agent = run_experiment.TrainRunner(LOG_PATH, create_agent)

            print(f'Loaded trained {FLAGS.agent} in {FLAGS.env}')
                
            LOG_PATH = os.path.join(f'{path}/{FLAGS.agent}/{FLAGS.env}/{FLAGS.exp}_{eps}_offline', f'test{i}')
            offline_runner = FixedReplayRunner(base_dir=LOG_PATH,
                                                create_agent_fn=functools.partial(
                                                    create_agent,
                                                    memory=trained_agent._agent._replay,
                                                ),
                                                use_wb=FLAGS.wb,
                                                num_iterations=30,
                                                training_steps=1000,
                                                evaluation_steps=200,
                                                create_environment_fn=gym_lib.create_gym_environment)
            print(f'Training fixed agent {i}, please be patient, may be a while...')
            offline_runner.run_experiment()
            print('Done fixed training!')
            if FLAGS.wb:
                run.finish()
        print('Finished!')


if __name__ == "__main__":
    app.run(main)