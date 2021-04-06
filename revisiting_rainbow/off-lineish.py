import numpy as np
import os

import dopamine
from dopamine.jax.agents.rainbow.rainbow_agent import JaxRainbowAgent
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment, checkpointer
from absl import flags
import gin.tf
import sys
path = "."
sys.path.append(".")

from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from agents.quantile_agent_new import *
from agents.implicit_quantile_agent_new import *

@gin.configurable
class OffRunner(run_experiment.Runner):
    def __init__(self, base_dir, create_agent_fn,
               create_environment_fn, _pretrained_agent):
        super(OffRunner, self).__init__(base_dir, create_agent_fn,
                                        create_environment_fn)
        self._pretrained_agent = _pretrained_agent

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.
        Returns:
        The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1

            # if self._clip_rewards:
                # Perform reward clipping.
            reward = np.clip(reward, -1, 1)

            if (self._environment.game_over or
                step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._end_episode(reward, is_terminal)
                action = self._pretrained_agent.begin_episode(observation)
            else:
                action = self._pretrained_agent.step(reward, observation)

        self._end_episode(reward, is_terminal)

        return step_number, total_reward

ags = {
    # 'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    # 'quantile': JaxQuantileAgentNew,
    # 'implicit': JaxImplicitQuantileAgentNew,
}

names = {
    # 'dqn': "JaxDQNAgentNew",
    'rainbow': "JaxRainbowAgentNew",
    # 'quantile': "JaxQuantileAgentNew",
    # 'implicit': "JaxImplicitQuantileAgentNew",
}


num_runs = 1
training_steps = 10
ckpt_path = "../../results/rainbow/512_test10/checkpoints/ckpt.29"
env = gym_lib.create_gym_environment("CartPole")


for agent in ags:
    for i in range (num_runs):
        def create_agent(sess, environment, summary_writer=None):
            return ags[agent](num_actions=environment.action_space.n)
        
        LOG_PATH = os.path.join(path, f'../../test_joao/{agent}/test{i}')
        gin_file = f'./Configs/{agent}_cartpole.gin'
        gin.parse_config_file(gin_file)

        new_agent = JaxRainbowAgentNew(num_actions=env.action_space.n)

        exp_data = checkpointer.Checkpointer(path)._load_data_from_file(ckpt_path)
        trained_agent = JaxRainbowAgentNew(num_actions=env.action_space.n, eval_mode=True)
        trained_agent.unbundle(ckpt_path, 29, exp_data)
        agent_runner = OffRunner(LOG_PATH, create_agent, gym_lib.create_gym_environment, trained_agent)
        print(f'Training agent {i+1}, please be patient, may be a while...')
        agent_runner.run_experiment()
        print('Done training!')
        # r = 0
        # obs = env.reset()
        # for _ in range(training_steps):
        #     a = trained_agent.step(r, obs)
        #     obs, r, terminal, _ = env.step(a)
        #     new_agent._store_transition(jnp.reshape(obs, (4,1)), a, r, terminal)
        #     new_agent._train_step()