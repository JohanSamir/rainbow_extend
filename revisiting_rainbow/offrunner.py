import numpy as np
import os

import jax.numpy as jnp
import dopamine
from dopamine.jax.agents.rainbow.rainbow_agent import JaxRainbowAgent
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment, checkpointer
from absl import flags
import gin.tf
import sys

@gin.configurable
class OffRunner(run_experiment.Runner):
    def __init__(self, base_dir, create_agent_fn,
               create_environment_fn, _pretrained_agent):
        super().__init__(base_dir, create_agent_fn,
                                        create_environment_fn)
        self._num_iterations = 30
        self._training_steps = 1000
        self._evaluation_steps = 200
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
                self._end_episode(reward)
                if self._agent.eval_mode:
                    action = self._agent.begin_episode(observation)
                else:
                    action = self._pretrained_agent.begin_episode(observation)
                    self._agent._store_transition(jnp.reshape(observation, (4,1)), action, reward, is_terminal)
                    self._agent._train_step()
            else:
                if self._agent.eval_mode:
                    action = self._agent.step(reward, observation)
                else:
                    action = self._pretrained_agent.step(reward, observation)
                    self._agent._store_transition(jnp.reshape(observation, (4,1)), action, reward, is_terminal)
                    self._agent._train_step()

        self._end_episode(reward)

        return step_number, total_reward