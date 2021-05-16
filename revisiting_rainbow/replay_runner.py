"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
    """Object that handles running Dopamine experiments with fixed replay buffer."""
    def __init__(self, base_dir, create_agent_fn, create_environment_fn,
                 num_iterations, training_steps, evaluation_steps):
        super().__init__(base_dir, create_agent_fn, create_environment_fn)
        self._num_iterations = num_iterations
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps

    def _run_train_phase(self):
        """Run training phase."""
        self._agent.eval_mode = False
        start_time = time.time()
        for _ in range(self._training_steps):
            self._agent._train_step()  # pylint: disable=protected-access
        time_delta = time.time() - start_time
        tf.logging.info('Average training steps per second: %.2f',
                        self._training_steps / time_delta)

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        self._run_train_phase()

        num_episodes_eval, average_reward_eval = self._run_eval_phase(
            statistics)

        self._save_tensorboard_summaries(iteration, num_episodes_eval,
                                         average_reward_eval)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration, num_episodes_eval,
                                    average_reward_eval):
        """Save statistics as tensorboard summaries.
        Args:
            iteration: int, The current iteration number.
            num_episodes_eval: int, number of evaluation episodes run.
            average_reward_eval: float, The average evaluation reward.
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)
        ])
        self._summary_writer.add_summary(summary, iteration)
