"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import wandb

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class WandBRunner(run_experiment.Runner):
    """Object that handles running Dopamine experiments with fixed replay buffer."""

    def _save_tensorboard_summaries(self, iteration, num_episodes_train, average_reward_train, num_episodes_eval,
                                    average_reward_eval, average_steps_per_second):
        """Save statistics as tensorboard summaries.
        Args:
            iteration: int, The current iteration number.
            num_episodes_eval: int, number of evaluation episodes run.
            average_reward_eval: float, The average evaluation reward.
        """
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(tag='Train/NumEpisodes', simple_value=num_episodes_train),
            tf.compat.v1.Summary.Value(tag='Train/AverageReturns', simple_value=average_reward_train),
            tf.compat.v1.Summary.Value(tag='Train/AverageStepsPerSecond', simple_value=average_steps_per_second),
            tf.compat.v1.Summary.Value(tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
            tf.compat.v1.Summary.Value(tag='Eval/AverageReturns', simple_value=average_reward_eval)
        ])
        self._summary_writer.add_summary(summary, iteration)
        wandb.log({
            'Train/NumEpisodes': num_episodes_train,
            'Train/AverageReturns': average_reward_train,
            'Train/AverageStepsPerSecond': average_steps_per_second,
            'Eval/NumEpisodes': num_episodes_eval,
            'Eval/AverageReturns': average_reward_eval
        })
