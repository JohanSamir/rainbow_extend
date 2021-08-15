# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Launcher for Dopamine.

Usage:
xmanager launch examples/dopaminelauncher.py -- \
  --gin_file=https://raw.githubusercontent.com/google/dopamine/master/dopamine/agents/dqn/configs/dqn_mountaincar.gin
"""
import asyncio
import os

import itertools
from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local
from xmanager.cloud import caip

FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     'gin_file',
#     'https://raw.githubusercontent.com/JohanSamir/rainbow_extend/main/revisiting_rainbow/Configs/rainbow_cartpole.gin',
#     'Gin file pulled from https://github.com/JohanSamir/rainbow_extend.')
flags.DEFINE_string('tensorboard', None, 'Tensorboard instance.')


def main(_):
  with xm_local.create_experiment(experiment_title='rainbow-dqn') as experiment:
    # gin_file = os.path.basename(FLAGS.gin_file)
    # add_instruction = f'ADD {FLAGS.gin_file} {gin_file}'
    # if FLAGS.gin_file.startswith('http'):
      # add_instruction = f'RUN wget -O ./{gin_file} {FLAGS.gin_file}'
    spec = xm.PythonContainer(
        docker_instructions=[
            'RUN apt update && apt install -y python3-opencv',
            'RUN pip install dopamine-rl',
            'COPY . workdir',
            'WORKDIR workdir',
            # add_instruction,
        ],
        path='.',
        entrypoint=xm.ModuleName('main_offline_experiments'),
    )

    [executable] = experiment.package([
        xm.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Caip.Spec(),
            args={
                'env': "cartpole",
                'agent': "dqn"
            },
        ),
    ])

    batch_sizes = [64, 1024]
    learning_rates = [0.1, 0.001]
    trials = list(
        dict([('batch_size', bs), ('learning_rate', lr)])
        for (bs, lr) in itertools.product(batch_sizes, learning_rates))

    tensorboard = FLAGS.tensorboard
    if not tensorboard:
      tensorboard = caip.client().create_tensorboard('cifar10') # TODO add meaningful name
      tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)
    output_dir = os.environ['GOOGLE_CLOUD_BUCKET_NAME']
    output_dir = os.path.join(output_dir, str(experiment.experiment_id))
    tensorboard_capability = xm_local.TensorboardCapability(
        name=tensorboard, base_output_directory=output_dir)
    work_units = []
    for hyperparameters in trials:
      hyperparameters = dict(hyperparameters)
      experiment.add(
        xm.Job(
            executable=executable,
            executor=xm_local.Caip(xm.JobRequirements(t4=FLAGS.gpus_per_node), 
              tensorboard=tensorboard_capability),
            args=hyperparameters,
        ))


if __name__ == '__main__':
  app.run(main)