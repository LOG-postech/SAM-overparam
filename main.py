# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Some parts of the code are borrowed from https://github.com/google/flax/blob/main/examples/imagenet/main.py.

"""
Main file to run the code.
"""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

import train

import os
import shutil

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  # For reproducible results
  os.environ["XLA_FLAGS"] = "xla_gpu_deterministic_reductions"
  os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
  
  if FLAGS.config.optimizer == 'sam':
    assert (FLAGS.config.rho > 0.0)

    workdir_suffix = os.path.join(
        'dataset_' + FLAGS.config.dataset,
        'optimizer_' + FLAGS.config.optimizer,
        'model_' + FLAGS.config.model, 
        'lr_' + str(FLAGS.config.learning_rate),
        'wd_' + str(FLAGS.config.weight_decay),
        'rho_' + str(FLAGS.config.rho),
        'pruner_' + str(FLAGS.config.pruner),
        'sparsity_' + str(FLAGS.config.sparsity),
        'seed_' + str(FLAGS.config.seed)
        )

  output_dir = os.path.join(FLAGS.workdir, workdir_suffix)

  if not FLAGS.config.restore_checkpoint:
    if os.path.exists(output_dir): # job restarted by cluster
      for f in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, f)):
          shutil.rmtree(os.path.join(output_dir, f))
        else:
          os.remove(os.path.join(output_dir, f))
    else:
      os.makedirs(output_dir)

  train.train_and_evaluate(FLAGS.config, output_dir)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
