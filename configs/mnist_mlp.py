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

# Copyright 2021 The Flax Authors.
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
"""MNIST / 3-layer MLP"""

import ml_collections

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # As defined in the `models` module.
  config.model = 'MLP'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'mnist'

  config.lr_scheduler = 'step_mnist'
  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.weight_decay = 0.0001

  config.num_epochs = 100
  config.log_every_steps = 100

  config.cache = False
  config.half_precision = False

  config.optimizer = 'sam'
  config.rho = 0.05

  config.seed = 1

  config.pruner='random'
  config.sparsity = 0.0

  config.num_neurons=[300, 100]

  config.restore_checkpoint = False

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1
  return config