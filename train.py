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

# Some parts of the code are borrowed from https://github.com/google/flax/blob/main/examples/imagenet/train.py.


"""
File to train the network.
"""

import time

from absl import logging
from clu import periodic_actions
import numpy as np
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import ml_collections
import tensorflow as tf

import math
from functools import partial

import models
from pruner import apply_mask
from train_utils import *

def create_model(config, num_classes):
  """Create the model."""
  if config.model == 'MLP':
    model_cls = getattr(models, config.model)
    return model_cls(num_classes=num_classes, num_neurons=config.num_neurons)
  elif 'ResNet' in config.model:
    model_cls = getattr(models, config.model)
    return model_cls(num_classes=num_classes, num_filters=config.num_filters)

@partial(jax.jit, static_argnames = ["optimizer", "loss_type"])
def train_step(state, batch, key, weight_decay, optimizer, rho, loss_type):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    params = apply_mask(params, state.mask) # apply pruning mask
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'],
        rngs=dict(dropout=key),
        train=True,
        mutable=['batch_stats'])
    loss = loss_type(logits, batch['label'])
    return loss, (new_model_state, logits)
  
  def get_sam_gradient(params, rho):
    """Returns the gradient of the SAM loss loss, updated state and logits.

    See https://arxiv.org/abs/2010.01412 for more details.

    Args:
      model: The model that we are training.
      rho: Size of the perturbation.
    """
    # compute gradient on the whole batch
    (_, (inner_state, logits)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grad = dual_vector(grad)
    noised_params = jax.tree_map(lambda p, b: p + rho * b, params, grad)
    (_, (_, _)), grad = jax.value_and_grad(
        loss_fn, has_aux=True)(noised_params)
    return (inner_state, logits), grad

  if optimizer == 'sgd':  # SGD
    (_, (new_model_state, logits)), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(
            state.params)
  elif optimizer == 'sam':  # SAM
    (new_model_state, logits), grads = get_sam_gradient(state.params, rho)

  # We manually apply weight decay in this way.
  grads = jax.tree_map(lambda g, p: g + weight_decay * p, grads, state.params)
  
  grads = jax.lax.pmean(grads, axis_name='batch')

  metrics = compute_metrics(logits, batch['label'], loss_type)

  new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  
  return new_state, metrics

@partial(jax.jit, static_argnames = ["loss_type"])
def eval_step(state, batch, loss_type):
  """Evaluate the model on the test data."""
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(
      variables, batch['image'], train=False, mutable=False)
  return compute_metrics(logits, batch['label'], loss_type)

def restore_checkpoint(state, workdir):
  """Restore the model from the checkpoint."""
  return checkpoints.restore_checkpoint(workdir, state)

def save_checkpoint(state, workdir):
  """Save the model checkpoint."""
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if state.batch_stats == {}:
    return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    Final TrainState.
  """
  logging.info(config)

  rng = random.PRNGKey(config.seed)
  tf.random.set_seed(config.seed)
  np.random.seed(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')    
    
  ############################# Prepare Dataset #############################
  train_iter, eval_iter, num_train_samples, num_val_samples, num_classes, input_shape = prepare_dataset(config)
  
  steps_per_epoch = (
      math.ceil(num_train_samples / config.batch_size)
  )

  if config.steps_per_eval == -1:
    steps_per_eval = math.ceil(num_val_samples / config.batch_size)
  else:
    steps_per_eval = config.steps_per_eval
  
  ############################# Prepare lr & Model #############################

  base_learning_rate = config.learning_rate

  model = create_model(config, num_classes)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)


  ############################# Prepare / Restore State #############################

  loss_type = partial(cross_entropy_loss, num_classes=num_classes)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, model, input_shape, learning_rate_fn, loss_type, config, half_precision=config.half_precision, train_iter=train_iter)

  if config.restore_checkpoint:
    state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  epoch_offset = int(state.step) // steps_per_epoch
  state = jax_utils.replicate(state)
  
  
  ############################# jit / pmap train_step #############################
  
  jitted_train_step = jax.jit(train_step, static_argnames=["optimizer", "loss_type"])
  
  p_train_step = jax.pmap(
    partial(jitted_train_step, weight_decay=config.weight_decay, optimizer=config.optimizer, 
            rho=config.rho, loss_type=loss_type),
    axis_name='batch',
    )
  p_eval_step = jax.pmap(partial(eval_step, loss_type=loss_type), axis_name='batch')


  ############################# Start Training #############################

  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  logging.info('Initial compilation, this might take some minutes...')

  total_steps = 0

  best_test_acc = -1
  best_epoch = -1 

  for epoch in range(epoch_offset, int(config.num_epochs)):
    logging.info("Epoch %d / %d " % (epoch + 1, int(config.num_epochs)))

    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    start_time = time.time()
    for step in range(steps_per_epoch):
      batch = next(train_iter)
      rng, step_rng = jax.random.split(rng)
      sharded_keys = common_utils.shard_prng_key(step_rng)
      state, metrics = p_train_step(state, batch, sharded_keys)
      train_loss_meter.update(metrics['loss'].mean(), len(batch['label'][0]))
      train_acc_meter.update(metrics['accuracy'].mean(), len(batch['label'][0]))

      total_steps += 1

      if total_steps % config.log_every_steps == 0:
        logging.info("Epoch[%d] Step [%d/%d]: loss %.4f acc %.4f (time elapsed: %.4f)" % (epoch + 1, step, steps_per_epoch, metrics['loss'].mean(), metrics['accuracy'].mean(), time.time() - start_time))


    cur_time = time.time()
    test_loss_meter = AverageMeter()
    test_acc_meter = AverageMeter()

    lr = learning_rate_fn(steps_per_epoch * epoch)
    
    state = sync_batch_stats(state)
  
    for step in range(steps_per_eval):
      batch = next(eval_iter)
      metrics = p_eval_step(state, batch)
      test_loss_meter.update(metrics['loss'].mean(), len(batch['label'][0]))
      test_acc_meter.update(metrics['accuracy'].mean(), len(batch['label'][0]))

    if test_acc_meter.avg > best_test_acc:
      best_test_acc = test_acc_meter.avg
      best_epoch = epoch
    elapsed_time = cur_time - start_time
    logging.info("Train: loss %.4f acc %.4f; Val: loss %.4f acc %.4f (lr %.4f / took %.2f seconds) \n" % (train_loss_meter.avg, train_acc_meter.avg, test_loss_meter.avg, test_acc_meter.avg, lr, elapsed_time))

    if (epoch + 1) % 1 == 0 or (epoch + 1) == int(config.num_epochs):
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info("Best test acc %.4f at epoch %d" % (best_test_acc, best_epoch + 1))

  return state


