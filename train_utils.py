# Some parts of the code are borrowed from https://github.com/google/flax/blob/main/examples/imagenet/train.py.

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import tensorflow as tf
import optax
import ml_collections
import tensorflow_datasets as tfds

from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import common_utils, train_state
from flax import jax_utils
from flax import struct, core

from functools import partial

from pruner import compute_mask, apply_mask, compute_score
from input_pipeline import create_split

"""
Utility functions for training the network.
"""

class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale
  mask: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

@jax.jit
def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(sum(
      [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / (gradient_norm + 1e-12), y)
  return normalized_gradient

class AverageMeter (object):
  """Class for calculating the average"""
  def __init__(self):
      self.reset ()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

def compute_metrics(logits, labels, loss_fn):
  """Compute loss and the accuracy"""
  loss = loss_fn(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics

def cross_entropy_loss(logits, labels, num_classes):
  """Standard cross entropy loss"""
  one_hot_labels = common_utils.onehot(labels, num_classes)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)

def prepare_dataset(config):
  """Create data iterators and the related infos"""
  if config.dataset == 'cifar10':
    _, image_size, _, _ = input_shape = (1, 32, 32, 3)
    num_classes = 10
  elif config.dataset == 'imagenet2012':
    _, image_size, _, _ = input_shape = (1, 224, 224, 3)
    num_classes = 1000
  elif config.dataset == 'mnist':
    _, image_size, _, _ = input_shape = (1, 28, 28, 1)
    num_classes = 10

  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_builder = tfds.builder(config.dataset)
  if config.dataset == 'imagenet2012':
    manual_dataset_dir = "your_directory" # replace the directory
    imagenet_download_config = tfds.download.DownloadConfig(
      extract_dir='./tmp/',
    manual_dir = manual_dataset_dir)
    dataset_builder.download_and_prepare(download_config=imagenet_download_config)
    test_set = 'validation'
  else:
    dataset_builder.download_and_prepare()
    test_set = 'test'

  train_iter = create_input_iter(
      config.dataset,
      dataset_builder, local_batch_size, image_size, input_dtype, train=True,
      cache=config.cache)
  
  eval_iter = create_input_iter(
      config.dataset,
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=config.cache)

  num_train_samples = dataset_builder.info.splits['train'].num_examples
  num_val_samples = dataset_builder.info.splits[test_set].num_examples
  
  return train_iter, eval_iter, num_train_samples, num_val_samples, num_classes, input_shape

def initialized(key, input_shape, model, batch_stats=True):
  """Initialize the parameters and the batchnorm stats"""
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key, 'dropout': key}, jnp.ones(input_shape))
  if not batch_stats:
    return variables['params'], {}
  else:
    return variables['params'], variables['batch_stats']

def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_util.tree_map(_prepare, xs)

def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  
  if config.lr_scheduler == 'step':
    return optax.piecewise_constant_schedule(base_learning_rate, boundaries_and_scales={int(config.num_epochs * 0.5 * steps_per_epoch): 0.1, int(config.num_epochs * 0.75 * steps_per_epoch): 0.1})
  elif config.lr_scheduler == 'step_mnist':
    return optax.piecewise_constant_schedule(base_learning_rate, boundaries_and_scales={int(config.num_epochs * 0.25 * steps_per_epoch): 0.1, int(config.num_epochs * 0.5 * steps_per_epoch): 0.1, int(config.num_epochs * 0.75 * steps_per_epoch): 0.1})
  elif config.lr_scheduler == 'imagenet_cosine':
    # Taken from https://github.com/google-research/vision_transformer/blob/main/vit_jax/utils.py
    def step_fn(step):
      warmup_steps = config.warmup_steps
      lr = base_learning_rate
      total_steps = config.num_epochs * steps_per_epoch
      progress = (step - warmup_steps) / float(total_steps - warmup_steps)
      progress = jnp.clip(progress, 0.0, 1.0)
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
      if warmup_steps:
        lr = lr * jnp.minimum(1., step / warmup_steps)
      return jnp.asarray(lr, dtype=jnp.float32)
    return step_fn
    
def create_pruning_mask(params, pruner, sparsity, loss_fn, **kwargs):
  """Create pruning mask"""
  batch = next(kwargs['train_iter'])
  kwargs['train_iter'].__init__()

  kwargs['loss_fn'] = loss_fn

  p_compute_score = jax.pmap(partial(compute_score, sc_type=pruner, **kwargs), axis_name='batch')
  scores = p_compute_score(params=jax_utils.replicate(params), batch=batch)
  scores = jax_utils.unreplicate(scores)
  mask = compute_mask(scores, sparsity, pruner)
  masked_params = apply_mask(params, mask)

  return masked_params, mask

def create_input_iter(dataset, dataset_builder, batch_size, image_size, dtype, train,
                      cache):
  """Create data iterator"""
  ds = create_split(
    dataset, dataset_builder, batch_size, train=train, cache=cache)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


def create_train_state(rng, model, input_shape, learning_rate_fn, loss_fn, config, half_precision=False, **kwargs):
  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  params, batch_stats = initialized(rng, input_shape, model, batch_stats=('ResNet' in config.model))

  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=False,
  )

  # compute pai mask
  params, mask = create_pruning_mask(params, config.pruner, config.sparsity, loss_fn, key=rng, batch_stats=batch_stats, apply_fn=model.apply, **kwargs)

  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale,
      mask=mask)

  return state