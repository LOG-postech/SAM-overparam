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

# Some parts of the code are borrowed from https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py

"""{mnist, cifar-10, imagenet} input pipeline.
"""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

CIFAR10_MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
CIFAR10_STDDEV_RGB = [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]

CROP_PADDING = 32

IMAGENET_MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def distorted_bounding_box_crop(
    image_bytes,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
  shape = tf.io.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True,
  )
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _resize(image, image_size):
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
      area_range=(0.08, 1.0),
      max_attempts=10,
  )
  original_shape = tf.io.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: _resize(image, image_size),
  )

  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.io.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      (
          (image_size / (image_size + CROP_PADDING))
          * tf.cast(tf.minimum(image_height, image_width), tf.float32)
      ),
      tf.int32,
  )

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height,
      offset_width,
      padded_center_crop_size,
      padded_center_crop_size,
  ])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image):
  """Normalize the given image"""
  image -= tf.constant(IMAGENET_MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(IMAGENET_STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image

def cifar10_augment(image, crop_padding=4, flip_lr=True):
  """Augment small image with random crop and h-flip.
  Args:
    image: image to augment
    crop_padding: random crop range
    flip_lr: if True perform random horizontal flip
  Returns:
    augmented image
  """
  HEIGHT = 32
  WIDTH = 32
  NUM_CHANNELS = 3

  assert crop_padding >= 0
  if crop_padding > 0:
    # Pad with reflection padding
    # (See https://arxiv.org/abs/1605.07146)
    # Section 3
    image = tf.pad(
        image, [[crop_padding, crop_padding],
                [crop_padding, crop_padding], [0, 0]], 'CONSTANT')

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

  if flip_lr:
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  return image

def cifar10_process_train_sample(x):
  cifar10_mean_rgb = tf.constant(CIFAR10_MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
  cifar10_std_rgb = tf.constant(CIFAR10_STDDEV_RGB, shape=[1, 1, 3], dtype=tf.float32)
  image = tf.cast(x['image'], tf.float32)
  image = cifar10_augment(image, crop_padding=4, flip_lr=True)
  image = (image - cifar10_mean_rgb) / cifar10_std_rgb
  batch = {'image': image, 'label': x['label']}
  return batch

def cifar10_process_test_sample(x):
  cifar10_mean_rgb = tf.constant(CIFAR10_MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
  cifar10_std_rgb = tf.constant(CIFAR10_STDDEV_RGB, shape=[1, 1, 3], dtype=tf.float32)
  image = tf.cast(x['image'], tf.float32)
  image = (image - cifar10_mean_rgb) / cifar10_std_rgb
  batch = {'image': image, 'label': x['label']}
  return batch

def imagenet_preprocess_for_train(image_bytes, dtype=tf.float32, image_size=224):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.random_flip_left_right(image)
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image

def imagenet_preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=224):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image

def mnist_process_sample(x): 
  """Proprocess the mnist image (normalizing)"""
  image = tf.cast(x['image'], tf.float32)
  image = image / 255.
  batch = {'image': image, 'label': x['label']}
  return batch

def create_split(dataset, dataset_builder, batch_size, train, cache=False):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.
  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f'train[{start}:{start + split_size}]'
  else:
    if dataset == 'imagenet2012':
      validate_examples = dataset_builder.info.splits['validation'].num_examples
    elif dataset == 'cifar10' or dataset == 'mnist':
      validate_examples = dataset_builder.info.splits['test'].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    if dataset == 'imagenet2012':
      split = f'validation[{start}:{start + split_size}]'
    elif dataset == 'cifar10' or dataset == 'mnist':
      split = f'test[{start}:{start + split_size}]'

  def decode_example(example):
    if dataset == 'cifar10':
      if train:
        return cifar10_process_train_sample(example)
      else:
        return cifar10_process_test_sample(example)
    elif dataset == 'imagenet2012':
      if train:
        image = imagenet_preprocess_for_train(example['image'])
      else:
        image = imagenet_preprocess_for_eval(example['image'])
      return {'image': image, 'label': example['label']}
    elif dataset == 'mnist':
      return mnist_process_sample(example)
    

  kwargs=dict()
  num_train_samples = dataset_builder.info.splits['train'].num_examples

  
  if dataset == 'cifar10':
    ds = tfds.load(dataset, split=split, **kwargs).cache()
  elif dataset == 'imagenet2012':
    ds = dataset_builder.as_dataset(split=split, decoders={'image': tfds.decode.SkipDecoding(),})
  elif dataset == 'mnist':
    ds = dataset_builder.as_dataset(split=split)
  
  if dataset == 'imagenet2012':
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 0
    ds = ds.with_options(options)

    if cache:
      ds = ds.cache()

    if train:
      ds = ds.repeat()
      ds = ds.shuffle(2_000, seed=0)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train:
      ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
  else:
    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
      ds = ds.shuffle(num_train_samples, seed=0, reshuffle_each_iteration=True)

    if train:
      ds = ds.batch(batch_size)
      ds = ds.repeat()
    else:
      ds = ds.batch(batch_size)
      ds = ds.repeat()

    ds = ds.prefetch(10)

  return ds
