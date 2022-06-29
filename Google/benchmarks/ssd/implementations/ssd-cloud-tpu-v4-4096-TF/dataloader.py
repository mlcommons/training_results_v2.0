# Copyright 2018 Google. All Rights Reserved.
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
# ==============================================================================
"""Data loader and processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import math

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import preprocessor
from object_detection import region_similarity_calculator
from object_detection import target_assigner
from object_detection import tf_example_decoder
from ssd import ssd_constants
from ssd import ssd_matcher


class DefaultBoxes(object):
  """Default anchor bounding boxes for RetinaNet.

  (a) Boxes are generated for the finest-grained feature map (aka level) first
  (which has the largest feature map size), and coarse-grained feature map
  last.

  (b) Default bounding boxes generation follows the order of (anchor_sizes,
  W, H). Therefore, the tensor converted from DefaultBoxes has a shape of [H,
  W, anchor_sizes, 4]. The last dimension is the box coordinates; 'ltrb' is
  [ymin, xmin, ymax, xmax] while 'xywh' is [cy, cx, h, w].

  The behavior of (a) and (b) must match that of the detection head.
  """

  def __init__(self, snap_to_pixel=True):
    """Construct a fixed set of anchor boxes.

    Args:
      snap_to_pixel: Whether to adjust the normalized coordinates such that they
        snap to integer pixels as much as possible (modulo fp errors).
    """

    self.default_boxes = []
    for feature_size, anchor_sizes, aspect_ratios in zip(
        ssd_constants.FEATURE_SIZES, ssd_constants.ANCHOR_SIZES,
        ssd_constants.ASPECT_RATIOS):
      all_sizes = []
      for s, a in it.product(anchor_sizes, aspect_ratios):
        s /= ssd_constants.IMAGE_SIZE
        x = math.sqrt(a)
        all_sizes.append((s * x, s / x))
      assert len(all_sizes) == ssd_constants.NUM_ANCHORS_PER_LOCATION_PER_LEVEL
      for i, j in it.product(range(feature_size), repeat=2):
        cx, cy = (j + 0.5) / feature_size, (i + 0.5) / feature_size
        for h, w in all_sizes:
          box = np.array([cy, cx, h, w])
          # snap to integer pixel
          if snap_to_pixel:
            box = np.round(
                box * ssd_constants.IMAGE_SIZE) / ssd_constants.IMAGE_SIZE
          # clip to image boundary
          box = np.clip(box, 0., 1.)
          self.default_boxes.append(tuple(box))

    assert len(self.default_boxes) == ssd_constants.NUM_SSD_BOXES

    def to_ltrb(cy, cx, h, w):
      return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

    # For IoU calculation
    self.default_boxes_ltrb = tuple(to_ltrb(*i) for i in self.default_boxes)

  def __call__(self, order='ltrb'):
    if order == 'ltrb':
      return self.default_boxes_ltrb
    if order == 'xywh':
      return self.default_boxes


def calc_iou_tensor(box1, box2):
  """ Calculation of IoU based on two boxes tensor,

      Reference to https://github.com/kuangliu/pytorch-ssd
      input:
          box1 (N, 4)
          box2 (M, 4)
      output:
          IoU (N, M)
  """
  N = tf.shape(box1)[0]
  M = tf.shape(box2)[0]

  be1 = tf.tile(tf.expand_dims(box1, axis=1), (1, M, 1))
  be2 = tf.tile(tf.expand_dims(box2, axis=0), (N, 1, 1))

  # Left Top & Right Bottom
  lt = tf.maximum(be1[:, :, :2], be2[:, :, :2])

  rb = tf.minimum(be1[:, :, 2:], be2[:, :, 2:])

  delta = tf.maximum(rb - lt, 0)

  intersect = delta[:, :, 0] * delta[:, :, 1]

  delta1 = be1[:, :, 2:] - be1[:, :, :2]
  area1 = delta1[:, :, 0] * delta1[:, :, 1]
  delta2 = be2[:, :, 2:] - be2[:, :, :2]
  area2 = delta2[:, :, 0] * delta2[:, :, 1]

  iou = intersect / (area1 + area2 - intersect)
  return iou


def ssd_crop(image, boxes, classes):
  """IoU biassed random crop.

  Reference: https://github.com/chauhan-utk/ssd.DomainAdaptation
  """

  num_boxes = tf.shape(boxes)[0]

  def no_crop_check():
    return (tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32) <
            ssd_constants.P_NO_CROP_PER_PASS)

  def no_crop_proposal():
    return (
        tf.ones((), tf.bool),
        tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32),
        tf.ones((num_boxes,), tf.bool),
    )

  def crop_proposal():
    rand_vec = lambda minval, maxval: tf.random_uniform(
        shape=(ssd_constants.NUM_CROP_PASSES, 1),
        minval=minval,
        maxval=maxval,
        dtype=tf.float32)

    width, height = rand_vec(0.3, 1), rand_vec(0.3, 1)
    left, top = rand_vec(0, 1 - width), rand_vec(0, 1 - height)

    right = left + width
    bottom = top + height

    ltrb = tf.concat([left, top, right, bottom], axis=1)

    min_iou = tf.random_shuffle(ssd_constants.CROP_MIN_IOU_CHOICES)[0]
    ious = calc_iou_tensor(ltrb, boxes)

    # discard any bboxes whose center not in the cropped image
    xc, yc = [
        tf.tile(0.5 * (boxes[:, i + 0] + boxes[:, i + 2])[tf.newaxis, :],
                (ssd_constants.NUM_CROP_PASSES, 1)) for i in range(2)
    ]

    masks = tf.reduce_all(
        tf.stack([
            tf.greater(xc, tf.tile(left, (1, num_boxes))),
            tf.less(xc, tf.tile(right, (1, num_boxes))),
            tf.greater(yc, tf.tile(top, (1, num_boxes))),
            tf.less(yc, tf.tile(bottom, (1, num_boxes))),
        ],
                 axis=2),
        axis=2)

    # Checks of whether a crop is valid.
    valid_aspect = tf.logical_and(
        tf.less(height / width, 2), tf.less(width / height, 2))
    valid_ious = tf.reduce_all(tf.greater(ious, min_iou), axis=1, keepdims=True)
    valid_masks = tf.reduce_any(masks, axis=1, keepdims=True)

    valid_all = tf.cast(
        tf.reduce_all(
            tf.concat([valid_aspect, valid_ious, valid_masks], axis=1), axis=1),
        tf.int32)

    # One indexed, as zero is needed for the case of no matches.
    index = tf.range(1, 1 + ssd_constants.NUM_CROP_PASSES, dtype=tf.int32)

    # Either one-hot, or zeros if there is no valid crop.
    selection = tf.equal(tf.reduce_max(index * valid_all), index)

    use_crop = tf.reduce_any(selection)
    output_ltrb = tf.reduce_sum(
        tf.multiply(
            ltrb,
            tf.tile(tf.cast(selection, tf.float32)[:, tf.newaxis], (1, 4))),
        axis=0)
    output_masks = tf.reduce_any(
        tf.logical_and(masks, tf.tile(selection[:, tf.newaxis],
                                      (1, num_boxes))),
        axis=0)

    return use_crop, output_ltrb, output_masks

  def proposal(*args):
    return tf.cond(
        pred=no_crop_check(),
        true_fn=no_crop_proposal,
        false_fn=crop_proposal,
    )

  _, crop_bounds, box_masks = tf.while_loop(
      cond=lambda x, *_: tf.logical_not(x),
      body=proposal,
      loop_vars=[
          tf.zeros((), tf.bool),
          tf.zeros((4,), tf.float32),
          tf.zeros((num_boxes,), tf.bool)
      ],
  )

  filtered_boxes = tf.boolean_mask(boxes, box_masks, axis=0)

  # Clip boxes to the cropped region.
  filtered_boxes = tf.stack([
      tf.maximum(filtered_boxes[:, 0], crop_bounds[0]),
      tf.maximum(filtered_boxes[:, 1], crop_bounds[1]),
      tf.minimum(filtered_boxes[:, 2], crop_bounds[2]),
      tf.minimum(filtered_boxes[:, 3], crop_bounds[3]),
  ],
                            axis=1)

  left = crop_bounds[0]
  top = crop_bounds[1]
  width = crop_bounds[2] - left
  height = crop_bounds[3] - top

  cropped_boxes = tf.stack([
      (filtered_boxes[:, 0] - left) / width,
      (filtered_boxes[:, 1] - top) / height,
      (filtered_boxes[:, 2] - left) / width,
      (filtered_boxes[:, 3] - top) / height,
  ],
                           axis=1)

  cropped_image = tf.image.crop_and_resize(
      image=image[tf.newaxis, :, :, :],
      boxes=crop_bounds[tf.newaxis, :],
      box_indices=tf.zeros((1,), tf.int32),
      crop_size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE),
  )[0, :, :, :]

  cropped_classes = tf.boolean_mask(classes, box_masks, axis=0)

  return cropped_image, cropped_boxes, cropped_classes


def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    if brightness > 0:
      image = tf.image.random_brightness(image, max_delta=brightness)
    if contrast > 0:
      image = tf.image.random_contrast(
          image, lower=1 - contrast, upper=1 + contrast)
    if saturation > 0:
      image = tf.image.random_saturation(
          image, lower=1 - saturation, upper=1 + saturation)
    if hue > 0:
      image = tf.image.random_hue(image, max_delta=hue)
    return image


def encode_labels(gt_boxes, gt_labels, use_spatial_partitioning):
  """Labels anchors with ground truth inputs.

  Args:
    gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
      For each row, it stores [y0, x0, y1, x1] for four corners of a box.
    gt_labels: A integer tensor with shape [N, 1] representing groundtruth
      classes.
    use_spatial_partitioning: whether to use spatial partitioning.

  Returns:
    encoded_classes_per_level: A list of tensors with shape
      [num_anchors_level_i, 1] for each FPN level.
    encoded_boxes_per_level: A list of tensor with shape
      [num_anchors_level_i, 4] for each FPN level.
    match_results_per_level: A list of 1-d tensor with shape
      [num_anchors_level_i] based on `mather.Match.match_results`
    num_positives: scalar tensor storing number of positive boxes in an image.
  """
  similarity_calc = region_similarity_calculator.IouSimilarity()
  matcher = ssd_matcher.Matcher(
      matched_threshold=ssd_constants.MATCH_THRESHOLD_HI,
      unmatched_threshold=ssd_constants.MATCH_THRESHOLD_LO,
      negatives_lower_than_unmatched=True,
      force_match_for_each_row=False,
      allow_low_quality_matches=True)

  box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
      scale_factors=ssd_constants.BOX_CODER_SCALES)

  default_boxes = box_list.BoxList(tf.convert_to_tensor(DefaultBoxes()('ltrb')))
  target_boxes = box_list.BoxList(gt_boxes)

  assigner = target_assigner.TargetAssigner(
      similarity_calc,
      matcher,
      box_coder,
      unmatched_cls_target=tf.constant([ssd_constants.UNMATCHED_CLS_TARGET],
                                       tf.float32))

  encoded_classes, _, encoded_boxes, _, matches = assigner.assign(
      default_boxes, target_boxes, gt_labels)
  num_matched_boxes = tf.reduce_sum(
      tf.cast(tf.greater_equal(matches.match_results, 0), tf.float32))

  encoded_classes = tf.squeeze(tf.cast(encoded_classes, tf.int32), axis=1)
  if ssd_constants.SPLIT_LEVEL:
    num_anchors_per_level = [
        fs**2 * ssd_constants.NUM_ANCHORS_PER_LOCATION_PER_LEVEL
        for fs in ssd_constants.FEATURE_SIZES
    ]
    encoded_classes_per_level = tf.split(encoded_classes, num_anchors_per_level,
                                         0)
    encoded_boxes_per_level = tf.split(encoded_boxes, num_anchors_per_level, 0)
    match_results_per_level = tf.split(matches.match_results,
                                       num_anchors_per_level, 0)
  else:
    encoded_classes_per_level = [encoded_classes]
    encoded_boxes_per_level = [encoded_boxes]
    match_results_per_level = [matches.match_results]

  # FIXME: The if-branch is inconsistent with the new anchor
  # generation order. Need fix if we do spatial partitioning.
  # TODO: Non split-level implementation when
  # use_spatial_partition=True.
  if use_spatial_partitioning:
    assert ssd_constants.SPLIT_LEVEL, ('use_spatial_partitioning=True without '
                                       'SPLIT_LEVEL has not been implemented.')
    transposed_classes_per_level = []
    transposed_boxes_per_level = []
    for i, ecpl in enumerate(encoded_classes_per_level):
      transposed_classes_per_level.append(
          tf.transpose(
              tf.reshape(ecpl, [
                  ssd_constants.NUM_ANCHORS_PER_LOCATION_PER_LEVEL,
                  ssd_constants.FEATURE_SIZES[i], ssd_constants.FEATURE_SIZES[i]
              ]), [1, 2, 0]))
    for i, ebpl in enumerate(encoded_boxes_per_level):
      transposed_boxes_per_level.append(
          tf.transpose(
              tf.reshape(ebpl, [
                  ssd_constants.NUM_ANCHORS_PER_LOCATION_PER_LEVEL,
                  ssd_constants.FEATURE_SIZES[i],
                  ssd_constants.FEATURE_SIZES[i], 4
              ]), [1, 2, 0, 3]))
    return transposed_classes_per_level, transposed_boxes_per_level, match_results_per_level, num_matched_boxes
  else:
    return encoded_classes_per_level, encoded_boxes_per_level, match_results_per_level, num_matched_boxes


def fused_transpose_and_space_to_depth(
    images,
    block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
    transpose_input=True):
  """Fuses space-to-depth and transpose.

  Space-to-depth performas the following permutation, which is equivalent to
  tf.nn.space_to_depth.

  images = tf.reshape(images, [batch, h // block_size, block_size,
                               w // block_size, block_size, c])
  images = tf.transpose(images, [0, 1, 3, 2, 4, 5])
  images = tf.reshape(images, [batch, h // block_size, w // block_size,
                               c * (block_size ** 2)])

  Args:
    images: A tensor with a shape of [batch_size, h, w, c] as the images. The h
      and w can be dynamic sizes.
    block_size: A integer for space-to-depth block size.
    transpose_input: A boolean to indicate if the images tensor should be
      transposed.

  Returns:
    A transformed images tensor.

  """
  batch_size, h, w, c = images.get_shape().as_list()
  images = tf.reshape(
      images,
      [batch_size, h // block_size, block_size, w // block_size, block_size, c])
  if transpose_input:
    if batch_size > 8:
      # HWCN
      images = tf.transpose(images, [1, 3, 2, 4, 5, 0])
      images = tf.reshape(
          images,
          [h // block_size, w // block_size, c * (block_size**2), batch_size])
    else:
      # NWCH
      images = tf.transpose(images, [0, 3, 2, 4, 5, 1])
      images = tf.reshape(
          images,
          [batch_size, w // block_size, c * (block_size**2), h // block_size])
  else:
    images = tf.transpose(images, [0, 1, 3, 2, 4, 5])
    images = tf.reshape(
        images,
        [batch_size, h // block_size, w // block_size, c * (block_size**2)])
  return images


class SSDInputReader(object):
  """Input reader for dataset."""

  def __init__(self,
               file_pattern,
               transpose_input=False,
               is_training=False,
               use_fake_data=False,
               distributed_eval=False,
               count=-1,
               params=None):
    self._file_pattern = file_pattern
    self._transpose_input = transpose_input
    self._is_training = is_training
    self._use_fake_data = use_fake_data
    self._distributed_eval = distributed_eval
    self._count = count
    self._params = params

  @tf.function
  def _parse_example(self, data):
    """Example parser."""
    with tf.name_scope('augmentation'):
      source_id = data['source_id']
      image = data['image']  # dtype uint8
      raw_shape = tf.shape(image)
      boxes = data['groundtruth_boxes']
      classes = tf.reshape(data['groundtruth_classes'], [-1, 1])

      class_map = tf.convert_to_tensor(ssd_constants.CLASS_MAP)
      classes = tf.gather(class_map, classes)
      classes = tf.cast(classes, dtype=tf.float32)

      def normalize(image):
        image -= tf.constant(
            ssd_constants.NORMALIZATION_MEAN,
            shape=[1, 1, 3],
            dtype=image.dtype)
        image /= tf.constant(
            ssd_constants.NORMALIZATION_STD, shape=[1, 1, 3], dtype=image.dtype)
        return image

      if self._is_training:
        image = tf.cast(image, dtype=tf.float32) / 255.0
        # random_horizontal_flip() is hard coded to flip with 50% chance.
        image, boxes = preprocessor.random_horizontal_flip(
            image=image, boxes=boxes)
        image = normalize(image)
        image = tf.image.resize_images(
            image,
            size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        if self._params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)

        encoded_classes_per_level, encoded_boxes_per_level, match_results_per_level, num_matched_boxes = encode_labels(
            boxes, classes, self._params['use_spatial_partitioning'])

        # list can't be used as tf.data structure. Must use tuple instead.
        labels = {
            ssd_constants.NUM_MATCHED_BOXES: num_matched_boxes,
            ssd_constants.BOXES: tuple(encoded_boxes_per_level),
            ssd_constants.CLASSES: tuple(encoded_classes_per_level),
            ssd_constants.MATCH_RESULT: tuple(match_results_per_level)
        }

        return image, labels

      else:
        image = tf.cast(image, dtype=tf.float32) / 255.0
        image = normalize(image)
        image = tf.image.resize_images(
            image,
            size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        if self._params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)

        def trim_and_pad(inp_tensor, dim_1):
          """Limit the number of boxes, and pad if necessary."""
          inp_tensor = inp_tensor[:ssd_constants.MAX_NUM_EVAL_BOXES]
          num_pad = ssd_constants.MAX_NUM_EVAL_BOXES - tf.shape(inp_tensor)[0]
          inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
          return tf.reshape(inp_tensor,
                            [ssd_constants.MAX_NUM_EVAL_BOXES, dim_1])

        boxes, classes = trim_and_pad(boxes, 4), trim_and_pad(classes, 1)

        labels = {
            ssd_constants.BOXES: boxes,
            ssd_constants.CLASSES: classes,
            ssd_constants.SOURCE_ID: tf.string_to_number(source_id, tf.int32),
            ssd_constants.RAW_SHAPE: raw_shape,
        }

        if not self._is_training and self._count > self._params['eval_samples']:
          labels[ssd_constants.IS_PADDED] = data[ssd_constants.IS_PADDED]
        return image, labels

  @tf.function
  def _decode(self, data):
    example_decoder = tf_example_decoder.TfExampleDecoder()
    return example_decoder.decode(data)

  def __call__(self, params):
    batch_size = params['batch_size']
    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    if self._is_training or self._distributed_eval:
      dataset = dataset.shard(params['dataset_num_shards'],
                              params['dataset_index'])
      if self._is_training:
        dataset = dataset.shuffle(
            tf.to_int64(
                max(256, params['dataset_num_shards']) /
                params['dataset_num_shards']))

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    dataset = dataset.interleave(
        _prefetch_dataset,
        cycle_length=32,
        num_parallel_calls=32,
        deterministic=not self._is_training)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(self._decode, num_parallel_calls=64)

    def _mark_is_padded(data):
      sample = data
      sample[ssd_constants.IS_PADDED] = tf.constant(True, dtype=tf.bool)
      return sample

    def _mark_is_not_padded(data):
      sample = data
      sample[ssd_constants.IS_PADDED] = tf.constant(False, dtype=tf.bool)
      return sample

    # Pad dataset to the desired size and mark if the data is padded.
    # During eval/predict, if local_batch_size * num_shards > 5000,
    # original dataset size won't be fit for computations on that number
    # of shards. In this case, will take
    # (local_batch_size - 5000 / num_shards) data from the original dataset
    # on each shard and mark the padded data as `is_padded`.
    # Also mark the original data as `not_padded`.
    # Append the padded data to the original dataset.
    if not self._is_training and self._count > self._params['eval_samples']:
      padded_dataset = dataset.map(_mark_is_padded)
      dataset = dataset.map(_mark_is_not_padded)
      dataset = dataset.concatenate(padded_dataset).take(
          self._count // params['dataset_num_shards'])

    if self._is_training:
      dataset = dataset.filter(
          lambda data: tf.greater(tf.shape(data['groundtruth_boxes'])[0], 0))
      if not self._use_fake_data:
        dataset = dataset.cache().shuffle(64).repeat()
    else:
      dataset = dataset.prefetch(batch_size * 64)

    dataset = dataset.map(self._parse_example, num_parallel_calls=64)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    if self._params['conv0_space_to_depth']:

      def _space_to_depth_fn(images, labels):
        images = fused_transpose_and_space_to_depth(
            images,
            block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            transpose_input=self._transpose_input)
        if self._transpose_input:
          labels[ssd_constants.BOXES] = tuple(
              tf.transpose(box_per_level, [0, 2, 1])
              for box_per_level in labels[ssd_constants.BOXES])
          labels[ssd_constants.CLASSES] = tuple(
              tf.transpose(class_per_level)
              for class_per_level in labels[ssd_constants.CLASSES])
          labels[ssd_constants.MATCH_RESULT] = tuple(
              tf.transpose(match_per_level)
              for match_per_level in labels[ssd_constants.MATCH_RESULT])
        return images, labels

      dataset = dataset.map(_space_to_depth_fn, num_parallel_calls=64)
    elif self._transpose_input:
      # Manually apply the double transpose trick for training data.
      def _transpose_dataset(image, labels):
        if batch_size > 8:
          image = tf.transpose(image, [1, 2, 3, 0])
        else:
          image = tf.transpose(image, [0, 2, 3, 1])
        labels[ssd_constants.BOXES] = tuple(
            tf.transpose(box_per_level, [0, 2, 1])
            for box_per_level in labels[ssd_constants.BOXES])
        labels[ssd_constants.CLASSES] = tuple(
            tf.transpose(class_per_level)
            for class_per_level in labels[ssd_constants.CLASSES])
        labels[ssd_constants.MATCH_RESULT] = tuple(
            tf.transpose(match_per_level)
            for match_per_level in labels[ssd_constants.MATCH_RESULT])
        return image, labels

      dataset = dataset.map(_transpose_dataset, num_parallel_calls=64)

    dataset = dataset.prefetch(64)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = self._params[
        'dataset_threadpool_size']
    dataset = dataset.with_options(options)

    if self._use_fake_data:
      dataset = dataset.take(1).cache().repeat()

    return dataset
