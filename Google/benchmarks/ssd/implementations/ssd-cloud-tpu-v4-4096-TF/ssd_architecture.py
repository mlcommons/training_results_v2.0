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
"""RetinaNet (via ResNeXt50) model definition.

Defines the RetinaNet model and loss functions from this paper:

https://arxiv.org/pdf/1708.02002

Uses the ResNeXt model as backbone:

https://arxiv.org/pdf/1611.05431.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import tensorflow.compat.v1 as tf

from ssd import ssd_constants
from ssd import ssd_fpn
from ssd import ssd_head
from ssd import utils
from util import image_util

_NMS_TILE_SIZE = 256

FLAGS = tf.flags.FLAGS

_mllog_weight_init = utils.mllog_weight_init


def _bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.

  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  with tf.name_scope('bbox_overlap'):
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    return iou


def _self_suppression(iou, _, iou_sum):
  batch_size = tf.shape(iou)[0]
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
  iou_suppressed = tf.reshape(
      tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
  return [
      iou_suppressed,
      tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  batch_size = tf.shape(boxes)[0]
  new_slice = tf.slice(boxes, [0, inner_idx * _NMS_TILE_SIZE, 0],
                       [batch_size, _NMS_TILE_SIZE, 4])
  iou = _bbox_overlap(new_slice, box_slice)
  ret_slice = tf.expand_dims(
      tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  num_tiles = tf.shape(boxes)[1] // _NMS_TILE_SIZE
  batch_size = tf.shape(boxes)[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * _NMS_TILE_SIZE, 0],
                       [batch_size, _NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      _cross_suppression, [boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = _bbox_overlap(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(_NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(_NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _ = tf.while_loop(
      lambda _iou, loop_condition, _iou_sum: loop_condition, _self_suppression,
      [iou, tf.constant(True),
       tf.reduce_sum(iou, [1, 2])])
  suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(tf.expand_dims(
      box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
          boxes, [batch_size, num_tiles, _NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = tf.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = _bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.

  Returns:
    nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
      dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
      same dtype as input boxes.
  """
  # TODO: Filter out score <= ssd_constants.MIN_SCORE.
  with tf.name_scope('nms'):
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    pad = tf.cast(
        tf.ceil(tf.cast(num_boxes, tf.float32) / _NMS_TILE_SIZE),
        tf.int32) * _NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(tf.cast(scores, tf.float32), [[0, 0], [0, pad]])
    num_boxes += pad

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
      return tf.logical_and(
          tf.reduce_min(output_size) < max_output_size,
          idx < num_boxes // _NMS_TILE_SIZE)

    selected_boxes, _, output_size, _ = tf.while_loop(
        _loop_cond, _suppression_loop_body, [
            boxes, iou_threshold,
            tf.zeros([batch_size], tf.int32),
            tf.constant(0)
        ])
    idx = num_boxes - tf.cast(
        tf.nn.top_k(
            tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
            tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
        tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(
        idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
    boxes = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), idx),
        [batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
            output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(
        tf.gather(tf.reshape(scores, [-1, 1]), idx),
        [batch_size, max_output_size])
    scores = scores * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
            output_size, [-1, 1]), scores.dtype)
    return scores, boxes


class GroupConv2D(tf.layers.Layer):
  """A grouped 2D convolution layer.

  Attributes:
    filters: Number of output channels.
    kernel_size: Spatial convolution kernel size (width and height).
    strides: Spatial convolution stride.
    padding: One of "valid" or "same" (case-insensitive).
    data_format: `channels_last` (default) or `channels_first`.
    groups: The number of groups to divide the input channels into (aka
      "cardinality" in the paper). Convolution along the channel dimension is
      performed within a groups. Both input and output channels must be
      divisible by `groups`.
    kernel_initializer: An initializer for the convolution kernel.
    trainable: Whether the layer is trainable or not.
    name: Layer name.
    **kwargs: Arguments passed through to tf.layers.Layer.
  """

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int = 1,
               padding: str = 'same',
               data_format: str = 'channels_last',
               groups: int = 1,
               kernel_initializer: ... = None,
               trainable: bool = True,
               name: str = 'GroupConv2D',
               **kwargs):
    super(GroupConv2D, self).__init__(name=name, **kwargs)

    if filters % groups != 0:
      raise ValueError('`filters` must be divisible by `groups`.')

    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.data_format = data_format
    self.groups = groups
    self.kernel_initializer = kernel_initializer
    self.trainable = trainable

  def build(self, input_shape):
    in_channels = (
        input_shape[-1]
        if self.data_format == 'channels_last' else input_shape[1])
    if in_channels % self.groups != 0:
      raise ValueError('Input channels must be divisible by `groups`.'
                       ' Check input shape and data format.')

    filter_shape = [
        self.kernel_size, self.kernel_size, in_channels // self.groups,
        self.filters
    ]

    self.kernel = self.add_weight(
        name='kernel',
        shape=filter_shape,
        initializer=self.kernel_initializer,
        trainable=self.trainable)

    # Another option is using fixed 128, which may require setting
    # use_einsum_for_projection
    actual_in = in_channels
    self.kernel = tf.broadcast_to(
        tf.expand_dims(self.kernel, axis=2), [
            self.kernel_size, self.kernel_size, actual_in //
            (in_channels // self.groups), in_channels // self.groups,
            self.filters
        ])
    new_kernel_shape = [
        self.kernel_size, self.kernel_size, actual_in, self.filters
    ]
    self.kernel = tf.reshape(self.kernel, new_kernel_shape)
    mask1 = tf.expand_dims(
        tf.range(actual_in) // (in_channels // self.groups), axis=1)
    mask2 = tf.expand_dims(
        tf.range(self.filters) // (in_channels // self.groups) % actual_in,
        axis=0)
    mask = tf.broadcast_to(tf.equal(mask1, mask2), new_kernel_shape)
    self.kernel = tf.where(mask, self.kernel,
                           tf.zeros(new_kernel_shape, dtype=self.kernel.dtype))

    super(GroupConv2D, self).build(input_shape)

  def call(self, inputs):
    # tf.nn.conv2d natively supports group conv on TPU and GPU when input
    # channels doesn't match given kernel dimensions. Doesn't work on CPU.
    output = tf.nn.conv2d(
        inputs,
        filter=self.kernel,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=('NHWC' if self.data_format == 'channels_last' else 'NCHW'))
    return output


def group_conv2d(inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
  """Returns a grouped 2D convolution on inputs.

  Args:
    inputs: Input tensor.
    *args: Positional arguments passed to GroupConv2D.
    **kwargs: Keyword arguments passed to GroupConv2D.

  Returns:
      Output tensor.
  """
  obj = GroupConv2D(*args, dtype=inputs.dtype, **kwargs)
  obj.build(inputs.get_shape())
  return obj.call(inputs)


class FrozenBatchNorm2d(tf.layers.Layer):
  """A parallel of PyTorch's FrozenBatchNorm2d.

  Essentially, it's just an affine function without all the bells and whistles
  of batch_normalization.

  https://pytorch.org/vision/stable/_modules/torchvision/ops/misc.html#FrozenBatchNorm2d
  """

  def __init__(self,
               channels: int,
               eps: float = 1e-5,
               data_format: str = 'channels_last',
               name: str = 'batch_normalization',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.channels = channels
    self.eps = eps
    self.data_format = data_format

  def build(self, input_shape: tf.TensorShape):
    self.gamma = self.add_weight(
        name='gamma', shape=[self.channels], trainable=False)
    self.beta = self.add_weight(
        name='beta', shape=[self.channels], trainable=False)
    self.moving_mean = self.add_weight(
        name='moving_mean', shape=[self.channels], trainable=False)
    self.moving_variance = self.add_weight(
        name='moving_variance', shape=[self.channels], trainable=False)
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    if self.data_format == 'channels_last':
      # NHWC
      reshape_to = [1, 1, 1, -1]
    else:
      # NCHW
      reshape_to = [1, -1, 1, 1]
    gamma = tf.reshape(self.gamma, reshape_to)
    beta = tf.reshape(self.beta, reshape_to)
    moving_mean = tf.reshape(self.moving_mean, reshape_to)
    moving_variance = tf.reshape(self.moving_variance, reshape_to)
    scale = gamma * tf.math.rsqrt((moving_variance + self.eps))
    bias = beta - moving_mean * scale
    return inputs * scale + bias


def batch_norm_relu(inputs: tf.Tensor,
                    is_training_bn: bool,
                    trainable: bool,
                    params: Optional[Dict[str, Any]] = None,
                    relu: bool = True,
                    init_zero: bool = False,
                    data_format: str = 'channels_last',
                    name: Optional[str] = None,
                    enable_distributed_batch_norm: bool = False,
                    enable_frozen_batch_norm: bool = True) -> tf.Tensor:
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: Tensor of shape `[batch, ...]`.
    is_training_bn: Whether the BN layer is in training mode.
    trainable: Whether beta and gamma in BN layer are trainable.
    params: A dict including `distributed_group_size` and `num_shards`.
    relu: If False, omits the ReLU operation.
    init_zero: If True, initializes scale parameter of BN with 0. Default 1.
    data_format: `channels_last` or `channels_first`.
    name: Name of the batch normalization layer.
    enable_distributed_batch_norm: Whether to enable distributed BN. It will be
      enabled when this flag is true and params['distributed_group_size'] > 0.
    enable_frozen_batch_norm: Whether to use FrozenBatchNorm2d if both
      is_training_bn and trainable are false.

  Returns:
    A batched normalized `Tensor` with the same shape and data_format.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  if enable_frozen_batch_norm and not is_training_bn and not trainable:
    frozen_bn = FrozenBatchNorm2d(
        channels=inputs.shape[axis],
        eps=ssd_constants.BATCH_NORM_EPSILON,
        data_format=data_format)
    inputs = frozen_bn(inputs)
  elif enable_distributed_batch_norm and params is not None and params.get(
      'distributed_group_size', 0) > 0:
    if params['tpu_slice_row'] > 0 and params['tpu_slice_col'] > 0:
      physical_shape = (params['tpu_slice_row'], params['tpu_slice_col'])
    else:
      physical_shape = None

    if params['dbn_tile_row'] > 0 and params['dbn_tile_col'] > 0:
      tile_shape = (params['dbn_tile_row'], params['dbn_tile_col'])
    else:
      tile_shape = None

    input_partition_dims = FLAGS.input_partition_dims
    inputs = image_util.distributed_batch_norm(
        inputs=inputs,
        decay=ssd_constants.BATCH_NORM_DECAY,
        epsilon=ssd_constants.BATCH_NORM_EPSILON,
        is_training=is_training_bn,
        gamma_initializer=gamma_initializer,
        num_shards=params['num_shards'],
        distributed_group_size=params['distributed_group_size'],
        physical_shape=physical_shape,
        tile_shape=tile_shape,
        input_partition_dims=input_partition_dims,
        map_to_z_dim=params['map_to_z_dim'],
        logical_devices=params['logical_devices'],
        tpu_topology_dim_count=params['tpu_topology_dim_count'],
        trainable=trainable)
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=ssd_constants.BATCH_NORM_DECAY,
        epsilon=ssd_constants.BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        trainable=trainable,
        training=is_training_bn,
        fused=True,
        gamma_initializer=gamma_initializer,
        name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def space_to_depth_fixed_padding(inputs,
                                 kernel_size,
                                 data_format='channels_last',
                                 block_size=2):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    block_size: `int` block size for space-to-depth convolution.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = (pad_total // 2 + 1) // block_size
  pad_end = (pad_total // 2) // block_size
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def bottleneck_residual_block(inputs: tf.Tensor,
                              filters: int,
                              strides: int,
                              trainable: bool,
                              is_training_bn: bool,
                              bn_trainable: bool,
                              params: Dict[str, Any],
                              groups: int = 32,
                              data_format: str = 'channels_last',
                              downsample: bool = False) -> tf.Tensor:
  """Residual block with group Conv2D and BN after convolutions.

  A block consists of a 1x1 conv, a 3x3 group conv, a 1x1 conv, and a residual
  link adding the input to the output. Additionally, there's batch norm after
  the conv layers.

  Args:
    inputs: Tensor of size [batch, ...].
    filters: The first two convolutions will have filters * 2 channels. The
      final convolution will have filters * 4 channels.
    strides: Stride used in the 3x3 layer. If greater than 1, this block will
      ultimately downsample the input.
    trainable: Whether the conv layers are trainable.
    is_training_bn: Whether the BN is in training mode.
    bn_trainable: Whether BN layer weights are trainable.
    params: Parameters controlling optimizations.
    groups: Number of groups used in group conv.
    data_format: `channels_last` or `channels_first`.
    downsample: Whether shortcut needs be downsampled to match the conv output.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs

  # FIXME(zqfeng): TF's he_normal (equivalent to PyTorch's kaiming_normal)
  # doesn't have the `mode` and `nonlinearity` parameters. It's fine for MLPerf
  # submission because we will initialize the weights from a checkpoint.

  # 1x1 convolution
  if not params['use_einsum_for_projection']:
    inputs = utils.conv2d_fixed_padding(
        inputs=inputs,
        filters=2 * filters,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        kernel_initializer=tf.initializers.he_normal(),
        use_bias=False,
        trainable=trainable)
  else:
    inputs_shape = inputs.get_shape().as_list()
    kernel = tf.get_variable(
        'conv2d/kernel', [1, 1, inputs_shape[-1], 2 * filters],
        initializer=tf.initializers.he_normal(),
        trainable=trainable)
    kernel = tf.cast(kernel, inputs.dtype)
    kernel = tf.reshape(kernel, [inputs_shape[-1], 2 * filters // 128, 128])
    inputs = tf.einsum('nhwc,cgo->hwgno', inputs, kernel)
    # [H,W,G,N,C/G] -> [N,H,W,G,C/G] -> [N,H,W,C]
    inputs = tf.transpose(inputs, [3, 0, 1, 2, 4])
    inputs_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(
        inputs,
        (inputs_shape[0], inputs_shape[1], inputs_shape[2], 2 * filters))

  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      bn_trainable,
      params,
      data_format=data_format,
      relu=True)

  # 3x3 (group) convolution
  inputs = group_conv2d(
      inputs=utils.fixed_padding(inputs, 3, data_format=data_format),
      filters=2 * filters,
      kernel_size=3,
      strides=strides,
      padding='VALID',
      groups=groups,
      data_format=data_format,
      kernel_initializer=tf.initializers.he_normal(),
      trainable=trainable)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      bn_trainable,
      params,
      data_format=data_format,
      relu=True)

  # 1x1 convolution
  if not params['use_einsum_for_projection']:
    inputs = utils.conv2d_fixed_padding(
        inputs=inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        kernel_initializer=tf.initializers.he_normal(),
        use_bias=False,
        trainable=trainable)
  else:
    kernel = tf.get_variable(
        'conv2d_1/kernel', [1, 1, 2 * filters, 4 * filters],
        initializer=tf.initializers.he_normal(),
        trainable=trainable)
    kernel = tf.cast(kernel, inputs.dtype)
    kernel = tf.reshape(kernel, [2 * filters // 128, 128, 4 * filters])
    # [N,H,W,C] -> [N,H,W,G,C/G] -> [H,W,G,N,C/G]
    inputs_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, [
        inputs_shape[0], inputs_shape[1], inputs_shape[2],
        inputs_shape[3] // 128, 128
    ])
    inputs = tf.transpose(inputs, [1, 2, 3, 0, 4])
    inputs = tf.einsum('hwgnc,gco->nhwo', inputs, kernel)

  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      bn_trainable,
      params,
      data_format=data_format,
      relu=False)

  if downsample:
    with tf.variable_scope('downsample', reuse=tf.AUTO_REUSE):
      shortcut = utils.conv2d_fixed_padding(
          inputs=shortcut,
          filters=4 * filters,
          kernel_size=1,
          strides=strides,
          data_format=data_format,
          kernel_initializer=tf.initializers.he_normal(),
          use_bias=False,
          trainable=trainable)
      shortcut = batch_norm_relu(
          shortcut,
          is_training_bn,
          bn_trainable,
          params,
          data_format=data_format,
          relu=False)
  return tf.nn.relu(inputs + shortcut)


def resnext_block_group(inputs: tf.Tensor,
                        filters: int,
                        block_fn: Callable[..., tf.Tensor],
                        blocks: int,
                        strides: int,
                        trainable: bool,
                        is_training_bn: bool,
                        bn_trainable: bool,
                        name: str,
                        params: Dict[str, Any],
                        data_format: str = 'channels_last') -> tf.Tensor:
  """Creates one group of blocks for the ResNeXt model.

  Args:
    inputs: Tensor of size [batch, ... ].
    filters: Number of filters for the first convolution of the layer.
    block_fn: Function to construct a block.
    blocks: Number of blocks contained in the group.
    strides: Stride to use for the first block in the group. If greater than 1,
      it will downsample the input.
    trainable: Whether the conv layers are trainable
    is_training_bn: Whether to BN layers are in traing mode. Whether to return
      the output in training mode (normalized with statistics of the current
      batch) or in inference mode (normalized with moving statistics)
    bn_trainable: Whether the BN weigts are trainable.
    name: Name of the output tensor.
    params: Params controlling optimizations.
    data_format: `channels_first` or `channels_last`

  Returns:
    The output `Tensor` of the block group.
  """
  out = inputs
  for i in range(blocks):
    with tf.variable_scope(f'block{i}', reuse=tf.AUTO_REUSE):
      out = block_fn(
          out,
          filters,
          strides if i == 0 else 1,
          trainable=trainable,
          is_training_bn=is_training_bn,
          bn_trainable=bn_trainable,
          params=params,
          data_format=data_format,
          downsample=(i == 0))

  return tf.identity(out, name)


def resnext_generator(
    block_fn: Callable[..., tf.Tensor],
    block_group_sizes: Tuple[int, int, int, int],
    trainable_groups: int,
    first_conv_kernel_size: int = 7,
    first_conv_stride: int = 2,
    first_conv_trainable: bool = False,
    bn_trainable: bool = False,
    params: Optional[Dict[str, Any]] = None,
    data_format: str = 'channels_last') -> Callable[..., List[tf.Tensor]]:
  """Creates a generator of ResNeXt backbone with classification layers removed.

  A ResNeXt backbone consists of a first conv layer, BN, and max pooling;
  followed by 4 block groups.

  Args:
    block_fn: A function to construct a block to use within the model. Either
      `residual_block` or `bottleneck_block`.
    block_group_sizes: The number of blocks to include in each of the 4 block
      groups. Each group consists of blocks that take inputs of the same
      resolution.
    trainable_groups: How many block groups are trainable, counting backwards
      from the last group.
    first_conv_kernel_size: The kernel size of the first convolution layer in
      the network.
    first_conv_stride: The stride of the first convolution layer in the network.
    first_conv_trainable: Whether the first conv layer is trainable.
    bn_trainable: Whether BN layers are trainable.
    params: params of the model, a dict.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training_bn` and returns the
    list of output `Tensor` of all block groups.
  """
  if len(block_group_sizes) != 4:
    raise ValueError('`block_group_sizes` should have 4 int.')
  if not 0 <= trainable_groups <= len(block_group_sizes):
    raise ValueError('trainable_groups must be in [0, len(block_group_sizes)]')

  def model(inputs: tf.Tensor, is_training_bn=False) -> List[tf.Tensor]:
    block_outputs = []
    # Just use similar naming to PyTorch: conv1, layer1, layer2, layer3, ...
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
      if params['conv0_space_to_depth']:
        inputs = conv0_space_to_depth(
            inputs=inputs,
            data_format=data_format,
            trainable=first_conv_trainable)
      else:
        inputs = utils.conv2d_fixed_padding(
            inputs=inputs,
            filters=64,
            kernel_size=first_conv_kernel_size,
            strides=first_conv_stride,
            data_format=data_format,
            kernel_initializer=tf.initializers.he_normal(),
            use_bias=False,
            trainable=first_conv_trainable)
      _mllog_weight_init('module.backbone.body.conv1.weight')

      inputs = batch_norm_relu(
          inputs,
          is_training_bn,
          bn_trainable,
          params,
          data_format=data_format,
          relu=True)

      # For max pool we should actually pad -inf. But because it follows ReLU
      # immediately here, padding -1 is sufficient.
      inputs = tf.layers.max_pooling2d(
          inputs=utils.fixed_padding(inputs, 3, data_format, -1.),
          pool_size=3,
          strides=2,
          padding='VALID',
          data_format=data_format)

    for i, num_blocks in enumerate(block_group_sizes):
      with tf.variable_scope(f'layer{i+1}', reuse=tf.AUTO_REUSE):
        inputs = resnext_block_group(
            inputs=inputs,
            filters=int(64 * 2**i),
            block_fn=block_fn,
            blocks=num_blocks,
            strides=(1 if i == 0 else 2),
            trainable=(i + trainable_groups + 1 > len(block_group_sizes)),
            is_training_bn=is_training_bn,
            bn_trainable=bn_trainable,
            name=f'layer{i}_output',
            params=params,
            data_format=data_format)
        block_outputs.append(inputs)

      for k in range(num_blocks):
        _mllog_weight_init(f'module.backbone.body.layer{i+1}.{k}.conv1.weight')
        _mllog_weight_init(f'module.backbone.body.layer{i+1}.{k}.conv2.weight')
        _mllog_weight_init(f'module.backbone.body.layer{i+1}.{k}.conv3.weight')
        if k == 0:
          _mllog_weight_init(
              f'module.backbone.body.layer{i+1}.{k}.downsample.0.weight')

    return block_outputs

  return model


def resnext(resnext_depth: Any,
            trainable_groups: int = 3,
            first_conv_trainable: bool = False,
            params: Optional[Dict[str, Any]] = None,
            data_format: str = 'channels_last') -> Callable[..., tf.Tensor]:
  """Returns the ResNeXt model function for a predefined config.

  Args:
    resnext_depth: A pre-configured ResNeXt depth.
    trainable_groups: Number of block groups to be configured as trainable,
      counting backwards from the last.
    first_conv_trainable: Whether the first conv layer is trainable.
    params: Param controlling optimizations.
    data_format: `channels_{last,first}`
  """
  model_params = {
      50: {
          'block': bottleneck_residual_block,
          'layers': [3, 4, 6, 3]
      },
      101: {
          'block': bottleneck_residual_block,
          'layers': [3, 4, 23, 3]
      },
      152: {
          'block': bottleneck_residual_block,
          'layers': [3, 8, 36, 3]
      },
      200: {
          'block': bottleneck_residual_block,
          'layers': [3, 24, 36, 3]
      }
  }

  try:
    resnet_params = model_params[resnext_depth]
  except KeyError as e:
    raise ValueError(
        f'Currently supported resnext depths are {str(model_params.keys())}'
    ) from e

  return resnext_generator(
      resnet_params['block'],
      resnet_params['layers'],
      trainable_groups=trainable_groups,
      first_conv_kernel_size=7,
      first_conv_stride=2,
      first_conv_trainable=first_conv_trainable,
      params=params,
      data_format=data_format)


def conv0_space_to_depth(inputs: tf.Tensor,
                         data_format: str = 'channels_last',
                         trainable: bool = False) -> tf.Tensor:
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, height_in, width_in, channels]`.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    trainable: Whether the first conv layer is trainable.

  Returns:
    A `Tensor` with the same type as `inputs`.
  """
  # Create the conv0 kernel w.r.t. the original image size. (no space-to-depth).
  filters = 64
  kernel_size = 7
  space_to_depth_block_size = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
  strides = 2
  conv0 = tf.compat.v1.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=2,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      trainable=trainable)
  # Use the image size without space-to-depth transform as the input of conv0.
  batch_size, h, w, channel = inputs.get_shape().as_list()
  conv0.build([
      batch_size, h * space_to_depth_block_size, w * space_to_depth_block_size,
      channel // (space_to_depth_block_size**2)
  ])

  kernel = conv0.weights[0]
  # [7, 7, 3, 64] --> [8, 8, 3, 64]
  kernel = tf.pad(
      kernel,
      paddings=tf.constant([[1, 0], [1, 0], [0, 0], [0, 0]]),
      mode='CONSTANT',
      constant_values=0.)
  # Transform kernel follows the space-to-depth logic
  kernel = tf.reshape(
      kernel,
      [4, space_to_depth_block_size, 4, space_to_depth_block_size, 3, filters])
  kernel = tf.transpose(kernel, [0, 2, 1, 3, 4, 5])
  kernel = tf.reshape(kernel, [4, 4, int(channel), filters])
  kernel = tf.cast(kernel, inputs.dtype)

  inputs = space_to_depth_fixed_padding(inputs, kernel_size, data_format,
                                        space_to_depth_block_size)

  return tf.nn.conv2d(
      input=inputs,
      filter=kernel,
      strides=[1, 1, 1, 1],
      padding='VALID',
      data_format='NHWC' if data_format == 'channels_last' else 'NCHW',
      name='conv2d/Conv2D')


def retinanet(
    features: tf.Tensor,
    params: Dict[str, Any]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
  """Construct the RetinaNet object detection model.

  Args:
    features: Batch of processed images.
    params: Dict of hyper params.

  Returns:
    class_outputs: Tensor of [batch, total_num_anchors, num_classes]
    box_outputs: Tensor of [batch, total_num_anchors, 4]
  """
  with tf.variable_scope('resnext50', reuse=tf.AUTO_REUSE):
    backbone_fn = resnext(50, trainable_groups=3, params=params)
    # MLPerf 2.0 SSD requires BN be frozen under all circumstances.
    _, c3, c4, c5 = backbone_fn(features, is_training_bn=False)  # pylint: disable=unbalanced-tuple-unpacking

  with tf.variable_scope('fpn', reuse=tf.AUTO_REUSE):
    fpn_input = collections.OrderedDict([('c3', c3), ('c4', c4), ('c5', c5)])
    fpn_output: OrderedDict[Any, tf.Tensor] = ssd_fpn.retinanet_fpn(
        fpn_input, data_format='channels_last')
    assert len(fpn_output) == len(ssd_constants.FEATURE_SIZES)

  with tf.variable_scope('head', reuse=tf.AUTO_REUSE):
    detection_head_in = list(fpn_output.values())
    detection_head_out = ssd_head.retinanet_head(
        detection_head_in,
        num_anchors=ssd_constants.NUM_ANCHORS_PER_LOCATION_PER_LEVEL,
        num_classes=ssd_constants.NUM_CLASSES,
        data_format='channels_last')
    class_outputs = detection_head_out['cls_logits']
    box_outputs = detection_head_out['bbox_regression']

    assert len(class_outputs) == len(ssd_constants.FEATURE_SIZES)
    assert len(box_outputs) == len(ssd_constants.FEATURE_SIZES)
    assert sum(x.shape[1] for x in class_outputs) == ssd_constants.NUM_SSD_BOXES
    assert sum(x.shape[1] for x in box_outputs) == ssd_constants.NUM_SSD_BOXES
    assert all(x.shape[-1] == ssd_constants.NUM_CLASSES for x in class_outputs)
    assert all(x.shape[-1] == 4 for x in box_outputs)

  return class_outputs, box_outputs
