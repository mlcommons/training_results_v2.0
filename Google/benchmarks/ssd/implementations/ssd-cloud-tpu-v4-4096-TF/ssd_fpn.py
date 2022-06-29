# Copyright 2022 Google. All Rights Reserved.
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
"""Construct the FPN (feature pyramid network) on top of several output layers from backbone.

Roughly equivalent to
https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/feature_pyramid_network.py

Illustration:
                                          P7
                                          ^
                                          |(relu, 3x3, stride=2)
                                          P6
                                          ^
                                          |(3x3, stride=2)
C5 -(1x1)-> C5' ------------> I5 -(3x3)-> P5
                              |(upsample, +)
                              v
C4 -(1x1)-> C4' ----(+)-----> I4 -(3x3)-> P4
                              |(upsample, +)
                              v
C3 -(1x1)-> C3' ----(+)-----> I3 -(3x3)-> P3
"""
import collections
from typing import Any, Callable, List, NewType, Optional, OrderedDict, Tuple

import tensorflow.compat.v1 as tf
from ssd import utils

ExtraBlockFn = NewType(
    'ExtraBlockFn',
    Callable[[List[tf.Tensor], List[tf.Tensor], List[str], int, str],
             Tuple[List[tf.Tensor], List[str]]])

_mllog_weight_init = utils.mllog_weight_init


def fpn(
    in_feature_maps: OrderedDict[Any, tf.Tensor],
    out_channels: int,
    data_format: str = 'channels_last',
    extra_blocks: Optional[ExtraBlockFn] = None) -> OrderedDict[Any, tf.Tensor]:
  """Adds an FPN on top of F = len(in_feature_maps) input feature maps.

  Outputs F feature maps, each of which has the same spatial dimension as its
  corresponding input. All output layers have `out_channels` channels.

  Each output feauture map is computed with both a lateral link and a
  top-down link, except for the top-level (smallest size, most
  coarse-grained). The top-level only has a lateral link.

  Args:
    in_feature_maps: Order should be from bottom (largest size, finest-grained)
      to top (smallest size, most coarse-grained). Each feature map has shape
      [N, H_i, W_i, C_i].
    out_channels: Number of channels in all output feature maps.
    data_format: Only `channels_last` is supported at the moment.
    extra_blocks: A function that adds extra blocks to the output that are
      generated in different ways.

  Returns:
    Transformed feature maps with the same keys and order.
  """
  if not in_feature_maps:
    raise ValueError('in_feature_maps must be a non-empty OrderedDict.')
  if not out_channels > 0:
    raise ValueError('out_channels must be positive.')
  if data_format != 'channels_last':
    raise ValueError('`tf.image.resize_nearest_neighbor` currently only'
                     ' supports channels_last format.')

  def inner_block(x: tf.Tensor) -> tf.Tensor:
    """Conv 1x1 from various in_channels to out_channels."""
    return tf.layers.conv2d(
        x,
        filters=out_channels,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=tf.initializers.he_uniform(),
        bias_initializer=tf.zeros_initializer(),
        data_format=data_format)

  def layer_block(x: tf.Tensor) -> tf.Tensor:
    """Conv 3x3 from out_channels to out_channels."""
    assert x.shape[-1] == out_channels
    return tf.layers.conv2d(
        x,
        filters=out_channels,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=tf.initializers.he_uniform(),
        bias_initializer=tf.zeros_initializer(),
        data_format=data_format)

  keys = list(in_feature_maps.keys())
  in_features = list(in_feature_maps.values())
  out_top_down = []
  last_inner_block = None
  for i, (key, x) in enumerate(reversed(in_feature_maps.items())):
    with tf.variable_scope(key, reuse=tf.AUTO_REUSE):
      if i == 0:
        # top-level only has lateral link
        last_inner_block = inner_block(x)
      else:
        b, h, w, c = last_inner_block.get_shape().as_list()
        assert x.shape[1] == 2 * h
        assert x.shape[2] == 2 * w
        top_down = tf.reshape(
            tf.tile(
                tf.reshape(last_inner_block, [b, h, 1, w, 1, c]),
                [1, 1, 2, 1, 2, 1]), [b, x.shape[1], x.shape[2], c])

        last_inner_block = inner_block(x) + top_down
      out = layer_block(last_inner_block)
      out_top_down.append(out)
      _mllog_weight_init(f'module.backbone.fpn.inner_blocks.{i}.bias')
      _mllog_weight_init(f'module.backbone.fpn.inner_blocks.{i}.weight')
      _mllog_weight_init(f'module.backbone.fpn.layer_blocks.{i}.bias')
      _mllog_weight_init(f'module.backbone.fpn.layer_blocks.{i}.weight')

  out_features = list(reversed(out_top_down))

  if extra_blocks:
    out_features, keys = extra_blocks(out_features, in_features, keys,
                                      out_channels, data_format)

  return collections.OrderedDict(zip(keys, out_features))


def retinatnet_last_level_p6_p7(
    fpn_out: List[tf.Tensor],
    fpn_in: List[tf.Tensor],
    names: List[str],
    out_channels: int,
    data_format: str = 'channels_last',
    use_p5: bool = True) -> Tuple[List[tf.Tensor], List[str]]:
  """Adds the extra P6 and P7 layers in RetinaNet.

  Refer to footnote 2 on page 4 of the RetinaNet paper for details.

  Args:
    fpn_out: Output layers from the FPN (called P in paper).
    fpn_in: Input layers to the FPN (called C in paper).
    names: Layer names in the same order.
    out_channels: Output channels. Expected to be the same as P layers.
    data_format: `channels_last` or `channels_first`.
    use_p5: Whether to use P5 instead of C5 to compute P6. The original
      RetinaNet paper uses C5, but MLPerf 2.0 uses P5.

  Returns:
    Modified list of layers and names.
  """
  c5: tf.Tensor = fpn_in[-1]
  p5: tf.Tensor = fpn_out[-1]
  x = p5 if use_p5 else c5
  with tf.variable_scope('p6', reuse=tf.AUTO_REUSE):
    p6 = utils.conv2d_fixed_padding(
        x,
        filters=out_channels,
        kernel_size=3,
        strides=2,
        data_format=data_format,
        kernel_initializer=tf.initializers.he_uniform(),
        bias_initializer=tf.zeros_initializer()
    )  # note: output P6 from FPN is before ReLU
  with tf.variable_scope('p7', reuse=tf.AUTO_REUSE):
    p7 = utils.conv2d_fixed_padding(
        tf.nn.relu(p6),
        filters=out_channels,
        kernel_size=3,
        strides=2,
        data_format=data_format,
        kernel_initializer=tf.initializers.he_uniform(),
        bias_initializer=tf.zeros_initializer())

  _mllog_weight_init('module.backbone.fpn.extra_blocks.p6.bias')
  _mllog_weight_init('module.backbone.fpn.extra_blocks.p6.weight')
  _mllog_weight_init('module.backbone.fpn.extra_blocks.p7.bias')
  _mllog_weight_init('module.backbone.fpn.extra_blocks.p7.weight')

  return fpn_out + [p6, p7], names + ['p6', 'p7']


def retinanet_fpn(in_feature_maps: OrderedDict[str, tf.Tensor],
                  data_format: str = 'channels_last') -> tf.Tensor:
  return fpn(
      in_feature_maps,
      out_channels=256,
      data_format=data_format,
      extra_blocks=retinatnet_last_level_p6_p7)
