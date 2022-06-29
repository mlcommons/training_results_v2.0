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
"""Construct the SSD detection heads given a pyramid of multi-scale features.

Roughly equivalent to `RetinaNetHead` in
https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/retinanet.py
"""

import math
from typing import Dict, List, Optional

import tensorflow.compat.v1 as tf

from ssd import utils

_mllog_weight_init = utils.mllog_weight_init


def retinanet_head(inputs: List[tf.Tensor], num_anchors: int, num_classes: int,
                   **kwargs) -> Dict[str, List[tf.Tensor]]:
  return {
      'cls_logits':
          classification_head(inputs, num_anchors, num_classes, **kwargs),
      'bbox_regression':
          regression_head(inputs, num_anchors, **kwargs)
  }


def classification_head(
    inputs: List[tf.Tensor],
    num_anchors: int,
    num_classes: int,
    prior_probability: float = 0.01,
    num_conv_layers: int = 4,
    name: str = 'class_net',
    data_format: str = 'channels_last',
    in_channels: Optional[int] = None) -> List[tf.Tensor]:
  """Adds a classification head on top of a pyramid of multi-scale feature maps.

  There are two major differences between MLPerf 2.0 (which uses
  RetinaNet) and previous MLPerf rounds (which use canonical SSD):
    (1) In RetinaNet, all input levels have the same number of channels, so that
    we can share the conv kernel across all levels.

    (2) In RetinaNet, all levels have the same number of anchors per
    location, so that we can share the logits kernel across all levels.

  Args:
    inputs: A list of tensors from bottom (highest resolution) to top. All
      should have the same channels. This assumption is necessary in order to
      match the ground truth with prediction.
    num_anchors: The number of anchors at each location.
    num_classes: The number of classes (including background).
    prior_probability: The prior probability used to initialize address class
      imbalance. See the RetinaNet paper for more details.
    num_conv_layers: The number of conv layers used in feature transformation.
    name: Variable scope name.
    data_format: `channels_last` or `channels_first`
    in_channels: Explicitly specified input channels. If None, will infer from
      `inputs`.

  Returns:
    A list of Tensor of (N, number_of_anchors_level_i, num_classes)
  """
  if data_format == 'channels_last':
    in_channels = in_channels or inputs[0].shape[-1]
  else:
    in_channels = in_channels or inputs[0].shape[1]

  out: List[tf.Tensor] = []
  for x in inputs:
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      with tf.variable_scope('transform', reuse=tf.AUTO_REUSE):
        for i in range(num_conv_layers):
          x = tf.layers.conv2d(
              x,
              in_channels, (3, 3),
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu,
              kernel_initializer=tf.initializers.random_normal(stddev=0.01))
          _mllog_weight_init(f'module.head.classification_head.conv.{i*2}.bias')
          _mllog_weight_init(
              f'module.head.classification_head.conv.{i*2}.weight')
      with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
        cls_logits = tf.layers.conv2d(
            x,
            num_anchors * num_classes, (3, 3),
            padding='same',
            data_format=data_format,
            kernel_initializer=tf.initializers.random_normal(stddev=0.01),
            bias_initializer=tf.initializers.constant(
                -math.log((1 - prior_probability) / prior_probability),
                dtype=tf.float32))
        _mllog_weight_init('module.head.classification_head.cls_logits.bias')
        _mllog_weight_init(
            'module.head.classification_head.cls_logits.weight')
      if data_format == 'channels_first':
        # NCHW -> NHWC
        cls_logits = tf.transpose(cls_logits, (0, 2, 3, 1))
      # cls_logits is (N, H, W, A*K)
      n, h, w, _ = cls_logits.shape
      cls_logits = tf.reshape(
          cls_logits,
          (n, h * w * num_anchors, num_classes))  # cls_logits is (N, HWA, K)
      out.append(cls_logits)

  return out


def regression_head(
    inputs: List[tf.Tensor],
    num_anchors: int,
    num_conv_layers: int = 4,
    name: str = 'box_net',
    data_format: str = 'channels_last',
    in_channels: Optional[int] = None) -> List[tf.Tensor]:  # pylint:disable=g-doc-args
  """Adds a box regression head.

  For each location in each layer, it predicts num_anchor * 4 numbers. Each box
  adjustment is encoded with 4 numbers.

  Args:
    inputs: A list of tensors from bottom to top.
    num_anchors: The number of anchors at each location.
    num_conv_layers: The number of conv layers used in feature transformation.
    name: Variable scope name.
    data_format: `channels_last` or `channels_first`
    in_channels: Explicitly specified input channels. If None, will infer from
      `inputs`.

  Returns:
    A list of Tensor of (N, number_of_anchors_level_i, 4)
  """
  if data_format == 'channels_last':
    in_channels = in_channels or inputs[0].shape[-1]
  else:
    in_channels = in_channels or inputs[0].shape[1]

  out: List[tf.Tensor] = []
  for x in inputs:
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      with tf.variable_scope('transform', reuse=tf.AUTO_REUSE):
        for i in range(num_conv_layers):
          x = tf.layers.conv2d(
              x,
              in_channels, (3, 3),
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu,
              kernel_initializer=tf.initializers.random_normal(stddev=0.01))
          _mllog_weight_init(f'module.head.regression_head.conv.{i*2}.bias')
          _mllog_weight_init(f'module.head.regression_head.conv.{i*2}.weight')
      with tf.variable_scope('regression', reuse=tf.AUTO_REUSE):
        bbox_reg = tf.layers.conv2d(
            x,
            num_anchors * 4,  # each bbox encoded by 4 numbers
            (3, 3),
            padding='same',
            data_format=data_format,
            kernel_initializer=tf.initializers.random_normal(stddev=0.01))
        _mllog_weight_init('module.head.regression_head.bbox_reg.bias')
        _mllog_weight_init('module.head.regression_head.bbox_reg.weight')
      if data_format == 'channels_first':
        bbox_reg = tf.transpose(bbox_reg, (0, 2, 3, 1))
      # bbox_reg is (N, H, W, 4*A)
      n, h, w, _ = bbox_reg.shape
      bbox_reg = tf.reshape(
          bbox_reg, (n, h * w * num_anchors, 4))  # bbox_reg is (N, H*W*A, 4)
    out.append(bbox_reg)

  return out
