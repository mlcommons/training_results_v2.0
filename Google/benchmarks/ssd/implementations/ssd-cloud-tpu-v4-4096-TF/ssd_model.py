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
"""Model defination for the SSD Model.

Defines model_fn of SSD for TF Estimator. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import Any, Dict, List, Tuple

import tensorflow.compat.v1 as tf

from contrib import training as contrib_training
from contrib.tpu.python.tpu import bfloat16
from contrib.tpu.python.tpu import tpu_optimizer
from ssd import dataloader
from ssd import ssd_architecture
from ssd import ssd_constants
from ssd import utils


def batched_keep_top_k_nd(inputs: tf.Tensor, k: float) -> tf.Tensor:
  """Keep the top-k values in each multi-dimensional activations and zero others.

  Args:
    inputs: A tensor of shape [batch, d_1, d_2, ..., d_n]
    k: Top-k

  Returns:
    A tensor of the same shape of inputs. For each image in the batch, only the
      top-k values in the d_1*d_2...*d_n values are kept; others are set to 0.
  """
  batch_size = inputs.shape[0]
  inputs_flatten = tf.reshape(inputs, [batch_size, -1])
  k = min(k, inputs_flatten.shape[-1])
  topk_values, topk_indices = tf.nn.top_k(inputs_flatten, k=k, sorted=True)
  values_to_scatter = tf.reshape(topk_values, [-1])  # [batch*k]
  indices_to_scatter = tf.reshape(topk_indices, [-1])  # [batch*k]
  # Pad dimension-0 to be usable by tf.scatter_nd
  # It will be [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] where each idx repeats k times
  indices_dim_0 = []
  for idx_in_batch in range(batch_size):
    indices_dim_0.extend([idx_in_batch] * k)
  # indices_to_scatter is [batch*k, 2]
  indices_to_scatter = tf.stack([indices_dim_0, indices_to_scatter], axis=1)
  out_flatten = tf.scatter_nd(indices_to_scatter, values_to_scatter,
                              inputs_flatten.shape)
  return tf.reshape(out_flatten, inputs.shape)


def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: a Tensor with shape [batch_size, N, num_classes], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  batch_size, num_anchors, num_class = scores_in.get_shape().as_list()
  scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
  scores_trans = tf.reshape(scores_trans, [-1, num_anchors])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores_trans, k=pre_nms_num_detections, sorted=True)

  top_k_scores = tf.reshape(top_k_scores,
                            [batch_size, num_class, pre_nms_num_detections])
  top_k_indices = tf.reshape(top_k_indices,
                             [batch_size, num_class, pre_nms_num_detections])

  return tf.transpose(top_k_scores,
                      [0, 2, 1]), tf.transpose(top_k_indices, [0, 2, 1])


def _filter_scores(scores, boxes, min_score=ssd_constants.MIN_SCORE):
  mask = scores > min_score
  scores = tf.where(mask, scores, tf.zeros_like(scores))
  boxes = tf.where(
      tf.tile(tf.expand_dims(mask, 2), (1, 1, 4)), boxes, tf.zeros_like(boxes))
  return scores, boxes


def non_max_suppression(scores_in,
                        boxes_in,
                        top_k_indices,
                        labels,
                        num_detections=ssd_constants.MAX_NUM_EVAL_BOXES):
  """Implement Non-maximum suppression.

  Args:
    scores_in: a Tensor with shape [batch_size,
      ssd_constants.MAX_NUM_EVAL_BOXES, num_classes]. The top
      ssd_constants.MAX_NUM_EVAL_BOXES box scores for each class.
    boxes_in: a Tensor with shape [batch_size, N, 4], which stacks box
      regression outputs on all feature levels. The N is the number of total
      anchors on all levels.
    top_k_indices: a Tensor with shape [batch_size,
      ssd_constants.MAX_NUM_EVAL_BOXES, num_classes]. The indices for these top
      boxes for each class.
    labels: labels tensor.
    num_detections: maximum output length.

  Returns:
    A tensor size of [batch_size, num_detections, 6] represents boxes, labels
    and scores after NMS.
  """

  _, _, num_classes = scores_in.get_shape().as_list()
  source_id = tf.cast(
      tf.tile(
          tf.expand_dims(labels[ssd_constants.SOURCE_ID], 1),
          [1, num_detections]), scores_in.dtype)
  raw_shape = tf.cast(
      tf.tile(
          tf.expand_dims(labels[ssd_constants.RAW_SHAPE], 1),
          [1, num_detections, 1]), scores_in.dtype)

  list_of_all_boxes = []
  list_of_all_scores = []
  list_of_all_classes = []
  # We needn't and shouldn't to calculate for the background class
  for class_i in range(1 if ssd_constants.USE_BACKGROUND_CLASS else 0,
                       num_classes):
    boxes = tf.batch_gather(boxes_in, top_k_indices[:, :, class_i])
    class_i_scores = scores_in[:, :, class_i]
    class_i_scores, boxes = _filter_scores(class_i_scores, boxes)
    (class_i_post_scores,
     class_i_post_boxes) = ssd_architecture.non_max_suppression_padded(
         scores=tf.cast(class_i_scores, scores_in.dtype),
         boxes=tf.cast(boxes, scores_in.dtype),
         max_output_size=num_detections,
         iou_threshold=ssd_constants.OVERLAP_CRITERIA)
    class_i_classes = tf.fill(
        tf.shape(class_i_post_scores), ssd_constants.CLASS_INV_MAP[class_i])
    list_of_all_boxes.append(class_i_post_boxes)
    list_of_all_scores.append(class_i_post_scores)
    list_of_all_classes.append(class_i_classes)

  post_nms_boxes = tf.concat(list_of_all_boxes, axis=1)
  post_nms_scores = tf.concat(list_of_all_scores, axis=1)
  post_nms_classes = tf.concat(list_of_all_classes, axis=1)

  # sort all results.
  post_nms_scores, sorted_indices = tf.nn.top_k(
      tf.cast(post_nms_scores, scores_in.dtype), k=num_detections, sorted=True)

  post_nms_boxes = tf.gather(post_nms_boxes, sorted_indices, batch_dims=1)
  post_nms_classes = tf.gather(post_nms_classes, sorted_indices, batch_dims=1)
  detections_result = tf.stack([
      source_id,
      post_nms_boxes[:, :, 1] * raw_shape[:, :, 1],
      post_nms_boxes[:, :, 0] * raw_shape[:, :, 0],
      (post_nms_boxes[:, :, 3] - post_nms_boxes[:, :, 1]) * raw_shape[:, :, 1],
      (post_nms_boxes[:, :, 2] - post_nms_boxes[:, :, 0]) * raw_shape[:, :, 0],
      post_nms_scores,
      tf.cast(post_nms_classes, scores_in.dtype),
  ],
                               axis=2)

  return detections_result


@tf.custom_gradient
def _softmax_cross_entropy(logits, label):
  """Helper function to compute softmax cross entropy loss."""
  shifted_logits = logits - tf.expand_dims(tf.reduce_max(logits, -1), -1)
  exp_shifted_logits = tf.math.exp(shifted_logits)
  sum_exp = tf.reduce_sum(exp_shifted_logits, -1)
  log_sum_exp = tf.math.log(sum_exp)
  one_hot_label = tf.one_hot(label, ssd_constants.NUM_CLASSES)
  shifted_logits = tf.reduce_sum(shifted_logits * one_hot_label, -1)
  loss = log_sum_exp - shifted_logits

  def grad(dy):
    return (exp_shifted_logits / tf.expand_dims(sum_exp, -1) -
            one_hot_label) * tf.expand_dims(dy, -1), dy

  return loss, grad


def _topk_mask(scores, k):
  """Efficient implementation of topk_mask for TPUs."""

  def larger_count(data, limit):
    """Number of elements larger than limit along the most minor dimension."""
    ret = []
    for d in data:
      ret.append(
          tf.reduce_sum(
              tf.cast(d > tf.reshape(limit, [-1] + [1] * (d.shape.ndims - 1)),
                      tf.int32),
              axis=range(1, d.shape.ndims)))
    return tf.add_n(ret)

  def body(bit_index, value):
    """Body for the while loop executing the binary search."""
    new_value = tf.bitwise.bitwise_or(value,
                                      tf.bitwise.left_shift(1, bit_index))
    larger = larger_count(scores, tf.bitcast(new_value, tf.float32))
    next_value = tf.where(
        tf.logical_xor(larger >= k, kth_negative), new_value, value)
    return bit_index - 1, next_value

  kth_negative = (larger_count(scores, 0.0) < k)
  limit_sign = tf.where(kth_negative, tf.broadcast_to(1, kth_negative.shape),
                        tf.broadcast_to(0, kth_negative.shape))
  next_value = tf.bitwise.left_shift(limit_sign, 31)
  _, limit = tf.while_loop(lambda bit_index, _: bit_index >= 0, body,
                           (30, next_value))
  ret = []
  for score in scores:
    # Filter scores that are smaller than the threshold.
    ret.append(
        tf.where(
            score >= tf.reshape(
                tf.bitcast(limit, tf.float32), [-1] + [1] *
                (score.shape.ndims - 1)), tf.ones(score.shape),
            tf.zeros(score.shape)))
  return ret


def _classification_loss(pred_labels, gt_labels, num_matched_boxes):
  """Computes the classification loss.

  Computes the classification loss with hard negative mining.
  Args:
    pred_labels: a dict from index to tensor of predicted class. The shape of
      the tensor is [batch_size, num_anchors, num_classes].
    gt_labels: a list of tensor that represents the classification groundtruth
      targets. The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets. This is used as the loss normalizater.

  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  keys = sorted(pred_labels.keys())
  cross_entropy = []
  for i, k in enumerate(keys):
    gt_label = gt_labels[i]
    pred_label = tf.reshape(
        pred_labels[k],
        gt_label.get_shape().as_list() + [ssd_constants.NUM_CLASSES])
    cross_entropy.append(_softmax_cross_entropy(pred_label, gt_label))

  float_mask = [tf.cast(gt_label > 0, tf.float32) for gt_label in gt_labels]

  # Hard example mining
  neg_masked_cross_entropy = [
      ce * (1 - m) for ce, m in zip(cross_entropy, float_mask)
  ]

  num_neg_boxes = tf.minimum(
      tf.cast(num_matched_boxes, tf.int32) * ssd_constants.NEGS_PER_POSITIVE,
      ssd_constants.NUM_SSD_BOXES)
  top_k_neg_mask = _topk_mask(neg_masked_cross_entropy, num_neg_boxes)

  class_loss = tf.add_n([
      tf.reduce_sum(tf.multiply(ce, fm + tm), axis=range(1, ce.shape.ndims))
      for ce, fm, tm in zip(cross_entropy, float_mask, top_k_neg_mask)
  ])

  return tf.reduce_mean(class_loss / num_matched_boxes)


def _sigmoid_focal_loss(cls_logits_per_level: List[tf.Tensor],
                        gt_classes_per_level: List[tf.Tensor],
                        match_results_per_level: List[tf.Tensor],
                        num_matched_boxes: tf.Tensor,
                        alpha: float = 0.25,
                        gamma: float = 2) -> tf.Tensor:
  """Computes the binary Sigmoid focal loss (aka classification loss).

  Adapted from
  https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/focal_loss.py
  but we handle the whole batch here.

  Sigmoid focal loss treats each box as a multi-label (rather than
  multi-class) problem. Hence there's no need to use a background class. A
  negative box should have low scores across all classes; its label should be
  an all-zero vector.

  Args:
    cls_logits_per_level: Tensors of [batch, num_anchors_level_i, num_classes]
    gt_classes_per_level: Tensors of [batch, num_anchors_level_i]
    match_results_per_level: Tensors of [batch, num_anchors_level_i]. Used to
      selectively include boxes in focal loss.
    num_matched_boxes: Tensor of [batch]. The number of anchors that are matched
      to a groundtruth (foreground) box per image. Used to normalize per-image
      loss.
    alpha: Parameter that handles class imbalance. See the focal loss paper for
      details.
    gamma: Parameter that downweights loss of negative examples. See the focal
      loss paper for details.

  Returns:
    A scalar tensor representing the classification loss.
  """
  loss_per_level = []
  for cls_logits, gt_classes, match_results in zip(cls_logits_per_level,
                                                   gt_classes_per_level,
                                                   match_results_per_level):
    # Expand gt_classes to [batch, num_anchors, num_classes] in this way: if
    # label >= 0, convert to one-hot; else convert to all-zero
    gt_classes_padded = tf.where_v2(
        tf.greater_equal(gt_classes, 0), gt_classes, ssd_constants.NUM_CLASSES)
    gt_classes_one_hot_padded = tf.one_hot(
        gt_classes_padded, depth=ssd_constants.NUM_CLASSES + 1)
    gt_classes_expanded = tf.slice(gt_classes_one_hot_padded, [0, 0, 0],
                                   [-1, -1, ssd_constants.NUM_CLASSES])

    assert cls_logits.shape == gt_classes_expanded.shape
    assert gt_classes_expanded.shape[-1] == ssd_constants.NUM_CLASSES
    # TODO(zqfeng): We're computing sigmoid twice here. Optimize if it becomes a
    # bottleneck.
    p = tf.nn.sigmoid(cls_logits)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=gt_classes_expanded, logits=cls_logits)
    p_t = p * gt_classes_expanded + (1 - p) * (1 - gt_classes_expanded)
    loss = ce_loss * tf.pow(1 - p_t, gamma)
    alpha_t = alpha * gt_classes_expanded + (1 - alpha) * (1 -
                                                           gt_classes_expanded)
    loss = alpha_t * loss
    assert loss.shape == cls_logits.shape
    loss = tf.reduce_sum(loss, axis=-1)  # shape is [batch, num_anchors]
    # Ignore anchors in match results which have value = -2 (aka ignored)
    # TODO(zqfeng): selector and reduce_sum can be combined into a matmul.
    selector = tf.cast(
        tf.not_equal(match_results, ssd_constants.MATCH_IGNORED), tf.float32)
    assert loss.shape == selector.shape
    loss = selector * loss
    loss = tf.reduce_sum(loss, axis=-1)  # shape is [batch]
    loss_per_level.append(loss)

  # [batch]
  loss = functools.reduce(tf.add, loss_per_level)
  loss = loss / tf.maximum(1.0, num_matched_boxes)
  loss = tf.reduce_mean(loss, axis=-1)
  return loss


def _l1_localization_loss(pred_locs_per_level: List[tf.Tensor],
                          gt_locs_per_level: List[tf.Tensor],
                          gt_labels_per_level: List[tf.Tensor],
                          num_matched_boxes: tf.Tensor) -> tf.Tensor:
  """Compute the L1 localization loss (aka box regression loss).

  Matching reference:
  https://github.com/mlcommons/training/blob/a0671ab8c9668d4acf86977ad6c9d36995431197/single_stage_detector/ssd/model/retinanet.py#L194

  Args:
    pred_locs_per_level: Tensors of [batch, num_anchors_level_i, 4]
    gt_locs_per_level: Tensors of [batch, num_anchors_level_i, 4]
    gt_labels_per_level: Tensors of [batch, num_anchors_level_i]
    num_matched_boxes: A tensor of [batch]

  Returns:
    A scalar tensor representing the localization loss
  """
  loss_per_level = []
  for pred_locs, gt_locs, gt_labels in zip(pred_locs_per_level,
                                           gt_locs_per_level,
                                           gt_labels_per_level):
    assert pred_locs.shape == gt_locs.shape
    # Select only foreground boxes in localization loss
    selector = tf.cast(
        tf.greater_equal(gt_labels,
                         1 if ssd_constants.USE_BACKGROUND_CLASS else 0),
        tf.float32)
    loss = tf.reduce_sum(tf.math.abs(pred_locs - gt_locs), axis=-1)
    assert loss.shape == selector.shape  # [batch, num_anchors]
    # TODO(zqfeng) selector and reduce_sum can be combined into a matmul
    loss = loss * selector
    loss = tf.reduce_sum(loss, axis=-1)  # [batch]
    loss_per_level.append(loss)

  # [batch]
  loss = functools.reduce(tf.add, loss_per_level)
  loss = loss / tf.maximum(1.0, num_matched_boxes)
  loss = tf.reduce_mean(loss, axis=-1)
  return loss


# TODO(zqfeng): move loss functions to a separate file
def detection_loss(cls_outputs_per_level: List[tf.Tensor],
                   box_outputs_per_level: List[tf.Tensor],
                   labels: List[tf.Tensor]) -> tf.Tensor:
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs_per_level: List of tensor of [batch, number_anchors_level_i,
      num_classes] for each FPN level.
    box_outputs_per_level: List of tensor of [batch , number_anchors_level_i, 4]
      for each FPN level.
    labels: The dictionary that returned from dataloader that includes
      groundtruth targets.

  Returns:
    total_loss: a float32 representing total loss reducing from class and box
      losses from all levels.
  """
  gt_boxes_per_level = labels[ssd_constants.BOXES]
  gt_classes_per_level = labels[ssd_constants.CLASSES]
  match_results_per_level = labels[ssd_constants.MATCH_RESULT]
  assert isinstance(gt_boxes_per_level, (list, tuple))
  assert isinstance(gt_classes_per_level, (list, tuple))
  assert isinstance(match_results_per_level, (list, tuple))

  num_matched_boxes = tf.reshape(labels[ssd_constants.NUM_MATCHED_BOXES], [-1])
  box_loss = _l1_localization_loss(box_outputs_per_level, gt_boxes_per_level,
                                   gt_classes_per_level, num_matched_boxes)
  class_loss = _sigmoid_focal_loss(cls_outputs_per_level, gt_classes_per_level,
                                   match_results_per_level, num_matched_boxes)

  box_loss = utils.add_print_op(box_loss, 'box_loss', 1, force=True)
  class_loss = utils.add_print_op(class_loss, 'class_loss', 1, force=True)

  return class_loss + box_loss


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  Args:
    params: a parameter dictionary that includes learning_rate, lr_warmup_epoch.
  """
  batch_size = params['batch_size'] * params['num_shards']
  # Learning rate is proportional to the batch size
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)


def learning_rate_schedule(params, global_step):
  """Handles learning rate scaling, linear warmup, and learning rate decay.

  Args:
    params: A dictionary that defines hyperparameters of model.
    global_step: A tensor representing current global step.

  Returns:
    A tensor representing current learning rate.
  """
  base_learning_rate = params['base_learning_rate']
  lr_warmup_factor = params['lr_warmup_factor']
  lr_warmup_step = params['lr_warmup_step']
  # (zqfeng) Disabled scaling lr for vanilla baseline
  # batch_size = params['batch_size'] * params['num_shards']
  # scaling_factor = batch_size / ssd_constants.DEFAULT_BATCH_SIZE
  scaling_factor = 1.0
  adjusted_learning_rate = base_learning_rate * scaling_factor
  # between 0 and lr_warmup_step, do linear interpolate between
  # lr_warmup_factor * adjusted_learning_rate and adjusted_learning_rate
  starting_lr = adjusted_learning_rate * lr_warmup_factor
  warmup_learning_rate = starting_lr + (
      adjusted_learning_rate - starting_lr) * tf.cast(
          global_step, dtype=tf.float32) / lr_warmup_step
  learning_rate = tf.where(global_step < lr_warmup_step, warmup_learning_rate,
                           adjusted_learning_rate)
  learning_rate = utils.add_print_op(learning_rate, 'learning_rate', 1)
  return learning_rate


class WeightDecayOptimizer(tf.train.Optimizer):
  """Wrapper to apply weight decay on gradients before all reduce."""

  def __init__(self, opt, weight_decay=0.0, name='WeightDecayOptimizer'):
    super(WeightDecayOptimizer, self).__init__(False, name)
    self._opt = opt
    self._weight_decay = weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None:
        summed_grads_and_vars.append((grad, var))
      else:
        summed_grads_and_vars.append((grad + var * self._weight_decay, var))
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)


def ssd_model_fn(params: Dict[str, Any], features: tf.Tensor, labels: tf.Tensor,
                 is_training: bool) -> Tuple[Any, Any]:
  """Model defination for the SSD model based on ResNet-50.

  Args:
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    is_training: whether this is for training. If yes, will add loss function
      and optimizer. Otherwise will add post-processing of detections.

  Returns:
    [train_op, None] or [None, predictions]
  """

  # Manually apply the double transpose trick for training data.
  if params['transpose_input'] and is_training:
    if params['batch_size'] > 8:
      features = tf.transpose(features, [3, 0, 1, 2])
    else:
      features = tf.transpose(features, [0, 3, 1, 2])
    labels[ssd_constants.BOXES] = tuple(
        tf.transpose(box_per_level, [0, 2, 1])
        for box_per_level in labels[ssd_constants.BOXES])
    labels[ssd_constants.CLASSES] = tuple(
        tf.transpose(class_per_level)
        for class_per_level in labels[ssd_constants.CLASSES])
    labels[ssd_constants.MATCH_RESULT] = tuple(
        tf.transpose(match_per_level)
        for match_per_level in labels[ssd_constants.MATCH_RESULT])

  if params['use_bfloat16']:
    with bfloat16.bfloat16_scope():
      with tf.variable_scope('retinanet', reuse=tf.AUTO_REUSE):
        cls_outputs_per_level, box_outputs_per_level = ssd_architecture.retinanet(
            features, params)
  else:
    with tf.variable_scope('retinanet', reuse=tf.AUTO_REUSE):
      cls_outputs_per_level, box_outputs_per_level = ssd_architecture.retinanet(
          features, params)

  # Note: output from retinanet is always split by FPN level regardless of
  # ssd_constants.SPLIT_LEVEL

  # First check if it is in PREDICT mode.
  if not is_training:
    if params['use_bfloat16']:
      cls_outputs_per_level = [
          tf.cast(x, tf.float32) for x in cls_outputs_per_level
      ]
      box_outputs_per_level = [
          tf.cast(x, tf.float32) for x in box_outputs_per_level
      ]
    all_anchors = tf.convert_to_tensor(dataloader.DefaultBoxes()('xywh'))
    all_box_outputs = tf.concat(box_outputs_per_level, axis=1)

    all_box_outputs /= tf.reshape(
        tf.convert_to_tensor(ssd_constants.BOX_CODER_SCALES), [1, 1, 4])
    ycenter_a, xcenter_a, ha, wa = tf.unstack(all_anchors, axis=-1)
    ty, tx, th, tw = tf.unstack(all_box_outputs, axis=-1)
    # Avoid sending too large values into tf.exp()
    tw = tf.minimum(tw, ssd_constants.BOX_DECODE_HW_CLAMP)
    th = tf.minimum(th, ssd_constants.BOX_DECODE_HW_CLAMP)

    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    decoded_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    # Clip coordinates inside image
    decoded_boxes = tf.clip_by_value(decoded_boxes, 0., 1.)

    # [batch, num_anchors_level_i, num_classes]
    pred_scores_per_level = [tf.nn.sigmoid(x) for x in cls_outputs_per_level]

    top_k_scores_and_indices_per_level = []
    num_anchors_per_level = [
        fs**2 * ssd_constants.NUM_ANCHORS_PER_LOCATION_PER_LEVEL
        for fs in ssd_constants.FEATURE_SIZES
    ]

    # From now on until NMS, we output num_classes at dim-1 to reduce transpose
    for i, scores in enumerate(pred_scores_per_level):
      batch, num_anchors = scores.shape[:2]
      assert num_anchors == num_anchors_per_level[i]
      # minimal global box idx of this level
      box_idx_base = sum(num_anchors_per_level[:i])
      scores_flatten = tf.reshape(scores, [batch, -1])
      k = ssd_constants.CANDIDATE_LOGITS_PER_LEVEL
      # Outputs [batch, k]
      topk_values, topk_indices = tf.nn.top_k(scores_flatten, k=k, sorted=True)
      topk_cls_ids = tf.math.floormod(topk_indices, ssd_constants.NUM_CLASSES)
      topk_box_ids = tf.math.floordiv(topk_indices, ssd_constants.NUM_CLASSES)
      # Outputs [batch, num_classes, k]
      one_hot_mask = tf.one_hot(topk_cls_ids, ssd_constants.NUM_CLASSES, axis=1)
      pre_nms_scores = tf.multiply(one_hot_mask,
                                   tf.expand_dims(topk_values, axis=1))
      pre_nms_indices = tf.multiply(
          tf.cast(one_hot_mask, topk_box_ids.dtype),
          tf.expand_dims(topk_box_ids + box_idx_base, axis=1))

      top_k_scores_and_indices_per_level.append(
          (pre_nms_scores, pre_nms_indices))

    # Each piece is a sparse tensor of size [num_classes, 1000] with only 1000
    # valid elements. After concat, both are [batch, num_classes,
    # num_boxes=5000]
    pred_scores = tf.concat([t[0] for t in top_k_scores_and_indices_per_level],
                            axis=-1)
    box_indices = tf.concat([t[1] for t in top_k_scores_and_indices_per_level],
                            axis=-1)

    # Global sort pred_scores and indices across FPN levels
    # This is a full sort, but because we need both the values and the sort
    # order (which is needed to gather the box_indices), we use top_k.
    topk_scores, topk_indices = tf.nn.top_k(
        pred_scores, k=pred_scores.shape[-1], sorted=True)
    sorted_scores = topk_scores
    sorted_indices = tf.gather(box_indices, topk_indices, batch_dims=2)

    if ssd_constants.IS_PADDED in labels:
      # For TPU nms
      is_padded = tf.reshape(labels[ssd_constants.IS_PADDED], [-1, 1, 1])
      sorted_scores = tf.where_v2(is_padded, 0., sorted_scores)

    # TODO(zqfeng): nms expects num_classes to be the last dimension. we can
    # skip these transpose if we change the implementation of nms
    sorted_scores = tf.transpose(sorted_scores, [0, 2, 1])
    sorted_indices = tf.transpose(sorted_indices, [0, 2, 1])
    detections = non_max_suppression(
        scores_in=sorted_scores,
        boxes_in=decoded_boxes,
        top_k_indices=sorted_indices,
        labels=labels,
        num_detections=ssd_constants.MAX_NUM_EVAL_BOXES)

    predictions = dict(detections=detections)

    if ssd_constants.IS_PADDED in labels:
      # For CPU nms in coco_eval
      predictions[ssd_constants.IS_PADDED] = labels[ssd_constants.IS_PADDED]

    return None, predictions

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)

  if not ssd_constants.SPLIT_LEVEL:
    cls_outputs_per_level = [tf.concat(cls_outputs_per_level, axis=1)]
    box_outputs_per_level = [tf.concat(box_outputs_per_level, axis=1)]
  if params['use_bfloat16']:
    cls_outputs_per_level = [
        tf.cast(x, tf.float32) for x in cls_outputs_per_level
    ]
    box_outputs_per_level = [
        tf.cast(x, tf.float32) for x in box_outputs_per_level
    ]

  total_loss = detection_loss(cls_outputs_per_level, box_outputs_per_level,
                              labels)

  use_bf16_allreduce = params['num_shards'] <= params[
      'bfloat16_replica_threshold']
  optimizer = tf.train.AdamOptimizer(learning_rate)
  optimizer = optimizer if use_bf16_allreduce else tpu_optimizer.CrossShardOptimizer(
      optimizer)

  # Batch norm requires update_ops to be added as a train_op dependency.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    if use_bf16_allreduce:
      scaled_loss = total_loss / params['num_shards']
      tvars = tf.trainable_variables()
      grads = tf.gradients(scaled_loss, tvars)
      grads_tvars = zip(grads, tvars)
      grads_tvars = [(tf.cast(
          tf.tpu.cross_replica_sum(tf.cast(g, tf.bfloat16)), tf.float32), v)
                     for g, v in grads_tvars]
      train_op = optimizer.apply_gradients(grads_tvars, global_step=global_step)
      return train_op, None
    else:
      return optimizer.minimize(total_loss, global_step), None


def default_hparams():
  return contrib_training.HParams(
      use_bfloat16=True,
      num_examples_per_epoch=120000,
      weight_decay=ssd_constants.WEIGHT_DECAY,
      base_learning_rate=ssd_constants.BASE_LEARNING_RATE,
      lr_warmup_epoch=ssd_constants.LEARNING_RATE_WARMUP_EPOCHS,
      lr_warmup_factor=ssd_constants.LEARNING_RATE_WARMUP_FACTOR,
      distributed_group_size=1,
      tpu_slice_row=-1,
      tpu_slice_col=-1,
      dbn_tile_row=-1,  # number of rows in each distributed batch norm group.
      dbn_tile_col=-1,  # number of cols in each distributed batch norm group.
      eval_every_checkpoint=False,
      transpose_input=False,
      conv0_space_to_depth=True,
      use_einsum_for_projection=False,
      eval_samples=ssd_constants.EVAL_SAMPLES,
      use_spatial_partitioning=False,
      map_to_z_dim=False,
      logical_devices=2,
      tpu_topology_dim_count=2,
  )
