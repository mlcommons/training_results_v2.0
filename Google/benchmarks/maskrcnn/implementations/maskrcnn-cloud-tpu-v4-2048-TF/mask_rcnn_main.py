#) Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Training script for Mask-RCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from postprocess.python import postprocess
import functools
import math
import multiprocessing
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import control_flow_util

from mask_rcnn import coco_metric
from mask_rcnn import dataloader
from mask_rcnn import eval_multiprocess
from mask_rcnn import mask_rcnn_model
from mask_rcnn import mask_rcnn_params
from util import mllog
from util import train_and_eval_runner


flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_shards',
    default=8,
    help='Number of shards (TPU cores) for '
    'training.')
flags.DEFINE_integer(
    'num_all_tpu_cores',
    default=8,
    help='Number of all tpu cores for training.')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
tf.flags.DEFINE_integer('train_batch_size', 128, 'training batch size')
tf.flags.DEFINE_integer('eval_batch_size', 128, 'evaluation batch size')
tf.flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                        'evaluation.')
flags.DEFINE_string('resnet_checkpoint', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
tf.flags.DEFINE_integer('num_examples_per_epoch', 117266,
                        'Number of examples in one epoch')
tf.flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')

FLAGS = flags.FLAGS
_STOP = -1
ap_score = 0
mask_ap_score = 0
cur_epoch = 0


def run_mask_rcnn(hparams):
  """Runs the Mask RCNN train and eval loop."""

  global ap_score
  global mask_ap_score
  global cur_epoch

  mllogger = mllog.MLLogger()
  ap_score = 0
  mask_ap_score = 0
  cur_epoch = 0

  params = dict(
      hparams.values(),
      transpose_input=False if FLAGS.input_partition_dims is not None else True,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      num_cores_per_replica=int(np.prod(FLAGS.input_partition_dims))
      if FLAGS.input_partition_dims else 1,
      replicas_per_host=FLAGS.replicas_per_host)

  mllogger.event('cache_clear')
  mllogger.start('init_start')
  mllogger.event('submission_org', 'Google')

  mllogger.event('submission_platform', 'tpu-v4-%d' % (FLAGS.num_all_tpu_cores))
  mllogger.event('submission_status', 'cloud')
  mllogger.event('submission_benchmark', 'maskrcnn')
  mllogger.event('submission_division', 'closed')
  mllogger.event('global_batch_size', FLAGS.train_batch_size)
  mllogger.event('train_samples', FLAGS.num_examples_per_epoch)
  mllogger.event('eval_samples', FLAGS.eval_samples)
  mllogger.event('min_image_size', params['short_side_image_size'])
  mllogger.event('max_image_size', params['long_side_max_image_size'])
  mllogger.event('num_image_candidates', params['rpn_post_nms_topn'])
  mllogger.event('opt_base_learning_rate', params['learning_rate'])
  mllogger.event('opt_learning_rate_warmup_steps', params['lr_warmup_step'])
  mllogger.event('opt_learning_rate_warmup_factor',
                 params['learning_rate'] / params['lr_warmup_step'])
  mllogger.event('opt_learning_rate_decay_factor', value=0.1)
  mllogger.event('opt_learning_rate_decay_steps',
                 [params['first_lr_drop_step'], params['second_lr_drop_step']])
  mllogger.event('gradient_accumulation_steps', 1)
  seed = int(time.time())
  tf.set_random_seed(seed)
  mllogger.event('seed', seed)

  train_steps = (
      FLAGS.num_epochs * FLAGS.num_examples_per_epoch // FLAGS.train_batch_size)
  eval_steps = int(math.ceil(float(FLAGS.eval_samples) / FLAGS.eval_batch_size))
  if eval_steps > 0:
    # The eval dataset is not evenly divided. Adding step by one will make sure
    # all eval samples are covered.
    # TODO: regenerate the eval dataset to make all hosts get the
    #                    same amount of work.
    eval_steps += 1
  num_hosts = FLAGS.num_shards // FLAGS.replicas_per_host
  runner = train_and_eval_runner.TrainAndEvalRunner(
      FLAGS.num_examples_per_epoch // FLAGS.train_batch_size, train_steps,
      eval_steps, FLAGS.num_shards,
      num_outfeed_threads=min(4, num_hosts))
  train_input_fn = dataloader.InputReader(
      FLAGS.training_file_pattern,
      mode=tf.estimator.ModeKeys.TRAIN,
      use_fake_data=FLAGS.use_fake_data)
  eval_input_fn = functools.partial(
      dataloader.InputReader(
          FLAGS.validation_file_pattern,
          mode=tf.estimator.ModeKeys.PREDICT,
          distributed_eval=True),
      num_examples=eval_steps * FLAGS.eval_batch_size)
  eval_metric = coco_metric.EvaluationMetric(
      FLAGS.val_json_file, use_cpp_extension=True)

  def init_fn():
    if FLAGS.resnet_checkpoint:
      tf.train.init_from_checkpoint(FLAGS.resnet_checkpoint,
                                    {'resnet/': 'resnet50/'})

  runner.initialize(train_input_fn, eval_input_fn,
                    mask_rcnn_model.MaskRcnnModelFn(params, mllogger),
                    FLAGS.train_batch_size, FLAGS.eval_batch_size,
                    FLAGS.input_partition_dims, init_fn, params=params)
  mllogger.end('init_stop')
  mllogger.start('run_start')

  def eval_init_fn(cur_step):
    """Executed before every eval."""
    global cur_epoch
    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    cur_epoch = 0 if steps_per_epoch == 0 else cur_step // steps_per_epoch
    mllogger.start(
        'block_start',
        None,
        metadata={
            'first_epoch_num': cur_epoch,
            'epoch_count': 1
        })

  def eval_finish_fn(cur_step, eval_output, _):
    """Callback function that's executed after each eval."""
    global ap_score
    global mask_ap_score
    global cur_epoch

    if eval_steps == 0:
      return False

    # Concat eval_output as eval_output is a list from each host.
    predictions = dict(
        detections=np.concatenate(eval_output['detections'], axis=0),
        image_info=np.concatenate(
            eval_output['image_info'], axis=0).astype(np.int32),
        mask_outputs=np.concatenate(
            eval_output['mask_outputs'], axis=0))

    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    cur_epoch = 0 if steps_per_epoch == 0 else cur_step // steps_per_epoch
    mllogger.end(
        'block_stop',
        None,
        metadata={
            'first_epoch_num': cur_epoch,
            'epoch_count': 1})

    if mask_rcnn_params.MULTITHREADED_EVAL:
      eval_multithreaded = postprocess.EvalPostprocessor()
      detections, segmentations = eval_multithreaded.PostProcess(
          predictions['detections'].astype(np.float32),
          predictions['mask_outputs'].astype(np.float32),
          predictions['image_info'].astype(np.int32),
          mask_rcnn_params.EVAL_WORKER_COUNT)
      eval_metric.update(detections, [segmentations])
    else:
      eval_multiprocess.eval_multiprocessing(predictions, eval_metric,
                                             mask_rcnn_params.EVAL_WORKER_COUNT)

    mllogger.start('eval_start', metadata={'epoch_num': cur_epoch + 1})
    _, eval_results = eval_metric.evaluate()

    ap_score = eval_results['AP']
    mask_ap_score = eval_results['mask_AP']

    mllogger.event(
        'eval_accuracy', {
            'BBOX': float(eval_results['AP']),
            'SEGM': float(eval_results['mask_AP'])
        },
        metadata={'epoch_num': cur_epoch + 1})
    mllogger.end('eval_stop', metadata={'epoch_num': cur_epoch + 1})
    if (eval_results['AP'] >= mask_rcnn_params.BOX_EVAL_TARGET and
        eval_results['mask_AP'] >= mask_rcnn_params.MASK_EVAL_TARGET):
      mllogger.end('run_stop', metadata={'status': 'success'})
      return True
    return False

  def run_finish_fn(success):
    if not success:
      mllogger.end('run_stop', metadata={'status': 'abort'})

  runner.train_and_eval(eval_init_fn, eval_finish_fn, run_finish_fn)
  return cur_epoch, ap_score, mask_ap_score


def main(argv):
  del argv  # Unused.

  # TODO: remove this workaround that uses control flow v2.
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

  # Parse hparams
  hparams = mask_rcnn_params.default_hparams()
  hparams.parse(FLAGS.hparams)
  run_mask_rcnn(hparams)


if __name__ == '__main__':
  tf.disable_eager_execution()
  logging.set_verbosity(logging.INFO)
  app.run(main)
