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
"""Training script for SSD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import sys
import threading

from absl import app
from absl import logging
import tensorflow.compat.v1 as tf

from ssd import coco_metric
from ssd import dataloader
from ssd import ssd_constants
from ssd import ssd_model
from ssd import utils as ssd_utils
from util import train_and_eval_runner

tf.flags.DEFINE_string(
    'resnet_checkpoint', '/path/to/pretrained/backbone/ckpt',
    'Location of the ResNet checkpoint to use for model '
    'initialization.')
tf.flags.DEFINE_string('hparams', '',
                       'Comma separated k=v pairs of hyperparameters.')
tf.flags.DEFINE_integer(
    'num_shards',
    default=8,
    help='Number of shards (TPU cores) for '
    'training.')
tf.flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
tf.flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
tf.flags.DEFINE_integer('eval_samples', 24781, 'The number of samples for '
                        'evaluation.')
tf.flags.DEFINE_integer(
    'iterations_per_loop', 1000, 'Number of iterations per TPU training loop.'
    ' Evaluation is done at the end of each loop.')
tf.flags.DEFINE_string(
    'training_file_pattern', '/path/to/train-*.tfrecord',
    'Glob for training data files (e.g., OpenImage MLPerf training set)')
tf.flags.DEFINE_string(
    'validation_file_pattern', '/path/to/val-*.tfrecord',
    'Glob for evaluation tfrecords (e.g., OpenImage MLPerf validation set)')
tf.flags.DEFINE_bool(
    'use_fake_data', False,
    'Use fake data to reduce the input preprocessing overhead (for unit tests)')
tf.flags.DEFINE_string(
    'val_json_file', '/path/to/validation/file/in/coco/format/validation.json',
    'COCO validation JSON containing golden bounding boxes.')
tf.flags.DEFINE_integer('num_examples_per_epoch', 1170301,
                        'Number of examples in one epoch')
tf.flags.DEFINE_integer('num_epochs', 64, 'Number of epochs for training')
tf.flags.DEFINE_multi_integer(
    'input_partition_dims',
    default=None,
    help=('Number of partitions on each dimension of the input. Each TPU core'
          ' processes a partition of the input image in parallel using spatial'
          ' partitioning.'))
tf.flags.DEFINE_integer(
    'dataset_threadpool_size',
    default=48,
    help=('The size of the private datapool size in dataset.'))
tf.flags.DEFINE_bool('run_cocoeval', True, 'Whether to run cocoeval')
tf.flags.DEFINE_integer(
    'bfloat16_replica_threshold',
    default=128,
    help=('Threshold for enabling bfloat16 for cross replica sum.'))
tf.flags.DEFINE_bool('map_checkpoint', False,
                     'Map PyT checkpoint to TF and quit if set to True')
tf.flags.DEFINE_integer(
    'num_multiprocessing_workers',
    default=0,
    help=('The number of multiprocessing workers to use. If <=0, eval inline.'))

FLAGS = tf.flags.FLAGS
_STOP = -1


def construct_run_config(iterations_per_loop):
  """Construct the run config."""

  # Parse hparams
  hparams = ssd_model.default_hparams()
  hparams.parse(FLAGS.hparams)

  return dict(
      hparams.values(),
      num_shards=FLAGS.num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      model_dir=FLAGS.model_dir,
      iterations_per_loop=iterations_per_loop,
      steps_per_epoch=FLAGS.num_examples_per_epoch // FLAGS.train_batch_size,
      eval_samples=FLAGS.eval_samples,
      transpose_input=False if FLAGS.input_partition_dims is not None else True,
      use_spatial_partitioning=True
      if FLAGS.input_partition_dims is not None else False,
      dataset_threadpool_size=FLAGS.dataset_threadpool_size)


def predict_post_processing(q_in, q_out):
  """Run post-processing on CPU for predictions."""
  coco_gt = coco_metric.create_coco(FLAGS.val_json_file, use_cpp_extension=True)

  current_step, predictions = q_in.get()
  while current_step != _STOP and q_out is not None:
    q_out.put((current_step,
               coco_metric.compute_map(
                   predictions,
                   coco_gt,
                   use_cpp_extension=True,
                   nms_on_tpu=True)))
    current_step, predictions = q_in.get()


# converged_epoch marks the epoch convergence happens
# evals_completed is a large enough array whose entries are set to true when a
# eval finishes. Note we index the array by (ep//epochs_per_eval - 1)
converged_epoch = 0
evals_completed = [False] * 50


def main(argv):
  del argv  # Unused.

  global converged_epoch
  converged_epoch = 0

  params = construct_run_config(FLAGS.iterations_per_loop)
  params['batch_size'] = FLAGS.train_batch_size // FLAGS.num_shards
  params['bfloat16_replica_threshold'] = FLAGS.bfloat16_replica_threshold
  input_partition_dims = FLAGS.input_partition_dims
  train_steps = FLAGS.num_epochs * FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
  eval_steps = int(math.ceil(FLAGS.eval_samples / FLAGS.eval_batch_size))
  params['map_checkpoint'] = FLAGS.map_checkpoint
  runner = train_and_eval_runner.TrainAndEvalRunner(
      FLAGS.iterations_per_loop,
      train_steps,
      eval_steps,
      FLAGS.num_shards,
      num_outfeed_threads=min(64, FLAGS.num_shards // FLAGS.replicas_per_host))

  mllogger = ssd_utils.get_mllogger()

  mllogger.event('cache_clear')
  mllogger.start('init_start')
  mllogger.event('submission_org', 'Google')
  mllogger.event('submission_platform', 'tpu-v4-%d' % (FLAGS.num_shards * 2))
  mllogger.event('submission_status', 'cloud')
  mllogger.event('submission_benchmark', 'ssd')
  mllogger.event('submission_division', 'closed')
  mllogger.event('global_batch_size', FLAGS.train_batch_size)
  mllogger.event('opt_base_learning_rate', params['base_learning_rate'])
  mllogger.event('opt_weight_decay', params['weight_decay'])
  batch_size = params['batch_size'] * params['num_shards']
  mllogger.event('opt_learning_rate_warmup_factor', params['lr_warmup_factor'])
  mllogger.event('opt_learning_rate_warmup_epochs', params['lr_warmup_epoch'])
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  mllogger.event('max_samples', ssd_constants.NUM_CROP_PASSES)
  mllogger.event('train_samples', FLAGS.num_examples_per_epoch)
  mllogger.event('eval_samples', FLAGS.eval_samples)
  mllogger.event('gradient_accumulation_steps', 1)
  mllogger.event('opt_name', 'adam')

  coco_gt = None
  log_eval_result_thread = None
  processes = []

  train_input_fn = dataloader.SSDInputReader(
      FLAGS.training_file_pattern,
      params['transpose_input'],
      is_training=True,
      use_fake_data=FLAGS.use_fake_data,
      params=params)
  eval_input_fn = dataloader.SSDInputReader(
      FLAGS.validation_file_pattern,
      is_training=False,
      use_fake_data=FLAGS.use_fake_data,
      distributed_eval=True,
      count=eval_steps * FLAGS.eval_batch_size,
      params=params)

  def init_fn():
    # Adam variables may not be in checkpoint. So we just load whatever is
    # available.
    all_ckpt_keys = [
        k for k, _ in tf.train.list_variables(params['resnet_checkpoint'])
        if k.startswith('retinanet/resnext50')
    ]
    logging.info('init_from_checkpoint:\n%s', '\n'.join(all_ckpt_keys))
    tf.train.init_from_checkpoint(params['resnet_checkpoint'],
                                  dict(zip(all_ckpt_keys, all_ckpt_keys)))
    if FLAGS.run_cocoeval and FLAGS.num_multiprocessing_workers <= 0:
      nonlocal coco_gt
      if coco_gt is None:
        coco_gt = coco_metric.create_coco(
            FLAGS.val_json_file, use_cpp_extension=True)

  runner.initialize(
      train_input_fn,
      eval_input_fn,
      functools.partial(ssd_model.ssd_model_fn, params),
      FLAGS.train_batch_size,
      FLAGS.eval_batch_size,
      input_partition_dims,
      init_fn,
      mllogger=None)
  mllogger.end('init_stop')
  mllogger.start('run_start')

  if FLAGS.run_cocoeval:
    if FLAGS.num_multiprocessing_workers > 0:
      q_in = multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE)
      q_out = multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE)
      processes = [
          multiprocessing.Process(
              target=predict_post_processing, args=(q_in, q_out))
          for _ in range(FLAGS.num_multiprocessing_workers)
      ]
      for p in processes:
        p.start()

      def log_eval_results_fn():
        """Print out MLPerf log."""
        global evals_completed
        global converged_epoch
        result = q_out.get()
        success = False
        while result[0] != _STOP:
          if not success:
            steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
            epochs_per_eval = FLAGS.iterations_per_loop // steps_per_epoch
            epoch = (result[0] + FLAGS.iterations_per_loop) // steps_per_epoch
            mllogger.event(
                'eval_accuracy',
                result[1]['COCO/AP'],
                metadata={'epoch_num': epoch})
            mllogger.end('eval_stop', metadata={'epoch_num': epoch})

            # Mark this eval as completed
            evals_completed[(epoch // epochs_per_eval) - 1] = True

            if result[1]['COCO/AP'] >= ssd_constants.EVAL_TARGET:
              # Moving success setting to after the check that all evals up to
              # the converging one have finished.
              # success = True
              converged_epoch = epoch

            # Once we have converged, we check that all the evals up to that
            # epoch have completed.
            if converged_epoch > 0:
              for ep in range(epochs_per_eval, converged_epoch + 1,
                              epochs_per_eval):
                if not evals_completed[(ep // epochs_per_eval) - 1]:
                  print('Converged but have not evaluated yet: ', ep)
                  break
                if ep == converged_epoch:
                  print('Converged and evaluated all, converged at: ', ep)
                  success = True
                  mllogger.end('run_stop', metadata={'status': 'success'})

          result = q_out.get()
        if not success:
          mllogger.end('run_stop', metadata={'status': 'abort'})

      log_eval_result_thread = threading.Thread(target=log_eval_results_fn)
      log_eval_result_thread.start()

  def eval_init_fn(cur_step):
    """Executed before every eval."""
    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    epoch = cur_step // steps_per_epoch
    mllogger.start(
        'block_start',
        metadata={
            'first_epoch_num': epoch,
            'epoch_count': FLAGS.iterations_per_loop // steps_per_epoch
        })

  def eval_finish_fn(cur_step, eval_output, _):
    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    epoch = cur_step // steps_per_epoch
    mllogger.end(
        'block_stop',
        metadata={
            'first_epoch_num': epoch,
            'epoch_count': FLAGS.iterations_per_loop // steps_per_epoch
        })
    mllogger.start(
        'eval_start',
        metadata={
            'epoch_num': epoch + FLAGS.iterations_per_loop // steps_per_epoch
        })
    if FLAGS.run_cocoeval:
      # eval_output['detections'] is a list length=num_shard * num_eval_batches
      if FLAGS.num_multiprocessing_workers > 0:
        q_in.put((cur_step, eval_output['detections']))
      else:
        result = coco_metric.compute_map(
            eval_output['detections'],
            coco_gt,
            use_cpp_extension=True,
            nms_on_tpu=True)
        mllogger.event(
            'eval_accuracy',
            result['COCO/AP'],
            metadata={
                'epoch_num':
                    epoch + FLAGS.iterations_per_loop // steps_per_epoch
            })
        mllogger.end(
            'eval_stop',
            metadata={
                'epoch_num':
                    epoch + FLAGS.iterations_per_loop // steps_per_epoch
            })

        if result['COCO/AP'] >= ssd_constants.EVAL_TARGET:
          mllogger.end('run_stop', metadata={'status': 'success'})
          return True
      return False

  def run_finish_fn(success):
    if not success:
      mllogger.end('run_stop', metadata={'status': 'abort'})

  runner.train_and_eval(
      eval_init_fn, eval_finish_fn,
      None if FLAGS.num_multiprocessing_workers > 0 else run_finish_fn)

  if FLAGS.run_cocoeval:
    if FLAGS.num_multiprocessing_workers > 0:
      for _ in processes:
        q_in.put((_STOP, None))

      for p in processes:
        try:
          p.join(timeout=10)
        except Exception:  #  pylint: disable=broad-except
          pass

      q_out.put((_STOP, None))
      log_eval_result_thread.join()

      # Clear out all the queues to avoid deadlock.
      while not q_out.empty():
        q_out.get()
      while not q_in.empty():
        q_in.get()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_eager_execution()
  app.run(main)
