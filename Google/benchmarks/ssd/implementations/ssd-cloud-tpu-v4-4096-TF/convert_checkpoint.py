"""Convert a pickle PyTorch checkpoint to TF."""

import collections
from typing import Dict, Sequence

from absl import app
import numpy as np
import tensorflow.compat.v1 as tf


from ssd import ssd_architecture
from ssd import ssd_constants

tf.flags.DEFINE_string('pyt_pickle_file', ssd_constants.PYT_PICKLE_FILE,
                       'Path of pickle file of PyTorch checkpoint')
tf.flags.DEFINE_string('tf_checkpoint_file', ssd_constants.TF_CHECKPOINT_FILE,
                       'Prefix path of output TF checkpoint')

FLAGS = tf.flags.FLAGS


def _get_tensor_mappings(prefix='retinanet/resnext50/') -> Dict[str, str]:  # pylint: disable=missing-function-docstring

  out = collections.OrderedDict()

  def map_conv(tf_name: str, pt_name: str) -> Dict[str, str]:
    out.update({f'{prefix}{tf_name}/kernel:0': f'{pt_name}.weight'})

  def map_bn(tf_name: str, pt_name: str) -> Dict[str, str]:
    out.update({
        f'{prefix}{tf_name}/gamma:0': f'{pt_name}.weight',
        f'{prefix}{tf_name}/beta:0': f'{pt_name}.bias',
        f'{prefix}{tf_name}/moving_mean:0': f'{pt_name}.running_mean',
        f'{prefix}{tf_name}/moving_variance:0': f'{pt_name}.running_var',
    })

  # map conv1 (and bn1)
  map_conv('conv1/conv2d', 'conv1')
  map_bn('conv1/batch_normalization', 'bn1')

  # map layer 1 - 4
  for layer_id, num_blocks in zip([1, 2, 3, 4], [3, 4, 6, 3]):
    for block_id in range(num_blocks):
      # The first 1x1 conv and bn
      map_conv(f'layer{layer_id}/block{block_id}/conv2d',
               f'layer{layer_id}.{block_id}.conv1')
      map_bn(f'layer{layer_id}/block{block_id}/batch_normalization',
             f'layer{layer_id}.{block_id}.bn1')
      # The 3x3 group conv and bn
      map_conv(f'layer{layer_id}/block{block_id}/GroupConv2D',
               f'layer{layer_id}.{block_id}.conv2')
      map_bn(f'layer{layer_id}/block{block_id}/batch_normalization_1',
             f'layer{layer_id}.{block_id}.bn2')
      # The last 1x1 conv and bn
      map_conv(f'layer{layer_id}/block{block_id}/conv2d_1',
               f'layer{layer_id}.{block_id}.conv3')
      map_bn(f'layer{layer_id}/block{block_id}/batch_normalization_2',
             f'layer{layer_id}.{block_id}.bn3')
      if block_id == 0:
        # Map downsample conv and bn
        map_conv(f'layer{layer_id}/block{block_id}/downsample/conv2d',
                 f'layer{layer_id}.{block_id}.downsample.0')
        map_bn(
            f'layer{layer_id}/block{block_id}/downsample/batch_normalization',
            f'layer{layer_id}.{block_id}.downsample.1')

  return out


def main(argv: Sequence[str]):
  del argv
  with tf.io.gfile.GFile(FLAGS.pyt_pickle_file, mode='rb') as f:
    pickle_data = np.load(f, allow_pickle=True)
  mappings = _get_tensor_mappings()
  with tf.Session() as sess:
    # Dummy input for graph building
    images = tf.random.normal(
        shape=[10, 800, 800, 3],
        mean=0.5,
        stddev=0.2,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None)
    # Build the graph and create all variables.
    with tf.variable_scope('retinanet'):
      params = {
          'conv0_space_to_depth': False,
          'use_einsum_for_projection': False
      }
      _, _ = ssd_architecture.retinanet(images, params=params)
      sess.run(tf.global_variables_initializer())

    # Map and assign variables
    for v in tf.global_variables('retinanet/resnext50'):
      assert v.name in mappings
      pt_name = mappings[v.name]
      assert pt_name in pickle_data
      if 'conv2d' in v.name.lower():
        # Assume we don't use 'conv2d' in any variable scope name
        print(f'Assigning transposed {v.name} => {pt_name}')
        # PyT conv2d kernel is (out, in/group, kernel_h, kernel_w)
        # TF conv2d kernel is (kernel_h, kernel_w, in/group, out)
        tensor_data = np.transpose(pickle_data[pt_name], (2, 3, 1, 0))
      else:
        print(f'Assigning {v.name} => {pt_name}')
        tensor_data = pickle_data[pt_name]
      v = tf.assign(v, tensor_data)
      sess.run(v)
    print('Saving checkpoint')
    saver = tf.train.Saver()
    saver.save(sess, FLAGS.tf_checkpoint_file)


if __name__ == '__main__':
  app.run(main)
