# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddle.fluid.layers.collective import _c_broadcast as broadcast
from custom_setup_ops import sort_bert_inputs_across_devices


def run_test_case(batch_size, max_batch_size=None):
    if max_batch_size is None:
        max_batch_size = batch_size

    max_seq_len = 512
    vocab_size = 30528

    dtype = 'int64'
    ring_id = 0

    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()
    device_id = rank
    num_devices = world_size

    global_batch_size = batch_size * num_devices
    seq_lens = np.random.randint(
        low=100, high=max_seq_len + 1, size=[num_devices * batch_size])

    input_ids = []
    segment_ids = []
    input_mask = []
    masked_lm_labels = []
    next_sentence_labels = []

    for seq_len in seq_lens:
        input_id = np.random.randint(low=1, high=vocab_size, size=[max_seq_len])
        input_id[seq_len:] = 0
        input_ids.append(input_id)

        sentence_1_len = np.random.randint(
            low=10, high=seq_len - 10, size=[1])[0]
        segment_id = np.zeros([max_seq_len])
        segment_id[sentence_1_len:seq_len] = 1
        segment_ids.append(segment_id)

        mask = np.ones([max_seq_len])
        mask[seq_len:] = 0
        input_mask.append(mask)

        masked_lm_label = np.random.randint(
            low=0, high=vocab_size, size=[max_seq_len])
        masked_lm_labels.append(masked_lm_label)

        next_sentence_label = np.random.randint(low=0, high=2, size=[1])[0]
        next_sentence_labels.append(next_sentence_label)

    shape = [num_devices, batch_size, max_seq_len]

    input_ids = np.reshape(np.array(input_ids, dtype=dtype), shape)
    segment_ids = np.reshape(np.array(segment_ids, dtype=dtype), shape)
    input_mask = np.reshape(np.array(input_mask, dtype=dtype), shape)
    masked_lm_labels = np.reshape(
        np.array(
            masked_lm_labels, dtype=dtype), shape)
    next_sentence_labels = np.reshape(
        np.array(
            next_sentence_labels, dtype=dtype), [num_devices, batch_size])

    seq_lens = np.reshape(np.sum(input_mask, axis=2), [-1])
    sorted_indices = np.argsort(-seq_lens, kind='mergesort')
    sorted_indices = np.reshape(sorted_indices,
                                [batch_size, num_devices])[:, rank]
    unsorted_indices = np.array(
        range(rank * batch_size, (rank + 1) * batch_size),
        dtype=sorted_indices.dtype)

    def reorder_by_indices(indices):
        shape = [num_devices * batch_size, max_seq_len]
        input_ids_ret = np.reshape(input_ids, shape)[indices, :]
        segment_ids_ret = np.reshape(segment_ids, shape)[indices, :]
        input_mask_ret = np.reshape(input_mask, shape)[indices, :]
        masked_lm_labels_ret = np.reshape(masked_lm_labels, shape)[indices, :]
        next_sentence_labels_ret = np.reshape(next_sentence_labels,
                                              shape[:1])[indices]
        return input_ids_ret, segment_ids_ret, input_mask_ret, masked_lm_labels_ret, next_sentence_labels_ret

    input_ids_sorted, segment_ids_sorted, input_mask_sorted, masked_lm_labels_sorted, next_sentence_labels_sorted = reorder_by_indices(
        sorted_indices)
    input_ids_unsorted, segment_ids_unsorted, input_mask_unsorted, masked_lm_labels_unsorted, next_sentence_labels_unsorted = reorder_by_indices(
        unsorted_indices)

    input_ids_sorted_actual, segment_ids_sorted_actual, input_mask_sorted_actual, masked_lm_labels_sorted_actual, next_sentence_labels_sorted_actual = sort_bert_inputs_across_devices(
        paddle.to_tensor(input_ids_unsorted),
        paddle.to_tensor(segment_ids_unsorted),
        paddle.to_tensor(input_mask_unsorted),
        paddle.to_tensor(masked_lm_labels_unsorted),
        paddle.to_tensor(next_sentence_labels_unsorted), max_batch_size,
        ring_id, rank, world_size)

    def assert_equal(x, y):
        assert np.array_equal(np.array(x), np.array(y))

    assert_equal(input_ids_sorted, input_ids_sorted_actual)
    assert_equal(segment_ids_sorted, segment_ids_sorted_actual)
    assert_equal(input_mask_sorted, input_mask_sorted_actual)
    assert_equal(masked_lm_labels_sorted, masked_lm_labels_sorted_actual)
    assert_equal(next_sentence_labels_sorted,
                 next_sentence_labels_sorted_actual)
    print('Test Passed')


def gen_seed_if_not_exists():
    seed = os.environ.get('SEED')
    if seed:
        return int(seed)

    with paddle.no_grad():
        seed = np.random.randint(low=0, high=10000, size=[1])
        seed = paddle.to_tensor(seed)
        seed = broadcast(seed, use_calc_stream=True)
        seed = seed.numpy()[0]
        os.environ['SEED'] = str(seed)
        return seed


def run_main():
    fleet.init(is_collective=True)
    seed = gen_seed_if_not_exists()

    np.random.seed(seed)
    run_test_case(56)
    run_test_case(47, 56)


if __name__ == "__main__":
    run_main()
