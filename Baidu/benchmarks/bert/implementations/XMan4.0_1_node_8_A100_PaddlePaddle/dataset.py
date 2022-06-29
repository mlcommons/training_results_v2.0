# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import h5py
from functools import partial

import numpy as np
import paddle
import paddle.fluid.core as core
import custom_setup_ops
import time
from paddle.fluid.layer_helper import LayerHelper
from paddle.io import DataLoader, Dataset, RandomSampler
from stack import Stack

import utility
from init_env import get_context

context = get_context()
use_nv_input = utility.use_nv_input()
reader_place = utility.get_place()


def pd_collate_data(data, args, stack_fn=Stack()):
    num_fields = len(data[0])
    out = [None] * num_fields
    # input_ids, segment_ids, input_mask, masked_lm_positions,
    # masked_lm_labels, next_sentence_labels, mask_token_num
    for i in (0, 1, 2, 5):
        out[i] = stack_fn([x[i] for x in data])
    batch_size, seq_length = out[0].shape
    size = num_mask = sum(len(x[3]) for x in data)
    # Padding for divisibility by 8 for fp16 or int8 usage
    if size % 8 != 0:
        size += 8 - (size % 8)
    # masked_lm_positions
    # Organize as a 1D tensor for gather or use gather_nd
    out[3] = np.full(size, 0, dtype=np.int32)
    # masked_lm_labels
    out[4] = np.full([size, 1], -1, dtype=np.int64)
    mask_token_num = 0
    for i, x in enumerate(data):
        for j, pos in enumerate(x[3]):
            out[3][mask_token_num] = i * seq_length + pos
            out[4][mask_token_num] = x[4][j]
            mask_token_num += 1
    # mask_token_num
    out.append(np.asarray([mask_token_num], dtype=np.float32))
    if args.use_amp and args.use_pure_fp16:
        # cast input_mask to fp16
        out[2] = out[2].astype(np.float16)
        # cast masked_lm_scale to fp16
        out[-1] = out[-1].astype(np.float16)
    return out


def create_pretraining_dataset(data_holders, f_id, tolist=True):
    args = context.args
    if f_id == 0:
        context.shuffle_files()

    if args.use_uncompressed_dataset and use_nv_input:
        train_data = NVUncompressedPretrainingDataset(
            input_file=context.files[f_id])
    else:
        train_data = PretrainingDataset(
            input_file=context.files[f_id],
            max_pred_length=args.max_predictions_per_seq,
            max_seq_length=args.max_seq_length)
    train_batch_sampler = paddle.io.BatchSampler(
        train_data, batch_size=args.train_batch_size, shuffle=True)

    train_data_loader = DataLoader(
        dataset=train_data,
        places=[reader_place],
        feed_list=data_holders,
        batch_sampler=train_batch_sampler,
        collate_fn=None if use_nv_input else partial(
            _collate_data, args=args),
        num_workers=0 if args.train_batch_size <= 8 else 4,
        worker_init_fn=context.worker_init,
        return_list=False)
    return list(train_data_loader) if tolist else train_data_loader


def create_cpu_exchange_padding_pretraining_dataset(data_holders,
                                                    f_id,
                                                    tolist=True):
    data = context.read_file(f_id)
    train_data_loader = DataLoader.from_generator(
        feed_list=data_holders,
        capacity=len(data[0]),
        return_list=False,
        drop_last=False)
    train_data_loader.set_batch_generator(
        lambda: data[0], places=[reader_place])
    if tolist:
        return list(train_data_loader), data[1]
    else:
        return train_data_loader, data[1]


def create_new_eval_dataset(data_holders):
    data = context.read_eval_file()
    eval_data_loader = DataLoader.from_generator(
        feed_list=data_holders,
        capacity=len(data[0]),
        return_list=False,
        drop_last=False)
    eval_data_loader.set_batch_generator(lambda: data[0], places=[reader_place])
    return list(eval_data_loader), data[1]


def create_eval_dataset(args, data_holders, worker_init=None, places=None):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(
                PretrainingDataset(
                    eval_file_path,
                    max_pred_length=args.max_predictions_per_seq,
                    max_seq_length=args.max_seq_length))
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[:args.num_eval_examples]
                break
    chunk_size = args.num_eval_examples // utility.get_num_trainers()
    rank = utility.get_trainer_id()
    remainder = args.num_eval_examples % utility.get_num_trainers()
    if rank < remainder:
        eval_data = eval_data[(chunk_size + 1) * rank:(chunk_size + 1) * (rank +
                                                                          1)]
    else:
        eval_data = eval_data[chunk_size * rank + remainder:chunk_size * (
            rank + 1) + remainder]

    eval_data = ListDataset(eval_data)
    eval_batch_sampler = paddle.io.BatchSampler(
        eval_data, batch_size=args.eval_batch_size, shuffle=False)
    eval_dataloader = DataLoader(
        eval_data,
        places=places,
        feed_list=data_holders,
        batch_sampler=eval_batch_sampler,
        collate_fn=None if use_nv_input else partial(
            _collate_data, args=args),
        num_workers=0 if min(chunk_size, args.eval_batch_size) <= 10 else 4,
        worker_init_fn=worker_init,
        return_list=False)

    return eval_dataloader


def inplace_exchange_padding(input_ids, segment_ids, input_mask,
                             masked_lm_labels, next_sentence_labels,
                             max_batch_size):
    helper = LayerHelper('exchange_padding')
    inputs = {
        'InputIds': [input_ids],
        'SegmentIds': [segment_ids],
        'InputMask': [input_mask],
        'MaskedLMLabels': [masked_lm_labels],
        'NextSentenceLabels': [next_sentence_labels],
    }

    outputs = {
        'InputIdsOut': [input_ids],
        'SegmentIdsOut': [segment_ids],
        'InputMaskOut': [input_mask],
        'MaskedLMLabelsOut': [masked_lm_labels],
        'NextSentenceLabelsOut': [next_sentence_labels],
    }

    attrs = {
        'max_batch_size': max_batch_size,
        'ring_id': 0,
        'device_id': utility.get_trainer_id(),
        'num_devices': utility.get_num_trainers(),
        'need_pad': False,
    }

    helper.append_op(
        type='sort_bert_inputs_across_devices',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    return input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels


def pd_create_data_holder(args):
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[-1, -1], dtype="int64")
    input_mask = paddle.static.data(
        name="input_mask", shape=[-1, 1, 1, -1], dtype="float32")
    masked_lm_positions = paddle.static.data(
        name="masked_lm_positions", shape=[-1], dtype="int32")
    masked_lm_labels = paddle.static.data(
        name="masked_lm_labels", shape=[-1, 1], dtype="int64")
    next_sentence_labels = paddle.static.data(
        name="next_sentence_labels", shape=[-1, 1], dtype="int64")
    masked_lm_scale = paddle.static.data(
        name="masked_lm_scale", shape=[-1, 1], dtype="float32")
    inputs = [input_ids, segment_ids, input_mask, masked_lm_positions]
    labels = [masked_lm_labels, next_sentence_labels, masked_lm_scale]
    # None stands for position_ids argument in forward
    return inputs + labels, inputs[:2] + [None] + inputs[2:], labels


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class PDPretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_seq_length=None):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            input[index].astype(np.int64)
            if indice < 5 else np.asarray(input[index].astype(np.int64))
            for indice, input in enumerate(self.inputs)
        ]

        return self.process_sample(input_ids, input_mask, segment_ids,
                                   masked_lm_positions, masked_lm_ids,
                                   next_sentence_labels, self.max_pred_length)

    @classmethod
    def process_sample(cls, input_ids, input_mask, segment_ids,
                       masked_lm_positions, masked_lm_ids, next_sentence_labels,
                       max_pred_length):
        # TODO: whether to use reversed mask by changing 1s and 0s to be
        # consistent with nv bert
        input_mask = (1 - np.reshape(
            input_mask.astype(np.float32), [1, 1, input_mask.shape[0]])) * -1e9

        index = max_pred_length
        # store number of  masked tokens in index
        # outputs of torch.nonzero diff with that of numpy.nonzero by zip
        padded_mask_indices = (masked_lm_positions == 0).nonzero()[0]
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
            mask_token_num = index
        else:
            index = max_pred_length
            mask_token_num = max_pred_length
        # masked_lm_labels = np.full(input_ids.shape, -1, dtype=np.int64)
        # masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        masked_lm_labels = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]
        # softmax_with_cross_entropy enforce last dim size equal 1
        masked_lm_labels = np.expand_dims(masked_lm_labels, axis=-1)
        next_sentence_labels = np.expand_dims(next_sentence_labels, axis=-1)

        return [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels
        ]


def nv_create_data_holder(args):
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, args.max_seq_length], dtype="int16")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[-1, args.max_seq_length], dtype="int16")
    input_mask = paddle.static.data(
        name="input_mask", shape=[-1, args.max_seq_length], dtype="int16")
    masked_lm_labels = paddle.static.data(
        name="masked_lm_labels", shape=[-1, args.max_seq_length], dtype="int16")
    next_sentence_labels = paddle.static.data(
        name="next_sentence_labels", shape=[-1, 1], dtype="int16")

    # [bs]
    seq_len = paddle.static.data(name="seq_len", shape=[-1], dtype="int32")
    # [bs + 1]
    prefix_sum_seq_len = paddle.static.data(
        name="prefix_sum_seq_len", shape=[-1], dtype="int32")
    host_prefix_sum_seq_len = paddle.static.data(
        name="host_prefix_sum_seq_len", shape=[-1], dtype="int32")
    # [1]
    max_seq_len = paddle.static.data(
        name="max_seq_len", shape=[1], dtype="int32")
    # [max_seq_len]
    nonzeros_indices = paddle.static.data(
        name="nonzeros_indices", shape=[-1], dtype="int32")

    num_valid = paddle.static.data(name="num_valid", shape=[1], dtype="float32")
    masked_lm_position = paddle.static.data(
        name="masked_lm_position", shape=[-1], dtype="int32")
    masked_lm_ids = paddle.static.data(
        name="masked_lm_ids", shape=[-1], dtype="int32")

    if args.gpu_exchange_padding and utility.get_num_trainers() > 1:
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = inplace_exchange_padding(
            input_ids, segment_ids, input_mask, masked_lm_labels,
            next_sentence_labels, args.train_batch_size)
    inputs = [input_ids, segment_ids, input_mask, masked_lm_labels]
    labels = [masked_lm_labels, next_sentence_labels]
    varlen_info = [
        seq_len, prefix_sum_seq_len, host_prefix_sum_seq_len, max_seq_len,
        nonzeros_indices
    ]
    mlm_label_info = [num_valid, masked_lm_position, masked_lm_ids]

    # None stands for position_ids argument in forward
    data_holders = [
        input_ids, segment_ids, input_mask, masked_lm_labels,
        next_sentence_labels, seq_len, prefix_sum_seq_len, nonzeros_indices,
        masked_lm_position, masked_lm_ids, num_valid
    ]
    return data_holders, inputs[:2] + [None] + inputs[
        2:] + varlen_info + mlm_label_info, labels + mlm_label_info, varlen_info, mlm_label_info


create_data_holder = nv_create_data_holder if use_nv_input else pd_create_data_holder


class NVUncompressedPretrainingDataset(Dataset):
    def __init__(self, input_file):
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            input[index].astype(np.int16)
            if indice < 5 else np.asarray(input[index].astype(np.int16))
            for indice, input in enumerate(self.inputs)
        ]

        length = np.sum(masked_lm_positions)
        masked_lm_positions = masked_lm_positions[:length]
        masked_lm_ids = masked_lm_ids[:length]

        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int16)
        masked_lm_labels[masked_lm_positions] = masked_lm_ids
        return input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels


class NVPretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_seq_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
            'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = np.zeros(self.max_seq_length).astype(np.int16)
        input_mask = np.zeros(self.max_seq_length).astype(np.int16)
        segment_ids = np.zeros(self.max_seq_length).astype(np.int16)
        [
            _input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids,
            _next_sentence_labels
        ] = [
            input[index].astype(np.int16)
            if indice < 4 else np.asarray(input[index].astype(np.int16))
            for indice, input in enumerate(self.inputs)
        ]

        input_mask_len = _input_ids.shape[-1]
        input_ids[:input_mask_len] = _input_ids
        input_mask[:input_mask_len] = np.ones(
            (1, input_mask_len)).astype(np.int16)
        segment_ids[:input_mask_len] = _segment_ids
        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int16)
        masked_lm_labels[_masked_lm_positions] = _masked_lm_ids
        next_sentence_labels = _next_sentence_labels
        return input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels


if use_nv_input:
    _collate_data = None
    create_data_holder = nv_create_data_holder
    PretrainingDataset = NVPretrainingDataset
else:
    _collate_data = pd_collate_data
    create_data_holder = pd_create_data_holder
    PretrainingDataset = PDPretrainingDataset


def gen_tensor(place, shape, dtype, value=1):
    np_arr = (np.ones(shape) * value).astype(dtype)
    t = core.Tensor()
    t.set(np_arr, place)
    return t


def gen_prefix_sum_tensor(place, batch_size, dtype, value=1):
    shape = [batch_size]
    np_seq_len_arr = (np.ones(shape) * value).astype(dtype)
    np_arr = np.cumsum(np_seq_len_arr)
    np_arr = np.insert(np_arr, 0, 0).astype(dtype)
    t = core.Tensor()
    t.set(np_arr, place)
    return t


def gen_nonzeros_indices_tensor(place, num_elems, dtype, value=1):
    np_arr = np.arange(num_elems).reshape(num_elems).astype(dtype)
    t = core.Tensor()
    t.set(np_arr, place)
    return t


def prepare_warmup_data(args, batch_size, place, dtype=None, return_dict=True):
    if use_nv_input:
        if dtype is None:
            dtype = np.int16

        seqlen_data = gen_tensor(place, [batch_size], np.int32, value=512)
        prefix_sum_seq_len_data = gen_prefix_sum_tensor(
            place, batch_size, np.int32, value=512)
        host_prefix_sum_seq_len_data = gen_prefix_sum_tensor(
            core.CPUPlace(), batch_size, np.int32, value=512)
        max_seq_len_data = gen_tensor(core.CPUPlace(), [1], np.int32, value=512)
        nonzeros_indices_data = gen_nonzeros_indices_tensor(
            place, batch_size * args.max_seq_length, np.int32)
        num_valid_data = gen_tensor(
            place, [1], np.float32, value=(batch_size * args.max_seq_length))
        # [0, batch_size * max_seq_len), 1d
        masked_lm_position_data = gen_nonzeros_indices_tensor(
            place, batch_size * args.max_seq_length, np.int32)
        # [1, 1, 1, ...], 1d
        masked_lm_ids_data = gen_tensor(
            place, [batch_size * args.max_seq_length], np.int32)

        data = [
            gen_tensor(place, [batch_size, args.max_seq_length], dtype),
            gen_tensor(place, [batch_size, args.max_seq_length], dtype),
            gen_tensor(place, [batch_size, args.max_seq_length], dtype),
            gen_tensor(place, [batch_size, args.max_seq_length], dtype),
            gen_tensor(place, [batch_size], dtype),
            seqlen_data,
            prefix_sum_seq_len_data,
            host_prefix_sum_seq_len_data,
            max_seq_len_data,
            nonzeros_indices_data,
            num_valid_data,
            masked_lm_position_data,
            masked_lm_ids_data,
        ]
        names = [
            "input_ids",
            "segment_ids",
            "input_mask",
            "masked_lm_labels",
            "next_sentence_labels",
            "seq_len",
            "prefix_sum_seq_len",
            "host_prefix_sum_seq_len",
            "max_seq_len",
            "nonzeros_indices",
            "num_valid",
            "masked_lm_position",
            "masked_lm_ids",
        ]
    else:
        if dtype is None:
            dtype = np.int64
        data = []
        for _ in range(batch_size):
            input_ids = np.ones([args.max_seq_length], dtype=dtype)
            input_mask = np.ones([args.max_seq_length], dtype=dtype)
            segment_ids = np.ones([args.max_seq_length], dtype=dtype)
            masked_lm_positions = np.ones(
                [args.max_predictions_per_seq], dtype=dtype)
            masked_lm_ids = np.ones([args.max_predictions_per_seq], dtype=dtype)
            next_sentence_labels = 1
            sample = PDPretrainingDataset.process_sample(
                input_ids, input_mask, segment_ids, masked_lm_positions,
                masked_lm_ids, next_sentence_labels,
                args.max_predictions_per_seq)
            data.append(sample)
        data = pd_collate_data(data, args)
        names = [
            "input_ids", "segment_ids", "input_mask", "masked_lm_positions",
            "masked_lm_labels", "next_sentence_labels", "masked_lm_scale"
        ]
    assert len(names) == len(data)
    return dict(zip(names, data)) if return_dict else data
