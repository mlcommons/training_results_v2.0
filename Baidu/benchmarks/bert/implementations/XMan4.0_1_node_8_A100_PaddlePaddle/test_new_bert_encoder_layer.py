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

import argparse
import collections
import itertools
import os
import random
import time
import h5py
import json
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import distutils.util

import paddle
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from paddle.static import Program, program_guard

from models.utils.tools import TimeCostAverage
from mlperf_logging.mllog import constants
from bert_utils.mlperf_logging_helper import paddle_bert_print_start, paddle_bert_print_end, paddle_bert_print_event
import bert_utils.utility as utility

from models.modeling import NewBertEncoder
from models.modeling import BertConfig

x_type = np.float16
attention_mask_type = np.int32

batch_size = 8
max_seq_len = 512
embed_dim = 1024

attention_mask = np.random.randint(
    0, 5, size=[batch_size, max_seq_len]).astype(attention_mask_type)
pad_input = np.random.rand(batch_size, max_seq_len, embed_dim).astype(x_type)

# adjust nonzero value to 1.
# todo: set maxseqlen_in_batch to 512, otherwise will report error.
for i, val in enumerate(attention_mask):
    if i == 2:
        for j, ele in enumerate(val):
            val[j] = 1
    for j, ele in enumerate(val):
        if ele != 0:
            val[j] = 1

# todo
bert_config_path = '/zengjinle/dataset/bert_data/phase1/bert_config.json'
config = BertConfig.from_json_file(bert_config_path)

config.num_hidden_layer = 2
config.hidden_size = embed_dim

#config.num_attention_heads=16 
#config.intermediate_size=4096
#config.hidden_act='gelu'
config.attention_probs_dropout_prob = 0
config.hidden_dropout_prob = 0

config.unpad_fmha = True


def run_dynamic():
    paddle.set_default_dtype(x_type)

    encoder = NewBertEncoder(config)
    pad_input_tensor = paddle.to_tensor(pad_input, stop_gradient=False)
    attention_mask_tensor = paddle.to_tensor(attention_mask)
    print("pad_input:", pad_input_tensor)
    print("attention_mask:", attention_mask_tensor)
    encoder_out = encoder(pad_input_tensor, attention_mask_tensor)
    print("encoder_out: ", encoder_out[-1])
    print("encoder_out.shape: ", encoder_out[-1].shape)

    #
    dout = np.random.random(
        (batch_size, max_seq_len, embed_dim)).astype(np.float16)
    dout = paddle.to_tensor(dout)
    dout = paddle.cast(dout, 'float16')
    #dout = paddle.ones(shape=[5, 3], dtype='float16')
    paddle.autograd.backward([encoder_out[-1]], [dout], retain_graph=True)
    print("pad_input.grad: ", pad_input_tensor.grad)


def run_static():

    paddle.set_default_dtype(x_type)
    encoder = NewBertEncoder(config)

    x = paddle.static.data(
        name='X', shape=[batch_size, max_seq_len, embed_dim], dtype=x_type)
    attn_mask = paddle.static.data(
        name='SrcMask',
        shape=[
            batch_size,
            max_seq_len,
        ],
        dtype=attention_mask_type)

    encoder_out = encoder(x, attn_mask)

    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    out = exe.run(paddle.static.default_main_program(),
                  feed={"X": pad_input,
                        "SrcMask": attention_mask},
                  fetch_list=[encoder_out])
    return out


def test_dynamic():
    paddle.disable_static(place=paddle.CUDAPlace(0))
    run_dynamic()


def test_static():
    paddle.enable_static()
    with paddle.static.program_guard(Program()):
        out = run_static()
    print("encoder out: ", out[-1])


#test_dynamic()
test_static()
