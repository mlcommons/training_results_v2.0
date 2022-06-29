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
import paddle.nn as nn
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from paddle.static import Program, program_guard

from models.utils.tools import TimeCostAverage
from mlperf_logging.mllog import constants
from bert_utils.mlperf_logging_helper import paddle_bert_print_start, paddle_bert_print_end, paddle_bert_print_event
import bert_utils.utility as utility

from models.modeling import FusedDense
from models.modeling import BertConfig

x_type = np.float16

batch_size = 8
max_seq_len = 512
embed_dim = 1024

pad_input = np.random.rand(batch_size, max_seq_len, embed_dim).astype(x_type)
dout = np.random.random((batch_size, max_seq_len, embed_dim)).astype(x_type)

bert_config_path = '/zengjinle/dataset/bert_data/phase1/bert_config.json'
config = BertConfig.from_json_file(bert_config_path)

config.num_hidden_layer = 2
config.hidden_size = embed_dim

config.attention_probs_dropout_prob = 0
config.hidden_dropout_prob = 0

config.fused_dense_weight_transpose = True


def run_dynamic():
    paddle.set_default_dtype(x_type)
    fused_dense = FusedDense(config.hidden_size, config.hidden_size,
                             config.fused_dense_weight_transpose)
    pad_input_tensor = paddle.to_tensor(pad_input, stop_gradient=False)
    out = fused_dense(pad_input_tensor)

    dout_tensor = paddle.to_tensor(dout)
    paddle.autograd.backward([out], [dout_tensor], retain_graph=True)
    return out, pad_input_tensor.grad, fused_dense


def run_dynamic_ref(fused_dense):
    paddle.set_default_dtype(x_type)
    dense = nn.Linear(config.hidden_size, config.hidden_size)

    # set values for ln.
    if config.fused_dense_weight_transpose:
        weight_np = fused_dense.weight.numpy()
        weight_t_np = weight_np.transpose(1, 0)
        dense.weight.set_value(weight_t_np)
    else:
        dense.weight.set_value(fused_dense.weight)
    dense.bias.set_value(fused_dense.bias)

    pad_input_tensor = paddle.to_tensor(pad_input, stop_gradient=False)
    out = dense(pad_input_tensor)

    dout_tensor = paddle.to_tensor(dout)
    paddle.autograd.backward([out], [dout_tensor], retain_graph=True)
    return out, pad_input_tensor.grad


def run_static():
    paddle.set_default_dtype(x_type)
    fused_dense = FusedDense(config.hidden_size, config.hidden_size)

    x = paddle.static.data(
        name='X', shape=[batch_size, max_seq_len, embed_dim], dtype=x_type)

    fused_out = fused_dense(x)

    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    out = exe.run(paddle.static.default_main_program(),
                  feed={"X": pad_input},
                  fetch_list=[fused_out])
    return out


def test_dynamic(atol=1e-2):
    paddle.disable_static(place=paddle.CUDAPlace(0))
    out, input_grad, fused_dense = run_dynamic()
    ref_out, ref_input_grad = run_dynamic_ref(fused_dense)

    np.testing.assert_allclose(
        ref_out.numpy(), out.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        ref_input_grad.numpy(), input_grad.numpy(), rtol=1e-5, atol=atol)
    print("Dynamic Test success!")


def test_static():
    paddle.enable_static()
    with paddle.static.program_guard(Program()):
        out = run_static()
    # print("static out: ", out)


test_dynamic()
test_static()
