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
import sys
import random
import time
import h5py
import json
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import distutils.util

import torch
sys.path.append(
    "/limin29/mlperf-1.1-repo/submission_training_1.1/NVIDIA/benchmarks/bert/implementations/pytorch"
)
from modeling import BertEncoder

import paddle
import paddle.nn as nn
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from paddle.static import Program, program_guard

from models.utils.tools import TimeCostAverage
from mlperf_logging.mllog import constants
from bert_utils.mlperf_logging_helper import paddle_bert_print_start, paddle_bert_print_end, paddle_bert_print_event
import bert_utils.utility as utility

from models.modeling import NewBertEncoder
from models.modeling import BertConfig

iters = 1
x_type = np.float16
attention_mask_type = np.int32

batch_size = 56
max_seq_len = 512
embed_dim = 1024

attention_mask = np.random.randint(
    0, 5, size=[batch_size, max_seq_len]).astype(attention_mask_type)
pad_input = np.random.rand(batch_size, max_seq_len,
                           embed_dim).astype(x_type) * 0.25
dout_np = np.random.random(
    (batch_size, max_seq_len, embed_dim)).astype(np.float16) * 0.25

# adjust nonzero value to 1.
# todo: set maxseqlen_in_batch to 512, otherwise will report error.
for i, val in enumerate(attention_mask):
    if i == 2:
        for j, ele in enumerate(val):
            val[j] = 1
    for j, ele in enumerate(val):
        if ele != 0:
            val[j] = 1
attention_mask = np.ones((batch_size, max_seq_len)).astype(attention_mask_type)

bert_config_path = '/zengjinle/dataset/bert_data/phase1/bert_config.json'
config = BertConfig.from_json_file(bert_config_path)

config.num_hidden_layers = 1
config.hidden_size = embed_dim
config.max_seq_length = 512

config.num_attention_heads = 16
config.intermediate_size = 4096
config.hidden_act = 'gelu'
config.attention_probs_dropout_prob = 0
config.hidden_dropout_prob = 0

config.unpad_fmha = True
config.unpad = True
config.pad_fmha = False
config.pad = False

config.fuse_mask = False
config.enable_stream = False
config.fused_gelu_bias = False
config.fused_mha = False

# if true, the linear weight parameter is [out_feature, in_feature]
# fused qkv + bias
config.fused_bias_mha = True

# fused linear + bias
config.fused_bias_fc = True
if config.fused_bias_fc:
    config.weight_transpose = True
else:
    config.weight_transpose = False

# fused_dropout_reisudal_ln
config.fused_dropout_add = False
config.fused_dropout_add_ln = True
'''
Wq = (np.ones((config.hidden_size, config.hidden_size)).astype(x_type))*0.001
Wk = (np.ones((config.hidden_size, config.hidden_size)).astype(x_type))*0.001
Wv = (np.ones((config.hidden_size, config.hidden_size)).astype(x_type))*0.001
Linear_weight = (np.ones((config.hidden_size, config.hidden_size)).astype(x_type))*0.001
Bq = np.zeros((config.hidden_size)).astype(x_type)
Bk = np.zeros((config.hidden_size)).astype(x_type)
Bv = np.zeros((config.hidden_size)).astype(x_type)
Linear_bias = np.zeros((config.hidden_size)).astype(x_type)
'''
# fmha params, q, k, v and out_liear
Wq = np.random.rand(config.hidden_size,
                    config.hidden_size).astype(x_type) * 0.25
Wk = np.random.rand(config.hidden_size,
                    config.hidden_size).astype(x_type) * 0.25
Wv = np.random.rand(config.hidden_size,
                    config.hidden_size).astype(x_type) * 0.25
Linear_weight = np.random.rand(config.hidden_size,
                               config.hidden_size).astype(x_type) * 0.25
Bq = np.random.rand(config.hidden_size).astype(x_type) * 0.25
Bk = np.random.rand(config.hidden_size).astype(x_type) * 0.25
Bv = np.random.rand(config.hidden_size).astype(x_type) * 0.25
#Bq = np.zeros((config.hidden_size)).astype(x_type)
#Bk = np.zeros((config.hidden_size)).astype(x_type)
#Bv = np.zeros((config.hidden_size)).astype(x_type)
Linear_bias = np.random.rand(config.hidden_size).astype(x_type) * 0.25
#Linear_bias = np.zeros((config.hidden_size)).astype(x_type)

# ffn: linear_2 params
linear_2_weight = np.random.rand(config.hidden_size,
                                 config.intermediate_size).astype(x_type) * 0.25
linear_2_bias = np.random.rand(config.intermediate_size).astype(x_type) * 0.25
#linear_2_bias = np.zeros((config.intermediate_size)).astype(x_type)

# ffn: linear_3 params
linear_3_weight = np.random.rand(config.intermediate_size,
                                 config.hidden_size).astype(x_type) * 0.25
linear_3_bias = np.random.rand(config.hidden_size).astype(x_type) * 0.25

# layer_norm params
norm_1_weight = np.random.rand(config.hidden_size).astype(x_type) * 0.25
norm_1_bias = np.random.rand(config.hidden_size).astype(x_type) * 0.25
norm_2_weight = np.random.rand(config.hidden_size).astype(x_type) * 0.25
norm_2_bias = np.random.rand(config.hidden_size).astype(x_type) * 0.25

print("weight_transpose:", config.weight_transpose)


def run_dynamic():
    paddle.disable_static(place=paddle.CUDAPlace(0))
    paddle.set_default_dtype(x_type)

    encoder = NewBertEncoder(config)

    attention = encoder.layers[0].attention
    intermediate = encoder.layers[0].intermediate
    last_output = encoder.layers[0].output

    self_output = attention.output
    fmha = attention.fmha
    Wqkv = np.concatenate((Wq, Wk, Wv))
    Bqkv = np.concatenate((Bq, Bk, Bv))
    fmha.Wqkv.set_value(Wqkv)
    fmha.Bqkv.set_value(Bqkv)

    self_output.dense.bias.set_value(Linear_bias)
    if config.fused_dropout_add_ln:
        self_output.fused_dropout_add_ln.bias.set_value(norm_1_bias)
    self_output.layer_norm.bias.set_value(norm_1_bias)

    intermediate.dense.bias.set_value(linear_2_bias)

    last_output.dense.bias.set_value(linear_3_bias)
    last_output.layer_norm.bias.set_value(norm_2_bias)

    if config.fused_dropout_add_ln:
        self_output.fused_dropout_add_ln.weight.set_value(norm_1_weight)
    self_output.layer_norm.weight.set_value(norm_1_weight)

    last_output.layer_norm.weight.set_value(norm_2_weight)

    if not config.weight_transpose:
        self_output.dense.weight.set_value(Linear_weight)
        intermediate.dense.weight.set_value(linear_2_weight)
        last_output.dense.weight.set_value(linear_3_weight)
    else:
        self_output.dense.weight.set_value(Linear_weight.T)
        intermediate.dense.weight.set_value(linear_2_weight.T)
        last_output.dense.weight.set_value(linear_3_weight.T)

    print("new encoder: ", encoder)
    pad_input_tensor = paddle.to_tensor(pad_input, stop_gradient=False)
    attention_mask_tensor = paddle.to_tensor(attention_mask)
    encoder_out = encoder(pad_input_tensor, attention_mask_tensor)
    '''
    dout = paddle.to_tensor(dout_np)
    paddle.autograd.backward([encoder_out[-1]], [dout], retain_graph=True)
    '''
    return encoder_out[-1], pad_input_tensor.grad, encoder


def run_dynamic_pytorch_ref():

    encoder = BertEncoder(config)
    print("pytorch encoder:", encoder)

    attention = encoder.layer[0].attention
    intermediate = encoder.layer[0].intermediate
    last_output = encoder.layer[0].output

    self_output = attention.output
    fmha = attention.self

    Wqkv = np.concatenate((Wq, Wk, Wv))
    Bqkv = np.concatenate((Bq, Bk, Bv))
    fmha.Wqkv = torch.from_numpy(Wqkv).cuda()
    fmha.Bqkv = torch.from_numpy(Bqkv).cuda()

    self_output.dense.weight.data = torch.from_numpy(Linear_weight.T).cuda()
    self_output.dense.bias.data = torch.from_numpy(Linear_bias).cuda()
    self_output.LayerNorm.weight.data = torch.from_numpy(norm_1_weight).cuda()
    self_output.LayerNorm.bias.data = torch.from_numpy(norm_1_bias).cuda()

    intermediate.dense.weight.data = torch.from_numpy(linear_2_weight.T).cuda()
    intermediate.dense.bias.data = torch.from_numpy(linear_2_bias).cuda()

    last_output.dense.weight.data = torch.from_numpy(linear_3_weight.T).cuda()
    last_output.dense.bias.data = torch.from_numpy(linear_3_bias).cuda()
    last_output.LayerNorm.weight.data = torch.from_numpy(norm_2_weight).cuda()
    last_output.LayerNorm.bias.data = torch.from_numpy(norm_2_bias).cuda()

    print("pytorch encoder: ", encoder)
    pad_input_tensor = torch.from_numpy(pad_input).cuda()
    attention_mask_tensor = torch.from_numpy(attention_mask).cuda()
    encoder_out = encoder(
        pad_input_tensor,
        attention_mask_tensor,
        output_all_encoded_layers=False)
    '''
    dout = torch.from_numpy(dout_np).cuda()
    torch.autograd.backward(
            [encoder_out[-1]], [dout], retain_graph=True)
    '''

    # return encoder_out[-1], pad_input_tensor.grad
    return encoder_out[-1]


def run_dynamic_paddle_ref(fused_encoder):
    paddle.disable_static(place=paddle.CUDAPlace(0))
    paddle.set_default_dtype(x_type)
    ref_encoder_layer = nn.TransformerEncoderLayer(
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        activation=config.hidden_act,
        attn_dropout=config.attention_probs_dropout_prob,
        act_dropout=0)
    ref_encoder = nn.TransformerEncoder(ref_encoder_layer,
                                        config.num_hidden_layers)
    print("ref_encoder: ", ref_encoder)

    # generate src_mask, is it right? 
    input_mask = (1 - np.reshape(
        attention_mask.astype(np.float32),
        [attention_mask.shape[0], 1, 1, attention_mask.shape[1]])) * -1e9

    # set params
    attention = fused_encoder.layer[0].attention
    fmha = attention.fmha
    self_output = attention.output
    intermediate = fused_encoder.layer[0].intermediate
    last_output = fused_encoder.layer[0].output
    layer_norm_2 = last_output.layer_norm

    ref_self_attn = ref_encoder.layers[0].self_attn
    ref_norm_1 = ref_encoder.layers[0].norm1
    ref_norm_2 = ref_encoder.layers[0].norm2
    ref_linear_1 = ref_encoder.layers[0].linear1
    ref_linear_2 = ref_encoder.layers[0].linear2

    ref_self_attn.q_proj.weight.set_value(Wq)
    ref_self_attn.k_proj.weight.set_value(Wk)
    ref_self_attn.v_proj.weight.set_value(Wv)
    ref_self_attn.q_proj.bias.set_value(Bq)
    ref_self_attn.k_proj.bias.set_value(Bk)
    ref_self_attn.v_proj.bias.set_value(Bv)

    ref_self_attn.out_proj.weight.set_value(Linear_weight)
    ref_self_attn.out_proj.bias.set_value(Linear_bias)
    '''
    ref_self_attn.q_proj.weight.set_value(fmha.Wqkv[0:config.hidden_size, :])
    ref_self_attn.k_proj.weight.set_value(fmha.Wqkv[config.hidden_size:2*config.hidden_size, :])
    ref_self_attn.v_proj.weight.set_value(fmha.Wqkv[2*config.hidden_size:3*config.hidden_size, :])
    ref_self_attn.q_proj.bias.set_value(fmha.Bqkv[0:config.hidden_size])
    ref_self_attn.k_proj.bias.set_value(fmha.Bqkv[config.hidden_size:2*config.hidden_size])
    ref_self_attn.v_proj.bias.set_value(fmha.Bqkv[2*config.hidden_size:3*config.hidden_size])
    
    ref_self_attn.out_proj.weight.set_value(self_output.dense.weight)
    ref_self_attn.out_proj.bias.set_value(self_output.dense.bias)
    '''
    '''
    weight_np = self_output.layer_norm.weight.numpy().astype(np.float16)
    ref_norm_1.weight.set_value(weight_np)
    weight_np = self_output.layer_norm.bias.numpy().astype(np.float16)
    ref_norm_1.bias.set_value(weight_np)
    #ref_norm_1.weight.set_value(self_output.layer_norm.weight)
    #ref_norm_1.bias.set_value(self_output.layer_norm.bias)
    
    weight_np = layer_norm_2.weight.numpy().astype(np.float16)
    ref_norm_2.weight.set_value(weight_np)
    weight_np = layer_norm_2.bias.numpy().astype(np.float16)
    ref_norm_2.bias.set_value(weight_np)
    #ref_norm_2.weight.set_value(layer_norm_2.weight)
    #ref_norm_2.bias.set_value(layer_norm_2.bias)
    
    ref_linear_1.weight.set_value(intermediate.dense.weight)
    ref_linear_1.bias.set_value(intermediate.dense.bias)
    
    ref_linear_2.weight.set_value(last_output.dense.weight)
    ref_linear_2.bias.set_value(last_output.dense.bias)
    '''
    ref_norm_1.weight.set_value(norm_1_weight)
    ref_norm_1.bias.set_value(norm_1_weight)
    ref_norm_2.weight.set_value(norm_2_weight)
    ref_norm_2.bias.set_value(norm_2_bias)

    ref_linear_1.weight.set_value(linear_2_weight)
    ref_linear_1.bias.set_value(linear_2_bias)

    ref_linear_2.weight.set_value(linear_3_weight)
    ref_linear_2.bias.set_value(linear_3_bias)

    pad_input_tensor = paddle.to_tensor(pad_input, stop_gradient=False)
    input_mask_tensor = paddle.to_tensor(input_mask)
    output = pad_input_tensor
    encoder_outputs = []
    for mod in ref_encoder.layers:
        output = mod(output, src_mask=input_mask_tensor)
        encoder_outputs.append(output)

    return encoder_outputs[-1]


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


def test_dynamic(atol=1e-2):
    for i in range(iters):
        out, input_grad, fused_encoder = run_dynamic()
        ref_out = run_dynamic_pytorch_ref()
        # compared with pytorch 
        np.testing.assert_allclose(
            ref_out.cpu().detach().numpy(), out.numpy(), rtol=1e-5, atol=atol)
        res = np.array_equal(ref_out.cpu().detach().numpy(), out.numpy())
        if (res):
            print("array equal")
        else:
            print("array not equal!")
        #np.testing.assert_allclose(
        #    ref_input_grad.cpu().detach().numpy(), input_grad.numpy(), rtol=1e-5, atol=atol)

        # compared with paddle
        #ref_paddle_out = run_dynamic_paddle_ref(fused_encoder)
        #print("ref_paddle_out: ", ref_paddle_out)
        #np.testing.assert_allclose(
        #    ref_paddle_out.numpy(), out.numpy(), rtol=1e-5, atol=atol)
        print("out = ", out)


def test_static():
    paddle.enable_static()
    with paddle.static.program_guard(Program()):
        out = run_static()
    # print("encoder out: ", out[-1])


test_dynamic(atol=1e-2)
# test_static()
