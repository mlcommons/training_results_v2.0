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

import collections
import copy
import math
import numpy as np
import numbers
import json
import sys

import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh
from paddle.nn import Layer, LayerList
from paddle.fluid.initializer import Constant
import utility

from bert_padding import generate_mask
try:
    from custom_setup_ops import custom_fmha, custom_fused_dropout_residual_ln, custom_fused_dense
except ImportError as e:
    print('custom_setup_ops import error: {}'.format(e))

from .load_tf_checkpoint import load_pickled_tf_checkpoint, save_pickled_tf_checkpoint

from .mlperf_logging_helper import paddle_bert_print_event

__all__ = [
    'BertConfig',
    'BertModel',
    'BertForPretraining',
    'BertPretrainingCriterion',
    'BertPretrainingHeads',
]

use_nv_input = utility.use_nv_input()

GELU_APPROXIMATE = True


def get_activation():
    return nn.GELU(approximate=GELU_APPROXIMATE)
    # return nn.ReLU()


def mask_gather(var, mask):
    return paddle.gather_nd(var, paddle.fluid.layers.where(mask))


def gen_pos_id(input_ids):
    ones = paddle.ones_like(input_ids)
    seq_length = paddle.cumsum(ones, axis=-1)
    position_ids = seq_length - ones
    position_ids.stop_gradient = True
    return position_ids


def fuse_dense(x,
               y,
               bias,
               transx=False,
               transy=False,
               with_gelu=False,
               use_addto=False):
    out = custom_fused_dense(
        x=x, y=y, bias=bias, transx=transx, transy=transy, use_addto=use_addto)
    if with_gelu:
        out = get_activation()(out)
    return out


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, **kwargs):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (
                sys.version_info[0] == 2 and
                isinstance(vocab_size_or_config_json_file, unicode)):
            with open(
                    vocab_size_or_config_json_file, "r",
                    encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            assert not kwargs, "kwargs should be empty if config json file is provided"
            self._fill_dict(json_config)
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self._fill_dict(kwargs)
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        config._fill_dict(json_object)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def _fill_dict(self, kwargs=None):
        defaults = {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pool_act": "tanh",
            "pad_token_id": 0,
        }

        # fill defaults
        for key, value in defaults.items():
            if key not in self.__dict__:
                self.__dict__[key] = value

        # fill other values
        if kwargs:
            for key, value in kwargs.items():
                self.__dict__[key] = value

        assert self.pad_token_id == 0, "pad_token_id must be 0"


def transpose_2d(x):
    assert len(x.shape) == 2
    return np.ascontiguousarray(np.transpose(x, (1, 0)))


class TFCkptHelper:
    def __init__(self, args, config, checkpoint_path, place):
        self.pd_vars_to_tf_vars = collections.OrderedDict()
        self.transpose_vars = set()
        self.args = args
        self.config = config
        self.place = place
        self.checkpoint_path = checkpoint_path
        self._tf_vars = None

        self.fuse_attn_qkv = self.args.unpad_fmha
        self.attn_fused_qkv_weights = [
            None for _ in range(self.config.num_hidden_layers)
        ]
        self.attn_fused_qkv_biases = [
            None for _ in range(self.config.num_hidden_layers)
        ]

    @property
    def tf_vars(self):
        if self._tf_vars is None:
            self._tf_vars = load_pickled_tf_checkpoint(self.checkpoint_path)
        return self._tf_vars

    def save(self, output_path, get_parameter_func):
        tf_vars = {}
        if self.fuse_attn_qkv:
            for idx in range(self.config.num_hidden_layers):
                prefix = self._enc_prefix(idx) + "attention/self/"

                pd_weight_name = self.attn_fused_qkv_weights[idx]
                pd_weight_var = self._get_fp32_param(pd_weight_name,
                                                     get_parameter_func)
                assert len(pd_weight_var.shape) == 2

                pd_bias_name = self.attn_fused_qkv_biases[idx]
                pd_bias_var = self._get_fp32_param(pd_bias_name,
                                                   get_parameter_func)
                assert len(pd_bias_var.shape) == 1

                need_transpose = pd_weight_name in self.transpose_vars
                if need_transpose:  # split along dim 0 and transpose
                    assert pd_weight_var.shape[0] == 3 * pd_weight_var.shape[1]
                    assert pd_weight_var.shape[0] == pd_bias_var.shape[0]
                    tf_weights = np.split(pd_weight_var, 3, axis=0)
                    tf_weights = [transpose_2d(w) for w in tf_weights]
                else:  # split along dim 1 
                    assert pd_weight_var.shape[1] == 3 * pd_weight_var.shape[0]
                    assert pd_weight_var.shape[1] == pd_bias_var.shape[0]
                    tf_weights = np.split(pd_weight_var, 3, axis=1)
                tf_biases = np.split(pd_bias_var, 3, axis=0)

                assert len(tf_weights) == 3
                assert len(tf_biases) == 3

                for i, name in enumerate(["query", "key", "value"]):
                    tf_var_name = prefix + name + "/"
                    weight_name = tf_var_name + "kernel"
                    bias_name = tf_var_name + "bias"
                    tf_vars[weight_name] = tf_weights[i]
                    tf_vars[bias_name] = tf_biases[i]

        for pd_var_name, tf_var_name in self.pd_vars_to_tf_vars.items():
            pd_var = self._get_fp32_param(pd_var_name, get_parameter_func)
            if "output_weights" in tf_var_name or pd_var_name in self.transpose_vars:
                pd_var = transpose_2d(pd_var)
            tf_vars[tf_var_name] = pd_var

        for key, tf_value in self.tf_vars.items():
            if key not in tf_vars:
                tf_vars[key] = tf_value
                continue

            pd_value = tf_vars[key]
            if tf_value.shape == pd_value.shape:
                continue

            if key == 'bert/embeddings/word_embeddings':
                assert len(tf_value.shape) == 2
                assert len(pd_value.shape) == 2
                assert tf_value.shape[1] == pd_value.shape[1]
                pd_value = pd_value[0:tf_value.shape[0]]
            elif key == 'cls/predictions/output_bias':
                assert len(tf_value.shape) == 1
                assert len(pd_value.shape) == 1
                pd_value = pd_value[0:tf_value.shape[0]]
            else:
                raise ValueError("unsupported key {}".format(key))

            tf_vars[key] = pd_value

        return save_pickled_tf_checkpoint(tf_vars, output_path)

    def load(self, get_parameter_func):
        tf_vars = self.tf_vars
        loaded_var_names = set()

        if self.fuse_attn_qkv:
            for idx in range(self.config.num_hidden_layers):
                weights = []
                biases = []
                prefix = self._enc_prefix(idx) + "attention/self/"

                pd_weight_name = self.attn_fused_qkv_weights[idx]
                pd_bias_name = self.attn_fused_qkv_biases[idx]

                need_transpose = pd_weight_name in self.transpose_vars
                for name in ["query", "key", "value"]:
                    tf_var_name = prefix + name + "/"
                    weight_name = tf_var_name + "kernel"
                    bias_name = tf_var_name + "bias"
                    
                    if utility.get_trainer_id() == 0:
                        paddle_bert_print_event(key='weights_initialization', metadata={'tensor':weight_name})
                        paddle_bert_print_event(key='weights_initialization', metadata={'tensor':bias_name})

                    if need_transpose:
                        weights.append(transpose_2d(tf_vars[weight_name]))
                    else:
                        weights.append(tf_vars[weight_name])
                    biases.append(tf_vars[bias_name])

                    loaded_var_names.add(weight_name)
                    loaded_var_names.add(bias_name)

                weight = np.concatenate(
                    weights, axis=0 if need_transpose else 1)
                bias = np.concatenate(biases)
                weight_pd_vars = get_parameter_func(pd_weight_name)
                bias_pd_vars = get_parameter_func(pd_bias_name)

                self._set_var_value(
                    weight_pd_vars, weight,
                    self.attn_fused_qkv_weights[idx] + "/qkv/kernel",
                    prefix + "qkv/kernel")
                self._set_var_value(
                    bias_pd_vars, bias,
                    self.attn_fused_qkv_biases[idx] + "/qkv/bias",
                    prefix + "qkv/bias")

        for idx, (pd_var_name,
                  tf_var_name) in enumerate(self.pd_vars_to_tf_vars.items()):
            if utility.get_trainer_id() == 0:
                paddle_bert_print_event(key='weights_initialization', metadata={'tensor':tf_var_name})
            var_value = tf_vars[tf_var_name]
            if "output_weights" in tf_var_name or pd_var_name in self.transpose_vars:
                if utility.get_trainer_id() == 0:
                    print('{} needs to transpose'.format(tf_var_name))
                var_value = transpose_2d(var_value)

            pd_vars = get_parameter_func(pd_var_name)
            self._set_var_value(pd_vars, var_value, pd_var_name, tf_var_name)

            loaded_var_names.add(tf_var_name)

        left_var_names = set()
        for var_name in self.pd_vars_to_tf_vars.values():
            if var_name not in loaded_var_names:
                left_var_names.add(var_name)
            else:
                loaded_var_names.remove(var_name)

        if self.fuse_attn_qkv:
            assert len(loaded_var_names
                       ) == 6 * self.config.num_hidden_layers, loaded_var_names
        else:
            assert len(loaded_var_names) == 0, loaded_var_names
        assert len(left_var_names) == 0, left_var_names

    def _set_var_value(self, pd_vars, var_value, pd_var_name, tf_var_name):
        if isinstance(pd_vars, (list, tuple)):
            assert len(pd_vars) == 2
            pd_var, master_pd_var = pd_vars
        else:
            pd_var = pd_vars
            master_pd_var = None
        pd_var_shape = tuple(pd_var.shape())
        tf_var_shape = tuple(var_value.shape)
        if pd_var_shape != tf_var_shape:
            if utility.get_trainer_id() == 0:
                print("{} vs {} shape differs: {} vs {}".format(
                    pd_var_name, tf_var_name, pd_var_shape, tf_var_shape))
            assert len(pd_var_shape) == len(tf_var_shape)
            slices = []
            n = len(pd_var_shape)
            for i in range(n):
                assert pd_var_shape[i] >= tf_var_shape[i]
                slices.append(slice(0, tf_var_shape[i], 1))
            new_var_value = np.zeros(pd_var_shape, dtype=var_value.dtype)
            new_var_value[slices] = var_value
            var_value = new_var_value

        if pd_var._dtype() == paddle.float16:
            assert var_value.dtype == np.float32
            if master_pd_var is not None:
                assert master_pd_var._dtype() == paddle.float32
                if utility.get_trainer_id() == 0:
                    print("Set master weight for {} {}".format(pd_var_name,
                                                               tf_var_name))
                self._inplace_set_tensor(master_pd_var, var_value)
            self._inplace_set_tensor(pd_var, var_value.astype(np.float16))
        elif pd_var._dtype() == paddle.float32:
            assert var_value.dtype == np.float32
            assert master_pd_var is None
            self._inplace_set_tensor(pd_var, var_value)
        else:
            raise TypeError("unsupported data type {}".format(pd_var._dtype()))

    def _inplace_set_tensor(self, tensor, value):
        old_ptr = tensor._ptr()
        tensor.set(value, self.place)
        new_ptr = tensor._ptr()
        assert old_ptr == new_ptr

    def _get_fp32_param(self, pd_var_name, get_parameter_func):
        pd_var_name = self._to_pd_var_name(pd_var_name)
        pd_vars = get_parameter_func(pd_var_name)

        assert isinstance(pd_vars, (list, tuple))
        assert len(pd_vars) == 2
        pd_var, master_pd_var = pd_vars
        if master_pd_var is not None:
            assert pd_var._dtype() == paddle.float16
            assert master_pd_var._dtype() == paddle.float32
            assert pd_var.shape() == master_pd_var.shape()
            return np.array(master_pd_var)
        else:
            assert pd_var._dtype() == paddle.float32
            return np.array(pd_var)

    def _enc_prefix(self, idx):
        return "bert/encoder/layer_{}/".format(idx)

    def _to_pd_var_name(self, var):
        return var if isinstance(var, (str, bytes)) else var.name

    def _record_pd_vars(self,
                        pd_vars,
                        tf_vars,
                        tf_var_prefix="",
                        weight_transpose=None):
        if not isinstance(pd_vars, (list, tuple)):
            pd_vars = [pd_vars]
        pd_vars = [self._to_pd_var_name(v) for v in pd_vars]

        if not isinstance(tf_vars, (list, tuple)):
            tf_vars = [tf_vars]
        tf_vars = [tf_var_prefix + v for v in tf_vars]

        assert len(pd_vars) == len(tf_vars)
        for pd_var, tf_var in zip(pd_vars, tf_vars):
            assert pd_var not in self.pd_vars_to_tf_vars, pd_var
            self.pd_vars_to_tf_vars[pd_var] = tf_var

        if weight_transpose:
            assert len(pd_vars) == 2
            self.transpose_vars.add(pd_vars[0])

    def embeddings(self, pd_vars):
        return self._record_pd_vars(pd_vars, [
            "word_embeddings", "position_embeddings", "token_type_embeddings"
        ], "bert/embeddings/")

    def norm_after_embeddings(self, pd_vars):
        return self._record_pd_vars(pd_vars, ["gamma", "beta"],
                                    "bert/embeddings/LayerNorm/")

    def enc_attn_query_fc(self, pd_vars, idx, weight_transpose=None):
        assert not self.fuse_attn_qkv
        prefix = self._enc_prefix(idx) + "attention/self/query/"
        return self._record_pd_vars(pd_vars, ["kernel", "bias"], prefix,
                                    weight_transpose)

    def enc_attn_key_fc(self, pd_vars, idx, weight_transpose=False):
        assert not self.fuse_attn_qkv
        prefix = self._enc_prefix(idx) + "attention/self/key/"
        return self._record_pd_vars(pd_vars, ["kernel", "bias"], prefix,
                                    weight_transpose)

    def enc_attn_value_fc(self, pd_vars, idx, weight_transpose=False):
        assert not self.fuse_attn_qkv
        prefix = self._enc_prefix(idx) + "attention/self/value/"
        return self._record_pd_vars(pd_vars, ["kernel", "bias"], prefix,
                                    weight_transpose)

    def enc_fused_attn_qkv_fc(self, pd_vars, idx, weight_transpose=False):
        assert self.fuse_attn_qkv
        assert self.attn_fused_qkv_weights[idx] is None
        assert self.attn_fused_qkv_biases[idx] is None
        weight, bias = pd_vars
        self.attn_fused_qkv_weights[idx] = self._to_pd_var_name(weight)
        self.attn_fused_qkv_biases[idx] = self._to_pd_var_name(bias)
        if weight_transpose:
            self.transpose_vars.add(self.attn_fused_qkv_weights[idx])

    def enc_attn_proj_fc(self, pd_vars, idx, weight_transpose=False):
        prefix = self._enc_prefix(idx) + "attention/output/dense/"
        return self._record_pd_vars(pd_vars, ["kernel", "bias"], prefix,
                                    weight_transpose)

    def enc_attn_norm(self, pd_vars, idx):
        prefix = self._enc_prefix(idx) + "attention/output/LayerNorm/"
        return self._record_pd_vars(pd_vars, ["gamma", "beta"], prefix)

    def enc_intermediate_fc(self, pd_vars, idx, weight_transpose=False):
        prefix = self._enc_prefix(idx) + "intermediate/dense/"
        return self._record_pd_vars(pd_vars, ["kernel", "bias"], prefix,
                                    weight_transpose)

    def enc_output_fc(self, pd_vars, idx, weight_transpose=False):
        prefix = self._enc_prefix(idx) + "output/dense/"
        return self._record_pd_vars(pd_vars, ["kernel", "bias"], prefix,
                                    weight_transpose)

    def enc_output_norm(self, pd_vars, idx):
        prefix = self._enc_prefix(idx) + "output/LayerNorm/"
        return self._record_pd_vars(pd_vars, ["gamma", "beta"], prefix)

    def pooler_fc(self, pd_vars, weight_transpose=False):
        return self._record_pd_vars(pd_vars, ["kernel", "bias"],
                                    "bert/pooler/dense/", weight_transpose)

    def cls_pred_trans_fc(self, pd_vars, weight_transpose=False):
        return self._record_pd_vars(pd_vars, ["kernel", "bias"],
                                    "cls/predictions/transform/dense/",
                                    weight_transpose)

    def cls_pred_trans_norm(self, pd_vars):
        return self._record_pd_vars(pd_vars, ["gamma", "beta"],
                                    "cls/predictions/transform/LayerNorm/")

    def cls_pred_fc_bias(self, pd_vars):
        return self._record_pd_vars(pd_vars, "output_bias", "cls/predictions/")

    def cls_seq_relation_fc(self, pd_vars, weight_transpose=False):
        return self._record_pd_vars(pd_vars, ["output_weights", "output_bias"],
                                    "cls/seq_relationship/", weight_transpose)


class FMHA(Layer):
    def __init__(self, config):
        super(FMHA, self).__init__()
        self.p_dropout = config.attention_probs_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        self.fused_qkv_bias = config.fused_bias_mha
        self.weight_transpose = True
        self.use_unpad_fmha_mke_opt = config.unpad_fmha_mke_opt
        assert self.d * self.h == self.hidden_size, "Invalid hidden size/num_heads"

        # create_parameter
        self._dtype = self._helper.get_default_dtype()

        if self.weight_transpose:
            Wqkv_shape = [3 * config.hidden_size, config.hidden_size]
        else:
            Wqkv_shape = [config.hidden_size, 3 * config.hidden_size]

        self.Wqkv = self.create_parameter(
            shape=Wqkv_shape, attr=None, dtype=self._dtype, is_bias=False)
        self.Bqkv = self.create_parameter(
            shape=[3 * config.hidden_size],
            attr=None,
            dtype=self._dtype,
            is_bias=True)

    def forward(self,
                hidden_states,
                cu_seqlens,
                host_cu_seqlens,
                max_s,
                is_training=True):
        #print("fmha layer, input shape: ", hidden_states.shape)
        #print("Wqkv.shape", self.Wqkv.shape)
        #print("Bqkv.shape", self.Bqkv.shape)
        if not self.fused_qkv_bias:
            qkv = paddle.matmul(
                hidden_states,
                self.Wqkv,
                transpose_x=False,
                transpose_y=self.weight_transpose)
            qkv = qkv + self.Bqkv
        else:
            qkv = fuse_dense(
                hidden_states,
                self.Wqkv,
                self.Bqkv,
                transx=False,
                transy=self.weight_transpose)

        qkv = paddle.reshape(qkv, [-1, 3, self.h, self.d])
        #print("qkv.shape", qkv.shape)
        # FMHA: max_s =  var memcpy_0.tmp_0 : LOD_TENSOR.shape(1,).dtype(int32).stop_gradient(False)
        #print("FMHA: max_s = ", max_s)
        out, _ = custom_fmha(
            qkv,
            cu_seqlens,
            host_cu_seqlens,
            not is_training,
            self.p_dropout,
            zero_tensors=False,
            use_fmha_mke_opt=self.use_unpad_fmha_mke_opt)
        return paddle.reshape(out, [-1, self.hidden_size])


class BertSelfAttention(Layer):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.weight_transpose = False

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = [0] * (
            len(x.shape) - 1
        ) + [self.num_attention_heads, self.attention_head_size]
        x = paddle.reshape(x, new_x_shape)
        return paddle.transpose(x, [0, 2, 1, 3])

    def transpose_key_for_scores(self, x):
        new_x_shape = [0] * (
            len(x.shape) - 1
        ) + [self.num_attention_heads, self.attention_head_size]
        x = paddle.reshape(x, new_x_shape)
        return paddle.transpose(x, [0, 2, 3, 1])

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = paddle.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        attention_scores = attention_scores + attention_mask.unsqueeze(
            1).unsqueeze(2)

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = paddle.transpose(context_layer, [0, 2, 1, 3])
        new_context_layer_shape = [0] * (len(context_layer.shape) - 2
                                         ) + [self.all_head_size]
        context_layer = paddle.reshape(context_layer, new_context_layer_shape)
        return context_layer


class BertAttention(Layer):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertLayer(Layer):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        assert use_nv_input
        assert not config.pad_fmha
        self.unpad = config.unpad

        if config.unpad_fmha:
            assert self.unpad
            self.attention = UnpadFMHABertAttention(config)
        else:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask,
                seqlen=None,
                host_seqlen=None,
                batch=None):
        if self.unpad:
            attention_output = self.attention(hidden_states, attention_mask,
                                              seqlen, host_seqlen, batch)
        else:
            attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(Layer):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        if use_nv_input:
            self.layers = nn.LayerList(
                [BertLayer(config) for _ in range(config.num_hidden_layers)])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation=config.hidden_act,
                attn_dropout=config.attention_probs_dropout_prob,
                act_dropout=0)
            self.encoder = nn.TransformerEncoder(encoder_layer,
                                                 config.num_hidden_layers)

        self.num_attention_heads = config.num_attention_heads
        self.unpad = config.unpad
        self.unpad_embed = config.unpad_embed
        self.unpad_fmha = config.unpad_fmha
        self.pad_fmha = config.pad_fmha
        self.hidden_size = config.hidden_size
        self.maxseqlen = config.max_seq_length

    def record_ckpt_vars(self, ckpt, idx):
        if use_nv_input:
            layer = self.layers[idx]
            attn = layer.attention
            if isinstance(attn, UnpadFMHABertAttention):
                ckpt.enc_fused_attn_qkv_fc([attn.fmha.Wqkv, attn.fmha.Bqkv],
                                           idx, attn.fmha.weight_transpose)
            else:
                ckpt.enc_attn_query_fc(
                    [attn.self.query.weight, attn.self.query.bias], idx,
                    attn.self.weight_transpose)
                ckpt.enc_attn_key_fc(
                    [attn.self.key.weight, attn.self.key.bias], idx,
                    attn.self.weight_transpose)
                ckpt.enc_attn_value_fc(
                    [attn.self.value.weight, attn.self.value.bias], idx,
                    attn.self.weight_transpose)

            ckpt.enc_attn_proj_fc(
                [attn.output.dense.weight, attn.output.dense.bias], idx,
                attn.output.weight_transpose)
            if attn.output.fused_dropout:
                ckpt.enc_attn_norm([
                    attn.output.fused_dropout_add_ln.weight,
                    attn.output.fused_dropout_add_ln.bias
                ], idx)
            else:
                ckpt.enc_attn_norm([
                    attn.output.layer_norm.weight, attn.output.layer_norm.bias
                ], idx)

            intermediate = layer.intermediate
            last_output = layer.output
            ckpt.enc_intermediate_fc(
                [intermediate.dense.weight, intermediate.dense.bias], idx,
                intermediate.weight_transpose)
            ckpt.enc_output_fc(
                [last_output.dense.weight, last_output.dense.bias], idx,
                last_output.weight_transpose)
            if last_output.fused_dropout:
                ckpt.enc_output_norm([
                    last_output.fused_dropout_add_ln.weight,
                    last_output.fused_dropout_add_ln.bias
                ], idx)
            else:
                ckpt.enc_output_norm([
                    last_output.layer_norm.weight, last_output.layer_norm.bias
                ], idx)
        else:
            layer = self.encoder.layers[idx]
            attn = layer.self_attn
            ckpt.enc_attn_query_fc([attn.q_proj.weight, attn.q_proj.bias], idx,
                                   False)
            ckpt.enc_attn_key_fc([attn.k_proj.weight, attn.k_proj.bias], idx,
                                 False)
            ckpt.enc_attn_value_fc([attn.v_proj.weight, attn.v_proj.bias], idx,
                                   False)

            ckpt.enc_attn_proj_fc([attn.out_proj.weight, attn.out_proj.bias],
                                  idx, False)
            ckpt.enc_attn_norm([layer.norm1.weight, layer.norm1.bias], idx)
            ckpt.enc_intermediate_fc(
                [layer.linear1.weight, layer.linear1.bias], idx, False)
            ckpt.enc_output_fc([layer.linear2.weight, layer.linear2.bias], idx,
                               False)
            ckpt.enc_output_norm([layer.norm2.weight, layer.norm2.bias], idx)

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=False,
                batch=56,
                maxseqlen=512,
                hidden_size=1024,
                zero_tensor=None,
                attention_indices=None,
                new_attention_mask=None,
                seqlen=None,
                cu_seqlens=None,
                host_cu_seqlens=None,
                actual_seqlens=None,
                maxseqlen_in_batch=None):

        if use_nv_input:
            return self.forward_with_nv_input(
                hidden_states, attention_mask, output_all_encoded_layers, batch,
                maxseqlen, hidden_size, zero_tensor, attention_indices,
                new_attention_mask, seqlen, cu_seqlens, host_cu_seqlens,
                actual_seqlens, maxseqlen_in_batch)

        if output_all_encoded_layers:
            output = hidden_states
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            return encoder_outputs
        else:
            sequence_output = self.encoder(hidden_states, attention_mask)
            return [sequence_output]

    def forward_with_nv_input(self,
                              hidden_states,
                              attention_mask,
                              output_all_encoded_layers=False,
                              batch=56,
                              maxseqlen=512,
                              hidden_size=1024,
                              zero_tensor=None,
                              attention_indices=None,
                              new_attention_mask=None,
                              seqlen=None,
                              cu_seqlens=None,
                              host_cu_seqlens=None,
                              actual_seqlens=None,
                              maxseqlen_in_batch=None):
        # Unpad inputs and mask. It will remove tokens that are padded. Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens. Then unpadding performs the following compression of the inputs:
        #        hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]

        if not self.unpad_embed and self.unpad_fmha:
            #batch = None
            #seqlen = None
            if self.unpad_fmha:
                batch = hidden_states.shape[0]
                maxseqlen = hidden_states.shape[1]
                assert maxseqlen == self.maxseqlen
                hidden_size = hidden_states.shape[2]
                assert hidden_size == self.hidden_size

                zero_tensor = paddle.zeros_like(hidden_states)
                zero_tensor = paddle.reshape(zero_tensor,
                                             [-1, self.hidden_size])

                # attention_indices: 把attn_mask flatten后，提取非零元的下标。
                # seqlen： 对[bs, max_seq_len]，每一行求和，代表获取每一行的实际seq_len（一维）。
                # ntokens: 代表当前batch的所有实际seq_len之和。
                # cu_seqlens: 对seq_len求prefix_sum的结果。
                # actual_seqlens: 与seqlen相同的值。
                # maxseqlen_in_batch: the max seqlen in a batch.
                #attention_indices, attention_mask, seqlen, ntokens, cu_seqlens, actual_seqlens, maxseqlen_in_batch = generate_mask(
                #    attention_mask, unpad_fmha=self.unpad_fmha)
                print("maxseqlen_in_batch = ", maxseqlen_in_batch)
                hidden_states = paddle.reshape(hidden_states,
                                               [-1, self.hidden_size])
                hidden_states = paddle.gather(hidden_states, attention_indices)
                # print("unpad after shape: ", hidden_states)
        elif self.unpad_fmha:
            attention_mask = new_attention_mask

        all_encoder_layers = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layers[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        for i, layer_module in enumerate(self.layers):
            if seqlen is None and batch is None:
                hidden_states = layer_module(hidden_states, attention_mask)
            else:
                assert seqlen is not None
                assert batch is not None
                if self.unpad_fmha:
                    hidden_states = layer_module(hidden_states, cu_seqlens,
                                                 host_cu_seqlens,
                                                 maxseqlen_in_batch)
                    print("hidden_states:", hidden_states)
                else:
                    hidden_states = layer_module(hidden_states, attention_mask,
                                                 seqlen, batch)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # Pad inputs and mask. It will insert back zero-padded tokens. Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens. Then padding performs the following de-compression:
        #        hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
        if self.unpad_fmha:
            hidden_states = paddle.scatter(zero_tensor, attention_indices,
                                           hidden_states)
            # todo: is self.maxseqlen same as maxseqlen?
            hidden_states = paddle.reshape(
                hidden_states, [batch, self.maxseqlen, self.hidden_size])
            #print("hidden_states.shape:", hidden_states.shape)
            #print("hidden_states:", hidden_states)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UnpadFMHABertAttention(Layer):
    def __init__(self, config):
        super(UnpadFMHABertAttention, self).__init__()
        self.fmha = FMHA(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                input_tensor,
                cu_seqlens,
                host_cu_seqlens,
                max_s,
                batch_size=None):
        self_output = self.fmha(
            input_tensor,
            cu_seqlens,
            host_cu_seqlens,
            max_s,
            is_training=self.training)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class FusedDropoutResidualLn(Layer):
    def __init__(self, config, normalized_shape, epsilon=1e-12):
        super(FusedDropoutResidualLn, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]
        self._normalized_shape = list(normalized_shape)
        param_shape = [np.prod(self._normalized_shape)]

        self._weight_attr = None
        self._bias_attr = None
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=param_shape,
            default_initializer=Constant(1.0))
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=param_shape, is_bias=True)

        self.p = config.hidden_dropout_prob
        self.epsilon = epsilon
        self.is_test = not self.training
        # todo: use default configs.
        self.fix_seed = False
        self.is_upscale_in_train = True
        self.seed_val = 0

    def forward(self, hidden_states, input_tensor):
        out, dropout_mask, ln_mean, ln_var, dropout_residual_out = custom_fused_dropout_residual_ln(
            hidden_states, input_tensor, self.weight, self.bias, self.epsilon,
            self.is_test, self.fix_seed, self.seed_val,
            self.is_upscale_in_train, self.p)
        return out

    def extra_repr(self):
        return 'normalized_shape={}, epsilon={}'.format(self._normalized_shape,
                                                        self.epsilon)


# support nn(weight is not transposed) and nt(weight is transposed)
class FusedDense(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_transpose=False,
                 weight_attr=None,
                 bias_attr=None,
                 with_gelu=False,
                 name=None):
        super(FusedDense, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight_transpose = weight_transpose
        if weight_transpose:
            self.weight = self.create_parameter(
                shape=[out_features, in_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)
        else:
            self.weight = self.create_parameter(
                shape=[in_features, out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.with_gelu = with_gelu
        self.name = name

    def forward(self, hidden_states):
        out = fuse_dense(
            hidden_states,
            self.weight,
            self.bias,
            transx=False,
            transy=self.weight_transpose,
            with_gelu=self.with_gelu)
        return out


class BertSelfOutput(Layer):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.fused_fc_bias = config.fused_bias_fc
        self.fused_dropout = config.fused_dropout_add_ln
        if not self.fused_fc_bias:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.weight_transpose = False
        else:
            self.dense = FusedDense(
                config.hidden_size,
                config.hidden_size,
                weight_transpose=config.weight_transpose)
            self.weight_transpose = config.weight_transpose
        if self.fused_dropout:
            self.fused_dropout_add_ln = FusedDropoutResidualLn(
                config, config.hidden_size, epsilon=1e-12)
        else:
            self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        #print("selfoutput input: hidden_states.shape = ", hidden_states.shape)
        hidden_states = self.dense(hidden_states)
        if not self.fused_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + input_tensor
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = self.fused_dropout_add_ln(hidden_states,
                                                      input_tensor)
        return hidden_states


class BertIntermediate(Layer):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.fused_fc_bias = config.fused_bias_fc
        if not self.fused_fc_bias:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
            self.weight_transpose = False
            self.intermediate_act_fn = get_activation()
        else:
            self.weight_transpose = config.weight_transpose
            self.dense = FusedDense(
                config.hidden_size,
                config.intermediate_size,
                weight_transpose=self.weight_transpose,
                with_gelu=True)

    def forward(self, hidden_states):
        if not self.fused_fc_bias:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)
        else:
            hidden_states = self.dense(hidden_states)
        return hidden_states


class BertOutput(Layer):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.fused_fc_bias = config.fused_bias_fc
        self.fused_dropout = config.fused_dropout_add_ln
        if not self.fused_fc_bias:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.weight_transpose = False
        else:
            self.dense = FusedDense(
                config.intermediate_size,
                config.hidden_size,
                weight_transpose=config.weight_transpose)
            self.weight_transpose = config.weight_transpose
        if self.fused_dropout:
            self.fused_dropout_add_ln = FusedDropoutResidualLn(
                config, config.hidden_size, epsilon=1e-12)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)
        # todo: add fused_dropout opt.
        self.p = config.hidden_dropout_prob

    def forward(self, hidden_states, input_tensor):
        #print("BertOutput, input.shape = ", hidden_states.shape)
        hidden_states = self.dense(hidden_states)
        if not self.fused_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + input_tensor
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = self.fused_dropout_add_ln(hidden_states,
                                                      input_tensor)
        return hidden_states


class BertEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 config=None):
        super(BertEmbeddings, self).__init__()
        self.unpad_embed = False
        if config is not None:
            self.unpad_embed = config.unpad_embed
            self.unpad_fmha = config.unpad_fmha
            if self.unpad_embed:
                self.batch_size = config.batch_size
                self.max_seq_length = config.max_seq_length
        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                attention_indices=None,
                seqlen=None,
                cu_seqlens=None,
                actual_seqlens=None,
                maxseqlen_in_batch=None):
        if position_ids is None:
            position_ids = gen_pos_id(input_ids)
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        # todo(@limin29): in order to construct the shape of zero_tensor, we use pad method to compute token_type_embeddings.
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if self.unpad_embed:
            assert self.unpad_fmha
            assert attention_mask is not None
            assert attention_indices is not None
            assert seqlen is not None
            assert cu_seqlens is not None
            assert maxseqlen_in_batch is not None

            cur_batch_size = input_ids.shape[0]

            zero_tensor = paddle.zeros_like(token_type_embeddings)
            zero_tensor = paddle.reshape(zero_tensor, [-1, self.hidden_size])

            # attention_indices: 把attn_mask flatten后，提取非零元的下标。
            # seqlen： 对[bs, max_seq_len]，每一行求和，代表获取每一行的实际seq_len（一维）。
            # ntokens: 代表当前batch的所有实际seq_len之和。
            # cu_seqlens: 对seq_len求prefix_sum的结果。
            # actual_seqlens: 与seqlen相同的值。
            # maxseqlen_in_batch: the max seqlen in a batch.
            #attention_indices, attention_mask, seqlen, ntokens, cu_seqlens, actual_seqlens, maxseqlen_in_batch = generate_mask(
            #    attention_mask, unpad_fmha=self.unpad_fmha)
            print("maxseqlen_in_batch = ", maxseqlen_in_batch)

            input_ids = paddle.reshape(input_ids, [-1])
            input_ids = paddle.gather(input_ids, attention_indices)

            position_ids = paddle.reshape(position_ids, [-1])
            position_ids = paddle.gather(position_ids, attention_indices)

            #token_type_ids = paddle.reshape(token_type_ids, [-1])
            #token_type_ids = paddle.gather(token_type_ids, attention_indices)
            token_type_embeddings = paddle.reshape(token_type_embeddings,
                                                   [-1, self.hidden_size])
            token_type_embeddings = paddle.gather(token_type_embeddings,
                                                  attention_indices)

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        if not self.unpad_embed:
            return embeddings
        else:
            return embeddings, cur_batch_size, zero_tensor


class BertPooler(Layer):
    """
    Pool the result of BertEncoder.
    """

    def __init__(self, hidden_size, pool_act="tanh"):
        super(BertPooler, self).__init__()
        #self.dense = nn.Linear(hidden_size, hidden_size)
        #self.weight_transpose = False
        self.dense = FusedDense(hidden_size, hidden_size)
        self.weight_transpose = self.dense.weight_transpose
        self.activation = nn.Tanh()
        self.pool_act = pool_act
        assert self.pool_act == "tanh"

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Layer):
    """
    The bare BERT Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `BertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `BertModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `16`.

        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to 0.02.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

        pool_act (str, optional):
            The non-linear activation function in the pooling layer.
            Defaults to `"tanh"`.

    """

    def __init__(self, config):
        super(BertModel, self).__init__()
        self.unpad = config.unpad
        self.pad_fmha = config.pad_fmha
        self.unpad_embed = config.unpad_embed
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.num_hidden_layers = config.num_hidden_layers
        self.maxseqlen = config.max_seq_length
        self.hidden_size = config.hidden_size

        # todo: 
        self.embeddings = BertEmbeddings(
            config.vocab_size, config.hidden_size, config.hidden_dropout_prob,
            config.max_position_embeddings, config.type_vocab_size, config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config.hidden_size, config.pool_act)

    def record_ckpt_vars(self, ckpt):
        emb = self.embeddings
        ckpt.embeddings([
            emb.word_embeddings.weight,
            emb.position_embeddings.weight,
            emb.token_type_embeddings.weight,
        ])

        ckpt.norm_after_embeddings([
            emb.layer_norm.weight,
            emb.layer_norm.bias,
        ])

        for idx in range(self.num_hidden_layers):
            self.encoder.record_ckpt_vars(ckpt, idx)

        pooler = self.pooler
        ckpt.pooler_fc([pooler.dense.weight, pooler.dense.bias],
                       pooler.weight_transpose)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                seq_len=None,
                prefix_sum_seq_len=None,
                host_prefix_sum_seq_len=None,
                max_seq_len=None,
                nonzeros_indices=None,
                output_hidden_states=False):
        if use_nv_input:
            if attention_mask is None:
                attention_mask = paddle.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = paddle.zeros_like(input_ids)

            extended_attention_mask = attention_mask
            if not self.unpad and not self.pad_fmha:
                extended_attention_mask = extended_attention_mask.cast(
                    dtype=paddle.float32)
                extended_attention_mask = (
                    1.0 - extended_attention_mask) * -10000.0
            attention_mask = extended_attention_mask
        else:
            if attention_mask is None:
                attention_mask = paddle.unsqueeze(
                    (input_ids == self.pad_token_id
                     ).astype(self.pooler.dense.weight.dtype) * -1e9,
                    axis=[1, 2])

        new_attention_mask = attention_mask
        attention_indices = nonzeros_indices
        seqlen = seq_len
        cu_seqlens = prefix_sum_seq_len
        host_cu_seqlens = host_prefix_sum_seq_len
        maxseqlen_in_batch = max_seq_len
        actual_seqlens = seqlen

        if self.unpad_embed or self.unpad_fmha:
            assert attention_indices is not None
            assert seqlen is not None
            assert cu_seqlens is not None
            assert host_cu_seqlens is not None
            assert maxseqlen_in_batch is not None

        if not self.unpad_embed:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)
            cur_batch_size = None
            zero_tensor = None
            attention_indices = None
            new_attention_mask = None
            seqlen = None
            cu_seqlens = None
            host_cu_seqlens = None
            actual_seqlens = None
            maxseqlen_in_batch = None
        else:
            embedding_output, cur_batch_size, zero_tensor = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                attention_indices=attention_indices,
                seqlen=seqlen,
                cu_seqlens=cu_seqlens,
                actual_seqlens=seqlen,
                maxseqlen_in_batch=maxseqlen_in_batch)

        encoder_outputs = self.encoder(
            embedding_output, attention_mask, output_hidden_states,
            cur_batch_size, self.maxseqlen, self.hidden_size, zero_tensor,
            attention_indices, new_attention_mask, seqlen, cu_seqlens,
            host_cu_seqlens, actual_seqlens, maxseqlen_in_batch)
        pooled_output = self.pooler(encoder_outputs[-1])
        print("encoder_outputs[-1]:", encoder_outputs[-1])
        print("pooled_output:", pooled_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return encoder_outputs[-1], pooled_output


class BertLMPredictionHead(Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, config, embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = FusedDense(
            config.hidden_size, config.hidden_size, with_gelu=True)
        self.weight_transpose = self.transform.weight_transpose
        assert config.hidden_act == "gelu"

        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)
        self.decoder_weight = self.create_parameter(
            shape=[config.vocab_size, config.hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size],
            dtype=self.decoder_weight.dtype,
            is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = fuse_dense(
            hidden_states,
            self.decoder_weight,
            self.decoder_bias,
            transx=False,
            transy=True)
        return hidden_states


class BertPretrainingHeads(Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        hidden_size (int):
            See :class:`BertModel`.
        vocab_size (int):
            See :class:`BertModel`.
        activation (str):
            Activation function used in the language modeling task.
        embedding_weights (Tensor, optional):
            Decoding weights used to map hidden_states to logits of the masked token prediction.
            Its data type should be float32 and its shape is [vocab_size, hidden_size].
            Defaults to `None`, which means use the same weights of the embedding layer.

    """

    def __init__(self, config, embedding_weights=None):
        super(BertPretrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, embedding_weights)
        #self.seq_relationship = nn.Linear(config.hidden_size, 2)
        #self.seq_relationship_weight_transpose = False
        self.seq_relationship = FusedDense(config.hidden_size, 2)
        self.seq_relationship_weight_transpose = self.seq_relationship.weight_transpose
        self.dense_seq_output = config.dense_seq_output
        self.share_weight = embedding_weights is not None

    def record_ckpt_vars(self, ckpt):
        pred_trans = self.predictions.transform
        ckpt.cls_pred_trans_fc([pred_trans.weight, pred_trans.bias],
                               self.predictions.weight_transpose)

        norm = self.predictions.layer_norm
        ckpt.cls_pred_trans_norm([norm.weight, norm.bias])

        assert self.share_weight
        ckpt.cls_pred_fc_bias(self.predictions.decoder_bias)

        seq_relation_fc = self.seq_relationship
        ckpt.cls_seq_relation_fc(
            [seq_relation_fc.weight, seq_relation_fc.bias],
            self.seq_relationship_weight_transpose)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        """
        Args:
            sequence_output(Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            pooled_output(Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
            masked_positions(Tensor, optional):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

    def nv_forward(self,
                   sequence_output,
                   pooled_output,
                   masked_lm_labels,
                   num_valid=None,
                   masked_lm_ids=None,
                   masked_lm_positions=None):
        print("sequence_output: ", sequence_output)
        if self.dense_seq_output:
            # nonzero indices
            index = masked_lm_positions
            sequence_flattened = paddle.index_select(
                sequence_output.reshape((-1, sequence_output.shape[-1])),
                index=index,
                axis=0)
            sequence_output = sequence_flattened
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertForPretraining(nn.Layer):
    """
    Bert Model with pretraining tasks on top.

    Args:
        bert (:class:`BertModel`):
            An instance of :class:`BertModel`.

    """

    def __init__(self, bert, config):
        super(BertForPretraining, self).__init__()
        self.config = config
        self.bert = bert
        self.cls = BertPretrainingHeads(
            config,
            embedding_weights=self.bert.embeddings.word_embeddings.weight)

    def load_tf_ckpt(self, args, get_parameter_func):
        place = utility.get_place()
        ckpt = TFCkptHelper(args, self.config, args.tf_ckpt_path, place)
        self.bert.record_ckpt_vars(ckpt)
        self.cls.record_ckpt_vars(ckpt)
        ckpt.load(get_parameter_func)
        return ckpt

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                seq_len=None,
                prefix_sum_seq_len=None,
                host_prefix_sum_seq_len=None,
                max_seq_len=None,
                nonzeros_indices=None,
                num_valid=None,
                masked_lm_ids=None,
                masked_lm_positions=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids (Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.
            masked_positions(Tensor, optional):
                See :class:`BertPretrainingHeads`.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            seq_len=seq_len,
            prefix_sum_seq_len=prefix_sum_seq_len,
            host_prefix_sum_seq_len=host_prefix_sum_seq_len,
            max_seq_len=max_seq_len,
            nonzeros_indices=nonzeros_indices)

        sequence_output, pooled_output = outputs
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_positions, num_valid,
            masked_lm_ids, masked_lm_positions)
        return prediction_scores, seq_relationship_score


class BertPretrainingCriterion(paddle.nn.Layer):
    """

    Args:
        vocab_size(int):
            Vocabulary size of `inputs_ids` in `BertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `BertModel`.

    """

    def __init__(self, config):
        super(BertPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = config.vocab_size
        self.dense_seq_output = config.dense_seq_output

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale):
        """
        Args:
            prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size]
            seq_relationship_score(Tensor):
                The scores of next sentence prediction. Its data type should be float32 and
                its shape is [batch_size, 2]
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None, its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1]
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]
            masked_lm_scale(Tensor or int):
                The scale of masked tokens. Used for the normalization of masked language modeling loss.
                If it is a `Tensor`, its data type should be int64 and its shape is equal to `prediction_scores`.

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].


        """
        masked_lm_loss = F.cross_entropy(
            prediction_scores,
            masked_lm_labels,
            reduction='none',
            ignore_index=-1)
        masked_lm_loss = masked_lm_loss / masked_lm_scale
        next_sentence_loss = F.cross_entropy(
            seq_relationship_score, next_sentence_labels, reduction='none')
        valid_mask = masked_lm_labels != -1
        total_loss_before_cast = paddle.sum(masked_lm_loss) + paddle.mean(
            next_sentence_loss)

        def func():
            total_loss = total_loss_before_cast.astype('float32')
            mlm_acc = paddle.cast(
                paddle.sum((paddle.argmax(
                    prediction_scores, axis=-1, keepdim=True) ==
                            masked_lm_labels) * valid_mask),
                dtype=masked_lm_scale.dtype) / masked_lm_scale
            return total_loss, mlm_acc, masked_lm_scale

        return func

    def nv_forward(self,
                   prediction_scores,
                   seq_relationship_score,
                   masked_lm_labels,
                   next_sentence_labels,
                   num_valid=None,
                   masked_lm_ids=None,
                   masked_lm_positions=None):
        if self.dense_seq_output:
            loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        else:
            loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=0)
        if self.dense_seq_output:
            masked_lm_labels_dense = masked_lm_ids
            print("masked_lm_labels_dense = ", masked_lm_labels_dense)
            masked_lm_loss = loss_fct(
                prediction_scores.reshape((-1, self.vocab_size)),
                masked_lm_labels_dense)
            #with paddle.static.device_guard('cpu'):
            #    num_valid = paddle.numel(masked_lm_labels_dense)
        else:
            masked_lm_loss = loss_fct(
                prediction_scores.reshape((-1, self.vocab_size)),
                masked_lm_labels.reshape((-1, )))
        nsp_loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        next_sentence_loss = nsp_loss_fct(
            seq_relationship_score.reshape([-1, 2]),
            next_sentence_labels.reshape([-1]))
        total_loss = masked_lm_loss + next_sentence_loss

        # Masked Language Model Accuracy
        # NOTE: total_loss and mlm_acc use float32 in NV
        if not self.dense_seq_output:

            def func():
                valid_mask = masked_lm_labels_flat != 0
                num_valid_cnt = valid_mask.astype('int32').sum(dtype='float32')
                mlm_labels = masked_lm_labels_flat
                prediction_scores_flat = prediction_scores.reshape(
                    (-1, prediction_scores.shape[-1]))
                mlm_predictions_scores = prediction_scores_flat
                mlm_predictions = mlm_predictions_scores.argmax(axis=-1)
                mlm_acc = ((mlm_predictions == mlm_labels) *
                           valid_mask).sum(dtype='float32') / num_valid_cnt
                return total_loss, mlm_acc, num_valid_cnt
        else:
            mlm_labels = masked_lm_ids
            dtype = masked_lm_labels_dense.dtype
            if dtype == paddle.int32:
                dtype = 'int32'
            elif dtype == paddle.int64:
                dtype = 'int64'
            else:
                assert False
            mlm_predictions = prediction_scores.argmax(
                axis=-1, dtype=dtype, keepdim=False)
            assert len(mlm_predictions.shape) == 1
            num_valid_cnt = num_valid

            def func():
                mlm_acc = paddle.cast(mlm_predictions == masked_lm_labels_dense,
                                      'float32').mean()
                return total_loss, mlm_acc, num_valid_cnt

        return func


if use_nv_input:
    BertPretrainingHeads.forward = BertPretrainingHeads.nv_forward
    BertPretrainingCriterion.forward = BertPretrainingCriterion.nv_forward
