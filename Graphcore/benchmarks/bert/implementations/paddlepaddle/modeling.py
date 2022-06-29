# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import paddle
import paddle.nn as nn
from paddle.nn import Layer
import paddle.static

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S %a')

__all__ = [
    'BertEmbeddings', 'BertModel', 'BertPretrainingMLM', "BertPretrainingCLS",
    'BertPretrainingNSP'
]


class CustomNameScope(object):
    def __init__(self, name="", parent=None):
        self._children = dict()
        self._name = name
        self._parent = parent

    def child(self, prefix):
        if prefix not in self._children:
            new_child = paddle.fluid.framework.NameScope(prefix, self)
            self._children[prefix] = [new_child]
        else:
            # new_child = NameScope(prefix + "_%d" % len(self._children[prefix]),
            #                       self)
            # NOTE: not append post_fix
            new_child = paddle.fluid.framework.NameScope(prefix, self)
            self._children[prefix].append(new_child)
        return new_child

    def parent(self):
        return self._parent

    def name(self):
        return self._name


# monkey patch
paddle.fluid.framework.NameScope = CustomNameScope


class BertEmbeddings(Layer):
    """
    The definition of the embeddings from word, position and token_type.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `input_ids`. Also is the vocab size of token embedding matrix.
            Defaults to `30912`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer. Defaults to `128`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `2`.
        custom_ops (Module, optional):
            The callable python module contains all CustomOp Layer APIs. Defaults to `None`.
    """

    def __init__(self,
                 vocab_size=30912,
                 hidden_size=128,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 custom_ops=None):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings_weights = paddle.static.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype="float32",
            name="Embedding/Embedding_Dict")
        self.token_embeddings_weights = paddle.static.create_parameter(
            shape=[type_vocab_size, hidden_size],
            dtype="float32",
            name="Embedding/Segment_Dict")
        self.position_embeddings_weights = paddle.static.create_parameter(
            shape=[max_position_embeddings, hidden_size],
            dtype="float32",
            name="Embedding/Positional_Dict")
        layer_norm_weights = paddle.ParamAttr(name="Embedding/Gamma")
        layer_norm_biases = paddle.ParamAttr(name="Embedding/Beta")
        self.layer_norm = nn.LayerNorm(
            hidden_size,
            epsilon=0.001,
            weight_attr=layer_norm_weights,
            bias_attr=layer_norm_biases)

        self.hidden_dropout_prob = hidden_dropout_prob
        self.custom_ops = custom_ops
        self.mask_threshlod_attr = {
            'name': 'mask_threshold',
            'shape': [1],
            'dtype': 'int32',
            'value': 0,
        }

    def forward(self, input_ids, token_type_ids, position_ids, input_mask,
                is_training):
        '''
        The BertEmbeddings forward method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size * sequence_length].
            token_type_ids (Tensor):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size * sequence_length].
            position_ids (Tensor):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Its data type should be `int64` and it has a shape of [batch_size * sequence_length].
            input_mask (Tensor):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type should be `int32` and it has a shape of [batch_size * sequence_length].
            is_training (Tensor):
                Switch the mode of `Dropout` in runtime. It's used to enable or disable `Dropout` for fast validation in training.
                Its data type should be `int32` and it has a shape of [1].

                - 0 corresponds to enable `Dropout`,
                - 1 corresponds to disable `Dropout`.

        Returns:
            tuple: Returns tuple (`embeddings`, `word_embedding`).

            With the fields:

            - `embeddings` (Tensor):
                It's data type should be float32 or float16 and its shape is [batch_size * sequence_length, hidden_size].
            - `word_embedding` (Tensor):
                It's data type should be float32 or float16 and its shape is [hidden_size, vocab_size].
        '''

        logging.info("Emb Layer - ipu_index:%d, ipu_stage:%d" % (0, 0))
        with paddle.static.amp.fp16_guard():
            with paddle.static.ipu_shard_guard(index=0, stage=0):
                with paddle.static.name_scope("Embedding"):
                    # word embeddings
                    word_embeddings_weights = paddle.transpose(
                        self.word_embeddings_weights, [1, 0])
                    word_embeddings = paddle.gather(
                        word_embeddings_weights, input_ids, axis=0)

                    # position_embeddings
                    position_embeddings = paddle.gather(
                        self.position_embeddings_weights, position_ids, axis=0)

                    # input mask
                    mask_threshold = paddle.fluid.layers.fill_constant(
                        **self.mask_threshlod_attr)
                    remask = paddle.greater_than(input_mask, mask_threshold)
                    remask = paddle.reshape(remask, [-1, 1])

                    remask = paddle.cast(remask, "float32")
                    remask = self.custom_ops.custom_Detach(remask, 1)

                    # token_type_embeddings
                    token_type_embeddings = paddle.fluid.input.one_hot(
                        token_type_ids, depth=2)
                    token_type_embeddings = paddle.matmul(
                        token_type_embeddings, self.token_embeddings_weights)

                    embeddings = paddle.add(word_embeddings,
                                            position_embeddings)
                    embeddings = paddle.add(embeddings, token_type_embeddings)
                    embeddings = paddle.fluid.layers.elementwise_mul(embeddings,
                                                                     remask)
                    embeddings = self.layer_norm(embeddings)
                    embeddings = self.custom_ops.custom_DropoutWithTrainingSwitch(
                        embeddings, is_training, ratio=self.hidden_dropout_prob)

        return embeddings, self.word_embeddings_weights


class BertModel(Layer):
    """
    The definition of the transformer encoder layer.

    Args:
        attn_ipu_index (list):
            The indexs of attention layers used for IPU graph sharding.
            Its data type should be `int`.
        ff_ipu_index (list):
            The indexs of ff layers used for IPU graph sharding.
            Its data type should be `int`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer. Defaults to `1024`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `24`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        sequence_length (int, optional):
            The max length of sequence. Defaults to `512`.
        num_ipus (int, optional):
            The number of IPUs. Defaults to `4`.
        split_qkv (bool, optional):
            Split the query, key and value as different tensor. Defaults to `False`.
        activation_checkpoint_dtype (str, optional):
            The data type of the checkoutputpoint Op. Defaults to `FLOAT16`.
        available_memory (float, optional):
            Set the percentage of memory used for the matmul Op. Default to `1.0`.
        no_attn_dropout (bool, optional):
            Disable dropout Ops in attention layers. Default to `True`.
        custom_ops (Module, optional):
            The callable python module contains all CustomOp Layer APIs. Defaults to `None`.
    """

    def __init__(self,
                 attn_ipu_index,
                 ff_ipu_index,
                 hidden_size=1024,
                 num_hidden_layers=24,
                 attention_probs_dropout_prob=0.1,
                 sequence_length=512,
                 num_ipus=4,
                 split_qkv=True,
                 activation_checkpoint_dtype="FLOAT16",
                 available_memory=1.0,
                 no_attn_dropout=True,
                 custom_ops=None):
        super(BertModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.sequence_length = sequence_length
        self.attention_heads = 16
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.qkv_length = self.hidden_size // self.attention_heads
        self.split_qkv = split_qkv
        self.activation_checkpoint_dtype = activation_checkpoint_dtype
        self.available_memory = available_memory
        self.no_attn_dropout = no_attn_dropout

        # sharding and pipelining
        self.attn_ipu_index = attn_ipu_index
        self.ff_ipu_index = ff_ipu_index
        self.num_ipus = num_ipus

        # enable custom op
        self.custom_ops = custom_ops

        self.qk_scale_attrs = {
            'name': 'QK_scale',
            'shape': [1],
            'dtype': 'float32',
            'value': 0.125,
        }
        self.comb_shape = [
            -1, self.sequence_length, self.attention_heads, self.qkv_length
        ]

    def need_checkpoint_att(self, att_index):
        '''
        Whether the attention layers need a `CheckpointOutput` Op.
        '''
        att_stage = self.attn_ipu_index[att_index] if self.attn_ipu_index[
            att_index] > 0 else max(self.attn_ipu_index) + 1
        if att_stage == max(self.attn_ipu_index) + 1:
            return True
        return False

    def need_checkpoint_ffn(self, ffn_index):
        '''
        Whether the ff layers need a `CheckpointOutput` Op.
        '''
        if ffn_index == self.num_hidden_layers - 1:
            return True
        last_ffn_on_this_ipu = self.ff_ipu_index[
            ffn_index] != self.ff_ipu_index[ffn_index + 1]
        last_but_followed_by_att = self.ff_ipu_index[
            ffn_index] == self.attn_ipu_index[ffn_index + 1]
        if not last_ffn_on_this_ipu or last_but_followed_by_att:
            return True

    def forward(self, embedding_output, input_mask, is_training):
        '''
        The BertModel forward method.

        Args:
            embedding_output (Tensor):
                The output from Embedding Layer.
                Its data type should be `float32` or `float16`.
            input_mask (Tensor):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type should be `int32` and it has a shape of [batch_size * sequence_length].
            is_training (Tensor):
                Switch the mode of `Dropout` in runtime. It's used to enable or disable `Dropout` for fast validation in training.
                Its data type should be `int32` and it has a shape of [1].

        Returns:
            tuple: Returns Tensor `sequence_output`.

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size * sequence_length, hidden_size].
        '''
        with paddle.static.amp.fp16_guard():
            with paddle.static.ipu_shard_guard(index=1, stage=1):
                with paddle.static.name_scope("Layer0"):
                    self.input_mask = self.custom_ops.custom_Detach(input_mask,
                                                                    1)
            sequence_output = embedding_output
            for i in range(self.num_hidden_layers):
                # Attention
                att_index = self.attn_ipu_index[i]
                att_stage = self.attn_ipu_index[i] if self.attn_ipu_index[
                    i] > 0 else max(self.attn_ipu_index) + 1
                logging.info("Attention %d - ipu_index:%d, ipu_stage:%d" %
                             (i, att_index, att_stage))

                with paddle.static.ipu_shard_guard(
                        index=att_index, stage=att_stage):
                    with paddle.static.name_scope(f"Layer{i}/Attention"):
                        layer_input = sequence_output

                        # QKV(query, key, value)
                        if self.split_qkv:
                            q_weights = paddle.static.create_parameter(
                                shape=[self.hidden_size, self.hidden_size],
                                dtype="float32",
                                name="Layer" + str(i) + "/Attention/Q")
                            k_weights = paddle.static.create_parameter(
                                shape=[self.hidden_size, self.hidden_size],
                                dtype="float32",
                                name="Layer" + str(i) + "/Attention/K")
                            v_weights = paddle.static.create_parameter(
                                shape=[self.hidden_size, self.hidden_size],
                                dtype="float32",
                                name="Layer" + str(i) + "/Attention/V")
                            qkv_weights = paddle.concat(
                                [q_weights, k_weights, v_weights], axis=1)
                        else:
                            qkv_weights = paddle.static.create_parameter(
                                shape=[self.hidden_size, 3 * self.hidden_size],
                                dtype="float32",
                                name="Layer" + str(i) + "/Attention/QKV")
                        qkv_biases = paddle.static.create_parameter(
                            shape=[3 * self.hidden_size],
                            dtype="float32",
                            is_bias=True,
                            name="Layer" + str(i) + "/Attention/QKV_bias")
                        qkv = paddle.matmul(sequence_output, qkv_weights)
                        qkv.block.ops[-1]._set_attr('__available_memory',
                                                    self.available_memory)
                        qkv = paddle.add(qkv, qkv_biases)

                        q, k, v = paddle.split(
                            qkv, num_or_sections=[self.hidden_size] * 3, axis=1)

                        q = paddle.reshape(q, self.comb_shape)
                        q = paddle.transpose(q, [0, 2, 1, 3])

                        k = paddle.reshape(k, self.comb_shape)
                        k = paddle.transpose(k, [0, 2, 3, 1])

                        v = paddle.reshape(v, self.comb_shape)
                        v = paddle.transpose(v, [0, 2, 1, 3])

                        # Attention calculation
                        with paddle.static.name_scope(f"Z"):
                            qk = paddle.matmul(q, k)
                            qk.block.ops[-1]._set_attr('__available_memory',
                                                       self.available_memory)

                            with paddle.static.name_scope(f"MUL"):
                                qk_scale = paddle.fluid.layers.fill_constant(
                                    **self.qk_scale_attrs)
                                qk_scale = paddle.fluid.layers.elementwise_mul(
                                    qk, qk_scale)

                            qk = self.custom_ops.custom_AttentionMask(
                                self.input_mask, qk_scale, "FLOAT16")

                            qk = self.custom_ops.custom_Detach(qk, 1)
                            qk = paddle.fluid.layers.elementwise_add(qk_scale,
                                                                     qk)
                            qk = paddle.fluid.layers.softmax(qk)
                            if not self.no_attn_dropout:
                                qk = self.custom_ops.custom_DropoutWithTrainingSwitch(
                                    qk,
                                    is_training,
                                    ratio=self.attention_probs_dropout_prob)

                            qkv = paddle.matmul(qk, v)
                            qkv.block.ops[-1]._set_attr('__available_memory',
                                                        self.available_memory)

                            qkv = paddle.transpose(qkv, [0, 2, 1, 3])
                            qkv = paddle.reshape(qkv, [-1, self.hidden_size])

                        att_weights = paddle.static.create_parameter(
                            shape=[self.hidden_size, self.hidden_size],
                            dtype="float32",
                            name="Layer" + str(i) + "/Attention/Out")
                        att_biases = paddle.static.create_parameter(
                            shape=[self.hidden_size],
                            dtype="float32",
                            is_bias=True,
                            name="Layer" + str(i) + "/Attention/Out_bias")

                        qkv = paddle.matmul(qkv, att_weights)
                        qkv.block.ops[-1]._set_attr('__available_memory',
                                                    self.available_memory)

                        qkv = paddle.add(qkv, att_biases)

                        qkv = self.custom_ops.custom_DropoutWithTrainingSwitch(
                            qkv,
                            is_training,
                            ratio=self.attention_probs_dropout_prob)
                        qkv = paddle.add(layer_input, qkv)

                        attn_layer_norm_weights = paddle.ParamAttr(
                            name="Layer" + str(i) + "/Attention/Gamma")
                        attn_layer_norm_biases = paddle.ParamAttr(
                            name="Layer" + str(i) + "/Attention/Beta")
                        attn_layer_norm = nn.LayerNorm(
                            self.hidden_size,
                            epsilon=0.001,
                            weight_attr=attn_layer_norm_weights,
                            bias_attr=attn_layer_norm_biases)
                        attention = attn_layer_norm(qkv)

                    if self.need_checkpoint_att(i):
                        with paddle.static.name_scope(f"Layer{i}"):
                            logging.info(
                                f'add checkpointoutput for Attention_{i}')
                            if self.activation_checkpoint_dtype == "FLOAT8":
                                attention = self.custom_ops.custom_CastToFp8(
                                    attention, "4", "3", "0")
                                attention = self.custom_ops.checkpointoutput(
                                    attention)
                                attention = self.custom_ops.custom_CastFromFp8(
                                    attention, "FLOAT16", "4", "3", "0")
                            else:
                                attention = self.custom_ops.checkpointoutput(
                                    attention)

                # FF
                ff_index = self.ff_ipu_index[i]
                ff_stage = self.ff_ipu_index[i] if self.ff_ipu_index[
                    i] > 0 else max(self.ff_ipu_index) + 1
                logging.info("FF %d - ipu_index:%d, ipu_stage:%d" %
                             (i, ff_index, ff_stage))
                with paddle.static.ipu_shard_guard(
                        index=ff_index, stage=ff_stage):
                    with paddle.static.name_scope(f"Layer{i}/FF"):
                        with paddle.static.name_scope(f"1"):
                            ff_weights1 = paddle.static.create_parameter(
                                shape=[self.hidden_size, 4 * self.hidden_size],
                                dtype="float32",
                                name="Layer" + str(i) + "/FF/1/W")
                            ff_biases1 = paddle.static.create_parameter(
                                shape=[4 * self.hidden_size],
                                dtype="float32",
                                is_bias=True,
                                name="Layer" + str(i) + "/FF/1/B")

                            ff_weights2 = paddle.static.create_parameter(
                                shape=[4 * self.hidden_size, self.hidden_size],
                                dtype="float32",
                                name="Layer" + str(i) + "/FF/2/W")
                            ff_biases2 = paddle.static.create_parameter(
                                shape=[self.hidden_size],
                                dtype="float32",
                                is_bias=True,
                                name="Layer" + str(i) + "/FF/2/B")

                            ff = paddle.matmul(attention, ff_weights1)
                            ff.block.ops[-1]._set_attr('__available_memory',
                                                       self.available_memory)

                            ff = paddle.add(ff, ff_biases1)

                        ff = paddle.fluid.layers.gelu(ff, approximate=True)
                        with paddle.static.name_scope(f"2"):
                            ff = paddle.matmul(ff, ff_weights2)
                            ff.block.ops[-1]._set_attr('__available_memory',
                                                       self.available_memory)

                            ff = paddle.add(ff, ff_biases2)

                        ff = self.custom_ops.custom_DropoutWithTrainingSwitch(
                            ff,
                            is_training,
                            ratio=self.attention_probs_dropout_prob)
                        ff = paddle.add(attention, ff)
                        ff_layer_norm_weights = paddle.ParamAttr(
                            name="Layer" + str(i) + "/FF/Gamma")
                        ff_layer_norm_biases = paddle.ParamAttr(
                            name="Layer" + str(i) + "/FF/Beta")
                        ff_layer_norm = nn.LayerNorm(
                            self.hidden_size,
                            epsilon=0.001,
                            weight_attr=ff_layer_norm_weights,
                            bias_attr=ff_layer_norm_biases)
                        sequence_output = ff_layer_norm(ff)

                    if self.need_checkpoint_ffn(i):
                        with paddle.static.name_scope(f"Layer{i}"):
                            logging.info(f'add checkpointoutput for ff_{i}')
                            if self.activation_checkpoint_dtype == "FLOAT8" and i < 23:
                                sequence_output = self.custom_ops.custom_CastToFp8(
                                    sequence_output, "4", "3", "0")
                                sequence_output = self.custom_ops.checkpointoutput(
                                    sequence_output)
                                sequence_output = self.custom_ops.custom_CastFromFp8(
                                    sequence_output, "FLOAT16", "4", "3", "0")

                            else:
                                sequence_output = self.custom_ops.checkpointoutput(
                                    sequence_output)

        return sequence_output


class BertPretrainingNSP(Layer):
    """
    NSP task for pretraining.

    Args:
        hidden_size (int, optional):
            Dimensionality of the encoder layer. Defaults to `1024`.
        sequence_length (int, optional):
            The max length of sequence. Defaults to `512`.
        max_sequences_per_pack (int, optional):
            The maximum number of sequences to expect in a pack. Defaults to 3.

    """

    def __init__(self,
                 num_ipus=4,
                 hidden_size=1024,
                 sequence_length=512,
                 max_sequences_per_pack=3,
                 custom_ops=None):
        super(BertPretrainingNSP, self).__init__()
        self.num_ipus = num_ipus
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.max_sequences_per_pack = max_sequences_per_pack
        self.custom_ops = custom_ops

    def pooler(self, pooler_input):
        """
        Extract the cls tokens of all sequences that have been packed into a sample
        (these tokens have been rearranged to the back of the pack)
        """

        starts = self.sequence_length - self.max_sequences_per_pack
        ends = self.sequence_length
        pooler_input = paddle.slice(pooler_input, [1], [starts], [ends])
        pooler_input = paddle.reshape(
            pooler_input, [-1, self.max_sequences_per_pack, self.hidden_size])

        weights = paddle.static.create_parameter(
            shape=[self.hidden_size, self.hidden_size],
            dtype="float32",
            name="NSP/PoolW")
        bias = paddle.static.create_parameter(
            shape=[self.hidden_size],
            dtype="float32",
            name="NSP/PoolB",
            is_bias=True)
        x = paddle.add(paddle.matmul(pooler_input, weights), bias)
        x = paddle.tanh(x)
        return x

    def forward(self, encoders_output, nsp_labels, nsp_weights):
        """
        Args:
            encoders_output (Tensor):
                The output from Transformer Layers.
                Its data type should be `float32` or `float16`.
            nsp_labels (Tensor):
                Labels of next sentence.
                Its data type should be `int32`.
            nsp_weights (Tensor):
                Weights of next sentence.
                Its data type should be `int32`.

        Returns:
            tuple: Returns tuple (`nsp_loss`, `nsp_accuracy`).

            With the fields:

            - `nsp_loss` (Tensor):
                The loss of nsp task.
                Its data type should be float32 and its shape is [1].
            - `nsp_accuracy` (Tensor):
                The accuracy of nsp task.
                Itss data type should be float32 and its shape is [batch_size, 1].
        """
        logging.info("NSP Layer - ipu_index:%d, ipu_stage:%d" %
                     (0, self.num_ipus))
        with paddle.static.ipu_shard_guard(index=0, stage=self.num_ipus):
            with paddle.static.name_scope("NSP"):
                with paddle.static.amp.fp16_guard():
                    x = paddle.reshape(
                        encoders_output,
                        [-1, self.sequence_length, self.hidden_size])
                    x = self.pooler(x)
                    cls_weights = paddle.static.create_parameter(
                        shape=[self.hidden_size, 2],
                        dtype="float32",
                        name="NSP/NspW")
                    cls_bias = paddle.static.create_parameter(
                        shape=[2],
                        dtype="float32",
                        name="NSP/NspB",
                        is_bias=True)
                    logits = paddle.add(paddle.matmul(x, cls_weights), cls_bias)

                    predictions = paddle.fluid.layers.argmax(logits, axis=-1)
                    probs = paddle.fluid.layers.softmax(logits, axis=-1)

                    nsp_loss_per_token = self.custom_ops.custom_nll_loss(
                        probs, nsp_labels, 2, "None", False)
                    # nsp_loss_per_token = paddle.cast(nsp_loss_per_token, "float32")
                    nsp_accuracy_per_token = paddle.fluid.layers.equal(
                        nsp_labels, predictions)

                # nsp_loss_per_token = paddle.cast(nsp_loss_per_token, "float32")
                nsp_weights = paddle.cast(nsp_weights, "float32")
                attempted = paddle.fluid.layers.nn.reduce_sum(
                    nsp_weights, dim=-1, keep_dim=True)
                nsp_loss = paddle.fluid.layers.elementwise_mul(
                    nsp_loss_per_token, nsp_weights)
                nsp_loss = paddle.fluid.layers.nn.reduce_sum(
                    nsp_loss, dim=-1, keep_dim=False)
                nsp_loss = paddle.fluid.layers.elementwise_div(nsp_loss,
                                                               attempted)
                nsp_loss = paddle.fluid.layers.nn.reduce_mean(
                    nsp_loss, keep_dim=False)

                # nsp accuracy
                nsp_accuracy_per_token = paddle.cast(nsp_accuracy_per_token,
                                                     "float32")
                nsp_accuracy = paddle.fluid.layers.elementwise_mul(
                    nsp_accuracy_per_token, nsp_weights)
                nsp_accuracy = paddle.fluid.layers.elementwise_div(nsp_accuracy,
                                                                   attempted)
                nsp_accuracy = paddle.fluid.layers.nn.reduce_sum(
                    nsp_accuracy, dim=-1, keep_dim=False)
                nsp_accuracy = paddle.fluid.layers.nn.reduce_mean(
                    nsp_accuracy, keep_dim=False)
        return nsp_loss, nsp_accuracy


class BertPretrainingCLS(Layer):
    """
    CLS task for pretraining.

    Args:
        num_ipus (int, optional):
            The number of IPUs. Defaults to `4`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer. Defaults to `1024`.
    """

    def __init__(self, num_ipus, hidden_size):
        super(BertPretrainingCLS, self).__init__()
        self.num_ipus = num_ipus
        self.hidden_size = hidden_size

    def forward(self, encoders):
        """
        Args:
            encoders (Tensor):
                The output from Transformer Layers.
                Its data type should be `float32` or `float16`.

        Returns:
            tuple: Returns Tensor ``encoders``.

            With the fields:

            - `encoders` (Tensor):
                The output from the CLS Layer.
                It's data type should be float32 or float16 and its shape is [batch_size * sequence_length, hidden_size].
        """
        with paddle.static.amp.fp16_guard():
            with paddle.static.ipu_shard_guard(index=0, stage=self.num_ipus):
                with paddle.static.name_scope("CLS"):
                    weights = paddle.static.create_parameter(
                        shape=[self.hidden_size, self.hidden_size],
                        dtype="float32",
                        name="CLS/LMPredictionW")
                    biases = paddle.static.create_parameter(
                        shape=[self.hidden_size],
                        dtype="float32",
                        is_bias=True,
                        name="CLS/LMPredictionB")
                    encoders = paddle.add(
                        paddle.matmul(encoders, weights), biases)
                    encoders = paddle.fluid.layers.gelu(
                        encoders, approximate=True)

                    layer_norm_weights = paddle.ParamAttr(name="CLS/Gamma")
                    layer_norm_biases = paddle.ParamAttr(name="CLS/Beta")
                    layer_norm = paddle.nn.LayerNorm(
                        self.hidden_size,
                        epsilon=0.001,
                        weight_attr=layer_norm_weights,
                        bias_attr=layer_norm_biases)
                    encoders = layer_norm(encoders)
        return encoders


class BertPretrainingMLM(Layer):
    """
    MLM task for pretraining.

    Args:
        num_ipus (int, optional):
            The number of IPUs. Defaults to `4`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer. Defaults to `1024`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        vocab_size (int, optional):
            Vocabulary size of `input_ids`. Also is the vocab size of token embedding matrix.
            Defaults to `30912`.
        max_predictions_per_seq (int, optional):
            The maximum number of masked tokens in an un-packed example. Defaults to 76.
        max_sequences_per_pack (int, optional):
            The maximum number of sequences to expect in a pack. Defaults to 3.
        custom_ops (Module, optional):
            The callable python module contains all CustomOp Layer APIs. Defaults to `None`.
    """

    def __init__(self,
                 num_ipus=4,
                 hidden_size=1024,
                 max_position_embeddings=512,
                 vocab_size=30912,
                 max_predictions_per_seq=76,
                 max_sequences_per_pack=3,
                 custom_ops=None):
        super(BertPretrainingMLM, self).__init__()
        self.num_ipus = num_ipus
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.max_predictions_per_seq = max_predictions_per_seq
        self.max_sequences_per_pack = max_sequences_per_pack
        self.custom_ops = custom_ops

        self.mlm_threshold_attrs = {
            'name': 'mlm_threshold',
            'shape': [1],
            'dtype': 'float32',
            'value': 0,
        }

    def forward(self, encoders_output, word_embedding, masked_lm_ids,
                masked_lm_weights, nsp_loss):
        """
        Args:
            encoders_output (Tensor):
                The output from CLS Layer.
                Its data type should be `float32` or `float16`.
            word_embedding (Tensor):
                The word embedding matrix.
                It's data type should be `float32` or float16 and its shape is [hidden_size, vocab_size].
            masked_lm_ids (Tensor):
                Indices of masked tokens.
                Its data type should be `int32`, and its shape is [batch_size, max_predictions_per_seq + max_sequences_per_pack - 1].
            masked_lm_weights (Tensor):
                Weights of masked tokens.
                Its data type should be `int32`, and its shape is [batch_size, max_predictions_per_seq + max_sequences_per_pack - 1].
            nsp_loss (Tensor):
                The loss of nsp task.
                Its data type should be `float32` and its shape is [1].
        Returns:
            tuple: Returns tuple (``final_loss``, ``mlm_acc``).

            With the fields:

            - `final_loss` (Tensor):
                The mlm + nsp loss.
                It's data type should be float32 and its shape is [1].
            - `mlm_acc` (Tensor):
                The accuracy of mlm task.
                It's data type should be float32 and its shape is [batch_size].
        """
        logging.info("MLM Layer - ipu_index:%d, ipu_stage:%d" %
                     (0, self.num_ipus))
        with paddle.static.name_scope("MLM"):
            with paddle.static.ipu_shard_guard(index=0, stage=self.num_ipus):
                with paddle.static.amp.fp16_guard():
                    mlm = paddle.reshape(
                        encoders_output,
                        [-1, self.max_position_embeddings, self.hidden_size])
                    mlm = paddle.slice(mlm, [1], [0], [
                        self.max_predictions_per_seq +
                        self.max_sequences_per_pack - 1
                    ])
                    mlm = paddle.reshape(mlm, [-1, self.hidden_size])
                    mlm = self.custom_ops.checkpointoutput(mlm)

                    # serialized matmul
                    mlm = paddle.matmul(mlm, word_embedding)
                    op = mlm.block.ops[-1]
                    op._set_attr('serialize_factor', 4)

                    mlm = self.custom_ops.checkpointoutput(mlm)
                    biases = paddle.static.create_parameter(
                        shape=[self.vocab_size],
                        dtype="float32",
                        is_bias=True,
                        name='MLM/ProjectionB')
                    mlm = paddle.add(mlm, biases)

                    mlm = self.custom_ops.checkpointoutput(mlm)
                    mlm = paddle.reshape(mlm, [
                        -1, self.max_predictions_per_seq +
                        self.max_sequences_per_pack - 1, self.vocab_size
                    ])
                    mlm_acc = paddle.fluid.layers.argmax(mlm, axis=-1)
                    mlm_acc = paddle.cast(mlm_acc, "int32")
                    mlm_acc = paddle.fluid.layers.equal(mlm_acc, masked_lm_ids)
                    mlm_acc = self.custom_ops.custom_Detach(mlm_acc, 1)

                    softmax = paddle.fluid.layers.softmax(mlm, axis=-1)
                    mlm_loss = self.custom_ops.custom_nll_loss(
                        softmax, masked_lm_ids, 0, "0", False)

                casted_masked_lm_weights = paddle.cast(masked_lm_weights,
                                                       "float32")
                casted_masked_lm_weights = paddle.fluid.layers.flatten(
                    casted_masked_lm_weights, axis=1)
                mlm_threshold = paddle.fluid.layers.fill_constant(
                    **self.mlm_threshold_attrs)
                # mlm_threshold = paddle.cast(mlm_threshold, "float32")
                mlm_weights = paddle.greater_than(casted_masked_lm_weights,
                                                  mlm_threshold)
                mlm_weights = paddle.cast(mlm_weights, "float32")

                mlm_acc1 = paddle.sum(mlm_weights, axis=-1)
                mlm_acc = paddle.cast(mlm_acc, "float32")
                mlm_acc2 = paddle.fluid.layers.elementwise_mul(mlm_acc,
                                                               mlm_weights)
                mlm_acc2 = paddle.sum(mlm_acc2, axis=-1)

                mlm_acc = paddle.fluid.layers.elementwise_div(mlm_acc2,
                                                              mlm_acc1)

                # mlm_loss = paddle.cast(mlm_loss, "float32")
                final_loss = paddle.add(mlm_loss, nsp_loss)
        return final_loss, mlm_acc
