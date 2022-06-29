# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


def get_tf_mapping(args, task="PRETRAINING"):

    squad_mapping = { # TODO
        "cls/squad/output_weights": "Squad/SquadW",
        "cls/squad/output_bias": "Squad/SquadB"
    }

    nsp_mapping = { # TODO
        "bert/pooler/dense/kernel": "NSP/PoolW",
        "bert/pooler/dense/bias": "NSP/PoolB",
        "cls/seq_relationship/output_weights": "NSP/NspW",
        "cls/seq_relationship/output_bias": "NSP/NspB"
    }

    lm_mapping = {
        "cls/predictions/transform/dense/kernel": "CLS/LMPredictionW",
        "cls/predictions/transform/dense/bias": "CLS/LMPredictionB",
        "cls/predictions/transform/LayerNorm/gamma": "CLS/Gamma",
        "cls/predictions/transform/LayerNorm/beta": "CLS/Beta",
        "cls/predictions/output_bias": "MLM/ProjectionB"
    }

    tf_to_pdmodel = {
        "bert/embeddings/word_embeddings": "Embedding/Embedding_Dict",
        "bert/embeddings/position_embeddings": "Embedding/Positional_Dict",
        "bert/embeddings/token_type_embeddings": "Embedding/Segment_Dict",
        "bert/embeddings/LayerNorm/gamma": "Embedding/Gamma",
        "bert/embeddings/LayerNorm/beta": "Embedding/Beta"
    }

    for i in range(args.num_hidden_layers):
        layer = {
            f"bert/encoder/layer_{i}/attention/output/dense/kernel":
            f"Layer{i}/Attention/Out",
            f"bert/encoder/layer_{i}/attention/self/query/bias":
            f"Layer{i}/Attention/QKV_bias",
            f"bert/encoder/layer_{i}/attention/self/key/bias":
            f"Layer{i}/Attention/QKV_bias",
            f"bert/encoder/layer_{i}/attention/self/value/bias":
            f"Layer{i}/Attention/QKV_bias",
            f"bert/encoder/layer_{i}/attention/output/dense/bias":
            f"Layer{i}/Attention/Out_bias",
            f"bert/encoder/layer_{i}/attention/output/LayerNorm/gamma":
            f"Layer{i}/Attention/Gamma",
            f"bert/encoder/layer_{i}/attention/output/LayerNorm/beta":
            f"Layer{i}/Attention/Beta",
            f"bert/encoder/layer_{i}/intermediate/dense/kernel":
            f"Layer{i}/FF/1/W",
            f"bert/encoder/layer_{i}/intermediate/dense/bias":
            f"Layer{i}/FF/1/B",
            f"bert/encoder/layer_{i}/output/dense/kernel": f"Layer{i}/FF/2/W",
            f"bert/encoder/layer_{i}/output/dense/bias": f"Layer{i}/FF/2/B",
            f"bert/encoder/layer_{i}/output/LayerNorm/gamma":
            f"Layer{i}/FF/Gamma",
            f"bert/encoder/layer_{i}/output/LayerNorm/beta":
            f"Layer{i}/FF/Beta",
        }
        if args.split_qkv:
            layer[
                f"bert/encoder/layer_{i}/attention/self/query/kernel"] = f"Layer{i}/Attention/Q"
            layer[
                f"bert/encoder/layer_{i}/attention/self/key/kernel"] = f"Layer{i}/Attention/K"
            layer[
                f"bert/encoder/layer_{i}/attention/self/value/kernel"] = f"Layer{i}/Attention/V"
        else:
            layer[
                f"bert/encoder/layer_{i}/attention/self/query/kernel"] = f"Layer{i}/Attention/QKV"
            layer[
                f"bert/encoder/layer_{i}/attention/self/key/kernel"] = f"Layer{i}/Attention/QKV"
            layer[
                f"bert/encoder/layer_{i}/attention/self/value/kernel"] = f"Layer{i}/Attention/QKV"
        tf_to_pdmodel.update(**layer)

    if task == "PRETRAINING":
        tf_to_pdmodel.update(**lm_mapping)
        tf_to_pdmodel.update(**nsp_mapping)
    elif task == "SQUAD":
        tf_to_pdmodel.update(**squad_mapping)

    # Load optimizer state from checkpoint as well
    optimizer_states = {}
    for tf_var, pdmodel_var in tf_to_pdmodel.items():
        optimizer_states[tf_var + "/LAMB"] = pdmodel_var + '_moment1_0'
        optimizer_states[tf_var + "/LAMB_1"] = pdmodel_var + '_moment2_0'
    tf_to_pdmodel.update(**optimizer_states)

    return tf_to_pdmodel


def get_tf_transform(task="PRETRAINING"):
    # Some of the head weights are stored transposed in the Google Research checkpoint
    # compared to the Popart model.
    tf_to_pdmodel_tform = {}
    if task == "PRETRAINING":
        tf_to_pdmodel_tform.update(
            **{"cls/seq_relationship/output_weights": np.transpose})
    elif task == "SQUAD":
        tf_to_pdmodel_tform.update(**{"cls/squad/output_weights": np.transpose})

    return tf_to_pdmodel_tform


def generate_initializers(args, map_names, load_data, mapping, transform={}):
    """
    Generate a graph initializer dictionary from the tensor names and
    data loaded from either a checkpoint or frozen graph using one of
    the methods below (`load_tf_ckpt_data` or `load_tf_frozen_data`).

    In the general case, this will simply map the tensor names from the
    TF model to the Popart model.

    The exception is the query-key-value matrix which is formed by
    concatenating the weight tensors Q, K and V.
    """
    initializers = {}
    initializers_param = {}
    initializers_opt = {}

    qkv_tensor_range = {
        "query": (0, args.hidden_size),
        "key": (args.hidden_size, args.hidden_size * 2),
        "value": (args.hidden_size * 2, args.hidden_size * 3),
    }

    for name, array in zip(map_names, load_data):
        logger.debug(
            f"Initialising tensor from checkpoint {name} -> {mapping[name]}")

        # config["lamb_m_dtype"] is for setting the data type for accl1 of lamb
        # BERT can use FP16 for accl1 without lossing accuracy
        # accl2 is always in FP32
        lamb_m_dtype = np.float32
        dtype = np.float16

        # the initial data for the optimizer state is scaled by loss_scaling since the optimizer
        # option "scaleOptimizerState" is on.
        if "moment1" in mapping[name]:
            if array.dtype != lamb_m_dtype:
                array *= args.scale_loss
                array = array.astype(lamb_m_dtype)
        elif "moment2" in mapping[name]:
            if array.dtype != np.float32:
                array *= (args.scale_loss * args.scale_loss)
                array = array.astype(lamb_m_dtype)
        elif array.dtype != dtype:
            array = array.astype(dtype)

        # If it's part of QKV, we need to handle separately as those 3
        # tensors need concatenating into one
        if not args.split_qkv:
            if mapping[name].endswith("QKV") or mapping[name].endswith(
                    "QKV_moment1_0") or mapping[name].endswith("QKV_moment2_0"):
                qkv_part = name.split("/")[5]
                if mapping[name] not in initializers.keys():
                    qkv_shape = (array.shape[0], array.shape[1] * 3)
                    initializers[mapping[name]] = np.empty(
                        qkv_shape, dtype=array.dtype)

                start_idx = qkv_tensor_range[qkv_part][0]
                end_idx = qkv_tensor_range[qkv_part][1]
                initializers[mapping[name]][:, start_idx:end_idx] = array
                logger.debug(
                    f"Initialising QKV component {name}[{start_idx}:{end_idx}] from checkpoint"
                )
                continue

        # If it's part of QKV biases, we need to handle separately as those 3
        # tensors need concatenating into one
        if "QKV_bias" in mapping[name]:
            qkv_part = name.split("/")[5]
            if mapping[name] not in initializers.keys():
                qkv_shape = (array.shape[0] * 3)
                initializers[mapping[name]] = np.empty(
                    qkv_shape, dtype=array.dtype)

            start_idx = qkv_tensor_range[qkv_part][0]
            end_idx = qkv_tensor_range[qkv_part][1]
            initializers[mapping[name]][start_idx:end_idx] = array
            logger.debug(
                f"Initialising QKV_bias component {name}[{start_idx}:{end_idx}] from checkpoint"
            )
            continue

        if name in transform:
            array = transform[name](array)

        padded_vocab_length = args.vocab_size
        if "Embedding/Embedding_Dict" in mapping[name]:
            tf_vocab_length = array.shape[0]
            diff = padded_vocab_length - tf_vocab_length
            # Pad or Crop the vocab.
            if diff > 0:
                logger.info(
                    f"Padding the vocabulary. From {tf_vocab_length} to {padded_vocab_length}"
                )
                pad = np.zeros((diff, args.hidden_size)).astype(array.dtype)
                array = np.concatenate((array, pad), axis=0)
            else:
                logger.warning(
                    f"Cropping the vocabulary may negatively effect performance. From {tf_vocab_length} to {padded_vocab_length}"
                )
                array = np.array(array[:padded_vocab_length, :])
            array = np.transpose(array, [1, 0])

        if "MLM/ProjectionB" in mapping[name]:
            tf_vocab_length = array.shape[0]
            diff = padded_vocab_length - tf_vocab_length
            # Pad or Crop the vocab.
            if diff > 0:
                logger.info(
                    f"Padding the vocabulary. From {tf_vocab_length} to {padded_vocab_length}"
                )
                pad = np.zeros((diff)).astype(array.dtype)
                array = np.concatenate((array, pad), axis=0)
            else:
                logger.warning(
                    f"Cropping the projection bias from {tf_vocab_length} to {padded_vocab_length}"
                )
                array = np.array(array[:padded_vocab_length])

        if "Embedding/Positional_Dict" in mapping[name]:
            max_pos, hidden_len = array.shape
            if max_pos > args.max_position_embeddings:
                array = array[:args.max_position_embeddings, :]

            # Otherwise just copy the positional embeddings over and over again as is done in longformer
            elif max_pos < args.max_position_embeddings:
                logger.warning(
                    f"Not enough positional embeddings in checkpoint, copying to match length..."
                )
                array = array[np.mod(
                    np.arange(args.max_position_embeddings), max_pos)]

        if "NSP/NspW_" in mapping[name]:
            array = np.transpose(array)

        initializers[mapping[name]] = array.copy()
        for k in initializers:
            if "moment" in k:
                initializers_opt[k] = initializers[k]
            else:
                initializers_param[k] = initializers[k]
    return initializers_param, initializers_opt


def load_tf_ckpt_data(tf_checkpoint_path, mapping):
    """
    Parses a checkpoint file and outputs a tensors (lists of names and data)
    found in both the mapping and the checkpoint, ready for importing into the
    Bert model.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model requires TensorFlow to be installed. "
            "Please see https://www.tensorflow.org/install/ for installation "
            "instructions.")
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)

    map_names = [name for name, shape in init_vars if name in mapping.keys()]
    for name in (n for n, _ in init_vars if n not in mapping.keys()):
        logger.info(f"Skipping load of {name} - Not in mapping")

    load_data = [tf.train.load_variable(tf_path, name) for name in map_names]

    return map_names, load_data


def load_initializers_from_tf(args, task):
    """
    Loads weights, etc. from Tensorflow files into a dictionary of Numpy Arrays.

    Can read either checkpoint files, or frozen graphs, according to the
    `is_checkpoint` flag, passed in as the second argument.
    """
    file_path = args.tf_checkpoint
    mapping = get_tf_mapping(args, task=task)
    transform = get_tf_transform(task=task)

    names, data = load_tf_ckpt_data(file_path, mapping)
    return generate_initializers(args, names, data, mapping, transform)
