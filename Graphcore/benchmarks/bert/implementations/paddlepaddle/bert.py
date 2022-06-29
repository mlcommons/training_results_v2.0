# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import time
import os
import sys
import random
import logging
import re
import socket

import paddle
import paddle.static
import paddle.optimizer

from paddlenlp.transformers import LinearDecayWithWarmup

import numpy as np

from modeling import BertEmbeddings, BertModel, BertPretrainingNSP, BertPretrainingMLM, BertPretrainingCLS
from bert_data import get_pretraining_dataset, get_packed_pretraining_dataset
from load_tf_ckpt import load_initializers_from_tf, get_tf_mapping

import utils
import utils.popvision as popvision

from mlperf_logging import mllog
from utils import logging_wrapper, set_distribution_args, sum_distributed_data, sum_distributed_dataf, distributed_barrier, engine_cache_copy
from paddle.utils.cpp_extension import load

np.set_printoptions(threshold=np.inf)

mlm_accuracy_target = 0.720

logger = logging.getLogger('BERT')


def bert_input_shape(args):
    return {
        "indices": [args.micro_batch_size * args.seq_len],
        "input_mask": [args.micro_batch_size, args.seq_len],
        "segment_ids": [args.micro_batch_size * args.seq_len],
        "positions": [args.micro_batch_size * args.seq_len],
        "masked_lm_ids": [
            args.micro_batch_size,
            args.max_predictions_per_seq + args.max_sequences_per_pack - 1
        ],
        "masked_lm_weights": [
            args.micro_batch_size,
            args.max_predictions_per_seq + args.max_sequences_per_pack - 1
        ],
        "nsp_labels": [args.micro_batch_size, args.max_sequences_per_pack],
        "nsp_weights": [args.micro_batch_size, args.max_sequences_per_pack]
    }


def get_online_evaluation_dataset(args):
    input_files = []
    for i in os.listdir(args.validation_files):
        i = args.validation_files + '/' + i
        input_files.append(i)

    tensor_shapes = [(name, shape)
                     for name, shape in bert_input_shape(args).items()]

    dataset = get_packed_pretraining_dataset(
        tensor_shapes,
        input_files=input_files,
        seed=0,
        sequence_length=args.seq_len,
        mask_tokens=args.max_predictions_per_seq,
        max_sequences_per_pack=args.max_sequences_per_pack,
        vocab_length=args.vocab_size,
        batch_size=args.micro_batch_size,
        batches_per_step=args.batches_per_step,
        accumulation_factor=args.grad_acc_factor,
        replication_factor=args.local_num_replicas,
        duplication_factor=1,
        shuffle=False,
        generated_data=False,
        epochs_to_cache=1,
        drop_remainder=False,
        use_popdist=True if args.POD > 64 else False,
        popdist_size=args.popdist_size,
        popdist_rank=args.popdist_rank)
    return dataset


def get_bert_dataset(args):
    input_files = []
    for i in os.listdir(args.input_files):
        i = args.input_files + '/' + i
        input_files.append(i)

    if not args.use_prepacked_pretraining_dataset:
        tensor_shapes = [(name, shape)
                         for name, shape in bert_input_shape(args).items()]

        return get_pretraining_dataset(
            tensor_shapes,
            input_files=input_files,
            seed=args.seed,
            sequence_length=args.seq_len,
            mask_tokens=args.max_predictions_per_seq,
            vocab_length=args.vocab_size,
            batch_size=args.micro_batch_size,
            batches_per_step=args.batches_per_step,
            accumulation_factor=args.grad_acc_factor,
            replication_factor=args.local_num_replicas,
            duplication_factor=args.duplication_factor,
            shuffle=args.shuffle,
            generated_data=False,
            epochs_to_cache=False,
            continue_training_from_epoch=False,
            use_popdist=True if args.POD > 64 else False,
            popdist_size=args.popdist_size,
            popdist_rank=args.popdist_rank)

    if args.use_prepacked_pretraining_dataset:
        tensor_shapes = [(name, shape)
                         for name, shape in bert_input_shape(args).items()]

        return get_packed_pretraining_dataset(
            tensor_shapes,
            input_files=input_files,
            seed=args.seed,
            sequence_length=args.seq_len,
            mask_tokens=args.max_predictions_per_seq,
            max_sequences_per_pack=args.max_sequences_per_pack,
            vocab_length=args.vocab_size,
            batch_size=args.micro_batch_size,
            batches_per_step=args.batches_per_step,
            accumulation_factor=args.grad_acc_factor,
            replication_factor=args.local_num_replicas,
            duplication_factor=args.duplication_factor,
            shuffle=args.shuffle,
            generated_data=False,
            epochs_to_cache=1,
            drop_remainder=True,
            continue_training_from_epoch=0,
            use_popdist=True if args.POD > 64 else False,
            popdist_size=args.popdist_size,
            popdist_rank=args.popdist_rank)


class Iteration:
    def __init__(self, args, steps_per_epoch):
        self.epoch = 0
        self.count = self.epoch * steps_per_epoch
        self.args = args
        self.epochs = args.epochs
        self.total_sequences_so_far = 0
        self.total_duration_so_far = 0.0

        # Compute the global batch size
        gbs = args.batch_size
        if args.POD >= 128:
            gbs *= args.popdist_size

        # Account for packing (multiply by the average number of sequences in a pack)
        gbs *= args.avg_seq_per_pack
        self.global_batch_size = gbs

        # When running the benchmark, the frequency of evaluation depends on the global batch size
        # Implement rule
        multiple = 25000
        eval_interval_samples = int(
            (0.05 * (230.23 * gbs + 3000000)) // multiple) * multiple
        eval_start = eval_interval_samples
        triggers = list(
            range(eval_start, args.max_training_sequences +
                  eval_interval_samples, eval_interval_samples))
        self.on_the_spot_validation_triggers = triggers
        status_str = f"\nVALIDATION CONFIG:"
        status_str = f"\n\t Target accuracy: {mlm_accuracy_target}"
        status_str += f"\n\t Global batch size: {gbs}"
        status_str += f"\n\t Validation interval: {eval_interval_samples}"
        status_str += f"\n\t Validation triggers: {triggers}"
        logger.info(status_str)

    def get_throughput(self, data, duration):
        sequences_per_pack = data["input_mask"].max(-1, keepdims=True)
        # Determine how many sequences were processed during the last step
        # across all instances
        sequences_in_sample = int(sequences_per_pack.sum())
        return np.divide(sequences_in_sample, duration)

    def flush_total_sequences(self, data):
        sequences_per_pack = data["input_mask"].max(-1, keepdims=True)
        sequences_in_sample = int(sequences_per_pack.sum())
        self.total_sequences_so_far += sequences_in_sample


def bert_pretrained_initialisers(args):
    logger.info(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
    initializers_param, initializers_opt = load_initializers_from_tf(
        args, "PRETRAINING")

    return initializers_param, initializers_opt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_data_holder(args):
    if args.device == "ipu":
        bs = args.micro_batch_size
    else:
        bs = args.batch_size
    indices = paddle.static.data(
        name="indices", shape=[bs * args.seq_len], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[bs * args.seq_len], dtype="int64")
    input_mask = paddle.static.data(
        name="input_mask", shape=[bs, args.seq_len], dtype="int32")
    positions = paddle.static.data(
        name="positions", shape=[bs * args.seq_len], dtype="int64")
    masked_lm_ids = paddle.static.data(
        name="masked_lm_ids",
        shape=[
            bs, args.max_predictions_per_seq + args.max_sequences_per_pack - 1
        ],
        dtype="int32")
    masked_lm_weights = paddle.static.data(
        name="masked_lm_weights",
        shape=[
            bs, args.max_predictions_per_seq + args.max_sequences_per_pack - 1
        ],
        dtype="int32")
    nsp_labels = paddle.static.data(
        name="nsp_labels",
        shape=[bs, args.max_sequences_per_pack],
        dtype="int32")
    nsp_weights = paddle.static.data(
        name="nsp_weights",
        shape=[bs, args.max_sequences_per_pack],
        dtype="int32")
    is_training = paddle.static.data(
        name="is_training", shape=[1], dtype="int32")

    return [
        indices, segment_ids, input_mask, positions, masked_lm_ids,
        masked_lm_weights, nsp_labels, nsp_weights, is_training
    ]


def load_custom_ops():
    # for successful compilation using `paddle.utils.cpp_extension.load`
    # we need a empty paddle custom op which defined in `custom_nop_op.cc`
    # the custom popart pattern is defined in `custom_popart_pattern.cc`
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_dir = cur_dir + "/custom_ops"
    custom_ops = load(
        name="custom_ops",
        sources=[
            f"{custom_dir}/custom_ops.cc", f"{custom_dir}/attention_mask.cpp",
            f"{custom_dir}/cast_from_fp8.cpp", f"{custom_dir}/cast_to_fp8.cpp",
            f"{custom_dir}/disable_attn_dropout_bwd_pattern.cpp",
            f"{custom_dir}/dropout_with_switchx.cpp",
            f"{custom_dir}/embedding_gather.cpp",
            f"{custom_dir}/seed_modify.cpp",
            f"{custom_dir}/lamb_serialised_weight_pattern.cpp",
            f"{custom_dir}/sparse_accumulate_pattern.cpp",
            f"{custom_dir}/tied_gather_pattern.cpp",
            f"{custom_dir}/detach_shape_inference.cpp",
            f"{custom_dir}/workarounds/accumulate_priority_pattern.cpp",
            f"{custom_dir}/workarounds/improved_sum.cpp",
            f"{custom_dir}/workarounds/prevent_const_expr_folding_op.cpp"
        ],
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
        extra_ldflags=["-lpopfloat"])
    return custom_ops


def setup_logger(log_level, handler=None):
    # Define a root config with a format which is simpler for console use
    root = logging.getLogger()
    root.setLevel(log_level)
    root_handler = logging.StreamHandler(sys.stdout)
    root_formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    root_handler.setFormatter(root_formatter)
    root.handlers = [root_handler]
    if handler is not None:
        root.handlers += [handler]

    # Define a specific Handler for this file that removes the root name.
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if handler is not None:
        logger.addHandler(handler)
    logger.propagate = False


def check_weight(args, program, mllogger):
    # Check if the weights in TensorFlow checkpoint are also in PaddlePaddle BERT
    tf_weight_map = get_tf_mapping(args)
    missing_tensors = []
    paramters = [x.name for x in program.all_parameters()]
    for key, value in tf_weight_map.items():
        # ignore optimizer state
        if "moment1_0" in value or "moment2_0" in value:
            continue
        values = []
        values.append(value)

        for v in values:
            if v not in paramters:
                missing_tensors.append(v)

    if len(missing_tensors) == 0:
        # PaddlePaddle BERT weights exactly matched TensorFlow checkpoint
        # log WEIGHTS_INITIALIZATION
        for key, value in tf_weight_map.items():
            if "moment1_0" in value or "moment2_0" in value:
                continue
            mllogger.event(
                key=mllog.constants.WEIGHTS_INITIALIZATION,
                metadata=dict(tensor=key))
    else:
        logger.warning(
            f"The following weights {missing_tensors} in checkpoint can not be found in BERT. "
            "The model is different from the one that generated the checkpoint.")


def main(args):
    paddle.enable_static()
    place = paddle.set_device(args.device)

    set_seed(args.seed)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    # Log the model configuration
    mllogger = logging_wrapper.SimpleWrapper(args, mllog)
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="Graphcore")
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value='bert')
    mllogger.event(
        key=mllog.constants.SUBMISSION_DIVISION, value=mllog.constants.CLOSED)
    mllogger.event(
        key=mllog.constants.SUBMISSION_STATUS, value=mllog.constants.ONPREM)
    mllogger.event(
        key=mllog.constants.SUBMISSION_PLATFORM, value="PaddlePaddle")

    # Define the model structure
    logger.info("Building Model")

    # Create input tensors
    data_holders = create_data_holder(args)
    [
        indices, segment_ids, input_mask, positions, masked_lm_ids,
        masked_lm_weights, nsp_labels, nsp_weights, is_training
    ] = data_holders

    # IPU custom_ops
    custom_ops = load_custom_ops()

    # The sharding and pipelining of encoder layers
    ipus_per_replica = 8
    if args.num_hidden_layers == 24:
        if args.POD == 16:
            attn_ipu_index = [
                1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7,
                7, 7, 7
            ]
            ff_ipu_index = [
                1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7,
                7, 7, 0
            ]
        elif args.POD >= 64:
            attn_ipu_index = [
                1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7,
                7, 7, 0
            ]
            ff_ipu_index = [
                1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7,
                7, 7, 0
            ]
    elif args.num_hidden_layers == 2:
        attn_ipu_index = [1, 1]
        ff_ipu_index = [1, 1]
        ipus_per_replica = 2
    elif args.num_hidden_layers == 1:
        attn_ipu_index = [1]
        ff_ipu_index = [1]
        ipus_per_replica = 2
    else:
        logging.ERROR("Only support num_hidden_layers = 1, 2, 24.")
        return

    available_memory = 0.3
    if args.POD == 64:
        available_memory = 1.0
    elif args.POD >= 128:
        available_memory = 0.3

    # Embedding
    embedding_init = BertEmbeddings(args.vocab_size, args.hidden_size,
                                    args.hidden_dropout_prob,
                                    args.max_position_embeddings, 2, custom_ops)
    embedding, word_embedding = embedding_init(indices, segment_ids, positions,
                                               input_mask, is_training)

    # Encoder Layers
    encoders_init = BertModel(
        attn_ipu_index, ff_ipu_index, args.hidden_size, args.num_hidden_layers,
        args.attention_probs_dropout_prob, 512, ipus_per_replica,
        args.split_qkv, args.activation_checkpoint_dtype, available_memory,
        args.no_attn_dropout, custom_ops)
    encoders = encoders_init(embedding, input_mask, is_training)

    # NSP
    nsp_init = BertPretrainingNSP(ipus_per_replica, args.hidden_size,
                                  args.seq_len, args.max_sequences_per_pack,
                                  custom_ops)
    nsp_loss, nsp_acc = nsp_init(encoders, nsp_labels, nsp_weights)

    # CLS
    cls_init = BertPretrainingCLS(ipus_per_replica, args.hidden_size)
    cls = cls_init(encoders)

    # MLM
    mlm_init = BertPretrainingMLM(ipus_per_replica, args.hidden_size,
                                  args.max_position_embeddings, args.vocab_size,
                                  args.max_predictions_per_seq,
                                  args.max_sequences_per_pack, custom_ops)
    final_loss, mlm_acc = mlm_init(cls, word_embedding, masked_lm_ids,
                                   masked_lm_weights, nsp_loss)

    # Load the training dataset
    iteration = Iteration(args, steps_per_epoch=0)
    iteration.mllogger = mllogger

    # Report the required characteristics and hyperparameters of the run
    mllogger.event(mllog.constants.SEED, value=args.seed)
    mllogger.event(mllog.constants.MAX_SEQUENCE_LENGTH, value=args.seq_len)
    mllogger.event(
        mllog.constants.GLOBAL_BATCH_SIZE, value=iteration.global_batch_size)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS,
                   args.grad_acc_factor)
    mllogger.event("opt_base_learning_rate", args.learning_rate)
    mllogger.event("opt_lamb_weight_decay_rate", args.weight_decay)
    mllogger.event(mllog.constants.OPT_LAMB_BETA_1, args.beta1)
    mllogger.event(mllog.constants.OPT_LAMB_BETA_2, args.beta2)
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS, args.warmup_steps)
    mllogger.event("num_warmup_steps", args.warmup_steps)
    mllogger.event("start_warmup_step", 0)
    mllogger.event("opt_learning_rate_training_steps", args.warmup_steps)
    mllogger.event("opt_epsilon", 1e-6)  # default
    mllogger.event(mllog.constants.OPT_LAMB_LR_DECAY_POLY_POWER, 1.0)

    # Define the dynamic learing_reate scheduler and optimizer
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, args.max_steps,
                                         args.warmup_steps)

    if args.is_training:
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1000.0)
        if args.POD >= 128:
            grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=100.0)
        optimizer = paddle.optimizer.SGD(learning_rate=lr_scheduler,
                                         weight_decay=args.weight_decay,
                                         grad_clip=grad_clip)
        if args.optimizer_type == 'adam':
            optimizer = paddle.optimizer.Adam(
                learning_rate=lr_scheduler,
                weight_decay=args.weight_decay,
                grad_clip=grad_clip)
        if args.optimizer_type == 'lamb':

            def exclude_fn(param):
                return param.name.endswith('B') or param.name.endswith("bias") or param.name.endswith('Beta') or param.name.endswith('Gamma') or \
                        "B_moment" in param.name or "bias_moment" in param.name or "Beta_moment" in param.name or "Gamma_moment" in param.name

            optimizer = paddle.optimizer.Lamb(
                learning_rate=lr_scheduler,
                lamb_weight_decay=args.weight_decay,
                beta1=args.beta1,
                beta2=args.beta2,
                grad_clip=grad_clip,
                exclude_from_weight_decay_fn=exclude_fn)
        optimizer.minimize(final_loss)

    amp_list = paddle.static.amp.CustomOpLists()
    amp_list.unsupported_list = {}
    to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(main_program,
                                                             amp_list)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    paddle.static.amp.cast_parameters_to_fp16(
        paddle.CPUPlace(), main_program, to_fp16_var_names=to_fp16_var_names)

    # remove __use__fp16__ namescope
    for op in main_program.blocks[main_program.current_block_idx].ops:
        op_namescope: str = op.attr('op_namescope')
        pattern = r'/__use_fp16__[_]*[0-9]*'
        cleaned_op_namescope = re.sub(pattern, '', op_namescope)
        if cleaned_op_namescope != op_namescope:
            op._set_attr('op_namescope', cleaned_op_namescope)

    # change namescope for pod128/256
    if args.POD >= 128:
        for op in main_program.blocks[main_program.current_block_idx].ops:
            raw_op_namescope: str = op.attr('op_namescope')
            op_namescope = raw_op_namescope
            # post_fix = ':0'
            post_fix = ''

            if op.type == 'elementwise_add':
                op_namescope = f"{op_namescope}Add{post_fix}/"
            if op.type == 'elementwise_mul':
                op_namescope = f"{op_namescope}Mul{post_fix}/"
            elif op.type == 'greater_than':
                op_namescope = f"{op_namescope}Greater{post_fix}/"
            elif op.type == 'reshape2':
                op_namescope = f"{op_namescope}Reshape{post_fix}/"
            elif op.type == 'custom_Detach':
                op_namescope = f"{op_namescope}Detach{post_fix}/"
            elif op.type == 'one_hot_v2':
                op_namescope = f"{op_namescope}OneHot{post_fix}/"
            elif op.type == 'matmul_v2':
                op_namescope = f"{op_namescope}MatMul{post_fix}/"
            elif op.type == 'layer_norm':
                op_namescope = f"{op_namescope}GroupNormalization{post_fix}/"
            elif op.type == 'custom_DropoutWithTrainingSwitch':
                op_namescope = f"{op_namescope}DropoutWithTrainingSwitch{post_fix}/"
            elif op.type == 'transpose2':
                op_namescope = f"{op_namescope}Transpose{post_fix}/"
            elif op.type == 'custom_AttentionMask':
                op_namescope = f"{op_namescope}AttentionMask{post_fix}/"
            elif op.type == 'arg_max':
                op_namescope = f"{op_namescope}ArgMax{post_fix}/"
            elif op.type == 'reduce_sum':
                op_namescope = f"{op_namescope}ReduceSum{post_fix}/"
            elif op.type == 'reduce_mean':
                op_namescope = f"{op_namescope}ReduceMean{post_fix}/"

            if op_namescope != raw_op_namescope:
                op._set_attr('op_namescope', op_namescope)

    # Initialize weights
    mllogger.start(mllog.constants.INIT_START, None)
    initializers_param, initializers_opt = bert_pretrained_initialisers(args)
    paddle.static.set_program_state(
        main_program, { ** initializers_param, ** initializers_opt})
    check_weight(args, main_program, mllogger)

    # IPU config
    if args.device == "ipu":
        feed_list = [
            "indices", "segment_ids", "input_mask", "positions",
            "masked_lm_ids", "masked_lm_weights", "nsp_labels", "nsp_weights",
            "is_training"
        ]
        fetch_list = [final_loss.name, mlm_acc.name, nsp_acc.name]

        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(
            num_ipus=ipus_per_replica * args.local_num_replicas
            if args.enable_replica and args.enable_pipelining else
            ipus_per_replica,
            micro_batch_size=args.micro_batch_size,
            is_training=args.is_training,
            enable_manual_shard=True)
        ipu_strategy.set_pipelining_config(
            enable_pipelining=args.enable_pipelining,
            batches_per_step=args.batches_per_step,
            enable_gradient_accumulation=args.enable_grad_acc and
            args.enable_pipelining,
            accumulation_factor=args.grad_acc_factor)
        # Training options
        ipu_strategy.set_options({
            "loss_scaling": args.scale_loss,
            "max_weight_norm": 65504.0,
            "use_no_bias_optimizer": True,
            "random_seed": args.seed,
            "accl1_type": 'FLOAT16',
            "accl2_type": 'FLOAT16',
            "scaled_optimizer_state": True
        })

        # IPU options
        ipu_strategy.set_options({
            "enable_floating_point_checks": False,
            "enable_stochastic_rounding": args.is_training,
            "enable_prefetch_datastreams": True,
            "enable_outlining": True,
            "outline_threshold": 10.0,
            "subgraph_copying_strategy": 1,
            "enable_half_partial": True,
            "enable_replicated_graphs": args.enable_replica and
            args.enable_pipelining,
            "replicated_graph_count": args.local_num_replicas,
            "default_prefetch_buffering_depth": 3,
            "rearrange_anchors_on_host": False,
            "accumulation_and_replication_reduction_type": 1,
            "location_optimizer": {
                "on_chip": not args.optimizer_state_offchip,
                "use_replicated_tensor_sharding": True
            },
            "disable_grad_accumulation_tensor_streams": True,
            "auto_recomputation": 3,
            "create_implicit_pipelining_fwd_only_program": True,
            "compilation_progress_logger": utils.ProgressFunc,
            "engine_options": {
                "debug.allowOutOfMemory": "true",
                "opt.useAutoloader": "true",
                "target.syncReplicasIndependently": "true",
                "exchange.streamBufferOverlap": "hostRearrangeOnly"
            }
        })

        if args.merge_collectives:
            ipu_strategy.set_options({
                "replicated_collectives_settings": {
                    "prepare_schedule_for_merging_collectives": True,
                    "merge_all_reduce_collectives": True
                }
            })
        if args.optimizer_state_offchip:
            ipu_strategy.set_options({"accumulate_outer_fragment": {3: [0]}})
        else:
            ipu_strategy.set_options({"accumulate_outer_fragment": {3: []}})

        # cache
        if not args.profile:
            ipu_strategy.set_options({
                "enable_engine_caching": True,
                "cache_path": f"paddle_cache/POD{args.POD}"
            })
        else:
            ipu_strategy.set_options({"enable_engine_caching": False})

        # distribution
        if args.POD >= 128:
            ipu_strategy.set_options({
                "enable_distribution": True,
                "enable_distributed_replicated_graphs": True,
                "global_replica_offset": args.replica_index_offset,
                "global_replication_factor": args.global_replicas,
                "location_optimizer": {
                    "on_chip": True,
                    "use_replicated_tensor_sharding": True,
                    "sharding_domain_with_consecutive": args.local_num_replicas
                },
                "location_weight": {
                    "on_chip": True,
                    "use_replicated_tensor_sharding": False,
                    "sharding_domain_with_consecutive": args.local_num_replicas
                },
                "location_accumulator": {
                    "on_chip": True,
                    "use_replicated_tensor_sharding": False,
                    "sharding_domain_with_consecutive": args.local_num_replicas
                }
            })

        #custom Ops
        ipu_strategy.add_custom_op("custom_AttentionMask", "AttentionMask",
                                   "ai.graphcore")
        ipu_strategy.add_custom_op("custom_CastFromFp8", "CastFromFp8",
                                   "ai.graphcore")
        ipu_strategy.add_custom_op("custom_CastToFp8", "CastToFp8",
                                   "ai.graphcore")
        ipu_strategy.add_custom_op("custom_Detach", "Detach", "custom.ops")
        ipu_strategy.add_custom_op("custom_DropoutWithTrainingSwitch",
                                   "DropoutWithTrainingSwitch", "ai.graphcore")
        ipu_strategy.add_custom_op("custom_EmbeddingGather", "EmbeddingGather",
                                   "ai.graphcore")

        # enable patterns
        ipu_strategy.enable_pattern("AccumulatePriorityPattern")
        ipu_strategy.disable_pattern("RemoveUnnecessaryLossGradCast")
        ipu_strategy.disable_pattern("DisableAttnDropoutBwdPattern")

        ipu_compiler = paddle.static.IpuCompiledProgram(
            main_program, ipu_strategy=ipu_strategy)
        main_program = ipu_compiler.compile(feed_list, fetch_list)

        if args.POD >= 128:
            # Sync to make sure all processes start the benchmark at the same time
            distributed_barrier()
            if args.compile_only == True:
                # Only compile to create engine cache for POD 128/256
                if args.popdist_rank == 0:
                    engine_cache_copy(args.POD, args.hosts)
                    logger.info("POD{} engine cache has been generated.".format(
                        args.POD))
                    logger.info("POD{} is ready to run.".format(args.POD))
                sys.exit(0)

        if os.getenv('IPU_COMPILE_ONLY', False) == True:
            # Offline compile
            logger.info("Finished compile only.")
            sys.exit(0)

    # Start the benchmark run
    logger.info("Benchmark timer started")
    global_cost = time.time()
    load_data_cost = time.time()
    mllogger.event(mllog.constants.CACHE_CLEAR, True)
    mllogger.start(key=mllog.constants.INIT_STOP)
    mllogger.start(key=mllog.constants.RUN_START)

    # Load the training dataset
    dataset = get_bert_dataset(args)

    # Load the evaluation dataset (if provided)
    evaluation_dataset = None
    if args.enable_validation:
        evaluation_dataset = get_online_evaluation_dataset(args)
        logger.info(f"Validation dataset length: {len(evaluation_dataset)}")
    iteration.evaluation_dataset = evaluation_dataset

    logger.info(f"Dataset length: {len(dataset)}")

    mllogger = iteration.mllogger
    global_step = 0

    # During validation the optimizer should prevent all training
    start_epoch = iteration.epoch
    mlm_eval_accuracy = 0
    first_epoch_num = 0
    mllogger.start(
        mllog.constants.BLOCK_START,
        None,
        metadata={"epoch_count": 1,
                  "first_epoch_num": first_epoch_num})

    load_data_cost = time.time() - load_data_cost
    global_train_cost = time.time()
    # Run
    batch_start = time.time()
    for iteration.epoch in range(start_epoch, iteration.epochs):
        for batch in dataset:
            epoch = iteration.epoch
            feed = {
                "indices": batch['indices'].astype(np.int64),
                "segment_ids": batch['segment_ids'].astype(np.int64),
                "input_mask": batch['input_mask'].astype(np.int32),
                "positions": batch['positions'].astype(np.int64),
                "masked_lm_ids": batch['masked_lm_ids'].astype(np.int32),
                "masked_lm_weights":
                batch['masked_lm_weights'].astype(np.int32),
                "nsp_labels": batch['nsp_labels'].astype(np.int32),
                "nsp_weights": batch['nsp_weights'].astype(np.int32),
                "is_training": np.ones([
                    args.batches_per_step * args.local_num_replicas *
                    args.grad_acc_factor
                ]).astype(np.int32)
            }
            read_cost = time.time() - batch_start
            # Run
            train_start = time.time()
            loss_return = exe.run(main_program,
                                  feed=feed,
                                  fetch_list=[final_loss, mlm_acc, nsp_acc],
                                  use_program_cache=True)
            if args.profile:
                ipu_compiler.reset()
                sys.exit(0)

            train_cost = time.time() - train_start
            iteration.total_duration_so_far += train_cost

            # Update learning rate
            lr_scheduler.step()

            iteration.count += 1
            total_cost = time.time() - batch_start
            num_token_for_loss = (batch["masked_lm_weights"] > 0).sum(-1)

            tput = iteration.get_throughput(batch, total_cost)

            # Run evaluation on the same graph by zeroing out the optimizer
            if args.enable_validation:
                next_trigger = min(iteration.on_the_spot_validation_triggers)

                # How many samples have been processed
                total_sequences_so_far = iteration.count * iteration.global_batch_size > next_trigger
                per_instance = iteration.total_sequences_so_far > next_trigger
                if (args.POD >= 128 and total_sequences_so_far) or (
                        args.POD < 128 and per_instance):
                    # Exit training block and enter validation block
                    mllogger.end(mllog.constants.BLOCK_STOP,
                                 None,
                                 metadata={"first_epoch_num": first_epoch_num})
                    first_epoch_num += 1

                    iteration.on_the_spot_validation_triggers.remove(
                        next_trigger)
                    start = time.time()

                    # Stream the validation optimizer to device (this optimizer zeros out all learning)
                    ipu_strategy.set_options({
                        "runtime_options.enable_eval": True
                    })

                    # Loop through the evaluation dataset to determine the total accuracy
                    # The evaluation dataset always uses 1 seq/sample for transparency
                    mlm_accuracy_data = []
                    indexing_data = []
                    num_tokens = []
                    for i, eval_data in enumerate(iteration.evaluation_dataset):
                        feed = {
                            "indices": eval_data['indices'].astype(np.int64),
                            "segment_ids":
                            eval_data['segment_ids'].astype(np.int64),
                            "input_mask":
                            eval_data['input_mask'].astype(np.int32),
                            "positions":
                            eval_data['positions'].astype(np.int64),
                            "masked_lm_ids":
                            eval_data['masked_lm_ids'].astype(np.int32),
                            "masked_lm_weights":
                            eval_data['masked_lm_weights'].astype(np.int32),
                            "nsp_labels":
                            eval_data['nsp_labels'].astype(np.int32),
                            "nsp_weights":
                            eval_data['nsp_weights'].astype(np.int32),
                            "is_training": np.zeros([
                                args.batches_per_step *
                                args.local_num_replicas * args.grad_acc_factor
                            ]).astype(np.int32)
                        }
                        validation_return = exe.run(
                            main_program,
                            feed=feed,
                            fetch_list=[final_loss, mlm_acc, nsp_acc],
                            use_program_cache=True)

                        # Collect the accuracies from the anchors, and copy them over
                        if args.accuracy_averaging_basis == "pertoken":
                            mlm_accuracy_data.append(
                                np.reshape(validation_return[1].copy(),
                                           [-1, 1]))
                            num_tokens.append(
                                np.reshape((batch["masked_lm_weights"] > 0).sum(
                                    -1), [-1, 1]))

                        elif args.accuracy_averaging_basis == "persequence":
                            mlm_accuracy_data.append(
                                np.reshape(validation_return[1].copy(),
                                           [-1, args.max_sequences_per_pack]))

                        # Pick out the sequences which contain data
                        tmp = np.arange(args.max_sequences_per_pack).reshape(
                            1, args.max_sequences_per_pack)
                        indexing_data.append(
                            eval_data["input_mask"].copy().reshape(
                                [-1, args.seq_len]).max(-1,
                                                        keepdims=True) > tmp)

                    # Concatenate the results from all the steps
                    mlm_accuracy_data = np.concatenate(mlm_accuracy_data)
                    num_tokens = [] if len(num_tokens) == 0 else np.concatenate(
                        num_tokens)
                    indexing_data = np.concatenate(indexing_data)

                    # The dataset is padded up to batch size, such that the remainder is not dropped
                    # this padding should now be removed before determining accuracy
                    num_padding_samples = iteration.evaluation_dataset.loader.dataloader.num_padding_samples
                    if num_padding_samples != 0:
                        mlm_accuracy_data = mlm_accuracy_data[:
                                                              -num_padding_samples, :]
                        indexing_data = indexing_data[:-num_padding_samples, :]
                    eval_sample_count = indexing_data.sum()

                    if args.accuracy_averaging_basis == "pertoken":
                        # Weighted average using the token counts per sequence
                        num_tokens = num_tokens[:-num_padding_samples, :]
                        total_local_tokens = num_tokens.sum()
                        mlm_eval_accuracy = (mlm_accuracy_data * num_tokens /
                                             total_local_tokens).sum()

                        if args.POD >= 128:
                            total_global_tokens = sum_distributed_data(
                                total_local_tokens)
                            mlm_eval_accuracy = mlm_eval_accuracy * (
                                total_local_tokens / total_global_tokens)
                            mlm_eval_accuracy = sum_distributed_dataf(
                                mlm_eval_accuracy)

                    elif args.accuracy_averaging_basis == "persequence":
                        # Use the indexing information to slice out non-zero sequences from the packed results
                        mlm_accuracy_data = mlm_accuracy_data[indexing_data]

                        # Average the accuracies
                        mlm_eval_accuracy = mlm_accuracy_data.mean()

                        if args.POD >= 128:
                            total_global_count = sum_distributed_data(
                                eval_sample_count)
                            mlm_eval_accuracy = mlm_eval_accuracy * (
                                eval_sample_count / total_global_count)
                            mlm_eval_accuracy = sum_distributed_dataf(
                                mlm_eval_accuracy)

                    # Go back to using the training optimizer
                    ipu_strategy.set_options({
                        "runtime_options.enable_eval": False
                    })

                    # Log the accuracy
                    mllogger.event(
                        key=mllog.constants.EVAL_ACCURACY,
                        value=mlm_eval_accuracy,
                        metadata={'epoch_num': next_trigger},
                        clear_line=True)

                    # If accuracy not at target, and there are still evaluation points left, cotinue training
                    if mlm_eval_accuracy < mlm_accuracy_target and len(
                            iteration.on_the_spot_validation_triggers) > 0:
                        mllogger.start(
                            mllog.constants.BLOCK_START,
                            None,
                            metadata={
                                "epoch_count": 1,
                                "first_epoch_num": first_epoch_num
                            })
                    logger.info(f"Eval accuracy: {mlm_eval_accuracy:5.3f}")
                    logger.info(
                        f"Evaluation took: {time.time() - start:4.3f} seconds")

                if mlm_eval_accuracy >= mlm_accuracy_target or len(
                        iteration.on_the_spot_validation_triggers) == 0:
                    logger.info("Training Finished")
                    benchmark_duration = time.time() - global_cost

                    logger.info(
                        f"Final eval accuracy: {mlm_eval_accuracy:5.3f}")
                    logger.info(f"Load data cost: %f s" % (load_data_cost))
                    logger.info(f"Total training cost: %f s" %
                                (time.time() - global_train_cost))
                    logger.info(f"Total cost: %f s" % (benchmark_duration))

                    if args.POD >= 128:
                        eval_sample_count = sum_distributed_data(
                            eval_sample_count)
                    mllogger.event(
                        key=mllog.constants.EVAL_SAMPLES,
                        value=eval_sample_count)

                    train_samples = iteration.total_sequences_so_far
                    total_duration_so_far = iteration.total_duration_so_far
                    total_tput = train_samples / total_duration_so_far
                    if args.POD >= 128:
                        logger.info(
                            f"Average throughput on process {args.popdist_rank}: = {total_tput}"
                        )
                        total_tput = sum_distributed_data(total_tput)
                        if (args.popdist_rank == 0):
                            logger.info(
                                f"Average throughput (across all processes combined): = {total_tput}"
                            )
                        train_samples = sum_distributed_data(
                            iteration.total_sequences_so_far)
                    else:
                        logger.info(f"Average throughput = {total_tput}")
                    mllogger.event(
                        key=mllog.constants.TRAIN_SAMPLES, value=train_samples)
                    status = mllog.constants.SUCCESS if mlm_eval_accuracy >= mlm_accuracy_target else mllog.constants.ABORTED
                    mllogger.start(
                        key=mllog.constants.RUN_STOP,
                        metadata={"status": status,
                                  "TTT": benchmark_duration})
                    return

            iteration.flush_total_sequences(batch)
            if args.wandb:
                wandb.log({
                    "final_loss":
                    loss_return[0].sum() / num_token_for_loss.sum(),
                    "accuracy/MLM": np.mean(loss_return[1]),
                    "accuracy/NSP": np.mean(loss_return[2]),
                    "latency/read": read_cost,
                    "latency/train": train_cost,
                    "latency/e2e": total_cost,
                    "optimization/throughput": tput,
                    "global_step": global_step,
                })

            if global_step % args.logging_steps == 0:
                logger.info(
                    "epoch: %d, step: %d, final_loss: %f, acc/mlm: %f, acc/nsp: %f,"
                    "total_cost: %.5f sec, read_cost: %.5f sec, train_cost: %.5f sec, throughput: %.5f seq/s"
                    % (epoch, global_step,
                       loss_return[0].sum() / num_token_for_loss.sum(),
                       np.mean(loss_return[1]), np.mean(loss_return[2]),
                       total_cost, read_cost, train_cost, tput))

            batch_start = time.time()
            global_step += 1


if __name__ == "__main__":

    args = utils.parse_args()

    if args.profile:
        popvision.set_profiling_vars(args.profile_dir, args.profile_instrument)
        popvision.set_logging_vars()
        args_dict = vars(args)
        args_dict["hostname"] = socket.gethostname()
        args_dict["command"] = ' '.join(sys.argv)
        popvision.save_app_info(args_dict)
        logging_handler = popvision.get_profile_logging_handler()
    else:
        logging_handler = None

    setup_logger(logging.getLevelName("INFO"), logging_handler)

    if args.wandb:
        import wandb
        wandb.init(
            project="popart-mlperf-bert",
            settings=wandb.Settings(console='off'),
            name='paddle-mlperf-bert')
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size
        wandb.config.update(args)

    logger.info("Program Start")
    logger.info("Hostname: " + socket.gethostname())
    logger.info("Command Executed: " + str(sys.argv))

    if args.POD >= 128:
        set_distribution_args(args)

    main(args)

    logger.info("Program Finished")
