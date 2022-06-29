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

from init_env import get_context

context = get_context()
num_trainers = None
trainer_id = None

import argparse
import collections
import itertools
import math
import os
import random
import time
import h5py
import json
import distutils.util
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from mlperf_logging.mllog import constants
#from mlperf_logging_helper import paddle_bert_print_start, paddle_bert_print_end, paddle_bert_print_event
from models.mlperf_logging_helper import paddle_bert_print_start, paddle_bert_print_end, paddle_bert_print_event

import numpy as np

if context.is_trainer:
    import utility
    import dataset
    import models
    import paddle
    import paddle.fluid.core as core
    import paddle.distributed.fleet as fleet
    from paddle.fluid.clip import _allow_pure_fp16_global_norm_clip
    from paddle.fluid.contrib.mixed_precision.fp16_utils import _keep_layer_norm_scale_bias_to_fp32
    from paddle.io import DataLoader, Dataset
    from paddle.fluid.layer_helper import LayerHelper

    from models.modeling import BertForPretraining, BertModel, BertPretrainingCriterion
    from models.modeling import BertConfig
    from models.optimization import LinearWarmupPolyDecayScheduler
    from dataset import create_data_holder, create_pretraining_dataset, create_cpu_exchange_padding_pretraining_dataset, create_new_eval_dataset

    try:
        from paddle.incubate.optimizer import DistributedFusedLamb
    except ImportError:
        print('DistributedFusedLamb import error')

    try:
        from custom_setup_ops import custom_lr
    except ImportError as e:
        print('custom_setup_ops import error: {}'.format(e))


def append_lr_op(base_lr, max_step):
    helper = LayerHelper('lr_op')
    step_var = paddle.fluid.layers.create_global_var(
        shape=[1], value=0, dtype='int64', persistable=True)
    lr_var = helper.create_variable_for_type_inference(dtype=np.float32)
    helper.append_op(
        type='custom_lr',
        inputs={'X': [step_var]},
        outputs={'Out': [lr_var]},
        attrs={'base_lr': base_lr,
               'max_step': max_step})
    return lr_var, step_var


def append_acc_merge_op(eval_program, acc, total):
    block = eval_program.global_block()
    acc = block.vars.get(acc.name)
    total = block.vars.get(total.name)
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(eval_program, startup_program):
        acc_step = paddle.static.create_global_var(
            name="eval_step",
            shape=[2],
            value=0,
            dtype=paddle.int64,
            persistable=True,
            force_cpu=True)
        acc_out = paddle.static.create_global_var(
            name="eval_acc",
            shape=[2],
            value=0,
            dtype=paddle.float64,
            persistable=True)
        helper = LayerHelper('acc_merge_op')
        helper.append_op(
            type='acc_merge',
            inputs={'Acc': [acc],
                    'Total': [total]},
            outputs={'Out': [acc_out],
                     'Step': [acc_step]})
    scope = utility.get_scope()
    t = scope.var(acc_step.name).get_tensor()
    set_eval_step_func = lambda v: t.set(np.array([0, v], dtype=np.int64), paddle.CPUPlace())
    set_eval_step_func(0)
    return [acc_out], set_eval_step_func


def save_env(local_file):
    with open(local_file, "w") as f:
        f.write(json.dumps(dict(os.environ), sort_keys=True, indent=2))


def print_flags():
    print(json.dumps(dict(core.globals()), sort_keys=True, indent=2))


def initial_loss_scale():
    return float(os.getenv("INIT_LOSS_SCALE", 2**20))


def str2bool(s):
    return True if distutils.util.strtobool(s) else False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--eval_dir",
        default=None,
        type=str,
        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument(
        "--num_eval_examples",
        default=10000,
        type=int,
        help="number of eval examples to run eval on")
    parser.add_argument(
        "--max_predictions_per_seq",
        default=80,
        type=int,
        help="The maximum total of masked tokens in input sequence")

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay_rate",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--lamb_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--opt_lamb_beta_1", default=0.9, type=float, help="LAMB beta1.")
    parser.add_argument(
        "--opt_lamb_beta_2", default=0.999, type=float, help="LAMB beta2.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Linear warmup step proportion.")
    parser.add_argument(
        "--start_warmup_step",
        default=0,
        type=int,
        help="The default start warmup steps.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--log_freq", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--enable_addto",
        type=str2bool,
        default=False,
        help="Whether to enable the addto strategy.")
    parser.add_argument(
        "--use_pure_fp16",
        type=str2bool,
        default=False,
        help="Whether to use pure fp16 training.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Device for selecting for the training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of merge steps before gradient update."
        "global_batch_size = gradient_accumulation_steps * batch_size.")
    parser.add_argument(
        "--tf_ckpt_path",
        type=str,
        default=None,
        help="The pickled checkpoint path of TensorFlow.")
    parser.add_argument(
        "--bert_config_path",
        type=str,
        default=None,
        help="The bert config path.")
    parser.add_argument(
        "--unpad",
        type=str2bool,
        default=False,
        help="Whether to use unpad optimization.")
    parser.add_argument(
        "--pad",
        type=str2bool,
        default=False,
        help="Whether to use pad optimization.")
    parser.add_argument(
        "--unpad_fmha",
        type=str2bool,
        default=False,
        help="Whether to use unpad_fmha optimization.")
    parser.add_argument(
        "--unpad_fmha_mke_opt",
        type=str2bool,
        default=False,
        help="Whether to use unpad_fmha_mke_opt optimization.")
    parser.add_argument(
        "--pad_fmha",
        type=str2bool,
        default=False,
        help="Whether to use pad_fmha optimization.")
    parser.add_argument(
        "--fused_bias_mha",
        type=str2bool,
        default=False,
        help="Whether to use fused_bias_mha optimization.")
    parser.add_argument(
        "--fused_bias_fc",
        type=str2bool,
        default=False,
        help="Whether to use fused_bias_fc optimization.")
    parser.add_argument(
        "--fused_dropout_add_ln",
        type=str2bool,
        default=False,
        help="Whether to use fused_dropout_add_ln optimization.")
    parser.add_argument(
        "--weight_transpose",
        type=str2bool,
        default=False,
        help="Whether the weight of linear is stored in tranpose format.")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="The max seq length.")
    parser.add_argument(
        "--batch_size", type=int, default=56, help="The batch size.")
    parser.add_argument(
        "--num_epochs_to_generate_seeds_for",
        type=int,
        default=2,
        help="Number of epochs to plan seeds for. Same set across all workers.")
    parser.add_argument(
        "--distributed_lamb",
        type=str2bool,
        default=False,
        help="Whether to use distributed LAMB optimizer.")
    parser.add_argument(
        "--exchange_padding",
        type=str2bool,
        default=False,
        help="Whether to exchange padding across devices.")
    parser.add_argument(
        "--cpu_exchange_padding",
        type=str2bool,
        default=False,
        help="Whether to use CPU to do exchange padding.")
    parser.add_argument(
        "--target_mlm_accuracy",
        type=float,
        default=0.72,
        help="The target mlm accuracy to be coveraged.")
    parser.add_argument(
        "--max_samples_termination",
        type=int,
        default=4500000,
        help="The max samples threshold to terminate the training process.")
    parser.add_argument(
        "--use_uncompressed_dataset",
        type=str2bool,
        default=False,
        help="Whether to use the uncompressed dataset.")
    parser.add_argument(
        "--dense_seq_output",
        type=str2bool,
        default=False,
        help="Whether to use the dense_seq_output optimization.")
    parser.add_argument(
        "--unpad_embed",
        type=str2bool,
        default=False,
        help="Whether to use unpad_embed optimization.")
    parser.add_argument(
        "--sort_eval_data",
        type=str2bool,
        default=False,
        help="Whether to sort the eval data.")
    args = parser.parse_args()
    return args


'''
def select_dataset_file_for_each_worker(files, f_start_id, worker_num,
                                        worker_index):
    """  
    Spliting the train file according to the worker index.
    """
    num_files = len(files)
    if worker_num > num_files:
        remainder = worker_num % num_files
        data_file = files[(
            f_start_id * worker_num + worker_index + remainder * f_start_id) %
                          num_files]
    else:
        data_file = files[(f_start_id * worker_num + worker_index) % num_files]
    return data_file
'''


def create_strategy(args):
    """
    Create build strategy and exec strategy.
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = args.enable_addto
    build_strategy.allow_cuda_graph_capture = True
    if args.distributed_lamb:
        build_strategy.fuse_all_reduce_ops = False
        build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy._NoReduce
    else:
        build_strategy.fuse_all_reduce_ops = True
        build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.AllReduce

    #build_strategy.fuse_gemm_epilogue = True
    build_strategy.fix_op_run_order = True
    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000
    return build_strategy, exec_strategy


def dist_optimizer(args, optimizer):
    """
    Create a distributed optimizer based on a normal optimizer
    """
    build_strategy, exec_strategy = create_strategy(args)

    dist_strategy = fleet.DistributedStrategy()
    if args.distributed_lamb:
        dist_strategy.gradient_scale_configs = {'scale_strategy': 'sum'}
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy
    if args.distributed_lamb:
        dist_strategy.fuse_all_reduce_ops = False
    else:
        dist_strategy.fuse_all_reduce_ops = True

    dist_strategy.fuse_grad_size_in_MB = 0
    if args.use_amp:
        dist_strategy.amp = True
        custom_white_list = [
            'arg_max',
            'concat',
            # 'cumsum',
            'dropout',
            'dropout_grad',
            'elementwise_add',
            'elementwise_add_grad',
            'elementwise_div',
            'elementwise_div_grad',
            'elementwise_max',
            'elementwise_mul',
            'elementwise_mul_grad',
            'elementwise_sub',
            'fill_any_like',
            'fill_constant',
            'gather',
            'gather_grad',
            'gelu',
            'gelu_grad',
            'lamb',
            'layer_norm',
            'layer_norm_grad',
            'lookup_table_v2',
            'lookup_table_v2_grad',
            'matmul',
            'matmul_grad',
            'matmul_v2',
            'matmul_v2_grad',
            # 'reduce_mean',
            # 'reduce_mean_grad',
            'reduce_any',
            # 'reduce_sum',
            # 'reduce_sum_grad',
            'reshape2',
            'reshape2_grad',
            'scatter',
            'scatter_grad',
            'slice',
            'slice_grad',
            'softmax',
            'softmax_grad',
            'softmax_with_cross_entropy',
            'softmax_with_cross_entropy_grad',
            'sqrt',
            'square',
            # 'sum',
            'tanh',
            'tanh_grad',
            'transpose2',
            'custom_fmha',
            'custom_fmha_grad'
            'custom_fused_dropout_residual_ln',
            'custom_fused_dropout_residual_ln_grad',
            'custom_fused_dense',
            'custom_fused_dense_grad',
        ]

        custom_black_list = [
            'reduce_sum',
            'reduce_sum_grad',
        ]

        dist_strategy.amp_configs = {
            'custom_white_list': custom_white_list,
            'custom_black_list': custom_black_list,
            'init_loss_scaling': initial_loss_scale(),
            'incr_every_n_steps': 2000,
            'decr_every_n_nan_or_inf': 1,
            'incr_ratio': 2.0,
            'decr_ratio': 0.5,
            'use_dynamic_loss_scaling': True,
            'use_pure_fp16': args.use_pure_fp16,
            'use_fp16_guard': args.use_pure_fp16,
        }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def prune_exchange_padding_op(eval_program):
    ops = eval_program.global_block().ops
    assert ops[0].type == "sort_bert_inputs_across_devices"
    eval_program.global_block()._remove_op(0)
    eval_program._sync_with_cpp()


def do_eval(exe, eval_program, eval_dataloader, eval_dataloader_cpu_tensor,
            fetch_list):
    acc = None
    n = len(eval_dataloader)
    for i, batch in enumerate(eval_dataloader):
        batch[0]['host_prefix_sum_seq_len'] = eval_dataloader_cpu_tensor[i][0]
        if i + 1 == n:
            acc = exe.run(eval_program, feed=batch, fetch_list=fetch_list)[0]
        else:
            exe.run(eval_program, feed=batch)

    if num_trainers > 1:
        context.trainer_comm.Allreduce(acc, acc)
    return acc[0] / acc[1]


def get_found_nan_inf_flag_var(prog):
    flag_var = None
    for op in prog.global_block().ops:
        if op.type == "update_loss_scaling":
            cur_flag_var = op.input("FoundInfinite")
            assert len(cur_flag_var) == 1
            cur_flag_var = cur_flag_var[0]
            if flag_var is None:
                flag_var = cur_flag_var
                flag_var = prog.global_block().vars.get(flag_var)
                assert flag_var is not None
            else:
                assert flag_var.name == cur_flag_var
    return flag_var


def get_global_step_var(prog):
    flag_var = None
    for op in prog.global_block().ops:
        if op.type == "distributed_fused_lamb":
            cur_flag_var = op.output("Step")
            assert len(cur_flag_var) == 1
            cur_flag_var = cur_flag_var[0]
            if flag_var is None:
                flag_var = cur_flag_var
                flag_var = prog.global_block().vars.get(flag_var)
                assert flag_var is not None
            else:
                assert flag_var.name == cur_flag_var
    return flag_var


def get_loss_scaling_var_names_to_restored(prog):
    def retrieve_var_names(loss_scale_op):
        loss_scale_var_in = loss_scale_op.input("PrevLossScaling")
        loss_scale_var_out = loss_scale_op.output("LossScaling")
        assert len(loss_scale_var_in) == 1 and len(
            loss_scale_var_out) and loss_scale_var_in[0] == loss_scale_var_out[
                0]

        good_steps_var_in = loss_scale_op.input("InGoodSteps")
        good_steps_var_out = loss_scale_op.output("OutGoodSteps")
        assert len(good_steps_var_in) == 1 and len(
            good_steps_var_out) == 1 and good_steps_var_in[
                0] == good_steps_var_out[0]

        bad_steps_var_in = loss_scale_op.input("InBadSteps")
        bad_steps_var_out = loss_scale_op.output("OutBadSteps")
        assert len(bad_steps_var_in) == 1 and len(
            bad_steps_var_out) == 1 and bad_steps_var_in[
                0] == bad_steps_var_out[0]
        return loss_scale_var_in[0], good_steps_var_in[0], bad_steps_var_in[0]

    loss_scale_var, good_steps_var, bad_steps_var = None, None, None
    for op in prog.global_block().ops:
        if op.type == 'update_loss_scaling':
            cur_loss_scale_var, cur_good_steps_var, cur_bad_steps_var = retrieve_var_names(
                op)
            if loss_scale_var is None:
                loss_scale_var = cur_loss_scale_var
                good_steps_var = cur_good_steps_var
                bad_steps_var = cur_bad_steps_var

            assert loss_scale_var == cur_loss_scale_var
            assert good_steps_var == cur_good_steps_var
            assert bad_steps_var == cur_bad_steps_var

    assert loss_scale_var is not None
    return loss_scale_var, good_steps_var, bad_steps_var


def get_loss_scaling_var_values(prog):
    scope = utility.get_scope()
    var_names = get_loss_scaling_var_names_to_restored(prog)
    data = []
    for name in var_names:
        value = np.array(scope.find_var(name).get_tensor())
        data.append(value)
        if trainer_id == 0:
            print('Var {} value is: {}'.format(name, value))
    return data


def restore_loss_scaling_var_values(prog, data, place):
    scope = utility.get_scope()
    var_names = get_loss_scaling_var_names_to_restored(prog)
    assert len(var_names) == len(data)
    for name, np_arr in zip(var_names, data):
        var = scope.find_var(name).get_tensor()
        var.set(np_arr, place)


def paddle_dtype_to_np_dtype(pd_dtype):
    if pd_dtype == paddle.float32:
        return np.float32
    elif pd_dtype == paddle.int64:
        return np.int64
    else:
        raise ValueError('Unsupported tensor dtype {}'.format(pd_dtype))


def inplace_fill_constant(tensor, value):
    dtype = paddle_dtype_to_np_dtype(tensor._dtype())
    shape = tensor.shape()
    np_value = (np.ones(shape) * value).astype(dtype)
    place = tensor._place()
    if place._equals(paddle.CPUPlace()):
        place = paddle.CPUPlace()
    elif place._equals(utility.get_place()):
        place = utility.get_place()
    else:
        raise ValueError("Unsupported tensor place {}".format(place))

    old_ptr = tensor._ptr()
    tensor.set(np_value, place)
    new_ptr = tensor._ptr()
    assert old_ptr == new_ptr


def recover_lamb_status(train_prog, args):
    scope = utility.get_scope()

    def fill_input_value(op, input_slot, value):
        var_names = op.input(input_slot)
        assert len(var_names) == 1
        t = scope.find_var(var_names[0]).get_tensor()
        inplace_fill_constant(t, value)

    def fill_output_value(op, output_slot, value):
        var_names = op.output(output_slot)
        assert len(var_names) == 1
        t = scope.find_var(var_names[0]).get_tensor()
        inplace_fill_constant(t, value)

    found_cnt = 0
    for i in range(train_prog.num_blocks):
        block = train_prog.block(i)
        for op in block.ops:
            if args.distributed_lamb:
                assert op.type != 'lamb'
            else:
                assert op.type != 'distributed_fused_lamb'

            if op.type != 'lamb' and op.type != 'distributed_fused_lamb':
                continue

            fill_input_value(op, "Moment1", 0)
            fill_input_value(op, "Moment2", 0)
            fill_input_value(op, "Beta1Pow", args.opt_lamb_beta_1)
            fill_input_value(op, "Beta2Pow", args.opt_lamb_beta_2)
            if op.type == 'distributed_fused_lamb':
                fill_output_value(op, "Step", 0)
                if args.gradient_accumulation_steps > 1:
                    fill_output_value(op, "AccStep", 0)
            found_cnt += 1

    if args.distributed_lamb:
        assert found_cnt == 1
    else:
        assert found_cnt > 1


def run_warmup(args,
               exe,
               train_prog,
               lr_scheduler,
               eval_prog=None,
               warmup_iter=8):
    train_warmup_data = dataset.prepare_warmup_data(args, args.train_batch_size,
                                                    exe.place)
    list_train_warmup_data = [train_warmup_data]

    paddle.device.cuda.synchronize()
    restored_values = get_loss_scaling_var_values(train_prog)
    old_lr_func = lr_scheduler.__call__
    lr_scheduler.__call__ = lambda self: 0.0
    for _ in range(warmup_iter):
        exe.run(train_prog, feed=list_train_warmup_data)
    lr_scheduler.__call__ = old_lr_func
    restore_loss_scaling_var_values(train_prog, restored_values, exe.place)
    paddle.device.cuda.synchronize()

    # extra run as NV code
    exe.run(train_prog, feed=list_train_warmup_data)
    paddle.device.cuda.synchronize()

    if eval_prog is not None:
        eval_warmup_data = dataset.prepare_warmup_data(
            args, args.eval_batch_size, exe.place)
        for _ in range(warmup_iter):
            exe.run(eval_prog, feed=[eval_warmup_data])
        paddle.device.cuda.synchronize()

    #recover_lamb_moment_pows(train_prog, args)
    recover_lamb_status(train_prog, args)
    paddle.device.cuda.synchronize()
    if utility.get_trainer_id() == 0:
        print('warmup ends')


def do_train(args):
    # Initialize the paddle and paddle fleet execute enviroment
    if trainer_id == 0:
        paddle_bert_print_start(key=constants.INIT_START)
        paddle_bert_print_event(key=constants.SEED, val=args.seed)
        paddle_bert_print_event(
            key=constants.GLOBAL_BATCH_SIZE,
            val=args.train_batch_size * num_trainers *
            args.gradient_accumulation_steps)
        paddle_bert_print_event(key='d_batch_size', val=args.train_batch_size)
        paddle_bert_print_event(
            key=constants.GRADIENT_ACCUMULATION_STEPS,
            val=args.gradient_accumulation_steps)
        paddle_bert_print_event(
            key="max_predictions_per_seq", val=args.max_predictions_per_seq)
        paddle_bert_print_event(
            key='opt_learning_rate_training_steps', val=args.max_steps)
        paddle_bert_print_event(key='num_warmup_steps', val=args.warmup_steps)

    place = utility.get_place()
    fleet.init(is_collective=True)

    worker_num = num_trainers
    worker_index = trainer_id

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    data_holders, data_inputs, data_labels, varlen_info, mlm_label_info = create_data_holder(
        args)

    config = BertConfig.from_json_file(args.bert_config_path)
    #config.fused_mha = args.fused_mha
    #config.fused_gelu_bias = args.fused_gelu_bias
    config.fused_bias_mha = args.fused_bias_mha
    config.fused_bias_fc = args.fused_bias_fc
    #config.fused_bias_fc_loss_head = args.fused_bias_fc_loss_head
    config.dense_seq_output = args.dense_seq_output
    config.unpad = args.unpad
    config.unpad_fmha = args.unpad_fmha
    config.unpad_fmha_mke_opt = args.unpad_fmha_mke_opt
    config.pad_fmha = args.pad_fmha
    config.max_seq_length = args.max_seq_length
    #config.pad = args.pad
    #config.fuse_qkv = not args.disable_fuse_qkv
    #config.fuse_scale = not args.disable_fuse_scale
    #config.fuse_mask = not args.disable_fuse_mask 
    #config.fuse_dropout = args.enable_fuse_dropout
    config.fused_dropout_add_ln = args.fused_dropout_add_ln
    #config.apex_softmax = not args.disable_apex_softmax
    #config.enable_stream = args.enable_stream
    #if config.fuse_mask: config.apex_softmax = True
    #if not config.pad: config.enable_stream = True
    #if config.unpad: config.fused_mha = False

    config.unpad_embed = args.unpad_embed

    config.weight_transpose = args.weight_transpose
    config.batch_size = args.batch_size

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    int16_max = np.iinfo(np.int16).max
    assert config.vocab_size <= int16_max
    assert args.train_batch_size * args.max_seq_length <= int16_max
    assert args.eval_batch_size * args.max_seq_length <= int16_max

    def build_model():
        model = BertForPretraining(BertModel(config), config)
        criterion = BertPretrainingCriterion(config)

        prediction_scores, seq_relationship_score = model(*data_inputs)
        func = criterion(prediction_scores, seq_relationship_score,
                         *data_labels)
        return model, func

    if args.use_pure_fp16:
        with paddle.static.amp.fp16_guard():
            model, func = build_model()
    else:
        model, func = build_model()
    loss, mlm_acc, num_masked = func()

    eval_program = main_program.clone(for_test=True)
    if args.gpu_exchange_padding and utility.use_nv_input(
    ) and num_trainers > 1:
        prune_exchange_padding_op(eval_program)

    # Define the dynamic learing_reate scheduler and optimizer
    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_dataloader) * args.num_train_epochs

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step

    lr_scheduler = LinearWarmupPolyDecayScheduler(
        startup_warmup_steps=warmup_start,
        warmup_steps=warmup_steps,
        total_steps=args.max_steps,
        base_lr=args.learning_rate,
        end_lr=0.0,
        degree=1.0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    if args.max_grad_norm > 0:
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm)
    else:
        grad_clip = None
    exclude_from_weight_decay_fn = lambda var: var.name not in decay_params

    lr_var, step_var = append_lr_op(args.learning_rate, args.max_steps)
    optimizer_kwargs = {
        'learning_rate': lr_var,
        'lamb_weight_decay': args.weight_decay_rate,
        'epsilon': args.lamb_epsilon,
        'exclude_from_weight_decay_fn': exclude_from_weight_decay_fn,
        'grad_clip': grad_clip,
        'beta1': args.opt_lamb_beta_1,
        'beta2': args.opt_lamb_beta_2,
    }

    if trainer_id == 0:
        paddle_bert_print_event(
            key=constants.OPT_BASE_LR, val=args.learning_rate)
        paddle_bert_print_event(
            key="opt_learning_rate_warmup_steps", val=args.warmup_steps)
        paddle_bert_print_event(
            key="start_warmup_step", val=args.start_warmup_step)

    _keep_layer_norm_scale_bias_to_fp32(False)
    _allow_pure_fp16_global_norm_clip(True)
    if args.distributed_lamb:
        # Note: 
        # (1) broadcast_master_param is true means use fp32 to update parameter.
        # (2) if 'is_grad_scaled_by_nranks' is false, must set 'dist_strategy.gradient_scale_configs = {'scale_strategy': 'sum'}'.
        #optimizer_kwargs.update({"broadcast_master_param": True, "clip_after_allreduce": False, "is_grad_scaled_by_nranks": False})
        optimizer_kwargs.update({
            "use_master_param_norm": True,
            "clip_after_allreduce": False,
            "is_grad_scaled_by_nranks": False,
        })
        if args.gradient_accumulation_steps > 1:
            optimizer_kwargs.update({
                "gradient_accumulation_steps": args.gradient_accumulation_steps
            })
        optimizer = DistributedFusedLamb(**optimizer_kwargs)
        optimizer._set_step(step_var)
    else:
        optimizer = paddle.optimizer.Lamb(**optimizer_kwargs)
        if args.use_pure_fp16:
            optimizer._multi_precision = True

    if hasattr(optimizer, "_get_parameter"):
        get_param_func = optimizer._get_parameter
    else:
        get_param_func = lambda name: utility.get_scope().find_var(name).get_tensor()

    if trainer_id == 0:
        paddle_bert_print_event(key='opt_epsilon', val=args.lamb_epsilon)
        paddle_bert_print_event(key='opt_lamb_beta_1', val=args.opt_lamb_beta_1)
        paddle_bert_print_event(key='opt_lamb_beta_2', val=args.opt_lamb_beta_2)
        paddle_bert_print_event(
            key='opt_lamb_weight_decay_rate', val=args.weight_decay_rate)

    # Use the fleet api to compile the distributed optimizer
    optimizer = dist_optimizer(args, optimizer)
    optimizer.minimize(loss)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    if args.use_amp:
        optimizer.amp_init(place, test_program=eval_program, use_fp16_test=True)

    eval_fetch_list, set_eval_step = append_acc_merge_op(eval_program, mlm_acc,
                                                         num_masked)

    with open('startup_program_{}.txt'.format(trainer_id), 'w') as f:
        f.write(str(startup_program))

    with open('main_program_{}.txt'.format(trainer_id), 'w') as f:
        f.write(str(main_program))

    with open('eval_program_{}.txt'.format(trainer_id), 'w') as f:
        f.write(str(eval_program))

    found_nan_inf = get_found_nan_inf_flag_var(main_program)
    found_global_step_val = get_global_step_var(main_program)
    if args.use_amp:
        assert found_nan_inf is not None
    else:
        assert found_nan_inf is None
    assert found_global_step_val is not None

    build_strategy, exec_strategy = create_strategy(args)
    eval_program = paddle.static.CompiledProgram(
        eval_program).with_data_parallel(
            build_strategy=build_strategy, exec_strategy=exec_strategy)
    run_warmup(
        args,
        exe,
        main_program,
        lr_scheduler,
        eval_prog=eval_program,
        warmup_iter=8)

    ckpt = None
    if args.tf_ckpt_path:
        if trainer_id == 0:
            print('starts to load tf checkpoint...')
            start_t = time.time()
        ckpt = model.load_tf_ckpt(args, get_param_func)
        paddle.device.cuda.synchronize()
        if trainer_id == 0:
            end_t = time.time()
            print('checkpoint loading ends, time cost: {}s'.format(end_t -
                                                                   start_t))

    def save_ckpt(ckpt_saved_path):
        if trainer_id == 0 and ckpt is not None:
            dirname = 'log_{}'.format(num_trainers)
            ckpt_saved_path = os.path.join(dirname, ckpt_saved_path)
            ckpt.save(ckpt_saved_path, get_param_func)
            print('PaddlePaddle ckpt saved into: {}'.format(ckpt_saved_path))

    # save_ckpt('loaded_ckpt_0.pickled')

    pool = ThreadPoolExecutor(1)
    # pre-compute eval boundaries
    samples_trained_per_step = args.train_batch_size * num_trainers * args.gradient_accumulation_steps
    eval_interval_samples = 0.05 * (230.23 * samples_trained_per_step + 3000000)
    eval_interval_samples = int(eval_interval_samples / 25000) * 25000
    start, stop, step = eval_interval_samples, args.max_samples_termination, eval_interval_samples
    eval_steps = [
        math.ceil(i / samples_trained_per_step)
        for i in np.arange(start, stop, step)
    ]
    eval_count = 0
    next_eval_step = eval_steps[eval_count]
    next_eval_training_step = next_eval_step * args.gradient_accumulation_steps

    create_train_dataset = create_cpu_exchange_padding_pretraining_dataset if args.cpu_exchange_padding else create_pretraining_dataset

    context.trainer_comm.Barrier()
    if trainer_id == 0:
        paddle_bert_print_end(key=constants.INIT_STOP)
        paddle_bert_print_start(key=constants.RUN_START)

    f_start_id = 0
    train_dataloader, train_dataloader_cpu_tensor = create_train_dataset(
        data_holders, f_start_id, tolist=False)

    if args.eval_dir:
        eval_dataset_future = pool.submit(create_new_eval_dataset, data_holders)
    need_next_training_shard = args.train_batch_size * args.gradient_accumulation_steps * args.max_steps > 10000

    global_step = 0
    skipped_steps = 0
    training_steps = 0
    now_skipped = 0
    now_step = 0
    skip_interval = 0
    average_loss = 0.0
    end_training, converged = False, False
    coveraged = False
    epoch = 1  # to be consistent with NV

    file_num = context.file_num()

    while global_step < args.max_steps and not end_training:
        if trainer_id == 0:
            paddle_bert_print_start(
                key=constants.EPOCH_START, metadata={'epoch_num': epoch})
            paddle_bert_print_start(
                key=constants.BLOCK_START,
                metadata={'first_epoch_num': epoch,
                          'epoch_count': 1})
            now_time = time.time()

        empty_fetch_list = []
        full_fetch_list = [loss, mlm_acc, found_global_step_val]
        for f_id in range(f_start_id + 1, file_num):
            if need_next_training_shard:
                dataset_future = pool.submit(create_train_dataset, data_holders,
                                             f_id)
            # limin-todo:
            # limin_file_start = time.time()
            for step, batch in enumerate(train_dataloader):

                training_steps += 1
                update_step = training_steps % args.gradient_accumulation_steps == 0
                batch[0][
                    'host_prefix_sum_seq_len'] = train_dataloader_cpu_tensor[
                        step][0]

                if training_steps < next_eval_training_step:
                    cur_fetch_list = empty_fetch_list
                else:
                    cur_fetch_list = full_fetch_list
                return_val = exe.run(main_program,
                                     feed=batch,
                                     fetch_list=cur_fetch_list)

                if training_steps >= next_eval_training_step:
                    loss_return, mlm_acc_return, found_global_step_return = return_val

                if update_step:
                    if training_steps >= next_eval_training_step:
                        global_step = found_global_step_return[0]
                        has_nan_inf_step = (
                            training_steps / args.gradient_accumulation_steps -
                            global_step)
                        #print("has_nan_inf_step: ", has_nan_inf_step)
                        if has_nan_inf_step:
                            skipped_steps = has_nan_inf_step
                    else:
                        global_step = int(training_steps /
                                          args.gradient_accumulation_steps)

                    if args.eval_dir and global_step == next_eval_step:
                        skip_interval = skipped_steps
                        now_skipped = skip_interval
                        if True:
                            if eval_count == 0:
                                eval_dataloader, eval_dataloader_cpu_tensor = eval_dataset_future.result(
                                    timeout=None)
                                eval_dataloader = list(eval_dataloader)
                                set_eval_step(len(eval_dataloader))
                            samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * num_trainers
                            samples_trained_prev = samples_trained

                            eval_start_t = time.time()
                            eval_avg_mlm_accuracy = do_eval(
                                exe, eval_program, eval_dataloader,
                                eval_dataloader_cpu_tensor, eval_fetch_list)
                            eval_end_t = time.time()
                            # if trainer_id == 0:
                            #    print('Eval time: {} s'.format(eval_end_t -
                            #                                   eval_start_t))

                            if args.target_mlm_accuracy > 0 and eval_avg_mlm_accuracy >= args.target_mlm_accuracy:
                                end_training, converged = True, True
                                coveraged = True
                                if trainer_id == 0:
                                    print(
                                        "%f > %f, Target MLM Accuracy reached at global_step %d , training_step %d"
                                        % (eval_avg_mlm_accuracy,
                                           args.target_mlm_accuracy,
                                           global_step, training_steps))
                            eval_count += 1
                            next_eval_step = eval_steps[eval_count]
                            next_eval_training_step = next_eval_step * args.gradient_accumulation_steps
                            if trainer_id == 0:
                                paddle_bert_print_event(
                                    key=constants.EVAL_ACCURACY,
                                    val=eval_avg_mlm_accuracy,
                                    metadata={'epoch_num': epoch})
                                print({
                                    "global_steps": global_step,
                                    "eval_mlm_accuracy": eval_avg_mlm_accuracy
                                })

                if args.log_freq > 0 and training_steps % (
                        args.log_freq * args.gradient_accumulation_steps) == 0:
                    samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * num_trainers
                    if trainer_id == 0:
                        time_interval = time.time() - now_time
                        step_interval = global_step - now_step
                        now_time = time.time()
                        now_step = global_step
                        training_perf = args.train_batch_size * args.gradient_accumulation_steps * num_trainers * (
                            step_interval + skip_interval) / time_interval
                        skip_interval = 0
                        divisor = args.gradient_accumulation_steps
                        #now_lr = lr_scheduler()
                        #step_loss_to_print = loss_return[0] * args.gradient_accumulation_steps / divisor
                        #average_loss_to_print = average_loss / (args.log_freq * divisor)
                        print({
                            "training_steps": training_steps,
                            #"average_loss": average_loss_to_print,
                            #"step_loss": step_loss_to_print,
                            #"learning_rate": now_lr,
                            "seq/s": training_perf,
                            "global_steps": now_step,
                            "samples_trained": samples_trained,
                            "skipped_steps": now_skipped,
                            "timestamp": now_time,
                        })

                        paddle_bert_print_event(
                            key='tracked_stats',
                            val={
                                'seq/s': training_perf,
                                #'step_loss': step_loss_to_print,
                                #'avg_loss': average_loss_to_print,
                                #'lr': now_lr,
                            },
                            metadata={"step": (epoch, training_steps)})
                        paddle_bert_print_event(
                            key='throughput', val=training_perf)

                    average_loss = 0.0

                if global_step >= args.max_steps or end_training:
                    status = 'success' if coveraged else 'abort'
                    end_training = True
                    break

            # limin-todo:
            #limin_file_end = time.time()
            #print("limin: one file time: ", limin_file_end - limin_file_start)
            del train_dataloader

            if samples_trained >= args.max_samples_termination or end_training:
                status = 'success' if converged else 'aborted'
                end_training = True
                break

            if not need_next_training_shard:
                dataset_future = pool.submit(create_train_dataset, f_id)
            train_dataloader, train_dataloader_cpu_tensor = dataset_future.result(
                timeout=None)

        if trainer_id == 0:
            paddle_bert_print_end(
                key=constants.BLOCK_STOP, metadata={'first_epoch_num': epoch})
            paddle_bert_print_end(
                key=constants.EPOCH_STOP, metadata={'epoch_num': epoch})

        epoch += 1

    if trainer_id == 0:
        paddle_bert_print_event(constants.TRAIN_SAMPLES, val=samples_trained)
        paddle_bert_print_event(
            constants.EVAL_SAMPLES, val=args.num_eval_examples)
        paddle_bert_print_end(
            key=constants.RUN_STOP, metadata={'status': status})

        #save_ckpt('saved_ckpt_final.pickled')


def do_read(args):
    while context.read_file():
        pass


if __name__ == "__main__":
    args = parse_args()
    if not args.exchange_padding:
        args.cpu_exchange_padding = False
        args.gpu_exchange_padding = False
    else:
        args.gpu_exchange_padding = not args.cpu_exchange_padding

    context.init_args(args)
    trainer_id = context.trainer_id
    num_trainers = context.trainer_num

    if context.is_trainer:
        if trainer_id == 0:
            paddle_bert_print_event(key=constants.SUBMISSION_ORG, val="Baidu")
            paddle_bert_print_event(
                key=constants.SUBMISSION_PLATFORM, val="1 x NVIDIA A100 GPU")
            paddle_bert_print_event(
                key=constants.SUBMISSION_DIVISION, val="closed")
            paddle_bert_print_event(
                key=constants.SUBMISSION_STATUS, val="onprem")
            paddle_bert_print_event(
                key=constants.SUBMISSION_BENCHMARK, val="bert")
            paddle_bert_print_event(key=constants.CACHE_CLEAR, val=True)
        save_env('./paddle_env_{}.json'.format(trainer_id))
        if trainer_id == 0:
            print(args)
            print_flags()
        if args.use_pure_fp16:
            assert args.use_amp, "--use_amp must be True if --use_pure_fp16 is True"
        paddle.enable_static()
        with paddle.static.scope_guard(utility.get_scope()):
            do_train(args)
    else:
        if args.cpu_exchange_padding:
            do_read(args)
    if args.cpu_exchange_padding:
        context.stop_reader()
    print('Process ends')
