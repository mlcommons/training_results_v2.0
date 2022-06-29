# coding=utf-8
# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language     verning permissions and
# limitations under the License.

"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import h5py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import logging
import math
import multiprocessing
import numpy as np
import os
import random
import re
import time

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
#from modeling import BertForPretraining, BertConfig
from schedulers import LinearWarmupPolyDecayScheduler

import utils

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import amp_C

import apex_C
from apex import amp
from apex.amp import _amp_state
from apex.optimizers import FusedLAMB
from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel.distributed import flat_dist_call
from apex.multi_tensor_apply import multi_tensor_applier

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertConfig
from modeling import BertForPreTrainingSegmented
from modeling import BertForPretraining
from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler

import mlperf_logger
from scaleoutbridge import init_bridge, ScaleoutBridge as SBridge

from mhalib import *

from fwd_loss_bwd_trainer import FwdLossBwdTrainer, preprocess_batch
from torch.cuda.amp import GradScaler
grad_scaler = GradScaler(init_scale=float(os.getenv("INIT_LOSS_SCALE", 2**20)), growth_interval=2000)

# Global variables
skipped_steps = 0
cached_batches = []

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

def get_eval_batchsize_per_worker(args):
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        remainder = args.num_eval_examples % torch.distributed.get_world_size()
        if rank<remainder:
            return (chunk_size+1)
        else:
            return chunk_size

def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init_fn, synthetic_input=False):
    if synthetic_input: # synthetic, in-memory dataset only for performance testing
        train_data = synthetic_dataset(input_file=input_file, max_pred_length=max_pred_length, max_seq_length=args.max_seq_length)
    else:
        train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length, max_seq_length=args.max_seq_length, packed_samples=args.packed_samples)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=0 if args.train_batch_size<=8 else 4, worker_init_fn=worker_init_fn,
        pin_memory=True, drop_last=True)

    return train_dataloader, input_file

def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(pretraining_dataset(eval_file_path, max_pred_length=args.max_predictions_per_seq, max_seq_length=args.max_seq_length))
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[:args.num_eval_examples]
                break
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        remainder = args.num_eval_examples % torch.distributed.get_world_size()
        if rank<remainder:
            eval_data = eval_data[(chunk_size+1)*rank : (chunk_size+1)*(rank+1)]
        else:
            eval_data = eval_data[chunk_size*rank+remainder : chunk_size*(rank+1)+remainder]

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         num_workers=0 if min(chunk_size, args.eval_batch_size)<=10 else 4, worker_init_fn=worker_init_fn, pin_memory=True)

    return eval_dataloader

class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_seq_length, packed_samples=False):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.packed_samples = packed_samples

        f = h5py.File(input_file, "r")
        if not self.packed_samples:
            keys = ['input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_labels']
        else:
            keys = ['input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'packed_input_len', 'packed_masked_lm_len', 'next_sentence_labels', ]

        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        input_mask= np.zeros((self.max_seq_length)).astype(np.int64)
        segment_ids=np.zeros((self.max_seq_length)).astype(np.int64)
        next_sentence_labels=np.zeros((3)).astype(np.int64)
        packed_input_len = np.zeros((3)).astype(np.int64)

        if not self.packed_samples:
            [_input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids, _next_sentence_labels] = [
                input[index].astype(np.int64) if indice < 4 else 
                np.asarray(input[index].astype(np.int64)) for indice, input in enumerate(self.inputs)]
        else:
            [_input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids, _packed_input_len, _packed_masked_lm_len, _next_sentence_labels] = [
                input[index].astype(np.int64) for indice, input in enumerate(self.inputs)]
        
        input_mask_len = _input_ids.shape[-1]
        input_ids[:input_mask_len] = _input_ids
        input_mask[:input_mask_len] = np.ones((1,input_mask_len)).astype(np.int64)        
        segment_ids[:input_mask_len] = _segment_ids
        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64)
        masked_lm_labels[ _masked_lm_positions] = _masked_lm_ids

        if not self.packed_samples:
            next_sentence_labels = _next_sentence_labels

            return [torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_labels), torch.from_numpy(next_sentence_labels)]
        else:
            packed_seqs = _packed_input_len.shape[-1]
            next_sentence_labels[:packed_seqs] = _next_sentence_labels
            packed_input_len[:packed_seqs] = _packed_input_len

            return [torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_labels), torch.from_numpy(next_sentence_labels),
                    torch.from_numpy(packed_input_len)]

class synthetic_dataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_seq_length, number_of_samples=100):
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.samples = []
        self.number_of_samples = number_of_samples
 
        for _ in range(number_of_samples):
            input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
            input_mask= np.zeros((self.max_seq_length)).astype(np.int64)
            segment_ids=np.zeros((self.max_seq_length)).astype(np.int64)
            next_sentence_labels=np.asarray(np.int64(1))
            
            input_mask_len = torch.randint(max_pred_length+1,max_seq_length,(1,))            
            input_ids[:input_mask_len] = torch.randint(2048,30000,(input_mask_len,))
            input_mask[:input_mask_len] = np.ones((1,input_mask_len)).astype(np.int64)
            segment_ids[:input_mask_len] = np.zeros((1,input_mask_len)).astype(np.int64)
            masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64)
            masked_count = torch.randint(max_pred_length, (1,))
            masked_lm_labels[ torch.randint(max_pred_length, (masked_count,))] = torch.randint(2048,30000,(masked_count,))
            self.samples.append([torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_labels), torch.from_numpy(next_sentence_labels)])

    def __len__(self):
        return self.number_of_samples*10000

    def __getitem__(self, index):
        return self.samples[index % self.number_of_samples]

def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--packed_samples",
                        default=False,
                        action="store_true",
                        required=False,
                        help="Indicate whether the samples in .hdf5 files contain packed sequences")

    parser.add_argument("--max_pack_factor",
                        default=3,
                        type=int,
                        required=False,
                        help="Upto how many sequences can be packed within a sample.")

    parser.add_argument("--average_packing_rate",
                        default=2,
                        type=int,
                        required=False,
                        help="Average number of sequences per batch.")

    parser.add_argument("--synthetic_input",
                        default=False,
                        action='store_true',
                        help="Whether to use synthetic, in-memory dataset - used only in performance measurements")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--cuda_graph_mode", default="segmented", type=str,
                        help="'segmented' or 'full_iteration' options for CUDA graph capture. \n"
                        "'segmented' option: Pytorch Autograd orchestrates execution of backward ops every iteration. \n"  
                        "'full_iteration' option: CUDA graph orchestrates execution of bwd ops every iteration without \
                        Autograd involvement (has composability limitations but could be more performant allowing optimizer \
                        and collectives capture).")

    parser.add_argument("--max_iterations_per_graph",
                        default=4,
                        type=int,
                        help="Maximum number of iterations to capture in a single graph. Requires 'full_iteration' option  \
                                for '--cuda_graph_mode'.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--eval_iter_start_samples",
                        default=3000000,
                        type=int,
                        help="Sample to begin performing eval.")
    parser.add_argument("--eval_iter_samples",
                        default=-1,
                        type=int,
                        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training")
    parser.add_argument("--num_eval_examples",
                        default=10000,
                        type=int,
                        help="number of eval examples to run eval on")
    parser.add_argument("--cache_eval_data",
                        default=False,
                        action='store_true',
                        help="whether to cache evaluation data on GPU")

    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument("--init_tf_checkpoint",
                        default=None,
                        type=str,
                        help="The initial TF checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=76,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=18,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for LAMB.")
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help="weight decay rate for LAMB.")
    parser.add_argument("--opt_lamb_beta_1",
                        default=0.9,
                        type=float,
                        help="LAMB beta1.")
    parser.add_argument("--opt_lamb_beta_2",
                        default=0.999,
                        type=float,
                        help="LAMB beta2.")
    parser.add_argument("--max_steps",
                        default=1536,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--max_samples_termination",
                        default=14000000,
                        type=float,
                        help="Total number of training samples to run.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=float,
                        help="Number of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--start_warmup_step",
                        default=0,
                        type=float,
                        help="Starting step for warmup. ")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', 0),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss. If not positive, no logging is provided for training loss')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint")
    parser.add_argument('--keep_n_most_recent_checkpoints',
                        type=int,
                        default=20,
                        help="Number of checkpoints to keep (rolling basis).")
    parser.add_argument('--num_samples_per_checkpoint',
                        type=int,
                        default=500000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--min_samples_to_start_checkpoints',
                        type=int,
                        default=3000000,
                        help="Number of update steps until model checkpoints start saving to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Only required for checkpoint saving format")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--exchange_padding",
                        default=False,
                        action='store_true',
                        help="Whether to run with unpadding.")
    parser.add_argument("--unpad",
                        default=False,
                        action='store_true',
                        help="Whether to run with unpadding.")
    parser.add_argument("--unpad_fmha",
                        default=False,
                        action='store_true',
                        help="Whether to run fmha with unpadding.")
    parser.add_argument("--pad_fmha",
                        default=False,
                        action='store_true',
                        help="Whether to run fmha with padding.")
    parser.add_argument("--pad",
                        default=False,
                        action='store_true',
                        help="Whether to pad tokens.")
    parser.add_argument("--enable_fuse_dropout",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of attention mask to softmax and dropout.")
    parser.add_argument("--disable_fuse_mask",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the attention mask to softmax.")
    parser.add_argument("--disable_fuse_scale",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the scaling to BMM1.")
    parser.add_argument("--disable_fuse_qkv",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the QKV GEMMs.")
    parser.add_argument("--disable_apex_softmax",
                        default=False,
                        action='store_true',
                        help="Whether to disable apex softmax.")
    parser.add_argument("--enable_stream",
                        default=False,
                        action='store_true',
                        help="Enable use of streams for pad case.")
    parser.add_argument("--fused_gemm_gelu",
                        default=False,
                        action='store_true',
                        help="Whether to fuse gemm and gelu together.")
    parser.add_argument("--fused_mha",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--no-fused_mha",
                        dest='fused_mha',
                        action='store_false',
                        help="Disable fused MHA optimization.")
    parser.add_argument("--fused_gelu_bias",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_dropout_add",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_bias_mha",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_bias_fc",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_bias_fc_loss_head",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--dense_seq_output",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--target_mlm_accuracy',
                        type=float,
                        default=0.0,
                        help="Stop training after reaching this Masked-LM accuracy")
    parser.add_argument('--train_mlm_accuracy_window_size',
                        type=int,
                        default=0,
                        help="Average accuracy over this amount of batches before performing a stopping criterion test")
    parser.add_argument('--num_epochs_to_generate_seeds_for',
                        type=int,
                        default=2,
                        help="Number of epochs to plan seeds for. Same set across all workers.")

    parser.add_argument("--use_cuda_graph",
                        default=False,
                        action='store_true',
                        help="Enable CUDA graph execution.")
    parser.add_argument("--use_ddp",
                        default=False,
                        action='store_true',
                        help="Enable DDP.")
    parser.add_argument("--ddp_type",
                        default='apex',
                        type=str,
                        help="DDP type: 'apex' or 'native'.")
    parser.add_argument("--use_gradient_as_bucket_view",
                        default=False,
                        action='store_true',
                        help="Turn ON gradient_as_bucket_view optimization in native DDP.")
    parser.add_argument("--bypass_amp",
                        default=False,
                        action='store_true',
                        help="Bypass AMP unscaling and inf/nan checks for SOL measurements.")
    parser.add_argument('--distributed_lamb',
                        default=False,
                        action='store_true',
                        help="Whether to use distributed lamb.")
    parser.add_argument('--dwu-group-size', '--dwugs',
                        default=0,
                        type=int,
                        metavar='DWUGS',
                        help='distributed weight update group size. If arg is 0, defaults to one node')
    parser.add_argument('--dwu-num-blocks',
                        '--dwunb',
                        default=1,
                        type=int,
                        metavar='DWUNB',
                        help='number of blocks in dwu scheme')
    parser.add_argument('--dwu-num-chunks',
                        '--dwunc',
                        default=1,
                        type=int,
                        metavar='DWUNC',
                        help='number of chunks in dwu scheme')
    parser.add_argument('--dwu-num-rs-pg',
                        '--dwurspg',
                        default=2,
                        type=int,
                        metavar='DWURSPG',
                        help='number of reduction-scatter streams in dwu scheme')
    parser.add_argument('--dwu-num-ar-pg',
                        '--dwuarpg',
                        default=4,
                        type=int,
                        metavar='DWUARPG',
                        help='number of all-reduce streams in dwu scheme')
    parser.add_argument('--dwu-num-ag-pg',
                        '--dwuagpg',
                        default=2,
                        type=int,
                        metavar='DWUAGPG',
                        help='number of all-gather streams in dwu scheme')
    parser.add_argument('--dwu-overlap-reductions',
                        default=False,
                        action='store_true',
                        help='whether to overlap reductions with backprop')
    parser.add_argument('--dwu-e5m2-allgather',
                        default=False,
                        action='store_true',
                        help='do allgather with e5m2 floats')
    args = parser.parse_args()

    # Check we've been given a checkpoint
    assert args.init_checkpoint is not None or args.init_tf_checkpoint is not None or found_resume_checkpoint(args), \
        "Must specify --init_checkpoint, --init_tf_checkpoint or have ckpt to resume from in --output_dir of the form *.pt"

    assert not (args.init_checkpoint is not None and args.init_tf_checkpoint is not None), \
            "Can only specify one of --init_checkpoint and --init_tf_checkpoint"

    assert not (args.exchange_padding and args.packed_samples), \
        "Cannot balance batch load (exchange_padding) in case samples are packed"
    
    assert args.dwu_overlap_reductions or (args.dwu_num_blocks==1 and args.dwu_num_chunks==1), \
        "With overlap reductions turned off, dwu_num_chunks and dwu_num_blocks should be set to 1"

    # if not using packed dataset, we have only one sequence per sample
    if not args.packed_samples:
        args.max_pack_factor = 1
        args.average_packing_rate = 1

    return args

# Returns true only if resuming from a checkpoint found in output_dir.
# init_checkpoint and init_tf_checkpoint are not considered
def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    return args.resume_from_checkpoint and len(glob.glob(os.path.join(args.output_dir, checkpoint_str))) > 0

def setup_training(args):
    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        assert False, "code path not tested with cuda graphs"
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = torch.distributed.get_world_size()

    args.device = device
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    #torch.cuda.init()
    #torch.cuda.enable_manual_allocations(False)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not (args.do_train or (args.eval_dir and args.eval_iter_samples <= 0)):
        raise ValueError(" `do_train`  or should be in offline eval mode")

    if not args.resume_from_checkpoint or not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def remap_attn_parameters(model_dict, config):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'attention' in k:
            if 'self.query.weight' in k:
                new_k = k.replace('self.query.weight', 'multi_head_attention.q_weight')
            elif 'self.key.weight' in k:
                new_k = k.replace('self.key.weight', 'multi_head_attention.k_weight')
            elif 'self.value.weight' in k:
                new_k = k.replace('self.value.weight', 'multi_head_attention.v_weight')
            elif 'self.query.bias' in k:
                new_k = k.replace('self.query.bias', 'multi_head_attention.q_bias')
            elif 'self.key.bias' in k:
                new_k = k.replace('self.key.bias', 'multi_head_attention.k_bias')
            elif 'self.value.bias' in k:
                new_k = k.replace('self.value.bias', 'multi_head_attention.v_bias')
            elif 'output.dense.weight' in k:
                new_k = k.replace('output.dense.weight', 'multi_head_attention.out_proj_weight')
            elif 'output.dense.bias' in k:
                new_k = k.replace('output.dense.bias', 'multi_head_attention.out_proj_bias')
            elif 'output.LayerNorm.weight' in k:
                new_k = k.replace('output.LayerNorm.weight', 'layer_norm.weight')
            elif 'output.LayerNorm.bias' in k:
                new_k = k.replace('output.LayerNorm.bias', 'layer_norm.bias')
            elif 'bert.encoder.layer' in k and config.fused_gemm_gelu:
                new_k = k.replace('intermediate.dense.weight','intermediate.dense.weight1')
                new_k = k.replace('intermediate.dense.bias','intermediate.dense.bias1')
                new_k = k.replace('output.dense.weight','intermediate.dense.weight2')
                new_k = k.replace('output.dense.bias','intermediate.dense.bias2')
            else:
                new_k = k
        else:
            new_k = k    
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

def remap_segmented_model_parameters(model_dict, config):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'bert' in k:
            new_k = 'bert_model_segment.' + k
        elif 'cls' in k:
            new_k = 'heads_only_segment.' + k
        else:
            assert False, "shouldn't happen"
        if config.fused_bias_fc_loss_head and new_k == "heads_only_segment.cls.predictions.bias":
            new_k = "heads_only_segment.cls.predictions.decoder.bias"
        if config.fused_gemm_gelu:
            if 'bert.encoder.layer' in new_k and 'intermediate.dense.weight' in new_k and 'attention' not in new_k :
                new_k = new_k.replace('intermediate.dense.weight','intermediate.dense.weight1')
            elif 'bert.encoder.layer' in new_k and 'intermediate.dense.bias' in new_k  and 'attention' not in new_k :
                new_k = new_k.replace('intermediate.dense.bias','intermediate.dense.bias1')
            elif 'bert.encoder.layer' in new_k and 'output.dense.weight' in new_k   and 'attention' not in new_k:
                new_k = new_k.replace('output.dense.weight','intermediate.dense.weight2')
            elif 'bert.encoder.layer' in new_k and 'output.dense.bias' in new_k   and 'attention' not in new_k:
                new_k = new_k.replace('output.dense.bias','intermediate.dense.bias2')
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

def prepare_model_and_optimizer(args, device, stream):
    global_step = 0
    args.resume_step = 0
    checkpoint = None

    config = BertConfig.from_json_file(args.bert_config_path)
    config.fused_mha = args.fused_mha
    config.fused_gelu_bias = args.fused_gelu_bias
    config.fused_bias_mha = args.fused_bias_mha
    config.fused_bias_fc = args.fused_bias_fc
    config.fused_bias_fc_loss_head = args.fused_bias_fc_loss_head
    config.fused_gemm_gelu = args.fused_gemm_gelu
    config.dense_seq_output = args.dense_seq_output
    config.unpad = args.unpad
    config.unpad_fmha = args.unpad_fmha
    config.pad_fmha = args.pad_fmha
    config.max_seq_length = args.max_seq_length
    config.pad = args.pad
    config.fuse_qkv = not args.disable_fuse_qkv
    config.fuse_scale = not args.disable_fuse_scale
    config.fuse_mask = not args.disable_fuse_mask
    config.fuse_dropout = args.enable_fuse_dropout
    config.fused_dropout_add = args.fused_dropout_add
    config.apex_softmax = not args.disable_apex_softmax
    config.enable_stream = args.enable_stream
    if config.fuse_mask == True: config.apex_softmax = True
    if config.pad == False: config.enable_stream = True
    if config.unpad == True: config.fused_mha = False
    config.packed_samples = args.packed_samples

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Load from Pyt checkpoint - either given as init_checkpoint, or picked up from output_dir if found
    if args.init_checkpoint is not None or found_resume_checkpoint(args):
        # Prepare model
        #model = BertForPreTraining(config)
        model = BertForPreTrainingSegmented(config)

        # for k,v in model.state_dict().items():
        #     print(f'model-k,len(v)={k}, {v.numel()}')

        #model = BertForPretraining(config)
        if args.init_checkpoint is None: # finding checkpoint in output_dir
            assert False, "code path not tested with cuda graphs"
            checkpoint_str = "phase2_ckpt_*.pt" if args.phase2 else "phase1_ckpt_*.pt"
            model_names = [f for f in glob.glob(os.path.join(args.output_dir, checkpoint_str))]
            global_step = max([int(x.split('.pt')[0].split('_')[-1].strip()) for x in model_names])
            args.resume_step = global_step #used for throughput computation

            resume_init_checkpoint = os.path.join(args.output_dir, checkpoint_str.replace("*", str(global_step)))
            print("Setting init checkpoint to %s - which is the latest in %s" %(resume_init_checkpoint, args.output_dir))
            checkpoint=torch.load(resume_init_checkpoint, map_location="cpu")
        else:
            checkpoint=torch.load(args.init_checkpoint, map_location="cpu")["model"]

        #Log weight initializations
        for weight in utils.convert_weight_names(list(checkpoint.keys())):
            mlperf_logger.log_event(mlperf_logger.constants.WEIGHTS_INITIALIZATION, metadata={'tensor': weight})

        # Fused MHA requires a remapping of checkpoint parameters
        if config.fused_mha:
            #assert False, "code path not tested with cuda graphs"
            checkpoint_remapped = remap_attn_parameters(checkpoint, config)
            checkpoint_remapped = remap_segmented_model_parameters(checkpoint_remapped, config)
        else:
            checkpoint_remapped = remap_segmented_model_parameters(checkpoint, config)

        model.load_state_dict(checkpoint_remapped, strict=True)
    else: #Load from TF Checkpoint
        assert False, "code path not tested with cuda graphs"
        #model = BertForPreTraining.from_pretrained(args.init_tf_checkpoint, from_tf=True, config=config)

    model.to(device)
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer_grouped_parameters_names = [
        {'params': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)]},
        {'params': [n for n, p in param_optimizer if any(nd in n for nd in no_decay)]}]

    mlperf_logger.log_event(key=mlperf_logger.constants.OPT_BASE_LR,
                            value=args.learning_rate, sync=False)
    full_ar = torch.cuda.device_count() < torch.distributed.get_world_size()
    clip_after_ar = args.dwu_overlap_reductions
    set_param_views_to_flat_buffer = not args.dwu_overlap_reductions

    if args.distributed_lamb:
        #from optim import distributed_fused_lamb
        ## overwrite methods for gradient-clipping-before-allreduce support
        #DistributedFusedLAMB._pipeline_block_reductions=distributed_fused_lamb._pipeline_block_reductions_patched
        #DistributedFusedLAMB._pipeline_step=distributed_fused_lamb._pipeline_step_patched
        optimizer = DistributedFusedLAMB(optimizer_grouped_parameters, lr=args.learning_rate,
                betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
                eps=1e-6,
                max_grad_norm=1.0,
                overlap_reductions=args.dwu_overlap_reductions,
                dwu_group_size=args.dwu_group_size,
                dwu_num_blocks=args.dwu_num_blocks,
                dwu_num_chunks=args.dwu_num_chunks,
                dwu_num_rs_pg=args.dwu_num_rs_pg,
                dwu_num_ar_pg=args.dwu_num_ar_pg,
                dwu_num_ag_pg=args.dwu_num_ag_pg,
                use_nvlamb=False, clip_after_ar=clip_after_ar, fused_norm=True, fuse_scale=True, full_ar=full_ar, set_param_views_to_flat_buffer=set_param_views_to_flat_buffer,
                #use_nvlamb=False, clip_after_ar=False, fused_norm=True, fuse_scale=True, full_ar=full_ar, set_param_views_to_flat_buffer=False,
                e5m2_allgather=args.dwu_e5m2_allgather)
        optimizer.set_global_scale(float(os.getenv("INIT_LOSS_SCALE", 2**20)))
    else:
        optimizer = FusedLAMB(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2))


    # Prevent following communicators to lock the tree
    os.environ["NCCL_SHARP_DISABLE"] = "1"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"

    mlperf_logger.log_event(key='opt_epsilon', value=optimizer.defaults['eps'],
                            sync=False)
    b1, b2 = optimizer.defaults['betas']
    mlperf_logger.log_event(key='opt_lamb_beta_1', value=b1, sync=False)
    mlperf_logger.log_event(key='opt_lamb_beta_2', value=b2, sync=False)
    mlperf_logger.log_event(key='opt_lamb_weight_decay_rate',
                            value=optimizer.defaults['weight_decay'],
                            sync=False)

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step
    lr_scheduler = LinearWarmupPolyDecayScheduler(optimizer, start_warmup_steps=warmup_start, warmup_steps=warmup_steps,
                                                  total_steps=args.max_steps, end_learning_rate=0.0, degree=1.0)
    

    # Only for SOL testing
    if args.fp16 and args.bypass_amp:
        model.half()

    if args.fp16 and not args.bypass_amp:
        if args.distributed_lamb:
            model.half()
#            optimizer._init_everything()
        elif args.fp16:
            if args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", master_weights=True)
            else:
                assert False, "code path not tested with cuda graphs"
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale)
            amp._amp_state.loss_scalers[0]._loss_scale = float(os.getenv("INIT_LOSS_SCALE", 2**20))

 
    if found_resume_checkpoint(args):
        assert False, "code path not tested with cuda graphs"
        optimizer.load_state_dict(checkpoint['optimizer']) #restores m,v states (only if resuming checkpoint, not for init_checkpoint and init_tf_checkpoint for now)

        # Restore AMP master parameters
        if args.fp16 and not args.distributed_lamb:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  

        # the following optimization makes sure parameters are laid out in a flat buffer to avoid flatten/unflatten copies for optimizer
        # this is not tested for non-FMHA codepath, so only enabling for FMHA codepath
        if set_param_views_to_flat_buffer and (args.pad_fmha or args.unpad_fmha):
            p_names=[]
            l1=optimizer_grouped_parameters_names[0]['params']
            l2=optimizer_grouped_parameters_names[1]['params']
            for n in l1:
                p_names.append(n)
            for n in l2:
                p_names.append(n)
    
            p_offset=0
            prev=None
            param_storage = optimizer._new_params.storage()
            buffer_w_offsets=[]
            buffer_b_offsets=[]
            for i, p in enumerate(optimizer._model_params):
                p_param_size = p.numel()
                if 'self.Wq' in p_names[i]:
                    buffer_w_offsets.append(p_offset)
                if 'self.Bq' in p_names[i]:
                    buffer_b_offsets.append(p_offset)
                #if 'self.Wq' in p_names[i] or 'self.Wk' in p_names[i] or 'self.Wv' in p_names[i] or 'self.Bq' in p_names[i] or 'self.Bk' in p_names[i] or 'self.Bv' in p_names[i]  :
                #    continue
                with torch.no_grad():
                    p.set_(source=param_storage, storage_offset=p_offset, size=p.size())
                p_offset += p_param_size
                if prev is not None and (prev.data_ptr() + prev.numel() * prev.element_size() != p.data_ptr()):
                    p_offset = ((p_offset + 63) // 64) * 64
                prev = p
            for i in range(24):
                size_tmp = model.bert_model_segment.bert.encoder.layer[i].attention.self.Wqkv.size()
                model.bert_model_segment.bert.encoder.layer[i].attention.self.Wqkv.set_(source=param_storage, storage_offset=buffer_w_offsets[i], size=size_tmp)
                size_tmp = model.bert_model_segment.bert.encoder.layer[i].attention.self.Bqkv.size()
                model.bert_model_segment.bert.encoder.layer[i].attention.self.Bqkv.set_(source=param_storage, storage_offset=buffer_b_offsets[i], size=size_tmp)
        model.load_state_dict(checkpoint_remapped, strict=True)
    return model, optimizer, lr_scheduler, checkpoint, global_step        

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    global skipped_steps
    if args.allreduce_post_accumulation and args.use_cuda_graph:
        assert False, "code path not tested with cuda graphs"
    if args.distributed_lamb:
        optimizer.set_global_scale(grad_scaler._get_scale_async())
        optimizer.complete_reductions()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        found_inf = optimizer._overflow_buf # GPU tensor
        skipped_steps += found_inf
        global_step += 1 # increment anyways

    elif args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        #torch.nn.utils.clip_grad_norm_(parameters=amp.master_params(optimizer), max_norm=1.0, norm_type=2.0)
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (torch.distributed.get_world_size() * args.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overflow_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            if utils.is_main_process():
                print("Overflow detected, reduced loss_scaler to %f" % (scaler.loss_scale()))
            skipped_steps += 1
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
    else:
        optimizer.step()
        if args.bypass_amp:
            global_step += 1
        else:
            global_step += 0 if _amp_state.loss_scalers[0]._has_overflow else 1
        #global_step += 0 if _amp_state.loss_scalers[0]._has_overflow else 1
#    if args.distributed_lamb:
#        for param in model.parameters():
#            param.grad = None

    return global_step

def run_graphed_eval(args, graph_eval, batch_placeholder, eval_dataloader, loss_eval, mlm_accuracy_eval, num_valid_eval, first_eval=False):
    if first_eval:
        # there should be only 1 eval iteration if code reaches here
        for batch in eval_dataloader:
            #print(f"loading eval batch")
            batch = preprocess_batch(args, *batch)
            for t,t_gpu in zip(batch, batch_placeholder):
                t_gpu.copy_(t, non_blocking=True)
    graph_eval.replay()
    mlm_accuracy_eval *= num_valid_eval
    loss_eval *= num_valid_eval
    if torch.distributed.is_initialized():
        #Collect total scores from all ranks
        torch.distributed.all_reduce(mlm_accuracy_eval, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(loss_eval, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_valid_eval, op=torch.distributed.ReduceOp.SUM)

    # Average by number of examples
    mlm_accuracy_eval /= num_valid_eval
    loss_eval /= num_valid_eval

    return loss_eval.item(), mlm_accuracy_eval.item()

def run_eval(args, model, trainer, eval_dataloader, device, num_eval_examples, first_eval=False, use_cache=False):
    model.eval()

    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0.0

    # on first eval, load and cache data on GPU
    if first_eval and use_cache:
        for batch in eval_dataloader:
            batch = preprocess_batch(args, *batch)
            cached_batches.append([t.to(device) for t in batch])

    with torch.no_grad():
        for batch in cached_batches if use_cache else eval_dataloader:
            if not use_cache: 
                batch = preprocess_batch(args, *batch)
                batch = [t.to(device) for t in batch]
        #    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        #    loss, mlm_acc, num_masked = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
        #                          masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
            loss, mlm_acc, num_masked = trainer.eval_step(batch, model)
            total_eval_loss += loss * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked
            # TODO should this sync be here?
            torch.cuda.synchronize()
    model.train()

    #total_eval_mlm_acc and total_eval_loss are already tensors, total_masked is not
    #total_masked = torch.tensor(total_masked, device=device, dtype=torch.int64)

    if torch.distributed.is_initialized():
        #Collect total scores from all ranks
        torch.distributed.all_reduce(total_eval_mlm_acc, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_eval_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_masked, op=torch.distributed.ReduceOp.SUM)

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss.item(), total_eval_mlm_acc.item()

def exchange_padding_fast(device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, max_batch_size):
    torch.cuda.nvtx.range_push('exchangepadding')
    pad_size = max_batch_size - input_ids.shape[0]
    if pad_size > 0:
        input_ids = F.pad(input_ids, (0, 0, 0, pad_size))
        segment_ids = F.pad(segment_ids, (0, 0, 0, pad_size))
        input_mask = F.pad(input_mask, (0, 0, 0, pad_size))
        masked_lm_labels = F.pad(masked_lm_labels, (0, 0, 0, pad_size))
        next_sentence_labels = F.pad(next_sentence_labels, (0, pad_size))
    ngpus = torch.distributed.get_world_size()
    nseqs = input_mask.shape[0]
    ntokensperseq = input_mask.shape[1]
    igpu = torch.distributed.get_rank()

    flattened_length_seq = nseqs * ntokensperseq
    flattened_length_nsp = nseqs

    def get_local_packet_size():
        return 4 * flattened_length_seq + flattened_length_nsp

    # Storing tensors in same order as arguments
    def encode_packet(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels):

        packet = torch.zeros([get_local_packet_size()], device=device, dtype=torch.int16)
        
        curr_pos = 0

        packet[curr_pos:curr_pos + flattened_length_seq] = input_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_seq] = segment_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_seq] = input_mask.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_seq] = masked_lm_labels.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_nsp] = next_sentence_labels.view(-1)[:]

        return packet

    def decode_packet(flat_packet):
        packet = flat_packet.view(ngpus, get_local_packet_size())

        curr_pos = 0

        input_ids_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        segment_ids_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        input_mask_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        masked_lm_labels_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        next_sentence_labels_ = packet[:, curr_pos:curr_pos + flattened_length_nsp].contiguous().view(ngpus, nseqs)

        return input_ids_, segment_ids_, input_mask_, masked_lm_labels_, next_sentence_labels_

    tensors = encode_packet(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)

    tensors_ = torch.zeros([ngpus, get_local_packet_size()], device=device, dtype=torch.float16)
    tensors_ = list(torch.split(tensors_, 1))

    torch.distributed.all_gather(tensors_, tensors.view(torch.float16))

    tensors_ = torch.stack(tensors_).view(torch.int16).long()
    input_ids_, segment_ids_, input_mask_, masked_lm_labels_, next_sentence_labels_ = decode_packet(tensors_)

    seqlens_, indices = torch.sort(input_mask_.sum(dim=2).view(-1), descending=True)



    if pad_size > 0:
        input_ids_sorted = input_ids_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        segment_ids_sorted = segment_ids_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        input_mask_sorted = input_mask_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        masked_lm_labels_sorted = masked_lm_labels_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        next_sentence_labels_sorted = next_sentence_labels_.view(ngpus * nseqs)[indices[:]]
        # we need to remove the empty sequences we added to the batch
        valid_idx = seqlens_.view(nseqs, ngpus)[:, igpu] > 0
        input_ids_sorted = input_ids_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        segment_ids_sorted = segment_ids_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        input_mask_sorted = input_mask_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_sorted.view(nseqs, ngpus)[valid_idx, igpu].contiguous()
    else:
        indices_ = indices.view(nseqs, ngpus)[:, igpu]
        input_ids_sorted = input_ids_.view(nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        segment_ids_sorted = segment_ids_.view(nseqs* ngpus, ntokensperseq)[indices_, :].contiguous()
        input_mask_sorted = input_mask_.view(nseqs* ngpus, ntokensperseq)[indices_, :].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_.view(nseqs* ngpus, ntokensperseq)[indices_, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_.view(nseqs * ngpus)[indices_].contiguous()

    torch.cuda.nvtx.range_pop()
    return input_ids_sorted, segment_ids_sorted, input_mask_sorted, masked_lm_labels_sorted, next_sentence_labels_sorted



def main():
    args = parse_arguments()
    status = 'aborted'  # later set to 'success' if termination criteria met

    mlperf_logger.log_start(key=mlperf_logger.constants.INIT_START,
                            log_all_ranks=True, sync=False)

    # if args.use_env and 'LOCAL_RANK' in os.environ:
    #     args.local_rank = int(os.environ['LOCAL_RANK'])


    device, args = setup_training(args)

    mlperf_logger.mlperf_submission_log('bert')

    mlperf_logger.log_event(key=mlperf_logger.constants.SEED, value=args.seed,
                            sync=False)
    mlperf_logger.log_event(key=mlperf_logger.constants.GLOBAL_BATCH_SIZE,
                            value=global_batch_size(args), sync=False)
    mlperf_logger.log_event(key='d_batch_size',
                            value=args.train_batch_size, sync=False)
    mlperf_logger.log_event(key=mlperf_logger.constants.GRADIENT_ACCUMULATION_STEPS,
                            value=args.gradient_accumulation_steps, sync=False)
    mlperf_logger.log_event(key='max_predictions_per_seq',
                            value=args.max_predictions_per_seq, sync=False)
    mlperf_logger.log_event(key='opt_learning_rate_training_steps',
                            value=args.max_steps, sync=False)
    mlperf_logger.log_event(key='num_warmup_steps',
                            value=int(args.warmup_proportion*args.max_steps) if args.warmup_steps==0 else args.warmup_steps,
                            sync=False)

    if utils.is_main_process():
        print("parsed args:")
        print(args)
    # Prepare optimizer
    capture_stream = torch.cuda.Stream()
    model, optimizer, lr_scheduler, checkpoint, global_step = prepare_model_and_optimizer(args, device, capture_stream)
    
    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.num_epochs_to_generate_seeds_for, device)
    worker_seed = worker_seeds[torch.distributed.get_rank()]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)

    # run a dummy optimizer step for dist lamb
    if 0:#args.distributed_lamb:
        dummy_data = [
            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            torch.ones(args.train_batch_size, args.max_predictions_per_seq, dtype=torch.int64, device=device),
            torch.ones(args.train_batch_size, args.max_predictions_per_seq, dtype=torch.int64, device=device),
            torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
            ]
        loss, _, _ = model(input_ids=dummy_data[0], token_type_ids=dummy_data[1], attention_mask=dummy_data[2], masked_lm_labels=dummy_data[4], next_sentence_label=dummy_data[5])
        grad_scaler.scale(loss).backward()
        optimizer.complete_reductions()
    
    current_device = torch.device('cuda', torch.cuda.current_device())
    samples_trained = torch.zeros((1,), dtype=torch.int, device=current_device) # global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu
    global_samples_trained = torch.zeros((1,), dtype=torch.int, device=current_device) 

    if args.unpad:
        assert not args.use_cuda_graph, "code path not tested with cuda graphs"
        torch.cuda.synchronize()
        InitMHACUDAExtension()
        torch.cuda.synchronize()
    if args.train_batch_size>6:
        assert not args.use_cuda_graph, "training batch size larger than 6 is not graph capturable"
    final_loss = float("inf")
    train_time_raw = float("inf")
    raw_train_start = time.time()

    if args.do_train:
        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 1
        training_steps = 0
        end_training, converged = False, False
        samples_trained_prev = 0

        # pre-compute eval boundaries
        samples_trained_per_step = args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu * args.average_packing_rate
        start, stop, step = args.eval_iter_start_samples, args.max_samples_termination, args.eval_iter_samples
        eval_steps = [math.ceil(i/samples_trained_per_step) for i in np.arange(start, stop, step)]
        eval_count = 0
        next_eval_step = eval_steps[eval_count]
        pool = ProcessPoolExecutor(1)

        if args.target_mlm_accuracy:
            if args.train_mlm_accuracy_window_size > 0:
                accuracy_scores = []
                avg_mlm_accuracy = torch.Tensor([0]).cuda()


        first_epoch = True
        if found_resume_checkpoint(args):
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            num_files = len(files)
        else:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                     os.path.isfile(os.path.join(args.input_dir, f)) and 'part_' in f]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch]).shuffle(files)
            f_start_id = 0

        # - CUDA graph ------
        #assert args.gradient_accumulation_steps == 1 and args.fp16 and not args.fused_mha, "code path not tested with cudagraphs"
        assert (args.gradient_accumulation_steps == 1 and args.fp16) or not args.use_cuda_graph, "code path not tested with cudagraphs"
        assert (args.local_rank != -1 and not args.allreduce_post_accumulation) or not args.use_cuda_graph, "code path not tested with cudagraphs"
        assert args.checkpoint_activations == False, "code path not tested with cudagraphs"

        use_cuda_graph = args.use_cuda_graph
        use_DDP = args.use_ddp

#        if utils.get_world_size() > 1:
#            assert use_DDP, "Running multi-GPU without --use_ddp"

        if args.use_gradient_as_bucket_view:
            assert args.ddp_type == 'native', \
                "DDP type must be set to 'native' for --use_gradient_as_bucket_view"

        global skipped_steps
        skipped_steps = torch.zeros((1,), dtype=torch.int, device=current_device)
        skipped_samples = torch.zeros((1,), dtype=torch.int, device=current_device)
        actual_batch = torch.zeros((1,), dtype=torch.int, device=current_device)

        graphs={} #dictionary that holds palceholder and graphs for particular number of sequences in a batch
        if not args.packed_samples : # only a single graph/placeholder
            batch_gpu_placeholder = preprocess_batch(args,
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
                    )
            graphs[args.train_batch_size] = {'graph':None, 'placeholder':batch_gpu_placeholder}        
        else: # one graph/placeholder per sequnece count
            for seq_count in range(args.train_batch_size,args.train_batch_size*args.max_pack_factor+1):
                batch_gpu_placeholder = preprocess_batch(args,
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(seq_count, dtype=torch.int64, device=device),
                    torch.ones(seq_count, dtype=torch.int64, device=device),
                    )
                graphs[seq_count]={'graph': None, 'placeholder': batch_gpu_placeholder}


        graphs_multi={}
        if use_cuda_graph and args.cuda_graph_mode=='full_iteration':
            if not args.packed_samples:
                batch_gpu_placeholder_multi = [
                            preprocess_batch(args,                 
                            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                            torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                            torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
                            ) for _ in range(args.max_iterations_per_graph)]
                graphs_multi[args.train_batch_size] = {'graph': None, 'placeholder': batch_gpu_placeholder_multi}
            else:
                for seq_count in range(args.train_batch_size,args.train_batch_size*args.max_pack_factor+1):
                    batch_gpu_placeholder_multi = [
                        preprocess_batch(args,                 
                        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
                        torch.ones(seq_count, dtype=torch.int64, device=device),
                        torch.ones(seq_count, dtype=torch.int64, device=device),
                        ) for _ in range(args.max_iterations_per_graph)]
                    graphs_multi[seq_count]={'graph': None, 'placeholder': batch_gpu_placeholder_multi}

            eval_batch = get_eval_batchsize_per_worker(args)
            #eval_batch = 16 #get_eval_batchsize_per_worker(args)

            #print(f"eval_batch: {eval_batch}")
            batch_gpu_placeholder_eval = preprocess_batch(                    
                    args,
                    torch.ones(eval_batch, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(eval_batch, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(eval_batch, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(eval_batch, args.max_seq_length, dtype=torch.int64, device=device),
                    torch.ones(eval_batch, dtype=torch.int64, device=device),
                    )
        else:#segmenter option, can also be used when use_cuda_graph is disabled
            fwd_loss_bwd_trainer = FwdLossBwdTrainer(args, grad_scaler)
            model = fwd_loss_bwd_trainer.capture_bert_model_segment_graph(model, use_cuda_graph)
            current_device = torch.device('cuda', torch.cuda.current_device())
            # skipped_steps = torch.zeros((1,), dtype=torch.int, device=current_device)

        if use_DDP and not args.distributed_lamb:
            if args.ddp_type == 'native':
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                                device_ids=[args.local_rank],
                                                                bucket_cap_mb=8192,
                                                                gradient_as_bucket_view=args.use_gradient_as_bucket_view)
            elif args.ddp_type == 'apex':
                model = DDP(model,
                            message_size=250000000,
                            delay_allreduce=True,
                            gradient_predivide_factor=torch.distributed.get_world_size())
            else:
                assert False, "Invalid DDP type"
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )
            if args.cuda_graph_mode=='segmented' or not use_cuda_graph:
                loss, mlm_acc, _ = fwd_loss_bwd_trainer.step(-1,
                                                              graphs[args.train_batch_size]['placeholder'],
                                                              model,
                                                              optimizer)
                optimizer.zero_grad()    
                overflow_buf = None
                if args.allreduce_post_accumulation:
                    overflow_buf = torch.cuda.IntTensor([0])
                optimizer.set_global_scale(grad_scaler._get_scale_async())
                optimizer.complete_reductions()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                found_inf = optimizer._overflow_buf # GPU tensor
                # skipped_steps = torch.zeros((1,), dtype=torch.int, device=current_device)
            else:
                current_device = torch.device('cuda', torch.cuda.current_device())
                ambient_stream = torch.cuda.current_stream()
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(ambient_stream)
                loss_train = torch.zeros((1,), dtype=torch.float, device=current_device)
                from model.layers.activations import fast_gelu_impl
                with torch.cuda.stream(capture_stream):  
                    #do a warmup
                    # for i in range(3):
                    #     loss, _, _ = model(*graphs[args.train_batch_size]['placeholder'])
                    #     optimizer._lazy_init_stage1()
                    #     grad_scaler.scale(loss).backward()
                    if args.fused_gemm_gelu:
                        for i in range(10):
                            input_t = torch.ones(args.train_batch_size,512,1024).cuda().half()
                            input_t.requires_grad=True
                            output_t = fast_gelu_impl(input_t)
                            doutput_t = torch.ones_like(output_t)
                            output_t.backward(doutput_t)
                    loss, _, _ = model(*graphs[args.train_batch_size]['placeholder'])
                    optimizer._lazy_init_stage1()
                    grad_scaler.scale(loss).backward()
                    optimizer._lazy_init_stage2()
                    optimizer.zero_grad()
                    optimizer.set_global_scale(grad_scaler._get_scale_async())
                    optimizer.complete_reductions()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    found_inf = optimizer._overflow_buf # GPU tensor
                    skipped_steps += found_inf
                    #below is for graph capture
                    capture_stream.synchronize()
                capture_stream.wait_stream(ambient_stream)
                    # graph = torch.cuda.CUDAGraph()
                if use_cuda_graph :#and ( args.max_iterations_per_graph == 1 ): 
                    for seq_count in range(args.train_batch_size, args.train_batch_size*args.max_pack_factor+1):                   
                        graph = torch.cuda.CUDAGraph()
                        optimizer.zero_grad(set_to_none=True)
                        with torch.cuda.graph(graph):
                            loss, mlm_acc, _ = model(*(graphs[seq_count]['placeholder']))
                            samples_trained += graphs[seq_count]['placeholder'][-2].shape[0]
                            loss_train.copy_(loss)
                            optimizer._lazy_init_stage1()
                            grad_scaler.scale(loss).backward()
                            optimizer._lazy_init_stage2()
                            static_scale = grad_scaler._scale
                            lr_scheduler.step()
                            optimizer.set_global_scale(grad_scaler._get_scale_async())
                            optimizer.complete_reductions()
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            static_scale.copy_(grad_scaler._scale)
                            found_inf = optimizer._overflow_buf # GPU tensor
                            skipped_steps += found_inf
                            skipped_samples += found_inf*graphs[seq_count]['placeholder'][-2].shape[0] #current_samples
                        #capture_stream.synchronize()
                        #capture_stream.wait_stream(ambient_stream)
                        graphs[seq_count]['graph'] = graph
                    # graph_multi = torch.cuda.CUDAGraph()
                #    if use_cuda_graph and ( args.max_iterations_per_graph > 1 ):
                    grad_scaler.update(new_scale=static_scale)
                    for seq_count in range(args.train_batch_size, args.train_batch_size*args.max_pack_factor+1): 
                        graph_multi = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph_multi):
                            for i in range(args.max_iterations_per_graph):
                                optimizer.zero_grad(set_to_none=True)
                                loss, mlm_acc, _ = model(*(graphs_multi[seq_count]['placeholder'][i]))
                                samples_trained += graphs_multi[seq_count]['placeholder'][i][-2].shape[0]
                                optimizer._lazy_init_stage1()
                                grad_scaler.scale(loss).backward()
                                optimizer._lazy_init_stage2()
                                # static_scale = grad_scaler._scale
                                lr_scheduler.step()
                                optimizer.set_global_scale(grad_scaler._get_scale_async())
                                optimizer.complete_reductions()
                                grad_scaler.step(optimizer)
                                grad_scaler.update()
                                #static_scale.copy_(grad_scaler._scale)
                                found_inf = optimizer._overflow_buf # GPU tensor
                                skipped_steps += found_inf
                                skipped_samples += found_inf*graphs_multi[seq_count]['placeholder'][i][-2].shape[0] #current_samples
                            static_scale.copy_(grad_scaler._scale)
                        # capture_stream.synchronize()
                        graphs_multi[seq_count]['graph'] = graph_multi
                    # Capture evaluation graph
                    graph_eval = torch.cuda.CUDAGraph()
                    mlm_accuracy_eval = torch.zeros((1,), dtype=torch.float, device=current_device)
                    loss_eval = torch.zeros((1,), dtype=torch.float, device=current_device)
                    num_valid_eval = torch.zeros((1,), dtype=torch.int64, device=current_device)
                    torch.cuda.synchronize()
                    if use_cuda_graph:
                        model.eval()
                        # run a warmup iteration for JIT kernels
                        loss, mlm_acc, num_valid = model(*batch_gpu_placeholder_eval)
                        loss, mlm_acc, num_valid = model(*batch_gpu_placeholder_eval)
                        with torch.cuda.graph(graph_eval):
                            loss, mlm_acc, num_valid = model(*batch_gpu_placeholder_eval)
                            loss_eval.copy_(loss)
                            mlm_accuracy_eval.copy_(mlm_acc)
                            num_valid_eval.copy_(num_valid)
                    model.train()

                torch.cuda.current_stream().wait_stream(capture_stream)

        accumulated_batches = 0 # define ahead so we can use within the hook
        if args.cuda_graph_mode == 'segmented' or not use_cuda_graph:
            # we assum that parameter buffers are stable from this point and capture them in the hook
            parameters = list(model.parameters())

            def grads_to_none_hook(module, input, output):
                #torch.cuda.nvtx.range_push(f'p.grad=None loop')
                #bert_mha_train.grad_to_none(parameters)
                for p in parameters:
                    if p.requires_grad and accumulated_batches == 1 :
                        p.grad = None
                #torch.cuda.nvtx.range_pop()
            model.bert_model_segment.register_forward_hook(grads_to_none_hook)


        mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP, sync=False)
        mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START, sync=True)
        mlperf_logger.barrier()

        # Start prefetching eval dataset
        if args.eval_dir:
            eval_dataset_future = pool.submit(create_eval_dataset, args, worker_init_fn=worker_init)
        # comparing to number of samples in a shard. There are ~38k samples in 4096-way shard, comparing to 10k to be safe
        need_next_training_shard = args.train_batch_size * args.gradient_accumulation_steps *  args.max_steps > 10000 

        now_step, now_skipped, skip_interval = 0, 0, 0

        sbridge = init_bridge(torch.distributed.get_rank())
        while global_step < args.max_steps and not end_training:
            mlperf_logger.log_start(key=mlperf_logger.constants.EPOCH_START,
                                    metadata={'epoch_num': global_samples_trained.item()}, sync=False)
            mlperf_logger.log_start(key=mlperf_logger.constants.BLOCK_START,
                                    metadata={'first_epoch_num': epoch,
                                              'epoch_count': 1},
                                    sync=False)
            sbridge.start_epoch_prof()
            if utils.is_main_process():
                print("parsed args:")
                print(args)

                now_time = time.time()

                print("epoch:", epoch)

            # Reshuffle file list on subsequent epochs
            if not first_epoch:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
                files.sort()
                num_files = len(files)
                random.Random(shuffling_seeds[epoch]).shuffle(files)
                f_start_id = 0

            first_epoch = False

            shared_file_list = {}
            sbridge.start_prof(SBridge.LOAD_TIME)

            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                remainder = torch.distributed.get_world_size() % num_files
                data_file = files[(f_start_id*torch.distributed.get_world_size() + torch.distributed.get_rank() +
                                   remainder * f_start_id) % num_files]
            else:
                data_file = files[(f_start_id*torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files]

            mlperf_logger.log_event(key='data_file', value=data_file, sync=False)
            if args.synthetic_input:
                train_data = synthetic_dataset(data_file, args.max_predictions_per_seq, args.max_seq_length)
            else:
                train_data = pretraining_dataset(data_file, args.max_predictions_per_seq, args.max_seq_length, args.packed_samples)

            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size, num_workers=0 if args.train_batch_size<=8 else 4, worker_init_fn=worker_init, pin_memory=True,drop_last=True)

            overflow_buf = None
            if args.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])
            #global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)
#            if args.distributed_lamb:
#                optimizer._init_everything()
            now_lr = 0.0
            stats_stream = torch.cuda.Stream()
            send_lr_in_parallel = False
            lr_cpu = torch.tensor([0.0], dtype=torch.float32, device='cpu').pin_memory()
            sbridge.stop_prof(SBridge.LOAD_TIME)

            for f_id in range(f_start_id + 1, len(files)):
                if torch.distributed.get_world_size() > num_files:
                    data_file = files[(f_id*torch.distributed.get_world_size() + torch.distributed.get_rank() +
                                       remainder * f_id) % num_files]
                else:
                    data_file = files[(f_id*torch.distributed.get_world_size() + torch.distributed.get_rank())%num_files]

                if need_next_training_shard:
                    dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init_fn=worker_init, synthetic_input=args.synthetic_input)

                if use_cuda_graph and args.cuda_graph_mode=='full_iteration':
                    forming_multi_iter_batch=False
                    samples_in_multi_iter_batch=0

                accumulated_batches = 0
                #torch.cuda.cudart().cudaProfilerStart()
                for step, batch in enumerate(train_dataloader):
                    batch = preprocess_batch(args, *batch)
                    actual_batch = batch[-2].shape[0]                    
                    assert actual_batch >= args.train_batch_size, "Batch underflow"
                    # samples_trained += actual_batch
                    training_steps += 1

                    accumulated_batches += 1
                    update_step = (accumulated_batches % args.gradient_accumulation_steps == 0)

                    if accumulated_batches == 1 and not args.distributed_lamb: # distributed_lamb zeros gradinetss in a forward hook
                        optimizer.zero_grad()

                    if args.distributed_lamb:
                        optimizer.set_is_accumulation_step(not update_step)
                        optimizer.set_last_step(step == len(train_dataloader) - 1)

                    if args.exchange_padding == True:                        
                        batch = [t.to(device, non_blocking=True, dtype=torch.int16) for t in batch]
                        batch = exchange_padding_fast(device, *batch, args.train_batch_size)
                    elif args.cuda_graph_mode=='segmented' or not use_cuda_graph:
                        batch = [t.to(device, non_blocking=True) for t in batch]
                    else: # full_iteration capture mode
                        #for capturing multiple iterations in a graph, we copy the corresponding number of samples into the placeholder memory location.
                        #once the corresponding number of samples are ready, we replay the graph. We only run multi-iteration graph if the next evaluation
                        #is sufficiently far in terms of number of steps
                        sbridge.start_prof(SBridge.FWD_BWD_TIME)
                        if forming_multi_iter_batch:
                            for t,t_gpu in zip(batch, graphs_multi[actual_batch]['placeholder'][samples_in_multi_iter_batch]):
                                t_gpu.copy_(t, non_blocking=True)
                            samples_in_multi_iter_batch += 1
                            if samples_in_multi_iter_batch<args.max_iterations_per_graph:
                                continue
                            else:
                                samples_in_multi_iter_batch=0
                                forming_multi_iter_batch = False
                                #torch.cuda.synchronize()
                                # graph_multi.replay()
                                graphs_multi[actual_batch]['graph'].replay()
                                global_step += args.max_iterations_per_graph
                        else:
                            if args.max_iterations_per_graph>1 and next_eval_step-global_step>=args.max_iterations_per_graph and args.max_steps-global_step>=args.max_iterations_per_graph:
                                forming_multi_iter_batch = True
                                for t,t_gpu in zip(batch, graphs_multi[actual_batch]['placeholder'][0]):
                                    t_gpu.copy_(t, non_blocking=True)
                                samples_in_multi_iter_batch = 1
                                continue
                            else:
                                for t,t_gpu in zip(batch, graphs[actual_batch]['placeholder']):
                                    t_gpu.copy_(t, non_blocking=True)
                                torch.cuda.synchronize()
                                graphs[actual_batch]['graph'].replay()
                                global_step +=1
                        loss = loss_train
                        sbridge.stop_prof(SBridge.FWD_BWD_TIME)

                    if args.cuda_graph_mode=='segmented' or not use_cuda_graph:

                        loss, mlm_acc, sbridge = fwd_loss_bwd_trainer.step(step,
                                                                  batch,
                                                                  model,
                                                                  optimizer,
                                                                  sbridge)
                        samples_trained += actual_batch
                    divisor = args.gradient_accumulation_steps
                    if args.log_freq>0:
                        average_loss += loss.item()

                    if update_step:
                        sbridge.start_prof(SBridge.OPT_TIME)
                        if args.cuda_graph_mode=='segmented' or not use_cuda_graph:
                            lr_scheduler.step() 

                        if send_lr_in_parallel:
                            stats_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stats_stream):
                                lr_cpu.copy_(optimizer._lr.detach(), non_blocking=True)
                        else:
                            now_lr = optimizer.param_groups[0]['lr']

                        if args.cuda_graph_mode=='segmented' or not use_cuda_graph:
                            global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)

                        if send_lr_in_parallel:
                            stats_stream.synchronize()
                            now_lr = lr_cpu.numpy()[0]

                        sbridge.stop_prof(SBridge.OPT_TIME)

                        if (args.eval_dir and args.eval_iter_samples > 0 and global_step == next_eval_step):
                            sbridge.start_eval_prof()

                            # get number of skipped steps
                            skip_interval = skipped_steps.item()
                            if skip_interval > 0:
                                global_step -= skip_interval
                                now_skipped += skip_interval
                                skipped_steps.zero_()
                                samples_trained -= skipped_samples
                                skipped_samples.zero_()
                            else:
                                # on first eval, get eval_dataloader
                                if eval_count == 0:
                                   eval_dataloader = eval_dataset_future.result(timeout=None)

                                global_samples_trained = samples_trained.detach().clone()
                                torch.distributed.all_reduce(global_samples_trained, op=torch.distributed.ReduceOp.SUM)
                                samples_trained_prev = samples_trained
                                if args.cuda_graph_mode=='segmented' or not use_cuda_graph:
                                    eval_avg_loss, eval_avg_mlm_accuracy = run_eval(args, model, fwd_loss_bwd_trainer, eval_dataloader, device, args.num_eval_examples,
                                                                                    first_eval=(eval_count == 0), use_cache=args.cache_eval_data)
                                else:
                                    eval_avg_loss, eval_avg_mlm_accuracy = run_graphed_eval(args, graph_eval, batch_gpu_placeholder_eval, eval_dataloader, loss_eval, mlm_accuracy_eval, num_valid_eval, first_eval=(eval_count == 0))
                                if utils.is_main_process():
                                    mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_ACCURACY, value=eval_avg_mlm_accuracy, metadata={'epoch_num': global_samples_trained.item()}, sync=False)
                                    print({"global_steps": global_step, "eval_loss": eval_avg_loss, "eval_mlm_accuracy":eval_avg_mlm_accuracy})

                                if args.target_mlm_accuracy:
                                    if eval_avg_mlm_accuracy >= args.target_mlm_accuracy:
                                        end_training, converged = True, True
                                        if utils.is_main_process():
                                            print("%f > %f, Target MLM Accuracy reached at %d"%(eval_avg_mlm_accuracy, args.target_mlm_accuracy, global_step))

                                eval_count += 1
                                next_eval_step = eval_steps[eval_count]
                            sbridge.stop_eval_prof()

                        accumulated_batches = 0

                    if args.target_mlm_accuracy and args.train_mlm_accuracy_window_size > 0:
                        accuracy_scores.append(mlm_acc)
                        if update_step:
                            accuracy_scores = accuracy_scores[-args.train_mlm_accuracy_window_size * args.gradient_accumulation_steps:]
                            avg_mlm_accuracy[0] = sum(accuracy_scores) / len(accuracy_scores)
                            torch.distributed.all_reduce(avg_mlm_accuracy, op=torch.distributed.ReduceOp.SUM)
                            avg_mlm_accuracy /= torch.distributed.get_world_size()

                    if args.log_freq > 0 and training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if utils.is_main_process():
                            time_interval = time.time() - now_time
                            step_interval = global_step - now_step
                            now_time = time.time()
                            now_step = global_step
                            training_perf = args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu \
                                            * (step_interval + skip_interval) / time_interval
                            skip_interval = 0

                            if args.train_mlm_accuracy_window_size > 0:
                                print({"training_steps": training_steps,
                                      "average_loss": average_loss / (args.log_freq * divisor),
                                      "step_loss": loss.item() * args.gradient_accumulation_steps / divisor,
                                      "learning_rate": now_lr,
                                      "seq/s": training_perf,
                                      "global_steps": now_step,
                                      "samples_trained": global_samples_trained.item(),
                                      "skipped_steps": now_skipped,
                                      "timestamp": now_time,
                                      "mlm_accuracy": avg_mlm_accuracy[0].item()})
                            else:
                                print({"training_steps": training_steps,
                                      "average_loss": average_loss / (args.log_freq * divisor),
                                      "step_loss": loss.item() * args.gradient_accumulation_steps / divisor,
                                      "learning_rate": now_lr,
                                      "seq/s": training_perf,
                                      "global_steps": now_step,
                                      "samples_trained": global_samples_trained.item(),
                                      "skipped_steps": now_skipped,
                                      "timestamp": now_time})

                            # for DLFW CI/CD
                            mlperf_logger.log_event(key='tracked_stats', 
                                                    value= {'seq/sec': training_perf, 
                                                            'step_loss': loss.item() * args.gradient_accumulation_steps / divisor, 
                                                            'avg_loss': average_loss / (args.log_freq * divisor), 
                                                            'lr': now_lr},
                                                    metadata = {"step": (epoch, training_steps)},
                                                    sync=False)

                            mlperf_logger.log_event(key='throughput',
                                                    value= training_perf)

                        average_loss = 0

                    if global_step >= args.max_steps or end_training:
                        status = 'success' if converged else 'aborted'
                        end_training = True
                        train_time_raw = time.time() - raw_train_start
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        if args.log_freq > 0:
                            last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                            last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                            average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                        if utils.is_main_process():
                            if args.train_mlm_accuracy_window_size > 0:
                                print((epoch, training_steps / args.gradient_accumulation_steps, ), {"final_loss": final_loss,
                                    "final_mlm_accuracy": avg_mlm_accuracy[0].item()})
                            else:
                                print((epoch, training_steps / args.gradient_accumulation_steps, ), {"final_loss": final_loss})

                    if end_training or (samples_trained - samples_trained_prev >= args.num_samples_per_checkpoint and samples_trained >= args.min_samples_to_start_checkpoints):
                        samples_trained_prev = samples_trained
                        if utils.is_main_process() and not args.skip_checkpoint:
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.phase2:
                                output_save_file = os.path.join(args.output_dir, "phase2_ckpt_{}.pt".format(samples_trained.item()))
                            else:
                                output_save_file = os.path.join(args.output_dir, "phase1_ckpt_{}.pt".format(samples_trained.item()))
                            if args.do_train:
                                torch.save({'model': model_to_save.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'master params': list(amp.master_params(optimizer)),
                                            'files': [f_id] + files}, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > args.keep_n_most_recent_checkpoints:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if samples_trained >= args.max_samples_termination or end_training:
                            status = 'success' if converged else 'aborted'
                            end_training = True
                            break
                    #torch.cuda.nvtx.range_pop()
                    #torch.cuda.nvtx.range_pop()
                del train_dataloader
                #torch.cuda.cudart().cudaProfilerStop()

                if global_samples_trained >= args.max_samples_termination or end_training:
                    status = 'success' if converged else 'aborted'
                    end_training = True
                    break

                if not need_next_training_shard:
                    dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init_fn=worker_init, synthetic_input=args.synthetic_input)
                train_dataloader, data_file = dataset_future.result(timeout=None)

            mlperf_logger.log_end(key=mlperf_logger.constants.BLOCK_STOP,
                                  metadata={'first_epoch_num': epoch},
                                  sync=False)
            mlperf_logger.log_end(key=mlperf_logger.constants.EPOCH_STOP,
                                  metadata={'epoch_num': global_samples_trained.item()}, sync=False)

            epoch += 1

            sbridge.stop_epoch_prof()

        # torch.distributed.all_reduce(samples_trained, op=torch.distributed.ReduceOp.SUM)
        mlperf_logger.log_event(key=mlperf_logger.constants.TRAIN_SAMPLES,
                                value=global_samples_trained.item(),
                                sync=False)
        mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_SAMPLES,
                                value=args.num_eval_examples,
                                sync=False)
        mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP,
                              metadata={'status': status}, sync=False)

        pool.shutdown(wait=True)
        
    return args, final_loss, train_time_raw

def global_batch_size(args):
    return args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu * args.average_packing_rate

if __name__ == "__main__":
    #torch.backends.cuda._stateful_ops.state_on_device = True

    now = time.time()
    args, final_loss, train_time_raw = main()

    gpu_count = args.n_gpu
    if torch.distributed.is_initialized():
        gpu_count = torch.distributed.get_world_size()
    if utils.is_main_process():
        e2e_time = time.time() - now
        training_perf = global_batch_size(args) \
                        * (args.max_steps - args.resume_step + skipped_steps.item()) / train_time_raw
        if args.do_train:
            print({"e2e_time": e2e_time, "training_sequences_per_second": training_perf,
                                             "final_loss": final_loss, "raw_train_time": train_time_raw })
        else:
            print({"e2e_time": e2e_time})

