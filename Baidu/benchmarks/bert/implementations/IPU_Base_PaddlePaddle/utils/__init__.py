# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import argparse
import os
from logging import getLogger

import numpy as np
import tqdm

logger = getLogger(__name__)

DISTRIBUTED_COMM = None


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Override config file if not None")
    parser.add_argument(
        "--seq_len", default=128, type=int, help="The sequence length")
    parser.add_argument(
        "--vocab_size",
        default=30912,
        type=int,
        help="Set the size of the vocabulary")
    parser.add_argument(
        "--max_predictions_per_seq",
        default=20,
        type=int,
        help="The maximum total of masked tokens in input sequence")
    parser.add_argument(
        "--max_position_embeddings",
        default=512,
        type=int,
        help="the length of the input mask")
    parser.add_argument(
        "--max_training_sequences",
        default=6000000,
        type=int,
        help="The maximum number of sequences to process during training")
    parser.add_argument(
        "--micro_batch_size", type=int, default=1, help="micro batch size")
    parser.add_argument(
        "--hidden_size", default=1024, type=int, help="Set the hidden size")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="hidden_dropout_prob")
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.0,
        help="attention_probs_dropout_prob")
    parser.add_argument(
        "--is_training", type=str_to_bool, default=True, help="Training or not")
    parser.add_argument(
        "--split_qkv",
        type=str_to_bool,
        default=True,
        help="Split QKV weights or not")
    parser.add_argument(
        "--activation_checkpoint_dtype",
        type=str,
        default="FLOAT16",
        help="Set the data type of checkpointOutput Op.")
    parser.add_argument(
        "--no_attn_dropout",
        type=str_to_bool,
        default=True,
        help="Disable dropout Ops in attention layers.")

    # Data
    parser.add_argument(
        "--input_files", type=str, default="", help="Files to load data from.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Total batch size for training with IPUs.")
    parser.add_argument(
        "--shuffle",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Shuffle Dataset")

    # Packed config
    parser.add_argument(
        "--duplication_factor",
        type=int,
        default=1,
        help="Set the number of times the dataset has been duplicated. This reduces the samples per epoch to"
        " (# of samples in input-files)/duplication-factor")
    parser.add_argument(
        "--max_sequences_per_pack",
        type=int,
        default=2,
        help="For use when pretraining data is pre-packed. Maximum number of sequences to expect in a pack."
    )
    parser.add_argument(
        "--avg_seq_per_pack",
        help="The approximate number of sequences in the average pack of sequences",
        type=int,
        default=1)
    parser.add_argument(
        "--use_prepacked_pretraining_dataset",
        default=False,
        help="For use when pretraining data is pre-packed to reduce padding.")

    # Optimizer
    parser.add_argument(
        "--optimizer_type", type=str, default='sgd', help="type of optimizer")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=10,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=1.0,
        help="The value of scale_loss for fp16.")
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Set the Adam/Lamb beta1 value")
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Set the Adam/Lamb beta2 value")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="ipu",
        help="Device for selecting for the training.")
    parser.add_argument(
        "--POD", type=int, default=16, help="The number of the POD")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--enable_pipelining",
        type=str_to_bool,
        default=False,
        help="enable pipelining or not.")
    parser.add_argument(
        "--batches_per_step", type=int, default=1, help="batches per step")
    parser.add_argument(
        "--enable_replica",
        type=str_to_bool,
        default=False,
        help="enable replicat or not.")
    parser.add_argument(
        "--local_num_replicas",
        type=int,
        default=1,
        help="number of local replicas")
    parser.add_argument(
        "--enable_grad_acc",
        type=str_to_bool,
        default=False,
        help="enable gradient accumulation")
    parser.add_argument(
        "--grad_acc_factor",
        type=int,
        default=1,
        help="factor of gradient accumulation")
    parser.add_argument(
        "--optimizer_state_offchip",
        type=str_to_bool,
        default=True,
        help="Set the store location of the optimizer tensors")
    parser.add_argument(
        "--merge_collectives",
        type=str_to_bool,
        default=True,
        help="Whether to attempt to merge small cross-replicate collective operations into a larger one"
    )

    # Validation
    parser.add_argument(
        "--tf_checkpoint",
        type=str,
        help="The path of Tensorflow Checkpoint to initialise the model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="The iteration of the whole dataset")
    parser.add_argument(
        "--enable_validation",
        type=str_to_bool,
        default=False,
        help="enable the validation during training or not")
    parser.add_argument(
        "--validation_files", type=str, help="the path of the validation files")
    parser.add_argument(
        "--accuracy_averaging_basis",
        type=str,
        default="pertoken",
        help="the basis for validation accuracy")

    # popdist
    parser.add_argument(
        "--popdist_size",
        type=int,
        default=1,
        help="The number of distributed processes")
    parser.add_argument(
        "--popdist_rank",
        type=int,
        default=0,
        help="The index of each processes")
    parser.add_argument(
        "--replica_index_offset",
        type=int,
        default=0,
        help="The index offset of each replica.")
    parser.add_argument(
        "--global_replicas",
        type=int,
        default=1,
        help="The number of total replicas.")
    parser.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="The multi-hosts for POD128/256 distributed computing.")
    parser.add_argument(
        "--compile_only",
        type=str_to_bool,
        default=False,
        help="Only compile to create engine cache for POD128/256.")

    # MLPerf config
    parser.add_argument(
        "--submission_run_index",
        type=int,
        default=1,
        help="The index of submission run")

    # profile
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Logging per X steps")
    parser.add_argument(
        "--wandb",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable logging to Weights and Biases.")
    parser.add_argument(
        "--profile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Generate a profile directory to be analysed by popvision.")
    parser.add_argument(
        "--profile_dir", type=str, help="Path to profile directory.")
    parser.add_argument(
        "--profile_instrument",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Include cycle count instrumentation when profiling.")
    args = parser.parse_args()
    return args


def engine_cache_copy(pod, hosts):
    # Copy the engine cache from the main host to others
    cache_path = os.getcwd() + '/paddle_cache/POD{}/'.format(pod)
    cache_list = os.listdir(cache_path)
    if not cache_list:
        raise RuntimeError(
            "No IPU engine cache. Please check the dir: {}.".format(cache_path))
    cache_list.sort(key=lambda fn: os.path.getmtime(cache_path + fn))
    cache_name = cache_list[-1]
    cache_file = cache_path + cache_name

    host_names = hosts.split(",")
    for i in range(1, len(host_names)):
        host = host_names[i].strip()
        logger.info("Copy engine cache from {} to {}...".format(host_names[0]
                                                                .strip(), host))
        res = os.system("scp {} {}".format(cache_file, host + ":" + cache_file))
        if res is not 0:
            raise RuntimeError(
                "Copy engine cache from {} to {}...Failed. Please check the dir: {}".
                format(host_names[0].strip(), host, cache_path))
        logger.info("Copy engine cache from {} to {}...Done".format(host_names[
            0].strip(), host))


def set_distribution_args(args):
    if 'POPDIST_NUM_TOTAL_REPLICAS' not in os.environ:
        raise RuntimeError(
            "Please use poprun run the program with POD128 and POD256")

    if not args.is_training:
        raise RuntimeError(
            "Distributed execution is only supported for training")

    local_replicas = int(
        os.environ.get('POPDIST_NUM_LOCAL_REPLICAS', default='1'))
    total_replicas = int(
        os.environ.get('POPDIST_NUM_TOTAL_REPLICAS', default='1'))
    replica_index_offset = int(
        os.environ.get('POPDIST_REPLICA_INDEX_OFFSET', default='0'))
    if args.local_num_replicas > 1 and args.local_num_replicas != local_replicas:
        logger.warning(
            f"Overwriting the local replication factor {args.local_num_replicas} to {local_replicas}"
        )
        args.local_num_replicas = local_replicas

    args.popdist_size = total_replicas // local_replicas
    args.popdist_rank = replica_index_offset // local_replicas
    args.replica_index_offset = replica_index_offset
    args.global_replicas = total_replicas

    from mpi4py import MPI
    global DISTRIBUTED_COMM
    DISTRIBUTED_COMM = MPI.COMM_WORLD


def _get_comm():
    global DISTRIBUTED_COMM
    if DISTRIBUTED_COMM is None:
        raise RuntimeError(
            "Distributed Commumication not setup. Please run setup_comm(MPI.COMM_WORLD) first. "
            "See https://mpi4py.readthedocs.io/ for details on MPI.COMM_WORLD.")
    return DISTRIBUTED_COMM


def sum_distributed_data(data: int) -> int:
    comm = _get_comm()
    size = comm.Get_size()
    rank = comm.Get_rank()

    sendbuf = np.array([data])
    recvbuf = np.empty([size, 1], int)
    comm.Allgather(sendbuf, recvbuf)
    data = np.sum(recvbuf, axis=0)

    return data


def sum_distributed_dataf(data: float) -> float:
    comm = _get_comm()
    size = comm.Get_size()
    rank = comm.Get_rank()

    sendbuf = np.array([data])
    recvbuf = np.empty([size, 1], np.float64)
    comm.Allgather(sendbuf, recvbuf)
    data = np.sum(recvbuf, axis=0)

    return np.float64(data)


def distributed_barrier():
    _get_comm().barrier()


class ProgressBar:
    def __init__(self):
        self._bar = None
        self._last = 0

    def __call__(self, progress: int, total: int):
        if self._bar is None:
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            bar_format += "[{elapsed}<{remaining}]"
            self._bar = tqdm.tqdm(
                desc="Graph compilation", total=total, bar_format=bar_format)
        self._bar.update(progress - self._last)
        self._last = progress
        if progress == total:
            self._bar.close()
            self._bar = None


# need to set to 0 when start a new compilation
g_current_progress = 0


def ProgressFunc(progress, total):
    global g_current_progress
    if progress != g_current_progress:
        g_current_progress = progress
        print(f"Graph compilation: {progress}/{total}")
