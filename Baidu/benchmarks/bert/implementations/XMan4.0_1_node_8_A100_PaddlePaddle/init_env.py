# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
from mpi4py import MPI
import numpy as np
import time
import paddle
from pybind.functions import process_allgathered_inputs as process_bert_inputs
from pybind.functions import process_eval_inputs as process_bert_eval_inputs
import h5py
import random

global_comm = MPI.COMM_WORLD
global_rank = global_comm.rank
global_world_size = global_comm.size
assert global_world_size % 2 == 0


def create_group_comm(ranks):
    ranks = list(ranks)
    new_group = global_comm.group.Incl(ranks)
    new_comm = global_comm.Create_group(new_group)
    return new_comm


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(comm, seeds, root=0):
    seeds = np.array(seeds).astype(np.int64)
    comm.Bcast(seeds, root=root)
    return seeds.tolist()


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
    # limin-todo: 
    #data_file = "/data2/zengjinle/dataset/bert_data/hdf5/training-4320/hdf5_4320_shards_uncompressed/part_01799_of_04320.hdf5"
    #print("data_file: ", data_file)
    return data_file


def read_hdf5_file(input_file, dtype=np.int16):
    keys = [
        'input_ids',
        'input_mask',
        'segment_ids',
        'masked_lm_positions',
        'masked_lm_ids',
        'next_sentence_labels',
    ]
    if not os.path.exists(input_file):
        return None
    with h5py.File(input_file, 'r') as f:
        outputs = [np.array(f[key], dtype=dtype) for key in keys]
        n = outputs[0].shape[0]
        masked_lm_labels = np.zeros(outputs[0].shape, dtype=dtype)
        lengths = np.zeros(n, dtype=dtype)
        for i in range(n):
            masked_lm_positions = outputs[3][i]
            masked_lm_ids = outputs[4][i]
            length = np.count_nonzero(masked_lm_positions)
            masked_lm_labels[i][
                masked_lm_positions[:length]] = masked_lm_ids[:length]
            lengths[i] = np.count_nonzero(outputs[1][i])
        outputs = [
            outputs[0], outputs[2], outputs[1], masked_lm_labels, outputs[-1],
            lengths
        ]
        idx = np.random.choice(np.arange(n), n, replace=False)
        for i in range(len(outputs)):
            outputs[i] = outputs[i][idx]
    return outputs


def read_eval_hdf5_file(input_file, dtype=np.int16):
    keys = [
        'input_ids',
        'input_mask',
        'segment_ids',
        'masked_lm_positions',
        'masked_lm_ids',
        'next_sentence_labels',
    ]
    if not os.path.exists(input_file):
        return None
    with h5py.File(input_file, 'r') as f:
        outputs = [np.asarray(f[key][:]) for key in keys]
        nsamples = outputs[0].shape[0]

        all_data = []
        for index in range(nsamples):
            [
                input_ids, input_mask, segment_ids, masked_lm_positions,
                masked_lm_ids, next_sentence_labels
            ] = [
                input[index].astype(dtype)
                if indice < 5 else np.asarray(input[index].astype(dtype))
                for indice, input in enumerate(outputs)
            ]

            length = np.count_nonzero(masked_lm_positions)
            masked_lm_positions = masked_lm_positions[:length]
            masked_lm_ids = masked_lm_ids[:length]

            masked_lm_labels = np.zeros(input_ids.shape, dtype=dtype)
            masked_lm_labels[masked_lm_positions] = masked_lm_ids

            #if index == 0:
            #    print("masked_lm_labels = ", masked_lm_labels)
            #    print("masked_lm_positions = ", masked_lm_positions)
            #    print("masked_lm_ids = ", masked_lm_ids)
            seq_len = np.asarray(np.count_nonzero(input_mask))

            data = [
                input_ids,
                segment_ids,
                input_mask,
                masked_lm_labels,
                next_sentence_labels,
                seq_len,
            ]
            # (2050, ), i.e., 512 * 4 + 1 + 1
            one_sample_data = np.concatenate([d.flatten() for d in data])

            all_data.extend(one_sample_data)

        # (2050000, ) -> (10000, 2050)
    return np.asarray(all_data).reshape((nsamples, -1))


class WorkerInitObj(object):
    "Construct the object with different seed, and the Dataloader will generate the data "
    "with different seed in each worker."

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class Context:
    def __init__(self):
        half_size = int(global_world_size / 2)
        self.trainer_id = global_rank % half_size
        self.trainer_num = half_size
        self.is_trainer = (global_rank < half_size)

        self.reader_id = self.trainer_id
        self.reader_num = self.trainer_num
        self.is_reader = not self.is_trainer

        self.trainer_comm = create_group_comm(range(0, half_size))
        self.reader_comm = create_group_comm(
            range(half_size, global_world_size))
        self.trainer_reader_comm = create_group_comm(
            [self.trainer_id, self.trainer_id + half_size])
        self.global_comm = global_comm

    def init_args(self, args, dtype=np.int16):
        self.args = args
        self.files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
        ]
        self.files.sort()
        self.fid_buf = np.array([1], dtype=np.int64)
        with h5py.File(self.files[0], 'r') as f:
            self.num_samples = np.array(f["next_sentence_labels"][:]).size

        self.batch_size = args.train_batch_size
        self.max_seq_length = args.max_seq_length
        self.worker_seeds, self.shuffling_seeds = self._setup_seeds(
            args.seed, args.num_epochs_to_generate_seeds_for)
        self.epoch_idx = 0

        data_buf_size = self.num_samples * 4 * self.max_seq_length + self.num_samples * 2
        self.data_buf = np.empty(
            shape=[self.trainer_num * data_buf_size], dtype=dtype)

        self.eval_dir = args.eval_dir
        self.num_eval_examples = args.num_eval_examples
        self.eval_batch_size = args.eval_batch_size

        cur_seed = self.worker_seeds[self.trainer_id]
        np.random.seed(cur_seed)
        random.seed(cur_seed)
        paddle.seed(cur_seed)
        self.worker_init = WorkerInitObj(cur_seed)
        self.barrier()

    def shuffle_files(self):
        random.Random(self.shuffling_seeds[self.epoch_idx]).shuffle(self.files)
        self.epoch_idx += 1

    def _setup_seeds(self, master_seed, epochs):
        if master_seed is None:
            master_seed = random.SystemRandom().randint(0, 2**32 - 1)
            if self.trainer_id == 0:
                print('Using random master seed: {}'.format(master_seed))
        else:
            print('Using master seed from command line: {}'.format(master_seed))

        # initialize seeding RNG
        seeding_rng = random.Random(master_seed)

        # generate worker seeds, one seed for every distributed worker
        worker_seeds = generate_seeds(seeding_rng, self.trainer_num)

        # generate seeds for data shuffling, one seed for every epoch
        shuffling_seeds = generate_seeds(seeding_rng, epochs)

        worker_seeds = broadcast_seeds(self.global_comm, worker_seeds)
        shuffling_seeds = broadcast_seeds(self.global_comm, shuffling_seeds)
        return worker_seeds, shuffling_seeds

    def worker_seed(self):
        return self.worker_seeds[self.trainer_id]

    def barrier(self):
        self.global_comm.barrier()

    def stop_reader(self):
        if self.is_trainer:
            self.read_file(-1)

    def file_num(self):
        return len(self.files)

    def read_file(self, f_id=None):
        if self.is_trainer:
            self.fid_buf[0] = f_id
            self.trainer_reader_comm.Isend(self.fid_buf, dest=1)
            if f_id == 0:
                self.shuffle_files()
            elif f_id < 0:
                return

            self.trainer_reader_comm.Recv(self.data_buf, source=1)
            results = process_bert_inputs(self.data_buf, self.num_samples,
                                          self.max_seq_length, self.batch_size,
                                          self.trainer_id, self.trainer_num)

            return results
        else:
            self.trainer_reader_comm.Recv(self.fid_buf, 0)
            f_id = self.fid_buf[0]
            if f_id == 0:
                self.shuffle_files()
            elif f_id < 0:
                return False

            fname = select_dataset_file_for_each_worker(
                self.files, f_id, self.trainer_num, self.trainer_id)
            data = read_hdf5_file(fname, dtype=self.data_buf.dtype)
            send_buf = np.concatenate([d.flatten() for d in data])
            self.reader_comm.Allgather(send_buf, self.data_buf)
            self.trainer_reader_comm.Send(self.data_buf, dest=0)
            return True

    def read_eval_file(self):
        if self.is_trainer:

            eval_data = []
            for eval_file in sorted(os.listdir(self.eval_dir)):
                eval_file_path = os.path.join(self.eval_dir, eval_file)
                if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
                    data = read_eval_hdf5_file(
                        eval_file_path, dtype=self.data_buf.dtype)
                    eval_data.extend(data)
                    if len(eval_data) > self.num_eval_examples:
                        break

            chunk_size = self.num_eval_examples // self.trainer_num
            rank = self.trainer_id
            remainder = self.num_eval_examples % self.trainer_num
            if rank < remainder:
                eval_data = eval_data[(chunk_size + 1) * rank:(chunk_size + 1) *
                                      (rank + 1)]
            else:
                eval_data = eval_data[chunk_size * rank + remainder:chunk_size *
                                      (rank + 1) + remainder]

            results = process_bert_eval_inputs(eval_data, self.max_seq_length,
                                               self.eval_batch_size,
                                               self.args.sort_eval_data)
            return results


_context = Context()


def get_context():
    return _context
