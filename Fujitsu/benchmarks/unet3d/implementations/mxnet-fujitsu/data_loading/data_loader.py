# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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
import glob
from math import ceil
from time import time
from multiprocessing import Pool

import numpy as np
from mxnet import gluon, nd

from data_loading.dali_loader import get_dali_loader
from mlperf_logger import mllog_event, constants


def calculate_work(f):
    arr = np.load(f)
    image_shape = list(arr.shape[1:])
    return np.prod([image_shape[i] // 64 - 1 + (1 if image_shape[i] % 64 >= 32 else 0) for i in range(3)])


def make_val_split_even(x_val, y_val, num_shards, shard_id, shard_eval, batch_size, local_shard_size):
    t0 = time()
    p = Pool(processes=8)
    work = np.array(p.map(calculate_work, y_val))
    x_res = [[] for _ in range(num_shards)]
    y_res = [[] for _ in range(num_shards)]
    curr_work_per_shard = np.zeros(shape=num_shards)

    if shard_eval:
        bucket_size = batch_size * local_shard_size
        work = np.array([bucket_size * ceil(w/bucket_size) for w in work])
    x_val, y_val = np.array(x_val), np.array(y_val)

    sort_idx = np.argsort(work)[::-1]
    work = work[sort_idx]
    x_val, y_val = x_val[sort_idx], y_val[sort_idx]

    for w_idx, w in enumerate(work):
        idx = np.argmin(curr_work_per_shard)
        curr_work_per_shard[idx] += w
        x_res[idx].append(x_val[w_idx])
        y_res[idx].append(y_val[w_idx])

    return x_res[shard_id], y_res[shard_id]


def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def get_data_split(path: str):
    with open("evaluation_cases.txt", "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)

    mllog_event(key='train_samples', value=len(imgs_train), sync=False)
    mllog_event(key='eval_samples', value=len(imgs_val), sync=False)

    return imgs_train, imgs_val, lbls_train, lbls_val


class SyntheticDataLoader:
    def __init__(self, batch_size=1, channels_in=1, channels_out=3, shape=(128, 128, 128),
                 ctx=None, scalar=False, world_size=1):
        x_shape = tuple(shape) + (channels_in,)
        self.curr_pos = None
        self.global_batch_size = batch_size * world_size
        self.x = nd.random.uniform(shape=(batch_size, *x_shape), dtype=np.float32, ctx=ctx)
        if scalar:
            self.y = nd.random.randint(low=0, high=channels_out-1, shape=(batch_size, *shape), dtype=np.int32, ctx=ctx)
            self.y = nd.expand_dims(self.y, -1)
        else:
            y_shape = tuple(shape) + (channels_out,)
            self.y = nd.random.uniform(shape=(batch_size, *y_shape), dtype=np.float32, ctx=ctx)

    def __iter__(self):
        self.curr_pos = 0
        return self

    def __next__(self):
        if self.curr_pos < len(self):
            self.curr_pos += 1
            return self.x, self.y
        else:
            raise StopIteration

    def __len__(self):
        return 168 // self.global_batch_size

    def __getitem__(self, idx):
        return self.x, self.y


def get_data_loaders(flags, data_dir, seed, local_rank, global_rank, train_ranks, eval_ranks,
                     spatial_group_size, shard_eval, ctx, world_size):
    if flags.loader == "synthetic":
        return SyntheticDataLoader(flags.batch_size, ctx=ctx, scalar=True, world_size=world_size), None
    x_train, x_val, y_train, y_val = get_data_split(data_dir)
    if global_rank in train_ranks:
        shard_id = global_rank // spatial_group_size
        num_shards = len(train_ranks) // spatial_group_size
        dataset_len = len(x_train)
        if flags.stick_to_shard:
            shard_len = len(x_train) // num_shards
            x_train = x_train[shard_id * shard_len:(shard_id + 1) * shard_len]
            y_train = y_train[shard_id * shard_len:(shard_id + 1) * shard_len]
            num_shards = 1
            shard_id = 0
            dataset_len = len(x_train)
        train_dataloader = get_dali_loader(flags, x_train, y_train, mode="train", seed=seed, num_shards=num_shards,
                                           device_id=local_rank, shard_id=shard_id, global_rank=global_rank,
                                           dataset_len=dataset_len)
    else:
        train_dataloader = None

    if global_rank in eval_ranks:
        if shard_eval:
            shard_id = (global_rank - eval_ranks[0]) // min(len(eval_ranks), 8)
            num_shards = len(eval_ranks) // min(len(eval_ranks), 8)
        else:
            shard_id = (global_rank - eval_ranks[0]) // spatial_group_size
            num_shards = len(eval_ranks) // spatial_group_size
        x_val, y_val = make_val_split_even(x_val, y_val, num_shards=num_shards, shard_id=shard_id,
                                           shard_eval=shard_eval, batch_size=flags.val_batch_size,
                                           local_shard_size=min(len(eval_ranks), 8))
        val_dataloader = get_dali_loader(flags, x_val, y_val, mode="validation", seed=seed,
                                         num_shards=1, device_id=local_rank)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def get_dummy_loaders(flags, data_dir, seed, local_rank, global_rank, training_ranks, spatial_group_size):
    if spatial_group_size > 1:
        assert flags.batch_size == 1, f"batch_size must be equal to 1, got {flags.batch_size}"
        assert flags.val_batch_size == 1, f"val_batch_size must be equal to 1, got {flags.val_batch_size}"

    train_dataloader = None
    val_dataloader = None
    if global_rank in training_ranks:
        case_id = str(local_rank).zfill(5)
        create_dummy_dataset(data_dir, case_id=case_id)
        x_train = load_data(data_dir, f"*{case_id}_x.npy")
        y_train = load_data(data_dir, f"*{case_id}_y.npy")
        train_dataloader = get_dali_loader(flags, x_train, y_train, mode="train", seed=seed, num_shards=1,
                                           device_id=local_rank, shard_id=0, global_rank=global_rank)

    return train_dataloader, val_dataloader


def create_dummy_dataset(data_dir, case_id):
    os.makedirs(data_dir, exist_ok=True)
    x = np.random.rand(1, 256, 256, 256).astype(np.float32)
    y = np.random.randint(low=0, high=3, size=(1, 256, 256, 256), dtype=np.uint8)
    np.save(os.path.join(data_dir, f"dummy_{case_id}_x.npy"), x)
    np.save(os.path.join(data_dir, f"dummy_{case_id}_y.npy"), y)
