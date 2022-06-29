# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import utils
from torch.utils.data import Dataset
from engine import preprocessing, loss_preprocessing, compute_matched_idxs


def init_cache(model_ptr, dataset, device, args, cache_sz):
    cache_images_, cache_targets_ = [], []

    for j in range(int(cache_sz / args.batch_size)):
        images, targets = [], []
        for i in range(args.batch_size):
            images.append(dataset[j * int(cache_sz / args.batch_size) + i][0])
            targets.append(dataset[j * int(cache_sz / args.batch_size) + i][1])

        images = list(image.to(device, non_blocking=True) for image in images)
        targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}

        images, targets = preprocessing(images, targets, model_ptr, args.data_layout)

        with torch.cuda.amp.autocast(enabled=args.amp):
            targets['matched_idxs'] = compute_matched_idxs(targets['boxes'], model_ptr)

        for i in range(args.batch_size):
            cache_images_.append(images[i].cpu())
            cache_targets_.append({'matched_idxs': targets['matched_idxs'][i].cuda(),
                                   'labels': targets['labels'][i].cuda(),
                                   'boxes': targets['boxes'][i].cuda()})

    return cache_images_, cache_targets_


def get_cached_dataset(model, dataset, device, args, cache_sz=16, virtual_cache_sz_factor=32768):
    cache_images_, cache_targets_ = init_cache(model, dataset, device, args, cache_sz)
    cached_dataset = CachedDataset(cache_sz, virtual_cache_sz_factor, cache_images_, cache_targets_)

    if args.distributed:
        cached_train_sampler = torch.utils.data.distributed.DistributedSampler(cached_dataset)
    else:
        cached_train_sampler = torch.utils.data.RandomSampler(cached_dataset)

    cached_train_batch_sampler = torch.utils.data.BatchSampler(cached_train_sampler, args.batch_size, drop_last=True)

    cached_data_loader = torch.utils.data.DataLoader(cached_dataset, batch_sampler=cached_train_batch_sampler,
                                                     num_workers=0, pin_memory=False, collate_fn=utils.collate_fn)
    return cached_data_loader


class CachedDataset(Dataset):
    def __init__(self, cache_sz, virtual_cache_sz_factor, cache_images, cache_targets):
        self.cache_sz = cache_sz
        self.virtual_cache_sz_factor = virtual_cache_sz_factor
        self.virtual_dataset_sz = self.cache_sz * self.virtual_cache_sz_factor
        self.cache_images = cache_images
        self.cache_targets = cache_targets

    def __len__(self):
        return self.virtual_dataset_sz

    def __getitem__(self, idx):
        return self.cache_images[idx % self.cache_sz], self.cache_targets[idx % self.cache_sz]
