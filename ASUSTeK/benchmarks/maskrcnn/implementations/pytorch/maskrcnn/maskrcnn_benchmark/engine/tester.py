# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
import torch

import os

from mlperf_logging.mllog import constants

from maskrcnn_benchmark.data.build import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.mlperf_logger import log_event
from maskrcnn_benchmark.utils.comm import synchronize, get_world_size

_first_test = True
_eval_datasets = None


class CachingDataLoader(object):
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.dataset = dataloader.dataset
        self.length = len(dataloader)
        self.samples = []
        self.next = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        self.next = 0
        return self

    def __next__(self):
        if self.next < self.length:
            if self.dataloader is not None:
                images, targets, image_ids = next(self.dataloader)
                images.tensors = torch.empty_like(images.tensors).copy_(images.tensors)
                self.samples.append( (images, targets, image_ids) )
                if len(self.samples) == self.length:
                    self.dataloader = None
            self.next = self.next + 1
            return self.samples[self.next-1]
        else:
            raise StopIteration


def test(cfg, model, distributed, shapes, eval_ranks_comm=None, dryrun=False):
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    if cfg.DATALOADER.CACHE_EVAL_IMAGES:
        global _eval_datasets
        if _eval_datasets is None:
            data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, shapes=shapes)
            _eval_datasets = []
            for data_loader_val in data_loaders_val:
                data_loader_val.dataset.coco.createIndex(use_ext=True)
                _eval_datasets.append( CachingDataLoader(data_loader_val) )
        data_loaders_val = _eval_datasets
    else:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, shapes=shapes)
        for data_loader_val in data_loaders_val: data_loader_val.dataset.coco.createIndex(use_ext=True)

    global _first_test
    if _first_test:
        dedicated_evaluation_ranks = max(0,cfg.DEDICATED_EVALUATION_RANKS)
        first_rank = (get_world_size()-dedicated_evaluation_ranks) if dedicated_evaluation_ranks > 0 else 0
        log_event(key=constants.EVAL_SAMPLES, value=len(data_loaders_val),log_rank=first_rank)
        _first_test = False

    results = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            eval_segm_numprocs=cfg.EVAL_SEGM_NUMPROCS,
            eval_mask_virtual_paste=cfg.EVAL_MASK_VIRTUAL_PASTE,
            dedicated_evaluation_ranks=max(0,cfg.DEDICATED_EVALUATION_RANKS),
            eval_ranks_comm=eval_ranks_comm,
            dryrun=dryrun,
        )
        # Note: this synchronize() would break async results by not allowing them
        # to actually be async
        # synchronize()
        results.append(result)
    return results
