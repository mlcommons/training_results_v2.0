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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as math
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.plugin_manager as plugin_manager

import numpy as np
import os.path
import math
import torch
import ctypes
import pdb
import utils
from pycocotools.coco import COCO

plugin_manager.load_library('/usr/local/lib/lib_box_iou.so')
plugin_manager.load_library('/usr/local/lib/lib_proposal_matcher.so')


class DaliDataIterator():
    def __init__(self, data_path, anno_path, batch_size,
                 num_shards, shard_id,
                 is_training, anchors, compute_matched_idxs,
                 num_threads, seed, lazy_init=True, image_size=(800, 800), prefetch_queue_depth=2):
        self.data_path = data_path
        self.anno_path = anno_path
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.is_training = is_training
        self.anchors = anchors[0]
        self.compute_matched_idxs = compute_matched_idxs
        self.num_threads = num_threads
        self.seed = seed
        self.lazy_init = lazy_init
        self.image_size = image_size
        self.prefetch_queue_depth = prefetch_queue_depth

        self.pipe = Pipeline(batch_size=self.batch_size,
                             num_threads=self.num_threads,
                             seed=self.seed,
                             device_id=torch.cuda.current_device())
        with self.pipe:
            flip = fn.random.coin_flip(probability=0.5)
            inputs, bboxes, labels, image_ids = fn.readers.coco(
                name="coco",
                file_root=self.data_path,
                annotations_file=self.anno_path,
                num_shards=self.num_shards,
                shard_id=self.shard_id,
                stick_to_shard=not self.is_training,
                pad_last_batch=not self.is_training,
                lazy_init=self.lazy_init,
                ltrb=True,
                shuffle_after_epoch=self.is_training,
                avoid_class_remapping=True,
                image_ids=True,
                ratio=True,
                prefetch_queue_depth=self.prefetch_queue_depth,
                read_ahead=True,
                skip_empty=False)

            # Images
            images_shape = fn.peek_image_shape(inputs)  # HWC
            images = fn.decoders.image(inputs, device='mixed')
            images = fn.flip(images, horizontal=flip, device='gpu')
            images = fn.crop_mirror_normalize(images, device='gpu',
                                              mean=[255 * 0.485, 255 * 0.456, 255 * 0.406],
                                              std=[255 * 0.229, 255 * 0.224, 255 * 0.225])
            images = fn.resize(images, resize_x=self.image_size[0], resize_y=self.image_size[1])

            # Labels
            labels_shape = fn.shapes(labels)
            labels = fn.pad(labels, axes=(0,))
            labels = labels.gpu()
            labels = fn.cast(labels, dtype=types.INT64)

            # BBoxes
            bboxes = fn.bb_flip(bboxes, horizontal=flip, ltrb=True)
            lt_x = bboxes[:, 0] * self.image_size[0]
            lt_y = bboxes[:, 1] * self.image_size[1]
            rb_x = bboxes[:, 2] * self.image_size[0]
            rb_y = bboxes[:, 3] * self.image_size[1]
            bboxes = fn.stack(lt_x, lt_y, rb_x, rb_y, axis=1)
            bboxes_shape = fn.shapes(bboxes)
            bboxes = bboxes.gpu()
            if self.compute_matched_idxs:
                match_quality_matrix = fn.box_iou(bboxes, self.anchors, device='gpu')
                matched_idxs = fn.proposal_matcher(match_quality_matrix, device='gpu')
            bboxes = fn.pad(bboxes, axes=(0,))

            set_outputs = [images, images_shape, image_ids, bboxes, bboxes_shape, labels, labels_shape]
            if self.compute_matched_idxs:
                set_outputs.append(matched_idxs)

            self.pipe.set_outputs(*set_outputs)
        self.pipe.build()

        output_map = ['images', 'images_shape', 'images_id', 'boxes', 'boxes_shape', 'labels', 'labels_shape']
        if self.compute_matched_idxs:
            output_map.append('matched_idxs')

        self.dali_iter = DALIGenericIterator(pipelines=[self.pipe],
                                             reader_name="coco",
                                             output_map=output_map,
                                             auto_reset=True,
                                             last_batch_policy=LastBatchPolicy.FILL)

    def __len__(self):
        return len(self.dali_iter)

    def __iter__(self):
        for obj in self.dali_iter:
            obj = obj[0]
            
            images = obj['images']

            boxes = [b[0][:b[1][0]] for b in zip(obj['boxes'], obj['boxes_shape'])]
            labels = [b[0][:b[1][0]] for b in zip(obj['labels'].to(torch.int64), obj['labels_shape'])]
            image_id = obj['images_id']
            original_image_size = obj['images_shape']
            targets = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'original_image_size': original_image_size}

            if self.compute_matched_idxs:
                matched_idxs = obj['matched_idxs'][:, 0, :]
                targets['matched_idxs'] = matched_idxs

            yield images, targets


if __name__ == '__main__':
    device = torch.device(0)
    # dali_iter = DaliDataIterator(data_path='/datasets/open-images-v6-mlperf/train/data',
    #                              anno_path='/datasets/open-images-v6-mlperf/train/labels/openimages-mlperf.json',
    #                              batch_size=8, num_threads=4, world=1)
    dali_iter = DaliDataIterator(data_path='/datasets/coco2017/train2017',
                                 anno_path='/datasets/coco2017/annotations/instances_train2017.json',
                                 batch_size=2, num_threads=1, world=1, training=True)
    for images, targets in dali_iter:
        pdb.set_trace()

