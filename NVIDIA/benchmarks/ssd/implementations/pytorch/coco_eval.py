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

import functools
import tempfile
import copy
import time
from contextlib import redirect_stdout

import numpy as np
import torch
import torch._six

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from collections import defaultdict

import utils



class CocoEvaluator(object):
    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def get_stats(self):
        stats = {}
        for iou_type, coco_eval in self.coco_eval.items():
            stats[iou_type] = coco_eval.stats
        return stats

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": label,
                        "bbox": bbox,
                        "score": score,
                    }
                    for (bbox, label, score) in zip(boxes, labels, scores)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


class DefaultCocoEvaluator(CocoEvaluator):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def reset(self):
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(set(predictions.keys()))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            with redirect_stdout(None):
                coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = list(img_ids)
                coco_eval.evaluate()

            catIds = coco_eval.params.catIds if coco_eval.params.useCats else [-1]
            areaRng = coco_eval.params.areaRng
            imgIds = coco_eval.params.imgIds
            evalImgs = np.asarray(coco_eval.evalImgs).reshape(len(catIds), len(areaRng), len(imgIds))
            self.eval_imgs[iou_type].append(evalImgs)

    def synchronize_between_processes(self, group=None):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type], group=group)


class NVCocoEvaluator(CocoEvaluator):
    def __init__(self, annotations_file, iou_types, num_threads=1, group=None):
        assert isinstance(iou_types, (list, tuple))
        self.coco_gt = None
        self.annotations_file = annotations_file
        self.num_threads = num_threads
        self.iou_types = iou_types
        self.coco_eval = {}
        self._results = {}
        for iou_type in iou_types:
            self._results[iou_type] = []

    @property
    def results(self):
        return self._results

    def reset(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self._results[iou_type] = []

    def update(self, predictions):
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            self._results[iou_type].extend(results)

    def synchronize_between_processes(self, group=None):
        for iou_type in self.iou_types:
            gathered_results = utils.all_gather(self._results[iou_type], group=group)
            self._results[iou_type] = []
            for result in gathered_results:
                self._results[iou_type].extend(result)

    def accumulate(self):
        if self.coco_gt is None:
            self.coco_gt = get_coco_gt(annotations_file=self.annotations_file, use_ext=True)

        for iou_type in self.iou_types:
            results = self._results[iou_type]
            coco_gt = self.coco_gt

            with redirect_stdout(None):
                coco_dt = coco_gt.loadRes(results, use_ext=True, print_res=False) if results else COCO(use_ext=True)
                coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type, num_threads=self.num_threads, use_ext=True)
                coco_eval.evaluate()
                coco_eval.accumulate()

            self.coco_eval[iou_type] = coco_eval


def static_nvcocoevaluator(results, annotations_file, num_threads):
    coco_gt = get_coco_gt(annotations_file=annotations_file, use_ext=True)
    stats = {}
    for iou_type in results.keys():
        coco_eval = None
        with redirect_stdout(None):
            coco_dt = coco_gt.loadRes(results[iou_type], use_ext=True, print_res=False) if results[iou_type] else COCO(use_ext=True)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type, num_threads=num_threads, use_ext=True)
            coco_eval.evaluate()
            coco_eval.accumulate()

        print("IoU metric: {}".format(iou_type))
        coco_eval.summarize()
        stats[iou_type] = coco_eval.stats

    return stats


@functools.lru_cache(maxsize=None)
def get_coco_gt(annotations_file, use_ext):
    return COCO(annotation_file=annotations_file, use_ext=use_ext)


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs, group=None):
    all_img_ids = utils.all_gather(img_ids, group=group)
    all_eval_imgs = utils.all_gather(eval_imgs, group=group)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs, group=None):
    img_ids, eval_imgs = merge(img_ids, eval_imgs, group=group)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
