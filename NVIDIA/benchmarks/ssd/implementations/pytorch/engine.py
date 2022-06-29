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

import math
import sys
import time
import torch

from ssd_logger import mllogger
from mlperf_logging.mllog.constants import (EPOCH_START, EPOCH_STOP, EVAL_START, EVAL_STOP, EVAL_ACCURACY)

from coco_utils import get_coco_api_from_dataset
from coco_eval import (DefaultCocoEvaluator, NVCocoEvaluator, static_nvcocoevaluator)
import utils
from scaleoutbridge import EmptyObject, ScaleoutBridge as SBridge

from async_executor import async_executor

def preprocessing(images, targets, model_ptr, data_layout):
    # TODO: can be parallelized? should we use DALI? there must be a better way
    target_per_image = []
    for i in range(len(images)):
        # create List[Dict] mapping for targets, used only for preprocessing.
        # only 'boxes', and perhaps 'keypoints', are used for preprocessing.
        dict_ = {}
        dict_['boxes'] = targets['boxes'][i]
        assert ('keypoints' not in targets)
        target_per_image.append(dict_)

    images, targets_ = model_ptr.transform(images, target_per_image)

    # List[Dict] -> Dict[List]
    for i in range(len(targets_)):
        targets['boxes'][i] = targets_[i]['boxes']
    targets_ = None

    images = images.tensors
    if data_layout == 'channels_last':
        images = images.to(memory_format=torch.channels_last)

    return images, targets


def init_scratchpad(images, targets, batch_size, num_classes, amp, fused_focal_loss,
                    max_boxes, cls_head_padded, reg_head_padded, cuda_graphs):
    device = targets['labels'][0].device

    # TODO: should we skip validation in deployment?
    # model_ptr.validate_input(images, targets)

    # one-time init
    if utils.ScratchPad.target_labels_padded is None and cls_head_padded:
        utils.ScratchPad.target_labels_padded = torch.zeros([batch_size, max_boxes + 1], device=device, dtype=torch.int64)
        utils.ScratchPad.target_labels_padded[:, -1] = num_classes if not fused_focal_loss else -1
    if utils.ScratchPad.target_boxes_padded is None and reg_head_padded:
        utils.ScratchPad.target_boxes_padded = torch.zeros([batch_size, max_boxes, 4], device=device)
    if utils.ScratchPad.target_n is None and (cls_head_padded or reg_head_padded):
        utils.ScratchPad.target_n = torch.zeros([batch_size, 1], device=device, dtype=torch.int64)
    if utils.ScratchPad.target_matched_idxs is None and cuda_graphs:
        utils.ScratchPad.target_matched_idxs = torch.zeros_like(targets['matched_idxs'], device=device)

    # these allocations are used to avoid allocations per iteration
    if utils.ScratchPad.gt_classes_target is None:
        if not fused_focal_loss:
            utils.ScratchPad.gt_classes_target = \
                torch.zeros(torch.Size([batch_size, 120087, num_classes + (1 if cls_head_padded else 0)]),
                            dtype=torch.float32 if not amp else torch.float16).to(device)
        else:
            utils.ScratchPad.gt_classes_target = \
                torch.zeros(torch.Size([batch_size, 120087]), device=device, dtype=torch.int64)
    if utils.ScratchPad.batch_size_vector is None:
        utils.ScratchPad.batch_size_vector = torch.arange(len(targets['boxes'])).unsqueeze(1).cuda()

    # data init
    if cls_head_padded:
        utils.ScratchPad.target_labels_padded[:, :-1].fill_(0)
    if reg_head_padded:
        utils.ScratchPad.target_boxes_padded.fill_(0)
    if cuda_graphs:
        utils.ScratchPad.target_matched_idxs.copy_(targets['matched_idxs'])

    for i in range(images.size(0)):
        # debug
        # assert targets['labels'][i].size(0) < max_boxes
        labels_n = targets['labels'][i].size(0)
        if cls_head_padded:
            utils.ScratchPad.target_labels_padded[i][:labels_n] = targets['labels'][i][:labels_n]

            # debug: if args.apex_focal_loss than the -1 pos remains num_classes and not overridden
            # assert ((not fused_focal_loss and (utils.ScratchPad.target_labels_padded[:, -1] == num_classes).all())
            #        or fused_focal_loss)
        if reg_head_padded:
            utils.ScratchPad.target_boxes_padded[i][:labels_n] = targets['boxes'][i][:labels_n]
        if cls_head_padded or reg_head_padded:
            utils.ScratchPad.target_n[i] = labels_n

    utils.ScratchPad.gt_classes_target.fill_(0 if not fused_focal_loss else -1)


def compute_matched_idxs(targets_boxes, model_ptr):
    matched_idxs = model_ptr.get_matched_idxs(targets_boxes)

    return matched_idxs


def loss_preprocessing(targets_boxes, targets_labels, matched_idxs, model_ptr, fused_focal_loss, max_boxes,
                       cls_head_padded, reg_head_padded):
    # classification loss prologues
    if cls_head_padded:
        gt_classes_target, num_foreground, valid_idxs = \
            model_ptr.head.classification_head.compute_loss_prologue_padded(targets_labels,
                                                                            matched_idxs,
                                                                            one_hot=(not fused_focal_loss),
                                                                            max_boxes=max_boxes)
    else:
        gt_classes_target, num_foreground, valid_idxs = \
            model_ptr.head.classification_head.compute_loss_prologue(targets_labels, matched_idxs,
                                                                     one_hot=(not fused_focal_loss))

    # regression loss prologues
    if reg_head_padded:
        target_regression, _, foreground_idxs_mask = \
            model_ptr.head.regression_head.compute_loss_prologue_padded(targets_boxes, matched_idxs, model_ptr.anchors)
    else:
        target_regression, _, foreground_idxs_mask = \
            model_ptr.head.regression_head.compute_loss_prologue(targets_boxes, matched_idxs, model_ptr.anchors)

    return gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask


def compute_loss(model_ptr, cls_logits, bbox_regression, valid_idxs, gt_classes_target, num_foreground,
                 target_regression, foreground_idxs_mask, fused_focal_loss, reg_head_padded):

    cls_loss = model_ptr.head.classification_head.compute_loss_core(cls_logits, gt_classes_target,
                                                                    valid_idxs, num_foreground,
                                                                    fused_focal_loss=fused_focal_loss)

    if reg_head_padded:
        reg_loss = model_ptr.head.regression_head.compute_loss_core_padded(bbox_regression, target_regression,
                                                                           foreground_idxs_mask, num_foreground)
    else:
        reg_loss = model_ptr.head.regression_head.compute_loss_core(bbox_regression, target_regression,
                                                                    foreground_idxs_mask, num_foreground)

    return cls_loss, reg_loss


def train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, args,
                    graphed_model=None, static_input=None, static_loss=None, static_prologues_out=None,
                    sbridge=EmptyObject()):
    mllogger.start(key=EPOCH_START, value=epoch, metadata={"epoch_num": epoch}, sync=True, sync_group=args.train_group)
    sbridge.start_epoch_prof()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # direct pointer to the model
    model_ptr = model.module if args.distributed else model

    lr_scheduler = None
    if epoch < args.warmup_epochs:
        # Convert epochs to iterations
        # we want to control warmup at the epoch level, but update lr every iteration
        start_iter = epoch*len(data_loader)
        warmup_iters = args.warmup_epochs*len(data_loader)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, start_iter, warmup_iters, args.warmup_factor)

    accuracy = None
    for images, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        if args.syn_dataset:
            images = list(image.to(device, non_blocking=True) for image in images)
            images = torch.stack(images)

            targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}
            targets['matched_idxs'] = torch.stack(targets['matched_idxs'])
        else:
            # DALI iterator provides data as needed
            if not args.dali:
                images = list(image.to(device, non_blocking=True) for image in images)
                # arrange "targets" as a Dict[List], instead of a List[Dict], so later it will be easier to use targets
                # data in parallel (e.g., to get the entire batch "boxes", one can just use targets['boxes']).
                # TODO: there might be some unused fields in the targets tensor, so perhaps can avoid some transfers
                targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}

                # preprocessing
                images, targets = preprocessing(images, targets, model_ptr, args.data_layout)

            # DALI can compute matched_idxs and put it in targets, but if it doesn't do so, do it here
            if 'matched_idxs' not in targets:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    targets['matched_idxs'] = compute_matched_idxs(targets['boxes'], model_ptr)

        if not args.cuda_graphs:
            optimizer.zero_grad()

        # init necessary data in the scratchpad
        with torch.cuda.amp.autocast(enabled=args.amp):
            init_scratchpad(images, targets, args.batch_size, args.num_classes, args.amp,
                            args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad,
                            args.cuda_graphs)

        if args.cuda_graphs:
            if args.not_graphed_prologues:
                # loss prologue: preprocess everything that does not require model forward and backward
                # use the padded scratchpad buffers if reg_head_pad/cls_head_pad are toggled
                targets_boxes = targets['boxes'] if not args.reg_head_pad else utils.ScratchPad.target_boxes_padded
                targets_labels = targets['labels'] if not args.cls_head_pad else utils.ScratchPad.target_labels_padded

                gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                    loss_preprocessing(targets_boxes, targets_labels, targets['matched_idxs'], model_ptr,
                                       args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

            sbridge.start_prof(SBridge.FWD_BWD_TIME)

            static_input.copy_(images)
            # All necessary data is copied to the graph buffers in init_scratchpad
            # The graph is programmed to use pointers to the scratchpad (besides images)

            if args.not_graphed_prologues:
                static_prologues_out[0].copy_(gt_classes_target)
                static_prologues_out[1].copy_(target_regression)
                static_prologues_out[2].copy_(num_foreground)
                static_prologues_out[3].copy_(valid_idxs)
                static_prologues_out[4].copy_(foreground_idxs_mask)

            # graphed model comprises loss_preprocessing->forward->compute_loss->backward
            graphed_model.replay()
            if args.sync_after_graph_replay:
                torch.cuda.synchronize()
            sbridge.stop_start_prof(SBridge.FWD_BWD_TIME, SBridge.OPT_TIME)
            scaler.step(optimizer)
            scaler.update()
            sbridge.stop_prof(SBridge.OPT_TIME)

        else:
            with torch.cuda.amp.autocast(enabled=args.amp):
                # loss prologue: preprocess everything that does not require model forward and backward
                # use the padded scratchpad buffers if reg_head_pad/cls_head_pad are toggled
                targets_boxes = targets['boxes'] if not args.reg_head_pad else utils.ScratchPad.target_boxes_padded
                targets_labels = targets['labels'] if not args.cls_head_pad else utils.ScratchPad.target_labels_padded

                gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                    loss_preprocessing(targets_boxes, targets_labels, targets['matched_idxs'], model_ptr,
                                       args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

                # forward
                sbridge.start_prof(SBridge.FWD_TIME)

                model_output = model(images)
                # features = model_output[0:5]
                # head_outputs = {'cls_logits': model_output[5], 'bbox_regression': model_output[6]}

                # loss (given the prologue computations)
                cls_loss, reg_loss = compute_loss(model_ptr, model_output[5], model_output[6], valid_idxs,
                                                  gt_classes_target, num_foreground, target_regression,
                                                  foreground_idxs_mask, args.apex_focal_loss, args.reg_head_pad)
                loss_dict = {'classification': cls_loss, 'bbox_regression': reg_loss}
                losses = sum(loss for loss in loss_dict.values())

                # --- old loss (for debug)
                # loss_dict_ = model_ptr.compute_loss(targets, head_outputs)
                # assert(torch.allclose(loss_dict['classification'], loss_dict_['classification']))
                # assert(torch.allclose(loss_dict['bbox_regression'], loss_dict_['bbox_regression']))

                # reduce losses over all GPUs for logging purposes
                # TODO: remove
                loss_dict_reduced = utils.reduce_dict(loss_dict, group=args.train_group)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
                sbridge.stop_prof(SBridge.FWD_TIME)

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

            # backward
            sbridge.start_prof(SBridge.BWD_TIME)
            scaler.scale(losses).backward()
            sbridge.stop_start_prof(SBridge.BWD_TIME, SBridge.OPT_TIME)
            scaler.step(optimizer)
            scaler.update()
            sbridge.stop_prof(SBridge.OPT_TIME)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if not args.skip_metric_loss:
            if not args.cuda_graphs:
                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            else:
                metric_logger.update(loss=static_loss)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        # Check async coco result
        if args.async_coco and not (metric_logger.current_iter % args.async_coco_check_freq):
            # FIXME(ahmadki): --num-eval-ranks
            if args.eval_rank == 0:
                results = async_executor.pop_if_done()
                # in case of multiple results are returned, get the highest mAP
                if results and len(results) > 0:
                    accuracy = max([result['bbox'][0] for result in results.values() if result], default=-1)

            if args.distributed:
                accuracy = utils.broadcast(accuracy, 0, group=args.train_group)

            if args.target_map and accuracy and accuracy >= args.target_map:
                break


    sbridge.stop_epoch_prof()
    mllogger.end(key=EPOCH_STOP, value=epoch, metadata={"epoch_num": epoch}, sync=True, sync_group=args.train_group)
    summary = metric_logger.summary
    throughput = summary['samples'] / (summary['end_time'] - summary['start_time'])
    mllogger.event(key='tracked_stats', value={'imgs_sec': throughput}, metadata={'step': (epoch + 1)})
    mllogger.event(key='throughput', value=throughput)
    return metric_logger, accuracy


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, args, sbridge=EmptyObject()):
    sbridge.start_eval_prof()
    mllogger.start(key=EVAL_START, value=epoch, log_rank=args.eval_rank==0, metadata={"epoch_num": epoch}, sync=True, sync_group=args.eval_group)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = ["bbox"]
    if args.cocoeval == 'default':
        coco = get_coco_api_from_dataset(data_loader.dataset)
        coco_evaluator = DefaultCocoEvaluator(coco=coco, iou_types=iou_types, group=args.eval_group)
    elif args.cocoeval == 'nvidia':
        coco_evaluator = NVCocoEvaluator(annotations_file=args.val_annotations_file,
            iou_types=iou_types, num_threads=args.coco_threads, group=args.eval_group)
    else:
        assert False, f"Unknown coco evaluator implementation: {args.coco}"


    model_ptr = model.module if args.distributed else model

    for images, targets in metric_logger.log_every(data_loader, args.eval_print_freq, header):
        images = list(img.to(device, non_blocking=True) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # preprocessing
        for i, (image, target) in enumerate(zip(images, targets)):
            # add the original image size to targets
            targets[i]['original_image_size'] = image.shape[-2:]
        images, targets = model_ptr.transform(images, targets)

        images = images.tensors
        if args.data_layout == 'channels_last':
            images = images.to(memory_format=torch.channels_last)

        model_time = time.time()
        with torch.cuda.amp.autocast(enabled=args.amp):
            model_output = model(images)
            features = model_output[0:5]
            head_outputs = {'cls_logits': model_output[5], 'bbox_regression': model_output[6]}

            outputs = model_ptr.eval_postprocess(images, features, targets, head_outputs)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(group=args.eval_group)
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()


    def log_callback(future):
        stats = future.result()
        accuracy = stats['bbox'][0]
        mllogger.event(key=EVAL_ACCURACY, value=accuracy, log_rank=True, metadata={"epoch_num": epoch}, clear_line=True)
        mllogger.end(key=EVAL_STOP, value=epoch, log_rank=True, metadata={"epoch_num": epoch})


    accuracy = None
    if (not args.distributed) or args.eval_rank == 0:
        if args.async_coco:
            async_executor.submit(tag=str(epoch),
                                  fn=static_nvcocoevaluator,
                                  results=coco_evaluator.results,
                                  annotations_file=args.val_annotations_file,
                                  num_threads=args.coco_threads)
            async_executor.add_done_callback(tag=str(epoch), fn=log_callback)
        else:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            accuracy = coco_evaluator.get_stats()['bbox'][0]
            mllogger.event(key=EVAL_ACCURACY, value=accuracy, log_rank=True, metadata={"epoch_num": epoch}, clear_line=True)
            mllogger.end(key=EVAL_STOP, value=epoch, log_rank=True, metadata={"epoch_num": epoch})

    if (not args.async_coco) and args.distributed:
        accuracy = utils.broadcast(accuracy, 0, group=args.eval_group)

    torch.set_num_threads(n_threads)
    sbridge.stop_eval_prof()
    return accuracy
