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
from engine import preprocessing, init_scratchpad, loss_preprocessing, compute_loss, compute_matched_idxs


def whole_model_capture(model, optimizer, scaler, dataset, args):
    model.train()

    # direct pointer to the model
    model_ptr = model.module if args.distributed else model

    # extracting the device name from some layer
    device = model_ptr.backbone.body.conv1.weight.device

    # Convert epochs to iterations
    # we want to control warmup at the epoch level, but update lr every iteration
    start_iter = 0
    dataset_len = len(dataset) if dataset is not None else int(args.train_sz / args.batch_size)
    warmup_iters = args.warmup_epochs * dataset_len
    lr_scheduler = utils.warmup_lr_scheduler(optimizer, start_iter, warmup_iters, args.warmup_factor)

    if args.cuda_graphs_syn:
        assert (dataset is None)

        images, targets = [], {'boxes': [], 'labels': []}
        for b in range(args.batch_size):
            # These are just arbitrary sizes for model capture
            images.append(torch.rand([3, 1000, 1000], device=device))
            targets['boxes'].append(torch.tensor([[10, 20, 30, 40]], device=device))
            targets['labels'].append(torch.tensor([1], device=device))
        images, targets = preprocessing(images, targets, model_ptr, args.data_layout)
    else:
        images, targets = [], []

        # taking the first batch
        for images_, targets_ in dataset:
            images = images_
            targets = targets_
            break

        # if not DALI, then we should preprocess the data
        if not args.dali:
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}

            # --- preprocessing
            images, targets = preprocessing(images, targets, model_ptr, args.data_layout)

    # DALI can compute matched_idxs and put it in targets, but if it doesn't do so, do it here
    if 'matched_idxs' not in targets:
        with torch.cuda.amp.autocast(enabled=args.amp):
            targets['matched_idxs'] = compute_matched_idxs(targets['boxes'], model_ptr)

    with torch.cuda.amp.autocast(enabled=args.amp):
        init_scratchpad(images, targets, args.batch_size, args.num_classes, args.amp,
                        args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad,
                        args.cuda_graphs)

        if args.not_graphed_prologues:
            gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                loss_preprocessing(utils.ScratchPad.target_boxes_padded if args.reg_head_pad else targets['boxes'],
                                   utils.ScratchPad.target_labels_padded if args.cls_head_pad else targets['labels'],
                                   utils.ScratchPad.target_matched_idxs, model_ptr,
                                   args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

    static_matched_idxs = torch.zeros_like(targets['matched_idxs'])
    static_matched_idxs.copy_(targets['matched_idxs'])

    # --- warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for j in range(11):
            if args.apex_adam:
                # set_to_none is True by default
                optimizer.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                if not args.not_graphed_prologues:
                    # preprocess everything that does not require model forward and backward
                    gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                        loss_preprocessing(utils.ScratchPad.target_boxes_padded if args.reg_head_pad else targets['boxes'],
                                           utils.ScratchPad.target_labels_padded if args.cls_head_pad else targets['labels'],
                                           utils.ScratchPad.target_matched_idxs, model_ptr,
                                           args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

                # forward
                model_output = model(images)
                # features = model_output[0:5]
                # head_outputs = {'cls_logits': model_output[5], 'bbox_regression': model_output[6]}

                cls_loss, reg_loss = compute_loss(model_ptr, model_output[5], model_output[6], valid_idxs,
                                                  gt_classes_target, num_foreground, target_regression,
                                                  foreground_idxs_mask, args.apex_focal_loss, args.reg_head_pad)

                losses = cls_loss + reg_loss

            # backward
            scaler.scale(losses).backward()

            # optimizer
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)

    # --- capture
    g = torch.cuda.CUDAGraph()

    if args.apex_adam:
        # set_to_none is True by default
        optimizer.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)

    with torch.cuda.graph(g):
        with torch.cuda.amp.autocast(enabled=args.amp):
            if not args.not_graphed_prologues:
                # loss_preprocessing is now part of the graph
                gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                    loss_preprocessing(utils.ScratchPad.target_boxes_padded if args.reg_head_pad else targets['boxes'],
                                       utils.ScratchPad.target_labels_padded if args.cls_head_pad else targets['labels'],
                                       utils.ScratchPad.target_matched_idxs, model_ptr,
                                       args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

            # forward
            static_model_output = model(images)

            # loss
            static_cls_loss, static_reg_loss = compute_loss(model_ptr, static_model_output[5], static_model_output[6],
                                                            valid_idxs, gt_classes_target, num_foreground,
                                                            target_regression, foreground_idxs_mask,
                                                            args.apex_focal_loss, args.reg_head_pad)

            static_loss = static_cls_loss + static_reg_loss

        # backward
        scaler.scale(static_loss).backward()

    if args.not_graphed_prologues:
        static_prologues_out = [gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask]
    else:
        static_prologues_out = None

    return g, images, static_loss, static_prologues_out


