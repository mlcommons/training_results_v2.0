# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time
import math
import os

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

import apex_C, amp_C
from apex.multi_tensor_apply import multi_tensor_applier

from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.mlperf_logger import log_event, barrier
from mlperf_logging.mllog import constants
from scaleoutbridge import ScaleoutBridge as SBridge

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

# Instead of zeroing, set parameter grads to None
# Prevents extraneous copy as we're not accumulating
def set_grads_to_none(model):
    for param in model.parameters():
        param.grad = None


class Prefetcher:
    def __init__(self, data_loader, device, max_annotations_per_image):
        self.data_loader = iter(data_loader)
        self.device = device
        self.max_annotations_per_image = max_annotations_per_image
        self.images = None
        self.targets = None
        self.target_bboxes = None
        self.target_objectness = None
        self.target_labels = None
        self.loader_stream = torch.cuda.Stream()
        self.done = False

    def create_padded_target_tensors(self, targets):
        if targets is None:
            target_objectness, target_bboxes, target_labels = None, None, None
        else:
            num_images = len(targets)
            target_objectness = [torch.ones(target.bbox.shape[0], device=target.bbox.device) for target in targets]
            target_labels = [target.get_field("labels") for target in targets]
            if num_images > 1 or self.max_annotations_per_image <= 0:
                target_bboxes = pad_sequence([target.bbox for target in targets], batch_first=True, padding_value=-1)
                target_objectness = pad_sequence(target_objectness, batch_first=True, padding_value=-1)
                target_labels = pad_sequence(target_labels, batch_first = True, padding_value = -1)
            else: 
                target_bboxes = targets[0].bbox
                target_objectness = target_objectness[0]
                target_labels = target_labels[0]
                if self.max_annotations_per_image > 0:
                    #shapes_before = str([list(target_bboxes.shape), list(target_objectness.shape), list(target_labels.shape)])
                    num_anno = target_objectness.shape[0]
                    target_bboxes = pad(target_bboxes, [0,0,0,self.max_annotations_per_image-num_anno], value=-1)
                    target_objectness = pad(target_objectness, [0,self.max_annotations_per_image-num_anno], value=-1)
                    target_labels = pad(target_labels, [0,self.max_annotations_per_image-num_anno], value=-1)
                    #shapes_after = str([list(target_bboxes.shape), list(target_objectness.shape), list(target_labels.shape)])
                    #print("%s -> %s" % (shapes_before, shapes_after))
                target_bboxes = target_bboxes.unsqueeze(0)
                target_objectness = target_objectness.unsqueeze(0)
                target_labels = target_labels.unsqueeze(0)
            target_bboxes.requires_grad = False
            target_objectness.requires_grad = False
            target_labels.requires_grad = False
            #print("self.max_annotations_per_image = %d, target_bboxes = %s,%s, target_objectness = %s,%s, target_labels = %s,%s" % (self.max_annotations_per_image, str(list(target_bboxes.shape)), str(target_bboxes.dtype), str(list(target_objectness.shape)), str(target_objectness.dtype), str(list(target_labels.shape)), str(target_labels.dtype)))
        return target_bboxes, target_objectness, target_labels

    def __iter__(self):
        return self

    def prefetch_CPU(self):
        try:
            with torch.no_grad():
                with torch.cuda.stream(self.loader_stream):
                    self.images, self.targets, _ = next(self.data_loader)
                    self.target_bboxes, self.target_objectness, self.target_labels = None, None, None
        except StopIteration:
            self.images, self.targets, self.target_bboxes, self.target_objectness, self.target_labels = None, None, None, None, None
            self.done = True

    def prefetch_GPU(self):
        if self.images is not None:
            with torch.no_grad():
                with torch.cuda.stream(self.loader_stream):
                    self.images = self.images.to(self.device)
                    self.targets = [target.to(self.device, non_blocking=True) for target in self.targets]
                    self.target_bboxes, self.target_objectness, self.target_labels = self.create_padded_target_tensors(self.targets)

    def __next__(self):
        if self.images is None and not self.done:
            self.prefetch_CPU()
            self.prefetch_GPU()
        torch.cuda.current_stream().wait_stream(self.loader_stream)
        if self.done:
            raise StopIteration()
        else:
            targets = self.targets
            images, target_bboxes, target_objectness, target_labels = self.images, self.target_bboxes, self.target_objectness, self.target_labels
            self.images, self.targets, self.target_bboxes, self.target_objectness, self.target_labels = None, None, None, None, None
            return images, (target_bboxes, target_objectness, target_labels, targets)


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    disable_allreduce_for_logging,
    disable_loss_logging,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None,
    final_callback_fn=None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    distributed = arguments["distributed"]
    num_training_ranks = arguments["num_training_ranks"]
    training_comm = arguments["training_comm"]
    dedicated_evaluation_ranks = arguments["dedicated_evaluation_ranks"]
    images_per_gpu_train = arguments["images_per_gpu_train"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    sbridge = None

    time_tags = int(os.getenv('TIME_TAGS', 0))
    nvtx_flag = int(os.getenv('NVTX_FLAG', 0))

    synchronize(training_comm)
    optimizer.zero_grad()
    prefetcher = Prefetcher(data_loader, device, arguments["max_annotations_per_image"])

    if time_tags or nvtx_flag:
        torch.cuda.synchronize()
        epoch_start = time.time()
        sbridge = SBridge(300)

    if distributed:
        params = [p for p in model.parameters() if p.requires_grad]
        overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
        gradient_scaler = 1.0 / num_training_ranks
    else:
        overflow_buf = None

    vss = []
    for iteration, (images, targets) in enumerate(prefetcher, start_iter):
        if per_iter_start_callback_fn is not None:
            per_iter_start_callback_fn(iteration=iteration)

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        fwd_start = 0
        if time_tags:
            torch.cuda.synchronize()
            fwd_start = time.time()
        if nvtx_flag:
            sbridge.push_nvtx(SBridge.FWD_TIME)

        if images_per_gpu_train == 1:
            if distributed:
                torch.distributed.barrier(group=training_comm) # Sync all processes before training starts to prevent CPUs from getting too far ahead of GPUs
            else:
                torch.cuda.synchronize()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        bwd_start = 0
        if time_tags:
            torch.cuda.synchronize()
            bwd_start = time.time()
            sbridge.add_tag(SBridge.FWD_TIME, bwd_start-fwd_start, dist.get_rank())
        if nvtx_flag:
            sbridge.pop_nvtx()
            sbridge.push_nvtx(SBridge.BWD_TIME)

        # optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with optimizer.scale_loss(losses) as scaled_losses:
        optimizer.backward(losses)

        if distributed:
            # gradient reduction
            grads = [p.grad for p in params]
            flat_grads = apex_C.flatten(grads)
            grad_redux = torch.distributed.all_reduce(flat_grads, group=training_comm, async_op=True)

        # At this point we are waiting for kernels launched by cuda graph to finish, so CPU is idle.
        # Take advantage of this by loading next input batch before calling step
        prefetcher.prefetch_CPU()

        opt_start = 0
        if time_tags:
            torch.cuda.synchronize()
            opt_start = time.time()
            sbridge.add_tag(SBridge.BWD_TIME, opt_start-bwd_start, dist.get_rank())
        if nvtx_flag:
            sbridge.pop_nvtx()
            sbridge.push_nvtx(SBridge.OPT_TIME)

        if distributed:
            grad_redux.wait()
            overflow_buf.zero_()
            multi_tensor_applier(
                    amp_C.multi_tensor_scale,
                    overflow_buf,
                    [apex_C.unflatten(flat_grads, grads), grads],
                    gradient_scaler)

        optimizer.step(overflow_buf)  # This will sync
        prefetcher.prefetch_GPU()

        if time_tags:
            torch.cuda.synchronize()
            if not sbridge.add_tag(SBridge.OPT_TIME, time.time()-opt_start, dist.get_rank()):
                sbridge.empty_qs()
                time_tags = 0
        if nvtx_flag:
            sbridge.pop_nvtx()

        will_report_this_iteration = (iteration % 50 == 0 or iteration == max_iter)

        # reduce losses over all GPUs for logging purposes
        if not disable_loss_logging and not disable_allreduce_for_logging:
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if math.isfinite(losses_reduced):
                meters.update(loss=losses_reduced, **loss_dict_reduced)
        else:
            # Optimization: 
            # Cat all meter updates + finite check if they are all single item tensors
            # This reduces number of D2H transfers to 1.
            ks,vs = zip(*[(k, v.unsqueeze(dim=0)) for (k, v) in loss_dict.items()])
            if disable_loss_logging:
                vs = torch.zeros([len(vs)], dtype=torch.float32)
            else:
                vs = list(vs)
                vs.append( losses.unsqueeze(dim=0) )
                vs = torch.cat(vs)
            vss.append(vs)
            if will_report_this_iteration:
                vss = torch.stack(vss).cpu() # will sync
                for vs in vss:
                    vs = [v.item() for v in list(vs.split(split_size=1))]
                    losses_host = vs.pop(-1)
                    if math.isfinite(losses_host):
                        loss_dict = {k: v for (k, v) in zip(ks,vs)}
                        meters.update(loss=losses_host, **loss_dict)
                vss = []

        # set_grads_to_none(model)
        optimizer.zero_grad()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 50 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            # for mlperf throughput - DLFW CI/CD

            meter_str = {}
            for name, meter in meters.meters.items():
                meter_str[name] = meter.global_avg
            throughput = arguments['ims_per_batch'] / float(meter_str['time'])
            meter_str['throughput'] = throughput
            log_event(key='tracked_stats',
                      value= meter_str,
                      metadata={"step": (iteration)})
            log_event(key='throughput',
                      value= throughput)

        if iteration % checkpoint_period == 0 and arguments["save_checkpoints"]:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter and arguments["save_checkpoints"]:
            checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            # Note: iteration has been incremented previously for
            # human-readable checkpoint names (i.e. 60000 instead of 59999)
            # so need to adjust again here
            early_exit = per_iter_end_callback_fn(iteration=iteration-1)
            if early_exit:
                break

    if final_callback_fn is not None and not early_exit:
        if final_callback_fn():
            early_exit = True

    if time_tags or nvtx_flag:
        sbridge.empty_qs()
        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        log_event(key=SBRIDGE.EPOCH_TIME, value={'r':dist.get_rank(), 't':epoch_time}, log_all_ranks=True)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if per_iter_end_callback_fn is not None:
        if early_exit:
            return True
        else:
            return False
    else:
        return None
