# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.

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
from function import graph
from apex import amp
import time
from scaleoutbridge import EmptyObject, ScaleoutBridge as SBridge

def preprocess_batch(args, input_ids, segment_ids, input_mask, labels_mlm, labels_nsp, input_only=False):
    b, s = input_ids.shape
    if args.pad_fmha:
        seqlens = input_mask.sum(-1, dtype=torch.int32)
        cu_seqlens = torch.zeros(b+1, dtype=torch.int32, device=seqlens.device)
        cu_seqlens[1:] = torch.cumsum(seqlens, 0)

        position_ids = torch.cat([torch.arange(s, dtype=torch.int64, device=seqlens.device) for _ in range(b)]).view(b,s)

        def compact(t):
            '''Removes per-sequence padding and adds all padding to the end of the batch.
            Thus, the output will still be [batch_size x seq_len].
            '''
            t_compact = torch.zeros_like(t).view(-1)
            for it in range(b):
                si = seqlens[it]
                begin = cu_seqlens[it]
                end = cu_seqlens[it +  1]
                t_compact[begin:end] = t[it, :si]
            return t_compact.view(t.shape)

        iids = compact(input_ids)
        sids = compact(segment_ids)
        pids = compact(position_ids)
        lmlm = compact(labels_mlm)

        if input_only:
            return iids, sids, cu_seqlens, pids

        return iids, sids, cu_seqlens, lmlm, labels_nsp, pids

    if input_only:
        return input_ids, segment_ids, input_mask
    return input_ids, segment_ids, input_mask, labels_mlm, labels_nsp

class FwdLossBwdTrainer():

    def __init__(self, args, grad_scaler):
        super(FwdLossBwdTrainer, self).__init__()
        self.args = args
        self.grad_scaler = grad_scaler
        self.capture_stream = torch.cuda.Stream()

        self.send_stats_in_parallel = False
        self.stats_stream = torch.cuda.Stream()
        self.loss_cpu = torch.tensor(0.0, dtype=torch.float32, device='cpu').pin_memory()
        self.mlm_acc_cpu = torch.tensor(0.0, dtype=torch.float32, device='cpu').pin_memory()

    def capture_bert_model_segment_graph(self, bert_model, use_cuda_graph):
        # eval batch depends on the rank, since eval sample count isn't divisible by world size
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        eval_batch_min = self.args.num_eval_examples // world_size
        remainder = self.args.num_eval_examples % world_size
        if rank<remainder:
            eval_batch = eval_batch_min + 1
        else:
            eval_batch = eval_batch_min
        eval_batch = min(eval_batch, self.args.eval_batch_size)
        batches_to_graph = [eval_batch, self.args.train_batch_size]
        
        bert_model_segment = bert_model.bert_model_segment
        sample_train = [
                 torch.ones(self.args.train_batch_size, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(self.args.train_batch_size, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(self.args.train_batch_size, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(self.args.train_batch_size, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(self.args.train_batch_size, dtype=torch.int64, device=self.args.device),
                 ]
        sample_eval = [
                 torch.ones(eval_batch, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(eval_batch, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(eval_batch, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(eval_batch, self.args.max_seq_length, dtype=torch.int64, device=self.args.device),
                 torch.ones(eval_batch, dtype=torch.int64, device=self.args.device),
                 ]  
        sample_model_train = preprocess_batch(self.args, *sample_train, input_only=True)
        sample_model_eval = preprocess_batch(self.args, *sample_eval, input_only=True)
        bert_model_segment = graph(bert_model_segment,
                                    tuple(t.clone() for t in sample_model_train),
                                    tuple(t.clone() for t in sample_model_eval) if self.args.eval_batch_size * world_size >= self.args.num_eval_examples else None,
                                    self.capture_stream,
                                    warmup_iters=8,
                                    warmup_only=(not use_cuda_graph))

        bert_head_segment = bert_model.heads_only_segment
        sample_head_train = [
                torch.ones(self.args.train_batch_size, self.args.max_seq_length, 1024, dtype=torch.float16, device=self.args.device),
                torch.ones(self.args.train_batch_size,                           1024, dtype=torch.float16, device=self.args.device),
                torch.ones(self.args.train_batch_size, self.args.max_seq_length,       dtype=torch.int64, device=self.args.device),
                torch.ones(self.args.train_batch_size,                                 dtype=torch.int64, device=self.args.device),
                ]
        sample_head_eval = [
                torch.ones(eval_batch, self.args.max_seq_length, 1024, dtype=torch.float16, device=self.args.device),
                torch.ones(eval_batch,                           1024, dtype=torch.float16, device=self.args.device),
                torch.ones(eval_batch, self.args.max_seq_length,       dtype=torch.int64, device=self.args.device),
                torch.ones(eval_batch,                                 dtype=torch.int64, device=self.args.device),
                ]
        sample_head_tuple_train = tuple([sample_head_train[0].clone().requires_grad_(), sample_head_train[1].clone().requires_grad_(), sample_head_train[2].clone(), sample_head_train[3].clone()])
        sample_head_tuple_eval = tuple([sample_head_eval[0].clone(), sample_head_eval[1].clone(), sample_head_eval[2].clone(), sample_head_eval[3].clone()])
        bert_head_segment = graph(bert_head_segment,
                                               sample_head_tuple_train,
                                               sample_head_tuple_eval if self.args.eval_batch_size * world_size >= self.args.num_eval_examples else None,
                                               self.capture_stream,
                                               warmup_iters=8,
                                               warmup_only=(not use_cuda_graph))


        return bert_model

    def eval_step(self, batch, model):
        model.eval()
        loss = None
        mlm_acc = None

        loss, mlm_acc, num_valid = model(*batch)
        return loss, mlm_acc, num_valid

    def step(self, step, batch, model, optimizer, sbridge=EmptyObject()):
        loss = None
        mlm_acc = None

        sbridge.start_prof(SBridge.FWD_TIME)
        loss, mlm_acc, _ = model(*batch)

        if self.send_stats_in_parallel:
            self.stats_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stats_stream):
                self.loss_cpu.copy_(loss.detach(), non_blocking=True)
                self.mlm_acc_cpu.copy_(mlm_acc.detach(), non_blocking=True)

        sbridge.stop_start_prof(SBridge.FWD_TIME, SBridge.BWD_TIME)

        if self.args.bypass_amp:
            loss.backward()
        elif self.args.distributed_lamb:
            optimizer._lazy_init_stage1()
            self.grad_scaler.scale(loss).backward()
            optimizer._lazy_init_stage2()
        else:
            with amp.scale_loss(loss, optimizer, delay_overflow_check=self.args.allreduce_post_accumulation) as scaled_loss:
                scaled_loss.backward()
        sbridge.stop_prof(SBridge.BWD_TIME)

        if self.send_stats_in_parallel:
            self.stats_stream.synchronize()
            loss = self.loss_cpu
            mlm_acc = self.mlm_acc_cpu

        return loss, mlm_acc, sbridge
