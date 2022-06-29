# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" example train fit utility """
import logging
import os
import time
import re
import math
import mxnet as mx
import horovod.mxnet as hvd
import numpy as np

#### imports needed for fit monkeypatch
from mxnet.initializer import Uniform
from mxnet.context import cpu
from mxnet.monitor import Monitor
from mxnet.model import BatchEndParam
from mxnet.initializer import Uniform
from mxnet.io import DataDesc, DataIter, DataBatch
from mxnet.base import _as_list
from mxnet import cuda_utils as cu
import copy
##### imports needed for custom optimizer
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply, where,multi_sum_sq, multi_lars_v2, broadcast_greater,
                           broadcast_greater_equal, broadcast_mul, broadcast_div, broadcast_sub, broadcast_add, broadcast_power)
from mxnet.ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                           mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update,
                           signsgd_update, signum_update,
                           multi_sgd_update, multi_sgd_mom_update, multi_mp_sgd_update,
                           multi_sgd_mom_update_v2, multi_mp_sgd_mom_update_v2,
                           multi_mp_sgd_mom_update,
                           lars_multi_sgd_update, lars_multi_sgd_mom_update, lars_multi_sgd_mom_update_v2,
                           lars_multi_mp_sgd_update, lars_multi_mp_sgd_mom_update, lars_multi_mp_sgd_mom_update_v2)
from mxnet.ndarray import sparse
#####

from mlperf_log_utils import mx_resnet_print_event, mx_resnet_print_start, \
                             mx_resnet_print_end, all_reduce, mpiwrapper, \
                             mpiwrapper
from mlperf_logging.mllog import constants
from mxnet import cuda_utils as cu
from scaleoutbridge import ScaleoutBridge as SBridge

from common.data import SyntheticDataIter

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

@register
class SGDwFASTLARSV2(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    lazy_update : bool, optional
        Default is True. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.::

            False: results in using the same precision as the weights (default),
            True: makes internal 32-bit copy of the weights and applies gradients
            in 32-bit precision even if actual weights used in the model have lower precision.
            Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, base_lr, end_lr, lr_decay_poly_power, 
            warmup_steps, total_steps,
            momentum=0.0, lazy_update=True, lars=True, lars_eta=0.001, lars_eps=0, **kwargs):
        super(SGDwFASTLARSV2, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))
        self.lars = True
        self.lars_eta = lars_eta
        self.lars_eps = lars_eps
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.lr_decay_poly_power = lr_decay_poly_power
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        self.skip = 0
        self.last_lr = None
        self.cur_lr = None
        self.use_cached = False 
        self.use_sgd_cached = False 
        self.full_index = 55
        self.cur_step = mx.nd.array([1.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.next_step = mx.nd.array([1.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.new_lrs = mx.nd.array([0.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.base_momentum = mx.nd.array([self.momentum] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.scaled_momentum = mx.nd.array([self.momentum] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.poly_lrs = mx.nd.array([0.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.old_poly_lrs = mx.nd.array([1.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.new_wds = mx.nd.array([kwargs['wd']] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.sgd_wds = mx.nd.array([0.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.w_sum_sq = mx.nd.array([0.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.g_sum_sq = mx.nd.array([0.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
        self.ones_gpu = mx.nd.array([1.0] * self.full_index, ctx=mx.gpu(hvd.local_rank()), dtype='float32')
    
    def reset_steps(self):
        broadcast_mul(self.scaled_momentum, 
                self.ones_gpu, 
                out = self.scaled_momentum)
        broadcast_mul(self.ones_gpu, 
                self.ones_gpu, 
                out = self.ones_gpu)
    
    
    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            is_weight = n.endswith('_weight')
            is_fc_bias = 'fc' in n and 'bias' in n
            if not (is_weight or is_fc_bias):
                self.wd_mult[n] = 0.0

        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        wds = self._get_wds(indices)
        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum 
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if aggregate:
            nb_params = len(indices)
            names = [self.idx2name[i] if i in self.idx2name else str(i) for i in indices]
            lars_idx = [i for i in range(nb_params) if not(names[i].endswith('gamma')
                        or names[i].endswith('beta') or names[i].endswith('bias'))]
            if self.lars and len(lars_idx) > 0:
                nb_lars = len(lars_idx)
                no_lars_idx = [i for i in range(nb_params) if (names[i].endswith('gamma') or
                               names[i].endswith('beta') or names[i].endswith('bias'))]
                cur_ctx = weights[0].context
                full_idx = lars_idx + no_lars_idx
                if not self.use_cached:
                    self.use_cached = True
                else:
                    self.old_poly_lrs = self.poly_lrs.copy()
                new_weights = [weights[i] for i in full_idx]
                new_grads = [grads[i] for i in full_idx]
                multi_sum_sq(*new_weights[:nb_lars], num_arrays=nb_lars, out=self.w_sum_sq[:nb_lars])
                multi_sum_sq(*new_grads[:nb_lars], num_arrays=nb_lars, out=self.g_sum_sq[:nb_lars])
                multi_lars_v2(self.w_sum_sq[:nb_lars], self.g_sum_sq[:nb_lars],
                           self.new_wds[:nb_lars], self.cur_step[:nb_lars],
                           eta=self.lars_eta, eps=self.lars_eps, rescale_grad=self.rescale_grad,
                           total_steps=self.total_steps,
                           warmup_steps=self.warmup_steps,
                           base_lr=self.base_lr,
                           end_lr=self.end_lr,
                           lr_decay_poly_power=self.lr_decay_poly_power,
                           out = (self.new_lrs[:nb_lars],self.poly_lrs[:nb_lars], self.next_step[:nb_lars]))
                new_states = [states[i] for i in full_idx]
                broadcast_mul(self.base_momentum[:nb_lars], self.poly_lrs[:nb_lars], out = self.scaled_momentum[:nb_lars])
                broadcast_div(self.scaled_momentum[:nb_lars], self.old_poly_lrs[:nb_lars], out = self.scaled_momentum[:nb_lars])
                #We are doing self.new_lrs[nb_lars:] = self.poly_lrs[:len(full_idx)-nb_lars] but in place
                self.new_lrs.slice_assign(self.poly_lrs[:len(full_idx)-nb_lars], (nb_lars), (len(full_idx)), (None))
                self.next_step.copyto(self.cur_step[:])
                sidx = 0
                while sidx < len(indices):
                    eidx = sidx + len(new_weights[sidx:sidx+self.aggregate_num])
                    if not multi_precision:
                        if self.momentum > 0:
                            lars_multi_sgd_mom_update_v2(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           new_states[sidx:eidx])),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        self.scaled_momentum[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            lars_multi_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                            new_grads[sidx:eidx])),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    else:
                        if self.momentum > 0:
                            lars_multi_mp_sgd_mom_update_v2(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           *zip(*new_states[sidx:eidx]))),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        self.scaled_momentum[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            lars_multi_mp_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           list(zip(*new_states[sidx:eidx]))[1])),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    sidx += self.aggregate_num
            else:
                current_index = 0
                while current_index < len(indices):
                    sidx = current_index
                    eidx = current_index + self.aggregate_num
                    if not multi_precision:
                        if self.momentum > 0:
                            multi_sgd_mom_update_v2(*_flatten_list(zip(weights[sidx:eidx],
                                                                    grads[sidx:eidx],
                                                                    states[sidx:eidx])),
                                                 self.poly_lrs[0:self.aggregate_num],
                                                 self.sgd_wds[sidx:eidx],
                                                 self.base_momentum[0:self.aggregate_num],
                                                 out=weights[sidx:eidx],
                                                 num_weights=len(weights[sidx:eidx]),
                                                 **kwargs)
                        else:
                            assert False, "Mom always > 0" 
                    else:
                        if self.momentum > 0:
                            multi_mp_sgd_mom_update_v2(*_flatten_list(zip(weights[sidx:eidx],
                                                                       grads[sidx:eidx],
                                                                       *zip(*states[sidx:eidx]))),
                                                       self.poly_lrs[0:self.aggregate_num],
                                                       self.sgd_wds[sidx:eidx],
                                                       self.base_momentum[sidx:eidx],
                                                       out=weights[sidx:eidx],
                                                       num_weights=len(weights[sidx:eidx]),
                                                       **kwargs)
                        else:
                            assert False, "Mom always > 0" 
                    current_index += self.aggregate_num
        else:
            assert False, "aggregate for optimizer should be set to true" 
            
    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == np.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == np.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)
