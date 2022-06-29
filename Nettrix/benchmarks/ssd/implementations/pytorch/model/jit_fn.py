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

import warnings
import torch
import torch.nn.functional as F
from torch import Tensor, HalfTensor, BoolTensor
from typing import Callable, List, Optional, Tuple
from model.frozen_bn import FrozenBatchNorm2d


# For debugging backprop put the following in the function and uncomment
# import pydevd
# pydevd.settrace(suspend=False, trace_only_current_thread=True)


class bn_relu_wrapper(FrozenBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, n=None):
        super(bn_relu_wrapper, self).__init__(num_features, eps, n)

    def forward(self, x):
        return bn_relu_jit.apply(x, self.scale, self.bias_term)


class bn_relu_jit(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input, scale, bias):
        bn_relu_out, relu_mask = fwd_bn_relu_jit(input, scale, bias)

        ctx.save_for_backward(scale, relu_mask)
        return bn_relu_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        scale, relu_mask = ctx.saved_tensors

        grad_input = bwd_bn_relu_jit(grad_output, scale, relu_mask)
        return grad_input, None, None


@torch.jit.script
def fwd_bn_relu_jit(input: HalfTensor, scale: HalfTensor, bias: HalfTensor) -> Tuple[HalfTensor, BoolTensor]:
    bn = input * scale + bias
    bn_relu = torch.nn.functional.relu(bn)
    relu_mask = bn > 0
    return bn_relu, relu_mask


@torch.jit.script
def bwd_bn_relu_jit(grad_output: HalfTensor, scale: HalfTensor, relu_mask: BoolTensor) -> HalfTensor:
    grad_input = grad_output * scale
    grad_input = grad_input * relu_mask

    return grad_input


class bn_add_relu_jit(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input1, scale1, bias1, input2):
        bn_relu_out, relu_mask = fwd_bn_add_relu_jit(input1, scale1, bias1, input2)

        ctx.save_for_backward(scale1, relu_mask)
        return bn_relu_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        scale, relu_mask = ctx.saved_tensors

        grad_input1, grad_input2 = bwd_bn_add_relu_jit(grad_output, scale, relu_mask)
        return grad_input1, None, None, grad_input2


@torch.jit.script
def fwd_bn_add_relu_jit(input1: HalfTensor, scale1: HalfTensor, bias1: HalfTensor,
                        input2: HalfTensor) -> Tuple[HalfTensor, BoolTensor]:
    bn = input1 * scale1 + bias1
    bn_add = bn + input2
    bn_add_relu = torch.nn.functional.relu(bn_add)
    relu_mask = bn_add > 0
    return bn_add_relu, relu_mask


@torch.jit.script
def bwd_bn_add_relu_jit(grad_output: HalfTensor, scale: HalfTensor,
                        relu_mask: BoolTensor) -> Tuple[HalfTensor, HalfTensor]:
    grad_input2 = grad_output * relu_mask
    grad_input1 = grad_input2 * scale

    return grad_input1, grad_input2


class bn_bn_add_relu_jit(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input1, scale1, bias1, input2, scale2, bias2):
        bn_relu_out, relu_mask = fwd_bn_bn_add_relu_jit(input1, scale1, bias1,
                                                        input2, scale2, bias2)

        ctx.save_for_backward(scale1, scale2, relu_mask)
        return bn_relu_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        scale1, scale2, relu_mask = ctx.saved_tensors

        grad_input1, grad_input2 = bwd_bn_bn_add_relu_jit(grad_output, scale1, scale2, relu_mask)
        return grad_input1, None, None, grad_input2, None, None


@torch.jit.script
def fwd_bn_bn_add_relu_jit(input1: HalfTensor, scale1: HalfTensor, bias1: HalfTensor,
                           input2: HalfTensor, scale2: HalfTensor, bias2: HalfTensor) -> Tuple[HalfTensor, BoolTensor]:
    bn1 = input1 * scale1 + bias1
    bn2 = input2 * scale2 + bias2
    bn_add = bn1 + bn2
    bn_add_relu = torch.nn.functional.relu(bn_add)
    relu_mask = bn_add > 0
    return bn_add_relu, relu_mask


@torch.jit.script
def bwd_bn_bn_add_relu_jit(grad_output: HalfTensor, scale1: HalfTensor, scale2: HalfTensor,
                           relu_mask: BoolTensor) -> Tuple[HalfTensor, HalfTensor]:
    grad_output_masked = grad_output * relu_mask
    grad_input1 = grad_output_masked * scale1
    grad_input2 = grad_output_masked * scale2

    return grad_input1, grad_input2
