// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/custom_raw_op_kernel_func.h"

template <typename T, bool NeedAccumulate>
static __device__ __forceinline__ void AccMerge(T acc, T total, T *out) {
  auto correct = static_cast<int64_t>(acc * total + 0.5);
  if (NeedAccumulate) {
    out[0] += correct;
    out[1] += total;
  } else {
    out[0] = correct;
    out[1] = total;
  }
}

template <typename T1, typename T2, bool NeedAccumulate>
static __global__ void AccMergeKernelCPUTotal(const T1 *acc,
                                              int64_t total,
                                              T2 *out) {
  AccMerge<T2, NeedAccumulate>(*acc, total, out);
}

template <typename T1, typename T2, bool NeedAccumulate>
static __global__ void AccMergeKernelGPUTotal(const T1 *acc,
                                              const T1 *total,
                                              T2 *out) {
  AccMerge<T2, NeedAccumulate>(*acc, *total, out);
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(acc_merge, ctx) {
  namespace f = paddle::framework;
  namespace p = paddle::platform;

  auto &step_t = *ctx.Output<f::Tensor>("Step");
  auto *step = step_t.data<int64_t>();
  if (step[1] <= 0) return;

  const auto &total_t = *ctx.Input<f::Tensor>("Total");
  bool is_cpu_place = p::is_cpu_place(total_t.place());

  using Type1 = float;
  using Type2 = double;

  const auto &acc_t = *ctx.Input<f::Tensor>("Acc");
  auto *acc = acc_t.data<Type1>();

  auto &out_t = *ctx.Output<f::Tensor>("Out");
  out_t.Resize({2});
  auto *out = out_t.mutable_data<Type2>(acc_t.place());

  auto stream = ctx.cuda_device_context().stream();
  if (step[0] == 0) {
    if (is_cpu_place) {
      AccMergeKernelCPUTotal<Type1, Type2, false><<<1, 1, 0, stream>>>(
          acc, *total_t.data<int64_t>(), out);
    } else {
      AccMergeKernelGPUTotal<Type1, Type2, false><<<1, 1, 0, stream>>>(
          acc, total_t.data<float>(), out);
    }
  } else {
    if (is_cpu_place) {
      AccMergeKernelCPUTotal<Type1, Type2, true><<<1, 1, 0, stream>>>(
          acc, *total_t.data<int64_t>(), out);
    } else {
      AccMergeKernelGPUTotal<Type1, Type2, true><<<1, 1, 0, stream>>>(
          acc, total_t.data<Type1>(), out);
    }
  }

  step[0] = (step[0] + 1) % step[1];
}
