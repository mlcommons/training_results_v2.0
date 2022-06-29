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

#include "paddle/extension.h"
#include "paddle/fluid/framework/custom_raw_op_kernel_func.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

// x_data[0]: current step which is numbered from 0.
// Note: when computing, we should use x_data[0] + 1.
// y_data[0]: the lr var of this step
__global__ void compute_lr_fwd_kernel(const int64_t* x_data,
                                      float* y_data,
                                      float base_lr,
                                      int64_t max_step) {
  // float res = base_lr * (float(max_step - (x_data[0] + 1) + 1)/max_step);
  float res = base_lr * (static_cast<float>(max_step - x_data[0]) / max_step);
  y_data[0] = res;
}

void compute_lr_forward(const int64_t* x_data,
                        float* out_data,
                        float base_lr,
                        int64_t max_step,
                        cudaStream_t stream) {
  int block = 1;
  int grid = 1;
  compute_lr_fwd_kernel<<<grid, block, 0, stream>>>(
      x_data, out_data, base_lr, max_step);
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_lr, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  auto* out = ctx.Output<f::Tensor>("Out");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();

  float base_lr = ctx.Attr<float>("base_lr");
  int64_t max_step = ctx.Attr<int64_t>("max_step");

  const auto& x_dims = x->dims();
  if (x_dims.size() != 1 || x_dims[0] != 1) {
    PD_THROW("The shape of input x must be [1].");
  }
  auto out_dims = x_dims;
  out->Resize(out_dims);

  const int64_t* x_data = x->data<int64_t>();
  float* out_data = out->mutable_data<float>(x->place());

  compute_lr_forward(x_data, out_data, base_lr, max_step, stream);
}
