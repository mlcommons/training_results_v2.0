// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dropout_residual_ln, ctx) {
  namespace f = paddle::framework;
  const auto &x = *ctx.Input<f::Tensor>("X");
  const auto &residual = *ctx.Input<f::Tensor>("Residual");
  const auto &ln_scale = *ctx.Input<f::Tensor>("LnScale");
  const auto &ln_bias = *ctx.Input<f::Tensor>("LnBias");
  auto &final_out = *ctx.Output<f::Tensor>("Out");
  auto &dropout_mask_out = *ctx.Output<f::Tensor>("DropoutMask");
  auto &ln_mean = *ctx.Output<f::Tensor>("LnMean");
  auto &ln_var = *ctx.Output<f::Tensor>("LnVar");
  auto &dropout_residual_out = *ctx.Output<f::Tensor>("DropoutResidualOut");

  auto ln_epsilon = ctx.Attr<float>("ln_epsilon");
  auto is_test = ctx.Attr<bool>("is_test");
  auto fix_seed = ctx.Attr<bool>("fix_seed");
  auto seed_val = ctx.Attr<int>("seed_val");
  auto is_upscale_in_train = ctx.Attr<bool>("is_upscale_in_train");
  auto dropout_rate = ctx.Attr<float>("dropout_rate");
  auto &dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  // inputs
  const auto &x_dims = x.dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_n = x_dims[x_dims.size() - 1];

  // outputs
  final_out.Resize(x_dims);
  dropout_mask_out.Resize(x_dims);
  dropout_residual_out.Resize(x_dims);
  ln_mean.Resize({x_m});
  ln_var.Resize({x_m});

  paddle::operators::DropoutParam dropout_param(fix_seed,
                                                0,
                                                is_test,
                                                is_upscale_in_train,
                                                dropout_rate,
                                                nullptr,
                                                seed_val);

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x.dtype(), "LayernormResidualDropoutBias", ([&] {
        paddle::operators::FusedDropoutLayerNormHelper<data_t, uint8_t>
            fused_dropout_layernorm_helper(
                dev_ctx, x_m, x_n, dropout_param, ln_epsilon);
        fused_dropout_layernorm_helper
            .LayernormResidualDropoutBias<data_t, true>(
                dev_ctx,
                x.data<data_t>(),         // out_linear_out_data,
                residual.data<data_t>(),  // residual_data
                nullptr,                  // bias_data,
                ln_scale.data<data_t>(),
                ln_bias.data<data_t>(),
                dev_ctx.Alloc<data_t>(&dropout_residual_out),
                dev_ctx.Alloc<uint8_t>(&dropout_mask_out),
                dev_ctx.Alloc<data_t>(&final_out),
                dev_ctx.Alloc<paddle::operators::LayerNormParamType<data_t>>(
                    &ln_mean),
                dev_ctx.Alloc<paddle::operators::LayerNormParamType<data_t>>(
                    &ln_var));
      }));
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dropout_residual_ln_grad, ctx) {
  namespace f = paddle::framework;
  const auto &ln_scale = *ctx.Input<f::Tensor>("LnScale");
  const auto &dropout_mask_out = *ctx.Input<f::Tensor>("DropoutMask");
  const auto &ln_mean = *ctx.Input<f::Tensor>("LnMean");
  const auto &ln_var = *ctx.Input<f::Tensor>("LnVar");
  const auto &dropout_residual_out =
      *ctx.Input<f::Tensor>("DropoutResidualOut");
  const auto &grad_out = *ctx.Input<f::Tensor>(f::GradVarName("Out"));
  auto &grad_x = *ctx.Output<f::Tensor>(f::GradVarName("X"));
  auto &grad_residual = *ctx.Output<f::Tensor>(f::GradVarName("Residual"));
  auto &grad_ln_scale = *ctx.Output<f::Tensor>(f::GradVarName("LnScale"));
  auto &grad_ln_bias = *ctx.Output<f::Tensor>(f::GradVarName("LnBias"));
  f::Tensor grad_dropout_residual_out;

  auto ln_epsilon = ctx.Attr<float>("ln_epsilon");
  auto is_test = ctx.Attr<bool>("is_test");
  auto fix_seed = ctx.Attr<bool>("fix_seed");
  auto seed_val = ctx.Attr<int>("seed_val");
  auto is_upscale_in_train = ctx.Attr<bool>("is_upscale_in_train");
  auto dropout_rate = ctx.Attr<float>("dropout_rate");
  auto &dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  const auto &x_dims = grad_out.dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_n = x_dims[x_dims.size() - 1];

  // output
  grad_x.Resize(x_dims);
  grad_residual.Resize(x_dims);
  grad_dropout_residual_out.Resize(x_dims);

  grad_ln_scale.Resize(ln_scale.dims());
  grad_ln_bias.Resize(ln_scale.dims());

  paddle::operators::DropoutParam dropout_param(fix_seed,
                                                0,
                                                is_test,
                                                is_upscale_in_train,
                                                dropout_rate,
                                                nullptr,
                                                seed_val);

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      grad_out.dtype(), "LayernormResidualDropoutBiasGrad", ([&] {
        paddle::operators::FusedDropoutLayerNormHelper<data_t, uint8_t>
            fused_dropout_layernorm_helper(
                dev_ctx, x_m, x_n, dropout_param, ln_epsilon);
        fused_dropout_layernorm_helper
            .LayernormResidualDropoutBiasGrad<data_t, true>(
                dev_ctx,
                grad_out.data<data_t>(),
                dropout_residual_out.data<data_t>(),
                dropout_mask_out.data<uint8_t>(),
                ln_scale.data<data_t>(),
                ln_mean.data<paddle::operators::LayerNormParamType<data_t>>(),
                ln_var.data<paddle::operators::LayerNormParamType<data_t>>(),
                dev_ctx.Alloc<data_t>(&grad_dropout_residual_out),
                dev_ctx.Alloc<data_t>(&grad_ln_scale),
                dev_ctx.Alloc<data_t>(&grad_ln_bias),
                dev_ctx.Alloc<data_t>(&grad_x),  // d_out_linear_out_data,
                nullptr,                         // d_out_linear_bias_data,
                dev_ctx.Alloc<data_t>(&grad_residual));
      }));
}
