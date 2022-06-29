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

#include <vector>
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> FusedDropoutResidualLnInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& residual_shape,
    const std::vector<int64_t>& ln_scale_shape,
    const std::vector<int64_t>& ln_bias_shape) {
  int x_m = 1;
  for (int i = 0; i < x_shape.size() - 1; i++) {
    x_m *= x_shape[i];
  }
  const std::vector<int64_t> ln_out_shape = {x_m};
  return {x_shape, x_shape, ln_out_shape, ln_out_shape, x_shape};
}

// todo: now, ln_scale is fp16; how ot set ln_mean and ln_var is fp32?
std::vector<paddle::DataType> FusedDropoutResidualLnInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType residual_dtype,
    paddle::DataType ln_scale_dtype,
    paddle::DataType ln_bias_dtype) {
  // the type of ln_mean/ln_var is the same as ln_scale.
  return {x_dtype,
          paddle::DataType::UINT8,
          paddle::DataType::FLOAT32,
          paddle::DataType::FLOAT32,
          // ln_scale_dtype,
          // ln_scale_dtype,
          x_dtype};
}

PD_BUILD_OP(custom_fused_dropout_residual_ln)
    .Inputs({"X", "Residual", "LnScale", "LnBias"})
    .Outputs({"Out", "DropoutMask", "LnMean", "LnVar", "DropoutResidualOut"})
    .Attrs({"ln_epsilon: float",
            "is_test: bool",
            "fix_seed: bool",
            "seed_val: int",
            "is_upscale_in_train: bool",
            "dropout_rate: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(FusedDropoutResidualLnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedDropoutResidualLnInferDtype));

PD_BUILD_GRAD_OP(custom_fused_dropout_residual_ln)
    .Inputs({"X",
             "Residual",
             "LnScale",
             "LnBias",
             "DropoutMask",
             "LnMean",
             "LnVar",
             "DropoutResidualOut",
             paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"),
              paddle::Grad("Residual"),
              paddle::Grad("LnScale"),
              paddle::Grad("LnBias")})
    .Attrs({"ln_epsilon: float",
            "is_test: bool",
            "fix_seed: bool",
            "seed_val: int",
            "is_upscale_in_train: bool",
            "dropout_rate: float"});
