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

// @x: [x, in_feature] or [xx, xx, in_feature]
// @y: [out_feature, in_feature]
// @out: [x, out_feature] or [xx, xx, out_feature]
// support transx=false, transy=true/false.
std::vector<std::vector<int64_t>> FusedDenseInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& bias_shape,
    const bool& transx,
    const bool& transy,
    const bool& use_addto) {
  int x_size = x_shape.size();
  int x_m = 1;
  for (int i = 0; i < (x_size - 1); i++) {
    x_m *= x_shape[i];
  }
  int x_k = x_shape[x_size - 1];

  int y_k = y_shape[0];
  int y_n = y_shape[1];
  if (transy) {
    y_k = y_shape[1];
    y_n = y_shape[0];
  }

  if (x_k != y_k) {
    PD_THROW("The reudce dim of A and B in matmul is not equal.");
  }

  if (transx) {
    PD_THROW("Only support cases: transx is False, transy are True/False.");
  }

  std::vector<int64_t> out_shape(x_shape);
  out_shape[x_size - 1] = y_n;
  return {out_shape};
}

std::vector<paddle::DataType> FusedDenseInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType y_dtype,
    paddle::DataType bias_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_fused_dense)
    .Inputs({"X", "Y", "Bias"})
    .Outputs({"Out"})
    .Attrs({"transx: bool", "transy: bool", "use_addto: bool"})
    .SetInferShapeFn(PD_INFER_SHAPE(FusedDenseInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedDenseInferDtype));

PD_BUILD_GRAD_OP(custom_fused_dense)
    .Inputs({"X", "Y", "Bias", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y"), paddle::Grad("Bias")})
    .Attrs({"transx: bool", "transy: bool"});
