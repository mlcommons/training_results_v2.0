/**
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cudnn/Types.h>
#include <ATen/ATen.h>

namespace at { namespace native {

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  if (tensor.scalar_type() == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (tensor.scalar_type() == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (tensor.scalar_type() == at::kHalf) {
    return CUDNN_DATA_HALF;
  }
  std::string msg("getCudnnDataType() not supported for ");
  msg += toString(tensor.scalar_type());
  throw std::runtime_error(msg);
}

int64_t cudnn_version() {
  return CUDNN_VERSION;
}

}} 
