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

#include <torch/extension.h>
#include "nccl.h"  // NOLINT

#define ASSERT_CHECK(__cond)                          \
  do {                                                \
    if (!(__cond)) throw std::runtime_error(#__cond); \
  } while (0)

extern "C" {

extern void InitNCCLPreMulSum(const void *scalar,
                              ncclDataType_t dtype,
                              ncclScalarResidence_t residence);
}

void InitNCCLPreMulSumByTensor(const at::Tensor &t) {
  const void *scalar = t.data_ptr();
  auto dtype = t.options().dtype();
  ncclDataType_t nccl_dtype;
  if (dtype == at::ScalarType::Half) {
    nccl_dtype = ncclFloat16;
  } else if (dtype == at::ScalarType::Float) {
    nccl_dtype = ncclFloat32;
  } else if (dtype == at::ScalarType::Double) {
    nccl_dtype = ncclFloat64;
  } else {
    ASSERT_CHECK(false);
  }
  auto residence =
      (t.device().is_cuda() ? ncclScalarDevice : ncclScalarHostImmediate);
  InitNCCLPreMulSum(scalar, nccl_dtype, residence);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("make_nccl_premul_sum",
        &InitNCCLPreMulSumByTensor,
        py::call_guard<py::gil_scoped_release>());
}
