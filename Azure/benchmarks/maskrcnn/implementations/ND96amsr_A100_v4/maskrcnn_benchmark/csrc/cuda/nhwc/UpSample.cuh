/******************************************************************************
*
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
*

 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <math.h>

namespace at {
namespace native {

/* TODO: move this to a common place */
template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}


static inline void upsample_2d_shape_check_nhwc(
    const Tensor& input,
    const Tensor& grad_output,
    int nbatch,
    int nchannels,
    int input_height,
    int input_width,
    int output_height,
    int output_width) {
  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  if (input.defined()) {
    TORCH_CHECK(
        input.numel() != 0 && input.dim() == 4,
        "non-empty 4D input tensor expected but got a tensor with sizes ",
        input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 4, 0, nbatch);
    check_dim_size(grad_output, 4, 1, output_height);
    check_dim_size(grad_output, 4, 2, output_width);
    check_dim_size(grad_output, 4, 3, nchannels);
  }
}

__device__ __forceinline__ static int nearest_neighbor_compute_source_index(
    const float scale,
    int dst_index,
    int input_size) {
  const int src_index =
      min(static_cast<int>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}


} // namespace native
} // namespace at
