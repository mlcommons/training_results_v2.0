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

#define CHECK_INPUT(x)                                      \
  PD_CHECK(x.place().GetType() == phi::AllocationType::GPU, \
           #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> fmha_cuda_forward(const paddle::Tensor& qkv,
                                              const paddle::Tensor& cu_seqlen,
                                              const paddle::Tensor& host_seqlen,
                                              bool is_test,
                                              float dropout_rate,
                                              bool zero_tensors,
                                              bool use_fmha_mke_opt);

std::vector<paddle::Tensor> fmha_cuda_backward(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cu_seqlen,
    const paddle::Tensor& host_seqlen,
    const paddle::Tensor& softmax_input,
    const paddle::Tensor& d_ctx_out,
    bool is_test,
    float dropout_rate,
    bool zero_tensors,
    bool use_fmha_mke_opt);

/*
 *x_shape(fp16) = [total_tokens, 3, num_heads, head_size]
 *y_shape(int32) = [batch_size + 1]
*/
std::vector<std::vector<int64_t>> FmhaInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& host_y_shape,
    const bool& is_test,
    const float& dropout_rate,
    const bool& zero_tensors,
    const bool& use_fmha_mke_opt) {
  int total = x_shape[0];
  int num_heads = x_shape[2];
  int head_size = x_shape[3];
  int batch_size = y_shape[0] - 1;

  if (x_shape[1] != 3) {
    PD_THROW(
        "The shape for input QKV should be [total_tokens, 3, num_heas, "
        "head_size].");
  }
  int max_seq_len = 512;
  std::vector<int64_t> ctx_out_shape = {total, num_heads, head_size};
  std::vector<int64_t> s_out_shape = {
      batch_size, num_heads, max_seq_len, max_seq_len};

  return {ctx_out_shape, s_out_shape};
}

std::vector<paddle::DataType> FmhaInferDtype(paddle::DataType x_dtype,
                                             paddle::DataType y_dtype,
                                             paddle::DataType host_y_dtype) {
  return {x_dtype, x_dtype};
}

std::vector<paddle::Tensor> FmhaCUDAForward(const paddle::Tensor& qkv,
                                            const paddle::Tensor& cu_seqlen,
                                            const paddle::Tensor& host_seqlen,
                                            bool is_test,
                                            float dropout_rate,
                                            bool zero_tensors,
                                            bool use_fmha_mke_opt) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(cu_seqlen);
  // Note: should not use CHECK_INPUT(max_seq_len_host),
  // because it will enforce this input to be GPU tensor

  return fmha_cuda_forward(qkv,
                           cu_seqlen,
                           host_seqlen,
                           is_test,
                           dropout_rate,
                           zero_tensors,
                           use_fmha_mke_opt);
}

std::vector<paddle::Tensor> FmhaCUDABackward(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cu_seqlen,
    const paddle::Tensor& host_seqlen,
    const paddle::Tensor& softmax_input,
    const paddle::Tensor& d_ctx_out,
    bool is_test,
    float dropout_rate,
    bool zero_tensors,
    bool use_fmha_mke_opt) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(cu_seqlen);
  CHECK_INPUT(softmax_input);
  CHECK_INPUT(d_ctx_out);

  return fmha_cuda_backward(qkv,
                            cu_seqlen,
                            host_seqlen,
                            softmax_input,
                            d_ctx_out,
                            is_test,
                            dropout_rate,
                            zero_tensors,
                            use_fmha_mke_opt);
}

PD_BUILD_OP(custom_fmha)
    .Inputs({"QKV", "CuSeqLen", "HostSeqLen"})
    .Outputs({"CtxOut", "SOut"})
    .Attrs({"is_test: bool",
            "dropout_rate: float",
            "zero_tensors: bool",
            "use_fmha_mke_opt: bool"})
    .SetKernelFn(PD_KERNEL(FmhaCUDAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FmhaInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FmhaInferDtype));

PD_BUILD_GRAD_OP(custom_fmha)
    .Inputs({"QKV", "CuSeqLen", "HostSeqLen", "SOut", paddle::Grad("CtxOut")})
    .Outputs({paddle::Grad("QKV")})
    .Attrs({"is_test: bool",
            "dropout_rate: float",
            "zero_tensors: bool",
            "use_fmha_mke_opt: bool"})
    .SetKernelFn(PD_KERNEL(FmhaCUDABackward));
