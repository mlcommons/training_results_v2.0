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

#include "fmhalib.h"  // NOLINT
#include "paddle/extension.h"
#include "paddle/fluid/operators/dropout_impl_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

#define CHECK_FMHALIB_ERROR()                          \
  do {                                                 \
    const auto* __err_msg = fmhalib_error();           \
    if (__err_msg != nullptr) {                        \
      PD_THROW("fmhalib error code is %s", __err_msg); \
    }                                                  \
  } while (0)

static const paddle::platform::CUDADeviceContext&
GetCurrentCUDADeviceContext() {
  auto dev_id = paddle::platform::GetCurrentDeviceId();
  paddle::platform::CUDAPlace place(dev_id);
  return *paddle::platform::DeviceContextPool::Instance().GetByPlace(place);
}

static std::string CuSeqLenToStr(const int* cu_seqlen,
                                 int bs,
                                 cudaStream_t stream) {
  std::vector<int> cpu_len(bs + 1);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(cpu_len.data(),
                      cu_seqlen,
                      cpu_len.size() * sizeof(cpu_len[0]),
                      cudaMemcpyDeviceToHost,
                      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  std::stringstream ss;
  ss << cpu_len[0];
  for (int i = 1; i <= bs; ++i) {
    ss << ", " << cpu_len[i];
  }
  return std::string("[") + ss.str() + "]";
}

static void SeedIncFunc(uint64_t inc,
                        uint64_t* seed,
                        const int64_t** offset_ptr,
                        uint64_t* offset,
                        bool* is_device_rnd) {
  auto& dev_ctx = GetCurrentCUDADeviceContext();
  paddle::operators::GetSeedDataAndIncrement(
      dev_ctx, nullptr, false, 0, inc, seed, offset);
  *offset_ptr = nullptr;
  *is_device_rnd = false;
}

struct FMHALibVersionPrintClass {
  FMHALibVersionPrintClass() {
    LOG(INFO) << "FMHALib version: " << fmhalib_version();
  }
} _g_fmhalib_version_print_obj;

static const int kMaxSupportedSeqLength = 512;

static const int MAX_GROUP_SIZE = 4;
static cudaStream_t all_streams[MAX_GROUP_SIZE];
static cudaEvent_t all_events[MAX_GROUP_SIZE + 1];

static void InitStreamEventsOnce(cudaStream_t stream) {
  static std::once_flag flag;
  std::call_once(flag, [stream] {
    all_streams[0] = stream;
    for (int i = 1; i < MAX_GROUP_SIZE; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamCreateWithFlags(&all_streams[i], cudaStreamNonBlocking));
    }

    for (int i = 0; i < MAX_GROUP_SIZE + 1; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&all_events[i], cudaEventDisableTiming));
    }
  });
}

static cudaStream_t GetStream(cudaStream_t stream, size_t i) {
  i %= MAX_GROUP_SIZE;
  return i == 0 ? stream : all_streams[i];
}

static cudaEvent_t GetStartEvent() { return all_events[MAX_GROUP_SIZE]; }

static cudaEvent_t GetEndEvent(size_t i) {
  return all_events[i % MAX_GROUP_SIZE];
}

static void PreRecordEvent(cudaStream_t stream, size_t n) {
  auto event = GetStartEvent();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, stream));
  for (size_t i = 1; i < n; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(GetStream(stream, i), event));
  }
}

static void PostRecordEvent(cudaStream_t stream, size_t n) {
  for (size_t i = 1; i < n; ++i) {
    auto event = GetEndEvent(i);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, GetStream(stream, i)));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(stream, event));
  }
}

static int seq_len_round(int real_seq_len) {
  int ret = fmhalib_seq_len(real_seq_len);
  if (ret < 0 || ret > kMaxSupportedSeqLength) {
    PD_THROW("Error of seq_len_round when use_fmha_mke_opt=1.");
  }
  return ret;
}

struct FMHASeqGroup {
  FMHASeqGroup() {}
  FMHASeqGroup(
      int seq_offset, int token_offset, int batch_size, int total, int seq_len)
      : seq_offset(seq_offset),
        token_offset(token_offset),
        batch_size(batch_size),
        total(total),
        seq_len(seq_len) {}

  int seq_offset;
  int token_offset;
  int batch_size;
  int total;
  int seq_len;
};

static std::vector<FMHASeqGroup> GetFMHASeqGroup(const int* prefix_sum_seq_len,
                                                 const int batch_size,
                                                 const bool use_fmha_mke_opt) {
  if (!use_fmha_mke_opt) {
    int max_seq_len = 0;
    for (int i = 0; i < batch_size; ++i) {
      max_seq_len = std::max(max_seq_len,
                             prefix_sum_seq_len[i + 1] - prefix_sum_seq_len[i]);
    }
    return {FMHASeqGroup(0,
                         0,
                         batch_size,
                         prefix_sum_seq_len[batch_size],
                         seq_len_round(max_seq_len))};
  }

  std::vector<FMHASeqGroup> infos;
  infos.reserve(batch_size);
  int prev_max_seq_len =
      seq_len_round(prefix_sum_seq_len[1] - prefix_sum_seq_len[0]);
  int prev_idx = 0;
  for (int i = 1; i < batch_size; ++i) {
    int cur_seq_len =
        seq_len_round(prefix_sum_seq_len[i + 1] - prefix_sum_seq_len[i]);
    if (cur_seq_len != prev_max_seq_len) {
      infos.emplace_back(prev_idx,
                         prefix_sum_seq_len[prev_idx],
                         i - prev_idx,
                         prefix_sum_seq_len[i] - prefix_sum_seq_len[prev_idx],
                         prev_max_seq_len);
      prev_idx = i;
      prev_max_seq_len = cur_seq_len;
    }
  }

  infos.emplace_back(
      prev_idx,
      prefix_sum_seq_len[prev_idx],
      batch_size - prev_idx,
      prefix_sum_seq_len[batch_size] - prefix_sum_seq_len[prev_idx],
      prev_max_seq_len);
  return infos;
}

/*
qkv: fp16,  [total, 3, num_heads, head_size]
cu_seqlen: int32,  [batch_size + 1]

ctx_output: fp16, [total, num_heads, head_size]
s_output: fp16, [batch_size, num_heads, max_seq_len, max_seq_len]
*/
std::vector<paddle::Tensor> fmha_cuda_forward(const paddle::Tensor& qkv,
                                              const paddle::Tensor& cu_seqlen,
                                              const paddle::Tensor& host_seqlen,
                                              bool is_test,
                                              float dropout_rate,
                                              bool zero_tensors,
                                              bool use_fmha_mke_opt) {
  // To prevent linker to prune _g_fmhalib_version_print_obj
  VLOG(10) << (&_g_fmhalib_version_print_obj);
  auto qkv_dims = qkv.shape();
  int total = qkv_dims[0];
  int num_heads = qkv_dims[2];
  int head_size = qkv_dims[3];

  auto cu_seqlen_dims = cu_seqlen.shape();
  int batch_size = cu_seqlen_dims[0] - 1;

  auto groups = GetFMHASeqGroup(
      host_seqlen.data<int>(), batch_size, use_fmha_mke_opt && (!is_test));

  cudaStream_t stream = qkv.stream();
  InitStreamEventsOnce(stream);

  const std::vector<int64_t> ctx_out_shape = {total, num_heads, head_size};
  auto place = qkv.place();
  auto dtype = qkv.dtype();
  auto ctx_out = paddle::experimental::empty(ctx_out_shape, dtype, place);
  const std::vector<int64_t> s_out_shape = {
      batch_size, num_heads, kMaxSupportedSeqLength, kMaxSupportedSeqLength};
  auto s_out = paddle::experimental::empty(s_out_shape, dtype, place);

  if (qkv.type() != paddle::DataType::FLOAT16) {
    PD_THROW("FMHALib only supports float16 inputs.");
  }

  auto* ctx_out_data = ctx_out.data<paddle::float16>();
  auto* s_out_data = s_out.data<paddle::float16>();
  const auto* qkv_data = qkv.data<paddle::float16>();
  const auto* cu_seqlen_data = cu_seqlen.data<int>();

  bool is_training = (!is_test);

  auto ctx_stride = static_cast<int64_t>(num_heads) * head_size;
  auto qkv_stride = ctx_stride * 3;
  auto s_stride = static_cast<int64_t>(num_heads) * kMaxSupportedSeqLength *
                  kMaxSupportedSeqLength;

  PreRecordEvent(stream, groups.size());
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const bool is_nl = false;  // (info.batch_size < 4);

    const auto& group = groups[group_idx];
    fmhalib_fwd(qkv_data + group.token_offset * qkv_stride,
                cu_seqlen_data + group.seq_offset,
                group.total,
                num_heads,
                head_size,
                group.batch_size,
                dropout_rate,
                is_nl ? kMaxSupportedSeqLength : group.seq_len,
                is_training,
                /*is_nl=*/is_nl,
                zero_tensors,
                /*seed_inc_func=*/SeedIncFunc,
                GetStream(stream, group_idx),
                ctx_out_data + group.token_offset * ctx_stride,
                s_out_data + group.seq_offset * s_stride);
  }

  PostRecordEvent(stream, groups.size());
  CHECK_FMHALIB_ERROR();
  return {ctx_out, s_out};
}

std::vector<paddle::Tensor> fmha_cuda_backward(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cu_seqlen,
    const paddle::Tensor& host_seqlen,
    const paddle::Tensor& softmax_input,
    const paddle::Tensor& d_ctx_out,
    bool is_test,
    float dropout_rate,
    bool zero_tensors,
    bool use_fmha_mke_opt) {
  auto qkv_dims = qkv.shape();
  int total = qkv_dims[0];
  int num_heads = qkv_dims[2];
  int head_size = qkv_dims[3];

  auto cu_seqlen_dims = cu_seqlen.shape();
  int batch_size = cu_seqlen_dims[0] - 1;

  auto groups = GetFMHASeqGroup(
      host_seqlen.data<int>(), batch_size, use_fmha_mke_opt && (!is_test));

  cudaStream_t stream = qkv.stream();
  const auto& place = qkv.place();

  // output
  auto grad_qkv_out =
      paddle::experimental::empty(qkv.shape(), qkv.dtype(), place);
  // tmp tensor:
  // auto grad_softmax_out = paddle::Tensor(paddle::PlaceType::kGPU,
  // softmax_input.shape());

  auto* softmax_input_data = softmax_input.data<paddle::float16>();
  paddle::float16* softmax_output_data = (paddle::float16*)softmax_input_data;
  // todo: memcpy softmax_input -> grad_softmax_out

  paddle::float16* grad_qkv_out_data = grad_qkv_out.data<paddle::float16>();
  // paddle::float16 *grad_softmax_out_data =
  // grad_softmax_out.mutable_data<paddle::float16>(qkv.place());

  const auto* d_ctx_out_data = d_ctx_out.data<paddle::float16>();
  const auto* qkv_data = qkv.data<paddle::float16>();
  const auto* cu_seqlen_data = cu_seqlen.data<int>();

  auto ctx_stride = static_cast<int64_t>(num_heads) * head_size;
  auto qkv_stride = ctx_stride * 3;
  auto s_stride = static_cast<int64_t>(num_heads) * kMaxSupportedSeqLength *
                  kMaxSupportedSeqLength;

  uint64_t workspace_size = 0;
  paddle::Tensor workspace_tensor;
  void* workspace_ptr = nullptr;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (workspace_size > 0) {
      workspace_tensor =
          paddle::experimental::empty({static_cast<int64_t>(workspace_size)},
                                      paddle::DataType::INT8,
                                      place);
      workspace_ptr = workspace_tensor.data<int8_t>();
    }

    if (i > 0) {
      PreRecordEvent(stream, groups.size());
    }

    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      const bool is_nl = false;  // (info.batch_size < 4);

      const auto& group = groups[group_idx];
      uint64_t cur_workspace_size = workspace_size;
      fmhalib_bwd(d_ctx_out_data + group.token_offset * ctx_stride,
                  qkv_data + group.token_offset * qkv_stride,
                  cu_seqlen_data + group.seq_offset,
                  group.total,
                  num_heads,
                  head_size,
                  group.batch_size,
                  dropout_rate,
                  is_nl ? kMaxSupportedSeqLength : group.seq_len,
                  /*is_nl=*/is_nl,
                  zero_tensors,
                  GetStream(stream, group_idx),
                  softmax_output_data + group.seq_offset * s_stride,
                  i == 0 ? nullptr
                         : grad_qkv_out_data + group.token_offset * qkv_stride,
                  workspace_ptr,
                  &cur_workspace_size);
      if (i == 0) {
        workspace_size = std::max(workspace_size, cur_workspace_size);
      }
    }

    if (i > 0) {
      PostRecordEvent(stream, groups.size());
    }
  }

  CHECK_FMHALIB_ERROR();
  return {grad_qkv_out};
}
