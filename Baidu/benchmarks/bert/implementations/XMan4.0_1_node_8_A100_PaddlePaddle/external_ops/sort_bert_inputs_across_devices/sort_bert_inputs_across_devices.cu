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

#include <cstdint>
#include <iostream>
#include <sstream>
#include "glog/logging.h"
#include "paddle/extension.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/for_range.h"

namespace framework = paddle::framework;
namespace operators = paddle::operators;
namespace platform = paddle::platform;
namespace memory = paddle::memory;
namespace kps = paddle::operators::kernel_primitives;

template <typename T>
struct NCCLDataTypeTrait;

#define DEFINE_NCCL_DTYPE_TRAIT(__cpp_type, __nccl_dtype)    \
  template <>                                                \
  struct NCCLDataTypeTrait<__cpp_type> {                     \
    static constexpr ncclDataType_t DataType = __nccl_dtype; \
  }

DEFINE_NCCL_DTYPE_TRAIT(int16_t, ncclFloat16);
DEFINE_NCCL_DTYPE_TRAIT(int32_t, ncclInt32);
DEFINE_NCCL_DTYPE_TRAIT(int64_t, ncclInt64);

template <typename T>
static std::string GPUTensorToString(const T *ptr, size_t numel) {
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  auto &dev_ctx = *platform::DeviceContextPool::Instance().GetByPlace(place);
  auto stream = dev_ctx.stream();

  std::vector<T> cpu_data(numel);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      cpu_data.data(), ptr, sizeof(T) * numel, cudaMemcpyDeviceToHost, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  std::stringstream ss;
  ss << "[";
  for (decltype(numel) i = 0; i < numel; ++i) {
    if (i > 0) ss << ",";
    ss << cpu_data[i];
  }
  ss << "]";
  return ss.str();
}

template <typename T>
static std::string GPUTensorToString(const paddle::Tensor &t) {
  return GPUTensorToString<T>(t.data<T>(), t.numel());
}

template <typename T>
static paddle::Tensor PadTensor(const paddle::Tensor &x,
                                int max_batch_size,
                                cudaStream_t stream) {
#define CALL_PAD_ZERO_DATA                                                   \
  do {                                                                       \
    T *y_data = y.mutable_data<T>(x.place());                                \
    const T *x_data = x.data<T>();                                           \
    int n = batch_size * seq_len;                                            \
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(                              \
        y_data, x_data, n * sizeof(T), cudaMemcpyDeviceToDevice, stream));   \
    PADDLE_ENFORCE_GPU_SUCCESS(                                              \
        cudaMemsetAsync(y_data + n,                                          \
                        0,                                                   \
                        (max_batch_size - batch_size) * seq_len * sizeof(T), \
                        stream));                                            \
    return y;                                                                \
  } while (0)

  const auto &x_dim = x.dims();
  int batch_size = x_dim[0];
  if (x_dim.size() == 2) {
    int seq_len = x_dim[1];
    paddle::Tensor y(x.place(), {max_batch_size, seq_len});
    CALL_PAD_ZERO_DATA;
  } else {
    PADDLE_ENFORCE_EQ(x_dim.size(), 1);
    int seq_len = 1;
    paddle::Tensor y(x.place(), {max_batch_size});
    CALL_PAD_ZERO_DATA;
  }
}

template <typename T>
struct RetrieveAllGatheredInputMaskFunctor {
  RetrieveAllGatheredInputMaskFunctor(
      const T *x, T *y, int fused_numel, int each_mask_numel, int mask_offset)
      : x_(x),
        y_(y),
        fused_numel_(fused_numel),
        each_mask_numel_(each_mask_numel),
        mask_offset_(mask_offset) {}

  HOSTDEVICE void operator()(int idx) const {
    int rank = idx / each_mask_numel_;
    int offset = mask_offset_ + idx % each_mask_numel_;
    int x_idx = rank * fused_numel_ + offset;
    y_[idx] = x_[x_idx];
  }

 private:
  const T *x_;
  T *y_;
  int fused_numel_;
  int each_mask_numel_;
  int mask_offset_;
};

template <typename T>
struct IotaFunctor {
  explicit IotaFunctor(T *x) : x_(x) {}

  HOSTDEVICE void operator()(int idx) const { x_[idx] = static_cast<T>(idx); }

 private:
  T *x_;
};

template <typename T, typename IndexT>
struct IsNonZeroFunctor {
  HOSTDEVICE IndexT operator()(T x) const {
    return static_cast<IndexT>(x != 0);
  }
};

template <typename T, typename IndexT>
struct ReorderBERTInputTensorsFunctor {
 public:
  ReorderBERTInputTensorsFunctor(const T *fused_inputs,
                                 const IndexT *indices,
                                 int device_id,
                                 int num_devices,
                                 int max_batch_size,
                                 int seq_len,
                                 T *input_ids_out,
                                 T *segment_ids_out,
                                 T *input_mask_out,
                                 T *masked_lm_labels_out,
                                 T *next_sentence_labels_out)
      : fused_inputs_(fused_inputs),
        indices_(indices),
        device_id_(device_id),
        num_devices_(num_devices),
        max_batch_size_(max_batch_size),
        seq_len_(seq_len),
        input_ids_out_(input_ids_out),
        segment_ids_out_(segment_ids_out),
        input_mask_out_(input_mask_out),
        masked_lm_labels_out_(masked_lm_labels_out),
        next_sentence_labels_out_(next_sentence_labels_out) {}

  // idx is in range [0, new_bs * seq_len)
  HOSTDEVICE void operator()(int idx) const {
    int out_idx_i = idx / seq_len_;
    int seq_len_idx = idx % seq_len_;

    auto index_per_device = indices_[device_id_ + out_idx_i * num_devices_];
    auto gpu_idx = index_per_device / max_batch_size_;
    auto bs_idx = index_per_device % max_batch_size_;

    // [gpu_idx, bs_idx, seq_len_idx]
    int device_stride = 4 * max_batch_size_ * seq_len_ + max_batch_size_;
    int tensor_offset = max_batch_size_ * seq_len_;
    int device_offset = gpu_idx * device_stride;
    int offset = device_offset + bs_idx * seq_len_ + seq_len_idx;

    input_ids_out_[idx] = fused_inputs_[offset];
    segment_ids_out_[idx] = fused_inputs_[offset + tensor_offset];
    input_mask_out_[idx] = fused_inputs_[offset + 2 * tensor_offset];
    masked_lm_labels_out_[idx] = fused_inputs_[offset + 3 * tensor_offset];
    if (seq_len_idx == 0) {
      next_sentence_labels_out_[out_idx_i] =
          fused_inputs_[device_offset + 4 * max_batch_size_ * seq_len_ +
                        bs_idx];
    }
  }

 private:
  const T *fused_inputs_;
  const IndexT *indices_;
  int device_id_;
  int num_devices_;
  int max_batch_size_;
  int seq_len_;
  T *input_ids_out_;
  T *segment_ids_out_;
  T *input_mask_out_;
  T *masked_lm_labels_out_;
  T *next_sentence_labels_out_;
};

/**
 * input_ids: shape: [bs, seq_len], range: [0, vocab_size)
 * segment_ids: shape: [bs, seq_len], range: {0, 1}
 * input_mask: shape: [bs, seq_len], range: {0, 1}
 * masked_lm_labels: shape: [bs, seq_len], range: [0, vocab_size)
 * next_sentence_labels：shape: [bs] or [bs, 1], range: {0, 1}
 */
template <typename T>
std::vector<paddle::Tensor> GPUSortBERTInputsAcrossDevicesWithDType(
    paddle::Tensor input_ids,
    paddle::Tensor segment_ids,
    paddle::Tensor input_mask,
    paddle::Tensor masked_lm_labels,
    paddle::Tensor next_sentence_labels,
    int max_batch_size,
    int ring_id,
    int device_id,
    int num_devices) {
  const auto &dim = input_ids.dims();
  int batch_size = dim[0];
  int seq_len = dim[1];

  PADDLE_ENFORCE_EQ(dim.size(), 2);
  PADDLE_ENFORCE_LE(batch_size, max_batch_size);
  PADDLE_ENFORCE_EQ(dim, segment_ids.dims());
  PADDLE_ENFORCE_EQ(dim, input_mask.dims());
  PADDLE_ENFORCE_EQ(dim, masked_lm_labels.dims());

  const auto &nsl_dim = next_sentence_labels.dims();
  if (nsl_dim.size() == 2) {
    PADDLE_ENFORCE_EQ(nsl_dim[0], batch_size);
    PADDLE_ENFORCE_EQ(nsl_dim[1], 1);
  } else if (nsl_dim.size() == 1) {
    PADDLE_ENFORCE_EQ(nsl_dim[0], batch_size);
  } else {
    PADDLE_THROW("invalid next_sentence_labels rank, should be 1 or 2.");
  }

  bool need_pad = (batch_size < max_batch_size);
  // NOTE: device_id may be different from platform::GetCurrentDeviceId()!
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  auto &dev_ctx = *platform::DeviceContextPool::Instance().GetByPlace(place);
  auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place)->comm();
  auto stream = dev_ctx.stream();

  // Step 1: pad to max_batch_size
  if (need_pad) {
    input_ids = PadTensor<T>(input_ids, max_batch_size, stream);
    segment_ids = PadTensor<T>(segment_ids, max_batch_size, stream);
    input_mask = PadTensor<T>(input_mask, max_batch_size, stream);
    masked_lm_labels = PadTensor<T>(masked_lm_labels, max_batch_size, stream);
    next_sentence_labels =
        PadTensor<T>(next_sentence_labels, max_batch_size, stream);
  }

  VLOG(10) << "input_ids = " << GPUTensorToString<T>(input_ids);
  VLOG(10) << "segment_ids = " << GPUTensorToString<T>(segment_ids);
  VLOG(10) << "input_mask = " << GPUTensorToString<T>(input_mask);
  VLOG(10) << "masked_lm_labels = " << GPUTensorToString<T>(masked_lm_labels);
  VLOG(10) << "next_sentence_labels = "
           << GPUTensorToString<T>(next_sentence_labels);

  // Step 2: fuse to continous space
  int n = max_batch_size * seq_len;
  int numel = 4 * n + max_batch_size;
  auto buffer = memory::Alloc(place, numel * sizeof(T));
  T *buf_ptr = reinterpret_cast<T *>(buffer->ptr());
  const T *input_ids_ptr =
      static_cast<const paddle::Tensor &>(input_ids).data<T>();
  const T *segment_ids_ptr =
      static_cast<const paddle::Tensor &>(segment_ids).data<T>();
  const T *input_mask_ptr =
      static_cast<const paddle::Tensor &>(input_mask).data<T>();
  const T *masked_lm_labels_ptr =
      static_cast<const paddle::Tensor &>(masked_lm_labels).data<T>();
  const T *next_sentence_labels_ptr =
      static_cast<const paddle::Tensor &>(next_sentence_labels).data<T>();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      buf_ptr, input_ids_ptr, n * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(buf_ptr + n,
                                             segment_ids_ptr,
                                             n * sizeof(T),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(buf_ptr + 2 * n,
                                             input_mask_ptr,
                                             n * sizeof(T),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(buf_ptr + 3 * n,
                                             masked_lm_labels_ptr,
                                             n * sizeof(T),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(buf_ptr + 4 * n,
                                             next_sentence_labels_ptr,
                                             max_batch_size * sizeof(T),
                                             cudaMemcpyDeviceToDevice,
                                             stream));

  VLOG(10) << "fused input = " << GPUTensorToString<T>(buf_ptr, numel);

  // Step 3: allgather
  auto allgather_buffer = memory::Alloc(place, numel * num_devices * sizeof(T));
  T *allgather_buf_ptr = reinterpret_cast<T *>(allgather_buffer->ptr());
  auto nccl_dtype = NCCLDataTypeTrait<T>::DataType;
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      buf_ptr, allgather_buf_ptr, numel, nccl_dtype, comm, stream));
  buffer = nullptr;

  VLOG(10) << "allgather fused input = "
           << GPUTensorToString<T>(allgather_buf_ptr, numel * num_devices);

  // Step 4: sort by seq_len
  framework::Tensor allgather_input_mask;
  int gbs = num_devices * max_batch_size;
  allgather_input_mask.Resize({gbs, seq_len});
  T *allgather_input_mask_ptr = allgather_input_mask.mutable_data<T>(place);
  platform::ForRange<platform::CUDADeviceContext> retrieve_mask_for_range(
      dev_ctx, num_devices * n);
  int input_mask_offset = 2 * n;
  retrieve_mask_for_range(
      RetrieveAllGatheredInputMaskFunctor<T>(allgather_buf_ptr,
                                             allgather_input_mask_ptr,
                                             numel,
                                             n,
                                             input_mask_offset));
  VLOG(10) << "allgather mask = "
           << GPUTensorToString<T>(allgather_input_mask_ptr, gbs * seq_len);

  framework::Tensor allgather_seq_len;
  allgather_seq_len.Resize({gbs});
  allgather_seq_len.mutable_data<T>(place);
  operators::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      dev_ctx,
      allgather_input_mask,
      &allgather_seq_len,
      kps::IdentityFunctor<T>(),
      {1},
      stream);

  VLOG(10) << "allgather seq_len = "
           << GPUTensorToString<T>(allgather_seq_len.data<T>(), gbs);

  using IndexT = int;
  auto indices = memory::Alloc(place, gbs * sizeof(IndexT));
  auto *indices_ptr = reinterpret_cast<IndexT *>(indices->ptr());
  platform::ForRange<platform::CUDADeviceContext> itoa_for_range(dev_ctx, gbs);
  itoa_for_range(IotaFunctor<IndexT>(indices_ptr));
  auto sorted_indices = memory::Alloc(place, gbs * sizeof(IndexT));
  auto *sorted_indices_ptr = reinterpret_cast<IndexT *>(sorted_indices->ptr());

  auto *allgather_seq_len_ptr = allgather_seq_len.data<T>();
  auto sorted_allgather_seq_len = memory::Alloc(place, gbs * sizeof(T));
  auto *sorted_allgather_seq_len_ptr =
      reinterpret_cast<T *>(sorted_allgather_seq_len->ptr());
  memory::AllocationPtr tmp_storage;
  void *tmp_storage_ptr;
  size_t tmp_storage_size = 0;

  for (int i = 0; i < 2; ++i) {
    if (tmp_storage_size > 0 && tmp_storage == nullptr) {
      tmp_storage = memory::Alloc(place, tmp_storage_size);
    }
    tmp_storage_ptr = tmp_storage ? tmp_storage->ptr() : nullptr;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cub::DeviceRadixSort::SortPairsDescending(tmp_storage_ptr,
                                                  tmp_storage_size,
                                                  allgather_seq_len_ptr,
                                                  sorted_allgather_seq_len_ptr,
                                                  indices_ptr,
                                                  sorted_indices_ptr,
                                                  gbs,
                                                  0,
                                                  sizeof(T) * 8,
                                                  stream));
  }
  VLOG(10) << "allgather sorted seq_len = "
           << GPUTensorToString<T>(sorted_allgather_seq_len_ptr, gbs);
  VLOG(10) << "allgather sorted indices = "
           << GPUTensorToString<IndexT>(sorted_indices_ptr, gbs);

  // Step 5: reorder inputs
  memory::AllocationPtr ntokens_alloc;
  IndexT ntokens;
  if (need_pad) {
    // find the max valid length here
    ntokens_alloc = memory::Alloc(place, sizeof(IndexT));
    auto *ntokens_ptr = reinterpret_cast<IndexT *>(ntokens_alloc->ptr());
    using CubIterator =
        cub::TransformInputIterator<IndexT, IsNonZeroFunctor<T, IndexT>, T *>;
    CubIterator input_iter(sorted_allgather_seq_len_ptr,
                           IsNonZeroFunctor<T, IndexT>());
    tmp_storage_size = 0;
    for (int i = 0; i < 2; ++i) {
      if (i > 0 && tmp_storage_size > 0 &&
          (tmp_storage == nullptr || tmp_storage->size() < tmp_storage_size)) {
        tmp_storage = memory::Alloc(place, tmp_storage_size);
      }
      tmp_storage_ptr = tmp_storage ? tmp_storage->ptr() : nullptr;
      PADDLE_ENFORCE_GPU_SUCCESS(
          cub::DeviceReduce::Reduce(tmp_storage_ptr,
                                    tmp_storage_size,
                                    input_iter,
                                    ntokens_ptr,
                                    gbs,
                                    cub::Sum(),
                                    static_cast<IndexT>(0),
                                    stream));
      VLOG(10) << "ntokens_ptr = " << GPUTensorToString<IndexT>(ntokens_ptr, 1);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        &ntokens, ntokens_ptr, sizeof(IndexT), cudaMemcpyDeviceToHost, stream));
    // NOTE: Maybe we do not need this line? D2H copy is always
    // synchronous if we do not use pinned memory.
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    ntokens = gbs;
  }

  VLOG(10) << "ntokens = " << ntokens;

  // The indices in GPU:device_id would be:
  // indices[device_id], indices[device_id + num_devices], ...,
  // indices[device_id + (batch_size - 1) * num_devices]
  // Therefore, device_id + (batch_size - 1) * num_devices < ntokens
  // i.e., batch_size < (ntokens - device_id) / num_devices + 1
  int new_bs = (ntokens - device_id) / num_devices;
  if ((ntokens - device_id) % num_devices > 0) {
    ++new_bs;
  }
  VLOG(10) << "New batch size is: " << new_bs
           << " , original batch size is: " << batch_size;

  std::vector<int64_t> out_shape = {new_bs, seq_len};
  std::vector<int64_t> nsl_out_shape = next_sentence_labels.shape();
  nsl_out_shape[0] = new_bs;

  paddle::Tensor new_input_ids(input_ids.place(), out_shape);
  paddle::Tensor new_segment_ids(segment_ids.place(), out_shape);
  paddle::Tensor new_input_mask(input_mask.place(), out_shape);
  paddle::Tensor new_masked_lm_labels(masked_lm_labels.place(), out_shape);
  paddle::Tensor new_next_sentence_labels(next_sentence_labels.place(),
                                          nsl_out_shape);

  VLOG(10) << "starts to reorder";
  platform::ForRange<platform::CUDADeviceContext> reorder_for_range(
      dev_ctx, new_bs * seq_len);
  ReorderBERTInputTensorsFunctor<T, IndexT> reorder_functor(
      allgather_buf_ptr,
      sorted_indices_ptr,
      device_id,
      num_devices,
      max_batch_size,
      seq_len,
      new_input_ids.mutable_data<T>(input_ids.place()),
      new_segment_ids.mutable_data<T>(segment_ids.place()),
      new_input_mask.mutable_data<T>(input_mask.place()),
      new_masked_lm_labels.mutable_data<T>(masked_lm_labels.place()),
      new_next_sentence_labels.mutable_data<T>(next_sentence_labels.place()));
  reorder_for_range(reorder_functor);
  VLOG(10) << "ends to reorder";

  return {new_input_ids,
          new_segment_ids,
          new_input_mask,
          new_masked_lm_labels,
          new_next_sentence_labels};
}

std::vector<paddle::Tensor> GPUSortBERTInputsAcrossDevices(
    const paddle::Tensor &input_ids,
    const paddle::Tensor &segment_ids,
    const paddle::Tensor &input_mask,
    const paddle::Tensor &masked_lm_labels,
    const paddle::Tensor &next_sentence_labels,
    int max_batch_size,
    int ring_id,
    int device_id,
    int num_devices) {
  PADDLE_ENFORCE_GT(num_devices, 1);

  auto dtype = input_ids.dtype();
#define CALL_DTYPE_FUNC(__dtype, __cpp_type)                      \
  do {                                                            \
    if (dtype == paddle::DataType::__dtype) {                     \
      return GPUSortBERTInputsAcrossDevicesWithDType<__cpp_type>( \
          input_ids,                                              \
          segment_ids,                                            \
          input_mask,                                             \
          masked_lm_labels,                                       \
          next_sentence_labels,                                   \
          max_batch_size,                                         \
          ring_id,                                                \
          device_id,                                              \
          num_devices);                                           \
    }                                                             \
  } while (0)

  CALL_DTYPE_FUNC(INT16, int16_t);
  CALL_DTYPE_FUNC(INT32, int32_t);
  CALL_DTYPE_FUNC(INT64, int64_t);
  PD_THROW("Unsupported data type: %d", static_cast<int>(dtype));
}
