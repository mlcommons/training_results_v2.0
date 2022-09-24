/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "fmha.h"
#include <cmath>
#include <cstring>
#include <string>
#include <exception>
#include <stdexcept>
#include <mutex>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
#include "dlfcn.h"

using SeedIncFuncPtr = void (*)(uint64_t, uint64_t *, const int64_t **, uint64_t*, bool*);

#define ASSERT_CHECK(__cond)                             \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
		__FILE__ + ":" +                         \
		::std::to_string(__LINE__);              \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)

#ifdef TORCH_CHECK
#undef TORCH_CHECK
#endif
#define TORCH_CHECK ASSERT_CHECK

static thread_local std::unique_ptr<char[]> flash_attn_err_msg;

#ifdef __cplusplus
extern "C" {
#endif

static void flash_attn_set_error(const char *msg) {
  if (msg == nullptr || *msg == '\0') {
    msg = "unknown error";
  }

  auto n = strlen(msg);
  std::unique_ptr<char[]> new_err_msg(new char[n+1]);
  std::strcpy(new_err_msg.get(), msg);
  flash_attn_err_msg = std::move(new_err_msg);
}

const char *flash_attn_error() {
  return flash_attn_err_msg.get();
}
#ifdef __cplusplus
}
#endif

#define FMHALIB_BEGIN_FUNC try {
#define FMHALIB_END_FUNC } catch (::std::exception &__e) { flash_attn_set_error(__e.what()); } catch (...) { flash_attn_set_error(nullptr); }

static void set_params(Fused_multihead_attention_fprop_params &params,
                // sizes
                const size_t b,
                const size_t s,
                const size_t h,
                const size_t d,
                // device pointers
                void *qkv_packed_d,
                void *cu_seqlens_d,
                void *o_packed_d,
                void *o_tmp_d,
                void *do_packed_d,
                void *s_d,
                void *softmax_lse_d,
                void *dsoftmax_sum_d,
                float p_dropout,
                float softmax_scale,
                bool is_causal) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.qkv_ptr = qkv_packed_d;
    params.qkv_stride_in_elts = h * 3 * d;
    params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.o_ptr = o_packed_d;
    params.o_stride_in_elts = h * d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);
    params.do_ptr = do_packed_d;
    params.o_tmp_ptr = o_tmp_d;

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;
    params.dsoftmax_sum = dsoftmax_sum_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
}

static void SetPhiloxCudaState(at::PhiloxCudaState *state, SeedIncFuncPtr seed_inc_func, uint64_t increment) {
    uint64_t rnd_seed;
    const int64_t *offset_ptr;
    uint64_t rnd_offset;
    bool is_device_rnd;
    seed_inc_func(increment, &rnd_seed, &offset_ptr, &rnd_offset, &is_device_rnd);
    if (is_device_rnd) {
        *state = at::PhiloxCudaState(rnd_seed, const_cast<int64_t *>(offset_ptr), static_cast<uint32_t>(rnd_offset));
    } else {
        *state = at::PhiloxCudaState(rnd_seed, rnd_offset);
    }
}

static cudaDeviceProp g_prop;

static cudaDeviceProp *GetCurrentDeviceProperties() {
    static std::once_flag flag;   
    std::call_once(flag, [] {
      int dev_id;
      TORCH_CHECK(cudaGetDevice(&dev_id) == cudaSuccess);
      TORCH_CHECK(cudaGetDeviceProperties(&g_prop, dev_id) == cudaSuccess);
    });
    return &g_prop;
}   

static void SetZero(void *ptr, size_t sizeof_type, std::initializer_list<int> shapes, cudaStream_t stream) {
    size_t n = sizeof_type;
    for (int s : shapes) n *= s;
    TORCH_CHECK(cudaMemsetAsync(ptr, 0, n, stream) == cudaSuccess); 
}

template <typename T>
static __global__ void FillConstantKernel(T *ptr, T value, size_t n) {
  auto idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] = value;
  }
} 

template <typename T>
static void SetConstValue(void *ptr, T value, size_t n, cudaStream_t stream) {
  constexpr auto kNumThreads = 1024;
  auto block = (n + kNumThreads - 1) / kNumThreads; 
  FillConstantKernel<T><<<block, kNumThreads, 0, stream>>>(static_cast<T *>(ptr), value, n);
} 

#ifdef __cplusplus
extern "C" {
#endif

int flash_attn_seq_len(int head_size, int max_seq_len, int *base_N_ptr) {
    int base_N = head_size == 128 ? 128 : 256;
    if (base_N_ptr) *base_N_ptr = base_N;
    if( max_seq_len <= 128 ) { 
        return 128;
    } else if( max_seq_len <= 256 ) { 
        return 256;
    } else {
        return ((max_seq_len + base_N - 1) / base_N) * base_N;
    } 
}

// qkv_ptr: FP16, [total, num_heads, 3, head_size]
// cu_seqlens_ptr: INT32, [batch_size + 1]
// ctx_ptr: FP16, [total, num_heads, head_size]
// softmax_lse_ptr: FP32, [batch_size, num_heads, seq_len]   
// softmax_ptr: FP16, [batch_size, num_heads, seq_len, seq_len]
// workspace_ptr: FP32, [total, num_heads, head_size] 
void flash_attn_fwd(const void *qkv_ptr, const void *cu_seqlens_ptr, 
                    const int total, const int batch_size, const int num_heads, 
                    const int head_size, const int max_seq_len, 
                    const bool is_training, const bool zero_tensors, const bool is_causal,
                    const float p_dropout, const float softmax_scale, SeedIncFuncPtr seed_inc_func,
                    cudaStream_t stream, void *ctx_ptr, void *softmax_lse_ptr,
                    void *softmax_ptr, void *workspace_ptr, uint64_t *workspace_size) { 
    FMHALIB_BEGIN_FUNC             
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);
    
    int base_N;
    int seq_len = flash_attn_seq_len(head_size, max_seq_len, &base_N);
    bool loop = seq_len > base_N;
    if (ctx_ptr == nullptr) {
        *workspace_size = loop ? uint64_t(total) * num_heads * head_size * sizeof(float): 0; 
        return; 
    }

    auto dprops = GetCurrentDeviceProperties();   
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    bool is_dropout;
    if (is_training) {
      TORCH_CHECK(p_dropout > 0.0);
      is_dropout = true;
    } else {
      is_dropout = false;
    }
    const bool return_softmax = (softmax_ptr != nullptr);
    Launch_params<Fused_multihead_attention_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    void *o_tmp_ptr = workspace_ptr;

    if( zero_tensors ) {
        SetZero(ctx_ptr, 2, {total, num_heads, head_size}, stream); 
        SetConstValue<float>(softmax_lse_ptr, -std::numeric_limits<float>::infinity(), uint64_t(batch_size) * num_heads * seq_len, stream);   
        if (loop) SetZero(o_tmp_ptr, 4, {total, num_heads, head_size}, stream);
        if (return_softmax) SetZero(softmax_ptr, 2, {batch_size, num_heads, seq_len, seq_len}, stream);  
    }

    set_params(launch_params.params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               ctx_ptr,
               loop ? o_tmp_ptr : nullptr,
               nullptr,
               return_softmax ? softmax_ptr : nullptr,
               softmax_lse_ptr,
               nullptr,
               p_dropout,
               softmax_scale,
               is_causal);

    run_fmha_fp16_sm80(launch_params, /*configure=*/ true);
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        SetPhiloxCudaState(&rng_engine_inputs, seed_inc_func, counter_offset); 
    }

    run_fmha_fp16_sm80(launch_params, /*configure=*/false);
    FMHALIB_END_FUNC
}

// qkv_ptr: FP16, [total, num_heads, 3, head_size]
// cu_seqlens_ptr: INT32, [batch_size + 1]
// ctx_ptr: FP16, [total, num_heads, head_size]
// softmax_lse_ptr: FP32, [batch_size, num_heads, seq_len]   
// softmax_ptr: FP16, [batch_size, num_heads, seq_len, seq_len]
// workspace_ptr: FP32, [total, num_heads, head_size] 
// dctx_ptr: FP16, [total, num_heads, head_size]
void flash_attn_bwd(const void *dctx_ptr, const void *qkv_ptr, const void *ctx_ptr, 
                    const void *softmax_ptr, const void *softmax_lse_ptr, 
                    const void *cu_seqlens_ptr, 
                    const int total, const int batch_size, const int num_heads, 
                    const int head_size, const int max_seq_len, 
                    const bool zero_tensors, const bool is_causal,
                    const float p_dropout, const float softmax_scale, SeedIncFuncPtr seed_inc_func,
                    cudaStream_t stream, void *dqkv_ptr, void *workspace_ptr, uint64_t *workspace_size) { 
    FMHALIB_BEGIN_FUNC
    int base_N;
    int seq_len = flash_attn_seq_len(head_size, max_seq_len, &base_N);
    bool loop = seq_len > base_N;
    if (dqkv_ptr == nullptr) {
      *workspace_size = uint64_t(batch_size) * num_heads * seq_len * sizeof(float);    
      if (loop) {
        (*workspace_size) += uint64_t(total) * num_heads * head_size * sizeof(float);
      }
      return;
    }

    auto dprops = GetCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    auto launch = &run_fmha_dgrad_fp16_sm80;

    bool is_dropout = p_dropout > 0.0;

    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128); 

    auto *softmax_d_ptr = reinterpret_cast<float *>(workspace_ptr); 
    auto *dq_tmp_ptr = loop ? softmax_d_ptr + uint64_t(batch_size) * num_heads * seq_len : nullptr;

    if( zero_tensors ) {
        SetZero(dqkv_ptr,  2, {total, num_heads, 3, head_size}, stream);
        SetZero(softmax_d_ptr, 4, {batch_size, num_heads, seq_len}, stream);
        if (loop) SetZero(dq_tmp_ptr, 4, {total, num_heads, head_size}, stream);
    }

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               const_cast<void *>(qkv_ptr),
               const_cast<void *>(cu_seqlens_ptr),
               const_cast<void *>(ctx_ptr),
               loop ? dq_tmp_ptr : nullptr,
               const_cast<void *>(dctx_ptr),
               const_cast<void *>(softmax_ptr),  // softmax gets overwritten by dP!
               const_cast<void *>(softmax_lse_ptr),
               softmax_d_ptr,
               p_dropout,
               softmax_scale,
               is_causal);

    // We're gonna reset the rng state in Python after this kernel, so the counter offset
    // here doesn't matter at all. We just choose an arbitrary number;
    int64_t counter_offset = 4;
    at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        SetPhiloxCudaState(&rng_engine_inputs, seed_inc_func, counter_offset);
    }

    params.dqkv_ptr = dqkv_ptr;
    launch(params, stream);
    FMHALIB_END_FUNC
}

#ifdef __cplusplus
}
#endif
