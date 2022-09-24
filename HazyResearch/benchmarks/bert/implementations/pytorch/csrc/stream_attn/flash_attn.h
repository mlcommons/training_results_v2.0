#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int flash_attn_seq_len(int head_size, int max_seq_len, int *base_N_ptr = NULL);

const char *flash_attn_error();

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
                    const float p_dropout, const float softmax_scale,
                    void (*seed_inc_func)(uint64_t, uint64_t *, const int64_t **, uint64_t*, bool*),
                    cudaStream_t stream, void *ctx_ptr, void *softmax_lse_ptr,
                    void *softmax_ptr, void *workspace_ptr, uint64_t *workspace_size);

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
                    const float p_dropout, const float softmax_scale,
                    void (*seed_inc_func)(uint64_t, uint64_t *, const int64_t **, uint64_t*, bool*),
                    cudaStream_t stream, void *dqkv_ptr, void *workspace_ptr, uint64_t *workspace_size);

#ifdef __cplusplus
}
#endif
