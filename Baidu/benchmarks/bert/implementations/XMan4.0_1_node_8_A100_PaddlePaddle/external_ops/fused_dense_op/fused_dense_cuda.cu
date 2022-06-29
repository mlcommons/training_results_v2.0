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

#define CUBLAS_VERSION 13000

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
// includes cublaslt
#include <cublasLt.h>
#endif

#include "paddle/extension.h"
#include "paddle/fluid/framework/custom_raw_op_kernel_func.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

#define CHECK_CUBLAS_ERR(error_code)                 \
  do {                                               \
    if (error_code != CUBLAS_STATUS_SUCCESS) {       \
      PD_THROW("cublas error code is ", error_code); \
    }                                                \
  } while (0)

// todo: allocate 4MB. (the following code looks like 4MB * sizeof(T)?)
constexpr auto kWorkspaceSize = (1 << 22);

// FP64 Wrapper around cublas GEMMEx
// TODO(limin): in fact, alpha and beta are double type.
cublasStatus_t gemm_bias(cublasHandle_t handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const double* A,
                         int lda,
                         const double* B,
                         int ldb,
                         const float* beta,
                         double* C,
                         int ldc) {
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      CUDA_R_64F,
                      lda,
                      B,
                      CUDA_R_64F,
                      ldb,
                      beta,
                      C,
                      CUDA_R_64F,
                      ldc,
                      CUDA_R_64F,
                      CUBLAS_GEMM_DEFAULT);
}

// FP32 Wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(cublasHandle_t handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const float* A,
                         int lda,
                         const float* B,
                         int ldb,
                         const float* beta,
                         float* C,
                         int ldc) {
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      CUDA_R_32F,
                      lda,
                      B,
                      CUDA_R_32F,
                      ldb,
                      beta,
                      C,
                      CUDA_R_32F,
                      ldc,
                      CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT);
}

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(cublasHandle_t handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const paddle::float16* A,
                         int lda,
                         const paddle::float16* B,
                         int ldb,
                         const float* beta,
                         paddle::float16* C,
                         int ldc) {
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      CUDA_R_16F,
                      lda,
                      B,
                      CUDA_R_16F,
                      ldb,
                      beta,
                      C,
                      CUDA_R_16F,
                      ldc,
                      CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
// float16 and float32
template <typename T>
cublasStatus_t cublaslt_matmul_desc_init(
    cublasLtMatmulDescOpaque_t* operationDesc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status =
      cublasLtMatmulDescInit(operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  return status;
}

// float64
template <>
cublasStatus_t cublaslt_matmul_desc_init<double>(
    cublasLtMatmulDescOpaque_t* operationDesc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status =
      cublasLtMatmulDescInit(operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F);
  return status;
}

// float16
template <typename T>
cublasStatus_t set_cublaslt_matrix_layout_init(
    cublasLtMatrixLayoutOpaque_t* Adesc,
    cublasLtMatrixLayoutOpaque_t* Bdesc,
    cublasLtMatrixLayoutOpaque_t* Cdesc,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasLtMatrixLayoutInit(Adesc,
                                    CUDA_R_16F,
                                    transa == CUBLAS_OP_N ? m : k,
                                    transa == CUBLAS_OP_N ? k : m,
                                    lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Bdesc,
                                    CUDA_R_16F,
                                    transb == CUBLAS_OP_N ? k : n,
                                    transb == CUBLAS_OP_N ? n : k,
                                    ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Cdesc, CUDA_R_16F, m, n, ldc);

CLEANUP:
  return status;
}

template <>
cublasStatus_t set_cublaslt_matrix_layout_init<float>(
    cublasLtMatrixLayoutOpaque_t* Adesc,
    cublasLtMatrixLayoutOpaque_t* Bdesc,
    cublasLtMatrixLayoutOpaque_t* Cdesc,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasLtMatrixLayoutInit(Adesc,
                                    CUDA_R_32F,
                                    transa == CUBLAS_OP_N ? m : k,
                                    transa == CUBLAS_OP_N ? k : m,
                                    lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Bdesc,
                                    CUDA_R_32F,
                                    transb == CUBLAS_OP_N ? k : n,
                                    transb == CUBLAS_OP_N ? n : k,
                                    ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Cdesc, CUDA_R_32F, m, n, ldc);
CLEANUP:
  return status;
}

template <>
cublasStatus_t set_cublaslt_matrix_layout_init<double>(
    cublasLtMatrixLayoutOpaque_t* Adesc,
    cublasLtMatrixLayoutOpaque_t* Bdesc,
    cublasLtMatrixLayoutOpaque_t* Cdesc,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasLtMatrixLayoutInit(Adesc,
                                    CUDA_R_64F,
                                    transa == CUBLAS_OP_N ? m : k,
                                    transa == CUBLAS_OP_N ? k : m,
                                    lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Bdesc,
                                    CUDA_R_64F,
                                    transb == CUBLAS_OP_N ? k : n,
                                    transb == CUBLAS_OP_N ? n : k,
                                    ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Cdesc, CUDA_R_64F, m, n, ldc);

CLEANUP:
  return status;
}
#endif

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
template <typename T>
int gemm_bias_lt(cublasLtHandle_t ltHandle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float* alpha, /* host pointer */
                 const T* A,
                 int lda,
                 const T* B,
                 int ldb,
                 const float* beta, /* host pointer */
                 T* C,
                 int ldc,
                 void* workspace,
                 size_t workspaceSize,
                 cudaStream_t stream,
                 bool use_bias,
                 const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublaslt_matmul_desc_init<T>(&operationDesc);

  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(
        &operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue,
                                          sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = set_cublaslt_matrix_layout_init<T>(
      &Adesc, &Bdesc, &Cdesc, transa, transb, m, n, k, lda, ldb, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
      &preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                          &operationDesc,
                                          &Adesc,
                                          &Bdesc,
                                          &Cdesc,
                                          &Cdesc,
                                          &preference,
                                          1,
                                          &heuristicResult,
                                          &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          reinterpret_cast<void*>(C),
                          &Cdesc,
                          // &heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  PADDLE_ENFORCE_GPU_SUCCESS(status);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
#endif

template <typename T>
static int linear_bias_cuda_forward_impl(
    const paddle::platform::CUDADeviceContext& dev_ctx,
    const T* input_data,
    const T* weight_data,
    const T* bias_data,
    bool transx,
    bool transy,
    int in_features,
    int batch_size,
    int out_features,
    T* output_data,
    void* lt_workspace) {
  auto handle = dev_ctx.cublas_handle();
  auto stream = dev_ctx.stream();

  const float alpha = 1.0;
  const float beta_zero = 0.0;
  const float beta_one = 1.0;
  int status = 1;

  // nt
  cublasOperation_t transpose_x = CUBLAS_OP_T;
  cublasOperation_t transpose_y = CUBLAS_OP_N;
  if (transy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    status = gemm_bias_lt((cublasLtHandle_t)handle,
                          transpose_x,
                          transpose_y,
                          out_features,
                          batch_size,
                          in_features,
                          &alpha, /* host pointer */
                          weight_data,
                          in_features,
                          input_data,
                          in_features,
                          &beta_zero, /* host pointer */
                          output_data,
                          out_features,
                          lt_workspace,
                          kWorkspaceSize,
                          stream,
                          true,
                          bias_data);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
#if 0
        output.copy_(bias);
        status = gemm_bias(
            handle,
            transpose_x,
            transpose_y,
            out_features,
            batch_size,
            in_features,
            &alpha,
            weight,
            in_features,
            input_data,
            in_features,
            &beta_one,
            output_data,
            out_features);
#endif
    }
  } else {
    // nn
    transpose_x = CUBLAS_OP_N;
    transpose_y = CUBLAS_OP_N;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    status = gemm_bias_lt((cublasLtHandle_t)handle,
                          transpose_x,
                          transpose_y,
                          out_features,
                          batch_size,
                          in_features,
                          &alpha, /* host pointer */
                          weight_data,
                          out_features,
                          input_data,
                          in_features,
                          &beta_zero, /* host pointer */
                          output_data,
                          out_features,
                          lt_workspace,
                          kWorkspaceSize,
                          stream,
                          true,
                          bias_data);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
    }
  }
  return status;
}

template <typename T>
int gemm_bgradb_lt(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha, /* host pointer */
                   const T* A,
                   int lda,
                   const T* B,
                   int ldb,
                   const float* beta, /* host pointer */
                   T* C,
                   int ldc,
                   void* workspace,
                   size_t workspaceSize,
                   cudaStream_t stream,
                   bool use_bias,
                   const void* bgrad,
                   cublasLtEpilogue_t epilogue) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  // cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublaslt_matmul_desc_init<T>(&operationDesc);

  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc,
                                            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bgrad,
                                            sizeof(bgrad));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue,
                                          sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = set_cublaslt_matrix_layout_init<T>(
      &Adesc, &Bdesc, &Cdesc, transa, transb, m, n, k, lda, ldb, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
      &preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                          &operationDesc,
                                          &Adesc,
                                          &Bdesc,
                                          &Cdesc,
                                          &Cdesc,
                                          &preference,
                                          1,
                                          &heuristicResult,
                                          &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          // &heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

template <typename T>
int linear_bias_cuda_backward_impl(
    const paddle::platform::CUDADeviceContext& dev_ctx,
    const T* input,
    const T* weight,
    const T* d_output,
    bool transx,
    bool transy,
    bool use_addto,
    int in_features,
    int batch_size,
    int out_features,
    T* d_weight,
    T* d_bias,
    T* d_input,
    void* lt_workspace) {
  auto handle = dev_ctx.cublas_handle();
  auto stream = dev_ctx.stream();

  const float alpha = 1.0;
  const float beta_zero = 0.0;
  const float beta_one = 1.0;
  int status = 1;

  if (transy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    // cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;
    status = gemm_bgradb_lt((cublasLtHandle_t)handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            in_features,
                            out_features,
                            batch_size,
                            &alpha, /* host pointer */
                            input,
                            in_features,
                            d_output,
                            out_features,
                            &beta_zero, /* host pointer */
                            d_weight,
                            in_features,
                            lt_workspace,
                            kWorkspaceSize,
                            stream,
                            true,
                            static_cast<const void*>(d_bias),
                            epilogue);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
#if 0
      status = gemm_bias(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          in_features,
          out_features,
          batch_size,
          &alpha,
          input,
          in_features,
          d_output,
          out_features,
          &beta_zero,
          d_weight,
          in_features);
#endif
    }
  } else {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADA;
    status = gemm_bgradb_lt((cublasLtHandle_t)handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            out_features,
                            in_features,
                            batch_size,
                            &alpha, /* host pointer */
                            d_output,
                            out_features,
                            input,
                            in_features,
                            &beta_zero, /* host pointer */
                            d_weight,
                            out_features,
                            lt_workspace,
                            kWorkspaceSize,
                            stream,
                            true,
                            static_cast<const void*>(d_bias),
                            epilogue);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
    }
  }

  cublasOperation_t transpose_x = CUBLAS_OP_N;
  cublasOperation_t transpose_y = CUBLAS_OP_N;
  const float beta_dinput = (use_addto ? beta_one : beta_zero);
  if (transy) {
    status = gemm_bias(handle,
                       transpose_x,
                       transpose_y,
                       in_features,
                       batch_size,
                       out_features,
                       &alpha,
                       weight,
                       in_features,
                       d_output,
                       out_features,
                       &beta_dinput,
                       d_input,
                       in_features);
  } else {
    transpose_x = CUBLAS_OP_T;
    transpose_y = CUBLAS_OP_N;
    status = gemm_bias(handle,
                       transpose_x,
                       transpose_y,
                       in_features,
                       batch_size,
                       out_features,
                       &alpha,
                       weight,
                       out_features,
                       d_output,
                       out_features,
                       &beta_dinput,
                       d_input,
                       in_features);
  }
  return status;
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dense, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  const auto* y = ctx.Input<f::Tensor>("Y");
  const auto* bias = ctx.Input<f::Tensor>("Bias");
  auto* out = ctx.Output<f::Tensor>("Out");
  bool transx = ctx.Attr<bool>("transx");
  bool transy = ctx.Attr<bool>("transy");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  if (transx) {
    PD_THROW("Attr(transx) must be False currently.");
  }

  const auto& x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_k = x_dims[x_dims.size() - 1];

  const auto& y_dims = y->dims();
  int y_k = y_dims[0];
  int y_n = y_dims[1];
  if (transy) {
    y_k = y_dims[1];
    y_n = y_dims[0];
  }
  if (x_k != y_k) {
    PD_THROW("The reudce dim of A and B in matmul is not equal.");
  }

  auto out_dims = x_dims;
  out_dims[x_dims.size() - 1] = y_n;
  out->Resize(out_dims);

  f::Tensor lt_workspace;
  lt_workspace.Resize({kWorkspaceSize});

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x->dtype(), "linear_bias_cuda_forward_impl", ([&] {
        linear_bias_cuda_forward_impl<data_t>(
            dev_ctx,
            x->data<data_t>(),
            y->data<data_t>(),
            bias->data<data_t>(),
            transx,
            transy,
            x_k,
            x_m,
            y_n,
            out->mutable_data<data_t>(place),
            lt_workspace.mutable_data<data_t>(place));
      }));
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dense_grad, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  const auto* y = ctx.Input<f::Tensor>("Y");
  const auto* grad_out = ctx.Input<f::Tensor>(f::GradVarName("Out"));
  auto* grad_x = ctx.Output<f::Tensor>(f::GradVarName("X"));
  auto* grad_y = ctx.Output<f::Tensor>(f::GradVarName("Y"));
  auto* grad_bias = ctx.Output<f::Tensor>(f::GradVarName("Bias"));

  bool transx = ctx.Attr<bool>("transx");
  bool transy = ctx.Attr<bool>("transy");
  bool use_addto = ctx.Attr<bool>("use_addto");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  if (transx) {
    PD_THROW("Attr(transx) must be False currently.");
  }

  const auto& x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_k = x_dims[x_dims.size() - 1];

  const auto& y_dims = y->dims();
  int y_k = y_dims[0];
  int y_n = y_dims[1];
  if (transy) {
    y_k = y_dims[1];
    y_n = y_dims[0];
  }
  if (x_k != y_k) {
    PD_THROW("The reudce dim of A and B in matmul is not equal.");
  }

  grad_x->Resize(x_dims);
  grad_y->Resize(y_dims);
  grad_bias->Resize({y_n});

  f::Tensor lt_workspace;
  lt_workspace.Resize({kWorkspaceSize});

#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  PD_THROW(
      "fused_dense_cuda_backward is not supported on cuda_version < 11000");
#endif

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x->dtype(), "linear_bias_cuda_backward_impl", ([&] {
        linear_bias_cuda_backward_impl<data_t>(
            dev_ctx,
            x->data<data_t>(),
            y->data<data_t>(),
            grad_out->data<data_t>(),
            transx,
            transy,
            use_addto,
            x_k,
            x_m,
            y_n,
            grad_y->mutable_data<data_t>(place),
            grad_bias->mutable_data<data_t>(place),
            grad_x->mutable_data<data_t>(place),
            lt_workspace.mutable_data<data_t>(place));
      }));
}
