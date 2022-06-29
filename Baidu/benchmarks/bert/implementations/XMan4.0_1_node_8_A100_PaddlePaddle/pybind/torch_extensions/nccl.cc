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

#define ASSERT_CHECK(__cond)                          \
  do {                                                \
    if (!(__cond)) throw std::runtime_error(#__cond); \
  } while (0)

#include "nccl.h"  // NOLINT
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "dlfcn.h"  // NOLINT

constexpr ncclRedOp_t UNUSED = ncclProd;

using AllReduceT = decltype(&ncclAllReduce);
using ReduceScatterT = decltype(&ncclReduceScatter);
using RedOpCreatePreMulSumT = decltype(&ncclRedOpCreatePreMulSum);
using RedOpDestroyT = decltype(&ncclRedOpDestroy);

static std::string GetNCCLSoPath() {
  const char *env = std::getenv("NCCL_SO_PATH");
  return env ? std::string(env) : "libnccl.so";
}

struct NCCLHandle {
  NCCLHandle() {
    auto so_path = GetNCCLSoPath();
    void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    this->ncclAllReduce =
        reinterpret_cast<AllReduceT>(dlsym(handle, "ncclAllReduce"));
    ASSERT_CHECK(this->ncclAllReduce != nullptr);
    this->ncclReduceScatter =
        reinterpret_cast<ReduceScatterT>(dlsym(handle, "ncclReduceScatter"));
    ASSERT_CHECK(this->ncclReduceScatter != nullptr);
    this->ncclRedOpCreatePreMulSum = reinterpret_cast<RedOpCreatePreMulSumT>(
        dlsym(handle, "ncclRedOpCreatePreMulSum"));
    ASSERT_CHECK(this->ncclRedOpCreatePreMulSum != nullptr);
    this->ncclRedOpDestroy =
        reinterpret_cast<RedOpDestroyT>(dlsym(handle, "ncclRedOpDestroy"));
    ASSERT_CHECK(this->ncclRedOpDestroy != nullptr);
    fprintf(stderr, "%s loaded successfully\n", so_path.c_str());
  }

  AllReduceT ncclAllReduce = nullptr;
  ReduceScatterT ncclReduceScatter = nullptr;
  RedOpCreatePreMulSumT ncclRedOpCreatePreMulSum = nullptr;
  RedOpDestroyT ncclRedOpDestroy = nullptr;
} g_nccl_handle;

struct NCCLPreMulSumInfo {
  void Init(const void *scalar,
            ncclDataType_t dtype,
            ncclScalarResidence_t residence) {
    scalar_ = const_cast<void *>(scalar);
    dtype_ = dtype;
    residence_ = residence;
  }

  ncclRedOp_t CreateOrReturn(ncclRedOp_t op, ncclComm_t comm) {
    if (op != UNUSED) return op;

    ASSERT_CHECK(ncclSuccess ==
                 g_nccl_handle.ncclRedOpCreatePreMulSum(
                     &op_, scalar_, dtype_, residence_, comm));
    comm_ = comm;
    is_created_ = true;
    return op_;
  }

  void Destroy() {
    if (is_created_) {
      ASSERT_CHECK(ncclSuccess == g_nccl_handle.ncclRedOpDestroy(op_, comm_));
      op_ = UNUSED;
      comm_ = nullptr;
      is_created_ = false;
    }
  }

 private:
  ncclRedOp_t op_ = UNUSED;
  ncclComm_t comm_ = nullptr;
  bool is_created_ = false;

  void *scalar_ = nullptr;
  ncclDataType_t dtype_ = ncclFloat16;
  ncclScalarResidence_t residence_ = ncclScalarDevice;
} g_info;

extern "C" {

void InitNCCLPreMulSum(const void *scalar,
                       ncclDataType_t dtype,
                       ncclScalarResidence_t residence) {
  g_info.Init(scalar, dtype, residence);
}

ncclResult_t ncclAllReduce(const void *sendbuff,
                           void *recvbuff,
                           size_t count,
                           ncclDataType_t datatype,
                           ncclRedOp_t op,
                           ncclComm_t comm,
                           cudaStream_t stream) {
  op = g_info.CreateOrReturn(op, comm);
  auto ret = g_nccl_handle.ncclAllReduce(
      sendbuff, recvbuff, count, datatype, op, comm, stream);
  g_info.Destroy();
  return ret;
}

ncclResult_t ncclReduceScatter(const void *sendbuff,
                               void *recvbuff,
                               size_t recvcount,
                               ncclDataType_t datatype,
                               ncclRedOp_t op,
                               ncclComm_t comm,
                               cudaStream_t stream) {
  op = g_info.CreateOrReturn(op, comm);
  auto ret = g_nccl_handle.ncclReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  g_info.Destroy();
  return ret;
}
}  // extern "C"
