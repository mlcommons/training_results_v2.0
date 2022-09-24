// Pytorch also has an implementation of Philox RNG: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/codegen/cuda/runtime/random_numbers.cu
#pragma once
// Philox CUDA.

// No "#pragma once" because this is a raw definition that can be copied by jit codegen.
// Eager mode clients should not include this file directly, instead,
// they should #include <ATen/CUDAGeneratorImpl.h>, which has a #pragma once.

// Stores RNG state values. Passed as a kernel argument.
// See Note [CUDA Graph-safe RNG states].
//
// The raw definition lives in its own file so jit codegen can easily copy it.

#pragma once

#include <cstdint>
#include <tuple>
#include "math.h"

namespace {

class Philox {
public:
#ifdef __NVCC__
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset)
      : STATE(0)
      , key(reinterpret_cast<const uint2&>(seed)) {
    //key.x = (unsigned int)seed;
    //key.y = (unsigned int)(seed >> 32);
    //counter = make_uint4(0, 0, 0, 0);
    //counter.z = (unsigned int)(subsequence);
    //counter.w = (unsigned int)(subsequence >> 32);
    //STATE = 0;
    //incr_n(offset / 4);

    // key = reinterpret_cast<const uint2&>(seed);
    ull2 * tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset / 4;
    tmp->y = subsequence;
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Philox counter: %d, %d, %d, %d\n", counter.x, counter.y, counter.z, counter.w);
    // }
  }
  __device__ inline uint4 operator()() {
    // if (STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      // 7-round philox
      #pragma unroll
      for (int i = 0; i < 6; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A);
        key_.y += (kPhilox10B);
      }
      // output = single_round(counter_, key_);
      uint4 output = single_round(counter_, key_);
      // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
      //     printf("Philox counter: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
      //     printf("Philox output: %u, %u, %u, %u\n", output.x, output.y, output.z, output.w);
      // }
      incr();
    // }
    // return a float4 directly
    // unsigned long ret;
    // switch(STATE) {
    //  case 0: ret = output.x; break;
    //  case 1: ret = output.y; break;
    //  case 2: ret = output.z; break;
    //  case 3: ret = output.w; break;
    //}
    // STATE = (STATE + 1) % 4;
    return output;
  }
#endif

private:
  struct ull2 {
      uint64_t x;
      uint64_t y;
  };
  uint4 counter;
  // uint4 output;
  const uint2 key;
  unsigned int STATE;
#ifdef __NVCC__
  __device__ inline void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ uint4 incr128 (uint4 ctr)
  {
    uint4 res;
    asm ("add.cc.u32      %0, %4, %8;\n\t"
         "addc.cc.u32     %1, %5, %9;\n\t"
         "addc.cc.u32     %2, %6, %10;\n\t"
         "addc.u32        %3, %7, %11;\n\t"
         : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
         : "r"(ctr.x), "r"(ctr.y), "r"(ctr.z), "r"(ctr.w),
           "n"(1), "n"(0), "n"(0), "n"(0));
    return res;
  }

  __device__ inline void incr() {
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Counter before: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
    // }
    counter = incr128(counter);
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Counter after: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
    // }
  }
  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    *result_high = __umulhi(a, b);
    return a * b;
  }
  __device__ uint2 mulhilo32_v2 (const unsigned int a, const unsigned int b)
  {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32      %0, %1, %2;\n\t"
         : "=l"(tmp)
         : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
  }
  __device__ inline uint4 single_round(const uint4 ctr, const uint2 key) {
    //unsigned int hi0;
    //unsigned int hi1;
    //unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    //unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    //uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    uint2 res0 = mulhilo32_v2(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32_v2(kPhiloxSB, ctr.z); 
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};  
    return ret;
  }
#endif
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};
// Inverse of 2^32.
#ifdef __NVCC__
constexpr float M_RAN_INVM32 = 2.3283064e-10f;
__device__ __inline__ float4 uniform4(const uint4 x) {
  return make_float4(x.x * M_RAN_INVM32, x.y * M_RAN_INVM32, x.z * M_RAN_INVM32,
                     x.w * M_RAN_INVM32);
#endif
}

} // namespace

namespace at {

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed,
                  uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(uint64_t seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_ = 0;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

} // namespace at


// No "#pragma once" because this is a raw definition that can be copied by jit codegen.
// Eager mode clients should not include this file directly, instead,
// they should #include <ATen/cuda/CUDAGraphsUtils.cuh>, which has a #pragma once.

namespace at {
namespace cuda {
namespace philox {

// In-kernel call to retrieve philox seed and offset from a PhiloxCudaState instance whether
// that instance was created with graph capture underway or not.
// See Note [CUDA Graph-safe RNG states].
//
// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a __device__ unpack helper here, in ATen/cuda.
//
// The raw definition lives in its own file so jit codegen can easily copy it.
#if defined(__CUDA_ACC__) or defined(__CUDA_ARCH__) 
#define DEVICE __device__
#else
#define DEVICE
#endif

inline DEVICE std::tuple<uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire kernel.
    // For most threads' reads it will hit in cache, so it shouldn't hurt performance.
    return std::make_tuple(arg.seed_, static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_, arg.offset_.val);
  }
}

} // namespace philox
} // namespace cuda
} // namespace at

