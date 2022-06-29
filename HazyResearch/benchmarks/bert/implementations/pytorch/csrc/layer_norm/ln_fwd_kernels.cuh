#pragma once

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/detail/UnpackRaw.cuh>  // For at::cuda::philox::unpack
#include <curand_kernel.h>

#include "ln.h"

namespace layer_norm {

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) 
void ln_fwd_kernel(FwdParams params) {

    enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
    enum { WARPS_N = Ktraits::WARPS_N };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
    enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
    enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
    enum { LDGS = Ktraits::LDGS };
    enum { NUM_ELTS = Ktraits::NUM_ELTS };
    enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

    using input_t = typename Ktraits::input_t;
    using output_t = typename Ktraits::output_t;
    using index_t = typename Ktraits::index_t;
    using compute_t = typename Ktraits::compute_t;
    using mask_t = typename Ktraits::mask_t;
    using Ivec = typename Ktraits::Ivec;
    using Ovec = typename Ktraits::Ovec;
    using Wvec = typename Ktraits::Wvec;
    using Cvec = typename Ktraits::Cvec;
    using Mvec = typename Ktraits::Mvec;

    using Stats = typename Ktraits::Stats;
    using stats_t = typename Stats::stats_t;

    extern __shared__ char smem_[];

    const index_t tidx = threadIdx.x;
    const index_t bidn = blockIdx.x % CTAS_PER_ROW;
    const index_t bidm = blockIdx.x / CTAS_PER_ROW;
    const index_t lane = tidx % THREADS_PER_WARP;
    const index_t warp = tidx / THREADS_PER_WARP;
    const index_t warp_m = warp / WARPS_N;
    const index_t warp_n = warp % WARPS_N;

    const index_t r = bidm * ROWS_PER_CTA + warp_m;
    const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

    Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);

    compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
    compute_t *rs_ptr = static_cast<compute_t *>(params.rs);

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Dropout.cu
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    const index_t tidx_global = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(std::get<0>(seeds), tidx_global, std::get<1>(seeds), &state);

    Wvec gamma[LDGS];
    Wvec beta[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        gamma[it].load_from(params.gamma, idx);
        beta[it].load_from(params.beta, idx);
        idx += VEC_COLS_PER_LDG;
    }

    constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);

    for( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        Ivec x0[LDGS];
        Ivec x1[LDGS];
        Ivec x[LDGS];
        Mvec dmask[LDGS];
        index_t idx = row * Ktraits::VEC_COLS + c;
        compute_t xf[LDGS * NUM_ELTS];
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            x0[it].load_from(params.x0, idx);
            x1[it].load_from(params.x1, idx);
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                // TD [2022-04-22]: We're memory bound, not compute bound, so we don't need to use
                // the more efficient curand_uniform4.
                float rand = curand_uniform(&state);
                mask_t keep = mask_t(rand <= params.dropout_keep_p);
                compute_t x0_ij = compute_t(x0[it].data.elt[jt]);
                compute_t x1_ij = compute_t(x1[it].data.elt[jt]);
                compute_t x_ij = keep ? x0_ij * params.dropout_scale + x1_ij : x1_ij;
                x[it].data.elt[jt] = input_t(x_ij);
                xf[it * NUM_ELTS + jt] = x_ij;
                dmask[it].data.elt[jt] = keep;
            }
            x[it].store_to(params.x, idx);
            dmask[it].store_to(params.dmask, idx);
            idx += VEC_COLS_PER_LDG;
        }

        stats_t s = stats.compute(xf, rn);

        compute_t mu = layer_norm::Get<0>::of<stats_t, compute_t>(s);
        compute_t m2 = layer_norm::Get<1>::of<stats_t, compute_t>(s);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            mu_ptr[row] = mu;
        }

        compute_t rs = rsqrtf(rn * m2 + params.epsilon);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            rs_ptr[row] = rs;
        }

        Ovec z[LDGS];
        idx = row * Ktraits::VEC_COLS + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                output_t y_ij = output_t(rs * (xf[it * NUM_ELTS + jt] - mu));
                output_t g_ij = gamma[it].data.elt[jt];
                output_t b_ij = beta[it].data.elt[jt];
                z[it].data.elt[jt] = (g_ij * y_ij + b_ij);
            }
            z[it].store_to(params.z, idx);
            idx += VEC_COLS_PER_LDG;
        }

    }
}

}  // namespace layer_norm
