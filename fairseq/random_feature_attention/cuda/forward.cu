#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include <vector>
#include <stdio.h>
typedef torch::Tensor Tensor;
#define FULL_MASK 0xffffffff

const int DIM_PER_THREAD = 16;
const int HALF2_PER_THREAD = DIM_PER_THREAD / 2;
const int INT4_PER_THREAD = DIM_PER_THREAD / 8;

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}


__forceinline__ __device__ __half clamp_one(__half x) {
    __half one = __float2half(0.01);
    return __hgt(x, one) ? x : one;
}


__device__
void read_qkv(
    const __half * __restrict__ q_local,
    const __half * __restrict__ k_local,
    const __half * __restrict__ v_local,
    __half2 q_val[HALF2_PER_THREAD],
    __half2 k_val[HALF2_PER_THREAD],
    __half2 v_val[4]) {
    
    #pragma unroll
    for (int i = 0; i < INT4_PER_THREAD; ++ i) {
        *((int4 *) q_val + i) = ((int4*) q_local)[i];
    }

    #pragma unroll
    for (int i = 0; i < INT4_PER_THREAD; ++ i) {
        *((int4 *) k_val + i) = ((int4*) k_local)[i];
    }
    *((int4 *) v_val) = ((int4*) v_local)[0];
}


__device__
void rfa_step(
        __half * __restrict__ attn_local,
        __half2 q_val[HALF2_PER_THREAD], 
        __half2 k_val[HALF2_PER_THREAD], 
        __half2 v_val[4], 
        __half2 s1[4][HALF2_PER_THREAD],
        __half2 s2[4][HALF2_PER_THREAD],
        __half2 z[HALF2_PER_THREAD],
        int num_threads_per_head_dim) {
    __half2 qs[4] = {__float2half2_rn(0.f)};
    __half2 qz = __float2half2_rn(0.f);
    
    __half2 v_val_swapped[4] = {
        __lowhigh2highlow(v_val[0]),
        __lowhigh2highlow(v_val[1]),
        __lowhigh2highlow(v_val[2]),
        __lowhigh2highlow(v_val[3]),
    };

    #pragma unroll 
    for (int i = 0;i < 4; ++ i) {
        qs[i] = __float2half2_rn(0.f);
        __half2 qs_swapped = __float2half2_rn(0.f);
        #pragma unroll 
        for (int j = 0;j < HALF2_PER_THREAD; ++ j) {
            s1[i][j] = __hfma2(v_val[i], k_val[j], s1[i][j]);
            s2[i][j] = __hfma2(v_val_swapped[i], k_val[j], s2[i][j]);
            
            qs[i] = __hfma2(s1[i][j], q_val[j], qs[i]);
            qs_swapped = __hfma2(s2[i][j], q_val[j], qs_swapped);
        }
        qs[i] = __hadd2(qs[i], __lowhigh2highlow(qs_swapped));
    }

    #pragma unroll 
    for (int j = 0; j < HALF2_PER_THREAD; ++ j) {
        z[j] = __hadd2(k_val[j], z[j]);
        qz = __hfma2(z[j], q_val[j], qz);
    }

    #pragma unroll 
    for (int offset = num_threads_per_head_dim >> 1; 
        offset > 0; 
        offset >>= 1) {
        qz =  __hadd2(qz, __shfl_down_sync(FULL_MASK, qz, offset));
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs[i] =  __hadd2(
                qs[i], __shfl_down_sync(FULL_MASK, qs[i], offset));
        }
    }
    
    if (threadIdx.x == 0) {
        __half qz_half = __hadd(qz.x, qz.y);
        qz_half = clamp_one(qz_half);
        qz = __half2half2(qz_half);
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs[i] =  __h2div(qs[i], qz);
        }
        *((int4 *) attn_local) = ((int4 *) qs)[0];
    }
}


__global__ 
void rfa_forward(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        __half * __restrict__ attn,
        int tgt_len, 
        int bsz, 
        int head_dim, 
        int proj_dim,
        int num_threads_per_head_dim) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        k: [tgt_len, bsz, proj_dim]
        v: [tgt_len, bsz, head_dim]
        
    Return:
        attn: [tgt_len, bsz, head_dim]
    */

    int bid = blockIdx.x;
    int head_dim_offset = threadIdx.y << 3;
    int proj_dim_offset = threadIdx.x * DIM_PER_THREAD;

    const __half * __restrict__ q_local = q + bid * proj_dim + proj_dim_offset;
    const __half * __restrict__ k_local = k + bid * proj_dim  + proj_dim_offset;
    const __half * __restrict__ v_local = v + bid * head_dim + head_dim_offset;
    
    __half * __restrict__ attn_local = attn + bid * head_dim + head_dim_offset;
    
    __half2 q_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 k_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 v_val[4] = {__float2half2_rn(0.f)};
    __half2 s1[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 s2[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};

    const int qk_inc_t = bsz * proj_dim;
    const int v_inc_t = bsz * head_dim;
    for (int t = 0; t < tgt_len; ++ t) {
        read_qkv(q_local, k_local, v_local,
                 q_val, k_val, v_val);
        rfa_step(
            attn_local,
            q_val, k_val, v_val,
            s1, s2, z,
            num_threads_per_head_dim);
        q_local += qk_inc_t;
        k_local += qk_inc_t;
        v_local += v_inc_t;
        attn_local += v_inc_t;
    }
}

Tensor RFAForward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        k: [tgt_len, bsz, proj_dim]
        v: [tgt_len, bsz, head_dim]
        
    Return:
        attn: [tgt_len, bsz, head_dim]
    */
    // column major
    const int tgt_len = q.size(0);
    const int bsz = q.size(1);
    const int head_dim = v.size(2);
    const int proj_dim = q.size(2);

    auto act_options  = q.options().requires_grad(false);
    Tensor attn = torch::zeros({
        tgt_len, bsz, head_dim}, act_options);
    
    // num threads per head_dim;
    const int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    dim3 dim_grid(bsz);
    // [x, y]
    dim3 dim_block(num_threads_per_head_dim, head_dim / 8);
    rfa_forward <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (q.data_ptr()), 
            static_cast<const __half *> (k.data_ptr()), 
            static_cast<const __half *> (v.data_ptr()),
            static_cast<__half *> (attn.data_ptr()), 
            tgt_len,
            bsz,
            head_dim, 
            proj_dim,
            num_threads_per_head_dim
    );
 
    return attn;
}
