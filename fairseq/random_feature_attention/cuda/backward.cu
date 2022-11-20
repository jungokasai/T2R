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
typedef __half real;
typedef torch::Tensor Tensor;
const int MAX_THREAD_PER_BLOCK = 512;
#define FULL_MASK 0xffffffff

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

__forceinline__ __device__ void
read_int4(__half2 val[4], const __half2* __restrict__ ptr) {
    int4 in0 =((int4*) ptr)[0];

    val[0] = *((__half2 *) &in0.x);
    val[1] = *((__half2 *) &in0.y);
    val[2] = *((__half2 *) &in0.z);
    val[3] = *((__half2 *) &in0.w);
}

__forceinline__ __device__ void
read_int4(__half2 val[4], int4 in0) {

    val[0] = *((__half2 *) &in0.x);
    val[1] = *((__half2 *) &in0.y);
    val[2] = *((__half2 *) &in0.z);
    val[3] = *((__half2 *) &in0.w);
}

__forceinline__ __device__ void
read_2int4(__half2 val[8], const __half2* __restrict__ ptr) {
    int4 in0 =((int4*) ptr)[0];
    int4 in1 =((int4*) ptr)[1];

    val[0] = *((__half2 *) &in0.x);
    val[1] = *((__half2 *) &in0.y);
    val[2] = *((__half2 *) &in0.z);
    val[3] = *((__half2 *) &in0.w);

    val[4] = *((__half2 *) &in1.x);
    val[5] = *((__half2 *) &in1.y);
    val[6] = *((__half2 *) &in1.z);
    val[7] = *((__half2 *) &in1.w);
}

__device__
void calculate_gq(
        const __half2* __restrict__ q_local,
        const __half2* __restrict__ k_local,
        const __half2* __restrict__ v_local,
        __half                      &qz_half,
        const __half2* __restrict__ ga_local,
        __half2 gq_val[4],
        __half  &gqz_half,
        __half2 s_val[4][4],
        __half2 s_val_swapped[4][4],
        __half2 z_val[4],
        __half  shared_mem_half[MAX_THREAD_PER_BLOCK],
        int4    shared_mem_int4[MAX_THREAD_PER_BLOCK],
        const int NTPQK,
        const int NTPH) {


   __half2 q_val[4], k_val[4], v_val[4], qs_val[4], gqs_val[4];

   read_int4(q_val, q_local);
   read_int4(k_val, k_local);
   read_int4(v_val, v_local);
   read_int4(gqs_val, ga_local);

   __half2 v_val_swapped[4] = {
        __lowhigh2highlow(v_val[0]),
        __lowhigh2highlow(v_val[1]),
        __lowhigh2highlow(v_val[2]),
        __lowhigh2highlow(v_val[3]),
    };

    __half2 qz_val = __float2half2_rn(0.f);
    __half2 gqz_val = __float2half2_rn(0.f);

    /* QS QZ starts */
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        qs_val[j] = __float2half2_rn(0.f);
        __half2 qs_val_swapped = __float2half2_rn(0.f);
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            // S
            s_val[i][j] = __hfma2(v_val[j], k_val[i], s_val[i][j]);
            s_val_swapped[i][j] = __hfma2(v_val_swapped[j], k_val[i], s_val_swapped[i][j]);

            // QS
            qs_val[j] = __hfma2(s_val[i][j], q_val[i], qs_val[j]);
            qs_val_swapped = __hfma2(s_val_swapped[i][j], q_val[i], qs_val_swapped);

        }
        qs_val[j] = __hadd2(qs_val[j], __lowhigh2highlow(qs_val_swapped));
        z_val[j] = __hadd2(z_val[j], k_val[j]);
        qz_val = __hfma2(z_val[j], q_val[j], qz_val);
    }


    // 128 x 256 case:
    // sum through 256 / 8 = 32 proj_dim threads:
    // thread_idx:
    // 0 + 1 + 2 + ... + 31
    // 32 + 33 + 34 + ... + 63
    // ...
    #pragma unroll
    for (int offset = NTPH / 2; offset > 0; offset >>= 1) {
        qz_val =  __hadd2(qz_val, __shfl_down_sync(FULL_MASK, qz_val, offset));
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            qs_val[j] =  __hadd2(qs_val[j], __shfl_down_sync(FULL_MASK, qs_val[j], offset));
        }
    }
    qz_half = __hadd(qz_val.x, qz_val.y);
    qz_half = clamp_one(qz_half);

    int remain = threadIdx.x % NTPH;
    if (remain == 0) {
        shared_mem_half[threadIdx.x] = qz_half;
        shared_mem_int4[threadIdx.x] = ((int4*) qs_val)[0];
    }
    __syncthreads();
    if (remain > 0) {
        qz_half = shared_mem_half[threadIdx.x - remain];
        ((int4*) qs_val)[0] = shared_mem_int4[threadIdx.x - remain];
    }
    __syncthreads();
    qz_val = __half2half2(qz_half);

    /* QS QZ done */


    /* GQZ and GQS */
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        // here it is still g_attn
        gqz_val = __hfma2(gqs_val[j], qs_val[j], gqz_val);

        // from now on it is gqs
        gqs_val[j] = __h2div(gqs_val[j], qz_val);
    }

    __half2 gqs_val_swapped[4] = {
        __lowhigh2highlow(gqs_val[0]),
        __lowhigh2highlow(gqs_val[1]),
        __lowhigh2highlow(gqs_val[2]),
        __lowhigh2highlow(gqs_val[3]),
    };

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511

    gqz_half = __hadd(gqz_val.x, gqz_val.y);
    shared_mem_half[threadIdx.x] = gqz_half;
    __syncthreads();
    if (threadIdx.x < NTPH) {
        for (int i = NTPH;i < NTPH * NTPQK; i += NTPH) {
            gqz_half = __hadd(gqz_half, shared_mem_half[threadIdx.x + i]);
        }
    }
    __syncthreads();
    gqz_half = __hdiv(__hneg(gqz_half),
                      __hmul(qz_half, qz_half));
    gqz_val = __half2half2(gqz_half);
    /* GQZ and GQS done done */
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        gq_val[i] = __float2half2_rn(0.f);
        __half2 gq_val_swapped = __float2half2_rn(0.f);
        #pragma unroll
        for (int j = 0;j < 4; ++ j) {
            gq_val[i] = __hfma2(s_val[i][j], gqs_val[j], gq_val[i]);
            gq_val_swapped = __hfma2(s_val_swapped[i][j], gqs_val_swapped[j], gq_val_swapped);
        }
        gq_val[i] = __hadd2(gq_val[i], gq_val_swapped);
    }

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511
    shared_mem_int4[threadIdx.x] = ((int4*) gq_val)[0];
    __syncthreads();
    if (threadIdx.x < NTPH) {
        #pragma unroll
        for (int j = NTPH;j < NTPQK * NTPH; j += NTPH) {
            __half2 tmp[4];
            read_int4(tmp, shared_mem_int4[threadIdx.x + j]);
            #pragma unroll
            for (int i = 0;i < 4; ++ i) {
                gq_val[i] = __hadd2(gq_val[i], tmp[i]);
            }
        }
        //#pragma unroll
        for (int i = 0;i < 4; ++ i) {
            gq_val[i] = __hfma2(z_val[i], gqz_val,  gq_val[i]);
        }
    }
}

__device__
void calculate_gkv(
        const __half2* __restrict__ q_local,
        const __half2* __restrict__ k_local,
        const __half2* __restrict__ v_local,
        const __half   qz_half,
        const __half2* __restrict__ ga_local,
        __half2 gk_val[4],
        __half2 gv_val[4],
        __half  gqz_half,
        __half2 s_val[4][4],
        __half2 s_val_swapped[4][4],
        __half2 t_val[4],
        int4    shared_mem_int4[MAX_THREAD_PER_BLOCK],
        const int NTPQK,
        const int NTPH) {
    __half2 q_val[4], k_val[4], v_val[4], gqs_val[4];
    read_int4(q_val, q_local);
    read_int4(k_val, k_local);
    read_int4(v_val, v_local);
    read_int4(gqs_val, ga_local);

    __half2 qz_val = __half2half2(qz_half);
    __half2 gqz_val = __half2half2(gqz_half);
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        gqs_val[i] = __h2div(gqs_val[i], qz_val);
    }
    __half2 q_val_swapped[4] = {
        __lowhigh2highlow(q_val[0]),
        __lowhigh2highlow(q_val[1]),
        __lowhigh2highlow(q_val[2]),
        __lowhigh2highlow(q_val[3]),
    };

    __half2 k_val_swapped[4] = {
        __lowhigh2highlow(k_val[0]),
        __lowhigh2highlow(k_val[1]),
        __lowhigh2highlow(k_val[2]),
        __lowhigh2highlow(k_val[3]),
    };

    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        gk_val[i] = __float2half2_rn(0.f);
        __half2 gk_val_swapped = __float2half2_rn(0.f);
        t_val[i] = __hfma2(gqz_val, q_val[i], t_val[i]);
        #pragma unroll
        for (int j = 0;j < 4; ++ j) {
            s_val[i][j] = __hfma2(q_val[i], gqs_val[j], s_val[i][j]);
            s_val_swapped[i][j] = __hfma2(q_val_swapped[i], gqs_val[j], s_val_swapped[i][j]);

            gk_val[i] = __hfma2(s_val[i][j], v_val[j], gk_val[i]);
            gk_val_swapped = __hfma2(s_val_swapped[i][j], v_val[j], gk_val_swapped);
        }
        gk_val[i] = __hadd2(gk_val[i], __lowhigh2highlow(gk_val_swapped));
    }
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        gv_val[j] = __float2half2_rn(0.f);
        __half2 gv_val_swapped = __float2half2_rn(0.f);
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            gv_val[j] = __hfma2(s_val[i][j], k_val[i], gv_val[j]);
            gv_val_swapped = __hfma2(s_val_swapped[i][j], k_val_swapped[i], gv_val_swapped);
        }
        gv_val[j] = __hadd2(gv_val[j], gv_val_swapped);
    }

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511

    shared_mem_int4[threadIdx.x] = ((int4*) gk_val)[0];
    __syncthreads();
    if (threadIdx.x < NTPH) {
        #pragma unroll
        for (int j = NTPH;j < NTPH * NTPQK; j += NTPH) {
            __half2 tmp[4];
            read_int4(tmp, shared_mem_int4[threadIdx.x + j]);
            #pragma unroll
            for (int i = 0;i < 4; ++ i) {
                gk_val[i] = __hadd2(gk_val[i], tmp[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        gk_val[i] = __hadd2(gk_val[i], t_val[i]);
    }

    // 128 x 256 case:
    // sum through 256 / 8 = 32 proj_dim threads:
    // thread_idx:
    // 0 + 1 + 2 + ... + 31
    // 32 + 33 + 34 + ... + 63
    // ...
    #pragma unroll
    for (int offset = NTPH / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            gv_val[j] =  __hadd2(gv_val[j], __shfl_down_sync(FULL_MASK, gv_val[j], offset));
        }
    }
}

__device__
void grad_q(
        const __half2 * __restrict__ Q,
        const __half2 * __restrict__ K,
        const __half2 * __restrict__ V,
        __half        * __restrict__ QZ,
        const __half2 * __restrict__ grad_attn,
        __half        * __restrict__ grad_QZ,
        __half2       * __restrict__ grad_Q,
        __half        shared_mem_half[MAX_THREAD_PER_BLOCK],
        int4          shared_mem_int4[MAX_THREAD_PER_BLOCK],
        const int tgt_len,
        int       NTPQK,
        int       NTPH,
        const int QK_inc_t,
        const int V_inc_t) {
    /*
        K:         [tgt_len, proj_dim]
        V:         [tgt_len, head_dim]
        QZ:        [tgt_len].         shared memory
        grad_attn: [tgt_len, head_dim]

    return:
        grad_Q:    [tgt_len, proj_dim]
        grad_QZ:   [tgt_len]. shared memory
    */
    int head_dim_offset = (threadIdx.x / NTPH) << 2;  // should be 3, but 2 due to half2
    int proj_dim_offset = (threadIdx.x % NTPH) << 2;  // should be 4, but 3 due to half2
    const __half2* __restrict__ q_local = Q + proj_dim_offset;
    const __half2* __restrict__ k_local = K + proj_dim_offset;
    const __half2* __restrict__ v_local = V + head_dim_offset;
    const __half2* __restrict__ ga_local = grad_attn + head_dim_offset;

    __half2* __restrict__ gq_local = grad_Q + proj_dim_offset;

    __half2 gq_val[4];
    __half qz_half, gqz_half;

    __half2 s_val[4][4] = {__float2half2_rn(0.f)};
    __half2 s_val_swapped[4][4] = {__float2half2_rn(0.f)};
    __half2 z_val[4] = {__float2half2_rn(0.f)};
    for (int t = 0; t < tgt_len; ++ t) {
        calculate_gq(
            q_local, k_local, v_local,
            qz_half, ga_local,
            gq_val, gqz_half,
            s_val, s_val_swapped,
            z_val,
            shared_mem_half,
            shared_mem_int4,
            NTPQK,
            NTPH
        );
        if (head_dim_offset == 0) {
            ((int4 *) gq_local)[0] = ((int4 *) gq_val)[0];
        }

        if (threadIdx.x == 0) {
            QZ[t] = qz_half;
            grad_QZ[t] = gqz_half;
        }
        __syncthreads();
        q_local += QK_inc_t;
        k_local += QK_inc_t;
        v_local += V_inc_t;
        ga_local += V_inc_t;

        gq_local += QK_inc_t;
    }
}

__device__
void grad_kv(
        const __half2 * __restrict__ Q,
        const __half2 * __restrict__ K,
        const __half2 * __restrict__ V,
        const __half  * __restrict__ QZ,
        const __half2 * __restrict__ grad_attn,
        const __half  * __restrict__ grad_QZ,
        __half2       * __restrict__ grad_K,
        __half2       * __restrict__ grad_V,
        int4          shared_mem_int4[MAX_THREAD_PER_BLOCK],
        const int tgt_len,
        int       NTPQK,
        int       NTPH,
        const int QK_inc_t,
        const int V_inc_t) {
    /*
        Q:         [tgt_len, proj_dim]
        V:         [tgt_len, head_dim]
        QZ:        [tgt_len].         shared memory
        grad_attn: [tgt_len, head_dim]
        grad_QZ:   [tgt_len].         shared memory

    return:
        grad_K:    [tgt_len, proj_dim]
    */
    int head_dim_offset = (threadIdx.x / NTPH) << 2;  // should be 3, but 2 due to half2
    int proj_dim_offset = (threadIdx.x % NTPH) << 2;  // should be 4, but 3 due to half2

    const __half2* __restrict__ q_local = Q + proj_dim_offset;
    const __half2* __restrict__ k_local = K + proj_dim_offset;
    const __half2* __restrict__ v_local = V + head_dim_offset;
    const __half2* __restrict__ ga_local = grad_attn + head_dim_offset;

    __half2* __restrict__ gk_local = grad_K + proj_dim_offset;
    __half2* __restrict__ gv_local = grad_V + head_dim_offset;

    __half2 s_val[4][4] = {__float2half2_rn(0.f)};
    __half2 s_val_swapped[4][4] = {__float2half2_rn(0.f)};
    __half2 t_val[4] = {__float2half2_rn(0.f)};
    __half2 gk_val[4], gv_val[4];
    __half qz_half, gqz_half;

    int offset = tgt_len - 1;
    q_local += QK_inc_t * offset;
    k_local += QK_inc_t * offset;
    v_local += V_inc_t * offset;
    ga_local += V_inc_t * offset;

    gk_local += QK_inc_t * offset;
    gv_local += V_inc_t * offset;
    for (int t = 0; t < tgt_len; ++ t) {
        qz_half = QZ[tgt_len - t - 1];
        gqz_half = grad_QZ[tgt_len - t - 1];
        calculate_gkv(
            q_local, k_local, v_local, qz_half, ga_local,
            gk_val, gv_val, gqz_half,
            s_val, s_val_swapped,
            t_val,
            shared_mem_int4,
            NTPQK,
            NTPH
        );

        if (proj_dim_offset == 0) {
            ((int4 *) gv_local)[0] = ((int4 *) gv_val)[0];
        }
        if (head_dim_offset == 0) {
            ((int4 *) gk_local)[0] = ((int4 *) gk_val)[0];
        }
        __syncthreads();
        q_local -= QK_inc_t;
        k_local -= QK_inc_t;
        v_local -= V_inc_t;
        ga_local -= V_inc_t;

        gk_local -= QK_inc_t;
        gv_local -= V_inc_t;
    }
}

__global__
void grad_qkv(
        const __half * __restrict__ Q,
        const __half * __restrict__ K,
        const __half * __restrict__ V,
        const __half * __restrict__ grad_attn,
        __half * __restrict__ grad_Q,
        __half * __restrict__ grad_K,
        __half * __restrict__ grad_V,
        const int tgt_len,
        const int bsz,
        const int head_dim,
        const int proj_dim,
        const int NTPQK,
        const int NTPH,
        const int stride_QK,
        const int stride_V,
        const int QK_inc_t,
        const int V_inc_t) {
    /*
        Q:         [tgt_len, bsz, proj_dim]
        K:         [tgt_len, bsz, proj_dim]
        V:         [tgt_len, bsz, head_dim]
        grad_attn: [tgt_len, bsz, head_dim]

    return:
        grad_Q:    [tgt_len, bsz, proj_dim]
        grad_K:    [tgt_len, bsz, proj_dim]
        grad_V:    [tgt_len, bsz, head_dim]
    */
    int bid = blockIdx.x;
    const __half2 * __restrict__ Q_local =  (__half2 *) (Q + bid * stride_QK);
    const __half2 * __restrict__ K_local =  (__half2 *) (K + bid * stride_QK);
    const __half2 * __restrict__ V_local =  (__half2 *) (V + bid * stride_V);
    const __half2 * __restrict__ ga_local = (__half2 *) (grad_attn + bid * stride_V);

    __half2 * __restrict__ grad_Q_local = (__half2 *) (grad_Q + bid * stride_QK);
    __half2 * __restrict__ grad_K_local = (__half2 *) (grad_K + bid * stride_QK);
    __half2 * __restrict__ grad_V_local = (__half2 *) (grad_V + bid * stride_V);

    extern __shared__ __half qz_shared[];
    __shared__ __half shared_mem_half[MAX_THREAD_PER_BLOCK];
    __shared__ int4 shared_mem_int4[MAX_THREAD_PER_BLOCK];
    __half * __restrict__ gqz_shared = qz_shared + tgt_len;
    grad_q(
        Q_local,
        K_local,
        V_local,
        qz_shared,
        ga_local,
        gqz_shared,
        grad_Q_local,
        shared_mem_half,
        shared_mem_int4,
        tgt_len,
        NTPQK,
        NTPH,
        QK_inc_t,
        V_inc_t);
    __syncthreads();

    grad_kv(
        Q_local,
        K_local,
        V_local,
        qz_shared,
        ga_local,
        gqz_shared,
        grad_K_local,
        grad_V_local,
        shared_mem_int4,
        tgt_len,
        NTPQK,
        NTPH,
        QK_inc_t,
        V_inc_t);
}



std::vector<Tensor> RFABackward(
        Tensor const& Q,
        Tensor const& K,
        Tensor const& V,
        Tensor const& grad_attn) {
    /*
    Args:
        Q:         [tgt_len, bsz, proj_dim]
        K:         [tgt_len, bsz, proj_dim]
        V:         [tgt_len, bsz, head_dim]
        grad_attn: [tgt_len, bsz, head_dim]

    Return:
        grad_Q:     [tgt_len, bsz, proj_dim]
        grad_K:     [tgt_len, bsz, proj_dim]
        grad_V:     [tgt_len, bsz, head_dim]
    */
    // column major
    const int tgt_len = K.size(0);
    const int bsz = K.size(1);
    const int proj_dim = K.size(2);
    const int head_dim = V.size(2);
    const int stride_QK = proj_dim;
    const int stride_V = head_dim;
    const int QK_inc_t = bsz * proj_dim >> 1;
    const int V_inc_t = bsz * head_dim >> 1;

    auto act_options  = Q.options().requires_grad(false);
    Tensor grad_Q = torch::zeros({tgt_len, bsz, proj_dim}, act_options);
    Tensor grad_K = torch::zeros({tgt_len, bsz, proj_dim}, act_options);
    Tensor grad_V = torch::zeros({tgt_len, bsz, head_dim}, act_options);

    const int block_size = proj_dim / 8 * head_dim / 8;
    const int NTPQK = head_dim / 8;
    const int NTPH = proj_dim / 8;
    // grad_q: 4 threads per proj_dim
    // 2 blocks per batch
    dim3 dim_grid(bsz);
    dim3 dim_block(block_size);
    grad_qkv <<<dim_grid, dim_block, 2 * sizeof(real) * tgt_len>>>(
            static_cast<const real *> (Q.data_ptr()),
            static_cast<const real *> (K.data_ptr()),
            static_cast<const real *> (V.data_ptr()),
            static_cast<const real *> (grad_attn.data_ptr()),
            static_cast<real *> (grad_Q.data_ptr()),
            static_cast<real *> (grad_K.data_ptr()),
            static_cast<real *> (grad_V.data_ptr()),
            tgt_len,
            bsz,
            head_dim,
            proj_dim,
            NTPQK,
            NTPH,
            stride_QK,
            stride_V,
            QK_inc_t,
            V_inc_t
    );

    return {grad_Q, grad_K, grad_V};
}

