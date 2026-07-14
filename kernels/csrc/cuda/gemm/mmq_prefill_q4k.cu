// Weight-amortized Q4_K x Q8_1 dp4a GEMM for long-context prompt prefill ("MMQ").
//
// Token-by-token prefill re-reads every projection weight once per prompt token, so a
// 128k prompt streams the whole 5.6 GB weight set 131072 times and pins the GPU at ~91%
// of DRAM bandwidth (~228 tok/s, flat in context). This kernel amortizes that read: one
// warp owns an output row, loads each Q4_K superblock ONCE into registers, and dp4a's it
// against M_TILE token rows of pre-quantized Q8_1 activation. Weight traffic drops M_TILE x.
//
// Weights are never dequantized -- they stay Q4_K in DRAM and 4-bit in registers, and the
// arithmetic is the same vec_dot_q4_K_q8_1 the decode GEMV runs (gemv_q4k_dp4a_pq_kernel),
// with the same per-token sub-block accumulation order and no cross-token reduction.
// Measured vs the token path: identical at M_TILE=1; for M_TILE>1 the wider unroll lets nvcc
// contract the FMAs differently, so ~0.013% of outputs land 1 bf16 ULP apart (maxabs 1.0 on
// values ~256). That is far inside the prefill correctness gate (top1 >= 0.90, KL <= 0.20).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "sparkinfer/kernels/mmq_prefill.h"

namespace sparkinfer {
namespace kernels {

namespace {

__device__ __forceinline__ float mq_h2f(const unsigned char* p) {
    __half h;
    *((unsigned short*)&h) = *(const unsigned short*)p;
    return __half2float(h);
}

// Q4_K 6-bit packed scale/min for sub-block j (llama.cpp layout, 12 scale bytes at blk+4).
__device__ __forceinline__ void mq_scale_min(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

constexpr int MQ_WPB = 4;    // warps per block, all sharing the staged activation tile
constexpr int MQ_CSB = 32;   // K-chunk in 32-element sub-blocks (= 1024 elements)
constexpr int N_REG = 4;     // output rows per warp (register-blocked); 144 regs, measured best

// q8/ad/as hold the whole activation tile: q8 is [M, K] int8, ad/as are [M, K/32].
// W is [N, K/256] Q4_K superblocks of 144 B. y is [M, N] row-major.
//
// Two axes of reuse, both required:
//   M_TILE  - each weight superblock is loaded once and dp4a'd against M_TILE token rows,
//             which is what removes the per-token weight reload (the whole point).
//   N_REG   - each warp owns N_REG output rows, so each activation load feeds N_REG dp4a.
//             With N_REG=1 the inner loop issues one smem load per dp4a and the kernel sits
//             at ~3% of dp4a peak no matter how large M is (measured: a flat 2.2x).
// The activation tile is staged in shared memory per K-chunk so the MQ_WPB warps read it once.
//
// Lane L accumulates sub-blocks L, L+32, L+64, ... exactly as gemv_q4k_dp4a_pq_kernel does
// (chunking K by MQ_CSB=32 preserves that stride), so the per-token summation order is kept.
template <int M_TILE, typename OutT>
__global__ __launch_bounds__(MQ_WPB * 32) void mmq_q4k_kernel(
    const signed char* __restrict__ q8, const float* __restrict__ ad,
    const float* __restrict__ as, const unsigned char* __restrict__ W,
    OutT* __restrict__ y, int M, int N, int K) {
    __shared__ signed char s_q8[M_TILE][MQ_CSB * 32];
    __shared__ float s_ad[M_TILE][MQ_CSB];
    __shared__ float s_as[M_TILE][MQ_CSB];

    const int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * MQ_WPB + warpId) * N_REG;   // first of this warp's N_REG rows
    const int m0 = blockIdx.y * M_TILE;
    const int nsb = K >> 5;
    const int nrows = min(M_TILE, M - m0);
    const size_t rowbytes = (size_t)(K >> 8) * 144;
    const unsigned char* base = W + (size_t)min(n0, N - 1) * rowbytes;

    float acc[N_REG][M_TILE];
#pragma unroll
    for (int r = 0; r < N_REG; r++)
#pragma unroll
        for (int i = 0; i < M_TILE; i++) acc[r][i] = 0.f;

    for (int c0 = 0; c0 < nsb; c0 += MQ_CSB) {
        const int csb = min(MQ_CSB, nsb - c0);
        // Stage all M_TILE rows (zero-filling any tail past nrows) so the compute loop below
        // can run to the compile-time bound M_TILE.
        __syncthreads();
        for (int t = threadIdx.x; t < M_TILE * ((csb * 32) >> 4); t += blockDim.x) {
            const int mm = t / ((csb * 32) >> 4), col = t % ((csb * 32) >> 4);
            int4 v = make_int4(0, 0, 0, 0);
            if (mm < nrows) {
                v = reinterpret_cast<const int4*>(
                    q8 + (size_t)(m0 + mm) * (size_t)K + ((size_t)c0 << 5))[col];
            }
            reinterpret_cast<int4*>(&s_q8[mm][0])[col] = v;
        }
        for (int t = threadIdx.x; t < M_TILE * csb; t += blockDim.x) {
            const int mm = t / csb, sb = t % csb;
            const bool ok = mm < nrows;
            s_ad[mm][sb] = ok ? ad[(size_t)(m0 + mm) * (size_t)nsb + c0 + sb] : 0.f;
            s_as[mm][sb] = ok ? as[(size_t)(m0 + mm) * (size_t)nsb + c0 + sb] : 0.f;
        }
        __syncthreads();

        for (int sl = lane; sl < csb; sl += 32) {
            const int sb = c0 + sl;
            const int super = sb >> 3, sib = sb & 7;
            const bool hi = sib & 1;

            // N_REG weight rows are read once here and each is reused for every token in the
            // tile; conversely each activation read below feeds N_REG rows. Without this second
            // axis of reuse the inner loop issues one smem load per dp4a and stalls at ~3% of
            // dp4a peak regardless of M.
            // dscd/dmscm depend only on the weight row and sub-block, never on the token, so
            // they are hoisted out of the mm loop: Q4_K's per-sub-block rescale is ~5 float
            // ops per 8 dp4a, and leaving the two products inside cost N_REG*M_TILE fmuls
            // per sub-block instead of N_REG.
            float dscd[N_REG], dmscm[N_REG];
            int w[N_REG][8];
#pragma unroll
            for (int r = 0; r < N_REG; r++) {
                const unsigned char* blk = base + (size_t)r * rowbytes + (size_t)super * 144;
                int scd, scm;
                mq_scale_min(sib, blk + 4, &scd, &scm);
                dscd[r] = mq_h2f(blk) * (float)scd;
                dmscm[r] = mq_h2f(blk + 2) * (float)scm;
                const int* q = reinterpret_cast<const int*>(blk + 16 + (sib >> 1) * 32);
#pragma unroll
                for (int k = 0; k < 8; k++)
                    w[r][k] = hi ? ((q[k] >> 4) & 0x0F0F0F0F) : (q[k] & 0x0F0F0F0F);
            }

            // MUST run to the compile-time bound M_TILE, not to the runtime `nrows`: a runtime
            // trip count leaves acc[r][mm] dynamically indexed, which parks the whole
            // accumulator array in local memory (ptxas reports it as a stack frame, NOT as a
            // spill) and makes every accumulate a local load+store. Tail rows are zeroed in
            // shared memory above and masked at the store, so the wasted lanes are harmless.
#pragma unroll
            for (int mm = 0; mm < M_TILE; mm++) {
                const int* aint = reinterpret_cast<const int*>(&s_q8[mm][sl << 5]);
                int a[8];
#pragma unroll
                for (int k = 0; k < 8; k++) a[k] = aint[k];
                const float xd = s_ad[mm][sl], xs = s_as[mm][sl];
#pragma unroll
                for (int r = 0; r < N_REG; r++) {
                    int sumi = 0;
#pragma unroll
                    for (int k = 0; k < 8; k++) sumi = __dp4a(w[r][k], a[k], sumi);
                    acc[r][mm] = fmaf(dscd[r] * xd, (float)sumi, acc[r][mm] - dmscm[r] * xs);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < N_REG; r++) {
        if (n0 + r >= N) break;
#pragma unroll
        for (int mm = 0; mm < M_TILE; mm++) {
            if (mm >= nrows) break;
            float a = acc[r][mm];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0) y[(size_t)(m0 + mm) * (size_t)N + n0 + r] = (OutT)a;
        }
    }
}

// Per-token Q8_1 quantization of an [M, K] bf16 activation tile. Each 32-element sub-block
// is scaled by its own absmax, exactly as quantize_q8_1_kernel does for a single token, so
// the emitted int8/scales are identical to what the token path would produce.
__global__ void mmq_quant_q8_1_kernel(const __nv_bfloat16* __restrict__ x,
                                      signed char* __restrict__ q8, float* __restrict__ ad,
                                      float* __restrict__ as, int M, int K) {
    const int nsb = K >> 5;
    const int lane = threadIdx.x & 31;
    const long total = (long)M * nsb;
    const int nwarp = (blockDim.x >> 5) * gridDim.x;
    for (long idx = (long)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)); idx < total;
         idx += nwarp) {
        const long off = idx * 32 + lane;
        float xv = __bfloat162float(x[off]);
        float a = fabsf(xv);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
        const float d = a / 127.0f;
        const int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);
        q8[off] = (signed char)qi;
        int sm = qi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, m);
        if (lane == 0) {
            ad[idx] = d;
            as[idx] = d * (float)sm;
        }
    }
}

}  // namespace

void launch_mmq_quant_q8_1(const void* x, void* q8, float* ad, float* as, int M, int K,
                           cudaStream_t stream) {
    const int blocks = min(1024, (M * (K >> 5) + 3) / 4);
    mmq_quant_q8_1_kernel<<<max(blocks, 1), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<signed char*>(q8), ad, as, M, K);
}

void launch_mmq_q4k(const void* q8, const float* ad, const float* as, const void* W, void* y,
                    int M, int N, int K, cudaStream_t stream) {
    const signed char* q = reinterpret_cast<const signed char*>(q8);
    const unsigned char* w = reinterpret_cast<const unsigned char*>(W);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(y);
    const int rows_per_blk = MQ_WPB * N_REG;
    const int gx = (N + rows_per_blk - 1) / rows_per_blk;
    // M_TILE trades register pressure (acc is N_REG x M_TILE) for weight reuse.
    if (M >= 16) {
        mmq_q4k_kernel<16, __nv_bfloat16><<<dim3(gx, (M + 15) / 16), MQ_WPB * 32, 0, stream>>>(
            q, ad, as, w, out, M, N, K);
    } else if (M >= 8) {
        mmq_q4k_kernel<8, __nv_bfloat16><<<dim3(gx, (M + 7) / 8), MQ_WPB * 32, 0, stream>>>(
            q, ad, as, w, out, M, N, K);
    } else {
        mmq_q4k_kernel<4, __nv_bfloat16><<<dim3(gx, (M + 3) / 4), MQ_WPB * 32, 0, stream>>>(
            q, ad, as, w, out, M, N, K);
    }
}

}  // namespace kernels
}  // namespace sparkinfer
