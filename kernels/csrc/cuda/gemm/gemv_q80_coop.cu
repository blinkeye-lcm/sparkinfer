// Coalesced-staging Q8_0 x Q8_1 mmvq for the batch-1 decode GEMVs (attn/GDN projections).
//
// The default si_mmvq_q80_kfixed reads each 34-byte Q8_0 weight block (fp16 scale + 32 int8)
// with per-lane, stride-34, 2-byte-aligned narrow loads: every load instruction scatters the
// warp across 32 distinct sectors 34 B apart -> uncoalesced global traffic + heavy L1 sector
// pressure on the #1 decode hotspot (~32% of decode). This kernel instead has the block
// cooperatively stage its whole weight row into shared memory with fully coalesced 128-bit
// (uint4) loads, then runs the byte-for-byte identical dp4a dot from smem. The arithmetic,
// accumulation order, and reduction are unchanged, so the output is BIT-IDENTICAL to the
// default kernel (zero accuracy change) — only the global memory access pattern differs.
// Selected at runtime by SPARKINFER_Q80_COOP (default on) and only when the weight row base is
// 16-byte aligned (row stride NBLOCKS*34 is a multiple of 16, so a 16-aligned tensor base
// keeps every row aligned); otherwise the caller falls back to the default kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {
namespace {

struct cq8_1 { __half2 ds; signed char qs[32]; };   // == si_block_q8_1 (llama Q8_1 activation)

__device__ __forceinline__ float cq80_h2f(const unsigned char* p) {
    __half h; *reinterpret_cast<unsigned short*>(&h) = *reinterpret_cast<const unsigned short*>(p);
    return __half2float(h);
}
__device__ __forceinline__ int cq80_i32(const unsigned char* p, int i) {
    const unsigned short* u = reinterpret_cast<const unsigned short*>(p);
    return (int)u[2 * i] | ((int)u[2 * i + 1] << 16);
}
__device__ __forceinline__ float cq80_dot(const unsigned char* bw, const cq8_1* ba) {
    const float dw = cq80_h2f(bw);
    const int* a = reinterpret_cast<const int*>(ba->qs);
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) sumi = __dp4a(cq80_i32(bw + 2, i), a[i], sumi);
    return dw * __low2float(ba->ds) * (float)sumi;
}
__device__ __forceinline__ void cq80_write(float* p, float v) { *p = v; }
__device__ __forceinline__ void cq80_write(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename OutT, int NBLOCKS>
__global__ void si_mmvq_q80_coop_kernel(const cq8_1* __restrict__ vy,
                                        const unsigned char* __restrict__ W, OutT* __restrict__ y, int N) {
    constexpr int NW = 4, WS = 32;
    constexpr int ROWB = NBLOCKS * 34;     // bytes/row; 2176 (K=2048) or 4352 (K=4096), both %16==0
    constexpr int ROW16 = ROWB / 16;       // uint4 count/row
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row = blockIdx.x;
    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)row * ROWB);

    extern __shared__ uint4 s_w4[];        // ROWB bytes of the weight row
    for (int i = tid; i < ROW16; i += NW * WS) s_w4[i] = w_row[i];   // coalesced 128-bit loads
    __syncthreads();

    const unsigned char* s_w = reinterpret_cast<const unsigned char*>(s_w4);
    float tmp = 0.0f;
    #pragma unroll
    for (int kb = tid; kb < NBLOCKS; kb += NW * WS) tmp += cq80_dot(s_w + (size_t)kb * 34, vy + kb);

    __shared__ float tmp_shared[NW - 1][WS];
    if (warp > 0) tmp_shared[warp - 1][lane] = tmp;
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) tmp += tmp_shared[l][lane];
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, m);
    if (lane == 0) cq80_write(y + row, tmp);
}

} // anonymous namespace

// Dispatch helper called from gemv.cu's launch_mmvq_q80[_f32] when SPARKINFER_Q80_COOP is on and
// the weight base is 16-byte aligned. K is 2048 or 4096 (the two kfixed instantiations).
void launch_mmvq_q80_coop(const void* q81, const void* W, void* y, int N, int K, bool f32, cudaStream_t stream) {
    const cq8_1* q = reinterpret_cast<const cq8_1*>(q81);
    const unsigned char* w = reinterpret_cast<const unsigned char*>(W);
    const int smem = (K == 2048 ? 64 : 128) * 34;   // ROWB bytes of dynamic smem
    if (!f32) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(y);
        if (K == 2048) si_mmvq_q80_coop_kernel<__nv_bfloat16, 64><<<N, 4 * 32, smem, stream>>>(q, w, out, N);
        else           si_mmvq_q80_coop_kernel<__nv_bfloat16, 128><<<N, 4 * 32, smem, stream>>>(q, w, out, N);
    } else {
        float* out = reinterpret_cast<float*>(y);
        if (K == 2048) si_mmvq_q80_coop_kernel<float, 64><<<N, 4 * 32, smem, stream>>>(q, w, out, N);
        else           si_mmvq_q80_coop_kernel<float, 128><<<N, 4 * 32, smem, stream>>>(q, w, out, N);
    }
}

} // namespace kernels
} // namespace sparkinfer
