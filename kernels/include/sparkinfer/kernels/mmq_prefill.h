// Weight-amortized Q4_K x Q8_1 GEMM for long-context prompt prefill.
//
// The token path (gemv_q4k_dp4a_pq / mmvq_q4k) reads every weight once PER TOKEN, which is
// what pins a 128k prefill at DRAM bandwidth. These entry points take a tile of M token rows
// and read each weight once per TILE instead, with the same dp4a arithmetic and the same
// per-token accumulation order (so the result matches the token path bit for bit).

#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Quantize an [M, K] bf16 activation tile to Q8_1 (per 32-element sub-block absmax scaling).
//   x  : [M, K] bf16
//   q8 : [M, K] int8              ad/as: [M, K/32] float (scale, scale*sum)
// Emits exactly what quantize_q8_1_kernel would emit for each row independently.
void launch_mmq_quant_q8_1(const void* x, void* q8, float* ad, float* as, int M, int K,
                           cudaStream_t stream = nullptr);

// y[M, N] = Q8_1(x)[M, K] . Q4_K(W)[N, K]^T
//   q8/ad/as : output of launch_mmq_quant_q8_1
//   W        : [N, K/256] Q4_K superblocks (144 B each), the GGUF-native layout
//   y        : [M, N] bf16, row-major
void launch_mmq_q4k(const void* q8, const float* ad, const float* as, const void* W, void* y,
                    int M, int N, int K, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
