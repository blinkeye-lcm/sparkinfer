#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Pre-packed-weight variant of the int8 prefill GEMM (prefill_i8.h).
//
// The row-major kernel reads each mma B operand as two scalar 4B smem loads at an XOR-swizzled
// address, and that swizzle only separates rows 0..3 -- rows 4 apart still collide 2-way. Prefill
// weights are constant, so the runtime can hold them resident and pay one reshuffle at cache-build
// time; afterwards the 8 bytes a lane feeds to mma.sync.m16n8k32 are contiguous, so the B path
// becomes one conflict-free 8B load per fragment with no address math.
//
// Results are bit-identical to launch_prefill_gemm_i8: the int8 products accumulate in int32 and
// cannot overflow at these shapes (|sum| <= 127*127*12288 ~ 1.98e8 < 2^31), so integer accumulation
// is exact and independent of order, and the dequant epilogue is unchanged. Callers that cannot pack
// (unsupported shape, no spare VRAM) keep using the row-major kernel.

namespace sparkinfer { namespace kernels {

// Packing needs N a multiple of 8 (the mma B operand's n extent) and K a multiple of 64 (two k32
// blocks, the kernel's BK step -- the activation pack below stages a whole BK tile per mtile).
bool prefill_pack_weight_i8_supported(int N, int K);

// ---------------------------------------------------------------------------
// Packed activation.
//
// The weight pack above fixes the mma *B* operand; the *A* operand is still four swizzled scalar 4B
// smem loads per fragment. Activations change every pass, so they cannot be held resident -- but they
// are already rewritten once per pass by the row quantize, and that pass is free to emit fragment
// order instead of row-major at no extra cost. The GEMM's four LDS.32 then become one conflict-free
// LDS.128 with no address math, taking the inner loop from 16 loads per 16 mmas to 10.
//
// Bit-identical to launch_prefill_quantize_rows_i8: the same symmetric per-row scheme (d = amax/127,
// q = round(x/d)), and amax is an fmaxf reduction, which is order-independent -- so re-parallelizing
// it cannot change a byte.
// ---------------------------------------------------------------------------

// Bytes launch_prefill_quantize_pack_a_i8 writes for an [M,K] activation (M padded to the 16-row
// mma tile).
size_t prefill_packed_activation_bytes(int M, int K);

// Quantize x_bf16[M,K] to int8 in mma.sync.m16n8k32 A-operand order -> Ap, per-row scales -> scale.
// Rows past M in the final tile are zero-filled, so the GEMM needs no row predicate.
void launch_prefill_quantize_pack_a_i8(const void* x_bf16, signed char* Ap, float* scale,
                                       int rows, int cols, cudaStream_t stream = nullptr);

// Bytes launch_prefill_pack_weight_i8 writes for a [N,K] int8 weight.
size_t prefill_packed_weight_bytes(int N, int K);

// Reshuffle a row-major int8 weight [N,K] into tensor-core fragment order. One-time, per weight.
void launch_prefill_pack_weight_i8(const signed char* W, signed char* Wp,
                                   int N, int K, cudaStream_t stream = nullptr);

// C[M,N] = A[M,K] @ W^T with Wp produced by launch_prefill_pack_weight_i8. A, sx, sw, C as in
// launch_prefill_gemm_i8.
void launch_prefill_gemm_i8_packed(const signed char* A, const signed char* Wp,
                                   const float* sx, const float* sw, void* C,
                                   int M, int N, int K, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Fused SwiGLU FFN variant.
//
// The dense FFN runs gate and up as two [n_half,K] projections over the same activation, then a
// separate elementwise pass reads both back to form silu(gate)*up. Packing lets the two weights be
// interleaved row-wise -- dst row 2j = gate row j, dst row 2j+1 = up row j -- into one [2*n_half,K]
// weight. mma.sync.m16n8k32's accumulator then lands columns n and n+1 in the *same lane's* c0/c1
// (the epilogue's column index n0 + wn*64 + j*8 + tig*2 is always even), so gate_j and up_j meet in
// registers and SwiGLU collapses into the epilogue: one GEMM, no elementwise pass, and the two
// full-size gate/up writes become one.
//
// Bit-identical to gemm_i8_packed + launch_prefill_swiglu: both operands are rounded to bf16 before
// the silu, exactly as materializing them to memory would.
// ---------------------------------------------------------------------------

// The fused GEMM tiles 128 weight rows (= 64 outputs) per block, so the interleaved weight needs
// 2*n_half a multiple of 128, on top of the plain packing constraints.
bool prefill_gate_up_fusion_supported(int n_half, int K);

// Pack one parity of the interleaved gate/up weight: parity 0 takes W = gate into the even dst rows,
// parity 1 takes W = up into the odd ones. Two calls with the same Wp build the whole weight, so the
// row-major staging buffer only ever has to hold one [n_half,K] weight at a time.
void launch_prefill_pack_gate_up_i8(const signed char* W, signed char* Wp,
                                    int parity, int n_half, int K, cudaStream_t stream = nullptr);

// Scatter one parity's row scales to match: dst[2*j + parity] = s[j].
void launch_prefill_interleave_scales(const float* s, float* dst,
                                      int parity, int n_half, cudaStream_t stream = nullptr);

// C[M,n_half] = silu(A[M,K] @ Wgate^T) * (A[M,K] @ Wup^T), with Wp the interleaved packed weight.
// sw holds the interleaved row scales (2*n_half of them); sx, A as in launch_prefill_gemm_i8.
void launch_prefill_gemm_i8_packed_swiglu(const signed char* A, const signed char* Wp,
                                          const float* sx, const float* sw, void* C,
                                          int M, int n_half, int K, cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
