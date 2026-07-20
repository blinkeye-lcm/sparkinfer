// int8 prefill GEMM over a pre-packed weight (see prefill_i8_packed.h).
//
// Same math as prefill_gemm_i8.cu -- C[M,N] = A[M,K] @ W^T, int8 x int8 -> int32 with the
// sx[m]*sw[n] dequant folded into the store -- but the weight is staged in tensor-core fragment
// order instead of [N,K] row-major.
//
// The row-major kernel pays for that layout twice in its inner loop: each mma B operand is two
// scalar 4B smem loads at a swizzled address, and the XOR swizzle only spreads rows 0..3 (rows 4
// apart still collide 2-way). Because the prefill weights are constant, the runtime can hold them
// resident and pay the reshuffle once at cache-build time, so a lane's mma operand becomes a single
// contiguous load with no address math.
//
// Packing (N%16==0, K%64==0). For column group cg = n/8, k-block k32 = k/32 and lane = 0..31 with
// grp = lane>>2 (the mma B operand's n index) and tig = lane&3, one lane's 8 operand bytes are
//   [0..3] = W[cg*8+grp][k32*32 +      tig*4 .. +4]
//   [4..7] = W[cg*8+grp][k32*32 + 16 + tig*4 .. +4]
// and column groups are stored in INTERLEAVED PAIRS (pk_bdst): cg and cg+1 land adjacent for the
// same lane, so one 16B load feeds two mma B operands. With the activation packed the same way
// (prefill_i8_packed.h), the inner loop issues 6 loads per 16 mma -- 2 for A, 4 for B -- all
// conflict-free, where the row-major kernel issues 32.
//
// Bit-identical to the row-major kernel: int8 products accumulate in int32 and never overflow
// (|sum| <= 127*127*12288 ~ 1.98e8 < 2^31), so integer accumulation is exact and order-independent;
// the epilogue is unchanged. Every pack is a pure permutation of the same bytes.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include "sparkinfer/kernels/prefill_i8_packed.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int PK_BM = 128;
constexpr int PK_BN = 128;
constexpr int PK_BK = 64;          // 2 k32 blocks per stage
constexpr int PK_MFRAG = 2;        // 32 rows per warp / 16
constexpr int PK_NFRAG = 8;        // 64 cols per warp / 8
constexpr int PK_CGP = PK_BN / 16; // 8 column-group PAIRS per block tile (see the pack note)
constexpr int PK_CGB = 1024;       // bytes per pair per BK tile: 2 k32 * 32 lanes * 16B
constexpr int PK_MT = PK_BM / 16;  // 8 mma row-tiles per block tile
constexpr int PK_MTB = 1024;       // bytes per row-tile per BK tile: 2 k32 * 32 lanes * 16B
constexpr int PK_QP_THREADS = 512; // quantize+pack block: 16 warps, one per row of the mma tile
// cp.async depth. 2 stages = 32 KB, which keeps 2 blocks/SM at 124 registers. A 3-stage variant
// measured +0.4% (inside run-to-run noise) while needing exactly 49152 B of static smem -- the
// per-block limit, with no headroom left for anything else -- so the extra prefetch depth is not
// worth the fragility.
constexpr int PK_STAGES = 2;
// Row-blocks per rasterization group. Blocks issue in blockIdx.x-major order, so the natural
// (m,n) = (blockIdx.y, blockIdx.x) mapping puts every block resident at one time on a *different*
// weight column block, and a wave's weight working set becomes the whole [N,K] weight. Walking
// PK_GROUP row-blocks down M before stepping along N bounds it to ~wave/PK_GROUP column blocks.
// This is worth +3% on the interleaved gate|up GEMM, whose 100 MB weight is the one that does not
// fit L2 -- without it the SwiGLU fusion below is a net *loss*, its saved elementwise pass paid
// straight back in re-streamed weight. Neutral on the smaller projections, which fit either way, so
// it is applied uniformly rather than special-cased.
//
// 8 is measured, not derived: swept 1/2/4/8/16/32 at ctx=4096, and only the step off 1 matters --
// past 8 the sweep is flat. The tempting extrapolation ("taller stripes -> fewer passes over the
// weight -> keep growing it") does not hold.
constexpr int PK_GROUP = 8;

// Stripe height for a shape: PK_GROUP, but never more row-blocks than exist.
// SPARKINFER_PREFILL_PK_GROUP overrides it (A/B); 1 gives the natural rasterization.
int pk_group(int M) {
    static const int env_g = []() {
        const char* e = getenv("SPARKINFER_PREFILL_PK_GROUP");
        return e ? atoi(e) : 0;
    }();
    const int nblk_m = (M + PK_BM - 1) / PK_BM;
    const int g = env_g > 0 ? env_g : PK_GROUP;
    return g > nblk_m ? (nblk_m < 1 ? 1 : nblk_m) : (g < 1 ? 1 : g);
}

// Where column group cg's fragment for (k32, lane) lives. Pairs of column groups are interleaved:
// a lane's 8 bytes for cg and its 8 for cg+1 sit adjacent, so the GEMM fetches both mma B operands
// in ONE 16B load instead of two 8B ones. Pair p = cg>>1 occupies 512 B per k32 (32 lanes x 16 B).
__device__ __forceinline__ signed char* pk_bdst(signed char* Wp, int cg, int k32, int lane,
                                                int nk32) {
    return Wp + ((size_t)((cg >> 1) * nk32 + k32) * 32 + lane) * 16 + (cg & 1) * 8;
}

__device__ __forceinline__ void pk_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}
// Must match pf_silu/pf_to_f in batched_prefill.cu -- the fused epilogue reproduces
// pf_swiglu_kernel exactly, including the bf16 rounding of both operands before the silu.
__device__ __forceinline__ float pk_silu(float x) { return x / (1.f + __expf(-x)); }

// Map this block to its (row, col) tile in `g`-row-block-tall stripes (see PK_GROUP). Pure
// scheduling: every block still computes exactly one tile, so results are unchanged. g == 1 is
// exactly the natural (blockIdx.y, blockIdx.x) mapping.
__device__ __forceinline__ void pk_tile(int nblk_n, int nblk_m, int g, int& m_blk, int& n_blk) {
    const int bid       = blockIdx.y * nblk_n + blockIdx.x;
    const int per_group = g * nblk_n;
    const int gid       = bid / per_group;
    const int first_m   = gid * g;
    const int rows      = min(nblk_m - first_m, g);          // last group may be short
    const int idx       = bid - gid * per_group;
    m_blk = first_m + idx % rows;
    n_blk = idx / rows;
}

// One block per (column group, k32 block); 32 threads = the 32 lanes of the target fragment.
__global__ void pk_pack_weight_kernel(const signed char* __restrict__ W, signed char* __restrict__ Wp,
                                      int N, int K) {
    const int nk32 = K >> 5;
    const int cg   = blockIdx.x;
    const int k32  = blockIdx.y;
    const int lane = threadIdx.x;
    const int grp = lane >> 2, tig = lane & 3;
    const int n = cg * 8 + grp;
    signed char* dst = pk_bdst(Wp, cg, k32, lane, nk32);
    if (n >= N) { *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u); return; }
    const signed char* src = W + (size_t)n * K + k32 * 32;
    uint2 v;
    v.x = *reinterpret_cast<const unsigned*>(src + tig * 4);
    v.y = *reinterpret_cast<const unsigned*>(src + 16 + tig * 4);
    *reinterpret_cast<uint2*>(dst) = v;
}

// Interleaved gate/up pack: dst row 2j = gate row j, dst row 2j+1 = up row j. `parity` selects which
// half this call fills, so the caller only needs a staging buffer for one [n_half,K] weight. Within a
// fragment the dst row is cg*8 + grp, and cg*8 is even, so a lane's parity is just grp&1 -- the two
// calls write disjoint lanes.
__global__ void pk_pack_gate_up_kernel(const signed char* __restrict__ W, signed char* __restrict__ Wp,
                                       int parity, int n_half, int K) {
    const int nk32 = K >> 5;
    const int cg   = blockIdx.x;
    const int k32  = blockIdx.y;
    const int lane = threadIdx.x;
    const int grp = lane >> 2, tig = lane & 3;
    if ((grp & 1) != parity) return;
    const int n_src = (cg * 8 + grp) >> 1;
    signed char* dst = pk_bdst(Wp, cg, k32, lane, nk32);
    if (n_src >= n_half) { *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u); return; }
    const signed char* src = W + (size_t)n_src * K + k32 * 32;
    uint2 v;
    v.x = *reinterpret_cast<const unsigned*>(src + tig * 4);
    v.y = *reinterpret_cast<const unsigned*>(src + 16 + tig * 4);
    *reinterpret_cast<uint2*>(dst) = v;
}

__global__ void pk_interleave_scales_kernel(const float* __restrict__ s, float* __restrict__ dst,
                                            int parity, int n_half) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_half) dst[2 * j + parity] = s[j];
}

// Fused per-row int8 quantize + mma A-operand pack. One block per 16-row mma tile: warp w reduces
// row w's amax, then the whole block writes the tile in fragment order. Same scheme as the row-major
// quantize (d = amax/127, q = round(x/d), zero row -> zero), and amax is an fmaxf reduction, so it is
// order-independent and every byte matches regardless of how the rows are split across threads.
__global__ __launch_bounds__(PK_QP_THREADS) void pk_quant_pack_a_kernel(
        const __nv_bfloat16* __restrict__ x, signed char* __restrict__ Ap,
        float* __restrict__ scale, int rows, int cols) {
    __shared__ float sd[16];
    const int mt = blockIdx.x, tid = threadIdx.x;
    const int w = tid >> 5, lane = tid & 31;

    if (w < 16) {
        const int r = mt * 16 + w;
        float amax = 0.f;
        if (r < rows) {
            // 8 bf16 per lane per step: K is a multiple of 64, so the row base is 16B-aligned.
            const uint4* xv = reinterpret_cast<const uint4*>(x + (size_t)r * cols);
            for (int v = lane; v < (cols >> 3); v += 32) {
                const uint4 raw = xv[v];
                const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
                #pragma unroll
                for (int e = 0; e < 8; e++) amax = fmaxf(amax, fabsf(__bfloat162float(h[e])));
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        if (lane == 0) {
            const float d = amax / 127.0f;
            sd[w] = d;
            if (r < rows) scale[r] = d;
        }
    }
    __syncthreads();

    // One 16B fragment per (k32, lane): the 4 chunks a lane feeds to mma as {a0,a1,a2,a3} are rows
    // grp / grp+8 at k-offsets tig*4 and 16+tig*4 -- exactly the operands the GEMM reads back.
    const int nk32 = cols >> 5;
    for (int u = tid; u < nk32 * 32; u += PK_QP_THREADS) {
        const int k32 = u >> 5, l = u & 31;
        const int grp = l >> 2, tig = l & 3;
        signed char out[16];
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            const int rr = grp + (p & 1) * 8;
            const int cc = k32 * 32 + (p >> 1) * 16 + tig * 4;
            const int gr = mt * 16 + rr;
            const float d = sd[rr];
            // cc is a multiple of 4, so the 4 bf16 this chunk needs are one aligned 8B load.
            uint2 raw = make_uint2(0u, 0u);
            if (gr < rows) raw = *reinterpret_cast<const uint2*>(x + (size_t)gr * cols + cc);
            const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
            #pragma unroll
            for (int e = 0; e < 4; e++)
                out[p * 4 + e] = (signed char)((d == 0.f) ? 0 : (int)roundf(__bfloat162float(h[e]) / d));
        }
        *reinterpret_cast<uint4*>(Ap + ((size_t)mt * nk32 + k32) * 512 + l * 16) =
            *reinterpret_cast<const uint4*>(out);
    }
}

// SWIGLU=false: C[M,N] = dequant(A @ Wp^T), the plain packed GEMM.
// SWIGLU=true:  Wp is the interleaved gate/up weight, so a lane's c0/c1 hold gate_j/up_j for the same
//               output column j = gn>>1; the epilogue folds SwiGLU in and writes C[M,N/2].
template <bool SWIGLU>
__global__ __launch_bounds__(256, 2) void pk_gemm_i8_packed_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ Wp,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* __restrict__ C, int M, int N, int K, int g) {
    __shared__ signed char As[PK_STAGES][PK_MT][PK_MTB];
    __shared__ signed char Bs[PK_STAGES][PK_CGP][PK_CGB];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;
    const int tig  = lane & 3;
    const int wm   = warp & 3;
    const int wn   = warp >> 2;
    int m_blk, n_blk;
    pk_tile(gridDim.x, gridDim.y, g, m_blk, n_blk);
    const int m0   = m_blk * PK_BM;
    const int n0   = n_blk * PK_BN;
    const int cgp0 = n0 >> 4;          // first column-group pair of this block tile
    const int mt0  = m_blk * PK_MT;
    const int mtiles = (M + 15) >> 4;
    const int nk32 = K >> 5;
    const int nk   = (K + PK_BK - 1) / PK_BK;

    int acc[PK_MFRAG][PK_NFRAG][4];
    #pragma unroll
    for (int i = 0; i < PK_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < PK_NFRAG; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

    // Both operands are already in fragment order, so each stage is two straight contiguous copies
    // with no swizzle: A is 8 row-tiles x 1024B (the tile's two k32 blocks are adjacent by
    // construction), B is 16 column groups x 512B.
    auto stage = [&](int buf, int k0) {
        const int k320 = k0 >> 5;
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int mtl = s >> 6, aoff = (s & 63) << 4;
            const int gmt = mt0 + mtl;
            pk_cp16(&As[buf][mtl][aoff],
                    A + ((size_t)gmt * nk32 + k320) * 512 + aoff,
                    gmt < mtiles);

            const int cgpl = s >> 6, boff = (s & 63) << 4;
            const int gcgp = cgp0 + cgpl;
            pk_cp16(&Bs[buf][cgpl][boff],
                    Wp + ((size_t)gcgp * nk32 + k320) * 512 + boff,
                    gcgp * 16 < N && k0 < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PK_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < PK_BK; kk += 32) {
            const int k32l = kk >> 5;
            uint4 af[PK_MFRAG];
            unsigned bf[PK_NFRAG][2];
            // One 16B load per fragment: {a0,a1,a2,a3} are this lane's four operand registers, laid
            // out adjacently by the pack. Lanes read consecutive 16B, so it is conflict-free.
            #pragma unroll
            for (int i = 0; i < PK_MFRAG; i++)
                af[i] = *reinterpret_cast<const uint4*>(
                    &As[buf][wm * 2 + i][k32l * 512 + lane * 16]);
            // One 16B load per column-group PAIR: the lane's four operand registers for fragments
            // 2*jp and 2*jp+1 are adjacent by construction, so B costs 4 loads per k-step, not 8.
            #pragma unroll
            for (int jp = 0; jp < PK_NFRAG / 2; jp++) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    &Bs[buf][wn * 4 + jp][k32l * 512 + lane * 16]);
                bf[2 * jp][0]     = v.x; bf[2 * jp][1]     = v.y;
                bf[2 * jp + 1][0] = v.z; bf[2 * jp + 1][1] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < PK_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < PK_NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i].x), "r"(af[i].y), "r"(af[i].z), "r"(af[i].w),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        __syncthreads();
        buf ^= 1;
    }

    // Fused SwiGLU epilogue. gn is always even (n0 is a multiple of PK_BN, plus wn*64 + j*8 + tig*2),
    // so with the interleaved weight this lane's c0 is gate column gn>>1 and c1 is up column gn>>1 --
    // the pair meets in registers and never goes to memory. Rounding both to bf16 *before* the silu is
    // what keeps this bit-identical: the unfused path materializes ffg/ffu as bf16, and pf_swiglu
    // reads them back. prefill_gate_up_fusion_supported() guarantees the pair is in range.
    if constexpr (SWIGLU) {
        const int Nout = N >> 1;
        #pragma unroll
        for (int i = 0; i < PK_MFRAG; i++) {
            #pragma unroll
            for (int j = 0; j < PK_NFRAG; j++) {
                const int gn = n0 + wn * 64 + j * 8 + tig * 2;
                const float w0 = sw[gn], w1 = sw[gn + 1];
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                    if (gm >= M) continue;
                    const float s = sx[gm];
                    const float g = __bfloat162float(__float2bfloat16((float)acc[i][j][h * 2]     * s * w0));
                    const float u = __bfloat162float(__float2bfloat16((float)acc[i][j][h * 2 + 1] * s * w1));
                    C[(size_t)gm * Nout + (gn >> 1)] = __float2bfloat16(pk_silu(g) * u);
                }
            }
        }
        return;
    }

    // Identical epilogue to the row-major kernel.
    #pragma unroll
    for (int i = 0; i < PK_MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < PK_NFRAG; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            if (gn + 1 >= N) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N)
                        C[(size_t)gm * N + cn] = __float2bfloat16((float)acc[i][j][e] * sx[gm] * sw[cn]);
                }
                continue;
            }
            const float w0 = sw[gn], w1 = sw[gn + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                if (gm >= M) continue;
                const float s = sx[gm];
                const __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                               (float)acc[i][j][h * 2 + 1] * s * w1);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]) = v;
            }
        }
    }
}
} // namespace

bool prefill_pack_weight_i8_supported(int N, int K) {
    // N % 16: column groups are packed in interleaved pairs (see pk_bdst), so N must tile whole
    // pairs. K % 64: the packed path stages a whole BK=64 tile (both k32 blocks of a row-tile) per
    // copy, so a partial trailing tile has no packed representation. Every Qwythos projection
    // satisfies both; anything else keeps the row-major kernel.
    return (N & 15) == 0 && (K & 63) == 0;
}

size_t prefill_packed_activation_bytes(int M, int K) {
    return (size_t)((M + 15) / 16) * 16 * (size_t)K;
}

void launch_prefill_quantize_pack_a_i8(const void* x_bf16, signed char* Ap, float* scale,
                                       int rows, int cols, cudaStream_t stream) {
    pk_quant_pack_a_kernel<<<(rows + 15) / 16, PK_QP_THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), Ap, scale, rows, cols);
}

size_t prefill_packed_weight_bytes(int N, int K) {
    return (size_t)((N + 7) / 8) * 8 * (size_t)K;
}

void launch_prefill_pack_weight_i8(const signed char* W, signed char* Wp,
                                   int N, int K, cudaStream_t stream) {
    dim3 grid((N + 7) / 8, K >> 5);
    pk_pack_weight_kernel<<<grid, 32, 0, stream>>>(W, Wp, N, K);
}

void launch_prefill_gemm_i8_packed(const signed char* A, const signed char* Wp,
                                   const float* sx, const float* sw, void* C,
                                   int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + PK_BN - 1) / PK_BN, (M + PK_BM - 1) / PK_BM);
    pk_gemm_i8_packed_kernel<false><<<grid, 256, 0, stream>>>(
        A, Wp, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K, pk_group(M));
}

bool prefill_gate_up_fusion_supported(int n_half, int K) {
    // The fused epilogue reads the (gn, gn+1) pair unconditionally, so the interleaved weight must
    // tile PK_BN exactly -- no partial column block, no padded rows.
    return prefill_pack_weight_i8_supported(2 * n_half, K) && ((2 * n_half) % PK_BN) == 0;
}

void launch_prefill_pack_gate_up_i8(const signed char* W, signed char* Wp,
                                    int parity, int n_half, int K, cudaStream_t stream) {
    dim3 grid((2 * n_half + 7) / 8, K >> 5);
    pk_pack_gate_up_kernel<<<grid, 32, 0, stream>>>(W, Wp, parity, n_half, K);
}

void launch_prefill_interleave_scales(const float* s, float* dst,
                                      int parity, int n_half, cudaStream_t stream) {
    pk_interleave_scales_kernel<<<(n_half + 255) / 256, 256, 0, stream>>>(s, dst, parity, n_half);
}

void launch_prefill_gemm_i8_packed_swiglu(const signed char* A, const signed char* Wp,
                                          const float* sx, const float* sw, void* C,
                                          int M, int n_half, int K, cudaStream_t stream) {
    const int N = 2 * n_half;
    dim3 grid(N / PK_BN, (M + PK_BM - 1) / PK_BM);
    pk_gemm_i8_packed_kernel<true><<<grid, 256, 0, stream>>>(
        A, Wp, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K, pk_group(M));
}

}} // namespace sparkinfer::kernels
