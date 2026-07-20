// Batched prompt prefill for the Qwen3.5 dense-hybrid (Qwythos) model.
//
// forward_token ingests a prompt one token at a time, so every prompt token pays a full
// bandwidth-bound weight reload for each projection (a GEMV). prefill_batched_run() instead runs
// the whole prompt through the layer stack in one pass: the weight-bound Q/K/V/O + dense-SwiGLU-FFN
// projections become tensor-core (cp.async, wmma) GEMMs, the Gated-DeltaNet recurrence runs as a
// single sequential scan over all N tokens, and the full-attention layers fill the paged int8 KV
// cache in the exact layout the decode path reads. It fills the same KV cache and recurrent/conv
// state a forward_token loop would, so a subsequent decode is numerically faithful.
//
// This is its own translation unit — it reaches nothing but the explicit Qwen35PrefillCtx, so it
// shares no code with the decode path (qwen35.cpp keeps Impl private).

#include "qwen35_prefill.h"
#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/prefill_i8.h"
#include "sparkinfer/kernels/prefill_i8_packed.h"
#include "sparkinfer/kernels/prefill_moe.h"
#include "sparkinfer/kernels/moe.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

namespace {
using bf16 = unsigned short;
inline void pf_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[prefill] %s: %s\n", what, cudaGetErrorString(e));
}
// ---------------------------------------------------------------------------
// Resident weight cache.
//
// GGUF weights are constant for the model's lifetime, but proj() re-derives the int8 form of every
// weight (and the bf16 dequant of the tiny gate projections) on *every* prefill call: 200 weight
// conversions per 4k prefill, ~24 ms of the ~248 ms wall. prefill_i8.h already specifies the
// intended behaviour -- "the per-output-row weight scales are computed once and kept resident, the
// per-token activation scales are computed per prefill pass" -- so this makes the resident half of
// that contract true. The activation quant stays per-pass, as it must.
//
// The cached bytes are exactly what the per-call path would have produced (same kernels, same
// inputs), so every GEMM sees bit-identical operands. Entries are keyed by weight pointer and built
// on first use; a build failure caches a null so it is not retried, and the caller falls back to the
// per-call scratch path. Budget-capped, and disabled with SPARKINFER_PREFILL_WCACHE=0.
// ---------------------------------------------------------------------------
// q: resident int8 weight -- in tensor-core fragment order when `packed`, else [n_out,K] row-major.
struct CachedI8 { signed char* q; float* s; bool packed; };

std::unordered_map<const void*, CachedI8> g_wc_i8;   // weight -> resident int8 rows + row scales
std::unordered_map<const void*, CachedI8> g_wc_gu;   // gate weight -> resident interleaved gate|up
size_t      g_wc_bytes = 0;
const void* g_wc_model = nullptr;   // model tag; a different model drops the cache

void wcache_reset() {
    for (auto& kv : g_wc_i8) { cudaFree(kv.second.q); cudaFree(kv.second.s); }
    for (auto& kv : g_wc_gu) { cudaFree(kv.second.q); cudaFree(kv.second.s); }
    g_wc_i8.clear();
    g_wc_gu.clear();
    g_wc_bytes = 0;
}

// Cache budget. Qwythos's projection weights are ~7.1 GB as int8; the default clears that with room
// to spare, and the free-VRAM headroom below is what actually keeps the device safe.
size_t wcache_budget() {
    static size_t mb = []() -> size_t {
        const char* e = getenv("SPARKINFER_PREFILL_WCACHE_MB");
        return e ? (size_t)atol(e) : 10240;
    }();
    return mb << 20;
}
bool wcache_enabled() {
    static bool on = []() {
        const char* e = getenv("SPARKINFER_PREFILL_WCACHE");
        return !(e && e[0] == '0');
    }();
    return on;
}
// Only take the allocation if it fits the budget and leaves the device comfortable headroom -- the
// scratch arena for a 64k context is >10 GB, and starving it would drop the whole prefill back to
// the per-token loop, which is far slower than any dequant this cache removes. The scratch is
// always allocated before the cache is consulted, so it wins the race; this headroom protects the
// *next* context's scratch, and prefill_batched_run drops the cache outright if it still collides.
bool wcache_can_alloc(size_t bytes) {
    if (g_wc_bytes + bytes > wcache_budget()) return false;
    size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return false;
    return bytes + (3ull << 30) < freeb;
}
// Device-buffer arena, persistent across calls.
//
// The scratch is ~1 GB at a 4k context and its buffer sizes are a pure function of the context, so
// allocating and freeing all ~22 of it per prefill is repeated work -- and the big ones are not
// cheap (a 100 MB cudaMalloc runs into the hundreds of microseconds). Buffers are handed out in a
// fixed order, so on a repeat call at the same context every request matches the buffer already
// held and the whole arena is reused. A request whose size does not match drops that buffer and
// everything after it and reallocates, which is what makes a context change correct: the first
// buffer is sized from N, so a new context mismatches immediately and rebuilds the arena.
//
// Nothing here is read before it is written in a pass, so a reused buffer's stale contents cannot
// leak into a result.
struct Arena {
    struct Buf { void* p; size_t bytes; };
    std::vector<Buf> bufs;
    size_t idx = 0;
    bool ok = true;
    template <class T> T* alloc(size_t n) {
        if (n == 0) n = 1;
        const size_t bytes = n * sizeof(T);
        if (idx < bufs.size()) {
            if (bufs[idx].bytes == bytes) return static_cast<T*>(bufs[idx++].p);
            for (size_t i = idx; i < bufs.size(); i++) cudaFree(bufs[i].p);   // shape changed
            bufs.resize(idx);
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        bufs.push_back({p, bytes});
        idx++;
        return static_cast<T*>(p);
    }
    // Start a pass: hand out from the first buffer again, and clear any failure the last pass hit
    // (the arena outlives the call now, so `ok` must not carry over).
    void rewind() { idx = 0; ok = true; }
    size_t bytes() const {
        size_t t = 0;
        for (const auto& b : bufs) t += b.bytes;
        return t;
    }
    void free_all() { for (auto& b : bufs) cudaFree(b.p); bufs.clear(); idx = 0; }
};

// Hold the arena between prefills only while it is this small. The trade is context-dependent and
// lopsided: a 4k prefill's scratch is ~1 GB and its alloc/free is ~5% of a ~175 ms wall, well worth
// keeping; a 64k prefill's scratch is ~14 GB and the same alloc is under 0.5% of a multi-second
// wall, so there the memory is worth far more left free -- for the weight cache, the KV cache, and
// the decode that follows. Above the cap the arena is released at the end of the pass, exactly as
// before this change.
constexpr size_t PF_ARENA_KEEP_BYTES = 2ull << 30;

// Row-major int8 form of a native GGUF weight [n_out,K] -> dst, row scales -> s. Exactly the kernels
// the per-call path runs, in the same order, so the bytes are identical to what it would produce.
// `bfs` is staging for weight types the fused GGUF->int8 path does not cover.
void build_i8_rows(const void* W, int wtype, signed char* dst, float* s, int n_out, int K,
                   bf16* bfs, cudaStream_t st) {
    if (kernels::launch_gguf_dequant_rows_i8(wtype, W, dst, s, n_out, K, st)) return;
    const void* wb = W;
    if (wtype != 0) {
        kernels::launch_gguf_dequant(wtype, W, bfs, (long)n_out * K, st);
        wb = bfs;
    }
    kernels::launch_prefill_quantize_rows_i8(wb, dst, s, n_out, K, st);
}

// Finish a cache entry: commit its bytes, or free it and leave it null if the build failed.
void wcache_commit(CachedI8& c, size_t bytes, cudaStream_t st) {
    if (cudaStreamSynchronize(st) == cudaSuccess) {
        g_wc_bytes += bytes;
    } else {
        cudaFree(c.q); cudaFree(c.s);
        c.q = nullptr; c.s = nullptr;
    }
}

// Reserve a cache entry's device buffers. False (and c left null) if it does not fit.
bool wcache_alloc(CachedI8& c, size_t qbytes, size_t sbytes) {
    if (wcache_can_alloc(qbytes + sbytes) &&
        cudaMalloc((void**)&c.q, qbytes) == cudaSuccess &&
        cudaMalloc((void**)&c.s, sbytes) == cudaSuccess) return true;
    cudaFree(c.q);              // no-op on null; the scale alloc may have been the one that failed
    c.q = nullptr; c.s = nullptr;
    return false;
}

// Resident int8 form of weight W, built on first use. Returns null if uncacheable (the caller then
// converts into per-call scratch exactly as before). `bfs`/`i8s` are the shared per-call staging
// buffers: bfs for weight types the fused GGUF->int8 path does not cover, i8s to hold the row-major
// int8 while it is reshuffled into the resident fragment-order copy.
const CachedI8* wcache_i8_get(const void* W, int wtype, int n_out, int K, bf16* bfs,
                              signed char* i8s, cudaStream_t st) {
    auto it = g_wc_i8.find(W);
    if (it != g_wc_i8.end()) return it->second.q ? &it->second : nullptr;

    const bool pack = i8s && kernels::prefill_pack_weight_i8_supported(n_out, K);
    const size_t qbytes = pack ? kernels::prefill_packed_weight_bytes(n_out, K)
                               : (size_t)n_out * K;
    const size_t sbytes = (size_t)n_out * sizeof(float);
    CachedI8 c{nullptr, nullptr, pack};
    if (wcache_alloc(c, qbytes, sbytes)) {
        // The pack is a pure reshuffle on top of the same row-major bytes.
        signed char* dst = pack ? i8s : c.q;
        build_i8_rows(W, wtype, dst, c.s, n_out, K, bfs, st);
        if (pack) kernels::launch_prefill_pack_weight_i8(dst, c.q, n_out, K, st);
        wcache_commit(c, qbytes + sbytes, st);
    }
    auto ins = g_wc_i8.emplace(W, c);   // cache the failure too, so it is not retried every call
    return ins.first->second.q ? &ins.first->second : nullptr;
}

// Resident interleaved gate|up weight (dst row 2j = gate row j, 2j+1 = up row j) in fragment order,
// so one GEMM can produce silu(gate)*up with the pair meeting in registers. Keyed by the gate
// pointer. Built one parity at a time, which keeps the row-major staging at one weight's worth.
// Returns null if it does not fit; the caller then runs the two separate projections as before.
const CachedI8* wcache_gate_up_get(const void* Wg, int gtype, const void* Wu, int utype,
                                   int n_half, int K, bf16* bfs, signed char* i8s, float* ss,
                                   cudaStream_t st) {
    auto it = g_wc_gu.find(Wg);
    if (it != g_wc_gu.end()) return it->second.q ? &it->second : nullptr;

    const size_t qbytes = kernels::prefill_packed_weight_bytes(2 * n_half, K);
    const size_t sbytes = (size_t)2 * n_half * sizeof(float);
    CachedI8 c{nullptr, nullptr, true};
    if (wcache_alloc(c, qbytes, sbytes)) {
        const void* W[2] = {Wg, Wu};
        const int   t[2] = {gtype, utype};
        for (int p = 0; p < 2; p++) {
            build_i8_rows(W[p], t[p], i8s, ss, n_half, K, bfs, st);
            kernels::launch_prefill_pack_gate_up_i8(i8s, c.q, p, n_half, K, st);
            kernels::launch_prefill_interleave_scales(ss, c.s, p, n_half, st);
        }
        wcache_commit(c, qbytes + sbytes, st);
    }
    auto ins = g_wc_gu.emplace(Wg, c);
    return ins.first->second.q ? &ins.first->second : nullptr;
}

} // namespace

int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n) {
    const Qwen35Config& c = s.cfg;
    // Batched prefill supports the Qwen3.5 dense-hybrid (Qwythos) AND the Qwen3.6-35B-A3B MoE hybrid.
    // Both share the GDN + full-attention batched kernels (identical math at 128/16/32 GDN dims and
    // 256/64 attn dims); they differ ONLY in the FFN, branched below (dense SwiGLU vs the expert-
    // grouped int8 MoE path). The MoE path is specialized for 256 experts with a top-k router.
    const bool moe = !c.dense_ffn && c.n_experts > 0;
    if (!s.gguf || !c.hybrid || n <= 0) return -1;
    if (!c.dense_ffn && !moe) return -1;
    if (c.head_dim != 256 || c.linear_head_dim != 128) return -1;   // kernels specialize these
    if (moe && (c.n_experts != 256 || c.top_k <= 0)) return -1;     // grouped top-k path specialized for 256
    if (moe)
        for (int L = 0; L < c.n_layers; L++) {
            const Qwen35LayerWeights& w = s.w.layers[L];
            // grouped expert GEMMs need quantized experts (Q4_K/Q5_K/Q6_K rows-int8 dequant) + a router
            if (!w.gate_q || !w.up_q || !w.down_q || !w.router_w) {
                fprintf(stderr, "[prefill-moe] layer %d missing expert/router tensors -> token loop\n", L);
                return -1;
            }
            auto qok = [](int t) { return t == 12 || t == 13 || t == 14; };
            if (!qok(w.gate_qtype) || !qok(w.up_qtype) || !qok(w.down_qtype)) {
                fprintf(stderr, "[prefill-moe] layer %d expert qtypes %d/%d/%d unsupported -> token loop\n",
                        L, w.gate_qtype, w.up_qtype, w.down_qtype);
                return -1;
            }
        }

    const int H = c.hidden;
    const int N = n;
    cudaStream_t st = s.stream;

    const int qdim = s.qdim, kvdim = s.kvdim;            // full-attn: 4096 / 1024
    const int lqkv = s.linear_qkvdim;                    // 8192
    const int lvdim = s.linear_vdim;                     // 4096
    const int vh   = c.linear_v_heads;                   // 32
    const int ffn  = c.moe_ffn;                          // dense: 12288; MoE: per-expert 512
    const int wide = 2 * qdim;                           // 8192 (qraw); also >= lqkv
    // wbuf must hold the largest weight the `dq` lambda dequantizes: the dense FFN (ffn*H) OR, on the
    // MoE path (small ffn=512), the biggest projection (wide/lqkv * H). Cover all of them.
    size_t maxw = (size_t)wide * H;
    if ((size_t)lqkv * H > maxw) maxw = (size_t)lqkv * H;
    if (!moe && (size_t)ffn * H > maxw) maxw = (size_t)ffn * H;
    // int8 proj scratch dims: largest projection input K (A rows) and output n_out (channel scales).
    // On MoE the small per-expert ffn (512) is NOT the max, so size against the real projections.
    auto imax = [](int x, int y) { return x > y ? x : y; };
    const int maxAK = moe ? imax(qdim, lvdim) : imax(ffn, imax(qdim, lvdim));   // max proj input dim
    const int maxNO = moe ? imax(wide, lqkv) : imax(ffn, imax(wide, lqkv));     // max proj output dim
    // Dense FFN is processed in token-chunks so its ffn-wide scratch (ffg/ffu/A_i8) stays O(chunk)
    // instead of O(N) — at long context those full-width buffers dominate and OOM (~8 GB @128k). The
    // FFN is per-token independent, so chunking is numerically identical. Env override; default 32768.
    // (MoE doesn't use ffg/ffu — its grouped FFN has its own O(N*top_k) scratch, so chunking is moot.)
    const int ffn_chunk = []{ const char* e = getenv("SPARKINFER_PREFILL_FFN_CHUNK"); int c = e ? atoi(e) : 32768; return c > 0 ? c : 32768; }();
    const int FC = (N < ffn_chunk) ? N : ffn_chunk;
    bf16* lin_conv_state = static_cast<bf16*>(s.lin_conv_state);

    // ---- scratch ----
    static Arena a;                    // persistent across calls; see Arena
    a.rewind();
    bf16* x    = a.alloc<bf16>((size_t)N * H);
    bf16* xn   = a.alloc<bf16>((size_t)N * H);
    bf16* hn   = a.alloc<bf16>((size_t)N * H);
    bf16* ao   = a.alloc<bf16>((size_t)N * H);
    bf16* b8   = a.alloc<bf16>((size_t)N * wide);        // qraw / lin_qkv (8192)
    bf16* lz   = a.alloc<bf16>((size_t)N * lvdim);       // lin_z (4096)
    bf16* gq   = a.alloc<bf16>((size_t)N * s.linear_qdim);   // gdn q (2048)
    bf16* gk   = a.alloc<bf16>((size_t)N * s.linear_qdim);   // gdn k (2048)
    bf16* gv   = a.alloc<bf16>((size_t)N * lvdim);       // gdn v (4096)
    bf16* att  = a.alloc<bf16>((size_t)N * lvdim);       // attn out / gdn_out (4096)
    bf16* lnrm = a.alloc<bf16>((size_t)N * lvdim);       // lin_norm (4096)
    bf16* la   = a.alloc<bf16>((size_t)N * vh);          // lin_alpha (32)
    bf16* lb   = a.alloc<bf16>((size_t)N * vh);          // lin_beta (32)
    // Full-attention scratch ALIASES the GDN scratch: a layer is either linear-attn (GDN) or full
    // softmax-attn, never both, and qb/qg/kf/vf are pairwise-distinct within a full-attn layer while
    // the GDN buffers they map onto are unused there (and vice-versa). Saves ~10K bf16/token of peak
    // scratch at long context (each is <= its GDN host: qdim/kvdim <= lvdim/linear_qdim).
    bf16* qb   = gv;                                     // full q      (4096) <- gdn v    (4096)
    bf16* qg   = lnrm;                                   // full q-gate (4096) <- lin_norm (4096)
    bf16* kf   = gq;                                     // full k      (1024) <- gdn q    (2048)
    bf16* vf   = gk;                                     // full v      (1024) <- gdn k    (2048)
    bf16* ffg  = a.alloc<bf16>((size_t)FC * ffn);        // ffn gate (12288), bounded to FC tokens
    bf16* ffu  = a.alloc<bf16>((size_t)FC * ffn);        // ffn up,          bounded to FC tokens
    bf16* ffh  = ffg;                                    // SwiGLU computed in-place into ffg (down reads it)
    bf16* wbuf = a.alloc<bf16>(maxw);                    // dequantized-weight scratch (reused)
    int*  d_ids = a.alloc<int>((size_t)N);
    if (!a.ok) { a.free_all(); fprintf(stderr, "[prefill] scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }
    // int8 tensor-core projections (prefill_gemm_i8): ~2x the bf16 GEMM at int8==bf16 output fidelity
    // (GGUF weights are already Q4_K/Q6_K -> int8 weight-quant is lossless vs what's stored). Default
    // ON at every batched context; SPARKINFER_PREFILL_I8=0 disables (A/B). The int8 scratch lives in
    // its own arena so an alloc failure at huge N degrades to the bf16 GEMMs, not to the token loop.
    const char* _pi8 = getenv("SPARKINFER_PREFILL_I8");
    // Dense: int8 projections default ON. MoE: default OFF — the discrete top-k router amplifies the
    // per-token int8 projection error into different expert selections, which diverges from the
    // token-by-token path far more than in the dense FFN; bf16 projections keep the batched MoE
    // prefill faithful to the decode path. SPARKINFER_PREFILL_I8 overrides either way.
    bool use_i8 = _pi8 ? (_pi8[0] != '0') : !moe;
    // Long-context fidelity (dense): the near-1-decay GDN recurrence amplifies the per-row int8
    // activation-quant error across the sequence, so int8 prefill diverges from the token-by-token
    // path past ~96k (128k: top1 0.31 / KL 0.18). Above bf16_minctx (default 96k) fall back to bf16
    // projections, which stay faithful (128k: top1 0.69 / KL 0.04) at ~half the prefill throughput.
    // SPARKINFER_PREFILL_BF16_MINCTX overrides the threshold.
    static int bf16_minctx = []{ const char* e = getenv("SPARKINFER_PREFILL_BF16_MINCTX"); return e ? atoi(e) : 98304; }();
    if (N > bf16_minctx) use_i8 = false;
    static Arena a8;                   // persistent across calls; see Arena
    a8.rewind();
    // A_i8 holds the quantized activation. Dense non-FFN projs quantize N rows x K(<=H); the chunked
    // FFN quantizes at most FC rows x ffn. The packed path emits mma A-operand order, which rounds the
    // row count up to the 16-row mma tile, so size against that padded form. MoE: N rows x maxAK.
    size_t a_i8_sz;
    if (moe) a_i8_sz = (size_t)N * maxAK;
    else {
        const size_t t1 = kernels::prefill_packed_activation_bytes(N, H);
        const size_t t2 = kernels::prefill_packed_activation_bytes(FC, ffn);
        a_i8_sz = t1 > t2 ? t1 : t2;
    }
    signed char* A_i8 = use_i8 ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    signed char* W_i8 = use_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = use_i8 ? a8.alloc<float>((size_t)N) : nullptr;
    float* sw = use_i8 ? a8.alloc<float>((size_t)maxNO) : nullptr;
    if (use_i8 && !a8.ok) { a8.free_all(); use_i8 = false; }

    // Weights are constant, so their converted form is cached across calls (see wcache_* above). A
    // different model in the same process drops the cache (weight pointers are unique per model).
    // Dense-path only -- the MoE FFN dequantizes its 256 experts in one grouped launch.
    const bool wcache = wcache_enabled();
    if (g_wc_model != s.w.embed_tokens) { wcache_reset(); g_wc_model = s.w.embed_tokens; }

    // ---- MoE (Qwen3.6) scratch: expert-int8 weights + pair bucketing + pair-major hidden ----
    // The expert-grouped GEMMs run int8 tensor-core UNCONDITIONALLY (that is the speedup), so this
    // block carries its own int8 activation scratch (mA_i8/msx) and does not depend on the shared
    // `use_i8` flag, which upstream defaults OFF for MoE (it governs only the bf16-vs-int8 choice of
    // the attention/GDN/shared projections routed through `proj`). Full-N shared-expert buffers
    // (sfg/sfu/sfh) are dedicated here because the outer ffg/ffu are FC-chunked (dense path only).
    const int E = moe ? c.n_experts : 0, topk = moe ? c.top_k : 0, mffn = moe ? c.moe_ffn : 0;
    const int P = moe ? N * topk : 0;                          // routed (token, expert) pairs
    const int max_tiles = moe ? (P + 127) / 128 + E : 0;       // per-expert 128-row GEMM tiles
    Arena am;
    signed char *Wg_i8 = nullptr, *Wu_i8 = nullptr, *Wd_i8 = nullptr, *h_i8 = nullptr, *mA_i8 = nullptr;
    float *swg = nullptr, *swu = nullptr, *swd = nullptr, *sh = nullptr, *msx = nullptr;
    float *mlogits = nullptr, *mweights = nullptr, *pair_w = nullptr, *routed_f32 = nullptr, *dw = nullptr;
    int *mids = nullptr, *mcounts = nullptr, *moffsets = nullptr, *mcursors = nullptr;
    int *pair_tok = nullptr, *tilemap = nullptr, *d_ntiles = nullptr;
    bf16 *hg = nullptr, *hu = nullptr, *hh = nullptr, *sfg = nullptr, *sfu = nullptr, *sfh = nullptr;
    if (moe) {
        Wg_i8 = am.alloc<signed char>((size_t)E * mffn * H);
        Wu_i8 = am.alloc<signed char>((size_t)E * mffn * H);
        Wd_i8 = am.alloc<signed char>((size_t)E * H * mffn);
        swg = am.alloc<float>((size_t)E * mffn);
        swu = am.alloc<float>((size_t)E * mffn);
        swd = am.alloc<float>((size_t)E * H);
        mlogits = am.alloc<float>((size_t)N * E);
        mids = am.alloc<int>((size_t)P);
        mweights = am.alloc<float>((size_t)P);
        mcounts = am.alloc<int>(E);
        moffsets = am.alloc<int>(E + 1);
        mcursors = am.alloc<int>(E);
        pair_tok = am.alloc<int>((size_t)P);
        pair_w = am.alloc<float>((size_t)P);
        tilemap = am.alloc<int>((size_t)2 * max_tiles);
        d_ntiles = am.alloc<int>(1);
        hg = am.alloc<bf16>((size_t)P * mffn);
        hu = am.alloc<bf16>((size_t)P * mffn);
        hh = am.alloc<bf16>((size_t)P * mffn);
        h_i8 = am.alloc<signed char>((size_t)P * mffn);
        sh = am.alloc<float>((size_t)P);
        routed_f32 = am.alloc<float>((size_t)N * H);
        dw = am.alloc<float>((size_t)N);
        mA_i8 = am.alloc<signed char>((size_t)N * H);          // int8 activation for the grouped GEMMs
        msx = am.alloc<float>((size_t)N);
        sfg = am.alloc<bf16>((size_t)N * mffn);                // shared-expert gate/up/hidden (full N)
        sfu = am.alloc<bf16>((size_t)N * mffn);
        sfh = am.alloc<bf16>((size_t)N * mffn);
        if (!am.ok) {
            a.free_all(); a8.free_all(); am.free_all();
            fprintf(stderr, "[prefill] MoE scratch alloc failed (ctx=%d) -> fallback\n", N);
            return -1;
        }
    }

    pf_cu(cudaMemcpyAsync(d_ids, prompt_ids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "prefill ids");

    // Dequantize a native GGUF weight [n_out,K] to bf16 scratch; return a bf16 [n_out,K] ptr.
    auto dq = [&](const void* W, int wtype, int n_out, int K) -> const void* {
        if (wtype == 0) return W;   // already bf16 dense
        kernels::launch_gguf_dequant(wtype, W, wbuf, (long)n_out * K, st);
        return wbuf;
    };
    // C[N,n_out] = A[N,K] @ W^T  (W native quantized [n_out,K]).
    auto proj = [&](const bf16* A, const void* W, int wtype, bf16* C, int n_out, int K, int rows = 0) {
        const int R = rows > 0 ? rows : N;   // rows (M) to process; chunked FFN passes a sub-N count
        // int8 only for the big weight-bound projections; keep the tiny per-v-head gate
        // projections (ssm_alpha/ssm_beta, n_out == v_heads) in bf16 — they feed the GDN
        // sigmoid gates, where per-row int8 quant of a 32-wide weight costs more accuracy
        // than the negligible time it saves.
        if (use_i8 && n_out >= 128) {
            const CachedI8* cc = wcache ? wcache_i8_get(W, wtype, n_out, K, wbuf, W_i8, st) : nullptr;
            if (cc && cc->packed) {
                // Resident fragment-order weight -> packed activation: both mma operands reach the
                // tensor core as one contiguous load each (see prefill_i8_packed.h).
                kernels::launch_prefill_quantize_pack_a_i8(A, A_i8, sx, R, K, st);
                kernels::launch_prefill_gemm_i8_packed(A_i8, cc->q, sx, cc->s, C, R, n_out, K, st);
                return;
            }
            kernels::launch_prefill_quantize_rows_i8(A, A_i8, sx, R, K, st);
            const signed char* Wq = W_i8;
            const float*       Ws = sw;
            if (cc) { Wq = cc->q; Ws = cc->s; }
            // fused Q4_K/Q6_K -> int8 rows skips the dequant-to-bf16 scratch round trip
            else if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
                const void* wb = dq(W, wtype, n_out, K);
                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
            }
            kernels::launch_prefill_gemm_i8(A_i8, Wq, sx, Ws, C, R, n_out, K, st);
        } else {
            // mma.sync bf16 GEMM only for dense-hybrid long prefill (the >96k int8→bf16 fallback).
            // MoE always stays on wmma: its top-k router turns tiny GEMM differences into expert
            // flips that fail the Qwen3.6 accuracy gate.
            const bool prefer_mma = !moe && R > bf16_minctx;
            kernels::launch_prefill_gemm(A, dq(W, wtype, n_out, K), C, R, n_out, K, st, prefer_mma);
        }
    };

    // H_out[R,ffn] = silu(A @ Wgate^T) * (A @ Wup^T) in one GEMM off the resident interleaved
    // gate|up weight: gate_j and up_j land in the same lane's accumulator, so the elementwise SwiGLU
    // pass and one of the two full-size gate/up writes both disappear. Bit-identical (both operands
    // rounded to bf16 before the silu). False when unavailable -> caller runs gate/up/swiglu.
    const bool gu_fuse = use_i8 && wcache && !moe && kernels::prefill_gate_up_fusion_supported(ffn, H);
    auto ffn_fused = [&](const Qwen35LayerWeights& w, const bf16* A, bf16* H_out, int rows) -> bool {
        if (!gu_fuse) return false;
        const CachedI8* cc = wcache_gate_up_get(w.gate_q, w.gate_qtype, w.up_q, w.up_qtype,
                                                ffn, H, wbuf, W_i8, sw, st);
        if (!cc) return false;
        kernels::launch_prefill_quantize_pack_a_i8(A, A_i8, sx, rows, H, st);
        kernels::launch_prefill_gemm_i8_packed_swiglu(A_i8, cc->q, sx, cc->s, H_out, rows, ffn, H, st);
        return true;
    };

    const int* btable = s.kv->block_table(s.seq_id);
    const int  bs = s.kv->block_size();
    const int  mbs = s.kv->max_blocks_per_seq();
    const bool kv8 = s.kv->int8_kv();
    const int  kv_elem = kv8 ? 1 : 2;
    const float rope_theta = c.rope_theta, eps = c.rms_eps;
    const int rope_dim = (c.rope_dim > 0) ? c.rope_dim : c.head_dim;
    const float attn_scale = 1.f / sqrtf((float)c.head_dim);

    // embed -> x, prime xn = RMSNorm(x, layer0.input_norm)
    kernels::launch_embedding(d_ids, s.w.embed_tokens, x, N, H, st);
    kernels::launch_rmsnorm(x, s.w.layers[0].input_norm, xn, N, H, eps, st);

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (w.linear_attn) {
            // ---- Gated DeltaNet linear-attention layer ----
            proj(xn, w.wqkv,      w.wqkv_type,      b8, lqkv,  H);   // qkv
            proj(xn, w.wqkv_gate, w.wqkv_gate_type, lz, lvdim, H);   // z gate
            proj(xn, w.ssm_alpha, w.ssm_alpha_type, la, vh,    H);
            proj(xn, w.ssm_beta,  w.ssm_beta_type,  lb, vh,    H);
            bf16* conv_state = lin_conv_state + (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
            kernels::launch_prefill_gdn_conv(b8, w.ssm_conv, conv_state, gq, gk, gv,
                N, c.linear_q_heads, vh, c.linear_head_dim, c.linear_conv_kernel, eps, st);
            float* layer_state = s.lin_state + (size_t)L * vh * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_prefill_gdn_scan(gq, gk, gv, la, lb, w.ssm_dt, w.ssm_a,
                layer_state, att, N, c.linear_q_heads, vh, c.linear_head_dim, st);
            kernels::launch_prefill_gated_norm(att, lz, w.ssm_norm, lnrm, N, vh, c.linear_head_dim, eps, st);
            proj(lnrm, w.ssm_out, w.ssm_out_type, ao, H, lvdim);
        } else {
            // ---- full softmax-attention layer (q_has_gate, partial RoPE, int8 KV) ----
            proj(xn, w.wq, w.wq_type, b8, wide,  H);                 // qraw = [q|gate] per head
            proj(xn, w.wk, w.wk_type, kf, kvdim, H);
            proj(xn, w.wv, w.wv_type, vf, kvdim, H);
            kernels::launch_prefill_split_q_gate(b8, qb, qg, N, c.n_q_heads, c.head_dim, st);
            signed char* kpool = (signed char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            signed char* vpool = (signed char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            if (!kv8) { a.free_all(); a8.free_all(); am.free_all(); fprintf(stderr, "[prefill] batched prefill requires int8 KV\n"); return -1; }
            kernels::launch_prefill_qknorm_rope_kv_int8(qb, kf, vf, w.q_norm, w.k_norm,
                kpool, vpool, kscale, vscale, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                rope_dim, rope_theta, eps, bs, mbs, st);
            kernels::launch_prefill_attn_int8_paged(qb, kpool, vpool, kscale, vscale, btable, att,
                N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale, st);
            kernels::launch_prefill_mul_sigmoid(att, qg, N, qdim, st);
            proj(att, w.wo, w.wo_type, ao, H, qdim);
        }

        // x += ao (post-attn residual, in-place) ; hn = RMSNorm(x, post_attn_norm)
        kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
        kernels::launch_rmsnorm(x, w.post_attn_norm, hn, N, H, eps, st);

        if (!moe) {
            // dense SwiGLU FFN, chunked over tokens (upstream #530): ffg/ffu/A_i8 stay O(FC*ffn).
            // Per-token independent, so this is numerically identical to the full-width pass.
            for (int fo = 0; fo < N; fo += FC) {
                const int fn = (N - fo < FC) ? (N - fo) : FC;
                const bf16* hn_c = hn + (size_t)fo * H;
                // One fused GEMM emits silu(gate)*up straight into ffg; fall back to the two
                // projections + elementwise SwiGLU when the interleaved weight is unavailable.
                if (!ffn_fused(w, hn_c, ffg, fn)) {
                    proj(hn_c, w.gate_q, w.gate_qtype, ffg, ffn, H, fn);
                    proj(hn_c, w.up_q,   w.up_qtype,   ffu, ffn, H, fn);
                    kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                }
                proj(ffg, w.down_q, w.down_qtype, ao + (size_t)fo * H, H, ffn, fn);
            }
            // x += ffn_out (post-attn residual already folded into x above)
            kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
        } else {
            // ---- expert-grouped 256-expert int8 MoE FFN (this PR): route -> bucket routed
            // (token, expert) pairs by expert -> per-expert int8 tensor-core GEMMs, so each expert's
            // weights are read ONCE per layer instead of once per routed token (the ~1.1 GB/token
            // MoE weight re-read that pinned the token loop). Router logits use the decode-reference
            // gemv_f32-order dot; the router weight may itself be quantized in the UD GGUF. ----
            const void* rw = w.router_w_type ? dq(w.router_w, w.router_w_type, E, H) : w.router_w;
            kernels::launch_pfm_router_logits(hn, rw, mlogits, N, E, H, st);
            pf_cu(cudaMemsetAsync(mcounts, 0, E * sizeof(int), st), "moe counts zero");
            kernels::launch_moe_router(mlogits, mids, mweights, mcounts, N, E, topk, 1, st);
            kernels::launch_pfm_bucket_pairs(mids, mweights, mcounts, moffsets, mcursors,
                                             pair_tok, pair_w, tilemap, d_ntiles, N, E, topk, st);
            // Expert weights -> int8 rows ONCE per layer (one launch covers all 256 experts).
            kernels::launch_gguf_dequant_rows_i8(w.gate_qtype, w.gate_q, Wg_i8, swg, E * mffn, H, st);
            kernels::launch_gguf_dequant_rows_i8(w.up_qtype,   w.up_q,   Wu_i8, swu, E * mffn, H, st);
            kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q, Wd_i8, swd, E * H, mffn, st);
            kernels::launch_prefill_quantize_rows_i8(hn, mA_i8, msx, N, H, st);
            kernels::launch_pfm_moe_gemm_i8(mA_i8, msx, Wg_i8, swg, pair_tok, pair_w, moffsets,
                                            tilemap, d_ntiles, hg, nullptr, mffn, H, max_tiles,
                                            /*a_indirect=*/true, /*c_scatter=*/false, st);
            kernels::launch_pfm_moe_gemm_i8(mA_i8, msx, Wu_i8, swu, pair_tok, pair_w, moffsets,
                                            tilemap, d_ntiles, hu, nullptr, mffn, H, max_tiles,
                                            true, false, st);
            kernels::launch_prefill_swiglu(hg, hu, hh, (long)P * mffn, st);
            kernels::launch_prefill_quantize_rows_i8(hh, h_i8, sh, P, mffn, st);
            pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), st), "routed zero");
            kernels::launch_pfm_moe_gemm_i8(h_i8, sh, Wd_i8, swd, pair_tok, pair_w, moffsets,
                                            tilemap, d_ntiles, nullptr, routed_f32, H, mffn, max_tiles,
                                            /*a_indirect=*/false, /*c_scatter=*/true, st);
            // Shared expert (Qwen3.6 UD): out scaled by sigmoid(hn . gate_inp) per token.
            bf16* shared_out = nullptr;
            const void* sg = w.shared_gate_q ? w.shared_gate_q : w.shared_gate;
            if (c.n_shared > 0 && sg) {
                const int sgt = w.shared_gate_q ? w.shared_gate_qtype : 0;
                const void* su = w.shared_up_q ? w.shared_up_q : w.shared_up;
                const int sut = w.shared_up_q ? w.shared_up_qtype : 0;
                const void* sd = w.shared_down_q ? w.shared_down_q : w.shared_down;
                const int sdt = w.shared_down_q ? w.shared_down_qtype : 0;
                const bool has_gi = w.shared_gate_inp != nullptr;
                if (has_gi) {
                    const void* gi = w.shared_gate_inp_type
                        ? dq(w.shared_gate_inp, w.shared_gate_inp_type, 1, H) : w.shared_gate_inp;
                    kernels::launch_pfm_shared_gate(hn, gi, dw, N, H, st);
                }
                proj(hn, sg, sgt, sfg, mffn, H);
                proj(hn, su, sut, sfu, mffn, H);
                kernels::launch_pfm_shared_swiglu(sfg, sfu, has_gi ? dw : nullptr, sfh, N, mffn, st);
                proj(sfh, sd, sdt, ao, H, mffn);
                shared_out = ao;
            }
            // x = x + routed + shared (fp32 math); x already holds the post-attn residual, so this
            // fused add writes the final layer output directly (no separate ffn-out residual add).
            kernels::launch_pfm_resid3(x, routed_f32, shared_out, x, (long)N * H, st);
        }

        const void* next_norm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_rmsnorm(x, next_norm, xn, N, H, eps, st);
    }

    // Seed for the first decode step: argmax at the last prompt position (xn already = final norm).
    const bf16* xn_last = xn + (size_t)(N - 1) * H;
    if (s.w.lm_head_type)
        kernels::launch_gemv_q_f32(xn_last, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else
        kernels::launch_gemv_f32(xn_last, s.w.lm_head, s.logits, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    pf_cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "prefill seed");
    pf_cu(cudaStreamSynchronize(st), "prefill sync");
    int seed = *s.h_out_id;

    // Keep the dense scratch for the next prefill when it is small enough to be worth the VRAM (see
    // PF_ARENA_KEEP_BYTES); the MoE arena is huge and per-N, so it is always released.
    if (a.bytes() + a8.bytes() > PF_ARENA_KEEP_BYTES) { a.free_all(); a8.free_all(); }
    am.free_all();
    return seed;
}

} // namespace sparkinfer
