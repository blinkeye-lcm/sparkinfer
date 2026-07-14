// Chunked, weight-amortized prompt ingest for the Qwen3.5 dense hybrid (Qwythos-9B).
//
// The token path pays one full weight read per prompt token. Here the prompt is walked in
// chunks of MQ_CHUNK tokens and every projection in the stack (GDN qkv/gate/alpha/beta/out,
// full-attention q/k/v/o, dense SwiGLU gate/up/down) is issued as a single Q4_K x Q8_1 tile
// GEMM over the chunk, so each weight superblock is read once per tile of token rows. The
// weights stay 4-bit from DRAM into registers; nothing is dequantized.
//
// Everything that is *not* a weight read is left to the kernels the decode path already uses:
//   - the Gated-DeltaNet conv + recurrence run per token (they are a serial recurrence, and
//     they read state, not weights, so batching them buys nothing -- measured, see below),
//   - the full-attention layers call the same sink+sliding-window sparse KV kernels decode
//     calls, once per query, so prefill attends to exactly the keys decode will later read,
//   - RoPE + int8 KV append, the norms, SwiGLU and the split/gate helpers are the in-tree
//     kernels, driven over a whole chunk at once (they are already row- or head-indexed).
//
// The LM head runs for the FINAL prompt token only: it is 572 MB of Q4_K (10% of the weight
// set) and a one-pass prefill needs its logits only to seed the first decode step.
//
// Scratch is O(chunk), never O(prompt), so 128k fits in the same VRAM as 4k.

#include "qwen35_prefill_mmq.h"

#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/mmq_prefill.h"
#include "sparkinfer/kernels/quant.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace sparkinfer {

namespace {

inline int mq_env_int(const char* name, int def) {
    const char* v = getenv(name);
    if (!v) return def;
    const int x = atoi(v);
    return x > 0 ? x : def;
}

// The KV-split count the token path would pick for this seq_len (its depth-adaptive policy for
// hd256 GQA-4). Reproducing it is not a tuning choice: launch_flash_decode_split selects its
// scalar vs int8-MMA implementation from seq_len/n_splits, so a different split count silently
// runs a different kernel with different rounding than the decode this prefill has to match.
inline int mq_splits_for(int seqlen, int n_q_heads, int n_kv_heads) {
    static int fixed = -1;
    if (fixed < 0) {
        const char* e = getenv("SPARKINFER_NSPLITS");
        fixed = e ? atoi(e) : 0;
        if (fixed < 1 || fixed > 256) fixed = 0;
    }
    if (fixed) return fixed;
    int split_chunk = 256;
    if (const char* c = getenv("SPARKINFER_SPLIT_CHUNK")) {
        const int v = atoi(c);
        if (v > 0) split_chunk = v;
    }
    int want = 32;
    if ((long)seqlen > 2L * split_chunk) want = 128;
    if ((long)seqlen > 28L * split_chunk && (long)seqlen <= 48L * split_chunk) want = 256;
    if ((long)seqlen > 64L * split_chunk) want = 256;
    if (n_kv_heads > 0 && n_q_heads == n_kv_heads * 4 && want >= 128) {
        if ((long)seqlen > 98304L) want = 128;
        else if ((long)seqlen > 65536L) want = 192;
        else want = 160;
    }
    return want;
}

// Device scratch held for the lifetime of one prefill call. Sized from the chunk width, so
// the 128k prompt and a 4k prompt allocate the same buffers.
struct Scratch {
    std::vector<void*> owned;

    template <class T>
    T* get(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) {
            fprintf(stderr, "[prefill-mmq] scratch alloc of %zu bytes failed\n", n * sizeof(T));
            return nullptr;
        }
        owned.push_back(p);
        return static_cast<T*>(p);
    }
    ~Scratch() {
        for (void* p : owned) cudaFree(p);
    }
};

using bf16_t = unsigned short;

}  // namespace

int qwen35_prefill_mmq_min_ctx() {
    const char* off = getenv("SPARKINFER_PREFILL_MMQ");
    if (off && off[0] == '0') return INT32_MAX;
    // Read directly rather than through mq_env_int: 0 is a meaningful setting here (ingest every
    // prompt through this path, which is what an A/B against the token loop wants), and
    // mq_env_int would quietly read it as "unset" and hand back the 65536 default instead.
    const char* v = getenv("SPARKINFER_PREFILL_MMQ_MINCTX");
    if (!v || !*v) return 65536;
    const int x = atoi(v);
    return x >= 0 ? x : 65536;
}

int qwen35_prefill_mmq_run(const Qwen35Config& cfg, const Qwen35Weights& w, KVCacheManager* kv,
                           uint64_t seq_id, float* lin_state, void* lin_conv_state, float* logits,
                           int* d_out_id, int* h_out_id, const int* ids, int n_tokens,
                           cudaStream_t stream) {
    if (n_tokens <= 0 || !kv || !lin_state || !lin_conv_state) return -1;
    // This path is written against the Qwythos shape only: dense SwiGLU FFN, hd256 GQA-4
    // full attention every full_attn_interval layers, partial RoPE, int8 paged KV, and a
    // Q4_K weight set (the tile GEMM is Q4_K x Q8_1).
    if (!cfg.dense_ffn || !cfg.hybrid || cfg.head_dim != 256 || cfg.n_shared != 0) return -1;
    if (cfg.n_q_heads != cfg.n_kv_heads * 4 || cfg.full_attn_interval <= 0) return -1;
    if (!(cfg.rope_dim > 0 && cfg.rope_dim < cfg.head_dim)) return -1;
    if (!kv->int8_kv() || kv->block_size() != 16) return -1;
    if (w.lm_head_type != 12) return -1;
    for (int L = 0; L < cfg.n_layers; L++) {
        const Qwen35LayerWeights& lw = w.layers[L];
        if (lw.gate_qtype != 12 || lw.up_qtype != 12 || lw.down_qtype != 12) return -1;
        if (lw.linear_attn) {
            if (lw.wqkv_type != 12 || lw.wqkv_gate_type != 12 || lw.ssm_alpha_type != 12 ||
                lw.ssm_beta_type != 12 || lw.ssm_out_type != 12)
                return -1;
        } else if (lw.wq_type != 12 || lw.wk_type != 12 || lw.wv_type != 12 || lw.wo_type != 12) {
            return -1;
        }
        // The chunk driver folds the q-gate itself; a layer without one is a different model.
        if (!lw.linear_attn && !lw.q_has_gate) return -1;
        if (cfg.linear_head_dim != 128) return -1;
    }

    const int H = cfg.hidden;
    const int qdim = cfg.n_q_heads * cfg.head_dim;
    const int kvdim = cfg.n_kv_heads * cfg.head_dim;
    const int lin_qdim = cfg.linear_q_heads * cfg.linear_head_dim;
    const int lin_vdim = cfg.linear_v_heads * cfg.linear_head_dim;
    const int lin_qkvdim = 2 * lin_qdim + lin_vdim;
    const int ffn = cfg.moe_ffn;

    const int chunk = mq_env_int("SPARKINFER_PREFILL_MMQ_CHUNK", 1024);
    const int M = (chunk < n_tokens) ? chunk : n_tokens;
    // Widest activation any projection consumes (the SwiGLU down-projection input).
    const int kmax = (ffn > H) ? ffn : H;

    // Sink + sliding-window selection must mirror what the decode path will read back, so the
    // window/min-ctx knobs are read from the same environment the model reads them from.
    int sparse_window = mq_env_int("SPARKINFER_SPARSE_WINDOW", 256);
    if (const char* rw = getenv("SPARKINFER_SPARSE_RECENT")) {
        const int v = atoi(rw);
        if (v > 0) sparse_window = v;
    }
    if (const char* b = getenv("SPARKINFER_SPARSE_BUDGET")) {
        const int v = atoi(b);
        if (v > 1) sparse_window = v - 1;
    }
    const int sparse_min_ctx = mq_env_int("SPARKINFER_SPARSE_MIN_CTX", 8192);
    bool sparse_enable = true;
    if (const char* se = getenv("SPARKINFER_SPARSE_KV")) sparse_enable = (se[0] != '0');
    const int sparse_budget = 1 + sparse_window;
    const int max_splits = 256;   // the token path's ceiling; partials below are sized for it

    Scratch sc;
    bf16_t* x = sc.get<bf16_t>((size_t)M * H);
    bf16_t* xn = sc.get<bf16_t>((size_t)M * H);
    bf16_t* hres = sc.get<bf16_t>((size_t)M * H);
    bf16_t* hn = sc.get<bf16_t>((size_t)M * H);
    bf16_t* ao = sc.get<bf16_t>((size_t)M * H);
    bf16_t* ffn_out = sc.get<bf16_t>((size_t)M * H);
    signed char* q8 = sc.get<signed char>((size_t)M * kmax);
    float* q8d = sc.get<float>((size_t)M * (kmax >> 5));
    float* q8s = sc.get<float>((size_t)M * (kmax >> 5));
    bf16_t* qraw = sc.get<bf16_t>((size_t)M * qdim * 2);
    bf16_t* qb = sc.get<bf16_t>((size_t)M * qdim);
    bf16_t* qgate = sc.get<bf16_t>((size_t)M * qdim);
    bf16_t* kb = sc.get<bf16_t>((size_t)M * kvdim);
    bf16_t* vb = sc.get<bf16_t>((size_t)M * kvdim);
    bf16_t* attn = sc.get<bf16_t>((size_t)M * qdim);
    bf16_t* lin_qkv = sc.get<bf16_t>((size_t)M * lin_qkvdim);
    bf16_t* lin_z = sc.get<bf16_t>((size_t)M * lin_vdim);
    bf16_t* lin_alpha = sc.get<bf16_t>((size_t)M * cfg.linear_v_heads);
    bf16_t* lin_beta = sc.get<bf16_t>((size_t)M * cfg.linear_v_heads);
    bf16_t* lin_gdn = sc.get<bf16_t>((size_t)M * lin_vdim);
    bf16_t* lin_norm = sc.get<bf16_t>((size_t)M * lin_vdim);
    bf16_t* tq = sc.get<bf16_t>(lin_qdim);   // one token's conv output, consumed by the scan
    bf16_t* tk = sc.get<bf16_t>(lin_qdim);
    bf16_t* tv = sc.get<bf16_t>(lin_vdim);
    bf16_t* gate_buf = sc.get<bf16_t>((size_t)M * ffn);
    bf16_t* up_buf = sc.get<bf16_t>((size_t)M * ffn);
    bf16_t* ffn_h = sc.get<bf16_t>((size_t)M * ffn);
    int* d_ids = sc.get<int>(M);
    int* d_pos = sc.get<int>(M);
    int* d_seqlen = sc.get<int>(M);
    float* fa_m = sc.get<float>((size_t)cfg.n_q_heads * max_splits);
    float* fa_l = sc.get<float>((size_t)cfg.n_q_heads * max_splits);
    float* fa_acc = sc.get<float>((size_t)cfg.n_q_heads * max_splits * cfg.head_dim);
    int* sparse_sel = sc.get<int>((size_t)cfg.n_kv_heads * sparse_budget);
    float* d_one = sc.get<float>(1);
    void* aq81 = sc.get<char>(kernels::llama_q8_1_bytes(H));
    if (!x || !ffn_h || !sparse_sel || !d_one || !aq81) return -1;

    const float one = 1.f;
    cudaMemcpyAsync(d_one, &one, sizeof(float), cudaMemcpyHostToDevice, stream);

    int* btable = kv->block_table(seq_id);
    const int bs = kv->block_size();
    const int mbps = kv->max_blocks_per_seq();
    const float scale = 1.f / sqrtf((float)cfg.head_dim);
    const bool sparse_avail = sparse_enable && sparse_budget > 1;

    if (const char* dbg = getenv("SPARKINFER_PREFILL_MMQ_DEBUG"); dbg && dbg[0] == '1')
        fprintf(stderr, "[prefill-mmq] %d tokens, chunk=%d, splits@end=%d, sparse=%d (window=%d min_ctx=%d)\n",
                n_tokens, M, mq_splits_for(n_tokens, cfg.n_q_heads, cfg.n_kv_heads), (int)sparse_avail, sparse_window, sparse_min_ctx);

    // The recurrence starts from zero, exactly as the token path's position-0 reset does. This
    // path never runs a position-0 forward_token, so it owns the reset.
    cudaMemsetAsync(lin_state, 0,
                    (size_t)cfg.n_layers * cfg.linear_v_heads * cfg.linear_head_dim *
                        cfg.linear_head_dim * sizeof(float),
                    stream);
    cudaMemsetAsync(lin_conv_state, 0,
                    (size_t)cfg.n_layers * (cfg.linear_conv_kernel - 1) * lin_qkvdim * sizeof(bf16_t),
                    stream);

    std::vector<int> h_pos(M), h_seq(M);
    int last_id = -1;

    for (int t0 = 0; t0 < n_tokens; t0 += M) {
        const int m = (n_tokens - t0 < M) ? (n_tokens - t0) : M;
        for (int i = 0; i < m; i++) {
            h_pos[i] = t0 + i;
            h_seq[i] = t0 + i + 1;
        }
        cudaMemcpyAsync(d_ids, ids + t0, m * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_pos, h_pos.data(), m * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_seqlen, h_seq.data(), m * sizeof(int), cudaMemcpyHostToDevice, stream);

        kernels::launch_embedding(d_ids, w.embed_tokens, x, m, H, stream);
        kernels::launch_rmsnorm(x, w.layers[0].input_norm, xn, m, H, cfg.rms_eps, stream);

        for (int L = 0; L < cfg.n_layers; L++) {
            const Qwen35LayerWeights& lw = w.layers[L];

            // ---- token mixing: Gated DeltaNet, or full attention every full_attn_interval ----
            if (lw.linear_attn) {
                kernels::launch_mmq_quant_q8_1(xn, q8, q8d, q8s, m, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.wqkv, lin_qkv, m, lin_qkvdim, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.wqkv_gate, lin_z, m, lin_vdim, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.ssm_alpha, lin_alpha, m, cfg.linear_v_heads, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.ssm_beta, lin_beta, m, cfg.linear_v_heads, H, stream);

                bf16_t* conv_state =
                    (bf16_t*)lin_conv_state + (size_t)L * (cfg.linear_conv_kernel - 1) * lin_qkvdim;
                float* layer_state =
                    lin_state + (size_t)L * cfg.linear_v_heads * cfg.linear_head_dim * cfg.linear_head_dim;
                // The recurrence carries token to token; it reads state rather than weights, so
                // it stays exactly the decode kernel, stepped once per token of the chunk.
                for (int i = 0; i < m; i++) {
                    kernels::launch_qwen36_conv_split_l2norm_fused(
                        lin_qkv + (size_t)i * lin_qkvdim, lw.ssm_conv, conv_state, tq, tk, tv,
                        cfg.linear_q_heads, cfg.linear_v_heads, cfg.linear_head_dim,
                        cfg.linear_conv_kernel, cfg.rms_eps, stream);
                    kernels::launch_qwen36_gdn_ar(tq, tk, tv, lin_alpha + (size_t)i * cfg.linear_v_heads,
                                                  lin_beta + (size_t)i * cfg.linear_v_heads, lw.ssm_dt,
                                                  lw.ssm_a, layer_state, lin_gdn + (size_t)i * lin_vdim,
                                                  cfg.linear_q_heads, cfg.linear_v_heads,
                                                  cfg.linear_head_dim, stream);
                }
                // gated_norm normalizes each v-head independently, so a chunk of m tokens is
                // just m * v_heads consecutive heads.
                kernels::launch_qwen36_gated_norm(lin_gdn, lin_z, lw.ssm_norm, lin_norm,
                                                  cfg.linear_v_heads * m, cfg.linear_head_dim,
                                                  cfg.rms_eps, stream);
                kernels::launch_mmq_quant_q8_1(lin_norm, q8, q8d, q8s, m, lin_vdim, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.ssm_out, ao, m, H, lin_vdim, stream);
            } else {
                kernels::launch_mmq_quant_q8_1(xn, q8, q8d, q8s, m, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.wq, qraw, m, qdim * 2, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.wk, kb, m, kvdim, H, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.wv, vb, m, kvdim, H, stream);
                // q rows are [head][2*head_dim]; m tokens of n_q_heads heads split as one run.
                kernels::launch_qwen36_split_q_gate(qraw, qb, qgate, cfg.n_q_heads * m, cfg.head_dim,
                                                    stream);
                kernels::launch_rmsnorm_qk(qb, kb, lw.q_norm, lw.k_norm, cfg.n_q_heads * m,
                                           cfg.n_kv_heads * m, cfg.head_dim, cfg.rms_eps, stream);

                const int kv_elem = 1;   // int8 paged KV (checked above)
                void* kpool = (char*)kv->k_pool() + (size_t)L * kv->layer_stride_elems() * kv_elem;
                void* vpool = (char*)kv->v_pool() + (size_t)L * kv->layer_stride_elems() * kv_elem;
                void* kscale = (char*)kv->k_scale_pool() + (size_t)L * kv->scale_layer_stride_elems() * 2;
                void* vscale = (char*)kv->v_scale_pool() + (size_t)L * kv->scale_layer_stride_elems() * 2;

                // max_blocks_per_seq = 0 collapses this kernel's per-token block-table stride to
                // the single shared table one sequence has, which is what a prompt needs.
                kernels::launch_rope_kv_append_partial_int8(qb, kb, vb, kpool, vpool, kscale, vscale,
                                                            btable, d_pos, m, cfg.n_q_heads,
                                                            cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim,
                                                            cfg.rope_theta, bs, 0, stream);

                // Every query in the chunk is already in the cache, but each one is masked to its
                // own seq_len, so appending the whole chunk before attending stays causal.
                for (int i = 0; i < m; i++) {
                    const int sl = t0 + i + 1;
                    const int nsp = mq_splits_for(sl, cfg.n_q_heads, cfg.n_kv_heads);
                    const bool use_sparse = sparse_avail && sl >= sparse_min_ctx;
                    if (use_sparse) {
                        kernels::launch_fa_kv_window_select(d_seqlen + i, sparse_sel, cfg.n_kv_heads,
                                                            bs, sparse_budget, sparse_window, stream);
                        kernels::launch_flash_decode_split_sparse(
                            qb + (size_t)i * qdim, kpool, vpool, btable, d_seqlen + i, sparse_sel,
                            fa_m, fa_l, fa_acc, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, bs, mbps,
                            nsp, sparse_budget, scale, kscale, vscale, stream);
                        kernels::launch_fa_combine_hd256(fa_m, fa_l, fa_acc, attn + (size_t)i * qdim,
                                                         cfg.n_q_heads, nsp, nullptr, stream);
                    } else {
                        kernels::launch_flash_decode_split(
                            qb + (size_t)i * qdim, kpool, vpool, btable, d_seqlen + i,
                            attn + (size_t)i * qdim, fa_m, fa_l, fa_acc, 1, cfg.n_q_heads,
                            cfg.n_kv_heads, cfg.head_dim, bs, mbps, nsp, scale, stream, nullptr,
                            sl, kscale, vscale, 1, nullptr);
                    }
                }
                kernels::launch_qwen36_mul_sigmoid(attn, qgate, m * qdim, stream);
                kernels::launch_mmq_quant_q8_1(attn, q8, q8d, q8s, m, qdim, stream);
                kernels::launch_mmq_q4k(q8, q8d, q8s, lw.wo, ao, m, H, qdim, stream);
            }

            // hres = x + ao ; hn = RMSNorm(hres, post_attn_norm)
            kernels::launch_add_rmsnorm2(x, ao, lw.post_attn_norm, hres, hn, m, H, cfg.rms_eps, stream);

            // ---- dense SwiGLU FFN ----
            kernels::launch_mmq_quant_q8_1(hn, q8, q8d, q8s, m, H, stream);
            kernels::launch_mmq_q4k(q8, q8d, q8s, lw.gate_q, gate_buf, m, ffn, H, stream);
            kernels::launch_mmq_q4k(q8, q8d, q8s, lw.up_q, up_buf, m, ffn, H, stream);
            kernels::launch_qwen36_shared_swiglu(gate_buf, up_buf, d_one, ffn_h, m * ffn, stream);
            kernels::launch_mmq_quant_q8_1(ffn_h, q8, q8d, q8s, m, ffn, stream);
            kernels::launch_mmq_q4k(q8, q8d, q8s, lw.down_q, ffn_out, m, H, ffn, stream);

            // x = hres + ffn_out ; xn = RMSNorm(x, next layer's input norm / final norm)
            const void* nextnorm = (L + 1 < cfg.n_layers) ? w.layers[L + 1].input_norm : w.final_norm;
            kernels::launch_add_rmsnorm2(hres, ffn_out, nextnorm, x, xn, m, H, cfg.rms_eps, stream);
        }

        // xn now holds RMSNorm(x_final, final_norm) for every token of the chunk, but only the
        // last token of the prompt needs logits -- that is the 572 MB LM head skipped for all
        // the others.
        if (t0 + m >= n_tokens) {
            kernels::launch_quantize_q8_1_blocks(xn + (size_t)(m - 1) * H, aq81, H, stream);
            kernels::launch_mmvq_q4k_f32(aq81, w.lm_head, logits, cfg.vocab, H, stream);
            kernels::launch_argmax(logits, d_out_id, 1, cfg.vocab, stream);
            cudaMemcpyAsync(h_out_id, d_out_id, sizeof(int), cudaMemcpyDeviceToHost, stream);
        }
        // The whole chunk is asynchronous; this is where any copy or launch failure in it lands.
        const cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[prefill-mmq] chunk at token %d: %s\n", t0, cudaGetErrorString(err));
            return -1;
        }
    }
    last_id = *h_out_id;
    return (last_id >= 0 && last_id < cfg.vocab) ? last_id : -1;
}

}  // namespace sparkinfer
