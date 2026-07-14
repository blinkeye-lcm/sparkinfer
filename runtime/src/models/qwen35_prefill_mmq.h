#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"

namespace sparkinfer {

// Long-context prompt ingest for the Qwen3.5 dense hybrid (Qwythos).
//
// forward_token() re-reads every projection weight for each prompt token, so a 128k prompt
// streams the whole quantized weight set 131072 times. This entry point walks the prompt in
// token chunks and runs each chunk's projections through the Q4_K x Q8_1 tile GEMM in
// kernels/mmq_prefill.h, so a weight superblock is read once per tile instead of once per
// token. Weights stay 4-bit end to end (never dequantized).
//
// It leaves the paged int8 KV cache, the Gated-DeltaNet recurrent state and the conv state
// exactly where a forward_token loop would leave them, so a subsequent decode continues
// unchanged. Returns the argmax id of the final prompt token, or -1 if this path declined
// to run (caller then falls back to the token loop).
int qwen35_prefill_mmq_run(const Qwen35Config& cfg, const Qwen35Weights& w, KVCacheManager* kv,
                           uint64_t seq_id, float* lin_state, void* lin_conv_state, float* logits,
                           int* d_out_id, int* h_out_id, const int* ids, int n_tokens,
                           cudaStream_t stream);

// Prompts longer than this use the chunked path above; everything at or below it stays on the
// token loop. Defaults to 65536 (SPARKINFER_PREFILL_MMQ_MINCTX), and returns a value no prompt
// can exceed when the path is switched off with SPARKINFER_PREFILL_MMQ=0.
int qwen35_prefill_mmq_min_ctx();

}  // namespace sparkinfer
