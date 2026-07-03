![sparkinfer banner](docs/sparkinfer.png)

# _SP⚡RKINFER_

**Fastest MoE/LLM inference runtime for consumer and edge Blackwell GPUs.**

**SPARKINFER** is a _Blackwell-native_ inference runtime built for **high-speed**, **power-optimized** **local AI** on **NVIDIA RTX 50xx**, **RTX PRO 6000**, **RTX Spark**, and **Jetson Thor**.

_It is designed for the next generation of personal agents like **Openclaw**, local copilots, **robotics**, and edge AI systems where inference speed, memory efficiency, privacy, and power efficiency decide how usable local intelligence feels._

**SPARKINFER** is continuously optimized through [**SN74 Gittensor** competition](https://gittensor.io/) and **proprietary Kernel Design Agents**, turning frontier CUDA improvements into faster, power-optimized local MoE/LLM decode on real Blackwell hardware. _([Live dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/))_

## Benchmark

| context | sparkinfer<br>GGUF Q4_K_M | llama.cpp<br>GGUF Q4_K_M | vLLM<br>GPTQ Int4 | SGLang<br>GPTQ Int4 | TensorRT-LLM<br>NVFP4 |
|---:|---:|---:|---:|---:|---:|
| 128 | **493.56 tok/s** | 365.85 tok/s | 280.83 tok/s | 241.21 tok/s | 99.00 tok/s |
| 512 | **469.58 tok/s** | 342.59 tok/s | 270.86 tok/s | 239.82 tok/s | 98.59 tok/s |
| 4k | **392.65 tok/s** | 292.99 tok/s | 202.65 tok/s | 234.67 tok/s | failed |
| 16k | **266.14 tok/s** | 245.53 tok/s | 81.89 tok/s | 226.12 tok/s | not run |

Runtime footprint, excluding model weights and Python launcher scripts:

| runtime | measured artifact | size | sparkinfer is |
|---|---|---:|---:|
| sparkinfer | native runtime binary | **2.5 MB** | baseline |
| llama.cpp | CUDA runtime executable + shared libs | 80 MB | 33x smaller |
| vLLM | runtime package | 605 MB | 243x smaller |
| SGLang | runtime + native kernel packages | 1.9 GB | 743x smaller |
| TensorRT-LLM | runtime package | 3.6 GB | 1,430x smaller |

LLM quality check, 25% benchmark tier, 196 items:

| runtime | BFCL | GSM8K | HumanEval | IFEval | MMLU-Pro | overall |
|---|---:|---:|---:|---:|---:|---:|
| sparkinfer GGUF | 73.33% | 84.85% | 80.00% | 77.08% | 44.00% | 64.37% |
| llama.cpp GGUF | 72.00% | 90.91% | 80.00% | 64.58% | 48.00% | 65.90% |
| vLLM AWQ | 76.00% | 84.85% | 80.00% | 77.08% | 48.00% | 66.92% |

sparkinfer and llama.cpp use the same GGUF on the same RTX 5090. Other runtimes cannot load
GGUF, so the table uses their fastest successful HF quantized path: vLLM/SGLang GPTQ Int4,
TensorRT-LLM NVFP4. Details: [`bench/competitors/latest-results.md`](bench/competitors/latest-results.md).

## How we move fast on SN74

SN74 rewards verified speedups. The loop is intentionally tight:

1. Pick a narrow bottleneck in the Blackwell decode path.
2. Submit a PR with source changes and benchmark evidence.
3. The bot builds `main` and the PR on the same RTX 5090.
4. The bot checks correctness against llama.cpp and guards 128, 512, 4k, and 16k decode.
5. The strongest context improvement gets the score label; regressions get explicit `regression-*` labels.
6. A maintainer merges the best frontier PR, and the dashboard updates the matching context chart.

This keeps rewards tied to marginal speed on shipped code, not claims in a PR description.

## Why SPARKINFER

Most LLM inference engines were built for datacenter GPUs and cloud AI. On consumer GPUs they can be
hard to install, power-hungry, thermally awkward, and slow to adapt to new SOTA models or algorithms
because the codebases are large and maintenance-heavy. **SPARKINFER** is designed for next-generation
personal agents on devices like [**NVIDIA RTX Spark**](https://www.nvidia.com/en-us/products/rtx-spark/),
with up to **1 Petaflop** FP4 AI performance.

SPARKINFER solves this for local Blackwell AI:

- **Fastest.** Frontier decode on RTX 5090 across 128, 512, 4k, and 16k context.
- **Smallest.** A native runtime binary measured in megabytes, not gigabytes.
- **Power-optimized.** Built for consumer and edge GPUs where thermals and watts matter.
- **SOTA-ready.** Designed to move quickly with new MoE models, quantization paths, and decode algorithms.
- **Agent-native.** Local, private inference for your data without cloud dependency or operational worry.

## Quickstart

On an NVIDIA Blackwell box (CUDA 12.8+) — the scripts auto-detect your GPU arch, fetch **prebuilt binaries** (or build from source if incompatible), and download the model:

```bash
# decode throughput (fetches Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# head-to-head vs llama.cpp on the same GGUF + GPU
bench/scripts/bench.sh --download --compare

# accuracy gate — token-match / KL / perplexity vs llama.cpp
bench/scripts/accuracy.sh --download
```

Your own model: `bench/scripts/bench.sh /path/to/model.gguf --tokens 256`. All options: [`bench/scripts/README.md`](bench/scripts/README.md).

## Miner guide

If you are contributing for SN74 rewards, start with the clear miner workflow:
[`docs/miner-guide.md`](docs/miner-guide.md). It explains what scores, what gets
rejected, how the 128 / 512 / 4k / 16k guards work, and the local commands to run
before opening a PR.

## Layout & scoring

| Path | What |
|---|---|
| [`kernels/`](kernels) | CUDA kernels — flash-decode (hd128/256/512), decode GEMV, fused quantized MoE expert FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | sync-free MoE router + expert dispatch (on-device counts, CUDA-graph-ready) |
| [`bench/`](bench) | reproducible benchmarks + eval harness (the eval/scoring scripts are maintainer-owned) |

**Scoring is speedup-only.** SN74 pays each merged PR for its verified marginal speedup,
labeled **XL / L / M / S / XS** by the deterministic eval loop. A speedup can land in
128, 512, 4k, or 16k context; sub-2% gains are never aggregated across contexts.
Tooling, bench, docs, and refactors are welcome but score 0 unless they produce a verified
frontier speedup. See [`.gittensor/weights.json`](.gittensor/weights.json) and
[`docs/miner-guide.md`](docs/miner-guide.md).

## Build

Requires **CUDA Toolkit 12.8+** (first toolkit with `sm_120` / `sm_121` codegen).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120   # or 121 for RTX Spark / Jetson Thor
cmake --build build -j
ctest --test-dir build
```

The top-level `CMakeLists.txt` is a superbuild (`kernels → moe → runtime`); each subsystem also builds standalone (the sibling `../kernels` references resolve within the monorepo). A direct `nvcc` build from the repo root works too — see [`bench/scripts`](bench/scripts).

## Targets

**Blackwell only, by design:** `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121` (RTX Spark / GB10, Jetson Thor). **Not** `sm_100` (datacenter B200/GB200 — binary-incompatible).

## Roadmap

**Milestone 1 — RTX 5090 proof-of-concept (now).** Qwen3-MoE decode on `sm_120` with
source-verified, correctness-gated eval against llama.cpp. The 5090 is where we prove the kernels,
the scoring loop, and the long-context guards.

**Milestone 2 — long context.** Keep 128-token speed from regressing while pushing 16k and then
32k decode. This is now part of the eval surface, not a side benchmark.

**Milestone 3 — MoE on low-bandwidth unified memory.** Move the same engineering to RTX Spark /
GB10-class `sm_121` hardware. There the target shifts from raw occupancy to bytes-per-token:
NVFP4 experts, residency, prefetch, and eliminating redundant weight reads.

## Contributing

Source-required and reproducible — the validator builds your PR from source (the
prebuilt binaries are a run convenience, not a submission format). Before a PR, run
`bench/scripts/bench.sh` (speed) and `bench/scripts/accuracy.sh` (accuracy must hold:
token-match and KL must stay within the current eval thresholds vs the prior build).
Contributions are rewarded on SN74 by the
**verified marginal speedup** added over the live frontier, correctness-gated against a
frozen llama.cpp reference. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Automated evaluation

Open a PR and a bot evaluates it automatically (polls every ~30 min). For each new commit it
builds your branch **from source** on an RTX 5090, gates **correctness** (token-match / KL vs
llama.cpp), checks that **128-token, 512-context, 4k-context, and 16k-context decode do not
regress**, scores the **strongest verified context improvement**, and posts a comment with an
**`eval:<label>`** verdict plus a UI-only context label such as `4k-context`:
Mixed outcomes are explicit: a real >2% win in one context can score while regressions elsewhere
are marked with `regression-*` labels and blocked from auto-merge; if no single context clears 2%
and any context regresses, the PR is `eval:REJECT` and auto-closed. Sub-2% gains are never
aggregated across contexts.

| label | meaning |
|---|---|
| `XL · L · M · S · XS` | verified speedup over the live frontier, by **% gain** (`XS` 2–3.5% … `XL` >18%) |
| `none` | correct, but no verified improvement (within the significance gate) |
| `REJECT` | failed correctness, or regressed below a no-regression guard |
| `BASELINE` | first verified entry; establishes the frontier |

The label is a **deterministic function of the measurements**, so it's reproducible across
validators. The bot also tags the PR's **subsystem** — `area:kernels` / `runtime` / `moe` /
`bench` — from its changed paths (categorization only — scoring is speedup-only; deterministic, no AI).
The bot **never merges** — merging is manual after review. Runs the same evaluator you can run
yourself: [`eval/`](eval) (`vast_eval.py`, `pr_eval_bot.py`).

### Trust & verifiability

Results are **reproducible from source today** — build `main` and the PR on the same RTX 5090 and you
get the same same-box delta (already independently reproduced by the community on a rented 5090). We're
hardening it toward **attested, multi-source eval**: CPU-TEE-signed scoring receipts (Intel TDX),
immutable run logs, and independent-validator consensus. Consumer 5090s have **no GPU Confidential
Computing**, so the *speed number* is trusted via **reproduction + consensus**, not a GPU enclave — by
design, since we optimize the hardware people actually own. → **[EVAL-TRUST.md](EVAL-TRUST.md)**

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
