## Summary

<!-- What this PR optimizes and the core idea — which kernel/path, what changed. -->


## Proof of speedup

<!-- sparkinfer rewards verified speedups only. Show before -> after from a real benchmark run.
     (Non-speedup PRs — bug fixes, tooling, docs — are welcome too; they score 0, so you can
     skip this section and say so.) -->

- [ ] Tested on **RTX 5090** (`sm_120`)

**Benchmark log** — `bench/scripts/bench.sh --download` on the baseline build and on this PR:

```text
# paste the bench output here (baseline -> this PR): decode tok/s
```

| build | decode tok/s |
|---|--:|
| frontier (before) |  |
| this PR (after)   |  |

<!-- More PR-checklist items will be added here later: correctness / accuracy gate
     (bench/scripts/accuracy.sh), compute-sanitizer clean, both basket models (Qwen + Gemma).
     The auto-eval bot also rebuilds + benchmarks this PR on an RTX 5090 and posts an eval:* label. -->
