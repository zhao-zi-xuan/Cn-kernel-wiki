---
id: kernel-lora-bgmv
title: "LoRA bgmv kernels (bgmv_shrink / bgmv_expand)"
type: kernel
architectures:
  - ascend-910b
  - ascend-910c
tags:
  - gemv
  - grouped-gemm
  - operator-fusion
confidence: source-reported
reproducibility: snippet
kernel_types:
  - gemv
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-ub
  - kernel-vocab-parallel-embedding
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-1884
evidence_basis: >
  Synthesized from upstream vllm-ascend PR #1884, which adds the bgmv_shrink and
  bgmv_expand AscendC kernels for LoRA. The PR body reports an end-to-end benchmark
  (QWen2.5 7B, single card, vllm-ascend v0.9.2.rc1) comparing "no custom operator" vs the
  bgmv custom operators; the numbers below are quoted from that table (source-reported —
  we did not run them; chip/dtype are not stated in the table).
performance_claims:
  - chip: davinci
    dtype: fp16
    shape: "QWen2.5 7B, single card, in/out 256/256, concurrency 20 (vllm-ascend v0.9.2.rc1); chip not named in source"
    metric: "TTFT (ms), baseline -> bgmv custom operator"
    value: "2219 -> 544 ms (source-reported, PR #1884 table)"
    source_id: pr-vllm-ascend-1884
    measurement: source-reported
  - chip: davinci
    dtype: fp16
    shape: "QWen2.5 7B, single card, in/out 256/256, concurrency 20 (vllm-ascend v0.9.2.rc1); chip not named in source"
    metric: "throughput (tokens/s), baseline -> bgmv custom operator"
    value: "90 -> 149 (source-reported, PR #1884 table)"
    source_id: pr-vllm-ascend-1884
    measurement: source-reported
  - chip: davinci
    dtype: fp16
    shape: "QWen2.5 7B, single card, in/out 512/512, concurrency 20 (vllm-ascend v0.9.2.rc1); chip not named in source"
    metric: "TTFT (ms), baseline -> bgmv custom operator"
    value: "2758 -> 641 ms (source-reported, PR #1884 table)"
    source_id: pr-vllm-ascend-1884
    measurement: source-reported
  - chip: davinci
    dtype: fp16
    shape: "QWen2.5 7B, single card, across the reported input/output/concurrency rows"
    metric: "TTFT / TPOT / throughput (author summary)"
    value: "~70% improvement overall (author-reported in PR #1884)"
    source_id: pr-vllm-ascend-1884
    measurement: source-reported
---

# LoRA bgmv kernels (bgmv_shrink / bgmv_expand)

## Overview

LoRA inference applies, per request, a low-rank update `ΔW = B · A` on top of the base
weights. When a batch mixes **different adapters**, the apply step is a *batched, grouped*
gather matrix-vector product — each sequence indexes its own `(A, B)` matrices. PR
[#1884](../../sources/prs/vllm-ascend/PR-1884.md) adds two custom AscendC kernels,
`bgmv_shrink` and `bgmv_expand` (`csrc/kernels/bgmv_*.cpp`), to make this fast on Ascend.

- **shrink** — `x (hidden) → r (rank)`: multiply the input by the per-request `A` matrix,
  reducing hidden dim to the (small) LoRA rank.
- **expand** — `r (rank) → hidden`: multiply the rank-space vector by `B` and add back
  into the base output.

`bgmv` = **batched gather matrix-vector**: the "batched/gather" part is the per-request
adapter indexing; the compute is GEMV-shaped (small rank makes it thin).

## Structure

Both kernels gather the adapter matrix rows for each request into [UB](../hardware/ub.md),
run the (thin) matmul on the [Cube unit](../hardware/cube-unit.md), and write the result
back. Because the LoRA rank is small, these are latency-sensitive, gather-bound ops rather
than large dense matmuls.

```cpp
// AscendC sketch (snippet-level) — synthesized from the kernels' purpose,
// NOT a verbatim upstream excerpt.  bgmv_shrink: x[hidden] -> y[rank]
for (int req = 0; req < batch; ++req) {
    int adapter = loraIndex[req];        // gather: which adapter this request uses
    CopyInA(adapter);                    // MTE: A rows for this adapter -> UB
    CopyInX(req);                        // MTE: x -> UB
    GemvAccumulate(y[req], A, x[req]);   // Cube: thin GEMV (hidden -> rank)
    CopyOut(y[req]);                     // MTE: UB -> GM
}
// bgmv_expand mirrors this: r[rank] -> hidden, adding into the base output.
```

## Notes for adapters

- shrink and expand are a pair — the rank dimension links them; keep dtype/rank
  conventions aligned across both.
- The perf win is from avoiding a generic dense path for what is a thin, per-request
  gather-GEMV; the gather (adapter indexing) is as important as the matmul.
- **Reported numbers**: PR #1884 includes an end-to-end table (QWen2.5 7B, single card,
  vllm-ascend v0.9.2.rc1) — e.g. at in/out 256/256, concurrency 20: TTFT 2219→544 ms,
  TPOT 213→131 ms, throughput 90→149 tok/s; author summarizes ~70% overall. These are
  **source-reported** (the PR's own end-to-end run), not measured by this wiki, and the
  table does not name the chip/dtype. Confidence stays `source-reported` (single upstream
  source, no official-doc corroboration, not independently reproduced).

## See also

- [Cube Unit](../hardware/cube-unit.md) — runs the thin GEMV.
- [Unified Buffer (UB)](../hardware/ub.md) — staging for gathered adapter rows.
