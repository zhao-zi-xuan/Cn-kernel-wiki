---
id: kernel-triton-sampling
title: "Triton sampling kernels (rejection sampling / penalties)"
type: kernel
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - triton-ascend
  - decode
  - vector-unit
confidence: source-reported
reproducibility: snippet
kernel_types:
  - sampling
  - decode
languages:
  - triton-ascend
  - python
related:
  - lang-triton-ascend
  - hw-vector-unit
  - hw-ub
  - kernel-decode-attention
sources:
  - pr-vllm-ascend-5259
  - pr-vllm-ascend-5324
  - pr-vllm-ascend-7569
evidence_basis: >
  Synthesized from upstream vllm-ascend PRs adding Triton sampling kernels: optimized
  rejection sampling (#5259), a refactor moving them into ops/triton (#5324), and
  penalty-related kernels (#7569). The PRs claim latency/throughput gains but disclose no
  machine-readable numbers, so the performance_claims entries are qualitative and
  source-reported (D2).
performance_claims:
  - chip: davinci
    dtype: fp32
    shape: "per-request token logits, batch × vocab (shape not disclosed)"
    metric: "sampling-stage latency with repetition/frequency/presence penalties"
    value: "source-reported — get_token_bin_counts_and_mask + apply_penalties as Triton kernels significantly reduce sampling latency when penalties are enabled (#7569); no absolute figure disclosed"
    source_id: pr-vllm-ascend-7569
    measurement: source-reported
  - chip: davinci
    dtype: fp32
    shape: "rejection sampling across batch sizes / MTP configs (shape not disclosed)"
    metric: "rejection-sampling kernel performance"
    value: "source-reported — optimized Triton rejection_random_sample kernels report gains across batch sizes and MTP configs while preserving accuracy (#5259); no absolute figure disclosed"
    source_id: pr-vllm-ascend-5259
    measurement: source-reported
---

# Triton sampling kernels

## Overview

After the model produces logits, the **sampling stage** turns them into the next token(s):
apply penalties, then draw a sample (with rejection sampling for speculative/MTP decoding).
On Ascend these are implemented as [Triton-Ascend](../languages/triton-ascend.md) kernels
under `vllm_ascend/ops/triton/` — a good fit because sampling is element-wise / reduction
work over the `batch × vocab` logits grid, not matmul.

Two families:

- **Rejection sampling** (`reject_sample.py`) — optimized Triton kernels for
  `rejection_random_sample`, used in speculative / multi-token-prediction (MTP) decoding
  ([#5259](../../sources/prs/vllm-ascend/PR-5259.md); an earlier refactor
  [#5324](../../sources/prs/vllm-ascend/PR-5324.md) moved these into `ops/triton`).
- **Penalties** (`bincount.py`, `penalty.py`) — `get_token_bin_counts_and_mask` +
  `apply_penalties` for repetition/frequency/presence penalties
  ([#7569](../../sources/prs/vllm-ascend/PR-7569.md)).

## Why Triton here

Sampling is memory-/[Vector](../hardware/vector-unit.md)-bound over a wide `vocab`
dimension: count occurrences, mask, subtract penalties, compare against thresholds. The
Triton block model maps cleanly onto this (one program instance per block of the logits),
and keeping the reductions in [UB](../hardware/ub.md) avoids extra GM passes — the same
motivation as the AscendC fused kernels, expressed in Python.

```python
# Triton sketch (snippet-level) — apply repetition penalty, NOT a verbatim excerpt.
import triton, triton.language as tl

@triton.jit
def apply_penalty_kernel(logits_ptr, counts_ptr, penalty_ptr, n_vocab, BLOCK: tl.constexpr):
    row  = tl.program_id(0)                                 # one sequence
    offs = tl.arange(0, BLOCK)
    base = row * n_vocab
    for start in range(0, n_vocab, BLOCK):
        idx  = start + offs
        mask = idx < n_vocab
        logit = tl.load(logits_ptr + base + idx, mask=mask) # MTE -> UB
        cnt   = tl.load(counts_ptr + base + idx, mask=mask) # token bin counts
        pen   = tl.load(penalty_ptr + row)                  # per-request penalty
        logit = tl.where(cnt > 0, logit - pen * cnt, logit) # Vector, stays in UB
        tl.store(logits_ptr + base + idx, logit, mask=mask)
```

## Notes for adapters

- These run in the **decode** loop (once per step), so their latency is on the critical
  path — hence the Triton perf passes.
- Rejection sampling must stay **numerically faithful** to the reference (accepted/rejected
  token distribution); #5259 emphasizes preserving accuracy while optimizing.
- `bincount` counts token occurrences for the penalty; keep its output layout aligned with
  `apply_penalties`.
- Confidence `source-reported` (upstream code; PRs claim gains but disclose no numbers).

## See also

- [Triton-Ascend](../languages/triton-ascend.md) — the DSL.
- [Decode attention path](decode-attention.md) — sampling is the step after attention in decode.
