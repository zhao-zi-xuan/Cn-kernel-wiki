---
id: kernel-triton-rope
title: "Triton RoPE kernel (rope_dim != head_dim)"
type: kernel
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - rope
  - triton-ascend
  - operator-fusion
  - attention
confidence: source-reported
reproducibility: snippet
kernel_types:
  - rope
languages:
  - triton-ascend
  - python
related:
  - lang-triton-ascend
  - hw-vector-unit
  - hw-ub
  - technique-operator-fusion
  - kernel-decode-attention
sources:
  - pr-vllm-ascend-4413
  - pr-vllm-ascend-5918
evidence_basis: >
  Synthesized from the two upstream vllm-ascend PRs that add/optimize the Triton RoPE
  kernel (#4413 partial-rope fusion, #5918 high-performance rewrite + registration fixes).
  Both report single-op RoPE latency figures in their PR bodies (source-reported, quoted
  below — we did not run them; chip is not explicitly named).
performance_claims:
  - chip: davinci
    dtype: fp16
    shape: "single RoPE op latency, GLM4/MoE inference on Ascend NPU; chip not named in source"
    metric: "RoPE operator single-execution latency"
    value: "57.1 μs -> 9 μs, author-reported 6.34x / 84.24% latency reduction (source-reported, PR #5918)"
    source_id: pr-vllm-ascend-5918
    measurement: source-reported
  - chip: davinci
    dtype: fp16
    shape: "rope_dim != head_dim (partial rope), DS 3.2 piecewise aclgraph; chip not named in source"
    metric: "RoPE compute time (new Triton kernel)"
    value: "New Triton RoPE ~12 μs; replaces a 2×split + 2×rope + 2×slice + 2×concat sequence (source-reported, PR #4413; original timing shown via profiling image)"
    source_id: pr-vllm-ascend-4413
    measurement: source-reported
---

# Triton RoPE kernel

## Overview

Rotary Position Embedding (RoPE) rotates the query/key vectors by a position-dependent
angle before attention. When **`rope_dim != head_dim`** — only part of each head is
rotated — a naive framework implementation has to *split* the rotated/un-rotated parts
out, apply rope, then *slice* and *concat* them back. PR
[#4413](../../sources/prs/vllm-ascend/PR-4413.md) replaces that whole
`split → rope → slice → concat` dance with **one [Triton-Ascend](../languages/triton-ascend.md)
kernel** (`vllm_ascend/ops/triton/rope.py`), and [#5918](../../sources/prs/vllm-ascend/PR-5918.md)
follows up with a high-performance rewrite plus fixes to the kernel's registration/invocation.

This is a Triton counterpart to the AscendC fused kernels: the win is again
[operator fusion](../techniques/operator-fusion.md) — fewer ops, no intermediate GM
round-trips — expressed in the higher-level Python DSL rather than hand-written AscendC.

## Structure

The kernel is an element-wise [Vector](../hardware/vector-unit.md)-bound op: each Triton
program instance handles a block of the (token × head) grid, loads the relevant slice of
`q`/`k` and the `cos`/`sin` tables into [UB](../hardware/ub.md), applies the rotation only
on the `rope_dim` sub-range, and writes back — the un-rotated tail is passed through
without a separate split/concat.

```python
# Triton sketch (snippet-level) — partial RoPE, NOT a verbatim upstream excerpt.
import triton, triton.language as tl

@triton.jit
def rope_partial_kernel(q_ptr, cos_ptr, sin_ptr, out_ptr,
                        rope_dim, head_dim, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    d    = offs % head_dim
    rot  = d < rope_dim                       # only this sub-range is rotated
    q    = tl.load(q_ptr + offs)              # MTE -> UB
    cos  = tl.load(cos_ptr + offs, mask=rot, other=1.0)
    sin  = tl.load(sin_ptr + offs, mask=rot, other=0.0)
    # rotate pairs on [0, rope_dim); pass through [rope_dim, head_dim)
    q_rot = q * cos + rotate_half(q) * sin    # Vector math, stays in UB
    tl.store(out_ptr + offs, tl.where(rot, q_rot, q))   # no split/concat needed
```

## Notes for adapters

- The fusion win is specific to the **partial-rope (`rope_dim != head_dim`)** case; for
  full-head rope the split/concat overhead it removes doesn't exist.
- #5918's fixes (fake-impl name matching, torch-ops namespace, missing `self` in cos/sin
  slice) are a reminder that **op registration/invocation** is as error-prone as the
  kernel math for Triton-Ascend custom ops.
- **Reported numbers** (source-reported, not run by this wiki): #5918 reports the single
  RoPE op latency dropping 57.1 μs → 9 μs (6.34x / 84.24%); #4413 reports the new Triton
  RoPE at ~12 μs. Chip is not named in either PR. Confidence stays `source-reported`
  (single upstream source each, no official-doc corroboration).

## See also

- [Triton-Ascend](../languages/triton-ascend.md) — the DSL and its `tl.*` → hardware mapping.
- [Decode attention path](decode-attention.md) — where RoPE sits in the attention pre-path.
- [Operator fusion](../techniques/operator-fusion.md) — the underlying optimization.
