---
id: pattern-fragmented-op-chain
title: "Fragmented op chain (many small ops invoked per-layer)"
type: pattern
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - operator-fusion
  - host-tiling
  - kv-cache
confidence: source-reported
symptoms:
  - "a single logical step is implemented as a sequence of framework ops (load + transform + store)"
  - "that sequence is invoked once per layer / per iteration, so overhead multiplies"
  - "lots of small kernel launches and GM round-trips for intermediates that are never reused"
candidate_techniques:
  - technique-operator-fusion
  - technique-host-tiling
related:
  - kernel-transpose-kv-cache-by-block
  - kernel-mla-preprocess
  - hw-mte
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-6366
  - pr-vllm-ascend-3226
---

# Fragmented op chain

## Symptom

A conceptually-single operation is expressed as a **chain of generic framework ops** —
typically *load → transform → store* — and that chain is re-invoked **for every layer**
(or every decode step). Each op launches separately and round-trips its intermediates
through GM, so both launch overhead and [MTE](../hardware/mte.md) traffic scale with the
number of layers.

## Diagnosis

Look for host-side (Python) sequences like `loadA(); transform(); storeB()` wrapped in a
per-layer loop, where the intermediates (`A`, `B`) are only consumed by the very next op.
Two concrete upstream examples:

- **KV-cache transpose** was `npu_paged_cache_load + transpose + _npu_reshape_and_cache`,
  called per layer ([#6366](../../sources/prs/vllm-ascend/PR-6366.md)).
- **MLA preprocessing** was Python-side tensor shuffling + copies before attention
  ([#3226](../../sources/prs/vllm-ascend/PR-3226.md)).

## Candidate techniques

1. **[Operator fusion](../techniques/operator-fusion.md)** — collapse the chain into one
   custom AscendC op so the intermediates stay in [UB](../hardware/ub.md). This is exactly
   what `transpose_kv_cache_by_block` and `mla_preprocess` did.
2. **[Host tiling](../techniques/host-tiling.md)** — the fused op then needs its own
   tiling (and possibly a full-load vs general path) to cover all shapes on-device.

## Relationship to other patterns

A fragmented op chain is one specific *cause* of a
[memory-bound layer](memory-bound-layer.md); fusing it is the fix in both framings.

## Notes

- Confidence `source-reported`: both example rewrites are from upstream code; no absolute
  speedups are asserted.
