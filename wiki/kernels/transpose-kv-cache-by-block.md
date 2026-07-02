---
id: kernel-transpose-kv-cache-by-block
title: "transpose_kv_cache_by_block (GQA KV-cache transpose)"
type: kernel
architectures:
  - ascend-910c
tags:
  - kv-cache
  - attention
  - operator-fusion
  - host-tiling
confidence: source-reported
reproducibility: snippet
kernel_types:
  - kv-cache
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-ub
  - hw-mte
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-6366
evidence_basis: >
  Synthesized from upstream vllm-ascend PR #6366, which adds the
  csrc/transpose_kv_cache_by_block AscendC operator. The PR describes replacing a
  three-op sequence with one fused kernel but discloses no absolute numbers, so the
  performance_claims entry is qualitative and source-reported (D2).
performance_claims:
  - chip: ascend-910c
    dtype: fp16
    shape: "per-layer KV cache blocks (shape not disclosed)"
    metric: "per-layer KV-cache transpose cost during GQA transfer"
    value: "source-reported reduction — one fused operator replaces the previous npu_paged_cache_load + transpose + _npu_reshape_and_cache sequence that was invoked per layer (no absolute figure disclosed)"
    source_id: pr-vllm-ascend-6366
    measurement: source-reported
---

# transpose_kv_cache_by_block

## Overview

`transpose_kv_cache_by_block` is a custom AscendC operator in vllm-ascend
(`csrc/transpose_kv_cache_by_block/`) that **transposes the KV-cache layout after a GQA
KV transfer**, needed when prefill and decode run at **heterogeneous tensor-parallel
sizes** ([#6366](../../sources/prs/vllm-ascend/PR-6366.md)). In that setup the KV cache
produced under one TP layout must be re-laid-out (transposed by block) before decode can
consume it.

The motivation is fusion: the previous implementation chained three ops —
`npu_paged_cache_load` + `transpose` + `_npu_reshape_and_cache` — and had to invoke them
**for every layer**, which is inefficient. The custom kernel folds that load → transpose
→ reshape/store sequence into a single block-wise operator.

## Structure

The device side ships two code paths — a **full-load** path (`op_kernel/full_load.h`) and
a **general** path (`op_kernel/general.h`) — selected by host-side tiling based on whether
a block's data fits on-chip in one shot. Both follow the AscendC
`CopyIn → (transpose) → CopyOut` skeleton, operating a block at a time so the transpose
stays in [UB](../hardware/ub.md) instead of bouncing through global memory per layer.

```cpp
// AscendC sketch (snippet-level) — synthesized from the PR file layout,
// NOT a verbatim upstream excerpt. One fused op per KV block.
for (int blk = 0; blk < numBlocks; ++blk) {
    CopyIn(blk);              // MTE: GM (paged KV) -> UB   (was: npu_paged_cache_load)
    TransposeBlock(blk);      // in-UB block transpose      (was: transpose)
    CopyOut(blk);             // MTE: UB -> GM (reshaped)   (was: _npu_reshape_and_cache)
}
// full_load.h : whole block resident in UB; general.h : tiled fallback.
```

## Notes for adapters

- The two paths (full-load vs general) trade UB residency against block size — pick per
  the host tiling; keep them behaviourally identical.
- This kernel exists specifically for the **heterogeneous prefill/decode TP** case; a
  homogeneous deployment does not need the transpose.
- Confidence `source-reported` (upstream code, no benchmark). Distributed KV *transfer*
  itself (the HCCL side) is out of scope here — this page covers only the on-device
  transpose kernel.

## See also

- [Unified Buffer (UB)](../hardware/ub.md) / [MTE](../hardware/mte.md) — the copy/transpose data path.
- [Fused MoE (DispatchFFNCombine)](fused-moe.md) — sibling fusion case study.
