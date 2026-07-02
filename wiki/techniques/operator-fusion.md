---
id: technique-operator-fusion
title: "Operator Fusion (fewer GM round-trips)"
type: technique
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - operator-fusion
  - kernel-fusion
  - ub
  - mte
confidence: source-reported
reproducibility: snippet
prerequisites:
  - hw-ub
  - hw-mte
related:
  - kernel-fused-moe
  - kernel-mla-preprocess
  - kernel-transpose-kv-cache-by-block
  - technique-double-buffering
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-6366
  - pr-vllm-ascend-3226
---

# Operator Fusion

## What it is

Fusing several logical operators into **one AscendC kernel** so their intermediate
tensors stay in the on-chip [Unified Buffer](../hardware/ub.md) instead of being written
back to and re-read from global memory (GM) between steps. The win is not fewer FLOPs —
it is **less [MTE](../hardware/mte.md) traffic and fewer kernel launches**.

## Why it helps on Ascend

The DaVinci data path is `GM → MTE → UB → Cube/Vector → UB → MTE → GM`. Every operator
boundary that crosses GM pays a full round-trip on the MTE, which is the bottleneck for
memory-bound layers. Keeping the producer's output resident in UB for the consumer
removes that round-trip.

## Where it shows up (upstream evidence)

- **Fused MoE / DispatchFFNCombine** — dispatch + grouped-FFN + combine in one op
  ([#3532](../../sources/prs/vllm-ascend/PR-3532.md); see [kernel](../kernels/fused-moe.md)).
- **MLA preprocess** — normalize + latent/query projections fused, replacing host-side
  shuffles ([#3226](../../sources/prs/vllm-ascend/PR-3226.md);
  [kernel](../kernels/mla-preprocess.md)).
- **transpose_kv_cache_by_block** — load + transpose + reshape/store folded into one
  block-wise op ([#6366](../../sources/prs/vllm-ascend/PR-6366.md);
  [kernel](../kernels/transpose-kv-cache-by-block.md)).

## Sketch

```cpp
// Un-fused: two ops, two GM round-trips.
//   op1: GM -> UB -> compute -> UB -> GM   (writes tmp)
//   op2: GM(tmp) -> UB -> compute -> UB -> GM
// Fused: one op, tmp never leaves UB.
CopyIn(tile);          // MTE: GM -> UB
Stage1(tile);          // Vector/Cube, result stays in UB
Stage2(tile);          // consumes Stage1 output directly from UB
CopyOut(tile);         // MTE: UB -> GM   (only the final result crosses GM)
```

## Caveats

- Fused kernels hold more live tensors in UB at once, so UB capacity bounds the tile
  size (see [host tiling](host-tiling.md)) and offsets must stay aligned
  (see [UB alignment](ub-alignment.md)).
- Confidence `source-reported`: the pattern and its instances come from upstream code;
  no absolute speedups are asserted (the PRs disclose none).
