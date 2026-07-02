---
id: kernel-fused-moe
title: "Fused MoE on Ascend (DispatchFFNCombine)"
type: kernel
architectures:
  - davinci
tags:
  - fused-moe
  - moe
  - operator-fusion
  - host-tiling
  - fine-grained-quantization
  - w8a8
  - w4a8
  - l0c
  - ub
confidence: source-reported
reproducibility: snippet
kernel_types:
  - fused-moe
  - moe
  - quantization
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-ub
  - hw-mte
  - kernel-mla-preprocess
  - kernel-grouped-gemm
  - kernel-quantization-gemm
  - technique-operator-fusion
  - technique-host-tiling
  - technique-double-buffering
  - technique-ub-alignment
  - technique-fine-grained-quantization
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-6468
  - pr-vllm-ascend-7779
  - pr-vllm-ascend-8902
evidence_basis: >
  Synthesized from four upstream vllm-ascend PRs that build and evolve the
  csrc/dispatch_ffn_combine AscendC operator. None of the source PRs disclose
  absolute performance numbers, so every performance_claims entry below is a
  qualitative, source-reported statement — no figure is invented (D2).
performance_claims:
  - chip: davinci
    dtype: w8a8
    shape: "not disclosed by source"
    metric: "DispatchFFNCombine operator execution latency"
    value: "source-reported reduction — merges token+scale transmission and uses tile-granularity communication to cut inter-core waiting bubbles (no absolute figure disclosed)"
    source_id: pr-vllm-ascend-6468
    measurement: source-reported
  - chip: davinci
    dtype: w4a8
    shape: "not disclosed by source"
    metric: "W4A8 MoE dispatch-FFN-combine inference performance"
    value: "source-reported improvement via communication/computation overlapping in the fused operator (no absolute figure disclosed)"
    source_id: pr-vllm-ascend-7779
    measurement: source-reported
---

# Fused MoE on Ascend (DispatchFFNCombine)

## Overview

`DispatchFFNCombine` is a custom AscendC operator in vllm-ascend
(`csrc/dispatch_ffn_combine/`) that fuses the three stages of a Mixture-of-Experts
layer — **expert dispatch → grouped FFN (quantized matmul + activation) → combine** —
into a single on-device operator. Fusing them avoids materializing the intermediate
dispatched/expanded activations back to global memory (GM) between stages, which is
where a naive multi-operator MoE path spends most of its time: repeated
GM↔UB round-trips (MTE traffic) and inter-core synchronization bubbles.

The kernel targets **quantized** MoE (W8A8 and, later, W4A8), so the FFN matmul runs on
the Cube unit with per-group / fine-grained dequantization, while dispatch and combine
are memory- and communication-bound stages that must overlap with that compute.

> Chip note: the source PRs tag only the generic DaVinci AI Core — none names a specific
> chip (910B/910C), so this page stays at `davinci` rather than inventing one.

## Why fuse

A non-fused MoE layer issues separate dispatch, matmul, activation, and combine
operators. Each hand-off crosses GM, so the tokens (and their quantization scales) are
written out and read back multiple times. On Ascend that means:

- **MTE pressure** — every GM↔UB copy competes for [MTE](../hardware/mte.md) bandwidth into the [Unified Buffer](../hardware/ub.md).
- **Sync bubbles** — cores stall waiting for the whole tile to arrive before the next
  stage can start.

Fusing lets the operator keep intermediates in UB / L1 and pipeline the stages at
**tile granularity**, so combine on one tile overlaps with FFN compute on the next.

## Evolution (upstream PR line)

| PR | What it does | Why it matters |
|----|--------------|----------------|
| [#3532](../../sources/prs/vllm-ascend/PR-3532.md) | Introduces the full host + device `dispatch_ffn_combine` kernel (tiling, MoE routing, W8A8 path) and wires it into the runtime | Establishes the fused operator |
| [#6468](../../sources/prs/vllm-ascend/PR-6468.md) | Perf pass: merge token+scale transmission; decouple multi-core dependencies; tile-granularity communication to reduce combine bubbles | Cuts latency of the existing operator |
| [#7779](../../sources/prs/vllm-ascend/PR-7779.md) | Adds a **W4A8** fused variant with communication/computation overlapping | Extends fusion to lower-bit quant |
| [#8902](../../sources/prs/vllm-ascend/PR-8902.md) | Updates the fusion kernel and fixes a known performance-degradation regression | Keeps the optimized path stable |

## Structure & the UB-alignment hazard

The device side is a classic Ascend fused pipeline: `CopyIn` (MTE: GM→UB) → `Compute`
(Cube grouped-matmul FFN + Vector activation/dequant) → `CopyOut` (MTE: UB→GM), unrolled
across experts/tiles with double buffering so the three phases overlap.

One recurring hazard in this family of kernels is **UB alignment**: mis-aligned UB
accesses have produced vector errors in related upstream work and must be fixed by
padding tile shapes to the required alignment. Treat UB offsets/strides as an
invariant to check when adapting the tiling.

```cpp
// AscendC sketch (snippet-level) — synthesized from the PR descriptions above,
// NOT a verbatim upstream excerpt. Illustrates the fused dispatch/FFN/combine
// pipeline with tile-granularity overlap and per-group dequant.
for (int tile = 0; tile < numTiles; ++tile) {
    // Stage 1 — dispatch: gather this tile's tokens + quant scales into UB.
    // (merged token+scale transfer, per PR #6468)
    DispatchCopyIn(tile);                 // MTE: GM -> UB  (UB offsets MUST be aligned)

    // Stage 2 — grouped FFN on the Cube unit with fine-grained dequant.
    GroupedMatmulDequant(tile);           // Cube: W8A8/W4A8 -> L0C accumulate
    Activation(tile);                     // Vector: SwiGLU etc.

    // Stage 3 — combine: reduce expert outputs back for this tile.
    // Tile-granularity so combine(tile) overlaps FFN(tile+1) and hides bubbles.
    CombineCopyOut(tile);                 // MTE: UB -> GM
}
```

## Notes for adapters

- The W8A8 path (#3532) and W4A8 path (#7779) differ mainly in the dequant granularity
  and the matmul input width; the dispatch/combine scaffolding is shared.
- Because the source is upstream code without an accompanying benchmark report,
  confidence is `source-reported`, not `verified`. Promote to `verified` only if an
  official doc plus upstream code jointly confirm a claim.

## See also

- [Cube Unit](../hardware/cube-unit.md) — the matmul engine the FFN stage runs on.
- [AscendC Programming Guide](../../sources/docs/ascendc-programming-guide.md) — UB/MTE/tiling model.
