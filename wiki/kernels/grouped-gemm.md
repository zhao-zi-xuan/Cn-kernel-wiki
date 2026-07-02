---
id: kernel-grouped-gemm
title: "Grouped GEMM for MoE experts (DispatchGmmCombine)"
type: kernel
architectures:
  - davinci
  - ascend-910c
tags:
  - grouped-gemm
  - matmul
  - moe
  - operator-fusion
  - host-tiling
  - fine-grained-quantization
  - nz-format
confidence: source-reported
reproducibility: snippet
kernel_types:
  - grouped-gemm
  - matmul
  - moe
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-ub
  - technique-host-tiling
  - technique-operator-fusion
  - kernel-fused-moe
  - kernel-quantization-gemm
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-4139
  - pr-vllm-ascend-4790
  - pr-vllm-ascend-3804
evidence_basis: >
  Synthesized from upstream vllm-ascend PRs that build the grouped-matmul core of the
  MoE expert path (dispatch_gmm_combine / DispatchGmmCombineDecode /
  grouped_matmul_swiglu_quant_weight_nz_tensor_list). No PR discloses absolute numbers,
  so the performance_claims entry is qualitative and source-reported (D2).
performance_claims:
  - chip: ascend-910c
    dtype: w8a8
    shape: "per-expert token groups (shape not disclosed)"
    metric: "MoE expert grouped-matmul cost"
    value: "source-reported — a fused grouped GEMM computes all experts in one op; #4790 further removes weight permute (GMM1) and transpose (GMM2) so no layout pre-pass is needed (no absolute figure disclosed)"
    source_id: pr-vllm-ascend-4790
    measurement: source-reported
---

# Grouped GEMM for MoE experts

## Overview

The compute core of a Mixture-of-Experts layer is a **grouped GEMM**: after tokens are
routed to experts, each expert multiplies *its own* subset of tokens by *its own* weight
matrix. Rather than launching one matmul per expert, a grouped GEMM does them as a single
[Cube-unit](../hardware/cube-unit.md) operation with per-group offsets — this is the
`GMM` inside vllm-ascend's `dispatch_gmm_combine` / `DispatchGmmCombineDecode` operators.

This page covers the **matmul core**; the surrounding dispatch/combine fusion is described
in [Fused MoE (DispatchFFNCombine)](fused-moe.md), and the low-bit dequant in
[Quantization GEMM](quantization-gemm.md).

> Chip note: #4139 targets A3 (`ascend-910c`); the broader line is tagged generic
> `davinci`, so this page lists both.

## Structure

A grouped GEMM slices the token dimension by expert group and issues a matmul per slice,
accumulating in L0C. Two upstream refinements are worth noting:

- **Fused SwiGLU + per-token dequant epilogue** — the matmul epilogue applies SwiGLU and
  per-token dequant in place (`.../epilogue/block/block_epilogue_per_token_dequant_swiglu.h`,
  [#4790](../../sources/prs/vllm-ascend/PR-4790.md)), so the activation/dequant never
  round-trip GM (see [operator fusion](../techniques/operator-fusion.md)).
- **No layout pre-pass** — #4790 adapts the op so GMM1 weights/scales no longer need
  permuting and GMM2 weights no longer need transposing, removing a preprocessing step.
- **NZ-format quantized weights** — `grouped_matmul_swiglu_quant_weight_nz_tensor_list`
  takes weights already in the Cube's NZ (fractal) layout
  ([#3804](../../sources/prs/vllm-ascend/PR-3804.md)).

```cpp
// AscendC sketch (snippet-level) — synthesized from the PR file layout,
// NOT a verbatim upstream excerpt. One op over all expert groups.
for (int g = 0; g < numGroups; ++g) {              // g = expert
    int m0 = groupOffset[g], m = groupSize[g];     // this expert's token slice
    LoadWeightsNz(g);                              // MTE: expert-g weights (NZ) -> L1/L0B
    for (int tile = 0; tile < ceildiv(m, tileM); ++tile) {
        LoadTokens(m0, tile);                      // MTE: tokens -> L0A
        Mmad(accL0C, aTokens, bWeights, params);   // Cube: grouped matmul -> L0C
    }
    EpilogueSwigluDequant(g);                      // Vector: SwiGLU + per-token dequant (in UB)
}
```

## Notes for adapters

- The "grouped" part is bookkeeping: per-expert `(offset, size)` and weight base
  pointers; the inner matmul is ordinary Cube `Mmad`.
- Grouped GEMM pairs with [fine-grained quantization](../techniques/fine-grained-quantization.md):
  the epilogue dequant is per-token/per-group, not per-tensor.
- Confidence `source-reported` (upstream code, no benchmark).

## See also

- [Fused MoE (DispatchFFNCombine)](fused-moe.md) — the dispatch/combine wrapper around this GEMM.
- [Quantization GEMM](quantization-gemm.md) — the low-bit dequant detail.
- [Cube Unit](../hardware/cube-unit.md) — the matmul engine.
