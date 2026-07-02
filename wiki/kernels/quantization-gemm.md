---
id: kernel-quantization-gemm
title: "Quantized GEMM (W8A8 / W4A8 with per-group dequant)"
type: kernel
architectures:
  - davinci
  - ascend-910c
tags:
  - quantization
  - matmul
  - moe
  - fine-grained-quantization
  - w8a8
  - w4a8
  - cube-unit
confidence: source-reported
reproducibility: snippet
kernel_types:
  - quantization
  - matmul
  - moe
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-vector-unit
  - technique-fine-grained-quantization
  - pattern-quantization-accuracy-drop
  - kernel-fused-moe
  - kernel-grouped-gemm
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-7779
  - pr-vllm-ascend-8902
evidence_basis: >
  Synthesized from the upstream vllm-ascend PRs that add the W8A8 (#3532) and W4A8 (#7779)
  quantized matmul paths and a later fix (#8902). No PR discloses absolute numbers, so the
  performance_claims entry is qualitative and source-reported (D2).
performance_claims:
  - chip: ascend-910c
    dtype: w4a8
    shape: "MoE FFN matmul (shape not disclosed)"
    metric: "quantized MoE matmul bandwidth/latency"
    value: "source-reported — INT4/INT8 weights cut MTE traffic vs fp16, with communication/computation overlap in the W4A8 fused path (#7779); no absolute figure disclosed"
    source_id: pr-vllm-ascend-7779
    measurement: source-reported
---

# Quantized GEMM (W8A8 / W4A8)

## Overview

The heavy matmuls in MoE/FFN run in **low-bit integer** form on Ascend: **W8A8** (INT8
weights + INT8 activations) and, later, **W4A8** (INT4 weights). The
[Cube unit](../hardware/cube-unit.md) accumulates in integer; a **per-group scale** is
applied during dequant on the [Vector unit](../hardware/vector-unit.md), inside the matmul
epilogue. This page is the *quantized-matmul* view of the same operators described by
[Grouped GEMM](grouped-gemm.md) and [Fused MoE](fused-moe.md).

## Why quantize the GEMM

- **Bandwidth.** INT4/INT8 weights are 2–4× smaller than fp16, directly cutting the
  [MTE](../hardware/mte.md) traffic that bottlenecks MoE (see the
  [memory-bound-layer](../patterns/memory-bound-layer.md) pattern).
- **Accuracy.** A single per-tensor scale is too coarse and drops quality (see the
  [quantization-accuracy-drop](../patterns/quantization-accuracy-drop.md) pattern);
  **per-group** scales fix this — the technique is
  [fine-grained quantization](../techniques/fine-grained-quantization.md).

## Upstream line

| PR | What it does |
|----|--------------|
| [#3532](../../sources/prs/vllm-ascend/PR-3532.md) | W8A8 fused MoE with per-group dequant (establishes the quantized path) |
| [#7779](../../sources/prs/vllm-ascend/PR-7779.md) | Adds a **W4A8** fused variant with communication/computation overlap |
| [#8902](../../sources/prs/vllm-ascend/PR-8902.md) | Updates the fusion kernel and fixes a known perf-degradation regression |

## Sketch

```cpp
// AscendC sketch (snippet-level) — integer matmul + per-group dequant epilogue.
// NOT a verbatim upstream excerpt.
Mmad(accI32, aInt8, bInt4Or8, mmadParams);         // Cube: integer accumulate in L0C
for (int g = 0; g < numGroups; ++g) {              // per-group, NOT per-tensor
    // dequant = accumulate * (act_scale[g] * weight_scale[g]); Vector, stays in UB
    Muls(outFp16[g], accI32[g], combinedScale[g]);
}
// output flows into the next fused stage (SwiGLU / combine) with no GM round-trip
```

## Notes for adapters

- W8A8 vs W4A8 differ in weight width and dequant granularity; the surrounding fusion is
  shared but keep the dequant path in sync (a mismatch is exactly the class of bug #8902
  addresses).
- The scale layout (per-token / per-group) must match how the weights were quantized
  offline.
- Confidence `source-reported` (upstream code, no accuracy/throughput numbers disclosed).

## See also

- [Fine-grained quantization](../techniques/fine-grained-quantization.md) — the per-group scheme.
- [Grouped GEMM](grouped-gemm.md) / [Fused MoE](fused-moe.md) — the operators this runs inside.
