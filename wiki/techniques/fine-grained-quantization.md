---
id: technique-fine-grained-quantization
title: "Fine-grained Quantization (per-group dequant in fused matmul)"
type: technique
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - fine-grained-quantization
  - quantization
  - w8a8
  - w4a8
  - cube-unit
confidence: source-reported
reproducibility: snippet
prerequisites:
  - hw-cube-unit
  - hw-ub
related:
  - kernel-fused-moe
  - technique-operator-fusion
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-7779
---

# Fine-grained Quantization

## What it is

Running the FFN/MoE matmuls in **low-bit integer** form (W8A8, and later W4A8) with
**per-group scales** — each block of weights/activations carries its own dequant scale,
rather than one scale per tensor. The [Cube unit](../hardware/cube-unit.md) accumulates
in integer, and the per-group scale is applied on the [Vector](../hardware/vector-unit.md)
side during dequant, keeping the intermediate in [UB](../hardware/ub.md).

## Why it matters on Ascend

- **Bandwidth & capacity.** W8A8/W4A8 weights are 2–4× smaller than fp16, cutting the
  [MTE](../hardware/mte.md) traffic that dominates MoE, and letting more of a tile stay
  resident in UB.
- **Accuracy.** Per-group (fine-grained) scales preserve accuracy far better than a single
  per-tensor scale, which is what makes low-bit MoE usable.
- **Fusion-friendly.** Dequant is an element-wise Vector step, so it fuses naturally into
  the matmul epilogue rather than becoming a separate GM round-trip
  (see [operator fusion](operator-fusion.md)).

## Where it shows up (upstream evidence)

- **DispatchFFNCombine W8A8** — the original fused MoE operator
  ([#3532](../../sources/prs/vllm-ascend/PR-3532.md)).
- **W4A8 fused variant** — adds a 4-bit-weight path with comm/compute overlap
  ([#7779](../../sources/prs/vllm-ascend/PR-7779.md); see
  [kernel](../kernels/fused-moe.md)).

## Sketch

```cpp
// Per-group dequant fused into the matmul epilogue (snippet-level).
Mmad(accI32, aInt8, bInt8, mmadParams);        // Cube: integer accumulate in L0C
// dequant with a per-group scale (not one global scale) — Vector, stays in UB
for (int g = 0; g < numGroups; ++g)
    Muls(outFp16[g], accI32[g], groupScale[g]); // apply this group's scale
// result flows straight into the next fused stage; no GM round-trip for the tmp
```

## Caveats

- W8A8 and W4A8 differ in weight width and dequant granularity; the dispatch/combine
  scaffolding is shared but the dequant path is not (keep them in sync).
- Confidence `source-reported`: variants and their existence are from upstream code; no
  accuracy/throughput numbers are asserted (the PRs disclose none).
