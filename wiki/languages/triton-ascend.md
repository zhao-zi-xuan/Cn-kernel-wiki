---
id: lang-triton-ascend
title: "Triton-Ascend (Triton DSL on Ascend NPU)"
type: language
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - triton-ascend
  - python
  - vector-unit
confidence: source-reported
reproducibility: snippet
aliases:
  - Triton-Ascend
  - "triton ascend"
  - triton-npu
related:
  - lang-ascendc
  - hw-vector-unit
  - hw-ub
  - kernel-triton-rope
  - kernel-triton-sampling
sources:
  - pr-vllm-ascend-4304
  - pr-vllm-ascend-4413
  - pr-vllm-ascend-4595
  - pr-vllm-ascend-5918
---

# Triton-Ascend

## What it is

Triton-Ascend is the **Triton DSL targeting the Ascend NPU** — the same Python,
`@triton.jit`-decorated, block-programming model used on GPUs, compiled to run on DaVinci
cores. In vllm-ascend these kernels live under `vllm_ascend/ops/triton/` and are the
higher-level alternative to hand-written [AscendC](ascendc.md): you write per-block Python
with `tl.load` / `tl.store` / `tl.*` math and let the compiler handle the device details.

## Where it is used (upstream evidence)

Triton-Ascend is the vehicle for a growing set of vllm-ascend ops, especially
element-wise / normalization / sampling kernels that map well to the block model:

- partial **RoPE** ([#4413](../../sources/prs/vllm-ascend/PR-4413.md), [#5918](../../sources/prs/vllm-ascend/PR-5918.md))
  — see the [Triton RoPE kernel](../kernels/triton-rope.md) page.
- **sampling** — rejection sampling ([#5259](../../sources/prs/vllm-ascend/PR-5259.md)) and
  penalties ([#7569](../../sources/prs/vllm-ascend/PR-7569.md)); see the
  [Triton sampling kernels](../kernels/triton-sampling.md) page.
- `fused_gdn_gating` ([#4304](../../sources/prs/vllm-ascend/PR-4304.md)) and **l2norm**
  ([#4595](../../sources/prs/vllm-ascend/PR-4595.md)) — other Triton ops not yet given their own page.

## Programming model

- A kernel is a Python function decorated with `@triton.jit`; a **grid** of program
  instances each handle one block.
- `tl.arange` + a `mask` select this block's elements; `tl.load`/`tl.store` move them
  (the compiler maps these to the [MTE](../hardware/mte.md)→[UB](../hardware/ub.md) path);
  `tl.*` ops run on the [Vector](../hardware/vector-unit.md) unit.
- `BLOCK` size is a `tl.constexpr` autotuning knob — the Triton analogue of AscendC tiling.

## Sketch

```python
import triton
import triton.language as tl

@triton.jit
def add_bias_kernel(x_ptr, b_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)                     # this block's index in the grid
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                             # guard the ragged tail
    x = tl.load(x_ptr + offs, mask=mask)        # MTE -> UB (compiler-managed)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + b, mask=mask)    # Vector add, then UB -> MTE

# host launch: grid = (triton.cdiv(n, BLOCK),)
```

## When to use it vs AscendC

- **Triton-Ascend**: faster to write, portable-looking, good for element-wise /
  reduction / sampling kernels; less manual tiling and alignment bookkeeping.
- **[AscendC](ascendc.md)**: maximum control and peak performance for the heavy fused
  operators (MoE, MLA), at the cost of verbosity.

## Notes

- Confidence `source-reported`: existence and usage from upstream PRs; the snippet is a
  standard Triton skeleton, not a verbatim upstream excerpt. The exact set of supported
  `tl.*` ops / autotune behaviour on Ascend is not asserted here without a citable doc.
