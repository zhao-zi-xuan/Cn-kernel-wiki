---
id: hw-vector-unit
title: "Vector Unit (DaVinci AIV)"
type: hardware
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - vector-unit
  - aiv
  - ub
  - scalar-unit
confidence: source-reported
related:
  - hw-cube-unit
  - hw-ub
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-796
aliases:
  - Vector
  - AIV
  - "Vector单元"
  - vector-core
---

# Vector Unit (DaVinci AIV)

## Overview

The Vector unit (AIV) is the DaVinci AI Core's SIMD engine for **element-wise and
reduction operations** — activations (SwiGLU, GELU), normalization (RMSNorm/LayerNorm),
dequantization, masking, and the reductions that surround matmul. It is the counterpart
to the [Cube unit](cube-unit.md) (which does the matmuls): a typical kernel runs matmul
on Cube and everything around it on Vector, both reading and writing the
[Unified Buffer](ub.md).

In the AscendC data path the Vector unit sits between the two UB arrows:

```
GM → MTE → UB → [Vector / Cube] → UB → MTE → GM
```

Because Vector ops are UB-resident, they are cheap **relative to** the GM traffic that
feeds them — which is why fusing element-wise steps into a neighbouring kernel (so their
inputs never leave UB) is such a common Ascend optimization.

## Why it matters for kernels

- **Fuse element-wise work.** Activation / dequant / masking done on Vector immediately
  after a Cube matmul (results still in UB) avoids a GM round-trip. This is the pattern
  behind the fused-MoE and MLA-preprocess kernels.
- **Masking & gather.** Vocab-parallel embedding masks out-of-range token ids per
  tensor-parallel rank as a Vector op before the gather
  ([#796](../../sources/prs/vllm-ascend/PR-796.md)).
- **AIC/AIV split.** On DaVinci the Cube (AIC) and Vector (AIV) sides can run
  concurrently; well-pipelined kernels keep both busy rather than serializing.

```cpp
// AscendC sketch (snippet-level): a Vector element-wise op on a UB-resident tile.
LocalTensor<half> x = inQueue.DeQue<half>();   // in UB (produced by MTE or Cube)
LocalTensor<half> y = outQueue.AllocTensor<half>();
Add(y, x, bias);                               // Vector: element-wise, stays in UB
Relu(y, y);                                    // Vector: activation, still in UB
outQueue.EnQue(y);
inQueue.FreeTensor(x);
```

## Notes

- `source-reported`: role and data-path come from the official AscendC guide (doc stub
  here); the masking example is from upstream PR evidence. Per-generation Vector width /
  throughput are not asserted without a citable source.

## See also

- [Cube Unit](cube-unit.md) — the matmul engine; Vector handles the surrounding ops.
- [Unified Buffer (UB)](ub.md) — where Vector reads/writes its operands.
- [MTE](mte.md) — feeds UB from GM.
