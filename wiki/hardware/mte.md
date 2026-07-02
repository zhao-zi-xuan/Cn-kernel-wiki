---
id: hw-mte
title: "MTE (Memory Transfer Engine)"
type: hardware
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - mte
  - ub
  - gm
  - mte-overlap
confidence: source-reported
related:
  - hw-ub
  - hw-cube-unit
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
aliases:
  - MTE
  - "Memory Transfer Engine"
  - "搬运单元"
  - data-movement
---

# MTE (Memory Transfer Engine)

## Overview

The MTE is the DaVinci AI Core's data-movement engine — the hardware that copies tensors
between global memory (GM) and the on-chip buffers ([UB](ub.md), L1, and the L0 operand
buffers of the [Cube unit](cube-unit.md)). In the AscendC data path it is the arrows in:

```
GM → [MTE] → UB / L1 → Cube / Vector → UB → [MTE] → GM
```

Because compute units cannot read GM directly, **every operand crosses the MTE at least
once**. On memory-bound kernels the MTE, not the Cube/Vector units, is the bottleneck,
so the optimization goal is to (a) move less data and (b) overlap the moves with compute.

## Why it matters for kernels

- **MTE traffic is the cost of un-fused pipelines.** Each operator hand-off through GM is
  an extra MTE round-trip; fusing stages to keep data in UB removes those copies (see
  [Fused MoE](../kernels/fused-moe.md)).
- **Overlap hides latency.** With double buffering, the MTE loads tile *N+1* while the
  Cube/Vector units compute on tile *N* — the copy latency disappears behind compute.
  This `CopyIn → Compute → CopyOut` pipeline is the standard AscendC kernel skeleton.
- **Merge transfers to cut launches.** Upstream fused-MoE work merged the transmission of
  tokens and their quantization scales into fewer MTE moves
  ([#6468](../../sources/prs/vllm-ascend/PR-6468.md)); combining logically-separate copies
  reduces per-transfer overhead.

```cpp
// AscendC sketch (snippet-level): double-buffered MTE overlap.
// Ping-pong so the MTE copy of the next tile hides behind compute of the current one.
for (int t = 0; t < numTiles; ++t) {
    LocalTensor<half> in = inQueue.AllocTensor<half>();
    DataCopy(in, gmSrc[t * tileLen], tileLen);   // MTE: GM -> UB
    inQueue.EnQue(in);                           // hand to compute stage
    LocalTensor<half> x = inQueue.DeQue<half>();
    Compute(x);                                  // Vector/Cube overlaps next DataCopy
    DataCopy(gmDst[t * tileLen], x, tileLen);    // MTE: UB -> GM
    inQueue.FreeTensor(x);
}
```

## Notes

- `source-reported`: the GM↔UB data-path role is from the official AscendC guide (doc
  stub here); the merged-transfer example is from upstream PR evidence. MTE channel counts
  / bandwidth per generation are not asserted without a citable source.

## See also

- [Unified Buffer (UB)](ub.md) — the on-chip destination of most MTE loads.
- [Cube Unit](cube-unit.md) — its L0A/L0B operands are staged in via MTE.
- [Fused MoE (DispatchFFNCombine)](../kernels/fused-moe.md) — reduces MTE traffic by fusing.
