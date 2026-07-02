---
id: hw-ub
title: "Unified Buffer (UB)"
type: hardware
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - ub
  - mte
  - ub-alignment
  - vector-unit
confidence: source-reported
related:
  - hw-cube-unit
  - hw-mte
  - hw-vector-unit
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-6468
aliases:
  - UB
  - "Unified Buffer"
  - "统一缓冲区"
  - unified-buffer
---

# Unified Buffer (UB)

## Overview

The Unified Buffer (UB) is the DaVinci AI Core's large on-chip scratchpad that feeds the
**Vector** unit and acts as the staging area on the core-local side of the data path.
In the AscendC data-flow model the canonical path is:

```
GM → MTE → UB / L1 → Cube / Vector → UB → MTE → GM
```

Operands are moved from global memory (GM) into UB by the MTE (see [MTE](mte.md)),
computed on by the Vector unit (and, for matmul, staged through L1 → L0 into the
[Cube unit](cube-unit.md)), and results are written back out through UB → GM. Keeping
intermediates resident in UB instead of round-tripping through GM is the single biggest
lever for memory-bound kernels — it is exactly what operator fusion buys (see
[Fused MoE](../kernels/fused-moe.md)).

## Why it matters for kernels

- **Residency = fewer GM round-trips.** A fused kernel that keeps activations in UB
  between stages avoids re-reading/re-writing GM, cutting MTE traffic.
- **Capacity bounds tiling.** UB size caps how large a tile can be; host-side tiling
  must partition work so each tile's live tensors fit in UB alongside double buffers.
- **UB alignment is an invariant.** Mis-aligned UB accesses can produce vector errors;
  this class of bug showed up in upstream fused-MoE work
  ([#6468](../../sources/prs/vllm-ascend/PR-6468.md)) and is fixed by padding tile
  shapes/offsets to the required alignment. Treat UB offsets and strides as something to
  verify whenever you change a tiling.

```cpp
// AscendC sketch (snippet-level): allocate a UB tile, copy in from GM, compute, copy out.
// Offsets/lengths must satisfy the UB alignment requirement (pad if needed).
LocalTensor<half> ubTile = inQueue.AllocTensor<half>();   // resident in UB
DataCopy(ubTile, gmSrc[tileOffset], tileLen);             // MTE: GM -> UB (aligned!)
Add(ubTile, ubTile, bias);                                // Vector op, stays in UB
DataCopy(gmDst[tileOffset], ubTile, tileLen);             // MTE: UB -> GM
inQueue.FreeTensor(ubTile);
```

## Notes

- This page is `source-reported`: the data-path model comes from the official AscendC
  guide (a doc stub here) and the alignment hazard from upstream PR evidence. Exact UB
  capacity per chip generation is not asserted until a citable source is added.

## See also

- [MTE](mte.md) — the engine that moves data between GM and UB.
- [Cube Unit](cube-unit.md) — matmul engine fed via L1/L0, hands results back through UB.
- [Fused MoE (DispatchFFNCombine)](../kernels/fused-moe.md) — fusion that exploits UB residency.
