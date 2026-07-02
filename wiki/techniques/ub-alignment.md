---
id: technique-ub-alignment
title: "UB Alignment (avoiding mis-aligned Vector access)"
type: technique
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - ub-alignment
  - ub
  - vector-unit
confidence: source-reported
reproducibility: snippet
prerequisites:
  - hw-ub
symptoms:
  - "vector error / incorrect results after changing a tile shape or offset"
  - "kernel works for some shapes but fails for others (non-multiple-of-alignment)"
related:
  - hw-ub
  - hw-vector-unit
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-6468
---

# UB Alignment

## What it is

Accesses into the [Unified Buffer](../hardware/ub.md) must satisfy the hardware's
alignment requirement (offsets and lengths as multiples of the required element/byte
granularity). **Mis-aligned UB access can produce vector errors or wrong results** —
a hazard that recurs whenever you change a tile shape, offset, or stride.

## Why it bites

Tiling math frequently produces tile sizes that are *not* a multiple of the alignment
(e.g. an odd number of tokens, or a hidden dim that isn't a round number). If the kernel
indexes UB at those raw offsets, the [Vector](../hardware/vector-unit.md) unit reads
across an alignment boundary and misbehaves. This class of bug appeared in upstream
fused-MoE work and had to be fixed
([#6468](../../sources/prs/vllm-ascend/PR-6468.md)).

## How to handle it

- **Pad tile shapes** up to the alignment; compute on the padded extent and mask/ignore
  the tail.
- Keep UB **offsets and strides** as multiples of the alignment when partitioning buffers.
- Treat alignment as an **invariant to re-check** every time the tiling changes — it is a
  common regression source.

## Sketch

```cpp
// BAD: raw length may not be aligned -> mis-aligned UB access.
DataCopy(ubTile, gm[offset], rawLen);

// GOOD: round the working extent up to the UB alignment; process the pad, mask the tail.
constexpr int ALIGN = 32;                       // element/byte granularity (illustrative)
int padLen = (rawLen + ALIGN - 1) / ALIGN * ALIGN;
DataCopy(ubTile, gm[offset], padLen);           // aligned copy
Compute(ubTile, padLen);
// only [0, rawLen) is valid; ignore/mask [rawLen, padLen)
```

## Caveats

- The exact alignment granularity is hardware/dtype dependent; this page states the
  *pattern*, not a specific byte count (no citable per-chip number).
- Confidence `source-reported`: hazard evidenced by the upstream fix; mechanism from the
  AscendC UB model.
