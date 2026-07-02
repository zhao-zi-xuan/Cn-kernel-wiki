---
id: technique-host-tiling
title: "Host-side Tiling (two-stage AscendC program)"
type: technique
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - host-tiling
  - block-size-tuning
  - ub
confidence: source-reported
reproducibility: snippet
prerequisites:
  - hw-ub
related:
  - kernel-fused-moe
  - kernel-mla-preprocess
  - kernel-transpose-kv-cache-by-block
  - technique-operator-fusion
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3226
  - pr-vllm-ascend-6366
---

# Host-side Tiling

## What it is

An AscendC operator is a **two-stage program**: a *host-side tiling program* that decides
how to partition the problem, plus a *device-side kernel program* that executes each tile.
The host computes a **tiling struct** (tile shapes, loop counts, per-core work split,
buffer offsets) and passes it to the device kernel. Getting the tiling right is often the
difference between a correct-but-slow and a fast kernel.

## Why it matters on Ascend

- **UB capacity is the hard constraint.** A tile's live tensors (plus double buffers) must
  fit in the [Unified Buffer](../hardware/ub.md). The host picks tile sizes that maximize
  residency without overflowing UB.
- **Core work split.** The host divides blocks across AI Cores; uneven splits leave cores
  idle at the tail.
- **Path selection.** The host can pick between kernel variants — e.g.
  `transpose_kv_cache_by_block` chooses a **full-load** vs **general** device path based on
  whether a block fits on-chip ([#6366](../../sources/prs/vllm-ascend/PR-6366.md)).

## Sketch

```cpp
// Host side: compute tiling and hand it to the kernel.
TilingData t;
t.tileM   = ChooseTileM(problem, UB_CAPACITY);   // fit UB, keep double buffers
t.tileN   = ChooseTileN(problem, UB_CAPACITY);
t.coreNum = SplitAcrossCores(problem.blocks);    // balance work per AI Core
SetTilingData(context, t);

// Device side: kernel reads the tiling and loops over its assigned tiles.
for (int i = blockIdx; i < t.numTiles; i += t.coreNum) {
    CopyIn(i, t); Compute(i, t); CopyOut(i, t);
}
```

## Caveats

- Tiling structs live in `op_host/.../*_tiling.*`; MoE/MLA kernels ship substantial tiling
  logic there ([#3226](../../sources/prs/vllm-ascend/PR-3226.md)).
- Tile size is a tuning knob (`block-size-tuning`); the best value is shape- and
  chip-dependent.
- Confidence `source-reported`: structure from the official guide, instances from
  upstream code.
