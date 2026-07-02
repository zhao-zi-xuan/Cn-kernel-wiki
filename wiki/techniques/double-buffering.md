---
id: technique-double-buffering
title: "Double Buffering / Ping-Pong (MTE–compute overlap)"
type: technique
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - double-buffering
  - ping-pong-buffer
  - mte-overlap
  - software-pipeline
confidence: source-reported
reproducibility: snippet
prerequisites:
  - hw-ub
  - hw-mte
related:
  - technique-operator-fusion
  - technique-host-tiling
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
---

# Double Buffering / Ping-Pong

## What it is

Allocate **two UB buffers** for a stage and alternate ("ping-pong") between them so the
[MTE](../hardware/mte.md) can load tile *N+1* into one buffer while the Cube/Vector units
compute on tile *N* in the other. The copy latency disappears behind compute — the
standard `CopyIn → Compute → CopyOut` skeleton becomes a software pipeline.

## Why it matters on Ascend

Compute units cannot touch GM directly; every tile must first be moved into
[UB](../hardware/ub.md) by the MTE. Without overlap, each tile serializes as
*copy-in → wait → compute → copy-out*, and the MTE and compute engines are idle half the
time. Double buffering keeps both busy. AscendC's queue abstraction (`EnQue`/`DeQue` with
a depth-2 queue) expresses exactly this.

## Sketch

```cpp
// Depth-2 queue = double buffer. The framework overlaps DataCopy(t+1) with Compute(t).
TQue<TPosition::VECIN, /*BUFFER_NUM=*/2> inQueue;

for (int t = 0; t < numTiles; ++t) {
    LocalTensor<half> in = inQueue.AllocTensor<half>();
    DataCopy(in, gm[t * tileLen], tileLen);   // MTE: load tile t (overlaps compute of t-1)
    inQueue.EnQue(in);

    LocalTensor<half> x = inQueue.DeQue<half>();
    Compute(x);                               // Cube/Vector on tile t
    DataCopy(gmOut[t * tileLen], x, tileLen); // MTE: store
    inQueue.FreeTensor(x);
}
```

## Caveats

- Two buffers double the UB footprint of that stage, so it trades against tile size (see
  [host tiling](host-tiling.md)) — a bigger tile with no double buffer may lose to a
  smaller tile that overlaps.
- Only helps when the stage is actually MTE-bound and there is compute to hide behind.
- Confidence `source-reported`: the double-buffer/ping-pong model is from the official
  AscendC guide; no chip-specific numbers are asserted.
