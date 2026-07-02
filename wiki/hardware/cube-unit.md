---
id: hw-cube-unit
title: "Cube Unit (DaVinci matmul engine)"
type: hardware
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - cube-unit
  - l0a
  - l0b
  - l0c
  - nz-format
confidence: source-reported
related:
  - kernel-fused-moe
  - hw-ub
  - hw-mte
  - hw-vector-unit
sources:
  - doc-ascendc-programming-guide
aliases:
  - Cube
  - "Cube单元"
  - matmul-unit
---

# Cube Unit

## Overview

The Cube unit is the DaVinci AI Core's matrix-multiply engine (the rough analogue of
an NVIDIA tensor core). It consumes operands from the L0A / L0B buffers and accumulates
into L0C, typically requiring the **NZ (fractal) layout** for its inputs. This is a
scaffold page — expand with: supported tile shapes per generation, dtype matrix
(fp16/bf16/int8), and the Cube/Vector hand-off via UB.

```cpp
// AscendC sketch (snippet-level): Cube mmad accumulating into L0C
// (illustrative — replace with a compilable excerpt from an upstream kernel)
LocalTensor<half> a0 = inQueueA.DeQue<half>();   // in L0A (NZ)
LocalTensor<half> b0 = inQueueB.DeQue<half>();   // in L0B (NZ)
Mmad(c0, a0, b0, mmadParams);                    // accumulate in L0C
```
