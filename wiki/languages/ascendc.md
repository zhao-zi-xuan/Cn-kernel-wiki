---
id: lang-ascendc
title: "AscendC (C++ kernel language for DaVinci)"
type: language
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - ascendc
  - cpp
  - host-tiling
  - ub
confidence: source-reported
reproducibility: snippet
aliases:
  - AscendC
  - "Ascend C"
  - ascend-c
related:
  - lang-triton-ascend
  - hw-cube-unit
  - hw-ub
  - technique-host-tiling
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-233
  - pr-vllm-ascend-3226
  - pr-vllm-ascend-6366
---

# AscendC

## What it is

AscendC is the **C++-like kernel language** for programming the DaVinci AI Core directly.
It is the native, lowest-level path for custom operators on Ascend and the language behind
almost every `csrc/**/op_kernel/*.cpp` kernel in vllm-ascend (e.g.
[#233](../../sources/prs/vllm-ascend/PR-233.md) wired the first custom AscendC kernels into
the runtime).

## Programming model

An AscendC operator is a **two-stage program** (see
[host tiling](../techniques/host-tiling.md)):

- **Host side** — computes a *tiling struct* (tile shapes, core split, buffer offsets) and
  the op definition/prototype (`op_host/*.cpp`, `*_tiling.*`).
- **Device side** — the kernel that runs on each AI Core (`op_kernel/*.cpp`), moving data
  along `GM → MTE → UB/L1 → Cube/Vector → UB → MTE → GM`.

Key building blocks:

- `GlobalTensor<T>` / `LocalTensor<T>` — handles to GM and on-chip
  ([UB](../hardware/ub.md)/L1) tensors.
- `TQue<TPosition, BUFFER_NUM>` — pipe queues; `BUFFER_NUM=2` gives
  [double buffering](../techniques/double-buffering.md).
- `DataCopy` — MTE transfers; `Mmad` — [Cube](../hardware/cube-unit.md) matmul; plus Vector
  intrinsics (`Add`, `Muls`, `Relu`, …) for element-wise/reduction work.

## Sketch

```cpp
// Minimal AscendC device kernel: CopyIn -> Compute -> CopyOut with a depth-2 queue.
class MyKernel {
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, const TilingData& t) {
        srcGm.SetGlobalBuffer((__gm__ half*)src);
        dstGm.SetGlobalBuffer((__gm__ half*)dst);
        pipe.InitBuffer(inQ, /*BUFFER_NUM=*/2, t.tileLen * sizeof(half));
        tileLen = t.tileLen; numTiles = t.numTiles;
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < numTiles; ++i) {
            LocalTensor<half> x = inQ.AllocTensor<half>();
            DataCopy(x, srcGm[i * tileLen], tileLen);   // MTE: GM -> UB
            inQ.EnQue(x);
            LocalTensor<half> y = inQ.DeQue<half>();
            Relu(y, y);                                 // Vector op, stays in UB
            DataCopy(dstGm[i * tileLen], y, tileLen);   // MTE: UB -> GM
            inQ.FreeTensor(y);
        }
    }
private:
    TPipe pipe; TQue<TPosition::VECIN, 2> inQ;
    GlobalTensor<half> srcGm, dstGm; int tileLen, numTiles;
};
```

## When to use it

- Maximum control / peak performance for custom device operators (fused MoE, MLA
  preprocess, KV-cache transpose all use it).
- The cost is verbosity and manual tiling/alignment — contrast with
  [Triton-Ascend](triton-ascend.md), which trades some control for a higher-level Python
  DSL.

## Notes

- Confidence `source-reported`: model from the official AscendC guide; usage from upstream
  code. This page is a snippet-level orientation, not a full API reference.
