---
id: kernel-vocab-parallel-embedding
title: "Vocab-parallel embedding mask kernel (get_masked_input_and_mask)"
type: kernel
architectures:
  - davinci
tags:
  - embedding
  - vector-unit
  - operator-fusion
confidence: source-reported
reproducibility: snippet
kernel_types:
  - embedding
languages:
  - ascendc
  - cpp
related:
  - hw-vector-unit
  - hw-ub
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-796
evidence_basis: >
  Synthesized from upstream vllm-ascend PR #796, which adds the
  csrc/kernels/get_masked_input_and_mask_kernel AscendC kernel for vocab-parallel
  embedding. The PR shows test/benchmark screenshots but no machine-readable numbers, so
  the performance_claims entry is qualitative and source-reported (D2).
performance_claims:
  - chip: davinci
    dtype: fp16
    shape: "vocab range per TP rank (shape not disclosed)"
    metric: "vocab-parallel embedding input masking"
    value: "source-reported — replaces host-side masking with a single AscendC kernel; PR provides benchmark/test screenshots but no machine-readable figure"
    source_id: pr-vllm-ascend-796
    measurement: source-reported
---

# Vocab-parallel embedding mask kernel

## Overview

When the embedding table is **sharded across tensor-parallel (TP) ranks**
(vocab-parallel embedding), each rank owns only a contiguous slice of the vocabulary.
Before gathering, every token id must be checked against this rank's `[start, end)` range:
ids inside the range are shifted to a local index, ids outside are masked to zero (and
contributed by other ranks after the all-reduce). PR
[#796](../../sources/prs/vllm-ascend/PR-796.md) adds a custom AscendC kernel,
`get_masked_input_and_mask` (`csrc/kernels/get_masked_input_and_mask_kernel.cpp`), that
computes the masked local indices and the validity mask in one on-device
[Vector](../hardware/vector-unit.md) pass.

## Structure

This is a memory-/vector-bound elementwise kernel: load the token ids into
[UB](../hardware/ub.md), compare against the rank's vocab bounds, emit the shifted local
index and a 0/1 mask, write both back. No matmul — it runs entirely on the Vector unit.

```cpp
// AscendC sketch (snippet-level) — synthesized from the kernel's purpose,
// NOT a verbatim upstream excerpt.
LocalTensor<int32_t> ids = inQueue.DeQue<int32_t>();      // token ids in UB
LocalTensor<int32_t> local = outIdx.AllocTensor<int32_t>();
LocalTensor<int32_t> mask  = outMask.AllocTensor<int32_t>();
// in-range = (ids >= vocabStart) && (ids < vocabEnd)   -> Vector compares/selects
CompareRange(mask, ids, vocabStart, vocabEnd);           // 1 if in this rank's slice
Sub(local, ids, vocabStart);                             // shift to local index
Select(local, mask, local, /*else=*/0);                  // masked ids -> 0
outIdx.EnQue(local); outMask.EnQue(mask);
inQueue.FreeTensor(ids);
```

## Notes for adapters

- The mask output feeds the subsequent gather + all-reduce; keep the masked-to-zero
  convention consistent with the reduction that follows.
- Pure Vector/elementwise — a good minimal example of a non-matmul custom AscendC op.
- Confidence `source-reported` (upstream code; PR shows screenshots, no citable number).

## See also

- [Vector Unit](../hardware/vector-unit.md) — the engine this kernel runs on.
- [Unified Buffer (UB)](../hardware/ub.md) — where ids/mask are staged.
