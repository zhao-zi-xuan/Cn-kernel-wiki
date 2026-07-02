---
id: kernel-mla-preprocess
title: "MLA Preprocess kernel (Ascend)"
type: kernel
architectures:
  - davinci
tags:
  - mla
  - attention
  - operator-fusion
  - host-tiling
  - rmsnorm
  - bf16
  - fp16
confidence: source-reported
reproducibility: snippet
kernel_types:
  - mla
  - attention
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-ub
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3226
  - pr-vllm-ascend-3530
evidence_basis: >
  Synthesized from the two upstream vllm-ascend PRs that add (#3226) and then clean up
  (#3530) the csrc/mla_preprocess AscendC operator. Neither PR discloses absolute
  performance numbers, so the performance_claims entry is a qualitative, source-reported
  statement — no figure is invented (D2).
performance_claims:
  - chip: davinci
    dtype: bf16
    shape: "not disclosed by source"
    metric: "MLA pre-processing overhead (Python-side tensor shuffling + memory copies)"
    value: "source-reported reduction — a single fused custom operator replaces Python-side tensor shuffling/copies that previously bottlenecked the MLA path (no absolute figure disclosed)"
    source_id: pr-vllm-ascend-3226
    measurement: source-reported
---

# MLA Preprocess kernel (Ascend)

## Overview

`mla_preprocess` is a custom AscendC operator in vllm-ascend
(`csrc/mla_preprocess/`) that performs the **pre-processing step of Multi-head Latent
Attention (MLA)** as one on-device operator. MLA (the DeepSeek-style attention that
compresses the KV cache into a low-rank latent) needs a preprocessing stage — normalize,
project, and lay out the latent/query tensors — before the core attention runs. Doing
this as a single fused kernel replaces a sequence of Python-side tensor shuffles and
memory copies that previously bottlenecked the MLA path
([#3226](../../sources/prs/vllm-ascend/PR-3226.md)).

> Chip note: the source PRs tag only the generic DaVinci AI Core, so this page stays at
> `davinci` rather than naming a specific chip.

## Structure

The operator ships the usual AscendC custom-op layout — a host side (tiling + op
definition/prototype) and a device side (`op_kernel`) — with **separate mixed-precision
implementations for bf16 and fp16** (`mla_preprocess_mix_bf16.hpp` /
`mla_preprocess_mix_fp16.hpp`). The preprocessing fuses normalization (RMSNorm-style) and
the latent/query projections so intermediates stay on-chip (see [UB](../hardware/ub.md))
instead of round-tripping through global memory between steps.

```cpp
// AscendC sketch (snippet-level) — synthesized from the PR file layout,
// NOT a verbatim upstream excerpt. One fused op replaces host-side shuffling.
// Dispatched to a bf16 or fp16 mixed-precision path at compile time.
CopyIn(tile);                    // MTE: GM -> UB (hidden states + weights)
RmsNormFused(tile);              // Vector: normalize, stays in UB
LatentQueryProject(tile);        // Cube: low-rank projections -> L0C
CopyOut(tile);                   // MTE: UB -> GM (preprocessed latent/query)
```

## Evolution (upstream PR line)

| PR | What it does | Why it matters |
|----|--------------|----------------|
| [#3226](../../sources/prs/vllm-ascend/PR-3226.md) | Adds the `mla_preprocess` custom kernel (host + device, bf16/fp16 paths) and wires it into the C++ extension so vLLM invokes it directly | Establishes the fused MLA preprocessing operator |
| [#3530](../../sources/prs/vllm-ascend/PR-3530.md) | Removes redundant, unused `gamma/beta` params from the kernel and its call hierarchy (C++ kernel, bindings, Python) | Interface cleanup — shrinks the op signature, no behavior change |

## Notes for adapters

- bf16 vs fp16 are distinct kernel implementations selected per dtype; keep them in sync
  when changing the fused sequence.
- `#3530` is a signature cleanup, not a compute change — useful as a reference for how the
  op's parameters map across the C++/binding/Python boundary.
- Confidence is `source-reported` (upstream code, no accompanying benchmark). Promote to
  `verified` only with official-doc + upstream-code dual evidence.

## See also

- [Cube Unit](../hardware/cube-unit.md) — runs the projection matmuls.
- [Unified Buffer (UB)](../hardware/ub.md) — keeps the fused intermediates on-chip.
- [Fused MoE (DispatchFFNCombine)](fused-moe.md) — sibling fused-operator case study.
