---
id: pattern-memory-bound-layer
title: "Memory-bound layer (MTE-bound, Cube under-utilized)"
type: pattern
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - mte
  - ub
  - operator-fusion
  - moe
confidence: source-reported
symptoms:
  - "profiler shows the AI Core waiting on MTE / high GM traffic, low Cube utilization"
  - "throughput scales with memory bandwidth, not with FLOPs — adding compute does not help"
  - "MoE / attention layers dominated by data movement rather than matmul"
candidate_techniques:
  - technique-operator-fusion
  - technique-double-buffering
  - technique-fine-grained-quantization
  - technique-host-tiling
related:
  - kernel-fused-moe
  - hw-mte
  - hw-ub
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-6468
---

# Memory-bound layer

## Symptom

The layer is limited by **data movement, not compute**: the [MTE](../hardware/mte.md) and
GM bandwidth are saturated while the [Cube unit](../hardware/cube-unit.md) sits partly
idle. Adding arithmetic (or a faster matmul) does not improve end-to-end throughput —
the bottleneck is getting operands in and results out. This is the common case for MoE
and KV-heavy attention on Ascend.

## Diagnosis

Ask: how many times does each tensor cross GM? A layer built from many small operators
writes intermediates to GM and reads them back at every boundary. If the per-op compute
is small relative to that traffic, the layer is memory-bound.

## Candidate techniques (in rough priority order)

1. **[Operator fusion](../techniques/operator-fusion.md)** — the biggest lever: keep
   intermediates in [UB](../hardware/ub.md) so they never round-trip GM. Evidenced by
   DispatchFFNCombine ([#3532](../../sources/prs/vllm-ascend/PR-3532.md)).
2. **[Fine-grained quantization](../techniques/fine-grained-quantization.md)** — W8A8/W4A8
   shrinks the weights that must be moved, directly cutting MTE traffic.
3. **[Double buffering](../techniques/double-buffering.md)** — hide the remaining copies
   behind compute so the MTE and Cube/Vector run concurrently.
4. **[Host tiling](../techniques/host-tiling.md)** — size tiles to maximize UB residency
   and balance work across cores.

## Notes

- These compose: fusion + quant reduce *how much* moves; double buffering hides *what's
  left*; tiling makes both fit. The upstream MoE perf pass
  ([#6468](../../sources/prs/vllm-ascend/PR-6468.md)) applies several at once.
- Confidence `source-reported`: the pattern and its fixes are drawn from upstream code and
  the AscendC model; no absolute numbers are asserted.
