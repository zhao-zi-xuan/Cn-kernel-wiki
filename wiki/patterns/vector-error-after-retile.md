---
id: pattern-vector-error-after-retile
title: "Vector error / wrong results after changing a tile shape"
type: pattern
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - ub-alignment
  - ub
  - vector-unit
confidence: source-reported
symptoms:
  - "a kernel that was correct starts producing vector errors or wrong results after a tiling change"
  - "works for 'round' shapes but fails when a dim is not a multiple of the alignment"
  - "failure depends on offset/stride, not on the numerical values"
candidate_techniques:
  - technique-ub-alignment
related:
  - technique-ub-alignment
  - hw-ub
  - hw-vector-unit
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-6468
---

# Vector error / wrong results after changing a tile shape

## Symptom

A kernel that worked suddenly throws a **vector error** or returns **wrong results** after
you changed a tile shape, buffer offset, or stride — and the failure tracks *shape*
(fails for non-round dims) rather than *data values*.

## Diagnosis

This is almost always **[UB](../hardware/ub.md) misalignment**: the new tiling produced an
offset or length that is not a multiple of the hardware alignment, so the
[Vector unit](../hardware/vector-unit.md) reads across an alignment boundary. It is a
known hazard — upstream fused-MoE work hit and fixed exactly this
([#6468](../../sources/prs/vllm-ascend/PR-6468.md)).

## Candidate technique

- **[UB alignment](../techniques/ub-alignment.md)** — pad tile shapes up to the alignment,
  keep offsets/strides as multiples of it, and process the pad while masking the tail.
  Re-check alignment as an invariant every time the tiling changes.

## Quick checklist

1. Are tile length and every UB offset multiples of the alignment granularity?
2. Did the change introduce an odd/non-round dimension (tokens, hidden, experts)?
3. Can you reproduce by only changing the shape (values fixed)? → confirms alignment.

## Notes

- The exact alignment granularity is hardware/dtype dependent; this page names the
  *diagnosis and fix pattern*, not a specific byte count.
- Confidence `source-reported`: hazard evidenced by the upstream fix.
