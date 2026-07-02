---
id: pattern-quantization-accuracy-drop
title: "Accuracy drop after low-bit quantization"
type: pattern
architectures:
  - davinci
  - ascend-910b
  - ascend-910c
tags:
  - quantization
  - w8a8
  - w4a8
  - fine-grained-quantization
confidence: source-reported
symptoms:
  - "model quality regresses after switching weights/activations to INT8 or INT4"
  - "the drop is larger for outlier-heavy layers (MoE experts, attention projections)"
  - "a single per-tensor scale is too coarse to cover the value range"
candidate_techniques:
  - technique-fine-grained-quantization
related:
  - technique-fine-grained-quantization
  - kernel-quantization-gemm
  - kernel-fused-moe
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-3532
  - pr-vllm-ascend-7779
---

# Accuracy drop after low-bit quantization

## Symptom

End-to-end quality regresses once weights/activations are moved to **INT8 (W8A8)** or
**INT4 (W4A8)**. The regression is worst where the value distribution has outliers —
MoE experts and attention projections — and tends to correlate with using a **single,
coarse scale** for a whole tensor.

## Diagnosis

A per-tensor scale must cover the entire dynamic range, so a few outliers force a large
scale that wastes precision on the common (small) values. The finer the quantization
granularity, the less each scale has to stretch.

## Candidate technique

- **[Fine-grained quantization](../techniques/fine-grained-quantization.md)** — use
  **per-group scales** (a scale per block of weights/activations) instead of one per
  tensor. This is what makes low-bit MoE usable on Ascend; the dequant is applied on the
  Vector side inside the fused matmul epilogue, so accuracy improves without adding a GM
  round-trip. Evidenced by the W8A8 fused MoE
  ([#3532](../../sources/prs/vllm-ascend/PR-3532.md)) and the W4A8 variant
  ([#7779](../../sources/prs/vllm-ascend/PR-7779.md)).

## Notes

- This page names the *diagnosis and mitigation direction*; it does **not** assert any
  accuracy figures — the source PRs disclose none.
- See [kernel-quantization-gemm](../kernels/quantization-gemm.md) for where the per-group
  dequant actually runs.
