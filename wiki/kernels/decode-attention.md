---
id: kernel-decode-attention
title: "Decode attention path on Ascend (MLA / sparse flash-attention)"
type: kernel
architectures:
  - davinci
  - ascend-910c
tags:
  - attention
  - decode
  - mla
  - sparse-attention
  - flash-attention
  - kv-cache
  - operator-fusion
confidence: source-reported
reproducibility: snippet
kernel_types:
  - attention
  - decode
  - mla
languages:
  - ascendc
  - cpp
related:
  - hw-cube-unit
  - hw-ub
  - kernel-mla-preprocess
  - kernel-transpose-kv-cache-by-block
  - technique-operator-fusion
sources:
  - doc-ascendc-programming-guide
  - pr-vllm-ascend-4625
  - pr-vllm-ascend-3226
  - pr-vllm-ascend-6366
evidence_basis: >
  Synthesized from upstream vllm-ascend PRs on the decode attention path: the AscendC
  lightning_indexer + sparse_flash_attention operators (#4625), MLA preprocess (#3226),
  and the GQA KV-cache transpose (#6366). No PR discloses absolute numbers, so the
  performance_claims entry is qualitative and source-reported (D2).
performance_claims:
  - chip: ascend-910c
    dtype: bf16
    shape: "decode step, single query token per sequence (shape not disclosed)"
    metric: "decode attention operator performance"
    value: "source-reported — dedicated AscendC lightning_indexer + sparse_flash_attention operators boost DeepSeek v3.2 decode attention (#4625); no absolute figure disclosed"
    source_id: pr-vllm-ascend-4625
    measurement: source-reported
---

# Decode attention path on Ascend

## Overview

**Decode** attention (one new query token per sequence, attending over a long KV cache)
is memory-bound and latency-critical — the opposite of compute-bound prefill. On Ascend
the decode path is assembled from several custom AscendC operators rather than one
monolithic kernel. This page ties them together; the two building blocks each have their
own page.

The DeepSeek-style **MLA** (Multi-head Latent Attention) path is the primary case:

1. **[MLA preprocess](mla-preprocess.md)** ([#3226](../../sources/prs/vllm-ascend/PR-3226.md))
   — normalize + project the latent/query tensors as one fused op.
2. **[KV-cache transpose](transpose-kv-cache-by-block.md)**
   ([#6366](../../sources/prs/vllm-ascend/PR-6366.md)) — re-lay-out the KV cache after a
   GQA transfer when prefill/decode run at heterogeneous TP sizes.
3. **Core attention** — `lightning_indexer` + `sparse_flash_attention`
   ([#4625](../../sources/prs/vllm-ascend/PR-4625.md)): AscendC operators that index the
   relevant KV entries (sparse) and run a flash-attention-style streaming softmax over
   them, added to speed up DeepSeek v3.2 decode.

## Why it is split this way

Flash-attention keeps the running softmax state in [UB](../hardware/ub.md) and streams KV
blocks through, avoiding materializing the full attention matrix. The **sparse** variant
plus a **lightning indexer** only touches the KV blocks that matter — decisive when the
cost is dominated by moving KV out of GM. Preprocessing and KV-layout are separated so
each can be fused/optimized independently (see
[operator fusion](../techniques/operator-fusion.md)).

```cpp
// AscendC sketch (snippet-level) — flash-attention-style decode inner loop.
// NOT a verbatim upstream excerpt.
InitRunningSoftmax(m /*=-inf*/, l /*=0*/, acc /*=0*/);      // state in UB
for (int blk : indexer.SelectedKvBlocks(q)) {              // sparse: only relevant blocks
    LoadKV(blk);                                           // MTE: GM -> UB
    Mmad(s, q, kBlk, params);                              // Cube: q·Kᵀ for this block
    OnlineSoftmaxUpdate(m, l, acc, s, vBlk);               // Vector: streaming softmax
}
WriteOut(acc / l);                                         // MTE: UB -> GM
```

## Notes for adapters

- This is a *path*, not a single kernel: preprocess → (KV transpose if heterogeneous TP)
  → sparse flash-attention. Swap pieces independently.
- `lightning_indexer` decides *which* KV to read; `sparse_flash_attention` does the
  streaming-softmax compute. Keep their block/selection conventions aligned.
- Prefill attention (compute-bound, full) is a different regime and out of scope here.
- Confidence `source-reported` (upstream code, no benchmark numbers).

## See also

- [MLA preprocess](mla-preprocess.md) / [KV-cache transpose](transpose-kv-cache-by-block.md) — the two staged sub-ops.
- [Unified Buffer (UB)](../hardware/ub.md) — holds the running softmax state.
