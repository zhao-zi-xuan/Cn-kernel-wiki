# Index: PRs by Repository

## vllm-project/vllm-ascend (31 PRs)

- `pr-vllm-ascend-233` — [core] Support custom ascendc kernels in vllm-ascend
- `pr-vllm-ascend-796` — add custom ascendc kernel vocabparallelembedding
- `pr-vllm-ascend-814` — [Performance]: Custom AscendC Kernel of Multi-Step Prepare Input
- `pr-vllm-ascend-1884` — Add  Custom Kernels For LoRA Performance
- `pr-vllm-ascend-3226` — add mla_preprocess kernel
- `pr-vllm-ascend-3530` — remove redundant params in mla_preprocess kernel
- `pr-vllm-ascend-3532` — add `dispatch_gmm_combine` kernel
- `pr-vllm-ascend-3804` — [Kernel] add custom op GmmSwigluQuantWeightNzTensorList
- `pr-vllm-ascend-4139` — [Kernel] add custom op DispatchGmmCombineDecode
- `pr-vllm-ascend-4194` — [Kernel] add custom moe ops for prefill
- `pr-vllm-ascend-4304` — [task] Add fused gdn gating triton kernel
- `pr-vllm-ascend-4413` — [Ops][Triton] Add a triton kernel supporting partial rope.
- `pr-vllm-ascend-4595` — [Kernel] add l2norm triton kernel
- `pr-vllm-ascend-4606` — [Kernel] add custom op MatmulAllreduceAddRmsnorm
- `pr-vllm-ascend-4625` — [kernel] add AscendC op: lightning_indexer and sparse_flash_attention
- `pr-vllm-ascend-4790` — [kernel] Adapt DispatchGmmCombineDecode operator to parameters of small operators
- `pr-vllm-ascend-4810` — [Kernel] Add moe normal ops
- `pr-vllm-ascend-5259` — feat: implement high-performance Triton kernels for rejection sampling: optimization for rejection_random_sample_kernel
- `pr-vllm-ascend-5324` — [Refactor][Triton] Move reject sample triton kernels into ops/triton
- `pr-vllm-ascend-5356` — [Feat][Spec] Optimize token index calculation in spec decode with Triton kernel
- `pr-vllm-ascend-5518` — [Triton][Config] Add muls_add triton kernel and refactor AscendCompilationConfig
- `pr-vllm-ascend-5579` — [Kernel] Add moe_gating_top_k operator support for Ascend NPU
- `pr-vllm-ascend-5918` — [Ascend] perf: optimize rope embedding with triton kernel for huge performance gain
- `pr-vllm-ascend-6366` — [Kernel] Add AscendC fused op transpose_kv_cache_by_block to speed up GQA transfer
- `pr-vllm-ascend-6468` — [Kernel]: Optimize DispatchFFNCombine performance
- `pr-vllm-ascend-6537` — perf: adaptive block size selection in linear_persistent kernel
- `pr-vllm-ascend-7569` — [Triton][Sampler] Add penalty-related Triton kernel for better performance of penalties
- `pr-vllm-ascend-7779` — [Feature][Kernel add] Fuse W4A8 dispatch + FFN + combine into a single fused kernel
- `pr-vllm-ascend-8083` — [Performance][model_runner_v2]:optimize for triton op _temperature_kernel and _topk_log_softmax_kernel
- `pr-vllm-ascend-8243` — [Performance][model_runner_v2]:optimize the performance of the _min_p_kernel
- `pr-vllm-ascend-8902` — [Feature] update dataWithScale and combine in dispatch_ffn_kernel

