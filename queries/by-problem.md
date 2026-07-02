# Index: Problem -> Candidate Techniques

## Fragmented op chain (many small ops invoked per-layer) (`pattern-fragmented-op-chain`)

- symptoms: ['a single logical step is implemented as a sequence of framework ops (load + transform + store)', 'that sequence is invoked once per layer / per iteration, so overhead multiplies', 'lots of small kernel launches and GM round-trips for intermediates that are never reused']
- candidate_techniques: ['technique-operator-fusion', 'technique-host-tiling']

## Memory-bound layer (MTE-bound, Cube under-utilized) (`pattern-memory-bound-layer`)

- symptoms: ['profiler shows the AI Core waiting on MTE / high GM traffic, low Cube utilization', 'throughput scales with memory bandwidth, not with FLOPs — adding compute does not help', 'MoE / attention layers dominated by data movement rather than matmul']
- candidate_techniques: ['technique-operator-fusion', 'technique-double-buffering', 'technique-fine-grained-quantization', 'technique-host-tiling']

## Accuracy drop after low-bit quantization (`pattern-quantization-accuracy-drop`)

- symptoms: ['model quality regresses after switching weights/activations to INT8 or INT4', 'the drop is larger for outlier-heavy layers (MoE experts, attention projections)', 'a single per-tensor scale is too coarse to cover the value range']
- candidate_techniques: ['technique-fine-grained-quantization']

## Vector error / wrong results after changing a tile shape (`pattern-vector-error-after-retile`)

- symptoms: ['a kernel that was correct starts producing vector errors or wrong results after a tiling change', "works for 'round' shapes but fails when a dim is not a multiple of the alignment", 'failure depends on offset/stride, not on the numerical values']
- candidate_techniques: ['technique-ub-alignment']

