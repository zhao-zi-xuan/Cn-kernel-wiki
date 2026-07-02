# Primer — 主题地图 / Reading Paths

> 这是给 **agent 和工程师的导航入口**：不按页面类型、而按"你在做什么任务"组织，给出推荐阅读顺序和对应 wiki `id`。
> 先在这里选一条路径，再用 `python scripts/get_page.py <id> --follow-sources` 深入，或 `python scripts/query.py --type <t>` 横向浏览。
>
> 约定：所有 kernel 页的性能都是**定性 `source-reported`**（无真实硬件，绝不自测/编造数字，见 [CLAUDE.md](../CLAUDE.md) 的 D2）。芯片未在源 PR 指明时标 `davinci`（通用 AI Core）。

---

## 按任务选路径

### 1. MoE 推理优化（最常见）
从"为什么慢"到"怎么融合 + 量化"：

1. [pattern-memory-bound-layer](../wiki/patterns/memory-bound-layer.md) — 先确认瓶颈是访存（MTE-bound）而非算力
2. [kernel-fused-moe](../wiki/kernels/fused-moe.md) — DispatchFFNCombine：dispatch+FFN+combine 融合成一个 op
3. [kernel-grouped-gemm](../wiki/kernels/grouped-gemm.md) — MoE expert 计算核心（分组 matmul + SwiGLU/dequant 融合 epilogue）
4. [kernel-quantization-gemm](../wiki/kernels/quantization-gemm.md) — W8A8/W4A8 量化 matmul
5. [technique-fine-grained-quantization](../wiki/techniques/fine-grained-quantization.md) — per-group dequant，兼顾带宽与精度

### 2. Attention / Decode 推理优化
把昇腾上的 decode attention 路径讲清楚：

1. [pattern-fragmented-op-chain](../wiki/patterns/fragmented-op-chain.md) — 识别"一个逻辑步骤被拆成 per-layer 多 op"
2. [kernel-mla-preprocess](../wiki/kernels/mla-preprocess.md) — MLA 预处理融合成单 op
3. [kernel-decode-attention](../wiki/kernels/decode-attention.md) — lightning_indexer + sparse_flash_attention 的 decode 路径总览
4. [kernel-transpose-kv-cache-by-block](../wiki/kernels/transpose-kv-cache-by-block.md) — 异构 TP 下的 KV-cache 转置
5. [technique-double-buffering](../wiki/techniques/double-buffering.md) — MTE↔compute overlap 隐藏搬运延迟

### 3. 量化优化
从"掉精度"到"怎么在 kernel 里做对量化"：

1. [pattern-quantization-accuracy-drop](../wiki/patterns/quantization-accuracy-drop.md) — 低比特掉精度的诊断入口
2. [technique-fine-grained-quantization](../wiki/techniques/fine-grained-quantization.md) — per-group scale 是解法
3. [kernel-quantization-gemm](../wiki/kernels/quantization-gemm.md) — W8A8/W4A8 量化 matmul 的实现视角
4. [kernel-grouped-gemm](../wiki/kernels/grouped-gemm.md) — 量化 dequant 融进分组 GEMM 的 epilogue

### 4. 新算子开发（AscendC，从零手写 device kernel）
先懂语言和硬件，再懂 tiling 和对齐陷阱：

1. [lang-ascendc](../wiki/languages/ascendc.md) — C++ 式两段式（host tiling + device kernel）编程模型
2. [hw-cube-unit](../wiki/hardware/cube-unit.md) — matmul 引擎（L0A/L0B→L0C，NZ 布局）
3. [hw-ub](../wiki/hardware/ub.md) + [hw-mte](../wiki/hardware/mte.md) + [hw-vector-unit](../wiki/hardware/vector-unit.md) — 片上缓冲、搬运引擎、向量单元
4. [technique-host-tiling](../wiki/techniques/host-tiling.md) — 如何切 tile、分核、选路径
5. [technique-ub-alignment](../wiki/techniques/ub-alignment.md) — UB 对齐陷阱（最常见的 retile 回归源）
   - 排错时配 [pattern-vector-error-after-retile](../wiki/patterns/vector-error-after-retile.md)

### 5. Triton kernel 开发（更高层的 Python DSL）
适合 element-wise / 归约 / 采样类 kernel：

1. [lang-triton-ascend](../wiki/languages/triton-ascend.md) — `@triton.jit` / grid / `tl.load|store` / `BLOCK` autotune
2. [hw-vector-unit](../wiki/hardware/vector-unit.md) + [hw-ub](../wiki/hardware/ub.md) + [hw-mte](../wiki/hardware/mte.md) — Triton 的 `tl.*` / `tl.load` 映射到的底层单元
3. [kernel-triton-rope](../wiki/kernels/triton-rope.md) — partial RoPE（`rope_dim != head_dim`）融合，Triton 版算子融合范例
4. [kernel-triton-sampling](../wiki/kernels/triton-sampling.md) — rejection sampling / penalties，decode 路径的 Triton 部分
   - 其余 Triton 源码（l2norm #4595、fused_gdn_gating #4304 等）见 `sources/prs/vllm-ascend/`，暂未单独成页

---

## 贯穿全库的横切概念

- **算子融合**是主线优化：[technique-operator-fusion](../wiki/techniques/operator-fusion.md) —— 让中间结果驻留 UB、不回 GM。几乎每个 kernel 页都在用它。
- **数据通路**：`GM → MTE → UB/L1 → Cube/Vector → UB → MTE → GM`，理解它就理解了大部分优化的动机。
- **对齐**：改任何 tiling 后都要复查 UB 对齐（[technique-ub-alignment](../wiki/techniques/ub-alignment.md)）。

## 其他 kernel（未列入上面路径的）

- [kernel-lora-bgmv](../wiki/kernels/lora-bgmv.md) — LoRA 的 batched gather-GEMV（`bgmv_shrink`/`expand`）
- [kernel-vocab-parallel-embedding](../wiki/kernels/vocab-parallel-embedding.md) — TP 分片词表的越界掩码（纯 Vector op，非 matmul 的最小示例）

## 怎么用这个库（工具速查）

```bash
python scripts/query.py --type kernel                 # 按类型横向浏览
python scripts/get_page.py kernel-fused-moe --follow-sources   # 看某页并追溯它引用的 PR
python scripts/query.py --tag operator-fusion         # 按 tag 过滤
```

自动索引（勿手改，`generate-indices.py` 再生）：
- `queries/by-problem.md` — 症状 → 候选 technique（pattern 页喂的）
- `queries/by-kernel-type.md` / `by-technique.md` / `by-hardware-feature.md` / `by-language.md` / `by-repo.md`

## 覆盖现状（截至 23 个 wiki 页）

| 层 | 页 |
|---|---|
| pattern（诊断入口） | memory-bound-layer · fragmented-op-chain · vector-error-after-retile · quantization-accuracy-drop |
| technique（手法） | operator-fusion · host-tiling · double-buffering · ub-alignment · fine-grained-quantization |
| kernel（实例） | fused-moe · grouped-gemm · quantization-gemm · decode-attention · mla-preprocess · transpose-kv-cache-by-block · lora-bgmv · vocab-parallel-embedding |
| hardware（硬件） | cube-unit · ub · mte · vector-unit |
| language（语言） | ascendc · triton-ascend |

> 维护：新增 wiki 页后，若它开辟了新任务场景，在上面对应路径里补一行 id；页面成体系后此表也应同步。
