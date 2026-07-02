# NPUKernelWiki — Ascend-first 国产 NPU Kernel 优化知识库（脚手架）

对标 MIT HAN Lab 的 [KernelWiki](https://github.com/mit-han-lab/KernelWiki)，把同一套
"三层架构 + schema 驱动 + LLM 可检索"方法论迁移到国产 NPU，**昇腾优先**。本仓库是
**Phase 0 脚手架**：结构、schema、词表、查询工具、validator、索引生成器与 4 个样例页都已就位，
`scripts/validate.py` 通过（0 errors）。后续按 `plan.md` 填充内容。

## 架构（三层 + 支撑层）

- `sources/` — 原始数据（PR/doc/blog/contest 摘要，不可变）
- `wiki/` — 综合知识页（hardware/techniques/kernels/patterns/languages/migration）
- `queries/` — 自动生成的交叉索引（勿手改）
- 支撑层：`data/`（schema+词表）、`candidates/`（PR 候选账本）、`artifacts/`（代码+溯源）、`references/`

## 两条奠基约束（见 plan.md）

1. **Ascend-first**：昇腾为主，别家仅 interface-level（非昇腾页需 `secondary_vendor_note`）。
2. **No-hardware**：复现性封顶 `snippet`，性能只能 `source-reported`。validator 强制。

## 快速开始

```bash
pip install -r requirements.txt
python3 scripts/validate.py            # schema + D1/D2 规则校验
python3 scripts/generate-indices.py    # 生成 queries/*.md
python3 scripts/query.py --type kernel
python3 scripts/get_page.py kernel-fused-moe --follow-sources
```

## 复用 vs 重写

直接复用 KernelWiki：`query.py / get_page.py / grep_wiki.py / _wiki_root.py`、三层目录、PROVENANCE。
本仓库已重写：`validate.py`（去 NVIDIA 耦合 + 加 D1/D2）、`generate-indices.py`、全部 `data/*.yaml`。
待做：`generate-pr-pages.py` 多源（GitHub + gitcode/Gitee + web）适配。
