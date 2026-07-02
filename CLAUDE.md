# CLAUDE.md — NPUKernelWiki 项目交接（给 Claude Code 读）

> 你（Claude Code）正在一个**国产 NPU kernel 优化知识库**仓库里工作。本文件是项目上下文 +
> 工作约定 + 当前进度 + 下一步。动手前先读完，并读 `plan.md`（完整计划）和 `data/schemas.yaml`（schema）。

## 这是什么

对标 MIT HAN Lab 的 KernelWiki（NVIDIA Blackwell/Hopper kernel 知识库），把同一套
"三层架构 + schema 驱动 + LLM 可检索" 方法论迁移到**国产 NPU，昇腾(Ascend)优先**。
仓库根即 Claude Code skill 目录（`SKILL.md` 在根），装到 `~/.claude/skills/` 即可被检索。

## 两条铁律（validator 强制，违反即报错）

- **D1 Ascend-first**：昇腾为主。寒武纪/摩尔/海光仅作 interface-level 提及；任何
  `architectures` 含 secondary vendor 但不含 ascend-*/davinci 的 **wiki 页必须带 `secondary_vendor_note`**。
- **D2 No-hardware**：无真实硬件。`reproducibility` 封顶 `snippet`（**禁止 runnable/benchmarked**）；
  kernel 页 `performance_claims` 的 `measurement` 只能 `source-reported`/`official-doc`，
  **绝不自测、绝不编造性能数字**。

## 架构（三层 + 支撑层）

- `sources/` 原始数据：`prs/<repo-slug>/PR-<N>.md`、`docs/`、`blogs/`、`contests/`（不可变摘要）
- `wiki/` 综合知识页：`hardware/ techniques/ kernels/ patterns/ languages/ migration/`（带 YAML frontmatter，靠 `id` 互引）
- `queries/` 自动生成索引（**勿手改**，跑 `generate-indices.py` 再生）
- 支撑层：`data/`（schema+词表）、`candidates/`（PR 候选账本）、`artifacts/`（代码+PROVENANCE 溯源）、`references/`

## 常用命令

```bash
pip install -r requirements.txt
python3 scripts/validate.py            # schema + D1/D2 校验，必须 0 errors
python3 scripts/generate-indices.py    # 从 frontmatter 重生 queries/*.md
python3 scripts/query.py --type kernel
python3 scripts/get_page.py <id> --follow-sources
python3 scripts/generate-pr-pages.py candidates/vllm-ascend.yaml   # 需 GITHUB_TOKEN
```

## schema 要点（细节见 data/schemas.yaml）

- 每页有唯一 `id` + 类型前缀：`pr-* doc-* blog-* contest-* hw-* technique-* kernel-* pattern-* lang-* migration-*`
- `tags / architectures / hardware_features / techniques / kernel_types / languages` 的取值**必须**在 `data/tags.yaml`（别名见 `data/aliases.yaml`）
- technique/kernel/language 页 `reproducibility >= snippet` 且必须含可编译 AscendC/TileLang 片段
- kernel 页必须有 `performance_claims`，每条含 `chip dtype shape metric value source_id measurement`
- confidence：`verified`（需 official-doc + upstream-code 双证据）> `source-reported` > `inferred` > `experimental`
- 改完任何内容，**以 `python3 scripts/validate.py` 跑出 0 errors 为"完成"的判据**

## 当前进度（Phase 0 脚手架 = DONE）

- 三层结构 + schema + 词表 + 通用查询工具 + 重写的 validator(含 D1/D2) + 索引生成器 + 多源 PR 摄入脚本，均就位
- 4 个互引样例页：`doc-ascendc-programming-guide`、`pr-vllm-ascend-6366`（**含 `<TO-FILL>` 占位**）、
  `hw-cube-unit`、`kernel-fused-moe`
- `candidates/vllm-ascend.yaml` 起步账本（3 行示例 + 搜索关键词）
- `validate.py` 通过（0 errors）。摄入脚本渲染逻辑已离线验证；**实时抓取尚未跑**（需 token）

## 下一步（按依赖排序，逐项做）

1. **跑通真实 PR 摄入**：设 `GITHUB_TOKEN`（无 token 时 GitHub 限 60/小时），
   `python3 scripts/generate-pr-pages.py candidates/vllm-ascend.yaml`，
   生成的页面会用真实 author/date/merge_sha/changed_paths 替换 `PR-6366.md` 里的 `<TO-FILL>`。
   服务器上跑需先 `export https_proxy=http://127.0.0.1:<隧道端口>`（本地 Clash 混合端口 7897）。
2. **扩账本**：把 vllm-ascend 近期 merged 的 kernel PR 逐条填进 `candidates/vllm-ascend.yaml`，
   标 include/defer/exclude + reason（kernel-only：动 `.cce`/`csrc/kernels/`/`ops/` 才 include）。目标 ≥25 个 include。
3. **写硬件页**：`wiki/hardware/` 补 cube-unit(已起)、vector-unit、ub、mte、nz-format、pto-isa 等，
   confidence 基于 `doc-ascendc-programming-guide` 等官方文档，**只写文档/代码能佐证的**。
4. **写 technique/kernel/pattern 页**：从已摄入 PR + 官方文档提炼，每页带 snippet 级 AscendC 代码。
5. **gitcode/Gitee 适配器**：CANN 的 `cann-recipes-infer/train` 不在 GitHub，给 `generate-pr-pages.py`
   加一个 gitcode API 分支（这是和 KernelWiki 差异最大的工程点）。
6. **artifacts 溯源**（可选）：把真实 kernel 代码拉进 `artifacts/<...>/`，每 bundle 一个 `PROVENANCE.yaml`（sha256+license）。
   CANN 闭源库（libatb 等）拿不到源码，只能停在 `concept`/`pseudocode`。

## 红线（不要做）

- 不要编造 PR 的 author/date/sha 或任何性能数字；拿不到就留 `<TO-FILL>` 或不写。
- 不要把 `reproducibility` 写成 `runnable`/`benchmarked`（无硬件）。
- 不要在只有博客来源时标 `confidence: verified`。
- 不要纳入分布式系统话题（HCCL / 超节点 / 大 EP 通信）——超出 kernel-only scope。
- 不要手改 `queries/*.md`——它们是生成物。
- 任何改动以 `validate.py` 0 errors 收尾，再 `generate-indices.py` 重生索引。
