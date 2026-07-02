# NPUKernelWiki — Build Plan (Ascend-first 国产芯片 kernel 知识库)

> 对标 MIT HAN Lab 的 KernelWiki（NVIDIA Blackwell/Hopper），把同一套
> "三层架构 + schema 驱动 + LLM 可检索" 的方法论迁移到国产 NPU，**昇腾优先**。

## 目标 (Goal)

构建一个面向 LLM agent 检索的国产 NPU kernel 优化知识库。覆盖以华为昇腾
(DaVinci AI Core) 为主的算子实现、优化技术、硬件特性、DSL，数据来自开源
PR / 官方文档 / 博客 / 竞赛。**单设备 kernel 优化为主**，排除分布式 (HCCL/超节点)。

## 两条奠基性约束 (Locked Decisions)

- **D1 Ascend-first**：昇腾为主，寒武纪/摩尔/海光仅作 interface-level 提及；
  非昇腾的 wiki 页必须带 `secondary_vendor_note`（validator 强制）。
- **D2 No-hardware**：无真实硬件，只挖代码/文档。`reproducibility` 封顶 `snippet`；
  `performance_claims.measurement` 只能 `source-reported`/`official-doc`，
  禁止 `runnable`/`benchmarked` 与自测数字（validator 强制）。

## 数据源 (Sources to Mine)

| 层 | 来源 | 备注 |
|---|---|---|
| PR/代码 | `vllm-project/vllm-ascend`（主力金矿）、`sgl-project/sglang`(ascend 部分) | GitHub，可用 gh API |
| 官方代码/文档 | CANN `cann-recipes-infer` / `cann-recipes-train`（gitcode）、hiascend.com | **非 GitHub，需适配 gitcode/Gitee API + web fetch** |
| DSL/编译器 | `tile-ai/tilelang-ascend`、`triton-ascend`、PTO/毕昇编译器资料 | language 层 |
| 论文 | AscendOptimizer 等体系结构论文 | doc(paper) 层 |
| 竞赛 | 昇腾 CANN 训练营 / 算子赛事 | 待确认具体赛事 |

## 验收标准 (Acceptance Criteria)

- **AC-1 结构**：三层 `sources/ | wiki/ | queries/` + 支撑层 `data/ candidates/ artifacts/ references/`，与本仓库布局一致；无 `wiki/systems/`（分布式排除）。
- **AC-2 source schema**：每个 `sources/prs/*/PR-*.md` 含全部必填字段；tag 全部在 `data/tags.yaml`。
- **AC-3 wiki schema**：六类 wiki 页各自满足 schema；technique/kernel/language `reproducibility >= snippet`。
- **AC-4 索引**：`queries/*.md` 全部由 `scripts/generate-indices.py` 从 frontmatter 生成，可重复再生。
- **AC-5 覆盖（下界）**：vllm-ascend ≥ 25 个 kernel PR；昇腾核心硬件页齐全（cube-unit / vector-unit / ub / mte / nz-format / pto-isa）；≥6 technique、≥4 kernel、≥3 pattern、≥3 language 页。
- **AC-6 代码**：每个 technique/kernel/language 页含可编译 AscendC/TileLang 片段（snippet 级）。
- **AC-7 导航**：SKILL.md / README.md / index.md + `data/*.yaml` 完整。
- **AC-8 性能元数据**：kernel 页每条 `performance_claims` 含 chip/dtype/shape/metric/value/source_id/measurement。
- **AC-9 D2 规则**：validator 拒绝 `runnable`/`benchmarked` 与非 source-reported 的 measurement。
- **AC-10 D1 规则**：validator 对缺 `secondary_vendor_note` 的非昇腾页报错。

## 分阶段 (Phases)

**Phase 0 — 脚手架（本仓库已完成）**：目录 + schema + tags/aliases + 通用查询工具 +
validator（含 D1/D2 规则）+ generate-indices + 4 个样例页跑通 `validate.py` 0 errors。

**Phase 1 — 词表与硬件页**：补全 `data/tags.yaml` 与昇腾硬件 wiki 页
（cube/vector/ub/l0/mte/nz/pto 等），全部 `confidence` 基于官方文档。

**Phase 2 — 摄入管线（关键工程）**：
1. 改造 `generate-pr-pages.py`：GitHub(gh) 用于 vllm-ascend/sglang；新增 gitcode/Gitee
   适配器抓 CANN recipes；官方文档走 web fetch。
2. 建 `candidates/*.yaml` 候选账本（已起 vllm-ascend），逐 PR 标 include/defer/exclude。
3. 生成 `sources/prs/**` 页面；填 author/date/merge_sha（替换 `<TO-FILL>`）。

**Phase 3 — wiki 合成 + artifacts**：从 PR/doc 提炼 technique/kernel/pattern/language 页；
把真实 kernel 代码拉进 `artifacts/`，每 bundle 一个 `PROVENANCE.yaml`（sha256 + license）。
注意 CANN 闭源库（libatb 等）只能 `concept`/`pseudocode` 级。

**Phase 4（可选，后期）— 抗漂移**：仿 KernelWiki 加 version-claims 注册表
（CANN/AscendC 版本钉定）+ 刷新工具 + 新鲜度检查。

## 直接复用自 KernelWiki 的部分

`scripts/{query,get_page,grep_wiki,_wiki_root}.py`、三层目录、PROVENANCE 机制、
schema 驱动思路。**重写/适配**：`validate.py`（去 NVIDIA 耦合，加 D1/D2）、
`generate-pr-pages.py`（多源适配）、全部 `data/*.yaml` 词表。
