# WORKLOG — NPUKernelWiki 项目工作日志

> 这个文件是**给人看的项目进度日志**：记录每一步"做了什么、为什么做、当前处于什么状态、下一步是什么"。
> 和 `CLAUDE.md`（给 AI 的交接约定）、`plan.md`（完整计划）互补——那两个讲"规则和蓝图"，这个讲"实际发生了什么"。
> 维护约定：**每完成一段有意义的工作，就在下面"进度日志"顶部加一条（倒序，最新在上）**，写清动机和结果。

---

## 一句话项目目标

对标 MIT HAN Lab 的 KernelWiki（NVIDIA kernel 知识库），把"三层架构 + schema 驱动 + LLM 可检索"的方法论迁移到**国产 NPU（昇腾 Ascend 优先）**，做一个**可被 Claude Code skill 检索的 kernel 优化知识库**。

## 为什么这么做（设计动机）

- **三层架构**：`sources/`（不可变原始摘要）→ `wiki/`（综合知识页）→ `queries/`（自动索引）。原始事实和综合结论分离，可溯源、可再生。
- **schema 驱动 + validator 强制**：每页带 YAML frontmatter，靠 `id` 互引；`scripts/validate.py` 跑出 **0 errors** 才算"完成"。防止知识库随规模增长而失控。
- **两条铁律**（validator 强制）：
  - **D1 Ascend-first**：昇腾为主，其他厂商仅 interface-level 提及。
  - **D2 No-hardware**：无真实硬件，所以**绝不自测、绝不编造性能数字**，性能只能引用官方/PR 自述（`source-reported`/`official-doc`）。
- **两阶段摄入管道**：先建**候选账本**（`candidates/*.yaml`，人工/半自动判定 include/defer/exclude），再由 `generate-pr-pages.py` 把 include 行拉成 `sources/` 页。保证进入知识库的都是真正动了 device kernel 的 PR。

---

## 进度日志（倒序，最新在上）

### 2026-07-02 — 上传到 GitHub（独立仓库，不含个人文件）

**做了什么**
- 发现本地 wiki 目录嵌在 `E:/` 这个大 git 仓库里（装着个人文件），为避免误传，在 `e:/npu-kernel-wiki` 内**单独 `git init` 一个独立仓库**（`main` 分支）。
- 加 `.gitignore`（Python 缓存 / 编辑器 / `*.token`、`.env` 等，防止误提交密钥），确认 62 个待提交文件全是 wiki 内容、无父目录个人文件。
- 首次提交推送到 `https://github.com/zhao-zi-xuan/Cn-kernel-wiki`（默认分支 main）。

**为什么**
- 给知识库一个可协作、可备份的远端；独立仓库隔离了 `E:/` 上的无关个人数据。

**结果 / 现状**
- 远端 62 blobs，与本地一致；`origin` 用干净 HTTPS URL（**未把 token 写进 config**）。推送走系统凭据管理器（那个只读 token 无写权限，已 403 拒绝，未使用）。
- 备注：仓库名是 `Cn-kernel-wiki`（用户自建），与项目内部标题 NPUKernelWiki 略有出入，无碍。

---

### 2026-07-02 — 第三步续：补两个硬件页 + MLA preprocess kernel 页（wiki 5 页互引成网）

**做了什么**
- 新增 `wiki/hardware/ub.md`（`hw-ub`）、`wiki/hardware/mte.md`（`hw-mte`）：讲清 `GM → MTE → UB/L1 → Cube/Vector → UB → GM` 数据通路、UB 驻留=减少 GM 往返、MTE 是访存瓶颈、double-buffer overlap、UB-alignment 陷阱。均带 snippet 级 AscendC。
- 新增 `wiki/kernels/mla-preprocess.md`（`kernel-mla-preprocess`）：从 #3226（建）+ #3530（清理冗余 gamma/beta 参数）提炼，含 bf16/fp16 双精度路径、演进线表、snippet。
- 把之前 `kernel-fused-moe` 里指向 `cube-unit` 的"UB & MTE"占位链接改成真正的 `hw-ub`/`hw-mte`；`cube-unit` 也回引这两页 → 5 个 wiki 页互引成网。

**为什么**
- `kernel-fused-moe` 已引用 UB/MTE 概念却没有对应页（悬空概念）；先补基础硬件页再扩 kernel 页，符合依赖顺序。MLA preprocess 是第二条素材齐全的 kernel 线。

**结果 / 现状**
- `validate.py` = **0 errors**（32 source / **5 wiki** / 37 ids），引用完整性通过，索引重生。两个 kernel 页的 `performance_claims` 均为定性 `source-reported`（源 PR 无绝对数字），芯片如实 `davinci`。

---

### 2026-07-02 — 第三步开工：写出首个 wiki 综合页（kernel-fused-moe / DispatchFFNCombine）

**做了什么**
- 扩写 `wiki/kernels/fused-moe.md`（`kernel-fused-moe`），把原本带 `<TO-FILL>` 的占位页改成完整的 DispatchFFNCombine 案例页：Overview、为什么融合（MTE 压力 / 同步 bubble）、**四个 PR 的演进线表格**（#3532 建 → #6468 通信优化 → #7779 W4A8 通算 overlap → #8902 修回退）、UB-alignment 陷阱、snippet 级 AscendC 融合流水线、adapter 注意事项、互引。
- 抓了 4 个 PR 的完整正文核对性能：**均无绝对性能数字** → `performance_claims` 只写定性 `source-reported`（注明"未披露绝对数值"），严守 D2 不编造。
- 芯片：源 PR 只标通用 `davinci`（未指明 910B/C）→ 页面如实用 `davinci`，不臆造。
- `sources` 引 5 个（doc + 4 PR），`related` 引 `hw-cube-unit`；`validate.py` 通过引用完整性校验。

**为什么**
- sources 层已厚，进入第三步"写综合页"。选 DispatchFFNCombine 是因为它有一条清晰的 4-PR 演进线，素材最全，适合做首个范例页。

**结果 / 现状**
- `validate.py` = **0 errors**（32 source / 2 wiki / 34 ids），索引重生。confidence 落 `source-reported`（无官方文档+代码双证据，不够 `verified`）。

---

### 2026-07-02 — Stage-2 批量摄入完成，sources 层建成

**做了什么**
- 对 31 个 include 行跑 `generate-pr-pages.py`，一次性摄入 **31 个真实 vllm-ascend PR 页**到 `sources/prs/vllm-ascend/`。
- `validate.py` = **0 errors**（32 source 页 / 2 wiki 页 / 34 ids），索引重生。

**为什么**
- 账本已就绪且质量可信，进入两阶段管道的第二阶段——把候选变成可检索的原始事实页，为后续写 wiki 综合页备料。

**结果 / 现状**
- 每页 author/date/merge_sha/changed_paths 均为 GitHub 真实抓取值；AscendC PR 标 `cpp/python`，Triton PR 标 `triton-ascend`；芯片未在标题指明的谨慎落 `davinci`（通用 AI Core）。

---

### 2026-07-02 — 扩充候选账本（含两轮质量收紧）

**做了什么**
- GitHub 搜索 `repo:vllm-project/vllm-ascend is:pr is:merged kernel in:title` → 86 个 merged PR。
- 逐个拉 `files_changed`，据"是否真正动了 device-kernel 源文件"自动分类 + 编辑分层，写回 `candidates/vllm-ascend.yaml`。
- **两轮收紧**（应用户复查反馈）：
  1. 只动 `vllm_ascend/ops/*.py` 这类 host 封装的（非 Triton）从 include 降为 defer。
  2. 重抓**完整 diff**核对：`#7221/7575/7757/7767/8030` 这批 `model_runner_v2` "优化 _xxx_kernel" 其实只改了 host 调用侧 + 测试，kernel 本体没动 → 降为 defer；`#5259/5324/5356/5918/6537` 确有真实 Triton 源码 → 保留。
  3. 修正 3 个 backport 行"自己重复自己"的 reason（正文无原始号引用，改为如实"待 GitHub 核对"）。
  4. `#8083/8243` 是 `patch_triton.py` monkey-patch（调编译配置/调用参数，非计算逻辑）→ include 保留，但 reason 注明写 wiki 时 confidence 应落 `source-reported`。

**为什么**
- CLAUDE.md 要求 ≥25 个高质量 include。光看标题不可靠，必须用实际改动文件判定，才符合 kernel-only scope。

**结果 / 现状**
- 账本：**86 total / 31 include / 37 defer / 18 exclude**。所有 include 行都有真实 kernel 源文件背书。

---

### 2026-07-02 — 跑通真实 PR 摄入（管道验证）

**做了什么**
- 用 GITHUB_TOKEN 先**匿名核对** PR 真实标题/作者/sha（防编造），确认 #6366/#6468 为真实 merged 的 `[Kernel]` PR。
- 跑 `generate-pr-pages.py`，把 `PR-6366.md` 里的 4 处 `<TO-FILL>` 换成真实数据，并新增 `PR-6468.md`。

**为什么**
- 验证两阶段摄入管道端到端可用；同时消除样例页里的占位符。

**结果 / 现状**
- 摄入管道验证通过，`validate.py` 0 errors。

---

### （更早）Phase 0 — 脚手架

- 三层结构 + schema（`data/schemas.yaml`）+ 词表（`data/tags.yaml`/`aliases.yaml`）+ 查询工具 + validator（含 D1/D2）+ 索引生成器 + 多源 PR 摄入脚本，均就位。
- 4 个互引样例页：`doc-ascendc-programming-guide`、`pr-vllm-ascend-6366`、`hw-cube-unit`、`kernel-fused-moe`。

---

## 当前状态快照（截至 2026-07-02）

| 层 | 内容 | 数量 |
|---|---|---|
| sources | vllm-ascend PR 页 + 官方文档摘要 | 31 PR + 1 doc |
| wiki | `hardware/`：cube-unit、ub、mte；`kernels/`：fused-moe（DispatchFFNCombine）、mla-preprocess | 5 页 |
| candidates | vllm-ascend 账本 | 31 incl / 37 defer / 18 excl |
| queries | 自动索引 | 6 个（生成物，勿手改） |

- `validate.py`：**0 errors**（32 source / 2 wiki / 34 ids）。

## 下一步（按 CLAUDE.md 依赖排序）

1. **写 wiki 综合页**（当前重点，素材已充分）：
   - kernel 页：~~`dispatch_ffn_combine`~~（✅）、~~`mla_preprocess`~~（✅）、`transpose_kv_cache_by_block`（#6366）、LoRA bgmv（#1884）、vocabparallel embedding（#796）。
   - hardware 页：~~`ub`~~（✅）、~~`mte`~~（✅），待补 `vector-unit / nz-format / pto-isa / l1-buffer`。
   - technique 页（尚未开张）：`operator-fusion`、`host-tiling`、`double-buffering`、`ub-alignment`、`fine-grained-quantization` —— 这些 tag 已在多页出现，可提炼成独立 technique 页。
   - 每页带 snippet 级 AscendC/TileLang 代码、靠 `id` 互引 sources、confidence 严格按证据分级。
2. **gitcode/Gitee 适配器**：CANN 的 `cann-recipes-infer/train` 不在 GitHub，需给 `generate-pr-pages.py` 加 gitcode API 分支。
3. **artifacts 溯源**（可选）：拉真实 kernel 代码进 `artifacts/`，每 bundle 一个 `PROVENANCE.yaml`。

## 备忘 / 注意事项

- 改完任何内容，**以 `python scripts/validate.py` 跑出 0 errors 为"完成"判据**，再 `generate-indices.py` 重生索引。
- 摄入需 GITHUB_TOKEN（无 token 限 60/hr）。**每次用完的临时 token 应及时 revoke**。
- `#8083/8243` 是 monkey-patch 类 PR，写 wiki 时措辞要区分"调配置/参数"而非"改计算逻辑"。
- 红线：不编造 PR 元数据/性能数字；`reproducibility` 封顶 `snippet`；不纳入分布式话题（HCCL/超节点）；不手改 `queries/*.md`。
