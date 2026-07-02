---
name: NPUKernelWiki
description: Use when the user asks about optimizing Chinese-domestic NPU kernels — primarily Huawei Ascend (DaVinci AI Core, Cube/Vector units, UB, MTE, AscendC / TileLang-Ascend / Triton-Ascend), operators in vllm-ascend / sglang-ascend / CANN, fused MoE, MLA, sparse/flash attention, KV-cache, quantization (W8A8/W4A8) on Ascend, or wants concrete PR references from the Ascend open-source ecosystem. Secondary vendors (Cambricon MLU, Moore Threads MUSA, Hygon DCU) are covered only as interface-level mentions. Do NOT use for generic CUDA/NVIDIA kernels, host-side framework integration, or distributed systems (HCCL / super-node).
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# NPUKernelWiki — Ascend-first NPU Kernel Optimization Wiki

> **Scope decisions (locked at kickoff):**
> - **Ascend-first.** 昇腾 is primary; other domestic vendors are interface-only.
> - **No hardware.** All claims are mined from open code/docs. Reproducibility is
>   capped at `snippet`; performance numbers are always `source-reported`
>   (never self-benchmarked). The validator enforces both.

## How To Query

```bash
python3 scripts/query.py "how is fused MoE structured on Ascend"
python3 scripts/query.py --tag cube-unit --type hardware
python3 scripts/query.py --repo vllm-project/vllm-ascend --limit 20
python3 scripts/get_page.py kernel-fused-moe --follow-sources
python3 scripts/grep_wiki.py "UB align"
```

`--tag` and `--architecture` are alias-aware (`--tag Cube` matches `cube-unit`,
`--architecture A3` matches `ascend-910c`). See `data/aliases.yaml`.

## Output Pattern

1. Cite pages by path + id.
2. Follow `sources:` back to PR / doc / blog.
3. Respect confidence (`verified` > `source-reported` > `inferred` > `experimental`).
4. Include the `snippet`-level code on technique/kernel/language pages.
5. Report perf claims with all fields incl. `source_id` and `measurement: source-reported`.

## Maintenance

```bash
pip install -r requirements.txt
python3 scripts/validate.py            # schema + no-hardware + ascend-first rules
python3 scripts/generate-indices.py    # regenerate queries/*.md
```
