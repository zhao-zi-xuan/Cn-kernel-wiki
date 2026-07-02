---
id: doc-ascendc-programming-guide
title: "AscendC Programming Guide (CANN)"
url: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/
source_category: official-doc
architectures:
  - davinci
  - ascend-910b
tags:
  - ascendc
  - cube-unit
  - vector-unit
  - ub
retrieved_at: '2026-04-27'
lang: zh
---

## Summary

AscendC 编程模型：host 侧 tiling program + device 侧 kernel program 的两段式结构；
核心数据通路 GM → MTE → UB/L1 → Cube/Vector → UB → GM。本页是官方文档摘要 stub，
后续按真实文档章节补全（搬运/计算/同步 API、流水编排、double buffer 等）。
