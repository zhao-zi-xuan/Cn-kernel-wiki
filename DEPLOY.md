# DEPLOY.md — 部署与同步

> **核心前提**：本项目遵循 D2「无硬件」，整条流水线（挖 PR / 写 wiki / validate /
> 生成索引 / 当 skill 查询）只用到 **Python + 网络 + Claude Code**，**不碰任何 GPU/NPU**。
> 所以在哪台机器上做都一样——轻薄本完全够用。skill 本质是一堆可移植的文件。
>
> 工作流就一句话：**哪台机器方便就在哪 build → 用 git 同步 → 在真正跑 Claude Code 的机器上 clone/pull 进 `~/.claude/skills/`。**

## 0. 一次性：把仓库纳入 git

```bash
cd npu-kernel-wiki
git init && git add -A && git commit -m "Phase 0 scaffold"
# 推到你的远端（GitHub / Gitee 私库均可），便于本地↔服务器同步：
git remote add origin <your-remote-url>
git push -u origin main
```

## 1. 安装为 Claude Code skill（本地或服务器，步骤相同）

个人级 skill 放在 `~/.claude/skills/<name>/`，Claude Code 会自动检测加载。

```bash
mkdir -p ~/.claude/skills && cd ~/.claude/skills
git clone <your-remote-url> npu-kernel-wiki      # 或：unzip npu-kernel-wiki.zip
pip install -r npu-kernel-wiki/requirements.txt
cd npu-kernel-wiki
python3 scripts/validate.py                      # 期望：0 errors
python3 scripts/query.py --type kernel           # 冒烟测试：能查出样例页
```

确认 `~/.claude/skills/npu-kernel-wiki/SKILL.md` 存在即安装成功。

**唯一的坑**：若 `~/.claude/skills/` 这个目录在当前 Claude Code 会话启动时还不存在，
新建后要**重启一次 Claude Code**它才会监视；目录已存在则热加载、无需重启。

## 2. 抓真实 PR（唯一需要联网的一步）

无 token 时 GitHub 限 60 次/小时，务必设 token（5000/小时）：

```bash
export GITHUB_TOKEN=ghp_你的只读token
python3 scripts/generate-pr-pages.py candidates/vllm-ascend.yaml
python3 scripts/validate.py
```

**在本地跑最省事**（Clash 原生）。若一定要在并行智算服务器上跑，需让流量走你的反向隧道：

```bash
export https_proxy=http://127.0.0.1:<隧道在服务器侧暴露的端口>   # 本地 Clash 混合端口是 7897
export http_proxy=$https_proxy
```

（脚本用 urllib，默认读 `http_proxy/https_proxy`，无需改代码。Clash 混合端口支持 HTTP CONNECT；
若你暴露的是纯 SOCKS 端口，urllib 不原生支持，那就退回本地跑。并行智算会定期重置凭证，跑前确认 token/隧道仍有效。）

## 3. 日常同步

在任一机器改完内容、`validate.py` 跑通后：

```bash
git add -A && git commit -m "..." && git push
# 另一台机器：
cd ~/.claude/skills/npu-kernel-wiki && git pull
```

`queries/*.md` 是生成物——改了 frontmatter 后跑 `python3 scripts/generate-indices.py` 再提交。

## 关于硬件（FAQ）

- **服务器那张卡对本项目有用吗？** 没用。无硬件阶段不需要任何加速卡。
- **将来想真跑 benchmark 呢？** 即便那天，你服务器的 NVIDIA 卡也不行——AscendC 算子只能跑在
  达芬奇核上。要测得有**真正的昇腾 NPU**。所以本项目无论现在还是将来，服务器都不构成优势。
