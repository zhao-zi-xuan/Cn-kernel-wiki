#!/usr/bin/env python3
"""Generate PR source pages from candidate ledgers (Ascend-first).

Adapted from KernelWiki's generator. Differences:
  - Fetches via GitHub REST API using urllib (no gh dependency), with optional
    GITHUB_TOKEN (env or --token) raising the limit from 60/hr to 5000/hr, and
    a `gh api` fallback if the CLI is installed and authenticated.
  - Ascend controlled vocabulary for auto-tagging.
  - Architecture defaults to `davinci` (generic AI Core) unless the title names a
    specific chip — we never invent a chip we can't verify.

Usage:
    export GITHUB_TOKEN=ghp_xxx            # strongly recommended
    python3 scripts/generate-pr-pages.py candidates/vllm-ascend.yaml
    python3 scripts/generate-pr-pages.py candidates/vllm-ascend.yaml --dry-run
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TAGS = yaml.safe_load((ROOT / "data" / "tags.yaml").read_text(encoding="utf-8"))
VALID_HW = set(TAGS.get("hardware_features", []))
VALID_TECH = set(TAGS.get("techniques", []))
VALID_KT = set(TAGS.get("kernel_types", []))
VALID_LANG = set(TAGS.get("languages", []))
ALL = VALID_HW | VALID_TECH | VALID_KT | VALID_LANG

# Ascend keyword -> controlled-vocab tag.
KW_TO_HW = {
    "cube": "cube-unit", "vector": "vector-unit", "ub ": "ub", "unified buffer": "ub",
    "mte": "mte", "l0c": "l0c", "l1": "l1-buffer", "nz": "nz-format", "pto": "pto-isa",
    "int8": "int8", "bf16": "bf16", "fp16": "fp16", "fp8": "fp8",
    "w8a8": "w8a8", "w4a8": "w4a8",
}
KW_TO_KT = {
    "matmul": "matmul", "gemm": "gemm", "gemv": "gemv", "attention": "attention",
    "flash": "flash-attention", "sparse": "sparse-attention", "mla": "mla",
    "moe": "moe", "ffncombine": "fused-moe", "dispatchffn": "fused-moe",
    "kv_cache": "kv-cache", "kv cache": "kv-cache", "kvcache": "kv-cache",
    "rope": "rope", "rmsnorm": "rmsnorm", "layernorm": "layernorm",
    "conv1d": "conv1d", "quant": "quantization", "decode": "decode", "prefill": "prefill",
}
KW_TO_TECH = {
    "fused": "operator-fusion", "fusion": "operator-fusion", "tiling": "host-tiling",
    "prefetch": "weight-prefetch", "double buffer": "double-buffering",
    "double_buffer": "double-buffering", "pingpong": "ping-pong-buffer",
    "ping-pong": "ping-pong-buffer", "block size": "block-size-tuning",
    "align": "ub-alignment", "quant": "fine-grained-quantization",
}
KW_TO_LANG = {
    ".cce": "ascendc", "ascendc": "ascendc", "ascend c": "ascendc",
    "tilelang": "tilelang-ascend", "triton": "triton-ascend", ".py": "python",
    ".cpp": "cpp", ".cc": "cpp",
}
CHIP_KW = {
    "910b": "ascend-910b", "a2": "ascend-910b", "910c": "ascend-910c",
    "a3": "ascend-910c", "310p": "ascend-310p", "950": "ascend-950",
}
EXCLUDE_TITLE = [r"\[ci\]", r"\[doc", r"bump", r"typo", r"format", r"lint",
                 r"\bnit\b", r"revert", r"readme", r"changelog", r"pre-commit",
                 r"release note", r"version"]
KERNEL_EXTS = {".cce", ".cpp", ".cc", ".cu", ".cuh"}
KERNEL_DIRS = ("csrc/", "kernels/", "ops/", "ascendc", "operator")


# ----------------------------- fetching ------------------------------------
def _http_get(url: str, token: str | None):
    headers = {"User-Agent": "npu-kernel-wiki", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gh_cli(endpoint: str):
    try:
        out = subprocess.run(["gh", "api", endpoint], capture_output=True,
                             text=True, timeout=30)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def fetch_pr(repo: str, number: int, token: str | None):
    base = f"https://api.github.com/repos/{repo}/pulls/{number}"
    try:
        pr = _http_get(base, token)
        files = _http_get(base + "/files?per_page=100", token)
        return pr, [f["filename"] for f in files]
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):   # rate limited -> try gh CLI
            pr = gh_cli(f"repos/{repo}/pulls/{number}")
            if pr:
                fl = gh_cli(f"repos/{repo}/pulls/{number}/files?per_page=100") or []
                return pr, [f["filename"] for f in fl]
            raise RuntimeError(
                f"GitHub API {e.code} (rate limit). Set GITHUB_TOKEN to raise the "
                f"limit to 5000/hr, or wait for the window to reset.") from e
        raise


# ----------------------------- tagging -------------------------------------
def auto_tag(title: str, files: list[str]):
    text = (title + " " + " ".join(files)).lower()
    hw = {v for k, v in KW_TO_HW.items() if k in text} & VALID_HW
    kt = {v for k, v in KW_TO_KT.items() if k in text} & VALID_KT
    tech = {v for k, v in KW_TO_TECH.items() if k in text} & VALID_TECH
    lang = {v for k, v in KW_TO_LANG.items() if k in text} & VALID_LANG
    if not lang:
        if any(f.endswith(".cce") for f in files):
            lang.add("ascendc")
        elif any(f.endswith(".py") for f in files):
            lang.add("python")
    tags = (hw | kt | tech) & ALL
    return sorted(tags), sorted(hw), sorted(kt), sorted(tech), sorted(lang)


def detect_chips(text: str) -> list[str]:
    chips = sorted({v for k, v in CHIP_KW.items() if k in text.lower()})
    return chips or ["davinci"]   # generic AI Core if unspecified


def is_kernel_related(title: str, files: list[str]):
    t = title.lower()
    for pat in EXCLUDE_TITLE:
        if re.search(pat, t):
            return False, "excluded by title pattern"
    for f in files:
        if os.path.splitext(f)[1] in KERNEL_EXTS or any(d in f.lower() for d in KERNEL_DIRS):
            return True, "kernel file change"
    return False, "no kernel file touched"


# ----------------------------- rendering -----------------------------------
def render_page(repo: str, pr: dict, files: list[str], reason: str, captured_at: str) -> str:
    slug = repo.split("/")[1]
    tags, hw, kt, tech, lang = auto_tag(pr["title"], files)
    body = re.sub(r"<!--.*?-->", "", pr.get("body") or "", flags=re.DOTALL).strip()
    kernel_paths = [f for f in files if os.path.splitext(f)[1] in KERNEL_EXTS][:10]
    fm = {
        "id": f"pr-{slug}-{pr['number']}",
        "repo": repo,
        "pr": pr["number"],
        "title": pr["title"],
        "author": pr["user"]["login"],
        "date": (pr.get("merged_at") or pr.get("created_at") or "")[:10],
        "url": pr["html_url"],
        "source_category": "upstream-code",
        "architectures": detect_chips(pr["title"] + " " + body),
        "tags": tags,
        "techniques": tech,
        "hardware_features": hw,
        "kernel_types": kt,
        "languages": lang or ["ascendc"],
        "captured_at": captured_at,
        "status": "merged" if pr.get("merged_at") else pr.get("state", "open"),
        "merge_sha": (pr.get("merge_commit_sha") or "")[:8],
        "inclusion_reason": reason,
        "changed_paths": kernel_paths,
    }
    out = "---\n" + yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False) + "---\n\n"
    out += f"## Summary\n\n{(body[:400] or 'No description provided.')}\n\n"
    out += "## Changed Files\n\n" + "".join(f"- `{f}`\n" for f in files[:15]) + "\n"
    return out


# ----------------------------- driver --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", help="candidates/<repo>.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    args = ap.parse_args()

    ledger = yaml.safe_load(Path(args.ledger).read_text(encoding="utf-8"))
    repo = ledger["repo"]
    slug = repo.split("/")[1]
    captured_at = ledger.get("searched_at", "")
    out_dir = ROOT / "sources" / "prs" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    include = [r for r in ledger.get("prs", []) if r.get("decision") == "include"]
    print(f"{repo}: {len(include)} include row(s)"
          + (" [dry-run]" if args.dry_run else "")
          + ("  (no token: 60 req/hr limit)" if not args.token else "  (token: 5000/hr)"))

    written, skipped = 0, []
    for row in include:
        n = row["number"]
        if args.dry_run:
            print(f"  would fetch PR #{n}: {row.get('title','')}")
            continue
        try:
            pr, files = fetch_pr(repo, n, args.token)
        except Exception as e:
            print(f"  PR #{n}: FETCH FAILED — {e}")
            return 2
        ok, why = is_kernel_related(pr["title"], files)
        if not ok:
            skipped.append((n, why))
            print(f"  PR #{n}: SKIP ({why})")
            continue
        (out_dir / f"PR-{n}.md").write_text(
            render_page(repo, pr, files, row.get("reason", why), captured_at),
            encoding="utf-8")
        written += 1
        print(f"  PR #{n}: wrote sources/prs/{slug}/PR-{n}.md")

    if not args.dry_run:
        print(f"Done: {written} written, {len(skipped)} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
