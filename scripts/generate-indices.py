#!/usr/bin/env python3
"""Regenerate queries/*.md cross-reference indices from page frontmatter.

Generic (no vendor-specific assumptions). Do NOT hand-edit queries/* —
they are overwritten on every run. Outputs:
  by-repo.md, by-hardware-feature.md, by-technique.md,
  by-kernel-type.md, by-language.md, by-problem.md
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed."); sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "queries"


def fm_of(p: Path):
    t = p.read_text(encoding="utf-8")
    if not t.startswith("---"):
        return None
    parts = t.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None


def all_pages():
    out = []
    for sub in ("sources", "wiki"):
        for p in sorted((ROOT / sub).rglob("*.md")):
            fm = fm_of(p)
            if fm:
                out.append((p.relative_to(ROOT).as_posix(), fm))
    return out


def write(name: str, lines: list[str]):
    (Q / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    Q.mkdir(exist_ok=True)
    pages = all_pages()

    # by-repo
    by_repo = defaultdict(list)
    for path, fm in pages:
        if path.startswith("sources/prs/"):
            by_repo[fm.get("repo", "unknown")].append(fm)
    lines = ["# Index: PRs by Repository\n"]
    for repo in sorted(by_repo):
        prs = sorted(by_repo[repo], key=lambda f: f.get("pr", 0))
        lines.append(f"## {repo} ({len(prs)} PRs)\n")
        for f in prs:
            lines.append(f"- `{f['id']}` — {f.get('title','')}")
        lines.append("")
    write("by-repo.md", lines)

    # tag-bucketed indices
    def bucket(field, title, fname):
        m = defaultdict(list)
        for path, fm in pages:
            for v in fm.get(field, []) or []:
                m[v].append(fm.get("id", path))
        out = [f"# Index: {title}\n"]
        for k in sorted(m):
            out.append(f"## {k} ({len(m[k])})\n")
            for pid in sorted(set(m[k])):
                out.append(f"- `{pid}`")
            out.append("")
        write(fname, out)

    bucket("hardware_features", "Pages by Hardware Feature", "by-hardware-feature.md")
    bucket("techniques", "Pages by Technique", "by-technique.md")
    bucket("kernel_types", "Pages by Kernel Type", "by-kernel-type.md")
    bucket("languages", "Pages by Language/DSL", "by-language.md")

    # by-problem (pattern symptoms -> candidate techniques)
    out = ["# Index: Problem -> Candidate Techniques\n"]
    for path, fm in pages:
        if fm.get("type") == "pattern":
            out.append(f"## {fm.get('title','')} (`{fm.get('id')}`)\n")
            out.append(f"- symptoms: {fm.get('symptoms', [])}")
            out.append(f"- candidate_techniques: {fm.get('candidate_techniques', [])}\n")
    write("by-problem.md", out)

    print(f"Regenerated 6 indices in {Q.relative_to(ROOT)}/ from {len(pages)} pages")


if __name__ == "__main__":
    main()
