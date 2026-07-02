#!/usr/bin/env python3
"""Validate frontmatter + project rules for NPUKernelWiki (Ascend-first).

Checks, per page:
  - YAML frontmatter parses and required fields per page-type are present
  - controlled-vocabulary fields use only values from data/tags.yaml
  - id carries the correct type prefix
  - reproducibility >= snippet for technique/kernel/language pages
  - referential integrity: every `sources:` / `related:` / `candidate_techniques:`
    id resolves to a real page id

Project-specific rules (the two decisions locked at kickoff):
  R1 NO-HARDWARE   : reproducibility may never be `runnable`/`benchmarked`;
                     every performance_claims entry needs source_id +
                     measurement in {source-reported, official-doc}.
  R2 ASCEND-FIRST  : a wiki page whose architectures include a secondary vendor
                     (cambricon-mlu/moore-musa/hygon-dcu) but NO ascend-*/davinci
                     arch must carry `secondary_vendor_note`.

Plus PROVENANCE.yaml shape checks for artifact bundles.

Exit code 0 iff zero errors.
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install -r requirements.txt")
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPRO_ORDER = ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]
SECONDARY_VENDORS = {"cambricon-mlu", "moore-musa", "hygon-dcu"}
ASCEND_ARCHS = {"ascend-910b", "ascend-910c", "ascend-310p", "ascend-950", "davinci"}

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def load_yaml(p: Path):
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_frontmatter(p: Path):
    text = p.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        err(f"{p.relative_to(ROOT)}: YAML parse error: {e}")
        return None


def page_type_of(p: Path, fm: dict) -> str | None:
    rel = p.relative_to(ROOT).as_posix()
    if rel.startswith("sources/prs/"):
        return "source-pr"
    if rel.startswith("sources/docs/"):
        return "source-doc"
    if rel.startswith("sources/blogs/"):
        return "source-blog"
    if rel.startswith("sources/contests/"):
        return "source-contest"
    if rel.startswith("wiki/"):
        t = fm.get("type")
        return f"wiki-{t}" if t else None
    return None


def main() -> int:
    schemas = load_yaml(DATA / "schemas.yaml")
    tags = load_yaml(DATA / "tags.yaml")

    vocab = {k: set(v) for k, v in tags.items() if isinstance(v, list)}
    global_tag_vocab = (
        vocab.get("architectures", set())
        | vocab.get("hardware_features", set())
        | vocab.get("techniques", set())
        | vocab.get("kernel_types", set())
        | vocab.get("languages", set())
    )

    md_files = sorted(
        list((ROOT / "sources").rglob("*.md")) + list((ROOT / "wiki").rglob("*.md"))
    )

    # Pass 1: collect all ids for referential integrity.
    pages = []  # (path, fm, ptype)
    known_ids: set[str] = set()
    for p in md_files:
        fm = parse_frontmatter(p)
        if fm is None:
            err(f"{p.relative_to(ROOT)}: missing or invalid frontmatter")
            continue
        ptype = page_type_of(p, fm)
        if ptype is None:
            err(f"{p.relative_to(ROOT)}: cannot determine page type")
            continue
        if "id" in fm:
            known_ids.add(str(fm["id"]))
        pages.append((p, fm, ptype))

    # Pass 2: per-page validation.
    for p, fm, ptype in pages:
        rel = p.relative_to(ROOT).as_posix()
        schema = schemas.get(ptype)
        if not schema:
            err(f"{rel}: unknown page type '{ptype}'")
            continue

        for field in schema.get("required", []):
            if field not in fm:
                err(f"{rel}: missing required field '{field}'")

        cons = schema.get("constraints", {})
        prefix = cons.get("id_prefix")
        if prefix and not str(fm.get("id", "")).startswith(prefix):
            err(f"{rel}: id '{fm.get('id')}' must start with '{prefix}'")

        # controlled vocabulary
        for field, cat in [
            ("architectures", "architectures"),
            ("hardware_features", "hardware_features"),
            ("techniques", "techniques"),
            ("kernel_types", "kernel_types"),
            ("languages", "languages"),
        ]:
            for val in fm.get(field, []) or []:
                if val not in vocab.get(cat, set()):
                    err(f"{rel}: {field} value '{val}' not in tags.yaml::{cat}")
        for val in fm.get("tags", []) or []:
            if val not in global_tag_vocab:
                err(f"{rel}: tag '{val}' not in any tags.yaml category")
        if "confidence" in fm and fm["confidence"] not in vocab.get("confidence", set()):
            err(f"{rel}: confidence '{fm['confidence']}' invalid")

        # reproducibility: global no-hardware cap + per-type minimum
        repro = fm.get("reproducibility")
        if repro is not None:
            if repro not in vocab.get("reproducibility", set()):
                err(f"{rel}: reproducibility '{repro}' not allowed "
                    f"(no-hardware: runnable/benchmarked are forbidden)")
            minimum = cons.get("reproducibility_minimum")
            if minimum and repro in REPRO_ORDER and minimum in REPRO_ORDER:
                if REPRO_ORDER.index(repro) < REPRO_ORDER.index(minimum):
                    err(f"{rel}: reproducibility '{repro}' below minimum '{minimum}'")

        # R1 performance_claims (kernel pages)
        if ptype == "wiki-kernel":
            pc_schema = schemas["performance-claim"]
            allowed_meas = pc_schema["constraints"]["measurement"]
            for i, claim in enumerate(fm.get("performance_claims", []) or []):
                for f in pc_schema["required"]:
                    if f not in claim:
                        err(f"{rel}: performance_claims[{i}] missing '{f}'")
                meas = claim.get("measurement")
                if meas is not None and meas not in allowed_meas:
                    err(f"{rel}: performance_claims[{i}].measurement '{meas}' "
                        f"forbidden (no self-benchmarking without hardware)")

        # R2 ascend-first secondary-vendor note
        if ptype.startswith("wiki-"):
            archs = set(fm.get("architectures", []) or [])
            if archs & SECONDARY_VENDORS and not (archs & ASCEND_ARCHS):
                if "secondary_vendor_note" not in fm:
                    err(f"{rel}: targets secondary vendor {archs & SECONDARY_VENDORS} "
                        f"without ascend arch; add 'secondary_vendor_note'")

        # referential integrity
        for field in ("sources", "related", "candidate_techniques", "prerequisites"):
            for ref in fm.get(field, []) or []:
                if ref not in known_ids:
                    err(f"{rel}: {field} references unknown id '{ref}'")

    # PROVENANCE bundles
    for prov in (ROOT / "artifacts").rglob("PROVENANCE.yaml"):
        data = load_yaml(prov)
        ab = schemas["artifact-bundle-provenance"]
        for f in ab["required"]:
            if f not in (data or {}):
                err(f"{prov.relative_to(ROOT)}: missing '{f}'")

    n_src = len(list((ROOT / "sources").rglob("*.md")))
    n_wiki = len(list((ROOT / "wiki").rglob("*.md")))
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK — {n_src} source pages, {n_wiki} wiki pages, {len(known_ids)} ids, 0 errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
