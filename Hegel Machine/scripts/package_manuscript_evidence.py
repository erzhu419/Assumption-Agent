#!/usr/bin/env python3
"""Package linked manuscript evidence and source; never traverse result-JSON paths."""

import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = "paper-2026-10-12-v3"
DOCUMENTS = tuple(f"paper/manuscript.{extension}" for extension in ("md", "tex", "pdf"))
SEEDS = (*DOCUMENTS, "paper/ARTIFACT_INDEX.md", "paper/README.md", "paper/FINALIZATION.md",
         "paper/jmlr_style", "scripts/package_manuscript_evidence.py",
         "scripts/build_manuscript.py", "scripts/regenerate_manuscript_tables.py", "pyproject.toml",
         *(f"artifacts/{study}/summary.json" for study in (
             "online_inquiry_luna_subscription_20261005_main_30",
             "temporal_counterexample_learning_luna_20261006",
             "temporal_evidence_scope_learning_luna_20261006",
             "temporal_selective_revision_luna_20261006",
             "temporal_selective_preservation_luna_20261006",
             "paper_scoped_retention_20261005", "sc_kg_structural_prior_comparison_20261007")),
         "artifacts/online_inquiry_luna_subscription_20261005_main_30/protocol.json")
SUFFIXES = {".md", ".tex", ".pdf", ".png", ".svg", ".jpg", ".jpeg", ".json",
            ".py", ".toml", ".yaml", ".yml", ".txt", ".bib", ".cls", ".sty", ".bst", ".bbl", ".patch"}
OMIT_PARTS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
              "provider_logs", "provider_calls", "sessions", "requests", "responses"}
OMIT_NAMES = {"auth.json", "credentials.json", "token.json", "tokens.json", "stdout.txt", "stderr.txt"}
COMPANIONS = ("protocol.json", "plan.json", "summary.json", "brief.json", "derived_analysis.json")
LINKS = re.compile(r"\]\((<[^>]+>|[^)]+)\)")


def permitted(path):
    relative = path.relative_to(ROOT)
    return ((path.suffix.lower() in SUFFIXES or path.name == "LICENSE")
            and not set(relative.parts).intersection(OMIT_PARTS)
            and path.name.lower() not in OMIT_NAMES and not path.name.startswith(".env"))


def local_links(path):
    for matched in LINKS.findall(path.read_text(encoding="utf-8")):
        target = matched[1:-1] if matched.startswith("<") else matched
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        yield (path.parent / unquote(parsed.path)).resolve()


def local_imports(path):
    """Follow ordinary local Python imports, including saved snapshot packages."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    search = [path.parent]
    for parent in path.parents:
        if parent == ROOT.parent:
            break
        if parent.name in {"src", "scripts"}:
            search.append(parent)
        for name in ("src", "scripts"):
            if (parent / name).is_dir():
                search.append(parent / name)
    search.extend((ROOT / "src", ROOT / "scripts", ROOT))

    def resolve(module, roots):
        for base in roots:
            stem = base.joinpath(*module.split(".")) if module else base
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                if candidate.is_file() and candidate.is_relative_to(ROOT):
                    return candidate.resolve()
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names, roots = [entry.name for entry in node.names], search
        elif isinstance(node, ast.ImportFrom):
            base = path.parent
            for _ in range(max(0, node.level - 1)):
                base = base.parent
            roots = [base] if node.level else search
            module = node.module or ""
            names = [module] + [f"{module}.{entry.name}".strip(".") for entry in node.names
                                if entry.name != "*"]
        else:
            continue
        for name in names:
            dependency = resolve(name, roots)
            if dependency:
                yield dependency


def collect():
    files, missing, omitted = set(), set(), set()
    queue = [(ROOT / name, True) for name in SEEDS]
    queue.extend((path, False) for path in (ROOT / "paper/figures").glob("*"))
    expanded = set()
    while queue:
        path, follow_markdown = queue.pop()
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            omitted.add(str(path))
            continue
        relative = str(path.relative_to(ROOT))
        # The active paper includes its appendices; previous paper revisions
        # remain in the checkout and their existing release, not this bundle.
        if path.is_relative_to(ROOT / "paper/revisions"):
            omitted.add(relative)
            continue
        if not path.exists():
            missing.add(relative)
            continue
        if path.is_dir():
            queue.extend((child, False) for child in path.rglob("*") if child.is_file())
            continue
        if not permitted(path):
            omitted.add(relative)
            continue
        files.add(path)
        key = (path, follow_markdown)
        if key in expanded:
            continue
        expanded.add(key)
        if path.suffix == ".md" and follow_markdown:
            # Historical design documents link to the entire research history.
            # Include a cited document itself; continue through paper pages and
            # study status records, which link the actual reported evidence.
            queue.extend((linked, linked.is_relative_to(ROOT / "paper")
                          or linked.name == "RUN_STATUS.md") for linked in local_links(path))
            if path.name == "RUN_STATUS.md":
                queue.extend((path.parent / name, False) for name in COMPANIONS
                             if (path.parent / name).is_file())
        if path.suffix == ".py":
            queue.extend((dependency, False) for dependency in local_imports(path))
    return sorted(files), sorted(missing), sorted(omitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/releases/hegel_machine_paper_2026-10-12_v3.zip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print included paths during preview or packaging")
    args = parser.parse_args()
    files, missing, omitted = collect()
    contents = {"release": RELEASE, "purpose": "selective manuscript review materials", "paths": [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size} for path in files],
        "generated_paths": ["paper/BUNDLE_CONTENTS.json"],
        "current_source_paths": [str(path.relative_to(ROOT)) for path in files
            if path.is_relative_to(ROOT / "src") or path.is_relative_to(ROOT / "scripts")],
        "selection": "Local Markdown links, known result companions, saved source directories and local Python import dependencies; JSON path strings are not followed.",
        "omitted_paths": omitted}
    report = {"files": len(files), "uncompressed_bytes": sum(row["bytes"] for row in contents["paths"]),
              "missing": missing, "omitted_count": len(omitted), "dry_run": args.dry_run}
    if args.list:
        for row in contents["paths"]:
            print(row["path"])
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    if missing:
        raise SystemExit("Missing linked or compiled materials: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            bundle.write(path, str(path.relative_to(ROOT)))
        bundle.writestr("paper/BUNDLE_CONTENTS.json", json.dumps(contents, indent=2) + "\n")
    report.update(output=str(args.output), compressed_bytes=args.output.stat().st_size,
                  archive_members=len(files) + 1)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
