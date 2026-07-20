"""Prospective BRIGHT v2 execution wrapper for the single U+0000 repair.

This module leaves the frozen v1 implementation byte-identical.  In a fresh
process it redirects every formal path/schema to v2 and replaces only the
document text reader: U+0000 becomes U+FFFD before the already-frozen
3,000-character truncation.  All acquisition packs, recipes, metrics,
evaluator candidates, promotion logic, arm execution, and concurrency remain
the v1 implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as base


VERSION = "bright_reasoning_retrieval_study_v2"
DESIGN_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_execution_repair_design_v2.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_study_implementation_freeze_v2.json"
)
DESIGN_SELF_SHA256 = "e72c4b0150c6509d35760b821ba3f0fd66dff4412818ed62f69250b7df3ac7b4"
DESIGN_FILE_SHA256 = "2c14445836498b212e45c66860718038234d4d24998b3c44eb8ace7c080efe52"
FORMAL_ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_study_v2")

CORPUS_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_corpus_tensor_v2.json"
)
G_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_G_form_v2.json")
A_FORM_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_A_form_v2.json"
)
F_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_F_search_v2.json")
A_HOLD_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_A_hold_v2.json"
)
M_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_M_search_v2.json")


def normalize_document_content(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise base.BrightStudyError("document content is invalid")
    normalized = value.replace("\x00", "\ufffd")
    if not normalized.strip() or "\x00" in normalized:
        raise base.BrightStudyError("normalized document content is invalid")
    return normalized[: base.DOCUMENT_TEXT_CHARACTERS]


def _read_source_documents_v2(
    project_root: Path, family: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if family not in base.core.FAMILY_ORDER:
        raise base.BrightStudyError("document family is invalid")
    binding = base.SOURCE_DOCUMENTS[family]
    path = project_root / base.SOURCE_ROOT_RELATIVE / binding["path"]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding["size"]
        or base.file_sha256(path) != binding["sha256"]
    ):
        raise base.BrightStudyError("document source binding drifted")
    try:
        import pyarrow.parquet as parquet

        reader = parquet.ParquetFile(path)
        if tuple(reader.schema_arrow.names) != base.DOCUMENT_SCHEMA:
            raise base.BrightStudyError("document parquet schema drifted")
        rows = reader.read(
            columns=list(base.DOCUMENT_SCHEMA), use_threads=False
        ).to_pylist()
    except base.BrightStudyError:
        raise
    except Exception as exc:
        raise base.BrightStudyError("document parquet read failed") from exc
    if len(rows) != binding["rows"]:
        raise base.BrightStudyError("document row count drifted")
    ids: list[str] = []
    contents: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(base.DOCUMENT_SCHEMA):
            raise base.BrightStudyError("document row shape drifted")
        ids.append(base._required_text(raw.get("id"), "document ID"))
        contents.append(normalize_document_content(raw.get("content")))
    if len(set(ids)) != len(ids):
        raise base.BrightStudyError("document IDs are duplicated")
    return tuple(ids), tuple(contents)


def _activate_v2() -> None:
    public = {
        "G_form": G_RESULT_RELATIVE,
        "A_form": A_FORM_RESULT_RELATIVE,
        "F_search": F_RESULT_RELATIVE,
        "A_hold": A_HOLD_RESULT_RELATIVE,
        "M_search": M_RESULT_RELATIVE,
    }
    predecessors = {
        "G_form": CORPUS_RESULT_RELATIVE,
        "A_form": G_RESULT_RELATIVE,
        "F_search": A_FORM_RESULT_RELATIVE,
        "A_hold": F_RESULT_RELATIVE,
        "M_search": A_HOLD_RESULT_RELATIVE,
    }
    updates: dict[str, Any] = {
        "VERSION": VERSION,
        "DESIGN_SCHEMA": f"{VERSION}_design",
        "FREEZE_SCHEMA": f"{VERSION}_implementation_freeze",
        "CORPUS_RESULT_SCHEMA": f"{VERSION}_corpus_result",
        "STAGE_RESULT_SCHEMA": f"{VERSION}_stage_result",
        "ACTION_SCHEMA": f"{VERSION}_local_action_pack",
        "SCORED_SCHEMA": f"{VERSION}_scored_pack",
        "MARKER_SCHEMA": f"{VERSION}_attempt",
        "DESIGN_RELATIVE": DESIGN_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "DESIGN_SELF_SHA256": DESIGN_SELF_SHA256,
        "DESIGN_FILE_SHA256": DESIGN_FILE_SHA256,
        "FORMAL_ROOT_RELATIVE": FORMAL_ROOT_RELATIVE,
        "CORPUS_RESULT_RELATIVE": CORPUS_RESULT_RELATIVE,
        "G_RESULT_RELATIVE": G_RESULT_RELATIVE,
        "A_FORM_RESULT_RELATIVE": A_FORM_RESULT_RELATIVE,
        "F_RESULT_RELATIVE": F_RESULT_RELATIVE,
        "A_HOLD_RESULT_RELATIVE": A_HOLD_RESULT_RELATIVE,
        "M_RESULT_RELATIVE": M_RESULT_RELATIVE,
        "PUBLIC_STAGE_RESULTS": public,
        "STAGE_PREDECESSORS": predecessors,
        "_read_source_documents": _read_source_documents_v2,
    }
    for name, value in updates.items():
        setattr(base, name, value)


def run(command: str, project_root: Path) -> dict[str, Any]:
    _activate_v2()
    functions = {
        "prepare-corpus": base.prepare_corpus,
        "G-form": base.run_g_form,
        "A-form": base.run_a_form,
        "F-search": base.run_f_search,
        "A-hold": base.run_a_hold,
        "M-search": base.run_m_search,
    }
    if command not in functions:
        raise base.BrightStudyError("v2 command is invalid")
    return functions[command](project_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare-corpus", "G-form", "A-form", "F-search", "A-hold", "M-search"),
    )
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run(arguments.command, arguments.project_root)
    print(base.canonical_json_bytes({
        "result_sha256": result["result_sha256"],
        "status": result["status"],
    }).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
