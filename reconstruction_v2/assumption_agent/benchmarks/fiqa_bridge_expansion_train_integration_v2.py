"""TRAIN-derived FiQA integration with explicit empty-document handling.

Version 1 established that FiQA contains corpus rows whose title and text are
both empty.  This non-claim version treats those rows as source-unretrievable,
applies the same rule to TRAIN qrels, and freezes the resulting filtered
corpus before any DEV or TEST qrel member is opened.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_integration_v1 as v1,
)


SCHEMA = "fiqa_bridge_expansion_train_integration_result_v2"
ATTEMPT_SCHEMA = "fiqa_bridge_expansion_train_integration_attempt_v2"
FREEZE_SCHEMA = "fiqa_bridge_expansion_train_integration_implementation_freeze_v2"

FAILURE_V1_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_integration_failure_v1.json"
)
FAILURE_V1_FILE_SHA256 = "e98bb0d060956bda813bac294420ae79122f933fecce62f88f3907703eec28a7"
FAILURE_V1_SELF_SHA256 = "8a2997443b309a48ec7b6eb766c7e4da6139cf3749c5efe5382e5c15ec68461c"
FREEZE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_integration_implementation_freeze_v2.json"
)
RESULT_RELATIVE = Path("manifests/fiqa_bridge_expansion_train_integration_result_v2.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/fiqa_bridge_expansion_train_integration_v2.py"
)
TEST_RELATIVE = Path("tests/test_fiqa_bridge_expansion_train_integration_v2.py")
RUN_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_train_integration_v2")


class FiqaTrainIntegrationV2Error(v1.FiqaTrainIntegrationError):
    """The frozen v2 TRAIN integration contract failed closed."""


class OneShotRefusal(FiqaTrainIntegrationV2Error):
    """The v2 formal attempt or result path is already consumed."""


@dataclass(frozen=True)
class ParsedTrainSourceV2:
    documents: Mapping[str, Mapping[str, str]]
    queries: Mapping[str, str]
    positive_qrels: Mapping[str, tuple[str, ...]]
    filtered_corpus_raw: bytes
    source_corpus_document_count: int
    usable_corpus_document_count: int
    empty_document_count: int
    shared_query_count: int
    train_qrel_query_count: int
    train_qrel_row_count: int
    positive_qrel_row_count: int
    unknown_document_positive_qrel_row_count: int
    empty_document_positive_qrel_row_count: int
    nonpositive_qrel_row_count: int
    self_document_positive_qrel_row_count: int


def parse_corpus_v2(
    raw: bytes,
) -> tuple[dict[str, dict[str, str]], frozenset[str], bytes, int]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaTrainIntegrationV2Error("corpus is not UTF-8") from exc
    documents: dict[str, dict[str, str]] = {}
    empty_ids: set[str] = set()
    seen_ids: set[str] = set()
    filtered_rows: list[dict[str, str]] = []
    source_count = 0
    for line in text.splitlines():
        if not line.strip():
            raise FiqaTrainIntegrationV2Error("corpus contains an empty line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FiqaTrainIntegrationV2Error("corpus JSONL is malformed") from exc
        if not isinstance(row, Mapping):
            raise FiqaTrainIntegrationV2Error("corpus row is not an object")
        source_count += 1
        document_id = v1._required_text(row.get("_id"), "document ID")
        title = v1._required_text(row.get("title", ""), "document title", allow_empty=True)
        body = v1._required_text(row.get("text", ""), "document text", allow_empty=True)
        if document_id in seen_ids:
            raise FiqaTrainIntegrationV2Error("document ID is duplicated")
        seen_ids.add(document_id)
        if not title.strip() and not body.strip():
            empty_ids.add(document_id)
            continue
        documents[document_id] = {"title": title, "text": body}
        filtered_rows.append({"_id": document_id, "title": title, "text": body})
    if not documents or source_count != len(documents) + len(empty_ids):
        raise FiqaTrainIntegrationV2Error("filtered corpus accounting drifted")
    filtered_raw = b"".join(v1.canonical_json(row) + b"\n" for row in filtered_rows)
    return documents, frozenset(empty_ids), filtered_raw, source_count


def parse_train_qrels_v2(
    raw: bytes,
    *,
    usable_document_ids: Sequence[str],
    empty_document_ids: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], set[str], dict[str, int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaTrainIntegrationV2Error("TRAIN qrels are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise FiqaTrainIntegrationV2Error("TRAIN qrel header drifted")
    usable = set(usable_document_ids)
    empty = set(empty_document_ids)
    all_source_ids = usable | empty
    qrel_query_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    positive: dict[str, set[str]] = {}
    counts = {
        "train_qrel_row_count": 0,
        "positive_qrel_row_count": 0,
        "unknown_document_positive_qrel_row_count": 0,
        "empty_document_positive_qrel_row_count": 0,
        "nonpositive_qrel_row_count": 0,
        "self_document_positive_qrel_row_count": 0,
    }
    for row in reader:
        query_id = v1._required_text(row.get("query-id"), "qrel query ID")
        document_id = v1._required_text(row.get("corpus-id"), "qrel document ID")
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise FiqaTrainIntegrationV2Error("TRAIN qrel pair is duplicated")
        seen_pairs.add(pair)
        qrel_query_ids.add(query_id)
        counts["train_qrel_row_count"] += 1
        try:
            score = int(v1._required_text(row.get("score"), "qrel score"))
        except ValueError as exc:
            raise FiqaTrainIntegrationV2Error("qrel score is not an integer") from exc
        if score <= 0:
            counts["nonpositive_qrel_row_count"] += 1
            continue
        counts["positive_qrel_row_count"] += 1
        if document_id not in all_source_ids:
            counts["unknown_document_positive_qrel_row_count"] += 1
            continue
        if document_id in empty:
            counts["empty_document_positive_qrel_row_count"] += 1
            continue
        if document_id == query_id:
            counts["self_document_positive_qrel_row_count"] += 1
            continue
        positive.setdefault(query_id, set()).add(document_id)
    return (
        {
            query_id: tuple(sorted(document_ids_for_query))
            for query_id, document_ids_for_query in sorted(positive.items())
            if document_ids_for_query
        },
        qrel_query_ids,
        counts,
    )


def parse_train_source_v2(
    *,
    corpus_raw: bytes,
    queries_raw: bytes,
    train_qrels_raw: bytes,
) -> ParsedTrainSourceV2:
    documents, empty_ids, filtered_raw, source_count = parse_corpus_v2(corpus_raw)
    positive, train_query_ids, counts = parse_train_qrels_v2(
        train_qrels_raw,
        usable_document_ids=tuple(documents),
        empty_document_ids=tuple(empty_ids),
    )
    try:
        queries, shared_query_count = v1.parse_shared_queries_for_train(
            queries_raw,
            train_query_ids=tuple(train_query_ids),
        )
    except v1.FiqaTrainIntegrationError as exc:
        raise FiqaTrainIntegrationV2Error(str(exc)) from exc
    positive = {
        query_id: document_ids
        for query_id, document_ids in positive.items()
        if query_id in queries and document_ids
    }
    if len(positive) < v1.TRAIN_DIAGNOSTIC_SIZE:
        raise FiqaTrainIntegrationV2Error("insufficient TRAIN eligible query capacity")
    return ParsedTrainSourceV2(
        documents=documents,
        queries=queries,
        positive_qrels=positive,
        filtered_corpus_raw=filtered_raw,
        source_corpus_document_count=source_count,
        usable_corpus_document_count=len(documents),
        empty_document_count=len(empty_ids),
        shared_query_count=shared_query_count,
        train_qrel_query_count=len(train_query_ids),
        train_qrel_row_count=counts["train_qrel_row_count"],
        positive_qrel_row_count=counts["positive_qrel_row_count"],
        unknown_document_positive_qrel_row_count=counts[
            "unknown_document_positive_qrel_row_count"
        ],
        empty_document_positive_qrel_row_count=counts[
            "empty_document_positive_qrel_row_count"
        ],
        nonpositive_qrel_row_count=counts["nonpositive_qrel_row_count"],
        self_document_positive_qrel_row_count=counts[
            "self_document_positive_qrel_row_count"
        ],
    )


def _load_failure_v1(project_root: Path) -> Mapping[str, Any]:
    path = project_root / "reconstruction_v2" / FAILURE_V1_RELATIVE
    if not path.is_file() or path.is_symlink() or v1.file_sha256(path) != FAILURE_V1_FILE_SHA256:
        raise FiqaTrainIntegrationV2Error("v1 failure receipt file binding drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaTrainIntegrationV2Error("v1 failure receipt is invalid") from exc
    if not isinstance(value, Mapping):
        raise FiqaTrainIntegrationV2Error("v1 failure receipt root drifted")
    v1.verify_self_hash(value, "self_sha256", FAILURE_V1_SELF_SHA256)
    return value


def _verify_freeze(project_root: Path) -> Mapping[str, Any]:
    base = project_root / "reconstruction_v2"
    path = base / FREEZE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze is invalid") from exc
    if not isinstance(value, Mapping) or value.get("schema") != FREEZE_SCHEMA:
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze hash is absent")
    v1.verify_self_hash(value, "self_sha256", declared)
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in value.get("implementation_bindings", ())
        if isinstance(row, Mapping)
    }
    expected = {
        IMPLEMENTATION_RELATIVE.as_posix(): v1.file_sha256(base / IMPLEMENTATION_RELATIVE),
        TEST_RELATIVE.as_posix(): v1.file_sha256(base / TEST_RELATIVE),
    }
    if observed != expected:
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze bindings drifted")
    if value.get("design_self_sha256") != v1.MANIFEST_BINDINGS[v1.DESIGN_RELATIVE]["self_sha256"]:
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze design binding drifted")
    if value.get("failure_v1_self_sha256") != FAILURE_V1_SELF_SHA256:
        raise FiqaTrainIntegrationV2Error("v2 implementation freeze failure binding drifted")
    return value


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    if result_path.exists():
        raise OneShotRefusal("v2 TRAIN integration result already exists")
    for relative in v1.MANIFEST_BINDINGS:
        v1._load_manifest(project_root, relative)
    _load_failure_v1(project_root)
    freeze = _verify_freeze(project_root)
    archive_path = v1._verify_archive(project_root)

    run_root = base / RUN_ROOT_RELATIVE
    try:
        run_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("v2 TRAIN integration root already exists") from exc
    marker = {
        "failure_v1_self_sha256": FAILURE_V1_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "study_design_self_sha256": v1.MANIFEST_BINDINGS[v1.DESIGN_RELATIVE]["self_sha256"],
    }
    marker_path = run_root / "attempt.marker"
    v1._exclusive_write_json(marker_path, marker)

    raw_by_role, member_bindings = v1.read_train_members(archive_path)
    parsed = parse_train_source_v2(
        corpus_raw=raw_by_role["corpus"],
        queries_raw=raw_by_role["queries"],
        train_qrels_raw=raw_by_role["train_qrels"],
    )
    corpus_path = run_root / "source_members" / "corpus.filtered.jsonl"
    v1._exclusive_write_bytes(corpus_path, parsed.filtered_corpus_raw)
    selected = v1.select_train_diagnostic(parsed)
    pack = v1.materialize_train_pack(parsed=parsed, selected=selected, run_root=run_root)

    body = {
        "claim_boundary": {
            "action_retrieval_model_evaluator_or_score_count": 0,
            "claim_eligible": False,
            "DEV_qrel_member_open_count": 0,
            "external_network_call_count": 0,
            "online_evaluator_call_count": 0,
            "TEST_qrel_member_open_count": 0,
            "TRAIN_query_ID_or_text_published_in_public_receipt": False,
        },
        "filtered_corpus_binding": {
            "relative_path": corpus_path.relative_to(base).as_posix(),
            "sha256": v1.file_sha256(corpus_path),
            "size_bytes": corpus_path.stat().st_size,
        },
        "formal_binding": {
            "attempt_marker_sha256": v1.file_sha256(marker_path),
            "failure_v1_self_sha256": FAILURE_V1_SELF_SHA256,
            "formal_implementation_commit": _git_head(project_root),
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "source_access_self_sha256": v1.MANIFEST_BINDINGS[v1.ACCESS_RELATIVE]["self_sha256"],
            "source_custody_self_sha256": v1.MANIFEST_BINDINGS[v1.CUSTODY_RELATIVE]["self_sha256"],
            "study_design_self_sha256": v1.MANIFEST_BINDINGS[v1.DESIGN_RELATIVE]["self_sha256"],
        },
        "source_aggregates": {
            "eligible_TRAIN_query_count": len(parsed.positive_qrels),
            "empty_document_count": parsed.empty_document_count,
            "empty_document_positive_qrel_row_count": parsed.empty_document_positive_qrel_row_count,
            "nonpositive_qrel_row_count": parsed.nonpositive_qrel_row_count,
            "positive_qrel_row_count": parsed.positive_qrel_row_count,
            "self_document_positive_qrel_row_count": parsed.self_document_positive_qrel_row_count,
            "shared_query_count": parsed.shared_query_count,
            "source_corpus_document_count": parsed.source_corpus_document_count,
            "train_qrel_query_count": parsed.train_qrel_query_count,
            "train_qrel_row_count": parsed.train_qrel_row_count,
            "unknown_document_positive_qrel_row_count": parsed.unknown_document_positive_qrel_row_count,
            "usable_corpus_document_count": parsed.usable_corpus_document_count,
        },
        "source_member_bindings": member_bindings,
        "source_rule": "empty_title_and_text_documents_are_source_unretrievable_and_are_excluded_before_retrieval;_their_qrels_are_excluded_and_a_query_requires_one_remaining_existing_nonself_positive",
        "TRAIN_diagnostic_pack": pack,
        "schema": SCHEMA,
        "status": "TRAIN_source_integration_v2_complete_DEV_and_TEST_qrels_unopened_no_action",
    }
    receipt = v1.self_hashed(body, "integration_sha256")
    v1._exclusive_write_json(result_path, receipt, mode=0o644)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    receipt = run_formal(arguments.project_root)
    print(
        v1.canonical_json(
            {
                "integration_sha256": receipt["integration_sha256"],
                "schema": SCHEMA,
                "status": receipt["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
