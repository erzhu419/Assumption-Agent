"""One-shot DEV acquisition for the frozen FiQA bridge-expansion study.

The formal entrypoint first verifies its implementation freeze and the
completed non-claim TRAIN runtime.  It then creates one private HMAC cohort,
opens the shared query member and DEV qrels exactly once, and materializes
separate view and label packs.  TEST qrels, retrieval, models, and scoring are
outside this stage.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_integration_v1 as integration_v1,
)


SCHEMA = "fiqa_bridge_expansion_dev_acquisition_result_v1"
ATTEMPT_SCHEMA = "fiqa_bridge_expansion_dev_acquisition_attempt_v1"
FREEZE_SCHEMA = "fiqa_bridge_expansion_dev_acquisition_implementation_freeze_v1"
COHORT_SIZE = 48

FREEZE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_dev_acquisition_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_dev_acquisition_result_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/fiqa_bridge_expansion_dev_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_fiqa_bridge_expansion_dev_acquisition_v1.py")
RUN_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_dev_acquisition_v1")
ARCHIVE_RELATIVE = Path("artifacts/fiqa_bridge_expansion_source_v1/archives/fiqa.zip")
INTEGRATION_RESULT_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_integration_result_v2.json"
)
TRAIN_RUNTIME_RESULT_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_runtime_result_v2.json"
)

INTEGRATION_RESULT_FILE_SHA256 = (
    "ff24838e9a238c606462b7142cf29571435a63226a8559a02cedd5bdf7c30890"
)
INTEGRATION_RESULT_SELF_SHA256 = (
    "c194ed16cd83e89b01a1058dbde5f77a4139671893a5170332959953badeb032"
)
TRAIN_RUNTIME_RESULT_FILE_SHA256 = (
    "79a66c4e04bc8af7845572354916cb98b9203885fe2a9a7ef6d738ba2810513f"
)
TRAIN_RUNTIME_RESULT_SELF_SHA256 = (
    "6a380300caadb53c0329fce7b342122ad78b27f392ed4eb0d7cfe774c0150f4c"
)

TARGET_MEMBERS = {
    "queries": "fiqa/queries.jsonl",
    "dev_qrels": "fiqa/qrels/dev.tsv",
}


class FiqaDevAcquisitionError(RuntimeError):
    """The frozen FiQA DEV acquisition failed closed."""


class OneShotRefusal(FiqaDevAcquisitionError):
    """The formal DEV acquisition root or result is already consumed."""


@dataclass(frozen=True)
class ParsedDevSource:
    queries: Mapping[str, str]
    positive_qrels: Mapping[str, tuple[str, ...]]
    shared_query_count: int
    dev_qrel_query_count: int
    dev_qrel_row_count: int
    positive_qrel_row_count: int
    source_unretrievable_positive_qrel_row_count: int
    nonpositive_qrel_row_count: int
    self_document_positive_qrel_row_count: int


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FiqaDevAcquisitionError("non-canonical value") from exc


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(body)
    output[field] = hashlib.sha256(canonical_json(output)).hexdigest()
    return output


def verify_self_hash(value: Mapping[str, Any], field: str, expected: str) -> None:
    body = dict(value)
    declared = body.pop(field, None)
    if declared != expected or hashlib.sha256(canonical_json(body)).hexdigest() != expected:
        raise FiqaDevAcquisitionError("self hash drifted")


def _required_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    try:
        return integration_v1._required_text(value, name, allow_empty=allow_empty)
    except integration_v1.FiqaTrainIntegrationError as exc:
        raise FiqaDevAcquisitionError(str(exc)) from exc


def parse_filtered_corpus_ids(raw: bytes) -> tuple[str, ...]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise FiqaDevAcquisitionError("filtered corpus is not canonical JSONL")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FiqaDevAcquisitionError("filtered corpus JSONL is malformed") from exc
        if (
            not isinstance(row, Mapping)
            or set(row) != {"_id", "text", "title"}
            or canonical_json(row) + b"\n" != line
        ):
            raise FiqaDevAcquisitionError("filtered corpus row drifted")
        identifier = _required_text(row.get("_id"), "document ID")
        title = _required_text(row.get("title"), "document title", allow_empty=True)
        text = _required_text(row.get("text"), "document text", allow_empty=True)
        if identifier in seen or (not title.strip() and not text.strip()):
            raise FiqaDevAcquisitionError("filtered corpus identity drifted")
        seen.add(identifier)
        identifiers.append(identifier)
    if not identifiers:
        raise FiqaDevAcquisitionError("filtered corpus is empty")
    return tuple(identifiers)


def parse_shared_queries(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaDevAcquisitionError("shared queries are not UTF-8") from exc
    queries: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            raise FiqaDevAcquisitionError("shared queries contain an empty line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FiqaDevAcquisitionError("shared query JSONL is malformed") from exc
        if not isinstance(row, Mapping):
            raise FiqaDevAcquisitionError("shared query row is not an object")
        query_id = _required_text(row.get("_id"), "query ID")
        query = _required_text(row.get("text"), "query text")
        if query_id in queries:
            raise FiqaDevAcquisitionError("shared query ID is duplicated")
        queries[query_id] = query
    if not queries:
        raise FiqaDevAcquisitionError("shared queries are empty")
    return queries


def parse_dev_qrels(
    raw: bytes,
    *,
    query_ids: Sequence[str],
    usable_document_ids: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], set[str], dict[str, int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaDevAcquisitionError("DEV qrels are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise FiqaDevAcquisitionError("DEV qrel header drifted")
    known_queries = set(query_ids)
    usable_documents = set(usable_document_ids)
    seen_pairs: set[tuple[str, str]] = set()
    qrel_queries: set[str] = set()
    positive: dict[str, set[str]] = {}
    counts = {
        "dev_qrel_row_count": 0,
        "positive_qrel_row_count": 0,
        "source_unretrievable_positive_qrel_row_count": 0,
        "nonpositive_qrel_row_count": 0,
        "self_document_positive_qrel_row_count": 0,
    }
    for row in reader:
        query_id = _required_text(row.get("query-id"), "qrel query ID")
        document_id = _required_text(row.get("corpus-id"), "qrel document ID")
        if query_id not in known_queries:
            raise FiqaDevAcquisitionError("DEV qrel references an unknown query ID")
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise FiqaDevAcquisitionError("DEV qrel pair is duplicated")
        seen_pairs.add(pair)
        qrel_queries.add(query_id)
        counts["dev_qrel_row_count"] += 1
        try:
            score = int(_required_text(row.get("score"), "qrel score"))
        except ValueError as exc:
            raise FiqaDevAcquisitionError("DEV qrel score is not an integer") from exc
        if score <= 0:
            counts["nonpositive_qrel_row_count"] += 1
            continue
        counts["positive_qrel_row_count"] += 1
        if document_id not in usable_documents:
            counts["source_unretrievable_positive_qrel_row_count"] += 1
            continue
        if document_id == query_id:
            counts["self_document_positive_qrel_row_count"] += 1
            continue
        positive.setdefault(query_id, set()).add(document_id)
    return (
        {
            query_id: tuple(sorted(document_ids))
            for query_id, document_ids in sorted(positive.items())
            if document_ids
        },
        qrel_queries,
        counts,
    )


def parse_dev_source(
    *,
    queries_raw: bytes,
    dev_qrels_raw: bytes,
    usable_document_ids: Sequence[str],
) -> ParsedDevSource:
    queries = parse_shared_queries(queries_raw)
    positive, qrel_query_ids, counts = parse_dev_qrels(
        dev_qrels_raw,
        query_ids=tuple(queries),
        usable_document_ids=usable_document_ids,
    )
    if len(positive) < COHORT_SIZE:
        raise FiqaDevAcquisitionError("insufficient eligible DEV query capacity")
    return ParsedDevSource(
        queries=queries,
        positive_qrels=positive,
        shared_query_count=len(queries),
        dev_qrel_query_count=len(qrel_query_ids),
        dev_qrel_row_count=counts["dev_qrel_row_count"],
        positive_qrel_row_count=counts["positive_qrel_row_count"],
        source_unretrievable_positive_qrel_row_count=counts[
            "source_unretrievable_positive_qrel_row_count"
        ],
        nonpositive_qrel_row_count=counts["nonpositive_qrel_row_count"],
        self_document_positive_qrel_row_count=counts[
            "self_document_positive_qrel_row_count"
        ],
    )


def select_dev_cohort(parsed: ParsedDevSource, secret: bytes) -> tuple[str, ...]:
    if len(secret) != 32:
        raise FiqaDevAcquisitionError("DEV selection secret length drifted")
    ordered = sorted(
        parsed.positive_qrels,
        key=lambda query_id: (
            hmac.new(
                secret,
                f"FIQA_DEV\x00{query_id}".encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            query_id,
        ),
    )
    selected = tuple(ordered[:COHORT_SIZE])
    if len(selected) != COHORT_SIZE or len(set(selected)) != COHORT_SIZE:
        raise FiqaDevAcquisitionError("DEV cohort selection drifted")
    return selected


def _item_key(secret: bytes, query_id: str) -> str:
    return hmac.new(
        secret,
        f"FIQA_DEV_ITEM\x00{query_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _exclusive_write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_write_json(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> None:
    _exclusive_write_bytes(path, canonical_json(payload) + b"\n", mode=mode)


def _exclusive_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _exclusive_write_bytes(
        path,
        b"".join(canonical_json(row) + b"\n" for row in rows),
    )


def materialize_dev_pack(
    *,
    parsed: ParsedDevSource,
    selected: Sequence[str],
    usable_document_ids: Sequence[str],
    secret: bytes,
    run_root: Path,
) -> dict[str, Any]:
    if len(selected) != COHORT_SIZE or len(set(selected)) != COHORT_SIZE:
        raise FiqaDevAcquisitionError("DEV materialization cohort drifted")
    usable = set(usable_document_ids)
    view_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for query_id in selected:
        if query_id not in parsed.positive_qrels or query_id not in parsed.queries:
            raise FiqaDevAcquisitionError("selected DEV identity drifted")
        item_key = _item_key(secret, query_id)
        view_rows.append(
            {
                "excluded_document_ids": [query_id] if query_id in usable else [],
                "family": "FIQA",
                "item_key": item_key,
                "query": parsed.queries[query_id],
                "source_query_id": query_id,
            }
        )
        label_rows.append(
            {
                "family": "FIQA",
                "gold_document_ids": list(parsed.positive_qrels[query_id]),
                "item_key": item_key,
            }
        )
    view_path = run_root / "C_confirm.view.jsonl"
    label_path = run_root / "C_confirm.labels.jsonl"
    _exclusive_write_jsonl(view_path, view_rows)
    _exclusive_write_jsonl(label_path, label_rows)
    return {
        "item_count": COHORT_SIZE,
        "label_file_sha256": integration_v1.file_sha256(label_path),
        "label_file_size_bytes": label_path.stat().st_size,
        "view_file_sha256": integration_v1.file_sha256(view_path),
        "view_file_size_bytes": view_path.stat().st_size,
    }


def _safe_member(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise FiqaDevAcquisitionError("archive member path is unsafe")
    return value


def read_dev_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    targets = {role: _safe_member(value) for role, value in TARGET_MEMBERS.items()}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            output: dict[str, bytes] = {}
            bindings: dict[str, dict[str, Any]] = {}
            for role in ("queries", "dev_qrels"):
                target = targets[role]
                if names.count(target) != 1:
                    raise FiqaDevAcquisitionError(
                        "required DEV member is absent or duplicated"
                    )
                info = next(info for info in infos if info.filename == target)
                if info.is_dir() or info.file_size > 2_000_000_000:
                    raise FiqaDevAcquisitionError("required DEV member is invalid")
                raw = archive.read(info)
                output[role] = raw
                bindings[role] = {
                    "archive_member": target,
                    "byte_size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
    except FiqaDevAcquisitionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise FiqaDevAcquisitionError("DEV archive member access failed") from exc
    return output, bindings


def _load_bound_json(
    base: Path,
    relative: Path,
    *,
    file_sha256: str,
    self_field: str,
    self_sha256: str,
) -> Mapping[str, Any]:
    path = base / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or integration_v1.file_sha256(path) != file_sha256
    ):
        raise FiqaDevAcquisitionError("bound prerequisite file drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaDevAcquisitionError("bound prerequisite is invalid") from exc
    if not isinstance(value, Mapping):
        raise FiqaDevAcquisitionError("bound prerequisite root drifted")
    verify_self_hash(value, self_field, self_sha256)
    return value


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    path = base / FREEZE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise FiqaDevAcquisitionError("DEV acquisition freeze is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaDevAcquisitionError("DEV acquisition freeze is invalid") from exc
    if not isinstance(value, Mapping) or value.get("schema") != FREEZE_SCHEMA:
        raise FiqaDevAcquisitionError("DEV acquisition freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise FiqaDevAcquisitionError("DEV acquisition freeze identity is absent")
    verify_self_hash(value, "self_sha256", declared)
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in value.get("implementation_bindings", ())
        if isinstance(row, Mapping)
    }
    expected = {
        IMPLEMENTATION_RELATIVE.as_posix(): integration_v1.file_sha256(
            base / IMPLEMENTATION_RELATIVE
        ),
        TEST_RELATIVE.as_posix(): integration_v1.file_sha256(base / TEST_RELATIVE),
    }
    if observed != expected:
        raise FiqaDevAcquisitionError("DEV acquisition freeze bindings drifted")
    if (
        value.get("study_design_self_sha256")
        != integration_v1.MANIFEST_BINDINGS[integration_v1.DESIGN_RELATIVE][
            "self_sha256"
        ]
        or value.get("TRAIN_runtime_result_self_sha256")
        != TRAIN_RUNTIME_RESULT_SELF_SHA256
    ):
        raise FiqaDevAcquisitionError("DEV acquisition freeze prerequisite drifted")
    return value


def _verify_archive(project_root: Path) -> Path:
    try:
        return integration_v1._verify_archive(project_root)
    except integration_v1.FiqaTrainIntegrationError as exc:
        raise FiqaDevAcquisitionError(str(exc)) from exc


def _load_filtered_corpus_ids(
    base: Path,
    integration: Mapping[str, Any],
) -> tuple[str, ...]:
    binding = integration.get("filtered_corpus_binding")
    if not isinstance(binding, Mapping):
        raise FiqaDevAcquisitionError("filtered corpus binding drifted")
    relative = binding.get("relative_path")
    if not isinstance(relative, str):
        raise FiqaDevAcquisitionError("filtered corpus path drifted")
    path = base / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != binding.get("size_bytes")
        or integration_v1.file_sha256(path) != binding.get("sha256")
    ):
        raise FiqaDevAcquisitionError("filtered corpus file drifted")
    identifiers = parse_filtered_corpus_ids(path.read_bytes())
    aggregates = integration.get("source_aggregates")
    if (
        not isinstance(aggregates, Mapping)
        or len(identifiers) != aggregates.get("usable_corpus_document_count")
    ):
        raise FiqaDevAcquisitionError("filtered corpus count drifted")
    return identifiers


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
        raise OneShotRefusal("DEV acquisition result already exists")
    for relative in integration_v1.MANIFEST_BINDINGS:
        try:
            integration_v1._load_manifest(project_root, relative)
        except integration_v1.FiqaTrainIntegrationError as exc:
            raise FiqaDevAcquisitionError(str(exc)) from exc
    integration = _load_bound_json(
        base,
        INTEGRATION_RESULT_RELATIVE,
        file_sha256=INTEGRATION_RESULT_FILE_SHA256,
        self_field="integration_sha256",
        self_sha256=INTEGRATION_RESULT_SELF_SHA256,
    )
    train_runtime = _load_bound_json(
        base,
        TRAIN_RUNTIME_RESULT_RELATIVE,
        file_sha256=TRAIN_RUNTIME_RESULT_FILE_SHA256,
        self_field="result_sha256",
        self_sha256=TRAIN_RUNTIME_RESULT_SELF_SHA256,
    )
    train_boundary = train_runtime.get("claim_boundary")
    if (
        train_runtime.get("status")
        != "TRAIN_end_to_end_runtime_v2_complete_nonclaim_true_late_label_DEV_and_TEST_unopened"
        or not isinstance(train_boundary, Mapping)
        or train_boundary.get("DEV_qrel_member_open_count") != 0
        or train_boundary.get("TEST_qrel_member_open_count") != 0
    ):
        raise FiqaDevAcquisitionError("TRAIN runtime completion boundary drifted")
    freeze = _verify_freeze(base)
    archive_path = _verify_archive(project_root)
    usable_document_ids = _load_filtered_corpus_ids(base, integration)

    run_root = base / RUN_ROOT_RELATIVE
    try:
        run_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("DEV acquisition root already exists") from exc
    marker = {
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "study_design_self_sha256": integration_v1.MANIFEST_BINDINGS[
            integration_v1.DESIGN_RELATIVE
        ]["self_sha256"],
        "TRAIN_runtime_result_self_sha256": TRAIN_RUNTIME_RESULT_SELF_SHA256,
    }
    marker_path = run_root / "attempt.marker"
    _exclusive_write_json(marker_path, marker)
    secret = os.urandom(32)
    secret_path = run_root / "selection.secret"
    _exclusive_write_bytes(secret_path, secret)

    raw_by_role, member_bindings = read_dev_members(archive_path)
    qrel_member_path = run_root / "source_members" / "qrels.dev.tsv"
    _exclusive_write_bytes(qrel_member_path, raw_by_role["dev_qrels"])
    member_bindings = {
        role: dict(binding) for role, binding in member_bindings.items()
    }
    member_bindings["dev_qrels"]["extracted_relative_path"] = (
        qrel_member_path.relative_to(base).as_posix()
    )
    parsed = parse_dev_source(
        queries_raw=raw_by_role["queries"],
        dev_qrels_raw=raw_by_role["dev_qrels"],
        usable_document_ids=usable_document_ids,
    )
    selected = select_dev_cohort(parsed, secret)
    pack = materialize_dev_pack(
        parsed=parsed,
        selected=selected,
        usable_document_ids=usable_document_ids,
        secret=secret,
        run_root=run_root,
    )

    receipt = self_hashed(
        {
            "C_confirm_pack": pack,
            "claim_boundary": {
                "action_evaluator_model_or_score_count": 0,
                "archive_member_payload_open_count": 2,
                "DEV_qrel_member_open_count": 1,
                "external_network_call_count": 0,
                "gold_or_qrel_exposed_to_model": False,
                "online_evaluator_call_count": 0,
                "selection_secret_published": False,
                "TEST_qrel_member_open_count": 0,
            },
            "formal_binding": {
                "attempt_marker_sha256": integration_v1.file_sha256(marker_path),
                "formal_implementation_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
                "source_access_self_sha256": integration_v1.MANIFEST_BINDINGS[
                    integration_v1.ACCESS_RELATIVE
                ]["self_sha256"],
                "source_custody_self_sha256": integration_v1.MANIFEST_BINDINGS[
                    integration_v1.CUSTODY_RELATIVE
                ]["self_sha256"],
                "study_design_self_sha256": integration_v1.MANIFEST_BINDINGS[
                    integration_v1.DESIGN_RELATIVE
                ]["self_sha256"],
                "TRAIN_integration_result_self_sha256": INTEGRATION_RESULT_SELF_SHA256,
                "TRAIN_runtime_result_self_sha256": TRAIN_RUNTIME_RESULT_SELF_SHA256,
            },
            "schema": SCHEMA,
            "source_aggregates": {
                "dev_qrel_query_count": parsed.dev_qrel_query_count,
                "dev_qrel_row_count": parsed.dev_qrel_row_count,
                "eligible_DEV_query_count": len(parsed.positive_qrels),
                "nonpositive_qrel_row_count": parsed.nonpositive_qrel_row_count,
                "positive_qrel_row_count": parsed.positive_qrel_row_count,
                "self_document_positive_qrel_row_count": parsed.self_document_positive_qrel_row_count,
                "shared_query_count": parsed.shared_query_count,
                "source_unretrievable_positive_qrel_row_count": parsed.source_unretrievable_positive_qrel_row_count,
                "usable_corpus_document_count": len(usable_document_ids),
            },
            "source_member_bindings": member_bindings,
            "source_rule": "reuse_the_frozen_57600_document_filtered_corpus;_exclude_positive_qrels_to_source_unretrievable_or_self_documents;_require_one_remaining_positive",
            "status": "one_shot_FiQA_DEV_C_confirm_acquired_labels_separated_no_action_TEST_unopened",
        },
        "acquisition_sha256",
    )
    _exclusive_write_json(result_path, receipt, mode=0o644)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_formal(arguments.project_root)
    print(
        canonical_json(
            {
                "acquisition_sha256": result["acquisition_sha256"],
                "schema": SCHEMA,
                "status": result["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
