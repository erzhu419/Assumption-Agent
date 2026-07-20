"""TRAIN-only source and runtime integration for the FiQA P10 study.

This non-claim stage opens only the shared corpus/queries members and TRAIN
qrels from the hash-pinned FiQA archive.  It materializes only TRAIN query
values, extracts a deterministic public-order diagnostic pack, and never
opens DEV or TEST qrels.  It performs no retrieval, model call, or scoring.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile


SCHEMA = "fiqa_bridge_expansion_train_integration_result_v1"
ATTEMPT_SCHEMA = "fiqa_bridge_expansion_train_integration_attempt_v1"
FREEZE_SCHEMA = "fiqa_bridge_expansion_train_integration_implementation_freeze_v1"
TRAIN_DIAGNOSTIC_SIZE = 12

CUSTODY_RELATIVE = Path("manifests/fiqa_bridge_expansion_source_custody_v1.json")
ACCESS_RELATIVE = Path("manifests/fiqa_bridge_expansion_source_access_v1.json")
DESIGN_RELATIVE = Path("manifests/fiqa_bridge_expansion_study_design_v1.json")
CORE_FREEZE_RELATIVE = Path("manifests/bright_bridge_expansion_implementation_freeze_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_integration_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path("manifests/fiqa_bridge_expansion_train_integration_result_v1.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/fiqa_bridge_expansion_train_integration_v1.py"
)
TEST_RELATIVE = Path("tests/test_fiqa_bridge_expansion_train_integration_v1.py")
RUN_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_train_integration_v1")
ARCHIVE_RELATIVE = Path("artifacts/fiqa_bridge_expansion_source_v1/archives/fiqa.zip")

MANIFEST_BINDINGS = {
    CUSTODY_RELATIVE: {
        "file_sha256": "4b7e82973f22213edf3a2df24632528f17f9220f85509185f8be95d95e6b9021",
        "self_sha256": "1dca30855419ac37816cd8efa1a405c7c013cd796d22a619dea6a6e418a822ef",
    },
    ACCESS_RELATIVE: {
        "file_sha256": "86d723a1106c5f581397c34452aaa7eb586f8fe12faa2cd97039de94cdb526aa",
        "self_sha256": "b08d72e2588188ea2de5775ceba2087310f40e8a855ff821d6060bb3e9fb3137",
    },
    DESIGN_RELATIVE: {
        "file_sha256": "84792a973d1f8620de2f7b939c7330c4c7f67203408853d562d5c72a8942d705",
        "self_sha256": "7db0cb44f243d2aab9a76b58107db9a529a2e3f54ef08f8368af01ca36afb7c7",
    },
    CORE_FREEZE_RELATIVE: {
        "file_sha256": "483095de8f2c1aeaf6c1a1c2ac2ae6e5caf95950e756907356c20b9e75e85c11",
        "self_sha256": "47e102f3da12a3021929a48c525cb9c4a6b69f5d6cb4f3cc260e4a15ddac6f8b",
    },
}

ARCHIVE_BINDING = {
    "md5": "17918ed23cd04fb15047f73e6c3bd9d9",
    "sha256": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
    "size": 17_948_027,
}

TARGET_MEMBERS = {
    "corpus": "fiqa/corpus.jsonl",
    "queries": "fiqa/queries.jsonl",
    "train_qrels": "fiqa/qrels/train.tsv",
}


class FiqaTrainIntegrationError(RuntimeError):
    """The frozen TRAIN integration contract failed closed."""


class OneShotRefusal(FiqaTrainIntegrationError):
    """The formal integration attempt or result path is already consumed."""


@dataclass(frozen=True)
class ParsedTrainSource:
    documents: Mapping[str, Mapping[str, str]]
    queries: Mapping[str, str]
    positive_qrels: Mapping[str, tuple[str, ...]]
    corpus_document_count: int
    shared_query_count: int
    train_qrel_query_count: int
    train_qrel_row_count: int
    positive_qrel_row_count: int
    unknown_document_positive_qrel_row_count: int
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
        raise FiqaTrainIntegrationError("non-canonical value") from exc


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return file_digest(path, "sha256")


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(body)
    output[field] = hashlib.sha256(canonical_json(output)).hexdigest()
    return output


def verify_self_hash(value: Mapping[str, Any], field: str, expected: str) -> None:
    body = dict(value)
    declared = body.pop(field, None)
    if declared != expected or hashlib.sha256(canonical_json(body)).hexdigest() != expected:
        raise FiqaTrainIntegrationError("self hash drifted")


def _required_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise FiqaTrainIntegrationError(f"{name} is not valid text")
    if not allow_empty and not value.strip():
        raise FiqaTrainIntegrationError(f"{name} is empty")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise FiqaTrainIntegrationError(f"{name} is not UTF-8 encodable") from exc
    return value


def parse_corpus(raw: bytes) -> dict[str, dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaTrainIntegrationError("corpus is not UTF-8") from exc
    documents: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.strip():
            raise FiqaTrainIntegrationError("corpus contains an empty line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FiqaTrainIntegrationError("corpus JSONL is malformed") from exc
        if not isinstance(row, Mapping):
            raise FiqaTrainIntegrationError("corpus row is not an object")
        document_id = _required_text(row.get("_id"), "document ID")
        title = _required_text(row.get("title", ""), "document title", allow_empty=True)
        body = _required_text(row.get("text", ""), "document text", allow_empty=True)
        if not title.strip() and not body.strip():
            raise FiqaTrainIntegrationError("document title and text are both empty")
        if document_id in documents:
            raise FiqaTrainIntegrationError("document ID is duplicated")
        documents[document_id] = {"title": title, "text": body}
    if not documents:
        raise FiqaTrainIntegrationError("corpus is empty")
    return documents


def parse_train_qrels(
    raw: bytes,
    *,
    document_ids: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], set[str], dict[str, int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaTrainIntegrationError("TRAIN qrels are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise FiqaTrainIntegrationError("TRAIN qrel header drifted")
    known_documents = set(document_ids)
    qrel_query_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    positive: dict[str, set[str]] = {}
    counts = {
        "train_qrel_row_count": 0,
        "positive_qrel_row_count": 0,
        "unknown_document_positive_qrel_row_count": 0,
        "nonpositive_qrel_row_count": 0,
        "self_document_positive_qrel_row_count": 0,
    }
    for row in reader:
        query_id = _required_text(row.get("query-id"), "qrel query ID")
        document_id = _required_text(row.get("corpus-id"), "qrel document ID")
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise FiqaTrainIntegrationError("TRAIN qrel pair is duplicated")
        seen_pairs.add(pair)
        qrel_query_ids.add(query_id)
        counts["train_qrel_row_count"] += 1
        try:
            score = int(_required_text(row.get("score"), "qrel score"))
        except ValueError as exc:
            raise FiqaTrainIntegrationError("qrel score is not an integer") from exc
        if score <= 0:
            counts["nonpositive_qrel_row_count"] += 1
            continue
        counts["positive_qrel_row_count"] += 1
        if document_id not in known_documents:
            counts["unknown_document_positive_qrel_row_count"] += 1
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


def parse_shared_queries_for_train(
    raw: bytes,
    *,
    train_query_ids: Sequence[str],
) -> tuple[dict[str, str], int]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FiqaTrainIntegrationError("shared queries are not UTF-8") from exc
    requested = set(train_query_ids)
    seen: set[str] = set()
    train_queries: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            raise FiqaTrainIntegrationError("shared queries contain an empty line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FiqaTrainIntegrationError("shared query JSONL is malformed") from exc
        if not isinstance(row, Mapping):
            raise FiqaTrainIntegrationError("shared query row is not an object")
        query_id = _required_text(row.get("_id"), "query ID")
        if query_id in seen:
            raise FiqaTrainIntegrationError("shared query ID is duplicated")
        seen.add(query_id)
        query_text = _required_text(row.get("text"), "query text")
        if query_id in requested:
            train_queries[query_id] = query_text
    if set(train_queries) != requested:
        raise FiqaTrainIntegrationError("TRAIN qrel references an unknown query ID")
    return train_queries, len(seen)


def parse_train_source(
    *,
    corpus_raw: bytes,
    queries_raw: bytes,
    train_qrels_raw: bytes,
) -> ParsedTrainSource:
    documents = parse_corpus(corpus_raw)
    positive, train_query_ids, counts = parse_train_qrels(
        train_qrels_raw,
        document_ids=tuple(documents),
    )
    queries, shared_query_count = parse_shared_queries_for_train(
        queries_raw,
        train_query_ids=tuple(train_query_ids),
    )
    positive = {
        query_id: document_ids
        for query_id, document_ids in positive.items()
        if query_id in queries and document_ids
    }
    if len(positive) < TRAIN_DIAGNOSTIC_SIZE:
        raise FiqaTrainIntegrationError("insufficient TRAIN eligible query capacity")
    return ParsedTrainSource(
        documents=documents,
        queries=queries,
        positive_qrels=positive,
        corpus_document_count=len(documents),
        shared_query_count=shared_query_count,
        train_qrel_query_count=len(train_query_ids),
        train_qrel_row_count=counts["train_qrel_row_count"],
        positive_qrel_row_count=counts["positive_qrel_row_count"],
        unknown_document_positive_qrel_row_count=counts[
            "unknown_document_positive_qrel_row_count"
        ],
        nonpositive_qrel_row_count=counts["nonpositive_qrel_row_count"],
        self_document_positive_qrel_row_count=counts[
            "self_document_positive_qrel_row_count"
        ],
    )


def select_train_diagnostic(parsed: ParsedTrainSource) -> tuple[str, ...]:
    eligible = tuple(parsed.positive_qrels)
    ordered = sorted(
        eligible,
        key=lambda query_id: (
            hashlib.sha256(f"FIQA_TRAIN\x00{query_id}".encode("utf-8")).digest(),
            query_id,
        ),
    )
    selected = tuple(ordered[:TRAIN_DIAGNOSTIC_SIZE])
    if len(selected) != TRAIN_DIAGNOSTIC_SIZE or len(set(selected)) != len(selected):
        raise FiqaTrainIntegrationError("TRAIN diagnostic selection drifted")
    return selected


def _safe_member(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise FiqaTrainIntegrationError("archive member path is unsafe")
    return value


def read_train_members(archive_path: Path) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    targets = {role: _safe_member(value) for role, value in TARGET_MEMBERS.items()}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            output: dict[str, bytes] = {}
            bindings: dict[str, dict[str, Any]] = {}
            for role in ("corpus", "train_qrels", "queries"):
                target = targets[role]
                if names.count(target) != 1:
                    raise FiqaTrainIntegrationError(
                        "required TRAIN integration member is absent or duplicated"
                    )
                info = next(info for info in infos if info.filename == target)
                if info.is_dir() or info.file_size > 2_000_000_000:
                    raise FiqaTrainIntegrationError("required archive member is invalid")
                raw = archive.read(info)
                output[role] = raw
                bindings[role] = {
                    "archive_member": target,
                    "byte_size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
    except FiqaTrainIntegrationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise FiqaTrainIntegrationError("TRAIN archive member access failed") from exc
    return output, bindings


def _exclusive_write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_write_json(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> None:
    _exclusive_write_bytes(path, canonical_json(payload) + b"\n", mode=mode)


def _exclusive_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    mode: int = 0o600,
) -> None:
    _exclusive_write_bytes(
        path,
        b"".join(canonical_json(row) + b"\n" for row in rows),
        mode=mode,
    )


def materialize_train_pack(
    *,
    parsed: ParsedTrainSource,
    selected: Sequence[str],
    run_root: Path,
) -> dict[str, Any]:
    view_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for query_id in selected:
        item_key = hashlib.sha256(
            f"FIQA_TRAIN_INTEGRATION\x00{query_id}".encode("utf-8")
        ).hexdigest()
        view_rows.append(
            {
                "excluded_document_ids": [query_id] if query_id in parsed.documents else [],
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
    view_path = run_root / "train_integration.view.jsonl"
    label_path = run_root / "train_integration.labels.jsonl"
    _exclusive_write_jsonl(view_path, view_rows)
    _exclusive_write_jsonl(label_path, label_rows)
    return {
        "item_count": len(view_rows),
        "label_file_sha256": file_sha256(label_path),
        "label_file_size_bytes": label_path.stat().st_size,
        "view_file_sha256": file_sha256(view_path),
        "view_file_size_bytes": view_path.stat().st_size,
    }


def _load_manifest(project_root: Path, relative: Path) -> Mapping[str, Any]:
    path = project_root / "reconstruction_v2" / relative
    binding = MANIFEST_BINDINGS[relative]
    if not path.is_file() or path.is_symlink() or file_sha256(path) != binding["file_sha256"]:
        raise FiqaTrainIntegrationError("public manifest file binding drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaTrainIntegrationError("public manifest is invalid") from exc
    if not isinstance(value, Mapping):
        raise FiqaTrainIntegrationError("public manifest is not an object")
    verify_self_hash(value, "self_sha256", binding["self_sha256"])
    return value


def _verify_freeze(project_root: Path) -> Mapping[str, Any]:
    base = project_root / "reconstruction_v2"
    path = base / FREEZE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise FiqaTrainIntegrationError("implementation freeze is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaTrainIntegrationError("implementation freeze is invalid") from exc
    if not isinstance(value, Mapping) or value.get("schema") != FREEZE_SCHEMA:
        raise FiqaTrainIntegrationError("implementation freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise FiqaTrainIntegrationError("implementation freeze hash is absent")
    verify_self_hash(value, "self_sha256", declared)
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in value.get("implementation_bindings", ())
        if isinstance(row, Mapping)
    }
    expected = {
        IMPLEMENTATION_RELATIVE.as_posix(): file_sha256(base / IMPLEMENTATION_RELATIVE),
        TEST_RELATIVE.as_posix(): file_sha256(base / TEST_RELATIVE),
    }
    if observed != expected:
        raise FiqaTrainIntegrationError("implementation freeze bindings drifted")
    if value.get("design_self_sha256") != MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"]:
        raise FiqaTrainIntegrationError("implementation freeze design binding drifted")
    return value


def _verify_archive(project_root: Path) -> Path:
    path = project_root / "reconstruction_v2" / ARCHIVE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise FiqaTrainIntegrationError("FiQA archive is unavailable")
    if (
        path.stat().st_size != ARCHIVE_BINDING["size"]
        or file_sha256(path) != ARCHIVE_BINDING["sha256"]
        or file_digest(path, "md5") != ARCHIVE_BINDING["md5"]
    ):
        raise FiqaTrainIntegrationError("FiQA archive binding drifted")
    return path


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
        raise OneShotRefusal("TRAIN integration result already exists")
    for relative in MANIFEST_BINDINGS:
        _load_manifest(project_root, relative)
    freeze = _verify_freeze(project_root)
    archive_path = _verify_archive(project_root)

    run_root = base / RUN_ROOT_RELATIVE
    try:
        run_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("TRAIN integration root already exists") from exc
    marker = {
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "study_design_self_sha256": MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"],
    }
    marker_path = run_root / "attempt.marker"
    _exclusive_write_json(marker_path, marker)

    raw_by_role, member_bindings = read_train_members(archive_path)
    corpus_path = run_root / "source_members" / "corpus.jsonl"
    _exclusive_write_bytes(corpus_path, raw_by_role["corpus"])
    parsed = parse_train_source(
        corpus_raw=raw_by_role["corpus"],
        queries_raw=raw_by_role["queries"],
        train_qrels_raw=raw_by_role["train_qrels"],
    )
    selected = select_train_diagnostic(parsed)
    pack = materialize_train_pack(parsed=parsed, selected=selected, run_root=run_root)

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
        "formal_binding": {
            "attempt_marker_sha256": file_sha256(marker_path),
            "formal_implementation_commit": _git_head(project_root),
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "source_access_self_sha256": MANIFEST_BINDINGS[ACCESS_RELATIVE]["self_sha256"],
            "source_custody_self_sha256": MANIFEST_BINDINGS[CUSTODY_RELATIVE]["self_sha256"],
            "study_design_self_sha256": MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"],
        },
        "source_aggregates": {
            "corpus_document_count": parsed.corpus_document_count,
            "eligible_TRAIN_query_count": len(parsed.positive_qrels),
            "nonpositive_qrel_row_count": parsed.nonpositive_qrel_row_count,
            "positive_qrel_row_count": parsed.positive_qrel_row_count,
            "self_document_positive_qrel_row_count": parsed.self_document_positive_qrel_row_count,
            "shared_query_count": parsed.shared_query_count,
            "train_qrel_query_count": parsed.train_qrel_query_count,
            "train_qrel_row_count": parsed.train_qrel_row_count,
            "unknown_document_positive_qrel_row_count": parsed.unknown_document_positive_qrel_row_count,
        },
        "source_member_bindings": member_bindings,
        "TRAIN_diagnostic_pack": pack,
        "schema": SCHEMA,
        "status": "TRAIN_source_integration_complete_DEV_and_TEST_qrels_unopened_no_action",
    }
    receipt = self_hashed(body, "integration_sha256")
    _exclusive_write_json(result_path, receipt, mode=0o644)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    receipt = run_formal(arguments.project_root)
    print(
        canonical_json(
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
