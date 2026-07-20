"""One-shot direct acquisition for the BEIR bridge-expansion study.

The formal entrypoint verifies three frozen BEIR archives, consumes a marker
and fresh HMAC secret, opens exactly corpus/queries/test-qrels members, and
writes label-separated private blocks.  It performs no retrieval, model call,
or scoring.
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


SCHEMA = "beir_bridge_expansion_acquisition_result_v1"
ATTEMPT_SCHEMA = "beir_bridge_expansion_acquisition_attempt_v1"
FREEZE_SCHEMA = "beir_bridge_expansion_acquisition_implementation_freeze_v1"
FAMILY_ORDER = ("NFCORPUS", "ARGUANA", "SCIDOCS")
BLOCK_SIZES = (
    ("G_form", 16),
    ("C_confirm", 24),
    ("A_form", 12),
    ("F_search", 12),
    ("A_hold", 16),
    ("M_search", 16),
)
TOTAL_PER_FAMILY = sum(size for _, size in BLOCK_SIZES)

CUSTODY_RELATIVE = Path("manifests/beir_bridge_expansion_source_custody_v1.json")
ACCESS_RELATIVE = Path("manifests/beir_bridge_expansion_source_access_v1.json")
DESIGN_RELATIVE = Path("manifests/beir_bridge_expansion_study_design_v1.json")
CORE_FREEZE_RELATIVE = Path(
    "manifests/bright_bridge_expansion_implementation_freeze_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/beir_bridge_expansion_acquisition_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path("manifests/beir_bridge_expansion_acquisition_result_v1.json")
ACQUISITION_RELATIVE = Path(
    "assumption_agent/benchmarks/beir_bridge_expansion_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_beir_bridge_expansion_acquisition_v1.py")
ATTEMPT_ROOT_RELATIVE = Path("artifacts/beir_bridge_expansion_acquisition_v1")
ARCHIVE_ROOT_RELATIVE = Path("artifacts/beir_bridge_expansion_source_v1/archives")

MANIFEST_BINDINGS = {
    CUSTODY_RELATIVE: {
        "file_sha256": "bdad390370e094dc2d7b7f71566261ed77d23b564dbfa9bd5abe062636043026",
        "self_sha256": "e680c9f49a3ee4dfe62e11201d6f0e9a466ebebe907455b73c34c9655a016fb4",
    },
    ACCESS_RELATIVE: {
        "file_sha256": "3b67cb9c6427380a1b79c99dfd1fd3740a1d547a11d8778a551dafcd319e97d2",
        "self_sha256": "c6340669946d4c6753e6024ccefafd738cff6382df4d051107229eecee75a905",
    },
    DESIGN_RELATIVE: {
        "file_sha256": "5d1dc8c0ff3f9d28ad91174a052cb9fdc45f0cba28bc147aa9494f3a42d54e8c",
        "self_sha256": "088460f99d647a02cc7b8f1d5b4577fe8cfbd5ce109d714a7bf73f6a67f0905e",
    },
    CORE_FREEZE_RELATIVE: {
        "file_sha256": "483095de8f2c1aeaf6c1a1c2ac2ae6e5caf95950e756907356c20b9e75e85c11",
        "self_sha256": "47e102f3da12a3021929a48c525cb9c4a6b69f5d6cb4f3cc260e4a15ddac6f8b",
    },
}

ARCHIVE_BINDINGS = {
    "NFCORPUS": {
        "filename": "nfcorpus.zip",
        "md5": "a89dba18a62ef92f7d323ec890a0d38d",
        "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "size": 2_448_432,
        "root": "nfcorpus",
    },
    "ARGUANA": {
        "filename": "arguana.zip",
        "md5": "8ad3e3c2a5867cdced806d6503f29b99",
        "sha256": "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
        "size": 3_773_617,
        "root": "arguana",
    },
    "SCIDOCS": {
        "filename": "scidocs.zip",
        "md5": "38121350fc3a4d2f48850f6aff52e4a9",
        "sha256": "96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1",
        "size": 142_471_588,
        "root": "scidocs",
    },
}


class BeirAcquisitionError(RuntimeError):
    """The frozen direct-acquisition contract failed closed."""


class OneShotRefusal(BeirAcquisitionError):
    """The formal attempt or result path was already consumed."""


@dataclass(frozen=True)
class ParsedFamily:
    documents: Mapping[str, Mapping[str, str]]
    queries: Mapping[str, str]
    positive_qrels: Mapping[str, tuple[str, ...]]
    member_bindings: Mapping[str, Mapping[str, Any]]
    self_document_query_count: int


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
        raise BeirAcquisitionError("non-canonical value") from exc


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
        raise BeirAcquisitionError("self hash drifted")


def _required_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise BeirAcquisitionError(f"{name} is not valid text")
    if not allow_empty and not value.strip():
        raise BeirAcquisitionError(f"{name} is empty")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise BeirAcquisitionError(f"{name} is not UTF-8 encodable") from exc
    return value


def parse_corpus(raw: bytes) -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BeirAcquisitionError("corpus is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise BeirAcquisitionError("corpus contains an empty line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BeirAcquisitionError("corpus JSONL is malformed") from exc
        if not isinstance(row, Mapping):
            raise BeirAcquisitionError("corpus row is not an object")
        document_id = _required_text(row.get("_id"), "document ID")
        title = _required_text(row.get("title", ""), "document title", allow_empty=True)
        body = _required_text(row.get("text", ""), "document text", allow_empty=True)
        if not title.strip() and not body.strip():
            raise BeirAcquisitionError("document title and text are both empty")
        if document_id in documents:
            raise BeirAcquisitionError("document ID is duplicated")
        documents[document_id] = {"title": title, "text": body}
    if not documents:
        raise BeirAcquisitionError("corpus is empty")
    return documents


def parse_queries(raw: bytes) -> dict[str, str]:
    queries: dict[str, str] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BeirAcquisitionError("queries are not UTF-8") from exc
    for line in text.splitlines():
        if not line.strip():
            raise BeirAcquisitionError("queries contain an empty line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BeirAcquisitionError("query JSONL is malformed") from exc
        if not isinstance(row, Mapping):
            raise BeirAcquisitionError("query row is not an object")
        query_id = _required_text(row.get("_id"), "query ID")
        query = _required_text(row.get("text"), "query text")
        if query_id in queries:
            raise BeirAcquisitionError("query ID is duplicated")
        queries[query_id] = query
    if not queries:
        raise BeirAcquisitionError("queries are empty")
    return queries


def parse_qrels(
    raw: bytes,
    *,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BeirAcquisitionError("qrels are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise BeirAcquisitionError("qrel header drifted")
    known_queries = set(query_ids)
    known_documents = set(document_ids)
    positive: dict[str, set[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in reader:
        query_id = _required_text(row.get("query-id"), "qrel query ID")
        document_id = _required_text(row.get("corpus-id"), "qrel document ID")
        if query_id not in known_queries or document_id not in known_documents:
            raise BeirAcquisitionError("qrel references an unknown ID")
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise BeirAcquisitionError("qrel pair is duplicated")
        seen_pairs.add(pair)
        try:
            score = int(_required_text(row.get("score"), "qrel score"))
        except ValueError as exc:
            raise BeirAcquisitionError("qrel score is not an integer") from exc
        if score > 0:
            positive.setdefault(query_id, set()).add(document_id)
    return {
        query_id: tuple(sorted(documents))
        for query_id, documents in sorted(positive.items())
        if documents
    }


def _safe_target_member(root: str, role: str) -> str:
    if role == "corpus":
        suffix = "corpus.jsonl"
    elif role == "queries":
        suffix = "queries.jsonl"
    elif role == "qrels":
        suffix = "qrels/test.tsv"
    else:
        raise BeirAcquisitionError("unknown source member role")
    value = f"{root}/{suffix}"
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise BeirAcquisitionError("target member path is unsafe")
    return value


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
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    _exclusive_write_bytes(path, raw, mode=mode)


def open_family_archive(
    archive_path: Path,
    *,
    family: str,
    extraction_root: Path,
) -> ParsedFamily:
    if family not in FAMILY_ORDER:
        raise BeirAcquisitionError("family is not frozen")
    binding = ARCHIVE_BINDINGS[family]
    targets = {
        role: _safe_target_member(binding["root"], role)
        for role in ("corpus", "queries", "qrels")
    }
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            for target in targets.values():
                if names.count(target) != 1:
                    raise BeirAcquisitionError("required archive member is absent or duplicated")
            raw_by_role: dict[str, bytes] = {}
            member_bindings: dict[str, dict[str, Any]] = {}
            for role, target in targets.items():
                info = next(info for info in infos if info.filename == target)
                if info.is_dir() or info.file_size > 2_000_000_000:
                    raise BeirAcquisitionError("required archive member is invalid")
                raw = archive.read(info)
                raw_by_role[role] = raw
                output_path = extraction_root / family.lower() / (
                    "qrels.test.tsv" if role == "qrels" else f"{role}.jsonl"
                )
                _exclusive_write_bytes(output_path, raw)
                member_bindings[role] = {
                    "archive_member": target,
                    "byte_size": len(raw),
                    "extracted_relative_path": output_path.relative_to(
                        extraction_root.parent
                    ).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
    except BeirAcquisitionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise BeirAcquisitionError("archive member access failed") from exc

    documents = parse_corpus(raw_by_role["corpus"])
    queries = parse_queries(raw_by_role["queries"])
    qrels = parse_qrels(
        raw_by_role["qrels"],
        query_ids=tuple(queries),
        document_ids=tuple(documents),
    )
    self_document_query_count = sum(query_id in documents for query_id in queries)
    return ParsedFamily(
        documents=documents,
        queries=queries,
        positive_qrels=qrels,
        member_bindings=member_bindings,
        self_document_query_count=self_document_query_count,
    )


def allocate_blocks(
    *,
    family: str,
    parsed: ParsedFamily,
    secret: bytes,
) -> Mapping[str, tuple[str, ...]]:
    if family not in FAMILY_ORDER or len(secret) != 32:
        raise BeirAcquisitionError("allocation binding drifted")
    eligible = tuple(
        query_id
        for query_id in parsed.queries
        if parsed.positive_qrels.get(query_id)
        and any(
            document_id != query_id
            for document_id in parsed.positive_qrels[query_id]
        )
    )
    if len(eligible) < TOTAL_PER_FAMILY:
        raise BeirAcquisitionError("insufficient eligible query capacity")
    ordered = tuple(
        sorted(
            eligible,
            key=lambda query_id: (
                hmac.new(
                    secret,
                    f"{family}\x00{query_id}".encode("utf-8"),
                    hashlib.sha256,
                ).digest(),
                query_id,
            ),
        )
    )
    output: dict[str, tuple[str, ...]] = {}
    offset = 0
    for block, size in BLOCK_SIZES:
        output[block] = ordered[offset : offset + size]
        offset += size
    if offset != TOTAL_PER_FAMILY or len(set().union(*map(set, output.values()))) != offset:
        raise BeirAcquisitionError("block allocation drifted")
    return output


def _item_key(secret: bytes, family: str, query_id: str) -> str:
    return hmac.new(
        secret,
        f"ITEM\x00{family}\x00{query_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def materialize_blocks(
    *,
    parsed_by_family: Mapping[str, ParsedFamily],
    allocation_by_family: Mapping[str, Mapping[str, Sequence[str]]],
    secret: bytes,
    block_root: Path,
) -> Mapping[str, Mapping[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for block, expected_per_family in BLOCK_SIZES:
        view_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        for family in FAMILY_ORDER:
            parsed = parsed_by_family[family]
            query_ids = tuple(allocation_by_family[family][block])
            if len(query_ids) != expected_per_family:
                raise BeirAcquisitionError("block family size drifted")
            for query_id in query_ids:
                key = _item_key(secret, family, query_id)
                excluded = [query_id] if query_id in parsed.documents else []
                gold = tuple(
                    document_id
                    for document_id in parsed.positive_qrels[query_id]
                    if document_id != query_id
                )
                if not gold:
                    raise BeirAcquisitionError("selected query has no non-self qrel")
                view_rows.append(
                    {
                        "excluded_document_ids": excluded,
                        "family": family,
                        "item_key": key,
                        "query": parsed.queries[query_id],
                        "source_query_id": query_id,
                    }
                )
                label_rows.append(
                    {
                        "family": family,
                        "gold_document_ids": list(gold),
                        "item_key": key,
                    }
                )
        view_path = block_root / f"{block}.view.jsonl"
        label_path = block_root / f"{block}.labels.jsonl"
        _exclusive_write_jsonl(view_path, view_rows)
        _exclusive_write_jsonl(label_path, label_rows)
        receipts[block] = {
            "item_count": len(view_rows),
            "items_per_family": expected_per_family,
            "label_file_sha256": file_sha256(label_path),
            "label_file_size_bytes": label_path.stat().st_size,
            "view_file_sha256": file_sha256(view_path),
            "view_file_size_bytes": view_path.stat().st_size,
        }
    return receipts


def _load_manifest(project_root: Path, relative: Path) -> Mapping[str, Any]:
    path = project_root / relative
    binding = MANIFEST_BINDINGS[relative]
    if not path.is_file() or path.is_symlink() or file_sha256(path) != binding["file_sha256"]:
        raise BeirAcquisitionError("public manifest file binding drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BeirAcquisitionError("public manifest is invalid") from exc
    if not isinstance(value, Mapping):
        raise BeirAcquisitionError("public manifest is not an object")
    verify_self_hash(value, "self_sha256", binding["self_sha256"])
    return value


def _verify_freeze(project_root: Path) -> Mapping[str, Any]:
    path = project_root / FREEZE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise BeirAcquisitionError("implementation freeze is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BeirAcquisitionError("implementation freeze is invalid") from exc
    if not isinstance(value, Mapping) or value.get("schema") != FREEZE_SCHEMA:
        raise BeirAcquisitionError("implementation freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise BeirAcquisitionError("implementation freeze hash is absent")
    verify_self_hash(value, "self_sha256", declared)
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in value.get("implementation_bindings", ())
        if isinstance(row, Mapping)
    }
    expected = {
        ACQUISITION_RELATIVE.as_posix(): file_sha256(
            project_root / ACQUISITION_RELATIVE
        ),
        TEST_RELATIVE.as_posix(): file_sha256(project_root / TEST_RELATIVE),
    }
    if observed != expected:
        raise BeirAcquisitionError("implementation freeze bindings drifted")
    if value.get("design_self_sha256") != MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"]:
        raise BeirAcquisitionError("implementation freeze design binding drifted")
    return value


def _verify_archives(project_root: Path) -> Mapping[str, Path]:
    root = project_root / ARCHIVE_ROOT_RELATIVE
    output: dict[str, Path] = {}
    for family in FAMILY_ORDER:
        binding = ARCHIVE_BINDINGS[family]
        path = root / binding["filename"]
        if not path.is_file() or path.is_symlink():
            raise BeirAcquisitionError("archive is unavailable")
        if (
            path.stat().st_size != binding["size"]
            or file_sha256(path) != binding["sha256"]
            or file_digest(path, "md5") != binding["md5"]
        ):
            raise BeirAcquisitionError("archive binding drifted")
        output[family] = path
    return output


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
    result_path = project_root / RESULT_RELATIVE
    if result_path.exists():
        raise OneShotRefusal("acquisition result already exists")
    for relative in MANIFEST_BINDINGS:
        _load_manifest(project_root, relative)
    freeze = _verify_freeze(project_root)
    archives = _verify_archives(project_root)

    attempt_root = project_root / ATTEMPT_ROOT_RELATIVE
    try:
        attempt_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("formal acquisition root already exists") from exc
    marker = {
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "study_design_self_sha256": MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"],
    }
    marker_path = attempt_root / "attempt.marker"
    _exclusive_write_json(marker_path, marker)
    secret = os.urandom(32)
    secret_path = attempt_root / "selection.secret"
    _exclusive_write_bytes(secret_path, secret)

    extraction_root = attempt_root / "source_members"
    parsed_by_family: dict[str, ParsedFamily] = {}
    allocation_by_family: dict[str, Mapping[str, tuple[str, ...]]] = {}
    family_aggregates: dict[str, dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        parsed = open_family_archive(
            archives[family],
            family=family,
            extraction_root=extraction_root,
        )
        parsed_by_family[family] = parsed
        allocation = allocate_blocks(family=family, parsed=parsed, secret=secret)
        allocation_by_family[family] = allocation
        family_aggregates[family] = {
            "corpus_document_count": len(parsed.documents),
            "eligible_query_count": sum(
                bool(parsed.positive_qrels.get(query_id))
                and any(
                    document_id != query_id
                    for document_id in parsed.positive_qrels[query_id]
                )
                for query_id in parsed.queries
            ),
            "positive_qrel_pair_count": sum(
                len(value) for value in parsed.positive_qrels.values()
            ),
            "query_count": len(parsed.queries),
            "self_document_query_count": parsed.self_document_query_count,
            "source_member_bindings": parsed.member_bindings,
        }

    block_receipts = materialize_blocks(
        parsed_by_family=parsed_by_family,
        allocation_by_family=allocation_by_family,
        secret=secret,
        block_root=attempt_root / "blocks",
    )
    body = {
        "block_receipts": block_receipts,
        "claim_boundary": {
            "action_evaluator_model_or_score_count": 0,
            "archive_member_payload_open_count": 9,
            "gold_or_qrel_exposed_to_model": False,
            "item_assignment_count": len(FAMILY_ORDER) * TOTAL_PER_FAMILY,
            "network_call_count": 0,
            "online_evaluator_call_count": 0,
            "selection_secret_published": False,
        },
        "family_aggregates": family_aggregates,
        "family_order": list(FAMILY_ORDER),
        "formal_binding": {
            "attempt_marker_sha256": file_sha256(marker_path),
            "formal_implementation_commit": _git_head(project_root),
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
            "source_access_self_sha256": MANIFEST_BINDINGS[ACCESS_RELATIVE]["self_sha256"],
            "source_custody_self_sha256": MANIFEST_BINDINGS[CUSTODY_RELATIVE]["self_sha256"],
            "study_design_self_sha256": MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"],
        },
        "schema": SCHEMA,
        "status": "one_shot_BEIR_blocks_acquired_labels_separated_no_action",
    }
    receipt = self_hashed(body, "acquisition_sha256")
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
                "acquisition_sha256": receipt["acquisition_sha256"],
                "schema": SCHEMA,
                "status": receipt["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
