"""One-shot aggregate-only qualification for the pinned BIRCO P1 JSON source.

The formal entry point consumes durable markers before it opens the source and
never emits a query, document, qrel row, or source identifier.  The qualifier
only establishes exact source identity, the documented mapping schema, and
aggregate capacity for the already-frozen query-local reranking study.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


VERSION = "birco_p1_source_qualification_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = Path(__file__).resolve()
TEST_PATH = PROJECT_ROOT / "tests/test_birco_p1_source_qualification_v1.py"
SOURCE_PATH = PROJECT_ROOT / "artifacts/birco_p1_official_source_v1/BIRCO_dataset.json"
CUSTODY_PATH = PROJECT_ROOT / "manifests/birco_p1_source_custody_v1.json"
DESIGN_PATH = PROJECT_ROOT / "manifests/birco_p1_typed_constraint_e4_study_design_v1.json"
DOWNLOAD_AUTHORIZATION_PATH = PROJECT_ROOT / "manifests/birco_p1_source_download_authorization_v1.json"
DOWNLOAD_RECEIPT_PATH = PROJECT_ROOT / "manifests/birco_p1_source_download_receipt_v1.json"
FREEZE_PATH = PROJECT_ROOT / "manifests/birco_p1_source_qualification_freeze_v1.json"
MARKER_PATH = PROJECT_ROOT / "artifacts/birco_p1_source_qualification_v1/qualification.one_shot_marker.json"
SOURCE_OPEN_MARKER_PATH = PROJECT_ROOT / "artifacts/birco_p1_source_qualification_v1/source_open.one_shot_marker.json"
FAILURE_PATH = PROJECT_ROOT / "artifacts/birco_p1_source_qualification_v1/qualification.terminal_failure.json"
RESULT_PATH = PROJECT_ROOT / "manifests/birco_p1_source_qualification_result_v1.json"

EXPECTED_CUSTODY_SELF_SHA256 = "190cddaf78d807d791713301cdaa95fe6239c7c541a385429f2cb7973599af12"
EXPECTED_DESIGN_SELF_SHA256 = "47f88edd3c322ad602f8d3ed4bbe64dc9a94acb6fe20a78791f93ce8e6d747c4"
EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256 = "TO_BE_PATCHED_AFTER_AUTHORIZATION"
EXPECTED_DOWNLOAD_RECEIPT_SELF_SHA256 = "TO_BE_PATCHED_AFTER_OPAQUE_DOWNLOAD"


class BircoP1SourceQualificationError(RuntimeError):
    """The fixed source or aggregate-only qualification contract failed."""


@dataclass(frozen=True)
class FamilyContract:
    query_count: int
    corpus_count: int
    allowed_scores: tuple[float, ...] | None
    minimum_score: float = 0.0
    maximum_score: float = 2.0


@dataclass(frozen=True)
class QualificationContract:
    source_size_bytes: int
    source_md5: str
    source_sha256: str
    families: Mapping[str, FamilyContract]
    minimum_pool_size: int = 10
    selected_query_capacity: int = 40
    maximum_id_characters: int = 1024
    maximum_query_characters: int = 250_000
    maximum_document_characters: int = 2_000_000


FORMAL_CONTRACT = QualificationContract(
    source_size_bytes=20_134_244,
    source_md5="548cad5d25ce8c0714274ba0ec17fa78",
    source_sha256="TO_BE_PATCHED_AFTER_OPAQUE_DOWNLOAD",
    families={
        "doris-mae": FamilyContract(60, 5_543, None),
        "clinical-trial": FamilyContract(50, 3_256, (0.0, 1.0, 2.0)),
        "wtb": FamilyContract(100, 1_767, (0.0, 1.0)),
    },
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BircoP1SourceQualificationError("receipt is not canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise BircoP1SourceQualificationError("receipt parent is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise BircoP1SourceQualificationError("receipt parent is unavailable")
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BircoP1SourceQualificationError("receipt parent is unsafe")
        break
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    _ensure_durable_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _load_verified_manifest(path: Path, expected_self_sha256: str) -> Mapping[str, Any]:
    if len(expected_self_sha256) != 64:
        raise BircoP1SourceQualificationError("manifest binding is not frozen")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoP1SourceQualificationError("bound manifest is unavailable") from exc
    if not isinstance(value, Mapping) or value.get("self_sha256") != expected_self_sha256:
        raise BircoP1SourceQualificationError("bound manifest self hash drifted")
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != expected_self_sha256:
        raise BircoP1SourceQualificationError("bound manifest semantic hash drifted")
    return value


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BircoP1SourceQualificationError("bound implementation file is unavailable") from exc


def _load_and_verify_freeze() -> tuple[Mapping[str, Any], str]:
    """Verify the final freeze without creating a qualifier/freeze hash cycle."""

    try:
        raw = FREEZE_PATH.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoP1SourceQualificationError("qualification freeze is unavailable") from exc
    if not isinstance(value, Mapping):
        raise BircoP1SourceQualificationError("qualification freeze shape drifted")
    claimed = value.get("self_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise BircoP1SourceQualificationError("qualification freeze self hash is invalid")
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != claimed:
        raise BircoP1SourceQualificationError("qualification freeze semantic hash drifted")
    required = {
        "schema": "birco_p1_source_qualification_freeze_v1",
        "status": "frozen_before_unique_formal_qualification",
        "source_custody_self_sha256": EXPECTED_CUSTODY_SELF_SHA256,
        "study_design_self_sha256": EXPECTED_DESIGN_SELF_SHA256,
        "download_authorization_self_sha256": EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256,
        "download_receipt_self_sha256": EXPECTED_DOWNLOAD_RECEIPT_SELF_SHA256,
        "source_sha256": FORMAL_CONTRACT.source_sha256,
        "qualifier_sha256": _file_sha256(QUALIFIER_PATH),
        "test_sha256": _file_sha256(TEST_PATH),
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise BircoP1SourceQualificationError("qualification freeze binding drifted")
    return value, claimed


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BircoP1SourceQualificationError("source JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise BircoP1SourceQualificationError("source JSON contains a non-finite number")


def _parse_json(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except BircoP1SourceQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoP1SourceQualificationError("source JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise BircoP1SourceQualificationError("source top level is not a mapping")
    return value


def _read_stable_regular_file(path: Path, contract: QualificationContract) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise BircoP1SourceQualificationError("fixed source is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BircoP1SourceQualificationError("fixed source is not a private regular file")
        if before.st_size != contract.source_size_bytes:
            raise BircoP1SourceQualificationError("fixed source size drifted")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise BircoP1SourceQualificationError("fixed source ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BircoP1SourceQualificationError("fixed source grew during read")
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if not stable:
            raise BircoP1SourceQualificationError("fixed source changed during read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if hashlib.md5(raw).hexdigest() != contract.source_md5:  # nosec B303: publisher identity
        raise BircoP1SourceQualificationError("fixed source MD5 identity drifted")
    if len(contract.source_sha256) != 64 or hashlib.sha256(raw).hexdigest() != contract.source_sha256:
        raise BircoP1SourceQualificationError("fixed source SHA256 identity drifted")
    return raw


def _safe_id(value: object, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BircoP1SourceQualificationError("source identity schema drifted")
    return value


def _safe_text(value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BircoP1SourceQualificationError("source text schema drifted")


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]

    def aggregate(self) -> dict[str, int]:
        sizes: dict[int, int] = {}
        for value in range(len(self.parent)):
            root = self.find(value)
            sizes[root] = sizes.get(root, 0) + 1
        values = tuple(sizes.values())
        return {
            "component_count": len(values),
            "largest_component_query_count": max(values, default=0),
            "singleton_component_count": sum(size == 1 for size in values),
        }


def _family_aggregate(
    family_value: object,
    family_contract: FamilyContract,
    contract: QualificationContract,
) -> dict[str, Any]:
    if not isinstance(family_value, Mapping):
        raise BircoP1SourceQualificationError("family schema drifted")
    if not {"query", "corpus", "qrel"}.issubset(family_value):
        raise BircoP1SourceQualificationError("family required fields are missing")
    queries = family_value.get("query")
    corpus = family_value.get("corpus")
    qrels = family_value.get("qrel")
    if not isinstance(queries, Mapping) or not isinstance(corpus, Mapping) or not isinstance(qrels, Mapping):
        raise BircoP1SourceQualificationError("family mappings drifted")
    if len(queries) != family_contract.query_count or len(corpus) != family_contract.corpus_count:
        raise BircoP1SourceQualificationError("family documented counts drifted")
    query_ids: list[str] = []
    for raw_id, text in queries.items():
        query_ids.append(_safe_id(raw_id, contract.maximum_id_characters))
        _safe_text(text, contract.maximum_query_characters)
    corpus_ids: set[str] = set()
    for raw_id, _text_value in corpus.items():
        corpus_ids.add(_safe_id(raw_id, contract.maximum_id_characters))
    if set(qrels) != set(query_ids):
        raise BircoP1SourceQualificationError("qrel and query identity sets drifted")

    pool_sizes: list[int] = []
    score_count = zero_count = threshold_positive_count = fractional_count = 0
    score_min = math.inf
    score_max = -math.inf
    owners: dict[str, int] = {}
    components = _DisjointSet(len(query_ids))
    distinct_candidate_ids: set[str] = set()
    for query_ordinal, query_id in enumerate(query_ids):
        row = qrels.get(query_id)
        if not isinstance(row, Mapping) or len(row) < contract.minimum_pool_size:
            raise BircoP1SourceQualificationError("candidate pool schema or minimum capacity drifted")
        pool_sizes.append(len(row))
        row_has_positive_gain = False
        for raw_candidate_id, raw_score in row.items():
            candidate_id = _safe_id(raw_candidate_id, contract.maximum_id_characters)
            if candidate_id not in corpus_ids:
                raise BircoP1SourceQualificationError("candidate is absent from the family corpus")
            _safe_text(corpus[candidate_id], contract.maximum_document_characters)
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise BircoP1SourceQualificationError("qrel score is not numeric")
            score = float(raw_score)
            if not math.isfinite(score) or not family_contract.minimum_score <= score <= family_contract.maximum_score:
                raise BircoP1SourceQualificationError("qrel score is outside the frozen domain")
            if family_contract.allowed_scores is not None and score not in family_contract.allowed_scores:
                raise BircoP1SourceQualificationError("qrel discrete score domain drifted")
            score_count += 1
            zero_count += score == 0.0
            threshold_positive_count += score >= 1.0
            fractional_count += not score.is_integer()
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            row_has_positive_gain = row_has_positive_gain or score > 0.0
            distinct_candidate_ids.add(candidate_id)
            previous = owners.setdefault(candidate_id, query_ordinal)
            components.union(query_ordinal, previous)
        if not row_has_positive_gain:
            raise BircoP1SourceQualificationError("candidate pool has no positive gain")
    if len(query_ids) < contract.selected_query_capacity or not threshold_positive_count:
        raise BircoP1SourceQualificationError("family formal capacity drifted")
    return {
        "candidate_membership": {
            "distinct_candidate_count": len(distinct_candidate_ids),
            "maximum_pool_size": max(pool_sizes),
            "minimum_pool_size": min(pool_sizes),
            "pool_entry_count": sum(pool_sizes),
        },
        "candidate_overlap_components": components.aggregate(),
        "corpus_count": len(corpus),
        "qrel_score_domain": {
            "fractional_score_count": fractional_count,
            "maximum_score": score_max,
            "minimum_score": score_min,
            "score_count": score_count,
            "threshold_ge_1_count": threshold_positive_count,
            "zero_score_count": zero_count,
        },
        "query_count": len(queries),
        "query_disjoint_selected_capacity": contract.selected_query_capacity,
    }


def _analyze_source(path: Path, contract: QualificationContract) -> dict[str, Any]:
    raw = _read_stable_regular_file(path, contract)
    source = _parse_json(raw)
    aggregates: dict[str, Any] = {}
    for family, family_contract in contract.families.items():
        if family not in source:
            raise BircoP1SourceQualificationError("selected family is absent")
        aggregates[family] = _family_aggregate(source[family], family_contract, contract)
    return {
        "family_aggregates": aggregates,
        "formal_family_count": len(aggregates),
        "qualified": True,
        "source_identity": {
            "md5": contract.source_md5,
            "sha256": contract.source_sha256,
            "size_bytes": contract.source_size_bytes,
        },
        "top_level_field_count": len(source),
    }


def _analyze_fixed_source() -> dict[str, Any]:
    return _analyze_source(SOURCE_PATH, FORMAL_CONTRACT)


def _consume_marker() -> str:
    body = {
        "model_action_or_score_count": 0,
        "qrel_value_output_count": 0,
        "retry_replay_resample_or_contract_revision": 0,
        "schema": f"{VERSION}_one_shot_marker_v1",
        "source_expected_MD5": FORMAL_CONTRACT.source_md5,
        "source_expected_size_bytes": FORMAL_CONTRACT.source_size_bytes,
        "source_item_query_document_qrel_or_ID_output_count": 0,
        "status": "started_before_manifest_validation_or_source_open",
    }
    value = {**body, "self_sha256": _semantic_hash(body)}
    return _write_exclusive(MARKER_PATH, value)


def _consume_source_open_marker() -> str:
    body = {
        "model_action_or_score_count": 0,
        "qrel_value_output_count": 0,
        "schema": f"{VERSION}_source_open_marker_v1",
        "source_item_query_document_qrel_or_ID_output_count": 0,
        "status": "consumed_immediately_before_fixed_source_open",
    }
    value = {**body, "self_sha256": _semantic_hash(body)}
    return _write_exclusive(SOURCE_OPEN_MARKER_PATH, value)


def run_formal_qualification() -> Mapping[str, Any]:
    marker_file_sha256 = _consume_marker()
    stage = "validate_frozen_bindings"
    try:
        _load_verified_manifest(CUSTODY_PATH, EXPECTED_CUSTODY_SELF_SHA256)
        _load_verified_manifest(DESIGN_PATH, EXPECTED_DESIGN_SELF_SHA256)
        _load_verified_manifest(
            DOWNLOAD_AUTHORIZATION_PATH, EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256
        )
        _load_verified_manifest(DOWNLOAD_RECEIPT_PATH, EXPECTED_DOWNLOAD_RECEIPT_SELF_SHA256)
        if len(FORMAL_CONTRACT.source_sha256) != 64:
            raise BircoP1SourceQualificationError("source SHA256 contract is not frozen")
        _freeze, freeze_self_sha256 = _load_and_verify_freeze()
        stage = "open_and_aggregate_fixed_source"
        source_open_marker_file_sha256 = _consume_source_open_marker()
        aggregate = _analyze_fixed_source()
        stage = "write_qualification_result"
        body: dict[str, Any] = {
            **aggregate,
            "binding_self_sha256": {
                "download_authorization": EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256,
                "download_receipt": EXPECTED_DOWNLOAD_RECEIPT_SELF_SHA256,
                "freeze": freeze_self_sha256,
                "source_custody": EXPECTED_CUSTODY_SELF_SHA256,
                "study_design": EXPECTED_DESIGN_SELF_SHA256,
            },
            "marker_file_sha256": marker_file_sha256,
            "model_action_or_score_count": 0,
            "online_evaluator_call_count": 0,
            "qrel_value_output_count": 0,
            "schema": f"{VERSION}_result_v1",
            "source_open_marker_file_sha256": source_open_marker_file_sha256,
            "source_item_query_document_qrel_or_ID_output_count": 0,
            "status": "qualified_aggregate_only",
        }
        value = {**body, "self_sha256": _semantic_hash(body)}
        _write_exclusive(RESULT_PATH, value)
        return value
    except Exception as exc:
        failure_body = {
            "error_message": str(exc),
            "error_type": type(exc).__name__,
            "failure_stage": stage,
            "marker_file_sha256": marker_file_sha256,
            "model_action_or_score_count": 0,
            "online_evaluator_call_count": 0,
            "qrel_value_output_count": 0,
            "qualified": False,
            "retry_replay_resample_or_contract_revision": 0,
            "schema": f"{VERSION}_terminal_failure_v1",
            "source_item_query_document_qrel_or_ID_output_count": 0,
            "status": "terminal_failure_no_retry",
        }
        failure = {**failure_body, "self_sha256": _semantic_hash(failure_body)}
        _write_exclusive(FAILURE_PATH, failure)
        raise


def main() -> int:
    value = run_formal_qualification()
    print(_canonical_bytes(value).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
