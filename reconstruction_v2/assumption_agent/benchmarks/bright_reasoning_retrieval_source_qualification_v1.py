"""One-shot aggregate-only BRIGHT source qualification.

Only document identifiers and the query/id/excluded_ids/gold_ids example
projection are decoded.  Human reasoning, answers, and document content are
deliberately outside this source-qualification program.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


SCHEMA = "bright_reasoning_retrieval_source_qualification_result_v1"
ATTEMPT_SCHEMA = "bright_reasoning_retrieval_source_qualification_attempt_v1"
FREEZE_SCHEMA = (
    "bright_reasoning_retrieval_source_qualification_implementation_freeze_v1"
)
FAMILY_ORDER = ("BIOLOGY", "ECONOMICS", "ROBOTICS")
DEMANDS = {family: 75 for family in FAMILY_ORDER}

CUSTODY_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_source_custody_v1.json"
)
ACCESS_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_source_access_v1.json"
)
DESIGN_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_source_qualification_design_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_source_qualification_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_source_qualification_result_v1.json"
)
QUALIFIER_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_reasoning_retrieval_source_qualification_v1.py"
)
TEST_RELATIVE = Path(
    "tests/test_bright_reasoning_retrieval_source_qualification_v1.py"
)
ATTEMPT_ROOT_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_source_qualification_v1"
)
SOURCE_ROOT_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_source_v1/dataset"
)

MANIFEST_BINDINGS = {
    CUSTODY_RELATIVE: {
        "file_sha256": "1bc98d38a016b0afc04f16017702cba92f027137a56bd9896e12b2390624444a",
        "self_field": "self_sha256",
        "self_sha256": "7a545a8168579a72fc4cbfbef7dbcb4ca37885b843ff46e8737c7c2ea05dc12d",
    },
    ACCESS_RELATIVE: {
        "file_sha256": "fd514be4a428904a22d4ab40016394461b4f0a5da3ad0cebf8b75e986c7d357c",
        "self_field": "self_sha256",
        "self_sha256": "2de28f8e9bb58fc53206311a54c0e5d29cc7649a08ba12c97f6df06b950d0051",
    },
    DESIGN_RELATIVE: {
        "file_sha256": "a0bf6e8a83ce397fd94d4a20f557bbd673cc6a62008a902da6c22ea1bff90185",
        "self_field": "self_sha256",
        "self_sha256": "3d6b12d6ac500bdaba551eb5f551dbe3ea5db302c6e1084cb36dea9f0777041d",
    },
}

SOURCE_BINDINGS = {
    "BIOLOGY": {
        "documents": {
            "relative": Path("documents/biology-00000-of-00001.parquet"),
            "sha256": "8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670",
            "size": 11_046_045,
            "rows": 57_359,
        },
        "examples": {
            "relative": Path("examples/biology-00000-of-00001.parquet"),
            "sha256": "6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829",
            "size": 200_655,
            "rows": 103,
        },
    },
    "ECONOMICS": {
        "documents": {
            "relative": Path("documents/economics-00000-of-00001.parquet"),
            "sha256": "f3ba8a0fbc9a9aed07b4970cc686e32cfefcd06d6922402587adf871f006394c",
            "size": 10_969_621,
            "rows": 50_220,
        },
        "examples": {
            "relative": Path("examples/economics-00000-of-00001.parquet"),
            "sha256": "2a79f0f3a881c7c03a258cf8ef8ac2db1ca9080963252d9a020bb45a264aa037",
            "size": 219_518,
            "rows": 103,
        },
    },
    "ROBOTICS": {
        "documents": {
            "relative": Path("documents/robotics-00000-of-00001.parquet"),
            "sha256": "2c83f286006a3b2e11a677abe88f382009c5ee79f97c1f43f6a571f3f94e6d15",
            "size": 7_874_186,
            "rows": 61_961,
        },
        "examples": {
            "relative": Path("examples/robotics-00000-of-00001.parquet"),
            "sha256": "621484c87c9ebae12f81e32a0a8c5d085af4b95cbe1b575ab40ae4b659adb53a",
            "size": 178_820,
            "rows": 101,
        },
    },
}

DOCUMENT_SCHEMA = ("id", "content")
EXAMPLE_SCHEMA = (
    "query",
    "reasoning",
    "id",
    "excluded_ids",
    "gold_ids_long",
    "gold_ids",
    "gold_answer",
)


class BrightQualificationError(RuntimeError):
    """Fail-closed qualification error without private row content."""


class OneShotRefusal(BrightQualificationError):
    """Raised when the formal attempt path is not pristine."""


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
        raise BrightQualificationError("non-canonical public value") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(body)
    output[field] = hashlib.sha256(canonical_json(output)).hexdigest()
    return output


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _required_text(value: Any, reason: str) -> str:
    if not isinstance(value, str) or "\x00" in value or not value.strip():
        raise ValueError(reason)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(reason) from exc
    return value


def _text_list(value: Any, reason: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(reason)
    output = tuple(_required_text(row, reason) for row in value)
    return output


@dataclass(frozen=True)
class Candidate:
    family: str
    item_id: str
    normalized_query: str
    gold_ids: tuple[str, ...]
    query_length: int
    present_excluded_count: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _component_sizes(candidates: Sequence[Candidate]) -> tuple[int, ...]:
    union = _UnionFind(len(candidates))
    registries: list[dict[Any, int]] = [{}, {}, {}]
    for index, candidate in enumerate(candidates):
        keys = (
            candidate.item_id,
            candidate.normalized_query,
            candidate.gold_ids,
        )
        for registry, key in zip(registries, keys):
            prior = registry.setdefault(key, index)
            union.union(index, prior)
    counts = Counter(union.find(index) for index in range(len(candidates)))
    return tuple(sorted(counts.values(), reverse=True))


def _candidate(
    family: str,
    row: Mapping[str, Any],
    document_ids: frozenset[str],
) -> Candidate:
    item_id = _required_text(row.get("id"), "example_id")
    query = _required_text(row.get("query"), "query")
    normalized_query = normalize(query)
    if not normalized_query:
        raise ValueError("query")
    gold_ids = _text_list(row.get("gold_ids"), "gold_ids")
    if not gold_ids or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("gold_ids")
    if any(identifier not in document_ids for identifier in gold_ids):
        raise ValueError("gold_id_absent_from_documents")
    excluded = _text_list(row.get("excluded_ids"), "excluded_ids")
    present_excluded = tuple(
        identifier for identifier in excluded if identifier in document_ids
    )
    if set(present_excluded).intersection(gold_ids):
        raise ValueError("excluded_gold_overlap")
    return Candidate(
        family=family,
        item_id=item_id,
        normalized_query=normalized_query,
        gold_ids=tuple(sorted(gold_ids)),
        query_length=len(query),
        present_excluded_count=len(present_excluded),
    )


def qualify_decoded_rows(
    *,
    document_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    example_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_binding: Mapping[str, Any],
    demands: Mapping[str, int] | None = None,
    expected_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Return a content-free capacity result from already projected rows."""

    demands = dict(DEMANDS if demands is None else demands)
    if set(demands) != set(FAMILY_ORDER):
        raise BrightQualificationError("family demand keys drifted")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in demands.values()):
        raise BrightQualificationError("family demand is invalid")
    if set(document_rows) != set(FAMILY_ORDER) or set(example_rows) != set(FAMILY_ORDER):
        raise BrightQualificationError("family row keys drifted")

    family_results: dict[str, Any] = {}
    all_satisfied = True
    for family in FAMILY_ORDER:
        raw_documents = document_rows[family]
        raw_examples = example_rows[family]
        if expected_counts is not None:
            expected = expected_counts[family]
            if len(raw_documents) != expected["documents"] or len(raw_examples) != expected["examples"]:
                all_satisfied = False

        document_invalid = Counter()
        document_ids: list[str] = []
        for row in raw_documents:
            try:
                if not isinstance(row, Mapping):
                    raise ValueError("document_row")
                document_ids.append(_required_text(row.get("id"), "document_id"))
            except ValueError as exc:
                document_invalid[str(exc)] += 1
        duplicate_document_id_count = len(document_ids) - len(set(document_ids))
        if document_invalid or duplicate_document_id_count:
            all_satisfied = False
        document_id_set = frozenset(document_ids)

        invalid = Counter()
        candidates: list[Candidate] = []
        for row in raw_examples:
            try:
                if not isinstance(row, Mapping):
                    raise ValueError("example_row")
                candidates.append(_candidate(family, row, document_id_set))
            except ValueError as exc:
                invalid[str(exc)] += 1

        sizes = _component_sizes(candidates) if candidates else ()
        component_count = len(sizes)
        satisfied = component_count >= demands[family]
        all_satisfied = all_satisfied and satisfied
        family_results[family] = {
            "component_capacity": component_count,
            "component_capacity_demand": demands[family],
            "component_capacity_satisfied": satisfied,
            "document_count": len(raw_documents),
            "document_duplicate_id_count": duplicate_document_id_count,
            "document_invalid_reason_counts": dict(sorted(document_invalid.items())),
            "eligible_example_count": len(candidates),
            "example_count": len(raw_examples),
            "example_invalid_reason_counts": dict(sorted(invalid.items())),
            "gold_count_max": max((len(row.gold_ids) for row in candidates), default=0),
            "gold_count_min": min((len(row.gold_ids) for row in candidates), default=0),
            "largest_component_size": max(sizes, default=0),
            "multirow_component_count": sum(size > 1 for size in sizes),
            "present_excluded_id_total": sum(row.present_excluded_count for row in candidates),
            "query_character_count_max": max((row.query_length for row in candidates), default=0),
            "query_character_count_min": min((row.query_length for row in candidates), default=0),
        }

    status = (
        "qualified_source_capacity_no_selection"
        if all_satisfied
        else "terminal_source_infeasible_no_selection"
    )
    body = {
        "claim_boundary": {
            "action_evaluator_model_or_score_count": 0,
            "document_content_column_read": False,
            "gold_answer_column_read": False,
            "item_assignment_or_selection_count": 0,
            "network_call_count": 0,
            "reasoning_column_read": False,
            "selection_secret_created": False,
        },
        "family_aggregates": family_results,
        "family_order": list(FAMILY_ORDER),
        "schema": SCHEMA,
        "source_binding": dict(source_binding),
        "status": status,
    }
    return self_hashed(body, "qualification_sha256")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightQualificationError("bound JSON cannot be read") from exc
    if not isinstance(value, dict):
        raise BrightQualificationError("bound JSON root is invalid")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, expected: str) -> None:
    body = dict(value)
    declared = body.pop(field, None)
    actual = hashlib.sha256(canonical_json(body)).hexdigest()
    if declared != expected or actual != expected:
        raise BrightQualificationError("bound manifest self hash drifted")


def _verify_public_bindings(project_root: Path) -> None:
    for relative, binding in MANIFEST_BINDINGS.items():
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise BrightQualificationError("bound manifest is unavailable")
        if file_sha256(path) != binding["file_sha256"]:
            raise BrightQualificationError("bound manifest file hash drifted")
        value = _load_json(path)
        _verify_self_hash(value, binding["self_field"], binding["self_sha256"])


def _verify_freeze(project_root: Path) -> dict[str, Any]:
    freeze_path = project_root / FREEZE_RELATIVE
    value = _load_json(freeze_path)
    if value.get("schema") != FREEZE_SCHEMA:
        raise BrightQualificationError("implementation freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise BrightQualificationError("implementation freeze hash is absent")
    _verify_self_hash(value, "self_sha256", declared)
    bindings = value.get("implementation_bindings")
    if not isinstance(bindings, list):
        raise BrightQualificationError("implementation bindings are invalid")
    expected = {
        QUALIFIER_RELATIVE.as_posix(): file_sha256(project_root / QUALIFIER_RELATIVE),
        TEST_RELATIVE.as_posix(): file_sha256(project_root / TEST_RELATIVE),
    }
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in bindings
        if isinstance(row, Mapping)
    }
    if observed != expected:
        raise BrightQualificationError("implementation freeze bindings drifted")
    if value.get("design_self_sha256") != MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"]:
        raise BrightQualificationError("implementation freeze design binding drifted")
    return value


def _verify_source_files(source_root: Path) -> None:
    for family in FAMILY_ORDER:
        for role in ("documents", "examples"):
            binding = SOURCE_BINDINGS[family][role]
            path = source_root / binding["relative"]
            if not path.is_file() or path.is_symlink():
                raise BrightQualificationError("source file is unavailable")
            if path.stat().st_size != binding["size"] or file_sha256(path) != binding["sha256"]:
                raise BrightQualificationError("source file binding drifted")


def _project_parquet(
    path: Path,
    *,
    expected_schema: Sequence[str],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet

        reader = parquet.ParquetFile(path)
        if tuple(reader.schema_arrow.names) != tuple(expected_schema):
            raise BrightQualificationError("parquet schema drifted")
        table = reader.read(columns=list(columns), use_threads=False)
        return table.to_pylist()
    except BrightQualificationError:
        raise
    except Exception as exc:
        raise BrightQualificationError("parquet projection failed") from exc


def _exclusive_write(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> None:
    raw = canonical_json(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _create_attempt(project_root: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / ATTEMPT_ROOT_RELATIVE
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("formal qualification root already exists") from exc
    marker = {
        "design_self_sha256": MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"],
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
    }
    marker_path = root / "attempt.marker"
    _exclusive_write(marker_path, marker)
    return {
        "attempt_marker_mode": "0600",
        "attempt_marker_sha256": file_sha256(marker_path),
        "attempt_marker_size_bytes": marker_path.stat().st_size,
    }


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
        raise OneShotRefusal("qualification result already exists")
    _verify_public_bindings(project_root)
    freeze = _verify_freeze(project_root)
    source_root = project_root / SOURCE_ROOT_RELATIVE
    _verify_source_files(source_root)
    attempt = _create_attempt(project_root, freeze)

    document_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
    example_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
    expected: dict[str, dict[str, int]] = {}
    for family in FAMILY_ORDER:
        document_binding = SOURCE_BINDINGS[family]["documents"]
        example_binding = SOURCE_BINDINGS[family]["examples"]
        document_rows[family] = _project_parquet(
            source_root / document_binding["relative"],
            expected_schema=DOCUMENT_SCHEMA,
            columns=("id",),
        )
        example_rows[family] = _project_parquet(
            source_root / example_binding["relative"],
            expected_schema=EXAMPLE_SCHEMA,
            columns=("query", "id", "excluded_ids", "gold_ids"),
        )
        expected[family] = {
            "documents": document_binding["rows"],
            "examples": example_binding["rows"],
        }
    receipt = qualify_decoded_rows(
        document_rows=document_rows,
        example_rows=example_rows,
        source_binding={
            "dataset_commit": "3066d29c9651a576c8aba4832d249807b181ecae",
            "document_file_read_count": 3,
            "example_file_read_count": 3,
            "formal_implementation_commit": _git_head(project_root),
            "formal_attempt": attempt,
            "source_access_self_sha256": MANIFEST_BINDINGS[ACCESS_RELATIVE]["self_sha256"],
            "source_custody_self_sha256": MANIFEST_BINDINGS[CUSTODY_RELATIVE]["self_sha256"],
            "source_qualification_design_self_sha256": MANIFEST_BINDINGS[DESIGN_RELATIVE]["self_sha256"],
        },
        expected_counts=expected,
    )
    _exclusive_write(result_path, receipt, mode=0o644)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    receipt = run_formal(arguments.project_root)
    print(canonical_json({
        "qualification_sha256": receipt["qualification_sha256"],
        "schema": SCHEMA,
        "status": receipt["status"],
    }).decode("ascii"))
    return 0 if receipt["status"] == "qualified_source_capacity_no_selection" else 2


if __name__ == "__main__":
    raise SystemExit(main())
