"""One-shot HMAC assignment and gold-separated BRIGHT block acquisition.

The formal path projects only ``query``, ``id``, ``excluded_ids`` and
``gold_ids`` from the three frozen example parquet files.  It does not open
document content, human reasoning, long-context labels, or gold answers.  A
fresh secret randomizes singleton components into preregistered blocks; the
query view and gold labels are then written to distinct private files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


VERSION = "bright_reasoning_retrieval_acquisition_v1"
RESULT_SCHEMA = f"{VERSION}_result"
VIEW_SCHEMA = f"{VERSION}_block_view"
LABEL_SCHEMA = f"{VERSION}_block_labels"
ATTEMPT_SCHEMA = f"{VERSION}_attempt"
FREEZE_SCHEMA = f"{VERSION}_implementation_freeze"

FAMILY_ORDER = ("BIOLOGY", "ECONOMICS", "ROBOTICS")
BLOCK_ORDER = ("G_form", "A_form", "F_search", "A_hold", "M_search")
BLOCK_COUNTS = {
    "G_form": 10,
    "A_form": 20,
    "F_search": 15,
    "A_hold": 15,
    "M_search": 15,
}
DATASET_COMMIT = "3066d29c9651a576c8aba4832d249807b181ecae"
SELECTION_DOMAIN = b"bright_acquisition_v1\x00"

DESIGN_RELATIVE = Path("manifests/bright_reasoning_retrieval_acquisition_design_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_acquisition_implementation_freeze_v1.json"
)
QUALIFICATION_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_source_qualification_result_v1.json"
)
RUNTIME_CUSTODY_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_offline_runtime_custody_v1.json"
)
RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_acquisition_result_v1.json")
SOURCE_ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_source_v1/dataset")
ATTEMPT_ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_acquisition_v1")
PRIVATE_ROOT_NAME = "private"

SOURCE_FILES = {
    "BIOLOGY": {
        "path": Path("examples/biology-00000-of-00001.parquet"),
        "sha256": "6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829",
        "size": 200_655,
        "rows": 103,
    },
    "ECONOMICS": {
        "path": Path("examples/economics-00000-of-00001.parquet"),
        "sha256": "2a79f0f3a881c7c03a258cf8ef8ac2db1ca9080963252d9a020bb45a264aa037",
        "size": 219_518,
        "rows": 103,
    },
    "ROBOTICS": {
        "path": Path("examples/robotics-00000-of-00001.parquet"),
        "sha256": "621484c87c9ebae12f81e32a0a8c5d085af4b95cbe1b575ab40ae4b659adb53a",
        "size": 178_820,
        "rows": 101,
    },
}
EXAMPLE_SCHEMA = (
    "query",
    "reasoning",
    "id",
    "excluded_ids",
    "gold_ids_long",
    "gold_ids",
    "gold_answer",
)
PROJECTED_COLUMNS = ("query", "id", "excluded_ids", "gold_ids")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BrightAcquisitionError(RuntimeError):
    """The formal acquisition contract failed closed."""


class BrightAcquisitionOneShotRefusal(BrightAcquisitionError):
    """A consumed formal root or published result cannot be replayed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BrightAcquisitionError("value is not canonical JSON") from exc


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise BrightAcquisitionError("self-hash field already exists")
    return {**body, field: stable_hash(body)}


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = payload.get(field)
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise BrightAcquisitionError(f"{field} is absent")
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise BrightAcquisitionError(f"{field} drifted")
    return declared


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BrightAcquisitionError(f"{field} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise BrightAcquisitionError(f"{field} is invalid") from exc
    return value


def _text_list(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BrightAcquisitionError(f"{field} is invalid")
    return tuple(_required_text(item, field) for item in value)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


@dataclass(frozen=True)
class SourceItem:
    family: str
    source_id: str
    query: str
    excluded_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.family not in FAMILY_ORDER:
            raise BrightAcquisitionError("source family is invalid")
        _required_text(self.source_id, "source id")
        if not normalize(_required_text(self.query, "query")):
            raise BrightAcquisitionError("query is empty after normalization")
        if not self.gold_ids or len(set(self.gold_ids)) != len(self.gold_ids):
            raise BrightAcquisitionError("gold ids are empty or duplicated")
        if set(self.excluded_ids).intersection(self.gold_ids):
            raise BrightAcquisitionError("excluded ids overlap gold ids")

    @property
    def commitment_sha256(self) -> str:
        return stable_hash(
            {
                "family": self.family,
                "query": self.query,
                "source_id": self.source_id,
            }
        )


def decode_source_rows(
    family: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[SourceItem, ...]:
    if family not in FAMILY_ORDER:
        raise BrightAcquisitionError("family is invalid")
    items: list[SourceItem] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(PROJECTED_COLUMNS):
            raise BrightAcquisitionError("projected source row shape drifted")
        items.append(
            SourceItem(
                family=family,
                source_id=_required_text(raw.get("id"), "source id"),
                query=_required_text(raw.get("query"), "query"),
                excluded_ids=_text_list(raw.get("excluded_ids"), "excluded ids"),
                gold_ids=_text_list(raw.get("gold_ids"), "gold ids"),
            )
        )
    if len({item.source_id for item in items}) != len(items):
        raise BrightAcquisitionError("source ids are duplicated")
    if len({normalize(item.query) for item in items}) != len(items):
        raise BrightAcquisitionError("normalized queries are duplicated")
    if len({tuple(sorted(item.gold_ids)) for item in items}) != len(items):
        raise BrightAcquisitionError("gold id sets are duplicated")
    if len({item.commitment_sha256 for item in items}) != len(items):
        raise BrightAcquisitionError("item commitments are duplicated")
    return tuple(items)


def selection_priority(secret: bytes, item: SourceItem) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise BrightAcquisitionError("selection secret must contain exactly 32 bytes")
    message = b"\x00".join(
        (
            SELECTION_DOMAIN.rstrip(b"\x00"),
            DATASET_COMMIT.encode("ascii"),
            item.family.encode("ascii"),
            item.source_id.encode("utf-8"),
        )
    )
    return hmac.new(secret, message, hashlib.sha256).digest()


def assign_blocks(
    items_by_family: Mapping[str, Sequence[SourceItem]], secret: bytes
) -> dict[str, tuple[SourceItem, ...]]:
    if set(items_by_family) != set(FAMILY_ORDER):
        raise BrightAcquisitionError("family registry drifted")
    allocations: dict[str, list[SourceItem]] = {
        block: [] for block in (*BLOCK_ORDER, "RESERVE")
    }
    required = sum(BLOCK_COUNTS.values())
    for family in FAMILY_ORDER:
        rows = tuple(items_by_family[family])
        if len(rows) < required or any(item.family != family for item in rows):
            raise BrightAcquisitionError("family capacity or identity drifted")
        ranked = sorted(
            rows,
            key=lambda item: (selection_priority(secret, item), item.source_id),
        )
        cursor = 0
        for block in BLOCK_ORDER:
            count = BLOCK_COUNTS[block]
            allocations[block].extend(ranked[cursor : cursor + count])
            cursor += count
        allocations["RESERVE"].extend(ranked[cursor:])
    output = {block: tuple(rows) for block, rows in allocations.items()}
    all_commitments = [
        item.commitment_sha256 for rows in output.values() for item in rows
    ]
    expected_total = sum(len(rows) for rows in items_by_family.values())
    if len(all_commitments) != expected_total or len(set(all_commitments)) != expected_total:
        raise BrightAcquisitionError("assignment is incomplete or overlapping")
    for block in BLOCK_ORDER:
        counts = Counter(item.family for item in output[block])
        if counts != Counter({family: BLOCK_COUNTS[block] for family in FAMILY_ORDER}):
            raise BrightAcquisitionError("block family balance drifted")
    return output


def block_view(block: str, rows: Sequence[SourceItem]) -> dict[str, Any]:
    if block not in (*BLOCK_ORDER, "RESERVE"):
        raise BrightAcquisitionError("view block is invalid")
    body = {
        "block": block,
        "excluded_fields": [
            "source_example_id",
            "reasoning",
            "gold_ids_long",
            "gold_ids",
            "gold_answer",
        ],
        "item_count": len(rows),
        "items": [
            {
                "excluded_ids": list(item.excluded_ids),
                "family": item.family,
                "item_commitment_sha256": item.commitment_sha256,
                "ordinal": ordinal,
                "query": item.query,
            }
            for ordinal, item in enumerate(rows)
        ],
        "schema": VIEW_SCHEMA,
    }
    return self_hashed(body, "pack_sha256")


def block_labels(block: str, rows: Sequence[SourceItem]) -> dict[str, Any]:
    if block not in BLOCK_ORDER:
        raise BrightAcquisitionError("label block is invalid")
    body = {
        "block": block,
        "item_count": len(rows),
        "items": [
            {
                "gold_ids": list(item.gold_ids),
                "item_commitment_sha256": item.commitment_sha256,
                "ordinal": ordinal,
            }
            for ordinal, item in enumerate(rows)
        ],
        "schema": LABEL_SCHEMA,
    }
    return self_hashed(body, "pack_sha256")


def _exclusive_write(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    _exclusive_write(path, canonical_json_bytes(payload) + b"\n", mode=mode)


def _read_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightAcquisitionError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightAcquisitionError(f"{field} is unreadable") from exc
    if not isinstance(value, dict):
        raise BrightAcquisitionError(f"{field} root is invalid")
    return value


def _verify_bound_manifest(
    path: Path, *, field: str, file_sha256_expected: str, self_field: str, self_expected: str
) -> dict[str, Any]:
    if file_sha256(path) != file_sha256_expected:
        raise BrightAcquisitionError(f"{field} file binding drifted")
    value = _read_json(path, field)
    if verify_self_hash(value, self_field) != self_expected:
        raise BrightAcquisitionError(f"{field} self binding drifted")
    return value


def _verify_preconditions(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    design = _read_json(project_root / DESIGN_RELATIVE, "acquisition design")
    design_sha = verify_self_hash(design, "self_sha256")
    freeze = _read_json(project_root / FREEZE_RELATIVE, "implementation freeze")
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise BrightAcquisitionError("implementation freeze schema drifted")
    verify_self_hash(freeze, "self_sha256")
    if freeze.get("design_self_sha256") != design_sha:
        raise BrightAcquisitionError("implementation freeze design binding drifted")
    qualification = _verify_bound_manifest(
        project_root / QUALIFICATION_RELATIVE,
        field="source qualification",
        file_sha256_expected="a648068f398b28aabc0bd30bbfdd50aee6f266b3290e220e00394706dd9b73d9",
        self_field="qualification_sha256",
        self_expected="fdc36b8cff050196f6da7f7ab18b99bd365757d988399358a9af424242278234",
    )
    runtime = _verify_bound_manifest(
        project_root / RUNTIME_CUSTODY_RELATIVE,
        field="runtime custody",
        file_sha256_expected="0f3a7857de24ea35e9588464bc2ab15cb71e2894f1d3bfa5b7bb4ed71ddf4b2f",
        self_field="self_sha256",
        self_expected="80f4d2f3900e2f9bfb406ee2a697aaa266b371e650e5e79a51df6b9742ff4671",
    )
    if qualification.get("status") != "qualified_source_capacity_no_selection":
        raise BrightAcquisitionError("source qualification did not authorize acquisition")
    return design, runtime


def _verify_freeze_implementation(project_root: Path, freeze: Mapping[str, Any]) -> None:
    bindings = freeze.get("implementation_bindings")
    if not isinstance(bindings, list):
        raise BrightAcquisitionError("implementation binding set is invalid")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in bindings
        if isinstance(row, Mapping)
    }
    expected_paths = (
        "assumption_agent/benchmarks/bright_reasoning_retrieval_acquisition_v1.py",
        "tests/test_bright_reasoning_retrieval_acquisition_v1.py",
    )
    expected = {
        relative: file_sha256(project_root / relative) for relative in expected_paths
    }
    if observed != expected:
        raise BrightAcquisitionError("implementation files drifted from freeze")


def _project_examples(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet

        reader = parquet.ParquetFile(path)
        if tuple(reader.schema_arrow.names) != EXAMPLE_SCHEMA:
            raise BrightAcquisitionError("example parquet schema drifted")
        return reader.read(columns=list(PROJECTED_COLUMNS), use_threads=False).to_pylist()
    except BrightAcquisitionError:
        raise
    except Exception as exc:
        raise BrightAcquisitionError("example parquet projection failed") from exc


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_formal_root(project_root: Path, design: Mapping[str, Any], freeze: Mapping[str, Any]) -> Path:
    root = project_root / ATTEMPT_ROOT_RELATIVE
    if (project_root / RESULT_RELATIVE).exists():
        raise BrightAcquisitionOneShotRefusal("public acquisition result already exists")
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise BrightAcquisitionOneShotRefusal("formal acquisition root already exists") from exc
    marker = self_hashed(
        {
            "design_self_sha256": design["self_sha256"],
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "schema": ATTEMPT_SCHEMA,
        },
        "attempt_sha256",
    )
    _write_json(root / "attempt.marker", marker)
    (root / PRIVATE_ROOT_NAME).mkdir(mode=0o700)
    return root


def _source_items(project_root: Path) -> dict[str, tuple[SourceItem, ...]]:
    source_root = project_root / SOURCE_ROOT_RELATIVE
    output: dict[str, tuple[SourceItem, ...]] = {}
    for family in FAMILY_ORDER:
        binding = SOURCE_FILES[family]
        path = source_root / binding["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size"]
            or file_sha256(path) != binding["sha256"]
        ):
            raise BrightAcquisitionError("source example file binding drifted")
        rows = _project_examples(path)
        if len(rows) != binding["rows"]:
            raise BrightAcquisitionError("source example row count drifted")
        output[family] = decode_source_rows(family, rows)
    return output


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    design, _runtime = _verify_preconditions(project_root)
    freeze = _read_json(project_root / FREEZE_RELATIVE, "implementation freeze")
    _verify_freeze_implementation(project_root, freeze)
    formal_root = _create_formal_root(project_root, design, freeze)
    private_root = formal_root / PRIVATE_ROOT_NAME
    secret = os.urandom(32)
    _exclusive_write(private_root / "selection.secret.bin", secret)
    items_by_family = _source_items(project_root)
    assignments = assign_blocks(items_by_family, secret)

    block_rows: dict[str, Any] = {}
    for block in (*BLOCK_ORDER, "RESERVE"):
        rows = assignments[block]
        view = block_view(block, rows)
        view_path = private_root / f"{block}.view.json"
        _write_json(view_path, view)
        row: dict[str, Any] = {
            "family_counts": dict(sorted(Counter(item.family for item in rows).items())),
            "item_count": len(rows),
            "view_pack_file_sha256": file_sha256(view_path),
            "view_pack_sha256": view["pack_sha256"],
        }
        if block in BLOCK_ORDER:
            labels = block_labels(block, rows)
            label_path = private_root / f"{block}.labels.json"
            _write_json(label_path, labels)
            row.update(
                {
                    "label_pack_file_sha256": file_sha256(label_path),
                    "label_pack_sha256": labels["pack_sha256"],
                }
            )
        block_rows[block] = row

    attempt_path = formal_root / "attempt.marker"
    result_body = {
        "block_aggregates": block_rows,
        "claim_boundary": {
            "document_content_column_read": False,
            "gold_answer_column_read": False,
            "model_or_evaluator_call_count": 0,
            "network_call_count": 0,
            "reasoning_column_read": False,
            "retrieval_or_score_count": 0,
        },
        "formal_binding": {
            "attempt_marker_file_sha256": file_sha256(attempt_path),
            "dataset_commit": DATASET_COMMIT,
            "formal_implementation_commit": _git_head(project_root),
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
            "source_qualification_sha256": "fdc36b8cff050196f6da7f7ab18b99bd365757d988399358a9af424242278234",
        },
        "schema": RESULT_SCHEMA,
        "status": "acquired_gold_separated_blocks_G_only_authorized",
    }
    result = self_hashed(result_body, "result_sha256")
    _write_json(project_root / RESULT_RELATIVE, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_formal(arguments.project_root)
    print(
        canonical_json_bytes(
            {
                "result_sha256": result["result_sha256"],
                "schema": result["schema"],
                "status": result["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
