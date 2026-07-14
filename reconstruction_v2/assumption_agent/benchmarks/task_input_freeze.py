from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .task_input_closure import (
    TASK_INPUT_CLOSURE_POLICY_VERSION,
    TASK_INPUT_PREPARATION_VERSION,
    default_task_input_cache_root,
    load_task_input_closure,
)


TASK_INPUT_CLOSURE_PROTOCOL_VERSION = "3.19.0"
TASK_INPUT_CLOSURE_PROTOCOL_VERSIONS = frozenset({"3.19.0", "3.20.0"})

_SHA256_FIELDS = {
    "preparation_receipt_file_sha256",
    "preparation_receipt_hash",
    "closure_ledger_hash",
    "closure_set_hash",
    "object_set_hash",
    "benchmark_source_environment_set_hash",
}
_COUNT_FIELDS = {
    "closure_count",
    "closure_ledger_item_count",
    "content_object_count",
}
_SOURCE_FIELDS = {
    "preparation_receipt",
    *_SHA256_FIELDS,
    *_COUNT_FIELDS,
}
_LEDGER_ROW_FIELDS = {
    "family_hash",
    "item_id_hash",
    "source_environment_hash",
    "closure_hash",
    "object_count",
    "object_hashes",
    "object_set_hash",
}


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_input_closure_source_contract_issues(source: object) -> list[str]:
    if not isinstance(source, Mapping):
        return ["task_input_closure_source_missing"]
    if set(source) != _SOURCE_FIELDS:
        return ["task_input_closure_source_fields_mismatch"]
    issues: list[str] = []
    receipt_path = str(source.get("preparation_receipt") or "")
    candidate = Path(receipt_path)
    if (
        not receipt_path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.suffix != ".json"
    ):
        issues.append("task_input_closure_preparation_receipt_path_invalid")
    for field in sorted(_SHA256_FIELDS):
        if not _is_sha256(source.get(field)):
            issues.append(f"task_input_closure_source_hash_invalid:{field}")
    for field in sorted(_COUNT_FIELDS):
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            issues.append(f"task_input_closure_source_count_invalid:{field}")
    if (
        isinstance(source.get("closure_count"), int)
        and source.get("closure_ledger_item_count") != source.get("closure_count")
    ):
        issues.append("task_input_closure_source_ledger_count_mismatch")
    return issues


def task_input_closure_policy_for_protocol_payload(
    protocol_payload: Mapping[str, Any],
) -> str | None:
    execution = protocol_payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("paper protocol execution contract is missing")
    declared = execution.get("task_input_closure_policy")
    enabled = (
        str(protocol_payload.get("protocol_version") or "")
        in TASK_INPUT_CLOSURE_PROTOCOL_VERSIONS
    )
    if enabled:
        if declared != TASK_INPUT_CLOSURE_POLICY_VERSION:
            raise ValueError("task input closure policy is missing or drifted")
        if task_input_closure_source_contract_issues(
            execution.get("task_input_closure_source")
        ):
            raise ValueError("task input closure source is missing or malformed")
        return TASK_INPUT_CLOSURE_POLICY_VERSION
    if declared is not None or "task_input_closure_source" in execution:
        raise ValueError("task input closure policy is not permitted by this protocol")
    return None


@dataclass(frozen=True)
class FrozenTaskInputClosure:
    source: Mapping[str, Any]
    receipt: Mapping[str, Any]
    receipt_path: Path
    ledger_by_item_hash: Mapping[str, Mapping[str, Any]]

    @property
    def policy(self) -> str:
        return TASK_INPUT_CLOSURE_POLICY_VERSION

    @property
    def freeze_hash(self) -> str:
        return stable_hash(dict(self.source))


def _validate_ledger(receipt: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = receipt.get("closure_ledger")
    if not isinstance(rows, list) or not rows:
        raise ValueError("task input preparation closure ledger is missing")
    normalized: list[Mapping[str, Any]] = []
    item_hashes: set[str] = set()
    closure_hashes: list[str] = []
    all_object_hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _LEDGER_ROW_FIELDS:
            raise ValueError("task input preparation closure ledger row is malformed")
        if any(
            not _is_sha256(row.get(field))
            for field in (
                "family_hash",
                "item_id_hash",
                "source_environment_hash",
                "closure_hash",
                "object_set_hash",
            )
        ):
            raise ValueError("task input preparation closure ledger hash is invalid")
        item_hash = str(row["item_id_hash"])
        if item_hash in item_hashes:
            raise ValueError("task input preparation closure ledger item is duplicated")
        item_hashes.add(item_hash)
        object_hashes = row.get("object_hashes")
        if (
            not isinstance(object_hashes, list)
            or not object_hashes
            or object_hashes != sorted(set(str(value) for value in object_hashes))
            or any(not _is_sha256(value) for value in object_hashes)
            or row.get("object_count") != len(object_hashes)
            or row.get("object_set_hash") != stable_hash(object_hashes)
        ):
            raise ValueError("task input preparation closure object ledger is invalid")
        closure_hashes.append(str(row["closure_hash"]))
        all_object_hashes.update(str(value) for value in object_hashes)
        normalized.append(dict(row))
    if normalized != sorted(
        normalized,
        key=lambda row: (
            str(row["family_hash"]),
            str(row["item_id_hash"]),
        ),
    ):
        raise ValueError("task input preparation closure ledger order is not canonical")
    if receipt.get("closure_ledger_item_count") != len(normalized):
        raise ValueError("task input preparation closure ledger count mismatch")
    if receipt.get("closure_count") != len(normalized):
        raise ValueError("task input preparation closure count mismatch")
    if receipt.get("closure_ledger_hash") != stable_hash(normalized):
        raise ValueError("task input preparation closure ledger hash mismatch")
    if receipt.get("closure_set_hash") != stable_hash(sorted(closure_hashes)):
        raise ValueError("task input preparation closure set hash mismatch")
    if receipt.get("content_object_count") != len(all_object_hashes):
        raise ValueError("task input preparation object count mismatch")
    if receipt.get("object_set_hash") != stable_hash(sorted(all_object_hashes)):
        raise ValueError("task input preparation object set hash mismatch")
    source_hashes = sorted(str(row["source_environment_hash"]) for row in normalized)
    if receipt.get("benchmark_source_environment_set_hash") != stable_hash(source_hashes):
        raise ValueError("task input preparation source environment set hash mismatch")
    return tuple(normalized)


def load_frozen_task_input_closure(
    protocol_payload: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> FrozenTaskInputClosure | None:
    policy = task_input_closure_policy_for_protocol_payload(protocol_payload)
    if policy is None:
        return None
    execution = protocol_payload["execution"]
    assert isinstance(execution, Mapping)
    source = execution["task_input_closure_source"]
    assert isinstance(source, Mapping)
    project = Path(project_root).expanduser().resolve()
    receipt_path = (project / str(source["preparation_receipt"])).resolve()
    if project not in receipt_path.parents or not receipt_path.is_file():
        raise PermissionError("task input preparation receipt escaped or is missing")
    if _file_sha256(receipt_path) != source["preparation_receipt_file_sha256"]:
        raise PermissionError("task input preparation receipt file hash mismatch")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError("task input preparation receipt is malformed") from exc
    if not isinstance(receipt, Mapping):
        raise PermissionError("task input preparation receipt must contain one object")
    calculated_receipt_hash = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    if (
        receipt.get("receipt_hash") != calculated_receipt_hash
        or receipt.get("receipt_hash") != source["preparation_receipt_hash"]
        or receipt.get("preparation_version") != TASK_INPUT_PREPARATION_VERSION
        or receipt.get("policy") != policy
        or receipt.get("passed") is not True
        or receipt.get("trial_runtime_download_required") is not False
        or receipt.get("test_or_solution_content_accessed") is not False
        or receipt.get("raw_content_persisted") is not False
    ):
        raise PermissionError("task input preparation receipt contract mismatch")
    try:
        rows = _validate_ledger(receipt)
    except ValueError as exc:
        raise PermissionError(str(exc)) from exc
    for field in (
        "closure_count",
        "closure_ledger_item_count",
        "closure_ledger_hash",
        "closure_set_hash",
        "content_object_count",
        "object_set_hash",
        "benchmark_source_environment_set_hash",
    ):
        if receipt.get(field) != source.get(field):
            raise PermissionError(f"task input preparation freeze mismatch: {field}")
    return FrozenTaskInputClosure(
        source=dict(source),
        receipt=dict(receipt),
        receipt_path=receipt_path,
        ledger_by_item_hash={str(row["item_id_hash"]): row for row in rows},
    )


def verify_current_task_input_closure(
    frozen: FrozenTaskInputClosure,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    root = Path(cache_root or default_task_input_cache_root()).expanduser().resolve()
    closures_root = root / "closures"
    if not closures_root.is_dir():
        raise PermissionError("frozen task input closure cache is missing")
    observed: dict[str, Mapping[str, Any]] = {}
    for path in sorted(closures_root.glob("*/*.json")):
        family = path.parent.name
        item_id = path.stem
        item_hash = stable_hash({"item_id": item_id})
        expected = frozen.ledger_by_item_hash.get(item_hash)
        if expected is None:
            continue
        manifest = load_task_input_closure(root, family, item_id)
        object_hashes = sorted(
            {str(row["sha256"]) for row in manifest.get("entries", ())}
        )
        row = {
            "family_hash": stable_hash({"family": family}),
            "item_id_hash": item_hash,
            "source_environment_hash": manifest.get("source_environment_hash"),
            "closure_hash": manifest.get("closure_hash"),
            "object_count": len(object_hashes),
            "object_hashes": object_hashes,
            "object_set_hash": stable_hash(object_hashes),
        }
        if row != expected:
            raise PermissionError("current task input closure differs from frozen ledger")
        if item_hash in observed:
            raise PermissionError("current task input closure identity is duplicated")
        observed[item_hash] = row
    if set(observed) != set(frozen.ledger_by_item_hash):
        raise PermissionError("current task input closure ledger is incomplete")
    return root


def expected_prewarm_closure_rows(
    receipt: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    rows = receipt.get("items")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("development prewarm item rows are missing")
    expected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("task_input_closure_required"):
            continue
        item_hash = str(row.get("item_id_hash") or "")
        if not _is_sha256(item_hash) or item_hash in expected:
            raise ValueError("development prewarm closure row identity is invalid")
        expected[item_hash] = dict(row)
    return expected
