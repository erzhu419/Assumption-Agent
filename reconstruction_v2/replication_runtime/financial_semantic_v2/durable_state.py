from __future__ import annotations

"""Crash-durable, content-addressed stage transition receipts."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from assumption_agent.models import stable_hash


DURABLE_STAGE_RECEIPT_VERSION = "financial_semantic_durable_stage_receipt_v2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STAGE_FILE = re.compile(r"^\d{3}_[a-z][a-z0-9_]{0,63}\.stage\.json$")


class DurableStateError(RuntimeError):
    """A durable state receipt or transition failed closed."""


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DurableStateError(f"{label} must be a lowercase sha256")
    return value


def _normalized_stage_order(stage_order: Sequence[str]) -> tuple[str, ...]:
    stages = tuple(stage_order)
    if (
        not stages
        or len(stages) > 999
        or len(set(stages)) != len(stages)
        or any(
            not isinstance(stage, str)
            or _STAGE_NAME.fullmatch(stage) is None
            for stage in stages
        )
    ):
        raise DurableStateError("stage order is empty, duplicated, or unsafe")
    return stages


def _stage_order_hash(stages: Sequence[str]) -> str:
    return stable_hash({"stages": list(stages)})


def _stage_path(root: Path, index: int, stage: str) -> Path:
    return root / f"{index:03d}_{stage}.stage.json"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_hashed_json_v2(
    path: str | Path,
    body: Mapping[str, Any],
    *,
    hash_field: str = "stage_hash",
    refuse_existing: bool = True,
) -> dict[str, Any]:
    """Atomically persist a self-hashed JSON object and fsync file + parent.

    With ``refuse_existing=True`` a fully written temporary inode is linked to
    the destination using an atomic no-clobber operation.  Concurrent writers
    cannot silently replace one another.
    """

    target = Path(path)
    if not hash_field or hash_field in body:
        raise DurableStateError("hash field must be absent from the body")
    if target.exists() or target.is_symlink():
        if refuse_existing:
            raise FileExistsError(target)
        if target.is_symlink() or not target.is_file():
            raise DurableStateError("destination is not a regular file")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise DurableStateError("destination parent is not a regular directory")

    normalized_body = dict(body)
    receipt = {
        **normalized_body,
        hash_field: stable_hash(normalized_body),
    }
    encoded = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_existing:
            os.link(temporary_path, target)
            temporary_path.unlink()
            temporary_path = None
        else:
            os.replace(temporary_path, target)
            temporary_path = None
        _fsync_directory(target.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return receipt


def read_hashed_json_v2(
    path: str | Path,
    *,
    hash_field: str = "stage_hash",
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DurableStateError("hashed JSON source is not a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DurableStateError("hashed JSON source is unreadable") from exc
    if not isinstance(value, dict):
        raise DurableStateError("hashed JSON source must contain one object")
    body = dict(value)
    declared = body.pop(hash_field, None)
    if not isinstance(declared, str) or declared != stable_hash(body):
        raise DurableStateError("hashed JSON self-hash mismatch")
    return value


@dataclass(frozen=True)
class DurableStageReceiptV2:
    receipt_version: str
    stage: str
    stage_index: int
    stage_order_hash: str
    work_unit_hash: str
    request_hash: str
    predecessor_stage_hash: str | None
    payload: Mapping[str, Any]
    stage_hash: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DurableStageReceiptV2":
        required = {
            "receipt_version",
            "stage",
            "stage_index",
            "stage_order_hash",
            "work_unit_hash",
            "request_hash",
            "predecessor_stage_hash",
            "payload",
            "stage_hash",
        }
        if set(value) != required:
            raise DurableStateError("stage receipt fields drifted")
        hash_body = dict(value)
        declared_hash = hash_body.pop("stage_hash", None)
        if (
            not isinstance(declared_hash, str)
            or declared_hash != stable_hash(hash_body)
        ):
            raise DurableStateError("stage receipt self-hash mismatch")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise DurableStateError("stage receipt payload must be an object")
        stage_index = value.get("stage_index")
        stage = value.get("stage")
        if not isinstance(stage, str) or _STAGE_NAME.fullmatch(stage) is None:
            raise DurableStateError("stage receipt name is malformed")
        if (
            isinstance(stage_index, bool)
            or not isinstance(stage_index, int)
            or stage_index < 0
        ):
            raise DurableStateError("stage receipt index is malformed")
        predecessor = value.get("predecessor_stage_hash")
        if predecessor is not None:
            _require_hash(predecessor, "predecessor stage hash")
        receipt = cls(
            receipt_version=str(value.get("receipt_version") or ""),
            stage=stage,
            stage_index=stage_index,
            stage_order_hash=_require_hash(
                value.get("stage_order_hash"), "stage order hash"
            ),
            work_unit_hash=_require_hash(
                value.get("work_unit_hash"), "work unit hash"
            ),
            request_hash=_require_hash(
                value.get("request_hash"), "request hash"
            ),
            predecessor_stage_hash=predecessor,
            payload=MappingProxyType(dict(payload)),
            stage_hash=_require_hash(value.get("stage_hash"), "stage hash"),
        )
        if receipt.receipt_version != DURABLE_STAGE_RECEIPT_VERSION:
            raise DurableStateError("stage receipt version drifted")
        return receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_order_hash": self.stage_order_hash,
            "work_unit_hash": self.work_unit_hash,
            "request_hash": self.request_hash,
            "predecessor_stage_hash": self.predecessor_stage_hash,
            "payload": dict(self.payload),
            "stage_hash": self.stage_hash,
        }


def load_durable_stage_chain_v2(
    root: str | Path,
    *,
    stage_order: Sequence[str],
    work_unit_hash: str,
    request_hash: str,
) -> tuple[DurableStageReceiptV2, ...]:
    """Load and validate a contiguous, predecessor-linked stage prefix."""

    stages = _normalized_stage_order(stage_order)
    work_hash = _require_hash(work_unit_hash, "work unit hash")
    request = _require_hash(request_hash, "request hash")
    order_hash = _stage_order_hash(stages)
    directory = Path(root)
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise DurableStateError("durable state root is not a regular directory")

    expected_paths = {
        _stage_path(directory, index, stage).name
        for index, stage in enumerate(stages)
    }
    observed_paths = {
        path.name
        for path in directory.iterdir()
        if _STAGE_FILE.fullmatch(path.name)
    }
    if not observed_paths.issubset(expected_paths):
        raise DurableStateError("durable state contains an unknown stage file")

    chain: list[DurableStageReceiptV2] = []
    gap_seen = False
    predecessor: str | None = None
    for index, stage in enumerate(stages):
        path = _stage_path(directory, index, stage)
        if not path.exists() and not path.is_symlink():
            gap_seen = True
            continue
        if gap_seen:
            raise DurableStateError("durable stage chain contains a gap")
        value = read_hashed_json_v2(path)
        receipt = DurableStageReceiptV2.from_dict(value)
        if (
            receipt.stage != stage
            or receipt.stage_index != index
            or receipt.stage_order_hash != order_hash
            or receipt.work_unit_hash != work_hash
            or receipt.request_hash != request
            or receipt.predecessor_stage_hash != predecessor
        ):
            raise DurableStateError("durable stage predecessor or identity drifted")
        chain.append(receipt)
        predecessor = receipt.stage_hash
    return tuple(chain)


def transition_durable_stage_v2(
    root: str | Path,
    *,
    stage_order: Sequence[str],
    work_unit_hash: str,
    request_hash: str,
    stage: str,
    predecessor_stage_hash: str | None,
    payload: Mapping[str, Any],
) -> DurableStageReceiptV2:
    """Persist exactly the next stage in a declared order.

    The caller must name the current predecessor hash.  Missing, stale,
    skipped, duplicated, or concurrently written transitions fail closed.
    """

    stages = _normalized_stage_order(stage_order)
    if stage not in stages:
        raise DurableStateError("requested stage is not in the frozen order")
    work_hash = _require_hash(work_unit_hash, "work unit hash")
    request = _require_hash(request_hash, "request hash")
    if predecessor_stage_hash is not None:
        _require_hash(predecessor_stage_hash, "predecessor stage hash")
    if not isinstance(payload, Mapping):
        raise DurableStateError("stage payload must be an object")

    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    chain = load_durable_stage_chain_v2(
        directory,
        stage_order=stages,
        work_unit_hash=work_hash,
        request_hash=request,
    )
    target_index = stages.index(stage)
    if len(chain) != target_index:
        raise DurableStateError("stage transition skipped or repeated a stage")
    actual_predecessor = chain[-1].stage_hash if chain else None
    if predecessor_stage_hash != actual_predecessor:
        raise DurableStateError("declared predecessor stage hash is stale")

    body: dict[str, Any] = {
        "receipt_version": DURABLE_STAGE_RECEIPT_VERSION,
        "stage": stage,
        "stage_index": target_index,
        "stage_order_hash": _stage_order_hash(stages),
        "work_unit_hash": work_hash,
        "request_hash": request,
        "predecessor_stage_hash": actual_predecessor,
        "payload": dict(payload),
    }
    path = _stage_path(directory, target_index, stage)
    value = atomic_write_hashed_json_v2(
        path,
        body,
        hash_field="stage_hash",
        refuse_existing=True,
    )
    return DurableStageReceiptV2.from_dict(value)
