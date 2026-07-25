"""Bounded two-process coordinate workers for MMQA P1.

This module turns a sequence of opaque ``work_id`` plus validated anonymous
items into either a MiniLM or cross-encoder private coordinate archive.  One
worker process initializes exactly one local model once, then consumes the
entire block in fixed-size chunks.  MiniLM and cross-encoder roles are
independent and can therefore run concurrently on physical GPU0 and GPU1.

The one-shot output is canonical JSON, exclusively created with mode 0600,
and contains only work IDs, source-local ordinals, float64-hex coordinates,
and audit identities/counts.  It contains no question or unit text, source
identifier, gold, family, support, answer, or model path.  A controller must
validate both archives against the same anonymous block before merging them;
missing, duplicate, additional, or reordered rows fail closed.

There is no adaptive batch sizing, retry, replay, resampling, network/API
fallback, or per-item process fan-out.  Any initialization, OOM, inference,
shape, or persistence failure terminates the worker.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
from typing import Callable, Iterable, Mapping, Sequence

from . import mmqa_p1_action_integration_v1 as integration
from . import mmqa_p1_local_action_executor_v1 as executor
from . import mmqa_p1_typed_proof_e5_core_v1 as core


VERSION = "mmqa_p1_block_coordinate_worker_v1"
STUDY_ID = core.STUDY_ID
STUDY_DESIGN_SELF_SHA256 = executor.STUDY_DESIGN_SELF_SHA256

ROLE_MINILM = "MINILM"
ROLE_CROSS_ENCODER = "CROSS_ENCODER"
ROLES = (ROLE_MINILM, ROLE_CROSS_ENCODER)
ROLE_DEVICE = {
    ROLE_MINILM: "cuda:0",
    ROLE_CROSS_ENCODER: "cuda:1",
}
ROLE_BATCH_SIZE = {
    ROLE_MINILM: executor.MINILM_BATCH_SIZE,
    ROLE_CROSS_ENCODER: executor.CROSS_ENCODER_BATCH_SIZE,
}
ROLE_MAX_LENGTH = {
    ROLE_MINILM: executor.MINILM_MAX_LENGTH,
    ROLE_CROSS_ENCODER: executor.CROSS_ENCODER_MAX_LENGTH,
}
ROLE_MODEL_ID = {
    ROLE_MINILM: executor.MINILM_MODEL_ID,
    ROLE_CROSS_ENCODER: executor.CROSS_ENCODER_MODEL_ID,
}
ROLE_REQUIRED_TREE_SHA256 = {
    ROLE_MINILM: executor.MINILM_REQUIRED_TREE_SHA256,
    ROLE_CROSS_ENCODER: executor.CROSS_ENCODER_REQUIRED_TREE_SHA256,
}

ANONYMOUS_BLOCK_SCHEMA = f"{VERSION}_anonymous_block_v1"
COORDINATE_ARCHIVE_SCHEMA = f"{VERSION}_private_coordinate_archive_v1"
BLOCK_MERGE_RECEIPT_SCHEMA = f"{VERSION}_block_merge_receipt_v1"

_WORK_ID = re.compile(r"mmqa-work-v1-[0-9a-f]{64}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_FIELDS = frozenset(
    {
        "schema",
        "study_id",
        "study_design_self_sha256",
        "item_count",
        "items",
        "block_sha256",
    }
)
_BLOCK_ITEM_FIELDS = frozenset({"work_id", "anonymous_work_item"})
_ARCHIVE_FIELDS = frozenset(
    {
        "schema",
        "study_id",
        "study_design_self_sha256",
        "role",
        "device",
        "model_id",
        "required_tree_sha256",
        "model_path_sha256",
        "local_runtime_identity_sha256",
        "anonymous_block_sha256",
        "item_count",
        "unit_count",
        "inference_input_count",
        "frozen_batch_size",
        "frozen_max_length",
        "model_initialization_count",
        "batch_call_count",
        "dynamic_batch_resize_count",
        "retry_replay_resample_count",
        "network_or_api_call_count",
        "rows",
        "archive_sha256",
    }
)
_ROW_FIELDS = frozenset(
    {"work_id", "ordinal", "coordinate_float64_hex"}
)


class MmqaP1BlockCoordinateWorkerError(RuntimeError):
    """A block, local worker, private archive, or merge contract drifted."""


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            "block worker value is not canonical JSON"
        ) from exc
    return (text + ("\n" if newline else "")).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise MmqaP1BlockCoordinateWorkerError(f"{label} schema drifted")


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MmqaP1BlockCoordinateWorkerError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _work_id(value: object) -> str:
    if not isinstance(value, str) or _WORK_ID.fullmatch(value) is None:
        raise MmqaP1BlockCoordinateWorkerError(
            "work_id must be an opaque MMQA work identity"
        )
    return value


def _absolute_lexical_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "\x00" in value
        or "//" in value
        or "/./" in value
        or "/../" in value
        or value.endswith("/.")
        or value.endswith("/..")
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "model path must be one normalized absolute lexical path"
        )
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MmqaP1BlockCoordinateWorkerError(
            f"{field} must be an exact integer at least {minimum}"
        )
    return value


def _coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate must be one finite scalar"
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate escaped the frozen [0, 1] range"
        )
    return 0.0 if result == 0.0 else result


def _float_from_canonical_hex(value: object) -> float:
    if not isinstance(value, str):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate encoding must be float64 hex"
        )
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate encoding must be float64 hex"
        ) from exc
    result = _coordinate(result)
    if value != result.hex():
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate float64 hex is noncanonical"
        )
    return result


@dataclass(frozen=True)
class AnonymousBlockItem:
    work_id: str
    work_item: integration.AnonymousWorkItem

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_id", _work_id(self.work_id))
        if not isinstance(self.work_item, integration.AnonymousWorkItem):
            raise MmqaP1BlockCoordinateWorkerError(
                "block item requires a validated AnonymousWorkItem"
            )

    def payload(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "anonymous_work_item": self.work_item.anonymous_payload(),
        }


def validate_block_items(
    items: Sequence[AnonymousBlockItem],
) -> tuple[AnonymousBlockItem, ...]:
    if (
        isinstance(items, (str, bytes))
        or not isinstance(items, Sequence)
        or not items
        or not all(isinstance(item, AnonymousBlockItem) for item in items)
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "anonymous block must contain validated block items"
        )
    checked = tuple(items)
    identifiers = tuple(item.work_id for item in checked)
    if len(set(identifiers)) != len(identifiers):
        raise MmqaP1BlockCoordinateWorkerError(
            "anonymous block contains duplicate work_id"
        )
    return checked


def anonymous_block_payload(
    items: Sequence[AnonymousBlockItem],
) -> dict[str, object]:
    checked = validate_block_items(items)
    body: dict[str, object] = {
        "schema": ANONYMOUS_BLOCK_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "item_count": len(checked),
        "items": [item.payload() for item in checked],
    }
    return {**body, "block_sha256": _semantic_hash(body)}


def validate_anonymous_block_payload(
    value: Mapping[str, object],
) -> tuple[AnonymousBlockItem, ...]:
    if not isinstance(value, Mapping):
        raise MmqaP1BlockCoordinateWorkerError(
            "anonymous block must be a mapping"
        )
    _exact_fields(value, _BLOCK_FIELDS, "anonymous block")
    body = {key: value[key] for key in value if key != "block_sha256"}
    if (
        value.get("schema") != ANONYMOUS_BLOCK_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("block_sha256") != _semantic_hash(body)
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "anonymous block identity drifted"
        )
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        raise MmqaP1BlockCoordinateWorkerError(
            "anonymous block items must be an array"
        )
    parsed = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise MmqaP1BlockCoordinateWorkerError(
                "anonymous block item must be a mapping"
            )
        _exact_fields(raw, _BLOCK_ITEM_FIELDS, "anonymous block item")
        anonymous = raw.get("anonymous_work_item")
        if not isinstance(anonymous, Mapping):
            raise MmqaP1BlockCoordinateWorkerError(
                "anonymous work item must be a mapping"
            )
        try:
            item = integration.validate_anonymous_work_item(anonymous)
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1BlockCoordinateWorkerError(
                "anonymous work item drifted"
            ) from exc
        parsed.append(
            AnonymousBlockItem(
                work_id=_work_id(raw.get("work_id")),
                work_item=item,
            )
        )
    checked = validate_block_items(tuple(parsed))
    if value.get("item_count") != len(checked):
        raise MmqaP1BlockCoordinateWorkerError(
            "anonymous block item count drifted"
        )
    return checked


def _read_private_canonical_mapping(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().absolute()
    descriptor = -1
    try:
        before = source.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "private input must be a regular mode-0600 file"
            )
        descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "private input identity changed while opening"
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            "private input cannot be read"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            "private input is not canonical JSON"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_json_bytes(value, newline=True)
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "private input is not canonical JSON"
        )
    return dict(value)


def load_anonymous_block(path: str | Path) -> tuple[AnonymousBlockItem, ...]:
    return validate_anonymous_block_payload(
        _read_private_canonical_mapping(path)
    )


def write_private_anonymous_block(
    path: str | Path,
    items: Sequence[AnonymousBlockItem],
) -> str:
    """Exclusively persist one canonical mode-0600 input for both CLI roles."""

    payload = anonymous_block_payload(items)
    destination = Path(path).expanduser().absolute()
    raw = _canonical_json_bytes(payload, newline=True)
    descriptor = -1
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            "private anonymous block already exists or cannot be created"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = destination.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or destination.read_bytes() != raw
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "private anonymous block reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BlockModelBinding:
    role: str
    model_path: str
    required_tree_sha256: str
    local_runtime_identity_sha256: str
    asset_identity_verified: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = False
    network_disabled: bool = True
    retry_count: int = 0

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise MmqaP1BlockCoordinateWorkerError(
                "block model role drifted"
            )
        object.__setattr__(
            self, "model_path", _absolute_lexical_path(self.model_path)
        )
        if (
            self.required_tree_sha256
            != ROLE_REQUIRED_TREE_SHA256[self.role]
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "block model required tree identity drifted"
            )
        _sha256(
            self.local_runtime_identity_sha256,
            "local runtime identity",
        )
        if (
            self.asset_identity_verified is not True
            or self.local_files_only is not True
            or self.trust_remote_code is not False
            or self.network_disabled is not True
            or type(self.retry_count) is not int
            or self.retry_count != 0
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "block model offline policy drifted"
            )

    @property
    def device(self) -> str:
        return ROLE_DEVICE[self.role]

    @property
    def batch_size(self) -> int:
        return ROLE_BATCH_SIZE[self.role]

    @property
    def max_length(self) -> int:
        return ROLE_MAX_LENGTH[self.role]

    @property
    def model_id(self) -> str:
        return ROLE_MODEL_ID[self.role]

    @property
    def model_path_sha256(self) -> str:
        return hashlib.sha256(self.model_path.encode("utf-8")).hexdigest()

    @property
    def binding_sha256(self) -> str:
        return _semantic_hash(
            {
                "role": self.role,
                "model_id": self.model_id,
                "model_path_sha256": self.model_path_sha256,
                "required_tree_sha256": self.required_tree_sha256,
                "local_runtime_identity_sha256": (
                    self.local_runtime_identity_sha256
                ),
                "asset_identity_verified": True,
                "local_files_only": True,
                "trust_remote_code": False,
                "network_disabled": True,
                "retry_count": 0,
            }
        )


@dataclass(frozen=True, order=True)
class CoordinateRow:
    work_id: str
    ordinal: int
    coordinate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_id", _work_id(self.work_id))
        object.__setattr__(
            self, "ordinal", _strict_int(self.ordinal, "coordinate ordinal")
        )
        object.__setattr__(
            self, "coordinate", _coordinate(self.coordinate)
        )

    def payload(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "ordinal": self.ordinal,
            "coordinate_float64_hex": self.coordinate.hex(),
        }


@dataclass(frozen=True)
class BlockCoordinateArchive:
    role: str
    device: str
    model_id: str
    required_tree_sha256: str
    model_path_sha256: str
    local_runtime_identity_sha256: str
    anonymous_block_sha256: str
    item_count: int
    unit_count: int
    inference_input_count: int
    frozen_batch_size: int
    frozen_max_length: int
    model_initialization_count: int
    batch_call_count: int
    rows: tuple[CoordinateRow, ...]

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise MmqaP1BlockCoordinateWorkerError(
                "coordinate archive role drifted"
            )
        expected = {
            "device": ROLE_DEVICE[self.role],
            "model_id": ROLE_MODEL_ID[self.role],
            "required_tree_sha256": ROLE_REQUIRED_TREE_SHA256[self.role],
            "frozen_batch_size": ROLE_BATCH_SIZE[self.role],
            "frozen_max_length": ROLE_MAX_LENGTH[self.role],
            "model_initialization_count": 1,
        }
        observed = {
            "device": self.device,
            "model_id": self.model_id,
            "required_tree_sha256": self.required_tree_sha256,
            "frozen_batch_size": self.frozen_batch_size,
            "frozen_max_length": self.frozen_max_length,
            "model_initialization_count": self.model_initialization_count,
        }
        if observed != expected:
            raise MmqaP1BlockCoordinateWorkerError(
                "coordinate archive frozen role contract drifted"
            )
        for value, field in (
            (self.required_tree_sha256, "required tree identity"),
            (self.model_path_sha256, "model path identity"),
            (
                self.local_runtime_identity_sha256,
                "local runtime identity",
            ),
            (self.anonymous_block_sha256, "anonymous block identity"),
        ):
            _sha256(value, field)
        for value, field, minimum in (
            (self.item_count, "archive item count", 1),
            (self.unit_count, "archive unit count", 1),
            (
                self.inference_input_count,
                "archive inference input count",
                1,
            ),
            (self.batch_call_count, "archive batch call count", 1),
        ):
            _strict_int(value, field, minimum=minimum)
        rows = tuple(self.rows)
        if (
            len(rows) != self.unit_count
            or not all(isinstance(row, CoordinateRow) for row in rows)
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "coordinate archive row count drifted"
            )
        if self.batch_call_count != (
            self.inference_input_count + self.frozen_batch_size - 1
        ) // self.frozen_batch_size:
            raise MmqaP1BlockCoordinateWorkerError(
                "coordinate archive batch call count drifted"
            )
        object.__setattr__(self, "rows", rows)

    def body(self) -> dict[str, object]:
        return {
            "schema": COORDINATE_ARCHIVE_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "role": self.role,
            "device": self.device,
            "model_id": self.model_id,
            "required_tree_sha256": self.required_tree_sha256,
            "model_path_sha256": self.model_path_sha256,
            "local_runtime_identity_sha256": (
                self.local_runtime_identity_sha256
            ),
            "anonymous_block_sha256": self.anonymous_block_sha256,
            "item_count": self.item_count,
            "unit_count": self.unit_count,
            "inference_input_count": self.inference_input_count,
            "frozen_batch_size": self.frozen_batch_size,
            "frozen_max_length": self.frozen_max_length,
            "model_initialization_count": self.model_initialization_count,
            "batch_call_count": self.batch_call_count,
            "dynamic_batch_resize_count": 0,
            "retry_replay_resample_count": 0,
            "network_or_api_call_count": 0,
            "rows": [row.payload() for row in self.rows],
        }

    def payload(self) -> dict[str, object]:
        body = self.body()
        return {**body, "archive_sha256": _semantic_hash(body)}

    @property
    def archive_sha256(self) -> str:
        return str(self.payload()["archive_sha256"])


def parse_coordinate_archive_payload(
    value: Mapping[str, object],
) -> BlockCoordinateArchive:
    if not isinstance(value, Mapping):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate archive must be a mapping"
        )
    _exact_fields(value, _ARCHIVE_FIELDS, "coordinate archive")
    body = {key: value[key] for key in value if key != "archive_sha256"}
    if (
        value.get("schema") != COORDINATE_ARCHIVE_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("archive_sha256") != _semantic_hash(body)
        or value.get("dynamic_batch_resize_count") != 0
        or value.get("retry_replay_resample_count") != 0
        or value.get("network_or_api_call_count") != 0
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate archive identity or policy drifted"
        )
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate archive rows must be an array"
        )
    rows = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise MmqaP1BlockCoordinateWorkerError(
                "coordinate row must be a mapping"
            )
        _exact_fields(raw, _ROW_FIELDS, "coordinate row")
        rows.append(
            CoordinateRow(
                work_id=_work_id(raw.get("work_id")),
                ordinal=_strict_int(
                    raw.get("ordinal"), "coordinate ordinal"
                ),
                coordinate=_float_from_canonical_hex(
                    raw.get("coordinate_float64_hex")
                ),
            )
        )
    return BlockCoordinateArchive(
        role=value.get("role"),  # type: ignore[arg-type]
        device=value.get("device"),  # type: ignore[arg-type]
        model_id=value.get("model_id"),  # type: ignore[arg-type]
        required_tree_sha256=value.get(  # type: ignore[arg-type]
            "required_tree_sha256"
        ),
        model_path_sha256=value.get("model_path_sha256"),  # type: ignore[arg-type]
        local_runtime_identity_sha256=value.get(  # type: ignore[arg-type]
            "local_runtime_identity_sha256"
        ),
        anonymous_block_sha256=value.get(  # type: ignore[arg-type]
            "anonymous_block_sha256"
        ),
        item_count=_strict_int(
            value.get("item_count"), "archive item count", minimum=1
        ),
        unit_count=_strict_int(
            value.get("unit_count"), "archive unit count", minimum=1
        ),
        inference_input_count=_strict_int(
            value.get("inference_input_count"),
            "archive inference input count",
            minimum=1,
        ),
        frozen_batch_size=_strict_int(
            value.get("frozen_batch_size"),
            "archive frozen batch size",
            minimum=1,
        ),
        frozen_max_length=_strict_int(
            value.get("frozen_max_length"),
            "archive frozen maximum length",
            minimum=1,
        ),
        model_initialization_count=_strict_int(
            value.get("model_initialization_count"),
            "archive model initialization count",
            minimum=1,
        ),
        batch_call_count=_strict_int(
            value.get("batch_call_count"),
            "archive batch call count",
            minimum=1,
        ),
        rows=tuple(rows),
    )


def _expected_row_keys(
    items: Sequence[AnonymousBlockItem],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (item.work_id, unit.ordinal)
        for item in items
        for unit in item.work_item.units
    )


def validate_coordinate_archive_for_block(
    archive: BlockCoordinateArchive,
    items: Sequence[AnonymousBlockItem],
    *,
    expected_role: str,
) -> BlockCoordinateArchive:
    checked = validate_block_items(items)
    if (
        not isinstance(archive, BlockCoordinateArchive)
        or expected_role not in ROLES
        or archive.role != expected_role
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate archive role does not match the controller"
        )
    expected_keys = _expected_row_keys(checked)
    observed_keys = tuple(
        (row.work_id, row.ordinal) for row in archive.rows
    )
    unit_count = len(expected_keys)
    expected_inputs = (
        unit_count + len(checked)
        if expected_role == ROLE_MINILM
        else unit_count
    )
    block_sha256 = str(
        anonymous_block_payload(checked)["block_sha256"]
    )
    if (
        observed_keys != expected_keys
        or len(set(observed_keys)) != len(observed_keys)
        or archive.item_count != len(checked)
        or archive.unit_count != unit_count
        or archive.inference_input_count != expected_inputs
        or archive.anonymous_block_sha256 != block_sha256
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate rows are missing, duplicated, added, reordered, "
            "or bound to another anonymous block"
        )
    return archive


def _chunks(
    values: Iterable[object], size: int
) -> Iterable[tuple[object, ...]]:
    chunk = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk = []
    if chunk:
        yield tuple(chunk)


def _minilm_inputs(
    items: Sequence[AnonymousBlockItem],
) -> Iterable[tuple[str, int | None, bool, str]]:
    for item in items:
        units = item.work_item.units
        yield (item.work_id, None, False, item.work_item.question)
        for index, unit in enumerate(units):
            yield (
                item.work_id,
                unit.ordinal,
                index + 1 == len(units),
                unit.serialized_content,
            )


def _cross_encoder_inputs(
    items: Sequence[AnonymousBlockItem],
) -> Iterable[tuple[str, int, tuple[str, str]]]:
    for item in items:
        for unit in item.work_item.units:
            yield (
                item.work_id,
                unit.ordinal,
                (item.work_item.question, unit.serialized_content),
            )


def _initialize_once(
    binding: BlockModelBinding,
    initialize_model: Callable[..., object],
) -> Callable[..., object]:
    if not callable(initialize_model):
        raise MmqaP1BlockCoordinateWorkerError(
            "model initializer must be callable"
        )
    try:
        backend = initialize_model(
            role=binding.role,
            model_path=binding.model_path,
            model_id=binding.model_id,
            required_tree_sha256=binding.required_tree_sha256,
            local_runtime_identity_sha256=(
                binding.local_runtime_identity_sha256
            ),
            device=binding.device,
            local_files_only=True,
            trust_remote_code=False,
            network_disabled=True,
            deterministic=True,
        )
    except Exception as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            f"{binding.role} model initialization failed; no retry permitted"
        ) from exc
    if not callable(backend):
        raise MmqaP1BlockCoordinateWorkerError(
            "initialized local model backend is not callable"
        )
    return backend


def _run_minilm(
    items: tuple[AnonymousBlockItem, ...],
    binding: BlockModelBinding,
    backend: Callable[..., object],
) -> tuple[CoordinateRow, ...]:
    rows = []
    active_query: dict[str, tuple[float, ...]] = {}
    calls = 0
    for raw_chunk in _chunks(_minilm_inputs(items), binding.batch_size):
        chunk = tuple(raw_chunk)  # typed after construction above
        texts = tuple(row[3] for row in chunk)
        try:
            raw_embeddings = backend(
                texts=texts,
                batch_size=binding.batch_size,
                max_length=binding.max_length,
                normalize_embeddings=True,
                convert_to_numpy=True,
                precision="float32",
                show_progress_bar=False,
                device=binding.device,
                deterministic=True,
            )
        except Exception as exc:
            raise MmqaP1BlockCoordinateWorkerError(
                "MINILM inference batch failed; no resize or retry permitted"
            ) from exc
        calls += 1
        try:
            matrix = executor._validated_embedding_matrix(  # noqa: SLF001
                raw_embeddings, len(chunk)
            )
        except executor.MmqaP1LocalActionExecutorError as exc:
            raise MmqaP1BlockCoordinateWorkerError(
                "MINILM inference batch shape or value drifted"
            ) from exc
        for descriptor, vector in zip(chunk, matrix, strict=True):
            work_id, ordinal, last, _text = descriptor
            if ordinal is None:
                if work_id in active_query:
                    raise MmqaP1BlockCoordinateWorkerError(
                        "MINILM question alignment duplicated"
                    )
                active_query[work_id] = vector
                continue
            question = active_query.get(work_id)
            if question is None:
                raise MmqaP1BlockCoordinateWorkerError(
                    "MINILM unit arrived without its question"
                )
            try:
                score = executor._cosine_scores(  # noqa: SLF001
                    (question, vector)
                )[0]
            except executor.MmqaP1LocalActionExecutorError as exc:
                raise MmqaP1BlockCoordinateWorkerError(
                    "MINILM cosine calculation drifted"
                ) from exc
            rows.append(CoordinateRow(work_id, ordinal, score))
            if last:
                del active_query[work_id]
    expected_calls = (
        sum(1 + len(item.work_item.units) for item in items)
        + binding.batch_size
        - 1
    ) // binding.batch_size
    if active_query or calls != expected_calls:
        raise MmqaP1BlockCoordinateWorkerError(
            "MINILM block alignment or fixed batch count drifted"
        )
    return tuple(rows)


def _run_cross_encoder(
    items: tuple[AnonymousBlockItem, ...],
    binding: BlockModelBinding,
    backend: Callable[..., object],
) -> tuple[CoordinateRow, ...]:
    rows = []
    calls = 0
    for raw_chunk in _chunks(
        _cross_encoder_inputs(items), binding.batch_size
    ):
        chunk = tuple(raw_chunk)
        pairs = tuple(row[2] for row in chunk)
        try:
            raw_logits = backend(
                pairs=pairs,
                batch_size=binding.batch_size,
                max_length=binding.max_length,
                return_logits=True,
                device=binding.device,
                deterministic=True,
            )
        except Exception as exc:
            raise MmqaP1BlockCoordinateWorkerError(
                "CROSS_ENCODER inference batch failed; no resize or retry "
                "permitted"
            ) from exc
        calls += 1
        try:
            scores = executor._validated_ce_scores(  # noqa: SLF001
                raw_logits, len(chunk)
            )
        except executor.MmqaP1LocalActionExecutorError as exc:
            raise MmqaP1BlockCoordinateWorkerError(
                "CROSS_ENCODER inference batch shape or value drifted"
            ) from exc
        rows.extend(
            CoordinateRow(descriptor[0], descriptor[1], score)
            for descriptor, score in zip(chunk, scores, strict=True)
        )
    expected_calls = (
        sum(len(item.work_item.units) for item in items)
        + binding.batch_size
        - 1
    ) // binding.batch_size
    if calls != expected_calls:
        raise MmqaP1BlockCoordinateWorkerError(
            "CROSS_ENCODER fixed batch count drifted"
        )
    return tuple(rows)


def run_block_coordinate_worker(
    items: Sequence[AnonymousBlockItem],
    *,
    model_binding: BlockModelBinding,
    initialize_model: Callable[..., object],
) -> BlockCoordinateArchive:
    """Initialize one model once and process the complete block in chunks."""

    checked = validate_block_items(items)
    if not isinstance(model_binding, BlockModelBinding):
        raise MmqaP1BlockCoordinateWorkerError(
            "worker requires a frozen block model binding"
        )
    backend = _initialize_once(model_binding, initialize_model)
    if model_binding.role == ROLE_MINILM:
        rows = _run_minilm(checked, model_binding, backend)
        inference_inputs = sum(
            1 + len(item.work_item.units) for item in checked
        )
    else:
        rows = _run_cross_encoder(checked, model_binding, backend)
        inference_inputs = sum(
            len(item.work_item.units) for item in checked
        )
    expected_keys = _expected_row_keys(checked)
    if tuple((row.work_id, row.ordinal) for row in rows) != expected_keys:
        raise MmqaP1BlockCoordinateWorkerError(
            "worker coordinate alignment drifted"
        )
    archive = BlockCoordinateArchive(
        role=model_binding.role,
        device=model_binding.device,
        model_id=model_binding.model_id,
        required_tree_sha256=model_binding.required_tree_sha256,
        model_path_sha256=model_binding.model_path_sha256,
        local_runtime_identity_sha256=(
            model_binding.local_runtime_identity_sha256
        ),
        anonymous_block_sha256=str(
            anonymous_block_payload(checked)["block_sha256"]
        ),
        item_count=len(checked),
        unit_count=len(rows),
        inference_input_count=inference_inputs,
        frozen_batch_size=model_binding.batch_size,
        frozen_max_length=model_binding.max_length,
        model_initialization_count=1,
        batch_call_count=(
            inference_inputs + model_binding.batch_size - 1
        )
        // model_binding.batch_size,
        rows=rows,
    )
    return validate_coordinate_archive_for_block(
        archive, checked, expected_role=model_binding.role
    )


def write_private_coordinate_archive(
    path: str | Path,
    archive: BlockCoordinateArchive,
) -> str:
    """Exclusively persist one canonical mode-0600 archive."""

    if not isinstance(archive, BlockCoordinateArchive):
        raise MmqaP1BlockCoordinateWorkerError(
            "private archive writer requires a validated archive"
        )
    destination = Path(path).expanduser().absolute()
    raw = _canonical_json_bytes(archive.payload(), newline=True)
    descriptor = -1
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MmqaP1BlockCoordinateWorkerError(
            "private coordinate archive already exists or cannot be created"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = destination.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or destination.read_bytes() != raw
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "private coordinate archive reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def load_coordinate_archive(path: str | Path) -> BlockCoordinateArchive:
    return parse_coordinate_archive_payload(
        _read_private_canonical_mapping(path)
    )


@dataclass(frozen=True)
class BlockIntegratedItem:
    work_id: str
    actions: integration.IntegratedActions


@dataclass(frozen=True)
class BlockIntegrationResult:
    items: tuple[BlockIntegratedItem, ...]
    anonymous_block_sha256: str
    minilm_archive_sha256: str
    cross_encoder_archive_sha256: str

    def receipt(self) -> dict[str, object]:
        return {
            "schema": BLOCK_MERGE_RECEIPT_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "anonymous_block_sha256": self.anonymous_block_sha256,
            "minilm_archive_sha256": self.minilm_archive_sha256,
            "cross_encoder_archive_sha256": (
                self.cross_encoder_archive_sha256
            ),
            "item_count": len(self.items),
            "coordinate_archive_text_field_count": 0,
            "gold_answer_support_family_qid_read_count": 0,
            "network_or_api_call_count": 0,
            "retry_replay_resample_count": 0,
        }


def merge_coordinate_archives(
    items: Sequence[AnonymousBlockItem],
    *,
    minilm_archive: BlockCoordinateArchive,
    cross_encoder_archive: BlockCoordinateArchive,
    e5_model: core.E5Model | None = None,
    e5_models_by_work_id: Mapping[str, core.E5Model | None] | None = None,
) -> BlockIntegrationResult:
    """Strictly align both private archives, then form each sealed action."""

    checked = validate_block_items(items)
    mini = validate_coordinate_archive_for_block(
        minilm_archive, checked, expected_role=ROLE_MINILM
    )
    ce = validate_coordinate_archive_for_block(
        cross_encoder_archive,
        checked,
        expected_role=ROLE_CROSS_ENCODER,
    )
    if (
        mini.local_runtime_identity_sha256
        != ce.local_runtime_identity_sha256
        or mini.anonymous_block_sha256 != ce.anonymous_block_sha256
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "coordinate archives do not share runtime and block identities"
        )
    if e5_model is not None and e5_models_by_work_id is not None:
        raise MmqaP1BlockCoordinateWorkerError(
            "global and per-item E5 models are mutually exclusive"
        )
    if e5_models_by_work_id is not None:
        if set(e5_models_by_work_id) != {
            item.work_id for item in checked
        } or any(
            model is not None and not isinstance(model, core.E5Model)
            for model in e5_models_by_work_id.values()
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "per-item E5 model mapping drifted"
            )
    if e5_model is not None and not isinstance(e5_model, core.E5Model):
        raise MmqaP1BlockCoordinateWorkerError(
            "global E5 model drifted"
        )
    mini_by_key = {
        (row.work_id, row.ordinal): row.coordinate for row in mini.rows
    }
    ce_by_key = {
        (row.work_id, row.ordinal): row.coordinate for row in ce.rows
    }
    integrated = []
    for item in checked:
        keys = tuple(
            (item.work_id, unit.ordinal) for unit in item.work_item.units
        )
        model = (
            e5_models_by_work_id[item.work_id]
            if e5_models_by_work_id is not None
            else e5_model
        )
        try:
            actions = executor.form_actions_from_local_coordinate_vectors(
                item.work_item,
                tuple(mini_by_key[key] for key in keys),
                tuple(ce_by_key[key] for key in keys),
                e5_model=model,
            )
        except executor.MmqaP1LocalActionExecutorError as exc:
            raise MmqaP1BlockCoordinateWorkerError(
                "coordinate archive merge could not form actions"
            ) from exc
        integrated.append(BlockIntegratedItem(item.work_id, actions))
    return BlockIntegrationResult(
        items=tuple(integrated),
        anonymous_block_sha256=mini.anonymous_block_sha256,
        minilm_archive_sha256=mini.archive_sha256,
        cross_encoder_archive_sha256=ce.archive_sha256,
    )


def _production_initializer(**configuration: object) -> Callable[..., object]:
    """Lazy local-only runtime loader used solely by the explicit worker CLI."""

    role = configuration.get("role")
    model_path = str(configuration.get("model_path"))
    device = str(configuration.get("device"))
    if (
        role not in ROLES
        or configuration.get("local_files_only") is not True
        or configuration.get("trust_remote_code") is not False
        or configuration.get("network_disabled") is not True
        or configuration.get("deterministic") is not True
    ):
        raise MmqaP1BlockCoordinateWorkerError(
            "production initializer policy drifted"
        )
    # Belt-and-suspenders offline flags supplement local_files_only=True.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    if role == ROLE_MINILM:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={
                "local_files_only": True,
                "torch_dtype": torch.float32,
                "use_safetensors": True,
            },
            config_kwargs={
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
        model.max_seq_length = executor.MINILM_MAX_LENGTH
        model.float()
        model.eval()

        def encode(**kwargs: object) -> object:
            if (
                kwargs.get("batch_size") != executor.MINILM_BATCH_SIZE
                or kwargs.get("max_length") != executor.MINILM_MAX_LENGTH
                or kwargs.get("device") != ROLE_DEVICE[ROLE_MINILM]
            ):
                raise MmqaP1BlockCoordinateWorkerError(
                    "production MiniLM fixed batch contract drifted"
                )
            return model.encode(
                kwargs["texts"],
                batch_size=executor.MINILM_BATCH_SIZE,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device=device,
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            )

        return encode

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval().to(device)

    def score(**kwargs: object) -> object:
        if (
            kwargs.get("batch_size") != executor.CROSS_ENCODER_BATCH_SIZE
            or kwargs.get("max_length") != executor.CROSS_ENCODER_MAX_LENGTH
            or kwargs.get("device")
            != ROLE_DEVICE[ROLE_CROSS_ENCODER]
        ):
            raise MmqaP1BlockCoordinateWorkerError(
                "production cross-encoder fixed batch contract drifted"
            )
        pairs = kwargs["pairs"]
        queries = [pair[0] for pair in pairs]  # type: ignore[index]
        passages = [pair[1] for pair in pairs]  # type: ignore[index]
        encoded = tokenizer(
            queries,
            passages,
            max_length=executor.CROSS_ENCODER_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().cpu().reshape(-1)
        return logits.numpy()

    return score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--input-block", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--required-tree-sha256", required=True)
    parser.add_argument("--local-runtime-identity-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    items = load_anonymous_block(arguments.input_block)
    binding = BlockModelBinding(
        role=arguments.role,
        model_path=str(arguments.model.absolute()),
        required_tree_sha256=arguments.required_tree_sha256,
        local_runtime_identity_sha256=(
            arguments.local_runtime_identity_sha256
        ),
    )
    archive = run_block_coordinate_worker(
        items,
        model_binding=binding,
        initialize_model=_production_initializer,
    )
    write_private_coordinate_archive(arguments.output, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION",
    "STUDY_ID",
    "STUDY_DESIGN_SELF_SHA256",
    "ROLE_MINILM",
    "ROLE_CROSS_ENCODER",
    "ROLES",
    "ROLE_DEVICE",
    "ROLE_BATCH_SIZE",
    "ROLE_MAX_LENGTH",
    "ROLE_MODEL_ID",
    "ROLE_REQUIRED_TREE_SHA256",
    "ANONYMOUS_BLOCK_SCHEMA",
    "COORDINATE_ARCHIVE_SCHEMA",
    "BLOCK_MERGE_RECEIPT_SCHEMA",
    "MmqaP1BlockCoordinateWorkerError",
    "AnonymousBlockItem",
    "BlockModelBinding",
    "CoordinateRow",
    "BlockCoordinateArchive",
    "BlockIntegratedItem",
    "BlockIntegrationResult",
    "validate_block_items",
    "anonymous_block_payload",
    "validate_anonymous_block_payload",
    "write_private_anonymous_block",
    "load_anonymous_block",
    "parse_coordinate_archive_payload",
    "validate_coordinate_archive_for_block",
    "run_block_coordinate_worker",
    "write_private_coordinate_archive",
    "load_coordinate_archive",
    "merge_coordinate_archives",
    "main",
]
