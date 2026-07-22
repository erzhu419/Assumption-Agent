"""Trusted, local-only source qualification and acquisition for TAT-QA P19.

This module is the only P19 component permitted to read the official raw and
TagOp source rows.  It validates their exact shared identity, projects the
official TagOp mapping to whole canonical table rows and paragraphs, forms the
four fixed balanced blocks, and writes label-free views separately from sealed
late labels.  It contains no downloader, model, network, or online evaluator.

The formal entry point is deliberately fail-closed.  A committed implementation
freeze and a self-hashed source-download receipt must already exist.  Once those
metadata-only preconditions pass, an exclusive one-shot root and marker are
created *before* any dataset payload byte is opened.  Every subsequent failure
leaves that root terminal and non-reusable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import unicodedata
from typing import Any, Callable, Mapping, Sequence


VERSION = "tatqa_p19_acquisition_v1"
STUDY_ID = "TATQA_P19_TYPED_EVIDENCE_COEVOLUTION_V1"

DESIGN_RELATIVE = Path("manifests/tatqa_p19_typed_evaluator_study_design_v1.json")
DESIGN_FILE_SHA256 = "ee842e4065232670ecd7e12b184d1efefdc14b0bee1c30f553fd71b0d6420e53"
DESIGN_SELF_SHA256 = "c83fc46cecfcaf34455f09ce5356259445f61ef6b623d2baa8998eb532ccc2a7"
CUSTODY_RELATIVE = Path("manifests/tatqa_p19_public_source_custody_v1.json")
CUSTODY_FILE_SHA256 = "c619e6d9091bd5c3d8d70df960e632ed8194a8c18273aa2b0c3fcd701fc6acef"
CUSTODY_SELF_SHA256 = "e37eb1ca699e2b0bbdd6b032fe92b6ae5146894b7118c0a6fa32a21cc09a7d56"
SOURCE_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"

IMPLEMENTATION_FREEZE_RELATIVE = Path("manifests/tatqa_p19_implementation_freeze_v1.json")
IMPLEMENTATION_FREEZE_SCHEMA = "tatqa_p19_implementation_freeze_v1"
RUNTIME_FINGERPRINT_SCHEMA = "tatqa_p19_composite_runtime_fingerprint_v1"
RUNTIME_SUBFINGERPRINT_SCHEMAS = {
    "typed_plan_minilm_runtime_python": (
        "tatqa_p19_typed_minilm_runtime_python_subfingerprint_v1"
    ),
    "hipporag_runtime_python": (
        "tatqa_p19_hipporag_runtime_python_subfingerprint_v1"
    ),
}
RUNTIME_SUBFINGERPRINT_HASHES_FIELD = (
    "runtime_python_subfingerprint_self_sha256s"
)
REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    {
        "assumption_agent/benchmarks/tatqa_p19_acquisition_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_formal_adapters_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_formal_controller_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_formal_study_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_implementation_freeze_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_label_free_runtime_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_offline_finalize_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_public_canary_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_runtime_qualification_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_source_download_v1.py",
        "assumption_agent/benchmarks/tatqa_p19_typed_evaluator_core_v1.py",
        "replication_runtime/qasper_minilm_v1/__init__.py",
        "replication_runtime/qasper_minilm_v1/binding.py",
        "replication_runtime/tatqa_p19_v1/__init__.py",
        "replication_runtime/tatqa_p19_v1/formal_runtime.py",
        "replication_runtime/tatqa_p19_v1/hipporag_contract.py",
        "replication_runtime/tatqa_p19_v1/hipporag_worker.py",
        "replication_runtime/tatqa_p19_v1/runtime_attestation_v1.py",
        "replication_runtime/tatqa_p19_v1/typed_plan_contract.py",
        "replication_runtime/tatqa_p19_v1/typed_plan_worker.py",
        "tests/test_tatqa_p19_acquisition_v1.py",
        "tests/test_tatqa_p19_formal_adapters_v1.py",
        "tests/test_tatqa_p19_formal_controller_v1.py",
        "tests/test_tatqa_p19_formal_runtime_v1.py",
        "tests/test_tatqa_p19_formal_study_v1.py",
        "tests/test_tatqa_p19_hipporag_runtime_v1.py",
        "tests/test_tatqa_p19_implementation_freeze_v1.py",
        "tests/test_tatqa_p19_label_free_runtime_v1.py",
        "tests/test_tatqa_p19_offline_finalize_v1.py",
        "tests/test_tatqa_p19_public_canary_v1.py",
        "tests/test_tatqa_p19_runtime_attestation_v1.py",
        "tests/test_tatqa_p19_runtime_qualification_v1.py",
        "tests/test_tatqa_p19_source_download_v1.py",
        "tests/test_tatqa_p19_study_design_v1.py",
        "tests/test_tatqa_p19_typed_evaluator_core_v1.py",
        "tests/test_tatqa_p19_typed_plan_runtime_v1.py",
    }
)

SOURCE_RECEIPT_RELATIVE = Path(
    "artifacts/tatqa_p19_official_source_v1/source.download.receipt.json"
)
SOURCE_RECEIPT_SCHEMA = "tatqa_p19_source_download_receipt_v1"
SOURCE_ROOT_RELATIVE = Path("artifacts/tatqa_p19_official_source_v1/TAT-QA")
SOURCE_FILES = (
    "LICENSE",
    "dataset_raw/tatqa_dataset_dev.json",
    "dataset_raw/tatqa_dataset_train.json",
    "dataset_tagop/tatqa_dataset_dev.json",
    "dataset_tagop/tatqa_dataset_train.json",
)

FORMAL_ROOT_RELATIVE = Path("artifacts/tatqa_p19_formal_v1")
ACQUISITION_ROOT_RELATIVE = FORMAL_ROOT_RELATIVE / "acquisition"
MARKER_FILENAME = "acquisition.one_shot_marker.json"
SECRET_FILENAME = "selection.secret.private.bin"
LEDGER_FILENAME = "acquisition.ledger.private.json"
PUBLIC_RECEIPT_FILENAME = "acquisition.public.json"
FAILURE_FILENAME = "acquisition.terminal_failure.json"

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
FAMILY_ORDER = ("TABLE", "TEXT", "TABLE_TEXT")
SPLIT_ORDER = ("train", "dev")
BLOCK_SPLIT = {
    "A_form": "train",
    "F_search": "train",
    "A_hold": "train",
    "M_search": "dev",
}
PER_FAMILY_QUOTA = {
    "A_form": 16,
    "F_search": 12,
    "A_hold": 10,
    "M_search": 10,
}
BLOCK_COUNTS = {
    block: PER_FAMILY_QUOTA[block] * len(FAMILY_ORDER) for block in BLOCK_ORDER
}
TOTAL_SELECTED_ITEMS = sum(BLOCK_COUNTS.values())
FAMILY_FROM_SOURCE = {
    "table": "TABLE",
    "text": "TEXT",
    "table-text": "TABLE_TEXT",
}

MIN_TABLE_ROWS = 2
MIN_TABLE_COLUMNS = 2
MIN_PARAGRAPHS = 2
MIN_CANONICAL_UNITS = 5
MAX_CANONICAL_UNITS = 96
MIN_GOLD_UNITS = 1
MAX_GOLD_UNITS = 5

PUBLIC_QUESTION_UIDS = frozenset(
    {
        "9337c3e6-c53f-45a9-836a-02c474ceac16",
        "c4170232-e89c-487a-97c5-afad45e9d702",
        "d81d1ae7-363c-4b47-8eea-1906fef33856",
        "eb787966-fa02-401f-bfaf-ccabf3828b23",
    }
)
PUBLIC_TABLE_UIDS = frozenset({"3ffd9053-a45d-491c-957a-1b2fa0af0570"})

VIEW_FILENAMES = {
    "A_form": "A_form.view.private.json",
    "F_search": "F_search.view.private.json",
    "A_hold": "A_hold.view.private.json",
    "M_search": "M_search.view.presealed.json",
}
LABEL_FILENAMES = {
    "A_form": "A_form.labels.sealed.json",
    "A_hold": "A_hold.labels.sealed.json",
    "M_search": "M_search.labels.presealed.json",
}

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*\Z")
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_SELECT_HMAC_DOMAIN = b"TATQA_P19_SELECT_V1"
_SHUFFLE_HMAC_DOMAIN = b"TATQA_P19_BLOCK_SHUFFLE_V1"
_SECRET_COMMITMENT_DOMAIN = b"TATQA_P19_SELECTION_SECRET_COMMITMENT_V1\0"

RAW_CONTEXT_KEYS = frozenset({"table", "paragraphs", "questions"})
TAGOP_CONTEXT_KEYS = RAW_CONTEXT_KEYS
TABLE_KEYS = frozenset({"uid", "table"})
PARAGRAPH_KEYS = frozenset({"uid", "order", "text"})
RAW_QUESTION_KEYS = frozenset(
    {
        "uid",
        "order",
        "question",
        "answer",
        "derivation",
        "answer_type",
        "answer_from",
        "rel_paragraphs",
        "req_comparison",
        "scale",
    }
)
TAGOP_QUESTION_KEYS = frozenset(
    {
        "uid",
        "question",
        "answer",
        "derivation",
        "answer_type",
        "answer_from",
        "scale",
        "facts",
        "mapping",
    }
)


class TatqaP19AcquisitionError(RuntimeError):
    """The trusted source, matching, firewall, or one-shot contract drifted."""


class TatqaP19OneShotRefusal(TatqaP19AcquisitionError):
    """The formal acquisition root has already been consumed."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP19AcquisitionError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    if field in body:
        raise TatqaP19AcquisitionError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    if not isinstance(value, Mapping):
        raise TatqaP19AcquisitionError("self-hashed value is not an object")
    body = dict(value)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise TatqaP19AcquisitionError("self-hash is missing or malformed")
    if stable_hash(body) != declared:
        raise TatqaP19AcquisitionError("self-hash mismatch")
    return declared


def _reject_constant(value: str) -> None:
    raise TatqaP19AcquisitionError(f"strict JSON contains {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TatqaP19AcquisitionError("strict JSON contains a duplicate object key")
        result[key] = value
    return result


def strict_json_loads(raw: bytes, *, label: str) -> Any:
    if not isinstance(raw, bytes):
        raise TatqaP19AcquisitionError(f"{label} bytes are invalid")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except TatqaP19AcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP19AcquisitionError(f"{label} is not strict UTF-8 JSON") from exc


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare source values without Python's bool/int equality coercion."""

    return _canonical_bytes(left) == _canonical_bytes(right)


def canonical_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise TatqaP19AcquisitionError(f"{field} is not NUL-free text")
    normalized = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    if not normalized and not allow_empty:
        raise TatqaP19AcquisitionError(f"{field} is empty")
    return normalized


def _strict_source_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise TatqaP19AcquisitionError(f"{field} is not an exact nonempty source ID")
    return value


def _unit_order(unit_id: str) -> tuple[int, int]:
    if not isinstance(unit_id, str) or ":" not in unit_id:
        raise TatqaP19AcquisitionError("canonical unit ID is malformed")
    prefix, ordinal = unit_id.split(":", 1)
    if prefix not in {"T", "P"} or not ordinal.isdecimal() or str(int(ordinal)) != ordinal:
        raise TatqaP19AcquisitionError("canonical unit ID is malformed")
    number = int(ordinal)
    if prefix == "P" and number <= 0:
        raise TatqaP19AcquisitionError("paragraph unit order must be positive")
    return (0 if prefix == "T" else 1, number)


@dataclass(frozen=True)
class CanonicalUnit:
    unit_id: str
    text: str

    def __post_init__(self) -> None:
        _unit_order(self.unit_id)
        canonical = canonical_text(self.text, field="canonical unit text")
        if canonical != self.text:
            raise TatqaP19AcquisitionError("canonical unit text is not normalized")

    def payload(self) -> dict[str, str]:
        return {"unit_id": self.unit_id, "text": self.text}


@dataclass(frozen=True)
class Candidate:
    split: str
    source_context_ordinal: int
    source_question_ordinal: int
    table_uid: str
    question_uid: str
    question: str
    units: tuple[CanonicalUnit, ...]
    family: str
    gold_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split not in SPLIT_ORDER:
            raise TatqaP19AcquisitionError("candidate split is invalid")
        if (
            type(self.source_context_ordinal) is not int
            or self.source_context_ordinal < 0
            or type(self.source_question_ordinal) is not int
            or self.source_question_ordinal < 0
        ):
            raise TatqaP19AcquisitionError("candidate source ordinal is invalid")
        _strict_source_id(self.table_uid, field="table UID")
        _strict_source_id(self.question_uid, field="question UID")
        canonical_text(self.question, field="question")
        if self.family not in FAMILY_ORDER:
            raise TatqaP19AcquisitionError("candidate family is invalid")
        if not MIN_GOLD_UNITS <= len(self.gold_unit_ids) <= MAX_GOLD_UNITS:
            raise TatqaP19AcquisitionError("candidate gold-unit count is invalid")
        if tuple(sorted(set(self.gold_unit_ids), key=_unit_order)) != self.gold_unit_ids:
            raise TatqaP19AcquisitionError("candidate gold-unit IDs are not canonical")
        unit_ids = tuple(unit.unit_id for unit in self.units)
        if len(unit_ids) != len(set(unit_ids)) or not set(self.gold_unit_ids).issubset(unit_ids):
            raise TatqaP19AcquisitionError("candidate units do not contain exact gold")
    @property
    def context_key(self) -> str:
        return self.table_uid


@dataclass(frozen=True)
class Slot:
    ordinal: int
    block: str
    split: str
    family: str
    family_slot_ordinal: int


@dataclass(frozen=True)
class Qualification:
    candidates: tuple[Candidate, ...]
    public_aggregate: Mapping[str, Any]


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TatqaP19AcquisitionError(f"{field} is not an object")
    return value


def _runtime_subfingerprint_self_hashes(
    runtime_fingerprint: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    if runtime_fingerprint.get("schema") != RUNTIME_FINGERPRINT_SCHEMA:
        raise TatqaP19AcquisitionError("runtime fingerprint schema drifted")
    fingerprint_self = verify_self_hash(runtime_fingerprint)
    inventory = _require_mapping(
        runtime_fingerprint.get("runtime_inventory"),
        field="runtime fingerprint inventory",
    )
    nested = _require_mapping(
        inventory.get("runtime_python_subfingerprints"),
        field="runtime Python subfingerprints",
    )
    if set(nested) != set(RUNTIME_SUBFINGERPRINT_SCHEMAS):
        raise TatqaP19AcquisitionError(
            "runtime Python subfingerprint registry drifted"
        )
    result: dict[str, str] = {}
    for key, schema in RUNTIME_SUBFINGERPRINT_SCHEMAS.items():
        value = _require_mapping(nested.get(key), field=f"{key} subfingerprint")
        if value.get("schema") != schema:
            raise TatqaP19AcquisitionError(
                f"{key} subfingerprint schema drifted"
            )
        result[key] = verify_self_hash(value)
    return fingerprint_self, result


def validate_production_canary_capability_receipts(
    value: Mapping[str, Any],
    *,
    runtime_fingerprint: Mapping[str, Any],
) -> None:
    """Recompute capability proof and its two runtime cross-bindings."""

    fingerprint_self, expected_subfingerprints = (
        _runtime_subfingerprint_self_hashes(runtime_fingerprint)
    )
    observed_subfingerprints = _require_mapping(
        value.get(RUNTIME_SUBFINGERPRINT_HASHES_FIELD),
        field="canary runtime Python subfingerprint hashes",
    )
    if (
        value.get("runtime_fingerprint_self_sha256") != fingerprint_self
        or set(observed_subfingerprints) != set(expected_subfingerprints)
    ):
        raise TatqaP19AcquisitionError(
            "canary runtime subfingerprint cross-binding drifted"
        )
    for key, expected in expected_subfingerprints.items():
        observed = observed_subfingerprints.get(key)
        if (
            not isinstance(observed, str)
            or _HEX64.fullmatch(observed) is None
            or observed != expected
        ):
            raise TatqaP19AcquisitionError(
                "canary runtime subfingerprint cross-binding drifted"
            )

    network = ["IPAddressDeny=any", "RestrictAddressFamilies=AF_UNIX"]
    isolation = "systemd_InaccessiblePaths_official_source_and_acquisition_v1"

    def require_hash(row: Mapping[str, Any], field: str) -> None:
        observed = row.get(field)
        if not isinstance(observed, str) or _HEX64.fullmatch(observed) is None:
            raise TatqaP19AcquisitionError("canary capability hash drifted")

    def require_unit_closure(
        value: object, *, unit_name_sha256: str
    ) -> Mapping[str, Any]:
        row = _require_mapping(value, field="systemd unit closure")
        expected = {
            "active_state",
            "control_group_process_count",
            "control_group_sha256",
            "control_group_thread_count",
            "load_state",
            "main_pid",
            "schema",
            "sub_state",
            "systemctl_reset_failed_returncode",
            "systemctl_reset_failed_stderr_sha256",
            "systemctl_reset_failed_stdout_sha256",
            "systemctl_show_returncode",
            "systemctl_show_stderr_sha256",
            "systemctl_show_stdout_sha256",
            "unit_name_sha256",
        }
        if (
            set(row) != expected
            or row.get("schema")
            != "tatqa_p19_formal_runtime_v1_systemd_unit_closure_v1"
            or row.get("unit_name_sha256") != unit_name_sha256
            or row.get("load_state") != "not-found"
            or row.get("active_state") != "inactive"
            or row.get("sub_state") != "dead"
            or type(row.get("main_pid")) is not int
            or row["main_pid"] != 0
            or type(row.get("control_group_process_count")) is not int
            or row["control_group_process_count"] != 0
            or type(row.get("control_group_thread_count")) is not int
            or row["control_group_thread_count"] != 0
            or type(row.get("systemctl_show_returncode")) is not int
            or row["systemctl_show_returncode"] != 0
            or type(row.get("systemctl_reset_failed_returncode")) is not int
            or row["systemctl_reset_failed_returncode"] not in {0, 1}
        ):
            raise TatqaP19AcquisitionError("canary systemd closure drifted")
        for field in (
            "control_group_sha256",
            "systemctl_reset_failed_stderr_sha256",
            "systemctl_reset_failed_stdout_sha256",
            "systemctl_show_stderr_sha256",
            "systemctl_show_stdout_sha256",
            "unit_name_sha256",
        ):
            require_hash(row, field)
        return row

    def require_start_policy(
        value: object, *, unit_name_sha256: str
    ) -> Mapping[str, Any]:
        row = _require_mapping(value, field="systemd start policy")
        expected = {
            "active_state",
            "control_group_sha256",
            "kill_mode",
            "load_state",
            "main_pid",
            "schema",
            "sub_state",
            "systemctl_show_returncode",
            "systemctl_show_stderr_sha256",
            "systemctl_show_stdout_sha256",
            "tasks_max",
            "unit_name_sha256",
        }
        if (
            set(row) != expected
            or row.get("schema")
            != "tatqa_p19_formal_runtime_v1_systemd_start_policy_v1"
            or row.get("unit_name_sha256") != unit_name_sha256
            or row.get("load_state") != "loaded"
            or row.get("active_state") != "active"
            or row.get("sub_state") != "running"
            or type(row.get("main_pid")) is not int
            or row["main_pid"] <= 1
            or type(row.get("tasks_max")) is not int
            or row["tasks_max"] != 3
            or row.get("kill_mode") != "control-group"
            or type(row.get("systemctl_show_returncode")) is not int
            or row["systemctl_show_returncode"] != 0
        ):
            raise TatqaP19AcquisitionError("canary systemd start policy drifted")
        for field in (
            "control_group_sha256",
            "systemctl_show_stderr_sha256",
            "systemctl_show_stdout_sha256",
            "unit_name_sha256",
        ):
            require_hash(row, field)
        return row

    typed = _require_mapping(
        value.get("typed_plan_worker_receipt_snapshot"),
        field="typed capability snapshot",
    )
    typed_rows = typed.get("receipts")
    if (
        set(typed) != {"schema", "capability_class", "receipts"}
        or typed.get("schema")
        != "tatqa_p19_typed_plan_capability_receipt_snapshot_v1"
        or typed.get("capability_class") != "SystemdTypedPlanBatchRunner"
        or not isinstance(typed_rows, list)
        or len(typed_rows) != 2
    ):
        raise TatqaP19AcquisitionError("typed capability snapshot drifted")
    typed_keys = {
        "schema",
        "block",
        "item_count",
        "input_sha256",
        "model_execution_finished_monotonic_ns",
        "model_execution_started_monotonic_ns",
        "output_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "batch_size",
        "physical_GPU",
        "worker_pid",
        "model_context_tokens",
        "filesystem_isolation",
        "network_properties",
        "systemd_unit_closure",
        "systemd_unit_name_sha256",
    }
    for index, row_value in enumerate(typed_rows, start=1):
        row = _require_mapping(row_value, field="typed capability receipt")
        started = row.get("model_execution_started_monotonic_ns")
        finished = row.get("model_execution_finished_monotonic_ns")
        if (
            set(row) != typed_keys
            or row.get("schema")
            != "tatqa_p19_formal_runtime_v1_typed_plan_transport_receipt_v1"
            or row.get("block") != f"PUBLIC_CANARY_REPEAT_{index}"
            or type(row.get("item_count")) is not int
            or row["item_count"] != 1
            or type(row.get("batch_size")) is not int
            or row["batch_size"] != 4
            or row.get("physical_GPU") != "1"
            or isinstance(row.get("worker_pid"), bool)
            or not isinstance(row.get("worker_pid"), int)
            or row["worker_pid"] <= 1
            or isinstance(row.get("model_context_tokens"), bool)
            or not isinstance(row.get("model_context_tokens"), int)
            or row["model_context_tokens"] < 16_640
            or row.get("filesystem_isolation") != isolation
            or row.get("network_properties") != network
            or isinstance(started, bool)
            or not isinstance(started, int)
            or started <= 0
            or isinstance(finished, bool)
            or not isinstance(finished, int)
            or finished <= started
            or row.get("input_sha256")
            != value.get("typed_plan_input_file_sha256")
            or row.get("output_sha256")
            != value.get("typed_plan_output_file_sha256")
        ):
            raise TatqaP19AcquisitionError("typed capability receipt drifted")
        for field in (
            "input_sha256",
            "output_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "systemd_unit_name_sha256",
        ):
            require_hash(row, field)
        require_unit_closure(
            row.get("systemd_unit_closure"),
            unit_name_sha256=row["systemd_unit_name_sha256"],
        )

    minilm = _require_mapping(
        value.get("minilm_worker_receipt_snapshot"),
        field="MiniLM capability snapshot",
    )
    runtime_receipt = _require_mapping(
        minilm.get("runtime_receipt"), field="MiniLM runtime receipt"
    )
    startup = _require_mapping(
        minilm.get("startup_canary_receipt"), field="MiniLM startup receipt"
    )
    execution = _require_mapping(
        minilm.get("execution"), field="MiniLM execution receipt"
    )
    if (
        set(minilm)
        != {
            "schema",
            "execution",
            "omitted_absolute_path_fields",
            "runtime_receipt",
            "startup_canary_receipt",
        }
        or minilm.get("schema")
        != "tatqa_p19_minilm_capability_receipt_snapshot_v1"
        or minilm.get("omitted_absolute_path_fields")
        != ["asset_manifest_path", "model_root"]
        or execution
        != {
            "capability_class": "BoundMiniLMEncoder",
            "device": "cpu",
            "dtype": "float32",
            "in_process": True,
            "torch_threads": 1,
        }
        or set(runtime_receipt)
        != {
            "asset_file_sha256",
            "asset_sha256",
            "embedding_dimension",
            "maximum_sequence_length",
            "model_tree_sha256",
            "runtime_versions",
            "status",
            "weights_sha256",
        }
        or runtime_receipt.get("status")
        != "verified_offline_immutable_qasper_minilm_runtime"
        or runtime_receipt.get("embedding_dimension") != 384
        or set(startup)
        != {
            "float32_bytes_sha256",
            "quantized_embedding_matrix_sha256",
            "qasper_rows_or_archives_accessed_by_canary",
            "repeat_count",
            "repeat_exact",
            "sentence_count",
            "status",
            "text_vector_sha256",
        }
        or startup.get("status") != "passed_exact_row_free_synthetic_canary"
        or startup.get("repeat_count") != 2
        or startup.get("repeat_exact") is not True
        or startup.get("qasper_rows_or_archives_accessed_by_canary") is not False
    ):
        raise TatqaP19AcquisitionError("MiniLM capability receipt drifted")
    for field in (
        "asset_file_sha256",
        "asset_sha256",
        "model_tree_sha256",
        "weights_sha256",
    ):
        require_hash(runtime_receipt, field)
    for field in (
        "float32_bytes_sha256",
        "quantized_embedding_matrix_sha256",
        "text_vector_sha256",
    ):
        require_hash(startup, field)

    hippo = _require_mapping(
        value.get("hippo_worker_receipt_snapshot"),
        field="HippoRAG capability snapshot",
    )
    hippo_rows = hippo.get("receipts")
    if (
        set(hippo) != {"schema", "capability_class", "receipts"}
        or hippo.get("schema")
        != "tatqa_p19_hipporag_capability_receipt_snapshot_v1"
        or hippo.get("capability_class") != "SystemdHippoByteRunner"
        or not isinstance(hippo_rows, list)
        or len(hippo_rows) != 1
    ):
        raise TatqaP19AcquisitionError("HippoRAG capability snapshot drifted")
    row = _require_mapping(hippo_rows[0], field="HippoRAG capability receipt")
    if (
        set(row)
        != {
            "schema",
            "block",
            "configured_torch_interop_threads",
            "configured_torch_intraop_threads",
            "item_commitment_sha256",
            "input_file_sha256",
            "input_semantic_sha256",
            "model_execution_finished_monotonic_ns",
            "model_execution_started_monotonic_ns",
            "observed_process_thread_peak",
            "output_file_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "CPU_threads",
            "worker_pid",
            "filesystem_isolation",
            "visible_GPU",
            "network_properties",
            "maximum_worker_process_threads",
            "systemd_start_policy",
            "systemd_start_policy_sha256",
            "systemd_tasks_max",
            "systemd_unit_closure",
            "systemd_unit_name_sha256",
            "thread_monitor_process_reservation",
        }
        or row.get("schema")
        != "tatqa_p19_formal_runtime_v1_hippo_transport_receipt_v1"
        or row.get("block") != "PUBLIC_CANARY_HIPPO"
        or row.get("item_commitment_sha256")
        != value.get("public_synthetic_fixture_sha256")
        or row.get("input_file_sha256")
        != value.get("hippo_canary_input_file_sha256")
        or row.get("input_semantic_sha256")
        != value.get("hippo_canary_input_semantic_sha256")
        or row.get("output_file_sha256")
        != value.get("hippo_canary_output_file_sha256")
        or type(row.get("CPU_threads")) is not int
        or row["CPU_threads"] != 2
        or isinstance(row.get("worker_pid"), bool)
        or not isinstance(row.get("worker_pid"), int)
        or row["worker_pid"] <= 1
        or row.get("filesystem_isolation") != isolation
        or row.get("visible_GPU") != ""
        or row.get("network_properties") != network
        or type(row.get("systemd_tasks_max")) is not int
        or row["systemd_tasks_max"] != 3
        or type(row.get("thread_monitor_process_reservation")) is not int
        or row["thread_monitor_process_reservation"] != 1
        or type(row.get("maximum_worker_process_threads")) is not int
        or row["maximum_worker_process_threads"] != 2
        or row["thread_monitor_process_reservation"]
        + row["maximum_worker_process_threads"]
        != row["systemd_tasks_max"]
        or isinstance(row.get("model_execution_started_monotonic_ns"), bool)
        or not isinstance(row.get("model_execution_started_monotonic_ns"), int)
        or row["model_execution_started_monotonic_ns"] <= 0
        or isinstance(row.get("model_execution_finished_monotonic_ns"), bool)
        or not isinstance(row.get("model_execution_finished_monotonic_ns"), int)
        or row["model_execution_finished_monotonic_ns"]
        <= row["model_execution_started_monotonic_ns"]
        or any(
            isinstance(row.get(field), bool)
            or not isinstance(row.get(field), int)
            or not 1 <= row[field] <= 2
            for field in (
                "configured_torch_intraop_threads",
                "configured_torch_interop_threads",
                "observed_process_thread_peak",
            )
        )
    ):
        raise TatqaP19AcquisitionError("HippoRAG capability receipt drifted")
    for field in (
        "item_commitment_sha256",
        "input_file_sha256",
        "input_semantic_sha256",
        "output_file_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "systemd_start_policy_sha256",
        "systemd_unit_name_sha256",
    ):
        require_hash(row, field)
    start_policy = require_start_policy(
        row.get("systemd_start_policy"),
        unit_name_sha256=row["systemd_unit_name_sha256"],
    )
    closure = require_unit_closure(
        row.get("systemd_unit_closure"),
        unit_name_sha256=row["systemd_unit_name_sha256"],
    )
    if (
        start_policy.get("main_pid") != row.get("worker_pid")
        or start_policy.get("control_group_sha256")
        != closure.get("control_group_sha256")
        or stable_hash(start_policy) != row.get("systemd_start_policy_sha256")
    ):
        raise TatqaP19AcquisitionError("canary systemd binding drifted")

    for prefix, snapshot in (
        ("typed_plan", typed),
        ("minilm", minilm),
        ("hippo", hippo),
    ):
        if value.get(f"{prefix}_worker_receipt_source") != "capability_receipt_snapshot":
            raise TatqaP19AcquisitionError("canary used a fallback capability receipt")
        if value.get(f"{prefix}_worker_receipt_sha256") != stable_hash(snapshot):
            raise TatqaP19AcquisitionError("canary capability snapshot hash drifted")


def _require_list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TatqaP19AcquisitionError(f"{field} is not an array")
    return value


def _canonical_units(
    table: Mapping[str, Any], paragraphs_value: object
) -> tuple[tuple[CanonicalUnit, ...], int, int, dict[int, str]]:
    if set(table) != TABLE_KEYS:
        raise TatqaP19AcquisitionError("table schema drifted")
    matrix = _require_list(table.get("table"), field="table matrix")
    if len(matrix) < MIN_TABLE_ROWS:
        raise TatqaP19AcquisitionError("table has fewer than two rows")
    first = _require_list(matrix[0], field="table header row")
    width = len(first)
    if width < MIN_TABLE_COLUMNS:
        raise TatqaP19AcquisitionError("table has fewer than two columns")
    normalized_rows: list[list[str]] = []
    for row_ordinal, raw_row in enumerate(matrix):
        row = _require_list(raw_row, field=f"table row {row_ordinal}")
        if len(row) != width:
            raise TatqaP19AcquisitionError("table is not rectangular")
        normalized_rows.append(
            [
                canonical_text(cell, field="table cell", allow_empty=True)
                for cell in row
            ]
        )

    paragraphs = _require_list(paragraphs_value, field="paragraphs")
    if len(paragraphs) < MIN_PARAGRAPHS:
        raise TatqaP19AcquisitionError("context has fewer than two paragraphs")
    paragraph_text_by_order: dict[int, str] = {}
    paragraph_uids: set[str] = set()
    for raw_paragraph in paragraphs:
        paragraph = _require_mapping(raw_paragraph, field="paragraph")
        if set(paragraph) != PARAGRAPH_KEYS:
            raise TatqaP19AcquisitionError("paragraph schema drifted")
        uid = _strict_source_id(paragraph.get("uid"), field="paragraph UID")
        if uid in paragraph_uids:
            raise TatqaP19AcquisitionError("paragraph UID is duplicated")
        paragraph_uids.add(uid)
        order = paragraph.get("order")
        if type(order) is not int or order <= 0 or order in paragraph_text_by_order:
            raise TatqaP19AcquisitionError("paragraph order is not unique and positive")
        text = paragraph.get("text")
        if not isinstance(text, str) or "\x00" in text:
            raise TatqaP19AcquisitionError("paragraph source text is invalid")
        # Preserve the exact source string for span bounds.  Canonicalization is
        # used only for the shared retrieval unit.
        paragraph_text_by_order[order] = text

    canonical_unit_count = len(matrix) + len(paragraph_text_by_order)
    if canonical_unit_count < MIN_CANONICAL_UNITS:
        raise TatqaP19AcquisitionError(
            "context has fewer than five canonical units required by top-k"
        )
    if canonical_unit_count > MAX_CANONICAL_UNITS:
        raise TatqaP19AcquisitionError("context exceeds 96 canonical units")

    headers = normalized_rows[0]
    units: list[CanonicalUnit] = []
    header_text = "TABLE_HEADER|" + "||".join(
        f"C{column}={value}" for column, value in enumerate(headers)
    )
    units.append(CanonicalUnit("T:0", header_text))
    for row_ordinal, row in enumerate(normalized_rows[1:], start=1):
        cells = []
        for column, value in enumerate(row):
            header = headers[column] or f"COLUMN_{column}"
            cells.append(f"{header}={value}")
        units.append(
            CanonicalUnit(
                f"T:{row_ordinal}",
                f"TABLE_ROW_{row_ordinal}|" + "||".join(cells),
            )
        )
    for order in sorted(paragraph_text_by_order):
        text = canonical_text(
            paragraph_text_by_order[order], field="paragraph text"
        )
        units.append(CanonicalUnit(f"P:{order}", f"PARAGRAPH_{order}|{text}"))
    return tuple(units), len(matrix), width, paragraph_text_by_order


def project_gold_mapping(
    mapping_value: object,
    *,
    table_row_count: int,
    table_column_count: int,
    paragraph_text_by_order: Mapping[int, str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate TagOp mapping and project cells/spans to whole units."""

    mapping = _require_mapping(mapping_value, field="TagOp mapping")
    if set(mapping) != {"table", "paragraph"}:
        raise TatqaP19AcquisitionError("TagOp mapping schema drifted")

    table_units: set[str] = set()
    coordinates = _require_list(mapping["table"], field="table mapping")
    for coordinate in coordinates:
        pair = _require_list(coordinate, field="table mapping coordinate")
        if len(pair) != 2 or any(type(value) is not int for value in pair):
            raise TatqaP19AcquisitionError("table mapping coordinate is malformed")
        row, column = pair
        if not (0 <= row < table_row_count and 0 <= column < table_column_count):
            raise TatqaP19AcquisitionError("table mapping coordinate is out of bounds")
        table_units.add(f"T:{row}")

    paragraph_units: set[str] = set()
    paragraph_mapping = _require_mapping(mapping["paragraph"], field="paragraph mapping")
    for raw_order, raw_spans in paragraph_mapping.items():
        if not isinstance(raw_order, str) or _POSITIVE_DECIMAL.fullmatch(raw_order) is None:
            raise TatqaP19AcquisitionError("paragraph mapping key is not canonical positive decimal")
        order = int(raw_order)
        if order not in paragraph_text_by_order:
            raise TatqaP19AcquisitionError("paragraph mapping order does not exist exactly once")
        spans = _require_list(raw_spans, field="paragraph mapping spans")
        if not spans:
            raise TatqaP19AcquisitionError("paragraph mapping span list is empty")
        source_text = paragraph_text_by_order[order]
        for raw_span in spans:
            span = _require_list(raw_span, field="paragraph mapping span")
            if len(span) != 2 or any(type(value) is not int for value in span):
                raise TatqaP19AcquisitionError("paragraph mapping span is malformed")
            start, end = span
            if not (0 <= start < end <= len(source_text)):
                raise TatqaP19AcquisitionError("paragraph mapping span is out of bounds")
            if not source_text[start:end].strip():
                raise TatqaP19AcquisitionError("paragraph mapping span is blank")
        paragraph_units.add(f"P:{order}")

    table_tuple = tuple(sorted(table_units, key=_unit_order))
    paragraph_tuple = tuple(sorted(paragraph_units, key=_unit_order))
    union = tuple(sorted(table_units | paragraph_units, key=_unit_order))
    return table_tuple, paragraph_tuple, union


def _validate_shared_question_identity(raw: Mapping[str, Any], tagop: Mapping[str, Any]) -> None:
    if set(raw) != RAW_QUESTION_KEYS:
        raise TatqaP19AcquisitionError("raw question schema drifted")
    if set(tagop) != TAGOP_QUESTION_KEYS:
        raise TatqaP19AcquisitionError("TagOp question schema drifted")
    # The frozen identity contract is deliberately narrower than equality of
    # the entire derived TagOp question object.  TagOp is allowed to add facts
    # and heuristic mapping fields.  The source-native question identity and
    # answer_from family, however, must remain byte-exact before normalization.
    for key in ("uid", "question", "answer_from"):
        if key not in tagop or not _exact_json_equal(raw[key], tagop[key]):
            raise TatqaP19AcquisitionError(
                "raw and TagOp question identity is not exact"
            )


def parse_source_pair(
    raw_payload: object,
    tagop_payload: object,
    *,
    split: str,
    public_question_uids: frozenset[str] = PUBLIC_QUESTION_UIDS,
    public_table_uids: frozenset[str] = PUBLIC_TABLE_UIDS,
) -> tuple[tuple[Candidate, ...], dict[str, Any]]:
    """Parse one exact raw/TagOp split without publishing any source ID."""

    if split not in SPLIT_ORDER:
        raise TatqaP19AcquisitionError("source split is invalid")
    raw_contexts = _require_list(raw_payload, field=f"raw {split} root")
    tagop_contexts = _require_list(tagop_payload, field=f"TagOp {split} root")
    if len(raw_contexts) != len(tagop_contexts):
        raise TatqaP19AcquisitionError("raw and TagOp context counts differ")

    candidates: list[Candidate] = []
    seen_tables: set[str] = set()
    seen_questions: set[str] = set()
    counters: Counter[str] = Counter()
    unit_counts: list[int] = []
    for context_ordinal, (raw_value, tagop_value) in enumerate(
        zip(raw_contexts, tagop_contexts, strict=True)
    ):
        raw = _require_mapping(raw_value, field="raw context")
        tagop = _require_mapping(tagop_value, field="TagOp context")
        if set(raw) != RAW_CONTEXT_KEYS or set(tagop) != TAGOP_CONTEXT_KEYS:
            raise TatqaP19AcquisitionError("source context schema drifted")
        if not _exact_json_equal(raw["table"], tagop["table"]):
            raise TatqaP19AcquisitionError("raw and TagOp table identity is not exact")
        if not _exact_json_equal(raw["paragraphs"], tagop["paragraphs"]):
            raise TatqaP19AcquisitionError("raw and TagOp paragraph identity is not exact")

        raw_table = _require_mapping(raw["table"], field="raw table")
        if set(raw_table) != TABLE_KEYS:
            raise TatqaP19AcquisitionError("table schema drifted")
        for paragraph_value in _require_list(
            raw["paragraphs"], field="raw paragraphs"
        ):
            paragraph = _require_mapping(paragraph_value, field="raw paragraph")
            if set(paragraph) != PARAGRAPH_KEYS:
                raise TatqaP19AcquisitionError("paragraph schema drifted")
        table_uid = _strict_source_id(raw_table.get("uid"), field="table UID")
        if table_uid in seen_tables:
            raise TatqaP19AcquisitionError("table UID is duplicated within a split")
        seen_tables.add(table_uid)
        raw_questions = _require_list(raw["questions"], field="raw questions")
        tagop_questions = _require_list(tagop["questions"], field="TagOp questions")
        if len(raw_questions) != len(tagop_questions):
            raise TatqaP19AcquisitionError("raw and TagOp question counts differ")
        question_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
        for raw_question_value, tagop_question_value in zip(
            raw_questions, tagop_questions, strict=True
        ):
            raw_question = _require_mapping(raw_question_value, field="raw question")
            tagop_question = _require_mapping(tagop_question_value, field="TagOp question")
            _validate_shared_question_identity(raw_question, tagop_question)
            question_uid = _strict_source_id(raw_question.get("uid"), field="question UID")
            if question_uid in seen_questions:
                raise TatqaP19AcquisitionError("question UID is duplicated within a split")
            seen_questions.add(question_uid)
            question_pairs.append((raw_question, tagop_question, question_uid))

        counters["source_context_count"] += 1
        counters["source_question_count"] += len(question_pairs)
        if table_uid in public_table_uids or any(
            question_uid in public_question_uids
            for _raw, _tagop, question_uid in question_pairs
        ):
            counters["public_example_excluded_context_count"] += 1
            counters["public_example_excluded_question_count"] += len(question_pairs)
            continue

        units, row_count, column_count, paragraphs_by_order = _canonical_units(
            raw_table, raw["paragraphs"]
        )
        unit_counts.append(len(units))
        for question_ordinal, (raw_question, tagop_question, question_uid) in enumerate(
            question_pairs
        ):
            question = canonical_text(raw_question["question"], field="question")
            source_family = raw_question["answer_from"]
            if source_family not in FAMILY_FROM_SOURCE:
                raise TatqaP19AcquisitionError("answer_from is outside the frozen registry")
            family = FAMILY_FROM_SOURCE[source_family]
            table_gold, paragraph_gold, gold = project_gold_mapping(
                tagop_question["mapping"],
                table_row_count=row_count,
                table_column_count=column_count,
                paragraph_text_by_order=paragraphs_by_order,
            )
            observed_family = (
                "TABLE_TEXT"
                if table_gold and paragraph_gold
                else "TABLE"
                if table_gold
                else "TEXT"
                if paragraph_gold
                else None
            )
            if observed_family != family:
                counters["family_mapping_inconsistent_question_count"] += 1
                continue
            if not MIN_GOLD_UNITS <= len(gold) <= MAX_GOLD_UNITS:
                counters["gold_unit_count_outside_range_question_count"] += 1
                continue
            candidates.append(
                Candidate(
                    split=split,
                    source_context_ordinal=context_ordinal,
                    source_question_ordinal=question_ordinal,
                    table_uid=table_uid,
                    question_uid=question_uid,
                    question=question,
                    units=units,
                    family=family,
                    gold_unit_ids=gold,
                )
            )
            counters[f"eligible_{family}_question_count"] += 1

    aggregate = {
        "source_context_count": counters["source_context_count"],
        "source_question_count": counters["source_question_count"],
        "public_example_excluded_context_count": counters[
            "public_example_excluded_context_count"
        ],
        "public_example_excluded_question_count": counters[
            "public_example_excluded_question_count"
        ],
        "family_mapping_inconsistent_question_count": counters[
            "family_mapping_inconsistent_question_count"
        ],
        "gold_unit_count_outside_range_question_count": counters[
            "gold_unit_count_outside_range_question_count"
        ],
        "eligible_question_count_by_family": {
            family: counters[f"eligible_{family}_question_count"]
            for family in FAMILY_ORDER
        },
        "canonical_unit_count": {
            "context_count": len(unit_counts),
            "minimum": min(unit_counts) if unit_counts else 0,
            "maximum": max(unit_counts) if unit_counts else 0,
            "sum": sum(unit_counts),
        },
    }
    return tuple(candidates), aggregate


def _slots() -> tuple[Slot, ...]:
    result: list[Slot] = []
    for block in BLOCK_ORDER:
        for family in FAMILY_ORDER:
            for family_slot in range(PER_FAMILY_QUOTA[block]):
                result.append(
                    Slot(
                        ordinal=len(result),
                        block=block,
                        split=BLOCK_SPLIT[block],
                        family=family,
                        family_slot_ordinal=family_slot,
                    )
                )
    if len(result) != TOTAL_SELECTED_ITEMS:
        raise TatqaP19AcquisitionError("fixed slot registry drifted")
    return tuple(result)


def _candidate_public_order(candidate: Candidate) -> tuple[object, ...]:
    return (
        SPLIT_ORDER.index(candidate.split),
        FAMILY_ORDER.index(candidate.family),
        candidate.table_uid.encode("utf-8"),
        candidate.question_uid.encode("utf-8"),
        candidate.source_context_ordinal,
        candidate.source_question_ordinal,
    )


def deterministic_augmenting_match(
    slots: Sequence[Slot],
    candidates: Sequence[Candidate],
    *,
    order_key: Callable[[Candidate], object],
) -> dict[int, Candidate]:
    """Return a deterministic maximum matching from fixed slots to contexts.

    A right-side node is an entire table context, not a question.  Therefore a
    context containing several eligible questions can still contribute at most
    one selected question.  Recursive augmenting paths make the result complete
    whenever a complete matching exists; a greedy first-fit is not sufficient
    when different families compete for the same context.
    """

    slot_rows = tuple(slots)
    candidate_rows = tuple(candidates)
    if len({slot.ordinal for slot in slot_rows}) != len(slot_rows):
        raise TatqaP19AcquisitionError("matching slot ordinal is duplicated")
    identities = {(row.split, row.question_uid) for row in candidate_rows}
    if len(identities) != len(candidate_rows):
        raise TatqaP19AcquisitionError("matching candidate question identity is duplicated")
    by_slot: dict[int, tuple[Candidate, ...]] = {}
    for slot in slot_rows:
        eligible = [
            row
            for row in candidate_rows
            if row.split == slot.split and row.family == slot.family
        ]
        eligible.sort(key=lambda row: (order_key(row), _candidate_public_order(row)))
        by_slot[slot.ordinal] = tuple(eligible)

    assignment: dict[int, Candidate] = {}
    context_owner: dict[str, int] = {}

    def place(slot_ordinal: int, seen_contexts: set[str]) -> bool:
        for candidate in by_slot[slot_ordinal]:
            context = candidate.context_key
            if context in seen_contexts:
                continue
            seen_contexts.add(context)
            incumbent = context_owner.get(context)
            if incumbent is None or place(incumbent, seen_contexts):
                context_owner[context] = slot_ordinal
                assignment[slot_ordinal] = candidate
                return True
        return False

    # Insert in reverse slot priority.  The standard augmenting-path update
    # grants the newly inserted slot first choice, so reversing here makes the
    # frozen forward block/family/slot order lexicographically prior: A_form
    # receives the earliest feasible private HMAC members before F/A_hold.
    for slot in reversed(slot_rows):
        if not place(slot.ordinal, set()):
            raise TatqaP19AcquisitionError(
                "fixed block/family context matching capacity is unavailable"
            )
    if len(assignment) != len(slot_rows) or len(context_owner) != len(slot_rows):
        raise TatqaP19AcquisitionError("augmenting matching completeness drifted")
    return assignment


def _frame_bytes(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) >= 2**32:
        raise TatqaP19AcquisitionError("HMAC frame is invalid")
    return len(value).to_bytes(4, "big") + value


def selection_hmac_message(candidate: Candidate) -> bytes:
    """Exact frozen bytes for split/family/table-UID/question-UID ordering."""

    fields = (
        candidate.split.encode("utf-8"),
        candidate.family.encode("ascii"),
        candidate.table_uid.encode("utf-8"),
        candidate.question_uid.encode("utf-8"),
    )
    return _SELECT_HMAC_DOMAIN + b"".join(_frame_bytes(value) for value in fields)


def _shuffle_hmac_message(block: str, candidate: Candidate) -> bytes:
    if block not in BLOCK_ORDER:
        raise TatqaP19AcquisitionError("shuffle block is invalid")
    # Family is intentionally absent: presentation order cannot be a family
    # sort, even though the trusted labels can later recover family by ordinal.
    fields = (
        block.encode("ascii"),
        candidate.table_uid.encode("utf-8"),
        candidate.question_uid.encode("utf-8"),
    )
    return _SHUFFLE_HMAC_DOMAIN + b"".join(_frame_bytes(value) for value in fields)


def _require_secret(secret: bytes) -> None:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise TatqaP19AcquisitionError("selection secret must be exactly 32 bytes")


def _selection_digest(secret: bytes, candidate: Candidate) -> bytes:
    _require_secret(secret)
    return hmac.new(secret, selection_hmac_message(candidate), hashlib.sha256).digest()


def _shuffle_digest(secret: bytes, block: str, candidate: Candidate) -> bytes:
    _require_secret(secret)
    return hmac.new(
        secret, _shuffle_hmac_message(block, candidate), hashlib.sha256
    ).digest()


def selection_secret_commitment(secret: bytes) -> str:
    _require_secret(secret)
    return hashlib.sha256(_SECRET_COMMITMENT_DOMAIN + secret).hexdigest()


def select_blocks(
    candidates: Sequence[Candidate], *, secret: bytes
) -> dict[str, tuple[Candidate, ...]]:
    """Select all 144 fixed slots, then independently blind each block order."""

    _require_secret(secret)
    assignment = deterministic_augmenting_match(
        _slots(), candidates, order_key=lambda row: _selection_digest(secret, row)
    )
    result: dict[str, tuple[Candidate, ...]] = {}
    for block in BLOCK_ORDER:
        rows = [
            assignment[slot.ordinal]
            for slot in _slots()
            if slot.block == block
        ]
        rows.sort(
            key=lambda row: (
                _shuffle_digest(secret, block, row),
                row.table_uid.encode("utf-8"),
                row.question_uid.encode("utf-8"),
            )
        )
        result[block] = tuple(rows)
    _verify_selected_blocks(result)
    return result


def _verify_selected_blocks(selected: Mapping[str, Sequence[Candidate]]) -> None:
    if set(selected) != set(BLOCK_ORDER):
        raise TatqaP19AcquisitionError("selected block set drifted")
    all_rows: list[Candidate] = []
    for block in BLOCK_ORDER:
        rows = tuple(selected[block])
        if len(rows) != BLOCK_COUNTS[block]:
            raise TatqaP19AcquisitionError("selected block count drifted")
        if any(row.split != BLOCK_SPLIT[block] for row in rows):
            raise TatqaP19AcquisitionError("selected block split drifted")
        observed = Counter(row.family for row in rows)
        if observed != Counter(
            {family: PER_FAMILY_QUOTA[block] for family in FAMILY_ORDER}
        ):
            raise TatqaP19AcquisitionError("selected block family quota drifted")
        all_rows.extend(rows)
    if (
        len(all_rows) != TOTAL_SELECTED_ITEMS
        or len({row.context_key for row in all_rows}) != TOTAL_SELECTED_ITEMS
        or len({(row.split, row.question_uid) for row in all_rows})
        != TOTAL_SELECTED_ITEMS
    ):
        raise TatqaP19AcquisitionError("one-context-one-question selection drifted")


def qualify_decoded_sources(
    *, raw_by_split: Mapping[str, object], tagop_by_split: Mapping[str, object]
) -> Qualification:
    """Perform aggregate-only qualification and a secret-free capacity proof."""

    if set(raw_by_split) != set(SPLIT_ORDER) or set(tagop_by_split) != set(SPLIT_ORDER):
        raise TatqaP19AcquisitionError("formal source split registry drifted")
    candidates: list[Candidate] = []
    split_aggregates: dict[str, Any] = {}
    table_split: dict[str, str] = {}
    question_split: dict[str, str] = {}
    for split in SPLIT_ORDER:
        rows, aggregate = parse_source_pair(
            raw_by_split[split], tagop_by_split[split], split=split
        )
        # Include public-example and mapping-ineligible rows in the global UID
        # audit.  Eligibility must not be able to hide cross-split collisions.
        for context_value in _require_list(
            raw_by_split[split], field=f"raw {split} root"
        ):
            context = _require_mapping(context_value, field="raw context")
            table = _require_mapping(context.get("table"), field="raw table")
            table_uid = _strict_source_id(table.get("uid"), field="table UID")
            prior_split = table_split.setdefault(table_uid, split)
            if prior_split != split:
                raise TatqaP19AcquisitionError("table UID is duplicated across splits")
            for question_value in _require_list(
                context.get("questions"), field="raw questions"
            ):
                question = _require_mapping(question_value, field="raw question")
                question_uid = _strict_source_id(
                    question.get("uid"), field="question UID"
                )
                prior_question_split = question_split.setdefault(question_uid, split)
                if prior_question_split != split:
                    raise TatqaP19AcquisitionError(
                        "question UID is duplicated across splits"
                    )
        candidates.extend(rows)
        split_aggregates[split] = aggregate

    # This source-order matching publishes nothing.  It proves that the frozen
    # quotas are structurally feasible before the private secret is created.
    deterministic_augmenting_match(
        _slots(), candidates, order_key=_candidate_public_order
    )
    counts = Counter((row.split, row.family) for row in candidates)
    contexts = defaultdict(set)
    for row in candidates:
        contexts[(row.split, row.family)].add(row.context_key)
    aggregate_body = {
        "schema": f"{VERSION}_aggregate_qualification",
        "status": "qualified_for_one_shot_acquisition",
        "splits": split_aggregates,
        "eligible_question_count_by_split_and_family": {
            split: {family: counts[(split, family)] for family in FAMILY_ORDER}
            for split in SPLIT_ORDER
        },
        "eligible_context_count_by_split_and_family": {
            split: {
                family: len(contexts[(split, family)]) for family in FAMILY_ORDER
            }
            for split in SPLIT_ORDER
        },
        "fixed_block_counts": dict(BLOCK_COUNTS),
        "fixed_per_family_quota": dict(PER_FAMILY_QUOTA),
        "total_selected_item_count": TOTAL_SELECTED_ITEMS,
        "one_context_one_question_capacity_proven": True,
        "source_or_item_identifiers_included": False,
    }
    public_aggregate = self_hashed(
        aggregate_body, "aggregate_qualification_sha256"
    )
    _assert_aggregate_has_no_identifiers(public_aggregate)
    return Qualification(tuple(candidates), public_aggregate)


def _assert_aggregate_has_no_identifiers(value: Mapping[str, Any]) -> None:
    forbidden = {
        "uid",
        "id",
        "table_uid",
        "question_uid",
        "question",
        "answer",
        "mapping",
        "gold_unit_ids",
        "items",
    }

    def visit(row: object) -> None:
        if isinstance(row, Mapping):
            for key, nested in row.items():
                if key in forbidden:
                    raise TatqaP19AcquisitionError(
                        "aggregate qualification contains a source identifier or row"
                    )
                visit(nested)
        elif isinstance(row, list):
            for nested in row:
                visit(nested)

    visit(value)


def item_commitment(
    *, block: str, ordinal: int, question: str, units: Sequence[CanonicalUnit]
) -> str:
    if (
        block not in BLOCK_ORDER
        or type(ordinal) is not int
        or not 0 <= ordinal < BLOCK_COUNTS[block]
    ):
        raise TatqaP19AcquisitionError("item commitment coordinate is invalid")
    return stable_hash(
        {
            "version": VERSION,
            "block": block,
            "ordinal": ordinal,
            "question": question,
            "units": [unit.payload() for unit in units],
        }
    )


_VIEW_FORBIDDEN_KEYS = frozenset(
    {
        "family",
        "answer_from",
        "answer",
        "answer_type",
        "derivation",
        "scale",
        "mapping",
        "source_mapping",
        "gold",
        "gold_unit_ids",
        "split",
        "table_uid",
        "question_uid",
        "source_context_ordinal",
        "source_question_ordinal",
    }
)


def assert_view_firewall(view: Mapping[str, Any]) -> None:
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in _VIEW_FORBIDDEN_KEYS:
                    raise TatqaP19AcquisitionError(
                        f"view contains forbidden late/source key {key}"
                    )
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(view)


def materialize_private_payloads(
    *,
    selected: Mapping[str, Sequence[Candidate]],
    selection_secret_commitment_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build four views, exactly three late-label packs, and one ledger."""

    _verify_selected_blocks(selected)
    if _HEX64.fullmatch(selection_secret_commitment_sha256) is None:
        raise TatqaP19AcquisitionError("selection-secret commitment is malformed")
    views: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    ledger_items: list[dict[str, Any]] = []
    for block in BLOCK_ORDER:
        rows = tuple(selected[block])
        commitments = [
            item_commitment(
                block=block,
                ordinal=ordinal,
                question=row.question,
                units=row.units,
            )
            for ordinal, row in enumerate(rows)
        ]
        view_body = {
            "schema": f"{VERSION}_block_view",
            "version": VERSION,
            "block": block,
            "access_state": (
                "presealed_until_valid_A_hold_promotion"
                if block == "M_search"
                else "available_only_at_frozen_lifecycle_stage"
            ),
            "item_count": len(rows),
            "late_fields_included": False,
            "items": [
                {
                    "ordinal": ordinal,
                    "item_commitment_sha256": commitment,
                    "question": row.question,
                    "canonical_units": [unit.payload() for unit in row.units],
                }
                for ordinal, (row, commitment) in enumerate(
                    zip(rows, commitments, strict=True)
                )
            ],
        }
        view = self_hashed(view_body, "block_view_sha256")
        assert_view_firewall(view)
        views[block] = view

        if block != "F_search":
            label_body = {
                "schema": f"{VERSION}_sealed_labels",
                "version": VERSION,
                "block": block,
                "access_state": (
                    "presealed_until_valid_A_hold_promotion"
                    if block == "M_search"
                    else "sealed_until_corresponding_actions_and_postflight"
                ),
                "item_count": len(rows),
                "block_view_sha256": view["block_view_sha256"],
                "items": [
                    {
                        "ordinal": ordinal,
                        "item_commitment_sha256": commitment,
                        "family": row.family,
                        "gold_unit_ids": list(row.gold_unit_ids),
                    }
                    for ordinal, (row, commitment) in enumerate(
                        zip(rows, commitments, strict=True)
                    )
                ],
            }
            labels[block] = self_hashed(label_body, "label_pack_sha256")

        for ordinal, (row, commitment) in enumerate(
            zip(rows, commitments, strict=True)
        ):
            ledger_items.append(
                {
                    "block": block,
                    "ordinal": ordinal,
                    "split": row.split,
                    "table_uid": row.table_uid,
                    "question_uid": row.question_uid,
                    "source_context_ordinal": row.source_context_ordinal,
                    "source_question_ordinal": row.source_question_ordinal,
                    "family": row.family,
                    "item_commitment_sha256": commitment,
                }
            )

    if set(views) != set(BLOCK_ORDER) or set(labels) != {
        "A_form",
        "A_hold",
        "M_search",
    }:
        raise TatqaP19AcquisitionError("view/late-label pack set drifted")
    ledger_body = {
        "schema": f"{VERSION}_private_ledger",
        "version": VERSION,
        "selection_secret_commitment_sha256": selection_secret_commitment_sha256,
        "item_count": len(ledger_items),
        "one_context_one_question": True,
        "items": ledger_items,
    }
    ledger = self_hashed(ledger_body, "ledger_sha256")
    return views, labels, ledger


def _safe_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TatqaP19AcquisitionError(f"{field} is not a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TatqaP19AcquisitionError(f"{field} is unsafe")
    return path.as_posix()


def _read_regular(project: Path, relative: str, *, field: str) -> bytes:
    safe = _safe_relative(relative, field=field)
    current = project
    parts = PurePosixPath(safe).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise TatqaP19AcquisitionError(f"{field} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TatqaP19AcquisitionError(f"{field} contains a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise TatqaP19AcquisitionError(f"{field} ancestor is not a directory")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise TatqaP19AcquisitionError(f"{field} is not a regular file")
    try:
        return current.read_bytes()
    except OSError as exc:
        raise TatqaP19AcquisitionError(f"{field} is unreadable") from exc


def _load_json_object(project: Path, relative: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(project, relative.as_posix(), field=field)
    value = strict_json_loads(raw, label=field)
    if not isinstance(value, dict):
        raise TatqaP19AcquisitionError(f"{field} root is not an object")
    return value, raw


def _verify_contracts(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    design, design_raw = _load_json_object(project, DESIGN_RELATIVE, field="P19 design")
    custody, custody_raw = _load_json_object(project, CUSTODY_RELATIVE, field="P19 custody")
    if (
        hashlib.sha256(design_raw).hexdigest() != DESIGN_FILE_SHA256
        or verify_self_hash(design) != DESIGN_SELF_SHA256
        or design.get("schema") != "tatqa_p19_typed_evaluator_study_design_v1"
    ):
        raise TatqaP19AcquisitionError("P19 design binding drifted")
    if (
        hashlib.sha256(custody_raw).hexdigest() != CUSTODY_FILE_SHA256
        or verify_self_hash(custody) != CUSTODY_SELF_SHA256
        or custody.get("schema") != "tatqa_p19_public_source_custody_v1"
    ):
        raise TatqaP19AcquisitionError("P19 custody binding drifted")
    source = design.get("source_binding")
    if (
        not isinstance(source, Mapping)
        or source.get("custody_self_sha256") != CUSTODY_SELF_SHA256
        or source.get("source_commit") != SOURCE_COMMIT
    ):
        raise TatqaP19AcquisitionError("P19 design/custody chain drifted")
    evidence = design.get("canonical_evidence_contract")
    qualification = (
        evidence.get("qualification") if isinstance(evidence, Mapping) else None
    )
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("minimum_canonical_unit_count")
        != MIN_CANONICAL_UNITS
        or qualification.get("maximum_canonical_unit_count")
        != MAX_CANONICAL_UNITS
    ):
        raise TatqaP19AcquisitionError("canonical-unit qualification binding drifted")
    exclusion = custody.get("public_example_exclusion")
    if (
        not isinstance(exclusion, Mapping)
        or frozenset(exclusion.get("question_uids", ())) != PUBLIC_QUESTION_UIDS
        or frozenset(exclusion.get("table_uids", ())) != PUBLIC_TABLE_UIDS
        or exclusion.get("whole_context_excluded_when_any_public_UID_matches") is not True
    ):
        raise TatqaP19AcquisitionError("public-example exclusion binding drifted")
    return design, custody


def _git_is_ancestor(project: Path, commit: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=project,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _git_commit_is_ancestor_of(
    project: Path, ancestor: str, descendant: str
) -> bool:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=project,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _git_committed_path(project: Path, commit: str, relative: str) -> bytes | None:
    try:
        prefix_process = subprocess.run(
            ["git", "rev-parse", "--show-prefix"],
            cwd=project,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if prefix_process.returncode != 0:
            return None
        prefix = prefix_process.stdout.decode("utf-8").strip()
        completed = subprocess.run(
            ["git", "show", f"{commit}:{prefix}{relative}"],
            cwd=project,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, UnicodeDecodeError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _git_paths_are_committed_and_clean(project: Path, relatives: Sequence[str]) -> bool:
    try:
        for relative in relatives:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=project,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if tracked.returncode != 0:
                return False
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *relatives,
            ],
            cwd=project,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return status.returncode == 0 and not status.stdout


def _verify_freeze(project: Path) -> dict[str, Any]:
    value, _raw = _load_json_object(
        project, IMPLEMENTATION_FREEZE_RELATIVE, field="P19 implementation freeze"
    )
    if (
        value.get("schema") != IMPLEMENTATION_FREEZE_SCHEMA
        or value.get("status") != "implementation_frozen"
        or value.get("study_design_self_sha256") != DESIGN_SELF_SHA256
        or value.get("source_custody_self_sha256") != CUSTODY_SELF_SHA256
        or value.get("implementation_binding_registry_is_exact") is not True
        or value.get("runtime_and_canary_committed_and_clean") is not True
        or value.get("runtime_qualification_terminal_committed_and_clean")
        is not True
        or value.get("implementation_bytes_unchanged_since_runtime_qualification")
        is not True
        or value.get("formal_source_opened") is not False
        or value.get("formal_source_download_receipt_present") is not False
        or value.get("formal_acquisition_root_present") is not False
        or value.get("external_network_calls_by_freeze_builder") != 0
        or value.get("api_or_online_evaluator_calls_by_freeze_builder") != 0
        or value.get("retry_replay_resample_provider_switch") != 0
    ):
        raise TatqaP19AcquisitionError("P19 implementation freeze contract drifted")
    verify_self_hash(value)
    commit = value.get("formal_implementation_commit")
    if (
        not isinstance(commit, str)
        or _HEX40.fullmatch(commit) is None
        or not _git_is_ancestor(project, commit)
    ):
        raise TatqaP19AcquisitionError("P19 formal implementation commit is not committed")
    qualification_commit = value.get("runtime_qualification_implementation_commit")
    if (
        not isinstance(qualification_commit, str)
        or _HEX40.fullmatch(qualification_commit) is None
        or not _git_is_ancestor(project, qualification_commit)
    ):
        raise TatqaP19AcquisitionError(
            "P19 runtime qualification implementation commit is not committed"
        )
    bindings = value.get("implementation_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise TatqaP19AcquisitionError("P19 implementation binding registry is absent")
    observed_paths: list[str] = []
    for row_value in bindings:
        row = _require_mapping(row_value, field="implementation binding")
        if set(row) != {"relative_path", "sha256"}:
            raise TatqaP19AcquisitionError("implementation binding schema drifted")
        relative = _safe_relative(row["relative_path"], field="implementation path")
        expected = row["sha256"]
        if not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
            raise TatqaP19AcquisitionError("implementation binding hash is malformed")
        raw = _read_regular(project, relative, field="frozen implementation member")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise TatqaP19AcquisitionError("frozen implementation member drifted")
        observed_paths.append(relative)
    if (
        observed_paths != sorted(set(observed_paths))
        or frozenset(observed_paths) != REQUIRED_IMPLEMENTATION_PATHS
        or value.get("formal_implementation_tree_sha256")
        != stable_hash(bindings)
    ):
        raise TatqaP19AcquisitionError("implementation binding path registry drifted")
    if not _git_commit_is_ancestor_of(project, qualification_commit, commit):
        raise TatqaP19AcquisitionError(
            "runtime qualification commit is not an ancestor of formal implementation"
        )
    for row in bindings:
        qualified = _git_committed_path(
            project, qualification_commit, str(row["relative_path"])
        )
        if (
            qualified is None
            or hashlib.sha256(qualified).hexdigest() != row["sha256"]
        ):
            raise TatqaP19AcquisitionError(
                "implementation changed after runtime qualification"
            )

    evidence_values: dict[str, dict[str, Any]] = {}
    evidence_paths: list[str] = []
    for field, schema in (
        (
            "runtime_fingerprint_binding",
            "tatqa_p19_composite_runtime_fingerprint_v1",
        ),
        (
            "production_canary_binding",
            "tatqa_p19_public_synthetic_production_canary_v1",
        ),
        (
            "runtime_qualification_terminal_binding",
            "tatqa_p19_runtime_qualification_v1_terminal_success_v1",
        ),
    ):
        binding = _require_mapping(value.get(field), field=field)
        if set(binding) != {"relative_path", "file_sha256", "self_sha256"}:
            raise TatqaP19AcquisitionError("freeze evidence binding schema drifted")
        relative = _safe_relative(binding["relative_path"], field=field)
        expected_file = binding.get("file_sha256")
        expected_self = binding.get("self_sha256")
        if (
            not isinstance(expected_file, str)
            or _HEX64.fullmatch(expected_file) is None
            or not isinstance(expected_self, str)
            or _HEX64.fullmatch(expected_self) is None
        ):
            raise TatqaP19AcquisitionError("freeze evidence hash drifted")
        raw = _read_regular(project, relative, field=field)
        if hashlib.sha256(raw).hexdigest() != expected_file:
            raise TatqaP19AcquisitionError("freeze evidence file drifted")
        parsed = strict_json_loads(raw, label=field)
        if not isinstance(parsed, dict) or parsed.get("schema") != schema:
            raise TatqaP19AcquisitionError("freeze evidence schema drifted")
        if verify_self_hash(parsed) != expected_self:
            raise TatqaP19AcquisitionError("freeze evidence self hash drifted")
        evidence_values[field] = parsed
        evidence_paths.append(relative)
    fingerprint = evidence_values["runtime_fingerprint_binding"]
    canary = evidence_values["production_canary_binding"]
    qualification_terminal = evidence_values[
        "runtime_qualification_terminal_binding"
    ]
    _fingerprint_self, fingerprint_subfingerprint_hashes = (
        _runtime_subfingerprint_self_hashes(fingerprint)
    )
    if canary.get(
        RUNTIME_SUBFINGERPRINT_HASHES_FIELD
    ) != fingerprint_subfingerprint_hashes:
        raise TatqaP19AcquisitionError(
            "freeze evidence runtime subfingerprint cross-binding drifted"
        )
    if (
        fingerprint.get("status") != "verified_before_formal_source_open"
        or fingerprint.get("study_design_self_sha256") != DESIGN_SELF_SHA256
        or fingerprint.get("formal_source_opened") is not False
        or fingerprint.get("external_network_calls") != 0
        or fingerprint.get("api_or_online_evaluator_calls") != 0
        or canary.get("status") != "qualified_before_formal_source_open"
        or canary.get("qualified") is not True
        or canary.get("runtime_fingerprint_self_sha256")
        != fingerprint.get("self_sha256")
        or canary.get("formal_source_opened") is not False
        or canary.get("public_synthetic_distinct_rankings") is not True
        or canary.get("hippo_canary_ran") is not True
        or canary.get("P1_retains_ordered_P0_top3") is not True
        or isinstance(canary.get("P1_outside_P0_unit_count"), bool)
        or not isinstance(canary.get("P1_outside_P0_unit_count"), int)
        or canary["P1_outside_P0_unit_count"] < 1
        or canary.get("typed_plan_worker_receipt_source")
        != "capability_receipt_snapshot"
        or canary.get("minilm_worker_receipt_source")
        != "capability_receipt_snapshot"
        or canary.get("hippo_worker_receipt_source")
        != "capability_receipt_snapshot"
        or canary.get("external_network_calls") != 0
        or canary.get("api_or_online_evaluator_calls") != 0
        or fingerprint.get("runtime_implementation_commit")
        != qualification_commit
        or qualification_terminal.get("status")
        != "qualified_before_formal_source_open"
        or qualification_terminal.get("runtime_fingerprint_self_sha256")
        != fingerprint.get("self_sha256")
        or qualification_terminal.get("production_canary_self_sha256")
        != canary.get("self_sha256")
        or qualification_terminal.get("formal_source_opened") is not False
        or qualification_terminal.get("retry_requalification") != 0
    ):
        raise TatqaP19AcquisitionError("freeze evidence qualification drifted")
    validate_production_canary_capability_receipts(
        canary,
        runtime_fingerprint=fingerprint,
    )
    committed_paths = [
        IMPLEMENTATION_FREEZE_RELATIVE.as_posix(),
        *evidence_paths,
        *observed_paths,
    ]
    if not _git_paths_are_committed_and_clean(project, committed_paths):
        raise TatqaP19AcquisitionError(
            "freeze, evidence, or implementation is not committed and clean"
        )
    return value


def verify_implementation_freeze(
    project_root: str | Path,
    *,
    runtime_fingerprint_path: str | Path,
    production_canary_path: str | Path,
) -> dict[str, Any]:
    """Verify the committed freeze and its exact configured evidence paths.

    This public metadata-only wrapper is used by formal entry before any model
    initialization.  It deliberately cannot open the formal source payload.
    """

    try:
        project = Path(project_root).resolve(strict=True)
    except OSError as exc:
        raise TatqaP19AcquisitionError("P19 project root is unavailable") from exc
    if not project.is_dir():
        raise TatqaP19AcquisitionError("P19 project root is not a directory")
    freeze = _verify_freeze(project)
    for field, configured in (
        ("runtime_fingerprint_binding", runtime_fingerprint_path),
        ("production_canary_binding", production_canary_path),
    ):
        binding = _require_mapping(freeze.get(field), field=field)
        relative = _safe_relative(binding.get("relative_path"), field=field)
        expected = project / relative
        observed = Path(configured).expanduser().absolute()
        if observed != expected:
            raise TatqaP19AcquisitionError(
                "formal evidence path differs from the implementation freeze"
            )
    return freeze


def _verify_source_download_receipt(project: Path) -> dict[str, Any]:
    value, _raw = _load_json_object(
        project, SOURCE_RECEIPT_RELATIVE, field="P19 source download receipt"
    )
    if (
        value.get("schema") != SOURCE_RECEIPT_SCHEMA
        or value.get("status") != "source_download_complete_unopened_by_acquisition"
        or value.get("source_commit") != SOURCE_COMMIT
        or value.get("source_custody_self_sha256") != CUSTODY_SELF_SHA256
        or value.get("source_root_relative") != SOURCE_ROOT_RELATIVE.as_posix()
        or not isinstance(value.get("implementation_freeze_self_sha256"), str)
        or _HEX64.fullmatch(value["implementation_freeze_self_sha256"]) is None
    ):
        raise TatqaP19AcquisitionError("P19 source download receipt contract drifted")
    verify_self_hash(value)
    files = value.get("files")
    if not isinstance(files, list):
        raise TatqaP19AcquisitionError("P19 source file registry is absent")
    observed: list[str] = []
    for row_value in files:
        row = _require_mapping(row_value, field="source file binding")
        if set(row) != {"relative_path", "sha256", "size_bytes"}:
            raise TatqaP19AcquisitionError("source file binding schema drifted")
        relative = _safe_relative(row["relative_path"], field="source file path")
        if (
            not isinstance(row["sha256"], str)
            or _HEX64.fullmatch(row["sha256"]) is None
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
        ):
            raise TatqaP19AcquisitionError("source file binding value is malformed")
        observed.append(relative)
    if tuple(observed) != SOURCE_FILES:
        raise TatqaP19AcquisitionError("source file registry drifted")
    return value


def _read_and_verify_source_files(
    project: Path, receipt: Mapping[str, Any]
) -> dict[str, bytes]:
    files = receipt.get("files")
    if not isinstance(files, list):
        raise TatqaP19AcquisitionError("source file registry is absent")
    result: dict[str, bytes] = {}
    root = SOURCE_ROOT_RELATIVE.as_posix()
    for row_value in files:
        row = _require_mapping(row_value, field="source file binding")
        relative = _safe_relative(row.get("relative_path"), field="source file path")
        raw = _read_regular(
            project, f"{root}/{relative}", field=f"official source file {relative}"
        )
        if (
            len(raw) != row.get("size_bytes")
            or hashlib.sha256(raw).hexdigest() != row.get("sha256")
        ):
            raise TatqaP19AcquisitionError("official source file identity drifted")
        result[relative] = raw
    if tuple(result) != SOURCE_FILES:
        raise TatqaP19AcquisitionError("official source file set drifted")
    return result


def _write_exclusive(path: Path, raw: bytes, *, mode: int = 0o600) -> tuple[str, int]:
    if not isinstance(raw, bytes):
        raise TatqaP19AcquisitionError("exclusive output is not bytes")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except OSError as exc:
        raise TatqaP19AcquisitionError("exclusive output already exists or is unavailable") from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
                raise TatqaP19AcquisitionError("exclusive output mode drifted")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, mode: int = 0o600
) -> tuple[str, int]:
    return _write_exclusive(path, _canonical_bytes(payload, newline=True), mode=mode)


def _write_secret(path: Path, secret: bytes) -> tuple[str, int]:
    _require_secret(secret)
    return _write_exclusive(path, secret, mode=0o600)


def _terminal_failure(root: Path, *, stage: str, exc: BaseException) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure",
        "version": VERSION,
        "status": "terminal_no_retry_replay_resample_or_smaller_blocks",
        "failure_stage": stage,
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "source_or_item_identifier_included": False,
        "online_or_network_call_count": 0,
    }
    try:
        _write_json_exclusive(
            root / FAILURE_FILENAME,
            self_hashed(body, "failure_sha256"),
            mode=0o600,
        )
    except BaseException:
        pass


def _private_binding(
    *, filename: str, file_sha256: str, semantic_sha256: str, size_bytes: int
) -> dict[str, Any]:
    return {
        "filename": filename,
        "file_sha256": file_sha256,
        "semantic_sha256": semantic_sha256,
        "size_bytes": size_bytes,
        "mode": "0600",
    }


def _assert_public_receipt_safe(value: Mapping[str, Any]) -> None:
    _assert_aggregate_has_no_identifiers(value.get("aggregate_qualification", {}))
    forbidden = {
        "items",
        "table_uid",
        "question_uid",
        "question",
        "answer",
        "mapping",
        "gold_unit_ids",
        "selection_secret",
        "source_context_ordinal",
        "source_question_ordinal",
    }

    def visit(row: object) -> None:
        if isinstance(row, Mapping):
            for key, nested in row.items():
                if key in forbidden:
                    raise TatqaP19AcquisitionError(
                        "public receipt contains a private row field"
                    )
                visit(nested)
        elif isinstance(row, list):
            for nested in row:
                visit(nested)

    visit(value)


def run_trusted_acquisition(project_root: str | Path) -> dict[str, Any]:
    """Run the sole formal local TAT-QA qualification/acquisition attempt."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise TatqaP19AcquisitionError("project root is invalid")
    acquisition_root = project / ACQUISITION_ROOT_RELATIVE
    if acquisition_root.exists() or acquisition_root.is_symlink():
        raise TatqaP19OneShotRefusal("formal acquisition root is already consumed")

    # Metadata-only preconditions.  Missing freeze/receipt fails before the
    # attempt root exists and, critically, before any source payload is opened.
    _design, _custody = _verify_contracts(project)
    freeze = _verify_freeze(project)
    source_receipt = _verify_source_download_receipt(project)
    if (
        source_receipt.get("implementation_freeze_self_sha256")
        != freeze.get("self_sha256")
    ):
        raise TatqaP19AcquisitionError(
            "source download is not bound to the current implementation freeze"
        )

    formal_root = project / FORMAL_ROOT_RELATIVE
    formal_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        formal_root.mkdir(mode=0o700)
    except FileExistsError:
        metadata = formal_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise TatqaP19OneShotRefusal("formal root is unsafe")
    try:
        acquisition_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise TatqaP19OneShotRefusal(
            "formal acquisition root is already consumed"
        ) from exc

    marker_body = {
        "schema": f"{VERSION}_one_shot_marker",
        "version": VERSION,
        "status": "formal_attempt_started_before_source_payload_open",
        "study_design_self_sha256": DESIGN_SELF_SHA256,
        "source_custody_self_sha256": CUSTODY_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "source_download_receipt_self_sha256": source_receipt["self_sha256"],
        "retry_replay_or_resample": 0,
    }
    _write_json_exclusive(
        acquisition_root / MARKER_FILENAME,
        self_hashed(marker_body, "marker_sha256"),
    )

    try:
        source_bytes = _read_and_verify_source_files(project, source_receipt)
        raw_by_split = {
            "train": strict_json_loads(
                source_bytes["dataset_raw/tatqa_dataset_train.json"],
                label="official raw train",
            ),
            "dev": strict_json_loads(
                source_bytes["dataset_raw/tatqa_dataset_dev.json"],
                label="official raw dev",
            ),
        }
        tagop_by_split = {
            "train": strict_json_loads(
                source_bytes["dataset_tagop/tatqa_dataset_train.json"],
                label="official TagOp train",
            ),
            "dev": strict_json_loads(
                source_bytes["dataset_tagop/tatqa_dataset_dev.json"],
                label="official TagOp dev",
            ),
        }
        qualification = qualify_decoded_sources(
            raw_by_split=raw_by_split, tagop_by_split=tagop_by_split
        )
        secret = os.urandom(32)
        _require_secret(secret)
        _secret_file_sha256, secret_size = _write_secret(
            acquisition_root / SECRET_FILENAME, secret
        )
        secret_commitment = selection_secret_commitment(secret)
        selected = select_blocks(qualification.candidates, secret=secret)
        views, labels, ledger = materialize_private_payloads(
            selected=selected,
            selection_secret_commitment_sha256=secret_commitment,
        )

        private_bindings: dict[str, Any] = {}
        for block in BLOCK_ORDER:
            filename = VIEW_FILENAMES[block]
            payload = views[block]
            file_sha, size = _write_json_exclusive(
                acquisition_root / filename, payload
            )
            private_bindings[filename] = _private_binding(
                filename=filename,
                file_sha256=file_sha,
                semantic_sha256=payload["block_view_sha256"],
                size_bytes=size,
            )
        for block in ("A_form", "A_hold", "M_search"):
            filename = LABEL_FILENAMES[block]
            payload = labels[block]
            file_sha, size = _write_json_exclusive(
                acquisition_root / filename, payload
            )
            private_bindings[filename] = _private_binding(
                filename=filename,
                file_sha256=file_sha,
                semantic_sha256=payload["label_pack_sha256"],
                size_bytes=size,
            )
        ledger_file_sha, ledger_size = _write_json_exclusive(
            acquisition_root / LEDGER_FILENAME, ledger
        )
        private_bindings[LEDGER_FILENAME] = _private_binding(
            filename=LEDGER_FILENAME,
            file_sha256=ledger_file_sha,
            semantic_sha256=ledger["ledger_sha256"],
            size_bytes=ledger_size,
        )

        receipt_body = {
            "schema": f"{VERSION}_public_receipt",
            "version": VERSION,
            "status": "trusted_one_shot_acquisition_complete",
            "study_design_self_sha256": DESIGN_SELF_SHA256,
            "source_custody_self_sha256": CUSTODY_SELF_SHA256,
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "source_download_receipt_self_sha256": source_receipt["self_sha256"],
            "aggregate_qualification": dict(qualification.public_aggregate),
            "selection_secret_commitment_sha256": secret_commitment,
            "selection_secret_size_bytes": secret_size,
            "selection_secret_persisted_publicly": False,
            "fixed_block_counts": dict(BLOCK_COUNTS),
            "fixed_per_family_quota": dict(PER_FAMILY_QUOTA),
            "selected_context_count": TOTAL_SELECTED_ITEMS,
            "selected_question_count": TOTAL_SELECTED_ITEMS,
            "private_file_bindings": dict(sorted(private_bindings.items())),
            "view_file_count": 4,
            "label_file_count": 3,
            "ledger_file_count": 1,
            "F_search_label_pack_created": False,
            "M_search_view_and_labels_presealed": True,
            "source_item_or_identifier_persisted_publicly": False,
            "network_download_online_evaluator_or_model_calls": 0,
            "retry_replay_resample_or_smaller_blocks": 0,
        }
        receipt = self_hashed(receipt_body, "acquisition_receipt_sha256")
        _assert_public_receipt_safe(receipt)
        _write_json_exclusive(
            acquisition_root / PUBLIC_RECEIPT_FILENAME, receipt
        )
        acquisition_root.chmod(0o500)
        metadata = acquisition_root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o500:
            raise TatqaP19AcquisitionError("completed acquisition root seal drifted")
        return receipt
    except BaseException as exc:
        _terminal_failure(acquisition_root, stage="source_qualification_or_acquisition", exc=exc)
        raise TatqaP19AcquisitionError(
            "formal source qualification/acquisition failed terminally"
        ) from exc


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_trusted_acquisition(arguments.project)
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACQUISITION_ROOT_RELATIVE",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "Candidate",
    "CanonicalUnit",
    "FAMILY_ORDER",
    "LABEL_FILENAMES",
    "PER_FAMILY_QUOTA",
    "PUBLIC_QUESTION_UIDS",
    "PUBLIC_TABLE_UIDS",
    "Qualification",
    "Slot",
    "TatqaP19AcquisitionError",
    "TatqaP19OneShotRefusal",
    "TOTAL_SELECTED_ITEMS",
    "VERSION",
    "VIEW_FILENAMES",
    "assert_view_firewall",
    "deterministic_augmenting_match",
    "materialize_private_payloads",
    "parse_source_pair",
    "project_gold_mapping",
    "qualify_decoded_sources",
    "run_trusted_acquisition",
    "select_blocks",
    "selection_hmac_message",
    "selection_secret_commitment",
    "self_hashed",
    "stable_hash",
    "strict_json_loads",
    "validate_production_canary_capability_receipts",
    "verify_implementation_freeze",
    "verify_self_hash",
]
