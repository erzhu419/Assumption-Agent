"""Authoritative, zero-execution application of the Q1 prestart amendment.

The pure candidate and wire modules define what fourteen admitted Q1 gates
mean.  This module owns only their local state transition: a post-clean-commit
capsule, an explicit owner-authorization record, one consumption record, and
one final 14/20 artifact.  It never starts Q1 and has no process or network
facility.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Final, Iterator, Mapping, NoReturn

from . import phase3_q1_prestart_candidate_v1 as _candidate
from . import phase3_q1_prestart_wire_v1 as _wire
from .strict_cbor_v1 import canonical_cbor_decode, content_hash


CONTRACT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-contract/1"
)
CONTRACT_ID: Final = "hegel-phase3a-q1-prestart-authoritative-amendment-v1"
ACTION_ID: Final = "apply-q1-prestart-admission-14-of-20-v1"
CAPSULE_INPUTS_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-capsule-inputs/1"
)
CAPSULE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-capsule/1"
)
AUTHORIZATION_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-authorization/1"
)
CONSUMPTION_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-consumption/1"
)
ARTIFACT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-artifact/1"
)
PENDING_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-pending/1"
)
TERMINAL_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-prestart-authoritative-amendment-terminal/1"
)

CAPSULE_READY: Final = "Q1_PRESTART_AMENDMENT_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = (
    "Q1_PRESTART_AMENDMENT_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
NOT_AUTHORIZED: Final = "Q1_PRESTART_AMENDMENT_NOT_AUTHORIZED_NOT_CONSUMED"
APPLIED: Final = "Q1_PRESTART_14_OF_20_APPLIED_Q1_NOT_RUN"
PENDING_NOT_CONSUMED: Final = "Q1_PRESTART_AMENDMENT_PENDING_NOT_CONSUMED"
TERMINAL_NO_RESUME: Final = "Q1_PRESTART_AMENDMENT_TERMINAL_NO_RESUME"

Q1_STATE: Final = "NOT_RUN"
Q1_GATE_COUNT: Final = 14
Q1_GATE_TOTAL: Final = 20
Q1_GATE_MASK: Final = 0x003FFF
NULL_OUTPUT_SLOTS: Final = [None] * 8
MAX_SMALL_JSON: Final = 16 * 1024 * 1024

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q1/PRESTART/AUTHORITATIVE_AMENDMENT/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q1/PRESTART/AUTHORITATIVE_AMENDMENT/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q1/PRESTART/AUTHORITATIVE_AMENDMENT/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q1/PRESTART/AUTHORITATIVE_AMENDMENT/CONSUMPTION/V1\x00",
    "actor_source_manifest": b"HEGEL/Q1/PRESTART/FORMAL_ACTOR_SOURCE_MANIFEST/V1\x00",
    "actor_dependency_identity": b"HEGEL/Q1/PRESTART/FORMAL_ACTOR_DEPENDENCY_IDENTITY/V1\x00",
    "actor_runtime_identity": b"HEGEL/Q1/PRESTART/FORMAL_ACTOR_RUNTIME_IDENTITY/V1\x00",
    "actor_offline_isolation": b"HEGEL/Q1/PRESTART/FORMAL_ACTOR_OFFLINE_ISOLATION/V1\x00",
    "mount_manifest": b"HEGEL/Q1/PRESTART/FORMAL_MOUNT_MANIFEST/V1\x00",
}

DEPENDENCY_IDENTITY_SCHEMA_VERSION: Final = (
    "hegel-q1-formal-actor-dependency-identity/1"
)
RUNTIME_IDENTITY_SCHEMA_VERSION: Final = (
    "hegel-q1-formal-actor-runtime-identity/1"
)
OFFLINE_ISOLATION_SCHEMA_VERSION: Final = (
    "hegel-q1-formal-actor-offline-isolation/1"
)
MOUNT_MANIFEST_SCHEMA_VERSION: Final = "hegel-q1-formal-mount-manifest/1"
INPUT_KEY_ABSENCE_SCHEMA_VERSION: Final = "hegel-q1-formal-input-key-absence/1"

_TAGGED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


class Q1PrestartAmendmentError(RuntimeError):
    """Base error for the authoritative amendment state machine."""


class AmendmentValidationError(Q1PrestartAmendmentError):
    """A committed contract, capsule, or record differs from its wire."""


class AmendmentPending(Q1PrestartAmendmentError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail


class AmendmentTerminal(Q1PrestartAmendmentError):
    def __init__(
        self,
        code: str,
        detail: str,
        *,
        authorization_consumption_count: int = 0,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.authorization_consumption_count = authorization_consumption_count


def _validation(detail: str) -> NoReturn:
    raise AmendmentValidationError(detail)


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _validation(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _validation(f"non-finite JSON token {value!r}")


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _validation("non-finite JSON number")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _validation("JSON object keys must be strings")
        for child in value.values():
            _finite(child)
    elif type(value) in (list, tuple):
        for child in value:
            _finite(child)


def canonical_json_bytes_v1(value: object) -> bytes:
    _finite(value)
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
    except (TypeError, ValueError, UnicodeError) as error:
        _validation(f"canonical JSON rejected: {error}")


def strict_json_bytes_v1(
    payload: bytes,
    label: str,
    maximum_bytes: int = MAX_SMALL_JSON,
) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > maximum_bytes:
        _validation(f"{label} exceeds the small-file bound")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _validation(f"{label} is not strict ASCII JSON: {error}")
    if type(value) is not dict:
        _validation(f"{label} must be a JSON object")
    if canonical_json_bytes_v1(value) != payload:
        _validation(f"{label} is not canonical JSON")
    return value


def _read_small(path: Path, label: str, maximum_bytes: int = MAX_SMALL_JSON) -> bytes:
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size > maximum_bytes:
            raise AmendmentPending(
                "PATH_AND_FORMAT",
                "SMALL_FILE_UNAVAILABLE",
                f"{label} is absent, not regular, or exceeds {maximum_bytes} bytes: {path}",
            )
        return path.read_bytes()
    except AmendmentPending:
        raise
    except OSError as error:
        raise AmendmentPending(
            "PATH_AND_FORMAT", "SMALL_FILE_UNAVAILABLE", f"{label}: {error}"
        ) from error


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _TAGGED_SHA256.fullmatch(value) is None:
        _validation(f"{label} must be tagged lowercase SHA-256")
    return value


def _hex40(value: object, label: str) -> str:
    if type(value) is not str or _HEX40.fullmatch(value) is None:
        _validation(f"{label} must be exactly 40 lowercase hex digits")
    return value


def _hex32(value: object, label: str) -> str:
    if type(value) is not str or _HEX32.fullmatch(value) is None:
        _validation(f"{label} must be exactly 32 lowercase hex digits")
    return value


def _digest(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes_v1(value)).hexdigest()


def _exact_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        _validation(f"{label} differs")


def _raw_sha(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedContractV1:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


def validate_contract_v1(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _validation("contract must be an object")
    contract = deepcopy(value)
    required = {
        "schema_version",
        "contract_id",
        "status",
        "action_id",
        "capsule_inputs_schema_version",
        "fixed_paths",
        "source_closure_relative_paths",
        "formal_actor_source_manifest_relative_paths",
        "prestart_wire_contract_relative_path",
        "evidence_inputs",
        "space_policy",
        "state_machine",
        "authority_boundary",
    }
    if set(contract) != required:
        _validation("contract field registry differs")
    if (
        contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["contract_id"] != CONTRACT_ID
        or contract["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or contract["action_id"] != ACTION_ID
        or contract["capsule_inputs_schema_version"]
        != CAPSULE_INPUTS_SCHEMA_VERSION
    ):
        _validation("contract header differs")
    paths = contract["fixed_paths"]
    if type(paths) is not dict or set(paths) != {
        "project_root",
        "capsule_path",
        "ledger_root",
        "authorization_path",
        "consumption_path",
        "artifact_path",
    }:
        _validation("fixed path registry differs")
    if any(type(item) is not str or not Path(item).is_absolute() for item in paths.values()):
        _validation("all fixed paths must be nonempty absolute paths")
    ledger_root = Path(paths["ledger_root"])
    if Path(paths["authorization_path"]).parent != ledger_root or Path(
        paths["consumption_path"]
    ).parent != ledger_root:
        _validation("authorization and consumption paths must belong to ledger root")
    source_paths = contract["source_closure_relative_paths"]
    if (
        type(source_paths) is not list
        or not source_paths
        or any(
            type(path) is not str
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in source_paths
        )
    ):
        _validation("source closure path registry differs")
    if len(source_paths) != len(set(source_paths)):
        _validation("source closure path registry repeats a path")
    actor_manifest_registry = contract[
        "formal_actor_source_manifest_relative_paths"
    ]
    if type(actor_manifest_registry) is not list or len(actor_manifest_registry) != 3:
        _validation("formal actor source-manifest registry differs")
    for row, frozen in zip(
        actor_manifest_registry,
        _wire.FORMAL_ACTOR_REGISTRY,
        strict=True,
    ):
        if type(row) is not dict or set(row) != {"actor_id", "relative_paths"}:
            _validation("formal actor source-manifest registry row differs")
        relative_paths = row["relative_paths"]
        if (
            row["actor_id"] != frozen[0].decode("ascii")
            or type(relative_paths) is not list
            or not relative_paths
            or any(type(path) is not str or path not in source_paths for path in relative_paths)
            or len(relative_paths) != len(set(relative_paths))
        ):
            _validation("formal actor source-manifest registry binding differs")
    wire_path = contract["prestart_wire_contract_relative_path"]
    if (
        type(wire_path) is not str
        or not wire_path
        or Path(wire_path).is_absolute()
        or ".." in Path(wire_path).parts
    ):
        _validation("prestart wire contract path differs")
    evidence = contract["evidence_inputs"]
    if type(evidence) is not list or len(evidence) != 7:
        _validation("exactly seven small predecessor evidence inputs are required")
    labels: list[str] = []
    for row in evidence:
        if type(row) is not dict or set(row) != {"label", "path", "maximum_bytes"}:
            _validation("evidence input row differs")
        if (
            type(row["label"]) is not str
            or not row["label"]
            or type(row["path"]) is not str
            or not Path(row["path"]).is_absolute()
            or type(row["maximum_bytes"]) is not int
            or not 1 <= row["maximum_bytes"] <= MAX_SMALL_JSON
        ):
            _validation("evidence input binding differs")
        labels.append(row["label"])
    if len(labels) != len(set(labels)):
        _validation("evidence labels must be unique")
    space = contract["space_policy"]
    if (
        type(space) is not dict
        or set(space) != {"minimum_free_bytes"}
        or type(space["minimum_free_bytes"]) is not int
        or space["minimum_free_bytes"] < 0
    ):
        _validation("space policy differs")
    state = contract["state_machine"]
    expected_state = {
        "same_digest_preconsumption_pending": True,
        "preflight_capsule_publication_atomic_no_replace": True,
        "authorization_consumption_count": 1,
        "final_artifact_publication_atomic_no_replace": True,
        "consumed_artifact_exact_read_only_replay": True,
        "consumed_artifact_absent_or_mismatch_terminal_no_resume": True,
    }
    if type(state) is not dict or set(state) != set(expected_state):
        _validation("state-machine policy differs")
    for field in expected_state.keys() - {"authorization_consumption_count"}:
        if state[field] is not True:
            _validation(f"state-machine {field} must be exact true")
    _exact_int(
        state.get("authorization_consumption_count"),
        1,
        "state-machine authorization consumption count",
    )
    authority = contract["authority_boundary"]
    expected_authority = {
        "q1_state": Q1_STATE,
        "q1_gate_count": Q1_GATE_COUNT,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_gate_mask": Q1_GATE_MASK,
        "q1_output_slots": NULL_OUTPUT_SLOTS,
        "future_q1_start_requires_distinct_authorization": True,
        "q1_execution_started": False,
        "maximum_popen_count": 0,
        "network_allowed": False,
        "target_split_role_access_allowed": False,
        "adoption_promotion_current_pointer_active_allowed": False,
    }
    if type(authority) is not dict or set(authority) != set(expected_authority):
        _validation("authority boundary differs")
    if (
        authority["q1_state"] != Q1_STATE
        or type(authority["q1_output_slots"]) is not list
        or authority["q1_output_slots"] != NULL_OUTPUT_SLOTS
    ):
        _validation("authority state or output slots differ")
    for field, expected in (
        ("future_q1_start_requires_distinct_authorization", True),
        ("q1_execution_started", False),
        ("network_allowed", False),
        ("target_split_role_access_allowed", False),
        ("adoption_promotion_current_pointer_active_allowed", False),
    ):
        if authority[field] is not expected:
            _validation(f"authority {field} must be exact {expected!r}")
    for field, expected in (
        ("q1_gate_count", Q1_GATE_COUNT),
        ("q1_gate_total", Q1_GATE_TOTAL),
        ("q1_gate_mask", Q1_GATE_MASK),
        ("maximum_popen_count", 0),
    ):
        _exact_int(authority.get(field), expected, f"authority {field}")
    return contract


def load_contract_v1(path: Path) -> LoadedContractV1:
    raw = _read_small(path, "amendment contract")
    value = validate_contract_v1(strict_json_bytes_v1(raw, "amendment contract"))
    return LoadedContractV1(path, raw, _raw_sha(raw), value)


def source_closure_sha256_v1(rows: object) -> str:
    if type(rows) is not list:
        _validation("source closure rows must be a list")
    return _digest(DOMAINS["source_closure"], rows)


def actor_source_manifest_root_v1(actor_id: str, rows: object) -> str:
    if type(actor_id) is not str or not actor_id or type(rows) is not list:
        _validation("actor source-manifest root input differs")
    return _digest(
        DOMAINS["actor_source_manifest"],
        {"actor_id": actor_id, "rows": rows},
    )


def actor_dependency_identity_root_v1(value: object) -> str:
    if type(value) is not dict or type(value.get("actor_id")) is not str:
        _validation("actor dependency-identity root input differs")
    checked = _dependency_identity_v1(value, value["actor_id"])
    return _digest(DOMAINS["actor_dependency_identity"], checked)


def actor_runtime_identity_root_v1(value: object) -> str:
    if type(value) is not dict or type(value.get("actor_id")) is not str:
        _validation("actor runtime-identity root input differs")
    checked = _runtime_identity_v1(value, value["actor_id"])
    return _digest(DOMAINS["actor_runtime_identity"], checked)


def actor_offline_isolation_root_v1(value: object) -> str:
    if type(value) is not dict or type(value.get("actor_id")) is not str:
        _validation("actor offline-isolation root input differs")
    checked = _offline_isolation_v1(value, value["actor_id"])
    return _digest(DOMAINS["actor_offline_isolation"], checked)


def mount_manifest_root_v1(value: object) -> str:
    if type(value) is not dict:
        _validation("mount-manifest root input differs")
    return _digest(DOMAINS["mount_manifest"], value)


def formal_input_key_absence_root_v1(value: object) -> str:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "formal_input_keys",
        "target_truth_accessed",
        "split_accessed",
        "role_evaluation_performed",
    }:
        _validation("formal input-key absence object differs")
    if (
        value["schema_version"] != INPUT_KEY_ABSENCE_SCHEMA_VERSION
        or value["formal_input_keys"] != []
        or value["target_truth_accessed"] is not False
        or value["split_accessed"] is not False
        or value["role_evaluation_performed"] is not False
    ):
        _validation("formal input-key absence boundary differs")
    root = content_hash(
        _candidate.INPUT_KEY_ABSENCE_DOMAIN,
        (1, b"hegel-q1-formal-input-key-absence/1", (), False, False, False),
    )
    return "sha256:" + root.hex()


def _validate_source_rows_v1(
    rows: object, required_paths: list[str]
) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(required_paths):
        _validation("source closure row count differs")
    checked: list[dict[str, Any]] = []
    for raw, expected_path in zip(rows, required_paths, strict=True):
        if type(raw) is not dict or set(raw) != {
            "relative_path",
            "byte_count",
            "raw_sha256",
        }:
            _validation("source closure row differs")
        if (
            raw["relative_path"] != expected_path
            or type(raw["byte_count"]) is not int
            or raw["byte_count"] < 0
        ):
            _validation("source closure row binding differs")
        _tagged(raw["raw_sha256"], "source closure raw_sha256")
        checked.append(deepcopy(raw))
    return checked


def _dependency_identity_v1(value: object, actor_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version", "actor_id", "ordered_rows"
    }:
        _validation("formal actor dependency identity differs")
    if (
        value["schema_version"] != DEPENDENCY_IDENTITY_SCHEMA_VERSION
        or value["actor_id"] != actor_id
        or type(value["ordered_rows"]) is not list
        or not value["ordered_rows"]
    ):
        _validation("formal actor dependency identity header differs")
    names: list[str] = []
    for row in value["ordered_rows"]:
        if type(row) is not dict or set(row) != {"name", "identity"}:
            _validation("formal actor dependency row differs")
        if (
            type(row["name"]) is not str
            or not row["name"]
            or type(row["identity"]) is not str
            or not row["identity"]
        ):
            _validation("formal actor dependency row value differs")
        names.append(row["name"])
    if len(names) != len(set(names)):
        _validation("formal actor dependency names differ")
    return deepcopy(value)


def _runtime_identity_v1(value: object, actor_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version", "actor_id", "runtime_kind", "runtime_identity"
    }:
        _validation("formal actor runtime identity differs")
    if (
        value["schema_version"] != RUNTIME_IDENTITY_SCHEMA_VERSION
        or value["actor_id"] != actor_id
        or type(value["runtime_kind"]) is not str
        or not value["runtime_kind"]
        or type(value["runtime_identity"]) is not str
        or not value["runtime_identity"]
    ):
        _validation("formal actor runtime identity value differs")
    return deepcopy(value)


def _offline_isolation_v1(value: object, actor_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "actor_id",
        "network_mode",
        "formal_input_keys",
        "target_truth_mounted",
        "split_mounted",
        "role_mounted",
    }:
        _validation("formal actor offline-isolation object differs")
    if (
        value["schema_version"] != OFFLINE_ISOLATION_SCHEMA_VERSION
        or value["actor_id"] != actor_id
        or value["network_mode"] != "none"
        or value["formal_input_keys"] != []
        or value["target_truth_mounted"] is not False
        or value["split_mounted"] is not False
        or value["role_mounted"] is not False
    ):
        _validation("formal actor offline-isolation boundary differs")
    return deepcopy(value)


def _validate_actor_rows_v1(
    rows: object,
    source_closure_rows: list[dict[str, Any]],
    expected_manifest_registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != 3:
        _validation("exactly three formal actor identities are required")
    checked: list[dict[str, Any]] = []
    expected = _wire.FORMAL_ACTOR_REGISTRY
    required = {
        "actor_id",
        "implementation_id",
        "action_id",
        "source_manifest_root",
        "entrypoint_raw_sha256",
        "dependency_closure_root",
        "runtime_identity_root",
        "offline_isolation_root",
        "entrypoint_relative_path",
        "source_manifest_rows",
        "dependency_identity",
        "runtime_identity",
        "offline_isolation",
    }
    source_registry = {
        row["relative_path"]: row for row in source_closure_rows
    }
    for raw, frozen, expected_manifest in zip(
        rows,
        expected,
        expected_manifest_registry,
        strict=True,
    ):
        if type(raw) is not dict or set(raw) != required:
            _validation("formal actor identity row differs")
        if (
            raw["actor_id"] != frozen[0].decode("ascii")
            or raw["implementation_id"] != frozen[1].decode("ascii")
            or raw["action_id"] != frozen[2].decode("ascii")
        ):
            _validation("formal actor registry differs")
        for field in required - {"actor_id", "implementation_id", "action_id"}:
            if field.endswith("_root") or field == "entrypoint_raw_sha256":
                _tagged(raw[field], f"formal actor {field}")
        actor_id = raw["actor_id"]
        manifest_rows = raw["source_manifest_rows"]
        if type(manifest_rows) is not list or not manifest_rows:
            _validation("formal actor source manifest must be nonempty")
        checked_manifest: list[dict[str, Any]] = []
        seen: list[str] = []
        for manifest_row in manifest_rows:
            if type(manifest_row) is not dict or set(manifest_row) != {
                "relative_path", "byte_count", "raw_sha256"
            }:
                _validation("formal actor source manifest row differs")
            relative = manifest_row["relative_path"]
            if type(relative) is not str:
                _validation("formal actor source manifest path differs")
            if relative not in source_registry or manifest_row != source_registry[relative]:
                _validation("formal actor source manifest is not an exact source-closure subset")
            seen.append(relative)
            checked_manifest.append(deepcopy(manifest_row))
        if len(seen) != len(set(seen)):
            _validation("formal actor source manifest repeats a path")
        if seen != expected_manifest["relative_paths"]:
            _validation("formal actor source manifest differs from its exact registry")
        if (
            type(raw["entrypoint_relative_path"]) is not str
            or raw["entrypoint_relative_path"] not in seen
        ):
            _validation("formal actor entrypoint is absent from its source manifest")
        entrypoint = source_registry[raw["entrypoint_relative_path"]]
        if raw["entrypoint_raw_sha256"] != entrypoint["raw_sha256"]:
            _validation("formal actor entrypoint raw SHA-256 differs")
        source_manifest_object = {
            "actor_id": actor_id,
            "rows": checked_manifest,
        }
        if raw["source_manifest_root"] != _digest(
            DOMAINS["actor_source_manifest"], source_manifest_object
        ):
            _validation("formal actor source manifest root differs")
        dependency = _dependency_identity_v1(raw["dependency_identity"], actor_id)
        runtime = _runtime_identity_v1(raw["runtime_identity"], actor_id)
        offline = _offline_isolation_v1(raw["offline_isolation"], actor_id)
        if raw["dependency_closure_root"] != _digest(
            DOMAINS["actor_dependency_identity"], dependency
        ):
            _validation("formal actor dependency identity root differs")
        if raw["runtime_identity_root"] != _digest(
            DOMAINS["actor_runtime_identity"], runtime
        ):
            _validation("formal actor runtime identity root differs")
        if raw["offline_isolation_root"] != _digest(
            DOMAINS["actor_offline_isolation"], offline
        ):
            _validation("formal actor offline-isolation root differs")
        checked.append(deepcopy(raw))
    return checked


def _validate_mount_manifest_v1(
    value: object,
    source_paths: list[str],
    actor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "rows",
        "formal_input_keys",
        "target_truth_mounted",
        "split_mounted",
        "role_mounted",
        "network_enabled",
    }:
        _validation("mount manifest object differs")
    rows = value["rows"]
    if (
        value["schema_version"] != MOUNT_MANIFEST_SCHEMA_VERSION
        or type(rows) is not list
        or len(rows) != 3
        or value["formal_input_keys"] != []
        or value["target_truth_mounted"] is not False
        or value["split_mounted"] is not False
        or value["role_mounted"] is not False
        or value["network_enabled"] is not False
    ):
        _validation("mount manifest boundary differs")
    for row, frozen, actor in zip(
        rows, _wire.FORMAL_ACTOR_REGISTRY, actor_rows, strict=True
    ):
        if type(row) is not dict or set(row) != {
            "actor_id", "read_only_source_relative_paths", "output_layout_root"
        }:
            _validation("mount manifest row differs")
        mounted = row["read_only_source_relative_paths"]
        if (
            row["actor_id"] != frozen[0].decode("ascii")
            or type(mounted) is not list
            or not mounted
            or any(type(path) is not str or path not in source_paths for path in mounted)
            or mounted
            != [item["relative_path"] for item in actor["source_manifest_rows"]]
            or row["output_layout_root"]
            != "sha256:" + _wire.FORMAL_OUTPUT_LAYOUT_ROOT.hex()
        ):
            _validation("mount manifest row binding differs")
        if len(mounted) != len(set(mounted)):
            _validation("mount manifest row repeats a source path")
    return deepcopy(value)


def _validate_evidence_rows_v1(
    rows: object, contract_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(contract_rows):
        _validation("evidence binding row count differs")
    checked: list[dict[str, Any]] = []
    for raw, frozen in zip(rows, contract_rows, strict=True):
        if type(raw) is not dict or set(raw) != {
            "label",
            "path",
            "byte_count",
            "raw_sha256",
        }:
            _validation("evidence binding row differs")
        if (
            raw["label"] != frozen["label"]
            or raw["path"] != frozen["path"]
            or type(raw["byte_count"]) is not int
            or not 0 <= raw["byte_count"] <= frozen["maximum_bytes"]
        ):
            _validation("evidence binding differs from contract")
        _tagged(raw["raw_sha256"], "evidence raw_sha256")
        checked.append(deepcopy(raw))
    return checked


def build_capsule_v1(
    contract: LoadedContractV1,
    capsule_inputs: object,
) -> dict[str, Any]:
    if type(capsule_inputs) is not dict:
        _validation("capsule inputs must be an object")
    inputs = deepcopy(capsule_inputs)
    required = {
        "schema_version",
        "source_identity",
        "source_closure_rows",
        "formal_actor_identities",
        "run_id",
        "created_at_unix_seconds",
        "prestart_wire_contract_raw_sha256",
        "mount_manifest",
        "mount_manifest_root",
        "formal_input_key_absence",
        "formal_input_key_absence_root",
        "evidence_inputs",
    }
    if set(inputs) != required or inputs["schema_version"] != CAPSULE_INPUTS_SCHEMA_VERSION:
        _validation("capsule input registry differs")
    source = inputs["source_identity"]
    if type(source) is not dict or set(source) != {
        "source_commit", "source_tree", "source_closure_sha256"
    }:
        _validation("source identity input differs")
    _hex40(source["source_commit"], "source commit")
    _hex40(source["source_tree"], "source tree")
    _tagged(source["source_closure_sha256"], "source closure")
    source_rows = _validate_source_rows_v1(
        inputs["source_closure_rows"],
        contract.value["source_closure_relative_paths"],
    )
    if source_closure_sha256_v1(source_rows) != source["source_closure_sha256"]:
        _validation("source closure digest differs")
    actors = _validate_actor_rows_v1(
        inputs["formal_actor_identities"],
        source_rows,
        contract.value["formal_actor_source_manifest_relative_paths"],
    )
    _hex32(inputs["run_id"], "run ID")
    if (
        type(inputs["created_at_unix_seconds"]) is not int
        or inputs["created_at_unix_seconds"] < 0
        or inputs["created_at_unix_seconds"] > (1 << 64) - 1
    ):
        _validation("created_at_unix_seconds must be uint64")
    for field in (
        "prestart_wire_contract_raw_sha256",
        "mount_manifest_root",
        "formal_input_key_absence_root",
    ):
        _tagged(inputs[field], field)
    mount_manifest = _validate_mount_manifest_v1(
        inputs["mount_manifest"],
        contract.value["source_closure_relative_paths"],
        actors,
    )
    if inputs["mount_manifest_root"] != _digest(
        DOMAINS["mount_manifest"], mount_manifest
    ):
        _validation("mount manifest root differs")
    expected_absence = formal_input_key_absence_root_v1(
        inputs["formal_input_key_absence"]
    )
    if (
        expected_absence
        != "sha256:" + _candidate.FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT.hex()
        or inputs["formal_input_key_absence_root"] != expected_absence
    ):
        _validation("formal input-key absence root differs")
    evidence = _validate_evidence_rows_v1(
        inputs["evidence_inputs"], contract.value["evidence_inputs"]
    )
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": "Q1_PRESTART_AMENDMENT_PREFLIGHT_ADMITTED",
        "contract_id": CONTRACT_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "action_id": ACTION_ID,
        "source_identity": deepcopy(source),
        "source_closure_rows": source_rows,
        "formal_actor_identities": actors,
        "run_id": inputs["run_id"],
        "created_at_unix_seconds": inputs["created_at_unix_seconds"],
        "prestart_wire_contract_raw_sha256": inputs[
            "prestart_wire_contract_raw_sha256"
        ],
        "mount_manifest": mount_manifest,
        "mount_manifest_root": inputs["mount_manifest_root"],
        "formal_input_key_absence": deepcopy(inputs["formal_input_key_absence"]),
        "formal_input_key_absence_root": inputs["formal_input_key_absence_root"],
        "evidence_inputs": evidence,
        "fixed_paths": deepcopy(contract.value["fixed_paths"]),
        "q1_state": Q1_STATE,
        "q1_gate_count_before_apply": 0,
        "q1_gate_mask_before_apply": 0,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "intended_q1_gate_count": Q1_GATE_COUNT,
        "intended_q1_gate_total": Q1_GATE_TOTAL,
        "intended_q1_gate_mask": Q1_GATE_MASK,
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule_v1(
    value: object,
    contract: LoadedContractV1,
) -> dict[str, Any]:
    if type(value) is not dict:
        _validation("capsule must be an object")
    capsule = deepcopy(value)
    required = {
        "schema_version",
        "status",
        "contract_id",
        "contract_raw_sha256",
        "action_id",
        "source_identity",
        "source_closure_rows",
        "formal_actor_identities",
        "run_id",
        "created_at_unix_seconds",
        "prestart_wire_contract_raw_sha256",
        "mount_manifest",
        "mount_manifest_root",
        "formal_input_key_absence",
        "formal_input_key_absence_root",
        "evidence_inputs",
        "fixed_paths",
        "q1_state",
        "q1_gate_count_before_apply",
        "q1_gate_mask_before_apply",
        "q1_output_slots",
        "intended_q1_gate_count",
        "intended_q1_gate_total",
        "intended_q1_gate_mask",
        "capsule_digest",
    }
    if set(capsule) != required:
        _validation("capsule field registry differs")
    for field, expected in (
        ("q1_gate_count_before_apply", 0),
        ("q1_gate_mask_before_apply", 0),
        ("intended_q1_gate_count", Q1_GATE_COUNT),
        ("intended_q1_gate_total", Q1_GATE_TOTAL),
        ("intended_q1_gate_mask", Q1_GATE_MASK),
    ):
        _exact_int(capsule.get(field), expected, f"capsule {field}")
    observed = _tagged(capsule.get("capsule_digest"), "capsule digest")
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest", None)
    if (
        capsule.get("schema_version") != CAPSULE_SCHEMA_VERSION
        or capsule.get("status") != "Q1_PRESTART_AMENDMENT_PREFLIGHT_ADMITTED"
        or capsule.get("contract_id") != CONTRACT_ID
        or capsule.get("contract_raw_sha256") != contract.raw_sha256
        or capsule.get("action_id") != ACTION_ID
        or capsule.get("fixed_paths") != contract.value["fixed_paths"]
        or capsule.get("q1_state") != Q1_STATE
        or capsule.get("q1_gate_count_before_apply") != 0
        or capsule.get("q1_gate_mask_before_apply") != 0
        or capsule.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or capsule.get("intended_q1_gate_count") != Q1_GATE_COUNT
        or capsule.get("intended_q1_gate_total") != Q1_GATE_TOTAL
        or capsule.get("intended_q1_gate_mask") != Q1_GATE_MASK
        or _digest(DOMAINS["capsule"], unsigned) != observed
    ):
        _validation("capsule wire differs")
    source = capsule.get("source_identity")
    if type(source) is not dict or set(source) != {
        "source_commit", "source_tree", "source_closure_sha256"
    }:
        _validation("capsule source identity differs")
    _hex40(source.get("source_commit"), "capsule source commit")
    _hex40(source.get("source_tree"), "capsule source tree")
    source_rows = _validate_source_rows_v1(
        capsule.get("source_closure_rows"),
        contract.value["source_closure_relative_paths"],
    )
    if source_closure_sha256_v1(source_rows) != source.get("source_closure_sha256"):
        _validation("capsule source closure differs")
    actors = _validate_actor_rows_v1(
        capsule.get("formal_actor_identities"),
        source_rows,
        contract.value["formal_actor_source_manifest_relative_paths"],
    )
    _validate_evidence_rows_v1(
        capsule.get("evidence_inputs"), contract.value["evidence_inputs"]
    )
    _hex32(capsule.get("run_id"), "capsule run ID")
    if (
        type(capsule.get("created_at_unix_seconds")) is not int
        or capsule["created_at_unix_seconds"] < 0
        or capsule["created_at_unix_seconds"] > (1 << 64) - 1
    ):
        _validation("capsule timestamp differs")
    for field in (
        "prestart_wire_contract_raw_sha256",
        "mount_manifest_root",
        "formal_input_key_absence_root",
    ):
        _tagged(capsule.get(field), f"capsule {field}")
    mount_manifest = _validate_mount_manifest_v1(
        capsule.get("mount_manifest"),
        contract.value["source_closure_relative_paths"],
        actors,
    )
    if capsule.get("mount_manifest_root") != _digest(
        DOMAINS["mount_manifest"], mount_manifest
    ):
        _validation("capsule mount manifest root differs")
    expected_absence = formal_input_key_absence_root_v1(
        capsule.get("formal_input_key_absence")
    )
    if (
        expected_absence
        != "sha256:" + _candidate.FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT.hex()
        or capsule.get("formal_input_key_absence_root") != expected_absence
    ):
        _validation("capsule formal input-key absence root differs")
    return capsule


def _load_canonical_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_small(path, label)
    return raw, strict_json_bytes_v1(raw, label)


def _load_published_capsule_v1(
    contract: LoadedContractV1, expected_digest: str
) -> tuple[bytes, dict[str, Any]]:
    consumed_exists = Path(
        contract.value["fixed_paths"]["consumption_path"]
    ).exists()
    try:
        _tagged(expected_digest, "requested capsule digest")
    except AmendmentValidationError as error:
        raise AmendmentTerminal(
            "REQUESTED_CAPSULE_DIGEST_INVALID",
            str(error),
            authorization_consumption_count=1 if consumed_exists else 0,
        ) from error
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    try:
        raw, value = _load_canonical_object(path, "published capsule")
    except AmendmentPending as error:
        if consumed_exists:
            raise AmendmentTerminal(
                "CONSUMED_CAPSULE_UNAVAILABLE",
                error.detail,
                authorization_consumption_count=1,
            ) from error
        raise AmendmentPending(
            "PATH_AND_FORMAT", "PUBLISHED_CAPSULE_UNAVAILABLE", error.detail
        ) from error
    try:
        capsule = validate_capsule_v1(value, contract)
    except AmendmentValidationError as error:
        if consumed_exists:
            raise AmendmentTerminal(
                "CONSUMED_CAPSULE_MISMATCH",
                str(error),
                authorization_consumption_count=1,
            ) from error
        raise AmendmentPending(
            "PATH_AND_FORMAT",
            "PUBLISHED_CAPSULE_FORMAT_OR_BINDING_DIFFERS",
            str(error),
        ) from error
    if capsule["capsule_digest"] != expected_digest:
        raise AmendmentTerminal(
            "CAPSULE_DIGEST_MISMATCH",
            "published capsule does not equal the explicitly requested digest",
            authorization_consumption_count=1 if consumed_exists else 0,
        )
    return raw, capsule


def _path_payloads_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> list[bytes]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    source_rows = capsule["source_closure_rows"]
    for row in source_rows:
        path = project_root / row["relative_path"]
        payload = _read_small(path, f"source closure {row['relative_path']}")
        if len(payload) != row["byte_count"] or _raw_sha(payload) != row["raw_sha256"]:
            raise AmendmentPending(
                "PATH_AND_FORMAT",
                "SOURCE_CLOSURE_BINDING_DIFFERS",
                f"source closure bytes differ: {path}",
            )
    wire_path = project_root / contract.value["prestart_wire_contract_relative_path"]
    wire_payload = _read_small(wire_path, "prestart wire contract")
    if _raw_sha(wire_payload) != capsule["prestart_wire_contract_raw_sha256"]:
        raise AmendmentPending(
            "PATH_AND_FORMAT",
            "PRESTART_WIRE_CONTRACT_DIFFERS",
            f"prestart wire contract bytes differ: {wire_path}",
        )
    payloads: list[bytes] = []
    for row, frozen in zip(
        capsule["evidence_inputs"],
        contract.value["evidence_inputs"],
        strict=True,
    ):
        payload = _read_small(Path(row["path"]), row["label"], frozen["maximum_bytes"])
        if len(payload) != row["byte_count"] or _raw_sha(payload) != row["raw_sha256"]:
            raise AmendmentPending(
                "PATH_AND_FORMAT",
                "PREDECESSOR_EVIDENCE_BINDING_DIFFERS",
                f"predecessor bytes differ: {row['path']}",
            )
        payloads.append(payload)
    return payloads


def _wire_inputs_v1(
    capsule: Mapping[str, Any],
) -> tuple[_wire.Q1SourceIdentityV1, tuple[_wire.Q1FormalActorIdentityV1, ...]]:
    source = capsule["source_identity"]
    source_identity = _wire.Q1SourceIdentityV1(
        bytes.fromhex(source["source_commit"]),
        bytes.fromhex(source["source_tree"]),
        bytes.fromhex(source["source_closure_sha256"][7:]),
    )
    actors = tuple(
        _wire.Q1FormalActorIdentityV1(
            actor_id=row["actor_id"].encode("ascii"),
            implementation_id=row["implementation_id"].encode("ascii"),
            action_id=row["action_id"].encode("ascii"),
            source_manifest_root=bytes.fromhex(row["source_manifest_root"][7:]),
            entrypoint_raw_sha256=bytes.fromhex(row["entrypoint_raw_sha256"][7:]),
            dependency_closure_root=bytes.fromhex(row["dependency_closure_root"][7:]),
            runtime_identity_root=bytes.fromhex(row["runtime_identity_root"][7:]),
            offline_isolation_root=bytes.fromhex(row["offline_isolation_root"][7:]),
        )
        for row in capsule["formal_actor_identities"]
    )
    return source_identity, actors


def _derive_and_replay_candidate_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = _path_payloads_v1(contract, capsule)
    source, actors = _wire_inputs_v1(capsule)
    try:
        value = _candidate.derive_q1_prestart_candidate_v1(
            *payloads,
            source_identity=source,
            actor_identities=actors,
            run_id=bytes.fromhex(capsule["run_id"]),
            created_at_unix_seconds=capsule["created_at_unix_seconds"],
            prestart_wire_contract_raw_sha256=bytes.fromhex(
                capsule["prestart_wire_contract_raw_sha256"][7:]
            ),
            mount_manifest_root=bytes.fromhex(capsule["mount_manifest_root"][7:]),
        )
        if type(value) is not dict:
            _validation("candidate derivation did not return an object")
        genesis_payload = bytes.fromhex(value["run_genesis_cbor_hex"])
        receipt_payload = bytes.fromhex(value["prestart_admission_receipt_cbor_hex"])
        genesis = _wire.decode_run_genesis_v1(genesis_payload)
        receipt = _wire.decode_prestart_admission_receipt_v1(receipt_payload)
        subjects = tuple(
            canonical_cbor_decode(bytes.fromhex(row["subjects_cbor_hex"]))
            for row in value["gate_evidence_rows"]
        )
        _wire.verify_prestart_gate_evidence_v1(receipt, subjects)
    except AmendmentValidationError:
        raise
    except Exception as error:
        _validation(f"pure candidate replay failed: {error}")
    for field, expected in (
        ("intended_q1_gate_count", Q1_GATE_COUNT),
        ("intended_q1_gate_mask", Q1_GATE_MASK),
        ("q1_gate_count_before_authoritative_apply", 0),
        ("q1_gate_mask_before_authoritative_apply", 0),
        ("popen_count", 0),
    ):
        _exact_int(value.get(field), expected, f"candidate {field}")
    if (
        value.get("status") != _candidate.STATUS
        or value.get("candidate_only") is not True
        or value.get("published") is not False
        or value.get("q1_execution_started") is not False
        or value.get("intended_q1_state") != Q1_STATE
        or value.get("intended_q1_gate_count") != Q1_GATE_COUNT
        or value.get("intended_q1_gate_mask") != Q1_GATE_MASK
        or value.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or value.get("popen_count") != 0
        or value.get("network_accessed") is not False
        or value.get("target_truth_accessed") is not False
        or value.get("split_accessed") is not False
        or value.get("role_evaluation_performed") is not False
        or value.get("run_genesis_root") != "sha256:" + genesis.genesis_root.hex()
        or value.get("prestart_admission_receipt_root")
        != "sha256:" + receipt.receipt_root.hex()
        or receipt.genesis != genesis
        or len(subjects) != Q1_GATE_COUNT
    ):
        _validation("candidate authority boundary or replay differs")
    return deepcopy(value)


def _authorization_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule["capsule_digest"],
        "action_id": ACTION_ID,
        "authority_role": "PROJECT_OWNER",
        "authorized_scope": "APPLY_Q1_PRESTART_14_OF_20_ONLY",
        "run_genesis_root": candidate["run_genesis_root"],
        "prestart_admission_receipt_root": candidate[
            "prestart_admission_receipt_root"
        ],
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "authorization_consumption_limit": 1,
        "q1_state": Q1_STATE,
        "q1_gate_count": Q1_GATE_COUNT,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_gate_mask": Q1_GATE_MASK,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "q1_start_authorized": False,
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def _artifact_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    candidate: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": APPLIED,
        "action_id": ACTION_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "source_commit": capsule["source_identity"]["source_commit"],
        "source_tree": capsule["source_identity"]["source_tree"],
        "actor_implementation_roots": deepcopy(
            candidate["actor_implementation_roots"]
        ),
        "run_genesis_root": candidate["run_genesis_root"],
        "run_genesis_cbor_hex": candidate["run_genesis_cbor_hex"],
        "prestart_admission_receipt_root": candidate[
            "prestart_admission_receipt_root"
        ],
        "prestart_admission_receipt_cbor_hex": candidate[
            "prestart_admission_receipt_cbor_hex"
        ],
        "gate_evidence_rows": deepcopy(candidate["gate_evidence_rows"]),
        "q1_state": Q1_STATE,
        "q1_gate_count": Q1_GATE_COUNT,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_gate_mask": Q1_GATE_MASK,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "campaign_execution_count": 1,
        "authorization_consumption_count": 1,
        "future_q1_start_requires_distinct_authorization": True,
        "q1_execution_started": False,
        "popen_count": 0,
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "adopted": False,
        "promoted": False,
        "current_pointer_updated": False,
        "active_transition_performed": False,
    }


def _consumption_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    artifact_bytes: bytes,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "status": "Q1_PRESTART_AMENDMENT_AUTHORIZATION_CONSUMED",
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "expected_artifact_raw_sha256": _raw_sha(artifact_bytes),
        "run_genesis_root": artifact["run_genesis_root"],
        "prestart_admission_receipt_root": artifact[
            "prestart_admission_receipt_root"
        ],
        "q1_state": Q1_STATE,
        "q1_gate_count": Q1_GATE_COUNT,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_gate_mask": Q1_GATE_MASK,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "campaign_execution_count": 1,
        "q1_execution_started": False,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def _validate_authorization_v1(
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _validation("authorization must be an object")
    authorization = deepcopy(value)
    required = {
        "schema_version",
        "status",
        "contract_raw_sha256",
        "capsule_digest",
        "action_id",
        "authority_role",
        "authorized_scope",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "campaign_execution_count",
        "authorization_consumption_count",
        "authorization_consumption_limit",
        "q1_state",
        "q1_gate_count",
        "q1_gate_total",
        "q1_gate_mask",
        "q1_output_slots",
        "q1_start_authorized",
        "authorization_digest",
    }
    if set(authorization) != required:
        _validation("authorization field registry differs")
    for field, expected in (
        ("campaign_execution_count", 0),
        ("authorization_consumption_count", 0),
        ("authorization_consumption_limit", 1),
        ("q1_gate_count", Q1_GATE_COUNT),
        ("q1_gate_total", Q1_GATE_TOTAL),
        ("q1_gate_mask", Q1_GATE_MASK),
    ):
        _exact_int(authorization.get(field), expected, f"authorization {field}")
    observed = _tagged(
        authorization.get("authorization_digest"), "authorization digest"
    )
    unsigned = deepcopy(authorization)
    unsigned.pop("authorization_digest", None)
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("status") != AUTHORIZATION_RECORDED
        or authorization.get("contract_raw_sha256") != contract.raw_sha256
        or authorization.get("capsule_digest") != capsule["capsule_digest"]
        or authorization.get("action_id") != ACTION_ID
        or authorization.get("authority_role") != "PROJECT_OWNER"
        or authorization.get("authorized_scope")
        != "APPLY_Q1_PRESTART_14_OF_20_ONLY"
        or authorization.get("authorization_consumption_limit") != 1
        or authorization.get("q1_state") != Q1_STATE
        or authorization.get("q1_gate_count") != Q1_GATE_COUNT
        or authorization.get("q1_gate_total") != Q1_GATE_TOTAL
        or authorization.get("q1_gate_mask") != Q1_GATE_MASK
        or authorization.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or authorization.get("q1_start_authorized") is not False
        or _digest(DOMAINS["authorization"], unsigned) != observed
    ):
        _validation("authorization record differs")
    return authorization


def _validate_consumption_v1(
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _validation("consumption must be an object")
    consumed = deepcopy(value)
    required = {
        "schema_version",
        "status",
        "contract_raw_sha256",
        "capsule_digest",
        "authorization_digest",
        "authorization_consumption_count",
        "expected_artifact_raw_sha256",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "q1_state",
        "q1_gate_count",
        "q1_gate_total",
        "q1_gate_mask",
        "q1_output_slots",
        "campaign_execution_count",
        "q1_execution_started",
        "record_digest",
    }
    if set(consumed) != required:
        _validation("consumption field registry differs")
    for field, expected in (
        ("authorization_consumption_count", 1),
        ("campaign_execution_count", 1),
        ("q1_gate_count", Q1_GATE_COUNT),
        ("q1_gate_total", Q1_GATE_TOTAL),
        ("q1_gate_mask", Q1_GATE_MASK),
    ):
        _exact_int(consumed.get(field), expected, f"consumption {field}")
    observed = _tagged(consumed.get("record_digest"), "consumption record digest")
    unsigned = deepcopy(consumed)
    unsigned.pop("record_digest", None)
    if (
        consumed.get("schema_version") != CONSUMPTION_SCHEMA_VERSION
        or consumed.get("status")
        != "Q1_PRESTART_AMENDMENT_AUTHORIZATION_CONSUMED"
        or consumed.get("contract_raw_sha256") != contract.raw_sha256
        or consumed.get("capsule_digest") != capsule["capsule_digest"]
        or consumed.get("authorization_consumption_count") != 1
        or consumed.get("q1_state") != Q1_STATE
        or consumed.get("q1_gate_count") != Q1_GATE_COUNT
        or consumed.get("q1_gate_total") != Q1_GATE_TOTAL
        or consumed.get("q1_gate_mask") != Q1_GATE_MASK
        or consumed.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or consumed.get("q1_execution_started") is not False
        or _TAGGED_SHA256.fullmatch(
            str(consumed.get("expected_artifact_raw_sha256"))
        )
        is None
        or _digest(DOMAINS["consumption"], unsigned) != observed
    ):
        _validation("consumption record differs")
    return consumed


def _validate_replayed_artifact_v1(
    raw: bytes,
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    consumed: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _validation("final artifact must be an object")
    artifact = deepcopy(value)
    required = {
        "schema_version",
        "status",
        "action_id",
        "contract_raw_sha256",
        "capsule_digest",
        "authorization_digest",
        "source_commit",
        "source_tree",
        "actor_implementation_roots",
        "run_genesis_root",
        "run_genesis_cbor_hex",
        "prestart_admission_receipt_root",
        "prestart_admission_receipt_cbor_hex",
        "gate_evidence_rows",
        "q1_state",
        "q1_gate_count",
        "q1_gate_total",
        "q1_gate_mask",
        "q1_output_slots",
        "campaign_execution_count",
        "authorization_consumption_count",
        "future_q1_start_requires_distinct_authorization",
        "q1_execution_started",
        "popen_count",
        "network_accessed",
        "target_truth_accessed",
        "split_accessed",
        "role_evaluation_performed",
        "adopted",
        "promoted",
        "current_pointer_updated",
        "active_transition_performed",
    }
    if (
        set(artifact) != required
        or _raw_sha(raw) != consumed["expected_artifact_raw_sha256"]
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != APPLIED
        or artifact.get("action_id") != ACTION_ID
        or artifact.get("contract_raw_sha256") != contract.raw_sha256
        or artifact.get("capsule_digest") != capsule["capsule_digest"]
        or artifact.get("authorization_digest") != consumed["authorization_digest"]
        or artifact.get("run_genesis_root") != consumed["run_genesis_root"]
        or artifact.get("prestart_admission_receipt_root")
        != consumed["prestart_admission_receipt_root"]
        or artifact.get("q1_state") != Q1_STATE
        or artifact.get("q1_gate_count") != Q1_GATE_COUNT
        or artifact.get("q1_gate_total") != Q1_GATE_TOTAL
        or artifact.get("q1_gate_mask") != Q1_GATE_MASK
        or artifact.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or artifact.get("authorization_consumption_count") != 1
        or artifact.get("future_q1_start_requires_distinct_authorization") is not True
        or artifact.get("q1_execution_started") is not False
        or artifact.get("popen_count") != 0
        or artifact.get("network_accessed") is not False
        or artifact.get("target_truth_accessed") is not False
        or artifact.get("split_accessed") is not False
        or artifact.get("role_evaluation_performed") is not False
        or artifact.get("adopted") is not False
        or artifact.get("promoted") is not False
        or artifact.get("current_pointer_updated") is not False
        or artifact.get("active_transition_performed") is not False
    ):
        _validation("final artifact differs from consumed authority")
    for field, expected in (
        ("campaign_execution_count", 1),
        ("authorization_consumption_count", 1),
        ("q1_gate_count", Q1_GATE_COUNT),
        ("q1_gate_total", Q1_GATE_TOTAL),
        ("q1_gate_mask", Q1_GATE_MASK),
        ("popen_count", 0),
    ):
        _exact_int(artifact.get(field), expected, f"artifact {field}")
    return artifact


@contextmanager
def _ledger_lock_v1(ledger_root: Path) -> Iterator[None]:
    try:
        fd = os.open(ledger_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise AmendmentPending(
            "PATH_AND_FORMAT", "LEDGER_ROOT_UNAVAILABLE", str(error)
        ) from error
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise AmendmentPending(
                "PATH_AND_FORMAT", "LEDGER_LOCK_UNAVAILABLE", str(error)
            ) from error
        yield
    finally:
        os.close(fd)


def _fsync_directory_v1(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage_path_v1(path: Path, capsule_digest: str) -> Path:
    return path.with_name(path.name + ".stage-" + capsule_digest[-16:])


def _clear_transient_stage_v1(path: Path, capsule_digest: str) -> None:
    stage = _stage_path_v1(path, capsule_digest)
    if not stage.exists():
        return
    try:
        stage.unlink()
        _fsync_directory_v1(path.parent)
    except OSError as error:
        raise AmendmentPending(
            "PATH_SPACE_AND_PROBE",
            "TRANSIENT_STAGE_CLEANUP_FAILED",
            f"{stage}: {error}",
        ) from error


def _probe_directory_v1(directory: Path, capsule_digest: str) -> None:
    probe = directory / (".q1-prestart-probe-" + capsule_digest[-16:])
    fd: int | None = None
    try:
        fd = os.open(
            probe,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        os.write(fd, b"q1-prestart-probe-v1\n")
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.unlink(probe)
        _fsync_directory_v1(directory)
    except OSError as error:
        if fd is not None:
            os.close(fd)
        try:
            if probe.exists():
                probe.unlink()
                _fsync_directory_v1(directory)
        except OSError:
            pass
        raise AmendmentPending(
            "PATH_SPACE_AND_PROBE", "DIRECTORY_PROBE_FAILED", f"{directory}: {error}"
        ) from error


def _publication_directories_v1(contract: LoadedContractV1) -> tuple[Path, ...]:
    fixed = contract.value["fixed_paths"]
    candidates = (
        Path(fixed["capsule_path"]).parent,
        Path(fixed["ledger_root"]),
        Path(fixed["artifact_path"]).parent,
    )
    return tuple(dict.fromkeys(candidates))


def _preconsumption_storage_gate_v1(
    contract: LoadedContractV1,
    capsule_digest: str,
    *,
    target_paths: tuple[Path, ...],
) -> None:
    for directory in _publication_directories_v1(contract):
        if not directory.is_dir():
            raise AmendmentPending(
                "PATH_SPACE_AND_PROBE",
                "PUBLICATION_DIRECTORY_UNAVAILABLE",
                str(directory),
            )
        try:
            free = os.statvfs(directory).f_bavail * os.statvfs(directory).f_frsize
        except OSError as error:
            raise AmendmentPending(
                "PATH_SPACE_AND_PROBE", "FREE_SPACE_UNAVAILABLE", str(error)
            ) from error
        if free < contract.value["space_policy"]["minimum_free_bytes"]:
            raise AmendmentPending(
                "PATH_SPACE_AND_PROBE",
                "INSUFFICIENT_FREE_SPACE",
                f"{directory}: {free} bytes available",
            )
    for path in target_paths:
        if not path.exists():
            _clear_transient_stage_v1(path, capsule_digest)
        if path.exists():
            raise AmendmentPending(
                "PATH_SPACE_AND_PROBE",
                "NO_REPLACE_PATH_OCCUPIED",
                str(path),
            )
    for directory in _publication_directories_v1(contract):
        _probe_directory_v1(directory, capsule_digest)


def _atomic_no_replace_v1(path: Path, payload: bytes, capsule_digest: str) -> None:
    stage = _stage_path_v1(path, capsule_digest)
    fd = os.open(
        stage,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("zero-byte write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.link(stage, path, follow_symlinks=False)
    _fsync_directory_v1(path.parent)
    os.unlink(stage)
    _fsync_directory_v1(path.parent)


def _pending_result_v1(
    contract: LoadedContractV1,
    capsule_digest: str | None,
    error: AmendmentPending,
) -> dict[str, Any]:
    return {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": PENDING_NOT_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule_digest,
        "phase": error.phase,
        "condition": error.condition,
        "detail": error.detail,
        "same_capsule_digest_reentry_allowed": capsule_digest is not None,
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "popen_count": 0,
        "network_accessed": False,
        "q1_state": Q1_STATE,
        "q1_gate_count": 0,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_gate_mask": 0,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
    }


def _unbound_contract_pending_result_v1(
    capsule_digest: str | None,
    contract_path: Path,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": PENDING_NOT_CONSUMED,
        "contract_raw_sha256": None,
        "capsule_digest": capsule_digest,
        "phase": "PATH_AND_FORMAT",
        "condition": "CONTRACT_UNAVAILABLE_OR_INVALID",
        "detail": f"{contract_path}: {error}",
        "same_capsule_digest_reentry_allowed": capsule_digest is not None,
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "popen_count": 0,
        "network_accessed": False,
        "q1_state": Q1_STATE,
        "q1_gate_count": 0,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_gate_mask": 0,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
    }


def _load_contract_entry_v1(
    contract_path: Path,
    capsule_digest: str | None,
) -> tuple[LoadedContractV1 | None, dict[str, Any] | None]:
    try:
        return load_contract_v1(contract_path), None
    except (AmendmentPending, AmendmentValidationError, OSError) as error:
        return None, _unbound_contract_pending_result_v1(
            capsule_digest,
            contract_path,
            error,
        )


def _terminal_result_v1(
    contract: LoadedContractV1,
    capsule_digest: str | None,
    error: AmendmentTerminal,
) -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": TERMINAL_NO_RESUME,
        "code": error.code,
        "detail": error.detail,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule_digest,
        "authorization_consumption_count": error.authorization_consumption_count,
        "retry_or_resume_allowed": False,
        "popen_count": 0,
        "network_accessed": False,
        "q1_execution_started": False,
    }


def _preflight_failure_result_v1(
    contract: LoadedContractV1,
    capsule_digest: str | None,
    pending: AmendmentPending,
) -> dict[str, Any]:
    if Path(contract.value["fixed_paths"]["consumption_path"]).exists():
        return _terminal_result_v1(
            contract,
            capsule_digest,
            AmendmentTerminal(
                "POST_CONSUMPTION_PREFLIGHT_NOT_REPLAYABLE",
                str(pending),
                authorization_consumption_count=1,
            ),
        )
    return _pending_result_v1(contract, capsule_digest, pending)


def preflight_v1(
    contract_path: Path,
    capsule_inputs_path: Path,
) -> dict[str, Any]:
    contract, contract_pending = _load_contract_entry_v1(contract_path, None)
    if contract_pending is not None:
        return contract_pending
    assert contract is not None
    digest_hint: str | None = None
    try:
        inputs_raw = _read_small(capsule_inputs_path, "capsule inputs")
        inputs = strict_json_bytes_v1(inputs_raw, "capsule inputs")
        capsule = build_capsule_v1(contract, inputs)
        digest_hint = capsule["capsule_digest"]
        capsule_path = Path(contract.value["fixed_paths"]["capsule_path"])
        ledger_root = Path(contract.value["fixed_paths"]["ledger_root"])
        with _ledger_lock_v1(ledger_root):
            consumed_exists = Path(
                contract.value["fixed_paths"]["consumption_path"]
            ).exists()
            if capsule_path.exists():
                try:
                    raw, existing = _load_canonical_object(
                        capsule_path, "published capsule"
                    )
                    validated = validate_capsule_v1(existing, contract)
                except (AmendmentPending, AmendmentValidationError) as error:
                    if consumed_exists:
                        raise AmendmentTerminal(
                            "CONSUMED_CAPSULE_MISMATCH",
                            str(error),
                            authorization_consumption_count=1,
                        ) from error
                    raise AmendmentPending(
                        "PATH_AND_FORMAT",
                        "PUBLISHED_CAPSULE_FORMAT_OR_BINDING_DIFFERS",
                        str(error),
                    ) from error
                if raw != canonical_json_bytes_v1(capsule) or validated != capsule:
                    if consumed_exists:
                        raise AmendmentTerminal(
                            "CONSUMED_CAPSULE_MISMATCH",
                            "published capsule differs from the exact preflight inputs",
                            authorization_consumption_count=1,
                        )
                    raise AmendmentPending(
                        "PATH_SPACE_AND_PROBE",
                        "CAPSULE_PATH_OCCUPIED_BY_DIFFERENT_BYTES",
                        str(capsule_path),
                    )
                consumed_replay = _consumed_replay_or_terminal_v1(
                    contract, capsule
                )
                if consumed_replay is not None:
                    return consumed_replay
                candidate = _derive_and_replay_candidate_v1(contract, capsule)
                replay = True
            else:
                if consumed_exists:
                    raise AmendmentTerminal(
                        "CONSUMED_CAPSULE_UNAVAILABLE",
                        str(capsule_path),
                        authorization_consumption_count=1,
                    )
                candidate = _derive_and_replay_candidate_v1(contract, capsule)
                _preconsumption_storage_gate_v1(
                    contract, digest_hint, target_paths=(capsule_path,)
                )
                _atomic_no_replace_v1(
                    capsule_path, canonical_json_bytes_v1(capsule), digest_hint
                )
                replay = False
        return {
            "schema_version": "hegel-phase3a-q1-prestart-amendment-preflight-result/1",
            "status": CAPSULE_READY,
            "contract_raw_sha256": contract.raw_sha256,
            "capsule_digest": digest_hint,
            "run_genesis_root": candidate["run_genesis_root"],
            "prestart_admission_receipt_root": candidate[
                "prestart_admission_receipt_root"
            ],
            "read_only_replay": replay,
            "authorization_recorded": False,
            "campaign_execution_count": 0,
            "authorization_consumption_count": 0,
            "q1_state": Q1_STATE,
            "q1_gate_count": 0,
            "q1_gate_total": Q1_GATE_TOTAL,
            "q1_gate_mask": 0,
            "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
            "popen_count": 0,
            "network_accessed": False,
        }
    except AmendmentPending as error:
        return _preflight_failure_result_v1(contract, digest_hint, error)
    except AmendmentTerminal as error:
        return _terminal_result_v1(contract, digest_hint, error)
    except AmendmentValidationError as error:
        return _preflight_failure_result_v1(
            contract,
            digest_hint,
            AmendmentPending(
                "PATH_AND_FORMAT", "PREFLIGHT_VALIDATION_DIFFERS", str(error)
            ),
        )
    except OSError as error:
        return _preflight_failure_result_v1(
            contract,
            digest_hint,
            AmendmentPending(
                "PATH_SPACE_AND_PROBE", "CAPSULE_PUBLICATION_FAILED", str(error)
            ),
        )


def _consumed_replay_or_terminal_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any] | None:
    fixed = contract.value["fixed_paths"]
    consumed_path = Path(fixed["consumption_path"])
    artifact_path = Path(fixed["artifact_path"])
    if not consumed_path.exists():
        return None
    try:
        _consumed_raw, consumed_value = _load_canonical_object(
            consumed_path, "consumption record"
        )
        consumed = _validate_consumption_v1(consumed_value, contract, capsule)
    except (AmendmentPending, AmendmentValidationError) as error:
        raise AmendmentTerminal(
            "CONSUMED_RECORD_MISMATCH",
            str(error),
            authorization_consumption_count=1,
        ) from error
    if not artifact_path.exists():
        raise AmendmentTerminal(
            "CONSUMED_ARTIFACT_ABSENT",
            str(artifact_path),
            authorization_consumption_count=1,
        )
    try:
        raw, value = _load_canonical_object(artifact_path, "final artifact")
        artifact = _validate_replayed_artifact_v1(
            raw, value, contract, capsule, consumed
        )
    except (AmendmentPending, AmendmentValidationError) as error:
        raise AmendmentTerminal(
            "CONSUMED_ARTIFACT_MISMATCH",
            str(error),
            authorization_consumption_count=1,
        ) from error
    return {
        **deepcopy(artifact),
        "read_only_replay": True,
        "consumption_record_digest": consumed["record_digest"],
    }


def record_authorization_v1(
    contract_path: Path,
    capsule_digest: str,
) -> dict[str, Any]:
    contract, contract_pending = _load_contract_entry_v1(
        contract_path, capsule_digest
    )
    if contract_pending is not None:
        return contract_pending
    assert contract is not None
    try:
        _raw, capsule = _load_published_capsule_v1(contract, capsule_digest)
        ledger_root = Path(contract.value["fixed_paths"]["ledger_root"])
        with _ledger_lock_v1(ledger_root):
            replay = _consumed_replay_or_terminal_v1(contract, capsule)
            if replay is not None:
                return replay
            candidate = _derive_and_replay_candidate_v1(contract, capsule)
            authorization = _authorization_v1(contract, capsule, candidate)
            authorization_path = Path(
                contract.value["fixed_paths"]["authorization_path"]
            )
            if authorization_path.exists():
                raw, value = _load_canonical_object(
                    authorization_path, "authorization record"
                )
                existing = _validate_authorization_v1(value, contract, capsule)
                if raw != canonical_json_bytes_v1(authorization) or existing != authorization:
                    raise AmendmentPending(
                        "PATH_AND_FORMAT",
                        "AUTHORIZATION_RECORD_MISMATCH",
                        "existing authorization differs for this capsule",
                    )
                replayed = True
            else:
                _preconsumption_storage_gate_v1(
                    contract, capsule_digest, target_paths=(authorization_path,)
                )
                _atomic_no_replace_v1(
                    authorization_path,
                    canonical_json_bytes_v1(authorization),
                    capsule_digest,
                )
                replayed = False
        return {
            **authorization,
            "read_only_replay": replayed,
            "campaign_execution_count": 0,
            "authorization_consumption_count": 0,
            "popen_count": 0,
            "network_accessed": False,
        }
    except AmendmentPending as error:
        return _pending_result_v1(contract, capsule_digest, error)
    except OSError as error:
        return _pending_result_v1(
            contract,
            capsule_digest,
            AmendmentPending(
                "PATH_SPACE_AND_PROBE",
                "AUTHORIZATION_PUBLICATION_FAILED",
                str(error),
            ),
        )
    except AmendmentTerminal as error:
        return _terminal_result_v1(contract, capsule_digest, error)
    except AmendmentValidationError as error:
        return _pending_result_v1(
            contract,
            capsule_digest,
            AmendmentPending(
                "PATH_AND_FORMAT", "PRECONSUMPTION_VALIDATION_DIFFERS", str(error)
            ),
        )


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, contract_pending = _load_contract_entry_v1(
        contract_path, capsule_digest
    )
    if contract_pending is not None:
        return contract_pending
    assert contract is not None
    try:
        _raw, capsule = _load_published_capsule_v1(contract, capsule_digest)
        fixed = contract.value["fixed_paths"]
        ledger_root = Path(fixed["ledger_root"])
        with _ledger_lock_v1(ledger_root):
            replay = _consumed_replay_or_terminal_v1(contract, capsule)
            if replay is not None:
                return replay
            authorization_path = Path(fixed["authorization_path"])
            if not authorization_path.exists():
                return {
                    "schema_version": "hegel-phase3a-q1-prestart-amendment-not-authorized/1",
                    "status": NOT_AUTHORIZED,
                    "contract_raw_sha256": contract.raw_sha256,
                    "capsule_digest": capsule_digest,
                    "campaign_execution_count": 0,
                    "authorization_consumption_count": 0,
                    "q1_state": Q1_STATE,
                    "q1_gate_count": 0,
                    "q1_gate_total": Q1_GATE_TOTAL,
                    "q1_gate_mask": 0,
                    "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
                    "popen_count": 0,
                    "network_accessed": False,
                }
            auth_raw, auth_value = _load_canonical_object(
                authorization_path, "authorization record"
            )
            authorization = _validate_authorization_v1(
                auth_value, contract, capsule
            )
            if auth_raw != canonical_json_bytes_v1(authorization):
                _validation("authorization record bytes differ")
            candidate = _derive_and_replay_candidate_v1(contract, capsule)
            expected_authorization = _authorization_v1(contract, capsule, candidate)
            if authorization != expected_authorization:
                _validation("authorization does not bind replayed candidate")
            artifact = _artifact_v1(contract, capsule, candidate, authorization)
            artifact_bytes = canonical_json_bytes_v1(artifact)
            consumed = _consumption_v1(
                contract, capsule, authorization, artifact_bytes, artifact
            )
            consumed_path = Path(fixed["consumption_path"])
            artifact_path = Path(fixed["artifact_path"])
            _preconsumption_storage_gate_v1(
                contract,
                capsule_digest,
                target_paths=(consumed_path, artifact_path),
            )
            try:
                _atomic_no_replace_v1(
                    consumed_path,
                    canonical_json_bytes_v1(consumed),
                    capsule_digest,
                )
            except OSError as error:
                if consumed_path.exists():
                    raise AmendmentTerminal(
                        "CONSUMPTION_PUBLICATION_UNCERTAIN",
                        str(error),
                        authorization_consumption_count=1,
                    ) from error
                raise AmendmentPending(
                    "PATH_SPACE_AND_PROBE",
                    "CONSUMPTION_PUBLICATION_FAILED_BEFORE_CONSUMPTION",
                    str(error),
                ) from error
            try:
                _atomic_no_replace_v1(artifact_path, artifact_bytes, capsule_digest)
            except OSError:
                replay_after_publication_error = _consumed_replay_or_terminal_v1(
                    contract, capsule
                )
                if replay_after_publication_error is not None:
                    return replay_after_publication_error
                raise AssertionError(
                    "consumption exists but post-publication replay returned no state"
                )
        return {
            **artifact,
            "read_only_replay": False,
            "consumption_record_digest": consumed["record_digest"],
        }
    except AmendmentPending as error:
        return _pending_result_v1(contract, capsule_digest, error)
    except AmendmentTerminal as error:
        return _terminal_result_v1(contract, capsule_digest, error)
    except AmendmentValidationError as error:
        return _pending_result_v1(
            contract,
            capsule_digest,
            AmendmentPending(
                "PATH_AND_FORMAT", "PRECONSUMPTION_VALIDATION_DIFFERS", str(error)
            ),
        )


__all__ = [
    "ACTION_ID",
    "APPLIED",
    "AUTHORIZATION_RECORDED",
    "CAPSULE_INPUTS_SCHEMA_VERSION",
    "CAPSULE_READY",
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "DEPENDENCY_IDENTITY_SCHEMA_VERSION",
    "INPUT_KEY_ABSENCE_SCHEMA_VERSION",
    "MOUNT_MANIFEST_SCHEMA_VERSION",
    "NOT_AUTHORIZED",
    "OFFLINE_ISOLATION_SCHEMA_VERSION",
    "PENDING_NOT_CONSUMED",
    "Q1PrestartAmendmentError",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "TERMINAL_NO_RESUME",
    "actor_dependency_identity_root_v1",
    "actor_offline_isolation_root_v1",
    "actor_runtime_identity_root_v1",
    "actor_source_manifest_root_v1",
    "build_capsule_v1",
    "canonical_json_bytes_v1",
    "formal_input_key_absence_root_v1",
    "load_contract_v1",
    "mount_manifest_root_v1",
    "preflight_v1",
    "record_authorization_v1",
    "run_v1",
    "source_closure_sha256_v1",
    "strict_json_bytes_v1",
    "validate_capsule_v1",
    "validate_contract_v1",
]
