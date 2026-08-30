"""Pure contract, capsule, and ledger wire for one target-blind Q1 start.

This module performs no process launch and has no network facility.  The outer
tool owns live gates and execution; this file makes the authority and durable
state transitions exact and replayable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Final, Mapping, NoReturn


CONTRACT_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-contract/1"
CONTRACT_ID: Final = "hegel-phase3a-q1-start-v1"
ACTION_ID: Final = "phase3-q1-start-v1"
CAPSULE_INPUTS_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-capsule-inputs/1"
CAPSULE_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-capsule/1"
AUTHORIZATION_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-authorization/1"
CONSUMPTION_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-consumption/1"
PENDING_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-pending/1"
TERMINAL_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-terminal/1"

CAPSULE_READY: Final = "Q1_START_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = "Q1_START_AUTHORIZATION_RECORDED_NOT_CONSUMED"
AUTHORIZATION_CONSUMED: Final = "Q1_START_AUTHORIZATION_CONSUMED"
PENDING_NOT_CONSUMED: Final = "Q1_START_PENDING_NOT_CONSUMED"
PENDING_CLEAN_CHECKPOINT: Final = "Q1_START_PENDING_AT_CLEAN_CHECKPOINT"
COMPLETE: Final = "Q1_COMPLETE_20_OF_20"
INCONCLUSIVE: Final = "Q1_START_INCONCLUSIVE_TERMINAL_NO_RESUME"
OUTCOME_UNKNOWN: Final = "Q1_START_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"

PYTHON_IMAGE: Final = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE: Final = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)

Q1_GATE_TOTAL: Final = 20
Q1_PRESTART_GATE_COUNT: Final = 14
Q1_PRESTART_GATE_MASK: Final = 0x003FFF
Q1_COMPLETE_GATE_MASK: Final = 0x0FFFFF
NULL_OUTPUT_SLOTS: Final = [None] * 8
MAX_SMALL_JSON: Final = 16 * 1024 * 1024
MAX_RUST_BINARY_BYTES: Final = 64 * 1024 * 1024

DOMAINS: Final = {
    "contract": b"HEGEL/Q1/START/CONTRACT/V1\x00",
    "source_closure": b"HEGEL/Q1/START/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q1/START/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q1/START/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q1/START/CONSUMPTION/V1\x00",
    "pending": b"HEGEL/Q1/START/PENDING/V1\x00",
    "terminal": b"HEGEL/Q1/START/TERMINAL/V1\x00",
}

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


class Q1StartCampaignValidationError(ValueError):
    pass


def _fail(detail: str) -> NoReturn:
    raise Q1StartCampaignValidationError(detail)


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON token {value!r}")


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("non-finite JSON number")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail("JSON object keys must be strings")
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
        _fail(f"canonical JSON rejected: {error}")


def strict_json_bytes_v1(
    payload: bytes,
    label: str,
    maximum_bytes: int = MAX_SMALL_JSON,
) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > maximum_bytes:
        _fail(f"{label} exceeds its byte bound")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict ASCII JSON: {error}")
    if type(value) is not dict or canonical_json_bytes_v1(value) != payload:
        _fail(f"{label} is not one canonical JSON object")
    return value


def _digest(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes_v1(value)).hexdigest()


def raw_sha256_v1(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail("raw SHA-256 input must be bytes")
    return "sha256:" + sha256(payload).hexdigest()


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be tagged lowercase SHA-256")
    return value


def _hex40(value: object, label: str) -> str:
    if type(value) is not str or _HEX40.fullmatch(value) is None:
        _fail(f"{label} must be 40 lowercase hex digits")
    return value


def _hex32(value: object, label: str) -> str:
    if type(value) is not str or _HEX32.fullmatch(value) is None:
        _fail(f"{label} must be 32 lowercase hex digits")
    return value


def _exact_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        _fail(f"{label} differs")


def _absolute_path(value: object, label: str) -> str:
    if type(value) is not str or not value or not Path(value).is_absolute():
        _fail(f"{label} must be one nonempty absolute path")
    return value


def _relative_paths(value: object, label: str) -> list[str]:
    if type(value) is not list or not value:
        _fail(f"{label} must be a nonempty list")
    checked: list[str] = []
    for path in value:
        if (
            type(path) is not str
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            _fail(f"{label} contains an invalid path")
        checked.append(path)
    if len(checked) != len(set(checked)):
        _fail(f"{label} repeats a path")
    return checked


@dataclass(frozen=True, slots=True)
class LoadedContractV1:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


def _validate_predecessor(value: object) -> dict[str, Any]:
    required = {
        "prestart_artifact_path",
        "prestart_artifact_raw_sha256",
        "prestart_capsule_path",
        "prestart_capsule_digest",
        "prestart_authorization_path",
        "prestart_authorization_digest",
        "prestart_consumption_path",
        "prestart_consumption_record_digest",
        "formal_source_commit",
        "formal_source_tree",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "actor_implementation_roots",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
    }
    if type(value) is not dict or set(value) != required:
        _fail("predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "prestart_artifact_path",
        "prestart_capsule_path",
        "prestart_authorization_path",
        "prestart_consumption_path",
    ):
        _absolute_path(checked[field], f"predecessor {field}")
    for field in (
        "prestart_artifact_raw_sha256",
        "prestart_capsule_digest",
        "prestart_authorization_digest",
        "prestart_consumption_record_digest",
        "run_genesis_root",
        "prestart_admission_receipt_root",
    ):
        _tagged(checked[field], f"predecessor {field}")
    _hex40(checked["formal_source_commit"], "formal source commit")
    _hex40(checked["formal_source_tree"], "formal source tree")
    roots = checked["actor_implementation_roots"]
    if type(roots) is not dict or set(roots) != {"host", "python", "rust"}:
        _fail("predecessor actor implementation registry differs")
    for actor, root in roots.items():
        _tagged(root, f"{actor} implementation root")
    if checked["q1_state"] != "NOT_RUN" or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS:
        _fail("predecessor Q1 state differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "predecessor gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "predecessor gate mask")
    return checked


def _validate_runtime(value: object) -> dict[str, Any]:
    required = {
        "docker_path",
        "supervisor_python_path",
        "python_image",
        "rust_image",
        "seccomp_profile_path",
        "seccomp_profile_raw_sha256",
        "python_cpuset",
        "rust_cpuset",
        "host_cpuset",
        "endpoint_memory_bytes",
        "endpoint_output_bytes",
        "endpoint_scratch_bytes",
        "endpoint_wall_seconds",
        "host_memory_bytes",
        "host_output_bytes",
        "host_scratch_bytes",
        "host_wall_seconds",
        "pids_limit",
        "minimum_available_memory_bytes",
        "minimum_available_storage_bytes",
    }
    if type(value) is not dict or set(value) != required:
        _fail("runtime registry differs")
    checked = deepcopy(value)
    if (
        checked["docker_path"] != "/usr/bin/docker"
        or checked["supervisor_python_path"] != "/usr/bin/python3.10"
        or checked["python_image"] != PYTHON_IMAGE
        or checked["rust_image"] != RUST_IMAGE
        or checked["python_cpuset"] != "0-11"
        or checked["rust_cpuset"] != "12-23"
        or checked["host_cpuset"] != "0-11"
    ):
        _fail("runtime executable, image, or CPU schedule differs")
    _absolute_path(checked["seccomp_profile_path"], "seccomp profile path")
    _tagged(checked["seccomp_profile_raw_sha256"], "seccomp profile raw SHA-256")
    expected_ints = {
        "endpoint_memory_bytes": 2_147_483_648,
        "endpoint_output_bytes": 16_777_216,
        "endpoint_scratch_bytes": 4_194_304,
        "endpoint_wall_seconds": 3600,
        "host_memory_bytes": 67_108_864,
        "host_output_bytes": 2_097_152,
        "host_scratch_bytes": 1,
        "host_wall_seconds": 3600,
        "pids_limit": 128,
        "minimum_available_memory_bytes": 4_294_967_296,
        "minimum_available_storage_bytes": 134_217_728,
    }
    for field, expected in expected_ints.items():
        _exact_int(checked[field], expected, f"runtime {field}")
    return checked


def validate_contract_v1(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "contract_id",
        "status",
        "action_id",
        "capsule_inputs_schema_version",
        "fixed_paths",
        "controller_source_closure_relative_paths",
        "predecessor",
        "runtime",
        "execution_policy",
        "authority_boundary",
    }
    if type(value) is not dict or set(value) != required:
        _fail("contract registry differs")
    contract = deepcopy(value)
    if (
        contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["contract_id"] != CONTRACT_ID
        or contract["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or contract["action_id"] != ACTION_ID
        or contract["capsule_inputs_schema_version"] != CAPSULE_INPUTS_SCHEMA_VERSION
    ):
        _fail("contract header differs")
    fixed_required = {
        "project_root",
        "campaign_lock_path",
        "run_root",
        "ledger_root",
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "checkpoint_root",
        "outcome_path",
        "terminal_path",
        "artifact_path",
        "rust_binary_path",
        "admission_path",
        "run_spec_path",
        "python_source_root",
        "rust_source_root",
        "host_source_root",
        "python_output_path",
        "python_scratch_path",
        "rust_output_path",
        "rust_scratch_path",
        "host_output_path",
        "host_scratch_path",
    }
    fixed = contract["fixed_paths"]
    if type(fixed) is not dict or set(fixed) != fixed_required:
        _fail("fixed path registry differs")
    for field, path in fixed.items():
        _absolute_path(path, f"fixed path {field}")
    if Path(fixed["capsule_path"]).parent != Path(fixed["run_root"]):
        _fail("capsule path must belong to run root")
    if Path(fixed["authorization_path"]).parent != Path(fixed["ledger_root"]):
        _fail("authorization path must belong to ledger root")
    if Path(fixed["consumption_path"]).parent != Path(fixed["ledger_root"]):
        _fail("consumption path must belong to ledger root")
    if Path(fixed["terminal_path"]).parent != Path(fixed["ledger_root"]):
        _fail("terminal path must belong to ledger root")
    run_children = (
        "checkpoint_root",
        "outcome_path",
        "admission_path",
        "run_spec_path",
        "python_source_root",
        "rust_source_root",
        "host_source_root",
        "python_output_path",
        "python_scratch_path",
        "rust_output_path",
        "rust_scratch_path",
        "host_output_path",
        "host_scratch_path",
    )
    if any(Path(fixed[field]).parent != Path(fixed["run_root"]) for field in run_children):
        _fail("run-owned path must be one direct run-root child")
    if len({fixed[field] for field in run_children}) != len(run_children):
        _fail("run-owned paths must be distinct")
    contract["controller_source_closure_relative_paths"] = _relative_paths(
        contract["controller_source_closure_relative_paths"],
        "controller source closure",
    )
    contract["predecessor"] = _validate_predecessor(contract["predecessor"])
    contract["runtime"] = _validate_runtime(contract["runtime"])
    policy = contract["execution_policy"]
    expected_policy = {
        "actor_order": ["python", "rust", "host"],
        "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "maximum_concurrently_live_actor_containers": 1,
        "clean_checkpoint_resume_allowed": True,
        "checkpoint_resume_repeats_completed_actor": False,
        "intent_or_partial_without_exact_completion_terminal_no_resume": True,
        "preconsumption_pending_consumes_authorization": False,
        "preconsumption_pending_starts_actor": False,
        "network_allowed": False,
    }
    if type(policy) is not dict or policy != expected_policy:
        _fail("execution policy differs")
    authority = contract["authority_boundary"]
    expected_authority = {
        "q1_start_authorized_by_future_exact_capsule": True,
        "q1_state_before": "NOT_RUN",
        "q1_gate_count_before": Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask_before": Q1_PRESTART_GATE_MASK,
        "q1_success_state": "COMPLETE",
        "q1_success_gate_count": Q1_GATE_TOTAL,
        "q1_success_gate_mask": Q1_COMPLETE_GATE_MASK,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "role_evaluation_performed": False,
        "outside_certificate_issued": False,
        "adoption_allowed": False,
        "promotion_allowed": False,
        "current_pointer_update_allowed": False,
        "active_transition_allowed": False,
    }
    if type(authority) is not dict or authority != expected_authority:
        _fail("authority boundary differs")
    return contract


def load_contract_v1(path: Path) -> LoadedContractV1:
    raw = path.read_bytes()
    value = validate_contract_v1(strict_json_bytes_v1(raw, "Q1 start contract"))
    return LoadedContractV1(path, raw, raw_sha256_v1(raw), value)


def source_closure_root_v1(rows: object) -> str:
    if type(rows) is not list:
        _fail("source closure rows must be a list")
    return _digest(DOMAINS["source_closure"], rows)


def _source_rows(value: object, paths: list[str]) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(paths):
        _fail("source closure row count differs")
    checked: list[dict[str, Any]] = []
    for row, expected in zip(value, paths, strict=True):
        if type(row) is not dict or set(row) != {"relative_path", "byte_count", "raw_sha256"}:
            _fail("source closure row differs")
        if (
            row["relative_path"] != expected
            or type(row["byte_count"]) is not int
            or row["byte_count"] < 0
        ):
            _fail("source closure row binding differs")
        _tagged(row["raw_sha256"], "source raw SHA-256")
        checked.append(deepcopy(row))
    return checked


def _rust_binary(value: object, expected_path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"path", "byte_count", "raw_sha256"}:
        _fail("Rust binary binding differs")
    if (
        value["path"] != expected_path
        or type(value["byte_count"]) is not int
        or not 1 <= value["byte_count"] <= MAX_RUST_BINARY_BYTES
    ):
        _fail("Rust binary path or byte count differs")
    _tagged(value["raw_sha256"], "Rust binary raw SHA-256")
    return deepcopy(value)


def build_capsule_v1(
    contract: LoadedContractV1,
    capsule_inputs: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "controller_source_identity",
        "controller_source_closure_rows",
        "rust_binary",
        "run_id",
        "created_at_unix_seconds",
    }
    if type(capsule_inputs) is not dict or set(capsule_inputs) != required:
        _fail("capsule input registry differs")
    inputs = deepcopy(capsule_inputs)
    if inputs["schema_version"] != CAPSULE_INPUTS_SCHEMA_VERSION:
        _fail("capsule input schema differs")
    source = inputs["controller_source_identity"]
    if type(source) is not dict or set(source) != {
        "source_commit", "source_tree", "source_closure_root"
    }:
        _fail("controller source identity differs")
    _hex40(source["source_commit"], "controller source commit")
    _hex40(source["source_tree"], "controller source tree")
    _tagged(source["source_closure_root"], "controller source closure root")
    rows = _source_rows(
        inputs["controller_source_closure_rows"],
        contract.value["controller_source_closure_relative_paths"],
    )
    if source_closure_root_v1(rows) != source["source_closure_root"]:
        _fail("controller source closure root differs")
    binary = _rust_binary(
        inputs["rust_binary"], contract.value["fixed_paths"]["rust_binary_path"]
    )
    _hex32(inputs["run_id"], "run ID")
    if (
        type(inputs["created_at_unix_seconds"]) is not int
        or not 0 <= inputs["created_at_unix_seconds"] <= (1 << 64) - 1
    ):
        _fail("created_at_unix_seconds must be uint64")
    predecessor = deepcopy(contract.value["predecessor"])
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": CAPSULE_READY,
        "contract_id": CONTRACT_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "action_id": ACTION_ID,
        "controller_source_identity": source,
        "controller_source_closure_rows": rows,
        "formal_source_commit": predecessor["formal_source_commit"],
        "formal_source_tree": predecessor["formal_source_tree"],
        "predecessor": predecessor,
        "runtime": deepcopy(contract.value["runtime"]),
        "execution_policy": deepcopy(contract.value["execution_policy"]),
        "fixed_paths": deepcopy(contract.value["fixed_paths"]),
        "rust_binary": binary,
        "run_id": inputs["run_id"],
        "created_at_unix_seconds": inputs["created_at_unix_seconds"],
        "q1_state_before": "NOT_RUN",
        "q1_gate_count_before": Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask_before": Q1_PRESTART_GATE_MASK,
        "q1_output_slots_before": deepcopy(NULL_OUTPUT_SLOTS),
        "intended_success_state": "COMPLETE",
        "intended_success_gate_count": Q1_GATE_TOTAL,
        "intended_success_gate_mask": Q1_COMPLETE_GATE_MASK,
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule_v1(value: object, contract: LoadedContractV1) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("capsule must be an object")
    capsule = deepcopy(value)
    digest = capsule.get("capsule_digest")
    _tagged(digest, "capsule digest")
    reconstructed_inputs = {
        "schema_version": CAPSULE_INPUTS_SCHEMA_VERSION,
        "controller_source_identity": capsule.get("controller_source_identity"),
        "controller_source_closure_rows": capsule.get("controller_source_closure_rows"),
        "rust_binary": capsule.get("rust_binary"),
        "run_id": capsule.get("run_id"),
        "created_at_unix_seconds": capsule.get("created_at_unix_seconds"),
    }
    expected = build_capsule_v1(contract, reconstructed_inputs)
    if capsule != expected or digest != expected["capsule_digest"]:
        _fail("capsule replay differs")
    return capsule


def build_authorization_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_capsule_v1(dict(capsule), contract)
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": validated["capsule_digest"],
        "action_id": ACTION_ID,
        "authority_role": "PROJECT_OWNER",
        "authorized_scope": "TARGET_BLIND_FULL_NODE6_Q1_START_ONCE",
        "authorization_consumption_limit": 1,
        "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "clean_checkpoint_resume_allowed": True,
        "completed_actor_remeasurement_allowed": False,
        "network_allowed": False,
        "q1_start_authorized": True,
        "q2_start_authorized": False,
        "adoption_promotion_current_active_authorized": False,
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def validate_authorization_v1(
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization_v1(contract, capsule)
    if value != expected:
        _fail("authorization replay differs")
    return deepcopy(expected)


def build_consumption_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    checked_capsule = validate_capsule_v1(dict(capsule), contract)
    checked_authorization = validate_authorization_v1(
        dict(authorization), contract, checked_capsule
    )
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "status": AUTHORIZATION_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
        "q1_state_at_consumption": "NOT_RUN",
        "q1_gate_count_at_consumption": Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask_at_consumption": Q1_PRESTART_GATE_MASK,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def validate_consumption_v1(
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption_v1(contract, capsule, authorization)
    if value != expected:
        _fail("consumption replay differs")
    return deepcopy(expected)


def build_pending_record_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    condition: str,
    detail: str,
    authorization_digest: str | None = None,
    consumption_record_digest: str | None = None,
    completed_actors: list[str] | None = None,
) -> dict[str, Any]:
    checked_capsule = validate_capsule_v1(dict(capsule), contract)
    if status not in {PENDING_NOT_CONSUMED, PENDING_CLEAN_CHECKPOINT}:
        _fail("pending status differs")
    if any(type(item) is not str or not item for item in (phase, condition, detail)):
        _fail("pending phase, condition, and detail must be nonempty strings")
    completed = list(completed_actors or [])
    expected_authorization = build_authorization_v1(contract, checked_capsule)
    expected_consumption = build_consumption_v1(
        contract,
        checked_capsule,
        expected_authorization,
    )
    if status == PENDING_NOT_CONSUMED:
        if completed or consumption_record_digest is not None:
            _fail("pre-consumption PENDING cannot bind completed work")
        if authorization_digest is not None:
            _tagged(authorization_digest, "pending authorization digest")
            if authorization_digest != expected_authorization["authorization_digest"]:
                _fail("pending authorization digest differs")
        consumption_count = 0
        campaign_count = 0
    else:
        if completed not in ([], ["python"], ["python", "rust"]):
            _fail("clean-checkpoint PENDING must bind a clean actor prefix")
        _tagged(authorization_digest, "pending authorization digest")
        _tagged(consumption_record_digest, "pending consumption record digest")
        if (
            authorization_digest != expected_authorization["authorization_digest"]
            or consumption_record_digest != expected_consumption["record_digest"]
        ):
            _fail("clean-checkpoint ledger chain differs")
        consumption_count = 1
        campaign_count = 1
    starts = {
        "host": int("host" in completed),
        "python": int("python" in completed),
        "rust": int("rust" in completed),
    }
    body: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": status,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": authorization_digest,
        "consumption_record_digest": consumption_record_digest,
        "authorization_consumption_count": consumption_count,
        "logical_campaign_count": campaign_count,
        "completed_actors": completed,
        "actor_start_count_by_actor": starts,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "q1_state": "NOT_RUN",
        "q1_gate_count": Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask": Q1_PRESTART_GATE_MASK,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_pending_record_v1(
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("pending record must be an object")
    expected = build_pending_record_v1(
        contract,
        capsule,
        status=value.get("status"),
        phase=value.get("phase"),
        condition=value.get("condition"),
        detail=value.get("detail"),
        authorization_digest=value.get("authorization_digest"),
        consumption_record_digest=value.get("consumption_record_digest"),
        completed_actors=value.get("completed_actors"),
    )
    if value != expected:
        _fail("pending record replay differs")
    return deepcopy(expected)


def build_terminal_record_v1(
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    artifact_raw_sha256: str | None,
) -> dict[str, Any]:
    checked_capsule = validate_capsule_v1(dict(capsule), contract)
    checked_authorization = validate_authorization_v1(
        dict(authorization), contract, checked_capsule
    )
    if type(outcome) is not dict:
        _fail("terminal outcome must be an object")
    status = outcome.get("status")
    if status not in {COMPLETE, INCONCLUSIVE, OUTCOME_UNKNOWN}:
        _fail("terminal outcome status differs")
    if outcome.get("capsule_digest") != checked_capsule["capsule_digest"]:
        _fail("terminal outcome capsule differs")
    if (
        type(outcome.get("actor_start_count_by_actor")) is not dict
        or set(outcome["actor_start_count_by_actor"])
        != {"host", "python", "rust"}
    ):
        _fail("terminal actor start registry differs")
    for actor in ("python", "rust", "host"):
        count = outcome["actor_start_count_by_actor"].get(actor)
        if type(count) is not int or not 0 <= count <= 1:
            _fail("terminal actor start count differs")
    starts = outcome["actor_start_count_by_actor"]
    if starts not in (
        {"host": 0, "python": 0, "rust": 0},
        {"host": 0, "python": 1, "rust": 0},
        {"host": 0, "python": 1, "rust": 1},
        {"host": 1, "python": 1, "rust": 1},
    ):
        _fail("terminal actor starts are not a strict sequential prefix")
    if outcome.get("network_accessed") is not False:
        _fail("terminal network boundary differs")
    if (
        outcome.get("target_truth_accessed") is not False
        or outcome.get("split_accessed") is not False
        or outcome.get("role_evaluation_performed") is not False
    ):
        _fail("terminal target-blind authority differs")
    if status == COMPLETE:
        _tagged(artifact_raw_sha256, "terminal artifact raw SHA-256")
        if (
            outcome["actor_start_count_by_actor"]
            != {"host": 1, "python": 1, "rust": 1}
            or outcome.get("q1_state") != "COMPLETE"
            or outcome.get("q1_gate_count") != Q1_GATE_TOTAL
            or outcome.get("q1_gate_mask") != Q1_COMPLETE_GATE_MASK
            or type(outcome.get("q1_output_slots")) is not list
            or len(outcome["q1_output_slots"]) != 8
            or any(
                type(item) is not str or _SHA256.fullmatch(item) is None
                for item in outcome["q1_output_slots"]
            )
        ):
            _fail("successful terminal Q1 authority differs")
    else:
        if artifact_raw_sha256 is not None:
            _fail("non-success terminal cannot bind a final artifact")
        if (
            outcome.get("q1_state") != "NOT_RUN"
            or outcome.get("q1_gate_count") != Q1_PRESTART_GATE_COUNT
            or outcome.get("q1_gate_mask") != Q1_PRESTART_GATE_MASK
            or outcome.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        ):
            _fail("non-success terminal cannot publish Q1 authority")
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": status,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": build_consumption_v1(
            contract,
            checked_capsule,
            checked_authorization,
        )["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "artifact_raw_sha256": artifact_raw_sha256,
        "outcome": deepcopy(dict(outcome)),
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["terminal"], unsigned)
    return body


def validate_terminal_record_v1(
    value: object,
    contract: LoadedContractV1,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "status",
        "contract_raw_sha256",
        "capsule_digest",
        "authorization_digest",
        "consumption_record_digest",
        "authorization_consumption_count",
        "logical_campaign_count",
        "artifact_raw_sha256",
        "outcome",
        "record_digest",
    }:
        _fail("terminal record registry differs")
    observed = _tagged(value["record_digest"], "terminal record digest")
    expected = build_terminal_record_v1(
        contract,
        capsule,
        authorization,
        value["outcome"],
        artifact_raw_sha256=value["artifact_raw_sha256"],
    )
    if value != expected or observed != expected["record_digest"]:
        _fail("terminal record replay differs")
    return deepcopy(expected)


__all__ = [
    "ACTION_ID",
    "AUTHORIZATION_CONSUMED",
    "AUTHORIZATION_RECORDED",
    "AUTHORIZATION_SCHEMA_VERSION",
    "CAPSULE_INPUTS_SCHEMA_VERSION",
    "CAPSULE_READY",
    "CAPSULE_SCHEMA_VERSION",
    "COMPLETE",
    "CONSUMPTION_SCHEMA_VERSION",
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "INCONCLUSIVE",
    "LoadedContractV1",
    "OUTCOME_UNKNOWN",
    "PENDING_CLEAN_CHECKPOINT",
    "PENDING_NOT_CONSUMED",
    "PENDING_SCHEMA_VERSION",
    "PYTHON_IMAGE",
    "Q1StartCampaignValidationError",
    "RUST_IMAGE",
    "TERMINAL_SCHEMA_VERSION",
    "build_authorization_v1",
    "build_capsule_v1",
    "build_consumption_v1",
    "build_pending_record_v1",
    "build_terminal_record_v1",
    "canonical_json_bytes_v1",
    "load_contract_v1",
    "raw_sha256_v1",
    "source_closure_root_v1",
    "strict_json_bytes_v1",
    "validate_authorization_v1",
    "validate_capsule_v1",
    "validate_consumption_v1",
    "validate_contract_v1",
    "validate_terminal_record_v1",
    "validate_pending_record_v1",
]
