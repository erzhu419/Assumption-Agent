"""Pure wire for one target-blind Q1-start successor campaign.

The v7 authority is intentionally separate from the terminal v6 campaign.  A
v7 capsule binds the authoritative 14/20 base, the immutable v2/v3/v4/v5/v6
terminal lineage, and one complete, pure successor-prestart bundle.  This module
launches no process and has no network facility.
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

from hegel_machine import phase3_q1_prestart_successor_candidate_v7 as _successor


CONTRACT_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-contract/7"
CONTRACT_ID: Final = "hegel-phase3a-q1-start-v7"
ACTION_ID: Final = "phase3-q1-start-v7"
CAPSULE_INPUTS_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-capsule-inputs/7"
CAPSULE_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-capsule/7"
AUTHORIZATION_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-authorization/7"
CONSUMPTION_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-consumption/7"
PENDING_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-pending/7"
TERMINAL_SCHEMA_VERSION: Final = "hegel-phase3a-q1-start-terminal/7"

CAPSULE_READY: Final = "Q1_START_V7_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = "Q1_START_V7_AUTHORIZATION_RECORDED_NOT_CONSUMED"
AUTHORIZATION_CONSUMED: Final = "Q1_START_V7_AUTHORIZATION_CONSUMED"
PENDING_NOT_CONSUMED: Final = "Q1_START_V7_PENDING_NOT_CONSUMED"
PENDING_CLEAN_CHECKPOINT: Final = "Q1_START_V7_PENDING_AT_CLEAN_CHECKPOINT"
COMPLETE: Final = "Q1_COMPLETE_20_OF_20"
INCONCLUSIVE: Final = "Q1_START_V7_INCONCLUSIVE_TERMINAL_NO_RESUME"
OUTCOME_UNKNOWN: Final = "Q1_START_V7_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"

FAILED_V2_STATUS: Final = "Q1_START_V2_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
FAILED_V2_CODE: Final = "Q1_START_SUPERVISOR_OUTCOME_UNKNOWN"
FAILED_V3_STATUS: Final = "Q1_START_V3_INCONCLUSIVE_TERMINAL_NO_RESUME"
FAILED_V3_CODE: Final = "REJECT_Q1_START_SUCCESSOR_BUNDLE"
FAILED_V3_DETAIL: Final = (
    "validator import failed: AttributeError: module 'importlib' has no attribute 'util'"
)
FAILED_V4_STATUS: Final = "Q1_START_V4_INCONCLUSIVE_TERMINAL_NO_RESUME"
FAILED_V4_CODE: Final = "Q1_START_ENDPOINT_REPLAY_MISMATCH"
FAILED_V4_DETAIL: Final = (
    "python:Q1FormalHostReplayError:REJECT_Q1_FORMAL_ENDPOINT_RESULT: "
    "same_run_record_set_root differs from replay"
)
FAILED_V5_STATUS: Final = "Q1_START_V5_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
FAILED_V5_CODE: Final = "Q1_START_POST_INTENT_EXCEPTION"
FAILED_V5_DETAIL: Final = (
    "PermissionError: [Errno 1] Operation not permitted: "
    "'/home/erzhu419/.local/state/hegel-machine/q1-start-v5-run/"
    "python-output/signature-02/stream-04/chunk-000000.framed'"
)
FAILED_V6_STATUS: Final = "Q1_START_V6_INCONCLUSIVE_TERMINAL_NO_RESUME"
FAILED_V6_CODE: Final = "Q1_START_ACTOR_NONZERO"
FAILED_V6_DETAIL: Final = "host:137"
AUTHORIZED_SCOPE: Final = (
    "TARGET_BLIND_Q1_START_V7_ONCE_WITH_BOUND_SUCCESSOR_PRESTART_BUNDLE"
)

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
ENDPOINT_SCRATCH_BYTES: Final = 134_217_728
NULL_OUTPUT_SLOTS: Final = [None] * 8
MAX_SMALL_JSON: Final = 16 * 1024 * 1024
MAX_RUST_BINARY_BYTES: Final = 64 * 1024 * 1024

DOMAINS: Final = {
    "contract": b"HEGEL/Q1/START/CONTRACT/V7\x00",
    "source_closure": b"HEGEL/Q1/START/SOURCE_CLOSURE/V7\x00",
    "capsule": b"HEGEL/Q1/START/CAPSULE/V7\x00",
    "authorization": b"HEGEL/Q1/START/AUTHORIZATION/V7\x00",
    "consumption": b"HEGEL/Q1/START/CONSUMPTION/V7\x00",
    "pending": b"HEGEL/Q1/START/PENDING/V7\x00",
    "terminal": b"HEGEL/Q1/START/TERMINAL/V7\x00",
}

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


class Q1StartCampaignV7ValidationError(ValueError):
    pass


def _fail(detail: str) -> NoReturn:
    raise Q1StartCampaignV7ValidationError(detail)


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


def canonical_json_bytes_v7(value: object) -> bytes:
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


def strict_json_bytes_v7(
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
    if type(value) is not dict or canonical_json_bytes_v7(value) != payload:
        _fail(f"{label} is not one canonical JSON object")
    return value


def _digest(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes_v7(value)).hexdigest()


def raw_sha256_v7(payload: bytes) -> str:
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


def _under(path: str, root: str) -> bool:
    candidate = Path(path)
    ancestor = Path(root)
    return candidate == ancestor or ancestor in candidate.parents


def _actor_roots(value: object, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {"host", "python", "rust"}:
        _fail(f"{label} registry differs")
    checked = deepcopy(value)
    for actor, root in checked.items():
        _tagged(root, f"{label} {actor}")
    return checked


@dataclass(frozen=True, slots=True)
class LoadedContractV7:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


def _validate_base_authoritative_predecessor(value: object) -> dict[str, Any]:
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
        _fail("base authoritative predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "prestart_artifact_path",
        "prestart_capsule_path",
        "prestart_authorization_path",
        "prestart_consumption_path",
    ):
        _absolute_path(checked[field], f"base predecessor {field}")
    for field in (
        "prestart_artifact_raw_sha256",
        "prestart_capsule_digest",
        "prestart_authorization_digest",
        "prestart_consumption_record_digest",
        "run_genesis_root",
        "prestart_admission_receipt_root",
    ):
        _tagged(checked[field], f"base predecessor {field}")
    _hex40(checked["formal_source_commit"], "base formal source commit")
    _hex40(checked["formal_source_tree"], "base formal source tree")
    checked["actor_implementation_roots"] = _actor_roots(
        checked["actor_implementation_roots"], "base actor implementation root"
    )
    if checked["q1_state"] != "NOT_RUN" or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS:
        _fail("base authoritative Q1 state differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "base gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "base gate mask")
    return checked


def _validate_failed_start_v2_predecessor(value: object) -> dict[str, Any]:
    required = {
        "run_root",
        "ledger_root",
        "capsule_path",
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_path",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_path",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_path",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "status",
        "failure_code",
        "failure_detail",
        "actor_start_count_by_actor",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
        "terminal_no_resume",
        "prior_actor_actions_conservatively_counted",
        "work_product_reuse_allowed",
    }
    if type(value) is not dict or set(value) != required:
        _fail("failed Q1-start-v2 predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "run_root",
        "ledger_root",
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "terminal_path",
    ):
        _absolute_path(checked[field], f"failed-v2 {field}")
    if Path(checked["capsule_path"]).parent != Path(checked["run_root"]):
        _fail("failed-v2 capsule path must belong to its run root")
    if Path(checked["authorization_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v2 authorization path must belong to its ledger root")
    if Path(checked["consumption_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v2 consumption path must belong to its ledger root")
    if Path(checked["terminal_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v2 terminal path must belong to its ledger root")
    for field in (
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_raw_sha256",
        "terminal_record_digest",
    ):
        _tagged(checked[field], f"failed-v2 {field}")
    if (
        checked["capsule_digest"] != _successor.Q1_START_V2_CAPSULE_DIGEST
        or checked["capsule_raw_sha256"] != _successor.Q1_START_V2_CAPSULE_RAW_SHA256
        or checked["authorization_digest"]
        != _successor.Q1_START_V2_AUTHORIZATION_DIGEST
        or checked["authorization_raw_sha256"]
        != _successor.Q1_START_V2_AUTHORIZATION_RAW_SHA256
        or checked["consumption_record_digest"]
        != _successor.Q1_START_V2_CONSUMPTION_RECORD_DIGEST
        or checked["consumption_raw_sha256"]
        != _successor.Q1_START_V2_CONSUMPTION_RAW_SHA256
        or checked["terminal_raw_sha256"]
        != _successor.Q1_START_V2_TERMINAL_RAW_SHA256
        or checked["terminal_record_digest"]
        != _successor.Q1_START_V2_TERMINAL_RECORD_DIGEST
        or checked["status"] != FAILED_V2_STATUS
        or checked["failure_code"] != FAILED_V2_CODE
        or type(checked["failure_detail"]) is not str
        or checked["actor_start_count_by_actor"]
        != {"host": 0, "python": 0, "rust": 0}
        or checked["q1_state"] != "NOT_RUN"
        or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or checked["terminal_no_resume"] is not True
        or checked["prior_actor_actions_conservatively_counted"] is not False
        or checked["work_product_reuse_allowed"] is not False
    ):
        _fail("failed Q1-start-v2 terminal boundary differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "failed-v2 gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "failed-v2 gate mask")
    return checked



def _validate_failed_start_v3_predecessor(value: object) -> dict[str, Any]:
    required = {
        "run_root",
        "ledger_root",
        "capsule_path",
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_path",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_path",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_path",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "status",
        "failure_code",
        "failure_detail",
        "actor_start_count_by_actor",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
        "terminal_no_resume",
        "prior_actor_actions_conservatively_counted",
        "work_product_reuse_allowed",
    }
    if type(value) is not dict or set(value) != required:
        _fail("failed Q1-start-v3 predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "run_root",
        "ledger_root",
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "terminal_path",
    ):
        _absolute_path(checked[field], f"failed-v3 {field}")
    if Path(checked["capsule_path"]).parent != Path(checked["run_root"]):
        _fail("failed-v3 capsule path must belong to its run root")
    if Path(checked["authorization_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v3 authorization path must belong to its ledger root")
    if Path(checked["consumption_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v3 consumption path must belong to its ledger root")
    if Path(checked["terminal_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v3 terminal path must belong to its ledger root")
    for field in (
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_raw_sha256",
        "terminal_record_digest",
    ):
        _tagged(checked[field], f"failed-v3 {field}")
    if (
        checked["capsule_digest"] != _successor.Q1_START_V3_CAPSULE_DIGEST
        or checked["capsule_raw_sha256"] != _successor.Q1_START_V3_CAPSULE_RAW_SHA256
        or checked["authorization_digest"]
        != _successor.Q1_START_V3_AUTHORIZATION_DIGEST
        or checked["authorization_raw_sha256"]
        != _successor.Q1_START_V3_AUTHORIZATION_RAW_SHA256
        or checked["consumption_record_digest"]
        != _successor.Q1_START_V3_CONSUMPTION_RECORD_DIGEST
        or checked["consumption_raw_sha256"]
        != _successor.Q1_START_V3_CONSUMPTION_RAW_SHA256
        or checked["terminal_raw_sha256"]
        != _successor.Q1_START_V3_TERMINAL_RAW_SHA256
        or checked["terminal_record_digest"]
        != _successor.Q1_START_V3_TERMINAL_RECORD_DIGEST
        or checked["status"] != FAILED_V3_STATUS
        or checked["failure_code"] != FAILED_V3_CODE
        or checked["failure_detail"] != FAILED_V3_DETAIL
        or checked["actor_start_count_by_actor"]
        != {"host": 0, "python": 0, "rust": 0}
        or checked["q1_state"] != "NOT_RUN"
        or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or checked["terminal_no_resume"] is not True
        or checked["prior_actor_actions_conservatively_counted"] is not False
        or checked["work_product_reuse_allowed"] is not False
    ):
        _fail("failed Q1-start-v3 terminal boundary differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "failed-v3 gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "failed-v3 gate mask")
    return checked


def _validate_failed_start_v4_predecessor(value: object) -> dict[str, Any]:
    required = {
        "run_root",
        "ledger_root",
        "capsule_path",
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_path",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_path",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_path",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "status",
        "failure_code",
        "failure_detail",
        "actor_start_count_by_actor",
        "completed_prefix",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
        "terminal_no_resume",
        "prior_actor_actions_conservatively_counted",
        "work_product_reuse_allowed",
    }
    if type(value) is not dict or set(value) != required:
        _fail("failed Q1-start-v4 predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "run_root",
        "ledger_root",
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "terminal_path",
    ):
        _absolute_path(checked[field], f"failed-v4 {field}")
    if Path(checked["capsule_path"]).parent != Path(checked["run_root"]):
        _fail("failed-v4 capsule path must belong to its run root")
    if Path(checked["authorization_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v4 authorization path must belong to its ledger root")
    if Path(checked["consumption_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v4 consumption path must belong to its ledger root")
    if Path(checked["terminal_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v4 terminal path must belong to its ledger root")
    for field in (
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_raw_sha256",
        "terminal_record_digest",
    ):
        _tagged(checked[field], f"failed-v4 {field}")
    if (
        checked["capsule_digest"] != _successor.Q1_START_V4_CAPSULE_DIGEST
        or checked["capsule_raw_sha256"]
        != _successor.Q1_START_V4_CAPSULE_RAW_SHA256
        or checked["authorization_digest"]
        != _successor.Q1_START_V4_AUTHORIZATION_DIGEST
        or checked["authorization_raw_sha256"]
        != _successor.Q1_START_V4_AUTHORIZATION_RAW_SHA256
        or checked["consumption_record_digest"]
        != _successor.Q1_START_V4_CONSUMPTION_RECORD_DIGEST
        or checked["consumption_raw_sha256"]
        != _successor.Q1_START_V4_CONSUMPTION_RAW_SHA256
        or checked["terminal_raw_sha256"]
        != _successor.Q1_START_V4_TERMINAL_RAW_SHA256
        or checked["terminal_record_digest"]
        != _successor.Q1_START_V4_TERMINAL_RECORD_DIGEST
        or checked["status"] != FAILED_V4_STATUS
        or checked["failure_code"] != FAILED_V4_CODE
        or checked["failure_detail"] != FAILED_V4_DETAIL
        or checked["actor_start_count_by_actor"]
        != {"host": 0, "python": 1, "rust": 0}
        or checked["completed_prefix"] != []
        or checked["q1_state"] != "NOT_RUN"
        or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or checked["terminal_no_resume"] is not True
        or checked["prior_actor_actions_conservatively_counted"] is not True
        or checked["work_product_reuse_allowed"] is not False
    ):
        _fail("failed Q1-start-v4 terminal boundary differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "failed-v4 gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "failed-v4 gate mask")
    return checked


def _validate_failed_start_v5_predecessor(value: object) -> dict[str, Any]:
    required = {
        "run_root",
        "ledger_root",
        "capsule_path",
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_path",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_path",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_path",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "supervisor_outcome_raw_sha256",
        "status",
        "failure_code",
        "failure_detail",
        "actor_start_count_by_actor",
        "completed_prefix",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
        "terminal_no_resume",
        "prior_actor_actions_conservatively_counted",
        "work_product_reuse_allowed",
    }
    if type(value) is not dict or set(value) != required:
        _fail("failed Q1-start-v5 predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "run_root",
        "ledger_root",
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "terminal_path",
    ):
        _absolute_path(checked[field], f"failed-v5 {field}")
    if Path(checked["capsule_path"]).parent != Path(checked["run_root"]):
        _fail("failed-v5 capsule path must belong to its run root")
    if Path(checked["authorization_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v5 authorization path must belong to its ledger root")
    if Path(checked["consumption_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v5 consumption path must belong to its ledger root")
    if Path(checked["terminal_path"]).parent != Path(checked["ledger_root"]):
        _fail("failed-v5 terminal path must belong to its ledger root")
    for field in (
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "supervisor_outcome_raw_sha256",
    ):
        _tagged(checked[field], f"failed-v5 {field}")
    if (
        checked["capsule_digest"] != _successor.Q1_START_V5_CAPSULE_DIGEST
        or checked["capsule_raw_sha256"]
        != _successor.Q1_START_V5_CAPSULE_RAW_SHA256
        or checked["authorization_digest"]
        != _successor.Q1_START_V5_AUTHORIZATION_DIGEST
        or checked["authorization_raw_sha256"]
        != _successor.Q1_START_V5_AUTHORIZATION_RAW_SHA256
        or checked["consumption_record_digest"]
        != _successor.Q1_START_V5_CONSUMPTION_RECORD_DIGEST
        or checked["consumption_raw_sha256"]
        != _successor.Q1_START_V5_CONSUMPTION_RAW_SHA256
        or checked["terminal_raw_sha256"]
        != _successor.Q1_START_V5_TERMINAL_RAW_SHA256
        or checked["terminal_record_digest"]
        != _successor.Q1_START_V5_TERMINAL_RECORD_DIGEST
        or checked["supervisor_outcome_raw_sha256"]
        != _successor.Q1_START_V5_SUPERVISOR_OUTCOME_RAW_SHA256
        or checked["status"] != FAILED_V5_STATUS
        or checked["failure_code"] != FAILED_V5_CODE
        or checked["failure_detail"] != FAILED_V5_DETAIL
        or checked["actor_start_count_by_actor"]
        != {"host": 0, "python": 1, "rust": 0}
        or checked["completed_prefix"] != []
        or checked["q1_state"] != "NOT_RUN"
        or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or checked["terminal_no_resume"] is not True
        or checked["prior_actor_actions_conservatively_counted"] is not True
        or checked["work_product_reuse_allowed"] is not False
    ):
        _fail("failed Q1-start-v5 terminal boundary differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "failed-v5 gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "failed-v5 gate mask")
    return checked


def _validate_failed_start_v6_predecessor(value: object) -> dict[str, Any]:
    required = {
        "run_root",
        "ledger_root",
        "capsule_path",
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_path",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_path",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "supervisor_outcome_path",
        "supervisor_outcome_raw_sha256",
        "controller_outcome_path",
        "controller_outcome_raw_sha256",
        "terminal_path",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "oom_event_receipt_path",
        "oom_event_receipt_raw_sha256",
        "status",
        "failure_code",
        "failure_detail",
        "actor_start_count_by_actor",
        "completed_prefix",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
        "terminal_no_resume",
        "prior_actor_actions_conservatively_counted",
        "resume_allowed",
        "work_product_reuse_allowed",
        "diagnostic_host_resource_peak_reuse_allowed",
    }
    if type(value) is not dict or set(value) != required:
        _fail("failed Q1-start-v6 predecessor registry differs")
    checked = deepcopy(value)
    for field in (
        "run_root",
        "ledger_root",
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "supervisor_outcome_path",
        "controller_outcome_path",
        "terminal_path",
        "oom_event_receipt_path",
    ):
        _absolute_path(checked[field], f"failed-v6 {field}")
    run_root = Path(checked["run_root"])
    ledger_root = Path(checked["ledger_root"])
    if run_root != Path(
        "/home/erzhu419/.local/state/hegel-machine/q1-start-v6-run"
    ) or ledger_root != Path(
        "/home/erzhu419/.local/state/hegel-machine/q1-start-v6-ledger"
    ):
        _fail("failed-v6 run or ledger root differs")
    if Path(checked["capsule_path"]).parent != run_root:
        _fail("failed-v6 capsule path must belong to its run root")
    if Path(checked["authorization_path"]).parent != ledger_root:
        _fail("failed-v6 authorization path must belong to its ledger root")
    if Path(checked["consumption_path"]).parent != ledger_root:
        _fail("failed-v6 consumption path must belong to its ledger root")
    if Path(checked["terminal_path"]).parent != ledger_root:
        _fail("failed-v6 terminal path must belong to its ledger root")
    if Path(checked["supervisor_outcome_path"]) != Path(
        _successor.Q1_START_V6_SUPERVISOR_OUTCOME_PATH
    ):
        _fail("failed-v6 supervisor outcome path differs")
    if Path(checked["controller_outcome_path"]) != Path(
        _successor.Q1_START_V6_CONTROLLER_OUTCOME_PATH
    ):
        _fail("failed-v6 controller outcome path differs")
    if Path(checked["oom_event_receipt_path"]) != Path(
        _successor.Q1_START_V6_OOM_EVENT_RECEIPT_PATH
    ):
        _fail("failed-v6 OOM event receipt path differs")
    for field in (
        "capsule_digest",
        "capsule_raw_sha256",
        "authorization_digest",
        "authorization_raw_sha256",
        "consumption_record_digest",
        "consumption_raw_sha256",
        "supervisor_outcome_raw_sha256",
        "controller_outcome_raw_sha256",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "oom_event_receipt_raw_sha256",
    ):
        _tagged(checked[field], f"failed-v6 {field}")
    if (
        checked["capsule_digest"] != _successor.Q1_START_V6_CAPSULE_DIGEST
        or checked["capsule_raw_sha256"]
        != _successor.Q1_START_V6_CAPSULE_RAW_SHA256
        or checked["authorization_digest"]
        != _successor.Q1_START_V6_AUTHORIZATION_DIGEST
        or checked["authorization_raw_sha256"]
        != _successor.Q1_START_V6_AUTHORIZATION_RAW_SHA256
        or checked["consumption_record_digest"]
        != _successor.Q1_START_V6_CONSUMPTION_RECORD_DIGEST
        or checked["consumption_raw_sha256"]
        != _successor.Q1_START_V6_CONSUMPTION_RAW_SHA256
        or checked["supervisor_outcome_raw_sha256"]
        != _successor.Q1_START_V6_SUPERVISOR_OUTCOME_RAW_SHA256
        or checked["controller_outcome_raw_sha256"]
        != _successor.Q1_START_V6_CONTROLLER_OUTCOME_RAW_SHA256
        or checked["terminal_raw_sha256"]
        != _successor.Q1_START_V6_TERMINAL_RAW_SHA256
        or checked["terminal_record_digest"]
        != _successor.Q1_START_V6_TERMINAL_RECORD_DIGEST
        or checked["oom_event_receipt_raw_sha256"]
        != _successor.Q1_START_V6_OOM_EVENT_RECEIPT_RAW_SHA256
        or checked["status"] != FAILED_V6_STATUS
        or checked["failure_code"] != FAILED_V6_CODE
        or checked["failure_detail"] != FAILED_V6_DETAIL
        or checked["actor_start_count_by_actor"]
        != {"host": 1, "python": 1, "rust": 1}
        or checked["completed_prefix"] != ["python", "rust"]
        or checked["q1_state"] != "NOT_RUN"
        or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or checked["terminal_no_resume"] is not True
        or checked["prior_actor_actions_conservatively_counted"] is not True
        or checked["resume_allowed"] is not False
        or checked["work_product_reuse_allowed"] is not False
        or checked["diagnostic_host_resource_peak_reuse_allowed"] is not False
    ):
        _fail("failed Q1-start-v6 terminal boundary differs")
    _exact_int(checked["q1_gate_count"], Q1_PRESTART_GATE_COUNT, "failed-v6 gate count")
    _exact_int(checked["q1_gate_mask"], Q1_PRESTART_GATE_MASK, "failed-v6 gate mask")
    return checked


_SUCCESSOR_POLICY_FIELDS: Final = {
    "path",
    "schema_version",
    "status",
    "canonical_json_required",
    "endpoint_scratch_bytes",
    "authorized",
    "authorization_consumption_count",
    "campaign_execution_count",
    "popen_count",
}


def _validate_successor_descriptor(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _SUCCESSOR_POLICY_FIELDS:
        _fail("successor-prestart bundle policy descriptor registry differs")
    checked = deepcopy(value)
    _absolute_path(checked["path"], "successor-prestart bundle path")
    if (
        checked["schema_version"] != _successor.SCHEMA_VERSION
        or checked["status"] != _successor.STATUS
        or checked["canonical_json_required"] is not True
        or checked["authorized"] is not False
    ):
        _fail("successor-prestart bundle policy header differs")
    _exact_int(
        checked["endpoint_scratch_bytes"],
        ENDPOINT_SCRATCH_BYTES,
        "successor endpoint scratch bytes",
    )
    for field in (
        "authorization_consumption_count",
        "campaign_execution_count",
        "popen_count",
    ):
        _exact_int(checked[field], 0, f"successor policy {field}")
    return checked


def _bundle_descriptor_projection(
    bundle: Mapping[str, Any],
    descriptor_path: str,
) -> dict[str, Any]:
    source = bundle["source_identity"]
    base_chain = bundle["base_chain"]
    raw = _successor.canonical_json_bytes_v7(dict(bundle))
    return {
        "path": descriptor_path,
        "schema_version": bundle["schema_version"],
        "status": bundle["status"],
        "byte_count": len(raw),
        "raw_sha256": raw_sha256_v7(raw),
        "bundle_root": _successor.successor_bundle_root_v7(dict(bundle)),
        "receipt_root": bundle["receipt_root"],
        "run_genesis_root": bundle["run_genesis_root"],
        "source_commit": source["source_commit"],
        "source_tree": source["source_tree"],
        "source_closure_root": source["source_closure_root"],
        "actor_source_closure_root": bundle["actor_source_closure_root"],
        "actor_source_manifest_roots": bundle["actor_source_manifest_roots"],
        "actor_implementation_roots": bundle["actor_implementation_roots"],
        "actor_offline_isolation_roots": bundle["actor_offline_isolation_roots"],
        "formal_output_layout_root": bundle["formal_output_layout_root"],
        "operator_owned_output_freezer_compatibility_descriptor": bundle[
            "operator_owned_output_freezer_compatibility_descriptor"
        ],
        "corrected_resource_envelope_root": bundle["corrected_resource_envelope_root"],
        "mount_manifest_root": bundle["mount_manifest_root"],
        "input_absence_root": bundle["input_absence_root"],
        "base_prestart_artifact_raw_sha256": base_chain[
            "base_prestart_artifact_raw_sha256"
        ],
        "failed_start_v2_terminal_raw_sha256": base_chain[
            "q1_start_v2_terminal_raw_sha256"
        ],
        "failed_start_v2_terminal_record_digest": base_chain[
            "q1_start_v2_terminal_record_digest"
        ],
        "failed_start_v2_capsule_raw_sha256": base_chain[
            "q1_start_v2_capsule_raw_sha256"
        ],
        "failed_start_v2_capsule_digest": base_chain["q1_start_v2_capsule_digest"],
        "failed_start_v2_authorization_raw_sha256": base_chain[
            "q1_start_v2_authorization_raw_sha256"
        ],
        "failed_start_v2_authorization_digest": base_chain[
            "q1_start_v2_authorization_digest"
        ],
        "failed_start_v2_consumption_raw_sha256": base_chain[
            "q1_start_v2_consumption_raw_sha256"
        ],
        "failed_start_v2_consumption_record_digest": base_chain[
            "q1_start_v2_consumption_record_digest"
        ],
        "failed_start_v3_terminal_raw_sha256": base_chain[
            "q1_start_v3_terminal_raw_sha256"
        ],
        "failed_start_v3_terminal_record_digest": base_chain[
            "q1_start_v3_terminal_record_digest"
        ],
        "failed_start_v3_capsule_raw_sha256": base_chain[
            "q1_start_v3_capsule_raw_sha256"
        ],
        "failed_start_v3_capsule_digest": base_chain["q1_start_v3_capsule_digest"],
        "failed_start_v3_authorization_raw_sha256": base_chain[
            "q1_start_v3_authorization_raw_sha256"
        ],
        "failed_start_v3_authorization_digest": base_chain[
            "q1_start_v3_authorization_digest"
        ],
        "failed_start_v3_consumption_raw_sha256": base_chain[
            "q1_start_v3_consumption_raw_sha256"
        ],
        "failed_start_v3_consumption_record_digest": base_chain[
            "q1_start_v3_consumption_record_digest"
        ],
        "failed_start_v4_terminal_raw_sha256": base_chain[
            "q1_start_v4_terminal_raw_sha256"
        ],
        "failed_start_v4_terminal_record_digest": base_chain[
            "q1_start_v4_terminal_record_digest"
        ],
        "failed_start_v4_capsule_raw_sha256": base_chain[
            "q1_start_v4_capsule_raw_sha256"
        ],
        "failed_start_v4_capsule_digest": base_chain["q1_start_v4_capsule_digest"],
        "failed_start_v4_authorization_raw_sha256": base_chain[
            "q1_start_v4_authorization_raw_sha256"
        ],
        "failed_start_v4_authorization_digest": base_chain[
            "q1_start_v4_authorization_digest"
        ],
        "failed_start_v4_consumption_raw_sha256": base_chain[
            "q1_start_v4_consumption_raw_sha256"
        ],
        "failed_start_v4_consumption_record_digest": base_chain[
            "q1_start_v4_consumption_record_digest"
        ],
        "failed_start_v5_terminal_raw_sha256": base_chain[
            "q1_start_v5_terminal_raw_sha256"
        ],
        "failed_start_v5_terminal_record_digest": base_chain[
            "q1_start_v5_terminal_record_digest"
        ],
        "failed_start_v5_capsule_raw_sha256": base_chain[
            "q1_start_v5_capsule_raw_sha256"
        ],
        "failed_start_v5_capsule_digest": base_chain["q1_start_v5_capsule_digest"],
        "failed_start_v5_authorization_raw_sha256": base_chain[
            "q1_start_v5_authorization_raw_sha256"
        ],
        "failed_start_v5_authorization_digest": base_chain[
            "q1_start_v5_authorization_digest"
        ],
        "failed_start_v5_consumption_raw_sha256": base_chain[
            "q1_start_v5_consumption_raw_sha256"
        ],
        "failed_start_v5_consumption_record_digest": base_chain[
            "q1_start_v5_consumption_record_digest"
        ],
        "failed_start_v5_supervisor_outcome_raw_sha256": base_chain[
            "q1_start_v5_supervisor_outcome_raw_sha256"
        ],
        "failed_start_v6_terminal_raw_sha256": base_chain[
            "q1_start_v6_terminal_raw_sha256"
        ],
        "failed_start_v6_terminal_record_digest": base_chain[
            "q1_start_v6_terminal_record_digest"
        ],
        "failed_start_v6_capsule_raw_sha256": base_chain[
            "q1_start_v6_capsule_raw_sha256"
        ],
        "failed_start_v6_capsule_digest": base_chain["q1_start_v6_capsule_digest"],
        "failed_start_v6_authorization_raw_sha256": base_chain[
            "q1_start_v6_authorization_raw_sha256"
        ],
        "failed_start_v6_authorization_digest": base_chain[
            "q1_start_v6_authorization_digest"
        ],
        "failed_start_v6_consumption_raw_sha256": base_chain[
            "q1_start_v6_consumption_raw_sha256"
        ],
        "failed_start_v6_consumption_record_digest": base_chain[
            "q1_start_v6_consumption_record_digest"
        ],
        "failed_start_v6_supervisor_outcome_raw_sha256": base_chain[
            "q1_start_v6_supervisor_outcome_raw_sha256"
        ],
        "failed_start_v6_controller_outcome_raw_sha256": base_chain[
            "q1_start_v6_controller_outcome_raw_sha256"
        ],
        "failed_start_v6_oom_event_receipt_raw_sha256": base_chain[
            "q1_start_v6_oom_event_receipt_raw_sha256"
        ],
        "q1_state": bundle["intended_q1_state"],
        "q1_gate_count": bundle["intended_q1_gate_count"],
        "q1_gate_mask": bundle["intended_q1_gate_mask"],
        "q1_output_slots": bundle["q1_output_slots"],
        "authorized": bundle["authorized"],
        "authorization_consumption_count": bundle["authorization_consumption_count"],
        "campaign_execution_count": bundle["campaign_execution_count"],
        "popen_count": bundle["popen_count"],
    }


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
        "actor_uid",
        "actor_gid",
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
        "endpoint_scratch_bytes": ENDPOINT_SCRATCH_BYTES,
        "endpoint_wall_seconds": 3600,
        "host_memory_bytes": 2_147_483_648,
        "host_output_bytes": 2_097_152,
        "host_scratch_bytes": 1,
        "host_wall_seconds": 3600,
        "pids_limit": 128,
        "minimum_available_memory_bytes": 4_294_967_296,
        "minimum_available_storage_bytes": 134_217_728,
        "actor_uid": 1000,
        "actor_gid": 1000,
    }
    for field, expected in expected_ints.items():
        _exact_int(checked[field], expected, f"runtime {field}")
    return checked


def validate_contract_v7(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "contract_id",
        "status",
        "action_id",
        "capsule_inputs_schema_version",
        "fixed_paths",
        "controller_source_closure_relative_paths",
        "base_authoritative_predecessor",
        "failed_start_v2_predecessor",
        "failed_start_v3_predecessor",
        "failed_start_v4_predecessor",
        "failed_start_v5_predecessor",
        "failed_start_v6_predecessor",
        "successor_prestart_bundle_descriptor",
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
        _fail("capsule path must belong to v7 run root")
    for field in ("authorization_path", "consumption_path", "terminal_path"):
        if Path(fixed[field]).parent != Path(fixed["ledger_root"]):
            _fail(f"{field} must belong to v7 ledger root")
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
        _fail("v7 run-owned path must be one direct run-root child")
    if len({fixed[field] for field in run_children}) != len(run_children):
        _fail("v7 run-owned paths must be distinct")
    contract["controller_source_closure_relative_paths"] = _relative_paths(
        contract["controller_source_closure_relative_paths"],
        "controller source closure",
    )
    contract["base_authoritative_predecessor"] = (
        _validate_base_authoritative_predecessor(
            contract["base_authoritative_predecessor"]
        )
    )
    contract["failed_start_v2_predecessor"] = _validate_failed_start_v2_predecessor(
        contract["failed_start_v2_predecessor"]
    )
    contract["failed_start_v3_predecessor"] = _validate_failed_start_v3_predecessor(
        contract["failed_start_v3_predecessor"]
    )
    contract["failed_start_v4_predecessor"] = _validate_failed_start_v4_predecessor(
        contract["failed_start_v4_predecessor"]
    )
    contract["failed_start_v5_predecessor"] = _validate_failed_start_v5_predecessor(
        contract["failed_start_v5_predecessor"]
    )
    contract["failed_start_v6_predecessor"] = _validate_failed_start_v6_predecessor(
        contract["failed_start_v6_predecessor"]
    )
    contract["successor_prestart_bundle_descriptor"] = _validate_successor_descriptor(
        contract["successor_prestart_bundle_descriptor"]
    )
    failed_predecessors = (
        contract["failed_start_v2_predecessor"],
        contract["failed_start_v3_predecessor"],
        contract["failed_start_v4_predecessor"],
        contract["failed_start_v5_predecessor"],
        contract["failed_start_v6_predecessor"],
    )
    if any(
        _under(fixed[field], failed[root])
        for field in ("run_root", "ledger_root", "campaign_lock_path") + run_children
        for failed in failed_predecessors
        for root in ("run_root", "ledger_root")
    ):
        _fail("v7 execution paths overlap a terminal predecessor campaign root")
    if fixed["run_root"] == fixed["ledger_root"] or _under(
        fixed["run_root"], fixed["ledger_root"]
    ) or _under(fixed["ledger_root"], fixed["run_root"]):
        _fail("v7 run and ledger roots must be separate")
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
        "preconsumption_startup_smoke_popen_limit_per_entry": 1,
        "startup_smoke_failure_preconsumption_pending": True,
        "startup_smoke_ownership_probe": True,
        "output_freezer_same_identity_required": True,
        "successor_bundle_replay_before_each_actor_intent": True,
        "failed_start_v2_resume_allowed": False,
        "failed_start_v2_work_product_reuse_allowed": False,
        "failed_start_v3_resume_allowed": False,
        "failed_start_v3_work_product_reuse_allowed": False,
        "failed_start_v4_resume_allowed": False,
        "failed_start_v4_work_product_reuse_allowed": False,
        "failed_start_v5_resume_allowed": False,
        "failed_start_v5_work_product_reuse_allowed": False,
        "failed_start_v6_resume_allowed": False,
        "failed_start_v6_work_product_reuse_allowed": False,
        "failed_start_v6_oom_event_receipt_replay_required": True,
        "formal_host_memory_bound_basis": (
            _successor.FORMAL_HOST_MEMORY_BOUND_BASIS
        ),
        "diagnostic_host_resource_peak_reuse_allowed": False,
        "network_allowed": False,
    }
    if type(policy) is not dict or policy != expected_policy:
        _fail("execution policy differs")
    authority = contract["authority_boundary"]
    expected_authority = {
        "q1_start_authorized_by_future_exact_capsule": True,
        "fresh_v7_actor_actions_require_exact_authorization": True,
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


def load_contract_v7(path: Path) -> LoadedContractV7:
    raw = path.read_bytes()
    value = validate_contract_v7(strict_json_bytes_v7(raw, "Q1 start v7 contract"))
    return LoadedContractV7(path, raw, raw_sha256_v7(raw), value)


def source_closure_root_v7(rows: object) -> str:
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


def _validated_actor_source_manifests(
    value: object,
    expected_roots: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"host", "python", "rust"}:
        _fail("actor source manifest registry differs")
    checked = deepcopy(value)
    for actor in ("python", "rust", "host"):
        manifest = checked[actor]
        if type(manifest) is not dict or set(manifest) != {
            "source_manifest_root",
            "rows",
        }:
            _fail(f"{actor} actor source manifest differs")
        try:
            computed = _successor.actor_source_manifest_root_v7(
                actor, manifest["rows"]
            )
        except _successor.Q1PrestartSuccessorCandidateError as error:
            _fail(f"{actor} actor source manifest rejected: {error}")
        if (
            manifest["source_manifest_root"] != computed
            or expected_roots.get(actor) != computed
        ):
            _fail(f"{actor} actor source manifest root differs")
    return checked


def _validated_successor_bundle(
    value: object,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        checked = _successor.validate_q1_prestart_successor_candidate_v7(value)
    except _successor.Q1PrestartSuccessorCandidateError as error:
        _fail(f"successor-prestart bundle rejected: {error}")
    projection = _bundle_descriptor_projection(checked, policy["path"])
    if (
        projection["schema_version"] != policy["schema_version"]
        or projection["status"] != policy["status"]
        or projection["authorized"] != policy["authorized"]
        or projection["authorization_consumption_count"]
        != policy["authorization_consumption_count"]
        or projection["campaign_execution_count"]
        != policy["campaign_execution_count"]
        or projection["popen_count"] != policy["popen_count"]
    ):
        _fail("successor-prestart bundle policy replay differs")
    corrected = checked["corrected_resource_envelope"]
    endpoint = corrected.get("endpoint")
    host = corrected.get("trusted_host_replay")
    if (
        type(endpoint) is not dict
        or endpoint.get("scratch_bytes") != ENDPOINT_SCRATCH_BYTES
        or type(host) is not dict
        or host.get("memory_bytes") != 2_147_483_648
        or host.get("memory_bound_basis")
        != _successor.FORMAL_HOST_MEMORY_BOUND_BASIS
        or host.get("diagnostic_host_resource_peak_reuse_allowed") is not False
        or host.get("failed_start_v6_oom_event_receipt_raw_sha256")
        != _successor.Q1_START_V6_OOM_EVENT_RECEIPT_RAW_SHA256
    ):
        _fail("successor-prestart corrected resource envelope differs")
    return deepcopy(checked), projection


def build_capsule_v7(
    contract: LoadedContractV7,
    capsule_inputs: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "controller_source_identity",
        "controller_source_closure_rows",
        "rust_binary",
        "successor_prestart_bundle",
        "actor_source_manifests",
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
    if source_closure_root_v7(rows) != source["source_closure_root"]:
        _fail("controller source closure root differs")
    binary = _rust_binary(
        inputs["rust_binary"], contract.value["fixed_paths"]["rust_binary_path"]
    )
    policy = contract.value["successor_prestart_bundle_descriptor"]
    bundle, descriptor = _validated_successor_bundle(
        inputs["successor_prestart_bundle"], policy
    )
    if bundle["source_identity"] != source:
        _fail(
            "successor-prestart bundle source identity differs from controller source identity"
        )
    actor_source_manifests = _validated_actor_source_manifests(
        inputs["actor_source_manifests"],
        bundle["actor_source_manifest_roots"],
    )
    if actor_source_manifests != bundle["actor_source_manifests"]:
        _fail("capsule actor source manifests differ from successor bundle")
    _hex32(inputs["run_id"], "run ID")
    if (
        type(inputs["created_at_unix_seconds"]) is not int
        or not 0 <= inputs["created_at_unix_seconds"] <= (1 << 64) - 1
    ):
        _fail("created_at_unix_seconds must be uint64")
    base = deepcopy(contract.value["base_authoritative_predecessor"])
    failed_v2 = deepcopy(contract.value["failed_start_v2_predecessor"])
    failed_v3 = deepcopy(contract.value["failed_start_v3_predecessor"])
    failed_v4 = deepcopy(contract.value["failed_start_v4_predecessor"])
    failed_v5 = deepcopy(contract.value["failed_start_v5_predecessor"])
    failed_v6 = deepcopy(contract.value["failed_start_v6_predecessor"])
    if (
        descriptor["base_prestart_artifact_raw_sha256"]
        != base["prestart_artifact_raw_sha256"]
        or descriptor["failed_start_v2_terminal_raw_sha256"]
        != failed_v2["terminal_raw_sha256"]
        or descriptor["failed_start_v2_terminal_record_digest"]
        != failed_v2["terminal_record_digest"]
        or descriptor["failed_start_v2_capsule_raw_sha256"]
        != failed_v2["capsule_raw_sha256"]
        or descriptor["failed_start_v2_capsule_digest"]
        != failed_v2["capsule_digest"]
        or descriptor["failed_start_v2_authorization_raw_sha256"]
        != failed_v2["authorization_raw_sha256"]
        or descriptor["failed_start_v2_authorization_digest"]
        != failed_v2["authorization_digest"]
        or descriptor["failed_start_v2_consumption_raw_sha256"]
        != failed_v2["consumption_raw_sha256"]
        or descriptor["failed_start_v2_consumption_record_digest"]
        != failed_v2["consumption_record_digest"]
        or descriptor["failed_start_v3_terminal_raw_sha256"]
        != failed_v3["terminal_raw_sha256"]
        or descriptor["failed_start_v3_terminal_record_digest"]
        != failed_v3["terminal_record_digest"]
        or descriptor["failed_start_v3_capsule_raw_sha256"]
        != failed_v3["capsule_raw_sha256"]
        or descriptor["failed_start_v3_capsule_digest"]
        != failed_v3["capsule_digest"]
        or descriptor["failed_start_v3_authorization_raw_sha256"]
        != failed_v3["authorization_raw_sha256"]
        or descriptor["failed_start_v3_authorization_digest"]
        != failed_v3["authorization_digest"]
        or descriptor["failed_start_v3_consumption_raw_sha256"]
        != failed_v3["consumption_raw_sha256"]
        or descriptor["failed_start_v3_consumption_record_digest"]
        != failed_v3["consumption_record_digest"]
        or descriptor["failed_start_v4_terminal_raw_sha256"]
        != failed_v4["terminal_raw_sha256"]
        or descriptor["failed_start_v4_terminal_record_digest"]
        != failed_v4["terminal_record_digest"]
        or descriptor["failed_start_v4_capsule_raw_sha256"]
        != failed_v4["capsule_raw_sha256"]
        or descriptor["failed_start_v4_capsule_digest"]
        != failed_v4["capsule_digest"]
        or descriptor["failed_start_v4_authorization_raw_sha256"]
        != failed_v4["authorization_raw_sha256"]
        or descriptor["failed_start_v4_authorization_digest"]
        != failed_v4["authorization_digest"]
        or descriptor["failed_start_v4_consumption_raw_sha256"]
        != failed_v4["consumption_raw_sha256"]
        or descriptor["failed_start_v4_consumption_record_digest"]
        != failed_v4["consumption_record_digest"]
        or descriptor["failed_start_v5_terminal_raw_sha256"]
        != failed_v5["terminal_raw_sha256"]
        or descriptor["failed_start_v5_terminal_record_digest"]
        != failed_v5["terminal_record_digest"]
        or descriptor["failed_start_v5_capsule_raw_sha256"]
        != failed_v5["capsule_raw_sha256"]
        or descriptor["failed_start_v5_capsule_digest"]
        != failed_v5["capsule_digest"]
        or descriptor["failed_start_v5_authorization_raw_sha256"]
        != failed_v5["authorization_raw_sha256"]
        or descriptor["failed_start_v5_authorization_digest"]
        != failed_v5["authorization_digest"]
        or descriptor["failed_start_v5_consumption_raw_sha256"]
        != failed_v5["consumption_raw_sha256"]
        or descriptor["failed_start_v5_consumption_record_digest"]
        != failed_v5["consumption_record_digest"]
        or descriptor["failed_start_v5_supervisor_outcome_raw_sha256"]
        != failed_v5["supervisor_outcome_raw_sha256"]
        or descriptor["failed_start_v6_terminal_raw_sha256"]
        != failed_v6["terminal_raw_sha256"]
        or descriptor["failed_start_v6_terminal_record_digest"]
        != failed_v6["terminal_record_digest"]
        or descriptor["failed_start_v6_capsule_raw_sha256"]
        != failed_v6["capsule_raw_sha256"]
        or descriptor["failed_start_v6_capsule_digest"]
        != failed_v6["capsule_digest"]
        or descriptor["failed_start_v6_authorization_raw_sha256"]
        != failed_v6["authorization_raw_sha256"]
        or descriptor["failed_start_v6_authorization_digest"]
        != failed_v6["authorization_digest"]
        or descriptor["failed_start_v6_consumption_raw_sha256"]
        != failed_v6["consumption_raw_sha256"]
        or descriptor["failed_start_v6_consumption_record_digest"]
        != failed_v6["consumption_record_digest"]
        or descriptor["failed_start_v6_supervisor_outcome_raw_sha256"]
        != failed_v6["supervisor_outcome_raw_sha256"]
        or descriptor["failed_start_v6_controller_outcome_raw_sha256"]
        != failed_v6["controller_outcome_raw_sha256"]
        or descriptor["failed_start_v6_oom_event_receipt_raw_sha256"]
        != failed_v6["oom_event_receipt_raw_sha256"]
    ):
        _fail("successor-prestart bundle lineage differs")
    compatibility = bundle["operator_owned_output_freezer_compatibility_descriptor"]
    runtime = contract.value["runtime"]
    if (
        compatibility != _successor.EXPECTED_OUTPUT_FREEZER_COMPATIBILITY
        or compatibility["actor_uid"] != runtime["actor_uid"]
        or compatibility["actor_gid"] != runtime["actor_gid"]
        or compatibility["output_freezer_same_identity_required"] is not True
    ):
        _fail("operator-owned-output/freezer compatibility differs")
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": CAPSULE_READY,
        "contract_id": CONTRACT_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "action_id": ACTION_ID,
        "controller_source_identity": source,
        "controller_source_closure_rows": rows,
        "formal_source_commit": descriptor["source_commit"],
        "formal_source_tree": descriptor["source_tree"],
        "base_authoritative_predecessor": base,
        "failed_start_v2_predecessor": failed_v2,
        "failed_start_v3_predecessor": failed_v3,
        "failed_start_v4_predecessor": failed_v4,
        "failed_start_v5_predecessor": failed_v5,
        "failed_start_v6_predecessor": failed_v6,
        "successor_prestart_bundle_descriptor": deepcopy(descriptor),
        "successor_prestart_bundle": bundle,
        "actor_source_manifests": actor_source_manifests,
        "runtime": deepcopy(runtime),
        "operator_owned_output_freezer_compatibility_descriptor": deepcopy(
            compatibility
        ),
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
        "failed_start_v2_work_product_reuse_allowed": False,
        "failed_start_v3_work_product_reuse_allowed": False,
        "failed_start_v4_work_product_reuse_allowed": False,
        "failed_start_v5_work_product_reuse_allowed": False,
        "failed_start_v6_work_product_reuse_allowed": False,
        "failed_start_v6_resume_allowed": False,
        "formal_host_memory_bound_basis": _successor.FORMAL_HOST_MEMORY_BOUND_BASIS,
        "diagnostic_host_resource_peak_reuse_allowed": False,
        "preconsumption_startup_smoke_popen_limit_per_entry": 1,
        "startup_smoke_failure_preconsumption_pending": True,
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule_v7(value: object, contract: LoadedContractV7) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("capsule must be an object")
    capsule = deepcopy(value)
    digest = capsule.get("capsule_digest")
    _tagged(digest, "capsule digest")
    inputs = {
        "schema_version": CAPSULE_INPUTS_SCHEMA_VERSION,
        "controller_source_identity": capsule.get("controller_source_identity"),
        "controller_source_closure_rows": capsule.get("controller_source_closure_rows"),
        "rust_binary": capsule.get("rust_binary"),
        "successor_prestart_bundle": capsule.get("successor_prestart_bundle"),
        "actor_source_manifests": capsule.get("actor_source_manifests"),
        "run_id": capsule.get("run_id"),
        "created_at_unix_seconds": capsule.get("created_at_unix_seconds"),
    }
    expected = build_capsule_v7(contract, inputs)
    if capsule != expected or digest != expected["capsule_digest"]:
        _fail("capsule replay differs")
    return capsule


def build_authorization_v7(
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_capsule_v7(dict(capsule), contract)
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "successor_prestart_bundle_raw_sha256": checked[
            "successor_prestart_bundle_descriptor"
        ]["raw_sha256"],
        "successor_prestart_bundle_root": checked[
            "successor_prestart_bundle_descriptor"
        ]["bundle_root"],
        "base_authoritative_artifact_raw_sha256": checked[
            "base_authoritative_predecessor"
        ]["prestart_artifact_raw_sha256"],
        "failed_start_v2_terminal_raw_sha256": checked[
            "failed_start_v2_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v2_terminal_record_digest": checked[
            "failed_start_v2_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v2_capsule_raw_sha256": checked["failed_start_v2_predecessor"][
            "capsule_raw_sha256"
        ],
        "failed_start_v2_capsule_digest": checked["failed_start_v2_predecessor"][
            "capsule_digest"
        ],
        "failed_start_v2_authorization_raw_sha256": checked[
            "failed_start_v2_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v2_authorization_digest": checked[
            "failed_start_v2_predecessor"
        ]["authorization_digest"],
        "failed_start_v2_consumption_raw_sha256": checked[
            "failed_start_v2_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v2_consumption_record_digest": checked[
            "failed_start_v2_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v3_terminal_raw_sha256": checked[
            "failed_start_v3_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v3_terminal_record_digest": checked[
            "failed_start_v3_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v3_capsule_raw_sha256": checked["failed_start_v3_predecessor"][
            "capsule_raw_sha256"
        ],
        "failed_start_v3_capsule_digest": checked["failed_start_v3_predecessor"][
            "capsule_digest"
        ],
        "failed_start_v3_authorization_raw_sha256": checked[
            "failed_start_v3_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v3_authorization_digest": checked[
            "failed_start_v3_predecessor"
        ]["authorization_digest"],
        "failed_start_v3_consumption_raw_sha256": checked[
            "failed_start_v3_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v3_consumption_record_digest": checked[
            "failed_start_v3_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v4_terminal_raw_sha256": checked[
            "failed_start_v4_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v4_terminal_record_digest": checked[
            "failed_start_v4_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v4_capsule_raw_sha256": checked[
            "failed_start_v4_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v4_capsule_digest": checked["failed_start_v4_predecessor"][
            "capsule_digest"
        ],
        "failed_start_v4_authorization_raw_sha256": checked[
            "failed_start_v4_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v4_authorization_digest": checked[
            "failed_start_v4_predecessor"
        ]["authorization_digest"],
        "failed_start_v4_consumption_raw_sha256": checked[
            "failed_start_v4_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v4_consumption_record_digest": checked[
            "failed_start_v4_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v5_terminal_raw_sha256": checked[
            "failed_start_v5_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v5_terminal_record_digest": checked[
            "failed_start_v5_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v5_capsule_raw_sha256": checked[
            "failed_start_v5_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v5_capsule_digest": checked["failed_start_v5_predecessor"][
            "capsule_digest"
        ],
        "failed_start_v5_authorization_raw_sha256": checked[
            "failed_start_v5_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v5_authorization_digest": checked[
            "failed_start_v5_predecessor"
        ]["authorization_digest"],
        "failed_start_v5_consumption_raw_sha256": checked[
            "failed_start_v5_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v5_consumption_record_digest": checked[
            "failed_start_v5_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v5_supervisor_outcome_raw_sha256": checked[
            "failed_start_v5_predecessor"
        ]["supervisor_outcome_raw_sha256"],
        "failed_start_v6_terminal_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v6_terminal_record_digest": checked[
            "failed_start_v6_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v6_capsule_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v6_capsule_digest": checked["failed_start_v6_predecessor"]
        ["capsule_digest"],
        "failed_start_v6_authorization_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v6_authorization_digest": checked[
            "failed_start_v6_predecessor"
        ]["authorization_digest"],
        "failed_start_v6_consumption_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v6_consumption_record_digest": checked[
            "failed_start_v6_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v6_supervisor_outcome_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["supervisor_outcome_raw_sha256"],
        "failed_start_v6_controller_outcome_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["controller_outcome_raw_sha256"],
        "failed_start_v6_oom_event_receipt_raw_sha256": checked[
            "failed_start_v6_predecessor"
        ]["oom_event_receipt_raw_sha256"],
        "action_id": ACTION_ID,
        "authority_role": "PROJECT_OWNER",
        "authorized_scope": AUTHORIZED_SCOPE,
        "authorization_consumption_limit": 1,
        "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "clean_checkpoint_resume_allowed": True,
        "completed_actor_remeasurement_allowed": False,
        "fresh_v7_actor_actions_authorized": True,
        "failed_start_v2_resume_authorized": False,
        "failed_start_v2_work_product_reuse_authorized": False,
        "failed_start_v3_resume_authorized": False,
        "failed_start_v3_work_product_reuse_authorized": False,
        "failed_start_v4_resume_authorized": False,
        "failed_start_v4_work_product_reuse_authorized": False,
        "failed_start_v5_resume_authorized": False,
        "failed_start_v5_work_product_reuse_authorized": False,
        "failed_start_v6_resume_authorized": False,
        "failed_start_v6_work_product_reuse_authorized": False,
        "failed_start_v6_oom_event_receipt_replay_required": True,
        "formal_host_memory_bytes": checked["runtime"]["host_memory_bytes"],
        "formal_host_memory_bound_basis": _successor.FORMAL_HOST_MEMORY_BOUND_BASIS,
        "diagnostic_host_resource_peak_reuse_authorized": False,
        "actor_uid": checked["runtime"]["actor_uid"],
        "actor_gid": checked["runtime"]["actor_gid"],
        "output_freezer_same_identity_required": checked[
            "operator_owned_output_freezer_compatibility_descriptor"
        ]["output_freezer_same_identity_required"],
        "preconsumption_startup_smoke_popen_limit_per_entry": 1,
        "startup_smoke_failure_preconsumption_pending": True,
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


def validate_authorization_v7(
    value: object,
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization_v7(contract, capsule)
    if value != expected:
        _fail("authorization replay differs")
    return deepcopy(expected)


def build_consumption_v7(
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    checked_capsule = validate_capsule_v7(dict(capsule), contract)
    checked_authorization = validate_authorization_v7(
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


def validate_consumption_v7(
    value: object,
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption_v7(contract, capsule, authorization)
    if value != expected:
        _fail("consumption replay differs")
    return deepcopy(expected)


def build_pending_record_v7(
    contract: LoadedContractV7,
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
    checked_capsule = validate_capsule_v7(dict(capsule), contract)
    if status not in {PENDING_NOT_CONSUMED, PENDING_CLEAN_CHECKPOINT}:
        _fail("pending status differs")
    if any(type(item) is not str or not item for item in (phase, condition, detail)):
        _fail("pending phase, condition, and detail must be nonempty strings")
    completed = list(completed_actors or [])
    expected_authorization = build_authorization_v7(contract, checked_capsule)
    expected_consumption = build_consumption_v7(
        contract, checked_capsule, expected_authorization
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


def validate_pending_record_v7(
    value: object,
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("pending record must be an object")
    expected = build_pending_record_v7(
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


def build_terminal_record_v7(
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    artifact_raw_sha256: str | None,
) -> dict[str, Any]:
    checked_capsule = validate_capsule_v7(dict(capsule), contract)
    checked_authorization = validate_authorization_v7(
        dict(authorization), contract, checked_capsule
    )
    if type(outcome) is not dict:
        _fail("terminal outcome must be an object")
    status = outcome.get("status")
    if status not in {COMPLETE, INCONCLUSIVE, OUTCOME_UNKNOWN}:
        _fail("terminal outcome status differs")
    if outcome.get("capsule_digest") != checked_capsule["capsule_digest"]:
        _fail("terminal outcome capsule differs")
    if type(outcome.get("actor_start_count_by_actor")) is not dict or set(
        outcome["actor_start_count_by_actor"]
    ) != {"host", "python", "rust"}:
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
            starts != {"host": 1, "python": 1, "rust": 1}
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
        "consumption_record_digest": build_consumption_v7(
            contract, checked_capsule, checked_authorization
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


def validate_terminal_record_v7(
    value: object,
    contract: LoadedContractV7,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
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
    }
    if type(value) is not dict or set(value) != required:
        _fail("terminal record registry differs")
    observed = _tagged(value["record_digest"], "terminal record digest")
    expected = build_terminal_record_v7(
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
    "AUTHORIZED_SCOPE",
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
    "ENDPOINT_SCRATCH_BYTES",
    "FAILED_V3_CODE",
    "FAILED_V3_DETAIL",
    "FAILED_V3_STATUS",
    "FAILED_V4_CODE",
    "FAILED_V4_DETAIL",
    "FAILED_V4_STATUS",
    "FAILED_V5_CODE",
    "FAILED_V5_DETAIL",
    "FAILED_V5_STATUS",
    "FAILED_V6_CODE",
    "FAILED_V6_DETAIL",
    "FAILED_V6_STATUS",
    "INCONCLUSIVE",
    "LoadedContractV7",
    "OUTCOME_UNKNOWN",
    "PENDING_CLEAN_CHECKPOINT",
    "PENDING_NOT_CONSUMED",
    "PENDING_SCHEMA_VERSION",
    "PYTHON_IMAGE",
    "Q1StartCampaignV7ValidationError",
    "RUST_IMAGE",
    "TERMINAL_SCHEMA_VERSION",
    "build_authorization_v7",
    "build_capsule_v7",
    "build_consumption_v7",
    "build_pending_record_v7",
    "build_terminal_record_v7",
    "canonical_json_bytes_v7",
    "load_contract_v7",
    "raw_sha256_v7",
    "source_closure_root_v7",
    "strict_json_bytes_v7",
    "validate_authorization_v7",
    "validate_capsule_v7",
    "validate_consumption_v7",
    "validate_contract_v7",
    "validate_pending_record_v7",
    "validate_terminal_record_v7",
]
