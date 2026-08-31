"""Pure state wire for one completion-only Q1 successor campaign.

This module deliberately has no execution surface.  It records authority for a
controller that may *read* the frozen v7r1 outputs and construct the final Q1
completion receipt, but may not launch a process, actor, or container.  The
v7r1 terminal campaign remains terminal-no-resume and its work product remains
immutable.
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


CONTRACT_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-contract/v8c"
CONTRACT_ID: Final = "hegel-phase3a-q1-completion-replay-v8c"
ACTION_ID: Final = "Q1_COMPLETION_REPLAY_V8C"
CAPSULE_INPUTS_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-capsule-inputs/v8c"
CAPSULE_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-capsule/v8c"
AUTHORIZATION_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-authorization/v8c"
CONSUMPTION_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-consumption/v8c"
PENDING_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-pending/v8c"
TERMINAL_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-terminal/v8c"

CAPSULE_READY: Final = "Q1_COMPLETION_REPLAY_V8C_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = (
    "Q1_COMPLETION_REPLAY_V8C_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED: Final = "Q1_COMPLETION_REPLAY_V8C_AUTHORIZATION_CONSUMED"
PENDING_NOT_CONSUMED: Final = "Q1_COMPLETION_REPLAY_V8C_PENDING_NOT_CONSUMED"
COMPLETE: Final = "Q1_COMPLETE_20_OF_20"
TERMINAL_NO_RESUME: Final = (
    "Q1_COMPLETION_REPLAY_V8C_FAILURE_TERMINAL_NO_RESUME"
)

V7R1_TERMINAL_STATUS: Final = "Q1_START_V7_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
V7R1_FAILURE_CODE: Final = "Q1_START_COMPLETION_ADAPTER_FAILED"
V7R1_FAILURE_DETAIL: Final = (
    "REJECT_Q1_DUAL_REPLAY: host resource guard was exceeded"
)
AUTHORIZED_SCOPE: Final = (
    "Q1_COMPLETION_REPLAY_V8C_EXACT_READONLY_REPLAY_ONCE"
)

Q1_GATE_TOTAL: Final = 20
Q1_PRESTART_GATE_COUNT: Final = 14
Q1_PRESTART_GATE_MASK: Final = 16_383
Q1_COMPLETE_GATE_MASK: Final = 1_048_575
NULL_OUTPUT_SLOTS: Final = [None] * 8
MAX_JSON_BYTES: Final = 16 * 1024 * 1024

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q1/COMPLETION_REPLAY/SOURCE_CLOSURE/V8C\x00",
    "capsule": b"HEGEL/Q1/COMPLETION_REPLAY/CAPSULE/V8C\x00",
    "authorization": b"HEGEL/Q1/COMPLETION_REPLAY/AUTHORIZATION/V8C\x00",
    "consumption": b"HEGEL/Q1/COMPLETION_REPLAY/CONSUMPTION/V8C\x00",
    "pending": b"HEGEL/Q1/COMPLETION_REPLAY/PENDING/V8C\x00",
    "terminal": b"HEGEL/Q1/COMPLETION_REPLAY/TERMINAL/V8C\x00",
}

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")


class Q1CompletionReplayValidationError(ValueError):
    """Raised when a completion-replay state object is not exact."""


@dataclass(frozen=True, slots=True)
class LoadedContract:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


def _fail(detail: str) -> NoReturn:
    raise Q1CompletionReplayValidationError(detail)


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


def canonical_json_bytes(value: object) -> bytes:
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


def strict_json_bytes(
    payload: bytes,
    label: str,
    maximum_bytes: int = MAX_JSON_BYTES,
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
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        _fail(f"{label} is not one canonical JSON object")
    return value


def raw_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail("raw SHA-256 input must be bytes")
    return "sha256:" + sha256(payload).hexdigest()


def _digest(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes(value)).hexdigest()


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be one tagged SHA-256")
    return value


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a nonempty string")
    return value


def _absolute(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != value:
        _fail(f"{label} must be a normalized absolute path")
    return value


def _under(child: str, parent: str) -> bool:
    try:
        Path(child).relative_to(Path(parent))
    except ValueError:
        return False
    return True


_FIXED_PATH_KEYS: Final = {
    "project_root",
    "campaign_lock_path",
    "run_root",
    "ledger_root",
    "capsule_path",
    "authorization_path",
    "consumption_path",
    "pending_path",
    "outcome_path",
    "terminal_path",
    "artifact_path",
}

_PREDECESSOR_KEYS: Final = {
    "run_id",
    "run_root",
    "ledger_root",
    "contract_path",
    "contract_raw_sha256",
    "capsule_path",
    "capsule_raw_sha256",
    "capsule_digest",
    "authorization_path",
    "authorization_raw_sha256",
    "authorization_digest",
    "consumption_path",
    "consumption_raw_sha256",
    "consumption_record_digest",
    "run_spec_path",
    "run_spec_raw_sha256",
    "python_intent_path",
    "python_intent_raw_sha256",
    "python_complete_path",
    "python_complete_raw_sha256",
    "rust_intent_path",
    "rust_intent_raw_sha256",
    "rust_complete_path",
    "rust_complete_raw_sha256",
    "host_intent_path",
    "host_intent_raw_sha256",
    "host_complete_path",
    "host_complete_raw_sha256",
    "supervisor_outcome_path",
    "supervisor_outcome_raw_sha256",
    "controller_outcome_path",
    "controller_outcome_raw_sha256",
    "host_result_path",
    "host_result_raw_sha256",
    "terminal_path",
    "terminal_raw_sha256",
    "terminal_record_digest",
    "status",
    "failure_code",
    "failure_detail",
    "authorization_consumption_count",
    "logical_campaign_count",
    "actor_start_count_by_actor",
    "completed_stages",
    "q1_state",
    "q1_gate_count",
    "q1_gate_mask",
    "q1_output_slots",
    "terminal_no_resume",
}

_READ_ONLY_REPLAY_KEYS: Final = {
    "python_output_root",
    "rust_output_root",
    "host_output_root",
    "access_mode",
    "mutation_allowed",
    "reuse_as_actor_work_product_allowed",
}

EXECUTION_POLICY: Final = {
    "maximum_popen_count": 0,
    "maximum_actor_start_count": 0,
    "maximum_docker_invocation_count": 0,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_pending_allowed": False,
    "postconsumption_failure_terminal_no_resume": True,
    "exact_partial_finalization_recovery_allowed": True,
    "v7r1_resume_allowed": False,
    "v7r1_work_product_mutation_allowed": False,
    "v7r1_outputs_read_only_replay_required": True,
    "network_allowed": False,
}

AUTHORITY_BOUNDARY: Final = {
    "q1_state_before": "NOT_RUN",
    "q1_gate_count_before": Q1_PRESTART_GATE_COUNT,
    "q1_gate_mask_before": Q1_PRESTART_GATE_MASK,
    "q1_output_slots_before": NULL_OUTPUT_SLOTS,
    "q1_success_state": "COMPLETE",
    "q1_success_gate_count": Q1_GATE_TOTAL,
    "q1_success_gate_mask": Q1_COMPLETE_GATE_MASK,
    "q2_state": "NOT_RUN",
    "m3_formal_roots": None,
    "outside_certificate_issued": False,
    "target_truth_access_allowed": False,
    "split_access_allowed": False,
    "role_evaluation_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "active_transition_allowed": False,
    "current_pointer_update_allowed": False,
}


def _validate_fixed_paths(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != _FIXED_PATH_KEYS:
        _fail("fixed path registry differs")
    fixed = {key: _absolute(child, f"fixed path {key}") for key, child in value.items()}
    run_root = Path(fixed["run_root"])
    ledger_root = Path(fixed["ledger_root"])
    if run_root == ledger_root or _under(str(run_root), str(ledger_root)) or _under(
        str(ledger_root), str(run_root)
    ):
        _fail("fresh run and ledger roots must be separate")
    for field in ("capsule_path", "pending_path", "outcome_path"):
        if Path(fixed[field]).parent != run_root:
            _fail(f"{field} must be one direct fresh run-root child")
    for field in ("authorization_path", "consumption_path", "terminal_path"):
        if Path(fixed[field]).parent != ledger_root:
            _fail(f"{field} must be one direct fresh ledger-root child")
    if len({fixed[field] for field in _FIXED_PATH_KEYS}) != len(_FIXED_PATH_KEYS):
        _fail("fixed paths must be distinct")
    return fixed


def _validate_predecessor(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PREDECESSOR_KEYS:
        _fail("v7r1 predecessor registry differs")
    checked = deepcopy(value)
    if type(checked["run_id"]) is not str or _RUN_ID.fullmatch(checked["run_id"]) is None:
        _fail("v7r1 run id differs")
    path_fields = {key for key in _PREDECESSOR_KEYS if key.endswith("_path")} | {
        "run_root",
        "ledger_root",
    }
    for field in path_fields:
        checked[field] = _absolute(checked[field], f"v7r1 {field}")
    digest_fields = {
        key
        for key in _PREDECESSOR_KEYS
        if key.endswith("_sha256") or key.endswith("_digest")
    }
    for field in digest_fields:
        checked[field] = _tagged(checked[field], f"v7r1 {field}")
    run_root = Path(checked["run_root"])
    ledger_root = Path(checked["ledger_root"])
    if run_root == ledger_root or _under(str(run_root), str(ledger_root)) or _under(
        str(ledger_root), str(run_root)
    ):
        _fail("v7r1 run and ledger roots differ")
    for field in (
        "capsule_path",
        "run_spec_path",
        "controller_outcome_path",
    ):
        if Path(checked[field]).parent != run_root:
            _fail(f"v7r1 {field} must belong to its run root")
    checkpoint_root = run_root / "checkpoints"
    for field in (
        "python_complete_path",
        "python_intent_path",
        "rust_complete_path",
        "rust_intent_path",
        "host_complete_path",
        "host_intent_path",
        "supervisor_outcome_path",
    ):
        if Path(checked[field]).parent != checkpoint_root:
            _fail(f"v7r1 {field} must belong to its checkpoint root")
    for field in ("authorization_path", "consumption_path", "terminal_path"):
        if Path(checked[field]).parent != ledger_root:
            _fail(f"v7r1 {field} must belong to its ledger root")
    if Path(checked["host_result_path"]).parent != run_root / "host-output":
        _fail("v7r1 host result must belong to its host output root")
    if (
        checked["status"] != V7R1_TERMINAL_STATUS
        or checked["failure_code"] != V7R1_FAILURE_CODE
        or checked["failure_detail"] != V7R1_FAILURE_DETAIL
        or checked["authorization_consumption_count"] != 1
        or checked["logical_campaign_count"] != 1
        or checked["actor_start_count_by_actor"]
        != {"host": 1, "python": 1, "rust": 1}
        or checked["completed_stages"] != ["python", "rust", "host"]
        or checked["q1_state"] != "NOT_RUN"
        or checked["q1_gate_count"] != Q1_PRESTART_GATE_COUNT
        or checked["q1_gate_mask"] != Q1_PRESTART_GATE_MASK
        or checked["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or checked["terminal_no_resume"] is not True
    ):
        _fail("v7r1 terminal boundary differs")
    return checked


def _validate_read_only_replay(
    value: object,
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _READ_ONLY_REPLAY_KEYS:
        _fail("read-only replay registry differs")
    checked = deepcopy(value)
    for field in ("python_output_root", "rust_output_root", "host_output_root"):
        checked[field] = _absolute(checked[field], f"read-only replay {field}")
    expected = {
        "python_output_root": str(Path(predecessor["run_root"]) / "python-output"),
        "rust_output_root": str(Path(predecessor["run_root"]) / "rust-output"),
        "host_output_root": str(Path(predecessor["run_root"]) / "host-output"),
        "access_mode": "READ_ONLY_REPLAY",
        "mutation_allowed": False,
        "reuse_as_actor_work_product_allowed": False,
    }
    if checked != expected:
        _fail("v7r1 outputs are not bound as exact read-only replay inputs")
    return checked


def validate_contract(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "contract_id",
        "status",
        "action_id",
        "fixed_paths",
        "predecessor_v7r1",
        "read_only_replay",
        "execution_policy",
        "authority_boundary",
    }
    if type(value) is not dict or set(value) != required:
        _fail("contract registry differs")
    contract = deepcopy(value)
    if (
        contract["schema_version"] != CONTRACT_SCHEMA
        or contract["contract_id"] != CONTRACT_ID
        or contract["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or contract["action_id"] != ACTION_ID
    ):
        _fail("contract header differs")
    fixed = _validate_fixed_paths(contract["fixed_paths"])
    predecessor = _validate_predecessor(contract["predecessor_v7r1"])
    replay = _validate_read_only_replay(contract["read_only_replay"], predecessor)
    for fresh in (
        fixed["run_root"],
        fixed["ledger_root"],
        fixed["campaign_lock_path"],
        fixed["artifact_path"],
    ):
        if _under(fresh, predecessor["run_root"]) or _under(
            fresh, predecessor["ledger_root"]
        ):
            _fail("fresh completion paths overlap the terminal v7r1 campaign")
    if contract["execution_policy"] != EXECUTION_POLICY:
        _fail("execution policy differs")
    if contract["authority_boundary"] != AUTHORITY_BOUNDARY:
        _fail("authority boundary differs")
    contract["fixed_paths"] = fixed
    contract["predecessor_v7r1"] = predecessor
    contract["read_only_replay"] = replay
    return contract


def load_contract(path: Path) -> LoadedContract:
    raw = path.read_bytes()
    value = validate_contract(strict_json_bytes(raw, "completion-replay contract"))
    return LoadedContract(path, raw, raw_sha256(raw), value)


def source_closure_root(rows: object) -> str:
    checked = _source_closure_rows(rows)
    return _digest(DOMAINS["source_closure"], checked)


def _source_closure_rows(value: object) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        _fail("source closure rows must be one nonempty list")
    checked: list[dict[str, Any]] = []
    prior = ""
    for row in value:
        if type(row) is not dict or set(row) != {
            "relative_path",
            "byte_count",
            "raw_sha256",
        }:
            _fail("source closure row registry differs")
        relative = row["relative_path"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or ".." in Path(relative).parts
            or relative <= prior
        ):
            _fail("source closure paths must be unique normalized sorted relative paths")
        count = row["byte_count"]
        if type(count) is not int or count < 0:
            _fail("source closure byte count differs")
        checked.append(
            {
                "relative_path": relative,
                "byte_count": count,
                "raw_sha256": _tagged(
                    row["raw_sha256"], "source closure raw SHA-256"
                ),
            }
        )
        prior = relative
    return checked


def validate_capsule_inputs(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_closure_rows",
        "source_closure_root",
        "run_id",
        "created_at_unix_seconds",
        "expected_completion_artifact_raw_sha256",
        "expected_completion_receipt_root",
        "expected_q1_output_slots",
    }
    if type(value) is not dict or set(value) != required:
        _fail("capsule inputs registry differs")
    checked = deepcopy(value)
    if checked["schema_version"] != CAPSULE_INPUTS_SCHEMA:
        _fail("capsule inputs schema differs")
    rows = _source_closure_rows(checked["source_closure_rows"])
    if checked["source_closure_root"] != source_closure_root(rows):
        _fail("source closure root differs")
    if type(checked["run_id"]) is not str or _RUN_ID.fullmatch(checked["run_id"]) is None:
        _fail("capsule run id must be 32 lowercase hexadecimal characters")
    if (
        type(checked["created_at_unix_seconds"]) is not int
        or checked["created_at_unix_seconds"] < 0
    ):
        _fail("capsule creation time differs")
    checked["expected_completion_artifact_raw_sha256"] = _tagged(
        checked["expected_completion_artifact_raw_sha256"],
        "expected completion artifact raw SHA-256",
    )
    checked["expected_completion_receipt_root"] = _tagged(
        checked["expected_completion_receipt_root"],
        "expected completion receipt root",
    )
    slots = _output_slots(checked["expected_q1_output_slots"])
    if slots[-1] != checked["expected_completion_receipt_root"]:
        _fail("Q1 output slot 8 must be the expected completion receipt root")
    checked["source_closure_rows"] = rows
    checked["expected_q1_output_slots"] = slots
    return checked


def build_capsule(
    contract: LoadedContract,
    capsule_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = validate_capsule_inputs(dict(capsule_inputs))
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "contract_id": CONTRACT_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "action_id": ACTION_ID,
        "source_closure_rows": deepcopy(inputs["source_closure_rows"]),
        "source_closure_root": inputs["source_closure_root"],
        "run_id": inputs["run_id"],
        "created_at_unix_seconds": inputs["created_at_unix_seconds"],
        "expected_completion_artifact_raw_sha256": inputs[
            "expected_completion_artifact_raw_sha256"
        ],
        "expected_completion_receipt_root": inputs[
            "expected_completion_receipt_root"
        ],
        "expected_q1_output_slots": deepcopy(inputs["expected_q1_output_slots"]),
        "fixed_paths": deepcopy(contract.value["fixed_paths"]),
        "predecessor_v7r1": deepcopy(contract.value["predecessor_v7r1"]),
        "read_only_replay": deepcopy(contract.value["read_only_replay"]),
        "execution_policy": deepcopy(contract.value["execution_policy"]),
        "authority_boundary": deepcopy(contract.value["authority_boundary"]),
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule(value: object, contract: LoadedContract) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("capsule must be an object")
    digest = _tagged(value.get("capsule_digest"), "capsule digest")
    inputs = {
        "schema_version": CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": value.get("source_closure_rows"),
        "source_closure_root": value.get("source_closure_root"),
        "run_id": value.get("run_id"),
        "created_at_unix_seconds": value.get("created_at_unix_seconds"),
        "expected_completion_artifact_raw_sha256": value.get(
            "expected_completion_artifact_raw_sha256"
        ),
        "expected_completion_receipt_root": value.get(
            "expected_completion_receipt_root"
        ),
        "expected_q1_output_slots": value.get("expected_q1_output_slots"),
    }
    expected = build_capsule(contract, inputs)
    if value != expected or digest != expected["capsule_digest"]:
        _fail("capsule replay differs")
    return deepcopy(expected)


def build_authorization(
    contract: LoadedContract,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_capsule(dict(capsule), contract)
    predecessor = checked["predecessor_v7r1"]
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "action_id": ACTION_ID,
        "authority_role": "PROJECT_OWNER",
        "authorized_scope": AUTHORIZED_SCOPE,
        "authorization_consumption_limit": 1,
        "source_closure_root": checked["source_closure_root"],
        "expected_completion_artifact_raw_sha256": checked[
            "expected_completion_artifact_raw_sha256"
        ],
        "expected_completion_receipt_root": checked[
            "expected_completion_receipt_root"
        ],
        "expected_q1_output_slots": deepcopy(checked["expected_q1_output_slots"]),
        "maximum_popen_count": 0,
        "maximum_actor_start_count": 0,
        "maximum_docker_invocation_count": 0,
        "network_allowed": False,
        "q1_completion_authorized": True,
        "q2_start_authorized": False,
        "v7r1_resume_authorized": False,
        "v7r1_output_mutation_authorized": False,
        "v7r1_run_id": predecessor["run_id"],
        "v7r1_contract_raw_sha256": predecessor["contract_raw_sha256"],
        "v7r1_capsule_raw_sha256": predecessor["capsule_raw_sha256"],
        "v7r1_capsule_digest": predecessor["capsule_digest"],
        "v7r1_authorization_raw_sha256": predecessor[
            "authorization_raw_sha256"
        ],
        "v7r1_authorization_digest": predecessor["authorization_digest"],
        "v7r1_consumption_raw_sha256": predecessor["consumption_raw_sha256"],
        "v7r1_consumption_record_digest": predecessor[
            "consumption_record_digest"
        ],
        "v7r1_terminal_record_digest": predecessor["terminal_record_digest"],
        "v7r1_terminal_raw_sha256": predecessor["terminal_raw_sha256"],
        "v7r1_run_spec_raw_sha256": predecessor["run_spec_raw_sha256"],
        "v7r1_python_intent_raw_sha256": predecessor[
            "python_intent_raw_sha256"
        ],
        "v7r1_supervisor_outcome_raw_sha256": predecessor[
            "supervisor_outcome_raw_sha256"
        ],
        "v7r1_python_complete_raw_sha256": predecessor[
            "python_complete_raw_sha256"
        ],
        "v7r1_rust_intent_raw_sha256": predecessor["rust_intent_raw_sha256"],
        "v7r1_rust_complete_raw_sha256": predecessor["rust_complete_raw_sha256"],
        "v7r1_host_intent_raw_sha256": predecessor["host_intent_raw_sha256"],
        "v7r1_host_complete_raw_sha256": predecessor["host_complete_raw_sha256"],
        "v7r1_controller_outcome_raw_sha256": predecessor[
            "controller_outcome_raw_sha256"
        ],
        "v7r1_host_result_raw_sha256": predecessor["host_result_raw_sha256"],
        "read_only_replay": deepcopy(checked["read_only_replay"]),
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def validate_authorization(
    value: object,
    contract: LoadedContract,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization(contract, capsule)
    if value != expected:
        _fail("authorization replay differs")
    return deepcopy(expected)


def build_consumption(
    contract: LoadedContract,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    checked_capsule = validate_capsule(dict(capsule), contract)
    checked_authorization = validate_authorization(
        dict(authorization), contract, checked_capsule
    )
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": AUTHORIZATION_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "popen_count": 0,
        "actor_start_count": 0,
        "predecessor_actor_start_count_by_actor": {
            "host": 1,
            "python": 1,
            "rust": 1,
        },
        "completion_successor_actor_start_count_by_actor": {
            "host": 0,
            "python": 0,
            "rust": 0,
        },
        "docker_invocation_count": 0,
        "q1_state_at_consumption": "NOT_RUN",
        "q1_gate_count_at_consumption": Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask_at_consumption": Q1_PRESTART_GATE_MASK,
        "q2_state_at_consumption": "NOT_RUN",
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def validate_consumption(
    value: object,
    contract: LoadedContract,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption(contract, capsule, authorization)
    if value != expected:
        _fail("consumption replay differs")
    return deepcopy(expected)


def build_pending(
    contract: LoadedContract,
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(dict(capsule), contract)
    for value, label in ((phase, "phase"), (condition, "condition"), (detail, "detail")):
        _nonempty(value, f"pending {label}")
    authorization_digest: str | None = None
    if authorization is not None:
        authorization_digest = validate_authorization(
            dict(authorization), contract, checked_capsule
        )["authorization_digest"]
    body: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA,
        "status": PENDING_NOT_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "resume_capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": authorization_digest,
        "consumption_record_digest": None,
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "popen_count": 0,
        "actor_start_count": 0,
        "predecessor_actor_start_count_by_actor": {
            "host": 1,
            "python": 1,
            "rust": 1,
        },
        "completion_successor_actor_start_count_by_actor": {
            "host": 0,
            "python": 0,
            "rust": 0,
        },
        "docker_invocation_count": 0,
        "same_digest_reentry_allowed": True,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "q1_state": "NOT_RUN",
        "q1_gate_count": Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask": Q1_PRESTART_GATE_MASK,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "q2_state": "NOT_RUN",
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_pending(
    value: object,
    contract: LoadedContract,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("pending record must be an object")
    expected = build_pending(
        contract,
        capsule,
        phase=value.get("phase"),
        condition=value.get("condition"),
        detail=value.get("detail"),
        authorization=authorization,
    )
    if value != expected:
        _fail("pending record replay differs")
    return deepcopy(expected)


def _output_slots(value: object) -> list[str]:
    if type(value) is not list or len(value) != 8:
        _fail("successful Q1 terminal requires exactly eight output slots")
    return [_tagged(item, f"Q1 output slot {index}") for index, item in enumerate(value, 1)]


def build_terminal(
    contract: LoadedContract,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    success: bool,
    q1_output_slots: list[str] | None = None,
    artifact_raw_sha256: str | None = None,
    failure_code: str | None = None,
    failure_detail: str | None = None,
    read_only_replay_completed: bool,
) -> dict[str, Any]:
    if type(success) is not bool or type(read_only_replay_completed) is not bool:
        _fail("terminal success/replay flags must be booleans")
    checked_capsule = validate_capsule(dict(capsule), contract)
    checked_authorization = validate_authorization(
        dict(authorization), contract, checked_capsule
    )
    checked_consumption = validate_consumption(
        dict(consumption), contract, checked_capsule, checked_authorization
    )
    if success:
        slots: list[str | None] = _output_slots(q1_output_slots)
        artifact = _tagged(artifact_raw_sha256, "completion artifact raw SHA-256")
        if slots != checked_capsule["expected_q1_output_slots"]:
            _fail("successful Q1 output slots differ from the capsule")
        if artifact != checked_capsule["expected_completion_artifact_raw_sha256"]:
            _fail("successful completion artifact differs from the capsule")
        if failure_code is not None or failure_detail is not None:
            _fail("successful terminal cannot contain a failure")
        if read_only_replay_completed is not True:
            _fail("successful terminal requires completed read-only replay")
        status = COMPLETE
        q1_state = "COMPLETE"
        q1_count = Q1_GATE_TOTAL
        q1_mask = Q1_COMPLETE_GATE_MASK
        completion_receipt_root: str | None = checked_capsule[
            "expected_completion_receipt_root"
        ]
        terminal_no_resume = False
    else:
        if q1_output_slots not in (None, NULL_OUTPUT_SLOTS):
            _fail("failed terminal cannot publish Q1 output slots")
        if artifact_raw_sha256 is not None:
            _fail("failed terminal cannot publish an artifact")
        _nonempty(failure_code, "terminal failure code")
        _nonempty(failure_detail, "terminal failure detail")
        slots = deepcopy(NULL_OUTPUT_SLOTS)
        artifact = None
        status = TERMINAL_NO_RESUME
        q1_state = "NOT_RUN"
        q1_count = Q1_PRESTART_GATE_COUNT
        q1_mask = Q1_PRESTART_GATE_MASK
        completion_receipt_root = None
        terminal_no_resume = True
    outcome = {
        "status": status,
        "capsule_digest": checked_capsule["capsule_digest"],
        "popen_count": 0,
        "actor_start_count": 0,
        "predecessor_actor_start_count_by_actor": {
            "host": 1,
            "python": 1,
            "rust": 1,
        },
        "completion_successor_actor_start_count_by_actor": {
            "host": 0,
            "python": 0,
            "rust": 0,
        },
        "docker_invocation_count": 0,
        "network_accessed": False,
        "v7r1_resumed": False,
        "v7r1_outputs_mutated": False,
        "v7r1_outputs_read_only_replayed": read_only_replay_completed,
        "q1_state": q1_state,
        "q1_gate_count": q1_count,
        "q1_gate_mask": q1_mask,
        "q1_output_slots": slots,
        "q2_state": "NOT_RUN",
        "completion_receipt_root": completion_receipt_root,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "terminal_no_resume": terminal_no_resume,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": status,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "artifact_raw_sha256": artifact,
        "outcome": outcome,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["terminal"], unsigned)
    return body


def validate_terminal(
    value: object,
    contract: LoadedContract,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("terminal record must contain one outcome object")
    outcome = value["outcome"]
    status = value.get("status")
    if status == COMPLETE:
        expected = build_terminal(
            contract,
            capsule,
            authorization,
            consumption,
            success=True,
            q1_output_slots=outcome.get("q1_output_slots"),
            artifact_raw_sha256=value.get("artifact_raw_sha256"),
            read_only_replay_completed=outcome.get(
                "v7r1_outputs_read_only_replayed"
            ),
        )
    elif status == TERMINAL_NO_RESUME:
        expected = build_terminal(
            contract,
            capsule,
            authorization,
            consumption,
            success=False,
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
            read_only_replay_completed=outcome.get(
                "v7r1_outputs_read_only_replayed"
            ),
        )
    else:
        _fail("terminal status differs")
    if value != expected:
        _fail("terminal record replay differs")
    return deepcopy(expected)


__all__ = [
    "ACTION_ID",
    "AUTHORIZATION_CONSUMED",
    "AUTHORIZATION_RECORDED",
    "AUTHORIZATION_SCHEMA",
    "AUTHORITY_BOUNDARY",
    "CAPSULE_READY",
    "CAPSULE_INPUTS_SCHEMA",
    "CAPSULE_SCHEMA",
    "COMPLETE",
    "CONSUMPTION_SCHEMA",
    "CONTRACT_ID",
    "CONTRACT_SCHEMA",
    "EXECUTION_POLICY",
    "LoadedContract",
    "NULL_OUTPUT_SLOTS",
    "PENDING_NOT_CONSUMED",
    "PENDING_SCHEMA",
    "Q1CompletionReplayValidationError",
    "Q1_COMPLETE_GATE_MASK",
    "Q1_GATE_TOTAL",
    "Q1_PRESTART_GATE_COUNT",
    "Q1_PRESTART_GATE_MASK",
    "TERMINAL_NO_RESUME",
    "TERMINAL_SCHEMA",
    "V7R1_FAILURE_CODE",
    "V7R1_FAILURE_DETAIL",
    "V7R1_TERMINAL_STATUS",
    "build_authorization",
    "build_capsule",
    "build_consumption",
    "build_pending",
    "build_terminal",
    "canonical_json_bytes",
    "load_contract",
    "raw_sha256",
    "source_closure_root",
    "strict_json_bytes",
    "validate_authorization",
    "validate_capsule",
    "validate_capsule_inputs",
    "validate_consumption",
    "validate_contract",
    "validate_pending",
    "validate_terminal",
]
