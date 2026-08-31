"""Pure state wire for one authorized Phase-3A-Q2 membership campaign.

The module defines capsule, authorization, consumption, pending, and terminal
objects.  It has no filesystem mutation and never opens target truth.
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

from . import phase3_q2_admission_v1 as _admission
from . import phase3_q2_role_membership_v1 as _membership


CONTRACT_SCHEMA: Final = "hegel-phase3a-q2-role-membership-contract/1"
CONTRACT_ID: Final = "hegel-phase3a-q2-role-membership-actual-v1"
ACTION_ID: Final = "Q2_ROLE_MEMBERSHIP_ACTUAL_V1"
CAPSULE_INPUTS_SCHEMA: Final = "hegel-phase3a-q2-role-membership-capsule-inputs/1"
CAPSULE_SCHEMA: Final = "hegel-phase3a-q2-role-membership-capsule/1"
AUTHORIZATION_SCHEMA: Final = "hegel-phase3a-q2-role-membership-authorization/1"
CONSUMPTION_SCHEMA: Final = "hegel-phase3a-q2-role-membership-consumption/1"
PENDING_SCHEMA: Final = "hegel-phase3a-q2-role-membership-pending/1"
TERMINAL_SCHEMA: Final = "hegel-phase3a-q2-role-membership-terminal/1"

CAPSULE_READY: Final = "Q2_ROLE_MEMBERSHIP_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = "Q2_ROLE_MEMBERSHIP_AUTHORIZATION_RECORDED_NOT_CONSUMED"
AUTHORIZATION_CONSUMED: Final = "Q2_ROLE_MEMBERSHIP_AUTHORIZATION_CONSUMED"
PENDING_NOT_CONSUMED: Final = "Q2_ROLE_MEMBERSHIP_PENDING_NOT_CONSUMED"
INVALIDATED_TERMINAL_NO_RESUME: Final = (
    "Q2_ROLE_MEMBERSHIP_INVALIDATED_TERMINAL_NO_RESUME"
)
AUTHORIZED_SCOPE: Final = "Q2_EXACT_FROZEN_ROLE_MEMBERSHIP_ONCE"

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q2/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q2/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q2/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q2/CONSUMPTION/V1\x00",
    "pending": b"HEGEL/Q2/PENDING/V1\x00",
    "terminal": b"HEGEL/Q2/TERMINAL/V1\x00",
}

EXECUTION_POLICY: Final = {
    "logical_campaign_limit": 1,
    "maximum_internal_popen_count": 0,
    "maximum_actor_start_count": 0,
    "maximum_docker_invocation_count": 0,
    "maximum_target_open_count_by_role": {
        "IN_LANGUAGE_NULL": 1,
        "OUTSIDE_TARGET": 1,
    },
    "maximum_role_evaluation_count_by_role": {
        "IN_LANGUAGE_NULL": 1,
        "OUTSIDE_TARGET": 1,
    },
    "network_allowed": False,
    "target_truth_access_only_after_consumption": True,
    "split_access_allowed": False,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_role_evaluation_reentry_allowed": False,
    "postconsumption_q1_archive_reread_allowed": False,
    "postconsumption_target_reopen_allowed": False,
    "partial_finalization_requires_sealed_artifact": True,
    "partial_finalization_artifact_to_terminal_only": True,
    "postconsumption_unknown_terminal_no_resume": True,
    "q1_archive_read_only": True,
    "q1_actor_work_product_copy_allowed": False,
}

AUTHORITY_BOUNDARY: Final = {
    "q1_state_before": "COMPLETE",
    "q1_gate_count_before": 20,
    "q1_gate_mask_before": 1_048_575,
    "q2_state_before": "NOT_RUN",
    "q2_start_authorized_by_contract": False,
    "m3_formal_roots": None,
    "outside_certificate_issued": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "active_transition_allowed": False,
    "production_allowed": False,
    "current_pointer_update_allowed": False,
    "external_scientific_claim_allowed": False,
}

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")

_FIXED_PATH_KEYS: Final = {
    "project_root",
    "campaign_lock_path",
    "run_root",
    "ledger_root",
    "capsule_path",
    "authorization_path",
    "consumption_path",
    "pending_path",
    "terminal_path",
    "artifact_path",
    "trajectory_path",
}

_Q1_INPUT_KEYS: Final = {
    "terminal_path",
    "completion_artifact_path",
    "archive_root",
    "selected_endpoint_kind",
    "access_mode",
}


class Q2CampaignValidationError(ValueError):
    """One Q2 campaign state object differs from the frozen wire."""


def _fail(detail: str) -> NoReturn:
    raise Q2CampaignValidationError(detail)


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


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > 16 * 1024 * 1024:
        _fail(f"{label} exceeds its byte bound")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict JSON: {error}")
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


def _absolute(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != value:
        _fail(f"{label} must be one normalized absolute path")
    return value


@dataclass(frozen=True, slots=True)
class LoadedContract:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


def _load_pretty_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict JSON: {error}")
    if type(value) is not dict:
        _fail(f"{label} must be one JSON object")
    return value


def validate_preregistration_payload(
    payload: bytes,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    _tagged(expected_raw_sha256, "preregistration raw SHA-256")
    value = _load_pretty_json(payload, "Q2 preregistration")
    if (
        raw_sha256(payload) != expected_raw_sha256
        or value != _admission.q2_preregistration_object_v1()
    ):
        _fail("Q2 preregistration raw bytes or projection differ")
    return value


def load_contract(path: Path) -> LoadedContract:
    if not isinstance(path, Path):
        raise TypeError("contract path must be pathlib.Path")
    raw = path.read_bytes()
    value = _load_pretty_json(raw, "Q2 contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "action_id",
        "preregistration_path",
        "preregistration_raw_sha256",
        "fixed_paths",
        "q1_inputs",
        "resource_envelope",
        "execution_policy",
        "authority_boundary",
    }
    if set(value) != required:
        _fail("Q2 contract key registry differs")
    if (
        value["schema_version"] != CONTRACT_SCHEMA
        or value["contract_id"] != CONTRACT_ID
        or value["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or value["action_id"] != ACTION_ID
        or value["execution_policy"] != EXECUTION_POLICY
        or value["authority_boundary"] != AUTHORITY_BOUNDARY
        or value["resource_envelope"]
        != _admission.q2_preregistration_object_v1()["resource_envelope"]
    ):
        _fail("Q2 contract fixed policy differs")
    _absolute(value["preregistration_path"], "preregistration path")
    _tagged(value["preregistration_raw_sha256"], "preregistration raw SHA-256")
    fixed = value["fixed_paths"]
    q1_inputs = value["q1_inputs"]
    if type(fixed) is not dict or set(fixed) != _FIXED_PATH_KEYS:
        _fail("Q2 fixed-path registry differs")
    if type(q1_inputs) is not dict or set(q1_inputs) != _Q1_INPUT_KEYS:
        _fail("Q2 Q1-input registry differs")
    for key, item in fixed.items():
        _absolute(item, f"fixed path {key}")
    for key in ("terminal_path", "completion_artifact_path", "archive_root"):
        _absolute(q1_inputs[key], f"Q1 input {key}")
    if q1_inputs != {
        "terminal_path": q1_inputs["terminal_path"],
        "completion_artifact_path": q1_inputs["completion_artifact_path"],
        "archive_root": q1_inputs["archive_root"],
        "selected_endpoint_kind": "python",
        "access_mode": "READ_ONLY_PREDECESSOR",
    }:
        _fail("Q2 Q1-input policy differs")
    return LoadedContract(path, raw, raw_sha256(raw), value)


def replay_loaded_contract(contract: LoadedContract) -> LoadedContract:
    if type(contract) is not LoadedContract:
        raise TypeError("contract must be LoadedContract")
    replayed = load_contract(contract.path)
    if (
        replayed.raw != contract.raw
        or replayed.raw_sha256 != contract.raw_sha256
        or replayed.value != contract.value
    ):
        _fail("loaded Q2 contract no longer replays exactly")
    return replayed


def _untag(value: object, label: str) -> bytes:
    return bytes.fromhex(_tagged(value, label)[7:])


def source_closure_root(rows: object) -> str:
    if type(rows) is not list or not rows:
        _fail("source closure must be a nonempty list")
    checked: list[dict[str, object]] = []
    previous = ""
    for row in rows:
        if type(row) is not dict or set(row) != {
            "relative_path",
            "byte_count",
            "raw_sha256",
        }:
            _fail("source closure row differs")
        relative = row["relative_path"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or relative <= previous
        ):
            _fail("source closure order/path differs")
        if type(row["byte_count"]) is not int or row["byte_count"] < 0:
            _fail("source closure byte count differs")
        _tagged(row["raw_sha256"], "source closure row SHA-256")
        checked.append(deepcopy(row))
        previous = relative
    return _digest(DOMAINS["source_closure"], checked)


def _preflight_receipt(value: object) -> dict[str, object]:
    required = {
        "endpoint_kind",
        "odd_class_count",
        "sink_class_count",
        "odd_universe_root",
        "sink_universe_root",
        "odd_signature_manifest_root",
        "sink_signature_manifest_root",
        "odd_saturation_state_root",
        "sink_saturation_state_root",
        "odd_class_archive_root",
        "sink_class_archive_root",
        "odd_class_stream_framed_bytes",
        "sink_class_stream_framed_bytes",
        "total_class_stream_framed_bytes",
        "selected_reader_total_bytes",
        "target_truth_accessed",
        "split_accessed",
        "role_evaluation_performed",
    }
    if type(value) is not dict or set(value) != required:
        _fail("preflight receipt registry differs")
    checked = deepcopy(value)
    if checked["endpoint_kind"] != "python":
        _fail("preflight endpoint differs")
    for key in (
        "odd_universe_root",
        "sink_universe_root",
        "odd_signature_manifest_root",
        "sink_signature_manifest_root",
        "odd_saturation_state_root",
        "sink_saturation_state_root",
        "odd_class_archive_root",
        "sink_class_archive_root",
    ):
        _tagged(checked[key], key)
    for key in (
        "odd_class_count",
        "sink_class_count",
        "odd_class_stream_framed_bytes",
        "sink_class_stream_framed_bytes",
        "total_class_stream_framed_bytes",
        "selected_reader_total_bytes",
    ):
        if type(checked[key]) is not int or checked[key] < 0:
            _fail(f"preflight {key} differs")
    if (
        checked["odd_class_count"] != 418
        or checked["sink_class_count"] != 89
        or checked["odd_class_stream_framed_bytes"] != 852_581
        or checked["sink_class_stream_framed_bytes"] != 55_084
        or checked["total_class_stream_framed_bytes"] != 907_665
        or checked["selected_reader_total_bytes"] != 919_757
        or checked["odd_universe_root"]
        != _admission.Q2_ROLE_COMMITMENTS[0].universe_root
        or checked["sink_universe_root"]
        != _admission.Q2_ROLE_COMMITMENTS[1].universe_root
        or checked["odd_signature_manifest_root"]
        != _admission.Q1_V8C_BINDING.ordered_q1_output_slots[0]
        or checked["odd_saturation_state_root"]
        != _admission.Q1_V8C_BINDING.ordered_q1_output_slots[1]
        or checked["sink_signature_manifest_root"]
        != _admission.Q1_V8C_BINDING.ordered_q1_output_slots[2]
        or checked["sink_saturation_state_root"]
        != _admission.Q1_V8C_BINDING.ordered_q1_output_slots[3]
        or checked["target_truth_accessed"] is not False
        or checked["split_accessed"] is not False
        or checked["role_evaluation_performed"] is not False
    ):
        _fail("preflight target-blind projection differs")
    return checked


def validate_preflight_receipt(value: object) -> dict[str, object]:
    return _preflight_receipt(value)


def validate_capsule_inputs(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_closure_rows",
        "source_closure_root",
        "run_id",
        "created_at_unix_seconds",
        "preflight_receipt",
    }
    if type(value) is not dict or set(value) != required:
        _fail("capsule inputs registry differs")
    checked = deepcopy(value)
    if checked["schema_version"] != CAPSULE_INPUTS_SCHEMA:
        _fail("capsule inputs schema differs")
    if checked["source_closure_root"] != source_closure_root(
        checked["source_closure_rows"]
    ):
        _fail("source closure root differs")
    if type(checked["run_id"]) is not str or _RUN_ID.fullmatch(checked["run_id"]) is None:
        _fail("run id must be 32 lowercase hexadecimal characters")
    if type(checked["created_at_unix_seconds"]) is not int or checked[
        "created_at_unix_seconds"
    ] < 0:
        _fail("capsule creation time differs")
    checked["preflight_receipt"] = _preflight_receipt(checked["preflight_receipt"])
    return checked


def build_capsule(
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    contract = replay_loaded_contract(contract)
    inputs = validate_capsule_inputs(dict(capsule_inputs))
    if dict(preregistration) != _admission.q2_preregistration_object_v1():
        _fail("Q2 preregistration replay differs")
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "contract_id": CONTRACT_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "preregistration_raw_sha256": contract.value[
            "preregistration_raw_sha256"
        ],
        "action_id": ACTION_ID,
        "source_closure_rows": deepcopy(inputs["source_closure_rows"]),
        "source_closure_root": inputs["source_closure_root"],
        "run_id": inputs["run_id"],
        "created_at_unix_seconds": inputs["created_at_unix_seconds"],
        "preflight_receipt": deepcopy(inputs["preflight_receipt"]),
        "q1_completion_binding": deepcopy(
            preregistration["q1_completion_binding"]
        ),
        "role_match_profile_id": preregistration["role_match_profile_id"],
        "role_commitments": deepcopy(preregistration["role_commitments"]),
        "result_contract": deepcopy(preregistration["result_contract"]),
        "resource_envelope": deepcopy(preregistration["resource_envelope"]),
        "fixed_paths": deepcopy(contract.value["fixed_paths"]),
        "q1_inputs": deepcopy(contract.value["q1_inputs"]),
        "execution_policy": deepcopy(EXECUTION_POLICY),
        "authority_boundary": deepcopy(AUTHORITY_BOUNDARY),
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule(
    value: object,
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("capsule must be an object")
    inputs = {
        "schema_version": CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": value.get("source_closure_rows"),
        "source_closure_root": value.get("source_closure_root"),
        "run_id": value.get("run_id"),
        "created_at_unix_seconds": value.get("created_at_unix_seconds"),
        "preflight_receipt": value.get("preflight_receipt"),
    }
    expected = build_capsule(contract, preregistration, inputs)
    if value != expected:
        _fail("capsule replay differs")
    return deepcopy(expected)


def build_authorization(
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_capsule(dict(capsule), contract, preregistration)
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "action_id": ACTION_ID,
        "authority_role": "PROJECT_OWNER",
        "authorized_scope": AUTHORIZED_SCOPE,
        "authorization_consumption_limit": 1,
        "logical_campaign_limit": 1,
        "q2_start_authorized": True,
        "target_truth_access_after_consumption_authorized": True,
        "target_open_limit_by_role": {
            "IN_LANGUAGE_NULL": 1,
            "OUTSIDE_TARGET": 1,
        },
        "role_evaluation_limit_by_role": {
            "IN_LANGUAGE_NULL": 1,
            "OUTSIDE_TARGET": 1,
        },
        "pass_terminal_authorized": True,
        "complete_fail_terminal_authorized": True,
        "partial_finalization_artifact_to_terminal_only": True,
        "split_access_authorized": False,
        "maximum_internal_popen_count": 0,
        "maximum_actor_start_count": 0,
        "maximum_docker_invocation_count": 0,
        "network_allowed": False,
        "outside_certificate_issuance_allowed": False,
        "external_scientific_claim_allowed": False,
        "active_transition_allowed": False,
        "production_allowed": False,
        "adoption_allowed": False,
        "promotion_allowed": False,
        "current_pointer_update_allowed": False,
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def validate_authorization(
    value: object,
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization(contract, preregistration, capsule)
    if value != expected:
        _fail("authorization replay differs")
    return deepcopy(expected)


def build_consumption(
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    checked_capsule = validate_capsule(dict(capsule), contract, preregistration)
    checked_authorization = validate_authorization(
        dict(authorization), contract, preregistration, checked_capsule
    )
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": AUTHORIZATION_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "internal_popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "q1_state_at_consumption": "COMPLETE",
        "q2_state_at_consumption": "NOT_RUN",
        "target_truth_accessed_before_consumption": False,
        "split_accessed_before_consumption": False,
        "role_evaluation_performed_before_consumption": False,
        "network_accessed_before_consumption": False,
        "target_blind_preflight": deepcopy(checked_capsule["preflight_receipt"]),
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def validate_consumption(
    value: object,
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption(
        contract, preregistration, capsule, authorization
    )
    if value != expected:
        _fail("consumption replay differs")
    return deepcopy(expected)


def build_pending(
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked = validate_capsule(dict(capsule), contract, preregistration)
    if any(type(value) is not str or not value for value in (phase, condition, detail)):
        _fail("pending phase/condition/detail must be nonempty strings")
    authorization_digest: str | None = None
    if authorization is not None:
        authorization_digest = validate_authorization(
            dict(authorization), contract, preregistration, checked
        )["authorization_digest"]
    body: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA,
        "status": PENDING_NOT_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "authorization_digest": authorization_digest,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "resume_capsule_digest": checked["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "internal_popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "q1_state": "COMPLETE",
        "q2_state": "NOT_RUN",
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_pending(
    value: object,
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("pending record must be an object")
    expected = build_pending(
        contract,
        preregistration,
        capsule,
        phase=value.get("phase"),
        condition=value.get("condition"),
        detail=value.get("detail"),
        authorization=authorization,
    )
    if value != expected:
        _fail("pending replay differs")
    return deepcopy(expected)


def _role_result_projection(
    result: _membership.Q2RoleMembershipResultV1,
) -> dict[str, object]:
    return {
        "role_id": result.commitment.role_id,
        "input_signature_id": result.commitment.input_signature_id,
        "universe_root": result.commitment.universe_root,
        "truth_root": result.commitment.truth_root,
        "q1_signature_manifest_root": "sha256:"
        + result.q1_signature_manifest_root.hex(),
        "q1_saturation_state_root": "sha256:"
        + result.q1_saturation_state_root.hex(),
        "q1_class_archive_root": "sha256:" + result.q1_class_archive_root.hex(),
        "evaluated_class_count": result.evaluated_class_count,
        "ordered_matching_class_ids": [
            "sha256:" + value.hex() for value in result.ordered_matching_class_ids
        ],
        "match_count": result.match_count,
        "membership_verdict": result.membership_verdict,
        "role_result_root": "sha256:" + result.result_root.hex(),
    }


def validate_artifact_binding(
    artifact: _membership.Q2MembershipArtifactV1,
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> _membership.Q2MembershipArtifactV1:
    """Bind one complete artifact to this exact consumed Q2 campaign."""

    if type(artifact) is not _membership.Q2MembershipArtifactV1:
        raise TypeError("artifact must be Q2MembershipArtifactV1")
    checked_capsule = validate_capsule(dict(capsule), contract, preregistration)
    checked_authorization = validate_authorization(
        dict(authorization), contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        dict(consumption),
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
    )
    binding = preregistration["q1_completion_binding"]
    slots = binding["ordered_output_slots"]
    preflight = checked_capsule["preflight_receipt"]
    role_expectations = (
        (
            _admission.Q2_ROLE_COMMITMENTS[0],
            418,
            slots[0],
            slots[1],
            preflight["odd_class_archive_root"],
        ),
        (
            _admission.Q2_ROLE_COMMITMENTS[1],
            89,
            slots[2],
            slots[3],
            preflight["sink_class_archive_root"],
        ),
    )
    if (
        artifact.capsule_digest
        != _untag(checked_capsule["capsule_digest"], "capsule digest")
        or artifact.authorization_digest
        != _untag(
            checked_authorization["authorization_digest"],
            "authorization digest",
        )
        or artifact.consumption_record_digest
        != _untag(checked_consumption["record_digest"], "consumption digest")
        or artifact.q1_completion_receipt_root
        != _untag(slots[7], "Q1 completion receipt root")
        or artifact.q1_closure_bundle_root
        != _untag(slots[4], "Q1 closure bundle root")
        or artifact.q1_dual_replay_agreement_root
        != _untag(slots[5], "Q1 dual replay agreement root")
        or artifact.q1_target_blind_access_ledger_root
        != _untag(slots[6], "Q1 target-blind access ledger root")
    ):
        _fail("artifact campaign/predecessor binding differs")
    for result, expectation in zip(
        artifact.role_results, role_expectations, strict=True
    ):
        commitment, class_count, manifest_root, saturation_root, archive_root = (
            expectation
        )
        if (
            result.commitment != commitment
            or result.evaluated_class_count != class_count
            or result.q1_signature_manifest_root
            != _untag(manifest_root, "Q1 signature manifest root")
            or result.q1_saturation_state_root
            != _untag(saturation_root, "Q1 saturation root")
            or result.q1_class_archive_root
            != _untag(archive_root, "Q1 class archive root")
        ):
            _fail("artifact role/Q1 archive binding differs")
    envelope = checked_capsule["resource_envelope"]
    if (
        artifact.observed_wall_time_ns
        > envelope["maximum_wall_time_seconds"] * 1_000_000_000
        or artifact.observed_peak_rss_bytes
        > envelope["maximum_peak_rss_bytes"]
        or len(artifact.canonical_bytes) > envelope["maximum_result_bytes"]
    ):
        _fail("artifact resource envelope differs")
    return artifact


def build_complete_terminal(
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    artifact: _membership.Q2MembershipArtifactV1,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(dict(capsule), contract, preregistration)
    checked_authorization = validate_authorization(
        dict(authorization), contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        dict(consumption),
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
    )
    artifact = validate_artifact_binding(
        artifact,
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
        checked_consumption,
    )
    outcome = {
        "status": artifact.status,
        "capsule_digest": checked_capsule["capsule_digest"],
        "q1_state": "COMPLETE",
        "q1_gate_count": 20,
        "q1_gate_mask": 1_048_575,
        "q2_state": "COMPLETE",
        "q2_disposition": artifact.status,
        "artifact_raw_sha256": raw_sha256(artifact.canonical_bytes),
        "artifact_root": "sha256:" + artifact.artifact_root.hex(),
        "role_results": [
            _role_result_projection(row) for row in artifact.role_results
        ],
        "target_open_count_by_role": {
            "IN_LANGUAGE_NULL": 1,
            "OUTSIDE_TARGET": 1,
        },
        "role_evaluation_count_by_role": {
            "IN_LANGUAGE_NULL": 1,
            "OUTSIDE_TARGET": 1,
        },
        "observed_wall_time_ns": artifact.observed_wall_time_ns,
        "observed_peak_rss_bytes": artifact.observed_peak_rss_bytes,
        "target_truth_accessed": True,
        "split_accessed": False,
        "role_evaluation_performed": True,
        "internal_popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "network_accessed": False,
        "external_scientific_claim_made": False,
        "active_transition_performed": False,
        "production_accessed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "adoption_performed": False,
        "promotion_performed": False,
        "current_pointer_updated": False,
        "failure_code": None,
        "failure_detail": None,
        "terminal_no_resume": False,
        "campaign_complete": True,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": artifact.status,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": outcome,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["terminal"], unsigned)
    return body


def build_invalidated_terminal(
    contract: LoadedContract,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
    target_open_count_by_role: Mapping[str, int | None],
    role_evaluation_count_by_role: Mapping[str, int | None],
) -> dict[str, Any]:
    checked_capsule = validate_capsule(dict(capsule), contract, preregistration)
    checked_authorization = validate_authorization(
        dict(authorization), contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        dict(consumption),
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
    )
    if any(
        type(value) is not str or not value
        for value in (failure_phase, failure_code, failure_detail)
    ):
        _fail("invalidated failure phase/code/detail must be nonempty strings")
    action_keys = {"IN_LANGUAGE_NULL", "OUTSIDE_TARGET"}
    target_counts = dict(target_open_count_by_role)
    evaluation_counts = dict(role_evaluation_count_by_role)
    if (
        set(target_counts) != action_keys
        or set(evaluation_counts) != action_keys
        or any(value not in (0, 1, None) for value in target_counts.values())
        or any(value not in (0, 1, None) for value in evaluation_counts.values())
    ):
        _fail("invalidated action audit differs")
    target_accessed = (
        None
        if any(value is None for value in target_counts.values())
        else any(value == 1 for value in target_counts.values())
    )
    evaluation_performed = (
        None
        if any(value is None for value in evaluation_counts.values())
        else any(value == 1 for value in evaluation_counts.values())
    )
    outcome = {
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": checked_capsule["capsule_digest"],
        "q1_state": "COMPLETE",
        "q1_gate_count": 20,
        "q1_gate_mask": 1_048_575,
        "q2_state": "INVALIDATED",
        "artifact_raw_sha256": None,
        "artifact_root": None,
        "role_results": None,
        "target_open_count_by_role": target_counts,
        "role_evaluation_count_by_role": evaluation_counts,
        "target_truth_accessed": target_accessed,
        "split_accessed": False,
        "role_evaluation_performed": evaluation_performed,
        "internal_popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "network_accessed": False,
        "external_scientific_claim_made": False,
        "active_transition_performed": False,
        "production_accessed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "adoption_performed": False,
        "promotion_performed": False,
        "current_pointer_updated": False,
        "failure_phase": failure_phase,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "terminal_no_resume": True,
        "campaign_complete": False,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
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
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    artifact_payload: bytes | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("terminal must contain an outcome")
    if value.get("status") in (
        _membership.PASS_STATUS,
        _membership.COMPLETE_FAIL_STATUS,
    ):
        if artifact_payload is None:
            _fail("complete terminal requires the sealed artifact")
        artifact = _membership.decode_q2_membership_artifact_v1(artifact_payload)
        expected = build_complete_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            artifact,
        )
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME:
        if artifact_payload is not None:
            _fail("invalidated terminal forbids a completion artifact")
        outcome = value["outcome"]
        expected = build_invalidated_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            failure_phase=outcome.get("failure_phase"),
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
            target_open_count_by_role=outcome.get(
                "target_open_count_by_role"
            ),
            role_evaluation_count_by_role=outcome.get(
                "role_evaluation_count_by_role"
            ),
        )
    else:
        _fail("terminal status differs")
    if value != expected:
        _fail("terminal replay differs")
    return deepcopy(expected)


__all__ = [
    "ACTION_ID",
    "AUTHORIZATION_CONSUMED",
    "AUTHORIZATION_RECORDED",
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZED_SCOPE",
    "AUTHORITY_BOUNDARY",
    "CAPSULE_INPUTS_SCHEMA",
    "CAPSULE_READY",
    "CAPSULE_SCHEMA",
    "CONSUMPTION_SCHEMA",
    "CONTRACT_ID",
    "CONTRACT_SCHEMA",
    "DOMAINS",
    "EXECUTION_POLICY",
    "INVALIDATED_TERMINAL_NO_RESUME",
    "LoadedContract",
    "PENDING_NOT_CONSUMED",
    "PENDING_SCHEMA",
    "Q2CampaignValidationError",
    "TERMINAL_SCHEMA",
    "build_authorization",
    "build_capsule",
    "build_complete_terminal",
    "build_consumption",
    "build_invalidated_terminal",
    "build_pending",
    "canonical_json_bytes",
    "load_contract",
    "raw_sha256",
    "replay_loaded_contract",
    "source_closure_root",
    "strict_json_bytes",
    "validate_authorization",
    "validate_artifact_binding",
    "validate_capsule",
    "validate_capsule_inputs",
    "validate_consumption",
    "validate_pending",
    "validate_preflight_receipt",
    "validate_preregistration_payload",
    "validate_terminal",
]
