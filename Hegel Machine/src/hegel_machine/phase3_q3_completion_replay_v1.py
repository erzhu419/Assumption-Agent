"""Pure state wire for the Phase-3A-Q3 completion-only admission replay.

The module has no process-launch or target-production surface.  It validates
the five sealed Q2 records, builds a producer-free bounded certificate
candidate, and defines capsule/authorization/consumption/pending/terminal
objects for a later separately authorized logical campaign.
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

from . import phase3_q3_bounded_quotient_certificate_v1 as _wire


CONTRACT_SCHEMA: Final = (
    "hegel-phase3a-q3-bounded-quotient-certificate-contract/1"
)
CONTRACT_ID: Final = "hegel-phase3a-q3-bounded-quotient-certificate-actual-v1"
ACTION_ID: Final = "Q3_BOUNDED_QUOTIENT_CERTIFICATE_COMPLETION_REPLAY_V1"
CAPSULE_INPUTS_SCHEMA: Final = (
    "hegel-phase3a-q3-bounded-quotient-certificate-capsule-inputs/1"
)
CAPSULE_SCHEMA: Final = "hegel-phase3a-q3-bounded-quotient-certificate-capsule/1"
AUTHORIZATION_SCHEMA: Final = (
    "hegel-phase3a-q3-bounded-quotient-certificate-authorization/1"
)
CONSUMPTION_SCHEMA: Final = (
    "hegel-phase3a-q3-bounded-quotient-certificate-consumption/1"
)
PENDING_SCHEMA: Final = "hegel-phase3a-q3-bounded-quotient-certificate-pending/1"
TERMINAL_SCHEMA: Final = (
    "hegel-phase3a-q3-bounded-quotient-certificate-terminal/1"
)

CAPSULE_READY: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_CAPSULE_READY_NOT_AUTHORIZED"
)
AUTHORIZATION_RECORDED: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_AUTHORIZATION_CONSUMED"
)
PENDING_NOT_CONSUMED: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_PENDING_NOT_CONSUMED"
)
INVALIDATED_TERMINAL_NO_RESUME: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_INVALIDATED_TERMINAL_NO_RESUME"
)
PARTIAL_FINALIZATION_REQUIRED: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_PARTIAL_FINALIZATION_REQUIRED"
)
AUTHORIZED_SCOPE: Final = (
    "Q3_EXACT_SEALED_Q2_BOUNDED_CERTIFICATE_ADMISSION_REPLAY_ONCE"
)
NOT_AUTHORIZED: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_NOT_AUTHORIZED_NOT_CONSUMED"
)

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q3/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q3/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q3/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q3/CONSUMPTION/V1\x00",
    "pending": b"HEGEL/Q3/PENDING/V1\x00",
    "terminal": b"HEGEL/Q3/TERMINAL/V1\x00",
}
Q2_DOMAINS: Final = {
    "capsule": b"HEGEL/Q2/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q2/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q2/CONSUMPTION/V1\x00",
    "terminal": b"HEGEL/Q2/TERMINAL/V1\x00",
}

Q3_SOURCE_PATHS: Final = (
    "src/hegel_machine/phase3_q3_bounded_quotient_certificate_v1.py",
    "src/hegel_machine/phase3_q3_completion_replay_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase3_q3_bounded_quotient_certificate_v1.py",
)

EXECUTION_POLICY: Final = {
    "logical_campaign_limit": 1,
    "maximum_internal_popen_count": 0,
    "maximum_actor_start_count": 0,
    "maximum_docker_invocation_count": 0,
    "maximum_measurement_count": 0,
    "maximum_target_open_count": 0,
    "maximum_role_evaluation_count": 0,
    "maximum_split_access_count": 0,
    "maximum_q1_archive_read_count": 0,
    "network_allowed": False,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_pending_allowed": False,
    "partial_finalization_artifact_to_terminal_only": True,
    "postconsumption_missing_or_unverifiable_artifact_terminal_no_resume": True,
}

AUTHORITY_BOUNDARY: Final = {
    "q1_state": "COMPLETE",
    "q1_gate_count": 20,
    "q1_gate_mask": 1_048_575,
    "q2_state": "COMPLETE",
    "q2_disposition": "Q2_ROLE_MEMBERSHIP_PASS",
    "phase3b_state": "NOT_RUN",
    "legacy_m3_resume_allowed": False,
    "m3_formal_roots": None,
    "outside_certificate_issued": False,
    "external_authority": False,
    "external_scientific_claim_allowed": False,
    "synthesis_allowed": False,
    "mdl_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "active_transition_allowed": False,
    "production_allowed": False,
    "current_pointer_update_allowed": False,
}

ZERO_ACTION_AUDIT: Final = {
    "target_open_count": 0,
    "target_truth_access_count": 0,
    "split_access_count": 0,
    "role_evaluation_count": 0,
    "q1_archive_read_count": 0,
    "legacy_m3_read_count": 0,
    "measurement_count": 0,
    "internal_popen_count": 0,
    "actor_start_count": 0,
    "docker_invocation_count": 0,
    "synthesis_action_count": 0,
    "mdl_action_count": 0,
    "certificate_signature_count": 0,
    "network_accessed": False,
    "outside_certificate_issued": False,
    "external_scientific_claim_made": False,
    "adoption_performed": False,
    "promotion_performed": False,
    "active_transition_performed": False,
    "production_accessed": False,
    "current_pointer_updated": False,
}

MAX_JSON_BYTES: Final = 32 * 1024 * 1024
MAX_Q2_ARTIFACT_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class Q3CompletionReplayError(ValueError):
    """One Q3 state object or sealed Q2 predecessor differs."""


def _fail(detail: str) -> NoReturn:
    raise Q3CompletionReplayError(detail)


@dataclass(frozen=True, slots=True)
class LoadedJson:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Q2SealedReplayV1:
    preflight_receipt: dict[str, Any]
    candidate: _wire.Q3CertificateCandidateV1


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


def pretty_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > MAX_JSON_BYTES:
        _fail(f"{label} exceeds its byte bound")
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


def _untag(value: object, label: str) -> bytes:
    return bytes.fromhex(_tagged(value, label)[7:])


def _absolute(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != value:
        _fail(f"{label} must be a normalized absolute path")
    return value


def load_preregistration(path: Path) -> LoadedJson:
    raw = path.read_bytes()
    value = pretty_json_bytes(raw, "Q3 preregistration")
    required = {
        "schema_version",
        "preregistration_id",
        "status",
        "predecessor_q2",
        "bounded_claim",
        "null_control",
        "completion_contract",
        "execution_policy",
        "authority_boundary",
    }
    if set(value) != required:
        _fail("Q3 preregistration key registry differs")
    if (
        value["schema_version"]
        != "hegel-phase3a-q3-bounded-quotient-certificate-preregistration/1"
        or value["preregistration_id"]
        != "hegel-phase3a-q3-bounded-quotient-certificate-v1"
        or value["status"] != "Q3_SPEC_FROZEN_NOT_AUTHORIZED"
        or value["execution_policy"] != EXECUTION_POLICY
        or value["authority_boundary"] != AUTHORITY_BOUNDARY
    ):
        _fail("Q3 preregistration fixed policy differs")
    _validate_predecessor_registry(value["predecessor_q2"])
    _validate_claim_registry(value["bounded_claim"], value["null_control"])
    completion = value["completion_contract"]
    if (
        type(completion) is not dict
        or completion.get("success_status") != _wire.Q3_SUCCESS_STATUS
        or completion.get("claim_level")
        != "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"
        or completion.get("producer_free_read_only_replay_required") is not True
        or any(
            completion.get(key) is not False
            for key in (
                "q2_measurement_reentry_allowed",
                "q1_archive_reread_allowed",
                "target_or_split_access_allowed",
                "certificate_signing_or_issuance_allowed",
            )
        )
    ):
        _fail("Q3 completion contract differs")
    return LoadedJson(path, raw, raw_sha256(raw), value)


def _validate_predecessor_registry(value: object) -> dict[str, Any]:
    required = {
        "contract_path",
        "contract_raw_sha256",
        "preregistration_path",
        "preregistration_raw_sha256",
        "source_closure_root",
        "capsule_path",
        "capsule_raw_sha256",
        "capsule_digest",
        "authorization_path",
        "authorization_raw_sha256",
        "authorization_digest",
        "consumption_path",
        "consumption_raw_sha256",
        "consumption_record_digest",
        "terminal_path",
        "terminal_raw_sha256",
        "terminal_record_digest",
        "artifact_path",
        "artifact_raw_sha256",
        "artifact_root",
        "q1_completion_receipt_root",
        "q1_closure_bundle_root",
        "q1_dual_replay_agreement_root",
        "q1_target_blind_access_ledger_root",
    }
    if type(value) is not dict or set(value) != required:
        _fail("Q2 predecessor registry differs")
    for key, item in value.items():
        if key.endswith("_path"):
            _absolute(item, f"Q2 predecessor {key}")
        else:
            _tagged(item, f"Q2 predecessor {key}")
    return deepcopy(value)


def _validate_claim_registry(outside: object, null: object) -> None:
    if type(outside) is not dict or type(null) is not dict:
        _fail("Q3 claim registry differs")
    if (
        outside.get("claim_kind") != "OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
        or outside.get("dsl_version") != "hegel-old-dsl-v1.6.0"
        or outside.get("closure_semantics_version")
        != "hegel-quotient-closure-v1.0.1"
        or outside.get("equivalence") != _wire.Q3_EQUIVALENCE_ID
        or outside.get("input_signature_id") != 1
        or outside.get("evaluated_class_count") != 418
        or outside.get("match_count") != 0
        or null.get("membership_verdict") != "IN_FROZEN_QUOTIENT_CLOSURE"
        or null.get("input_signature_id") != 2
        or null.get("evaluated_class_count") != 89
        or null.get("match_count") != 1
        or type(null.get("ordered_matching_class_ids")) is not list
        or len(null["ordered_matching_class_ids"]) != 1
    ):
        _fail("Q3 claim disposition differs")
    for mapping, keys in (
        (
            outside,
            ("frozen_universe_root", "target_truth_root", "role_result_root"),
        ),
        (
            null,
            ("frozen_universe_root", "target_truth_root", "role_result_root"),
        ),
    ):
        for key in keys:
            _tagged(mapping.get(key), f"claim {key}")
    _tagged(null["ordered_matching_class_ids"][0], "null matching class ID")


def load_contract(path: Path) -> LoadedJson:
    raw = path.read_bytes()
    value = pretty_json_bytes(raw, "Q3 contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "action_id",
        "preregistration_path",
        "preregistration_raw_sha256",
        "fixed_paths",
        "q2_inputs",
        "execution_policy",
        "authority_boundary",
    }
    if set(value) != required:
        _fail("Q3 contract key registry differs")
    if (
        value["schema_version"] != CONTRACT_SCHEMA
        or value["contract_id"] != CONTRACT_ID
        or value["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or value["action_id"] != ACTION_ID
        or value["execution_policy"] != EXECUTION_POLICY
        or value["authority_boundary"] != AUTHORITY_BOUNDARY
    ):
        _fail("Q3 contract fixed policy differs")
    _absolute(value["preregistration_path"], "Q3 preregistration path")
    _tagged(value["preregistration_raw_sha256"], "Q3 preregistration raw SHA-256")
    fixed_keys = {
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
    q2_keys = {"capsule", "authorization", "consumption", "terminal", "artifact"}
    if type(value["fixed_paths"]) is not dict or set(value["fixed_paths"]) != fixed_keys:
        _fail("Q3 fixed-path registry differs")
    if type(value["q2_inputs"]) is not dict or set(value["q2_inputs"]) != q2_keys:
        _fail("Q3 selected-Q2 input registry differs")
    for key, item in value["fixed_paths"].items():
        _absolute(item, f"Q3 fixed path {key}")
    for key, item in value["q2_inputs"].items():
        _absolute(item, f"Q3 Q2 input {key}")
    return LoadedJson(path, raw, raw_sha256(raw), value)


def source_closure_root(rows: object) -> str:
    checked = validate_source_closure_rows(rows)
    return _digest(DOMAINS["source_closure"], checked)


def validate_source_closure_rows(rows: object) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(Q3_SOURCE_PATHS):
        _fail("Q3 source closure row count differs")
    checked: list[dict[str, Any]] = []
    for row, expected_path in zip(rows, Q3_SOURCE_PATHS, strict=True):
        if type(row) is not dict or set(row) != {
            "relative_path",
            "byte_count",
            "raw_sha256",
        }:
            _fail("Q3 source closure row differs")
        if row["relative_path"] != expected_path:
            _fail("Q3 source closure is not the producer-free allowlist")
        if type(row["byte_count"]) is not int or row["byte_count"] < 1:
            _fail("Q3 source byte count differs")
        _tagged(row["raw_sha256"], "Q3 source raw SHA-256")
        checked.append(deepcopy(row))
    return checked


def _semantic_record(
    payload: bytes,
    label: str,
    domain: bytes,
    digest_key: str,
    expected_digest: str,
) -> dict[str, Any]:
    value = strict_json_bytes(payload, label)
    if value.get(digest_key) != expected_digest:
        _fail(f"{label} semantic digest field differs")
    unsigned = deepcopy(value)
    unsigned.pop(digest_key, None)
    if _digest(domain, unsigned) != expected_digest:
        _fail(f"{label} semantic digest replay differs")
    return value


def _expect_digest(value: object, expected: str, label: str) -> None:
    if value != expected:
        _fail(f"{label} differs")


def replay_q2_predecessor_v1(
    preregistration: Mapping[str, Any],
    *,
    capsule_payload: bytes,
    authorization_payload: bytes,
    consumption_payload: bytes,
    terminal_payload: bytes,
    artifact_payload: bytes,
) -> Q2SealedReplayV1:
    predecessor = _validate_predecessor_registry(
        preregistration.get("predecessor_q2")
    )
    supplied = {
        "capsule": capsule_payload,
        "authorization": authorization_payload,
        "consumption": consumption_payload,
        "terminal": terminal_payload,
        "artifact": artifact_payload,
    }
    for key, payload in supplied.items():
        maximum = MAX_Q2_ARTIFACT_BYTES if key == "artifact" else MAX_JSON_BYTES
        if type(payload) is not bytes or len(payload) > maximum:
            _fail(f"Q2 {key} exceeds its byte bound")
        _expect_digest(
            raw_sha256(payload),
            predecessor[f"{key}_raw_sha256"],
            f"Q2 {key} raw SHA-256",
        )
    capsule = _semantic_record(
        capsule_payload,
        "Q2 capsule",
        Q2_DOMAINS["capsule"],
        "capsule_digest",
        predecessor["capsule_digest"],
    )
    authorization = _semantic_record(
        authorization_payload,
        "Q2 authorization",
        Q2_DOMAINS["authorization"],
        "authorization_digest",
        predecessor["authorization_digest"],
    )
    consumption = _semantic_record(
        consumption_payload,
        "Q2 consumption",
        Q2_DOMAINS["consumption"],
        "record_digest",
        predecessor["consumption_record_digest"],
    )
    terminal = _semantic_record(
        terminal_payload,
        "Q2 terminal",
        Q2_DOMAINS["terminal"],
        "record_digest",
        predecessor["terminal_record_digest"],
    )
    if (
        capsule.get("contract_raw_sha256") != predecessor["contract_raw_sha256"]
        or capsule.get("preregistration_raw_sha256")
        != predecessor["preregistration_raw_sha256"]
        or capsule.get("source_closure_root") != predecessor["source_closure_root"]
        or authorization.get("capsule_digest") != predecessor["capsule_digest"]
        or authorization.get("outside_certificate_issuance_allowed") is not False
        or consumption.get("capsule_digest") != predecessor["capsule_digest"]
        or consumption.get("authorization_digest")
        != predecessor["authorization_digest"]
        or consumption.get("authorization_consumption_count") != 1
        or consumption.get("logical_campaign_count") != 1
        or terminal.get("capsule_digest") != predecessor["capsule_digest"]
        or terminal.get("authorization_digest") != predecessor["authorization_digest"]
        or terminal.get("consumption_record_digest")
        != predecessor["consumption_record_digest"]
        or terminal.get("status") != "Q2_ROLE_MEMBERSHIP_PASS"
    ):
        _fail("Q2 authority chain cross-binding differs")
    artifact = _wire.decode_sealed_q2_artifact_v1(artifact_payload)
    if (
        artifact.capsule_digest != _untag(predecessor["capsule_digest"], "Q2 capsule")
        or artifact.authorization_digest
        != _untag(predecessor["authorization_digest"], "Q2 authorization")
        or artifact.consumption_record_digest
        != _untag(predecessor["consumption_record_digest"], "Q2 consumption")
        or artifact.artifact_root
        != _untag(predecessor["artifact_root"], "Q2 artifact root")
        or artifact.q1_completion_receipt_root
        != _untag(
            predecessor["q1_completion_receipt_root"],
            "Q1 completion receipt root",
        )
        or artifact.q1_closure_bundle_root
        != _untag(predecessor["q1_closure_bundle_root"], "Q1 closure root")
        or artifact.q1_dual_replay_agreement_root
        != _untag(predecessor["q1_dual_replay_agreement_root"], "Q1 dual root")
        or artifact.q1_target_blind_access_ledger_root
        != _untag(
            predecessor["q1_target_blind_access_ledger_root"],
            "Q1 access-ledger root",
        )
    ):
        _fail("Q2 artifact authority/predecessor binding differs")
    outside_spec = preregistration["bounded_claim"]
    null_spec = preregistration["null_control"]
    outside, null = artifact.role_results
    if (
        outside.result_root_id != outside_spec["role_result_root"]
        or outside.universe_root
        != _untag(outside_spec["frozen_universe_root"], "outside universe")
        or outside.truth_root
        != _untag(outside_spec["target_truth_root"], "outside truth")
        or outside.evaluated_class_count != 418
        or outside.match_count != 0
        or null.result_root_id != null_spec["role_result_root"]
        or null.universe_root
        != _untag(null_spec["frozen_universe_root"], "null universe")
        or null.truth_root
        != _untag(null_spec["target_truth_root"], "null truth")
        or null.evaluated_class_count != 89
        or null.match_count != 1
        or tuple("sha256:" + item.hex() for item in null.ordered_matching_class_ids)
        != tuple(null_spec["ordered_matching_class_ids"])
    ):
        _fail("Q2 role projection differs from the Q3 freeze")
    outcome = terminal.get("outcome")
    if (
        type(outcome) is not dict
        or outcome.get("artifact_raw_sha256") != predecessor["artifact_raw_sha256"]
        or outcome.get("artifact_root") != predecessor["artifact_root"]
        or outcome.get("q1_state") != "COMPLETE"
        or outcome.get("q2_state") != "COMPLETE"
        or outcome.get("q2_disposition") != "Q2_ROLE_MEMBERSHIP_PASS"
        or outcome.get("outside_certificate_issued") is not False
        or outcome.get("m3_formal_roots") is not None
        or any(
            outcome.get(key) is not False
            for key in (
                "network_accessed",
                "external_scientific_claim_made",
                "active_transition_performed",
                "production_accessed",
                "adoption_performed",
                "promotion_performed",
                "current_pointer_updated",
            )
        )
    ):
        _fail("Q2 terminal outcome differs")
    candidate = _wire.Q3CertificateCandidateV1(
        q2_capsule_digest=artifact.capsule_digest,
        q2_authorization_digest=artifact.authorization_digest,
        q2_consumption_record_digest=artifact.consumption_record_digest,
        q2_terminal_record_digest=_untag(
            predecessor["terminal_record_digest"], "Q2 terminal record"
        ),
        q2_artifact_raw_sha256=_untag(
            predecessor["artifact_raw_sha256"], "Q2 artifact raw"
        ),
        q2_artifact_root=artifact.artifact_root,
        q1_completion_receipt_root=artifact.q1_completion_receipt_root,
        q1_closure_bundle_root=artifact.q1_closure_bundle_root,
        q1_dual_replay_agreement_root=artifact.q1_dual_replay_agreement_root,
        q1_target_blind_access_ledger_root=(
            artifact.q1_target_blind_access_ledger_root
        ),
        outside_role_result_root=outside.result_root,
        null_role_result_root=null.result_root,
        dsl_version=outside_spec["dsl_version"],
        closure_semantics_version=outside_spec["closure_semantics_version"],
        outside_universe_root=outside.universe_root,
        outside_truth_root=outside.truth_root,
        null_universe_root=null.universe_root,
        null_truth_root=null.truth_root,
        null_matching_class_ids=null.ordered_matching_class_ids,
    )
    if _wire.decode_q3_certificate_candidate_v1(candidate.canonical_bytes) != candidate:
        _fail("Q3 candidate self-replay differs")
    preflight = {
        "selected_q2_inputs": [
            {"kind": key, "byte_count": len(supplied[key]), "raw_sha256": raw_sha256(supplied[key])}
            for key in ("capsule", "authorization", "consumption", "terminal", "artifact")
        ],
        "selected_q2_input_total_bytes": sum(len(value) for value in supplied.values()),
        "q2_status": artifact.status,
        "q2_capsule_digest": predecessor["capsule_digest"],
        "q2_authorization_digest": predecessor["authorization_digest"],
        "q2_consumption_record_digest": predecessor["consumption_record_digest"],
        "q2_terminal_record_digest": predecessor["terminal_record_digest"],
        "q2_artifact_raw_sha256": predecessor["artifact_raw_sha256"],
        "q2_artifact_root": predecessor["artifact_root"],
        "outside_role_result_root": outside.result_root_id,
        "outside_evaluated_class_count": outside.evaluated_class_count,
        "outside_match_count": outside.match_count,
        "outside_membership_verdict": outside.membership_verdict,
        "null_role_result_root": null.result_root_id,
        "null_evaluated_class_count": null.evaluated_class_count,
        "null_match_count": null.match_count,
        "null_membership_verdict": null.membership_verdict,
        "expected_candidate_artifact_raw_sha256": raw_sha256(candidate.canonical_bytes),
        "expected_candidate_root": "sha256:" + candidate.candidate_root.hex(),
        "producer_free_read_only_replay_completed": True,
        "sealed_q2_truth_commitment_replay_completed": True,
        "q3_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }
    validate_preflight_receipt(preflight)
    return Q2SealedReplayV1(preflight, candidate)


def validate_preflight_receipt(value: object) -> dict[str, Any]:
    required = {
        "selected_q2_inputs",
        "selected_q2_input_total_bytes",
        "q2_status",
        "q2_capsule_digest",
        "q2_authorization_digest",
        "q2_consumption_record_digest",
        "q2_terminal_record_digest",
        "q2_artifact_raw_sha256",
        "q2_artifact_root",
        "outside_role_result_root",
        "outside_evaluated_class_count",
        "outside_match_count",
        "outside_membership_verdict",
        "null_role_result_root",
        "null_evaluated_class_count",
        "null_match_count",
        "null_membership_verdict",
        "expected_candidate_artifact_raw_sha256",
        "expected_candidate_root",
        "producer_free_read_only_replay_completed",
        "sealed_q2_truth_commitment_replay_completed",
        "q3_action_audit",
    }
    if type(value) is not dict or set(value) != required:
        _fail("Q3 preflight receipt registry differs")
    checked = deepcopy(value)
    selected = checked["selected_q2_inputs"]
    kinds = ("capsule", "authorization", "consumption", "terminal", "artifact")
    if (
        type(selected) is not list
        or len(selected) != 5
        or tuple(row.get("kind") for row in selected if type(row) is dict) != kinds
        or any(
            type(row) is not dict
            or set(row) != {"kind", "byte_count", "raw_sha256"}
            or type(row["byte_count"]) is not int
            or row["byte_count"] < 1
            for row in selected
        )
        or checked["selected_q2_input_total_bytes"]
        != sum(row["byte_count"] for row in selected)
        or checked["q2_status"] != "Q2_ROLE_MEMBERSHIP_PASS"
        or checked["outside_evaluated_class_count"] != 418
        or checked["outside_match_count"] != 0
        or checked["outside_membership_verdict"]
        != "OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
        or checked["null_evaluated_class_count"] != 89
        or checked["null_match_count"] != 1
        or checked["null_membership_verdict"] != "IN_FROZEN_QUOTIENT_CLOSURE"
        or checked["producer_free_read_only_replay_completed"] is not True
        or checked["sealed_q2_truth_commitment_replay_completed"] is not True
        or checked["q3_action_audit"] != ZERO_ACTION_AUDIT
    ):
        _fail("Q3 preflight receipt disposition differs")
    for key in (
        "q2_capsule_digest",
        "q2_authorization_digest",
        "q2_consumption_record_digest",
        "q2_terminal_record_digest",
        "q2_artifact_raw_sha256",
        "q2_artifact_root",
        "outside_role_result_root",
        "null_role_result_root",
        "expected_candidate_artifact_raw_sha256",
        "expected_candidate_root",
    ):
        _tagged(checked[key], f"Q3 preflight {key}")
    for row in selected:
        _tagged(row["raw_sha256"], "selected Q2 raw SHA-256")
    return checked


def validate_capsule_inputs(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_closure_rows",
        "source_closure_root",
        "source_commit",
        "run_id",
        "created_at_unix_seconds",
        "preflight_receipt",
    }
    if type(value) is not dict or set(value) != required:
        _fail("Q3 capsule-input registry differs")
    checked = deepcopy(value)
    if checked["schema_version"] != CAPSULE_INPUTS_SCHEMA:
        _fail("Q3 capsule-input schema differs")
    rows = validate_source_closure_rows(checked["source_closure_rows"])
    if checked["source_closure_root"] != source_closure_root(rows):
        _fail("Q3 source closure root differs")
    if type(checked["source_commit"]) is not str or _SOURCE_COMMIT.fullmatch(checked["source_commit"]) is None:
        _fail("Q3 source commit must be 40 lowercase hexadecimal characters")
    if type(checked["run_id"]) is not str or _RUN_ID.fullmatch(checked["run_id"]) is None:
        _fail("Q3 run id must be 32 lowercase hexadecimal characters")
    if type(checked["created_at_unix_seconds"]) is not int or checked["created_at_unix_seconds"] < 0:
        _fail("Q3 capsule creation time differs")
    checked["source_closure_rows"] = rows
    checked["preflight_receipt"] = validate_preflight_receipt(
        checked["preflight_receipt"]
    )
    return checked


def build_capsule(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = validate_capsule_inputs(dict(capsule_inputs))
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "contract_id": CONTRACT_ID,
        "contract_raw_sha256": contract.raw_sha256,
        "preregistration_raw_sha256": contract.value["preregistration_raw_sha256"],
        "action_id": ACTION_ID,
        "source_closure_rows": deepcopy(inputs["source_closure_rows"]),
        "source_closure_root": inputs["source_closure_root"],
        "source_commit": inputs["source_commit"],
        "run_id": inputs["run_id"],
        "created_at_unix_seconds": inputs["created_at_unix_seconds"],
        "preflight_receipt": deepcopy(inputs["preflight_receipt"]),
        "predecessor_q2": deepcopy(preregistration["predecessor_q2"]),
        "bounded_claim": deepcopy(preregistration["bounded_claim"]),
        "null_control": deepcopy(preregistration["null_control"]),
        "completion_contract": deepcopy(preregistration["completion_contract"]),
        "fixed_paths": deepcopy(contract.value["fixed_paths"]),
        "q2_inputs": deepcopy(contract.value["q2_inputs"]),
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
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q3 capsule must be an object")
    inputs = {
        "schema_version": CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": value.get("source_closure_rows"),
        "source_closure_root": value.get("source_closure_root"),
        "source_commit": value.get("source_commit"),
        "run_id": value.get("run_id"),
        "created_at_unix_seconds": value.get("created_at_unix_seconds"),
        "preflight_receipt": value.get("preflight_receipt"),
    }
    expected = build_capsule(contract, preregistration, inputs)
    if value != expected:
        _fail("Q3 capsule replay differs")
    return deepcopy(expected)


def build_authorization(
    contract: LoadedJson,
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
        "completion_replay_authorized": True,
        "producer_free_q2_read_only_replay_required": True,
        "maximum_internal_popen_count": 0,
        "maximum_actor_start_count": 0,
        "maximum_docker_invocation_count": 0,
        "maximum_measurement_count": 0,
        "maximum_target_open_count": 0,
        "maximum_role_evaluation_count": 0,
        "maximum_split_access_count": 0,
        "maximum_q1_archive_read_count": 0,
        "network_allowed": False,
        "certificate_signing_allowed": False,
        "outside_certificate_issuance_allowed": False,
        "synthesis_allowed": False,
        "mdl_allowed": False,
        "external_scientific_claim_allowed": False,
        "active_transition_allowed": False,
        "production_allowed": False,
        "adoption_allowed": False,
        "promotion_allowed": False,
        "current_pointer_update_allowed": False,
        "partial_finalization_artifact_to_terminal_only": True,
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def validate_authorization(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization(contract, preregistration, capsule)
    if value != expected:
        _fail("Q3 authorization replay differs")
    return deepcopy(expected)


def build_consumption(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": AUTHORIZATION_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "q1_state_at_consumption": "COMPLETE",
        "q2_state_at_consumption": "COMPLETE",
        "phase3b_state_at_consumption": "NOT_RUN",
        "q3_action_audit_before_consumption": deepcopy(ZERO_ACTION_AUDIT),
        "preflight_receipt": deepcopy(checked_capsule["preflight_receipt"]),
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def validate_consumption(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption(
        contract, preregistration, capsule, authorization
    )
    if value != expected:
        _fail("Q3 consumption replay differs")
    return deepcopy(expected)


def build_pending(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked = validate_capsule(capsule, contract, preregistration)
    if any(type(value) is not str or not value for value in (phase, condition, detail)):
        _fail("Q3 pending phase/condition/detail must be nonempty strings")
    authorization_digest = None
    if authorization is not None:
        authorization_digest = validate_authorization(
            authorization, contract, preregistration, checked
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
        "q1_state": "COMPLETE",
        "q2_state": "COMPLETE",
        "phase3b_state": "NOT_RUN",
        "q3_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_candidate_binding(
    payload: bytes,
    capsule: Mapping[str, Any],
) -> _wire.Q3CertificateCandidateV1:
    candidate = _wire.decode_q3_certificate_candidate_v1(payload)
    preflight = validate_preflight_receipt(capsule.get("preflight_receipt"))
    if (
        raw_sha256(payload) != preflight["expected_candidate_artifact_raw_sha256"]
        or "sha256:" + candidate.candidate_root.hex()
        != preflight["expected_candidate_root"]
        or "sha256:" + candidate.q2_terminal_record_digest.hex()
        != preflight["q2_terminal_record_digest"]
        or "sha256:" + candidate.q2_artifact_root.hex()
        != preflight["q2_artifact_root"]
    ):
        _fail("Q3 candidate/capsule binding differs")
    return candidate


def build_complete_terminal(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    artifact_payload: bytes,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption,
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
    )
    candidate = validate_candidate_binding(artifact_payload, checked_capsule)
    outcome = {
        "status": _wire.Q3_SUCCESS_STATUS,
        "claim_level": "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL",
        "capsule_digest": checked_capsule["capsule_digest"],
        "q1_state": "COMPLETE",
        "q1_gate_count": 20,
        "q1_gate_mask": 1_048_575,
        "q2_state": "COMPLETE",
        "q2_disposition": "Q2_ROLE_MEMBERSHIP_PASS",
        "phase3b_state": "NOT_RUN",
        "artifact_raw_sha256": raw_sha256(artifact_payload),
        "candidate_root": "sha256:" + candidate.candidate_root.hex(),
        "producer_free_read_only_replay_completed": True,
        "sealed_q2_truth_commitment_replay_completed": True,
        "q3_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "formal_certificate_signed": False,
        "external_authority": False,
        "synthesis_released": False,
        "terminal_no_resume": False,
        "campaign_complete": True,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": _wire.Q3_SUCCESS_STATUS,
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
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption,
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
    )
    if any(
        type(value) is not str or not value
        for value in (failure_phase, failure_code, failure_detail)
    ):
        _fail("Q3 invalidated failure fields differ")
    outcome = {
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": checked_capsule["capsule_digest"],
        "q1_state": "COMPLETE",
        "q2_state": "COMPLETE",
        "phase3b_state": "NOT_RUN",
        "artifact_raw_sha256": None,
        "candidate_root": None,
        "q3_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "formal_certificate_signed": False,
        "external_authority": False,
        "synthesis_released": False,
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
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    artifact_payload: bytes | None = None,
) -> dict[str, Any]:
    """Replay one durable terminal from its bound predecessor state."""

    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("Q3 terminal must contain an outcome")
    if value.get("status") == _wire.Q3_SUCCESS_STATUS:
        if artifact_payload is None:
            _fail("Q3 complete terminal requires the sealed candidate artifact")
        expected = build_complete_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            artifact_payload,
        )
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME:
        if artifact_payload is not None:
            _fail("Q3 invalidated terminal forbids a valid candidate artifact")
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
        )
    else:
        _fail("Q3 terminal status differs")
    if value != expected:
        _fail("Q3 terminal replay differs")
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
    "LoadedJson",
    "NOT_AUTHORIZED",
    "PARTIAL_FINALIZATION_REQUIRED",
    "PENDING_NOT_CONSUMED",
    "PENDING_SCHEMA",
    "Q2SealedReplayV1",
    "Q3CompletionReplayError",
    "Q3_SOURCE_PATHS",
    "TERMINAL_SCHEMA",
    "ZERO_ACTION_AUDIT",
    "build_authorization",
    "build_capsule",
    "build_complete_terminal",
    "build_consumption",
    "build_invalidated_terminal",
    "build_pending",
    "canonical_json_bytes",
    "load_contract",
    "load_preregistration",
    "raw_sha256",
    "replay_q2_predecessor_v1",
    "source_closure_root",
    "strict_json_bytes",
    "validate_authorization",
    "validate_candidate_binding",
    "validate_capsule",
    "validate_capsule_inputs",
    "validate_consumption",
    "validate_terminal",
    "validate_preflight_receipt",
    "validate_source_closure_rows",
]
