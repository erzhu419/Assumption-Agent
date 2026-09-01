"""Pure state and completion replay for the Q5b Purpose-5 trust amendment.

Q5b reads exactly six sealed, public Q5a inputs.  It never opens a custody
root or private key and performs no process, cryptographic-key, signing,
certificate, target, or network action.  Its sole state transition pins the
three public Purpose-5 signer identities and accepts their epoch-1 current
status.  This public signer activation is deliberately distinct from the
project-wide ``ACTIVE`` transition, which remains false.
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

from . import phase3_q5b_purpose5_trust_amendment_v1 as _wire


CONTRACT_SCHEMA: Final = "hegel-phase3a-q5b-purpose5-trust-amendment-contract/1"
CONTRACT_ID: Final = "hegel-phase3a-q5b-purpose5-trust-amendment-actual-v1"
PREREGISTRATION_SCHEMA: Final = (
    "hegel-phase3a-q5b-purpose5-trust-amendment-preregistration/1"
)
PREREGISTRATION_ID: Final = "hegel-phase3a-q5b-purpose5-trust-amendment-v1"
ACTION_ID: Final = _wire.ACTION_ID
CAPSULE_SCHEMA: Final = "hegel-phase3a-q5b-purpose5-trust-amendment-capsule/1"
AUTHORIZATION_SCHEMA: Final = (
    "hegel-phase3a-q5b-purpose5-trust-amendment-authorization/1"
)
CONSUMPTION_SCHEMA: Final = (
    "hegel-phase3a-q5b-purpose5-trust-amendment-consumption/1"
)
PENDING_SCHEMA: Final = "hegel-phase3a-q5b-purpose5-trust-amendment-pending/1"
TERMINAL_SCHEMA: Final = "hegel-phase3a-q5b-purpose5-trust-amendment-terminal/1"

CAPSULE_READY: Final = "Q5B_PURPOSE5_TRUST_AMENDMENT_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = (
    "Q5B_PURPOSE5_TRUST_AMENDMENT_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED: Final = (
    "Q5B_PURPOSE5_TRUST_AMENDMENT_AUTHORIZATION_CONSUMED"
)
PENDING_NOT_CONSUMED: Final = (
    "Q5B_PURPOSE5_TRUST_AMENDMENT_PENDING_NOT_CONSUMED"
)
INVALIDATED_TERMINAL_NO_RESUME: Final = (
    "Q5B_PURPOSE5_TRUST_AMENDMENT_INVALIDATED_TERMINAL_NO_RESUME"
)
PARTIAL_FINALIZATION_REQUIRED: Final = (
    "Q5B_PURPOSE5_TRUST_AMENDMENT_PARTIAL_FINALIZATION_REQUIRED"
)
NOT_AUTHORIZED: Final = "Q5B_PURPOSE5_TRUST_AMENDMENT_NOT_AUTHORIZED_NOT_CONSUMED"
AUTHORIZED_SCOPE: Final = (
    "Q5B_EXACT_SEALED_Q5A_PUBLIC_TRUST_PIN_AND_CURRENT_STATUS_ACCEPTANCE_ONCE"
)
Q5A_SUCCESS_STATUS_TEXT: Final = _wire.Q5A_SUCCESS_STATUS.decode("ascii")

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q5B/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q5B/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q5B/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q5B/CONSUMPTION/V1\x00",
    "pending": b"HEGEL/Q5B/PENDING/V1\x00",
    "terminal": b"HEGEL/Q5B/TERMINAL/V1\x00",
}
Q5A_DOMAINS: Final = {
    "capsule": b"HEGEL/Q5A/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q5A/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q5A/CONSUMPTION/V1\x00",
    "terminal": b"HEGEL/Q5A/TERMINAL/V1\x00",
}

Q5B_SOURCE_PATHS: Final = (
    "src/hegel_machine/phase3_q5b_purpose5_trust_amendment_v1.py",
    "src/hegel_machine/phase3_q5b_completion_replay_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase3_q5b_purpose5_trust_amendment_v1.py",
)

EXECUTION_POLICY: Final = {
    "logical_campaign_limit": 1,
    "maximum_internal_popen_count": 0,
    "maximum_worker_popen_count": 0,
    "maximum_worker_openssl_popen_count": 0,
    "maximum_actor_start_count": 0,
    "maximum_docker_invocation_count": 0,
    "maximum_measurement_count": 0,
    "maximum_target_open_count": 0,
    "maximum_target_truth_access_count": 0,
    "maximum_split_access_count": 0,
    "maximum_role_evaluation_count": 0,
    "maximum_q1_read_count": 0,
    "maximum_q2_read_count": 0,
    "maximum_q3_read_count": 0,
    "maximum_q4_read_count": 0,
    "maximum_archive_read_count": 0,
    "maximum_custody_root_access_count": 0,
    "maximum_private_key_access_count": 0,
    "maximum_key_generation_count": 0,
    "maximum_key_import_count": 0,
    "maximum_signature_count": 0,
    "maximum_certificate_envelope_publication_count": 0,
    "maximum_certificate_issuance_count": 0,
    "maximum_trust_genesis_pin_count": 1,
    "maximum_current_key_status_acceptance_count": 1,
    "purpose5_active_signer_count_after_success": 3,
    "global_active_transition_allowed": False,
    "network_allowed": False,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_pending_allowed": False,
    "partial_finalization_artifact_to_terminal_only": True,
    "partial_finalization_terminal_to_trajectory_suffix_only": True,
}

AUTHORITY_BOUNDARY: Final = {
    "q5a_state": Q5A_SUCCESS_STATUS_TEXT,
    "q5b_disposition": (
        "PURPOSE5_TRUST_GENESIS_PINNED_CURRENT_STATUS_ACCEPTED_NOT_SIGNED_NOT_ISSUED"
    ),
    "purpose5_candidate_signer_count": 3,
    "purpose5_active_signer_count_after_success": 3,
    "purpose5_trust_genesis_pinned_after_success": True,
    "purpose5_current_key_status_accepted_after_success": True,
    "global_active_transition_allowed": False,
    "formal_certificate_signed": False,
    "certificate_envelope_published": False,
    "outside_certificate_issued": False,
    "external_authority": False,
    "synthesis_allowed": False,
    "mdl_allowed": False,
    "external_scientific_claim_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "production_allowed": False,
    "current_pointer_update_allowed": False,
}

_ZERO_RESTRICTED_ACTIONS: Final = {
    "internal_popen_count": 0,
    "worker_popen_count": 0,
    "worker_openssl_popen_count": 0,
    "actor_start_count": 0,
    "docker_invocation_count": 0,
    "measurement_count": 0,
    "target_open_count": 0,
    "target_truth_access_count": 0,
    "split_access_count": 0,
    "role_evaluation_count": 0,
    "q1_read_count": 0,
    "q2_read_count": 0,
    "q3_read_count": 0,
    "q4_read_count": 0,
    "archive_read_count": 0,
    "custody_root_access_count": 0,
    "private_key_access_count": 0,
    "key_generation_count": 0,
    "key_import_count": 0,
    "signature_count": 0,
    "certificate_envelope_publication_count": 0,
    "certificate_issuance_count": 0,
    "synthesis_action_count": 0,
    "mdl_action_count": 0,
    "network_accessed": False,
    "external_scientific_claim_made": False,
    "adoption_performed": False,
    "promotion_performed": False,
    "global_active_transition_performed": False,
    "production_accessed": False,
    "current_pointer_updated": False,
}


def q5b_action_audit(*, completed: bool) -> dict[str, Any]:
    if type(completed) is not bool:
        raise TypeError("completed must be one exact boolean")
    result = {
        "trust_genesis_pin_count": 1 if completed else 0,
        "current_key_status_acceptance_count": 1 if completed else 0,
        "purpose5_active_signer_count": 3 if completed else 0,
    }
    result.update(deepcopy(_ZERO_RESTRICTED_ACTIONS))
    return result


ZERO_ACTION_AUDIT: Final = q5b_action_audit(completed=False)
SUCCESS_ACTION_AUDIT: Final = q5b_action_audit(completed=True)

_Q5A_ZERO_RESTRICTED: Final = {
    "actor_start_count": 0,
    "docker_invocation_count": 0,
    "measurement_count": 0,
    "target_open_count": 0,
    "target_truth_access_count": 0,
    "split_access_count": 0,
    "role_evaluation_count": 0,
    "q1_read_count": 0,
    "q2_read_count": 0,
    "q3_read_count": 0,
    "key_import_count": 0,
    "signature_count": 0,
    "certificate_envelope_publication_count": 0,
    "certificate_issuance_count": 0,
    "synthesis_action_count": 0,
    "mdl_action_count": 0,
    "network_accessed": False,
    "external_scientific_claim_made": False,
    "private_key_exported": False,
    "private_key_repository_write_performed": False,
    "trust_genesis_pinned": False,
    "current_key_status_accepted": False,
    "adoption_performed": False,
    "promotion_performed": False,
    "active_transition_performed": False,
    "production_accessed": False,
    "current_pointer_updated": False,
}


def q5a_action_audit(completed_signer_count: int) -> dict[str, Any]:
    if type(completed_signer_count) is not int or not 0 <= completed_signer_count <= 3:
        raise ValueError("Q5a completed signer count differs")
    result = {
        "worker_popen_count": completed_signer_count,
        "worker_openssl_popen_count": 2 * completed_signer_count,
        "key_generation_count": completed_signer_count,
        "public_key_derivation_count": completed_signer_count,
        "public_receipt_count": completed_signer_count,
        "candidate_signer_count": completed_signer_count,
        "active_signer_count": 0,
    }
    result.update(deepcopy(_Q5A_ZERO_RESTRICTED))
    return result


Q5A_SUCCESS_ACTION_AUDIT: Final = q5a_action_audit(3)
Q5A_TRAJECTORY_EVENTS: Final = (
    "CAPSULE_PREPARED_NOT_AUTHORIZED",
    "AUTHORIZATION_RECORDED_NOT_CONSUMED",
    "Q5A_AUTHORIZATION_CONSUMED",
    "SIGNER_1_PUBLIC_RECEIPT_COMPLETE",
    "SIGNER_2_PUBLIC_RECEIPT_COMPLETE",
    "SIGNER_3_PUBLIC_RECEIPT_COMPLETE",
    Q5A_SUCCESS_STATUS_TEXT,
)

MAX_JSON_BYTES: Final = 32 * 1024 * 1024
MAX_Q5A_ARTIFACT_BYTES: Final = 1024 * 1024
MAX_TRAJECTORY_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class Q5bCompletionReplayError(ValueError):
    """One Q5b object or sealed public Q5a predecessor differs."""


def _fail(detail: str) -> NoReturn:
    raise Q5bCompletionReplayError(detail)


@dataclass(frozen=True, slots=True)
class LoadedJson:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Q5aSealedReplayV1:
    preflight_receipt: dict[str, Any]
    amendment_receipt: Any

    @property
    def artifact_payload(self) -> bytes:
        payload = getattr(self.amendment_receipt, "canonical_bytes", None)
        if type(payload) is not bytes:
            _fail("Q5b amendment receipt has no canonical bytes")
        return payload


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


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > MAX_JSON_BYTES:
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


def _source_commit(value: object, label: str) -> str:
    if type(value) is not str or _SOURCE_COMMIT.fullmatch(value) is None:
        _fail(f"{label} must be one Git SHA-1")
    return value


def load_pretty_json(path: Path, label: str) -> LoadedJson:
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail(f"{label} unavailable: {error}")
    return LoadedJson(path, raw, raw_sha256(raw), pretty_json_bytes(raw, label))


def load_preregistration(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q5b preregistration")
    value = loaded.value
    if (
        value.get("schema_version") != PREREGISTRATION_SCHEMA
        or value.get("preregistration_id") != PREREGISTRATION_ID
        or value.get("status") != "Q5B_SPEC_FROZEN_NOT_AUTHORIZED"
        or type(value.get("predecessor_q5a")) is not dict
    ):
        _fail("Q5b preregistration identity differs")
    return loaded


def load_contract(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q5b contract")
    value = loaded.value
    if (
        value.get("schema_version") != CONTRACT_SCHEMA
        or value.get("contract_id") != CONTRACT_ID
        or value.get("action_id") != ACTION_ID
        or type(value.get("fixed_paths")) is not dict
        or type(value.get("q5a_inputs")) is not dict
        or value.get("execution_policy") != EXECUTION_POLICY
        or value.get("authority_boundary") != AUTHORITY_BOUNDARY
    ):
        _fail("Q5b contract identity differs")
    prereg = load_preregistration(Path(value.get("preregistration_path", "")))
    if prereg.raw_sha256 != value.get("preregistration_raw_sha256"):
        _fail("Q5b preregistration raw identity differs")
    return loaded


def _semantic_digest(value: Mapping[str, Any], kind: str, field: str) -> str:
    unsigned = deepcopy(dict(value))
    supplied = unsigned.pop(field, None)
    expected = _digest(Q5A_DOMAINS[kind], unsigned)
    if supplied != expected:
        _fail(f"Q5a {kind} semantic digest differs")
    return expected


def _expected_predecessor(preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    value = preregistration.get("predecessor_q5a")
    if type(value) is not dict:
        _fail("Q5b predecessor Q5a registry differs")
    return value


def _expected_input(predecessor: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    value = predecessor.get(kind)
    if type(value) is not dict:
        _fail(f"Q5b predecessor Q5a {kind} registry differs")
    _tagged(value.get("raw_sha256"), f"Q5a {kind} raw digest")
    return value


def _replay_q5a_trajectory(
    payload: bytes,
    *,
    capsule_digest: str,
    authorization_digest: str,
    consumption_digest: str,
    terminal_digest: str,
    candidate_root: str,
) -> list[dict[str, Any]]:
    if type(payload) is not bytes or len(payload) > MAX_TRAJECTORY_BYTES:
        _fail("Q5a trajectory exceeds its byte bound")
    if not payload.endswith(b"\n"):
        _fail("Q5a trajectory framing differs")
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(Q5A_TRAJECTORY_EVENTS):
        _fail("Q5a trajectory event count differs")
    rows = [
        strict_json_bytes(line, f"Q5a trajectory event {index}")
        for index, line in enumerate(lines, 1)
    ]
    prior_time = -1
    for index, row in enumerate(rows):
        observed_time = row.get("time_ns")
        if type(observed_time) is not int or observed_time < prior_time:
            _fail("Q5a trajectory time sequence differs")
        prior_time = observed_time
        if index == 0:
            authorization = consumption = None
            prefix: list[int] = []
            count = 0
            audit = q5a_action_audit(0)
        elif index == 1:
            authorization = authorization_digest
            consumption = None
            prefix = []
            count = 0
            audit = q5a_action_audit(0)
        elif index == 2:
            authorization = authorization_digest
            consumption = consumption_digest
            prefix = []
            count = 1
            audit = q5a_action_audit(0)
        else:
            completed = min(index - 2, 3)
            authorization = authorization_digest
            consumption = consumption_digest
            prefix = list(range(1, completed + 1))
            count = 1
            audit = q5a_action_audit(completed)
        expected: dict[str, Any] = {
            "schema_version": "hegel-phase3a-q5a-trajectory/1",
            "event": Q5A_TRAJECTORY_EVENTS[index],
            "capsule_digest": capsule_digest,
            "authorization_digest": authorization,
            "consumption_record_digest": consumption,
            "completed_prefix": prefix,
            "authorization_consumption_count": count,
            "logical_campaign_count": count,
            "q5a_action_audit": audit,
            "time_ns": observed_time,
        }
        if index == 6:
            expected.update(
                {
                    "terminal_record_digest": terminal_digest,
                    "candidate_root": candidate_root,
                    "failure_code": None,
                }
            )
        if row != expected:
            _fail(f"Q5a trajectory event {index + 1} replay differs")
    return rows


def replay_q5a_predecessor_v1(
    preregistration: Mapping[str, Any],
    *,
    q5b_source_commit: str,
    q5b_source_closure_root: str,
    capsule_payload: bytes,
    authorization_payload: bytes,
    consumption_payload: bytes,
    terminal_payload: bytes,
    artifact_payload: bytes,
    trajectory_payload: bytes,
) -> Q5aSealedReplayV1:
    """Replay the exact six public Q5a inputs and derive one Q5b receipt."""

    predecessor = _expected_predecessor(preregistration)
    q5b_commit = _source_commit(q5b_source_commit, "Q5b source commit")
    q5b_closure = _tagged(q5b_source_closure_root, "Q5b source closure")
    payloads = {
        "capsule": capsule_payload,
        "authorization": authorization_payload,
        "consumption": consumption_payload,
        "terminal": terminal_payload,
        "artifact": artifact_payload,
        "trajectory": trajectory_payload,
    }
    selected: list[dict[str, Any]] = []
    for kind, payload in payloads.items():
        expected = _expected_input(predecessor, kind)
        actual = raw_sha256(payload)
        if actual != expected.get("raw_sha256"):
            _fail(f"Q5a {kind} raw SHA-256 differs")
        selected.append(
            {"kind": kind, "byte_count": len(payload), "raw_sha256": actual}
        )

    capsule = strict_json_bytes(capsule_payload, "Q5a capsule")
    authorization = strict_json_bytes(authorization_payload, "Q5a authorization")
    consumption = strict_json_bytes(consumption_payload, "Q5a consumption")
    terminal = strict_json_bytes(terminal_payload, "Q5a terminal")
    capsule_digest = _semantic_digest(capsule, "capsule", "capsule_digest")
    authorization_digest = _semantic_digest(
        authorization, "authorization", "authorization_digest"
    )
    consumption_digest = _semantic_digest(consumption, "consumption", "record_digest")
    terminal_digest = _semantic_digest(terminal, "terminal", "record_digest")
    for kind, field, actual in (
        ("capsule", "capsule_digest", capsule_digest),
        ("authorization", "authorization_digest", authorization_digest),
        ("consumption", "record_digest", consumption_digest),
        ("terminal", "record_digest", terminal_digest),
    ):
        if _expected_input(predecessor, kind).get(field) != actual:
            _fail(f"Q5a {kind} configured semantic digest differs")

    outcome = terminal.get("outcome")
    if (
        capsule.get("status")
        != "Q5A_PURPOSE5_SIGNER_GENESIS_CAPSULE_READY_NOT_AUTHORIZED"
        or authorization.get("status")
        != "Q5A_PURPOSE5_SIGNER_GENESIS_AUTHORIZATION_RECORDED_NOT_CONSUMED"
        or consumption.get("status")
        != "Q5A_PURPOSE5_SIGNER_GENESIS_AUTHORIZATION_CONSUMED"
        or terminal.get("status") != Q5A_SUCCESS_STATUS_TEXT
        or type(outcome) is not dict
        or outcome.get("status") != Q5A_SUCCESS_STATUS_TEXT
        or authorization.get("capsule_digest") != capsule_digest
        or consumption.get("capsule_digest") != capsule_digest
        or terminal.get("capsule_digest") != capsule_digest
        or consumption.get("authorization_digest") != authorization_digest
        or terminal.get("authorization_digest") != authorization_digest
        or terminal.get("consumption_record_digest") != consumption_digest
        or terminal.get("authorization_consumption_count") != 1
        or terminal.get("logical_campaign_count") != 1
        or capsule.get("source_commit") != predecessor.get("source_commit")
        or capsule.get("source_closure_root") != predecessor.get("source_closure_root")
        or outcome.get("completed_prefix") != [1, 2, 3]
        or type(outcome.get("completed_public_receipts")) is not list
        or len(outcome["completed_public_receipts"]) != 3
        or outcome.get("artifact_raw_sha256") != raw_sha256(artifact_payload)
        or outcome.get("q5a_action_audit") != Q5A_SUCCESS_ACTION_AUDIT
        or outcome.get("purpose5_candidate_signer_count") != 3
        or outcome.get("purpose5_active_signer_count") != 0
        or outcome.get("purpose5_trust_genesis_pinned") is not False
        or outcome.get("purpose5_current_key_status_accepted") is not False
        or outcome.get("formal_certificate_signed") is not False
        or outcome.get("certificate_envelope_published") is not False
        or outcome.get("outside_certificate_issued") is not False
        or outcome.get("external_authority") is not False
        or outcome.get("synthesis_released") is not False
        or outcome.get("terminal_no_resume") is not False
        or outcome.get("campaign_complete") is not True
    ):
        _fail("Q5a predecessor completion boundary differs")

    try:
        amendment = _wire.build_q5b_trust_amendment_v1(
            artifact_payload,
            q5a_capsule_digest=bytes.fromhex(capsule_digest[7:]),
            q5a_authorization_digest=bytes.fromhex(authorization_digest[7:]),
            q5a_consumption_record_digest=bytes.fromhex(consumption_digest[7:]),
            q5a_terminal_record_digest=bytes.fromhex(terminal_digest[7:]),
            q5a_source_commit=predecessor.get("source_commit"),
            q5a_source_closure_root=bytes.fromhex(
                _tagged(
                    predecessor.get("source_closure_root"),
                    "Q5a source closure",
                )[7:]
            ),
            q5b_source_commit=q5b_commit,
            q5b_source_closure_root=bytes.fromhex(q5b_closure[7:]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"Q5a candidate/Q5b amendment replay rejected: {error}")
    q4_lineage = capsule.get("preflight_receipt")
    if type(q4_lineage) is dict:
        q4_lineage = q4_lineage.get("q4_predecessor_receipt")
    candidate_object = amendment.q5a_candidate.canonical_object
    if (
        type(q4_lineage) is not dict
        or candidate_object[3] != bytes.fromhex(q4_lineage.get("q4_capsule_digest", "")[7:])
        or candidate_object[4] != bytes.fromhex(q4_lineage.get("q4_authorization_digest", "")[7:])
        or candidate_object[5] != bytes.fromhex(q4_lineage.get("q4_consumption_record_digest", "")[7:])
        or candidate_object[6] != bytes.fromhex(q4_lineage.get("q4_terminal_record_digest", "")[7:])
        or candidate_object[7] != bytes.fromhex(q4_lineage.get("q4_capability_receipt_root", "")[7:])
        or candidate_object[8] != bytes.fromhex(q4_lineage.get("q4_quotient_body_hash", "")[7:])
        or candidate_object[9] != (1, bytes.fromhex(q4_lineage.get("q4_source_commit", "")))
        or candidate_object[10] != bytes.fromhex(q4_lineage.get("q4_source_closure_root", "")[7:])
        or outcome.get("q4_capability_receipt_root")
        != q4_lineage.get("q4_capability_receipt_root")
        or outcome.get("q4_quotient_body_hash")
        != q4_lineage.get("q4_quotient_body_hash")
        or amendment.q5a_candidate.q5a_source_commit_raw.hex()
        != predecessor.get("source_commit")
        or "sha256:" + amendment.q5a_candidate.q5a_source_closure_root.hex()
        != predecessor.get("source_closure_root")
    ):
        _fail("Q5a capsule/Q4 lineage/artifact projection differs")
    candidate_root = "sha256:" + amendment.q5a_candidate_root.hex()
    unpinned_root = "sha256:" + amendment.q5a_unpinned_genesis_root.hex()
    genesis_only_status = "sha256:" + amendment.q5a_genesis_only_status_root.hex()
    manifest_roots = ["sha256:" + item.hex() for item in amendment.q5a_manifest_roots]
    key_ids = [item.hex() for item in amendment.q5a_key_ids]
    expected_artifact = _expected_input(predecessor, "artifact")
    if (
        outcome.get("candidate_root") != candidate_root
        or expected_artifact.get("candidate_root") != candidate_root
        or expected_artifact.get("unpinned_genesis_root") != unpinned_root
        or expected_artifact.get("genesis_only_status_root") != genesis_only_status
        or expected_artifact.get("manifest_roots") != manifest_roots
        or expected_artifact.get("key_ids") != key_ids
    ):
        _fail("Q5a candidate public object graph differs")

    rows = _replay_q5a_trajectory(
        trajectory_payload,
        capsule_digest=capsule_digest,
        authorization_digest=authorization_digest,
        consumption_digest=consumption_digest,
        terminal_digest=terminal_digest,
        candidate_root=candidate_root,
    )
    trajectory_expected = _expected_input(predecessor, "trajectory")
    if trajectory_expected.get("event_count") != 7:
        _fail("Q5a configured trajectory event count differs")

    q5b_payload = amendment.canonical_bytes
    derived = {
        "raw_sha256": raw_sha256(q5b_payload),
        "receipt_root": "sha256:" + amendment.receipt_root.hex(),
        "pinned_trust_genesis_root": (
            "sha256:" + amendment.pinned_trust_genesis.genesis_root.hex()
        ),
        "current_key_status_root": (
            "sha256:" + amendment.current_key_status.status_root.hex()
        ),
    }
    preflight = {
        "schema_version": "hegel-phase3a-q5b-purpose5-trust-amendment-preflight/1",
        "selected_q5a_inputs": selected,
        "selected_q5a_input_count": 6,
        "selected_q5a_input_total_bytes": sum(len(item) for item in payloads.values()),
        "q5a_capsule_digest": capsule_digest,
        "q5a_authorization_digest": authorization_digest,
        "q5a_consumption_record_digest": consumption_digest,
        "q5a_terminal_record_digest": terminal_digest,
        "q5a_artifact_raw_sha256": raw_sha256(artifact_payload),
        "q5a_source_commit": predecessor.get("source_commit"),
        "q5a_source_closure_root": predecessor.get("source_closure_root"),
        "q5b_source_commit": q5b_commit,
        "q5b_source_closure_root": q5b_closure,
        "q5a_candidate_root": candidate_root,
        "q5a_unpinned_genesis_root": unpinned_root,
        "q5a_genesis_only_status_root": genesis_only_status,
        "q5a_manifest_roots": manifest_roots,
        "q5a_key_ids": key_ids,
        "q5a_trajectory_audit": {
            "raw_sha256": raw_sha256(trajectory_payload),
            "byte_count": len(trajectory_payload),
            "event_count": 7,
            "events": [row["event"] for row in rows],
            "success_action_audit_replayed": True,
        },
        "expected_q5b_artifact_raw_sha256": derived["raw_sha256"],
        "expected_q5b_receipt_root": derived["receipt_root"],
        "expected_pinned_trust_genesis_root": derived["pinned_trust_genesis_root"],
        "expected_current_key_status_root": derived["current_key_status_root"],
        "producer_free_read_only_replay_completed": True,
        "formal_codec_round_trip_completed": True,
        "custody_root_access_count": 0,
        "private_key_access_count": 0,
        "q5b_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }
    return Q5aSealedReplayV1(preflight, amendment)


def validate_source_closure_rows(rows: object) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(Q5B_SOURCE_PATHS):
        _fail("Q5b source closure row count differs")
    checked: list[dict[str, Any]] = []
    for row, expected_path in zip(rows, Q5B_SOURCE_PATHS, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"relative_path", "byte_count", "raw_sha256"}
            or row.get("relative_path") != expected_path
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 1
        ):
            _fail("Q5b source closure row differs")
        _tagged(row.get("raw_sha256"), "Q5b source row digest")
        checked.append(deepcopy(row))
    return checked


def source_closure_root(rows: object) -> str:
    return _digest(DOMAINS["source_closure"], validate_source_closure_rows(rows))


def validate_preflight_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5b preflight receipt differs")
    required = {
        "schema_version", "selected_q5a_inputs", "selected_q5a_input_count",
        "selected_q5a_input_total_bytes", "q5a_capsule_digest",
        "q5a_authorization_digest", "q5a_consumption_record_digest",
        "q5a_terminal_record_digest", "q5a_artifact_raw_sha256",
        "q5a_source_commit", "q5a_source_closure_root",
        "q5b_source_commit", "q5b_source_closure_root",
        "q5a_candidate_root", "q5a_unpinned_genesis_root",
        "q5a_genesis_only_status_root", "q5a_manifest_roots", "q5a_key_ids",
        "q5a_trajectory_audit", "expected_q5b_artifact_raw_sha256",
        "expected_q5b_receipt_root", "expected_pinned_trust_genesis_root",
        "expected_current_key_status_root", "producer_free_read_only_replay_completed",
        "formal_codec_round_trip_completed", "custody_root_access_count",
        "private_key_access_count", "q5b_action_audit",
    }
    trajectory = value.get("q5a_trajectory_audit")
    if (
        set(value) != required
        or value.get("schema_version")
        != "hegel-phase3a-q5b-purpose5-trust-amendment-preflight/1"
        or value.get("selected_q5a_input_count") != 6
        or type(value.get("selected_q5a_inputs")) is not list
        or [item.get("kind") for item in value["selected_q5a_inputs"]]
        != ["capsule", "authorization", "consumption", "terminal", "artifact", "trajectory"]
        or type(trajectory) is not dict
        or trajectory.get("event_count") != 7
        or trajectory.get("events") != list(Q5A_TRAJECTORY_EVENTS)
        or trajectory.get("success_action_audit_replayed") is not True
        or value.get("producer_free_read_only_replay_completed") is not True
        or value.get("formal_codec_round_trip_completed") is not True
        or value.get("custody_root_access_count") != 0
        or value.get("private_key_access_count") != 0
        or value.get("q5b_action_audit") != ZERO_ACTION_AUDIT
        or type(value.get("q5a_manifest_roots")) is not list
        or len(value["q5a_manifest_roots"]) != 3
        or type(value.get("q5a_key_ids")) is not list
        or len(value["q5a_key_ids"]) != 3
    ):
        _fail("Q5b preflight receipt differs")
    for field in (
        "q5a_capsule_digest", "q5a_authorization_digest",
        "q5a_consumption_record_digest", "q5a_terminal_record_digest",
        "q5a_artifact_raw_sha256", "q5a_candidate_root",
        "q5a_unpinned_genesis_root", "q5a_genesis_only_status_root",
        "q5a_source_closure_root", "q5b_source_closure_root",
        "expected_q5b_artifact_raw_sha256", "expected_q5b_receipt_root",
        "expected_pinned_trust_genesis_root", "expected_current_key_status_root",
    ):
        _tagged(value.get(field), field)
    _source_commit(value.get("q5a_source_commit"), "Q5a source commit")
    _source_commit(value.get("q5b_source_commit"), "Q5b source commit")
    _tagged(trajectory.get("raw_sha256"), "Q5a trajectory raw digest")
    for item in value["q5a_manifest_roots"]:
        _tagged(item, "Q5a manifest root")
    if any(type(item) is not str or re.fullmatch(r"[0-9a-f]{32}", item) is None for item in value["q5a_key_ids"]):
        _fail("Q5a key id list differs")
    return deepcopy(value)


def _validate_preflight_binding(
    preflight: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    predecessor = _expected_predecessor(preregistration)
    actual_rows = [(item.get("kind"), item.get("raw_sha256")) for item in preflight["selected_q5a_inputs"]]
    expected_rows = [(kind, _expected_input(predecessor, kind).get("raw_sha256")) for kind in ("capsule", "authorization", "consumption", "terminal", "artifact", "trajectory")]
    expected = {
        "q5a_capsule_digest": _expected_input(predecessor, "capsule").get("capsule_digest"),
        "q5a_authorization_digest": _expected_input(predecessor, "authorization").get("authorization_digest"),
        "q5a_consumption_record_digest": _expected_input(predecessor, "consumption").get("record_digest"),
        "q5a_terminal_record_digest": _expected_input(predecessor, "terminal").get("record_digest"),
        "q5a_artifact_raw_sha256": _expected_input(predecessor, "artifact").get("raw_sha256"),
        "q5a_source_commit": predecessor.get("source_commit"),
        "q5a_source_closure_root": predecessor.get("source_closure_root"),
        "q5a_candidate_root": _expected_input(predecessor, "artifact").get("candidate_root"),
        "q5a_unpinned_genesis_root": _expected_input(predecessor, "artifact").get("unpinned_genesis_root"),
        "q5a_genesis_only_status_root": _expected_input(predecessor, "artifact").get("genesis_only_status_root"),
        "q5a_manifest_roots": _expected_input(predecessor, "artifact").get("manifest_roots"),
        "q5a_key_ids": _expected_input(predecessor, "artifact").get("key_ids"),
    }
    if (
        actual_rows != expected_rows
        or any(preflight.get(key) != value for key, value in expected.items())
        or preflight["q5a_trajectory_audit"].get("raw_sha256")
        != _expected_input(predecessor, "trajectory").get("raw_sha256")
        or predecessor.get("selected_input_total_bytes") not in (None, preflight["selected_q5a_input_total_bytes"])
    ):
        _fail("Q5b capsule predecessor binding differs")


def build_capsule(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    rows = validate_source_closure_rows(inputs.get("source_closure_rows"))
    closure = source_closure_root(rows)
    if inputs.get("source_closure_root") != closure:
        _fail("Q5b source closure root differs")
    source_commit = _source_commit(inputs.get("source_commit"), "Q5b source commit")
    run_id = inputs.get("run_id")
    created = inputs.get("created_at_unix_seconds")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        _fail("Q5b run id differs")
    if type(created) is not int or created < 0:
        _fail("Q5b creation time differs")
    preflight = validate_preflight_receipt(inputs.get("preflight_receipt"))
    _validate_preflight_binding(preflight, preregistration)
    if (
        preflight["q5b_source_commit"] != source_commit
        or preflight["q5b_source_closure_root"] != closure
    ):
        _fail("Q5b preflight/source closure binding differs")
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "contract_raw_sha256": contract.raw_sha256,
        "preregistration_raw_sha256": contract.value["preregistration_raw_sha256"],
        "action_id": ACTION_ID,
        "source_commit": source_commit,
        "source_closure_rows": rows,
        "source_closure_root": closure,
        "run_id": run_id,
        "created_at_unix_seconds": created,
        "preflight_receipt": preflight,
        "execution_policy": deepcopy(EXECUTION_POLICY),
        "authority_boundary": deepcopy(AUTHORITY_BOUNDARY),
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule(value: object, contract: LoadedJson, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5b capsule differs")
    expected = build_capsule(
        contract,
        preregistration,
        {key: value.get(key) for key in ("source_commit", "source_closure_rows", "source_closure_root", "run_id", "created_at_unix_seconds", "preflight_receipt")},
    )
    if value != expected:
        _fail("Q5b capsule replay differs")
    return deepcopy(expected)


def build_authorization(contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_capsule(capsule, contract, preregistration)
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "authorized_scope": AUTHORIZED_SCOPE,
        "logical_campaign_limit": 1,
        "trust_genesis_pin_count_limit": 1,
        "current_key_status_acceptance_count_limit": 1,
        "purpose5_active_signer_count_after_success": 3,
        "q5b_prohibited_action_audit_limit": deepcopy(_ZERO_RESTRICTED_ACTIONS),
        "global_active_transition_allowed": False,
        "certificate_signing_allowed": False,
        "certificate_envelope_publication_allowed": False,
        "certificate_issuance_allowed": False,
        "synthesis_allowed": False,
        "network_allowed": False,
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def validate_authorization(value: object, contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_authorization(contract, preregistration, capsule)
    if value != expected:
        _fail("Q5b authorization replay differs")
    return deepcopy(expected)


def build_consumption(contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any]) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(authorization, contract, preregistration, checked_capsule)
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": AUTHORIZATION_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "q5b_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "purpose5_active_signer_count": 0,
        "purpose5_trust_genesis_pinned": False,
        "purpose5_current_key_status_accepted": False,
        "global_active_transition_performed": False,
        "formal_certificate_signed": False,
        "outside_certificate_issued": False,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def validate_consumption(value: object, contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_consumption(contract, preregistration, capsule, authorization)
    if value != expected:
        _fail("Q5b consumption replay differs")
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
    authorization_digest = None
    if authorization is not None:
        authorization_digest = validate_authorization(authorization, contract, preregistration, checked)["authorization_digest"]
    if any(type(item) is not str or not item for item in (phase, condition, detail)):
        _fail("Q5b pending fields differ")
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
        "q5b_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_artifact_binding(payload: bytes, q5a_artifact_payload: bytes, capsule: Mapping[str, Any]) -> Any:
    if type(payload) is not bytes or type(q5a_artifact_payload) is not bytes:
        _fail("Q5b/Q5a artifact payload type differs")
    preflight = validate_preflight_receipt(capsule.get("preflight_receipt"))
    try:
        receipt = _wire.validate_q5b_receipt_authority_v1(
            payload,
            q5a_artifact_payload,
            expected_q5a_capsule_digest=bytes.fromhex(
                preflight["q5a_capsule_digest"][7:]
            ),
            expected_q5a_authorization_digest=bytes.fromhex(
                preflight["q5a_authorization_digest"][7:]
            ),
            expected_q5a_consumption_record_digest=bytes.fromhex(
                preflight["q5a_consumption_record_digest"][7:]
            ),
            expected_q5a_terminal_record_digest=bytes.fromhex(
                preflight["q5a_terminal_record_digest"][7:]
            ),
            expected_q5a_artifact_raw_sha256=bytes.fromhex(
                preflight["q5a_artifact_raw_sha256"][7:]
            ),
            expected_q5a_source_commit=preflight["q5a_source_commit"],
            expected_q5a_source_closure_root=bytes.fromhex(
                preflight["q5a_source_closure_root"][7:]
            ),
            expected_q5b_source_commit=capsule["source_commit"],
            expected_q5b_source_closure_root=bytes.fromhex(
                capsule["source_closure_root"][7:]
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(f"Q5b artifact replay rejected: {error}")
    if (
        raw_sha256(payload) != preflight["expected_q5b_artifact_raw_sha256"]
        or "sha256:" + receipt.receipt_root.hex() != preflight["expected_q5b_receipt_root"]
        or "sha256:" + receipt.pinned_trust_genesis.genesis_root.hex() != preflight["expected_pinned_trust_genesis_root"]
        or "sha256:" + receipt.current_key_status.status_root.hex() != preflight["expected_current_key_status_root"]
    ):
        _fail("Q5b artifact/capsule binding differs")
    return receipt


def build_complete_terminal(contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any], consumption: Mapping[str, Any], artifact_payload: bytes, q5a_artifact_payload: bytes) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(authorization, contract, preregistration, checked_capsule)
    checked_consumption = validate_consumption(consumption, contract, preregistration, checked_capsule, checked_authorization)
    receipt = validate_artifact_binding(artifact_payload, q5a_artifact_payload, checked_capsule)
    outcome = {
        "status": _wire.Q5B_SUCCESS_STATUS,
        "disposition": "PURPOSE5_TRUST_GENESIS_PINNED_CURRENT_STATUS_ACCEPTED_NOT_SIGNED_NOT_ISSUED",
        "capsule_digest": checked_capsule["capsule_digest"],
        "artifact_raw_sha256": raw_sha256(artifact_payload),
        "trust_amendment_receipt_root": "sha256:" + receipt.receipt_root.hex(),
        "pinned_trust_genesis_root": "sha256:" + receipt.pinned_trust_genesis.genesis_root.hex(),
        "current_key_status_root": "sha256:" + receipt.current_key_status.status_root.hex(),
        "q5a_candidate_root": "sha256:" + receipt.q5a_candidate_root.hex(),
        "q5a_manifest_roots": ["sha256:" + item.hex() for item in receipt.q5a_manifest_roots],
        "q5a_key_ids": [item.hex() for item in receipt.q5a_key_ids],
        "q5b_action_audit": deepcopy(SUCCESS_ACTION_AUDIT),
        "producer_free_q5a_replay_completed": True,
        "purpose5_active_signer_count": 3,
        "purpose5_trust_genesis_pinned": True,
        "purpose5_current_key_status_accepted": True,
        "global_active_transition_performed": False,
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
        "external_authority": False,
        "synthesis_released": False,
        "terminal_no_resume": False,
        "campaign_complete": True,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": _wire.Q5B_SUCCESS_STATUS,
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


def build_invalidated_terminal(contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any], consumption: Mapping[str, Any], *, failure_phase: str, failure_code: str, failure_detail: str) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(authorization, contract, preregistration, checked_capsule)
    checked_consumption = validate_consumption(consumption, contract, preregistration, checked_capsule, checked_authorization)
    if any(type(item) is not str or not item for item in (failure_phase, failure_code, failure_detail)):
        _fail("Q5b invalidated failure fields differ")
    outcome = {
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": checked_capsule["capsule_digest"],
        "artifact_raw_sha256": None,
        "trust_amendment_receipt_root": None,
        "pinned_trust_genesis_root": None,
        "current_key_status_root": None,
        "q5b_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "purpose5_active_signer_count": 0,
        "purpose5_trust_genesis_pinned": False,
        "purpose5_current_key_status_accepted": False,
        "global_active_transition_performed": False,
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
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


def validate_terminal(value: object, contract: LoadedJson, preregistration: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any], consumption: Mapping[str, Any], *, artifact_payload: bytes | None = None, q5a_artifact_payload: bytes | None = None) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("Q5b terminal must contain an outcome")
    if value.get("status") == _wire.Q5B_SUCCESS_STATUS:
        if artifact_payload is None or q5a_artifact_payload is None:
            _fail("Q5b complete terminal requires both sealed public artifacts")
        expected = build_complete_terminal(contract, preregistration, capsule, authorization, consumption, artifact_payload, q5a_artifact_payload)
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME:
        if artifact_payload is not None:
            _fail("Q5b invalidated terminal forbids a valid amendment artifact")
        outcome = value["outcome"]
        expected = build_invalidated_terminal(
            contract, preregistration, capsule, authorization, consumption,
            failure_phase=outcome.get("failure_phase"),
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
        )
    else:
        _fail("Q5b terminal status differs")
    if value != expected:
        _fail("Q5b terminal replay differs")
    return deepcopy(expected)


__all__ = [
    "ACTION_ID", "AUTHORIZATION_CONSUMED", "AUTHORIZATION_RECORDED",
    "AUTHORIZATION_SCHEMA", "AUTHORIZED_SCOPE", "AUTHORITY_BOUNDARY",
    "CAPSULE_READY", "CAPSULE_SCHEMA", "CONSUMPTION_SCHEMA", "CONTRACT_ID",
    "CONTRACT_SCHEMA", "DOMAINS", "EXECUTION_POLICY",
    "INVALIDATED_TERMINAL_NO_RESUME", "LoadedJson", "MAX_JSON_BYTES",
    "MAX_Q5A_ARTIFACT_BYTES", "MAX_TRAJECTORY_BYTES", "NOT_AUTHORIZED",
    "PARTIAL_FINALIZATION_REQUIRED", "PENDING_NOT_CONSUMED", "PENDING_SCHEMA",
    "PREREGISTRATION_ID", "PREREGISTRATION_SCHEMA", "Q5A_SUCCESS_ACTION_AUDIT",
    "Q5A_TRAJECTORY_EVENTS", "Q5B_SOURCE_PATHS", "Q5aSealedReplayV1",
    "Q5bCompletionReplayError", "SUCCESS_ACTION_AUDIT", "TERMINAL_SCHEMA",
    "ZERO_ACTION_AUDIT", "build_authorization", "build_capsule",
    "build_complete_terminal", "build_consumption", "build_invalidated_terminal",
    "build_pending", "canonical_json_bytes", "load_contract",
    "load_preregistration", "pretty_json_bytes", "q5a_action_audit",
    "q5b_action_audit", "raw_sha256", "replay_q5a_predecessor_v1",
    "source_closure_root", "strict_json_bytes", "validate_artifact_binding",
    "validate_authorization", "validate_capsule", "validate_consumption",
    "validate_preflight_receipt", "validate_source_closure_rows",
    "validate_terminal",
]
