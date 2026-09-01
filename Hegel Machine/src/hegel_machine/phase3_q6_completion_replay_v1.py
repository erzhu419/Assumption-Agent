"""Pure state and completion replay for Phase-3A Q6 issuance.

The module is deliberately producer-free with respect to Q4 and Q5b.  It
replays their sealed public artifacts through the Q6 wire, validates the six
Q5b outer-authority records, and models the three one-shot Purpose-5 signing
actions.  It never opens a custody root or invokes a process.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Final, Mapping, NoReturn, Sequence

from . import phase3_q6_formal_quotient_certificate_issuance_v1 as _wire


CONTRACT_SCHEMA: Final = "hegel-phase3a-q6-formal-quotient-certificate-issuance-contract/1"
CONTRACT_ID: Final = "hegel-phase3a-q6-formal-quotient-certificate-issuance-actual-v1"
PREREGISTRATION_SCHEMA: Final = (
    "hegel-phase3a-q6-formal-quotient-certificate-issuance-preregistration/1"
)
PREREGISTRATION_ID: Final = "hegel-phase3a-q6-formal-quotient-certificate-issuance-v1"
ACTION_ID: Final = _wire.Q6_ACTION_ID

CAPSULE_SCHEMA: Final = "hegel-phase3a-q6-formal-quotient-certificate-issuance-capsule/1"
AUTHORIZATION_SCHEMA: Final = (
    "hegel-phase3a-q6-formal-quotient-certificate-issuance-authorization/1"
)
CONSUMPTION_SCHEMA: Final = (
    "hegel-phase3a-q6-formal-quotient-certificate-issuance-consumption/1"
)
PENDING_SCHEMA: Final = "hegel-phase3a-q6-formal-quotient-certificate-issuance-pending/1"
CHECKPOINT_SCHEMA: Final = (
    "hegel-phase3a-q6-formal-quotient-certificate-issuance-checkpoint/1"
)
SIGNING_INTENT_SCHEMA: Final = "hegel-phase3a-q6-purpose5-signing-intent/1"
VERIFICATION_INTENT_SCHEMA: Final = (
    "hegel-phase3a-q6-purpose5-signature-verification-intent/1"
)
SIGNATURE_RECEIPT_SCHEMA: Final = (
    "hegel-phase3a-q6-purpose5-verified-signature-receipt/1"
)
WORKER_RECEIPT_SCHEMA: Final = "hegel-phase3-q6-purpose5-signing-public-receipt/1"
WORKER_SMOKE_SCHEMA: Final = (
    "hegel-phase3-q6-purpose5-signing-startup-smoke-receipt/1"
)
WORKER_SUCCESS_STATUS: Final = (
    "PURPOSE5_SIGNATURE_COMPLETE_PENDING_PRODUCER_FREE_VERIFICATION"
)
WORKER_SMOKE_STATUS: Final = "Q6_PURPOSE5_SIGNING_STARTUP_SMOKE_PASS"
TERMINAL_SCHEMA: Final = "hegel-phase3a-q6-formal-quotient-certificate-issuance-terminal/1"

CAPSULE_READY: Final = "Q6_PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = (
    "Q6_PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED: Final = (
    "Q6_PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_AUTHORIZATION_CONSUMED"
)
PENDING_NOT_CONSUMED: Final = "Q6_PURPOSE5_CERTIFICATE_ISSUANCE_PENDING_NOT_CONSUMED"
PENDING_CLEAN_CHECKPOINT: Final = (
    "Q6_PURPOSE5_CERTIFICATE_ISSUANCE_PENDING_CLEAN_CHECKPOINT"
)
CLEAN_CHECKPOINT: Final = "Q6_PURPOSE5_CERTIFICATE_ISSUANCE_CLEAN_CHECKPOINT"
SIGNING_INTENT_RECORDED: Final = "Q6_PURPOSE5_SIGNING_INTENT_RECORDED"
VERIFICATION_INTENT_RECORDED: Final = (
    "Q6_PURPOSE5_SIGNATURE_VERIFICATION_INTENT_RECORDED"
)
SIGNATURE_VERIFIED: Final = "Q6_PURPOSE5_SIGNATURE_VERIFIED"
INVALIDATED_TERMINAL_NO_RESUME: Final = (
    "Q6_PURPOSE5_CERTIFICATE_ISSUANCE_INVALIDATED_TERMINAL_NO_RESUME"
)
PARTIAL_FINALIZATION_REQUIRED: Final = (
    "Q6_PURPOSE5_CERTIFICATE_ISSUANCE_PARTIAL_FINALIZATION_REQUIRED"
)
NOT_AUTHORIZED: Final = (
    "Q6_PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_NOT_AUTHORIZED_NOT_CONSUMED"
)
AUTHORIZED_SCOPE: Final = (
    "Q6_EXACT_Q5B_CURRENT_PURPOSE5_THREE_OF_THREE_FORMAL_QUOTIENT_CERTIFICATE_ISSUANCE_ONCE"
)

SIGNER_ORDINALS: Final = _wire.SIGNER_ORDINALS
SIGNER_ROLE_NAMES: Final = _wire.SIGNER_ROLE_NAMES
NEXT_SIGNER_PREFIXES: Final = ((), (1,), (1, 2))
CLEAN_PREFIXES: Final = NEXT_SIGNER_PREFIXES
COMPLETE_PREFIX: Final = (1, 2, 3)
PENDING_PREFIXES: Final = NEXT_SIGNER_PREFIXES + (COMPLETE_PREFIX,)

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q6/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q6/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q6/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q6/CONSUMPTION/V1\x00",
    "intent": b"HEGEL/Q6/SIGNING_INTENT/V1\x00",
    "verification_intent": b"HEGEL/Q6/VERIFICATION_INTENT/V1\x00",
    "signature_receipt": b"HEGEL/Q6/SIGNATURE_RECEIPT/V1\x00",
    "checkpoint": b"HEGEL/Q6/CHECKPOINT/V1\x00",
    "pending": b"HEGEL/Q6/PENDING/V1\x00",
    "terminal": b"HEGEL/Q6/TERMINAL/V1\x00",
}
Q5B_DOMAINS: Final = {
    "capsule": b"HEGEL/Q5B/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q5B/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q5B/CONSUMPTION/V1\x00",
    "terminal": b"HEGEL/Q5B/TERMINAL/V1\x00",
}

Q6_SOURCE_PATHS: Final = (
    "src/hegel_machine/phase3_q6_formal_quotient_certificate_issuance_v1.py",
    "src/hegel_machine/phase3_q6_completion_replay_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase3_q6_purpose5_signing_worker_v1.py",
    "tools/phase3_q6_formal_quotient_certificate_issuance_v1.py",
)

EXECUTION_POLICY: Final = {
    "logical_campaign_limit": 1,
    "maximum_startup_smoke_popen_per_controller_entry": 3,
    "maximum_signing_worker_popen_count": 3,
    "maximum_worker_openssl_sign_popen_count": 3,
    "maximum_private_key_logical_load_count": 3,
    "maximum_postconsumption_preintent_custody_metadata_admission_count": 3,
    "maximum_signing_intent_publication_count": 3,
    "maximum_producer_free_signature_verification_count": 3,
    "maximum_signature_verification_intent_count": 3,
    "maximum_signature_output_count": 3,
    "maximum_signing_public_receipt_publication_count": 3,
    "maximum_verified_checkpoint_publication_count": 3,
    "per_signer_verified_checkpoint_required": True,
    "postcheckpoint_crypto_reverification_allowed": False,
    "required_valid_signature_count": 3,
    "maximum_certificate_envelope_publication_count": 1,
    "maximum_certificate_issuance_count": 1,
    "maximum_key_generation_count": 0,
    "maximum_key_import_count": 0,
    "maximum_key_export_count": 0,
    "maximum_key_copy_count": 0,
    "maximum_key_status_mutation_count": 0,
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
    "maximum_q4_ledger_read_count": 0,
    "maximum_q5a_ledger_read_count": 0,
    "maximum_archive_read_count": 0,
    "network_allowed": False,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_clean_prefix_pending_allowed": True,
    "partial_finalization_artifact_to_terminal_only": True,
    "partial_finalization_terminal_to_trajectory_suffix_only": True,
}

AUTHORITY_BOUNDARY: Final = {
    "q5b_state": _wire.Q5B_SUCCESS_STATUS.decode("ascii"),
    "q6_disposition": "THREE_OF_THREE_FORMAL_QUOTIENT_CERTIFICATE_ISSUANCE_ONLY",
    "purpose5_active_signer_count": 3,
    "purpose5_trust_genesis_pinned": True,
    "purpose5_current_key_status_accepted": True,
    "formal_certificate_signed_after_success": True,
    "certificate_envelope_published_after_success": True,
    "outside_certificate_issued_after_success": True,
    "legacy_closure_assessment_unlocked": False,
    "external_authority": False,
    "global_active_transition_allowed": False,
    "synthesis_allowed": False,
    "mdl_allowed": False,
    "external_scientific_claim_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "production_allowed": False,
    "current_pointer_update_allowed": False,
}

_ZERO_RESTRICTED_ACTIONS: Final = {
    "key_generation_count": 0,
    "key_import_count": 0,
    "key_export_count": 0,
    "key_copy_count": 0,
    "key_status_mutation_count": 0,
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
    "q4_ledger_read_count": 0,
    "q5a_ledger_read_count": 0,
    "archive_read_count": 0,
    "network_accessed": False,
    "synthesis_action_count": 0,
    "mdl_action_count": 0,
    "external_scientific_claim_made": False,
    "external_authority": False,
    "adoption_performed": False,
    "promotion_performed": False,
    "global_active_transition_performed": False,
    "production_accessed": False,
    "current_pointer_updated": False,
}


def q6_action_audit(
    completed_prefix: Sequence[int] = (), *, issued: bool = False
) -> dict[str, Any]:
    if type(completed_prefix) not in (tuple, list):
        raise ValueError("completed prefix must be an exact sequence")
    prefix = tuple(completed_prefix)
    if prefix not in PENDING_PREFIXES:
        raise ValueError("completed prefix differs")
    if type(issued) is not bool or (issued and prefix != COMPLETE_PREFIX):
        raise ValueError("Q6 issued audit requires the complete signer prefix")
    count = len(prefix)
    result: dict[str, Any] = {
        "completed_prefix": list(prefix),
        "signing_worker_popen_count": count,
        "worker_openssl_sign_popen_count": count,
        "private_key_logical_load_count": count,
        "postconsumption_preintent_custody_metadata_admission_count": count,
        "signing_intent_publication_count": count,
        "producer_free_signature_verification_count": count,
        "signature_verification_intent_count": count,
        "signature_output_count": count,
        "signing_public_receipt_publication_count": count,
        "verified_checkpoint_count": count,
        "verified_checkpoint_publication_count": count,
        "signature_count": count,
        "valid_signature_count": count,
        "signature_receipt_count": count,
        "certificate_envelope_publication_count": 1 if issued else 0,
        "certificate_issuance_count": 1 if issued else 0,
    }
    result.update(deepcopy(_ZERO_RESTRICTED_ACTIONS))
    return result


def validate_observed_action_audit(value: object) -> dict[str, Any]:
    """Validate a conservative post-consumption action accounting snapshot."""

    if type(value) is not dict:
        _fail("Q6 observed action audit differs")
    expected_keys = set(q6_action_audit(()))
    if set(value) != expected_keys:
        _fail("Q6 observed action audit field set differs")
    prefix = completed_prefix_tuple(value.get("completed_prefix"))
    counts = {
        name: value.get(name)
        for name in (
            "signing_worker_popen_count",
            "worker_openssl_sign_popen_count",
            "private_key_logical_load_count",
            "postconsumption_preintent_custody_metadata_admission_count",
            "producer_free_signature_verification_count",
            "signature_verification_intent_count",
            "signature_output_count",
            "signing_public_receipt_publication_count",
            "verified_checkpoint_count",
            "verified_checkpoint_publication_count",
            "signing_intent_publication_count",
            "signature_count",
            "valid_signature_count",
            "signature_receipt_count",
        )
    }
    if any(type(item) is not int or not (0 <= item <= 3) for item in counts.values()):
        _fail("Q6 observed action audit count differs")
    durable = len(prefix)
    if (
        counts["verified_checkpoint_count"] != durable
        or counts["verified_checkpoint_publication_count"] < durable
        or counts["verified_checkpoint_publication_count"]
        > counts["valid_signature_count"]
        or counts["signature_receipt_count"] != durable
        or counts["valid_signature_count"] < durable
        or counts["producer_free_signature_verification_count"]
        < counts["valid_signature_count"]
        or counts["signature_verification_intent_count"]
        < counts["producer_free_signature_verification_count"]
        or counts["signing_worker_popen_count"]
        > counts["signing_intent_publication_count"]
        or counts["signing_intent_publication_count"]
        > counts["postconsumption_preintent_custody_metadata_admission_count"]
        or counts["worker_openssl_sign_popen_count"]
        > counts["signing_worker_popen_count"]
        or counts["private_key_logical_load_count"]
        > counts["signing_worker_popen_count"]
        or counts["worker_openssl_sign_popen_count"]
        > counts["private_key_logical_load_count"]
        or counts["signature_output_count"]
        > counts["worker_openssl_sign_popen_count"]
        or counts["signing_public_receipt_publication_count"]
        > counts["signature_output_count"]
        or counts["signature_verification_intent_count"]
        > counts["signing_public_receipt_publication_count"]
        or counts["producer_free_signature_verification_count"]
        > counts["signature_verification_intent_count"]
        or counts["valid_signature_count"]
        > counts["producer_free_signature_verification_count"]
        or type(value.get("certificate_envelope_publication_count")) is not int
        or value.get("certificate_envelope_publication_count") not in (0, 1)
        or type(value.get("certificate_issuance_count")) is not int
        or value.get("certificate_issuance_count") not in (0, 1)
        or value.get("signature_count") != counts["signature_output_count"]
        or value.get("certificate_issuance_count")
        > value.get("certificate_envelope_publication_count")
    ):
        _fail("Q6 observed action audit ordering differs")
    for name, expected in _ZERO_RESTRICTED_ACTIONS.items():
        if value.get(name) != expected:
            _fail("Q6 prohibited action audit differs")
    return deepcopy(value)


def q6_observed_action_audit(
    completed_prefix: Sequence[int] = (),
    *,
    metadata_admission_count: int,
    signing_intent_count: int,
    worker_popen_count: int,
    openssl_sign_count: int,
    private_key_load_count: int,
    signature_output_count: int,
    worker_receipt_count: int,
    verification_intent_count: int,
    verification_count: int,
    valid_signature_count: int,
    verified_checkpoint_count: int,
    verified_checkpoint_publication_count: int,
    envelope_publication_count: int = 0,
    issuance_count: int = 0,
) -> dict[str, Any]:
    prefix = completed_prefix_tuple(completed_prefix)
    result = q6_action_audit(prefix)
    result.update({
        "postconsumption_preintent_custody_metadata_admission_count": metadata_admission_count,
        "signing_intent_publication_count": signing_intent_count,
        "signing_worker_popen_count": worker_popen_count,
        "worker_openssl_sign_popen_count": openssl_sign_count,
        "private_key_logical_load_count": private_key_load_count,
        "signature_output_count": signature_output_count,
        "signing_public_receipt_publication_count": worker_receipt_count,
        "signature_verification_intent_count": verification_intent_count,
        "producer_free_signature_verification_count": verification_count,
        "valid_signature_count": valid_signature_count,
        "verified_checkpoint_count": verified_checkpoint_count,
        "verified_checkpoint_publication_count": verified_checkpoint_publication_count,
        "signature_count": signature_output_count,
        "signature_receipt_count": verified_checkpoint_count,
        "certificate_envelope_publication_count": envelope_publication_count,
        "certificate_issuance_count": issuance_count,
    })
    return validate_observed_action_audit(result)


ZERO_ACTION_AUDIT: Final = q6_action_audit(())
SUCCESS_ACTION_AUDIT: Final = q6_action_audit(COMPLETE_PREFIX, issued=True)

MAX_JSON_BYTES: Final = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
MAX_TRAJECTORY_BYTES: Final = 4 * 1024 * 1024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_LOWER_HEX_16 = re.compile(r"[0-9a-f]{32}\Z")
_LOWER_HEX_32 = re.compile(r"[0-9a-f]{64}\Z")


class Q6CompletionReplayError(ValueError):
    pass


def _fail(detail: str) -> NoReturn:
    raise Q6CompletionReplayError(detail)


@dataclass(frozen=True, slots=True)
class LoadedJson:
    path: Path
    raw: bytes
    raw_digest: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PredecessorReplayV1:
    preflight_receipt: dict[str, Any]
    q5b_artifact: _wire.SealedQ5bTrustAmendmentArtifactV1
    q4_artifact: _wire.SealedQ4CapabilityArtifactV1


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def _finite(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        _fail("non-finite JSON number is forbidden")
    if type(value) is list:
        for item in value:
            _finite(item)
    elif type(value) is dict:
        for item in value.values():
            _finite(item)


def canonical_json_bytes(value: object) -> bytes:
    _finite(value)
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_JSON_BYTES:
        _fail(f"{label} byte bound differs")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, Q6CompletionReplayError) as error:
        _fail(f"{label} JSON rejected: {error}")
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        _fail(f"{label} canonical JSON differs")
    return value


def pretty_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_JSON_BYTES:
        _fail(f"{label} byte bound differs")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, Q6CompletionReplayError) as error:
        _fail(f"{label} JSON rejected: {error}")
    if type(value) is not dict:
        _fail(f"{label} top-level object differs")
    _finite(value)
    return value


def raw_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("raw SHA-256 input must be bytes")
    return "sha256:" + sha256(payload).hexdigest()


def validate_pending_history_receipt(value: object) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != {"raw_sha256", "byte_count", "event_count"}
        or _SHA256.fullmatch(str(value.get("raw_sha256"))) is None
        or type(value.get("byte_count")) is not int
        or value["byte_count"] < 0
        or type(value.get("event_count")) is not int
        or value["event_count"] < 0
        or (value["event_count"] == 0 and (
            value["byte_count"] != 0 or value["raw_sha256"] != raw_sha256(b"")
        ))
    ):
        _fail("Q6 pending history receipt differs")
    return deepcopy(value)


def _digest(domain: bytes, value: object) -> str:
    return raw_sha256(domain + canonical_json_bytes(value))


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be tagged SHA-256")
    return value


def _source_commit(value: object, label: str) -> str:
    if type(value) is not str or _SOURCE_COMMIT.fullmatch(value) is None:
        _fail(f"{label} must be 40 lower hex")
    return value


def load_pretty_json(path: Path, label: str) -> LoadedJson:
    raw = path.read_bytes()
    return LoadedJson(path, raw, raw_sha256(raw), pretty_json_bytes(raw, label))


def load_preregistration(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q6 preregistration")
    if (
        loaded.value.get("schema_version") != PREREGISTRATION_SCHEMA
        or loaded.value.get("preregistration_id") != PREREGISTRATION_ID
        or loaded.value.get("status") != "Q6_SPEC_FROZEN_NOT_AUTHORIZED"
        or loaded.value.get("execution_policy") != EXECUTION_POLICY
        or loaded.value.get("authority_boundary") != AUTHORITY_BOUNDARY
    ):
        _fail("Q6 preregistration boundary differs")
    return loaded


def load_contract(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q6 contract")
    value = loaded.value
    if (
        value.get("schema_version") != CONTRACT_SCHEMA
        or value.get("contract_id") != CONTRACT_ID
        or value.get("action_id") != ACTION_ID
        or value.get("status") != "IMPLEMENTED_NOT_EXECUTED"
        or value.get("execution_policy") != EXECUTION_POLICY
        or value.get("authority_boundary") != AUTHORITY_BOUNDARY
        or type(value.get("fixed_paths")) is not dict
        or type(value.get("q5b_inputs")) is not dict
        or type(value.get("q4_input")) is not dict
        or type(value.get("worker")) is not dict
    ):
        _fail("Q6 contract boundary differs")
    return loaded


def completed_prefix_tuple(value: object) -> tuple[int, ...]:
    if type(value) not in (tuple, list):
        raise ValueError("completed prefix must be an exact sequence")
    result = tuple(value)
    if result not in PENDING_PREFIXES:
        raise ValueError("completed prefix differs")
    return result


def _semantic_digest(value: Mapping[str, Any], kind: str, field: str) -> str:
    unsigned = deepcopy(dict(value))
    supplied = unsigned.pop(field, None)
    expected = _digest(Q5B_DOMAINS[kind], unsigned)
    if supplied != expected:
        _fail(f"Q5b {kind} semantic digest differs")
    return expected


def _expected_map(parent: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if type(value) is not dict:
        _fail(f"{label} registry differs")
    return value


def _replay_q5b_trajectory(
    payload: bytes,
    *,
    capsule_digest: str,
    authorization_digest: str,
    consumption_digest: str,
    terminal_digest: str,
    q5b_artifact: _wire.SealedQ5bTrustAmendmentArtifactV1,
) -> list[dict[str, Any]]:
    if type(payload) is not bytes or len(payload) > MAX_TRAJECTORY_BYTES or not payload.endswith(b"\n"):
        _fail("Q5b trajectory framing differs")
    lines = payload.splitlines(keepends=True)
    if len(lines) != 3:
        _fail("Q5b trajectory event count differs")
    rows = [strict_json_bytes(line, f"Q5b trajectory event {index}") for index, line in enumerate(lines, 1)]
    events = (
        "CAPSULE_PREPARED_NOT_AUTHORIZED",
        "AUTHORIZATION_RECORDED_NOT_CONSUMED",
        _wire.Q5B_SUCCESS_STATUS.decode("ascii"),
    )
    for index, row in enumerate(rows):
        if row.get("event") != events[index] or row.get("capsule_digest") != capsule_digest:
            _fail("Q5b trajectory event identity differs")
        if index == 0 and (row.get("authorization_digest") is not None or row.get("authorization_consumption_count") != 0):
            _fail("Q5b capsule trajectory authority differs")
        if index == 1 and (row.get("authorization_digest") != authorization_digest or row.get("authorization_consumption_count") != 0):
            _fail("Q5b authorization trajectory differs")
        if index == 2:
            if (
                row.get("authorization_digest") != authorization_digest
                or row.get("consumption_record_digest") != consumption_digest
                or row.get("terminal_record_digest") != terminal_digest
                or row.get("authorization_consumption_count") != 1
                or row.get("logical_campaign_count") != 1
                or row.get("trust_amendment_receipt_root")
                != "sha256:" + q5b_artifact.receipt_root.hex()
                or row.get("pinned_trust_genesis_root")
                != "sha256:" + q5b_artifact.pinned_genesis.genesis_root.hex()
                or row.get("current_key_status_root")
                != "sha256:" + q5b_artifact.current_status.status_root.hex()
            ):
                _fail("Q5b success trajectory differs")
    return rows


def replay_predecessors_v1(
    preregistration: Mapping[str, Any],
    *,
    q5b_payloads: Mapping[str, bytes],
    q4_artifact_payload: bytes,
) -> PredecessorReplayV1:
    predecessor = _expected_map(preregistration, "predecessor_q5b", "Q5b predecessor")
    q4_expected = _expected_map(
        preregistration, "selected_q4_capability_artifact", "Q4 artifact"
    )
    required = ("capsule", "authorization", "consumption", "terminal", "artifact", "trajectory")
    if tuple(q5b_payloads) != required:
        _fail("selected Q5b input set/order differs")
    selected: list[dict[str, Any]] = []
    for kind in required:
        payload = q5b_payloads[kind]
        expected = _expected_map(predecessor, kind, f"Q5b {kind}")
        actual = raw_sha256(payload)
        if actual != expected.get("raw_sha256") or len(payload) != expected.get("byte_count"):
            _fail(f"Q5b {kind} raw identity differs")
        selected.append({"kind": kind, "byte_count": len(payload), "raw_sha256": actual})

    capsule = strict_json_bytes(q5b_payloads["capsule"], "Q5b capsule")
    authorization = strict_json_bytes(q5b_payloads["authorization"], "Q5b authorization")
    consumption = strict_json_bytes(q5b_payloads["consumption"], "Q5b consumption")
    terminal = strict_json_bytes(q5b_payloads["terminal"], "Q5b terminal")
    capsule_digest = _semantic_digest(capsule, "capsule", "capsule_digest")
    authorization_digest = _semantic_digest(authorization, "authorization", "authorization_digest")
    consumption_digest = _semantic_digest(consumption, "consumption", "record_digest")
    terminal_digest = _semantic_digest(terminal, "terminal", "record_digest")
    for kind, field, actual in (
        ("capsule", "capsule_digest", capsule_digest),
        ("authorization", "authorization_digest", authorization_digest),
        ("consumption", "record_digest", consumption_digest),
        ("terminal", "record_digest", terminal_digest),
    ):
        if _expected_map(predecessor, kind, kind).get(field) != actual:
            _fail(f"configured Q5b {kind} semantic digest differs")

    try:
        q5b_artifact = _wire.decode_sealed_q5b_trust_amendment_artifact_v1(
            q5b_payloads["artifact"]
        )
        q4_artifact = _wire.decode_sealed_q4_capability_artifact_v1(q4_artifact_payload)
    except (TypeError, ValueError) as error:
        _fail(f"sealed Q4/Q5b artifact replay rejected: {error}")

    q5b_artifact_expected = _expected_map(predecessor, "artifact", "Q5b artifact")
    key_ids = [item.key_id.hex() for item in q5b_artifact.signer_manifests]
    public_keys = [item.public_key.hex() for item in q5b_artifact.signer_manifests]
    if (
        capsule.get("status") != "Q5B_PURPOSE5_TRUST_AMENDMENT_CAPSULE_READY_NOT_AUTHORIZED"
        or authorization.get("status")
        != "Q5B_PURPOSE5_TRUST_AMENDMENT_AUTHORIZATION_RECORDED_NOT_CONSUMED"
        or consumption.get("status") != "Q5B_PURPOSE5_TRUST_AMENDMENT_AUTHORIZATION_CONSUMED"
        or terminal.get("status") != _wire.Q5B_SUCCESS_STATUS.decode("ascii")
        or terminal.get("authorization_consumption_count") != 1
        or terminal.get("logical_campaign_count") != 1
        or capsule.get("source_commit") != predecessor.get("source_commit")
        or capsule.get("source_closure_root") != predecessor.get("source_closure_root")
        or raw_sha256(q5b_payloads["artifact"]) != q5b_artifact_expected.get("raw_sha256")
        or "sha256:" + q5b_artifact.receipt_root.hex()
        != q5b_artifact_expected.get("trust_amendment_receipt_root")
        or "sha256:" + q5b_artifact.pinned_genesis.genesis_root.hex()
        != q5b_artifact_expected.get("pinned_trust_genesis_root")
        or "sha256:" + q5b_artifact.current_status.status_root.hex()
        != q5b_artifact_expected.get("current_key_status_root")
        or key_ids != q5b_artifact_expected.get("key_ids")
        or public_keys != q5b_artifact_expected.get("public_keys")
    ):
        _fail("Q5b predecessor completion boundary differs")
    outcome = terminal.get("outcome")
    if (
        type(outcome) is not dict
        or outcome.get("status") != _wire.Q5B_SUCCESS_STATUS.decode("ascii")
        or outcome.get("artifact_raw_sha256") != raw_sha256(q5b_payloads["artifact"])
        or outcome.get("purpose5_active_signer_count") != 3
        or outcome.get("purpose5_trust_genesis_pinned") is not True
        or outcome.get("purpose5_current_key_status_accepted") is not True
        or outcome.get("formal_certificate_signed") is not False
        or outcome.get("certificate_envelope_published") is not False
        or outcome.get("outside_certificate_issued") is not False
        or outcome.get("campaign_complete") is not True
        or outcome.get("terminal_no_resume") is not False
    ):
        _fail("Q5b terminal outcome boundary differs")

    q4_raw = raw_sha256(q4_artifact_payload)
    if (
        q4_raw != q4_expected.get("raw_sha256")
        or len(q4_artifact_payload) != q4_expected.get("byte_count")
        or "sha256:" + q4_artifact.receipt_root.hex()
        != q4_expected.get("capability_receipt_root")
        or "sha256:" + q4_artifact.body.body_hash.hex()
        != q4_expected.get("quotient_body_hash")
        or q5b_artifact.q5a_candidate.q4_capability_receipt_root
        != q4_artifact.receipt_root
        or q5b_artifact.q5a_candidate.q4_body_hash != q4_artifact.body.body_hash
    ):
        _fail("Q4 artifact/Q5b lineage differs")

    rows = _replay_q5b_trajectory(
        q5b_payloads["trajectory"],
        capsule_digest=capsule_digest,
        authorization_digest=authorization_digest,
        consumption_digest=consumption_digest,
        terminal_digest=terminal_digest,
        q5b_artifact=q5b_artifact,
    )
    preflight = {
        "schema_version": "hegel-phase3a-q6-predecessor-preflight/1",
        "selected_q5b_inputs": selected,
        "selected_q5b_input_count": 6,
        "selected_q5b_input_total_bytes": sum(len(q5b_payloads[item]) for item in required),
        "q5b_capsule_digest": capsule_digest,
        "q5b_authorization_digest": authorization_digest,
        "q5b_consumption_record_digest": consumption_digest,
        "q5b_terminal_record_digest": terminal_digest,
        "q5b_artifact_raw_sha256": raw_sha256(q5b_payloads["artifact"]),
        "q5b_receipt_root": "sha256:" + q5b_artifact.receipt_root.hex(),
        "pinned_trust_genesis_root": "sha256:" + q5b_artifact.pinned_genesis.genesis_root.hex(),
        "current_key_status_root": "sha256:" + q5b_artifact.current_status.status_root.hex(),
        "current_key_epoch": _wire.INITIAL_KEY_EPOCH,
        "purpose5_active_signer_count": 3,
        "role_ordered_key_ids": key_ids,
        "role_ordered_public_keys": public_keys,
        "q4_artifact_raw_sha256": q4_raw,
        "q4_capability_receipt_root": "sha256:" + q4_artifact.receipt_root.hex(),
        "q4_quotient_body_hash": "sha256:" + q4_artifact.body.body_hash.hex(),
        "q5b_trajectory_audit": {
            "raw_sha256": raw_sha256(q5b_payloads["trajectory"]),
            "byte_count": len(q5b_payloads["trajectory"]),
            "event_count": len(rows),
            "events": [row["event"] for row in rows],
        },
        "producer_free_q5b_replay_completed": True,
        "producer_free_q4_replay_completed": True,
        "custody_root_access_count": 0,
        "private_key_access_count": 0,
        "q6_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }
    return PredecessorReplayV1(preflight, q5b_artifact, q4_artifact)


def validate_source_closure_rows(rows: object) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(Q6_SOURCE_PATHS):
        _fail("Q6 source closure row count differs")
    result: list[dict[str, Any]] = []
    for expected_path, row in zip(Q6_SOURCE_PATHS, rows):
        if (
            type(row) is not dict
            or row.get("relative_path") != expected_path
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 1
            or _SHA256.fullmatch(str(row.get("raw_sha256"))) is None
            or set(row) != {"relative_path", "byte_count", "raw_sha256"}
        ):
            _fail("Q6 source closure row differs")
        result.append(deepcopy(row))
    return result


def source_closure_root(rows: object) -> str:
    return _digest(DOMAINS["source_closure"], validate_source_closure_rows(rows))


def validate_worker_smoke_receipt(
    value: object,
    *,
    ordinal: int,
    public_key_hex: str,
    key_id_hex: str,
    message_raw_sha256: str,
    message_byte_count: int,
    expected_uid: int,
    expected_gid: int,
) -> dict[str, Any]:
    expected = {
        "expected_gid": expected_gid,
        "expected_key_id_hex": key_id_hex,
        "expected_message_byte_count": message_byte_count,
        "expected_message_raw_sha256": message_raw_sha256,
        "expected_public_key_hex": public_key_hex,
        "expected_uid": expected_uid,
        "key_generation_count": 0,
        "key_import_count": 0,
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 0,
        "private_key_access_count": 0,
        "purpose_id": 5,
        "schema": WORKER_SMOKE_SCHEMA,
        "signature_count": 0,
        "signer_ordinal": ordinal,
        "signer_role": SIGNER_ROLE_NAMES[ordinal - 1],
        "status": WORKER_SMOKE_STATUS,
    }
    if value != expected:
        _fail(f"signer {ordinal} startup smoke receipt differs")
    return deepcopy(expected)


def build_preflight_receipt(
    predecessor: PredecessorReplayV1,
    smoke_receipts: Sequence[Mapping[str, Any]],
    preimage: _wire.Q6SignaturePreimageV1,
    *,
    expected_uid: int = 1000,
    expected_gid: int = 1000,
) -> dict[str, Any]:
    if type(predecessor) is not PredecessorReplayV1 or len(smoke_receipts) != 3:
        _fail("Q6 preflight input shape differs")
    message = preimage.signature_message
    canonical = preimage.canonical_bytes
    key_ids = predecessor.preflight_receipt["role_ordered_key_ids"]
    public_keys = predecessor.preflight_receipt["role_ordered_public_keys"]
    validated = [
        validate_worker_smoke_receipt(
            item,
            ordinal=ordinal,
            public_key_hex=public_keys[ordinal - 1],
            key_id_hex=key_ids[ordinal - 1],
            message_raw_sha256=raw_sha256(message),
            message_byte_count=len(message),
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        for ordinal, item in enumerate(smoke_receipts, 1)
    ]
    return {
        "schema_version": "hegel-phase3a-q6-formal-issuance-preflight/1",
        "predecessor_receipt": deepcopy(predecessor.preflight_receipt),
        "certificate_created_at_unix_seconds": preimage.created_at_unix_seconds,
        "signature_preimage_raw_sha256": raw_sha256(canonical),
        "signature_preimage_byte_count": len(canonical),
        "signature_preimage_root": "sha256:" + preimage.preimage_root.hex(),
        "signing_message_raw_sha256": raw_sha256(message),
        "signing_message_byte_count": len(message),
        "startup_smoke_receipts": validated,
        "startup_smoke_popen_count": 3,
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "custody_root_access_count": 0,
        "private_key_access_count": 0,
        "signature_count": 0,
        "q6_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }


def validate_preflight_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q6 preflight receipt differs")
    required = {
        "schema_version",
        "predecessor_receipt",
        "certificate_created_at_unix_seconds",
        "signature_preimage_raw_sha256",
        "signature_preimage_byte_count",
        "signature_preimage_root",
        "signing_message_raw_sha256",
        "signing_message_byte_count",
        "startup_smoke_receipts",
        "startup_smoke_popen_count",
        "authorization_consumption_count",
        "logical_campaign_count",
        "custody_root_access_count",
        "private_key_access_count",
        "signature_count",
        "q6_action_audit",
    }
    if (
        set(value) != required
        or value.get("schema_version") != "hegel-phase3a-q6-formal-issuance-preflight/1"
        or type(value.get("certificate_created_at_unix_seconds")) is not int
        or value["certificate_created_at_unix_seconds"] < 0
        or any(value.get(name) != 0 for name in (
            "authorization_consumption_count", "logical_campaign_count",
            "custody_root_access_count", "private_key_access_count", "signature_count"
        ))
        or value.get("startup_smoke_popen_count") != 3
        or type(value.get("startup_smoke_receipts")) is not list
        or len(value["startup_smoke_receipts"]) != 3
        or value.get("q6_action_audit") != ZERO_ACTION_AUDIT
    ):
        _fail("Q6 preflight receipt boundary differs")
    for field in ("signature_preimage_raw_sha256", "signature_preimage_root", "signing_message_raw_sha256"):
        _tagged(value.get(field), field)
    predecessor = value["predecessor_receipt"]
    if type(predecessor) is not dict:
        _fail("Q6 preflight predecessor receipt differs")
    key_ids = predecessor.get("role_ordered_key_ids")
    public_keys = predecessor.get("role_ordered_public_keys")
    if type(key_ids) is not list or type(public_keys) is not list or len(key_ids) != 3 or len(public_keys) != 3:
        _fail("Q6 preflight signer bindings differ")
    for ordinal, receipt in enumerate(value["startup_smoke_receipts"], 1):
        validate_worker_smoke_receipt(
            receipt,
            ordinal=ordinal,
            public_key_hex=public_keys[ordinal - 1],
            key_id_hex=key_ids[ordinal - 1],
            message_raw_sha256=value["signing_message_raw_sha256"],
            message_byte_count=value["signing_message_byte_count"],
            expected_uid=1000,
            expected_gid=1000,
        )
    return deepcopy(value)


def build_capsule(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    *,
    source_commit: str,
    source_closure_rows: list[dict[str, Any]],
    run_id: str,
    created_at_unix_seconds: int,
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source_commit = _source_commit(source_commit, "Q6 source commit")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        _fail("Q6 run id differs")
    if type(created_at_unix_seconds) is not int or created_at_unix_seconds < 0:
        _fail("Q6 capsule creation time differs")
    rows = validate_source_closure_rows(source_closure_rows)
    preflight = validate_preflight_receipt(preflight_receipt)
    value: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "action_id": ACTION_ID,
        "contract_raw_sha256": contract.raw_digest,
        "preregistration_raw_sha256": contract.value.get("preregistration_raw_sha256"),
        "source_commit": source_commit,
        "source_closure_rows": rows,
        "source_closure_root": source_closure_root(rows),
        "run_id": run_id,
        "created_at_unix_seconds": created_at_unix_seconds,
        "certificate_created_at_unix_seconds": preflight["certificate_created_at_unix_seconds"],
        "preflight_receipt": preflight,
        "execution_policy": deepcopy(EXECUTION_POLICY),
        "authority_boundary": deepcopy(AUTHORITY_BOUNDARY),
    }
    value["capsule_digest"] = _digest(DOMAINS["capsule"], value)
    return value


def validate_capsule(
    value: object, contract: LoadedJson, preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q6 capsule differs")
    supplied = value.get("capsule_digest")
    unsigned = deepcopy(value)
    unsigned.pop("capsule_digest", None)
    if (
        supplied != _digest(DOMAINS["capsule"], unsigned)
        or value.get("schema_version") != CAPSULE_SCHEMA
        or value.get("status") != CAPSULE_READY
        or value.get("action_id") != ACTION_ID
        or value.get("contract_raw_sha256") != contract.raw_digest
        or value.get("preregistration_raw_sha256") != contract.value.get("preregistration_raw_sha256")
        or value.get("execution_policy") != EXECUTION_POLICY
        or value.get("authority_boundary") != AUTHORITY_BOUNDARY
        or value.get("source_closure_root") != source_closure_root(value.get("source_closure_rows"))
    ):
        _fail("Q6 capsule binding differs")
    preflight = validate_preflight_receipt(value.get("preflight_receipt"))
    if value.get("certificate_created_at_unix_seconds") != preflight["certificate_created_at_unix_seconds"]:
        _fail("Q6 capsule certificate time differs")
    return deepcopy(value)


def build_authorization(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    del preregistration
    value: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_RECORDED,
        "capsule_digest": capsule["capsule_digest"],
        "contract_raw_sha256": contract.raw_digest,
        "authorized_scope": AUTHORIZED_SCOPE,
        "logical_campaign_limit": 1,
        "signing_worker_popen_limit": 3,
        "signing_intent_publication_limit": 3,
        "worker_openssl_sign_popen_limit": 3,
        "private_key_logical_load_limit": 3,
        "postconsumption_preintent_custody_metadata_admission_limit": 3,
        "signature_output_limit": 3,
        "signing_public_receipt_publication_limit": 3,
        "signature_verification_intent_limit": 3,
        "producer_free_signature_verification_limit": 3,
        "verified_checkpoint_publication_limit": 3,
        "signature_count_limit": 3,
        "certificate_envelope_publication_limit": 1,
        "certificate_issuance_limit": 1,
        "network_allowed": False,
        "global_active_transition_allowed": False,
        "synthesis_allowed": False,
        "q6_prohibited_action_audit_limit": deepcopy(_ZERO_RESTRICTED_ACTIONS),
    }
    value["authorization_digest"] = _digest(DOMAINS["authorization"], value)
    return value


def validate_authorization(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization(contract, preregistration, capsule)
    if value != expected:
        _fail("Q6 authorization differs")
    return deepcopy(expected)


def build_consumption(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    del preregistration
    value: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": AUTHORIZATION_CONSUMED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "contract_raw_sha256": contract.raw_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "completed_prefix": [],
        "q6_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
        "global_active_transition_performed": False,
    }
    value["record_digest"] = _digest(DOMAINS["consumption"], value)
    return value


def validate_consumption(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption(contract, preregistration, capsule, authorization)
    if value != expected:
        _fail("Q6 consumption differs")
    return deepcopy(expected)


def _receipt_digest_row(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "signer_ordinal": receipt["signer_ordinal"],
        "signature_receipt_digest": receipt["signature_receipt_digest"],
        "signature_raw_sha256": receipt["signature_raw_sha256"],
    }


def build_signing_intent(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    *,
    created_at_unix_seconds: int,
) -> dict[str, Any]:
    del preregistration
    prefix = tuple(item["signer_ordinal"] for item in completed_receipts)
    if prefix not in NEXT_SIGNER_PREFIXES:
        _fail("Q6 intent completed prefix differs")
    ordinal = len(prefix) + 1
    predecessor = capsule["preflight_receipt"]["predecessor_receipt"]
    key_ids = predecessor["role_ordered_key_ids"]
    public_keys = predecessor["role_ordered_public_keys"]
    value: dict[str, Any] = {
        "schema_version": SIGNING_INTENT_SCHEMA,
        "status": SIGNING_INTENT_RECORDED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "contract_raw_sha256": contract.raw_digest,
        "signer_ordinal": ordinal,
        "signer_role": SIGNER_ROLE_NAMES[ordinal - 1],
        "key_epoch": 1,
        "key_id_hex": key_ids[ordinal - 1],
        "public_key_hex": public_keys[ordinal - 1],
        "signature_preimage_root": capsule["preflight_receipt"]["signature_preimage_root"],
        "signing_message_raw_sha256": capsule["preflight_receipt"]["signing_message_raw_sha256"],
        "signing_message_byte_count": capsule["preflight_receipt"]["signing_message_byte_count"],
        "completed_prefix_before": list(prefix),
        "completed_receipts_before": [_receipt_digest_row(item) for item in completed_receipts],
        "created_at_unix_seconds": created_at_unix_seconds,
        "worker_popen_limit": 1,
        "signature_count_limit": 1,
    }
    value["intent_digest"] = _digest(DOMAINS["intent"], value)
    return value


def validate_signing_intent(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("created_at_unix_seconds")) is not int:
        _fail("Q6 signing intent differs")
    expected = build_signing_intent(
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        completed_receipts,
        created_at_unix_seconds=value["created_at_unix_seconds"],
    )
    if value != expected:
        _fail("Q6 signing intent binding differs")
    return deepcopy(expected)


def expected_worker_receipt(
    *,
    intent: Mapping[str, Any],
    signature_payload: bytes,
) -> dict[str, Any]:
    if type(signature_payload) is not bytes or len(signature_payload) != 64:
        _fail("Q6 signature shape differs")
    return {
        "key_epoch": 1,
        "key_generation_count": 0,
        "key_id_hex": intent["key_id_hex"],
        "key_import_count": 0,
        "message_byte_count": intent["signing_message_byte_count"],
        "message_raw_sha256": intent["signing_message_raw_sha256"],
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 1,
        "private_key_copy_count": 0,
        "private_key_exported": False,
        "private_key_openssl_use_count": 1,
        "public_key_hex": intent["public_key_hex"],
        "purpose_id": 5,
        "schema": WORKER_RECEIPT_SCHEMA,
        "signature_byte_count": 64,
        "signature_count": 1,
        "signature_raw_sha256": raw_sha256(signature_payload),
        "signature_verification_count": 0,
        "signer_ordinal": intent["signer_ordinal"],
        "signer_role": intent["signer_role"],
        "status": WORKER_SUCCESS_STATUS,
    }


def build_verification_intent(
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    signing_intent: Mapping[str, Any],
    *,
    worker_receipt: Mapping[str, Any],
    signature_payload: bytes,
    created_at_unix_seconds: int,
) -> dict[str, Any]:
    prefix = tuple(item["signer_ordinal"] for item in completed_receipts)
    if prefix not in NEXT_SIGNER_PREFIXES:
        _fail("Q6 verification intent prefix differs")
    ordinal = len(prefix) + 1
    if (
        signing_intent.get("signer_ordinal") != ordinal
        or dict(worker_receipt)
        != expected_worker_receipt(intent=signing_intent, signature_payload=signature_payload)
        or type(created_at_unix_seconds) is not int
        or created_at_unix_seconds < 0
    ):
        _fail("Q6 verification intent input differs")
    value: dict[str, Any] = {
        "schema_version": VERIFICATION_INTENT_SCHEMA,
        "status": VERIFICATION_INTENT_RECORDED,
        "contract_raw_sha256": contract.raw_digest,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "signing_intent_digest": signing_intent["intent_digest"],
        "signer_ordinal": ordinal,
        "signer_role": signing_intent["signer_role"],
        "key_id_hex": signing_intent["key_id_hex"],
        "public_key_hex": signing_intent["public_key_hex"],
        "signature_preimage_root": signing_intent["signature_preimage_root"],
        "signing_message_raw_sha256": signing_intent["signing_message_raw_sha256"],
        "worker_receipt_raw_sha256": raw_sha256(canonical_json_bytes(dict(worker_receipt))),
        "signature_raw_sha256": raw_sha256(signature_payload),
        "signature_byte_count": 64,
        "completed_prefix_before": list(prefix),
        "producer_free_verification_limit": 1,
        "created_at_unix_seconds": created_at_unix_seconds,
    }
    value["verification_intent_digest"] = _digest(
        DOMAINS["verification_intent"], value
    )
    return value


def validate_verification_intent(
    value: object,
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    signing_intent: Mapping[str, Any],
    *,
    worker_receipt: Mapping[str, Any],
    signature_payload: bytes,
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("created_at_unix_seconds")) is not int:
        _fail("Q6 verification intent differs")
    expected = build_verification_intent(
        contract,
        capsule,
        authorization,
        consumption,
        completed_receipts,
        signing_intent,
        worker_receipt=worker_receipt,
        signature_payload=signature_payload,
        created_at_unix_seconds=value["created_at_unix_seconds"],
    )
    if value != expected:
        _fail("Q6 verification intent binding differs")
    return expected


def build_signature_receipt(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    intent: Mapping[str, Any],
    *,
    worker_receipt: Mapping[str, Any],
    signature_payload: bytes,
    verification_intent: Mapping[str, Any],
) -> dict[str, Any]:
    intent_value = validate_signing_intent(
        intent, contract, preregistration, capsule, authorization, consumption, completed_receipts
    )
    expected_worker = expected_worker_receipt(intent=intent_value, signature_payload=signature_payload)
    if dict(worker_receipt) != expected_worker:
        _fail("Q6 worker receipt differs")
    verification = validate_verification_intent(
        verification_intent,
        contract,
        capsule,
        authorization,
        consumption,
        completed_receipts,
        intent_value,
        worker_receipt=worker_receipt,
        signature_payload=signature_payload,
    )
    ordinal = intent_value["signer_ordinal"]
    prefix = tuple(item["signer_ordinal"] for item in completed_receipts) + (ordinal,)
    value: dict[str, Any] = {
        "schema_version": SIGNATURE_RECEIPT_SCHEMA,
        "status": SIGNATURE_VERIFIED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "intent_digest": intent_value["intent_digest"],
        "verification_intent_digest": verification["verification_intent_digest"],
        "signer_ordinal": ordinal,
        "signer_role": intent_value["signer_role"],
        "key_epoch": 1,
        "key_id_hex": intent_value["key_id_hex"],
        "public_key_hex": intent_value["public_key_hex"],
        "signature_preimage_root": intent_value["signature_preimage_root"],
        "signing_message_raw_sha256": intent_value["signing_message_raw_sha256"],
        "worker_receipt_raw_sha256": raw_sha256(canonical_json_bytes(expected_worker)),
        "signature_raw_sha256": raw_sha256(signature_payload),
        "signature_byte_count": 64,
        "producer_free_signature_verified": True,
        "completed_prefix": list(prefix),
        "q6_action_audit": q6_action_audit(prefix),
    }
    value["signature_receipt_digest"] = _digest(DOMAINS["signature_receipt"], value)
    return value


def validate_signature_receipt_sequence(
    receipts: Sequence[Mapping[str, Any]],
    signatures: Sequence[bytes],
    *,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    intents: Sequence[Mapping[str, Any]],
    worker_receipts: Sequence[Mapping[str, Any]],
    verification_intents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not (
        len(receipts)
        == len(signatures)
        == len(intents)
        == len(worker_receipts)
        == len(verification_intents)
        <= 3
    ):
        _fail("Q6 completed signer sequence lengths differ")
    validated: list[dict[str, Any]] = []
    for index, (receipt, signature, intent, worker, verification_intent) in enumerate(
        zip(receipts, signatures, intents, worker_receipts, verification_intents), 1
    ):
        expected = build_signature_receipt(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            validated,
            intent,
            worker_receipt=worker,
            signature_payload=signature,
            verification_intent=verification_intent,
        )
        if receipt != expected or expected["signer_ordinal"] != index:
            _fail("Q6 signature receipt sequence differs")
        validated.append(expected)
    return validated


def build_checkpoint(
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prefix = tuple(item["signer_ordinal"] for item in receipts)
    completed_prefix_tuple(prefix)
    value: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": CLEAN_CHECKPOINT,
        "contract_raw_sha256": contract.raw_digest,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "completed_prefix": list(prefix),
        "completed_receipts": [_receipt_digest_row(item) for item in receipts],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "q6_action_audit": q6_action_audit(prefix),
    }
    value["checkpoint_digest"] = _digest(DOMAINS["checkpoint"], value)
    return value


def build_pending(
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    observed_action_audit: Mapping[str, Any] | None = None,
    inflight_public_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prefix = tuple(item["signer_ordinal"] for item in receipts)
    if consumption is None:
        status = PENDING_NOT_CONSUMED
        count = 0
        prefix = ()
        audit = ZERO_ACTION_AUDIT
        if inflight_public_result is not None:
            _fail("Q6 pre-consumption PENDING cannot bind an inflight result")
    else:
        if prefix not in PENDING_PREFIXES:
            _fail("Q6 PENDING prefix is not clean")
        status = PENDING_CLEAN_CHECKPOINT
        count = 1
        audit = (
            q6_action_audit(prefix)
            if observed_action_audit is None
            else validate_observed_action_audit(observed_action_audit)
        )
        if tuple(audit["completed_prefix"]) != prefix:
            _fail("Q6 PENDING observed audit/prefix differs")
    inflight = None
    if inflight_public_result is not None:
        inflight = deepcopy(dict(inflight_public_result))
        if (
            set(inflight) != {
                "status", "signer_ordinal", "intent_digest",
                "worker_receipt_raw_sha256", "signature_raw_sha256",
                "verification_intent_digest",
            }
            or inflight.get("status")
            != "KNOWN_SEALED_WORKER_RESULT_AWAITING_VERIFICATION"
            or inflight.get("signer_ordinal") != len(prefix) + 1
            or inflight.get("verification_intent_digest") is not None
        ):
            _fail("Q6 PENDING inflight public result differs")
        for field in ("intent_digest", "worker_receipt_raw_sha256", "signature_raw_sha256"):
            _tagged(inflight.get(field), field)
    value: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA,
        "status": status,
        "contract_raw_sha256": contract.raw_digest,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": None if authorization is None else authorization["authorization_digest"],
        "consumption_record_digest": None if consumption is None else consumption["record_digest"],
        "completed_prefix": list(prefix),
        "authorization_consumption_count": count,
        "logical_campaign_count": count,
        "failure_phase": failure_phase,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "inflight_public_result": inflight,
        "q6_action_audit": deepcopy(audit),
    }
    value["pending_digest"] = _digest(DOMAINS["pending"], value)
    return value


def build_issuance_artifact(
    capsule: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    signatures: Sequence[bytes],
    *,
    q4_artifact_payload: bytes,
    q5b_artifact_payload: bytes,
    complete_checkpoint_digest: str,
) -> _wire.Q6FormalIssuanceReceiptV1:
    if tuple(item["signer_ordinal"] for item in receipts) != COMPLETE_PREFIX or len(signatures) != 3:
        _fail("Q6 issuance requires the complete signer prefix")
    predecessor = capsule["preflight_receipt"]["predecessor_receipt"]
    for ordinal, (receipt, signature) in enumerate(zip(receipts, signatures), 1):
        if (
            receipt.get("status") != SIGNATURE_VERIFIED
            or receipt.get("signer_ordinal") != ordinal
            or receipt.get("key_id_hex")
            != predecessor["role_ordered_key_ids"][ordinal - 1]
            or receipt.get("public_key_hex")
            != predecessor["role_ordered_public_keys"][ordinal - 1]
            or receipt.get("signature_preimage_root")
            != capsule["preflight_receipt"]["signature_preimage_root"]
            or receipt.get("signing_message_raw_sha256")
            != capsule["preflight_receipt"]["signing_message_raw_sha256"]
            or receipt.get("signature_raw_sha256") != raw_sha256(signature)
            or receipt.get("producer_free_signature_verified") is not True
        ):
            _fail("Q6 issuance signature/verified-checkpoint binding differs")
    _tagged(complete_checkpoint_digest, "Q6 complete checkpoint digest")
    try:
        result = _wire.build_q6_formal_issuance_receipt_v1(
            q4_artifact_payload,
            q5b_artifact_payload,
            q5b_capsule_digest=bytes.fromhex(predecessor["q5b_capsule_digest"][7:]),
            q5b_authorization_digest=bytes.fromhex(predecessor["q5b_authorization_digest"][7:]),
            q5b_consumption_record_digest=bytes.fromhex(predecessor["q5b_consumption_record_digest"][7:]),
            q5b_terminal_record_digest=bytes.fromhex(predecessor["q5b_terminal_record_digest"][7:]),
            created_at_unix_seconds=capsule["certificate_created_at_unix_seconds"],
            role_order_signatures=tuple(signatures),  # type: ignore[arg-type]
            q6_source_commit=capsule["source_commit"],
            q6_source_closure_root=bytes.fromhex(capsule["source_closure_root"][7:]),
            complete_checkpoint_digest=bytes.fromhex(complete_checkpoint_digest[7:]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"Q6 issuance artifact construction rejected: {error}")
    return result


def validate_artifact_binding(
    artifact_payload: bytes,
    envelope_payload: bytes,
    capsule: Mapping[str, Any],
    *,
    q4_artifact_payload: bytes,
    q5b_artifact_payload: bytes,
    complete_checkpoint_digest: str,
    verified_checkpoints: Sequence[Mapping[str, Any]],
) -> _wire.Q6FormalIssuanceReceiptV1:
    if type(artifact_payload) is not bytes or len(artifact_payload) > MAX_ARTIFACT_BYTES:
        _fail("Q6 artifact byte bound differs")
    try:
        receipt = _wire.decode_q6_formal_issuance_receipt_v1(artifact_payload)
    except (TypeError, ValueError) as error:
        _fail(f"Q6 issuance artifact rejected: {error}")
    predecessor = capsule["preflight_receipt"]["predecessor_receipt"]
    _tagged(complete_checkpoint_digest, "Q6 complete checkpoint digest")
    if tuple(item.get("signer_ordinal") for item in verified_checkpoints) != COMPLETE_PREFIX:
        _fail("Q6 artifact replay requires three ordered verified checkpoints")
    envelope_by_key = {
        record.key_id.hex(): record.signature for record in receipt.envelope.signatures
    }
    for checkpoint in verified_checkpoints:
        signature = envelope_by_key.get(checkpoint.get("key_id_hex"))
        unsigned_checkpoint = deepcopy(dict(checkpoint))
        supplied_checkpoint_digest = unsigned_checkpoint.pop(
            "signature_receipt_digest", None
        )
        ordinal = checkpoint.get("signer_ordinal")
        if (
            signature is None
            or checkpoint.get("status") != SIGNATURE_VERIFIED
            or supplied_checkpoint_digest
            != _digest(DOMAINS["signature_receipt"], unsigned_checkpoint)
            or type(ordinal) is not int
            or checkpoint.get("completed_prefix") != list(range(1, ordinal + 1))
            or checkpoint.get("q6_action_audit")
            != q6_action_audit(tuple(range(1, ordinal + 1)))
            or checkpoint.get("capsule_digest") != capsule["capsule_digest"]
            or checkpoint.get("signature_preimage_root")
            != capsule["preflight_receipt"]["signature_preimage_root"]
            or checkpoint.get("signing_message_raw_sha256")
            != capsule["preflight_receipt"]["signing_message_raw_sha256"]
            or checkpoint.get("signature_raw_sha256") != raw_sha256(signature)
            or checkpoint.get("producer_free_signature_verified") is not True
        ):
            _fail("Q6 artifact/verified-checkpoint structural binding differs")
    if (
        receipt.envelope.canonical_bytes != envelope_payload
        or receipt.q5b_artifact.canonical_bytes != q5b_artifact_payload
        or receipt.q4_artifact.canonical_bytes != q4_artifact_payload
        or "sha256:" + receipt.q5b_capsule_digest.hex() != predecessor["q5b_capsule_digest"]
        or "sha256:" + receipt.q5b_authorization_digest.hex() != predecessor["q5b_authorization_digest"]
        or "sha256:" + receipt.q5b_consumption_record_digest.hex()
        != predecessor["q5b_consumption_record_digest"]
        or "sha256:" + receipt.q5b_terminal_record_digest.hex()
        != predecessor["q5b_terminal_record_digest"]
        or receipt.signature_preimage.created_at_unix_seconds
        != capsule["certificate_created_at_unix_seconds"]
        or "sha256:" + receipt.signature_preimage.preimage_root.hex()
        != capsule["preflight_receipt"]["signature_preimage_root"]
        or receipt.q6_source_commit_raw.hex() != capsule["source_commit"]
        or "sha256:" + receipt.q6_source_closure_root.hex() != capsule["source_closure_root"]
        or "sha256:" + receipt.complete_checkpoint_digest.hex()
        != complete_checkpoint_digest
    ):
        _fail("Q6 issuance artifact authority differs")
    return receipt


def build_complete_terminal(
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    artifact_payload: bytes,
    envelope_payload: bytes,
    receipt: _wire.Q6FormalIssuanceReceiptV1,
    verified_checkpoints: Sequence[Mapping[str, Any]],
    *,
    complete_checkpoint_digest: str,
    pending_history_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(item.get("signer_ordinal") for item in verified_checkpoints) != COMPLETE_PREFIX:
        _fail("Q6 terminal requires three ordered verified checkpoints")
    if any(
        item.get("capsule_digest") != capsule["capsule_digest"]
        or item.get("authorization_digest") != authorization["authorization_digest"]
        or item.get("consumption_record_digest") != consumption["record_digest"]
        for item in verified_checkpoints
    ):
        _fail("Q6 terminal verified-checkpoint authority differs")
    _tagged(complete_checkpoint_digest, "Q6 complete checkpoint digest")
    expected_checkpoint = build_checkpoint(
        contract, capsule, authorization, consumption, verified_checkpoints
    )
    if expected_checkpoint["checkpoint_digest"] != complete_checkpoint_digest:
        _fail("Q6 terminal checkpoint/verified-receipt linkage differs")
    if "sha256:" + receipt.complete_checkpoint_digest.hex() != complete_checkpoint_digest:
        _fail("Q6 terminal complete checkpoint binding differs")
    history = validate_pending_history_receipt(pending_history_receipt)
    outcome = {
        "status": _wire.Q6_SUCCESS_STATUS,
        "capsule_digest": capsule["capsule_digest"],
        "completed_prefix": [1, 2, 3],
        "artifact_raw_sha256": raw_sha256(artifact_payload),
        "issuance_receipt_root": "sha256:" + receipt.receipt_root.hex(),
        "envelope_raw_sha256": raw_sha256(envelope_payload),
        "signature_preimage_root": "sha256:" + receipt.signature_preimage.preimage_root.hex(),
        "complete_checkpoint_digest": complete_checkpoint_digest,
        "pending_history_receipt": history,
        "valid_signature_count": 3,
        "formal_certificate_signed": True,
        "certificate_envelope_published": True,
        "outside_certificate_issued": True,
        "external_authority": False,
        "global_active_transition_performed": False,
        "synthesis_released": False,
        "q6_action_audit": deepcopy(SUCCESS_ACTION_AUDIT),
        "campaign_complete": True,
        "terminal_no_resume": False,
    }
    value: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": _wire.Q6_SUCCESS_STATUS,
        "contract_raw_sha256": contract.raw_digest,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": outcome,
    }
    value["record_digest"] = _digest(DOMAINS["terminal"], value)
    return value


def build_invalidated_terminal(
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
    observed_action_audit: Mapping[str, Any],
    pending_history_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    prefix = tuple(item["signer_ordinal"] for item in receipts)
    audit = validate_observed_action_audit(observed_action_audit)
    history = validate_pending_history_receipt(pending_history_receipt)
    if tuple(audit["completed_prefix"]) != prefix:
        _fail("Q6 invalidated audit/prefix binding differs")
    value: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "contract_raw_sha256": contract.raw_digest,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": {
            "status": INVALIDATED_TERMINAL_NO_RESUME,
            "completed_prefix": list(prefix),
            "failure_phase": failure_phase,
            "failure_code": failure_code,
            "failure_detail": failure_detail,
            "formal_certificate_signed": False,
            "certificate_envelope_published": False,
            "outside_certificate_issued": False,
            "external_authority": False,
            "global_active_transition_performed": False,
            "synthesis_released": False,
            "q6_action_audit": audit,
            "pending_history_receipt": history,
            "campaign_complete": False,
            "terminal_no_resume": True,
        },
    }
    value["record_digest"] = _digest(DOMAINS["terminal"], value)
    return value


def validate_terminal(
    value: object,
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    artifact_payload: bytes | None = None,
    envelope_payload: bytes | None = None,
    q4_artifact_payload: bytes | None = None,
    q5b_artifact_payload: bytes | None = None,
    verified_checkpoints: Sequence[Mapping[str, Any]] = (),
    complete_checkpoint_digest: str | None = None,
    pending_history_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q6 terminal differs")
    unsigned = deepcopy(value)
    supplied = unsigned.pop("record_digest", None)
    if (
        supplied != _digest(DOMAINS["terminal"], unsigned)
        or value.get("schema_version") != TERMINAL_SCHEMA
        or value.get("contract_raw_sha256") != contract.raw_digest
        or value.get("capsule_digest") != capsule["capsule_digest"]
        or value.get("authorization_digest") != authorization["authorization_digest"]
        or value.get("consumption_record_digest") != consumption["record_digest"]
        or value.get("authorization_consumption_count") != 1
        or value.get("logical_campaign_count") != 1
    ):
        _fail("Q6 terminal authority differs")
    if value.get("status") == _wire.Q6_SUCCESS_STATUS:
        if any(item is None for item in (artifact_payload, envelope_payload, q4_artifact_payload, q5b_artifact_payload)):
            _fail("Q6 success terminal requires sealed artifact inputs")
        expected_checkpoint = build_checkpoint(
            contract, capsule, authorization, consumption, verified_checkpoints
        )
        if complete_checkpoint_digest != expected_checkpoint["checkpoint_digest"]:
            _fail("Q6 terminal complete checkpoint replay differs")
        receipt = validate_artifact_binding(
            artifact_payload,  # type: ignore[arg-type]
            envelope_payload,  # type: ignore[arg-type]
            capsule,
            q4_artifact_payload=q4_artifact_payload,  # type: ignore[arg-type]
            q5b_artifact_payload=q5b_artifact_payload,  # type: ignore[arg-type]
            complete_checkpoint_digest=complete_checkpoint_digest,  # type: ignore[arg-type]
            verified_checkpoints=verified_checkpoints,
        )
        expected = build_complete_terminal(
            contract, capsule, authorization, consumption,
            artifact_payload, envelope_payload, receipt,  # type: ignore[arg-type]
            verified_checkpoints,
            complete_checkpoint_digest=complete_checkpoint_digest,  # type: ignore[arg-type]
            pending_history_receipt=pending_history_receipt,  # type: ignore[arg-type]
        )
        if value != expected:
            _fail("Q6 success terminal differs")
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME:
        outcome = value.get("outcome")
        if type(outcome) is not dict or outcome.get("terminal_no_resume") is not True:
            _fail("Q6 invalidated terminal differs")
        audit = validate_observed_action_audit(outcome.get("q6_action_audit"))
        prefix = completed_prefix_tuple(outcome.get("completed_prefix"))
        placeholder_receipts = [
            {"signer_ordinal": ordinal} for ordinal in prefix
        ]
        expected = build_invalidated_terminal(
            contract,
            capsule,
            authorization,
            consumption,
            placeholder_receipts,
            failure_phase=outcome.get("failure_phase"),
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
            observed_action_audit=audit,
            pending_history_receipt=pending_history_receipt,  # type: ignore[arg-type]
        )
        if value != expected:
            _fail("Q6 invalidated terminal exact replay differs")
    else:
        _fail("Q6 terminal status differs")
    return deepcopy(value)


__all__ = [
    "ACTION_ID", "AUTHORITY_BOUNDARY", "AUTHORIZATION_CONSUMED",
    "AUTHORIZATION_RECORDED", "CAPSULE_READY", "CLEAN_CHECKPOINT",
    "COMPLETE_PREFIX", "CONSUMPTION_SCHEMA", "CONTRACT_ID", "CONTRACT_SCHEMA",
    "DOMAINS", "EXECUTION_POLICY", "INVALIDATED_TERMINAL_NO_RESUME",
    "LoadedJson", "NOT_AUTHORIZED", "PARTIAL_FINALIZATION_REQUIRED",
    "NEXT_SIGNER_PREFIXES", "PENDING_CLEAN_CHECKPOINT", "PENDING_NOT_CONSUMED",
    "PENDING_PREFIXES", "PredecessorReplayV1",
    "Q6CompletionReplayError", "Q6_SOURCE_PATHS", "SIGNATURE_RECEIPT_SCHEMA",
    "SIGNATURE_VERIFIED", "SIGNER_ORDINALS", "SIGNER_ROLE_NAMES",
    "SIGNING_INTENT_RECORDED", "SUCCESS_ACTION_AUDIT", "VERIFICATION_INTENT_RECORDED",
    "VERIFICATION_INTENT_SCHEMA", "WORKER_RECEIPT_SCHEMA",
    "WORKER_SMOKE_SCHEMA", "WORKER_SMOKE_STATUS", "WORKER_SUCCESS_STATUS",
    "ZERO_ACTION_AUDIT", "build_authorization", "build_capsule",
    "build_checkpoint", "build_complete_terminal", "build_consumption",
    "build_invalidated_terminal", "build_issuance_artifact", "build_pending",
    "build_preflight_receipt", "build_signature_receipt", "build_signing_intent",
    "build_verification_intent",
    "canonical_json_bytes", "completed_prefix_tuple", "expected_worker_receipt",
    "load_contract", "load_preregistration", "q6_action_audit",
    "q6_observed_action_audit", "raw_sha256", "replay_predecessors_v1",
    "source_closure_root", "strict_json_bytes", "validate_artifact_binding",
    "validate_authorization", "validate_capsule", "validate_consumption",
    "validate_preflight_receipt", "validate_signature_receipt_sequence",
    "validate_signing_intent", "validate_source_closure_rows", "validate_terminal",
    "validate_observed_action_audit", "validate_verification_intent",
    "validate_worker_smoke_receipt",
]
