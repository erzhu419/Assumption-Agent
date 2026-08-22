"""Pure bytes-only owner-status replay core for Phase-2B Step 5.

This verifier independently re-freezes the owner-authorization status ``/1``
wire family.  It intentionally does not import the owner-status producer, CAS
service, logical-time service, runtime closure, auditor, or qualification
modules.  The only project dependencies are the pure authority/profile V2
verifier, the pure logical-time replay core, and the RFC-8785 implementation;
Ed25519 verification is performed directly with ``cryptography``.

A successful result proves canonical bytes, the locked authority/profile V2
graph, the governance two-purpose policy, governance event signatures, the
caller-supplied ACTIVE -> optional REVOKED chain, the complete-event-set root,
the legacy CAS-reference *candidate* shape, and cross-replay of the complete
caller-supplied logical-time chain.  It does not prove that the supplied final
head is externally current.  It also does not authenticate any CAS receipt,
observe an anti-fork anchor or runtime, authenticate an owner signature, grant
authority, satisfy R01, change readiness, or authorize execution.

The public verifier captures its call graph at import time.  Its result is
privately issued and retains exact immutable snapshots and strong object
identity, so direct ``object.__new__`` fabrication and ordinary
``object.__setattr__`` mutation fail closed.  This is a same-module KAT
property, not resistance to arbitrary interpreter or process-memory mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    Phase2BStep5AuthorityErrorV2 as _AuthorityErrorV2,
    load_step5_authority_profile_v2 as _load_authority_profile_v2,
    load_step5_signer_registry_v2 as _load_signer_registry_v2,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_trust_lock_v2,
    require_locked_step5_authority_graph_v2 as _require_authority_graph_v2,
)
from .phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 import (
    ExpectedLogicalEventCoreV1,
    Phase2BStep5LogicalTimeReplayCoreErrorV1 as _LogicalTimeCoreErrorV1,
    require_verified_logical_time_replay_core_v1 as _require_time_result_v1,
    verify_logical_time_replay_core_v1 as _verify_time_core_v1,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical_rfc8785_v1,
)

try:  # Importability is preserved; signature verification still fails closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - pinned verifier-image dependency.
    _InvalidSignature = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-replay-core/1"
)

OWNER_AUTHORIZATION_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-c1-owner-actual-unsealed-960-authorization/1"
)
OWNER_AUTHORIZATION_ID_V1: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)
OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1: Final = (
    "26ff84c3fe239158b1e8db24b5634ccfe8b3f3995d54bdc1f728c49a70f8fca1"
)
OWNER_AUTHORIZATION_SOURCE_SHA256_V1: Final = (
    "4c96d50706e520f1c67b2d2c31fce9ed605feacbf4c1177ddf9a7a2a9c1d9a73"
)
PROTOCOL_ID_V1: Final = (
    "phase2b_protocol_"
    "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
EXACT_FREEZE_VERSION_V1: Final = "hegel-freeze-p2b-p3-v1.0.1"
EXACT_FREEZE_ID_V1: Final = (
    "phase2b_exact_freeze_"
    "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
AUTHORITY_CLASS_V1: Final = "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
AUTHORITY_REQUIREMENT_ID_V1: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
ACTOR_PROFILE_CONTENT_ID_V1: Final = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
ACTOR_PROFILE_CANONICAL_SHA256_V1: Final = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)
LOGICAL_STAGE_WIRE_SCHEMA_ID_V1: Final = (
    "phase2b_c1_logical_stage_wire_schema_v1_"
    "1a625fbc369b057b9fc295fec02391d0421fc071ca41905b15901d4365833b5d"
)
LOGICAL_STAGE_WIRE_POLICY_ID_V1: Final = (
    "phase2b_c1_logical_stage_wire_policy_v2_"
    "c88653f2caa92eae1dfb7456217c4ca05c339e8f23e0540a06e435552dc59573"
)
STAGE_EVENT_CONTRACT_MANIFEST_ID_V1: Final = (
    "phase2b_c1_stage_event_contract_manifest_v1_"
    "c0929d05bdae56205e08700ab760e88a4668c686e73fe4be7a00ce6d69f9baf7"
)

EXACT_OWNER_AUTHORIZATION_BINDING_V1: Final = MappingProxyType({
    "authorization_schema": OWNER_AUTHORIZATION_SCHEMA_V1,
    "authorization_id": OWNER_AUTHORIZATION_ID_V1,
    "authorization_config_sha256": OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1,
    "authorization_source_sha256": OWNER_AUTHORIZATION_SOURCE_SHA256_V1,
    "protocol_id": PROTOCOL_ID_V1,
    "exact_freeze_version": EXACT_FREEZE_VERSION_V1,
    "exact_freeze_id": EXACT_FREEZE_ID_V1,
    "authority_class": AUTHORITY_CLASS_V1,
    "authority_requirement_id": AUTHORITY_REQUIREMENT_ID_V1,
    "actor_profile_content_id": ACTOR_PROFILE_CONTENT_ID_V1,
    "actor_profile_canonical_sha256": ACTOR_PROFILE_CANONICAL_SHA256_V1,
    "logical_stage_wire_schema_id": LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
    "logical_stage_wire_policy_id": LOGICAL_STAGE_WIRE_POLICY_ID_V1,
    "stage_event_contract_manifest_id": STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
})

OWNER_STATUS_DECISION_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-decision/1"
)
SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-owner-status-event/1"
)
OWNER_STATUS_CHECKPOINT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-checkpoint/1"
)
OWNER_STATUS_HEAD_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-head/1"
)
OWNER_STATUS_PROJECTION_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-projection/1"
)
CAS_DEPENDENCY_REFERENCE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-cas-dependency-reference/1"
)
LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-logical-time-dependency-reference/1"
)

AUTHORITY_PROFILE_PREFIX_V2: Final = "phase2b_step5_authority_profile_v2_"
SIGNER_REGISTRY_PREFIX_V2: Final = "phase2b_step5_signer_registry_v2_"
STAGE_B_TRUST_LOCK_PREFIX_V2: Final = "phase2b_step5_stage_b_trust_lock_v2_"
ROLE_KEY_PREFIX_V1: Final = "phase2b_step5_role_key_v1_"
LEDGER_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_ledger_v1_"
DECISION_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_decision_v1_"
EVENT_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_event_v1_"
CHECKPOINT_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_checkpoint_v1_"
HEAD_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_head_v1_"
PROJECTION_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_projection_v1_"
LOGICAL_TIME_TOKEN_ID_PREFIX_V1: Final = "phase2b_step5_logical_time_token_v1_"

GOVERNANCE_ROLE_V1: Final = "GOVERNANCE_ANCHOR"
GOVERNANCE_CONTROL_DOMAIN_V1: Final = "STEP5_GOVERNANCE_CONTROL"
GOVERNANCE_SERVICE_NAME_V2: Final = "governance_status"
SIGN_SIGNER_REGISTRY_PURPOSE_V1: Final = "SIGN_SIGNER_REGISTRY"
SIGN_OWNER_STATUS_EVENT_PURPOSE_V1: Final = (
    "SIGN_OWNER_AUTHORIZATION_STATUS_EVENT"
)
REQUIRED_GOVERNANCE_PURPOSES_V1: Final = (
    SIGN_SIGNER_REGISTRY_PURPOSE_V1,
    SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
)
GOVERNANCE_KEY_EPOCH_V1: Final = 1
ED25519_ALGORITHM_V1: Final = "Ed25519"

CAS_STORE_ROLE_V1: Final = "CAS_STORE"
CAS_STORE_RECEIPT_PURPOSE_V1: Final = "SIGN_CAS_IF_ABSENT_RECEIPT"
CAS_STORE_KEY_EPOCH_V1: Final = 1
CAS_STORE_SIGNATURE_DOMAIN_V1: Final = (
    "HEGEL_MACHINE_PHASE2B_FORMAL_WIRE_COVERT_STEP5_CAS_STORE_RECEIPT_V1"
)
CAS_SIGNED_RECEIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-cas-store-receipt/1"
)
CAS_CHECKPOINT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-checkpoint/1"
)
CAS_HEAD_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-head/1"
)
CAS_DEPENDENCY_VERIFICATION_STATUS_V1: Final = (
    "STORE_DEPENDENCY_CANDIDATE_NOT_VERIFIED"
)

LOGICAL_TIME_ROLE_V1: Final = "LOGICAL_TIME"
LOGICAL_TIME_SIGNATURE_PURPOSE_V1: Final = "SIGN_LOGICAL_TIME_TOKEN"
LOGICAL_TIME_EVENT_PURPOSE_V1: Final = "OWNER_AUTHORIZATION_STATUS_DECISION"
LOGICAL_TIME_KEY_EPOCH_V1: Final = 1
LOGICAL_TIME_TOKEN_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-logical-time-token/1"
)
LOGICAL_TIME_DEPENDENCY_VERIFICATION_STATUS_V1: Final = (
    "LOGICAL_TIME_TOKEN_SIGNATURE_VERIFIED_MECHANICS_ONLY_NOT_AUTHORITY"
)

STATUS_ACTIVE_V1: Final = "ACTIVE"
STATUS_REVOKED_V1: Final = "REVOKED"
STATUS_NONE_V1: Final = "NONE"
EVENT_INITIAL_ACTIVE_V1: Final = "OWNER_AUTHORIZATION_INITIAL_ACTIVE"
EVENT_REVOKED_V1: Final = "OWNER_AUTHORIZATION_REVOKED"
HEAD_PURPOSE_V1: Final = "OWNER_AUTHORIZATION_STATUS_APPEND_ONLY_HEAD"
CHECKPOINT_PURPOSE_V1: Final = (
    "OWNER_AUTHORIZATION_STATUS_APPEND_ONLY_CHECKPOINT"
)
PROJECTION_CLAIM_LEVEL_V1: Final = (
    "STATUS_LEDGER_REPLAY_PROJECTION_CANDIDATE_NOT_LATEST_AUTHORITY_NOT_R01"
)
EVENT_CLAIM_BOUNDARY_V1: Final = (
    "GOVERNANCE_SIGNED_STATUS_EVENT_MECHANICS_CANDIDATE_AUTHORITY_V2_PURPOSE_"
    "STATICALLY_FROZEN_NOT_RAW_REGISTRY_VERIFIED_NOT_R01_NOT_READINESS_NOT_"
    "EXECUTION"
)
AUTHORITY_CLAIM_BOUNDARY_V1: Final = (
    "PURE_OWNER_STATUS_REPLAY_CANONICAL_AUTHORITY_GRAPH_GOVERNANCE_SIGNATURE_"
    "AND_SUPPLIED_CHAIN_MECHANICS_ONLY_LEGACY_CAS_REFERENCE_CANDIDATE_NOT_"
    "CAS_VERIFIED_NOT_EXTERNAL_CURRENTNESS_NOT_ANTI_FORK_NOT_RUNTIME_NOT_"
    "OWNER_SIGNATURE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)

GENESIS_EVENT_ID_V1: Final = EVENT_ID_PREFIX_V1 + ("0" * 64)
GENESIS_CHECKPOINT_ID_V1: Final = CHECKPOINT_ID_PREFIX_V1 + ("0" * 64)

_ROLE_KEY_ID_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/ROLE_KEY/V1\x00"
_LEDGER_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/LEDGER_ID/V1\x00"
)
_DECISION_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/DECISION_ID/V1\x00"
)
_GOVERNANCE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/SIGN/OWNER_AUTHORIZATION_STATUS_EVENT/GOVERNANCE/V1\x00"
)
GOVERNANCE_SIGNATURE_DOMAIN_V1: Final = (
    "HEGEL/PHASE2B/STEP5/SIGN/OWNER_AUTHORIZATION_STATUS_EVENT/GOVERNANCE/V1"
)
_EVENT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/EVENT_ID/V1\x00"
)
_CHECKPOINT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/CHECKPOINT_ID/V1\x00"
)
_HEAD_ID_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/HEAD_ID/V1\x00"
_PROJECTION_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/PROJECTION_ID/V1\x00"
)
_COMPLETE_EVENT_SET_ROOT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/COMPLETE_EVENT_SET_ROOT/V1\x00"
)

MAXIMUM_OWNER_AUTHORIZATION_BYTES: Final = 128 * 1024
MAXIMUM_DECISION_BYTES: Final = 256 * 1024
MAXIMUM_EVENT_BYTES: Final = 768 * 1024
MAXIMUM_CHECKPOINT_BYTES: Final = 1024 * 1024
MAXIMUM_HEAD_BYTES: Final = 128 * 1024
MAXIMUM_PROJECTION_BYTES: Final = 256 * 1024
_MAXIMUM_SAFE_JSON_INTEGER: Final = (1 << 53) - 1
_MAXIMUM_IDENTIFIER_ASCII_BYTES: Final = 512

_LOWER_HEX_32_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX_64_PATTERN: Final = re.compile(r"[0-9a-f]{128}")
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_PATTERN: Final = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}"
)
_UPPER_TOKEN_PATTERN: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}")

_DECISION_KEYS: Final = frozenset(
    {
        "schema_version", "ledger_id", "authority_profile_id",
        "signer_registry_id", "governance_key_id", "governance_key_epoch",
        "attempt_context_id", "owner_authorization_binding",
        "owner_authorization_payload_sha256", "sequence", "event_type",
        "prior_effective_status", "effective_status", "revocation_count",
        "reason_code", "decision_nonce", "previous_event_id",
        "previous_head_id", "previous_checkpoint_id",
        "previous_complete_event_set_root", "decision_id",
    }
)
_CAS_DEPENDENCY_KEYS: Final = frozenset(
    {
        "schema_version", "verification_status", "authority_profile_id",
        "signer_registry_id", "attempt_context_id", "retention_session_id",
        "chain_id", "signer_role", "signer_key_id", "signer_key_epoch",
        "signature_purpose", "signature_domain", "bound_artifact_id",
        "bound_artifact_sha256", "object_id", "object_canonical_sha256",
        "signed_receipt_schema_version", "signed_receipt_id",
        "signed_receipt_canonical_sha256", "checkpoint_schema_version",
        "checkpoint_id", "checkpoint_canonical_sha256", "checkpoint_index",
        "head_schema_version", "head_id", "head_canonical_sha256",
        "head_last_checkpoint_id", "head_checkpoint_count",
    }
)
_LOGICAL_TIME_DEPENDENCY_KEYS: Final = frozenset(
    {
        "schema_version", "verification_status", "authority_profile_id",
        "signer_registry_id", "attempt_context_id", "signer_role",
        "signer_key_id", "signer_key_epoch", "signature_purpose",
        "token_schema_version", "token_id", "token_canonical_sha256",
        "token_bound_artifact_id", "token_bound_artifact_sha256",
        "token_sequence",
    }
)
_EVENT_KEYS: Final = frozenset(
    {
        "schema_version", "signature_purpose", "signature_domain",
        "authority_profile_id", "signer_registry_id", "governance_key_id",
        "governance_key_epoch", "governance_public_key_hex",
        "attempt_context_id", "decision", "decision_canonical_sha256",
        "cas_store_dependency", "logical_time_dependency", "algorithm",
        "signature_hex", "event_id", "claim_boundary",
    }
)
_CHECKPOINT_KEYS: Final = frozenset(
    {
        "schema_version", "purpose", "ledger_id", "authority_profile_id",
        "signer_registry_id", "governance_key_id", "governance_key_epoch",
        "attempt_context_id", "checkpoint_index", "previous_head_id",
        "previous_checkpoint_id", "previous_event_id",
        "previous_complete_event_set_root", "event_id",
        "signed_event_canonical_sha256", "signed_event",
        "complete_event_set_root", "checkpoint_id",
    }
)
_HEAD_KEYS: Final = frozenset(
    {
        "schema_version", "purpose", "ledger_id", "authority_profile_id",
        "signer_registry_id", "governance_key_id", "governance_key_epoch",
        "attempt_context_id", "event_count", "next_sequence",
        "last_event_id", "last_checkpoint_id", "complete_event_set_root",
        "latest_effective_status", "revocation_count", "terminal", "head_id",
    }
)
_PROJECTION_KEYS: Final = frozenset(
    {
        "schema_version", "claim_level", "ledger_id", "authority_profile_id",
        "signer_registry_id", "governance_key_id", "governance_key_epoch",
        "attempt_context_id", "owner_authorization_binding",
        "owner_authorization_payload_sha256", "event_count", "latest_sequence",
        "latest_effective_status", "revocation_count", "last_event_id",
        "last_checkpoint_id", "head_id", "complete_event_set_root",
        "governance_signatures_verified",
        "governance_status_purpose_authorized_by_pinned_registry",
        "owner_signature_authenticated", "logical_time_token_signatures_verified",
        "store_dependency_candidate_references_exact",
        "store_dependencies_verified", "latest_authority_status_verified",
        "r01_satisfied", "readiness_node_transition_issued",
        "execution_authorized", "authority_evidence", "projection_id",
    }
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_SCHEMA"
FAIL_AUTHORITY_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_AUTHORITY"
FAIL_BINDING_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_BINDING"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_CONTENT_ID"
FAIL_SIGNATURE_BACKEND_V1: Final = (
    "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_SIGNATURE_BACKEND"
)
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_SIGNATURE"
FAIL_CHAIN_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_CHAIN"
FAIL_LOGICAL_TIME_V1: Final = (
    "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_LOGICAL_TIME"
)
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_CLAIM_BOUNDARY"
)
FAIL_RESULT_V1: Final = "FAIL_STEP5_OWNER_STATUS_REPLAY_CORE_V1_RESULT"


class Phase2BStep5OwnerStatusReplayCoreErrorV1(ValueError):
    """Stable fail-closed error for the pure owner-status replay core."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5OwnerStatusReplayCoreErrorV1(code, detail)


def _canonical(
    value: object,
    _canonicalize=_canonicalize_rfc8785_v1,
    _rfc_error_type=_RFC8785Error,
    _error_type=Phase2BStep5OwnerStatusReplayCoreErrorV1,
) -> bytes:
    try:
        return _canonicalize(value)  # type: ignore[arg-type]
    except (_rfc_error_type, TypeError, ValueError, RecursionError) as exc:
        raise _error_type(FAIL_CANONICAL_V1, str(exc)) from exc


def _restore_integral_numbers(
    value: object,
    _maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
    _fail_exact=_fail,
) -> object:
    def restore(item: object) -> object:
        if type(item) is float:
            if not item.is_integer() or not abs(item) <= _maximum:
                _fail_exact(
                    FAIL_SCHEMA_V1,
                    "owner-status /1 wire schemas admit exact safe integers only",
                )
            return int(item)
        if type(item) is list:
            return [restore(child) for child in item]
        if type(item) is dict:
            return {key: restore(child) for key, child in item.items()}
        return item

    return restore(value)


def _decode(
    payload: bytes,
    maximum_bytes: int,
    label: str,
    _decode_exact=_decode_canonical_rfc8785_v1,
    _restore=_restore_integral_numbers,
    _canonical_exact=_canonical,
    _rfc_error_type=_RFC8785Error,
    _error_type=Phase2BStep5OwnerStatusReplayCoreErrorV1,
    _fail_exact=_fail,
) -> dict[str, object]:
    if type(payload) is not bytes:
        _fail_exact(FAIL_INPUT_V1, f"{label} must be exact bytes")
    if not 1 <= len(payload) <= maximum_bytes or payload.endswith(b"\n"):
        _fail_exact(FAIL_INPUT_V1, f"{label} is outside its exact byte boundary")
    try:
        decoded = _decode_exact(payload)
    except (_rfc_error_type, TypeError, ValueError, RecursionError) as exc:
        raise _error_type(FAIL_CANONICAL_V1, f"{label}: {exc}") from exc
    restored = _restore(decoded)
    if type(restored) is not dict or _canonical_exact(restored) != payload:
        _fail_exact(FAIL_CANONICAL_V1, f"{label} canonical readback differs")
    return restored


def _exact_object(
    value: object,
    fields: frozenset[str],
    label: str,
    _fail_exact=_fail,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields):
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exact field set differs")
    return value


def _identifier(
    value: object,
    label: str,
    _pattern=_IDENTIFIER_PATTERN,
    _maximum: int = _MAXIMUM_IDENTIFIER_ASCII_BYTES,
    _error_type=Phase2BStep5OwnerStatusReplayCoreErrorV1,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be one bounded identifier")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _error_type(FAIL_SCHEMA_V1, f"{label} must be exact ASCII") from exc
    if len(encoded) > _maximum:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exceeds its ASCII byte bound")
    return value


def _prefixed_id(
    value: object,
    prefix: str,
    label: str,
    _validate_identifier=_identifier,
    _digest_pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
) -> str:
    result = _validate_identifier(value, label)
    suffix = result[len(prefix) :] if result.startswith(prefix) else ""
    if not result.startswith(prefix) or _digest_pattern.fullmatch(suffix) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} prefix or digest differs")
    return result


def _artifact_id(
    value: object,
    label: str,
    _validate_identifier=_identifier,
    _sha_pattern=_SHA256_PATTERN,
    _digest_pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
) -> str:
    result = _validate_identifier(value, label)
    if _sha_pattern.fullmatch(result) is not None:
        return result
    if "_" not in result or _digest_pattern.fullmatch(result[-64:]) is None:
        _fail_exact(
            FAIL_SCHEMA_V1,
            f"{label} must carry one lowercase SHA-256 suffix",
        )
    return result


def _sha256_address(
    value: object,
    label: str,
    _pattern=_SHA256_PATTERN,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be sha256:<lowerhex64>")
    return value


def _hex32(
    value: object,
    label: str,
    _pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be exact lowercase 32-byte hex")
    return value


def _hex64(
    value: object,
    label: str,
    _pattern=_LOWER_HEX_64_PATTERN,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be exact lowercase 64-byte hex")
    return value


def _upper_token(
    value: object,
    label: str,
    _pattern=_UPPER_TOKEN_PATTERN,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be one uppercase token")
    return value


def _wire_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
    _fail_exact=_fail,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exact integer differs")
    return value


def _sha256_bytes(
    payload: bytes,
    _sha256=hashlib.sha256,
    _fail_exact=_fail,
) -> str:
    if type(payload) is not bytes:
        _fail_exact(FAIL_INPUT_V1, "SHA-256 input must be exact bytes")
    return "sha256:" + _sha256(payload).hexdigest()


def _content_id(
    prefix: str,
    domain: bytes,
    subject: object,
    _canonical_exact=_canonical,
    _sha256=hashlib.sha256,
) -> str:
    return prefix + _sha256(domain + _canonical_exact(subject)).hexdigest()


def _mapping_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
    _canonical_exact=_canonical,
) -> bool:
    return _canonical_exact(left) == _canonical_exact(right)


GENESIS_COMPLETE_EVENT_SET_ROOT_V1: Final = _sha256_bytes(
    _COMPLETE_EVENT_SET_ROOT_DOMAIN
    + _canonical(
        {
            "schema_version": (
                "hegel-machine-phase2b-step5-owner-status-complete-event-set-root/1"
            ),
            "events_in_sequence": [],
        }
    )
)


def _validate_owner_authorization_payload(
    payload: bytes,
    _decode_exact=_decode,
    _sha256=hashlib.sha256,
    _fail_exact=_fail,
) -> None:
    if type(payload) is not bytes:
        _fail_exact(FAIL_INPUT_V1, "owner-authorization payload must be exact bytes")
    if not 1 <= len(payload) <= MAXIMUM_OWNER_AUTHORIZATION_BYTES:
        _fail_exact(FAIL_INPUT_V1, "owner-authorization payload exceeds byte bound")
    if _sha256(payload).hexdigest() != OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1:
        _fail_exact(FAIL_BINDING_V1, "owner-authorization payload SHA-256 differs")
    wrapper = _decode_exact(
        payload,
        MAXIMUM_OWNER_AUTHORIZATION_BYTES,
        "owner-authorization payload",
    )
    if set(wrapper) != {"authorization_body", "authorization_id"}:
        _fail_exact(FAIL_SCHEMA_V1, "owner-authorization wrapper fields differ")
    body = wrapper["authorization_body"]
    if (
        type(body) is not dict
        or wrapper["authorization_id"] != OWNER_AUTHORIZATION_ID_V1
        or body.get("schema_version") != OWNER_AUTHORIZATION_SCHEMA_V1
    ):
        _fail_exact(FAIL_BINDING_V1, "owner-authorization ID or schema differs")


def _validate_owner_binding(
    value: object,
    _expected=EXACT_OWNER_AUTHORIZATION_BINDING_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(_expected):
        _fail_exact(FAIL_SCHEMA_V1, "owner-authorization binding fields differ")
    for key, expected in _expected.items():
        observed = value[key]
        if type(observed) is not type(expected) or observed != expected:
            _fail_exact(FAIL_BINDING_V1, "owner-authorization exact binding differs")
    return value


def _common_tuple(
    value: Mapping[str, object],
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_identifier=_identifier,
    _fail_exact=_fail,
) -> tuple[str, str, str, int, str]:
    profile_id = _validate_prefixed_id(
        value["authority_profile_id"],
        AUTHORITY_PROFILE_PREFIX_V2,
        "authority profile ID",
    )
    registry_id = _validate_prefixed_id(
        value["signer_registry_id"],
        SIGNER_REGISTRY_PREFIX_V2,
        "signer registry ID",
    )
    key_id = _validate_prefixed_id(
        value["governance_key_id"], ROLE_KEY_PREFIX_V1, "governance key ID"
    )
    epoch = _validate_integer(
        value["governance_key_epoch"], "governance key epoch", minimum=1
    )
    if epoch != GOVERNANCE_KEY_EPOCH_V1:
        _fail_exact(FAIL_BINDING_V1, "owner-status /1 requires key epoch 1")
    attempt = _validate_identifier(
        value["attempt_context_id"], "Step-5 attempt context ID"
    )
    return profile_id, registry_id, key_id, epoch, attempt


def _require_common(
    observed: tuple[str, str, str, int, str],
    expected: tuple[str, str, str, int, str],
    label: str,
    _fail_exact=_fail,
) -> None:
    if type(observed) is not tuple or observed != expected:
        _fail_exact(FAIL_BINDING_V1, f"{label} authority/key/attempt pins differ")


def _derive_governance_key_id(
    public_key: bytes,
    *,
    key_epoch: int = GOVERNANCE_KEY_EPOCH_V1,
    _validate_integer=_wire_integer,
    _content_address=_content_id,
    _fail_exact=_fail,
) -> str:
    if type(public_key) is not bytes or len(public_key) != 32:
        _fail_exact(FAIL_AUTHORITY_V1, "governance public key is not raw 32 bytes")
    epoch = _validate_integer(key_epoch, "governance key epoch", minimum=1)
    if epoch != GOVERNANCE_KEY_EPOCH_V1:
        _fail_exact(FAIL_AUTHORITY_V1, "governance key epoch differs")
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": GOVERNANCE_ROLE_V1,
        "control_domain": GOVERNANCE_CONTROL_DOMAIN_V1,
        "authorized_signature_purposes": list(REQUIRED_GOVERNANCE_PURPOSES_V1),
        "algorithm": ED25519_ALGORITHM_V1,
        "key_epoch": epoch,
        "public_key_hex": public_key.hex(),
    }
    return _content_address(ROLE_KEY_PREFIX_V1, _ROLE_KEY_ID_DOMAIN, subject)


def _ledger_id(
    common: tuple[str, str, str, int, str],
    _content_address=_content_id,
) -> str:
    subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-owner-status-ledger-identity/1"
        ),
        "authority_profile_id": common[0],
        "signer_registry_id": common[1],
        "governance_key_id": common[2],
        "governance_key_epoch": common[3],
        "attempt_context_id": common[4],
        "owner_authorization_id": OWNER_AUTHORIZATION_ID_V1,
        "owner_authorization_payload_sha256": (
            "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
        ),
    }
    return _content_address(LEDGER_ID_PREFIX_V1, _LEDGER_ID_DOMAIN, subject)


def _extract_governance_authority(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    _load_lock=_load_stage_b_trust_lock_v2,
    _load_profile=_load_authority_profile_v2,
    _load_registry=_load_signer_registry_v2,
    _require_graph=_require_authority_graph_v2,
    _derive_key_id=_derive_governance_key_id,
    _canonical_exact=_canonical,
    _authority_error_type=_AuthorityErrorV2,
    _error_type=Phase2BStep5OwnerStatusReplayCoreErrorV1,
    _public_key_pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
) -> tuple[str, str, str, str, int, bytes]:
    if any(
        type(value) is not bytes
        for value in (
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest,
        )
    ):
        _fail_exact(FAIL_INPUT_V1, "authority graph and OOB pins must be exact bytes")
    try:
        lock = _load_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
        )
        profile = _load_profile(authority_profile_payload, lock)
        registry = _load_registry(signer_registry_payload, lock, profile)
        lock, profile, registry = _require_graph(lock, profile, registry)
    except _authority_error_type as exc:
        raise _error_type(
            FAIL_AUTHORITY_V1, f"authority/profile V2 graph rejected: {exc}"
        ) from exc

    if registry.signature_backends != ("CRYPTOGRAPHY_ED25519",):
        _fail_exact(FAIL_AUTHORITY_V1, "authority signature backend lineage differs")
    rows = registry.body["role_keys_in_exact_order"]
    if type(rows) is not tuple or len(rows) != 8:
        _fail_exact(FAIL_AUTHORITY_V1, "authority registry is not exact-eight-role")
    governance_rows = tuple(
        row for row in rows if row["role"] == GOVERNANCE_ROLE_V1
    )
    if len(governance_rows) != 1:
        _fail_exact(FAIL_AUTHORITY_V1, "governance role cardinality differs")
    key = governance_rows[0]
    if not (
        key["schema_version"] == "hegel-machine-phase2b-step5-role-key/1"
        and key["control_domain"] == GOVERNANCE_CONTROL_DOMAIN_V1
        and type(key["authorized_signature_purposes"]) is tuple
        and key["authorized_signature_purposes"] == REQUIRED_GOVERNANCE_PURPOSES_V1
        and key["algorithm"] == ED25519_ALGORITHM_V1
        and type(key["key_epoch"]) is int
        and key["key_epoch"] == GOVERNANCE_KEY_EPOCH_V1
        and type(key["public_key_hex"]) is str
        and _public_key_pattern.fullmatch(key["public_key_hex"]) is not None
        and type(key["key_id"]) is str
    ):
        _fail_exact(FAIL_AUTHORITY_V1, "governance two-purpose policy differs")
    public_key = bytes.fromhex(key["public_key_hex"])
    expected_key_id = _derive_key_id(public_key, key_epoch=key["key_epoch"])
    if key["key_id"] != expected_key_id:
        _fail_exact(FAIL_AUTHORITY_V1, "governance role-key ID differs")

    services = profile.body["service_static_bindings_in_exact_order"]
    if type(services) is not tuple or len(services) != 5:
        _fail_exact(FAIL_AUTHORITY_V1, "profile static-service cardinality differs")
    governance_services = tuple(
        row for row in services if row["service_name"] == GOVERNANCE_SERVICE_NAME_V2
    )
    if len(governance_services) != 1:
        _fail_exact(FAIL_AUTHORITY_V1, "governance-status service cardinality differs")
    service = governance_services[0]
    if not (
        service["role"] == GOVERNANCE_ROLE_V1
        and service["control_domain"] == GOVERNANCE_CONTROL_DOMAIN_V1
        and service["role_key_id"] == expected_key_id
        and service["key_epoch"] == GOVERNANCE_KEY_EPOCH_V1
        and type(service["authorized_signature_purposes"]) is tuple
        and service["authorized_signature_purposes"] == REQUIRED_GOVERNANCE_PURPOSES_V1
    ):
        _fail_exact(FAIL_AUTHORITY_V1, "governance service/key purpose link differs")
    governance_root = profile.body["governance_root"]
    if not (
        type(governance_root) is type(key)
        and tuple(governance_root) == tuple(key)
        and all(
            type(governance_root[field]) is type(key[field])
            and governance_root[field] == key[field]
            for field in key
        )
    ):
        _fail_exact(FAIL_AUTHORITY_V1, "profile governance root differs from registry")
    _canonical_exact(
        {
            "role": key["role"],
            "control_domain": key["control_domain"],
            "authorized_signature_purposes": list(
                key["authorized_signature_purposes"]
            ),
            "algorithm": key["algorithm"],
            "key_epoch": key["key_epoch"],
            "public_key_hex": key["public_key_hex"],
            "key_id": key["key_id"],
        }
    )
    return (
        lock.artifact_id,
        profile.artifact_id,
        registry.artifact_id,
        expected_key_id,
        GOVERNANCE_KEY_EPOCH_V1,
        public_key,
    )


def _decision_id_subject(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "ledger_id": body["ledger_id"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "governance_key_id": body["governance_key_id"],
        "governance_key_epoch": body["governance_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "owner_authorization_binding": body["owner_authorization_binding"],
        "owner_authorization_payload_sha256": (
            body["owner_authorization_payload_sha256"]
        ),
        "sequence": body["sequence"],
        "event_type": body["event_type"],
        "prior_effective_status": body["prior_effective_status"],
        "effective_status": body["effective_status"],
        "revocation_count": body["revocation_count"],
        "reason_code": body["reason_code"],
        "decision_nonce": body["decision_nonce"],
        "previous_event_id": body["previous_event_id"],
        "previous_head_id": body["previous_head_id"],
        "previous_checkpoint_id": body["previous_checkpoint_id"],
        "previous_complete_event_set_root": (
            body["previous_complete_event_set_root"]
        ),
    }


def _validate_decision_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common_exact=_require_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_owner=_validate_owner_binding,
    _validate_integer=_wire_integer,
    _validate_upper=_upper_token,
    _validate_nonce=_hex32,
    _validate_sha256=_sha256_address,
    _ledger_identity=_ledger_id,
    _id_subject=_decision_id_subject,
    _content_address=_content_id,
    _decision_schema=OWNER_STATUS_DECISION_SCHEMA_V1,
    _event_initial=EVENT_INITIAL_ACTIVE_V1,
    _event_revoked=EVENT_REVOKED_V1,
    _status_none=STATUS_NONE_V1,
    _status_active=STATUS_ACTIVE_V1,
    _status_revoked=STATUS_REVOKED_V1,
    _genesis_event_id=GENESIS_EVENT_ID_V1,
    _genesis_checkpoint_id=GENESIS_CHECKPOINT_ID_V1,
    _genesis_root=GENESIS_COMPLETE_EVENT_SET_ROOT_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _DECISION_KEYS, "owner-status decision")
    if value["schema_version"] != _decision_schema:
        _fail_exact(FAIL_SCHEMA_V1, "owner-status decision schema differs")
    observed_common = _common_projection(value)
    _require_common_exact(observed_common, expected_common, "owner-status decision")
    _validate_prefixed_id(value["ledger_id"], LEDGER_ID_PREFIX_V1, "ledger ID")
    if value["ledger_id"] != _ledger_identity(expected_common):
        _fail_exact(FAIL_BINDING_V1, "decision ledger identity differs")
    _validate_owner(value["owner_authorization_binding"])
    if value["owner_authorization_payload_sha256"] != (
        "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
    ):
        _fail_exact(FAIL_BINDING_V1, "decision owner payload SHA-256 differs")
    sequence = _validate_integer(value["sequence"], "decision sequence", maximum=1)
    revocation_count = _validate_integer(
        value["revocation_count"], "decision revocation count", maximum=1
    )
    _validate_upper(value["reason_code"], "decision reason code")
    _validate_nonce(value["decision_nonce"], "decision nonce")
    _validate_prefixed_id(
        value["previous_event_id"], EVENT_ID_PREFIX_V1, "previous event ID"
    )
    _validate_prefixed_id(
        value["previous_head_id"], HEAD_ID_PREFIX_V1, "previous head ID"
    )
    _validate_prefixed_id(
        value["previous_checkpoint_id"],
        CHECKPOINT_ID_PREFIX_V1,
        "previous checkpoint ID",
    )
    _validate_sha256(
        value["previous_complete_event_set_root"],
        "previous complete-event-set root",
    )
    if sequence == 0:
        expected_transition = (
            _event_initial,
            _status_none,
            _status_active,
            0,
            _genesis_event_id,
            _genesis_checkpoint_id,
            _genesis_root,
        )
    else:
        expected_transition = (
            _event_revoked,
            _status_active,
            _status_revoked,
            1,
            value["previous_event_id"],
            value["previous_checkpoint_id"],
            value["previous_complete_event_set_root"],
        )
        if (
            value["previous_event_id"] == _genesis_event_id
            or value["previous_checkpoint_id"] == _genesis_checkpoint_id
            or value["previous_complete_event_set_root"]
            == _genesis_root
        ):
            _fail_exact(FAIL_CHAIN_V1, "REVOKED decision retains genesis linkage")
    observed_transition = (
        value["event_type"],
        value["prior_effective_status"],
        value["effective_status"],
        revocation_count,
        value["previous_event_id"],
        value["previous_checkpoint_id"],
        value["previous_complete_event_set_root"],
    )
    if observed_transition != expected_transition:
        _fail_exact(FAIL_CHAIN_V1, "owner-status decision transition differs")
    expected_id = _content_address(
        DECISION_ID_PREFIX_V1, _DECISION_ID_DOMAIN, _id_subject(value)
    )
    if value["decision_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "owner-status decision ID differs")
    return value


def _parse_decision(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    _decode_exact=_decode,
    _validate_mapping=_validate_decision_mapping,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, MAXIMUM_DECISION_BYTES, "owner-status decision"),
        expected_common,
    )


def _validate_cas_dependency_mapping(
    body: object,
    expected_profile_registry_attempt: tuple[str, str, str],
    *,
    _exact_mapping=_exact_object,
    _validate_prefixed_id=_prefixed_id,
    _validate_identifier=_identifier,
    _validate_artifact_id=_artifact_id,
    _validate_sha256=_sha256_address,
    _validate_integer=_wire_integer,
    _cas_dependency_schema=CAS_DEPENDENCY_REFERENCE_SCHEMA_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _CAS_DEPENDENCY_KEYS, "legacy CAS dependency")
    literals = (
        ("schema_version", _cas_dependency_schema),
        ("verification_status", CAS_DEPENDENCY_VERIFICATION_STATUS_V1),
        ("signer_role", CAS_STORE_ROLE_V1),
        ("signature_purpose", CAS_STORE_RECEIPT_PURPOSE_V1),
        ("signature_domain", CAS_STORE_SIGNATURE_DOMAIN_V1),
        ("signed_receipt_schema_version", CAS_SIGNED_RECEIPT_SCHEMA_V1),
        ("checkpoint_schema_version", CAS_CHECKPOINT_SCHEMA_V1),
        ("head_schema_version", CAS_HEAD_SCHEMA_V1),
    )
    if any(
        type(value[field]) is not str or value[field] != expected
        for field, expected in literals
    ):
        _fail_exact(FAIL_SCHEMA_V1, "legacy CAS dependency literal differs")
    profile = _validate_prefixed_id(
        value["authority_profile_id"],
        AUTHORITY_PROFILE_PREFIX_V2,
        "CAS authority profile ID",
    )
    registry = _validate_prefixed_id(
        value["signer_registry_id"],
        SIGNER_REGISTRY_PREFIX_V2,
        "CAS signer registry ID",
    )
    attempt = _validate_identifier(
        value["attempt_context_id"], "CAS attempt context ID"
    )
    if (profile, registry, attempt) != expected_profile_registry_attempt:
        _fail_exact(FAIL_BINDING_V1, "legacy CAS dependency graph forks")
    _validate_identifier(value["retention_session_id"], "CAS retention session ID")
    _validate_identifier(value["chain_id"], "CAS chain ID")
    _validate_prefixed_id(value["signer_key_id"], ROLE_KEY_PREFIX_V1, "CAS key ID")
    if _validate_integer(
        value["signer_key_epoch"], "CAS key epoch", minimum=1
    ) != CAS_STORE_KEY_EPOCH_V1:
        _fail_exact(FAIL_BINDING_V1, "legacy CAS key epoch differs")
    _validate_artifact_id(value["bound_artifact_id"], "CAS-bound artifact ID")
    _validate_sha256(value["bound_artifact_sha256"], "CAS-bound artifact SHA-256")
    _validate_artifact_id(value["object_id"], "CAS object ID")
    _validate_sha256(value["object_canonical_sha256"], "CAS object SHA-256")
    _validate_artifact_id(value["signed_receipt_id"], "CAS receipt ID")
    _validate_sha256(
        value["signed_receipt_canonical_sha256"], "CAS receipt SHA-256"
    )
    _validate_artifact_id(value["checkpoint_id"], "CAS checkpoint ID")
    _validate_sha256(
        value["checkpoint_canonical_sha256"], "CAS checkpoint SHA-256"
    )
    checkpoint_index = _validate_integer(
        value["checkpoint_index"], "CAS checkpoint index"
    )
    _validate_artifact_id(value["head_id"], "CAS head ID")
    _validate_sha256(value["head_canonical_sha256"], "CAS head SHA-256")
    _validate_artifact_id(
        value["head_last_checkpoint_id"], "CAS head last-checkpoint ID"
    )
    head_count = _validate_integer(
        value["head_checkpoint_count"], "CAS head checkpoint count", minimum=1
    )
    if not (
        value["object_id"] == value["object_canonical_sha256"]
        and value["signed_receipt_id"]
        == value["signed_receipt_canonical_sha256"]
        and value["checkpoint_id"] == value["checkpoint_canonical_sha256"]
        and value["head_id"] == value["head_canonical_sha256"]
        and value["head_last_checkpoint_id"] == value["checkpoint_id"]
        and head_count == checkpoint_index + 1
    ):
        _fail_exact(
            FAIL_BINDING_V1,
            "legacy CAS candidate content-address linkage differs",
        )
    return value


def _validate_logical_time_dependency_mapping(
    body: object,
    expected_profile_registry_attempt: tuple[str, str, str],
    *,
    _exact_mapping=_exact_object,
    _validate_prefixed_id=_prefixed_id,
    _validate_identifier=_identifier,
    _validate_artifact_id=_artifact_id,
    _validate_sha256=_sha256_address,
    _validate_integer=_wire_integer,
    _logical_dependency_schema=LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(
        body, _LOGICAL_TIME_DEPENDENCY_KEYS, "logical-time dependency"
    )
    literals = (
        ("schema_version", _logical_dependency_schema),
        (
            "verification_status",
            LOGICAL_TIME_DEPENDENCY_VERIFICATION_STATUS_V1,
        ),
        ("signer_role", LOGICAL_TIME_ROLE_V1),
        ("signature_purpose", LOGICAL_TIME_SIGNATURE_PURPOSE_V1),
        ("token_schema_version", LOGICAL_TIME_TOKEN_SCHEMA_V1),
    )
    if any(
        type(value[field]) is not str or value[field] != expected
        for field, expected in literals
    ):
        _fail_exact(FAIL_SCHEMA_V1, "logical-time dependency literal differs")
    profile = _validate_prefixed_id(
        value["authority_profile_id"],
        AUTHORITY_PROFILE_PREFIX_V2,
        "time dependency authority profile ID",
    )
    registry = _validate_prefixed_id(
        value["signer_registry_id"],
        SIGNER_REGISTRY_PREFIX_V2,
        "time dependency signer registry ID",
    )
    attempt = _validate_identifier(
        value["attempt_context_id"], "time dependency attempt context ID"
    )
    if (profile, registry, attempt) != expected_profile_registry_attempt:
        _fail_exact(FAIL_BINDING_V1, "logical-time dependency graph forks")
    _validate_prefixed_id(
        value["signer_key_id"], ROLE_KEY_PREFIX_V1, "logical-time key ID"
    )
    if _validate_integer(
        value["signer_key_epoch"], "logical-time key epoch", minimum=1
    ) != LOGICAL_TIME_KEY_EPOCH_V1:
        _fail_exact(FAIL_BINDING_V1, "logical-time key epoch differs")
    _validate_prefixed_id(
        value["token_id"], LOGICAL_TIME_TOKEN_ID_PREFIX_V1, "logical-time token ID"
    )
    _validate_sha256(value["token_canonical_sha256"], "logical-time token SHA-256")
    _validate_artifact_id(
        value["token_bound_artifact_id"], "logical-time bound artifact ID"
    )
    _validate_sha256(
        value["token_bound_artifact_sha256"],
        "logical-time bound artifact SHA-256",
    )
    _validate_integer(value["token_sequence"], "logical-time token sequence")
    return value


def _event_unsigned_body(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "signature_purpose": body["signature_purpose"],
        "signature_domain": body["signature_domain"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "governance_key_id": body["governance_key_id"],
        "governance_key_epoch": body["governance_key_epoch"],
        "governance_public_key_hex": body["governance_public_key_hex"],
        "attempt_context_id": body["attempt_context_id"],
        "decision": body["decision"],
        "decision_canonical_sha256": body["decision_canonical_sha256"],
        "cas_store_dependency": body["cas_store_dependency"],
        "logical_time_dependency": body["logical_time_dependency"],
        "algorithm": body["algorithm"],
        "claim_boundary": body["claim_boundary"],
    }


def _event_id_subject(
    body: Mapping[str, object],
    _unsigned=_event_unsigned_body,
) -> dict[str, object]:
    return {**_unsigned(body), "signature_hex": body["signature_hex"]}


def _event_signature_preimage(
    body: Mapping[str, object],
    _unsigned=_event_unsigned_body,
    _canonical_exact=_canonical,
    _signature_purpose=SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
) -> bytes:
    wrapper = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
        "artifact_kind": "owner_authorization_status",
        "role": GOVERNANCE_ROLE_V1,
        "purpose": _signature_purpose,
        "key_epoch": body["governance_key_epoch"],
        "signer_key_id": body["governance_key_id"],
        "body": _unsigned(body),
    }
    return _GOVERNANCE_SIGNATURE_DOMAIN + _canonical_exact(wrapper)


def _verify_governance_signature(
    event: Mapping[str, object],
    governance_public_key: bytes,
    *,
    _derive_key_id=_derive_governance_key_id,
    _preimage=_event_signature_preimage,
    _public_key_type=_Ed25519PublicKey,
    _invalid_signature_type=_InvalidSignature,
    _error_type=Phase2BStep5OwnerStatusReplayCoreErrorV1,
    _fail_exact=_fail,
) -> None:
    if event["governance_public_key_hex"] != governance_public_key.hex():
        _fail_exact(FAIL_BINDING_V1, "event governance public key differs")
    expected_key_id = _derive_key_id(
        governance_public_key,
        key_epoch=event["governance_key_epoch"],  # type: ignore[arg-type]
    )
    if event["governance_key_id"] != expected_key_id:
        _fail_exact(FAIL_BINDING_V1, "event governance role-key ID differs")
    if _public_key_type is None or _invalid_signature_type is None:
        _fail_exact(
            FAIL_SIGNATURE_BACKEND_V1,
            "cryptography Ed25519 verification backend is unavailable",
        )
    try:
        verifier = _public_key_type.from_public_bytes(governance_public_key)
        verifier.verify(
            bytes.fromhex(event["signature_hex"]),  # type: ignore[arg-type]
            _preimage(event),
        )
    except _invalid_signature_type as exc:
        raise _error_type(
            FAIL_SIGNATURE_V1, "governance owner-status signature rejected"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _error_type(
            FAIL_SIGNATURE_V1, "governance Ed25519 public key rejected"
        ) from exc


def _validate_event_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    decision_payload: bytes,
    governance_public_key: bytes,
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common_exact=_require_common,
    _validate_decision=_validate_decision_mapping,
    _validate_cas=_validate_cas_dependency_mapping,
    _validate_time=_validate_logical_time_dependency_mapping,
    _validate_hex32=_hex32,
    _validate_hex64=_hex64,
    _validate_sha256=_sha256_address,
    _canonical_exact=_canonical,
    _sha256_exact=_sha256_bytes,
    _id_subject=_event_id_subject,
    _content_address=_content_id,
    _verify_signature_exact=_verify_governance_signature,
    _event_schema=SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1,
    _signature_purpose=SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
    _signature_domain=GOVERNANCE_SIGNATURE_DOMAIN_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _EVENT_KEYS, "signed owner-status event")
    literals = (
        ("schema_version", _event_schema),
        ("signature_purpose", _signature_purpose),
        ("signature_domain", _signature_domain),
        ("algorithm", ED25519_ALGORITHM_V1),
        ("claim_boundary", EVENT_CLAIM_BOUNDARY_V1),
    )
    if any(
        type(value[field]) is not str or value[field] != expected
        for field, expected in literals
    ):
        _fail_exact(FAIL_SCHEMA_V1, "signed owner-status event literal differs")
    _require_common_exact(_common_projection(value), expected_common, "signed event")
    _validate_hex32(value["governance_public_key_hex"], "governance public key")
    _validate_hex64(value["signature_hex"], "governance signature")
    _validate_sha256(value["decision_canonical_sha256"], "decision SHA-256")
    decision = _validate_decision(value["decision"], expected_common)
    if (
        _canonical_exact(decision) != decision_payload
        or value["decision_canonical_sha256"] != _sha256_exact(decision_payload)
    ):
        _fail_exact(
            FAIL_BINDING_V1,
            "event embedded decision bytes differ from separate decision bytes",
        )
    graph = (expected_common[0], expected_common[1], expected_common[4])
    cas = _validate_cas(value["cas_store_dependency"], graph)
    logical_time = _validate_time(value["logical_time_dependency"], graph)
    decision_sha = _sha256_exact(decision_payload)
    if not (
        cas["bound_artifact_id"] == decision["decision_id"]
        and cas["bound_artifact_sha256"] == decision_sha
        and cas["object_id"] == decision_sha
        and cas["object_canonical_sha256"] == decision_sha
        and logical_time["token_bound_artifact_id"] == decision["decision_id"]
        and logical_time["token_bound_artifact_sha256"] == decision_sha
        and logical_time["token_sequence"] == decision["sequence"]
    ):
        _fail_exact(FAIL_BINDING_V1, "event dependencies do not bind exact decision")
    expected_id = _content_address(
        EVENT_ID_PREFIX_V1, _EVENT_ID_DOMAIN, _id_subject(value)
    )
    if value["event_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "signed owner-status event ID differs")
    _verify_signature_exact(value, governance_public_key)
    return value


def _parse_event(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    decision_payload: bytes,
    governance_public_key: bytes,
    _decode_exact=_decode,
    _validate_mapping=_validate_event_mapping,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, MAXIMUM_EVENT_BYTES, "signed owner-status event"),
        expected_common,
        decision_payload,
        governance_public_key,
    )


def _next_complete_event_set_root(
    previous_root: str,
    event: Mapping[str, object],
    event_payload: bytes,
    *,
    _validate_sha256=_sha256_address,
    _canonical_exact=_canonical,
    _sha256_exact=_sha256_bytes,
) -> str:
    _validate_sha256(previous_root, "previous complete-event-set root")
    subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-owner-status-complete-event-set-root/1"
        ),
        "previous_complete_event_set_root": previous_root,
        "sequence": event["decision"]["sequence"],  # type: ignore[index]
        "event_id": event["event_id"],
        "event_canonical_sha256": _sha256_exact(event_payload),
    }
    return _sha256_exact(_COMPLETE_EVENT_SET_ROOT_DOMAIN + _canonical_exact(subject))


def _checkpoint_id_subject(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "purpose": body["purpose"],
        "ledger_id": body["ledger_id"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "governance_key_id": body["governance_key_id"],
        "governance_key_epoch": body["governance_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "checkpoint_index": body["checkpoint_index"],
        "previous_head_id": body["previous_head_id"],
        "previous_checkpoint_id": body["previous_checkpoint_id"],
        "previous_event_id": body["previous_event_id"],
        "previous_complete_event_set_root": (
            body["previous_complete_event_set_root"]
        ),
        "event_id": body["event_id"],
        "signed_event_canonical_sha256": body["signed_event_canonical_sha256"],
        "signed_event": body["signed_event"],
        "complete_event_set_root": body["complete_event_set_root"],
    }


def _validate_checkpoint_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    previous_head: Mapping[str, object],
    event: Mapping[str, object],
    event_payload: bytes,
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common_exact=_require_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_sha256=_sha256_address,
    _ledger_identity=_ledger_id,
    _canonical_exact=_canonical,
    _mapping_equal_exact=_mapping_equal,
    _sha256_exact=_sha256_bytes,
    _next_root=_next_complete_event_set_root,
    _id_subject=_checkpoint_id_subject,
    _content_address=_content_id,
    _checkpoint_schema=OWNER_STATUS_CHECKPOINT_SCHEMA_V1,
    _checkpoint_purpose=CHECKPOINT_PURPOSE_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _CHECKPOINT_KEYS, "owner-status checkpoint")
    if not (
        value["schema_version"] == _checkpoint_schema
        and value["purpose"] == _checkpoint_purpose
    ):
        _fail_exact(FAIL_SCHEMA_V1, "owner-status checkpoint literal differs")
    _require_common_exact(
        _common_projection(value), expected_common, "owner-status checkpoint"
    )
    _validate_prefixed_id(value["ledger_id"], LEDGER_ID_PREFIX_V1, "ledger ID")
    if value["ledger_id"] != _ledger_identity(expected_common):
        _fail_exact(FAIL_BINDING_V1, "checkpoint ledger identity differs")
    checkpoint_index = _validate_integer(
        value["checkpoint_index"], "checkpoint index", maximum=1
    )
    _validate_prefixed_id(
        value["previous_head_id"], HEAD_ID_PREFIX_V1, "checkpoint previous head ID"
    )
    _validate_prefixed_id(
        value["previous_checkpoint_id"],
        CHECKPOINT_ID_PREFIX_V1,
        "checkpoint previous checkpoint ID",
    )
    _validate_prefixed_id(
        value["previous_event_id"],
        EVENT_ID_PREFIX_V1,
        "checkpoint previous event ID",
    )
    _validate_sha256(
        value["previous_complete_event_set_root"],
        "checkpoint previous complete-event-set root",
    )
    _validate_prefixed_id(value["event_id"], EVENT_ID_PREFIX_V1, "event ID")
    _validate_sha256(
        value["signed_event_canonical_sha256"], "signed-event SHA-256"
    )
    _validate_sha256(value["complete_event_set_root"], "complete-event-set root")
    if type(value["signed_event"]) is not dict:
        _fail_exact(FAIL_SCHEMA_V1, "checkpoint embedded event must be object")
    if (
        _canonical_exact(value["signed_event"]) != event_payload
        or not _mapping_equal_exact(value["signed_event"], event)  # type: ignore[arg-type]
    ):
        _fail_exact(
            FAIL_BINDING_V1,
            "checkpoint embedded event bytes differ from separate event bytes",
        )
    decision = event["decision"]
    if type(decision) is not dict:
        _fail_exact(FAIL_SCHEMA_V1, "event decision must be exact object")
    expected_root = _next_root(
        previous_head["complete_event_set_root"],  # type: ignore[arg-type]
        event,
        event_payload,
    )
    if not (
        checkpoint_index == decision["sequence"]
        and value["previous_head_id"] == previous_head["head_id"]
        and value["previous_checkpoint_id"] == previous_head["last_checkpoint_id"]
        and value["previous_event_id"] == previous_head["last_event_id"]
        and value["previous_complete_event_set_root"]
        == previous_head["complete_event_set_root"]
        and decision["previous_head_id"] == previous_head["head_id"]
        and decision["previous_checkpoint_id"]
        == previous_head["last_checkpoint_id"]
        and decision["previous_event_id"] == previous_head["last_event_id"]
        and decision["previous_complete_event_set_root"]
        == previous_head["complete_event_set_root"]
        and value["event_id"] == event["event_id"]
        and value["signed_event_canonical_sha256"] == _sha256_exact(event_payload)
        and value["complete_event_set_root"] == expected_root
    ):
        _fail_exact(FAIL_CHAIN_V1, "checkpoint/event/previous-head linkage differs")
    expected_id = _content_address(
        CHECKPOINT_ID_PREFIX_V1,
        _CHECKPOINT_ID_DOMAIN,
        _id_subject(value),
    )
    if value["checkpoint_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "owner-status checkpoint ID differs")
    return value


def _parse_checkpoint(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    previous_head: Mapping[str, object],
    event: Mapping[str, object],
    event_payload: bytes,
    _decode_exact=_decode,
    _validate_mapping=_validate_checkpoint_mapping,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, MAXIMUM_CHECKPOINT_BYTES, "owner-status checkpoint"),
        expected_common,
        previous_head,
        event,
        event_payload,
    )


def _head_id_subject(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "purpose": body["purpose"],
        "ledger_id": body["ledger_id"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "governance_key_id": body["governance_key_id"],
        "governance_key_epoch": body["governance_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "event_count": body["event_count"],
        "next_sequence": body["next_sequence"],
        "last_event_id": body["last_event_id"],
        "last_checkpoint_id": body["last_checkpoint_id"],
        "complete_event_set_root": body["complete_event_set_root"],
        "latest_effective_status": body["latest_effective_status"],
        "revocation_count": body["revocation_count"],
        "terminal": body["terminal"],
    }


def _expected_head_mapping(
    *,
    common: tuple[str, str, str, int, str],
    event_count: int,
    last_event_id: str,
    last_checkpoint_id: str,
    complete_event_set_root: str,
    latest_effective_status: str,
    revocation_count: int,
    terminal: bool,
    _content_address=_content_id,
    _ledger_identity=_ledger_id,
    _head_schema=OWNER_STATUS_HEAD_SCHEMA_V1,
    _head_purpose=HEAD_PURPOSE_V1,
) -> dict[str, object]:
    subject = {
        "schema_version": _head_schema,
        "purpose": _head_purpose,
        "ledger_id": _ledger_identity(common),
        "authority_profile_id": common[0],
        "signer_registry_id": common[1],
        "governance_key_id": common[2],
        "governance_key_epoch": common[3],
        "attempt_context_id": common[4],
        "event_count": event_count,
        "next_sequence": event_count,
        "last_event_id": last_event_id,
        "last_checkpoint_id": last_checkpoint_id,
        "complete_event_set_root": complete_event_set_root,
        "latest_effective_status": latest_effective_status,
        "revocation_count": revocation_count,
        "terminal": terminal,
    }
    return {
        **subject,
        "head_id": _content_address(HEAD_ID_PREFIX_V1, _HEAD_ID_DOMAIN, subject),
    }


def _validate_head_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common_exact=_require_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_sha256=_sha256_address,
    _ledger_identity=_ledger_id,
    _id_subject=_head_id_subject,
    _content_address=_content_id,
    _head_schema=OWNER_STATUS_HEAD_SCHEMA_V1,
    _head_purpose=HEAD_PURPOSE_V1,
    _status_none=STATUS_NONE_V1,
    _status_active=STATUS_ACTIVE_V1,
    _status_revoked=STATUS_REVOKED_V1,
    _genesis_event_id=GENESIS_EVENT_ID_V1,
    _genesis_checkpoint_id=GENESIS_CHECKPOINT_ID_V1,
    _genesis_root=GENESIS_COMPLETE_EVENT_SET_ROOT_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _HEAD_KEYS, "owner-status head")
    if not (
        value["schema_version"] == _head_schema
        and value["purpose"] == _head_purpose
    ):
        _fail_exact(FAIL_SCHEMA_V1, "owner-status head literal differs")
    _require_common_exact(_common_projection(value), expected_common, "owner-status head")
    _validate_prefixed_id(value["ledger_id"], LEDGER_ID_PREFIX_V1, "head ledger ID")
    if value["ledger_id"] != _ledger_identity(expected_common):
        _fail_exact(FAIL_BINDING_V1, "head ledger identity differs")
    count = _validate_integer(value["event_count"], "head event count", maximum=2)
    next_sequence = _validate_integer(
        value["next_sequence"], "head next sequence", maximum=2
    )
    if count != next_sequence:
        _fail_exact(FAIL_CHAIN_V1, "head count and next sequence differ")
    _validate_prefixed_id(value["last_event_id"], EVENT_ID_PREFIX_V1, "last event ID")
    _validate_prefixed_id(
        value["last_checkpoint_id"], CHECKPOINT_ID_PREFIX_V1, "last checkpoint ID"
    )
    _validate_sha256(value["complete_event_set_root"], "complete-event-set root")
    revocations = _validate_integer(
        value["revocation_count"], "head revocation count", maximum=1
    )
    if type(value["terminal"]) is not bool:
        _fail_exact(FAIL_SCHEMA_V1, "head terminal must be exact bool")
    if count == 0:
        expected_state = (
            _genesis_event_id,
            _genesis_checkpoint_id,
            _genesis_root,
            _status_none,
            0,
            False,
        )
    elif count == 1:
        expected_state = (
            value["last_event_id"],
            value["last_checkpoint_id"],
            value["complete_event_set_root"],
            _status_active,
            0,
            False,
        )
        if (
            value["last_event_id"] == _genesis_event_id
            or value["last_checkpoint_id"] == _genesis_checkpoint_id
            or value["complete_event_set_root"] == _genesis_root
        ):
            _fail_exact(FAIL_CHAIN_V1, "ACTIVE head retains genesis sentinel")
    else:
        expected_state = (
            value["last_event_id"],
            value["last_checkpoint_id"],
            value["complete_event_set_root"],
            _status_revoked,
            1,
            True,
        )
        if (
            value["last_event_id"] == _genesis_event_id
            or value["last_checkpoint_id"] == _genesis_checkpoint_id
            or value["complete_event_set_root"] == _genesis_root
        ):
            _fail_exact(FAIL_CHAIN_V1, "REVOKED head retains genesis sentinel")
    observed_state = (
        value["last_event_id"],
        value["last_checkpoint_id"],
        value["complete_event_set_root"],
        value["latest_effective_status"],
        revocations,
        value["terminal"],
    )
    if observed_state != expected_state:
        _fail_exact(FAIL_CHAIN_V1, "owner-status head state differs")
    expected_id = _content_address(
        HEAD_ID_PREFIX_V1, _HEAD_ID_DOMAIN, _id_subject(value)
    )
    if value["head_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "owner-status head ID differs")
    return value


def _parse_head(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    _decode_exact=_decode,
    _validate_mapping=_validate_head_mapping,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, MAXIMUM_HEAD_BYTES, "owner-status head"),
        expected_common,
    )


def _projection_id_subject(body: Mapping[str, object]) -> dict[str, object]:
    return {key: body[key] for key in _PROJECTION_KEYS if key != "projection_id"}


def _expected_projection_mapping(
    final_head: Mapping[str, object],
    *,
    _content_address=_content_id,
    _projection_schema=OWNER_STATUS_PROJECTION_SCHEMA_V1,
    _projection_claim=PROJECTION_CLAIM_LEVEL_V1,
) -> dict[str, object]:
    subject = {
        "schema_version": _projection_schema,
        "claim_level": _projection_claim,
        "ledger_id": final_head["ledger_id"],
        "authority_profile_id": final_head["authority_profile_id"],
        "signer_registry_id": final_head["signer_registry_id"],
        "governance_key_id": final_head["governance_key_id"],
        "governance_key_epoch": final_head["governance_key_epoch"],
        "attempt_context_id": final_head["attempt_context_id"],
        "owner_authorization_binding": dict(EXACT_OWNER_AUTHORIZATION_BINDING_V1),
        "owner_authorization_payload_sha256": (
            "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
        ),
        "event_count": final_head["event_count"],
        "latest_sequence": final_head["event_count"] - 1,  # type: ignore[operator]
        "latest_effective_status": final_head["latest_effective_status"],
        "revocation_count": final_head["revocation_count"],
        "last_event_id": final_head["last_event_id"],
        "last_checkpoint_id": final_head["last_checkpoint_id"],
        "head_id": final_head["head_id"],
        "complete_event_set_root": final_head["complete_event_set_root"],
        "governance_signatures_verified": True,
        "governance_status_purpose_authorized_by_pinned_registry": False,
        "owner_signature_authenticated": False,
        "logical_time_token_signatures_verified": True,
        "store_dependency_candidate_references_exact": True,
        "store_dependencies_verified": False,
        "latest_authority_status_verified": False,
        "r01_satisfied": False,
        "readiness_node_transition_issued": False,
        "execution_authorized": False,
        "authority_evidence": False,
    }
    return {
        **subject,
        "projection_id": _content_address(
            PROJECTION_ID_PREFIX_V1, _PROJECTION_ID_DOMAIN, subject
        ),
    }


def _validate_projection_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    final_head: Mapping[str, object],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common_exact=_require_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_sha256=_sha256_address,
    _validate_owner=_validate_owner_binding,
    _ledger_identity=_ledger_id,
    _expected_projection=_expected_projection_mapping,
    _mapping_equal_exact=_mapping_equal,
    _id_subject=_projection_id_subject,
    _content_address=_content_id,
    _projection_schema=OWNER_STATUS_PROJECTION_SCHEMA_V1,
    _projection_claim=PROJECTION_CLAIM_LEVEL_V1,
    _status_active=STATUS_ACTIVE_V1,
    _status_revoked=STATUS_REVOKED_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _PROJECTION_KEYS, "owner-status projection")
    if not (
        value["schema_version"] == _projection_schema
        and value["claim_level"] == _projection_claim
    ):
        _fail_exact(FAIL_SCHEMA_V1, "owner-status projection literal differs")
    _require_common_exact(
        _common_projection(value), expected_common, "owner-status projection"
    )
    _validate_prefixed_id(value["ledger_id"], LEDGER_ID_PREFIX_V1, "ledger ID")
    if value["ledger_id"] != _ledger_identity(expected_common):
        _fail_exact(FAIL_BINDING_V1, "projection ledger identity differs")
    _validate_owner(value["owner_authorization_binding"])
    if value["owner_authorization_payload_sha256"] != (
        "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
    ):
        _fail_exact(FAIL_BINDING_V1, "projection owner payload SHA differs")
    count = _validate_integer(
        value["event_count"], "projection event count", minimum=1, maximum=2
    )
    sequence = _validate_integer(
        value["latest_sequence"], "projection latest sequence", maximum=1
    )
    revocations = _validate_integer(
        value["revocation_count"], "projection revocation count", maximum=1
    )
    if not (
        sequence == count - 1
        and value["latest_effective_status"]
        == (_status_active if count == 1 else _status_revoked)
        and revocations == (0 if count == 1 else 1)
    ):
        _fail_exact(FAIL_CHAIN_V1, "projection status/count differs")
    _validate_prefixed_id(value["last_event_id"], EVENT_ID_PREFIX_V1, "last event ID")
    _validate_prefixed_id(
        value["last_checkpoint_id"], CHECKPOINT_ID_PREFIX_V1, "last checkpoint ID"
    )
    _validate_prefixed_id(value["head_id"], HEAD_ID_PREFIX_V1, "projection head ID")
    _validate_sha256(value["complete_event_set_root"], "projection event-set root")
    boolean_fields = (
        "governance_signatures_verified",
        "governance_status_purpose_authorized_by_pinned_registry",
        "owner_signature_authenticated",
        "logical_time_token_signatures_verified",
        "store_dependency_candidate_references_exact",
        "store_dependencies_verified",
        "latest_authority_status_verified",
        "r01_satisfied",
        "readiness_node_transition_issued",
        "execution_authorized",
        "authority_evidence",
    )
    if any(type(value[field]) is not bool for field in boolean_fields):
        _fail_exact(FAIL_SCHEMA_V1, "projection boolean field type differs")
    expected_flags = (
        True, False, False, True, True, False, False, False, False, False, False
    )
    if tuple(value[field] for field in boolean_fields) != expected_flags:
        _fail_exact(FAIL_CLAIM_BOUNDARY_V1, "projection safety flags differ")
    expected_id = _content_address(
        PROJECTION_ID_PREFIX_V1, _PROJECTION_ID_DOMAIN, _id_subject(value)
    )
    if value["projection_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "projection ID differs")
    if not _mapping_equal_exact(value, _expected_projection(final_head)):
        _fail_exact(
            FAIL_CHAIN_V1,
            "projection does not exact-project the caller-supplied current head",
        )
    return value


def _parse_projection(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    final_head: Mapping[str, object],
    _decode_exact=_decode,
    _validate_mapping=_validate_projection_mapping,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, MAXIMUM_PROJECTION_BYTES, "owner-status projection"),
        expected_common,
        final_head,
    )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedOwnerStatusReplayCoreV1:
    """Privately issued supplied-chain mechanics result; never authority."""

    schema_version: str
    claim_boundary: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    governance_key_id: str
    governance_key_epoch: int
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    owner_authorization_id: str
    owner_authorization_payload_sha256: str
    ledger_id: str
    verified_event_count: int
    verified_governance_signature_count: int
    verified_logical_time_signature_count: int
    decision_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    head_ids: tuple[str, ...]
    logical_time_request_ids: tuple[str, ...]
    logical_time_token_ids: tuple[str, ...]
    legacy_cas_receipt_ids: tuple[str, ...]
    current_head_id: str
    current_head_payload_sha256: str
    complete_event_set_root: str
    latest_effective_status: str
    revocation_count: int
    projection_id: str
    projection_payload_sha256: str
    logical_time_ledger_id: str
    logical_time_final_head_id: str
    authority_graph_signature_backend: str
    raw_authority_v2_graph_verified: bool
    governance_two_purpose_policy_verified: bool
    governance_static_service_binding_verified: bool
    governance_signatures_verified: bool
    logical_time_supplied_chain_verified: bool
    owner_status_supplied_chain_verified: bool
    owner_status_projection_verified: bool
    caller_supplied_current_head_byte_equal: bool
    legacy_cas_candidate_references_exact: bool
    legacy_cas_receipts_authenticated: bool
    external_cas_checkpoint_verified: bool
    external_currentness_verified: bool
    external_anti_fork_anchor_verified: bool
    runtime_closure_verified: bool
    owner_signature_authenticated: bool
    authority_evidence: bool
    r01_satisfied: bool
    readiness_evidence: bool
    execution_authorized: bool
    verifier_only: bool
    private_key_present: bool
    issuer_path_present: bool
    filesystem_observed: bool
    service_observed: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "VerifiedOwnerStatusReplayCoreV1":
        raise TypeError("verified owner-status replay results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified owner-status replay results are privately issued")


def _build_verified_result_lineage_v1():
    result_type = VerifiedOwnerStatusReplayCoreV1
    fail_exact = _fail
    validate_prefixed_id = _prefixed_id
    validate_identifier = _identifier
    validate_integer = _wire_integer
    validate_sha256 = _sha256_address
    object_new = object.__new__
    object_setattr = object.__setattr__
    schema = SCHEMA_VERSION
    claim_boundary = AUTHORITY_CLAIM_BOUNDARY_V1
    backend = "CRYPTOGRAPHY_ED25519"
    status_active = STATUS_ACTIVE_V1
    status_revoked = STATUS_REVOKED_V1
    issued: dict[
        int,
        tuple[
            VerifiedOwnerStatusReplayCoreV1,
            tuple[object, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ],
    ] = {}
    tuple_fields = (
        "decision_ids",
        "event_ids",
        "checkpoint_ids",
        "head_ids",
        "logical_time_request_ids",
        "logical_time_token_ids",
        "legacy_cas_receipt_ids",
    )
    scalar_fields = (
        "schema_version",
        "claim_boundary",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "owner_authorization_id",
        "owner_authorization_payload_sha256",
        "ledger_id",
        "verified_event_count",
        "verified_governance_signature_count",
        "verified_logical_time_signature_count",
        "current_head_id",
        "current_head_payload_sha256",
        "complete_event_set_root",
        "latest_effective_status",
        "revocation_count",
        "projection_id",
        "projection_payload_sha256",
        "logical_time_ledger_id",
        "logical_time_final_head_id",
        "authority_graph_signature_backend",
        "raw_authority_v2_graph_verified",
        "governance_two_purpose_policy_verified",
        "governance_static_service_binding_verified",
        "governance_signatures_verified",
        "logical_time_supplied_chain_verified",
        "owner_status_supplied_chain_verified",
        "owner_status_projection_verified",
        "caller_supplied_current_head_byte_equal",
        "legacy_cas_candidate_references_exact",
        "legacy_cas_receipts_authenticated",
        "external_cas_checkpoint_verified",
        "external_currentness_verified",
        "external_anti_fork_anchor_verified",
        "runtime_closure_verified",
        "owner_signature_authenticated",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
        "verifier_only",
        "private_key_present",
        "issuer_path_present",
        "filesystem_observed",
        "service_observed",
    )

    def exact_scalar_snapshot(
        observed: tuple[object, ...], expected: tuple[object, ...]
    ) -> bool:
        return (
            type(observed) is tuple
            and type(expected) is tuple
            and len(observed) == len(expected)
            and all(
                type(left) is type(right) and left == right
                for left, right in zip(observed, expected, strict=True)
            )
        )

    def require(value: object) -> VerifiedOwnerStatusReplayCoreV1:
        if type(value) is not result_type:
            fail_exact(FAIL_RESULT_V1, "verified owner-status result type differs")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            fail_exact(
                FAIL_RESULT_V1,
                "verified owner-status result lacks private issuance lineage",
            )
        expected_scalars = record[1]
        expected_tuples = record[2:]
        try:
            observed_scalars = tuple(
                object.__getattribute__(value, field) for field in scalar_fields
            )
            intact = exact_scalar_snapshot(observed_scalars, expected_scalars)
            intact = intact and all(
                object.__getattribute__(value, field) is expected_tuple
                for field, expected_tuple in zip(
                    tuple_fields, expected_tuples, strict=True
                )
            )
        except (AttributeError, TypeError):
            intact = False
        if not intact:
            fail_exact(FAIL_RESULT_V1, "verified owner-status result was mutated")

        count = validate_integer(value.verified_event_count, "event count", minimum=1)
        if count not in (1, 2):
            fail_exact(FAIL_RESULT_V1, "verified event count is not one or two")
        expected_status = status_active if count == 1 else status_revoked
        expected_revocations = 0 if count == 1 else 1
        if not (
            value.schema_version == schema
            and value.claim_boundary == claim_boundary
            and validate_prefixed_id(
                value.stage_b_trust_lock_id,
                STAGE_B_TRUST_LOCK_PREFIX_V2,
                "Stage-B lock ID",
            )
            == value.stage_b_trust_lock_id
            and validate_prefixed_id(
                value.authority_profile_id,
                AUTHORITY_PROFILE_PREFIX_V2,
                "authority profile ID",
            )
            == value.authority_profile_id
            and validate_prefixed_id(
                value.signer_registry_id,
                SIGNER_REGISTRY_PREFIX_V2,
                "signer registry ID",
            )
            == value.signer_registry_id
            and validate_prefixed_id(
                value.governance_key_id, ROLE_KEY_PREFIX_V1, "governance key ID"
            )
            == value.governance_key_id
            and value.governance_key_epoch == 1
            and validate_prefixed_id(
                value.logical_time_key_id,
                ROLE_KEY_PREFIX_V1,
                "logical-time key ID",
            )
            == value.logical_time_key_id
            and value.logical_time_key_epoch == 1
            and validate_identifier(value.attempt_context_id, "attempt context ID")
            == value.attempt_context_id
            and value.owner_authorization_id == OWNER_AUTHORIZATION_ID_V1
            and value.owner_authorization_payload_sha256
            == "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
            and validate_prefixed_id(value.ledger_id, LEDGER_ID_PREFIX_V1, "ledger ID")
            == value.ledger_id
            and value.verified_governance_signature_count == count
            and value.verified_logical_time_signature_count == count
            and type(value.decision_ids) is tuple
            and len(value.decision_ids) == count
            and type(value.event_ids) is tuple
            and len(value.event_ids) == count
            and type(value.checkpoint_ids) is tuple
            and len(value.checkpoint_ids) == count
            and type(value.head_ids) is tuple
            and len(value.head_ids) == count + 1
            and type(value.logical_time_request_ids) is tuple
            and len(value.logical_time_request_ids) == count
            and type(value.logical_time_token_ids) is tuple
            and len(value.logical_time_token_ids) == count
            and type(value.legacy_cas_receipt_ids) is tuple
            and len(value.legacy_cas_receipt_ids) == count
            and value.current_head_id == value.head_ids[-1]
            and validate_sha256(
                value.current_head_payload_sha256, "current head payload SHA-256"
            )
            == value.current_head_payload_sha256
            and validate_sha256(
                value.complete_event_set_root, "complete-event-set root"
            )
            == value.complete_event_set_root
            and value.latest_effective_status == expected_status
            and value.revocation_count == expected_revocations
            and validate_prefixed_id(
                value.projection_id, PROJECTION_ID_PREFIX_V1, "projection ID"
            )
            == value.projection_id
            and validate_sha256(
                value.projection_payload_sha256, "projection payload SHA-256"
            )
            == value.projection_payload_sha256
            and validate_identifier(value.logical_time_ledger_id, "time ledger ID")
            == value.logical_time_ledger_id
            and validate_identifier(
                value.logical_time_final_head_id, "time final head ID"
            )
            == value.logical_time_final_head_id
            and value.authority_graph_signature_backend == backend
            and value.raw_authority_v2_graph_verified is True
            and value.governance_two_purpose_policy_verified is True
            and value.governance_static_service_binding_verified is True
            and value.governance_signatures_verified is True
            and value.logical_time_supplied_chain_verified is True
            and value.owner_status_supplied_chain_verified is True
            and value.owner_status_projection_verified is True
            and value.caller_supplied_current_head_byte_equal is True
            and value.legacy_cas_candidate_references_exact is True
            and value.legacy_cas_receipts_authenticated is False
            and value.external_cas_checkpoint_verified is False
            and value.external_currentness_verified is False
            and value.external_anti_fork_anchor_verified is False
            and value.runtime_closure_verified is False
            and value.owner_signature_authenticated is False
            and value.authority_evidence is False
            and value.r01_satisfied is False
            and value.readiness_evidence is False
            and value.execution_authorized is False
            and value.verifier_only is True
            and value.private_key_present is False
            and value.issuer_path_present is False
            and value.filesystem_observed is False
            and value.service_observed is False
        ):
            fail_exact(FAIL_RESULT_V1, "verified owner-status result semantics differ")
        for item in value.decision_ids:
            validate_prefixed_id(item, DECISION_ID_PREFIX_V1, "decision ID")
        for item in value.event_ids:
            validate_prefixed_id(item, EVENT_ID_PREFIX_V1, "event ID")
        for item in value.checkpoint_ids:
            validate_prefixed_id(item, CHECKPOINT_ID_PREFIX_V1, "checkpoint ID")
        for item in value.head_ids:
            validate_prefixed_id(item, HEAD_ID_PREFIX_V1, "head ID")
        for item in value.logical_time_request_ids:
            validate_identifier(item, "logical-time request ID")
        for item in value.logical_time_token_ids:
            validate_prefixed_id(
                item, LOGICAL_TIME_TOKEN_ID_PREFIX_V1, "logical-time token ID"
            )
        for item in value.legacy_cas_receipt_ids:
            validate_sha256(item, "legacy CAS receipt candidate ID")
        return value

    def issue(
        *,
        lock_id: str,
        common: tuple[str, str, str, int, str],
        logical_time_key_id: str,
        logical_time_key_epoch: int,
        ledger_id: str,
        decision_ids: tuple[str, ...],
        event_ids: tuple[str, ...],
        checkpoint_ids: tuple[str, ...],
        head_ids: tuple[str, ...],
        logical_time_request_ids: tuple[str, ...],
        logical_time_token_ids: tuple[str, ...],
        legacy_cas_receipt_ids: tuple[str, ...],
        current_head_payload_sha256: str,
        complete_event_set_root: str,
        latest_effective_status: str,
        revocation_count: int,
        projection_id: str,
        projection_payload_sha256: str,
        logical_time_ledger_id: str,
        logical_time_final_head_id: str,
    ) -> VerifiedOwnerStatusReplayCoreV1:
        if (
            type(common) is not tuple
            or len(common) != 5
            or any(
                type(value) is not tuple
                for value in (
                    decision_ids,
                    event_ids,
                    checkpoint_ids,
                    head_ids,
                    logical_time_request_ids,
                    logical_time_token_ids,
                    legacy_cas_receipt_ids,
                )
            )
        ):
            fail_exact(FAIL_RESULT_V1, "private owner-status issuance shape differs")
        count = len(decision_ids)
        if not (
            count in (1, 2)
            and len(event_ids) == len(checkpoint_ids) == count
            and len(head_ids) == count + 1
            and len(logical_time_request_ids)
            == len(logical_time_token_ids)
            == len(legacy_cas_receipt_ids)
            == count
        ):
            fail_exact(FAIL_RESULT_V1, "private owner-status issuance counts differ")
        values: dict[str, object] = {
            "schema_version": schema,
            "claim_boundary": claim_boundary,
            "stage_b_trust_lock_id": lock_id,
            "authority_profile_id": common[0],
            "signer_registry_id": common[1],
            "governance_key_id": common[2],
            "governance_key_epoch": common[3],
            "logical_time_key_id": logical_time_key_id,
            "logical_time_key_epoch": logical_time_key_epoch,
            "attempt_context_id": common[4],
            "owner_authorization_id": OWNER_AUTHORIZATION_ID_V1,
            "owner_authorization_payload_sha256": (
                "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
            ),
            "ledger_id": ledger_id,
            "verified_event_count": count,
            "verified_governance_signature_count": count,
            "verified_logical_time_signature_count": count,
            "decision_ids": decision_ids,
            "event_ids": event_ids,
            "checkpoint_ids": checkpoint_ids,
            "head_ids": head_ids,
            "logical_time_request_ids": logical_time_request_ids,
            "logical_time_token_ids": logical_time_token_ids,
            "legacy_cas_receipt_ids": legacy_cas_receipt_ids,
            "current_head_id": head_ids[-1],
            "current_head_payload_sha256": current_head_payload_sha256,
            "complete_event_set_root": complete_event_set_root,
            "latest_effective_status": latest_effective_status,
            "revocation_count": revocation_count,
            "projection_id": projection_id,
            "projection_payload_sha256": projection_payload_sha256,
            "logical_time_ledger_id": logical_time_ledger_id,
            "logical_time_final_head_id": logical_time_final_head_id,
            "authority_graph_signature_backend": backend,
            "raw_authority_v2_graph_verified": True,
            "governance_two_purpose_policy_verified": True,
            "governance_static_service_binding_verified": True,
            "governance_signatures_verified": True,
            "logical_time_supplied_chain_verified": True,
            "owner_status_supplied_chain_verified": True,
            "owner_status_projection_verified": True,
            "caller_supplied_current_head_byte_equal": True,
            "legacy_cas_candidate_references_exact": True,
            "legacy_cas_receipts_authenticated": False,
            "external_cas_checkpoint_verified": False,
            "external_currentness_verified": False,
            "external_anti_fork_anchor_verified": False,
            "runtime_closure_verified": False,
            "owner_signature_authenticated": False,
            "authority_evidence": False,
            "r01_satisfied": False,
            "readiness_evidence": False,
            "execution_authorized": False,
            "verifier_only": True,
            "private_key_present": False,
            "issuer_path_present": False,
            "filesystem_observed": False,
            "service_observed": False,
        }
        result = object_new(result_type)
        for name, value in values.items():
            object_setattr(result, name, value)
        scalar_snapshot = tuple(values[field] for field in scalar_fields)
        tuple_snapshots = tuple(values[field] for field in tuple_fields)
        issued[id(result)] = (
            result,
            scalar_snapshot,
            *tuple_snapshots,  # type: ignore[arg-type]
        )
        return require(result)

    return issue, require


(
    _hidden_issue_verified_owner_status_replay_core_v1,
    require_verified_owner_status_replay_core_v1,
) = _build_verified_result_lineage_v1()
del _build_verified_result_lineage_v1


def _verify_owner_status_replay_core_implementation_v1(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    owner_authorization_payload: bytes,
    decision_payloads_in_exact_order: tuple[bytes, ...],
    signed_event_payloads_in_exact_order: tuple[bytes, ...],
    owner_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    owner_head_payloads_in_exact_order: tuple[bytes, ...],
    caller_supplied_current_head_payload: bytes,
    owner_projection_payload: bytes,
    logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_signed_token_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
    expected_logical_time_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
    _extract_authority=_extract_governance_authority,
    _validate_owner_payload=_validate_owner_authorization_payload,
    _validate_identifier=_identifier,
    _parse_decision_exact=_parse_decision,
    _parse_event_exact=_parse_event,
    _parse_checkpoint_exact=_parse_checkpoint,
    _parse_head_exact=_parse_head,
    _parse_projection_exact=_parse_projection,
    _ledger_identity=_ledger_id,
    _expected_head=_expected_head_mapping,
    _mapping_equal_exact=_mapping_equal,
    _canonical_exact=_canonical,
    _sha256_exact=_sha256_bytes,
    _verify_time_core=_verify_time_core_v1,
    _require_time_result=_require_time_result_v1,
    _time_event_type=ExpectedLogicalEventCoreV1,
    _time_error_type=_LogicalTimeCoreErrorV1,
    _error_type=Phase2BStep5OwnerStatusReplayCoreErrorV1,
    _issue_result=_hidden_issue_verified_owner_status_replay_core_v1,
    _logical_time_event_purpose=LOGICAL_TIME_EVENT_PURPOSE_V1,
    _status_revoked=STATUS_REVOKED_V1,
    _genesis_event_id=GENESIS_EVENT_ID_V1,
    _genesis_checkpoint_id=GENESIS_CHECKPOINT_ID_V1,
    _genesis_root=GENESIS_COMPLETE_EVENT_SET_ROOT_V1,
    _status_none=STATUS_NONE_V1,
    _fail_exact=_fail,
) -> VerifiedOwnerStatusReplayCoreV1:
    """Replay one complete raw ACTIVE or ACTIVE->REVOKED status chain.

    ``owner_head_payloads_in_exact_order`` contains genesis followed by one
    post-event head per decision.  The separately supplied current-head bytes
    must be byte-for-byte identical to the last element; this is only caller-
    supplied equality and deliberately is not external currentness.
    """

    owner_tuple_inputs = (
        (decision_payloads_in_exact_order, "owner decision payloads"),
        (signed_event_payloads_in_exact_order, "owner signed-event payloads"),
        (owner_checkpoint_payloads_in_exact_order, "owner checkpoint payloads"),
        (owner_head_payloads_in_exact_order, "owner head payloads"),
    )
    time_tuple_inputs = (
        (logical_time_request_payloads_in_exact_order, "logical-time requests"),
        (
            logical_time_signed_token_payloads_in_exact_order,
            "logical-time signed tokens",
        ),
        (
            logical_time_checkpoint_payloads_in_exact_order,
            "logical-time checkpoints",
        ),
        (logical_time_head_payloads_in_exact_order, "logical-time heads"),
    )
    for value, label in owner_tuple_inputs + time_tuple_inputs:
        if type(value) is not tuple:
            _fail_exact(FAIL_INPUT_V1, f"{label} must be an exact tuple")
        if any(type(item) is not bytes for item in value):
            _fail_exact(FAIL_INPUT_V1, f"every {label} item must be exact bytes")
    if type(expected_logical_time_event_plan) is not tuple:
        _fail_exact(FAIL_INPUT_V1, "logical-time event plan must be exact tuple")
    if any(
        type(item) is not _time_event_type
        for item in expected_logical_time_event_plan
    ):
        _fail_exact(
            FAIL_INPUT_V1,
            "logical-time event plan rows must use the exact pure-core DTO",
        )
    if (
        type(owner_authorization_payload) is not bytes
        or type(caller_supplied_current_head_payload) is not bytes
        or type(owner_projection_payload) is not bytes
    ):
        _fail_exact(FAIL_INPUT_V1, "owner raw singleton artifacts must be exact bytes")

    count = len(decision_payloads_in_exact_order)
    if count not in (1, 2):
        _fail_exact(FAIL_CHAIN_V1, "owner-status replay requires exactly one or two events")
    if not (
        len(signed_event_payloads_in_exact_order)
        == len(owner_checkpoint_payloads_in_exact_order)
        == len(logical_time_request_payloads_in_exact_order)
        == len(logical_time_signed_token_payloads_in_exact_order)
        == len(logical_time_checkpoint_payloads_in_exact_order)
        == len(expected_logical_time_event_plan)
        == count
        and len(owner_head_payloads_in_exact_order) == count + 1
        and len(logical_time_head_payloads_in_exact_order) == count + 1
    ):
        _fail_exact(
            FAIL_CHAIN_V1,
            "owner/time plan, decision, event, checkpoint, and head cardinalities differ",
        )
    if caller_supplied_current_head_payload != owner_head_payloads_in_exact_order[-1]:
        _fail_exact(
            FAIL_CHAIN_V1,
            "caller-supplied current head is not byte-equal to final replay head",
        )

    expected_attempt = _validate_identifier(
        expected_attempt_context_id, "expected Step-5 attempt context ID"
    )
    (
        lock_id,
        authority_profile_id,
        signer_registry_id,
        governance_key_id,
        governance_key_epoch,
        governance_public_key,
    ) = _extract_authority(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_payload_sha256_digest=(
            out_of_band_expected_payload_sha256_digest
        ),
    )
    common = (
        authority_profile_id,
        signer_registry_id,
        governance_key_id,
        governance_key_epoch,
        expected_attempt,
    )
    expected_ledger_id = _ledger_identity(common)
    _validate_owner_payload(owner_authorization_payload)

    try:
        logical_time_result = _verify_time_core(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            logical_time_request_payloads_in_exact_order,
            logical_time_signed_token_payloads_in_exact_order,
            logical_time_checkpoint_payloads_in_exact_order,
            logical_time_head_payloads_in_exact_order,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=expected_attempt,
            expected_event_plan=expected_logical_time_event_plan,
        )
        logical_time_result = _require_time_result(logical_time_result)
    except _time_error_type as exc:
        raise _error_type(
            FAIL_LOGICAL_TIME_V1,
            f"pure logical-time replay core rejected chain: {exc}",
        ) from exc
    if not (
        logical_time_result.stage_b_trust_lock_id == lock_id
        and logical_time_result.authority_profile_id == authority_profile_id
        and logical_time_result.signer_registry_id == signer_registry_id
        and logical_time_result.attempt_context_id == expected_attempt
        and logical_time_result.verified_event_count == count
        and logical_time_result.verified_request_count == count
        and logical_time_result.verified_signature_count == count
        and logical_time_result.verified_checkpoint_count == count
        and logical_time_result.verified_head_count == count + 1
        and logical_time_result.supplied_chain_final_head_verified is True
        and logical_time_result.external_currentness_verified is False
        and logical_time_result.external_cas_checkpoint_verified is False
        and logical_time_result.authority_evidence is False
        and logical_time_result.readiness_evidence is False
        and logical_time_result.r01_satisfied is False
        and logical_time_result.execution_authorized is False
    ):
        _fail_exact(FAIL_LOGICAL_TIME_V1, "logical-time result boundary differs")

    heads = tuple(
        _parse_head_exact(payload, common)
        for payload in owner_head_payloads_in_exact_order
    )
    expected_genesis = _expected_head(
        common=common,
        event_count=0,
        last_event_id=_genesis_event_id,
        last_checkpoint_id=_genesis_checkpoint_id,
        complete_event_set_root=_genesis_root,
        latest_effective_status=_status_none,
        revocation_count=0,
        terminal=False,
    )
    if (
        not _mapping_equal_exact(heads[0], expected_genesis)
        or _canonical_exact(expected_genesis) != owner_head_payloads_in_exact_order[0]
    ):
        _fail_exact(FAIL_CHAIN_V1, "owner-status genesis head differs from exact pins")

    decision_ids: list[str] = []
    event_ids: list[str] = []
    checkpoint_ids: list[str] = []
    head_ids: list[str] = [heads[0]["head_id"]]  # type: ignore[list-item]
    cas_receipt_ids: list[str] = []
    seen_decision_ids: set[str] = set()
    seen_decision_nonces: set[str] = set()
    seen_event_ids: set[str] = set()
    seen_checkpoint_ids: set[str] = set()
    seen_head_ids: set[str] = {heads[0]["head_id"]}  # type: ignore[arg-type]
    seen_cas_receipt_ids: set[str] = set()
    seen_cas_checkpoint_ids: set[str] = set()
    seen_cas_head_ids: set[str] = set()
    frozen_cas_context: tuple[str, str, str, int] | None = None
    previous_cas_checkpoint_index: int | None = None
    previous_head = heads[0]

    for index in range(count):
        if previous_head["terminal"] is True:
            _fail_exact(FAIL_CHAIN_V1, "terminal REVOKED head has a successor")
        decision_payload = decision_payloads_in_exact_order[index]
        event_payload = signed_event_payloads_in_exact_order[index]
        checkpoint_payload = owner_checkpoint_payloads_in_exact_order[index]
        decision = _parse_decision_exact(decision_payload, common)
        event = _parse_event_exact(
            event_payload,
            common,
            decision_payload,
            governance_public_key,
        )
        checkpoint = _parse_checkpoint_exact(
            checkpoint_payload,
            common,
            previous_head,
            event,
            event_payload,
        )
        next_head = heads[index + 1]
        if not (
            decision["sequence"] == index
            and checkpoint["checkpoint_index"] == index
            and decision["previous_head_id"] == previous_head["head_id"]
            and checkpoint["previous_head_id"] == previous_head["head_id"]
        ):
            _fail_exact(FAIL_CHAIN_V1, "owner-status sequence or prior head forks")

        expected_next_head = _expected_head(
            common=common,
            event_count=index + 1,
            last_event_id=event["event_id"],  # type: ignore[arg-type]
            last_checkpoint_id=checkpoint["checkpoint_id"],  # type: ignore[arg-type]
            complete_event_set_root=checkpoint["complete_event_set_root"],  # type: ignore[arg-type]
            latest_effective_status=decision["effective_status"],  # type: ignore[arg-type]
            revocation_count=decision["revocation_count"],  # type: ignore[arg-type]
            terminal=decision["effective_status"] == _status_revoked,
        )
        if (
            not _mapping_equal_exact(next_head, expected_next_head)
            or _canonical_exact(expected_next_head)
            != owner_head_payloads_in_exact_order[index + 1]
        ):
            _fail_exact(
                FAIL_CHAIN_V1,
                "next owner-status head does not exact-commit checkpoint",
            )

        plan_row = logical_time_result.verified_event_plan[index]
        if not (
            type(plan_row) is tuple
            and len(plan_row) == 4
            and plan_row[0] == _logical_time_event_purpose
            and plan_row[1] == decision["decision_id"]
            and plan_row[2] == _sha256_exact(decision_payload)
        ):
            _fail_exact(
                FAIL_LOGICAL_TIME_V1,
                "logical-time plan does not bind the exact owner decision",
            )
        logical_dependency = event["logical_time_dependency"]
        cas_dependency = event["cas_store_dependency"]
        if type(logical_dependency) is not dict or type(cas_dependency) is not dict:
            _fail_exact(FAIL_SCHEMA_V1, "event dependency objects differ")
        if not (
            logical_dependency["signer_key_id"]
            == logical_time_result.logical_time_key_id
            and logical_dependency["signer_key_epoch"]
            == logical_time_result.logical_time_key_epoch
            and logical_dependency["token_id"]
            == logical_time_result.token_ids[index]
            and logical_dependency["token_canonical_sha256"]
            == _sha256_exact(
                logical_time_signed_token_payloads_in_exact_order[index]
            )
            and logical_dependency["token_bound_artifact_id"]
            == decision["decision_id"]
            and logical_dependency["token_bound_artifact_sha256"]
            == _sha256_exact(decision_payload)
            and logical_dependency["token_sequence"] == index
        ):
            _fail_exact(
                FAIL_LOGICAL_TIME_V1,
                "event logical-time reference differs from pure replay result",
            )

        cas_context = (
            cas_dependency["retention_session_id"],
            cas_dependency["chain_id"],
            cas_dependency["signer_key_id"],
            cas_dependency["signer_key_epoch"],
        )
        cas_checkpoint_index = cas_dependency["checkpoint_index"]
        if index == 0:
            frozen_cas_context = cas_context  # type: ignore[assignment]
        elif not (
            cas_context == frozen_cas_context
            and type(cas_checkpoint_index) is int
            and previous_cas_checkpoint_index is not None
            and cas_checkpoint_index == previous_cas_checkpoint_index + 1
        ):
            _fail_exact(
                FAIL_CHAIN_V1,
                "legacy CAS candidate context or checkpoint order forks",
            )

        decision_id = decision["decision_id"]
        decision_nonce = decision["decision_nonce"]
        event_id = event["event_id"]
        checkpoint_id = checkpoint["checkpoint_id"]
        head_id = next_head["head_id"]
        cas_receipt_id = cas_dependency["signed_receipt_id"]
        cas_checkpoint_id = cas_dependency["checkpoint_id"]
        cas_head_id = cas_dependency["head_id"]
        if (
            decision_id in seen_decision_ids
            or decision_nonce in seen_decision_nonces
            or event_id in seen_event_ids
            or checkpoint_id in seen_checkpoint_ids
            or head_id in seen_head_ids
            or cas_receipt_id in seen_cas_receipt_ids
            or cas_checkpoint_id in seen_cas_checkpoint_ids
            or cas_head_id in seen_cas_head_ids
        ):
            _fail_exact(FAIL_CHAIN_V1, "owner-status or CAS candidate identity replayed")
        seen_decision_ids.add(decision_id)  # type: ignore[arg-type]
        seen_decision_nonces.add(decision_nonce)  # type: ignore[arg-type]
        seen_event_ids.add(event_id)  # type: ignore[arg-type]
        seen_checkpoint_ids.add(checkpoint_id)  # type: ignore[arg-type]
        seen_head_ids.add(head_id)  # type: ignore[arg-type]
        seen_cas_receipt_ids.add(cas_receipt_id)  # type: ignore[arg-type]
        seen_cas_checkpoint_ids.add(cas_checkpoint_id)  # type: ignore[arg-type]
        seen_cas_head_ids.add(cas_head_id)  # type: ignore[arg-type]
        decision_ids.append(decision_id)  # type: ignore[arg-type]
        event_ids.append(event_id)  # type: ignore[arg-type]
        checkpoint_ids.append(checkpoint_id)  # type: ignore[arg-type]
        head_ids.append(head_id)  # type: ignore[arg-type]
        cas_receipt_ids.append(cas_receipt_id)  # type: ignore[arg-type]
        previous_cas_checkpoint_index = cas_checkpoint_index  # type: ignore[assignment]
        previous_head = next_head

    final_head = heads[-1]
    if not (
        final_head["event_count"] == count
        and final_head["next_sequence"] == count
        and final_head["head_id"] == head_ids[-1]
        and _canonical_exact(final_head) == caller_supplied_current_head_payload
    ):
        _fail_exact(FAIL_CHAIN_V1, "final/current owner-status head differs")
    projection = _parse_projection_exact(
        owner_projection_payload, common, final_head
    )

    return _issue_result(
        lock_id=lock_id,
        common=common,
        logical_time_key_id=logical_time_result.logical_time_key_id,
        logical_time_key_epoch=logical_time_result.logical_time_key_epoch,
        ledger_id=expected_ledger_id,
        decision_ids=tuple(decision_ids),
        event_ids=tuple(event_ids),
        checkpoint_ids=tuple(checkpoint_ids),
        head_ids=tuple(head_ids),
        logical_time_request_ids=logical_time_result.request_ids,
        logical_time_token_ids=logical_time_result.token_ids,
        legacy_cas_receipt_ids=tuple(cas_receipt_ids),
        current_head_payload_sha256=_sha256_exact(caller_supplied_current_head_payload),
        complete_event_set_root=final_head["complete_event_set_root"],  # type: ignore[arg-type]
        latest_effective_status=final_head["latest_effective_status"],  # type: ignore[arg-type]
        revocation_count=final_head["revocation_count"],  # type: ignore[arg-type]
        projection_id=projection["projection_id"],  # type: ignore[arg-type]
        projection_payload_sha256=_sha256_exact(owner_projection_payload),
        logical_time_ledger_id=logical_time_result.ledger_id,
        logical_time_final_head_id=logical_time_result.final_head_id,
    )


def _build_public_replay_entry_v1(implementation):
    def verify_owner_status_replay_core_v1(
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        owner_authorization_payload: bytes,
        decision_payloads_in_exact_order: tuple[bytes, ...],
        signed_event_payloads_in_exact_order: tuple[bytes, ...],
        owner_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        owner_head_payloads_in_exact_order: tuple[bytes, ...],
        caller_supplied_current_head_payload: bytes,
        owner_projection_payload: bytes,
        logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
        logical_time_signed_token_payloads_in_exact_order: tuple[bytes, ...],
        logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_payload_sha256_digest: bytes,
        expected_attempt_context_id: str,
        expected_logical_time_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
    ) -> VerifiedOwnerStatusReplayCoreV1:
        return implementation(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            owner_authorization_payload,
            decision_payloads_in_exact_order,
            signed_event_payloads_in_exact_order,
            owner_checkpoint_payloads_in_exact_order,
            owner_head_payloads_in_exact_order,
            caller_supplied_current_head_payload,
            owner_projection_payload,
            logical_time_request_payloads_in_exact_order,
            logical_time_signed_token_payloads_in_exact_order,
            logical_time_checkpoint_payloads_in_exact_order,
            logical_time_head_payloads_in_exact_order,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=expected_attempt_context_id,
            expected_logical_time_event_plan=expected_logical_time_event_plan,
        )

    return verify_owner_status_replay_core_v1


verify_owner_status_replay_core_v1 = _build_public_replay_entry_v1(
    _verify_owner_status_replay_core_implementation_v1
)
del _build_public_replay_entry_v1
del _verify_owner_status_replay_core_implementation_v1
del _hidden_issue_verified_owner_status_replay_core_v1


def verify_external_currentness_v1(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: caller-supplied final-head equality is not currentness."""

    _fail(
        FAIL_CLAIM_BOUNDARY_V1,
        "external currentness and anti-fork anchoring are outside this core",
    )


def verify_cas_dependencies_v1(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: legacy embedded CAS/1 references are candidates only."""

    _fail(
        FAIL_CLAIM_BOUNDARY_V1,
        "legacy CAS references are not authenticated CAS sessions",
    )


__all__ = [
    "AUTHORITY_CLAIM_BOUNDARY_V1",
    "CAS_DEPENDENCY_REFERENCE_SCHEMA_V1",
    "CHECKPOINT_PURPOSE_V1",
    "EVENT_INITIAL_ACTIVE_V1",
    "EVENT_REVOKED_V1",
    "ExpectedLogicalEventCoreV1",
    "FAIL_AUTHORITY_V1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CHAIN_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_INPUT_V1",
    "FAIL_LOGICAL_TIME_V1",
    "FAIL_RESULT_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_SIGNATURE_BACKEND_V1",
    "FAIL_SIGNATURE_V1",
    "GENESIS_CHECKPOINT_ID_V1",
    "GENESIS_COMPLETE_EVENT_SET_ROOT_V1",
    "GENESIS_EVENT_ID_V1",
    "GOVERNANCE_SIGNATURE_DOMAIN_V1",
    "HEAD_PURPOSE_V1",
    "LOGICAL_TIME_EVENT_PURPOSE_V1",
    "OWNER_STATUS_CHECKPOINT_SCHEMA_V1",
    "OWNER_STATUS_DECISION_SCHEMA_V1",
    "OWNER_STATUS_HEAD_SCHEMA_V1",
    "OWNER_STATUS_PROJECTION_SCHEMA_V1",
    "PROJECTION_CLAIM_LEVEL_V1",
    "Phase2BStep5OwnerStatusReplayCoreErrorV1",
    "SCHEMA_VERSION",
    "SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1",
    "STATUS_ACTIVE_V1",
    "STATUS_REVOKED_V1",
    "VerifiedOwnerStatusReplayCoreV1",
    "require_verified_owner_status_replay_core_v1",
    "verify_cas_dependencies_v1",
    "verify_external_currentness_v1",
    "verify_owner_status_replay_core_v1",
]
