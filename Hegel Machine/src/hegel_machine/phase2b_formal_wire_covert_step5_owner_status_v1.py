"""Mechanics-only owner-authorization status ledger for Step 5.

This module freezes canonical RFC-8785 wire contracts and local append-only
filesystem mechanics for a monotone ``ACTIVE`` -> ``REVOKED`` owner-
authorization status chain.  Owner-status schema/1 keeps its field shape while
its authority-profile and signer-registry dependencies are replaced by the
profile/2 and registry/2 identifier lineages.  It deliberately imports neither
the Step-5 authority module nor the CAS-store or logical-time actor modules.
Cross-actor artifacts enter only as exact canonical bytes and purpose-separated
bindings.

Authority/profile V2 statically freezes ``GOVERNANCE_ANCHOR`` with the exact
two-purpose list ending in ``SIGN_OWNER_AUTHORIZATION_STATUS_EVENT``.  This
module byte-matches that role-key derivation policy but does not load raw
Stage-B lock/profile/registry bytes or verify their closed graph.  Nor does it
verify a live runtime closure, complete CAS session, complete logical-time
ledger, or external anti-fork anchor.  Consequently, artifacts produced here
remain construction candidates.  Nothing in this module sets R01, authority
evidence, or any readiness node true.

Local ``O_NOFOLLOW``/``O_EXCL`` publication, fsync, readback, and versioned
files expose crash gaps and accidental mutation.  They do not resist rollback
or cloning of the entire ledger by a privileged host administrator.  A later
authority integration must externally anchor the final complete-event-set root
and prove the distinct governance/CAS/logical-time control domains.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import os
import re
import stat
import threading
from types import MappingProxyType
from typing import Final, Iterator, Mapping

from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Importability remains; every crypto operation still fails closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - depends on the pinned actor image.
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status/1"
)
CAS_DEPENDENCY_REFERENCE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-cas-dependency-reference/1"
)
LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-owner-status-logical-time-dependency-reference/1"
)
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

GOVERNANCE_ROLE_V1: Final = "GOVERNANCE_ANCHOR"
GOVERNANCE_CONTROL_DOMAIN_V1: Final = "STEP5_GOVERNANCE_CONTROL"
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
LOGICAL_TIME_CONTROL_DOMAIN_V1: Final = "STEP5_LOGICAL_TIME_CONTROL"
LOGICAL_TIME_SIGNATURE_PURPOSE_V1: Final = "SIGN_LOGICAL_TIME_TOKEN"
LOGICAL_TIME_EVENT_PURPOSE_V1: Final = "OWNER_AUTHORIZATION_STATUS_DECISION"
LOGICAL_TIME_KEY_EPOCH_V1: Final = 1
LOGICAL_TIME_TOKEN_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-logical-time-token/1"
)
LOGICAL_TIME_REQUEST_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-logical-time-request/1"
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
AUTHORITY_CLAIM_BOUNDARY_V1: Final = (
    "OWNER_STATUS_CANONICAL_GOVERNANCE_SIGNATURE_AND_LOCAL_APPEND_ONLY_"
    "MECHANICS_ONLY_AUTHORITY_V2_PURPOSE_STATICALLY_FROZEN_NOT_RAW_LOCK_"
    "PROFILE_REGISTRY_RUNTIME_CAS_TIME_OR_EXTERNAL_ANCHOR_VERIFIED_NOT_LATEST_"
    "AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION_NOT_ROLLBACK_OR_CLONE_"
    "RESISTANT"
)
REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1: Final = (
    "OUT_OF_BAND_AUTHORITY_V2_PROFILE_AND_STAGE_B_TRUST_LOCK_PINS_VERIFIED",
    "RAW_AUTHORITY_V2_SIGNER_REGISTRY_BYTES_AND_PURPOSE_GRAPH_VERIFIED",
    "FRESH_STEP5_ATTEMPT_CONTEXT_VERIFIED",
    "PINNED_GOVERNANCE_STATUS_SERVICE_IMAGE_AND_RUNTIME_CLOSURE_VERIFIED",
    "EXPECTED_DISTINCT_GOVERNANCE_SERVICE_UID_AND_CONTROL_DOMAIN_VERIFIED",
    "GOVERNANCE_PRIVATE_KEY_CUSTODY_VERIFIED",
    "CAS_STORE_SIGNED_RECEIPT_AND_COMPLETE_HEAD_CHECKPOINT_VERIFIED",
    "LOGICAL_TIME_COMPLETE_LEDGER_AND_TOKEN_ORDER_VERIFIED",
    "EXTERNAL_ANTI_ROLLBACK_ANCHOR_OF_COMPLETE_EVENT_SET_ROOT_VERIFIED",
    "PRODUCER_FREE_PURE_REPLAY_VERIFIED",
)
REQUIRED_AUTHORITY_SCHEMA_CASCADE_V1: Final = (
    "OWNER_STATUS_SCHEMA_V1_FIELD_SHAPE_UNCHANGED",
    "AUTHORITY_DEPENDENCY_REPLACED_BY_PROFILE_V2_AND_SIGNER_REGISTRY_V2",
    "GOVERNANCE_TWO_PURPOSE_ROLE_KEY_POLICY_STATICALLY_FROZEN_IN_AUTHORITY_V2",
    "OWNER_STATUS_ROLE_KEY_DERIVATION_BYTE_MATCHES_AUTHORITY_V2",
    "RAW_STAGE_B_LOCK_PROFILE_REGISTRY_GRAPH_STILL_MUST_BE_BYTES_VERIFIED",
    "CAS_AND_LOGICAL_TIME_DEPENDENCIES_BIND_PROFILE_V2_REGISTRY_V2_ATTEMPT",
    "PINNED_DISTINCT_SERVICE_IMAGES_AND_RUNTIME_CLOSURES_STILL_MUST_BE_VERIFIED",
    "RUN_COMPLETE_BYTES_ONLY_PURE_REPLAY_BEFORE_ANY_REAL_R01_VERIFIER",
)
KNOWN_INTEGRATION_BLOCKERS_V1: Final = (
    "AUTHORITY_V2_GOVERNANCE_PURPOSE_IS_STATICALLY_FROZEN_BUT_RAW_REGISTRY_GRAPH_IS_NOT_VERIFIED_HERE",
    "AUTHORITY_V2_PROFILE_REGISTRY_IDS_ARE_PREFIX_VALIDATED_ONLY_NOT_CROSS_GRAPH_VERIFIED",
    "CAS_COMPLETE_SESSION_IS_NOT_VERIFIED_HERE",
    "LOGICAL_TIME_COMPLETE_LEDGER_IS_NOT_VERIFIED_HERE",
    "NO_EXTERNAL_ANTI_ROLLBACK_OR_ANTI_CLONE_ANCHOR_EXISTS",
    "PINNED_LIVE_GOVERNANCE_STATUS_SERVICE_IMAGE_AND_RUNTIME_CLOSURE_ARE_NOT_VERIFIED_HERE",
    "NO_REAL_R01_RECEIPT_OR_READINESS_OVERLAY_SCHEMA_IS_ISSUED_HERE",
)
EVENT_CLAIM_BOUNDARY_V1: Final = (
    "GOVERNANCE_SIGNED_STATUS_EVENT_MECHANICS_CANDIDATE_AUTHORITY_V2_PURPOSE_"
    "STATICALLY_FROZEN_NOT_RAW_REGISTRY_VERIFIED_NOT_R01_NOT_READINESS_NOT_"
    "EXECUTION"
)

AUTHORITY_PROFILE_PREFIX_V2: Final = "phase2b_step5_authority_profile_v2_"
SIGNER_REGISTRY_PREFIX_V2: Final = "phase2b_step5_signer_registry_v2_"
ROLE_KEY_PREFIX_V1: Final = "phase2b_step5_role_key_v1_"
LEDGER_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_ledger_v1_"
DECISION_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_decision_v1_"
EVENT_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_event_v1_"
CHECKPOINT_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_checkpoint_v1_"
HEAD_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_head_v1_"
PROJECTION_ID_PREFIX_V1: Final = "phase2b_step5_owner_status_projection_v1_"
LOGICAL_TIME_REQUEST_ID_PREFIX_V1: Final = (
    "phase2b_step5_logical_time_request_v1_"
)
LOGICAL_TIME_TOKEN_ID_PREFIX_V1: Final = "phase2b_step5_logical_time_token_v1_"
LOGICAL_TIME_GENESIS_TOKEN_ID_V1: Final = (
    LOGICAL_TIME_TOKEN_ID_PREFIX_V1 + ("0" * 64)
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
_LOGICAL_TIME_ROLE_KEY_ID_DOMAIN: Final = _ROLE_KEY_ID_DOMAIN
_LOGICAL_TIME_TOKEN_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/SIGN/LOGICAL_TIME_TOKEN/V1\x00"
)
_LOGICAL_TIME_REQUEST_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/REQUEST_ID/V1\x00"
)
_LOGICAL_TIME_TOKEN_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/TOKEN_ID/V1\x00"
)

GENESIS_COMPLETE_EVENT_SET_ROOT_V1: Final = "sha256:" + hashlib.sha256(
    _COMPLETE_EVENT_SET_ROOT_DOMAIN
    + canonicalize_rfc8785_v1(
        {
            "schema_version": (
                "hegel-machine-phase2b-step5-owner-status-complete-event-set-root/1"
            ),
            "events_in_sequence": [],
        }
    )
).hexdigest()

MAXIMUM_OWNER_AUTHORIZATION_BYTES: Final = 128 * 1024
MAXIMUM_DEPENDENCY_REFERENCE_BYTES: Final = 128 * 1024
MAXIMUM_DECISION_BYTES: Final = 256 * 1024
MAXIMUM_EVENT_BYTES: Final = 768 * 1024
MAXIMUM_CHECKPOINT_BYTES: Final = 1024 * 1024
MAXIMUM_HEAD_BYTES: Final = 128 * 1024
MAXIMUM_PROJECTION_BYTES: Final = 256 * 1024
MAXIMUM_LOGICAL_TIME_TOKEN_BYTES: Final = 128 * 1024
_MAXIMUM_SAFE_JSON_INTEGER: Final = (1 << 53) - 1
_MAXIMUM_IDENTIFIER_ASCII_BYTES: Final = 512

_LOWER_HEX_32_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX_64_PATTERN: Final = re.compile(r"[0-9a-f]{128}")
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_PATTERN: Final = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}"
)
_UPPER_TOKEN_PATTERN: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_VERSIONED_FILENAME_PATTERN: Final = re.compile(r"[0-9]{20}\.json")

_OWNER_BINDING_KEYS: Final = frozenset(EXACT_OWNER_AUTHORIZATION_BINDING_V1)
_CAS_DEPENDENCY_KEYS: Final = frozenset(
    {
        "schema_version",
        "verification_status",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "signer_role",
        "signer_key_id",
        "signer_key_epoch",
        "signature_purpose",
        "signature_domain",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "object_id",
        "object_canonical_sha256",
        "signed_receipt_schema_version",
        "signed_receipt_id",
        "signed_receipt_canonical_sha256",
        "checkpoint_schema_version",
        "checkpoint_id",
        "checkpoint_canonical_sha256",
        "checkpoint_index",
        "head_schema_version",
        "head_id",
        "head_canonical_sha256",
        "head_last_checkpoint_id",
        "head_checkpoint_count",
    }
)
_LOGICAL_TIME_DEPENDENCY_KEYS: Final = frozenset(
    {
        "schema_version",
        "verification_status",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "signer_role",
        "signer_key_id",
        "signer_key_epoch",
        "signature_purpose",
        "token_schema_version",
        "token_id",
        "token_canonical_sha256",
        "token_bound_artifact_id",
        "token_bound_artifact_sha256",
        "token_sequence",
    }
)
_DECISION_KEYS: Final = frozenset(
    {
        "schema_version",
        "ledger_id",
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "attempt_context_id",
        "owner_authorization_binding",
        "owner_authorization_payload_sha256",
        "sequence",
        "event_type",
        "prior_effective_status",
        "effective_status",
        "revocation_count",
        "reason_code",
        "decision_nonce",
        "previous_event_id",
        "previous_head_id",
        "previous_checkpoint_id",
        "previous_complete_event_set_root",
        "decision_id",
    }
)
_EVENT_KEYS: Final = frozenset(
    {
        "schema_version",
        "signature_purpose",
        "signature_domain",
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "governance_public_key_hex",
        "attempt_context_id",
        "decision",
        "decision_canonical_sha256",
        "cas_store_dependency",
        "logical_time_dependency",
        "algorithm",
        "signature_hex",
        "event_id",
        "claim_boundary",
    }
)
_CHECKPOINT_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "ledger_id",
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "attempt_context_id",
        "checkpoint_index",
        "previous_head_id",
        "previous_checkpoint_id",
        "previous_event_id",
        "previous_complete_event_set_root",
        "event_id",
        "signed_event_canonical_sha256",
        "signed_event",
        "complete_event_set_root",
        "checkpoint_id",
    }
)
_HEAD_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "ledger_id",
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "attempt_context_id",
        "event_count",
        "next_sequence",
        "last_event_id",
        "last_checkpoint_id",
        "complete_event_set_root",
        "latest_effective_status",
        "revocation_count",
        "terminal",
        "head_id",
    }
)
_PROJECTION_KEYS: Final = frozenset(
    {
        "schema_version",
        "claim_level",
        "ledger_id",
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "attempt_context_id",
        "owner_authorization_binding",
        "owner_authorization_payload_sha256",
        "event_count",
        "latest_sequence",
        "latest_effective_status",
        "revocation_count",
        "last_event_id",
        "last_checkpoint_id",
        "head_id",
        "complete_event_set_root",
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
        "projection_id",
    }
)
_LOGICAL_TIME_TOKEN_KEYS: Final = frozenset(
    {
        "schema_version",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "request_id",
        "request_nonce",
        "signature_purpose",
        "purpose",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "prior_token_id",
        "sequence",
        "algorithm",
        "signature_hex",
        "token_id",
    }
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_SCHEMA"
FAIL_BINDING_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_BINDING"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_CONTENT_ID"
FAIL_SIGNATURE_BACKEND_V1: Final = (
    "FAIL_STEP5_OWNER_STATUS_V1_SIGNATURE_BACKEND"
)
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_SIGNATURE"
FAIL_LEDGER_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_LEDGER"
FAIL_FILESYSTEM_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_FILESYSTEM"
FAIL_CLAIM_BOUNDARY_V1: Final = "FAIL_STEP5_OWNER_STATUS_V1_CLAIM_BOUNDARY"


class Phase2BStep5OwnerStatusErrorV1(ValueError):
    """Stable fail-closed exception for owner-status mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> None:
    raise Phase2BStep5OwnerStatusErrorV1(code, detail)


def _canonical(value: object) -> bytes:
    try:
        return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5OwnerStatusErrorV1(FAIL_CANONICAL_V1, str(exc)) from exc


def _restore_integral_numbers(value: object) -> object:
    if type(value) is float:
        if not value.is_integer() or abs(value) > _MAXIMUM_SAFE_JSON_INTEGER:
            _fail(FAIL_SCHEMA_V1, "owner-status schemas do not admit fractions")
        return int(value)
    if type(value) is list:
        return [_restore_integral_numbers(item) for item in value]
    if type(value) is dict:
        return {
            key: _restore_integral_numbers(item) for key, item in value.items()
        }
    return value


def _decode(payload: bytes, maximum_bytes: int, label: str) -> dict[str, object]:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, f"{label} must be exact bytes")
    if not 1 <= len(payload) <= maximum_bytes or payload.endswith(b"\n"):
        _fail(FAIL_INPUT_V1, f"{label} is outside its exact byte boundary")
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5OwnerStatusErrorV1(
            FAIL_CANONICAL_V1, f"{label}: {exc}"
        ) from exc
    restored = _restore_integral_numbers(decoded)
    if type(restored) is not dict or _canonical(restored) != payload:
        _fail(FAIL_CANONICAL_V1, f"{label} canonical readback differs")
    return restored


def _exact_object(
    value: object, keys: frozenset[str], label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        _fail(FAIL_SCHEMA_V1, f"{label} exact field set differs")
    return value


def _exact_equal(value: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping):
            return False
        return set(value) == set(expected) and all(
            _exact_equal(value[key], expected[key]) for key in expected
        )
    if type(value) is not type(expected):
        return False
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_equal(left, right)
            for left, right in zip(value, expected, strict=True)
        )
    return value == expected


def _exact_literal(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(FAIL_SCHEMA_V1, f"{label} exact literal differs")


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be one bounded ASCII identifier")
    if len(value.encode("ascii")) > _MAXIMUM_IDENTIFIER_ASCII_BYTES:
        _fail(FAIL_SCHEMA_V1, f"{label} exceeds its ASCII byte bound")
    return value


def _prefixed_id(value: object, prefix: str, label: str) -> str:
    result = _identifier(value, label)
    suffix = result[len(prefix) :] if result.startswith(prefix) else ""
    if not result.startswith(prefix) or _LOWER_HEX_32_PATTERN.fullmatch(suffix) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} prefix or digest differs")
    return result


def _artifact_id(value: object, label: str) -> str:
    result = _identifier(value, label)
    if _SHA256_PATTERN.fullmatch(result) is not None:
        return result
    if "_" not in result or _LOWER_HEX_32_PATTERN.fullmatch(result[-64:]) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must carry one lowercase SHA-256 suffix")
    return result


def _sha256_address(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be sha256:<lowerhex64>")
    return value


def _hex32(value: object, label: str) -> str:
    if type(value) is not str or _LOWER_HEX_32_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lowercase 32-byte hex")
    return value


def _hex64(value: object, label: str) -> str:
    if type(value) is not str or _LOWER_HEX_64_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lowercase 64-byte hex")
    return value


def _upper_token(value: object, label: str) -> str:
    if type(value) is not str or _UPPER_TOKEN_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be one frozen uppercase token")
    return value


def _wire_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(FAIL_SCHEMA_V1, f"{label} exact integer differs")
    return value


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "SHA-256 input must be exact bytes")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _content_id(prefix: str, domain: bytes, subject: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(subject)).hexdigest()


def _validate_owner_binding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(FAIL_SCHEMA_V1, "frozen owner-authorization binding is not mapping")
    binding = dict(value)
    if set(binding) != set(_OWNER_BINDING_KEYS):
        _fail(FAIL_SCHEMA_V1, "frozen owner-authorization binding keyset differs")
    if not _exact_equal(binding, EXACT_OWNER_AUTHORIZATION_BINDING_V1):
        _fail(FAIL_BINDING_V1, "owner-authorization full exact binding differs")
    return binding


def verify_exact_owner_authorization_payload_v1(payload: bytes) -> None:
    """Verify the frozen canonical owner payload without importing its actor."""

    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "owner-authorization payload must be exact bytes")
    if len(payload) > MAXIMUM_OWNER_AUTHORIZATION_BYTES:
        _fail(FAIL_INPUT_V1, "owner-authorization payload exceeds its frozen bound")
    if hashlib.sha256(payload).hexdigest() != OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1:
        _fail(FAIL_BINDING_V1, "owner-authorization payload SHA-256 differs")
    wrapper = _decode(
        payload, MAXIMUM_OWNER_AUTHORIZATION_BYTES, "owner-authorization payload"
    )
    if set(wrapper) != {"authorization_body", "authorization_id"}:
        _fail(FAIL_SCHEMA_V1, "owner-authorization wrapper field set differs")
    body = wrapper["authorization_body"]
    if type(body) is not dict:
        _fail(FAIL_SCHEMA_V1, "owner-authorization body must be an object")
    if (
        wrapper["authorization_id"] != OWNER_AUTHORIZATION_ID_V1
        or body.get("schema_version") != OWNER_AUTHORIZATION_SCHEMA_V1
    ):
        _fail(FAIL_BINDING_V1, "owner-authorization ID or schema differs")


def derive_required_governance_role_key_id_v1(
    governance_public_key: bytes,
    *,
    key_epoch: int = GOVERNANCE_KEY_EPOCH_V1,
) -> str:
    """Derive the authority/profile V2 role-key ID with both exact purposes.

    The subject byte-matches authority/profile V2's governance role-key/1 row.
    This function derives the key ID but does not load or verify registry bytes.
    """

    if type(governance_public_key) is not bytes or len(governance_public_key) != 32:
        _fail(FAIL_INPUT_V1, "governance public key must be exact raw 32 bytes")
    epoch = _wire_integer(key_epoch, "governance key epoch", minimum=1)
    if epoch != GOVERNANCE_KEY_EPOCH_V1:
        _fail(FAIL_BINDING_V1, "owner-status schema/1 requires key epoch 1")
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": GOVERNANCE_ROLE_V1,
        "control_domain": GOVERNANCE_CONTROL_DOMAIN_V1,
        "authorized_signature_purposes": list(REQUIRED_GOVERNANCE_PURPOSES_V1),
        "algorithm": ED25519_ALGORITHM_V1,
        "key_epoch": epoch,
        "public_key_hex": governance_public_key.hex(),
    }
    return _content_id(ROLE_KEY_PREFIX_V1, _ROLE_KEY_ID_DOMAIN, subject)


def _derive_logical_time_role_key_id_v1(public_key: bytes) -> str:
    if type(public_key) is not bytes or len(public_key) != 32:
        _fail(FAIL_INPUT_V1, "logical-time public key must be exact raw 32 bytes")
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": LOGICAL_TIME_ROLE_V1,
        "control_domain": LOGICAL_TIME_CONTROL_DOMAIN_V1,
        "authorized_signature_purposes": [LOGICAL_TIME_SIGNATURE_PURPOSE_V1],
        "algorithm": ED25519_ALGORITHM_V1,
        "key_epoch": LOGICAL_TIME_KEY_EPOCH_V1,
        "public_key_hex": public_key.hex(),
    }
    return _content_id(
        ROLE_KEY_PREFIX_V1, _LOGICAL_TIME_ROLE_KEY_ID_DOMAIN, subject
    )


def _validate_common_bindings(
    *,
    authority_profile_id: object,
    signer_registry_id: object,
    governance_key_id: object,
    governance_key_epoch: object,
    attempt_context_id: object,
) -> tuple[str, str, str, int, str]:
    profile = _prefixed_id(
        authority_profile_id, AUTHORITY_PROFILE_PREFIX_V2, "authority profile ID"
    )
    registry = _prefixed_id(
        signer_registry_id, SIGNER_REGISTRY_PREFIX_V2, "signer registry ID"
    )
    key_id = _prefixed_id(
        governance_key_id, ROLE_KEY_PREFIX_V1, "governance role-key ID"
    )
    epoch = _wire_integer(governance_key_epoch, "governance key epoch", minimum=1)
    if epoch != GOVERNANCE_KEY_EPOCH_V1:
        _fail(FAIL_BINDING_V1, "owner-status schema/1 requires key epoch 1")
    attempt = _identifier(attempt_context_id, "fresh Step-5 attempt context ID")
    return profile, registry, key_id, epoch, attempt


def _ledger_id_from_common(
    common: tuple[str, str, str, int, str]
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
    return _content_id(LEDGER_ID_PREFIX_V1, _LEDGER_ID_DOMAIN, subject)


@dataclass(frozen=True, slots=True)
class CasStoreDependencyReferenceV1:
    """Exact opaque CAS binding; schema/1 does not authenticate it yet."""

    schema_version: str
    verification_status: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    retention_session_id: str
    chain_id: str
    signer_role: str
    signer_key_id: str
    signer_key_epoch: int
    signature_purpose: str
    signature_domain: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    object_id: str
    object_canonical_sha256: str
    signed_receipt_schema_version: str
    signed_receipt_id: str
    signed_receipt_canonical_sha256: str
    checkpoint_schema_version: str
    checkpoint_id: str
    checkpoint_canonical_sha256: str
    checkpoint_index: int
    head_schema_version: str
    head_id: str
    head_canonical_sha256: str
    head_last_checkpoint_id: str
    head_checkpoint_count: int

    def __post_init__(self) -> None:
        if type(self) is not CasStoreDependencyReferenceV1:
            _fail(FAIL_INPUT_V1, "CAS dependency must use the exact DTO")
        for value, expected, label in (
            (self.schema_version, CAS_DEPENDENCY_REFERENCE_SCHEMA_V1, "CAS dependency schema"),
            (self.verification_status, CAS_DEPENDENCY_VERIFICATION_STATUS_V1, "CAS verification status"),
            (self.signer_role, CAS_STORE_ROLE_V1, "CAS signer role"),
            (self.signature_purpose, CAS_STORE_RECEIPT_PURPOSE_V1, "CAS signature purpose"),
            (self.signature_domain, CAS_STORE_SIGNATURE_DOMAIN_V1, "CAS signature domain"),
            (self.signed_receipt_schema_version, CAS_SIGNED_RECEIPT_SCHEMA_V1, "CAS signed-receipt schema"),
            (self.checkpoint_schema_version, CAS_CHECKPOINT_SCHEMA_V1, "CAS checkpoint schema"),
            (self.head_schema_version, CAS_HEAD_SCHEMA_V1, "CAS head schema"),
        ):
            _exact_literal(value, expected, label)
        _prefixed_id(
            self.authority_profile_id,
            AUTHORITY_PROFILE_PREFIX_V2,
            "CAS dependency authority profile ID",
        )
        _prefixed_id(
            self.signer_registry_id,
            SIGNER_REGISTRY_PREFIX_V2,
            "CAS dependency signer registry ID",
        )
        _identifier(self.attempt_context_id, "CAS dependency attempt context ID")
        _identifier(self.retention_session_id, "CAS retention session ID")
        _identifier(self.chain_id, "CAS chain ID")
        _prefixed_id(
            self.signer_key_id, ROLE_KEY_PREFIX_V1, "CAS signer role-key ID"
        )
        if _wire_integer(
            self.signer_key_epoch, "CAS dependency key epoch", minimum=1
        ) != CAS_STORE_KEY_EPOCH_V1:
            _fail(FAIL_BINDING_V1, "CAS dependency key epoch differs")
        _artifact_id(self.bound_artifact_id, "CAS-bound artifact ID")
        _sha256_address(
            self.bound_artifact_sha256, "CAS-bound artifact canonical SHA-256"
        )
        _artifact_id(self.object_id, "CAS object ID")
        _sha256_address(
            self.object_canonical_sha256, "CAS object canonical SHA-256"
        )
        _identifier(
            self.signed_receipt_schema_version, "CAS signed-receipt schema"
        )
        _artifact_id(self.signed_receipt_id, "CAS signed-receipt ID")
        _sha256_address(
            self.signed_receipt_canonical_sha256,
            "CAS signed-receipt canonical SHA-256",
        )
        _identifier(self.checkpoint_schema_version, "CAS checkpoint schema")
        _artifact_id(self.checkpoint_id, "CAS checkpoint ID")
        _sha256_address(
            self.checkpoint_canonical_sha256,
            "CAS checkpoint canonical SHA-256",
        )
        _wire_integer(self.checkpoint_index, "CAS checkpoint index")
        _identifier(self.head_schema_version, "CAS head schema")
        _artifact_id(self.head_id, "CAS head ID")
        _sha256_address(
            self.head_canonical_sha256, "CAS head canonical SHA-256"
        )
        _artifact_id(
            self.head_last_checkpoint_id, "CAS head last-checkpoint ID"
        )
        if (
            self.object_id != self.object_canonical_sha256
            or self.signed_receipt_id != self.signed_receipt_canonical_sha256
            or self.checkpoint_id != self.checkpoint_canonical_sha256
            or self.head_id != self.head_canonical_sha256
            or self.head_last_checkpoint_id != self.checkpoint_id
        ):
            _fail(
                FAIL_BINDING_V1,
                "CAS content-address IDs must equal their canonical SHA-256",
            )
        count = _wire_integer(
            self.head_checkpoint_count, "CAS head checkpoint count", minimum=1
        )
        if count != self.checkpoint_index + 1:
            _fail(FAIL_BINDING_V1, "CAS head is not exact post-checkpoint head")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verification_status": self.verification_status,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "chain_id": self.chain_id,
            "signer_role": self.signer_role,
            "signer_key_id": self.signer_key_id,
            "signer_key_epoch": self.signer_key_epoch,
            "signature_purpose": self.signature_purpose,
            "signature_domain": self.signature_domain,
            "bound_artifact_id": self.bound_artifact_id,
            "bound_artifact_sha256": self.bound_artifact_sha256,
            "object_id": self.object_id,
            "object_canonical_sha256": self.object_canonical_sha256,
            "signed_receipt_schema_version": self.signed_receipt_schema_version,
            "signed_receipt_id": self.signed_receipt_id,
            "signed_receipt_canonical_sha256": (
                self.signed_receipt_canonical_sha256
            ),
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_canonical_sha256": self.checkpoint_canonical_sha256,
            "checkpoint_index": self.checkpoint_index,
            "head_schema_version": self.head_schema_version,
            "head_id": self.head_id,
            "head_canonical_sha256": self.head_canonical_sha256,
            "head_last_checkpoint_id": self.head_last_checkpoint_id,
            "head_checkpoint_count": self.head_checkpoint_count,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_DEPENDENCY_REFERENCE_BYTES:
            _fail(FAIL_INPUT_V1, "CAS dependency reference exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "CasStoreDependencyReferenceV1":
        body = _exact_object(value, _CAS_DEPENDENCY_KEYS, "CAS dependency")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            verification_status=body["verification_status"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            signer_role=body["signer_role"],  # type: ignore[arg-type]
            signer_key_id=body["signer_key_id"],  # type: ignore[arg-type]
            signer_key_epoch=_wire_integer(
                body["signer_key_epoch"], "CAS signer key epoch", minimum=1
            ),
            signature_purpose=body["signature_purpose"],  # type: ignore[arg-type]
            signature_domain=body["signature_domain"],  # type: ignore[arg-type]
            bound_artifact_id=body["bound_artifact_id"],  # type: ignore[arg-type]
            bound_artifact_sha256=body["bound_artifact_sha256"],  # type: ignore[arg-type]
            object_id=body["object_id"],  # type: ignore[arg-type]
            object_canonical_sha256=body["object_canonical_sha256"],  # type: ignore[arg-type]
            signed_receipt_schema_version=body["signed_receipt_schema_version"],  # type: ignore[arg-type]
            signed_receipt_id=body["signed_receipt_id"],  # type: ignore[arg-type]
            signed_receipt_canonical_sha256=body["signed_receipt_canonical_sha256"],  # type: ignore[arg-type]
            checkpoint_schema_version=body["checkpoint_schema_version"],  # type: ignore[arg-type]
            checkpoint_id=body["checkpoint_id"],  # type: ignore[arg-type]
            checkpoint_canonical_sha256=body["checkpoint_canonical_sha256"],  # type: ignore[arg-type]
            checkpoint_index=_wire_integer(
                body["checkpoint_index"], "CAS checkpoint index"
            ),
            head_schema_version=body["head_schema_version"],  # type: ignore[arg-type]
            head_id=body["head_id"],  # type: ignore[arg-type]
            head_canonical_sha256=body["head_canonical_sha256"],  # type: ignore[arg-type]
            head_last_checkpoint_id=body["head_last_checkpoint_id"],  # type: ignore[arg-type]
            head_checkpoint_count=_wire_integer(
                body["head_checkpoint_count"],
                "CAS head checkpoint count",
                minimum=1,
            ),
        )

    @classmethod
    def from_canonical_bytes(
        cls, payload: bytes
    ) -> "CasStoreDependencyReferenceV1":
        value = cls.from_mapping(
            _decode(
                payload,
                MAXIMUM_DEPENDENCY_REFERENCE_BYTES,
                "CAS dependency reference",
            )
        )
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "CAS dependency DTO readback differs")
        return value


@dataclass(frozen=True, slots=True)
class LogicalTimeDependencyReferenceV1:
    schema_version: str
    verification_status: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    signer_role: str
    signer_key_id: str
    signer_key_epoch: int
    signature_purpose: str
    token_schema_version: str
    token_id: str
    token_canonical_sha256: str
    token_bound_artifact_id: str
    token_bound_artifact_sha256: str
    token_sequence: int

    def __post_init__(self) -> None:
        if type(self) is not LogicalTimeDependencyReferenceV1:
            _fail(FAIL_INPUT_V1, "logical-time dependency must use exact DTO")
        for value, expected, label in (
            (self.schema_version, LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1, "logical-time dependency schema"),
            (self.verification_status, LOGICAL_TIME_DEPENDENCY_VERIFICATION_STATUS_V1, "logical-time verification status"),
            (self.signer_role, LOGICAL_TIME_ROLE_V1, "logical-time signer role"),
            (self.signature_purpose, LOGICAL_TIME_SIGNATURE_PURPOSE_V1, "logical-time signature purpose"),
            (self.token_schema_version, LOGICAL_TIME_TOKEN_SCHEMA_V1, "logical-time token schema"),
        ):
            _exact_literal(value, expected, label)
        _prefixed_id(
            self.authority_profile_id,
            AUTHORITY_PROFILE_PREFIX_V2,
            "logical-time dependency authority profile ID",
        )
        _prefixed_id(
            self.signer_registry_id,
            SIGNER_REGISTRY_PREFIX_V2,
            "logical-time dependency signer registry ID",
        )
        _identifier(
            self.attempt_context_id, "logical-time dependency attempt context ID"
        )
        _prefixed_id(
            self.signer_key_id,
            ROLE_KEY_PREFIX_V1,
            "logical-time signer role-key ID",
        )
        if _wire_integer(
            self.signer_key_epoch, "logical-time dependency key epoch", minimum=1
        ) != LOGICAL_TIME_KEY_EPOCH_V1:
            _fail(FAIL_BINDING_V1, "logical-time dependency key epoch differs")
        _prefixed_id(
            self.token_id, LOGICAL_TIME_TOKEN_ID_PREFIX_V1, "logical-time token ID"
        )
        _sha256_address(
            self.token_canonical_sha256,
            "logical-time token canonical SHA-256",
        )
        _artifact_id(
            self.token_bound_artifact_id, "logical-time bound artifact ID"
        )
        _sha256_address(
            self.token_bound_artifact_sha256,
            "logical-time bound artifact canonical SHA-256",
        )
        _wire_integer(self.token_sequence, "logical-time token sequence")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verification_status": self.verification_status,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "attempt_context_id": self.attempt_context_id,
            "signer_role": self.signer_role,
            "signer_key_id": self.signer_key_id,
            "signer_key_epoch": self.signer_key_epoch,
            "signature_purpose": self.signature_purpose,
            "token_schema_version": self.token_schema_version,
            "token_id": self.token_id,
            "token_canonical_sha256": self.token_canonical_sha256,
            "token_bound_artifact_id": self.token_bound_artifact_id,
            "token_bound_artifact_sha256": self.token_bound_artifact_sha256,
            "token_sequence": self.token_sequence,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_DEPENDENCY_REFERENCE_BYTES:
            _fail(FAIL_INPUT_V1, "logical-time dependency exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(
        cls, value: object
    ) -> "LogicalTimeDependencyReferenceV1":
        body = _exact_object(
            value, _LOGICAL_TIME_DEPENDENCY_KEYS, "logical-time dependency"
        )
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            verification_status=body["verification_status"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            signer_role=body["signer_role"],  # type: ignore[arg-type]
            signer_key_id=body["signer_key_id"],  # type: ignore[arg-type]
            signer_key_epoch=_wire_integer(
                body["signer_key_epoch"],
                "logical-time signer key epoch",
                minimum=1,
            ),
            signature_purpose=body["signature_purpose"],  # type: ignore[arg-type]
            token_schema_version=body["token_schema_version"],  # type: ignore[arg-type]
            token_id=body["token_id"],  # type: ignore[arg-type]
            token_canonical_sha256=body["token_canonical_sha256"],  # type: ignore[arg-type]
            token_bound_artifact_id=body["token_bound_artifact_id"],  # type: ignore[arg-type]
            token_bound_artifact_sha256=body["token_bound_artifact_sha256"],  # type: ignore[arg-type]
            token_sequence=_wire_integer(
                body["token_sequence"], "logical-time token sequence"
            ),
        )

    @classmethod
    def from_canonical_bytes(
        cls, payload: bytes
    ) -> "LogicalTimeDependencyReferenceV1":
        value = cls.from_mapping(
            _decode(
                payload,
                MAXIMUM_DEPENDENCY_REFERENCE_BYTES,
                "logical-time dependency reference",
            )
        )
        if value.canonical_bytes() != payload:
            _fail(
                FAIL_CANONICAL_V1,
                "logical-time dependency DTO readback differs",
            )
        return value


@dataclass(frozen=True, slots=True)
class OwnerStatusDecisionV1:
    schema_version: str
    ledger_id: str
    authority_profile_id: str
    signer_registry_id: str
    governance_key_id: str
    governance_key_epoch: int
    attempt_context_id: str
    owner_authorization_binding: Mapping[str, object]
    owner_authorization_payload_sha256: str
    sequence: int
    event_type: str
    prior_effective_status: str
    effective_status: str
    revocation_count: int
    reason_code: str
    decision_nonce: str
    previous_event_id: str
    previous_head_id: str
    previous_checkpoint_id: str
    previous_complete_event_set_root: str
    decision_id: str

    def __post_init__(self) -> None:
        if type(self) is not OwnerStatusDecisionV1:
            _fail(FAIL_INPUT_V1, "owner-status decision must use exact DTO")
        _exact_literal(
            self.schema_version,
            OWNER_STATUS_DECISION_SCHEMA_V1,
            "owner-status decision schema",
        )
        _prefixed_id(self.ledger_id, LEDGER_ID_PREFIX_V1, "status ledger ID")
        common = _validate_common_bindings(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            governance_key_id=self.governance_key_id,
            governance_key_epoch=self.governance_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        if self.ledger_id != _ledger_id_from_common(common):
            _fail(FAIL_BINDING_V1, "decision ledger identity differs")
        binding = _validate_owner_binding(self.owner_authorization_binding)
        object.__setattr__(
            self,
            "owner_authorization_binding",
            MappingProxyType(dict(binding)),
        )
        if (
            self.owner_authorization_payload_sha256
            != "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
        ):
            _fail(FAIL_BINDING_V1, "decision owner payload SHA-256 differs")
        sequence = _wire_integer(self.sequence, "status decision sequence")
        revocations = _wire_integer(
            self.revocation_count, "status decision revocation count"
        )
        _upper_token(self.reason_code, "status decision reason code")
        _hex32(self.decision_nonce, "status decision nonce")
        _prefixed_id(
            self.previous_event_id, EVENT_ID_PREFIX_V1, "decision previous event ID"
        )
        _prefixed_id(
            self.previous_head_id, HEAD_ID_PREFIX_V1, "decision previous head ID"
        )
        _prefixed_id(
            self.previous_checkpoint_id,
            CHECKPOINT_ID_PREFIX_V1,
            "decision previous checkpoint ID",
        )
        _sha256_address(
            self.previous_complete_event_set_root,
            "decision previous complete-event-set root",
        )
        if sequence == 0:
            if (
                self.event_type != EVENT_INITIAL_ACTIVE_V1
                or self.prior_effective_status != STATUS_NONE_V1
                or self.effective_status != STATUS_ACTIVE_V1
                or revocations != 0
                or self.previous_event_id != GENESIS_EVENT_ID_V1
                or self.previous_checkpoint_id != GENESIS_CHECKPOINT_ID_V1
                or self.previous_complete_event_set_root
                != GENESIS_COMPLETE_EVENT_SET_ROOT_V1
            ):
                _fail(FAIL_LEDGER_V1, "initial ACTIVE transition differs")
        elif sequence == 1:
            if (
                self.event_type != EVENT_REVOKED_V1
                or self.prior_effective_status != STATUS_ACTIVE_V1
                or self.effective_status != STATUS_REVOKED_V1
                or revocations != 1
                or self.previous_event_id == GENESIS_EVENT_ID_V1
                or self.previous_checkpoint_id == GENESIS_CHECKPOINT_ID_V1
                or self.previous_complete_event_set_root
                == GENESIS_COMPLETE_EVENT_SET_ROOT_V1
            ):
                _fail(FAIL_LEDGER_V1, "terminal REVOKED transition differs")
        else:
            _fail(
                FAIL_LEDGER_V1,
                "schema/1 permits exactly initial ACTIVE then terminal REVOKED",
            )
        expected = _content_id(
            DECISION_ID_PREFIX_V1, _DECISION_ID_DOMAIN, self._id_subject()
        )
        if self.decision_id != expected:
            _fail(FAIL_CONTENT_ID_V1, "owner-status decision ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ledger_id": self.ledger_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "governance_key_id": self.governance_key_id,
            "governance_key_epoch": self.governance_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "owner_authorization_binding": dict(self.owner_authorization_binding),
            "owner_authorization_payload_sha256": (
                self.owner_authorization_payload_sha256
            ),
            "sequence": self.sequence,
            "event_type": self.event_type,
            "prior_effective_status": self.prior_effective_status,
            "effective_status": self.effective_status,
            "revocation_count": self.revocation_count,
            "reason_code": self.reason_code,
            "decision_nonce": self.decision_nonce,
            "previous_event_id": self.previous_event_id,
            "previous_head_id": self.previous_head_id,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "previous_complete_event_set_root": (
                self.previous_complete_event_set_root
            ),
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "decision_id": self.decision_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_DECISION_BYTES:
            _fail(FAIL_INPUT_V1, "owner-status decision exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "OwnerStatusDecisionV1":
        body = _exact_object(value, _DECISION_KEYS, "owner-status decision")
        owner_binding = _validate_owner_binding(body["owner_authorization_binding"])
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            ledger_id=body["ledger_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            governance_key_id=body["governance_key_id"],  # type: ignore[arg-type]
            governance_key_epoch=_wire_integer(
                body["governance_key_epoch"],
                "decision governance key epoch",
                minimum=1,
            ),
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            owner_authorization_binding=dict(owner_binding),
            owner_authorization_payload_sha256=body["owner_authorization_payload_sha256"],  # type: ignore[arg-type]
            sequence=_wire_integer(body["sequence"], "decision sequence"),
            event_type=body["event_type"],  # type: ignore[arg-type]
            prior_effective_status=body["prior_effective_status"],  # type: ignore[arg-type]
            effective_status=body["effective_status"],  # type: ignore[arg-type]
            revocation_count=_wire_integer(
                body["revocation_count"], "decision revocation count"
            ),
            reason_code=body["reason_code"],  # type: ignore[arg-type]
            decision_nonce=body["decision_nonce"],  # type: ignore[arg-type]
            previous_event_id=body["previous_event_id"],  # type: ignore[arg-type]
            previous_head_id=body["previous_head_id"],  # type: ignore[arg-type]
            previous_checkpoint_id=body["previous_checkpoint_id"],  # type: ignore[arg-type]
            previous_complete_event_set_root=body["previous_complete_event_set_root"],  # type: ignore[arg-type]
            decision_id=body["decision_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "OwnerStatusDecisionV1":
        value = cls.from_mapping(
            _decode(payload, MAXIMUM_DECISION_BYTES, "owner-status decision")
        )
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "owner-status decision DTO readback differs")
        return value


def _logical_time_token_unsigned_body(
    token: dict[str, object]
) -> dict[str, object]:
    return {
        "schema_version": token["schema_version"],
        "authority_profile_id": token["authority_profile_id"],
        "signer_registry_id": token["signer_registry_id"],
        "logical_time_key_id": token["logical_time_key_id"],
        "logical_time_key_epoch": token["logical_time_key_epoch"],
        "attempt_context_id": token["attempt_context_id"],
        "request_id": token["request_id"],
        "request_nonce": token["request_nonce"],
        "signature_purpose": token["signature_purpose"],
        "purpose": token["purpose"],
        "bound_artifact_id": token["bound_artifact_id"],
        "bound_artifact_sha256": token["bound_artifact_sha256"],
        "prior_token_id": token["prior_token_id"],
        "sequence": token["sequence"],
        "algorithm": token["algorithm"],
    }


def _verify_logical_time_token_v1(
    payload: bytes,
    reference: LogicalTimeDependencyReferenceV1,
    *,
    logical_time_public_key: bytes,
) -> dict[str, object]:
    """Verify the frozen token/1 wire and Ed25519 signature literally."""

    token = _exact_object(
        _decode(
            payload, MAXIMUM_LOGICAL_TIME_TOKEN_BYTES, "signed logical-time token"
        ),
        _LOGICAL_TIME_TOKEN_KEYS,
        "signed logical-time token",
    )
    if (
        token["schema_version"] != LOGICAL_TIME_TOKEN_SCHEMA_V1
        or token["signature_purpose"] != LOGICAL_TIME_SIGNATURE_PURPOSE_V1
        or token["purpose"] != LOGICAL_TIME_EVENT_PURPOSE_V1
        or token["algorithm"] != ED25519_ALGORITHM_V1
    ):
        _fail(FAIL_BINDING_V1, "logical-time token schema/purpose/algorithm differs")
    profile = _prefixed_id(
        token["authority_profile_id"],
        AUTHORITY_PROFILE_PREFIX_V2,
        "token authority profile ID",
    )
    registry = _prefixed_id(
        token["signer_registry_id"],
        SIGNER_REGISTRY_PREFIX_V2,
        "token signer registry ID",
    )
    key_id = _prefixed_id(
        token["logical_time_key_id"],
        ROLE_KEY_PREFIX_V1,
        "token logical-time role-key ID",
    )
    epoch = _wire_integer(
        token["logical_time_key_epoch"], "token logical-time key epoch", minimum=1
    )
    if epoch != LOGICAL_TIME_KEY_EPOCH_V1:
        _fail(FAIL_BINDING_V1, "logical-time token key epoch differs")
    attempt = _identifier(token["attempt_context_id"], "token attempt context ID")
    request_id = _prefixed_id(
        token["request_id"],
        LOGICAL_TIME_REQUEST_ID_PREFIX_V1,
        "logical-time token request ID",
    )
    request_nonce = _hex32(token["request_nonce"], "logical-time request nonce")
    _artifact_id(token["bound_artifact_id"], "logical-time bound artifact ID")
    _sha256_address(
        token["bound_artifact_sha256"],
        "logical-time bound artifact canonical SHA-256",
    )
    _prefixed_id(
        token["prior_token_id"],
        LOGICAL_TIME_TOKEN_ID_PREFIX_V1,
        "logical-time prior token ID",
    )
    sequence = _wire_integer(token["sequence"], "logical-time sequence")
    request_subject = {
        "schema_version": LOGICAL_TIME_REQUEST_SCHEMA_V1,
        "authority_profile_id": profile,
        "signer_registry_id": registry,
        "logical_time_key_id": key_id,
        "logical_time_key_epoch": epoch,
        "attempt_context_id": attempt,
        "request_nonce": request_nonce,
        "signature_purpose": token["signature_purpose"],
        "purpose": token["purpose"],
        "bound_artifact_id": token["bound_artifact_id"],
        "bound_artifact_sha256": token["bound_artifact_sha256"],
        "prior_token_id": token["prior_token_id"],
        "next_sequence": sequence,
    }
    expected_request_id = _content_id(
        LOGICAL_TIME_REQUEST_ID_PREFIX_V1,
        _LOGICAL_TIME_REQUEST_ID_DOMAIN,
        request_subject,
    )
    if request_id != expected_request_id:
        _fail(FAIL_CONTENT_ID_V1, "logical-time request ID differs")
    signature_hex = _hex64(token["signature_hex"], "logical-time signature")
    id_subject = {
        **_logical_time_token_unsigned_body(token),
        "signature_hex": signature_hex,
    }
    expected_token_id = _content_id(
        LOGICAL_TIME_TOKEN_ID_PREFIX_V1,
        _LOGICAL_TIME_TOKEN_ID_DOMAIN,
        id_subject,
    )
    if token["token_id"] != expected_token_id:
        _fail(FAIL_CONTENT_ID_V1, "logical-time token ID differs")
    expected_key_id = _derive_logical_time_role_key_id_v1(logical_time_public_key)
    if key_id != expected_key_id:
        _fail(FAIL_BINDING_V1, "logical-time public key/role-key ID differs")
    expected_reference = (
        reference.authority_profile_id,
        reference.signer_registry_id,
        reference.attempt_context_id,
        reference.signer_key_id,
        reference.signer_key_epoch,
        reference.token_id,
        reference.token_canonical_sha256,
        reference.token_bound_artifact_id,
        reference.token_bound_artifact_sha256,
        reference.token_sequence,
    )
    observed_reference = (
        profile,
        registry,
        attempt,
        key_id,
        epoch,
        expected_token_id,
        _sha256(payload),
        token["bound_artifact_id"],
        token["bound_artifact_sha256"],
        sequence,
    )
    if observed_reference != expected_reference:
        _fail(FAIL_BINDING_V1, "logical-time token/reference exact binding differs")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(
            FAIL_SIGNATURE_BACKEND_V1,
            "cryptography Ed25519 verification backend is unavailable",
        )
    preimage_wrapper = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
        "artifact_kind": "logical_time",
        "role": LOGICAL_TIME_ROLE_V1,
        "purpose": LOGICAL_TIME_SIGNATURE_PURPOSE_V1,
        "key_epoch": epoch,
        "signer_key_id": key_id,
        "body": _logical_time_token_unsigned_body(token),
    }
    try:
        public_key = _Ed25519PublicKey.from_public_bytes(logical_time_public_key)
        public_key.verify(
            bytes.fromhex(signature_hex),
            _LOGICAL_TIME_TOKEN_SIGNATURE_DOMAIN + _canonical(preimage_wrapper),
        )
    except (ValueError, _InvalidSignature) as exc:
        raise Phase2BStep5OwnerStatusErrorV1(
            FAIL_SIGNATURE_V1, "logical-time Ed25519 token signature is invalid"
        ) from exc
    return token


def _require_dependency_graph(
    decision: OwnerStatusDecisionV1,
    cas: CasStoreDependencyReferenceV1,
    logical_time: LogicalTimeDependencyReferenceV1,
) -> None:
    common = (
        decision.authority_profile_id,
        decision.signer_registry_id,
        decision.attempt_context_id,
    )
    if (
        common
        != (
            cas.authority_profile_id,
            cas.signer_registry_id,
            cas.attempt_context_id,
        )
        or common
        != (
            logical_time.authority_profile_id,
            logical_time.signer_registry_id,
            logical_time.attempt_context_id,
        )
    ):
        _fail(FAIL_BINDING_V1, "dependency authority/registry/attempt graph forks")
    decision_sha = _sha256(decision.canonical_bytes())
    if (
        cas.bound_artifact_id != decision.decision_id
        or cas.bound_artifact_sha256 != decision_sha
        or cas.object_id != decision_sha
        or cas.object_canonical_sha256 != decision_sha
        or logical_time.token_bound_artifact_id != decision.decision_id
        or logical_time.token_bound_artifact_sha256 != decision_sha
        or logical_time.token_sequence != decision.sequence
    ):
        _fail(FAIL_BINDING_V1, "CAS/time dependencies do not bind exact decision")


@dataclass(frozen=True, slots=True)
class SignedOwnerStatusEventV1:
    schema_version: str
    signature_purpose: str
    signature_domain: str
    authority_profile_id: str
    signer_registry_id: str
    governance_key_id: str
    governance_key_epoch: int
    governance_public_key_hex: str
    attempt_context_id: str
    decision: OwnerStatusDecisionV1
    decision_canonical_sha256: str
    cas_store_dependency: CasStoreDependencyReferenceV1
    logical_time_dependency: LogicalTimeDependencyReferenceV1
    algorithm: str
    signature_hex: str
    event_id: str
    claim_boundary: str

    def __post_init__(self) -> None:
        if type(self) is not SignedOwnerStatusEventV1:
            _fail(FAIL_INPUT_V1, "signed owner-status event must use exact DTO")
        for value, expected, label in (
            (self.schema_version, SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1, "signed-event schema"),
            (self.signature_purpose, SIGN_OWNER_STATUS_EVENT_PURPOSE_V1, "signed-event purpose"),
            (self.signature_domain, GOVERNANCE_SIGNATURE_DOMAIN_V1, "signed-event domain"),
            (self.algorithm, ED25519_ALGORITHM_V1, "signed-event algorithm"),
            (self.claim_boundary, EVENT_CLAIM_BOUNDARY_V1, "signed-event claim boundary"),
        ):
            _exact_literal(value, expected, label)
        common = _validate_common_bindings(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            governance_key_id=self.governance_key_id,
            governance_key_epoch=self.governance_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        if type(self.decision) is not OwnerStatusDecisionV1:
            _fail(FAIL_INPUT_V1, "event decision must use exact DTO")
        self.decision.__post_init__()
        self.cas_store_dependency.__post_init__()
        self.logical_time_dependency.__post_init__()
        if common != (
            self.decision.authority_profile_id,
            self.decision.signer_registry_id,
            self.decision.governance_key_id,
            self.decision.governance_key_epoch,
            self.decision.attempt_context_id,
        ):
            _fail(FAIL_BINDING_V1, "event/decision common binding differs")
        _require_dependency_graph(
            self.decision,
            self.cas_store_dependency,
            self.logical_time_dependency,
        )
        if self.decision_canonical_sha256 != _sha256(
            self.decision.canonical_bytes()
        ):
            _fail(FAIL_BINDING_V1, "event decision canonical SHA-256 differs")
        public_hex = _hex32(
            self.governance_public_key_hex, "governance public-key encoding"
        )
        expected_key_id = derive_required_governance_role_key_id_v1(
            bytes.fromhex(public_hex), key_epoch=self.governance_key_epoch
        )
        if self.governance_key_id != expected_key_id:
            _fail(FAIL_BINDING_V1, "governance public key/role-key ID differs")
        _hex64(self.signature_hex, "governance Ed25519 signature")
        expected_id = _content_id(
            EVENT_ID_PREFIX_V1, _EVENT_ID_DOMAIN, self._id_subject()
        )
        if self.event_id != expected_id:
            _fail(FAIL_CONTENT_ID_V1, "signed owner-status event ID differs")

    def _unsigned_body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signature_purpose": self.signature_purpose,
            "signature_domain": self.signature_domain,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "governance_key_id": self.governance_key_id,
            "governance_key_epoch": self.governance_key_epoch,
            "governance_public_key_hex": self.governance_public_key_hex,
            "attempt_context_id": self.attempt_context_id,
            "decision": self.decision.to_mapping(),
            "decision_canonical_sha256": self.decision_canonical_sha256,
            "cas_store_dependency": self.cas_store_dependency.to_mapping(),
            "logical_time_dependency": self.logical_time_dependency.to_mapping(),
            "algorithm": self.algorithm,
            "claim_boundary": self.claim_boundary,
        }

    def _id_subject(self) -> dict[str, object]:
        return {**self._unsigned_body(), "signature_hex": self.signature_hex}

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "event_id": self.event_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_EVENT_BYTES:
            _fail(FAIL_INPUT_V1, "signed owner-status event exceeds byte bound")
        return payload

    def signature_preimage(self) -> bytes:
        wrapper = {
            "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
            "artifact_kind": "owner_authorization_status",
            "role": GOVERNANCE_ROLE_V1,
            "purpose": SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
            "key_epoch": self.governance_key_epoch,
            "signer_key_id": self.governance_key_id,
            "body": self._unsigned_body(),
        }
        return _GOVERNANCE_SIGNATURE_DOMAIN + _canonical(wrapper)

    @classmethod
    def from_mapping(cls, value: object) -> "SignedOwnerStatusEventV1":
        body = _exact_object(value, _EVENT_KEYS, "signed owner-status event")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            signature_purpose=body["signature_purpose"],  # type: ignore[arg-type]
            signature_domain=body["signature_domain"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            governance_key_id=body["governance_key_id"],  # type: ignore[arg-type]
            governance_key_epoch=_wire_integer(
                body["governance_key_epoch"],
                "event governance key epoch",
                minimum=1,
            ),
            governance_public_key_hex=body["governance_public_key_hex"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            decision=OwnerStatusDecisionV1.from_mapping(body["decision"]),
            decision_canonical_sha256=body["decision_canonical_sha256"],  # type: ignore[arg-type]
            cas_store_dependency=CasStoreDependencyReferenceV1.from_mapping(
                body["cas_store_dependency"]
            ),
            logical_time_dependency=LogicalTimeDependencyReferenceV1.from_mapping(
                body["logical_time_dependency"]
            ),
            algorithm=body["algorithm"],  # type: ignore[arg-type]
            signature_hex=body["signature_hex"],  # type: ignore[arg-type]
            event_id=body["event_id"],  # type: ignore[arg-type]
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(
        cls, payload: bytes
    ) -> "SignedOwnerStatusEventV1":
        value = cls.from_mapping(
            _decode(payload, MAXIMUM_EVENT_BYTES, "signed owner-status event")
        )
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "signed event DTO readback differs")
        return value


def verify_signed_owner_status_event_v1(
    event_payload: bytes,
    *,
    governance_public_key: bytes,
    logical_time_token_payload: bytes,
    logical_time_public_key: bytes,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
    expected_governance_key_id: str,
    expected_governance_key_epoch: int,
    expected_attempt_context_id: str,
) -> SignedOwnerStatusEventV1:
    """Verify one event from bytes with concrete Ed25519 backends only."""

    event = SignedOwnerStatusEventV1.from_canonical_bytes(event_payload)
    expected_common = _validate_common_bindings(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        governance_key_id=expected_governance_key_id,
        governance_key_epoch=expected_governance_key_epoch,
        attempt_context_id=expected_attempt_context_id,
    )
    observed_common = (
        event.authority_profile_id,
        event.signer_registry_id,
        event.governance_key_id,
        event.governance_key_epoch,
        event.attempt_context_id,
    )
    if observed_common != expected_common:
        _fail(FAIL_BINDING_V1, "signed event differs from expected graph pins")
    if type(governance_public_key) is not bytes or len(governance_public_key) != 32:
        _fail(FAIL_INPUT_V1, "governance public key must be exact raw 32 bytes")
    if derive_required_governance_role_key_id_v1(governance_public_key) != (
        event.governance_key_id
    ):
        _fail(FAIL_BINDING_V1, "governance public key differs from event role key")
    logical_time_token = _verify_logical_time_token_v1(
        logical_time_token_payload,
        event.logical_time_dependency,
        logical_time_public_key=logical_time_public_key,
    )
    if (
        event.decision.sequence == 0
        and logical_time_token["prior_token_id"]
        != LOGICAL_TIME_GENESIS_TOKEN_ID_V1
    ):
        _fail(
            FAIL_LEDGER_V1,
            "initial ACTIVE logical-time token does not follow genesis",
        )
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(
            FAIL_SIGNATURE_BACKEND_V1,
            "cryptography Ed25519 verification backend is unavailable",
        )
    try:
        public_key = _Ed25519PublicKey.from_public_bytes(governance_public_key)
        public_key.verify(
            bytes.fromhex(event.signature_hex), event.signature_preimage()
        )
    except (ValueError, _InvalidSignature) as exc:
        raise Phase2BStep5OwnerStatusErrorV1(
            FAIL_SIGNATURE_V1, "governance owner-status signature is invalid"
        ) from exc
    return event


def _next_complete_event_set_root(
    previous_root: str, event: SignedOwnerStatusEventV1
) -> str:
    _sha256_address(previous_root, "previous complete-event-set root")
    subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-owner-status-complete-event-set-root/1"
        ),
        "previous_complete_event_set_root": previous_root,
        "sequence": event.decision.sequence,
        "event_id": event.event_id,
        "event_canonical_sha256": _sha256(event.canonical_bytes()),
    }
    return _sha256(_COMPLETE_EVENT_SET_ROOT_DOMAIN + _canonical(subject))


@dataclass(frozen=True, slots=True)
class OwnerStatusCheckpointV1:
    schema_version: str
    purpose: str
    ledger_id: str
    authority_profile_id: str
    signer_registry_id: str
    governance_key_id: str
    governance_key_epoch: int
    attempt_context_id: str
    checkpoint_index: int
    previous_head_id: str
    previous_checkpoint_id: str
    previous_event_id: str
    previous_complete_event_set_root: str
    event_id: str
    signed_event_canonical_sha256: str
    signed_event: SignedOwnerStatusEventV1
    complete_event_set_root: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        if type(self) is not OwnerStatusCheckpointV1:
            _fail(FAIL_INPUT_V1, "owner-status checkpoint must use exact DTO")
        _exact_literal(
            self.schema_version,
            OWNER_STATUS_CHECKPOINT_SCHEMA_V1,
            "owner-status checkpoint schema",
        )
        _exact_literal(
            self.purpose, CHECKPOINT_PURPOSE_V1, "owner-status checkpoint purpose"
        )
        _prefixed_id(self.ledger_id, LEDGER_ID_PREFIX_V1, "checkpoint ledger ID")
        common = _validate_common_bindings(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            governance_key_id=self.governance_key_id,
            governance_key_epoch=self.governance_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        if self.ledger_id != _ledger_id_from_common(common):
            _fail(FAIL_BINDING_V1, "checkpoint ledger identity differs")
        index = _wire_integer(
            self.checkpoint_index, "owner-status checkpoint index", maximum=1
        )
        _prefixed_id(
            self.previous_head_id,
            HEAD_ID_PREFIX_V1,
            "checkpoint previous head ID",
        )
        _prefixed_id(
            self.previous_checkpoint_id,
            CHECKPOINT_ID_PREFIX_V1,
            "checkpoint previous checkpoint ID",
        )
        _prefixed_id(
            self.previous_event_id,
            EVENT_ID_PREFIX_V1,
            "checkpoint previous event ID",
        )
        _sha256_address(
            self.previous_complete_event_set_root,
            "checkpoint previous complete-event-set root",
        )
        _prefixed_id(self.event_id, EVENT_ID_PREFIX_V1, "checkpoint event ID")
        _sha256_address(
            self.signed_event_canonical_sha256,
            "checkpoint signed-event canonical SHA-256",
        )
        _sha256_address(
            self.complete_event_set_root,
            "checkpoint complete-event-set root",
        )
        if type(self.signed_event) is not SignedOwnerStatusEventV1:
            _fail(FAIL_INPUT_V1, "checkpoint event must use exact DTO")
        self.signed_event.__post_init__()
        event = self.signed_event
        if (
            common
            != (
                event.authority_profile_id,
                event.signer_registry_id,
                event.governance_key_id,
                event.governance_key_epoch,
                event.attempt_context_id,
            )
            or event.decision.ledger_id != self.ledger_id
            or event.decision.sequence != index
            or event.decision.previous_head_id != self.previous_head_id
            or event.decision.previous_checkpoint_id
            != self.previous_checkpoint_id
            or event.decision.previous_event_id != self.previous_event_id
            or event.decision.previous_complete_event_set_root
            != self.previous_complete_event_set_root
            or event.event_id != self.event_id
            or _sha256(event.canonical_bytes())
            != self.signed_event_canonical_sha256
            or _next_complete_event_set_root(
                self.previous_complete_event_set_root, event
            )
            != self.complete_event_set_root
        ):
            _fail(FAIL_BINDING_V1, "checkpoint/event exact linkage differs")
        expected_id = _content_id(
            CHECKPOINT_ID_PREFIX_V1,
            _CHECKPOINT_ID_DOMAIN,
            self._id_subject(),
        )
        if self.checkpoint_id != expected_id:
            _fail(FAIL_CONTENT_ID_V1, "owner-status checkpoint ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "ledger_id": self.ledger_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "governance_key_id": self.governance_key_id,
            "governance_key_epoch": self.governance_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "checkpoint_index": self.checkpoint_index,
            "previous_head_id": self.previous_head_id,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "previous_event_id": self.previous_event_id,
            "previous_complete_event_set_root": (
                self.previous_complete_event_set_root
            ),
            "event_id": self.event_id,
            "signed_event_canonical_sha256": (
                self.signed_event_canonical_sha256
            ),
            "signed_event": self.signed_event.to_mapping(),
            "complete_event_set_root": self.complete_event_set_root,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "checkpoint_id": self.checkpoint_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_CHECKPOINT_BYTES:
            _fail(FAIL_INPUT_V1, "owner-status checkpoint exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "OwnerStatusCheckpointV1":
        body = _exact_object(value, _CHECKPOINT_KEYS, "owner-status checkpoint")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            ledger_id=body["ledger_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            governance_key_id=body["governance_key_id"],  # type: ignore[arg-type]
            governance_key_epoch=_wire_integer(
                body["governance_key_epoch"],
                "checkpoint governance key epoch",
                minimum=1,
            ),
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            checkpoint_index=_wire_integer(
                body["checkpoint_index"], "checkpoint index", maximum=1
            ),
            previous_head_id=body["previous_head_id"],  # type: ignore[arg-type]
            previous_checkpoint_id=body["previous_checkpoint_id"],  # type: ignore[arg-type]
            previous_event_id=body["previous_event_id"],  # type: ignore[arg-type]
            previous_complete_event_set_root=body["previous_complete_event_set_root"],  # type: ignore[arg-type]
            event_id=body["event_id"],  # type: ignore[arg-type]
            signed_event_canonical_sha256=body["signed_event_canonical_sha256"],  # type: ignore[arg-type]
            signed_event=SignedOwnerStatusEventV1.from_mapping(
                body["signed_event"]
            ),
            complete_event_set_root=body["complete_event_set_root"],  # type: ignore[arg-type]
            checkpoint_id=body["checkpoint_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(
        cls, payload: bytes
    ) -> "OwnerStatusCheckpointV1":
        value = cls.from_mapping(
            _decode(payload, MAXIMUM_CHECKPOINT_BYTES, "owner-status checkpoint")
        )
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "checkpoint DTO readback differs")
        return value


@dataclass(frozen=True, slots=True)
class OwnerStatusHeadV1:
    schema_version: str
    purpose: str
    ledger_id: str
    authority_profile_id: str
    signer_registry_id: str
    governance_key_id: str
    governance_key_epoch: int
    attempt_context_id: str
    event_count: int
    next_sequence: int
    last_event_id: str
    last_checkpoint_id: str
    complete_event_set_root: str
    latest_effective_status: str
    revocation_count: int
    terminal: bool
    head_id: str

    def __post_init__(self) -> None:
        if type(self) is not OwnerStatusHeadV1:
            _fail(FAIL_INPUT_V1, "owner-status head must use exact DTO")
        _exact_literal(
            self.schema_version,
            OWNER_STATUS_HEAD_SCHEMA_V1,
            "owner-status head schema",
        )
        _exact_literal(self.purpose, HEAD_PURPOSE_V1, "owner-status head purpose")
        _prefixed_id(self.ledger_id, LEDGER_ID_PREFIX_V1, "head ledger ID")
        common = _validate_common_bindings(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            governance_key_id=self.governance_key_id,
            governance_key_epoch=self.governance_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        if self.ledger_id != _ledger_id_from_common(common):
            _fail(FAIL_BINDING_V1, "head ledger identity differs")
        count = _wire_integer(self.event_count, "head event count", maximum=2)
        next_sequence = _wire_integer(
            self.next_sequence, "head next sequence", maximum=2
        )
        if count != next_sequence:
            _fail(FAIL_LEDGER_V1, "head count and next sequence differ")
        _prefixed_id(self.last_event_id, EVENT_ID_PREFIX_V1, "head last event ID")
        _prefixed_id(
            self.last_checkpoint_id,
            CHECKPOINT_ID_PREFIX_V1,
            "head last checkpoint ID",
        )
        _sha256_address(
            self.complete_event_set_root, "head complete-event-set root"
        )
        revocations = _wire_integer(
            self.revocation_count, "head revocation count", maximum=1
        )
        if type(self.terminal) is not bool:
            _fail(FAIL_SCHEMA_V1, "head terminal flag must be exact bool")
        if count == 0:
            expected_state = (
                GENESIS_EVENT_ID_V1,
                GENESIS_CHECKPOINT_ID_V1,
                GENESIS_COMPLETE_EVENT_SET_ROOT_V1,
                STATUS_NONE_V1,
                0,
                False,
            )
        elif count == 1:
            if (
                self.last_event_id == GENESIS_EVENT_ID_V1
                or self.last_checkpoint_id == GENESIS_CHECKPOINT_ID_V1
                or self.complete_event_set_root
                == GENESIS_COMPLETE_EVENT_SET_ROOT_V1
            ):
                _fail(FAIL_LEDGER_V1, "ACTIVE head retains a genesis sentinel")
            expected_state = (
                self.last_event_id,
                self.last_checkpoint_id,
                self.complete_event_set_root,
                STATUS_ACTIVE_V1,
                0,
                False,
            )
        else:
            if (
                self.last_event_id == GENESIS_EVENT_ID_V1
                or self.last_checkpoint_id == GENESIS_CHECKPOINT_ID_V1
                or self.complete_event_set_root
                == GENESIS_COMPLETE_EVENT_SET_ROOT_V1
            ):
                _fail(FAIL_LEDGER_V1, "REVOKED head retains a genesis sentinel")
            expected_state = (
                self.last_event_id,
                self.last_checkpoint_id,
                self.complete_event_set_root,
                STATUS_REVOKED_V1,
                1,
                True,
            )
        observed_state = (
            self.last_event_id,
            self.last_checkpoint_id,
            self.complete_event_set_root,
            self.latest_effective_status,
            revocations,
            self.terminal,
        )
        if observed_state != expected_state:
            _fail(FAIL_LEDGER_V1, "owner-status head state differs")
        expected_id = _content_id(
            HEAD_ID_PREFIX_V1, _HEAD_ID_DOMAIN, self._id_subject()
        )
        if self.head_id != expected_id:
            _fail(FAIL_CONTENT_ID_V1, "owner-status head ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "ledger_id": self.ledger_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "governance_key_id": self.governance_key_id,
            "governance_key_epoch": self.governance_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "event_count": self.event_count,
            "next_sequence": self.next_sequence,
            "last_event_id": self.last_event_id,
            "last_checkpoint_id": self.last_checkpoint_id,
            "complete_event_set_root": self.complete_event_set_root,
            "latest_effective_status": self.latest_effective_status,
            "revocation_count": self.revocation_count,
            "terminal": self.terminal,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "head_id": self.head_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_HEAD_BYTES:
            _fail(FAIL_INPUT_V1, "owner-status head exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "OwnerStatusHeadV1":
        body = _exact_object(value, _HEAD_KEYS, "owner-status head")
        if type(body["terminal"]) is not bool:
            _fail(FAIL_SCHEMA_V1, "head terminal field must be exact bool")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            ledger_id=body["ledger_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            governance_key_id=body["governance_key_id"],  # type: ignore[arg-type]
            governance_key_epoch=_wire_integer(
                body["governance_key_epoch"],
                "head governance key epoch",
                minimum=1,
            ),
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            event_count=_wire_integer(
                body["event_count"], "head event count", maximum=2
            ),
            next_sequence=_wire_integer(
                body["next_sequence"], "head next sequence", maximum=2
            ),
            last_event_id=body["last_event_id"],  # type: ignore[arg-type]
            last_checkpoint_id=body["last_checkpoint_id"],  # type: ignore[arg-type]
            complete_event_set_root=body["complete_event_set_root"],  # type: ignore[arg-type]
            latest_effective_status=body["latest_effective_status"],  # type: ignore[arg-type]
            revocation_count=_wire_integer(
                body["revocation_count"], "head revocation count", maximum=1
            ),
            terminal=body["terminal"],
            head_id=body["head_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "OwnerStatusHeadV1":
        value = cls.from_mapping(
            _decode(payload, MAXIMUM_HEAD_BYTES, "owner-status head")
        )
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "owner-status head DTO readback differs")
        return value


@dataclass(frozen=True, slots=True)
class OwnerStatusProjectionV1:
    schema_version: str
    claim_level: str
    ledger_id: str
    authority_profile_id: str
    signer_registry_id: str
    governance_key_id: str
    governance_key_epoch: int
    attempt_context_id: str
    owner_authorization_binding: Mapping[str, object]
    owner_authorization_payload_sha256: str
    event_count: int
    latest_sequence: int
    latest_effective_status: str
    revocation_count: int
    last_event_id: str
    last_checkpoint_id: str
    head_id: str
    complete_event_set_root: str
    governance_signatures_verified: bool
    governance_status_purpose_authorized_by_pinned_registry: bool
    owner_signature_authenticated: bool
    logical_time_token_signatures_verified: bool
    store_dependency_candidate_references_exact: bool
    store_dependencies_verified: bool
    latest_authority_status_verified: bool
    r01_satisfied: bool
    readiness_node_transition_issued: bool
    execution_authorized: bool
    authority_evidence: bool
    projection_id: str

    def __post_init__(self) -> None:
        if type(self) is not OwnerStatusProjectionV1:
            _fail(FAIL_INPUT_V1, "owner-status projection must use exact DTO")
        _exact_literal(
            self.schema_version,
            OWNER_STATUS_PROJECTION_SCHEMA_V1,
            "owner-status projection schema",
        )
        _exact_literal(
            self.claim_level,
            PROJECTION_CLAIM_LEVEL_V1,
            "owner-status projection claim level",
        )
        _prefixed_id(self.ledger_id, LEDGER_ID_PREFIX_V1, "projection ledger ID")
        common = _validate_common_bindings(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            governance_key_id=self.governance_key_id,
            governance_key_epoch=self.governance_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        if self.ledger_id != _ledger_id_from_common(common):
            _fail(FAIL_BINDING_V1, "projection ledger identity differs")
        binding = _validate_owner_binding(self.owner_authorization_binding)
        object.__setattr__(
            self,
            "owner_authorization_binding",
            MappingProxyType(dict(binding)),
        )
        if self.owner_authorization_payload_sha256 != (
            "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
        ):
            _fail(FAIL_BINDING_V1, "projection owner payload SHA differs")
        count = _wire_integer(
            self.event_count, "projection event count", minimum=1, maximum=2
        )
        sequence = _wire_integer(
            self.latest_sequence, "projection latest sequence", maximum=1
        )
        if sequence != count - 1:
            _fail(FAIL_LEDGER_V1, "projection count/latest sequence differs")
        revocations = _wire_integer(
            self.revocation_count, "projection revocation count", maximum=1
        )
        expected_status = STATUS_ACTIVE_V1 if count == 1 else STATUS_REVOKED_V1
        expected_revocations = 0 if count == 1 else 1
        if (
            self.latest_effective_status != expected_status
            or revocations != expected_revocations
        ):
            _fail(FAIL_LEDGER_V1, "projection latest status differs")
        _prefixed_id(
            self.last_event_id, EVENT_ID_PREFIX_V1, "projection last event ID"
        )
        _prefixed_id(
            self.last_checkpoint_id,
            CHECKPOINT_ID_PREFIX_V1,
            "projection last checkpoint ID",
        )
        _prefixed_id(self.head_id, HEAD_ID_PREFIX_V1, "projection head ID")
        _sha256_address(
            self.complete_event_set_root,
            "projection complete-event-set root",
        )
        expected_flags = (
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        )
        observed_flags = (
            self.governance_signatures_verified,
            self.governance_status_purpose_authorized_by_pinned_registry,
            self.owner_signature_authenticated,
            self.logical_time_token_signatures_verified,
            self.store_dependency_candidate_references_exact,
            self.store_dependencies_verified,
            self.latest_authority_status_verified,
            self.r01_satisfied,
            self.readiness_node_transition_issued,
            self.execution_authorized,
            self.authority_evidence,
        )
        if observed_flags != expected_flags or any(
            type(flag) is not bool for flag in observed_flags
        ):
            _fail(FAIL_CLAIM_BOUNDARY_V1, "projection safety flags differ")
        expected_id = _content_id(
            PROJECTION_ID_PREFIX_V1,
            _PROJECTION_ID_DOMAIN,
            self._id_subject(),
        )
        if self.projection_id != expected_id:
            _fail(FAIL_CONTENT_ID_V1, "owner-status projection ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "claim_level": self.claim_level,
            "ledger_id": self.ledger_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "governance_key_id": self.governance_key_id,
            "governance_key_epoch": self.governance_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "owner_authorization_binding": dict(self.owner_authorization_binding),
            "owner_authorization_payload_sha256": (
                self.owner_authorization_payload_sha256
            ),
            "event_count": self.event_count,
            "latest_sequence": self.latest_sequence,
            "latest_effective_status": self.latest_effective_status,
            "revocation_count": self.revocation_count,
            "last_event_id": self.last_event_id,
            "last_checkpoint_id": self.last_checkpoint_id,
            "head_id": self.head_id,
            "complete_event_set_root": self.complete_event_set_root,
            "governance_signatures_verified": self.governance_signatures_verified,
            "governance_status_purpose_authorized_by_pinned_registry": (
                self.governance_status_purpose_authorized_by_pinned_registry
            ),
            "owner_signature_authenticated": self.owner_signature_authenticated,
            "logical_time_token_signatures_verified": (
                self.logical_time_token_signatures_verified
            ),
            "store_dependency_candidate_references_exact": (
                self.store_dependency_candidate_references_exact
            ),
            "store_dependencies_verified": self.store_dependencies_verified,
            "latest_authority_status_verified": (
                self.latest_authority_status_verified
            ),
            "r01_satisfied": self.r01_satisfied,
            "readiness_node_transition_issued": (
                self.readiness_node_transition_issued
            ),
            "execution_authorized": self.execution_authorized,
            "authority_evidence": self.authority_evidence,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "projection_id": self.projection_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_PROJECTION_BYTES:
            _fail(FAIL_INPUT_V1, "owner-status projection exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "OwnerStatusProjectionV1":
        body = _exact_object(value, _PROJECTION_KEYS, "owner-status projection")
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
        if any(type(body[field]) is not bool for field in boolean_fields):
            _fail(FAIL_SCHEMA_V1, "projection bool field type differs")
        owner_binding = _validate_owner_binding(body["owner_authorization_binding"])
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            claim_level=body["claim_level"],  # type: ignore[arg-type]
            ledger_id=body["ledger_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            governance_key_id=body["governance_key_id"],  # type: ignore[arg-type]
            governance_key_epoch=_wire_integer(
                body["governance_key_epoch"],
                "projection governance key epoch",
                minimum=1,
            ),
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            owner_authorization_binding=dict(owner_binding),
            owner_authorization_payload_sha256=body["owner_authorization_payload_sha256"],  # type: ignore[arg-type]
            event_count=_wire_integer(
                body["event_count"],
                "projection event count",
                minimum=1,
                maximum=2,
            ),
            latest_sequence=_wire_integer(
                body["latest_sequence"],
                "projection latest sequence",
                maximum=1,
            ),
            latest_effective_status=body["latest_effective_status"],  # type: ignore[arg-type]
            revocation_count=_wire_integer(
                body["revocation_count"],
                "projection revocation count",
                maximum=1,
            ),
            last_event_id=body["last_event_id"],  # type: ignore[arg-type]
            last_checkpoint_id=body["last_checkpoint_id"],  # type: ignore[arg-type]
            head_id=body["head_id"],  # type: ignore[arg-type]
            complete_event_set_root=body["complete_event_set_root"],  # type: ignore[arg-type]
            governance_signatures_verified=body["governance_signatures_verified"],
            governance_status_purpose_authorized_by_pinned_registry=body["governance_status_purpose_authorized_by_pinned_registry"],
            owner_signature_authenticated=body["owner_signature_authenticated"],
            logical_time_token_signatures_verified=body["logical_time_token_signatures_verified"],
            store_dependency_candidate_references_exact=body["store_dependency_candidate_references_exact"],
            store_dependencies_verified=body["store_dependencies_verified"],
            latest_authority_status_verified=body["latest_authority_status_verified"],
            r01_satisfied=body["r01_satisfied"],
            readiness_node_transition_issued=body["readiness_node_transition_issued"],
            execution_authorized=body["execution_authorized"],
            authority_evidence=body["authority_evidence"],
            projection_id=body["projection_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(
        cls, payload: bytes
    ) -> "OwnerStatusProjectionV1":
        value = cls.from_mapping(
            _decode(payload, MAXIMUM_PROJECTION_BYTES, "owner-status projection")
        )
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "owner-status projection DTO readback differs")
        return value


def _head_from_state(
    *,
    common: tuple[str, str, str, int, str],
    event_count: int,
    last_event_id: str,
    last_checkpoint_id: str,
    complete_event_set_root: str,
    latest_effective_status: str,
    revocation_count: int,
    terminal: bool,
) -> OwnerStatusHeadV1:
    subject = {
        "schema_version": OWNER_STATUS_HEAD_SCHEMA_V1,
        "purpose": HEAD_PURPOSE_V1,
        "ledger_id": _ledger_id_from_common(common),
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
    return OwnerStatusHeadV1(
        **subject,
        head_id=_content_id(HEAD_ID_PREFIX_V1, _HEAD_ID_DOMAIN, subject),
    )


def build_owner_status_genesis_head_v1(
    *,
    authority_profile_id: str,
    signer_registry_id: str,
    governance_key_id: str,
    governance_key_epoch: int,
    attempt_context_id: str,
    governance_public_key: bytes,
    owner_authorization_payload: bytes,
) -> bytes:
    """Build the immutable empty head; this grants no active status."""

    verify_exact_owner_authorization_payload_v1(owner_authorization_payload)
    common = _validate_common_bindings(
        authority_profile_id=authority_profile_id,
        signer_registry_id=signer_registry_id,
        governance_key_id=governance_key_id,
        governance_key_epoch=governance_key_epoch,
        attempt_context_id=attempt_context_id,
    )
    if derive_required_governance_role_key_id_v1(
        governance_public_key, key_epoch=governance_key_epoch
    ) != governance_key_id:
        _fail(FAIL_BINDING_V1, "genesis governance key/role-key ID differs")
    head = _head_from_state(
        common=common,
        event_count=0,
        last_event_id=GENESIS_EVENT_ID_V1,
        last_checkpoint_id=GENESIS_CHECKPOINT_ID_V1,
        complete_event_set_root=GENESIS_COMPLETE_EVENT_SET_ROOT_V1,
        latest_effective_status=STATUS_NONE_V1,
        revocation_count=0,
        terminal=False,
    )
    return OwnerStatusHeadV1.from_canonical_bytes(
        head.canonical_bytes()
    ).canonical_bytes()


def build_owner_status_decision_v1(
    previous_head_payload: bytes,
    *,
    owner_authorization_payload: bytes,
    reason_code: str,
    decision_nonce: str,
) -> bytes:
    """Build the exact next ACTIVE or terminal REVOKED decision from a head."""

    verify_exact_owner_authorization_payload_v1(owner_authorization_payload)
    head = OwnerStatusHeadV1.from_canonical_bytes(previous_head_payload)
    if head.terminal:
        _fail(FAIL_LEDGER_V1, "REVOKED is terminal; no later status is permitted")
    sequence = head.next_sequence
    if sequence == 0:
        event_type = EVENT_INITIAL_ACTIVE_V1
        effective_status = STATUS_ACTIVE_V1
        revocation_count = 0
    elif sequence == 1 and head.latest_effective_status == STATUS_ACTIVE_V1:
        event_type = EVENT_REVOKED_V1
        effective_status = STATUS_REVOKED_V1
        revocation_count = 1
    else:
        _fail(FAIL_LEDGER_V1, "head does not admit one exact next transition")
    subject = {
        "schema_version": OWNER_STATUS_DECISION_SCHEMA_V1,
        "ledger_id": head.ledger_id,
        "authority_profile_id": head.authority_profile_id,
        "signer_registry_id": head.signer_registry_id,
        "governance_key_id": head.governance_key_id,
        "governance_key_epoch": head.governance_key_epoch,
        "attempt_context_id": head.attempt_context_id,
        "owner_authorization_binding": dict(EXACT_OWNER_AUTHORIZATION_BINDING_V1),
        "owner_authorization_payload_sha256": (
            "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
        ),
        "sequence": sequence,
        "event_type": event_type,
        "prior_effective_status": head.latest_effective_status,
        "effective_status": effective_status,
        "revocation_count": revocation_count,
        "reason_code": reason_code,
        "decision_nonce": decision_nonce,
        "previous_event_id": head.last_event_id,
        "previous_head_id": head.head_id,
        "previous_checkpoint_id": head.last_checkpoint_id,
        "previous_complete_event_set_root": head.complete_event_set_root,
    }
    decision = OwnerStatusDecisionV1(
        **subject,
        decision_id=_content_id(
            DECISION_ID_PREFIX_V1, _DECISION_ID_DOMAIN, subject
        ),
    )
    return OwnerStatusDecisionV1.from_canonical_bytes(
        decision.canonical_bytes()
    ).canonical_bytes()


def build_cas_store_dependency_reference_v1(
    *,
    decision_payload: bytes,
    retention_session_id: str,
    chain_id: str,
    signer_key_id: str,
    signed_receipt_schema_version: str,
    signed_receipt_id: str,
    signed_receipt_canonical_sha256: str,
    checkpoint_schema_version: str,
    checkpoint_id: str,
    checkpoint_canonical_sha256: str,
    checkpoint_index: int,
    head_schema_version: str,
    head_id: str,
    head_canonical_sha256: str,
    head_last_checkpoint_id: str,
    head_checkpoint_count: int,
) -> bytes:
    """Build an exact CAS candidate reference without claiming authentication."""

    decision = OwnerStatusDecisionV1.from_canonical_bytes(decision_payload)
    decision_sha = _sha256(decision_payload)
    value = CasStoreDependencyReferenceV1(
        schema_version=CAS_DEPENDENCY_REFERENCE_SCHEMA_V1,
        verification_status=CAS_DEPENDENCY_VERIFICATION_STATUS_V1,
        authority_profile_id=decision.authority_profile_id,
        signer_registry_id=decision.signer_registry_id,
        attempt_context_id=decision.attempt_context_id,
        retention_session_id=retention_session_id,
        chain_id=chain_id,
        signer_role=CAS_STORE_ROLE_V1,
        signer_key_id=signer_key_id,
        signer_key_epoch=CAS_STORE_KEY_EPOCH_V1,
        signature_purpose=CAS_STORE_RECEIPT_PURPOSE_V1,
        signature_domain=CAS_STORE_SIGNATURE_DOMAIN_V1,
        bound_artifact_id=decision.decision_id,
        bound_artifact_sha256=decision_sha,
        object_id=decision_sha,
        object_canonical_sha256=decision_sha,
        signed_receipt_schema_version=signed_receipt_schema_version,
        signed_receipt_id=signed_receipt_id,
        signed_receipt_canonical_sha256=signed_receipt_canonical_sha256,
        checkpoint_schema_version=checkpoint_schema_version,
        checkpoint_id=checkpoint_id,
        checkpoint_canonical_sha256=checkpoint_canonical_sha256,
        checkpoint_index=checkpoint_index,
        head_schema_version=head_schema_version,
        head_id=head_id,
        head_canonical_sha256=head_canonical_sha256,
        head_last_checkpoint_id=head_last_checkpoint_id,
        head_checkpoint_count=head_checkpoint_count,
    )
    return CasStoreDependencyReferenceV1.from_canonical_bytes(
        value.canonical_bytes()
    ).canonical_bytes()


def build_logical_time_dependency_reference_v1(
    *,
    decision_payload: bytes,
    logical_time_token_payload: bytes,
    logical_time_public_key: bytes,
) -> bytes:
    """Build and authenticate a literal token/1 dependency reference."""

    decision = OwnerStatusDecisionV1.from_canonical_bytes(decision_payload)
    token = _exact_object(
        _decode(
            logical_time_token_payload,
            MAXIMUM_LOGICAL_TIME_TOKEN_BYTES,
            "signed logical-time token",
        ),
        _LOGICAL_TIME_TOKEN_KEYS,
        "signed logical-time token",
    )
    reference = LogicalTimeDependencyReferenceV1(
        schema_version=LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1,
        verification_status=LOGICAL_TIME_DEPENDENCY_VERIFICATION_STATUS_V1,
        authority_profile_id=decision.authority_profile_id,
        signer_registry_id=decision.signer_registry_id,
        attempt_context_id=decision.attempt_context_id,
        signer_role=LOGICAL_TIME_ROLE_V1,
        signer_key_id=token["logical_time_key_id"],  # type: ignore[arg-type]
        signer_key_epoch=_wire_integer(
            token["logical_time_key_epoch"], "logical-time token key epoch"
        ),
        signature_purpose=LOGICAL_TIME_SIGNATURE_PURPOSE_V1,
        token_schema_version=LOGICAL_TIME_TOKEN_SCHEMA_V1,
        token_id=token["token_id"],  # type: ignore[arg-type]
        token_canonical_sha256=_sha256(logical_time_token_payload),
        token_bound_artifact_id=decision.decision_id,
        token_bound_artifact_sha256=_sha256(decision_payload),
        token_sequence=_wire_integer(token["sequence"], "logical-time sequence"),
    )
    _verify_logical_time_token_v1(
        logical_time_token_payload,
        reference,
        logical_time_public_key=logical_time_public_key,
    )
    return LogicalTimeDependencyReferenceV1.from_canonical_bytes(
        reference.canonical_bytes()
    ).canonical_bytes()


def _governance_signature_preimage(unsigned_body: dict[str, object]) -> bytes:
    wrapper = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
        "artifact_kind": "owner_authorization_status",
        "role": GOVERNANCE_ROLE_V1,
        "purpose": SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
        "key_epoch": unsigned_body["governance_key_epoch"],
        "signer_key_id": unsigned_body["governance_key_id"],
        "body": unsigned_body,
    }
    return _GOVERNANCE_SIGNATURE_DOMAIN + _canonical(wrapper)


def _issue_signed_event_service_side(
    *,
    decision: OwnerStatusDecisionV1,
    cas_dependency: CasStoreDependencyReferenceV1,
    logical_time_dependency: LogicalTimeDependencyReferenceV1,
    logical_time_token_payload: bytes,
    governance_public_key: bytes,
    logical_time_public_key: bytes,
    private_key: object,
) -> SignedOwnerStatusEventV1:
    if _Ed25519PrivateKey is None or _serialization is None:
        _fail(
            FAIL_SIGNATURE_BACKEND_V1,
            "cryptography Ed25519 signing backend is unavailable",
        )
    if not isinstance(private_key, _Ed25519PrivateKey):
        _fail(
            FAIL_INPUT_V1,
            "service issuance requires an Ed25519 private-key object",
        )
    observed_public = private_key.public_key().public_bytes(
        encoding=_serialization.Encoding.Raw,
        format=_serialization.PublicFormat.Raw,
    )
    if observed_public != governance_public_key:
        _fail(FAIL_BINDING_V1, "governance private/public key binding differs")
    _require_dependency_graph(decision, cas_dependency, logical_time_dependency)
    _verify_logical_time_token_v1(
        logical_time_token_payload,
        logical_time_dependency,
        logical_time_public_key=logical_time_public_key,
    )
    unsigned_body = {
        "schema_version": SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1,
        "signature_purpose": SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
        "signature_domain": GOVERNANCE_SIGNATURE_DOMAIN_V1,
        "authority_profile_id": decision.authority_profile_id,
        "signer_registry_id": decision.signer_registry_id,
        "governance_key_id": decision.governance_key_id,
        "governance_key_epoch": decision.governance_key_epoch,
        "governance_public_key_hex": governance_public_key.hex(),
        "attempt_context_id": decision.attempt_context_id,
        "decision": decision.to_mapping(),
        "decision_canonical_sha256": _sha256(decision.canonical_bytes()),
        "cas_store_dependency": cas_dependency.to_mapping(),
        "logical_time_dependency": logical_time_dependency.to_mapping(),
        "algorithm": ED25519_ALGORITHM_V1,
        "claim_boundary": EVENT_CLAIM_BOUNDARY_V1,
    }
    signature = private_key.sign(_governance_signature_preimage(unsigned_body))
    if type(signature) is not bytes or len(signature) != 64:
        _fail(FAIL_SIGNATURE_V1, "governance signer returned invalid bytes")
    id_subject = {**unsigned_body, "signature_hex": signature.hex()}
    event = SignedOwnerStatusEventV1(
        schema_version=SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1,
        signature_purpose=SIGN_OWNER_STATUS_EVENT_PURPOSE_V1,
        signature_domain=GOVERNANCE_SIGNATURE_DOMAIN_V1,
        authority_profile_id=decision.authority_profile_id,
        signer_registry_id=decision.signer_registry_id,
        governance_key_id=decision.governance_key_id,
        governance_key_epoch=decision.governance_key_epoch,
        governance_public_key_hex=governance_public_key.hex(),
        attempt_context_id=decision.attempt_context_id,
        decision=decision,
        decision_canonical_sha256=_sha256(decision.canonical_bytes()),
        cas_store_dependency=cas_dependency,
        logical_time_dependency=logical_time_dependency,
        algorithm=ED25519_ALGORITHM_V1,
        signature_hex=signature.hex(),
        event_id=_content_id(EVENT_ID_PREFIX_V1, _EVENT_ID_DOMAIN, id_subject),
        claim_boundary=EVENT_CLAIM_BOUNDARY_V1,
    )
    return verify_signed_owner_status_event_v1(
        event.canonical_bytes(),
        governance_public_key=governance_public_key,
        logical_time_token_payload=logical_time_token_payload,
        logical_time_public_key=logical_time_public_key,
        expected_authority_profile_id=decision.authority_profile_id,
        expected_signer_registry_id=decision.signer_registry_id,
        expected_governance_key_id=decision.governance_key_id,
        expected_governance_key_epoch=decision.governance_key_epoch,
        expected_attempt_context_id=decision.attempt_context_id,
    )


def _checkpoint_from_event(
    previous_head: OwnerStatusHeadV1,
    event: SignedOwnerStatusEventV1,
) -> OwnerStatusCheckpointV1:
    next_root = _next_complete_event_set_root(
        previous_head.complete_event_set_root, event
    )
    subject = {
        "schema_version": OWNER_STATUS_CHECKPOINT_SCHEMA_V1,
        "purpose": CHECKPOINT_PURPOSE_V1,
        "ledger_id": previous_head.ledger_id,
        "authority_profile_id": previous_head.authority_profile_id,
        "signer_registry_id": previous_head.signer_registry_id,
        "governance_key_id": previous_head.governance_key_id,
        "governance_key_epoch": previous_head.governance_key_epoch,
        "attempt_context_id": previous_head.attempt_context_id,
        "checkpoint_index": event.decision.sequence,
        "previous_head_id": previous_head.head_id,
        "previous_checkpoint_id": previous_head.last_checkpoint_id,
        "previous_event_id": previous_head.last_event_id,
        "previous_complete_event_set_root": (
            previous_head.complete_event_set_root
        ),
        "event_id": event.event_id,
        "signed_event_canonical_sha256": _sha256(event.canonical_bytes()),
        "signed_event": event.to_mapping(),
        "complete_event_set_root": next_root,
    }
    return OwnerStatusCheckpointV1(
        schema_version=OWNER_STATUS_CHECKPOINT_SCHEMA_V1,
        purpose=CHECKPOINT_PURPOSE_V1,
        ledger_id=previous_head.ledger_id,
        authority_profile_id=previous_head.authority_profile_id,
        signer_registry_id=previous_head.signer_registry_id,
        governance_key_id=previous_head.governance_key_id,
        governance_key_epoch=previous_head.governance_key_epoch,
        attempt_context_id=previous_head.attempt_context_id,
        checkpoint_index=event.decision.sequence,
        previous_head_id=previous_head.head_id,
        previous_checkpoint_id=previous_head.last_checkpoint_id,
        previous_event_id=previous_head.last_event_id,
        previous_complete_event_set_root=previous_head.complete_event_set_root,
        event_id=event.event_id,
        signed_event_canonical_sha256=_sha256(event.canonical_bytes()),
        signed_event=event,
        complete_event_set_root=next_root,
        checkpoint_id=_content_id(
            CHECKPOINT_ID_PREFIX_V1, _CHECKPOINT_ID_DOMAIN, subject
        ),
    )


def _next_head_from_checkpoint(
    previous_head: OwnerStatusHeadV1,
    checkpoint: OwnerStatusCheckpointV1,
) -> OwnerStatusHeadV1:
    event = checkpoint.signed_event
    common = (
        previous_head.authority_profile_id,
        previous_head.signer_registry_id,
        previous_head.governance_key_id,
        previous_head.governance_key_epoch,
        previous_head.attempt_context_id,
    )
    return _head_from_state(
        common=common,
        event_count=previous_head.event_count + 1,
        last_event_id=event.event_id,
        last_checkpoint_id=checkpoint.checkpoint_id,
        complete_event_set_root=checkpoint.complete_event_set_root,
        latest_effective_status=event.decision.effective_status,
        revocation_count=event.decision.revocation_count,
        terminal=event.decision.effective_status == STATUS_REVOKED_V1,
    )


def _common_tuple(value: object) -> tuple[str, str, str, int, str]:
    return (
        object.__getattribute__(value, "authority_profile_id"),
        object.__getattribute__(value, "signer_registry_id"),
        object.__getattribute__(value, "governance_key_id"),
        object.__getattribute__(value, "governance_key_epoch"),
        object.__getattribute__(value, "attempt_context_id"),
    )


def _verify_transition(
    previous_head: OwnerStatusHeadV1,
    checkpoint: OwnerStatusCheckpointV1,
    next_head: OwnerStatusHeadV1,
    *,
    governance_public_key: bytes,
    logical_time_token_payload: bytes,
    logical_time_public_key: bytes,
) -> None:
    previous_head.__post_init__()
    checkpoint.__post_init__()
    next_head.__post_init__()
    event = verify_signed_owner_status_event_v1(
        checkpoint.signed_event.canonical_bytes(),
        governance_public_key=governance_public_key,
        logical_time_token_payload=logical_time_token_payload,
        logical_time_public_key=logical_time_public_key,
        expected_authority_profile_id=previous_head.authority_profile_id,
        expected_signer_registry_id=previous_head.signer_registry_id,
        expected_governance_key_id=previous_head.governance_key_id,
        expected_governance_key_epoch=previous_head.governance_key_epoch,
        expected_attempt_context_id=previous_head.attempt_context_id,
    )
    if not (
        previous_head.ledger_id == checkpoint.ledger_id == next_head.ledger_id
        and _common_tuple(previous_head)
        == _common_tuple(checkpoint)
        == _common_tuple(event)
        == _common_tuple(next_head)
    ):
        _fail(FAIL_BINDING_V1, "status transition common graph forks")
    expected_index = previous_head.event_count
    if (
        previous_head.terminal
        or checkpoint.checkpoint_index != expected_index
        or event.decision.sequence != previous_head.next_sequence
        or event.decision.previous_head_id != previous_head.head_id
        or event.decision.previous_checkpoint_id
        != previous_head.last_checkpoint_id
        or event.decision.previous_event_id != previous_head.last_event_id
        or event.decision.previous_complete_event_set_root
        != previous_head.complete_event_set_root
        or checkpoint.previous_head_id != previous_head.head_id
        or checkpoint.previous_checkpoint_id != previous_head.last_checkpoint_id
        or checkpoint.previous_event_id != previous_head.last_event_id
        or checkpoint.previous_complete_event_set_root
        != previous_head.complete_event_set_root
    ):
        _fail(FAIL_LEDGER_V1, "checkpoint is not the exact next chain element")
    expected_next = _next_head_from_checkpoint(previous_head, checkpoint)
    if next_head != expected_next:
        _fail(FAIL_LEDGER_V1, "next head does not exact-commit checkpoint")


def verify_owner_status_ledger_transition_v1(
    previous_head_payload: bytes,
    checkpoint_payload: bytes,
    next_head_payload: bytes,
    *,
    governance_public_key: bytes,
    logical_time_token_payload: bytes,
    logical_time_public_key: bytes,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
    expected_governance_key_id: str,
    expected_governance_key_epoch: int,
    expected_attempt_context_id: str,
) -> OwnerStatusHeadV1:
    """Verify one bytes-only transition; no filesystem state is trusted."""

    previous = OwnerStatusHeadV1.from_canonical_bytes(previous_head_payload)
    checkpoint = OwnerStatusCheckpointV1.from_canonical_bytes(checkpoint_payload)
    next_head = OwnerStatusHeadV1.from_canonical_bytes(next_head_payload)
    expected = _validate_common_bindings(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        governance_key_id=expected_governance_key_id,
        governance_key_epoch=expected_governance_key_epoch,
        attempt_context_id=expected_attempt_context_id,
    )
    if _common_tuple(previous) != expected:
        _fail(FAIL_BINDING_V1, "status transition differs from expected pins")
    _verify_transition(
        previous,
        checkpoint,
        next_head,
        governance_public_key=governance_public_key,
        logical_time_token_payload=logical_time_token_payload,
        logical_time_public_key=logical_time_public_key,
    )
    return next_head


def _projection_from_head(head: OwnerStatusHeadV1) -> OwnerStatusProjectionV1:
    if head.event_count < 1:
        _fail(FAIL_LEDGER_V1, "empty genesis has no latest-status projection")
    subject = {
        "schema_version": OWNER_STATUS_PROJECTION_SCHEMA_V1,
        "claim_level": PROJECTION_CLAIM_LEVEL_V1,
        "ledger_id": head.ledger_id,
        "authority_profile_id": head.authority_profile_id,
        "signer_registry_id": head.signer_registry_id,
        "governance_key_id": head.governance_key_id,
        "governance_key_epoch": head.governance_key_epoch,
        "attempt_context_id": head.attempt_context_id,
        "owner_authorization_binding": dict(EXACT_OWNER_AUTHORIZATION_BINDING_V1),
        "owner_authorization_payload_sha256": (
            "sha256:" + OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
        ),
        "event_count": head.event_count,
        "latest_sequence": head.event_count - 1,
        "latest_effective_status": head.latest_effective_status,
        "revocation_count": head.revocation_count,
        "last_event_id": head.last_event_id,
        "last_checkpoint_id": head.last_checkpoint_id,
        "head_id": head.head_id,
        "complete_event_set_root": head.complete_event_set_root,
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
    return OwnerStatusProjectionV1(
        **subject,
        projection_id=_content_id(
            PROJECTION_ID_PREFIX_V1, _PROJECTION_ID_DOMAIN, subject
        ),
    )


def verify_owner_status_ledger_replay_v1(
    genesis_head_payload: bytes,
    checkpoint_payloads_in_sequence: tuple[bytes, ...],
    subsequent_head_payloads_in_sequence: tuple[bytes, ...],
    logical_time_token_payloads_in_sequence: tuple[bytes, ...],
    *,
    owner_authorization_payload: bytes,
    governance_public_key: bytes,
    logical_time_public_key: bytes,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
    expected_governance_key_id: str,
    expected_governance_key_epoch: int,
    expected_attempt_context_id: str,
) -> OwnerStatusProjectionV1:
    """Purely replay the complete ordered event set into a nonauthority view."""

    if (
        type(checkpoint_payloads_in_sequence) is not tuple
        or type(subsequent_head_payloads_in_sequence) is not tuple
        or type(logical_time_token_payloads_in_sequence) is not tuple
    ):
        _fail(FAIL_INPUT_V1, "replay sequences must be exact tuples of bytes")
    count = len(checkpoint_payloads_in_sequence)
    if not 1 <= count <= 2 or not (
        len(subsequent_head_payloads_in_sequence)
        == len(logical_time_token_payloads_in_sequence)
        == count
    ):
        _fail(FAIL_LEDGER_V1, "replay event/head/token cardinality differs")
    if any(
        type(payload) is not bytes
        for payload in (
            checkpoint_payloads_in_sequence
            + subsequent_head_payloads_in_sequence
            + logical_time_token_payloads_in_sequence
        )
    ):
        _fail(FAIL_INPUT_V1, "replay artifacts must be exact bytes")
    verify_exact_owner_authorization_payload_v1(owner_authorization_payload)
    expected_common = _validate_common_bindings(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        governance_key_id=expected_governance_key_id,
        governance_key_epoch=expected_governance_key_epoch,
        attempt_context_id=expected_attempt_context_id,
    )
    if derive_required_governance_role_key_id_v1(
        governance_public_key, key_epoch=expected_governance_key_epoch
    ) != expected_governance_key_id:
        _fail(FAIL_BINDING_V1, "replay governance public key differs")
    expected_genesis_payload = build_owner_status_genesis_head_v1(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        governance_key_id=expected_governance_key_id,
        governance_key_epoch=expected_governance_key_epoch,
        attempt_context_id=expected_attempt_context_id,
        governance_public_key=governance_public_key,
        owner_authorization_payload=owner_authorization_payload,
    )
    if genesis_head_payload != expected_genesis_payload:
        _fail(FAIL_LEDGER_V1, "replay genesis bytes differ from exact pins")
    previous = OwnerStatusHeadV1.from_canonical_bytes(genesis_head_payload)
    if _common_tuple(previous) != expected_common:
        _fail(FAIL_BINDING_V1, "replay genesis graph differs")

    decision_nonces: set[str] = set()
    logical_token_ids: set[str] = set()
    logical_request_nonces: set[str] = set()
    cas_receipt_ids: set[str] = set()
    cas_checkpoint_ids: set[str] = set()
    previous_logical_sequence: int | None = None
    previous_logical_token_id: str | None = None
    previous_cas_checkpoint_index: int | None = None
    frozen_cas_context: tuple[str, str, str, int] | None = None
    frozen_logical_time_key: tuple[str, int] | None = None
    for index in range(count):
        checkpoint = OwnerStatusCheckpointV1.from_canonical_bytes(
            checkpoint_payloads_in_sequence[index]
        )
        next_head = OwnerStatusHeadV1.from_canonical_bytes(
            subsequent_head_payloads_in_sequence[index]
        )
        token_payload = logical_time_token_payloads_in_sequence[index]
        _verify_transition(
            previous,
            checkpoint,
            next_head,
            governance_public_key=governance_public_key,
            logical_time_token_payload=token_payload,
            logical_time_public_key=logical_time_public_key,
        )
        event = checkpoint.signed_event
        logical_time_token = _verify_logical_time_token_v1(
            token_payload,
            event.logical_time_dependency,
            logical_time_public_key=logical_time_public_key,
        )
        nonce = event.decision.decision_nonce
        token_id = event.logical_time_dependency.token_id
        request_nonce = logical_time_token["request_nonce"]
        receipt_id = event.cas_store_dependency.signed_receipt_id
        cas_checkpoint_id = event.cas_store_dependency.checkpoint_id
        cas_context = (
            event.cas_store_dependency.retention_session_id,
            event.cas_store_dependency.chain_id,
            event.cas_store_dependency.signer_key_id,
            event.cas_store_dependency.signer_key_epoch,
        )
        logical_key = (
            event.logical_time_dependency.signer_key_id,
            event.logical_time_dependency.signer_key_epoch,
        )
        if (
            nonce in decision_nonces
            or token_id in logical_token_ids
            or request_nonce in logical_request_nonces
            or receipt_id in cas_receipt_ids
            or cas_checkpoint_id in cas_checkpoint_ids
        ):
            _fail(FAIL_LEDGER_V1, "replay dependency or decision replay detected")
        if index == 0:
            frozen_cas_context = cas_context
            frozen_logical_time_key = logical_key
            if (
                event.logical_time_dependency.token_sequence != 0
                or logical_time_token["prior_token_id"]
                != LOGICAL_TIME_GENESIS_TOKEN_ID_V1
            ):
                _fail(
                    FAIL_LEDGER_V1,
                    "initial ACTIVE logical-time token chain differs",
                )
        elif (
            cas_context != frozen_cas_context
            or logical_key != frozen_logical_time_key
            or previous_cas_checkpoint_index is None
            or event.cas_store_dependency.checkpoint_index
            != previous_cas_checkpoint_index + 1
            or previous_logical_sequence is None
            or event.logical_time_dependency.token_sequence
            != previous_logical_sequence + 1
            or previous_logical_token_id is None
            or logical_time_token["prior_token_id"]
            != previous_logical_token_id
        ):
            _fail(
                FAIL_LEDGER_V1,
                "cross-event CAS session/key or logical-time order forks",
            )
        decision_nonces.add(nonce)
        logical_token_ids.add(token_id)
        logical_request_nonces.add(request_nonce)  # type: ignore[arg-type]
        cas_receipt_ids.add(receipt_id)
        cas_checkpoint_ids.add(cas_checkpoint_id)
        previous_cas_checkpoint_index = (
            event.cas_store_dependency.checkpoint_index
        )
        previous_logical_sequence = event.logical_time_dependency.token_sequence
        previous_logical_token_id = token_id
        previous = next_head
    projection = _projection_from_head(previous)
    projection.__post_init__()
    return projection


@dataclass(frozen=True, slots=True)
class OwnerStatusServiceAppendV1:
    decision_id: str
    event_id: str
    checkpoint_id: str
    head_id: str
    complete_event_set_root: str
    signed_event_payload: bytes
    checkpoint_payload: bytes
    head_payload: bytes
    projection_payload: bytes
    mechanics_only: bool = True
    authority_evidence: bool = False
    r01_satisfied: bool = False

    def __post_init__(self) -> None:
        if type(self) is not OwnerStatusServiceAppendV1:
            _fail(FAIL_INPUT_V1, "append result must use exact DTO")
        _prefixed_id(
            self.decision_id, DECISION_ID_PREFIX_V1, "append decision ID"
        )
        _prefixed_id(self.event_id, EVENT_ID_PREFIX_V1, "append event ID")
        _prefixed_id(
            self.checkpoint_id,
            CHECKPOINT_ID_PREFIX_V1,
            "append checkpoint ID",
        )
        _prefixed_id(self.head_id, HEAD_ID_PREFIX_V1, "append head ID")
        _sha256_address(
            self.complete_event_set_root, "append complete-event-set root"
        )
        if (
            type(self.signed_event_payload) is not bytes
            or type(self.checkpoint_payload) is not bytes
            or type(self.head_payload) is not bytes
            or type(self.projection_payload) is not bytes
        ):
            _fail(FAIL_INPUT_V1, "append result payloads must be exact bytes")
        event = SignedOwnerStatusEventV1.from_canonical_bytes(
            self.signed_event_payload
        )
        checkpoint = OwnerStatusCheckpointV1.from_canonical_bytes(
            self.checkpoint_payload
        )
        head = OwnerStatusHeadV1.from_canonical_bytes(self.head_payload)
        projection = OwnerStatusProjectionV1.from_canonical_bytes(
            self.projection_payload
        )
        if (
            event.decision.decision_id != self.decision_id
            or event.event_id != self.event_id
            or checkpoint.checkpoint_id != self.checkpoint_id
            or head.head_id != self.head_id
            or head.complete_event_set_root != self.complete_event_set_root
            or projection.head_id != self.head_id
            or projection.complete_event_set_root
            != self.complete_event_set_root
        ):
            _fail(FAIL_BINDING_V1, "append result artifact linkage differs")
        if (
            self.mechanics_only is not True
            or self.authority_evidence is not False
            or self.r01_satisfied is not False
        ):
            _fail(FAIL_CLAIM_BOUNDARY_V1, "append result claim boundary differs")


def _require_filesystem_support() -> None:
    required = ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        _fail(FAIL_FILESYSTEM_V1, "required no-follow filesystem flags unavailable")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(fd, view[written:])
        if count <= 0:
            _fail(FAIL_FILESYSTEM_V1, "short ledger write")
        written += count


def _read_regular_at(
    directory_fd: int, name: str, maximum_bytes: int
) -> bytes:
    if type(name) is not str or "/" in name or name in {"", ".", ".."}:
        _fail(FAIL_FILESYSTEM_V1, "unsafe ledger entry name")
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise Phase2BStep5OwnerStatusErrorV1(
            FAIL_FILESYSTEM_V1, f"cannot stat ledger record {name}: {exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _fail(FAIL_FILESYSTEM_V1, "ledger record is not one-link regular file")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise Phase2BStep5OwnerStatusErrorV1(
            FAIL_FILESYSTEM_V1, f"cannot open ledger record {name}: {exc}"
        ) from exc
    try:
        held = os.fstat(fd)
        if (
            not _same_file(before, held)
            or not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or not 1 <= held.st_size <= maximum_bytes
        ):
            _fail(FAIL_FILESYSTEM_V1, "held ledger record identity/size differs")
        chunks: list[bytes] = []
        remaining = held.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(FAIL_FILESYSTEM_V1, "ledger record ended before st_size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) != b"":
            _fail(FAIL_FILESYSTEM_V1, "ledger record grew during read")
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        held_after = os.fstat(fd)
        if (
            not _same_file(before, after)
            or not _same_file(held, held_after)
            or after.st_size != held.st_size
            or after.st_nlink != 1
            or held_after.st_nlink != 1
        ):
            _fail(FAIL_FILESYSTEM_V1, "ledger record changed during read")
        payload = b"".join(chunks)
        if len(payload) != held.st_size:
            _fail(FAIL_FILESYSTEM_V1, "ledger read length differs from st_size")
        return payload
    finally:
        os.close(fd)


def _versioned_filename(index: int) -> str:
    value = _wire_integer(index, "versioned ledger index", maximum=2)
    name = f"{value:020d}.json"
    if _VERSIONED_FILENAME_PATTERN.fullmatch(name) is None:
        _fail(FAIL_FILESYSTEM_V1, "versioned ledger filename differs")
    return name


class OwnerStatusLedgerV1:
    """Local service primitive; deployment separation remains unproven."""

    __slots__ = (
        "authority_profile_id",
        "signer_registry_id",
        "governance_key_id",
        "governance_key_epoch",
        "attempt_context_id",
        "ledger_id",
        "governance_public_key",
        "logical_time_public_key",
        "owner_authorization_payload",
        "_root_fd",
        "_heads_fd",
        "_checkpoints_fd",
        "_tokens_fd",
        "_lock_fd",
        "_thread_lock",
        "_publication_nonce",
        "_closed",
    )

    def __init__(
        self,
        root_path: str,
        *,
        authority_profile_id: str,
        signer_registry_id: str,
        governance_key_id: str,
        governance_key_epoch: int,
        attempt_context_id: str,
        governance_public_key: bytes,
        logical_time_public_key: bytes,
        owner_authorization_payload: bytes,
        initialize_empty: bool = False,
    ) -> None:
        _require_filesystem_support()
        if (
            type(root_path) is not str
            or not os.path.isabs(root_path)
            or "\x00" in root_path
        ):
            _fail(FAIL_INPUT_V1, "ledger root must be one absolute path")
        if type(initialize_empty) is not bool:
            _fail(FAIL_INPUT_V1, "initialize_empty must be exact bool")
        verify_exact_owner_authorization_payload_v1(owner_authorization_payload)
        common = _validate_common_bindings(
            authority_profile_id=authority_profile_id,
            signer_registry_id=signer_registry_id,
            governance_key_id=governance_key_id,
            governance_key_epoch=governance_key_epoch,
            attempt_context_id=attempt_context_id,
        )
        if derive_required_governance_role_key_id_v1(
            governance_public_key, key_epoch=governance_key_epoch
        ) != governance_key_id:
            _fail(FAIL_BINDING_V1, "ledger governance public key differs")
        logical_key_id = _derive_logical_time_role_key_id_v1(
            logical_time_public_key
        )
        _prefixed_id(
            logical_key_id, ROLE_KEY_PREFIX_V1, "logical-time role-key ID"
        )
        self.authority_profile_id = common[0]
        self.signer_registry_id = common[1]
        self.governance_key_id = common[2]
        self.governance_key_epoch = common[3]
        self.attempt_context_id = common[4]
        self.ledger_id = _ledger_id_from_common(common)
        self.governance_public_key = bytes(governance_public_key)
        self.logical_time_public_key = bytes(logical_time_public_key)
        self.owner_authorization_payload = bytes(owner_authorization_payload)
        self._heads_fd = -1
        self._checkpoints_fd = -1
        self._tokens_fd = -1
        self._lock_fd = -1
        self._thread_lock = threading.Lock()
        self._publication_nonce = 0
        self._closed = False
        try:
            self._root_fd = os.open(
                root_path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise Phase2BStep5OwnerStatusErrorV1(
                FAIL_FILESYSTEM_V1, f"cannot open owner-status root: {exc}"
            ) from exc
        try:
            root_stat = os.fstat(self._root_fd)
            if not stat.S_ISDIR(root_stat.st_mode):
                _fail(FAIL_FILESYSTEM_V1, "owner-status root is not a directory")
            if initialize_empty:
                self._initialize_layout()
            self._open_layout(allow_genesis_creation=initialize_empty)
            with self._exclusive_lock():
                self._load_chain()
        except BaseException:
            self.close()
            raise

    def _initialize_layout(self) -> None:
        if tuple(os.listdir(self._root_fd)):
            _fail(FAIL_LEDGER_V1, "initial owner-status root is not exactly empty")
        try:
            os.mkdir("heads", 0o700, dir_fd=self._root_fd)
            os.mkdir("checkpoints", 0o700, dir_fd=self._root_fd)
            os.mkdir("logical-time-tokens", 0o700, dir_fd=self._root_fd)
            lock_fd = os.open(
                "ledger.lock",
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._root_fd,
            )
            os.fsync(lock_fd)
            os.close(lock_fd)
            os.fsync(self._root_fd)
        except OSError as exc:
            raise Phase2BStep5OwnerStatusErrorV1(
                FAIL_FILESYSTEM_V1,
                f"exclusive owner-status layout initialization failed: {exc}",
            ) from exc

    def _open_directory_at(self, name: str) -> int:
        try:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self._root_fd,
            )
        except OSError as exc:
            raise Phase2BStep5OwnerStatusErrorV1(
                FAIL_FILESYSTEM_V1, f"cannot open ledger directory {name}: {exc}"
            ) from exc
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            os.close(fd)
            _fail(FAIL_FILESYSTEM_V1, f"ledger entry {name} is not directory")
        return fd

    def _open_layout(self, *, allow_genesis_creation: bool) -> None:
        if set(os.listdir(self._root_fd)) != {
            "heads",
            "checkpoints",
            "logical-time-tokens",
            "ledger.lock",
        }:
            _fail(FAIL_LEDGER_V1, "owner-status root exact entry set differs")
        self._heads_fd = self._open_directory_at("heads")
        self._checkpoints_fd = self._open_directory_at("checkpoints")
        self._tokens_fd = self._open_directory_at("logical-time-tokens")
        try:
            self._lock_fd = os.open(
                "ledger.lock",
                os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self._root_fd,
            )
        except OSError as exc:
            raise Phase2BStep5OwnerStatusErrorV1(
                FAIL_FILESYSTEM_V1, f"cannot open ledger lock: {exc}"
            ) from exc
        lock_stat = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or lock_stat.st_size != 0
        ):
            _fail(FAIL_FILESYSTEM_V1, "ledger lock inode differs")
        if not tuple(os.listdir(self._heads_fd)):
            if tuple(os.listdir(self._checkpoints_fd)) or tuple(
                os.listdir(self._tokens_fd)
            ):
                _fail(FAIL_LEDGER_V1, "empty heads with dependency residue")
            if not allow_genesis_creation:
                _fail(
                    FAIL_LEDGER_V1,
                    "missing genesis head is a fail-closed initialization gap",
                )
            genesis = build_owner_status_genesis_head_v1(
                authority_profile_id=self.authority_profile_id,
                signer_registry_id=self.signer_registry_id,
                governance_key_id=self.governance_key_id,
                governance_key_epoch=self.governance_key_epoch,
                attempt_context_id=self.attempt_context_id,
                governance_public_key=self.governance_public_key,
                owner_authorization_payload=self.owner_authorization_payload,
            )
            self._publish_exclusive(
                self._heads_fd,
                _versioned_filename(0),
                genesis,
                MAXIMUM_HEAD_BYTES,
                "head",
            )

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if self._closed:
            _fail(FAIL_LEDGER_V1, "owner-status ledger is closed")
        with self._thread_lock:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _publish_exclusive(
        self,
        directory_fd: int,
        final_name: str,
        payload: bytes,
        maximum_bytes: int,
        artifact_kind: str,
    ) -> None:
        if type(payload) is not bytes or not 1 <= len(payload) <= maximum_bytes:
            _fail(FAIL_INPUT_V1, "ledger publication payload outside byte bound")
        if artifact_kind not in {"head", "checkpoint", "logical-time-token"}:
            _fail(FAIL_INPUT_V1, "internal ledger artifact kind differs")
        self._publication_nonce += 1
        temporary_name = (
            f".pending-{os.getpid()}-{self._publication_nonce}-{final_name}"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
        )
        temporary_fd: int | None = None
        final_fd: int | None = None
        try:
            temporary_fd = os.open(
                temporary_name, flags, 0o600, dir_fd=directory_fd
            )
            _write_all(temporary_fd, payload)
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            if _read_regular_at(
                directory_fd, temporary_name, maximum_bytes
            ) != payload:
                _fail(FAIL_FILESYSTEM_V1, "staged ledger readback differs")
            final_fd = os.open(final_name, flags, 0o600, dir_fd=directory_fd)
            _write_all(final_fd, payload)
            os.fsync(final_fd)
            os.close(final_fd)
            final_fd = None
            os.fsync(directory_fd)
            observed = _read_regular_at(directory_fd, final_name, maximum_bytes)
            if observed != payload:
                _fail(FAIL_FILESYSTEM_V1, "final ledger readback differs")
            if artifact_kind == "head":
                OwnerStatusHeadV1.from_canonical_bytes(observed)
            elif artifact_kind == "checkpoint":
                OwnerStatusCheckpointV1.from_canonical_bytes(observed)
            else:
                _decode(
                    observed,
                    MAXIMUM_LOGICAL_TIME_TOKEN_BYTES,
                    "persisted logical-time token",
                )
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            if _read_regular_at(directory_fd, final_name, maximum_bytes) != payload:
                _fail(FAIL_FILESYSTEM_V1, "post-fsync ledger readback differs")
        except FileExistsError as exc:
            raise Phase2BStep5OwnerStatusErrorV1(
                FAIL_LEDGER_V1,
                f"exclusive ledger publication collision for {final_name}",
            ) from exc
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            if final_fd is not None:
                os.close(final_fd)
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            # A created final file is never removed on failure.  A partial
            # sequence remains visible and poisons later cardinality/replay.

    def _exact_versioned_names(
        self, directory_fd: int, count: int, label: str
    ) -> tuple[str, ...]:
        names = tuple(sorted(os.listdir(directory_fd)))
        expected = tuple(_versioned_filename(index) for index in range(count))
        if names != expected or any(
            _VERSIONED_FILENAME_PATTERN.fullmatch(name) is None for name in names
        ):
            _fail(FAIL_LEDGER_V1, f"{label} exact versioned filename set differs")
        return names

    def _require_held_layout_identity(self) -> None:
        try:
            for name, fd in (
                ("heads", self._heads_fd),
                ("checkpoints", self._checkpoints_fd),
                ("logical-time-tokens", self._tokens_fd),
            ):
                current = os.stat(
                    name, dir_fd=self._root_fd, follow_symlinks=False
                )
                held = os.fstat(fd)
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or not stat.S_ISDIR(held.st_mode)
                    or not _same_file(current, held)
                ):
                    _fail(
                        FAIL_FILESYSTEM_V1,
                        f"held ledger directory {name} was rebound or replaced",
                    )
            current_lock = os.stat(
                "ledger.lock", dir_fd=self._root_fd, follow_symlinks=False
            )
            held_lock = os.fstat(self._lock_fd)
        except OSError as exc:
            raise Phase2BStep5OwnerStatusErrorV1(
                FAIL_FILESYSTEM_V1, f"cannot audit held ledger layout: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(current_lock.st_mode)
            or not stat.S_ISREG(held_lock.st_mode)
            or not _same_file(current_lock, held_lock)
            or current_lock.st_nlink != 1
            or held_lock.st_nlink != 1
            or current_lock.st_size != 0
            or held_lock.st_size != 0
        ):
            _fail(FAIL_FILESYSTEM_V1, "held ledger lock was rebound or replaced")

    def _load_chain(
        self,
    ) -> tuple[
        tuple[OwnerStatusHeadV1, ...],
        tuple[OwnerStatusCheckpointV1, ...],
        tuple[bytes, ...],
        tuple[bytes, ...],
        tuple[bytes, ...],
        OwnerStatusProjectionV1 | None,
    ]:
        if set(os.listdir(self._root_fd)) != {
            "heads",
            "checkpoints",
            "logical-time-tokens",
            "ledger.lock",
        }:
            _fail(FAIL_LEDGER_V1, "owner-status root exact entry set differs")
        self._require_held_layout_identity()
        head_count = len(tuple(os.listdir(self._heads_fd)))
        checkpoint_count = len(tuple(os.listdir(self._checkpoints_fd)))
        token_count = len(tuple(os.listdir(self._tokens_fd)))
        if (
            head_count < 1
            or head_count > 3
            or checkpoint_count > 2
            or token_count > 2
            or head_count != checkpoint_count + 1
            or token_count != checkpoint_count
        ):
            _fail(
                FAIL_LEDGER_V1,
                "head/checkpoint/token cardinality exposes crash gap or residue",
            )
        head_names = self._exact_versioned_names(
            self._heads_fd, head_count, "owner-status head"
        )
        checkpoint_names = self._exact_versioned_names(
            self._checkpoints_fd,
            checkpoint_count,
            "owner-status checkpoint",
        )
        token_names = self._exact_versioned_names(
            self._tokens_fd, token_count, "logical-time token"
        )
        head_payloads = tuple(
            _read_regular_at(self._heads_fd, name, MAXIMUM_HEAD_BYTES)
            for name in head_names
        )
        checkpoint_payloads = tuple(
            _read_regular_at(
                self._checkpoints_fd, name, MAXIMUM_CHECKPOINT_BYTES
            )
            for name in checkpoint_names
        )
        token_payloads = tuple(
            _read_regular_at(
                self._tokens_fd, name, MAXIMUM_LOGICAL_TIME_TOKEN_BYTES
            )
            for name in token_names
        )
        heads = tuple(
            OwnerStatusHeadV1.from_canonical_bytes(payload)
            for payload in head_payloads
        )
        checkpoints = tuple(
            OwnerStatusCheckpointV1.from_canonical_bytes(payload)
            for payload in checkpoint_payloads
        )
        expected_genesis = build_owner_status_genesis_head_v1(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            governance_key_id=self.governance_key_id,
            governance_key_epoch=self.governance_key_epoch,
            attempt_context_id=self.attempt_context_id,
            governance_public_key=self.governance_public_key,
            owner_authorization_payload=self.owner_authorization_payload,
        )
        if head_payloads[0] != expected_genesis:
            _fail(FAIL_LEDGER_V1, "persisted genesis head differs from pins")
        projection: OwnerStatusProjectionV1 | None = None
        if checkpoint_count:
            projection = verify_owner_status_ledger_replay_v1(
                head_payloads[0],
                checkpoint_payloads,
                head_payloads[1:],
                token_payloads,
                owner_authorization_payload=self.owner_authorization_payload,
                governance_public_key=self.governance_public_key,
                logical_time_public_key=self.logical_time_public_key,
                expected_authority_profile_id=self.authority_profile_id,
                expected_signer_registry_id=self.signer_registry_id,
                expected_governance_key_id=self.governance_key_id,
                expected_governance_key_epoch=self.governance_key_epoch,
                expected_attempt_context_id=self.attempt_context_id,
            )
            if projection.head_id != heads[-1].head_id:
                _fail(FAIL_LEDGER_V1, "replay projection/latest head differs")
        return (
            heads,
            checkpoints,
            head_payloads,
            checkpoint_payloads,
            token_payloads,
            projection,
        )

    def current_head_bytes(self) -> bytes:
        """Return reopened current head bytes after complete-chain replay."""

        with self._exclusive_lock():
            _heads, _checkpoints, head_payloads, _cp, _tokens, _projection = (
                self._load_chain()
            )
            return bytes(head_payloads[-1])

    def latest_projection_bytes(self) -> bytes:
        """Return nonauthority latest projection; genesis alone has none."""

        with self._exclusive_lock():
            _heads, _checkpoints, _hp, _cp, _tokens, projection = (
                self._load_chain()
            )
            if projection is None:
                _fail(FAIL_LEDGER_V1, "genesis ledger has no status projection")
            return projection.canonical_bytes()

    def append_service_side_v1(
        self,
        *,
        decision_payload: bytes,
        cas_store_dependency_reference_payload: bytes,
        logical_time_dependency_reference_payload: bytes,
        logical_time_token_payload: bytes,
        expected_previous_head_payload: bytes,
        governance_private_key: object,
    ) -> OwnerStatusServiceAppendV1:
        """Sign and durably append one event in the governance service.

        The private key must be a concrete ``cryptography`` Ed25519 key object;
        seed bytes, callbacks, injected signers, and injected verifiers are not
        accepted.  The key object is used only for this call and never retained.
        """

        decision = OwnerStatusDecisionV1.from_canonical_bytes(decision_payload)
        cas_dependency = CasStoreDependencyReferenceV1.from_canonical_bytes(
            cas_store_dependency_reference_payload
        )
        logical_dependency = (
            LogicalTimeDependencyReferenceV1.from_canonical_bytes(
                logical_time_dependency_reference_payload
            )
        )
        expected_head = OwnerStatusHeadV1.from_canonical_bytes(
            expected_previous_head_payload
        )
        expected_common = (
            self.authority_profile_id,
            self.signer_registry_id,
            self.governance_key_id,
            self.governance_key_epoch,
            self.attempt_context_id,
        )
        if (
            _common_tuple(decision) != expected_common
            or _common_tuple(expected_head) != expected_common
            or decision.ledger_id != self.ledger_id
            or expected_head.ledger_id != self.ledger_id
        ):
            _fail(FAIL_BINDING_V1, "append request differs from service pins")
        _require_dependency_graph(decision, cas_dependency, logical_dependency)
        logical_time_token = _verify_logical_time_token_v1(
            logical_time_token_payload,
            logical_dependency,
            logical_time_public_key=self.logical_time_public_key,
        )

        with self._exclusive_lock():
            heads, checkpoints, head_payloads, _cp, token_payloads, _projection = (
                self._load_chain()
            )
            previous_head = heads[-1]
            current_head_payload = head_payloads[-1]
            if (
                expected_previous_head_payload != current_head_payload
                or expected_head != previous_head
            ):
                _fail(
                    FAIL_LEDGER_V1,
                    "caller previous-head bytes are stale or compare unequal",
                )
            if previous_head.terminal:
                _fail(FAIL_LEDGER_V1, "REVOKED owner-status head is terminal")
            if (
                decision.sequence != previous_head.next_sequence
                or decision.previous_head_id != previous_head.head_id
                or decision.previous_checkpoint_id
                != previous_head.last_checkpoint_id
                or decision.previous_event_id != previous_head.last_event_id
                or decision.previous_complete_event_set_root
                != previous_head.complete_event_set_root
                or decision.prior_effective_status
                != previous_head.latest_effective_status
            ):
                _fail(FAIL_LEDGER_V1, "decision is not exact next head transition")
            existing_events = tuple(
                checkpoint.signed_event for checkpoint in checkpoints
            )
            if any(
                prior.decision.decision_nonce == decision.decision_nonce
                or prior.logical_time_dependency.token_id
                == logical_dependency.token_id
                or prior.cas_store_dependency.signed_receipt_id
                == cas_dependency.signed_receipt_id
                or prior.cas_store_dependency.checkpoint_id
                == cas_dependency.checkpoint_id
                for prior in existing_events
            ):
                _fail(FAIL_LEDGER_V1, "append dependency or decision replay detected")
            if existing_events:
                prior = existing_events[-1]
                prior_cas = prior.cas_store_dependency
                prior_time = prior.logical_time_dependency
                if (
                    (
                        cas_dependency.retention_session_id,
                        cas_dependency.chain_id,
                        cas_dependency.signer_key_id,
                        cas_dependency.signer_key_epoch,
                    )
                    != (
                        prior_cas.retention_session_id,
                        prior_cas.chain_id,
                        prior_cas.signer_key_id,
                        prior_cas.signer_key_epoch,
                    )
                    or cas_dependency.checkpoint_index
                    != prior_cas.checkpoint_index + 1
                    or (
                        logical_dependency.signer_key_id,
                        logical_dependency.signer_key_epoch,
                    )
                    != (prior_time.signer_key_id, prior_time.signer_key_epoch)
                    or logical_dependency.token_sequence
                    != prior_time.token_sequence + 1
                ):
                    _fail(
                        FAIL_LEDGER_V1,
                        "append CAS session/key or logical-time order forks",
                    )
                prior_token = _verify_logical_time_token_v1(
                    token_payloads[-1],
                    prior_time,
                    logical_time_public_key=self.logical_time_public_key,
                )
                if (
                    logical_time_token["prior_token_id"]
                    != prior_token["token_id"]
                    or logical_time_token["request_nonce"]
                    == prior_token["request_nonce"]
                ):
                    _fail(
                        FAIL_LEDGER_V1,
                        "append logical-time prior-token link or nonce forks",
                    )
            elif (
                logical_dependency.token_sequence != 0
                or logical_time_token["prior_token_id"]
                != LOGICAL_TIME_GENESIS_TOKEN_ID_V1
            ):
                _fail(
                    FAIL_LEDGER_V1,
                    "initial append logical-time token chain differs",
                )
            event = _issue_signed_event_service_side(
                decision=decision,
                cas_dependency=cas_dependency,
                logical_time_dependency=logical_dependency,
                logical_time_token_payload=logical_time_token_payload,
                governance_public_key=self.governance_public_key,
                logical_time_public_key=self.logical_time_public_key,
                private_key=governance_private_key,
            )
            checkpoint = _checkpoint_from_event(previous_head, event)
            next_head = _next_head_from_checkpoint(previous_head, checkpoint)
            _verify_transition(
                previous_head,
                checkpoint,
                next_head,
                governance_public_key=self.governance_public_key,
                logical_time_token_payload=logical_time_token_payload,
                logical_time_public_key=self.logical_time_public_key,
            )
            index_name = _versioned_filename(decision.sequence)
            event_payload = event.canonical_bytes()
            checkpoint_payload = checkpoint.canonical_bytes()
            next_head_payload = next_head.canonical_bytes()
            self._publish_exclusive(
                self._tokens_fd,
                index_name,
                logical_time_token_payload,
                MAXIMUM_LOGICAL_TIME_TOKEN_BYTES,
                "logical-time-token",
            )
            self._publish_exclusive(
                self._checkpoints_fd,
                index_name,
                checkpoint_payload,
                MAXIMUM_CHECKPOINT_BYTES,
                "checkpoint",
            )
            self._publish_exclusive(
                self._heads_fd,
                _versioned_filename(decision.sequence + 1),
                next_head_payload,
                MAXIMUM_HEAD_BYTES,
                "head",
            )
            (
                audited_heads,
                audited_checkpoints,
                audited_head_payloads,
                audited_checkpoint_payloads,
                audited_token_payloads,
                projection,
            ) = self._load_chain()
            if (
                audited_heads[-1] != next_head
                or audited_checkpoints[-1] != checkpoint
                or audited_head_payloads[-1] != next_head_payload
                or audited_checkpoint_payloads[-1] != checkpoint_payload
                or audited_token_payloads[-1] != logical_time_token_payload
                or projection is None
            ):
                _fail(FAIL_LEDGER_V1, "post-append complete-chain readback differs")
            result = OwnerStatusServiceAppendV1(
                decision_id=decision.decision_id,
                event_id=event.event_id,
                checkpoint_id=checkpoint.checkpoint_id,
                head_id=next_head.head_id,
                complete_event_set_root=next_head.complete_event_set_root,
                signed_event_payload=event_payload,
                checkpoint_payload=checkpoint_payload,
                head_payload=next_head_payload,
                projection_payload=projection.canonical_bytes(),
            )
            result.__post_init__()
            return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in (
            "_tokens_fd",
            "_checkpoints_fd",
            "_heads_fd",
            "_lock_fd",
            "_root_fd",
        ):
            try:
                fd = object.__getattribute__(self, name)
            except AttributeError:
                continue
            if fd >= 0:
                os.close(fd)
                object.__setattr__(self, name, -1)

    def __enter__(self) -> "OwnerStatusLedgerV1":
        if self._closed:
            _fail(FAIL_LEDGER_V1, "owner-status ledger is closed")
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> bool:
        self.close()
        return False


def load_cas_store_dependency_reference_v1(
    payload: bytes,
) -> CasStoreDependencyReferenceV1:
    return CasStoreDependencyReferenceV1.from_canonical_bytes(payload)


def load_logical_time_dependency_reference_v1(
    payload: bytes,
) -> LogicalTimeDependencyReferenceV1:
    return LogicalTimeDependencyReferenceV1.from_canonical_bytes(payload)


def load_owner_status_decision_v1(payload: bytes) -> OwnerStatusDecisionV1:
    return OwnerStatusDecisionV1.from_canonical_bytes(payload)


def load_signed_owner_status_event_v1(
    payload: bytes,
) -> SignedOwnerStatusEventV1:
    """Structurally load bytes; use the verifier for authenticity."""

    return SignedOwnerStatusEventV1.from_canonical_bytes(payload)


def load_owner_status_checkpoint_v1(
    payload: bytes,
) -> OwnerStatusCheckpointV1:
    return OwnerStatusCheckpointV1.from_canonical_bytes(payload)


def load_owner_status_head_v1(payload: bytes) -> OwnerStatusHeadV1:
    return OwnerStatusHeadV1.from_canonical_bytes(payload)


def load_owner_status_projection_v1(
    payload: bytes,
) -> OwnerStatusProjectionV1:
    return OwnerStatusProjectionV1.from_canonical_bytes(payload)


def owner_status_decision_id_preimage_v1(payload: bytes) -> bytes:
    decision = OwnerStatusDecisionV1.from_canonical_bytes(payload)
    return _DECISION_ID_DOMAIN + _canonical(decision._id_subject())


def owner_status_event_signature_preimage_v1(payload: bytes) -> bytes:
    event = SignedOwnerStatusEventV1.from_canonical_bytes(payload)
    return event.signature_preimage()


def owner_status_event_id_preimage_v1(payload: bytes) -> bytes:
    event = SignedOwnerStatusEventV1.from_canonical_bytes(payload)
    return _EVENT_ID_DOMAIN + _canonical(event._id_subject())


def owner_status_checkpoint_id_preimage_v1(payload: bytes) -> bytes:
    checkpoint = OwnerStatusCheckpointV1.from_canonical_bytes(payload)
    return _CHECKPOINT_ID_DOMAIN + _canonical(checkpoint._id_subject())


def owner_status_head_id_preimage_v1(payload: bytes) -> bytes:
    head = OwnerStatusHeadV1.from_canonical_bytes(payload)
    return _HEAD_ID_DOMAIN + _canonical(head._id_subject())


def owner_status_projection_id_preimage_v1(payload: bytes) -> bytes:
    projection = OwnerStatusProjectionV1.from_canonical_bytes(payload)
    return _PROJECTION_ID_DOMAIN + _canonical(projection._id_subject())


__all__ = [
    "ACTOR_PROFILE_CANONICAL_SHA256_V1",
    "ACTOR_PROFILE_CONTENT_ID_V1",
    "AUTHORITY_CLAIM_BOUNDARY_V1",
    "AUTHORITY_CLASS_V1",
    "AUTHORITY_PROFILE_PREFIX_V2",
    "AUTHORITY_REQUIREMENT_ID_V1",
    "CAS_CHECKPOINT_SCHEMA_V1",
    "CAS_DEPENDENCY_REFERENCE_SCHEMA_V1",
    "CAS_DEPENDENCY_VERIFICATION_STATUS_V1",
    "CAS_HEAD_SCHEMA_V1",
    "CAS_SIGNED_RECEIPT_SCHEMA_V1",
    "CAS_STORE_RECEIPT_PURPOSE_V1",
    "CAS_STORE_KEY_EPOCH_V1",
    "CAS_STORE_ROLE_V1",
    "CAS_STORE_SIGNATURE_DOMAIN_V1",
    "CasStoreDependencyReferenceV1",
    "CHECKPOINT_ID_PREFIX_V1",
    "CHECKPOINT_PURPOSE_V1",
    "DECISION_ID_PREFIX_V1",
    "ED25519_ALGORITHM_V1",
    "EVENT_CLAIM_BOUNDARY_V1",
    "EVENT_ID_PREFIX_V1",
    "EVENT_INITIAL_ACTIVE_V1",
    "EVENT_REVOKED_V1",
    "EXACT_FREEZE_ID_V1",
    "EXACT_FREEZE_VERSION_V1",
    "EXACT_OWNER_AUTHORIZATION_BINDING_V1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_FILESYSTEM_V1",
    "FAIL_INPUT_V1",
    "FAIL_LEDGER_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_SIGNATURE_BACKEND_V1",
    "FAIL_SIGNATURE_V1",
    "GENESIS_CHECKPOINT_ID_V1",
    "GENESIS_COMPLETE_EVENT_SET_ROOT_V1",
    "GENESIS_EVENT_ID_V1",
    "GOVERNANCE_CONTROL_DOMAIN_V1",
    "GOVERNANCE_KEY_EPOCH_V1",
    "GOVERNANCE_ROLE_V1",
    "GOVERNANCE_SIGNATURE_DOMAIN_V1",
    "HEAD_ID_PREFIX_V1",
    "HEAD_PURPOSE_V1",
    "KNOWN_INTEGRATION_BLOCKERS_V1",
    "LEDGER_ID_PREFIX_V1",
    "LOGICAL_STAGE_WIRE_POLICY_ID_V1",
    "LOGICAL_STAGE_WIRE_SCHEMA_ID_V1",
    "LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1",
    "LOGICAL_TIME_GENESIS_TOKEN_ID_V1",
    "LOGICAL_TIME_REQUEST_SCHEMA_V1",
    "LOGICAL_TIME_DEPENDENCY_VERIFICATION_STATUS_V1",
    "LOGICAL_TIME_CONTROL_DOMAIN_V1",
    "LOGICAL_TIME_EVENT_PURPOSE_V1",
    "LOGICAL_TIME_KEY_EPOCH_V1",
    "LOGICAL_TIME_REQUEST_ID_PREFIX_V1",
    "LOGICAL_TIME_ROLE_V1",
    "LOGICAL_TIME_SIGNATURE_PURPOSE_V1",
    "LOGICAL_TIME_TOKEN_ID_PREFIX_V1",
    "LOGICAL_TIME_TOKEN_SCHEMA_V1",
    "LogicalTimeDependencyReferenceV1",
    "OWNER_AUTHORIZATION_ID_V1",
    "OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1",
    "OWNER_AUTHORIZATION_SCHEMA_V1",
    "OWNER_AUTHORIZATION_SOURCE_SHA256_V1",
    "OWNER_STATUS_CHECKPOINT_SCHEMA_V1",
    "OWNER_STATUS_DECISION_SCHEMA_V1",
    "OWNER_STATUS_HEAD_SCHEMA_V1",
    "OWNER_STATUS_PROJECTION_SCHEMA_V1",
    "OwnerStatusCheckpointV1",
    "OwnerStatusDecisionV1",
    "OwnerStatusHeadV1",
    "OwnerStatusLedgerV1",
    "OwnerStatusProjectionV1",
    "OwnerStatusServiceAppendV1",
    "MAXIMUM_CHECKPOINT_BYTES",
    "MAXIMUM_DECISION_BYTES",
    "MAXIMUM_DEPENDENCY_REFERENCE_BYTES",
    "MAXIMUM_EVENT_BYTES",
    "MAXIMUM_HEAD_BYTES",
    "MAXIMUM_LOGICAL_TIME_TOKEN_BYTES",
    "MAXIMUM_OWNER_AUTHORIZATION_BYTES",
    "MAXIMUM_PROJECTION_BYTES",
    "Phase2BStep5OwnerStatusErrorV1",
    "PROJECTION_CLAIM_LEVEL_V1",
    "PROJECTION_ID_PREFIX_V1",
    "PROTOCOL_ID_V1",
    "REQUIRED_AUTHORITY_SCHEMA_CASCADE_V1",
    "REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1",
    "REQUIRED_GOVERNANCE_PURPOSES_V1",
    "ROLE_KEY_PREFIX_V1",
    "SCHEMA_VERSION",
    "SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1",
    "SIGNER_REGISTRY_PREFIX_V2",
    "SIGN_OWNER_STATUS_EVENT_PURPOSE_V1",
    "SIGN_SIGNER_REGISTRY_PURPOSE_V1",
    "STAGE_EVENT_CONTRACT_MANIFEST_ID_V1",
    "STATUS_ACTIVE_V1",
    "STATUS_NONE_V1",
    "STATUS_REVOKED_V1",
    "SignedOwnerStatusEventV1",
    "build_cas_store_dependency_reference_v1",
    "build_logical_time_dependency_reference_v1",
    "build_owner_status_decision_v1",
    "build_owner_status_genesis_head_v1",
    "derive_required_governance_role_key_id_v1",
    "load_cas_store_dependency_reference_v1",
    "load_logical_time_dependency_reference_v1",
    "load_owner_status_checkpoint_v1",
    "load_owner_status_decision_v1",
    "load_owner_status_head_v1",
    "load_owner_status_projection_v1",
    "load_signed_owner_status_event_v1",
    "owner_status_checkpoint_id_preimage_v1",
    "owner_status_decision_id_preimage_v1",
    "owner_status_event_id_preimage_v1",
    "owner_status_event_signature_preimage_v1",
    "owner_status_head_id_preimage_v1",
    "owner_status_projection_id_preimage_v1",
    "verify_exact_owner_authorization_payload_v1",
    "verify_owner_status_ledger_replay_v1",
    "verify_owner_status_ledger_transition_v1",
    "verify_signed_owner_status_event_v1",
]
