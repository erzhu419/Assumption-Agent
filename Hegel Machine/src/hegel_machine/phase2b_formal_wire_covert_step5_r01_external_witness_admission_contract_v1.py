"""Bytes-only candidate contract for real external-witness admission.

This module freezes, but does not operate, the prospective real R01
external-witness admission protocol.  The presented transcript is deliberately
acyclic::

    Q1(PREPARE) -> A(compare-and-append) -> Q2(CONFIRM) -> transcript
        -> one post-Q2 logical-time event -> candidate result

``A`` is the separately pinned durable-anchor key.  The Q1/Q2 online key, its
policy-bootstrap key, its custody-attestor key, A, and all eight authority-v2
role keys must be mutually disjoint.  The verifier consumes only exact canonical raw bytes plus two raw
out-of-band pins for the Stage-B lock and one ID/digest pin for each external
policy.  It verifies every Ed25519 signature directly and replays the supplied
complete logical-time chain through the producer-free replay core.  The chain
must end at the anchor-bound current-cut token/checkpoint/head immediately
before its final transcript-bound post-Q2 event.

Acceptance proves only internal consistency of a *presented* signed transcript.
This module has no channel, socket, filesystem, path, clock, process, service,
private key, decoded-object, callback, or caller-nonce input.  It cannot prove
that Q1 or Q2 was live, that A was durable or atomic, that the mapping was
current, or that any participant was non-equivocating.  Global anti-fork,
authority, R01, readiness, and execution predicates therefore remain false.
The real admission entry point is intentionally NOT_IMPLEMENTED.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from types import FunctionType, MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    DETACHED_SIGNATURE_FIELDS_V2 as _AUTH_SIGNATURE_FIELDS,
    DETACHED_SIGNATURE_SCHEMA_V2 as _AUTH_SIGNATURE_SCHEMA,
    ROLE_ORDER_V2 as _AUTH_ROLE_ORDER,
    Phase2BStep5AuthorityErrorV2 as _AuthorityError,
    load_step5_authority_profile_v2 as _load_authority_profile,
    load_step5_signer_registry_v2 as _load_signer_registry,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_lock,
    phase2b_step5_signature_preimage_v2 as _authority_signature_preimage,
    require_locked_step5_authority_graph_v2 as _require_authority_graph,
)
from .phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 import (
    ExpectedLogicalEventCoreV1 as _ExpectedLogicalEvent,
    Phase2BStep5LogicalTimeReplayCoreErrorV1 as _LogicalTimeError,
    require_verified_logical_time_replay_core_v1 as _require_logical_replay,
    verify_logical_time_replay_core_v1 as _verify_logical_replay,
)
from .phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 import (
    CANDIDATE_CLAIM_LEVEL_V1 as _EVIDENCE_CLAIM_LEVEL,
    CANDIDATE_STATUS_V1 as _EVIDENCE_STATUS,
    EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1 as _EXTERNAL_RECEIPT_FIELDS,
    EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1 as _EXTERNAL_RECEIPT_SCHEMA,
    EXTERNAL_WITNESS_POLICY_FIELDS_V1 as _BOOTSTRAP_POLICY_FIELDS,
    EXTERNAL_WITNESS_POLICY_SCHEMA_V1 as _BOOTSTRAP_POLICY_SCHEMA,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1 as _ANCHOR_BODY_FIELDS,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_FIELDS_V1 as _ANCHOR_FIELDS,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1 as _ANCHOR_SCHEMA,
    phase2b_step5_external_anchor_receipt_signature_preimage_v1 as _external_receipt_preimage,
    phase2b_step5_r01_evidence_content_id_v1 as _evidence_content_id,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical,
)

try:  # Import stays possible; verification still fails closed without crypto.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - verifier image dependency.
    _InvalidSignature = None
    _Ed25519PublicKey = None


AUTHORITY_V2_SOURCE_SHA256_V1: Final = (
    "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
)
EVIDENCE_CONTRACT_SOURCE_SHA256_V1: Final = (
    "574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d"
)
EXTERNAL_WITNESS_SOURCE_SHA256_V1: Final = (
    "b0622d5e50e2393e116211a64622e6d3d36d76d476afdcd6e5fd98a777061a88"
)

# Re-frozen from the source-pinned external-witness candidate.  Importing that
# issuer module would also import its filesystem implementation, which is
# intentionally outside this pure admission verifier's dependency graph.
_CANDIDATE_LEDGER_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-ledger/1"
)
_CANDIDATE_CHECKPOINT_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-checkpoint/1"
)
_CANDIDATE_HEAD_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-head/1"
)
_EXTERNAL_MECHANICS_BOUNDARY: Final = (
    "INITIAL_ONE_LEAF_EXTERNAL_WITNESS_SIGNATURE_AND_EXCLUSIVE_LOCAL_SPOOL_"
    "MECHANICS_ONLY_NOT_DISTINCT_UID_NOT_INDEPENDENT_OR_LOCAL_FILESYSTEM_"
    "NOT_HOST_ROLLBACK_RESISTANCE_NOT_KEY_CUSTODY_NOT_EXTERNAL_CONTROL_"
    "NOT_EXTERNAL_CURRENTNESS_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
LOGICAL_TIME_REPLAY_CORE_SOURCE_SHA256_V1: Final = (
    "a618291256723e2ce5ac8d77d4b77a3deec4bbf787ab3422b0b61b9c36358def"
)

ONLINE_WITNESS_POLICY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-online-witness-policy/1"
)
BOOTSTRAP_EXTERNAL_WITNESS_POLICY_SCHEMA_V1: Final = _BOOTSTRAP_POLICY_SCHEMA
DURABLE_ANCHOR_POLICY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-durable-anchor-policy/1"
)
Q1_PREPARE_REQUEST_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-q1-prepare-request/1"
)
Q1_PREPARE_RESPONSE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-q1-prepare-response/1"
)
A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-a-compare-and-append-request/1"
)
A_DURABLE_CHECKPOINT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-a-durable-checkpoint/1"
)
A_DURABLE_HEAD_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-a-durable-head/1"
)
A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-a-compare-and-append-receipt/1"
)
Q2_CONFIRM_REQUEST_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-q2-confirm-request/1"
)
Q2_CONFIRM_RESPONSE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-admission-q2-confirm-response/1"
)
ADMISSION_TRANSCRIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-admission-transcript/1"
)
ADMISSION_CANDIDATE_RESULT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-admission-candidate-result/1"
)

CANDIDATE_STATUS_V1: Final = "PRESENTED_TRANSCRIPT_CANDIDATE_NOT_ADMITTED"
CANDIDATE_CLAIM_LEVEL_V1: Final = (
    "PRESENTED_SIGNED_TRANSCRIPT_MECHANICS_ONLY_NOT_LIVE_CHANNEL_NOT_CURRENT_"
    "NOT_DURABLE_NOT_NON_EQUIVOCATING_NOT_GLOBAL_ANTI_FORK_NOT_RUNTIME_CLOSURE_"
    "NOT_KEY_CUSTODY_NOT_DISTINCT_LIVE_UID_GID_NOT_AUTHORITY_NOT_R01_"
    "NOT_READINESS_NOT_EXECUTION"
)
POST_Q2_LOGICAL_TIME_PURPOSE_V1: Final = (
    "COMMIT_EXTERNAL_WITNESS_ADMISSION_TRANSCRIPT"
)
NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1: Final = 14
VERIFIED_SIGNATURE_SCOPE_V1: Final = (
    "ADMISSION_LAYER_14_PLUS_LOGICAL_TOKEN_SIGNATURES_"
    "EXCLUDES_AUTHORITY_GRAPH_SIGNATURES"
)
VERIFIED_PROTOCOL_OBJECT_COUNT_V1: Final = 18
PREPARE_OUTCOME_V1: Final = "PREPARED_FOR_EXACT_COMPARE_AND_APPEND"
CAS_OUTCOME_V1: Final = "APPLIED_EXACT_COMPARE_AND_APPEND"
CONFIRM_OUTCOME_V1: Final = "CONFIRMED_PRESENTED_CURRENT_MAPPING"
ALGORITHM_V1: Final = "Ed25519"

SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1: Final = "SIGN_ONLINE_WITNESS_POLICY"
SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1: Final = "SIGN_DURABLE_ANCHOR_POLICY"
SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1: Final = "SIGN_Q1_PREPARE_REQUEST"
SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1: Final = "SIGN_Q1_PREPARE_RESPONSE"
SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1: Final = (
    "SIGN_A_COMPARE_AND_APPEND_REQUEST"
)
SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1: Final = "SIGN_A_DURABLE_CHECKPOINT"
SIGN_A_DURABLE_HEAD_PURPOSE_V1: Final = "SIGN_A_DURABLE_HEAD"
SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1: Final = (
    "SIGN_A_COMPARE_AND_APPEND_RECEIPT"
)
SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1: Final = "SIGN_Q2_CONFIRM_REQUEST"
SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1: Final = "SIGN_Q2_CONFIRM_RESPONSE"
SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1: Final = "SIGN_ADMISSION_TRANSCRIPT"
SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1: Final = (
    "SIGN_ADMISSION_CANDIDATE_RESULT"
)

BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1: Final = (
    SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
    SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1,
    SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1,
    SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1,
    SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1,
)
ONLINE_PURPOSES_IN_EXACT_ORDER_V1: Final = (
    SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1,
    SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
    SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1,
)
A_PURPOSES_IN_EXACT_ORDER_V1: Final = (
    SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1,
    SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1,
    SIGN_A_DURABLE_HEAD_PURPOSE_V1,
    SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1,
)

PRESENTED_TRANSCRIPT_CLAIM_BOUNDARY_V1: Final = CANDIDATE_CLAIM_LEVEL_V1

FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_CONTENT_ID"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_BINDING"
FAIL_TRUST_ANCHOR_V1: Final = (
    "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_TRUST_ANCHOR"
)
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_SIGNATURE"
FAIL_LOGICAL_TIME_V1: Final = (
    "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_LOGICAL_TIME"
)
FAIL_RESULT_V1: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_RESULT"
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_CLAIM_BOUNDARY"
)
FAIL_NOT_IMPLEMENTED_V1: Final = (
    "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_NOT_IMPLEMENTED"
)

SIGNATURE_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "algorithm",
        "signer_key_id",
        "key_epoch",
        "purpose",
        "signature_hex",
    }
)
ONLINE_WITNESS_POLICY_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "online_witness_policy_id", "bootstrap_signature"}
)
ONLINE_WITNESS_POLICY_BODY_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "policy_epoch",
        "witness_namespace",
        "witness_control_domain_id",
        "bootstrap_external_witness_policy_id",
        "bootstrap_external_witness_policy_payload_sha256",
        "bootstrap_key_id",
        "bootstrap_key_epoch",
        "bootstrap_public_key_hex",
        "bootstrap_authorized_signature_purposes_in_exact_order",
        "online_key_id",
        "online_key_epoch",
        "online_public_key_hex",
        "online_authorized_signature_purposes_in_exact_order",
        "service_binding_id",
        "service_uid",
        "service_gid",
        "service_user_namespace_inode",
        "runtime_closure_policy_id",
        "runtime_closure_policy_payload_sha256",
        "runtime_identity_profile",
        "custody_attestation_policy_id",
        "custody_attestation_policy_payload_sha256",
        "custody_attestor_key_id",
        "custody_attestor_key_epoch",
        "custody_attestor_public_key_hex",
        "online_key_nonexportable_required",
        "algorithm",
    }
)
BOOTSTRAP_EXTERNAL_WITNESS_POLICY_FIELDS_V1: Final = frozenset(
    _BOOTSTRAP_POLICY_FIELDS
)
DURABLE_ANCHOR_POLICY_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "durable_anchor_policy_id", "signature"}
)
DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "policy_epoch",
        "bootstrap_external_witness_policy_id",
        "bootstrap_external_witness_policy_payload_sha256",
        "online_witness_policy_id",
        "anchor_namespace",
        "anchor_control_domain_id",
        "anchor_key_id",
        "anchor_key_epoch",
        "anchor_public_key_hex",
        "algorithm",
        "authorized_signature_purposes_in_exact_order",
        "genesis_checkpoint_id",
        "genesis_head_id",
        "genesis_tree_root_sha256",
        "oob_high_water_checkpoint_id",
        "oob_high_water_checkpoint_payload_sha256",
        "oob_high_water_checkpoint_sequence",
        "oob_high_water_head_id",
        "oob_high_water_head_payload_sha256",
        "oob_high_water_head_sequence",
        "oob_high_water_tree_root_sha256",
        "oob_high_water_mapping",
        "admission_slot_key_profile",
        "compare_and_append_semantics",
        "durable_before_return_required",
    }
)

DURABLE_MAPPING_FIELDS_V1: Final = frozenset(
    {
        "external_anchor_id",
        "external_anchor_payload_sha256",
        "external_receipt_id",
        "external_receipt_payload_sha256",
        "external_ledger_id",
        "external_ledger_payload_sha256",
        "external_checkpoint_id",
        "external_checkpoint_payload_sha256",
        "external_head_id",
        "external_head_payload_sha256",
        "witness_sequence",
        "tree_size",
        "signed_tree_root_sha256",
        "attempt_context_id",
        "anchor_sequence",
        "effective_state",
        "superseded",
        "revoked",
        "map_leaf_index",
        "map_tree_size",
        "map_root_sha256",
        "map_inclusion_path_sha256_in_exact_order",
        "presented_to_live_consistency_path_sha256_in_exact_order",
    }
)

RUNTIME_CUSTODY_BINDING_FIELDS_V1: Final = frozenset(
    {
        "service_binding_id",
        "runtime_instance_id",
        "peer_pid",
        "peer_uid",
        "peer_gid",
        "user_namespace_inode",
        "process_start_ticks",
        "control_domain_id",
        "runtime_closure_policy_id",
        "runtime_closure_report_id",
        "runtime_closure_report_payload_sha256",
        "custody_attestation_policy_id",
        "custody_attestation_id",
        "custody_attestation_payload_sha256",
        "online_key_nonexportable",
    }
)

_COMMON_BODY_FIELDS: Final = frozenset(
    {
        "status",
        "claim_level",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "bootstrap_external_witness_policy_id",
        "online_witness_policy_id",
        "durable_anchor_policy_id",
    }
)
Q1_PREPARE_REQUEST_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "prepare_nonce",
        "candidate_mapping",
        "expected_previous_durable_checkpoint_id",
        "expected_previous_durable_head_id",
        "expected_previous_durable_checkpoint_payload_sha256",
        "expected_previous_durable_head_payload_sha256",
        "expected_previous_durable_checkpoint_sequence",
        "expected_previous_durable_head_sequence",
        "expected_previous_durable_tree_root_sha256",
    }
)
Q1_PREPARE_RESPONSE_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "q1_request_id",
        "q1_request_payload_sha256",
        "prepare_nonce",
        "confirmation_nonce",
        "observed_current_mapping",
        "prepared_candidate_mapping",
        "runtime_custody_binding",
        "prepare_outcome",
    }
)
A_COMPARE_AND_APPEND_REQUEST_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "q1_request_id",
        "q1_request_payload_sha256",
        "q1_response_id",
        "q1_response_payload_sha256",
        "prepare_nonce",
        "confirmation_nonce",
        "expected_current_mapping",
        "proposed_current_mapping",
        "expected_previous_durable_checkpoint_id",
        "expected_previous_durable_checkpoint_payload_sha256",
        "expected_previous_durable_head_id",
        "expected_previous_durable_head_payload_sha256",
        "expected_previous_durable_checkpoint_sequence",
        "expected_previous_durable_head_sequence",
        "expected_previous_durable_tree_root_sha256",
        "compare_and_append_mode",
    }
)
A_DURABLE_CHECKPOINT_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "a_request_id",
        "a_request_payload_sha256",
        "prepare_nonce",
        "confirmation_nonce",
        "previous_mapping",
        "committed_mapping",
        "cas_outcome",
        "checkpoint_sequence",
        "previous_a_checkpoint_id",
    }
)
A_DURABLE_HEAD_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "a_request_id",
        "a_request_payload_sha256",
        "a_checkpoint_id",
        "a_checkpoint_payload_sha256",
        "prepare_nonce",
        "confirmation_nonce",
        "committed_mapping",
        "cas_outcome",
        "head_sequence",
        "previous_a_head_id",
    }
)
A_COMPARE_AND_APPEND_RECEIPT_BODY_FIELDS_V1: Final = (
    _COMMON_BODY_FIELDS
    | frozenset(
        {
            "schema_version",
            "a_request_id",
            "a_request_payload_sha256",
            "a_checkpoint_id",
            "a_checkpoint_payload_sha256",
            "a_head_id",
            "a_head_payload_sha256",
            "prepare_nonce",
            "confirmation_nonce",
            "previous_mapping",
            "committed_mapping",
            "cas_outcome",
            "compare_matched",
            "append_applied",
            "oob_high_water_checkpoint_id",
            "oob_high_water_head_id",
            "durable_before_return",
        }
    )
)
Q2_CONFIRM_REQUEST_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "q1_response_id",
        "q1_response_payload_sha256",
        "a_request_id",
        "a_request_payload_sha256",
        "a_receipt_id",
        "a_receipt_payload_sha256",
        "a_checkpoint_id",
        "a_checkpoint_payload_sha256",
        "a_head_id",
        "a_head_payload_sha256",
        "prepare_nonce",
        "confirmation_nonce",
        "expected_current_mapping",
        "expected_cas_outcome",
    }
)
Q2_CONFIRM_RESPONSE_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "q2_request_id",
        "q2_request_payload_sha256",
        "a_receipt_id",
        "a_receipt_payload_sha256",
        "prepare_nonce",
        "confirmation_nonce",
        "observed_current_mapping",
        "runtime_custody_binding",
        "observed_cas_outcome",
        "confirm_outcome",
    }
)
ADMISSION_TRANSCRIPT_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "prepare_nonce",
        "confirmation_nonce",
        "candidate_mapping",
        "q1_request_id",
        "q1_request_payload_sha256",
        "q1_response_id",
        "q1_response_payload_sha256",
        "a_request_id",
        "a_request_payload_sha256",
        "a_checkpoint_id",
        "a_checkpoint_payload_sha256",
        "a_head_id",
        "a_head_payload_sha256",
        "a_receipt_id",
        "a_receipt_payload_sha256",
        "q2_request_id",
        "q2_request_payload_sha256",
        "q2_response_id",
        "q2_response_payload_sha256",
        "prepare_outcome",
        "cas_outcome",
        "confirm_outcome",
        "runtime_custody_binding",
        "post_q2_logical_time_request_nonce",
        "presented_q1_a_q2_chain_complete",
        "runtime_closure_verified",
        "key_custody_verified",
        "distinct_live_uid_gid_verified",
        "live_channel_verified",
        "external_currentness_verified",
        "durability_verified",
        "non_equivocation_verified",
        "global_anti_fork_verified",
    }
)
ADMISSION_CANDIDATE_RESULT_BODY_FIELDS_V1: Final = _COMMON_BODY_FIELDS | frozenset(
    {
        "schema_version",
        "admission_transcript_id",
        "admission_transcript_payload_sha256",
        "logical_time_ledger_id",
        "post_commit_logical_time_request_id",
        "post_commit_logical_time_token_id",
        "post_commit_logical_time_checkpoint_id",
        "post_commit_logical_time_head_id",
        "post_commit_logical_time_head_payload_sha256",
        "verified_signature_count",
        "verified_signature_scope",
        "presented_transcript_mechanics_verified",
        "post_q2_supplied_logical_time_chain_verified",
        "runtime_closure_verified",
        "key_custody_verified",
        "distinct_live_uid_gid_verified",
        "live_channel_verified",
        "external_currentness_verified",
        "durability_verified",
        "non_equivocation_verified",
        "global_anti_fork_verified",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
    }
)

Q1_PREPARE_REQUEST_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "q1_request_id", "signature"}
)
Q1_PREPARE_RESPONSE_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "q1_response_id", "signature"}
)
A_COMPARE_AND_APPEND_REQUEST_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "a_request_id", "signature"}
)
A_DURABLE_CHECKPOINT_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "a_checkpoint_id", "signature"}
)
A_DURABLE_HEAD_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "a_head_id", "signature"}
)
A_COMPARE_AND_APPEND_RECEIPT_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "a_receipt_id", "signature"}
)
Q2_CONFIRM_REQUEST_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "q2_request_id", "signature"}
)
Q2_CONFIRM_RESPONSE_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "q2_response_id", "signature"}
)
ADMISSION_TRANSCRIPT_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "admission_transcript_id", "signature"}
)
ADMISSION_CANDIDATE_RESULT_FIELDS_V1: Final = frozenset(
    {"schema_version", "body", "candidate_result_id", "signature"}
)

_CANDIDATE_LEDGER_FIELDS: Final = frozenset(
    {
        "schema_version", "status", "claim_level", "mechanics_claim_boundary",
        "stage_b_trust_lock_id", "authority_profile_id", "signer_registry_id",
        "attempt_context_id", "external_witness_policy_id", "witness_namespace",
        "witness_control_domain_id", "witness_key_id", "witness_key_epoch",
        "initial_anchor_id", "initial_anchor_payload_sha256", "initial_receipt_id",
        "initial_receipt_payload_sha256", "witness_sequence", "tree_size",
        "signed_tree_root_sha256", "ledger_id",
    }
)
_CANDIDATE_CHECKPOINT_FIELDS: Final = frozenset(
    {
        "schema_version", "status", "claim_level", "mechanics_claim_boundary",
        "external_witness_policy_id", "ledger_id", "initial_receipt_id",
        "initial_receipt_payload_sha256", "witness_sequence", "tree_size",
        "signed_tree_root_sha256", "checkpoint_sequence",
        "previous_checkpoint_id", "checkpoint_id",
    }
)
_CANDIDATE_HEAD_FIELDS: Final = frozenset(
    {
        "schema_version", "status", "claim_level", "mechanics_claim_boundary",
        "external_witness_policy_id", "ledger_id", "checkpoint_id",
        "checkpoint_payload_sha256", "initial_receipt_id",
        "initial_receipt_payload_sha256", "witness_sequence", "tree_size",
        "signed_tree_root_sha256", "previous_head_id", "head_id",
    }
)

_MAXIMUM_CANONICAL_BYTES: Final = 16 * 1024 * 1024
_MAXIMUM_SAFE_INTEGER: Final = (1 << 53) - 1
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX128 = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_CONTENT_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")

_PREFIXES: Final = MappingProxyType(
    {
        "online_witness_policy": "phase2b_step5_online_witness_policy_v1_",
        "durable_anchor_policy": "phase2b_step5_external_durable_anchor_policy_v1_",
        "q1_prepare_request": "phase2b_step5_external_admission_q1_request_v1_",
        "q1_prepare_response": "phase2b_step5_external_admission_q1_response_v1_",
        "a_compare_and_append_request": "phase2b_step5_external_admission_a_request_v1_",
        "a_durable_checkpoint": "phase2b_step5_external_admission_a_checkpoint_v1_",
        "a_durable_head": "phase2b_step5_external_admission_a_head_v1_",
        "a_compare_and_append_receipt": "phase2b_step5_external_admission_a_receipt_v1_",
        "q2_confirm_request": "phase2b_step5_external_admission_q2_request_v1_",
        "q2_confirm_response": "phase2b_step5_external_admission_q2_response_v1_",
        "admission_transcript": "phase2b_step5_external_admission_transcript_v1_",
        "admission_candidate_result": "phase2b_step5_external_admission_candidate_result_v1_",
    }
)
_CONTENT_ID_DOMAINS: Final = MappingProxyType(
    {
        kind: f"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/{kind.upper()}/ID/V1\0".encode("ascii")
        for kind in _PREFIXES
    }
)
_SIGNATURE_DOMAINS: Final = MappingProxyType(
    {
        "online_witness_policy": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/ONLINE_POLICY/V1\0",
        "durable_anchor_policy": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/DURABLE_ANCHOR_POLICY/V1\0",
        "q1_prepare_request": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/Q1_PREPARE_REQUEST/V1\0",
        "q1_prepare_response": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/Q1_PREPARE_RESPONSE/V1\0",
        "a_compare_and_append_request": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/A_COMPARE_AND_APPEND_REQUEST/V1\0",
        "a_durable_checkpoint": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/A_DURABLE_CHECKPOINT/V1\0",
        "a_durable_head": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/A_DURABLE_HEAD/V1\0",
        "a_compare_and_append_receipt": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/A_COMPARE_AND_APPEND_RECEIPT/V1\0",
        "q2_confirm_request": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/Q2_CONFIRM_REQUEST/V1\0",
        "q2_confirm_response": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/Q2_CONFIRM_RESPONSE/V1\0",
        "admission_transcript": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/TRANSCRIPT/V1\0",
        "admission_candidate_result": b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SIGN/CANDIDATE_RESULT/V1\0",
    }
)

ONLINE_WITNESS_POLICY_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "online_witness_policy"
]
DURABLE_ANCHOR_POLICY_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "durable_anchor_policy"
]
Q1_PREPARE_REQUEST_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "q1_prepare_request"
]
Q1_PREPARE_RESPONSE_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "q1_prepare_response"
]
A_COMPARE_AND_APPEND_REQUEST_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "a_compare_and_append_request"
]
A_DURABLE_CHECKPOINT_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "a_durable_checkpoint"
]
A_DURABLE_HEAD_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "a_durable_head"
]
A_COMPARE_AND_APPEND_RECEIPT_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "a_compare_and_append_receipt"
]
Q2_CONFIRM_REQUEST_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "q2_confirm_request"
]
Q2_CONFIRM_RESPONSE_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "q2_confirm_response"
]
ADMISSION_TRANSCRIPT_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "admission_transcript"
]
ADMISSION_CANDIDATE_RESULT_CONTENT_ID_DOMAIN_V1: Final = _CONTENT_ID_DOMAINS[
    "admission_candidate_result"
]

ONLINE_WITNESS_POLICY_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "online_witness_policy"
]
DURABLE_ANCHOR_POLICY_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "durable_anchor_policy"
]
Q1_PREPARE_REQUEST_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "q1_prepare_request"
]
Q1_PREPARE_RESPONSE_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "q1_prepare_response"
]
A_COMPARE_AND_APPEND_REQUEST_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "a_compare_and_append_request"
]
A_DURABLE_CHECKPOINT_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "a_durable_checkpoint"
]
A_DURABLE_HEAD_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS["a_durable_head"]
A_COMPARE_AND_APPEND_RECEIPT_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "a_compare_and_append_receipt"
]
Q2_CONFIRM_REQUEST_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "q2_confirm_request"
]
Q2_CONFIRM_RESPONSE_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "q2_confirm_response"
]
ADMISSION_TRANSCRIPT_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "admission_transcript"
]
ADMISSION_CANDIDATE_RESULT_SIGNATURE_DOMAIN_V1: Final = _SIGNATURE_DOMAINS[
    "admission_candidate_result"
]

_BOOTSTRAP_POLICY_PREFIX: Final = "phase2b_step5_external_witness_policy_v1_"
_BOOTSTRAP_POLICY_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
)
_DURABLE_POLICY_PREFIX: Final = _PREFIXES["durable_anchor_policy"]
_DURABLE_POLICY_DOMAIN: Final = DURABLE_ANCHOR_POLICY_CONTENT_ID_DOMAIN_V1
_ANCHOR_PREFIX: Final = "phase2b_step5_signed_owner_status_anti_fork_anchor_v1_"
_CANDIDATE_LEDGER_PREFIX: Final = "phase2b_step5_external_witness_ledger_v1_"
_CANDIDATE_LEDGER_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/LEDGER/V1\0"
_CANDIDATE_CHECKPOINT_PREFIX: Final = "phase2b_step5_external_witness_checkpoint_v1_"
_CANDIDATE_CHECKPOINT_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/CHECKPOINT/V1\0"
_CANDIDATE_HEAD_PREFIX: Final = "phase2b_step5_external_witness_head_v1_"
_CANDIDATE_HEAD_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/HEAD/V1\0"
_MERKLE_LEAF_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/MERKLE_LEAF/V1\0"
_CURRENT_MAP_LEAF_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/CURRENT_MAP_LEAF/V1\0"
)
_A_GENESIS_CHECKPOINT_ID: Final = _PREFIXES["a_durable_checkpoint"] + ("0" * 64)
_A_GENESIS_HEAD_ID: Final = _PREFIXES["a_durable_head"] + ("0" * 64)


class Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(ValueError):
    """Stable fail-closed error for presented admission transcript mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(code, detail)


def _canonical(value: object, _implementation=_canonicalize) -> bytes:
    try:
        return _implementation(value)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_CANONICAL_V1, "RFC-8785 canonicalization failed"
        ) from exc


def _restore_wire_integers(value: object) -> object:
    if type(value) is float and math.isfinite(value) and value.is_integer():
        if abs(value) > _MAXIMUM_SAFE_INTEGER:
            _fail(FAIL_SCHEMA_V1, "decoded integer exceeds I-JSON safe range")
        return int(value)
    if type(value) is list:
        return [_restore_wire_integers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_wire_integers(item) for key, item in value.items()}
    return value


def _decode(payload: bytes, label: str, _decoder=_decode_canonical) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAXIMUM_CANONICAL_BYTES
        or payload.endswith(b"\n")
    ):
        _fail(FAIL_INPUT_V1, f"{label} must be bounded exact bytes without newline")
    try:
        decoded = _restore_wire_integers(_decoder(payload))
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_CANONICAL_V1, f"{label} is not exact canonical RFC-8785"
        ) from exc
    if type(decoded) is not dict or _canonical(decoded) != payload:
        _fail(FAIL_CANONICAL_V1, f"{label} canonical readback differs")
    return decoded


def _object(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(FAIL_SCHEMA_V1, f"{label} keyset differs")
    return value


def _text(value: object, label: str, maximum: int = 4096) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        _fail(FAIL_SCHEMA_V1, f"{label} must be bounded nonempty text")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAXIMUM_SAFE_INTEGER:
        _fail(FAIL_SCHEMA_V1, f"{label} must be a safe integer >= {minimum}")
    return value


def _hex64(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lower hex SHA-256")
    return value


def _hex128(value: object, label: str) -> str:
    if type(value) is not str or _HEX128.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lower hex Ed25519 signature")
    return value


def _sha_address(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or _HEX64.fullmatch(value[7:]) is None
    ):
        _fail(FAIL_SCHEMA_V1, f"{label} must be one sha256: address")
    return value


def _identifier(value: object, label: str, prefix: str | None = None) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be an exact identifier")
    if prefix is not None and not value.startswith(prefix):
        _fail(FAIL_SCHEMA_V1, f"{label} prefix differs")
    return value


def _content_identifier(value: object, label: str, prefix: str) -> str:
    if (
        type(value) is not str
        or _CONTENT_IDENTIFIER.fullmatch(value) is None
        or not value.startswith(prefix)
    ):
        _fail(FAIL_SCHEMA_V1, f"{label} content ID differs")
    return value


def _nonce(value: object, label: str) -> str:
    nonce = _hex64(value, label)
    if nonce == "0" * 64:
        _fail(FAIL_SCHEMA_V1, f"{label} must be nonzero")
    return nonce


def _payload_sha(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "payload SHA input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _without(value: Mapping[str, object], *names: str) -> dict[str, object]:
    return {key: item for key, item in value.items() if key not in names}


def canonical_external_witness_admission_json_bytes_v1(value: object) -> bytes:
    """Return exact RFC-8785 bytes for fixture/protocol construction."""

    return _canonical(value)


def phase2b_step5_external_witness_admission_content_id_v1(
    artifact_kind: str, body: object
) -> str:
    """Derive one frozen admission content ID from an exact body."""

    if type(artifact_kind) is not str or artifact_kind not in _PREFIXES:
        _fail(FAIL_INPUT_V1, "admission artifact kind is unsupported")
    return _PREFIXES[artifact_kind] + hashlib.sha256(
        _CONTENT_ID_DOMAINS[artifact_kind] + _canonical(body)
    ).hexdigest()


def phase2b_step5_external_witness_admission_signature_preimage_v1(
    *,
    artifact_kind: str,
    body: object,
    artifact_id: str,
    signer_key_id: str,
    key_epoch: int,
    purpose: str,
) -> bytes:
    """Freeze the Ed25519 preimage for one admission protocol object."""

    if type(artifact_kind) is not str or artifact_kind not in _SIGNATURE_DOMAINS:
        _fail(FAIL_INPUT_V1, "admission signature artifact kind is unsupported")
    expected_id = phase2b_step5_external_witness_admission_content_id_v1(
        artifact_kind, body
    )
    if artifact_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "signature artifact ID differs from body")
    _identifier(signer_key_id, "signature signer key ID")
    _integer(key_epoch, "signature key epoch", 1)
    _text(purpose, "signature purpose", 128)
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-external-admission-signature-preimage/1",
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "body": body,
        "signer_key_id": signer_key_id,
        "key_epoch": key_epoch,
        "purpose": purpose,
    }
    return _SIGNATURE_DOMAINS[artifact_kind] + _canonical(subject)


def _decode_oob_id(value: bytes, label: str, prefix: str) -> str:
    if type(value) is not bytes or not 1 <= len(value) <= 512:
        _fail(FAIL_INPUT_V1, f"{label} OOB ID must be bounded exact bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_INPUT_V1, f"{label} OOB ID must be exact ASCII"
        ) from exc
    return _content_identifier(decoded, label, prefix)


def _decode_expected_identifier_bytes(value: bytes, label: str) -> str:
    if type(value) is not bytes or not 1 <= len(value) <= 512:
        _fail(FAIL_INPUT_V1, f"{label} must be bounded exact raw bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_INPUT_V1, f"{label} must be exact ASCII"
        ) from exc
    return _identifier(decoded, label)


def _load_raw_authority_graph(
    lock_payload: bytes,
    profile_payload: bytes,
    registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    _load_lock=_load_stage_b_lock,
    _load_profile=_load_authority_profile,
    _load_registry=_load_signer_registry,
    _require_graph=_require_authority_graph,
) -> tuple[object, object, object]:
    if (
        type(lock_payload) is not bytes
        or type(profile_payload) is not bytes
        or type(registry_payload) is not bytes
        or type(out_of_band_expected_lock_id_bytes) is not bytes
        or type(out_of_band_expected_lock_payload_sha256_digest) is not bytes
        or len(out_of_band_expected_lock_payload_sha256_digest) != 32
        or out_of_band_expected_lock_payload_sha256_digest == b"\x00" * 32
    ):
        _fail(FAIL_INPUT_V1, "authority graph and raw Stage-B pins must be exact bytes")
    try:
        lock = _load_lock(
            lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        profile = _load_profile(profile_payload, lock)
        registry = _load_registry(registry_payload, lock, profile)
        return _require_graph(lock, profile, registry)
    except _AuthorityError as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_TRUST_ANCHOR_V1, "raw authority-v2 graph rejected"
        ) from exc


def _role_keys(registry: object) -> tuple[Mapping[str, object], ...]:
    try:
        rows = registry.body["role_keys_in_exact_order"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_TRUST_ANCHOR_V1, "verified registry role keys unavailable"
        ) from exc
    if type(rows) is not tuple or len(rows) != len(_AUTH_ROLE_ORDER):
        _fail(FAIL_TRUST_ANCHOR_V1, "verified registry is not exact-eight-role")
    for expected_role, row in zip(_AUTH_ROLE_ORDER, rows, strict=True):
        if type(row) is not MappingProxyType or row.get("role") != expected_role:
            _fail(FAIL_TRUST_ANCHOR_V1, "verified registry role order differs")
    return rows


def _load_oob_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    prefix: str,
    label: str,
) -> tuple[dict[str, object], str]:
    if (
        type(payload) is not bytes
        or type(expected_payload_sha256_digest) is not bytes
        or len(expected_payload_sha256_digest) != 32
        or expected_payload_sha256_digest == b"\x00" * 32
    ):
        _fail(FAIL_INPUT_V1, f"{label} payload and OOB digest must be exact bytes")
    if hashlib.sha256(payload).digest() != expected_payload_sha256_digest:
        _fail(FAIL_TRUST_ANCHOR_V1, f"{label} raw SHA-256 pin differs")
    expected_id = _decode_oob_id(expected_id_bytes, label, prefix)
    return _decode(payload, label), expected_id


def _verify_ed25519(
    *, public_key_hex: str, signature_hex: str, preimage: bytes, label: str
) -> None:
    _hex64(public_key_hex, f"{label} public key")
    _hex128(signature_hex, f"{label} signature")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        key = _Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), preimage)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_SIGNATURE_V1, f"{label} Ed25519 signature rejected"
        ) from exc


def _signature(
    value: object,
    *,
    expected_key_id: str,
    expected_key_epoch: int,
    expected_purpose: str,
    public_key_hex: str,
    preimage: bytes,
    label: str,
) -> dict[str, object]:
    signature = _object(value, SIGNATURE_FIELDS_V1, f"{label} signature")
    if (
        signature["schema_version"]
        != "hegel-machine-phase2b-step5-external-admission-signature/1"
        or signature["algorithm"] != ALGORITHM_V1
        or signature["signer_key_id"] != expected_key_id
        or type(signature["key_epoch"]) is not int
        or signature["key_epoch"] != expected_key_epoch
        or signature["purpose"] != expected_purpose
    ):
        _fail(FAIL_SIGNATURE_V1, f"{label} signature metadata differs")
    _verify_ed25519(
        public_key_hex=public_key_hex,
        signature_hex=_hex128(signature["signature_hex"], f"{label} signature"),
        preimage=preimage,
        label=label,
    )
    return signature


def _load_bootstrap_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    role_keys: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    decoded, expected_id = _load_oob_policy(
        payload,
        expected_id_bytes=expected_id_bytes,
        expected_payload_sha256_digest=expected_payload_sha256_digest,
        prefix=_BOOTSTRAP_POLICY_PREFIX,
        label="bootstrap external-witness policy",
    )
    policy = _object(
        decoded,
        BOOTSTRAP_EXTERNAL_WITNESS_POLICY_FIELDS_V1,
        "bootstrap external-witness policy",
    )
    if (
        policy["schema_version"] != BOOTSTRAP_EXTERNAL_WITNESS_POLICY_SCHEMA_V1
        or policy["status"] != _EVIDENCE_STATUS
        or policy["claim_level"] != _EVIDENCE_CLAIM_LEVEL
        or policy["algorithm"] != ALGORITHM_V1
        or type(policy["witness_key_epoch"]) is not int
        or policy["witness_key_epoch"] != 1
    ):
        _fail(FAIL_SCHEMA_V1, "bootstrap external-witness policy header differs")
    _text(policy["witness_namespace"], "bootstrap witness namespace", 256)
    for field in ("witness_control_domain_id", "witness_key_id"):
        _identifier(policy[field], f"bootstrap policy {field}")
    public_key = _hex64(policy["public_key_hex"], "bootstrap public key")
    policy_id = _BOOTSTRAP_POLICY_PREFIX + hashlib.sha256(
        _BOOTSTRAP_POLICY_DOMAIN
        + _canonical(_without(policy, "external_witness_policy_id"))
    ).hexdigest()
    if policy["external_witness_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "bootstrap external-witness policy content ID differs")
    if (
        public_key in {row["public_key_hex"] for row in role_keys}
        or policy["witness_key_id"] in {row["key_id"] for row in role_keys}
        or policy["witness_control_domain_id"]
        in {row["control_domain"] for row in role_keys}
        or policy["witness_namespace"] in set(_AUTH_ROLE_ORDER)
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "bootstrap witness is not authority-disjoint")
    return policy


def _load_online_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    role_keys: tuple[Mapping[str, object], ...],
    authority_profile: object,
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
) -> dict[str, object]:
    decoded, expected_id = _load_oob_policy(
        payload,
        expected_id_bytes=expected_id_bytes,
        expected_payload_sha256_digest=expected_payload_sha256_digest,
        prefix=_PREFIXES["online_witness_policy"],
        label="online-witness policy",
    )
    root = _object(decoded, ONLINE_WITNESS_POLICY_FIELDS_V1, "online policy")
    body = _object(
        root["body"], ONLINE_WITNESS_POLICY_BODY_FIELDS_V1, "online policy body"
    )
    if (
        root["schema_version"] != ONLINE_WITNESS_POLICY_SCHEMA_V1
        or body["schema_version"] != ONLINE_WITNESS_POLICY_SCHEMA_V1
        or body["status"] != CANDIDATE_STATUS_V1
        or body["claim_level"] != CANDIDATE_CLAIM_LEVEL_V1
        or type(body["policy_epoch"]) is not int
        or body["policy_epoch"] != 1
        or body["algorithm"] != ALGORITHM_V1
        or type(body["bootstrap_key_epoch"]) is not int
        or body["bootstrap_key_epoch"] != 1
        or type(body["online_key_epoch"]) is not int
        or body["online_key_epoch"] != 1
        or type(body["bootstrap_authorized_signature_purposes_in_exact_order"])
        is not list
        or tuple(body["bootstrap_authorized_signature_purposes_in_exact_order"])
        != BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1
        or type(body["online_authorized_signature_purposes_in_exact_order"])
        is not list
        or tuple(body["online_authorized_signature_purposes_in_exact_order"])
        != ONLINE_PURPOSES_IN_EXACT_ORDER_V1
        or body["online_key_nonexportable_required"] is not True
        or body["runtime_identity_profile"]
        != "LINUX_AF_UNIX_SEQPACKET_PROCFS_ENTRY_EXIT_V1"
    ):
        _fail(FAIL_SCHEMA_V1, "online-witness policy header/purpose order differs")
    for field in (
        "bootstrap_key_id",
        "online_key_id",
        "witness_control_domain_id",
        "service_binding_id",
        "runtime_closure_policy_id",
        "custody_attestation_policy_id",
        "custody_attestor_key_id",
    ):
        _identifier(body[field], f"online policy {field}")
    for field in (
        "service_uid",
        "service_gid",
        "service_user_namespace_inode",
        "custody_attestor_key_epoch",
    ):
        _integer(body[field], f"online policy {field}", 1)
    for field in (
        "runtime_closure_policy_payload_sha256",
        "custody_attestation_policy_payload_sha256",
    ):
        _hex64(body[field], f"online policy {field}")
    _text(body["witness_namespace"], "online witness namespace", 256)
    bootstrap_public = _hex64(body["bootstrap_public_key_hex"], "bootstrap public key")
    online_public = _hex64(body["online_public_key_hex"], "online public key")
    custody_public = _hex64(
        body["custody_attestor_public_key_hex"], "custody attestor public key"
    )
    if (
        body["bootstrap_external_witness_policy_id"]
        != bootstrap_policy["external_witness_policy_id"]
        or body["bootstrap_external_witness_policy_payload_sha256"]
        != _payload_sha(bootstrap_policy_payload)
        or body["bootstrap_key_id"] != bootstrap_policy["witness_key_id"]
        or body["bootstrap_key_epoch"] != bootstrap_policy["witness_key_epoch"]
        or bootstrap_public != bootstrap_policy["public_key_hex"]
        or body["witness_namespace"] != bootstrap_policy["witness_namespace"]
        or body["witness_control_domain_id"]
        == bootstrap_policy["witness_control_domain_id"]
    ):
        _fail(FAIL_BINDING_V1, "online policy bootstrap-policy linkage differs")
    policy_id = phase2b_step5_external_witness_admission_content_id_v1(
        "online_witness_policy", body
    )
    if root["online_witness_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "online-witness policy content ID differs")
    preimage = phase2b_step5_external_witness_admission_signature_preimage_v1(
        artifact_kind="online_witness_policy",
        body=body,
        artifact_id=policy_id,
        signer_key_id=body["bootstrap_key_id"],
        key_epoch=body["bootstrap_key_epoch"],
        purpose=SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
    )
    _signature(
        root["bootstrap_signature"],
        expected_key_id=body["bootstrap_key_id"],
        expected_key_epoch=body["bootstrap_key_epoch"],
        expected_purpose=SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
        public_key_hex=bootstrap_public,
        preimage=preimage,
        label="online policy bootstrap",
    )
    authority_public = {row["public_key_hex"] for row in role_keys}
    authority_ids = {row["key_id"] for row in role_keys}
    authority_controls = {row["control_domain"] for row in role_keys}
    try:
        authority_services = authority_profile.body[
            "service_static_bindings_in_exact_order"
        ]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_TRUST_ANCHOR_V1,
            "verified authority service bindings unavailable",
        ) from exc
    if type(authority_services) is not tuple or len(authority_services) != 5:
        _fail(FAIL_TRUST_ANCHOR_V1, "verified authority service set differs")
    authority_service_binding_ids = {row["binding_id"] for row in authority_services}
    authority_service_uids_and_gids = {
        identity
        for row in authority_services
        for identity in (row["uid"], row["gid"])
    }
    authority_runtime_policy_shas = {
        row["static_runtime_closure_policy_sha256"] for row in authority_services
    }
    if (
        len({bootstrap_public, online_public, custody_public}) != 3
        or body["bootstrap_key_id"] == body["online_key_id"]
        or body["custody_attestor_key_id"]
        in {body["bootstrap_key_id"], body["online_key_id"]}
        or bootstrap_public in authority_public
        or online_public in authority_public
        or custody_public in authority_public
        or body["bootstrap_key_id"] in authority_ids
        or body["online_key_id"] in authority_ids
        or body["custody_attestor_key_id"] in authority_ids
        or body["witness_control_domain_id"] in authority_controls
        or body["service_binding_id"] in authority_service_binding_ids
        or body["service_uid"] == body["service_gid"]
        or body["service_uid"] in authority_service_uids_and_gids
        or body["service_gid"] in authority_service_uids_and_gids
        or body["runtime_closure_policy_payload_sha256"]
        in authority_runtime_policy_shas
        or body["witness_namespace"] in set(_AUTH_ROLE_ORDER)
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "online/bootstrap policy is not authority-disjoint")
    root["body"] = body
    return root


def _load_durable_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    role_keys: tuple[Mapping[str, object], ...],
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
    online_policy: Mapping[str, object],
) -> dict[str, object]:
    decoded, expected_id = _load_oob_policy(
        payload,
        expected_id_bytes=expected_id_bytes,
        expected_payload_sha256_digest=expected_payload_sha256_digest,
        prefix=_DURABLE_POLICY_PREFIX,
        label="durable-anchor policy",
    )
    root = _object(decoded, DURABLE_ANCHOR_POLICY_FIELDS_V1, "durable policy")
    body = _object(
        root["body"], DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1, "durable policy body"
    )
    if (
        root["schema_version"] != DURABLE_ANCHOR_POLICY_SCHEMA_V1
        or body["schema_version"] != DURABLE_ANCHOR_POLICY_SCHEMA_V1
        or body["status"] != CANDIDATE_STATUS_V1
        or body["claim_level"] != CANDIDATE_CLAIM_LEVEL_V1
        or type(body["policy_epoch"]) is not int
        or body["policy_epoch"] != 1
        or body["algorithm"] != ALGORITHM_V1
        or type(body["anchor_key_epoch"]) is not int
        or body["anchor_key_epoch"] != 1
        or type(body["authorized_signature_purposes_in_exact_order"]) is not list
        or tuple(body["authorized_signature_purposes_in_exact_order"])
        != A_PURPOSES_IN_EXACT_ORDER_V1
        or body["durable_before_return_required"] is not True
        or body["admission_slot_key_profile"]
        != "UNIQUE_BY_ATTEMPT_CONTEXT_ID_AND_ANCHOR_SEQUENCE"
        or body["compare_and_append_semantics"]
        != "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
    ):
        _fail(FAIL_SCHEMA_V1, "durable-anchor policy header differs")
    _text(body["anchor_namespace"], "durable anchor namespace", 256)
    for field in ("anchor_control_domain_id", "anchor_key_id"):
        _identifier(body[field], f"durable policy {field}")
    _content_identifier(
        body["oob_high_water_checkpoint_id"],
        "durable policy OOB high-water checkpoint ID",
        _PREFIXES["a_durable_checkpoint"],
    )
    _content_identifier(
        body["oob_high_water_head_id"],
        "durable policy OOB high-water head ID",
        _PREFIXES["a_durable_head"],
    )
    durable_public = _hex64(body["anchor_public_key_hex"], "A public key")
    for field in (
        "bootstrap_external_witness_policy_payload_sha256",
        "genesis_tree_root_sha256",
        "oob_high_water_checkpoint_payload_sha256",
        "oob_high_water_head_payload_sha256",
        "oob_high_water_tree_root_sha256",
    ):
        _hex64(body[field], f"durable policy {field}")
    for field in (
        "oob_high_water_checkpoint_sequence",
        "oob_high_water_head_sequence",
    ):
        _integer(body[field], f"durable policy {field}")
    if (
        body["bootstrap_external_witness_policy_id"]
        != bootstrap_policy["external_witness_policy_id"]
        or body["bootstrap_external_witness_policy_payload_sha256"]
        != _payload_sha(bootstrap_policy_payload)
        or body["online_witness_policy_id"]
        != online_policy["online_witness_policy_id"]
        or body["genesis_checkpoint_id"] != _A_GENESIS_CHECKPOINT_ID
        or body["genesis_head_id"] != _A_GENESIS_HEAD_ID
        or body["genesis_tree_root_sha256"] != "0" * 64
    ):
        _fail(FAIL_BINDING_V1, "durable policy bootstrap/online/genesis linkage differs")
    policy_id = phase2b_step5_external_witness_admission_content_id_v1(
        "durable_anchor_policy", body
    )
    if root["durable_anchor_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "durable-anchor policy content ID differs")
    online_body = online_policy["body"]
    authority_public = {row["public_key_hex"] for row in role_keys}
    authority_ids = {row["key_id"] for row in role_keys}
    authority_controls = {row["control_domain"] for row in role_keys}
    if (
        durable_public in authority_public
        or body["anchor_key_id"] in authority_ids
        or body["anchor_control_domain_id"] in authority_controls
        or durable_public
        in {
            bootstrap_policy["public_key_hex"],
            online_body["online_public_key_hex"],
            online_body["custody_attestor_public_key_hex"],
        }
        or body["anchor_key_id"]
        in {
            bootstrap_policy["witness_key_id"],
            online_body["online_key_id"],
            online_body["custody_attestor_key_id"],
        }
        or body["anchor_control_domain_id"]
        in {
            bootstrap_policy["witness_control_domain_id"],
            online_body["witness_control_domain_id"],
        }
        or body["anchor_namespace"] in {
            bootstrap_policy["witness_namespace"],
            online_body["witness_namespace"],
        }
        or body["anchor_namespace"] in set(_AUTH_ROLE_ORDER)
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "A key/control/namespace is not fully disjoint")
    preimage = phase2b_step5_external_witness_admission_signature_preimage_v1(
        artifact_kind="durable_anchor_policy",
        body=body,
        artifact_id=policy_id,
        signer_key_id=body["anchor_key_id"],
        key_epoch=body["anchor_key_epoch"],
        purpose=SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1,
    )
    _signature(
        root["signature"],
        expected_key_id=body["anchor_key_id"],
        expected_key_epoch=body["anchor_key_epoch"],
        expected_purpose=SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1,
        public_key_hex=durable_public,
        preimage=preimage,
        label="durable-anchor policy",
    )
    root["body"] = body
    return root


def _verify_pure_anchor_signature(
    root: Mapping[str, object], body: Mapping[str, object], role_keys: tuple[Mapping[str, object], ...]
) -> None:
    pure = role_keys[_AUTH_ROLE_ORDER.index("PURE_REPLAYER")]
    signature = _object(root["signature"], frozenset(_AUTH_SIGNATURE_FIELDS), "PURE anchor signature")
    if (
        signature["schema_version"] != _AUTH_SIGNATURE_SCHEMA
        or signature["algorithm"] != ALGORITHM_V1
        or signature["role"] != "PURE_REPLAYER"
        or signature["purpose"] != "SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR"
        or signature["signer_key_id"] != pure["key_id"]
        or type(signature["key_epoch"]) is not int
        or signature["key_epoch"] != pure["key_epoch"]
        or signature["purpose"] not in pure["authorized_signature_purposes"]
    ):
        _fail(FAIL_SIGNATURE_V1, "PURE anchor signature metadata differs")
    try:
        preimage = _authority_signature_preimage(
            artifact_kind="owner_status_anti_fork_anchor",
            body=body,
            role="PURE_REPLAYER",
            purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
            signer_key_id=pure["key_id"],
            key_epoch=pure["key_epoch"],
        )
    except _AuthorityError as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_SIGNATURE_V1, "PURE anchor signature preimage rejected"
        ) from exc
    _verify_ed25519(
        public_key_hex=pure["public_key_hex"],
        signature_hex=_hex128(signature["signature_hex"], "PURE anchor signature"),
        preimage=preimage,
        label="PURE anchor",
    )


def _current_map_leaf_root(mapping: Mapping[str, object]) -> str:
    subject = _without(
        mapping,
        "map_leaf_index",
        "map_tree_size",
        "map_root_sha256",
        "map_inclusion_path_sha256_in_exact_order",
        "presented_to_live_consistency_path_sha256_in_exact_order",
    )
    return hashlib.sha256(_CURRENT_MAP_LEAF_DOMAIN + _canonical(subject)).hexdigest()


def _load_candidate_artifacts(
    *,
    anchor_payload: bytes,
    receipt_payload: bytes,
    ledger_payload: bytes,
    checkpoint_payload: bytes,
    head_payload: bytes,
    lock: object,
    profile: object,
    registry: object,
    role_keys: tuple[Mapping[str, object], ...],
    bootstrap_policy: Mapping[str, object],
    expected_attempt_context_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    anchor = _object(_decode(anchor_payload, "candidate anchor"), frozenset(_ANCHOR_FIELDS), "candidate anchor")
    body = _object(anchor["body"], frozenset(_ANCHOR_BODY_FIELDS), "candidate anchor body")
    if (
        anchor["schema_version"] != _ANCHOR_SCHEMA
        or body["schema_version"] != _ANCHOR_SCHEMA
        or body["status"] != _EVIDENCE_STATUS
        or body["claim_level"] != _EVIDENCE_CLAIM_LEVEL
        or body["stage_b_trust_lock_id"] != lock.artifact_id
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["attempt_context_id"] != expected_attempt_context_id
        or body["external_witness_policy_id"] != bootstrap_policy["external_witness_policy_id"]
        or type(body["anchor_sequence"]) is not int
        or body["anchor_sequence"] != 1
        or body["previous_external_anchor_receipt_id"] is not None
    ):
        _fail(FAIL_BINDING_V1, "candidate anchor authority/attempt/policy/sequence differs")
    _content_identifier(
        body["current_cut_statement_id"],
        "candidate current-cut statement ID",
        "phase2b_step5_current_cut_statement_v1_",
    )
    _hex64(
        body["current_cut_statement_payload_sha256"],
        "candidate current-cut statement payload SHA",
    )
    for field, prefix in (
        ("current_cut_token_id", "phase2b_step5_logical_time_token_v1_"),
        ("current_cut_checkpoint_id", "phase2b_step5_logical_time_checkpoint_v1_"),
        ("current_cut_head_id", "phase2b_step5_logical_time_head_v1_"),
        ("owner_status_ledger_id", "phase2b_step5_owner_status_ledger_v1_"),
        ("owner_current_head_id", "phase2b_step5_owner_status_head_v1_"),
        ("projection_id", "phase2b_step5_owner_status_projection_v1_"),
    ):
        _content_identifier(body[field], f"candidate anchor {field}", prefix)
    for field in (
        "current_cut_token_payload_sha256",
        "current_cut_checkpoint_payload_sha256",
        "current_cut_head_payload_sha256",
        "complete_event_set_root",
    ):
        _sha_address(body[field], f"candidate anchor {field}")
    _text(body["anchor_nonce"], "candidate anchor nonce", 256)
    expected_anchor_id = _evidence_content_id("signed_owner_status_anti_fork_anchor", body)
    if anchor["anchor_id"] != expected_anchor_id:
        _fail(FAIL_CONTENT_ID_V1, "candidate anchor ID differs")
    _verify_pure_anchor_signature(anchor, body, role_keys)

    receipt = _object(_decode(receipt_payload, "candidate external receipt"), frozenset(_EXTERNAL_RECEIPT_FIELDS), "candidate external receipt")
    expected_root = hashlib.sha256(_MERKLE_LEAF_DOMAIN + anchor_payload).hexdigest()
    if (
        receipt["schema_version"] != _EXTERNAL_RECEIPT_SCHEMA
        or receipt["external_witness_policy_id"] != bootstrap_policy["external_witness_policy_id"]
        or receipt["witness_namespace"] != bootstrap_policy["witness_namespace"]
        or receipt["witness_control_domain_id"] != bootstrap_policy["witness_control_domain_id"]
        or receipt["witness_key_id"] != bootstrap_policy["witness_key_id"]
        or type(receipt["witness_key_epoch"]) is not int
        or receipt["witness_key_epoch"] != bootstrap_policy["witness_key_epoch"]
        or receipt["anchor_id"] != anchor["anchor_id"]
        or receipt["anchor_payload_sha256"] != _payload_sha(anchor_payload)
        or type(receipt["witness_sequence"]) is not int
        or receipt["witness_sequence"] != 1
        or receipt["previous_external_receipt_id"] is not None
        or type(receipt["leaf_index"]) is not int
        or receipt["leaf_index"] != 0
        or type(receipt["tree_size"]) is not int
        or receipt["tree_size"] != 1
        or receipt["signed_tree_root_sha256"] != expected_root
        or type(receipt["inclusion_path_sha256_in_exact_order"]) is not list
        or receipt["inclusion_path_sha256_in_exact_order"] != []
        or receipt["algorithm"] != ALGORITHM_V1
    ):
        _fail(FAIL_BINDING_V1, "candidate external receipt lineage differs")
    receipt_id = _evidence_content_id(
        "external_owner_status_anchor_receipt",
        _without(receipt, "signature_hex", "receipt_id"),
    )
    if receipt["receipt_id"] != receipt_id:
        _fail(FAIL_CONTENT_ID_V1, "candidate external receipt ID differs")
    try:
        receipt_preimage = _external_receipt_preimage(dict(receipt))
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_SIGNATURE_V1, "candidate receipt signature preimage rejected"
        ) from exc
    _verify_ed25519(
        public_key_hex=bootstrap_policy["public_key_hex"],
        signature_hex=_hex128(receipt["signature_hex"], "candidate receipt signature"),
        preimage=receipt_preimage,
        label="candidate A receipt",
    )

    ledger = _object(_decode(ledger_payload, "candidate external ledger"), _CANDIDATE_LEDGER_FIELDS, "candidate external ledger")
    checkpoint = _object(_decode(checkpoint_payload, "candidate external checkpoint"), _CANDIDATE_CHECKPOINT_FIELDS, "candidate external checkpoint")
    head = _object(_decode(head_payload, "candidate external head"), _CANDIDATE_HEAD_FIELDS, "candidate external head")
    ledger_subject = _without(ledger, "ledger_id")
    expected_ledger_id = _CANDIDATE_LEDGER_PREFIX + hashlib.sha256(
        _CANDIDATE_LEDGER_DOMAIN + _canonical(ledger_subject)
    ).hexdigest()
    checkpoint_subject = _without(checkpoint, "checkpoint_id")
    expected_checkpoint_id = _CANDIDATE_CHECKPOINT_PREFIX + hashlib.sha256(
        _CANDIDATE_CHECKPOINT_DOMAIN + _canonical(checkpoint_subject)
    ).hexdigest()
    head_subject = _without(head, "head_id")
    expected_head_id = _CANDIDATE_HEAD_PREFIX + hashlib.sha256(
        _CANDIDATE_HEAD_DOMAIN + _canonical(head_subject)
    ).hexdigest()
    common_candidate = (
        ledger["schema_version"] == _CANDIDATE_LEDGER_SCHEMA
        and checkpoint["schema_version"] == _CANDIDATE_CHECKPOINT_SCHEMA
        and head["schema_version"] == _CANDIDATE_HEAD_SCHEMA
        and ledger["status"] == checkpoint["status"] == head["status"] == _EVIDENCE_STATUS
        and ledger["claim_level"] == checkpoint["claim_level"] == head["claim_level"] == _EVIDENCE_CLAIM_LEVEL
        and ledger["mechanics_claim_boundary"] == checkpoint["mechanics_claim_boundary"] == head["mechanics_claim_boundary"] == _EXTERNAL_MECHANICS_BOUNDARY
    )
    if not common_candidate:
        _fail(FAIL_CLAIM_BOUNDARY_V1, "candidate external spool claim boundary differs")
    if (
        ledger["ledger_id"] != expected_ledger_id
        or checkpoint["checkpoint_id"] != expected_checkpoint_id
        or head["head_id"] != expected_head_id
        or ledger["stage_b_trust_lock_id"] != lock.artifact_id
        or ledger["authority_profile_id"] != profile.artifact_id
        or ledger["signer_registry_id"] != registry.artifact_id
        or ledger["attempt_context_id"] != expected_attempt_context_id
        or ledger["external_witness_policy_id"] != bootstrap_policy["external_witness_policy_id"]
        or ledger["witness_namespace"] != bootstrap_policy["witness_namespace"]
        or ledger["witness_control_domain_id"] != bootstrap_policy["witness_control_domain_id"]
        or ledger["witness_key_id"] != bootstrap_policy["witness_key_id"]
        or type(ledger["witness_key_epoch"]) is not int
        or ledger["witness_key_epoch"] != 1
        or ledger["initial_anchor_id"] != anchor["anchor_id"]
        or ledger["initial_anchor_payload_sha256"] != _payload_sha(anchor_payload)
        or ledger["initial_receipt_id"] != receipt["receipt_id"]
        or ledger["initial_receipt_payload_sha256"] != _payload_sha(receipt_payload)
        or type(ledger["witness_sequence"]) is not int
        or ledger["witness_sequence"] != 1
        or type(ledger["tree_size"]) is not int
        or ledger["tree_size"] != 1
        or ledger["signed_tree_root_sha256"] != expected_root
        or checkpoint["external_witness_policy_id"] != bootstrap_policy["external_witness_policy_id"]
        or checkpoint["ledger_id"] != ledger["ledger_id"]
        or checkpoint["initial_receipt_id"] != receipt["receipt_id"]
        or checkpoint["initial_receipt_payload_sha256"] != _payload_sha(receipt_payload)
        or type(checkpoint["witness_sequence"]) is not int
        or checkpoint["witness_sequence"] != 1
        or type(checkpoint["tree_size"]) is not int
        or checkpoint["tree_size"] != 1
        or checkpoint["signed_tree_root_sha256"] != expected_root
        or type(checkpoint["checkpoint_sequence"]) is not int
        or checkpoint["checkpoint_sequence"] != 1
        or checkpoint["previous_checkpoint_id"] is not None
        or head["external_witness_policy_id"] != bootstrap_policy["external_witness_policy_id"]
        or head["ledger_id"] != ledger["ledger_id"]
        or head["checkpoint_id"] != checkpoint["checkpoint_id"]
        or head["checkpoint_payload_sha256"] != _payload_sha(checkpoint_payload)
        or head["initial_receipt_id"] != receipt["receipt_id"]
        or head["initial_receipt_payload_sha256"] != _payload_sha(receipt_payload)
        or type(head["witness_sequence"]) is not int
        or head["witness_sequence"] != 1
        or type(head["tree_size"]) is not int
        or head["tree_size"] != 1
        or head["signed_tree_root_sha256"] != expected_root
        or head["previous_head_id"] is not None
    ):
        _fail(FAIL_BINDING_V1, "candidate anchor/receipt/ledger/checkpoint/head lineage differs")
    mapping = {
        "external_anchor_id": anchor["anchor_id"],
        "external_anchor_payload_sha256": _payload_sha(anchor_payload),
        "external_receipt_id": receipt["receipt_id"],
        "external_receipt_payload_sha256": _payload_sha(receipt_payload),
        "external_ledger_id": ledger["ledger_id"],
        "external_ledger_payload_sha256": _payload_sha(ledger_payload),
        "external_checkpoint_id": checkpoint["checkpoint_id"],
        "external_checkpoint_payload_sha256": _payload_sha(checkpoint_payload),
        "external_head_id": head["head_id"],
        "external_head_payload_sha256": _payload_sha(head_payload),
        "witness_sequence": 1,
        "tree_size": 1,
        "signed_tree_root_sha256": expected_root,
        "attempt_context_id": expected_attempt_context_id,
        "anchor_sequence": 1,
        "effective_state": "CURRENT_UNSUPERSEDED",
        "superseded": False,
        "revoked": False,
        "map_leaf_index": 0,
        "map_tree_size": 1,
        "map_root_sha256": None,
        "map_inclusion_path_sha256_in_exact_order": [],
        "presented_to_live_consistency_path_sha256_in_exact_order": [],
    }
    mapping["map_root_sha256"] = _current_map_leaf_root(mapping)
    return anchor, receipt, ledger, checkpoint, head, mapping


def _genesis_mapping(attempt_context_id: str) -> dict[str, object]:
    return {
        "external_anchor_id": None,
        "external_anchor_payload_sha256": None,
        "external_receipt_id": None,
        "external_receipt_payload_sha256": None,
        "external_ledger_id": None,
        "external_ledger_payload_sha256": None,
        "external_checkpoint_id": None,
        "external_checkpoint_payload_sha256": None,
        "external_head_id": None,
        "external_head_payload_sha256": None,
        "witness_sequence": 0,
        "tree_size": 0,
        "signed_tree_root_sha256": None,
        "attempt_context_id": attempt_context_id,
        "anchor_sequence": 0,
        "effective_state": "ABSENT",
        "superseded": False,
        "revoked": False,
        "map_leaf_index": None,
        "map_tree_size": 0,
        "map_root_sha256": None,
        "map_inclusion_path_sha256_in_exact_order": [],
        "presented_to_live_consistency_path_sha256_in_exact_order": [],
    }


def _mapping(
    value: object, label: str, expected_attempt_context_id: str
) -> dict[str, object]:
    mapping = _object(value, DURABLE_MAPPING_FIELDS_V1, label)
    sequence = _integer(mapping["witness_sequence"], f"{label} witness sequence")
    tree_size = _integer(mapping["tree_size"], f"{label} tree size")
    anchor_sequence = _integer(
        mapping["anchor_sequence"], f"{label} anchor sequence"
    )
    map_tree_size = _integer(mapping["map_tree_size"], f"{label} map tree size")
    if mapping["attempt_context_id"] != expected_attempt_context_id:
        _fail(FAIL_BINDING_V1, f"{label} attempt context differs")
    if mapping["superseded"] is not False or mapping["revoked"] is not False:
        _fail(FAIL_BINDING_V1, f"{label} state booleans differ")
    for field in (
        "map_inclusion_path_sha256_in_exact_order",
        "presented_to_live_consistency_path_sha256_in_exact_order",
    ):
        path = mapping[field]
        if type(path) is not list or len(path) > 64:
            _fail(FAIL_SCHEMA_V1, f"{label} {field} shape differs")
        for index, digest in enumerate(path):
            _hex64(digest, f"{label} {field} {index}")
    if sequence == 0 or tree_size == 0:
        if mapping != _genesis_mapping(expected_attempt_context_id):
            _fail(FAIL_BINDING_V1, f"{label} partial genesis mapping differs")
        return mapping
    if sequence != 1 or tree_size != 1 or anchor_sequence != 1:
        _fail(FAIL_BINDING_V1, f"{label} only initial mapping is frozen")
    for field in (
        "external_anchor_id", "external_receipt_id", "external_ledger_id",
        "external_checkpoint_id", "external_head_id",
    ):
        _identifier(mapping[field], f"{label} {field}")
    for field in (
        "external_anchor_payload_sha256", "external_receipt_payload_sha256",
        "external_ledger_payload_sha256", "external_checkpoint_payload_sha256",
        "external_head_payload_sha256", "signed_tree_root_sha256",
    ):
        _hex64(mapping[field], f"{label} {field}")
    if (
        mapping["effective_state"] != "CURRENT_UNSUPERSEDED"
        or mapping["superseded"] is not False
        or mapping["revoked"] is not False
        or type(mapping["map_leaf_index"]) is not int
        or mapping["map_leaf_index"] != 0
        or map_tree_size != 1
        or mapping["map_inclusion_path_sha256_in_exact_order"] != []
        or mapping["presented_to_live_consistency_path_sha256_in_exact_order"]
        != []
        or mapping["map_root_sha256"] != _current_map_leaf_root(mapping)
    ):
        _fail(FAIL_BINDING_V1, f"{label} authenticated current mapping differs")
    return mapping


def _runtime_custody_binding(
    value: object, policy_body: Mapping[str, object], label: str
) -> dict[str, object]:
    binding = _object(value, RUNTIME_CUSTODY_BINDING_FIELDS_V1, label)
    for field in (
        "service_binding_id",
        "runtime_instance_id",
        "control_domain_id",
        "runtime_closure_policy_id",
        "runtime_closure_report_id",
        "custody_attestation_policy_id",
        "custody_attestation_id",
    ):
        _identifier(binding[field], f"{label} {field}")
    for field in (
        "peer_pid",
        "peer_uid",
        "peer_gid",
        "user_namespace_inode",
        "process_start_ticks",
    ):
        _integer(binding[field], f"{label} {field}", 1)
    for field in (
        "runtime_closure_report_payload_sha256",
        "custody_attestation_payload_sha256",
    ):
        _hex64(binding[field], f"{label} {field}")
    if (
        binding["service_binding_id"] != policy_body["service_binding_id"]
        or binding["peer_uid"] != policy_body["service_uid"]
        or binding["peer_gid"] != policy_body["service_gid"]
        or binding["user_namespace_inode"]
        != policy_body["service_user_namespace_inode"]
        or binding["control_domain_id"]
        != policy_body["witness_control_domain_id"]
        or binding["runtime_closure_policy_id"]
        != policy_body["runtime_closure_policy_id"]
        or binding["custody_attestation_policy_id"]
        != policy_body["custody_attestation_policy_id"]
        or binding["online_key_nonexportable"] is not True
    ):
        _fail(FAIL_BINDING_V1, f"{label} policy/runtime/custody binding differs")
    return binding


def _common_body(
    body: Mapping[str, object], *, lock: object, profile: object, registry: object,
    attempt_context_id: str, bootstrap_policy_id: str,
    online_policy_id: str, durable_policy_id: str,
    label: str,
) -> None:
    if (
        body["status"] != CANDIDATE_STATUS_V1
        or body["claim_level"] != CANDIDATE_CLAIM_LEVEL_V1
        or body["stage_b_trust_lock_id"] != lock.artifact_id
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["attempt_context_id"] != attempt_context_id
        or body["bootstrap_external_witness_policy_id"] != bootstrap_policy_id
        or body["online_witness_policy_id"] != online_policy_id
        or body["durable_anchor_policy_id"] != durable_policy_id
    ):
        _fail(FAIL_BINDING_V1, f"{label} common authority/attempt/policy binding differs")


def _load_signed_protocol_object(
    payload: bytes,
    *,
    label: str,
    schema: str,
    fields: frozenset[str],
    body_fields: frozenset[str],
    artifact_kind: str,
    id_field: str,
    signature_field: str,
    expected_key_id: str,
    expected_key_epoch: int,
    expected_purpose: str,
    public_key_hex: str,
) -> tuple[dict[str, object], dict[str, object]]:
    root = _object(_decode(payload, label), fields, label)
    body = _object(root["body"], body_fields, f"{label} body")
    if root["schema_version"] != schema or body["schema_version"] != schema:
        _fail(FAIL_SCHEMA_V1, f"{label} schema differs")
    expected_id = phase2b_step5_external_witness_admission_content_id_v1(
        artifact_kind, body
    )
    if root[id_field] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, f"{label} content ID differs")
    preimage = phase2b_step5_external_witness_admission_signature_preimage_v1(
        artifact_kind=artifact_kind,
        body=body,
        artifact_id=expected_id,
        signer_key_id=expected_key_id,
        key_epoch=expected_key_epoch,
        purpose=expected_purpose,
    )
    _signature(
        root[signature_field],
        expected_key_id=expected_key_id,
        expected_key_epoch=expected_key_epoch,
        expected_purpose=expected_purpose,
        public_key_hex=public_key_hex,
        preimage=preimage,
        label=label,
    )
    root["body"] = body
    return root, body


@dataclass(frozen=True, slots=True, init=False)
class VerifiedExternalWitnessAdmissionCandidateV1:
    """Privately issued mechanics result; never real admission evidence."""

    schema_version: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    bootstrap_external_witness_policy_id: str
    online_witness_policy_id: str
    durable_anchor_policy_id: str
    admission_transcript_id: str
    admission_candidate_result_id: str
    logical_time_ledger_id: str
    post_commit_logical_time_head_id: str
    verified_signature_count: int
    verified_signature_scope: str
    verified_protocol_object_count: int
    raw_authority_graph_verified: bool
    three_oob_policy_pins_verified: bool
    key_sets_pairwise_disjoint: bool
    presented_q1_a_q2_chain_verified: bool
    supplied_post_q2_logical_time_chain_verified: bool
    runtime_closure_verified: bool
    key_custody_verified: bool
    distinct_live_uid_gid_verified: bool
    private_key_present: bool
    spool_path_present: bool
    decoded_mapping_input_present: bool
    callback_present: bool
    caller_nonce_present: bool
    live_channel_verified: bool
    external_currentness_verified: bool
    durability_verified: bool
    non_equivocation_verified: bool
    global_anti_fork_verified: bool
    authority_evidence: bool
    r01_satisfied: bool
    readiness_evidence: bool
    execution_authorized: bool

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiedExternalWitnessAdmissionCandidateV1":
        raise TypeError("verified external admission candidates are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified external admission candidates are privately issued")


def _build_result_lineage():
    result_type = VerifiedExternalWitnessAdmissionCandidateV1
    error_type = Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    fail_result = FAIL_RESULT_V1
    schema = ADMISSION_CANDIDATE_RESULT_SCHEMA_V1
    minimum_verified_signature_count = NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1 + 2
    verified_signature_scope = VERIFIED_SIGNATURE_SCOPE_V1
    verified_protocol_object_count = VERIFIED_PROTOCOL_OBJECT_COUNT_V1
    scalar_fields = tuple(result_type.__dataclass_fields__)
    issued: dict[int, tuple[VerifiedExternalWitnessAdmissionCandidateV1, tuple[object, ...]]] = {}

    def fail_exact(detail: str) -> NoReturn:
        raise error_type(fail_result, detail)

    def require(value: object) -> VerifiedExternalWitnessAdmissionCandidateV1:
        if type(value) is not result_type:
            fail_exact("verified admission candidate exact type differs")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            fail_exact("verified admission candidate lacks private lineage")
        try:
            observed = tuple(object.__getattribute__(value, name) for name in scalar_fields)
        except (AttributeError, TypeError):
            observed = ()
        if len(observed) != len(record[1]) or any(
            type(current) is not type(original) or current != original
            for current, original in zip(observed, record[1], strict=True)
        ):
            fail_exact("verified admission candidate was mutated")
        if not (
            value.schema_version == schema
            and type(value.verified_signature_count) is int
            and value.verified_signature_count >= minimum_verified_signature_count
            and value.verified_signature_scope == verified_signature_scope
            and value.verified_protocol_object_count
            == verified_protocol_object_count
            and value.raw_authority_graph_verified is True
            and value.three_oob_policy_pins_verified is True
            and value.key_sets_pairwise_disjoint is True
            and value.presented_q1_a_q2_chain_verified is True
            and value.supplied_post_q2_logical_time_chain_verified is True
            and value.runtime_closure_verified is False
            and value.key_custody_verified is False
            and value.distinct_live_uid_gid_verified is False
            and value.private_key_present is False
            and value.spool_path_present is False
            and value.decoded_mapping_input_present is False
            and value.callback_present is False
            and value.caller_nonce_present is False
            and value.live_channel_verified is False
            and value.external_currentness_verified is False
            and value.durability_verified is False
            and value.non_equivocation_verified is False
            and value.global_anti_fork_verified is False
            and value.authority_evidence is False
            and value.r01_satisfied is False
            and value.readiness_evidence is False
            and value.execution_authorized is False
        ):
            fail_exact("verified admission candidate semantics differ")
        return value

    def issue(**values: object) -> VerifiedExternalWitnessAdmissionCandidateV1:
        if frozenset(values) != frozenset(scalar_fields):
            fail_exact("private result issuance keyset differs")
        result = object.__new__(result_type)
        for name in scalar_fields:
            object.__setattr__(result, name, values[name])
        snapshot = tuple(values[name] for name in scalar_fields)
        issued[id(result)] = (result, snapshot)
        return require(result)

    return issue, require


(_issue_result, _require_result) = _build_result_lineage()
del _build_result_lineage


def require_verified_external_witness_admission_candidate_v1(
    value: object,
) -> VerifiedExternalWitnessAdmissionCandidateV1:
    return _require_result(value)


def _verify_external_witness_admission_transcript_candidate_implementation_v1(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    bootstrap_external_witness_policy_payload: bytes,
    online_witness_policy_payload: bytes,
    durable_anchor_policy_payload: bytes,
    candidate_anchor_payload: bytes,
    candidate_receipt_payload: bytes,
    candidate_ledger_payload: bytes,
    candidate_checkpoint_payload: bytes,
    candidate_head_payload: bytes,
    q1_prepare_request_payload: bytes,
    q1_prepare_response_payload: bytes,
    a_compare_and_append_request_payload: bytes,
    a_durable_checkpoint_payload: bytes,
    a_durable_head_payload: bytes,
    a_compare_and_append_receipt_payload: bytes,
    q2_confirm_request_payload: bytes,
    q2_confirm_response_payload: bytes,
    admission_transcript_payload: bytes,
    post_commit_logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
    post_commit_signed_logical_time_token_payloads_in_exact_order: tuple[bytes, ...],
    post_commit_logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    post_commit_logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
    admission_candidate_result_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
    out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_online_witness_policy_id_bytes: bytes,
    out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
    out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
    expected_attempt_context_id_bytes: bytes,
    _load_authority=_load_raw_authority_graph,
    _get_role_keys=_role_keys,
    _load_bootstrap=_load_bootstrap_policy,
    _load_online=_load_online_policy,
    _load_durable=_load_durable_policy,
    _load_candidates=_load_candidate_artifacts,
    _load_protocol=_load_signed_protocol_object,
    _check_common=_common_body,
    _check_mapping=_mapping,
    _check_runtime=_runtime_custody_binding,
    _genesis=_genesis_mapping,
    _sha=_payload_sha,
    _logical_event_type=_ExpectedLogicalEvent,
    _logical_verify=_verify_logical_replay,
    _logical_require=_require_logical_replay,
    _logical_error=_LogicalTimeError,
    _issue=_issue_result,
) -> VerifiedExternalWitnessAdmissionCandidateV1:
    attempt_id = _decode_expected_identifier_bytes(
        expected_attempt_context_id_bytes, "expected attempt context ID"
    )
    lock, profile, registry = _load_authority(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_lock_payload_sha256_digest=(
            out_of_band_expected_lock_payload_sha256_digest
        ),
    )
    role_keys = _get_role_keys(registry)
    bootstrap_policy = _load_bootstrap(
        bootstrap_external_witness_policy_payload,
        expected_id_bytes=(
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes
        ),
        expected_payload_sha256_digest=(
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest
        ),
        role_keys=role_keys,
    )
    online_policy = _load_online(
        online_witness_policy_payload,
        expected_id_bytes=out_of_band_expected_online_witness_policy_id_bytes,
        expected_payload_sha256_digest=(
            out_of_band_expected_online_witness_policy_payload_sha256_digest
        ),
        role_keys=role_keys,
        authority_profile=profile,
        bootstrap_policy_payload=bootstrap_external_witness_policy_payload,
        bootstrap_policy=bootstrap_policy,
    )
    durable_policy = _load_durable(
        durable_anchor_policy_payload,
        expected_id_bytes=out_of_band_expected_durable_anchor_policy_id_bytes,
        expected_payload_sha256_digest=(
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest
        ),
        role_keys=role_keys,
        bootstrap_policy_payload=bootstrap_external_witness_policy_payload,
        bootstrap_policy=bootstrap_policy,
        online_policy=online_policy,
    )
    (candidate_anchor, _, _, _, _, candidate_mapping) = _load_candidates(
        anchor_payload=candidate_anchor_payload,
        receipt_payload=candidate_receipt_payload,
        ledger_payload=candidate_ledger_payload,
        checkpoint_payload=candidate_checkpoint_payload,
        head_payload=candidate_head_payload,
        lock=lock,
        profile=profile,
        registry=registry,
        role_keys=role_keys,
        bootstrap_policy=bootstrap_policy,
        expected_attempt_context_id=attempt_id,
    )
    candidate_anchor_body = candidate_anchor["body"]
    online_body = online_policy["body"]
    durable_body = durable_policy["body"]
    bootstrap_policy_id = bootstrap_policy["external_witness_policy_id"]
    online_policy_id = online_policy["online_witness_policy_id"]
    durable_policy_id = durable_policy["durable_anchor_policy_id"]
    bootstrap_key = online_body["bootstrap_key_id"]
    bootstrap_epoch = online_body["bootstrap_key_epoch"]
    bootstrap_public = online_body["bootstrap_public_key_hex"]
    online_key = online_body["online_key_id"]
    online_epoch = online_body["online_key_epoch"]
    online_public = online_body["online_public_key_hex"]
    a_key = durable_body["anchor_key_id"]
    a_epoch = durable_body["anchor_key_epoch"]
    a_public = durable_body["anchor_public_key_hex"]
    durable_high_water_mapping = _check_mapping(
        durable_body["oob_high_water_mapping"],
        "durable policy OOB high-water mapping",
        attempt_id,
    )
    if (
        durable_high_water_mapping != _genesis(attempt_id)
        or durable_body["oob_high_water_checkpoint_sequence"] != 0
        or durable_body["oob_high_water_head_sequence"] != 0
        or durable_body["oob_high_water_checkpoint_id"]
        != durable_body["genesis_checkpoint_id"]
        or durable_body["oob_high_water_head_id"]
        != durable_body["genesis_head_id"]
        or durable_body["oob_high_water_checkpoint_payload_sha256"] != "0" * 64
        or durable_body["oob_high_water_head_payload_sha256"] != "0" * 64
        or durable_body["oob_high_water_tree_root_sha256"]
        != durable_body["genesis_tree_root_sha256"]
    ):
        _fail(
            FAIL_BINDING_V1,
            "durable policy OOB high-water sequence/root/genesis binding differs",
        )

    common_arguments = {
        "lock": lock,
        "profile": profile,
        "registry": registry,
        "attempt_context_id": attempt_id,
        "bootstrap_policy_id": bootstrap_policy_id,
        "online_policy_id": online_policy_id,
        "durable_policy_id": durable_policy_id,
    }
    q1_request, q1_request_body = _load_protocol(
        q1_prepare_request_payload,
        label="Q1 PREPARE request", schema=Q1_PREPARE_REQUEST_SCHEMA_V1,
        fields=Q1_PREPARE_REQUEST_FIELDS_V1,
        body_fields=Q1_PREPARE_REQUEST_BODY_FIELDS_V1,
        artifact_kind="q1_prepare_request", id_field="q1_request_id",
        signature_field="signature", expected_key_id=bootstrap_key,
        expected_key_epoch=bootstrap_epoch,
        expected_purpose=SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1,
        public_key_hex=bootstrap_public,
    )
    _check_common(q1_request_body, label="Q1 request", **common_arguments)
    prepare_nonce = _nonce(q1_request_body["prepare_nonce"], "Q1 prepare nonce")
    if (
        _check_mapping(
            q1_request_body["candidate_mapping"], "Q1 candidate mapping", attempt_id
        ) != candidate_mapping
        or q1_request_body["expected_previous_durable_checkpoint_id"]
        != durable_body["oob_high_water_checkpoint_id"]
        or q1_request_body["expected_previous_durable_head_id"]
        != durable_body["oob_high_water_head_id"]
        or q1_request_body["expected_previous_durable_checkpoint_payload_sha256"]
        != durable_body["oob_high_water_checkpoint_payload_sha256"]
        or q1_request_body["expected_previous_durable_head_payload_sha256"]
        != durable_body["oob_high_water_head_payload_sha256"]
        or q1_request_body["expected_previous_durable_checkpoint_sequence"]
        != durable_body["oob_high_water_checkpoint_sequence"]
        or q1_request_body["expected_previous_durable_head_sequence"]
        != durable_body["oob_high_water_head_sequence"]
        or q1_request_body["expected_previous_durable_tree_root_sha256"]
        != durable_body["oob_high_water_tree_root_sha256"]
    ):
        _fail(FAIL_BINDING_V1, "Q1 candidate or previous durable mapping differs")

    q1_response, q1_response_body = _load_protocol(
        q1_prepare_response_payload,
        label="Q1 PREPARE response", schema=Q1_PREPARE_RESPONSE_SCHEMA_V1,
        fields=Q1_PREPARE_RESPONSE_FIELDS_V1,
        body_fields=Q1_PREPARE_RESPONSE_BODY_FIELDS_V1,
        artifact_kind="q1_prepare_response", id_field="q1_response_id",
        signature_field="signature", expected_key_id=online_key,
        expected_key_epoch=online_epoch,
        expected_purpose=SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1,
        public_key_hex=online_public,
    )
    _check_common(q1_response_body, label="Q1 response", **common_arguments)
    confirmation_nonce = _nonce(q1_response_body["confirmation_nonce"], "confirmation nonce")
    q1_runtime_custody = _check_runtime(
        q1_response_body["runtime_custody_binding"],
        online_body,
        "Q1 runtime/custody binding",
    )
    if confirmation_nonce == prepare_nonce:
        _fail(FAIL_BINDING_V1, "prepare and confirmation nonces must be distinct")
    if (
        q1_response_body["q1_request_id"] != q1_request["q1_request_id"]
        or q1_response_body["q1_request_payload_sha256"] != _sha(q1_prepare_request_payload)
        or q1_response_body["prepare_nonce"] != prepare_nonce
        or _check_mapping(
            q1_response_body["observed_current_mapping"],
            "Q1 observed mapping",
            attempt_id,
        ) != candidate_mapping
        or _check_mapping(
            q1_response_body["prepared_candidate_mapping"],
            "Q1 prepared mapping",
            attempt_id,
        ) != candidate_mapping
        or q1_response_body["prepare_outcome"] != PREPARE_OUTCOME_V1
    ):
        _fail(FAIL_BINDING_V1, "Q1 PREPARE response linkage/outcome differs")

    a_request, a_request_body = _load_protocol(
        a_compare_and_append_request_payload,
        label="A compare-and-append request", schema=A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        fields=A_COMPARE_AND_APPEND_REQUEST_FIELDS_V1,
        body_fields=A_COMPARE_AND_APPEND_REQUEST_BODY_FIELDS_V1,
        artifact_kind="a_compare_and_append_request", id_field="a_request_id",
        signature_field="signature", expected_key_id=online_key,
        expected_key_epoch=online_epoch,
        expected_purpose=SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
        public_key_hex=online_public,
    )
    _check_common(a_request_body, label="A request", **common_arguments)
    if (
        a_request_body["q1_request_id"] != q1_request["q1_request_id"]
        or a_request_body["q1_request_payload_sha256"] != _sha(q1_prepare_request_payload)
        or a_request_body["q1_response_id"] != q1_response["q1_response_id"]
        or a_request_body["q1_response_payload_sha256"] != _sha(q1_prepare_response_payload)
        or a_request_body["prepare_nonce"] != prepare_nonce
        or a_request_body["confirmation_nonce"] != confirmation_nonce
        or _check_mapping(
            a_request_body["expected_current_mapping"], "A expected mapping", attempt_id
        ) != durable_high_water_mapping
        or _check_mapping(
            a_request_body["proposed_current_mapping"], "A proposed mapping", attempt_id
        ) != candidate_mapping
        or a_request_body["expected_previous_durable_checkpoint_id"]
        != durable_body["oob_high_water_checkpoint_id"]
        or a_request_body["expected_previous_durable_checkpoint_payload_sha256"]
        != durable_body["oob_high_water_checkpoint_payload_sha256"]
        or a_request_body["expected_previous_durable_head_id"]
        != durable_body["oob_high_water_head_id"]
        or a_request_body["expected_previous_durable_head_payload_sha256"]
        != durable_body["oob_high_water_head_payload_sha256"]
        or a_request_body["expected_previous_durable_checkpoint_sequence"]
        != durable_body["oob_high_water_checkpoint_sequence"]
        or a_request_body["expected_previous_durable_head_sequence"]
        != durable_body["oob_high_water_head_sequence"]
        or a_request_body["expected_previous_durable_tree_root_sha256"]
        != durable_body["oob_high_water_tree_root_sha256"]
        or a_request_body["compare_and_append_mode"]
        != "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
    ):
        _fail(FAIL_BINDING_V1, "A request Q1/CAS binding differs")

    a_checkpoint, a_checkpoint_body = _load_protocol(
        a_durable_checkpoint_payload,
        label="A durable checkpoint", schema=A_DURABLE_CHECKPOINT_SCHEMA_V1,
        fields=A_DURABLE_CHECKPOINT_FIELDS_V1,
        body_fields=A_DURABLE_CHECKPOINT_BODY_FIELDS_V1,
        artifact_kind="a_durable_checkpoint", id_field="a_checkpoint_id",
        signature_field="signature", expected_key_id=a_key,
        expected_key_epoch=a_epoch,
        expected_purpose=SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1,
        public_key_hex=a_public,
    )
    _check_common(a_checkpoint_body, label="A checkpoint", **common_arguments)
    if (
        a_checkpoint_body["a_request_id"] != a_request["a_request_id"]
        or a_checkpoint_body["a_request_payload_sha256"] != _sha(a_compare_and_append_request_payload)
        or a_checkpoint_body["prepare_nonce"] != prepare_nonce
        or a_checkpoint_body["confirmation_nonce"] != confirmation_nonce
        or _check_mapping(
            a_checkpoint_body["previous_mapping"],
            "A checkpoint previous mapping",
            attempt_id,
        ) != durable_high_water_mapping
        or _check_mapping(
            a_checkpoint_body["committed_mapping"],
            "A checkpoint committed mapping",
            attempt_id,
        ) != candidate_mapping
        or a_checkpoint_body["cas_outcome"] != CAS_OUTCOME_V1
        or type(a_checkpoint_body["checkpoint_sequence"]) is not int
        or a_checkpoint_body["checkpoint_sequence"]
        != durable_body["oob_high_water_checkpoint_sequence"] + 1
        or a_checkpoint_body["previous_a_checkpoint_id"]
        != durable_body["oob_high_water_checkpoint_id"]
    ):
        _fail(FAIL_BINDING_V1, "A checkpoint transition differs")

    a_head, a_head_body = _load_protocol(
        a_durable_head_payload,
        label="A durable head", schema=A_DURABLE_HEAD_SCHEMA_V1,
        fields=A_DURABLE_HEAD_FIELDS_V1,
        body_fields=A_DURABLE_HEAD_BODY_FIELDS_V1,
        artifact_kind="a_durable_head", id_field="a_head_id",
        signature_field="signature", expected_key_id=a_key,
        expected_key_epoch=a_epoch,
        expected_purpose=SIGN_A_DURABLE_HEAD_PURPOSE_V1,
        public_key_hex=a_public,
    )
    _check_common(a_head_body, label="A head", **common_arguments)
    if (
        a_head_body["a_request_id"] != a_request["a_request_id"]
        or a_head_body["a_request_payload_sha256"] != _sha(a_compare_and_append_request_payload)
        or a_head_body["a_checkpoint_id"] != a_checkpoint["a_checkpoint_id"]
        or a_head_body["a_checkpoint_payload_sha256"] != _sha(a_durable_checkpoint_payload)
        or a_head_body["prepare_nonce"] != prepare_nonce
        or a_head_body["confirmation_nonce"] != confirmation_nonce
        or _check_mapping(
            a_head_body["committed_mapping"],
            "A head committed mapping",
            attempt_id,
        ) != candidate_mapping
        or a_head_body["cas_outcome"] != CAS_OUTCOME_V1
        or type(a_head_body["head_sequence"]) is not int
        or a_head_body["head_sequence"]
        != durable_body["oob_high_water_head_sequence"] + 1
        or a_head_body["previous_a_head_id"]
        != durable_body["oob_high_water_head_id"]
    ):
        _fail(FAIL_BINDING_V1, "A head transition differs")

    a_receipt, a_receipt_body = _load_protocol(
        a_compare_and_append_receipt_payload,
        label="A compare-and-append receipt", schema=A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1,
        fields=A_COMPARE_AND_APPEND_RECEIPT_FIELDS_V1,
        body_fields=A_COMPARE_AND_APPEND_RECEIPT_BODY_FIELDS_V1,
        artifact_kind="a_compare_and_append_receipt", id_field="a_receipt_id",
        signature_field="signature", expected_key_id=a_key,
        expected_key_epoch=a_epoch,
        expected_purpose=SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1,
        public_key_hex=a_public,
    )
    _check_common(a_receipt_body, label="A receipt", **common_arguments)
    if (
        a_receipt_body["a_request_id"] != a_request["a_request_id"]
        or a_receipt_body["a_request_payload_sha256"] != _sha(a_compare_and_append_request_payload)
        or a_receipt_body["a_checkpoint_id"] != a_checkpoint["a_checkpoint_id"]
        or a_receipt_body["a_checkpoint_payload_sha256"] != _sha(a_durable_checkpoint_payload)
        or a_receipt_body["a_head_id"] != a_head["a_head_id"]
        or a_receipt_body["a_head_payload_sha256"] != _sha(a_durable_head_payload)
        or a_receipt_body["prepare_nonce"] != prepare_nonce
        or a_receipt_body["confirmation_nonce"] != confirmation_nonce
        or _check_mapping(
            a_receipt_body["previous_mapping"],
            "A receipt previous mapping",
            attempt_id,
        ) != durable_high_water_mapping
        or _check_mapping(
            a_receipt_body["committed_mapping"],
            "A receipt committed mapping",
            attempt_id,
        ) != candidate_mapping
        or a_receipt_body["cas_outcome"] != CAS_OUTCOME_V1
        or a_receipt_body["compare_matched"] is not True
        or a_receipt_body["append_applied"] is not True
        or a_receipt_body["oob_high_water_checkpoint_id"]
        != durable_body["oob_high_water_checkpoint_id"]
        or a_receipt_body["oob_high_water_head_id"]
        != durable_body["oob_high_water_head_id"]
        or a_receipt_body["durable_before_return"] is not True
    ):
        _fail(FAIL_BINDING_V1, "A receipt exact CAS outcome differs")

    q2_request, q2_request_body = _load_protocol(
        q2_confirm_request_payload,
        label="Q2 CONFIRM request", schema=Q2_CONFIRM_REQUEST_SCHEMA_V1,
        fields=Q2_CONFIRM_REQUEST_FIELDS_V1,
        body_fields=Q2_CONFIRM_REQUEST_BODY_FIELDS_V1,
        artifact_kind="q2_confirm_request", id_field="q2_request_id",
        signature_field="signature", expected_key_id=bootstrap_key,
        expected_key_epoch=bootstrap_epoch,
        expected_purpose=SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1,
        public_key_hex=bootstrap_public,
    )
    _check_common(q2_request_body, label="Q2 request", **common_arguments)
    if (
        q2_request_body["q1_response_id"] != q1_response["q1_response_id"]
        or q2_request_body["q1_response_payload_sha256"] != _sha(q1_prepare_response_payload)
        or q2_request_body["a_request_id"] != a_request["a_request_id"]
        or q2_request_body["a_request_payload_sha256"] != _sha(a_compare_and_append_request_payload)
        or q2_request_body["a_receipt_id"] != a_receipt["a_receipt_id"]
        or q2_request_body["a_receipt_payload_sha256"] != _sha(a_compare_and_append_receipt_payload)
        or q2_request_body["a_checkpoint_id"] != a_checkpoint["a_checkpoint_id"]
        or q2_request_body["a_checkpoint_payload_sha256"] != _sha(a_durable_checkpoint_payload)
        or q2_request_body["a_head_id"] != a_head["a_head_id"]
        or q2_request_body["a_head_payload_sha256"] != _sha(a_durable_head_payload)
        or q2_request_body["prepare_nonce"] != prepare_nonce
        or q2_request_body["confirmation_nonce"] != confirmation_nonce
        or _check_mapping(
            q2_request_body["expected_current_mapping"],
            "Q2 expected mapping",
            attempt_id,
        ) != candidate_mapping
        or q2_request_body["expected_cas_outcome"] != CAS_OUTCOME_V1
    ):
        _fail(FAIL_BINDING_V1, "Q2 request A/CAS binding differs")

    q2_response, q2_response_body = _load_protocol(
        q2_confirm_response_payload,
        label="Q2 CONFIRM response", schema=Q2_CONFIRM_RESPONSE_SCHEMA_V1,
        fields=Q2_CONFIRM_RESPONSE_FIELDS_V1,
        body_fields=Q2_CONFIRM_RESPONSE_BODY_FIELDS_V1,
        artifact_kind="q2_confirm_response", id_field="q2_response_id",
        signature_field="signature", expected_key_id=online_key,
        expected_key_epoch=online_epoch,
        expected_purpose=SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1,
        public_key_hex=online_public,
    )
    _check_common(q2_response_body, label="Q2 response", **common_arguments)
    q2_runtime_custody = _check_runtime(
        q2_response_body["runtime_custody_binding"],
        online_body,
        "Q2 runtime/custody binding",
    )
    if (
        q2_response_body["q2_request_id"] != q2_request["q2_request_id"]
        or q2_response_body["q2_request_payload_sha256"] != _sha(q2_confirm_request_payload)
        or q2_response_body["a_receipt_id"] != a_receipt["a_receipt_id"]
        or q2_response_body["a_receipt_payload_sha256"] != _sha(a_compare_and_append_receipt_payload)
        or q2_response_body["prepare_nonce"] != prepare_nonce
        or q2_response_body["confirmation_nonce"] != confirmation_nonce
        or _check_mapping(
            q2_response_body["observed_current_mapping"],
            "Q2 observed mapping",
            attempt_id,
        ) != candidate_mapping
        or q2_response_body["observed_cas_outcome"] != CAS_OUTCOME_V1
        or q2_response_body["confirm_outcome"] != CONFIRM_OUTCOME_V1
        or q2_runtime_custody != q1_runtime_custody
    ):
        _fail(FAIL_BINDING_V1, "Q2 CONFIRM response binding/outcome differs")

    transcript, transcript_body = _load_protocol(
        admission_transcript_payload,
        label="admission transcript", schema=ADMISSION_TRANSCRIPT_SCHEMA_V1,
        fields=ADMISSION_TRANSCRIPT_FIELDS_V1,
        body_fields=ADMISSION_TRANSCRIPT_BODY_FIELDS_V1,
        artifact_kind="admission_transcript", id_field="admission_transcript_id",
        signature_field="signature", expected_key_id=bootstrap_key,
        expected_key_epoch=bootstrap_epoch,
        expected_purpose=SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1,
        public_key_hex=bootstrap_public,
    )
    _check_common(transcript_body, label="admission transcript", **common_arguments)
    transcript_bindings = (
        transcript_body["prepare_nonce"] == prepare_nonce,
        transcript_body["confirmation_nonce"] == confirmation_nonce,
        _check_mapping(
            transcript_body["candidate_mapping"],
            "transcript candidate mapping",
            attempt_id,
        ) == candidate_mapping,
        transcript_body["q1_request_id"] == q1_request["q1_request_id"],
        transcript_body["q1_request_payload_sha256"] == _sha(q1_prepare_request_payload),
        transcript_body["q1_response_id"] == q1_response["q1_response_id"],
        transcript_body["q1_response_payload_sha256"] == _sha(q1_prepare_response_payload),
        transcript_body["a_request_id"] == a_request["a_request_id"],
        transcript_body["a_request_payload_sha256"] == _sha(a_compare_and_append_request_payload),
        transcript_body["a_checkpoint_id"] == a_checkpoint["a_checkpoint_id"],
        transcript_body["a_checkpoint_payload_sha256"] == _sha(a_durable_checkpoint_payload),
        transcript_body["a_head_id"] == a_head["a_head_id"],
        transcript_body["a_head_payload_sha256"] == _sha(a_durable_head_payload),
        transcript_body["a_receipt_id"] == a_receipt["a_receipt_id"],
        transcript_body["a_receipt_payload_sha256"] == _sha(a_compare_and_append_receipt_payload),
        transcript_body["q2_request_id"] == q2_request["q2_request_id"],
        transcript_body["q2_request_payload_sha256"] == _sha(q2_confirm_request_payload),
        transcript_body["q2_response_id"] == q2_response["q2_response_id"],
        transcript_body["q2_response_payload_sha256"] == _sha(q2_confirm_response_payload),
        transcript_body["prepare_outcome"] == PREPARE_OUTCOME_V1,
        transcript_body["cas_outcome"] == CAS_OUTCOME_V1,
        transcript_body["confirm_outcome"] == CONFIRM_OUTCOME_V1,
        _check_runtime(
            transcript_body["runtime_custody_binding"],
            online_body,
            "transcript runtime/custody binding",
        ) == q1_runtime_custody,
        transcript_body["presented_q1_a_q2_chain_complete"] is True,
        transcript_body["runtime_closure_verified"] is False,
        transcript_body["key_custody_verified"] is False,
        transcript_body["distinct_live_uid_gid_verified"] is False,
        transcript_body["live_channel_verified"] is False,
        transcript_body["external_currentness_verified"] is False,
        transcript_body["durability_verified"] is False,
        transcript_body["non_equivocation_verified"] is False,
        transcript_body["global_anti_fork_verified"] is False,
    )
    post_q2_nonce = _nonce(
        transcript_body["post_q2_logical_time_request_nonce"],
        "post-Q2 logical-time request nonce",
    )
    if post_q2_nonce in {prepare_nonce, confirmation_nonce} or not all(transcript_bindings):
        _fail(FAIL_BINDING_V1, "admission transcript closure or nonce separation differs")

    expected_event = _logical_event_type(
        purpose=POST_Q2_LOGICAL_TIME_PURPOSE_V1,
        bound_artifact_id=transcript["admission_transcript_id"],
        bound_artifact_sha256="sha256:" + _sha(admission_transcript_payload),
        request_nonce=post_q2_nonce,
    )
    supplied_logical_collections = (
        post_commit_logical_time_request_payloads_in_exact_order,
        post_commit_signed_logical_time_token_payloads_in_exact_order,
        post_commit_logical_time_checkpoint_payloads_in_exact_order,
        post_commit_logical_time_head_payloads_in_exact_order,
    )
    if (
        any(type(items) is not tuple for items in supplied_logical_collections)
        or len(post_commit_logical_time_request_payloads_in_exact_order) < 2
        or len(post_commit_signed_logical_time_token_payloads_in_exact_order)
        != len(post_commit_logical_time_request_payloads_in_exact_order)
        or len(post_commit_logical_time_checkpoint_payloads_in_exact_order)
        != len(post_commit_logical_time_request_payloads_in_exact_order)
        or len(post_commit_logical_time_head_payloads_in_exact_order)
        != len(post_commit_logical_time_request_payloads_in_exact_order) + 1
    ):
        _fail(
            FAIL_LOGICAL_TIME_V1,
            "supplied logical-time chain must contain pre-cut ancestry plus one final post-Q2 event",
        )
    try:
        expected_pre_cut_events = tuple(
            _logical_event_type(
                purpose=request["purpose"],
                bound_artifact_id=request["bound_artifact_id"],
                bound_artifact_sha256=request["bound_artifact_sha256"],
                request_nonce=request["request_nonce"],
            )
            for request in (
                _decode(payload, f"pre-cut logical-time request {index}")
                for index, payload in enumerate(
                    post_commit_logical_time_request_payloads_in_exact_order[:-1]
                )
            )
        )
    except (KeyError, TypeError, ValueError, _LogicalTimeError) as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_LOGICAL_TIME_V1,
            "pre-cut logical-time event projection rejected",
        ) from exc
    try:
        logical_result = _logical_verify(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            post_commit_logical_time_request_payloads_in_exact_order,
            post_commit_signed_logical_time_token_payloads_in_exact_order,
            post_commit_logical_time_checkpoint_payloads_in_exact_order,
            post_commit_logical_time_head_payloads_in_exact_order,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
            expected_attempt_context_id=attempt_id,
            expected_event_plan=expected_pre_cut_events + (expected_event,),
        )
        logical_result = _logical_require(logical_result)
    except _logical_error as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1(
            FAIL_LOGICAL_TIME_V1, "post-Q2 raw logical-time chain rejected"
        ) from exc

    penultimate_token_payload = (
        post_commit_signed_logical_time_token_payloads_in_exact_order[-2]
    )
    penultimate_checkpoint_payload = (
        post_commit_logical_time_checkpoint_payloads_in_exact_order[-2]
    )
    penultimate_head_payload = post_commit_logical_time_head_payloads_in_exact_order[-2]
    penultimate_token = _decode(
        penultimate_token_payload, "anchor-bound penultimate logical-time token"
    )
    penultimate_checkpoint = _decode(
        penultimate_checkpoint_payload,
        "anchor-bound penultimate logical-time checkpoint",
    )
    penultimate_head = _decode(
        penultimate_head_payload, "anchor-bound penultimate logical-time head"
    )
    if (
        penultimate_token.get("token_id")
        != candidate_anchor_body["current_cut_token_id"]
        or "sha256:" + _sha(penultimate_token_payload)
        != candidate_anchor_body["current_cut_token_payload_sha256"]
        or penultimate_checkpoint.get("checkpoint_id")
        != candidate_anchor_body["current_cut_checkpoint_id"]
        or "sha256:" + _sha(penultimate_checkpoint_payload)
        != candidate_anchor_body["current_cut_checkpoint_payload_sha256"]
        or penultimate_head.get("head_id")
        != candidate_anchor_body["current_cut_head_id"]
        or "sha256:" + _sha(penultimate_head_payload)
        != candidate_anchor_body["current_cut_head_payload_sha256"]
    ):
        _fail(
            FAIL_LOGICAL_TIME_V1,
            "supplied post-Q2 chain is not an exact descendant of the anchor current-cut head",
        )
    verified_signature_count = (
        NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1
        + logical_result.verified_signature_count
    )

    result_root, result_body = _load_protocol(
        admission_candidate_result_payload,
        label="admission candidate result", schema=ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        fields=ADMISSION_CANDIDATE_RESULT_FIELDS_V1,
        body_fields=ADMISSION_CANDIDATE_RESULT_BODY_FIELDS_V1,
        artifact_kind="admission_candidate_result", id_field="candidate_result_id",
        signature_field="signature", expected_key_id=bootstrap_key,
        expected_key_epoch=bootstrap_epoch,
        expected_purpose=SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1,
        public_key_hex=bootstrap_public,
    )
    _check_common(result_body, label="admission candidate result", **common_arguments)
    expected_result_values = {
        "admission_transcript_id": transcript["admission_transcript_id"],
        "admission_transcript_payload_sha256": _sha(admission_transcript_payload),
        "logical_time_ledger_id": logical_result.ledger_id,
        "post_commit_logical_time_request_id": logical_result.request_ids[-1],
        "post_commit_logical_time_token_id": logical_result.token_ids[-1],
        "post_commit_logical_time_checkpoint_id": logical_result.checkpoint_ids[-1],
        "post_commit_logical_time_head_id": logical_result.final_head_id,
        "post_commit_logical_time_head_payload_sha256": logical_result.final_head_payload_sha256,
        "verified_signature_count": verified_signature_count,
        "verified_signature_scope": VERIFIED_SIGNATURE_SCOPE_V1,
        "presented_transcript_mechanics_verified": True,
        "post_q2_supplied_logical_time_chain_verified": True,
        "runtime_closure_verified": False,
        "key_custody_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
    }
    for name, expected_value in expected_result_values.items():
        observed_value = result_body[name]
        if type(expected_value) is bool:
            if observed_value is not expected_value:
                _fail(
                    FAIL_CLAIM_BOUNDARY_V1,
                    "candidate result mechanics/claim boundary differs",
                )
        elif observed_value != expected_value:
            _fail(
                FAIL_CLAIM_BOUNDARY_V1,
                "candidate result mechanics/claim boundary differs",
            )

    return _issue(
        schema_version=ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        stage_b_trust_lock_id=lock.artifact_id,
        authority_profile_id=profile.artifact_id,
        signer_registry_id=registry.artifact_id,
        attempt_context_id=attempt_id,
        bootstrap_external_witness_policy_id=bootstrap_policy_id,
        online_witness_policy_id=online_policy_id,
        durable_anchor_policy_id=durable_policy_id,
        admission_transcript_id=transcript["admission_transcript_id"],
        admission_candidate_result_id=result_root["candidate_result_id"],
        logical_time_ledger_id=logical_result.ledger_id,
        post_commit_logical_time_head_id=logical_result.final_head_id,
        verified_signature_count=verified_signature_count,
        verified_signature_scope=VERIFIED_SIGNATURE_SCOPE_V1,
        verified_protocol_object_count=VERIFIED_PROTOCOL_OBJECT_COUNT_V1,
        raw_authority_graph_verified=True,
        three_oob_policy_pins_verified=True,
        key_sets_pairwise_disjoint=True,
        presented_q1_a_q2_chain_verified=True,
        supplied_post_q2_logical_time_chain_verified=True,
        runtime_closure_verified=False,
        key_custody_verified=False,
        distinct_live_uid_gid_verified=False,
        private_key_present=False,
        spool_path_present=False,
        decoded_mapping_input_present=False,
        callback_present=False,
        caller_nonce_present=False,
        live_channel_verified=False,
        external_currentness_verified=False,
        durability_verified=False,
        non_equivocation_verified=False,
        global_anti_fork_verified=False,
        authority_evidence=False,
        r01_satisfied=False,
        readiness_evidence=False,
        execution_authorized=False,
    )


def _build_public_verifier_entry_v1(implementation):
    """Expose the bytes-only contract without dependency-injection keywords."""

    def verify_external_witness_admission_transcript_candidate_v1(
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        bootstrap_external_witness_policy_payload: bytes,
        online_witness_policy_payload: bytes,
        durable_anchor_policy_payload: bytes,
        candidate_anchor_payload: bytes,
        candidate_receipt_payload: bytes,
        candidate_ledger_payload: bytes,
        candidate_checkpoint_payload: bytes,
        candidate_head_payload: bytes,
        q1_prepare_request_payload: bytes,
        q1_prepare_response_payload: bytes,
        a_compare_and_append_request_payload: bytes,
        a_durable_checkpoint_payload: bytes,
        a_durable_head_payload: bytes,
        a_compare_and_append_receipt_payload: bytes,
        q2_confirm_request_payload: bytes,
        q2_confirm_response_payload: bytes,
        admission_transcript_payload: bytes,
        post_commit_logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
        post_commit_signed_logical_time_token_payloads_in_exact_order: tuple[bytes, ...],
        post_commit_logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        post_commit_logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
        admission_candidate_result_payload: bytes,
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
        out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
        out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
        out_of_band_expected_online_witness_policy_id_bytes: bytes,
        out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
        out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
        out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
        expected_attempt_context_id_bytes: bytes,
    ) -> VerifiedExternalWitnessAdmissionCandidateV1:
        return implementation(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            bootstrap_external_witness_policy_payload,
            online_witness_policy_payload,
            durable_anchor_policy_payload,
            candidate_anchor_payload,
            candidate_receipt_payload,
            candidate_ledger_payload,
            candidate_checkpoint_payload,
            candidate_head_payload,
            q1_prepare_request_payload,
            q1_prepare_response_payload,
            a_compare_and_append_request_payload,
            a_durable_checkpoint_payload,
            a_durable_head_payload,
            a_compare_and_append_receipt_payload,
            q2_confirm_request_payload,
            q2_confirm_response_payload,
            admission_transcript_payload,
            post_commit_logical_time_request_payloads_in_exact_order,
            post_commit_signed_logical_time_token_payloads_in_exact_order,
            post_commit_logical_time_checkpoint_payloads_in_exact_order,
            post_commit_logical_time_head_payloads_in_exact_order,
            admission_candidate_result_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes=(
                out_of_band_expected_bootstrap_external_witness_policy_id_bytes
            ),
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=(
                out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest
            ),
            out_of_band_expected_online_witness_policy_id_bytes=(
                out_of_band_expected_online_witness_policy_id_bytes
            ),
            out_of_band_expected_online_witness_policy_payload_sha256_digest=(
                out_of_band_expected_online_witness_policy_payload_sha256_digest
            ),
            out_of_band_expected_durable_anchor_policy_id_bytes=(
                out_of_band_expected_durable_anchor_policy_id_bytes
            ),
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest=(
                out_of_band_expected_durable_anchor_policy_payload_sha256_digest
            ),
            expected_attempt_context_id_bytes=expected_attempt_context_id_bytes,
        )

    return verify_external_witness_admission_transcript_candidate_v1


def verify_real_external_witness_admission_v1(
    *args: object, **kwargs: object
) -> NoReturn:
    """Fail closed: no live real-admission channel is implemented here."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "real live/current/durable/non-equivocating external admission is NOT_IMPLEMENTED",
    )


def _definition_captured_function_graph_v1() -> dict[str, object]:
    """Clone every local function and local function-valued default."""

    module_namespace = globals()
    private_namespace = dict(module_namespace)
    originals: dict[str, FunctionType] = {
        name: value
        for name, value in module_namespace.items()
        if type(value) is FunctionType
        and value.__module__ == __name__
        and name != "_definition_captured_function_graph_v1"
    }
    provisional: dict[str, FunctionType] = {}
    provisional_by_identity: dict[int, FunctionType] = {}
    for name, implementation in originals.items():
        clone = FunctionType(
            implementation.__code__, private_namespace, implementation.__name__,
            implementation.__defaults__, implementation.__closure__
        )
        provisional[name] = clone
        provisional_by_identity[id(implementation)] = clone
    private_namespace.update(provisional)

    def provisional_default(value: object) -> object:
        return provisional_by_identity.get(id(value), value)

    captured: dict[str, FunctionType] = {}
    for name, implementation in originals.items():
        defaults = implementation.__defaults__
        clone = FunctionType(
            implementation.__code__, private_namespace, implementation.__name__,
            None if defaults is None else tuple(provisional_default(item) for item in defaults),
            implementation.__closure__,
        )
        clone.__kwdefaults__ = (
            None if implementation.__kwdefaults__ is None else {
                key: provisional_default(item)
                for key, item in implementation.__kwdefaults__.items()
            }
        )
        clone.__annotations__ = dict(implementation.__annotations__)
        clone.__doc__ = implementation.__doc__
        clone.__module__ = implementation.__module__
        clone.__qualname__ = implementation.__qualname__
        captured[name] = clone
    final_by_identity = {id(originals[name]): clone for name, clone in captured.items()}

    def final_default(value: object) -> object:
        return final_by_identity.get(id(value), value)

    for name, implementation in originals.items():
        clone = captured[name]
        defaults = implementation.__defaults__
        clone.__defaults__ = (
            None if defaults is None else tuple(final_default(item) for item in defaults)
        )
        clone.__kwdefaults__ = (
            None if implementation.__kwdefaults__ is None else {
                key: final_default(item)
                for key, item in implementation.__kwdefaults__.items()
            }
        )
    private_namespace.update(captured)
    return captured


_CAPTURED = _definition_captured_function_graph_v1()
canonical_external_witness_admission_json_bytes_v1 = _CAPTURED[
    "canonical_external_witness_admission_json_bytes_v1"
]
phase2b_step5_external_witness_admission_content_id_v1 = _CAPTURED[
    "phase2b_step5_external_witness_admission_content_id_v1"
]
phase2b_step5_external_witness_admission_signature_preimage_v1 = _CAPTURED[
    "phase2b_step5_external_witness_admission_signature_preimage_v1"
]
verify_external_witness_admission_transcript_candidate_v1 = _CAPTURED[
    "_build_public_verifier_entry_v1"
](
    _CAPTURED[
        "_verify_external_witness_admission_transcript_candidate_implementation_v1"
    ]
)
require_verified_external_witness_admission_candidate_v1 = _CAPTURED[
    "require_verified_external_witness_admission_candidate_v1"
]
verify_real_external_witness_admission_v1 = _CAPTURED[
    "verify_real_external_witness_admission_v1"
]
del _CAPTURED
del _definition_captured_function_graph_v1
del _build_public_verifier_entry_v1
del _verify_external_witness_admission_transcript_candidate_implementation_v1
del _issue_result


__all__ = [
    "AUTHORITY_V2_SOURCE_SHA256_V1",
    "EVIDENCE_CONTRACT_SOURCE_SHA256_V1",
    "EXTERNAL_WITNESS_SOURCE_SHA256_V1",
    "LOGICAL_TIME_REPLAY_CORE_SOURCE_SHA256_V1",
    "BOOTSTRAP_EXTERNAL_WITNESS_POLICY_SCHEMA_V1",
    "ONLINE_WITNESS_POLICY_SCHEMA_V1",
    "DURABLE_ANCHOR_POLICY_SCHEMA_V1",
    "Q1_PREPARE_REQUEST_SCHEMA_V1",
    "Q1_PREPARE_RESPONSE_SCHEMA_V1",
    "A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1",
    "A_DURABLE_CHECKPOINT_SCHEMA_V1",
    "A_DURABLE_HEAD_SCHEMA_V1",
    "A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1",
    "Q2_CONFIRM_REQUEST_SCHEMA_V1",
    "Q2_CONFIRM_RESPONSE_SCHEMA_V1",
    "ADMISSION_TRANSCRIPT_SCHEMA_V1",
    "ADMISSION_CANDIDATE_RESULT_SCHEMA_V1",
    "CANDIDATE_STATUS_V1",
    "CANDIDATE_CLAIM_LEVEL_V1",
    "PRESENTED_TRANSCRIPT_CLAIM_BOUNDARY_V1",
    "POST_Q2_LOGICAL_TIME_PURPOSE_V1",
    "NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1",
    "VERIFIED_SIGNATURE_SCOPE_V1",
    "VERIFIED_PROTOCOL_OBJECT_COUNT_V1",
    "PREPARE_OUTCOME_V1",
    "CAS_OUTCOME_V1",
    "CONFIRM_OUTCOME_V1",
    "ALGORITHM_V1",
    "SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1",
    "SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1",
    "SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1",
    "SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1",
    "SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1",
    "SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1",
    "SIGN_A_DURABLE_HEAD_PURPOSE_V1",
    "SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1",
    "SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1",
    "SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1",
    "SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1",
    "SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1",
    "BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1",
    "ONLINE_PURPOSES_IN_EXACT_ORDER_V1",
    "A_PURPOSES_IN_EXACT_ORDER_V1",
    "SIGNATURE_FIELDS_V1",
    "ONLINE_WITNESS_POLICY_FIELDS_V1",
    "ONLINE_WITNESS_POLICY_BODY_FIELDS_V1",
    "BOOTSTRAP_EXTERNAL_WITNESS_POLICY_FIELDS_V1",
    "DURABLE_ANCHOR_POLICY_FIELDS_V1",
    "DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1",
    "DURABLE_MAPPING_FIELDS_V1",
    "RUNTIME_CUSTODY_BINDING_FIELDS_V1",
    "Q1_PREPARE_REQUEST_FIELDS_V1",
    "Q1_PREPARE_REQUEST_BODY_FIELDS_V1",
    "Q1_PREPARE_RESPONSE_FIELDS_V1",
    "Q1_PREPARE_RESPONSE_BODY_FIELDS_V1",
    "A_COMPARE_AND_APPEND_REQUEST_FIELDS_V1",
    "A_COMPARE_AND_APPEND_REQUEST_BODY_FIELDS_V1",
    "A_DURABLE_CHECKPOINT_FIELDS_V1",
    "A_DURABLE_CHECKPOINT_BODY_FIELDS_V1",
    "A_DURABLE_HEAD_FIELDS_V1",
    "A_DURABLE_HEAD_BODY_FIELDS_V1",
    "A_COMPARE_AND_APPEND_RECEIPT_FIELDS_V1",
    "A_COMPARE_AND_APPEND_RECEIPT_BODY_FIELDS_V1",
    "Q2_CONFIRM_REQUEST_FIELDS_V1",
    "Q2_CONFIRM_REQUEST_BODY_FIELDS_V1",
    "Q2_CONFIRM_RESPONSE_FIELDS_V1",
    "Q2_CONFIRM_RESPONSE_BODY_FIELDS_V1",
    "ADMISSION_TRANSCRIPT_FIELDS_V1",
    "ADMISSION_TRANSCRIPT_BODY_FIELDS_V1",
    "ADMISSION_CANDIDATE_RESULT_FIELDS_V1",
    "ADMISSION_CANDIDATE_RESULT_BODY_FIELDS_V1",
    "ONLINE_WITNESS_POLICY_CONTENT_ID_DOMAIN_V1",
    "DURABLE_ANCHOR_POLICY_CONTENT_ID_DOMAIN_V1",
    "Q1_PREPARE_REQUEST_CONTENT_ID_DOMAIN_V1",
    "Q1_PREPARE_RESPONSE_CONTENT_ID_DOMAIN_V1",
    "A_COMPARE_AND_APPEND_REQUEST_CONTENT_ID_DOMAIN_V1",
    "A_DURABLE_CHECKPOINT_CONTENT_ID_DOMAIN_V1",
    "A_DURABLE_HEAD_CONTENT_ID_DOMAIN_V1",
    "A_COMPARE_AND_APPEND_RECEIPT_CONTENT_ID_DOMAIN_V1",
    "Q2_CONFIRM_REQUEST_CONTENT_ID_DOMAIN_V1",
    "Q2_CONFIRM_RESPONSE_CONTENT_ID_DOMAIN_V1",
    "ADMISSION_TRANSCRIPT_CONTENT_ID_DOMAIN_V1",
    "ADMISSION_CANDIDATE_RESULT_CONTENT_ID_DOMAIN_V1",
    "ONLINE_WITNESS_POLICY_SIGNATURE_DOMAIN_V1",
    "Q1_PREPARE_REQUEST_SIGNATURE_DOMAIN_V1",
    "Q1_PREPARE_RESPONSE_SIGNATURE_DOMAIN_V1",
    "A_COMPARE_AND_APPEND_REQUEST_SIGNATURE_DOMAIN_V1",
    "A_DURABLE_CHECKPOINT_SIGNATURE_DOMAIN_V1",
    "A_DURABLE_HEAD_SIGNATURE_DOMAIN_V1",
    "A_COMPARE_AND_APPEND_RECEIPT_SIGNATURE_DOMAIN_V1",
    "Q2_CONFIRM_REQUEST_SIGNATURE_DOMAIN_V1",
    "Q2_CONFIRM_RESPONSE_SIGNATURE_DOMAIN_V1",
    "ADMISSION_TRANSCRIPT_SIGNATURE_DOMAIN_V1",
    "ADMISSION_CANDIDATE_RESULT_SIGNATURE_DOMAIN_V1",
    "DURABLE_ANCHOR_POLICY_SIGNATURE_DOMAIN_V1",
    "FAIL_INPUT_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_BINDING_V1",
    "FAIL_TRUST_ANCHOR_V1",
    "FAIL_SIGNATURE_V1",
    "FAIL_LOGICAL_TIME_V1",
    "FAIL_RESULT_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1",
    "VerifiedExternalWitnessAdmissionCandidateV1",
    "canonical_external_witness_admission_json_bytes_v1",
    "phase2b_step5_external_witness_admission_content_id_v1",
    "phase2b_step5_external_witness_admission_signature_preimage_v1",
    "verify_external_witness_admission_transcript_candidate_v1",
    "require_verified_external_witness_admission_candidate_v1",
    "verify_real_external_witness_admission_v1",
]
