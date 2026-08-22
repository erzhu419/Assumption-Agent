"""Acyclic raw-bytes candidate contract for external-witness admission/2.

This module is a deterministic serialization, content-address, signature, and
presented-transcript integrity verifier.  It composes the frozen b480 admission
candidate as an immutable ancestor and the post-b480 live-overlay policy
mechanics, but deliberately versions every same-session wire object that must
name the new admission/2 result.

The frozen same-session dependency order is::

    live policy -> A/2 policy -> pre-response challenge
        -> W response -> source report -> W custody attestation
        -> admission/2 candidate result
        -> exact six-object CAS replay
        -> A/2 request -> checkpoint -> head -> receipt

The six stored objects, in exact order, are the live policy, challenge,
response, report, custody attestation, and admission/2 result.  No one of those
six may contain a same-session CAS or A/2 transcript reference.  No object
before the result may name that result.  In particular, the b480 candidate ID
in the policy lineage is a completed predecessor and is never reinterpreted as
the admission/2 result.

Acceptance proves only integrity of exact presented raw bytes.  The verifier
has no decoded mapping, callback, private-key, filesystem-path, socket, clock,
process, service, or live-channel input.  It does not establish actual runtime,
key custody, durability, currentness, non-equivocation, authority, R01,
readiness, or execution.  The real admission entry remains NOT_IMPLEMENTED.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct
import threading
from types import FunctionType, MappingProxyType, ModuleType
from typing import Final, Mapping, NoReturn

from . import phase2b_formal_wire_covert_step5_authority_v2 as authority
from . import phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 as admission_v1
from . import phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2 as overlay_v2
from . import phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure
from . import phase2b_formal_wire_covert_step5_store_replay_core_v2 as store_replay
from .rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Import remains possible; verification fails closed without Ed25519.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - deployment image dependent.
    _InvalidSignature = None
    _Ed25519PublicKey = None


SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-external-witness-"
    "admission-contract/2"
)
STATUS_V2: Final = "PRESENTED_ACYCLIC_EXTERNAL_WITNESS_ADMISSION_V2_CANDIDATE"
CLAIM_LEVEL_V2: Final = (
    "RAW_PRESENTED_SERIALIZATION_SIGNATURE_AND_REPLAY_INTEGRITY_ONLY_"
    "NOT_ACTUAL_RUNTIME_NOT_KEY_CUSTODY_NOT_DURABILITY_NOT_CURRENTNESS_"
    "NOT_NON_EQUIVOCATION_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
CANONICALIZATION_V2: Final = "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
ALGORITHM_V2: Final = "Ed25519"
REAL_ADMISSION_STATUS_V2: Final = "NOT_IMPLEMENTED"

INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256: Final = (
    "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285"
)
INSPECTED_LIVE_RUNTIME_OVERLAY_V2_SOURCE_SHA256: Final = (
    "5e4b23b79c04e4a613995ddec5aafe883e1aa694b945b86bda7ba67caff3be0e"
)
INSPECTED_AUTHORITY_V2_SOURCE_SHA256: Final = (
    "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
)
INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256: Final = (
    "f836b3d00f332398214d3ccf7787109abb8f2de96b7b8281748bbb0ec695f2f9"
)
INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256: Final = (
    "e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02"
)
DEPENDENCY_SOURCE_SHA256_IN_EXACT_ORDER_V2: Final = (
    (
        "external_witness_admission_contract_v1",
        INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256,
    ),
    ("live_runtime_overlay_v2", INSPECTED_LIVE_RUNTIME_OVERLAY_V2_SOURCE_SHA256),
    ("authority_v2", INSPECTED_AUTHORITY_V2_SOURCE_SHA256),
    ("runtime_closure_v1", INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256),
    ("store_replay_core_v2", INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256),
)

_AUTH_ROLE_ORDER_V2: Final = tuple(authority.ROLE_ORDER_V2)
_AUTH_DETACHED_SIGNATURE_FIELDS_V2: Final = frozenset(
    authority.DETACHED_SIGNATURE_FIELDS_V2
)
_AUTH_DETACHED_SIGNATURE_SCHEMA_V2: Final = authority.DETACHED_SIGNATURE_SCHEMA_V2
_ADMISSION_BOOTSTRAP_POLICY_FIELDS_V1: Final = frozenset(
    admission_v1.BOOTSTRAP_EXTERNAL_WITNESS_POLICY_FIELDS_V1
)
_ADMISSION_ONLINE_POLICY_FIELDS_V1: Final = frozenset(
    admission_v1.ONLINE_WITNESS_POLICY_FIELDS_V1
)
_ADMISSION_ONLINE_POLICY_BODY_FIELDS_V1: Final = frozenset(
    admission_v1.ONLINE_WITNESS_POLICY_BODY_FIELDS_V1
)
_ADMISSION_DURABLE_POLICY_FIELDS_V1: Final = frozenset(
    admission_v1.DURABLE_ANCHOR_POLICY_FIELDS_V1
)
_ADMISSION_DURABLE_POLICY_BODY_FIELDS_V1: Final = frozenset(
    admission_v1.DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1
)
_ADMISSION_RESULT_FIELDS_V1: Final = frozenset(
    admission_v1.ADMISSION_CANDIDATE_RESULT_FIELDS_V1
)
_ADMISSION_BOOTSTRAP_PURPOSES_V1: Final = tuple(
    admission_v1.BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1
)
_SIGN_ONLINE_POLICY_PURPOSE_V1: Final = (
    admission_v1.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1
)
_OVERLAY_ONLINE_PURPOSES_V2: Final = tuple(
    overlay_v2.LIVE_OVERLAY_ONLINE_PURPOSES_IN_EXACT_ORDER_V2
)
_OVERLAY_CUSTODY_PURPOSES_V2: Final = tuple(
    overlay_v2.CUSTODY_PURPOSES_IN_EXACT_ORDER_V2
)
_OVERLAY_ANCHOR_PURPOSES_V2: Final = tuple(
    overlay_v2.ANCHOR_PURPOSES_IN_EXACT_ORDER_V2
)

ADMISSION_V2_PRE_RESPONSE_CHALLENGE_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-pre-response-"
    "challenge/2"
)
ADMISSION_V2_LIVE_POLICY_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-live-policy/2"
)
ADMISSION_V2_WITNESS_RESPONSE_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-w-response/2"
)
ADMISSION_V2_WITNESS_REPORT_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-w-report/2"
)
ADMISSION_V2_CUSTODY_ATTESTATION_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-custody-"
    "attestation/2"
)
ADMISSION_V2_CANDIDATE_RESULT_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-candidate-result/2"
)
ADMISSION_V2_A2_MAPPING_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-a2-mapping/2"
)
ADMISSION_V2_A2_POLICY_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-a2-policy/2"
)
ADMISSION_V2_A2_REQUEST_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-a2-request/2"
)
ADMISSION_V2_A2_CHECKPOINT_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-a2-checkpoint/2"
)
ADMISSION_V2_A2_HEAD_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-a2-head/2"
)
ADMISSION_V2_A2_RECEIPT_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-a2-receipt/2"
)
ADMISSION_V2_CAS_COMPLETE_BINDING_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-witness-admission-cas-complete-"
    "binding/2"
)

SIGN_WITNESS_RESPONSE_PURPOSE_V2: Final = (
    overlay_v2.SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_PURPOSE_V2
)
ATTEST_WITNESS_CUSTODY_PURPOSE_V2: Final = (
    overlay_v2.ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2
)
SIGN_A2_POLICY_PURPOSE_V2: Final = (
    overlay_v2.SIGN_LIVE_RUNTIME_ANCHOR_POLICY_PURPOSE_V2
)
SIGN_A2_CHECKPOINT_PURPOSE_V2: Final = (
    overlay_v2.SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_PURPOSE_V2
)
SIGN_A2_HEAD_PURPOSE_V2: Final = (
    overlay_v2.SIGN_LIVE_RUNTIME_ANCHOR_HEAD_PURPOSE_V2
)
SIGN_A2_RECEIPT_PURPOSE_V2: Final = (
    overlay_v2.SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_PURPOSE_V2
)
SIGN_SOURCE_PURPOSE_V2: Final = overlay_v2.SIGN_RUNTIME_CLOSURE_PURPOSE_V2
SIGN_PURE_RESULT_PURPOSE_V2: Final = overlay_v2.SIGN_PURE_REPLAY_RESULT_PURPOSE_V2

CAS_PLAN_KIND_V2: Final = (
    "PHASE2B_STEP5_EXTERNAL_WITNESS_ADMISSION_V2_EXACT_SIX_OBJECTS"
)
CAS_OBJECTS_IN_EXACT_ORDER_V2: Final = (
    (0, "external_admission_v2_live_policy", 16_777_216),
    (1, "external_admission_v2_pre_response_challenge", 16_777_216),
    (2, "external_admission_v2_w_response", 16_777_216),
    (3, "external_admission_v2_w_report", 16_777_216),
    (4, "external_admission_v2_custody_attestation", 16_777_216),
    (5, "external_admission_v2_candidate_result", 16_777_216),
)
CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2: Final = tuple(
    row[1] for row in CAS_OBJECTS_IN_EXACT_ORDER_V2
)
PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2: Final = 9
PRIVATE_RESULT_VERIFIED_SIGNATURE_COUNT_V2: Final = 14
PRIVATE_RESULT_VERIFIED_SIGNATURE_SCOPE_V2: Final = (
    "CUSTODY_POLICY_LIVE_POLICY_SIGNATURES_PLUS_ADMISSION_V2_CHALLENGE_RESPONSE_REPORT_"
    "CUSTODY_RESULT_AND_A2_POLICY_REQUEST_CHECKPOINT_HEAD_RECEIPT_ONLY_"
    "EXCLUDES_AUTHORITY_GRAPH_B480_CAS_RECEIPTS_AND_OFFLINE_CLOSURE"
)
VERIFIED_PROTOCOL_OBJECT_COUNT_V2: Final = 12

FAIL_INPUT_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_INPUT"
FAIL_CANONICAL_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_CANONICAL"
FAIL_SCHEMA_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_SCHEMA"
FAIL_CONTENT_ID_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_CONTENT_ID"
FAIL_BINDING_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_BINDING"
FAIL_SIGNATURE_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_SIGNATURE"
FAIL_DEPENDENCY_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_DEPENDENCY"
FAIL_ACYCLIC_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_ACYCLIC"
FAIL_CLAIM_BOUNDARY_V2: Final = (
    "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_CLAIM_BOUNDARY"
)
FAIL_RESULT_V2: Final = "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_RESULT"
FAIL_NOT_IMPLEMENTED_V2: Final = (
    "FAIL_STEP5_R01_EXTERNAL_ADMISSION_V2_NOT_IMPLEMENTED"
)

_MAXIMUM_CANONICAL_BYTES: Final = 16 * 1024 * 1024
_SAFE_INTEGER: Final = (1 << 53) - 1
_HEX64_RE: Final = re.compile(r"[0-9a-f]{64}")
_HEX128_RE: Final = re.compile(r"[0-9a-f]{128}")
_LOWER_HEX_RE: Final = re.compile(r"[0-9a-f]+")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_STORE_SHA_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_DECIMAL_RE: Final = re.compile(r"0|[1-9][0-9]{0,63}")
_SHA256 = hashlib.sha256
_PACK = struct.pack


class Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(ValueError):
    """Stable fail-closed error for admission/2 presented mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(code, detail)


def _fields(text: str) -> frozenset[str]:
    return frozenset(text.split())


EXTERNAL_SIGNATURE_FIELDS_V2: Final = _fields(
    "schema_version algorithm signer_key_id key_epoch purpose signature_hex"
)
EXTERNAL_SIGNATURE_PREIMAGE_FIELDS_V2: Final = _fields(
    "schema_version artifact_kind content_id body signer_key_id key_epoch purpose"
)
POLICY_ALONE_NONCLAIMS_FIELDS_V2: Final = _fields(
    "actual_runtime_verified runtime_closure_verified live_channel_verified "
    "key_custody_verified online_key_nonexportable_verified distinct_live_uid_gid_verified "
    "external_cas_verified external_anchor_verified external_anti_rollback_verified "
    "external_currentness_verified durability_verified non_equivocation_verified "
    "global_anti_fork_verified malicious_root_resistance_verified authority_evidence "
    "r01_satisfied readiness_evidence execution_authorized"
)
PROCESS_SNAPSHOT_FIELDS_V2: Final = frozenset(overlay_v2.PROCESS_SNAPSHOT_FIELDS_V2)
OVERLAY_POLICY_ALONE_NONCLAIMS_FIELDS_V2: Final = frozenset(
    overlay_v2.POLICY_ALONE_NONCLAIMS_FIELDS_V2
)

_SESSION_LINEAGE_FIELDS: Final = _fields(
    "status claim_level stage_b_trust_lock_id authority_profile_id signer_registry_id "
    "attempt_context_id predecessor_b480_admission_candidate_result_id "
    "predecessor_b480_admission_candidate_result_payload_sha256 "
    "bootstrap_external_witness_policy_id online_witness_policy_id durable_anchor_policy_id "
    "runtime_closure_policy_id runtime_closure_evidence_id custody_attestation_policy_id "
    "admission_v2_live_policy_id admission_v2_a2_policy_id session_id session_sequence"
)

ADMISSION_V2_LIVE_POLICY_FIELDS: Final = _fields(
    "schema_version body admission_v2_live_policy_id signatures_in_exact_order"
)
ADMISSION_V2_PRE_RESPONSE_CHALLENGE_FIELDS: Final = _fields(
    "schema_version body pre_response_challenge_id signatures_in_exact_order"
)
ADMISSION_V2_WITNESS_RESPONSE_FIELDS: Final = _fields(
    "schema_version body witness_response_id signatures_in_exact_order"
)
ADMISSION_V2_WITNESS_REPORT_FIELDS: Final = _fields(
    "schema_version body witness_report_id signatures_in_exact_order"
)
ADMISSION_V2_CUSTODY_ATTESTATION_FIELDS: Final = _fields(
    "schema_version body custody_attestation_id signatures_in_exact_order"
)
ADMISSION_V2_CANDIDATE_RESULT_FIELDS: Final = _fields(
    "schema_version body admission_v2_candidate_result_id signatures_in_exact_order"
)

ADMISSION_V2_LIVE_POLICY_BODY_FIELDS: Final = (
    frozenset(overlay_v2.LIVE_RUNTIME_POLICY_BODY_FIELDS_V2)
    - _fields(
        "admission_candidate_result_id admission_candidate_result_payload_sha256"
    )
    | _fields(
        "predecessor_b480_admission_candidate_result_id "
        "predecessor_b480_admission_candidate_result_payload_sha256"
    )
)
ADMISSION_V2_A2_POLICY_FIELDS: Final = _fields(
    "schema_version body admission_v2_a2_policy_id signatures_in_exact_order"
)
ADMISSION_V2_A2_REQUEST_FIELDS: Final = _fields(
    "schema_version body admission_v2_a2_request_id signatures_in_exact_order"
)
ADMISSION_V2_A2_CHECKPOINT_FIELDS: Final = _fields(
    "schema_version body admission_v2_a2_checkpoint_id signatures_in_exact_order"
)
ADMISSION_V2_A2_HEAD_FIELDS: Final = _fields(
    "schema_version body admission_v2_a2_head_id signatures_in_exact_order"
)
ADMISSION_V2_A2_RECEIPT_FIELDS: Final = _fields(
    "schema_version body admission_v2_a2_receipt_id signatures_in_exact_order"
)

ADMISSION_V2_PRE_RESPONSE_CHALLENGE_BODY_FIELDS: Final = (
    _SESSION_LINEAGE_FIELDS
    | _fields(
        "schema_version challenge_nonce challenge_nonce_size_bytes challenger_role "
        "challenger_key_id challenger_key_epoch expected_responder_key_id "
        "expected_responder_key_epoch entry_process_snapshot "
        "entry_process_snapshot_payload_sha256 expected_previous_a2_mapping "
        "challenge_monotonic_nanoseconds_decimal maximum_round_trip_nanoseconds "
        "one_shot_required challenge_outcome"
    )
)
ADMISSION_V2_WITNESS_RESPONSE_BODY_FIELDS: Final = (
    _SESSION_LINEAGE_FIELDS
    | _fields(
        "schema_version pre_response_challenge_id pre_response_challenge_payload_sha256 "
        "challenge_nonce challenge_nonce_size_bytes responder_key_id responder_key_epoch "
        "response_nonce response_nonce_size_bytes entry_process_snapshot_id "
        "entry_process_snapshot_payload_sha256 response_outcome"
    )
)
ADMISSION_V2_WITNESS_REPORT_BODY_FIELDS: Final = (
    _SESSION_LINEAGE_FIELDS
    | _fields(
        "schema_version pre_response_challenge_id pre_response_challenge_payload_sha256 "
        "witness_response_id witness_response_payload_sha256 challenge_nonce response_nonce "
        "entry_process_snapshot entry_process_snapshot_payload_sha256 exit_process_snapshot "
        "exit_process_snapshot_payload_sha256 entry_exit_identity_fields_in_exact_order "
        "presented_entry_exit_identity_equal challenge_sent_monotonic_nanoseconds_decimal "
        "response_received_monotonic_nanoseconds_decimal round_trip_nanoseconds "
        "maximum_round_trip_nanoseconds presented_round_trip_within_policy "
        "presented_exact_one_challenge_packet_sent "
        "presented_exact_one_response_packet_received response_message_flags "
        "response_packet_size_bytes response_ancillary_item_count "
        "presented_message_truncation_absent presented_control_truncation_absent "
        "presented_unexpected_ancillary_absent presented_so_peercred_check_passed "
        "presented_pidfd_identity_check_passed presented_procfs_identity_check_passed "
        "presented_service_binding_check_passed "
        "presented_runtime_closure_policy_binding_check_passed "
        "presented_runtime_closure_evidence_binding_check_passed "
        "presented_runtime_executable_binding_check_passed "
        "presented_distinct_uid_gid_check_passed presented_live_channel_observed_claim "
        "report_outcome"
    )
)
ADMISSION_V2_CUSTODY_ATTESTATION_BODY_FIELDS: Final = (
    _SESSION_LINEAGE_FIELDS
    | _fields(
        "schema_version pre_response_challenge_id pre_response_challenge_payload_sha256 "
        "witness_response_id witness_response_payload_sha256 witness_report_id "
        "witness_report_payload_sha256 challenge_nonce response_nonce "
        "entry_process_snapshot_id entry_process_snapshot_payload_sha256 "
        "exit_process_snapshot_id exit_process_snapshot_payload_sha256 "
        "attestation_sequence attestor_service_instance_id attestor_control_domain_id "
        "attestor_key_id attestor_key_epoch attestation_purpose online_witness_key_id "
        "online_witness_key_epoch online_witness_public_key_hex opaque_key_handle_sha256 "
        "attestation_evidence_format attestation_evidence_hex "
        "certificate_chain_der_hex_in_exact_order certificate_chain_root_id "
        "endorsement_material_hex_in_exact_order revocation_material_hex_in_exact_order "
        "attested_challenge_nonce attested_report_id attested_report_payload_sha256 "
        "attested_online_key_public_key_sha256 attested_key_handle_sha256 "
        "presented_online_key_generated_in_attested_boundary_claim "
        "presented_online_key_nonexportable_claim "
        "presented_certificate_chain_check_passed presented_quote_check_passed "
        "presented_fresh_nonce_check_passed presented_report_binding_check_passed "
        "presented_online_key_binding_check_passed attestation_outcome"
    )
)
ADMISSION_V2_CANDIDATE_RESULT_BODY_FIELDS: Final = (
    _SESSION_LINEAGE_FIELDS
    | _fields(
        "schema_version pre_response_challenge_id pre_response_challenge_payload_sha256 "
        "witness_response_id witness_response_payload_sha256 witness_report_id "
        "witness_report_payload_sha256 custody_attestation_id "
        "custody_attestation_payload_sha256 verified_signature_count "
        "verified_signature_scope verified_protocol_object_count "
        "raw_authority_graph_verified frozen_b480_candidate_verified "
        "dependency_contract_constants_verified nineteen_expected_pins_verified "
        "key_sets_pairwise_disjoint no_same_session_forward_references_verified "
        "offline_static_image_graph_verified presented_response_report_chain_verified "
        "presented_custody_attestation_bytes_verified presented_only "
        "private_key_present filesystem_path_present socket_present "
        "decoded_mapping_input_present callback_present caller_nonce_present "
        "caller_clock_present actual_runtime_verified runtime_closure_verified "
        "live_channel_verified key_custody_verified online_key_nonexportable_verified "
        "distinct_live_uid_gid_verified external_cas_verified external_anchor_verified "
        "external_anti_rollback_verified external_currentness_verified durability_verified "
        "non_equivocation_verified global_anti_fork_verified "
        "malicious_root_resistance_verified authority_evidence r01_satisfied "
        "readiness_evidence execution_authorized first_failure_code"
    )
)

ADMISSION_V2_A2_MAPPING_FIELDS: Final = _fields(
    "schema_version attempt_context_id admission_v2_live_policy_id "
    "predecessor_b480_admission_candidate_result_id "
    "predecessor_b480_admission_candidate_result_payload_sha256 session_id "
    "session_sequence admission_v2_candidate_result_id "
    "admission_v2_candidate_result_payload_sha256 retention_plan_id "
    "retention_plan_payload_sha256 retention_session_id chain_id "
    "cas_complete_head_state_id cas_final_checkpoint_payload_sha256 "
    "cas_final_head_payload_sha256 cas_final_response_payload_sha256 "
    "anchor_sequence effective_state superseded revoked"
)
ADMISSION_V2_CAS_COMPLETE_BINDING_FIELDS: Final = _fields(
    "schema_version authority_profile_id signer_registry_id attempt_context_id "
    "session_id session_sequence retention_plan_id retention_plan_payload_sha256 "
    "retention_session_id chain_id plan_kind expected_object_count "
    "artifact_kinds_in_exact_order final_ordinal final_artifact_kind "
    "admission_v2_candidate_result_id admission_v2_candidate_result_payload_sha256 "
    "final_store_request_id final_store_request_payload_sha256 final_object_id "
    "final_object_sha256 final_object_size_bytes final_signed_receipt_payload_sha256 "
    "final_checkpoint_payload_sha256 final_head_state_id final_head_payload_sha256 "
    "final_response_payload_sha256 observed_object_count complete "
    "complete_head_required presented_durable_before_return_claim "
    "presented_readback_verified_claim"
)
ADMISSION_V2_A2_POLICY_BODY_FIELDS: Final = _fields(
    "schema_version status claim_level canonicalization policy_epoch "
    "stage_b_trust_lock_id stage_b_trust_lock_payload_sha256 authority_profile_id "
    "authority_profile_payload_sha256 signer_registry_id signer_registry_payload_sha256 "
    "attempt_context_id predecessor_b480_admission_candidate_result_id "
    "predecessor_b480_admission_candidate_result_payload_sha256 "
    "bootstrap_external_witness_policy_id bootstrap_external_witness_policy_payload_sha256 "
    "online_witness_policy_id online_witness_policy_payload_sha256 durable_anchor_policy_id "
    "durable_anchor_policy_payload_sha256 admission_v2_live_policy_id "
    "admission_v2_live_policy_payload_sha256 anchor_namespace anchor_control_domain_id "
    "anchor_service_binding_id anchor_service_uid anchor_service_gid "
    "anchor_service_user_namespace_inode anchor_key_id anchor_key_epoch anchor_public_key_hex "
    "algorithm authorized_signature_purposes_in_exact_order requester_role "
    "requester_control_domain requester_key_id requester_key_epoch requester_public_key_hex "
    "requester_signature_purpose genesis_checkpoint_id genesis_head_id genesis_mapping "
    "oob_high_water_checkpoint_id oob_high_water_checkpoint_payload_sha256 "
    "oob_high_water_checkpoint_sequence oob_high_water_head_id "
    "oob_high_water_head_payload_sha256 oob_high_water_head_sequence "
    "oob_high_water_mapping admission_slot_key_profile compare_and_append_semantics "
    "durable_before_return_required cas_complete_head_required policy_alone_nonclaims"
)
_A2_COMMON_FIELDS: Final = _SESSION_LINEAGE_FIELDS | _fields(
    "admission_v2_candidate_result_id admission_v2_candidate_result_payload_sha256"
)
ADMISSION_V2_A2_REQUEST_BODY_FIELDS: Final = _A2_COMMON_FIELDS | _fields(
    "schema_version admission_v2_live_policy_payload_sha256 pre_response_challenge_id "
    "pre_response_challenge_payload_sha256 witness_response_id "
    "witness_response_payload_sha256 witness_report_id witness_report_payload_sha256 "
    "custody_attestation_id custody_attestation_payload_sha256 cas_complete_binding "
    "expected_previous_a2_checkpoint_id expected_previous_a2_checkpoint_payload_sha256 "
    "expected_previous_a2_checkpoint_sequence expected_previous_a2_head_id "
    "expected_previous_a2_head_payload_sha256 expected_previous_a2_head_sequence "
    "expected_current_mapping proposed_current_mapping compare_and_append_mode"
)
ADMISSION_V2_A2_CHECKPOINT_BODY_FIELDS: Final = _A2_COMMON_FIELDS | _fields(
    "schema_version admission_v2_a2_request_id admission_v2_a2_request_payload_sha256 "
    "previous_mapping committed_mapping cas_complete_binding cas_outcome "
    "checkpoint_sequence previous_admission_v2_a2_checkpoint_id"
)
ADMISSION_V2_A2_HEAD_BODY_FIELDS: Final = _A2_COMMON_FIELDS | _fields(
    "schema_version admission_v2_a2_request_id admission_v2_a2_request_payload_sha256 "
    "admission_v2_a2_checkpoint_id admission_v2_a2_checkpoint_payload_sha256 "
    "committed_mapping cas_complete_binding cas_outcome head_sequence "
    "previous_admission_v2_a2_head_id"
)
ADMISSION_V2_A2_RECEIPT_BODY_FIELDS: Final = _A2_COMMON_FIELDS | _fields(
    "schema_version admission_v2_a2_request_id admission_v2_a2_request_payload_sha256 "
    "admission_v2_a2_checkpoint_id admission_v2_a2_checkpoint_payload_sha256 "
    "admission_v2_a2_head_id admission_v2_a2_head_payload_sha256 previous_mapping "
    "committed_mapping cas_complete_binding cas_outcome compare_matched append_applied "
    "oob_high_water_checkpoint_id oob_high_water_head_id "
    "presented_durable_before_return_claim"
)

CAS_FORBIDDEN_FIELDS_V2: Final = _fields(
    "retention_plan_id retention_plan_payload_sha256 retention_session_id chain_id "
    "cas_request_id cas_object_id cas_signed_receipt_payload_sha256 "
    "cas_checkpoint_payload_sha256 cas_head_state_id cas_head_payload_sha256 "
    "cas_response_payload_sha256 cas_complete_binding"
)
CURRENT_RESULT_FORBIDDEN_BEFORE_RESULT_FIELDS_V2: Final = _fields(
    "admission_v2_candidate_result_id admission_v2_candidate_result_payload_sha256"
)
A2_TRANSCRIPT_FORBIDDEN_IN_SIX_FIELDS_V2: Final = _fields(
    "admission_v2_a2_request_id admission_v2_a2_request_payload_sha256 "
    "admission_v2_a2_checkpoint_id admission_v2_a2_checkpoint_payload_sha256 "
    "admission_v2_a2_head_id admission_v2_a2_head_payload_sha256 "
    "admission_v2_a2_receipt_id admission_v2_a2_receipt_payload_sha256"
)

EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2: Final = tuple(
    overlay_v2.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2
)
EXPECTED_PIN_NAMES_V2: Final = frozenset(EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2)

_SCHEMA_BY_KIND: Final = MappingProxyType(
    {
        "admission_v2_live_policy": ADMISSION_V2_LIVE_POLICY_SCHEMA,
        "pre_response_challenge": ADMISSION_V2_PRE_RESPONSE_CHALLENGE_SCHEMA,
        "witness_response": ADMISSION_V2_WITNESS_RESPONSE_SCHEMA,
        "witness_report": ADMISSION_V2_WITNESS_REPORT_SCHEMA,
        "custody_attestation": ADMISSION_V2_CUSTODY_ATTESTATION_SCHEMA,
        "admission_v2_result": ADMISSION_V2_CANDIDATE_RESULT_SCHEMA,
        "admission_v2_a2_policy": ADMISSION_V2_A2_POLICY_SCHEMA,
        "admission_v2_a2_request": ADMISSION_V2_A2_REQUEST_SCHEMA,
        "admission_v2_a2_checkpoint": ADMISSION_V2_A2_CHECKPOINT_SCHEMA,
        "admission_v2_a2_head": ADMISSION_V2_A2_HEAD_SCHEMA,
        "admission_v2_a2_receipt": ADMISSION_V2_A2_RECEIPT_SCHEMA,
    }
)
_TOP_FIELDS_BY_KIND: Final = MappingProxyType(
    {
        "admission_v2_live_policy": ADMISSION_V2_LIVE_POLICY_FIELDS,
        "pre_response_challenge": ADMISSION_V2_PRE_RESPONSE_CHALLENGE_FIELDS,
        "witness_response": ADMISSION_V2_WITNESS_RESPONSE_FIELDS,
        "witness_report": ADMISSION_V2_WITNESS_REPORT_FIELDS,
        "custody_attestation": ADMISSION_V2_CUSTODY_ATTESTATION_FIELDS,
        "admission_v2_result": ADMISSION_V2_CANDIDATE_RESULT_FIELDS,
        "admission_v2_a2_policy": ADMISSION_V2_A2_POLICY_FIELDS,
        "admission_v2_a2_request": ADMISSION_V2_A2_REQUEST_FIELDS,
        "admission_v2_a2_checkpoint": ADMISSION_V2_A2_CHECKPOINT_FIELDS,
        "admission_v2_a2_head": ADMISSION_V2_A2_HEAD_FIELDS,
        "admission_v2_a2_receipt": ADMISSION_V2_A2_RECEIPT_FIELDS,
    }
)
_BODY_FIELDS_BY_KIND: Final = MappingProxyType(
    {
        "admission_v2_live_policy": ADMISSION_V2_LIVE_POLICY_BODY_FIELDS,
        "pre_response_challenge": ADMISSION_V2_PRE_RESPONSE_CHALLENGE_BODY_FIELDS,
        "witness_response": ADMISSION_V2_WITNESS_RESPONSE_BODY_FIELDS,
        "witness_report": ADMISSION_V2_WITNESS_REPORT_BODY_FIELDS,
        "custody_attestation": ADMISSION_V2_CUSTODY_ATTESTATION_BODY_FIELDS,
        "admission_v2_result": ADMISSION_V2_CANDIDATE_RESULT_BODY_FIELDS,
        "admission_v2_a2_policy": ADMISSION_V2_A2_POLICY_BODY_FIELDS,
        "admission_v2_a2_request": ADMISSION_V2_A2_REQUEST_BODY_FIELDS,
        "admission_v2_a2_checkpoint": ADMISSION_V2_A2_CHECKPOINT_BODY_FIELDS,
        "admission_v2_a2_head": ADMISSION_V2_A2_HEAD_BODY_FIELDS,
        "admission_v2_a2_receipt": ADMISSION_V2_A2_RECEIPT_BODY_FIELDS,
    }
)
_ID_FIELD_BY_KIND: Final = MappingProxyType(
    {
        "admission_v2_live_policy": "admission_v2_live_policy_id",
        "pre_response_challenge": "pre_response_challenge_id",
        "witness_response": "witness_response_id",
        "witness_report": "witness_report_id",
        "custody_attestation": "custody_attestation_id",
        "admission_v2_result": "admission_v2_candidate_result_id",
        "admission_v2_a2_policy": "admission_v2_a2_policy_id",
        "admission_v2_a2_request": "admission_v2_a2_request_id",
        "admission_v2_a2_checkpoint": "admission_v2_a2_checkpoint_id",
        "admission_v2_a2_head": "admission_v2_a2_head_id",
        "admission_v2_a2_receipt": "admission_v2_a2_receipt_id",
    }
)
_PREFIX_BY_KIND: Final = MappingProxyType(
    {
        kind: "phase2b_step5_external_admission_v2_" + kind + "_"
        for kind in _SCHEMA_BY_KIND
    }
)
_CONTENT_ID_DOMAIN_BY_KIND: Final = MappingProxyType(
    {
        kind: (
            "HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/"
            + kind.upper()
            + "/ID/V2\0"
        ).encode("ascii")
        for kind in _SCHEMA_BY_KIND
    }
)
_EXTERNAL_SIGNATURE_DOMAIN_BY_KIND: Final = MappingProxyType(
    {
        kind: (
            "HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/"
            + kind.upper()
            + "/EXTERNAL_SIGNATURE/V2\0"
        ).encode("ascii")
        for kind in (
            "admission_v2_live_policy",
            "witness_response",
            "custody_attestation",
            "admission_v2_a2_policy",
            "admission_v2_a2_checkpoint",
            "admission_v2_a2_head",
            "admission_v2_a2_receipt",
        )
    }
)
SESSION_ID_PREFIX_V2: Final = "phase2b_step5_external_admission_v2_session_"
SESSION_ID_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/SESSION/ID/V2\0"
)
CAS_CHAIN_ID_PREFIX_V2: Final = "phase2b_step5_external_admission_v2_cas_chain_"
CAS_CHAIN_ID_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/CAS_CHAIN/ID/V2\0"
)

ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2: Final = tuple(
    overlay_v2.ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2
)


def _canonical(value: object) -> bytes:
    try:
        return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_CANONICAL_V2, "value is outside exact RFC8785"
        ) from exc


def _restore_numbers(value: object) -> object:
    if type(value) is float and value.is_integer() and abs(value) <= _SAFE_INTEGER:
        return int(value)
    if type(value) is list:
        return [_restore_numbers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_numbers(item) for key, item in value.items()}
    return value


def _decode(payload: bytes, label: str) -> object:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAXIMUM_CANONICAL_BYTES:
        _fail(FAIL_INPUT_V2, f"{label} must be bounded exact bytes")
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_CANONICAL_V2, f"{label} is not exact RFC8785"
        ) from exc
    restored = _restore_numbers(decoded)
    if _canonical(restored) != payload:
        _fail(FAIL_CANONICAL_V2, f"{label} differs after integer restoration")
    return restored


def _object(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(FAIL_SCHEMA_V2, f"{label} exact field set differs")
    return value


def _exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return frozenset(left) == frozenset(right) and all(
            _exact_equal(left[key], right[key]) for key in left  # type: ignore[index]
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _exact_equal(a, b)
            for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return left == right


def _sha(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return _SHA256(payload).hexdigest()


def _store_sha(payload: bytes) -> str:
    return "sha256:" + _sha(payload)


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(FAIL_SCHEMA_V2, f"{label} must be one exact string")
    return value


def _identifier(value: object, label: str) -> str:
    text = _text(value, label)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} identifier syntax differs")
    return text


def _hex64(value: object, label: str) -> str:
    text = _text(value, label)
    if _HEX64_RE.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} must be 64 lowercase hex characters")
    return text


def _store_digest(value: object, label: str) -> str:
    text = _text(value, label)
    if _STORE_SHA_RE.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} store digest syntax differs")
    return text


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _SAFE_INTEGER:
        _fail(FAIL_SCHEMA_V2, f"{label} integer differs")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(FAIL_SCHEMA_V2, f"{label} must be one exact boolean")
    return value


def _sequence(value: object, label: str, *, nonempty: bool = False) -> tuple[object, ...]:
    if type(value) is not list or (nonempty and not value):
        _fail(FAIL_SCHEMA_V2, f"{label} must be one exact array")
    return tuple(value)


def _string_sequence(
    value: object, label: str, *, nonempty: bool = False
) -> tuple[str, ...]:
    items = _sequence(value, label, nonempty=nonempty)
    if any(type(item) is not str for item in items):
        _fail(FAIL_SCHEMA_V2, f"{label} contains a non-string")
    return items  # type: ignore[return-value]


def _content_identifier(value: object, prefix: str, label: str) -> str:
    text = _text(value, label)
    if not text.startswith(prefix) or _HEX64_RE.fullmatch(text[len(prefix):]) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} content-ID syntax differs")
    return text


def canonical_external_witness_admission_json_bytes_v2(value: object) -> bytes:
    return _canonical(value)


def phase2b_step5_external_witness_admission_content_id_v2(
    artifact_kind: str, body: Mapping[str, object]
) -> str:
    if artifact_kind not in _SCHEMA_BY_KIND or not isinstance(body, Mapping):
        _fail(FAIL_INPUT_V2, "admission/2 content-ID input differs")
    return _PREFIX_BY_KIND[artifact_kind] + _SHA256(
        _CONTENT_ID_DOMAIN_BY_KIND[artifact_kind] + _canonical(dict(body))
    ).hexdigest()


def phase2b_step5_external_witness_admission_signature_preimage_v2(
    *, artifact_kind: str, body: Mapping[str, object], content_id: str,
    signer_key_id: str, key_epoch: int, purpose: str,
) -> bytes:
    if artifact_kind not in _EXTERNAL_SIGNATURE_DOMAIN_BY_KIND:
        _fail(FAIL_INPUT_V2, "external signature artifact kind differs")
    if type(key_epoch) is not int or not 1 <= key_epoch <= _SAFE_INTEGER:
        _fail(FAIL_INPUT_V2, "external signature key epoch must be an exact integer")
    subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-external-witness-admission-"
            "external-signature-preimage/2"
        ),
        "artifact_kind": artifact_kind,
        "content_id": content_id,
        "body": dict(body),
        "signer_key_id": signer_key_id,
        "key_epoch": key_epoch,
        "purpose": purpose,
    }
    if frozenset(subject) != EXTERNAL_SIGNATURE_PREIMAGE_FIELDS_V2:
        _fail(FAIL_SCHEMA_V2, "external signature preimage field set differs")
    return _EXTERNAL_SIGNATURE_DOMAIN_BY_KIND[artifact_kind] + _canonical(subject)


def _load_root(
    payload: bytes, kind: str
) -> tuple[dict[str, object], dict[str, object]]:
    root = _object(_decode(payload, kind), _TOP_FIELDS_BY_KIND[kind], kind)
    body = _object(root["body"], _BODY_FIELDS_BY_KIND[kind], f"{kind} body")
    schema = _SCHEMA_BY_KIND[kind]
    if root["schema_version"] != schema or body["schema_version"] != schema:
        _fail(FAIL_SCHEMA_V2, f"{kind} schema differs")
    identifier_field = _ID_FIELD_BY_KIND[kind]
    expected = phase2b_step5_external_witness_admission_content_id_v2(kind, body)
    _content_identifier(root[identifier_field], _PREFIX_BY_KIND[kind], f"{kind} ID")
    if root[identifier_field] != expected:
        _fail(FAIL_CONTENT_ID_V2, f"{kind} content ID differs")
    if type(root["signatures_in_exact_order"]) is not list:
        _fail(FAIL_SCHEMA_V2, f"{kind} signatures must be one exact array")
    return root, body


def _verify_ed25519(
    public_key_hex: str, signature_hex: str, preimage: bytes, label: str
) -> None:
    _hex64(public_key_hex, f"{label} public key")
    if _HEX128_RE.fullmatch(signature_hex) is None:
        _fail(FAIL_SIGNATURE_V2, f"{label} signature encoding differs")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V2, "Ed25519 verification backend unavailable")
    try:
        _Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature_hex), preimage
        )
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_SIGNATURE_V2, f"{label} Ed25519 signature rejected"
        ) from exc


def _verify_external_signature(
    signature_value: object, *, kind: str, body: Mapping[str, object],
    content_id: str, key_id: str, key_epoch: int, public_key_hex: str,
    purpose: str, label: str,
) -> None:
    signature = _object(signature_value, EXTERNAL_SIGNATURE_FIELDS_V2, label)
    if (
        signature["schema_version"]
        != "hegel-machine-phase2b-step5-external-witness-admission-signature/2"
        or signature["algorithm"] != ALGORITHM_V2
        or signature["signer_key_id"] != key_id
        or _integer(signature["key_epoch"], f"{label} key epoch", minimum=1)
        != key_epoch
        or signature["purpose"] != purpose
    ):
        _fail(FAIL_SIGNATURE_V2, f"{label} metadata differs")
    _verify_ed25519(
        public_key_hex,
        _text(signature["signature_hex"], f"{label} signature"),
        phase2b_step5_external_witness_admission_signature_preimage_v2(
            artifact_kind=kind,
            body=body,
            content_id=content_id,
            signer_key_id=key_id,
            key_epoch=key_epoch,
            purpose=purpose,
        ),
        label,
    )


def _verify_authority_signature(
    signature_value: object, *, kind: str, body: Mapping[str, object],
    role_row: Mapping[str, object], purpose: str, label: str,
    _preimage=authority.phase2b_step5_signature_preimage_v2,
) -> None:
    signature = _object(
        signature_value,
        _AUTH_DETACHED_SIGNATURE_FIELDS_V2,
        label,
    )
    if (
        signature["schema_version"] != _AUTH_DETACHED_SIGNATURE_SCHEMA_V2
        or signature["algorithm"] != ALGORITHM_V2
        or signature["role"] != role_row["role"]
        or signature["purpose"] != purpose
        or _integer(signature["key_epoch"], f"{label} key epoch", minimum=1)
        != role_row["key_epoch"]
        or signature["signer_key_id"] != role_row["key_id"]
    ):
        _fail(FAIL_SIGNATURE_V2, f"{label} metadata differs")
    try:
        preimage = _preimage(
            artifact_kind=kind,
            body=dict(body),
            role=role_row["role"],
            purpose=purpose,
            signer_key_id=role_row["key_id"],
            key_epoch=role_row["key_epoch"],
        )
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2, f"{label} authority preimage rejected"
        ) from exc
    _verify_ed25519(
        _text(role_row["public_key_hex"], f"{label} public key"),
        _text(signature["signature_hex"], f"{label} signature"),
        preimage,
        label,
    )


def _verify_signature_order(
    root: Mapping[str, object], *, kind: str, body: Mapping[str, object],
    signers: tuple[tuple[str, Mapping[str, object] | None, str, str, int, str], ...],
) -> None:
    signatures = root["signatures_in_exact_order"]
    if type(signatures) is not list or len(signatures) != len(signers):
        _fail(FAIL_SIGNATURE_V2, f"{kind} signature order/cardinality differs")
    content_id = _text(root[_ID_FIELD_BY_KIND[kind]], f"{kind} ID")
    for index, (mode, role_row, purpose, key_id, key_epoch, public_key) in enumerate(signers):
        label = f"{kind} signature {index}"
        if mode == "authority":
            if role_row is None:
                _fail(FAIL_SIGNATURE_V2, f"{label} authority row absent")
            _verify_authority_signature(
                signatures[index], kind=kind, body=body,
                role_row=role_row, purpose=purpose, label=label,
            )
        elif mode == "external":
            _verify_external_signature(
                signatures[index], kind=kind, body=body, content_id=content_id,
                key_id=key_id, key_epoch=key_epoch, public_key_hex=public_key,
                purpose=purpose, label=label,
            )
        else:
            _fail(FAIL_SIGNATURE_V2, f"{label} signer mode differs")


def _all_false(value: object, label: str) -> dict[str, object]:
    body = _object(value, POLICY_ALONE_NONCLAIMS_FIELDS_V2, label)
    if any(item is not False for item in body.values()):
        _fail(FAIL_CLAIM_BOUNDARY_V2, f"{label} grants an operational claim")
    return body


def _require_raw_pin(value: bytes, expected: str, label: str) -> None:
    if type(value) is not bytes:
        _fail(FAIL_INPUT_V2, f"{label} pin must be exact bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeError as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_INPUT_V2, f"{label} pin is not ASCII"
        ) from exc
    if decoded != expected:
        _fail(FAIL_BINDING_V2, f"{label} out-of-band pin differs")


def _require_digest_pin(value: bytes, payload: bytes, label: str) -> None:
    if type(value) is not bytes or len(value) != 32:
        _fail(FAIL_INPUT_V2, f"{label} digest pin must be 32 exact bytes")
    if value != _SHA256(payload).digest():
        _fail(FAIL_BINDING_V2, f"{label} digest pin differs")


def _candidate_header(body: Mapping[str, object], schema: str, label: str) -> None:
    if (
        body["schema_version"] != schema
        or body["status"] != STATUS_V2
        or body["claim_level"] != CLAIM_LEVEL_V2
    ):
        _fail(FAIL_SCHEMA_V2, f"{label} header differs")


def _policy_header(body: Mapping[str, object], schema: str, label: str) -> None:
    _candidate_header(body, schema, label)
    if (
        body["canonicalization"] != CANONICALIZATION_V2
        or _integer(body["policy_epoch"], f"{label} policy epoch", minimum=1) != 1
    ):
        _fail(FAIL_SCHEMA_V2, f"{label} policy header differs")


def _recursive_keys(value: object) -> frozenset[str]:
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            found.update(item)
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
    return frozenset(found)


def _require_no_forbidden_fields(
    value: object, forbidden: frozenset[str], label: str
) -> None:
    overlap = _recursive_keys(value) & forbidden
    if overlap:
        _fail(FAIL_ACYCLIC_V2, f"{label} contains forbidden fields: {sorted(overlap)!r}")


def _capture_dependency_function_graph(module: ModuleType) -> dict[str, FunctionType]:
    """Freeze a dependency's local function graph before public verification."""

    namespace = dict(vars(module))
    originals = {
        name: value
        for name, value in namespace.items()
        if type(value) is FunctionType and value.__module__ == module.__name__
    }
    private_namespace = dict(namespace)
    provisional: dict[str, FunctionType] = {}
    provisional_by_id: dict[int, FunctionType] = {}
    for name, function in originals.items():
        clone = FunctionType(
            function.__code__, private_namespace, function.__name__,
            function.__defaults__, function.__closure__,
        )
        clone.__kwdefaults__ = function.__kwdefaults__
        provisional[name] = clone
        provisional_by_id[id(function)] = clone
    private_namespace.update(provisional)

    def captured_default(value: object) -> object:
        return provisional_by_id.get(id(value), value)

    captured: dict[str, FunctionType] = {}
    for name, function in originals.items():
        defaults = function.__defaults__
        clone = FunctionType(
            function.__code__, private_namespace, function.__name__,
            None if defaults is None else tuple(captured_default(item) for item in defaults),
            function.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if function.__kwdefaults__ is None
            else {
                key: captured_default(value)
                for key, value in function.__kwdefaults__.items()
            }
        )
        captured[name] = clone
    private_namespace.update(captured)
    return captured


_OVERLAY_FUNCTIONS = _capture_dependency_function_graph(overlay_v2)
_overlay_load_authority_context = _OVERLAY_FUNCTIONS["_load_authority_context"]
_overlay_load_custody_policy = _OVERLAY_FUNCTIONS["_load_custody_policy"]
_overlay_process_snapshot = _OVERLAY_FUNCTIONS["_process_snapshot"]
del _OVERLAY_FUNCTIONS


def _require_dependency_contract_constants() -> None:
    if (
        DEPENDENCY_SOURCE_SHA256_IN_EXACT_ORDER_V2
        != (
            (
                "external_witness_admission_contract_v1",
                INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256,
            ),
            (
                "live_runtime_overlay_v2",
                INSPECTED_LIVE_RUNTIME_OVERLAY_V2_SOURCE_SHA256,
            ),
            ("authority_v2", INSPECTED_AUTHORITY_V2_SOURCE_SHA256),
            ("runtime_closure_v1", INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256),
            ("store_replay_core_v2", INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256),
        )
        or tuple(authority.ROLE_ORDER_V2) != _AUTH_ROLE_ORDER_V2
        or frozenset(authority.DETACHED_SIGNATURE_FIELDS_V2)
        != _AUTH_DETACHED_SIGNATURE_FIELDS_V2
        or authority.DETACHED_SIGNATURE_SCHEMA_V2
        != _AUTH_DETACHED_SIGNATURE_SCHEMA_V2
        or tuple(admission_v1.BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1)
        != _ADMISSION_BOOTSTRAP_PURPOSES_V1
        or tuple(overlay_v2.LIVE_OVERLAY_ONLINE_PURPOSES_IN_EXACT_ORDER_V2)
        != _OVERLAY_ONLINE_PURPOSES_V2
        or tuple(overlay_v2.CUSTODY_PURPOSES_IN_EXACT_ORDER_V2)
        != _OVERLAY_CUSTODY_PURPOSES_V2
        or tuple(overlay_v2.ANCHOR_PURPOSES_IN_EXACT_ORDER_V2)
        != _OVERLAY_ANCHOR_PURPOSES_V2
        or
        overlay_v2.INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256
        != INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256
        or overlay_v2.INSPECTED_AUTHORITY_V2_SOURCE_SHA256
        != INSPECTED_AUTHORITY_V2_SOURCE_SHA256
        or overlay_v2.INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256
        != INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256
        or overlay_v2.INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256
        != INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256
        or tuple(overlay_v2.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2)
        != EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2
    ):
        _fail(FAIL_DEPENDENCY_V2, "dependency contract constants differ")


def _runtime_identity(value: object) -> dict[str, object]:
    body = _object(
        value, frozenset(overlay_v2.RUNTIME_IDENTITY_FIELDS_V2), "runtime identity"
    )
    for field in body:
        _text(body[field], f"runtime identity {field}")
    return body


def _overlay_nonclaims(value: object, label: str) -> dict[str, object]:
    body = _object(value, OVERLAY_POLICY_ALONE_NONCLAIMS_FIELDS_V2, label)
    if any(item is not False for item in body.values()):
        _fail(FAIL_CLAIM_BOUNDARY_V2, f"{label} grants an operational claim")
    return body


def _common_session(
    body: Mapping[str, object], *, context: Mapping[str, object], label: str
) -> None:
    expected = {
        "status": STATUS_V2,
        "claim_level": CLAIM_LEVEL_V2,
        "stage_b_trust_lock_id": context["lock_id"],
        "authority_profile_id": context["profile_id"],
        "signer_registry_id": context["registry_id"],
        "attempt_context_id": context["attempt_id"],
        "predecessor_b480_admission_candidate_result_id": context["b480_result_id"],
        "predecessor_b480_admission_candidate_result_payload_sha256": context["b480_result_sha"],
        "bootstrap_external_witness_policy_id": context["bootstrap_id"],
        "online_witness_policy_id": context["online_id"],
        "durable_anchor_policy_id": context["durable_id"],
        "runtime_closure_policy_id": context["closure_policy_id"],
        "runtime_closure_evidence_id": context["closure_evidence_id"],
        "custody_attestation_policy_id": context["custody_policy_id"],
        "admission_v2_live_policy_id": context["live_policy_id"],
        "admission_v2_a2_policy_id": context["a2_policy_id"],
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, f"{label} session lineage differs")


def _common_a2(
    body: Mapping[str, object], *, context: Mapping[str, object], label: str
) -> None:
    _common_session(body, context=context, label=label)
    if (
        body["admission_v2_candidate_result_id"] != context["result_id"]
        or body["admission_v2_candidate_result_payload_sha256"]
        != context["result_sha"]
    ):
        _fail(FAIL_BINDING_V2, f"{label} result lineage differs")


def _mapping(value: object, label: str) -> dict[str, object]:
    body = _object(value, ADMISSION_V2_A2_MAPPING_FIELDS, label)
    if body["schema_version"] != ADMISSION_V2_A2_MAPPING_SCHEMA:
        _fail(FAIL_SCHEMA_V2, f"{label} schema differs")
    _identifier(body["attempt_context_id"], f"{label} attempt context")
    _content_identifier(
        body["admission_v2_live_policy_id"],
        _PREFIX_BY_KIND["admission_v2_live_policy"],
        f"{label} live policy ID",
    )
    _content_identifier(
        body["predecessor_b480_admission_candidate_result_id"],
        "phase2b_step5_external_admission_candidate_result_v1_",
        f"{label} b480 result ID",
    )
    _hex64(
        body["predecessor_b480_admission_candidate_result_payload_sha256"],
        f"{label} b480 result SHA",
    )
    _identifier(body["session_id"], f"{label} session")
    _integer(body["session_sequence"], f"{label} session sequence")
    _content_identifier(
        body["admission_v2_candidate_result_id"],
        _PREFIX_BY_KIND["admission_v2_result"],
        f"{label} admission/2 result ID",
    )
    _hex64(
        body["admission_v2_candidate_result_payload_sha256"],
        f"{label} admission/2 result SHA",
    )
    _content_identifier(
        body["retention_plan_id"],
        "phase2b_step5_retention_plan_v1_",
        f"{label} retention plan ID",
    )
    _hex64(body["retention_plan_payload_sha256"], f"{label} plan SHA")
    _identifier(body["retention_session_id"], f"{label} retention session")
    _identifier(body["chain_id"], f"{label} CAS chain")
    _content_identifier(
        body["cas_complete_head_state_id"],
        "phase2b_step5_cas_head_state_v2_",
        f"{label} CAS head-state ID",
    )
    for field in (
        "cas_final_checkpoint_payload_sha256",
        "cas_final_head_payload_sha256",
        "cas_final_response_payload_sha256",
    ):
        _store_digest(body[field], f"{label} {field}")
    _integer(body["anchor_sequence"], f"{label} anchor sequence")
    if body["effective_state"] not in ("GENESIS", "COMMITTED"):
        _fail(FAIL_SCHEMA_V2, f"{label} effective state differs")
    _boolean(body["superseded"], f"{label} superseded")
    _boolean(body["revoked"], f"{label} revoked")
    return body


def _genesis_mapping(
    *, attempt_id: str, live_policy_id: str, b480_result_id: str,
    b480_result_sha: str,
) -> dict[str, object]:
    zeros = "0" * 64
    return {
        "schema_version": ADMISSION_V2_A2_MAPPING_SCHEMA,
        "attempt_context_id": attempt_id,
        "admission_v2_live_policy_id": live_policy_id,
        "predecessor_b480_admission_candidate_result_id": b480_result_id,
        "predecessor_b480_admission_candidate_result_payload_sha256": b480_result_sha,
        "session_id": "GENESIS",
        "session_sequence": 0,
        "admission_v2_candidate_result_id": (
            _PREFIX_BY_KIND["admission_v2_result"] + zeros
        ),
        "admission_v2_candidate_result_payload_sha256": zeros,
        "retention_plan_id": "phase2b_step5_retention_plan_v1_" + zeros,
        "retention_plan_payload_sha256": zeros,
        "retention_session_id": "GENESIS",
        "chain_id": "GENESIS",
        "cas_complete_head_state_id": "phase2b_step5_cas_head_state_v2_" + zeros,
        "cas_final_checkpoint_payload_sha256": "sha256:" + zeros,
        "cas_final_head_payload_sha256": "sha256:" + zeros,
        "cas_final_response_payload_sha256": "sha256:" + zeros,
        "anchor_sequence": 0,
        "effective_state": "GENESIS",
        "superseded": False,
        "revoked": False,
    }


def _require_policy_refs(
    body: Mapping[str, object], *, refs: Mapping[str, object], label: str
) -> None:
    fields = (
        "stage_b_trust_lock_id",
        "stage_b_trust_lock_payload_sha256",
        "authority_profile_id",
        "authority_profile_payload_sha256",
        "signer_registry_id",
        "signer_registry_payload_sha256",
        "attempt_context_id",
        "predecessor_b480_admission_candidate_result_id",
        "predecessor_b480_admission_candidate_result_payload_sha256",
        "bootstrap_external_witness_policy_id",
        "bootstrap_external_witness_policy_payload_sha256",
        "online_witness_policy_id",
        "online_witness_policy_payload_sha256",
        "durable_anchor_policy_id",
        "durable_anchor_policy_payload_sha256",
    )
    for field in fields:
        if field in body and not _exact_equal(body[field], refs[field]):
            _fail(FAIL_BINDING_V2, f"{label} {field} differs")


def _load_live_policy_v2(
    payload: bytes, *, refs: Mapping[str, object], bootstrap: Mapping[str, object],
    online: Mapping[str, object], durable: Mapping[str, object],
    auth: Mapping[str, object], closure_result: object,
    closure_policy_body: Mapping[str, object],
    custody_policy_root: Mapping[str, object], custody_policy_payload: bytes,
    oob_id: bytes, oob_digest: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "admission_v2_live_policy")
    _policy_header(body, ADMISSION_V2_LIVE_POLICY_SCHEMA, "admission/2 live policy")
    _require_policy_refs(body, refs=refs, label="admission/2 live policy")
    _require_raw_pin(oob_id, root["admission_v2_live_policy_id"], "live policy ID")
    _require_digest_pin(oob_digest, payload, "live policy")
    if (
        body["runtime_closure_policy_id"] != closure_result.closure_policy_id
        or body["runtime_closure_policy_payload_sha256"]
        != closure_result.closure_policy_payload_sha256
        or body["runtime_closure_evidence_id"] != closure_result.evidence_id
        or body["runtime_closure_evidence_payload_sha256"]
        != refs["runtime_closure_evidence_payload_sha256"]
        or body["runtime_closure_service_name"] != closure_result.service_name
        or not _exact_equal(
            _runtime_identity(body["runtime_identity"]),
            closure_policy_body["runtime_identity"],
        )
        or body["custody_attestation_policy_id"]
        != custody_policy_root["custody_attestation_policy_id"]
        or body["custody_attestation_policy_payload_sha256"]
        != _sha(custody_policy_payload)
    ):
        _fail(FAIL_BINDING_V2, "live policy closure/custody binding differs")

    roles = auth["roles"]
    services = auth["services"]
    if not isinstance(roles, Mapping) or not isinstance(services, Mapping):
        _fail(FAIL_BINDING_V2, "authority context mappings differ")
    source = roles["SOURCE_RUNTIME_CUSTODIAN"]
    pure = roles["PURE_REPLAYER"]
    cas = roles["CAS_STORE"]
    runner = services["runner"]
    pure_service = services["pure_replayer"]
    cas_service = services["cas_store"]
    expected = {
        "bootstrap_key_id": bootstrap["witness_key_id"],
        "bootstrap_key_epoch": bootstrap["witness_key_epoch"],
        "bootstrap_public_key_hex": bootstrap["public_key_hex"],
        "bootstrap_authorized_signature_purposes_in_exact_order": list(
            _ADMISSION_BOOTSTRAP_PURPOSES_V1
        ),
        "online_witness_key_id": online["online_key_id"],
        "online_witness_key_epoch": online["online_key_epoch"],
        "online_witness_public_key_hex": online["online_public_key_hex"],
        "online_witness_authorized_signature_purposes_in_exact_order": list(
            _OVERLAY_ONLINE_PURPOSES_V2
        ),
        "online_witness_service_binding_id": online["service_binding_id"],
        "online_witness_control_domain_id": online["witness_control_domain_id"],
        "online_witness_service_uid": online["service_uid"],
        "online_witness_service_gid": online["service_gid"],
        "online_witness_service_user_namespace_inode": online["service_user_namespace_inode"],
        "source_runtime_custodian_role": "SOURCE_RUNTIME_CUSTODIAN",
        "source_runtime_custodian_control_domain": source["control_domain"],
        "source_runtime_custodian_service_binding_id": runner["binding_id"],
        "source_runtime_custodian_service_uid": runner["uid"],
        "source_runtime_custodian_service_gid": runner["gid"],
        "source_runtime_custodian_key_id": source["key_id"],
        "source_runtime_custodian_key_epoch": source["key_epoch"],
        "source_runtime_custodian_public_key_hex": source["public_key_hex"],
        "source_runtime_custodian_authorized_signature_purposes_in_exact_order": list(
            source["authorized_signature_purposes"]
        ),
        "pure_replayer_role": "PURE_REPLAYER",
        "pure_replayer_control_domain": pure["control_domain"],
        "pure_replayer_service_binding_id": pure_service["binding_id"],
        "pure_replayer_service_uid": pure_service["uid"],
        "pure_replayer_service_gid": pure_service["gid"],
        "pure_replayer_key_id": pure["key_id"],
        "pure_replayer_key_epoch": pure["key_epoch"],
        "pure_replayer_public_key_hex": pure["public_key_hex"],
        "pure_replayer_authorized_signature_purposes_in_exact_order": list(
            pure["authorized_signature_purposes"]
        ),
        "cas_store_role": "CAS_STORE",
        "cas_store_control_domain": cas["control_domain"],
        "cas_store_service_binding_id": cas_service["binding_id"],
        "cas_store_service_uid": cas_service["uid"],
        "cas_store_service_gid": cas_service["gid"],
        "cas_store_key_id": cas["key_id"],
        "cas_store_key_epoch": cas["key_epoch"],
        "cas_store_public_key_hex": cas["public_key_hex"],
        "cas_store_authorized_signature_purposes_in_exact_order": list(
            cas["authorized_signature_purposes"]
        ),
        "custody_attestor_key_id": online["custody_attestor_key_id"],
        "custody_attestor_key_epoch": online["custody_attestor_key_epoch"],
        "custody_attestor_public_key_hex": online["custody_attestor_public_key_hex"],
        "custody_attestor_authorized_signature_purposes_in_exact_order": list(
            _OVERLAY_CUSTODY_PURPOSES_V2
        ),
        "independent_custody_attestor_required": True,
        "online_key_nonexportable_required": True,
        "live_runtime_anchor_key_id": durable["anchor_key_id"],
        "live_runtime_anchor_key_epoch": durable["anchor_key_epoch"],
        "live_runtime_anchor_public_key_hex": durable["anchor_public_key_hex"],
        "live_runtime_anchor_authorized_signature_purposes_in_exact_order": list(
            _OVERLAY_ANCHOR_PURPOSES_V2
        ),
        "live_runtime_anchor_control_domain_id": durable["anchor_control_domain_id"],
        "transport_address_family": "AF_UNIX",
        "transport_socket_type": "SOCK_SEQPACKET",
        "transport_peer_credential_option": "SO_PEERCRED",
        "transport_peer_credential_struct_format": "=iii",
        "transport_peer_credential_struct_size_bytes": 12,
        "exactly_one_challenge_packet_required": True,
        "exactly_one_response_packet_required": True,
        "scm_rights_allowed": False,
        "pidfd_required": True,
        "procfs_identity_required": True,
        "entry_exit_identity_equal_required": True,
        "challenge_nonce_size_bytes": 32,
        "response_nonce_size_bytes": 32,
        "caller_nonce_forbidden": True,
        "caller_sequence_forbidden": True,
        "one_active_session_per_attempt_and_online_key_required": True,
        "cas_retention_plan_kind": CAS_PLAN_KIND_V2,
        "cas_expected_object_count": 6,
        "cas_artifact_kinds_in_exact_order": list(
            CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2
        ),
        "cas_complete_head_required": True,
        "cas_durable_before_return_required": True,
        "cas_readback_required": True,
        "anchor_slot_key_profile": durable["admission_slot_key_profile"],
        "anchor_compare_and_append_semantics": durable["compare_and_append_semantics"],
        "anchor_durable_before_return_required": True,
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "live policy exact mechanics/key binding differs")
    for field in (
        "online_witness_expected_cgroup_sha256",
    ):
        _hex64(body[field], f"live policy {field}")
    for field in (
        "custody_attestation_profile",
        "live_runtime_anchor_control_domain_id",
        "cas_chain_id_profile",
        "anchor_slot_key_profile",
        "anchor_compare_and_append_semantics",
        "monotonic_clock",
    ):
        _identifier(body[field], f"live policy {field}")
    _integer(body["maximum_packet_bytes"], "maximum packet bytes", minimum=1)
    _integer(
        body["maximum_round_trip_nanoseconds"],
        "maximum round trip nanoseconds",
        minimum=1,
    )
    _overlay_nonclaims(body["policy_alone_nonclaims"], "live policy nonclaims")
    _verify_signature_order(
        root,
        kind="admission_v2_live_policy",
        body=body,
        signers=(
            (
                "external", None, _SIGN_ONLINE_POLICY_PURPOSE_V1,
                _text(bootstrap["witness_key_id"], "bootstrap key ID"),
                _integer(bootstrap["witness_key_epoch"], "bootstrap key epoch", minimum=1),
                _text(bootstrap["public_key_hex"], "bootstrap public key"),
            ),
            ("authority", source, SIGN_SOURCE_PURPOSE_V2, "", 0, ""),
        ),
    )
    return root, body


def _load_a2_policy(
    payload: bytes, *, refs: Mapping[str, object], live_policy_root: Mapping[str, object],
    live_policy_payload: bytes, live_policy: Mapping[str, object],
    durable: Mapping[str, object], auth: Mapping[str, object],
    oob_id: bytes, oob_digest: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "admission_v2_a2_policy")
    _policy_header(body, ADMISSION_V2_A2_POLICY_SCHEMA, "admission/2 A2 policy")
    _require_policy_refs(body, refs=refs, label="admission/2 A2 policy")
    _require_raw_pin(oob_id, root["admission_v2_a2_policy_id"], "A2 policy ID")
    _require_digest_pin(oob_digest, payload, "A2 policy")
    source = auth["roles"]["SOURCE_RUNTIME_CUSTODIAN"]
    expected = {
        "admission_v2_live_policy_id": live_policy_root["admission_v2_live_policy_id"],
        "admission_v2_live_policy_payload_sha256": _sha(live_policy_payload),
        "anchor_namespace": durable["anchor_namespace"],
        "anchor_control_domain_id": durable["anchor_control_domain_id"],
        "anchor_key_id": durable["anchor_key_id"],
        "anchor_key_epoch": durable["anchor_key_epoch"],
        "anchor_public_key_hex": durable["anchor_public_key_hex"],
        "algorithm": ALGORITHM_V2,
        "authorized_signature_purposes_in_exact_order": list(
            _OVERLAY_ANCHOR_PURPOSES_V2
        ),
        "requester_role": "SOURCE_RUNTIME_CUSTODIAN",
        "requester_control_domain": source["control_domain"],
        "requester_key_id": source["key_id"],
        "requester_key_epoch": source["key_epoch"],
        "requester_public_key_hex": source["public_key_hex"],
        "requester_signature_purpose": SIGN_SOURCE_PURPOSE_V2,
        "admission_slot_key_profile": durable["admission_slot_key_profile"],
        "compare_and_append_semantics": durable["compare_and_append_semantics"],
        "durable_before_return_required": True,
        "cas_complete_head_required": True,
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "A2 policy key/reference binding differs")
    if (
        live_policy["live_runtime_anchor_key_id"] != body["anchor_key_id"]
        or live_policy["live_runtime_anchor_key_epoch"] != body["anchor_key_epoch"]
        or live_policy["live_runtime_anchor_public_key_hex"] != body["anchor_public_key_hex"]
        or live_policy["live_runtime_anchor_control_domain_id"]
        != body["anchor_control_domain_id"]
    ):
        _fail(FAIL_BINDING_V2, "live policy and A2 policy anchor binding differs")
    for field in (
        "anchor_namespace",
        "anchor_control_domain_id",
        "anchor_service_binding_id",
        "admission_slot_key_profile",
        "compare_and_append_semantics",
    ):
        _identifier(body[field], f"A2 policy {field}")
    for field in (
        "anchor_service_uid",
        "anchor_service_gid",
        "anchor_service_user_namespace_inode",
    ):
        _integer(body[field], f"A2 policy {field}")
    for field in (
        "oob_high_water_checkpoint_payload_sha256",
        "oob_high_water_head_payload_sha256",
    ):
        _hex64(body[field], f"A2 policy {field}")
    _content_identifier(
        body["genesis_checkpoint_id"],
        _PREFIX_BY_KIND["admission_v2_a2_checkpoint"],
        "A2 genesis checkpoint ID",
    )
    _content_identifier(
        body["genesis_head_id"],
        _PREFIX_BY_KIND["admission_v2_a2_head"],
        "A2 genesis head ID",
    )
    _content_identifier(
        body["oob_high_water_checkpoint_id"],
        _PREFIX_BY_KIND["admission_v2_a2_checkpoint"],
        "A2 OOB high-water checkpoint ID",
    )
    _content_identifier(
        body["oob_high_water_head_id"],
        _PREFIX_BY_KIND["admission_v2_a2_head"],
        "A2 OOB high-water head ID",
    )
    checkpoint_sequence = _integer(
        body["oob_high_water_checkpoint_sequence"], "A2 checkpoint high-water"
    )
    head_sequence = _integer(body["oob_high_water_head_sequence"], "A2 head high-water")
    genesis = _genesis_mapping(
        attempt_id=refs["attempt_context_id"],
        live_policy_id=live_policy_root["admission_v2_live_policy_id"],
        b480_result_id=refs["predecessor_b480_admission_candidate_result_id"],
        b480_result_sha=refs[
            "predecessor_b480_admission_candidate_result_payload_sha256"
        ],
    )
    observed_genesis = _mapping(body["genesis_mapping"], "A2 genesis mapping")
    high_water = _mapping(body["oob_high_water_mapping"], "A2 high-water mapping")
    zero = "0" * 64
    if (
        not _exact_equal(observed_genesis, genesis)
        or body["genesis_checkpoint_id"]
        != _PREFIX_BY_KIND["admission_v2_a2_checkpoint"] + zero
        or body["genesis_head_id"] != _PREFIX_BY_KIND["admission_v2_a2_head"] + zero
        or checkpoint_sequence != head_sequence
        or high_water["anchor_sequence"] != head_sequence
        or high_water["session_sequence"] != head_sequence
        or high_water["attempt_context_id"] != refs["attempt_context_id"]
        or high_water["admission_v2_live_policy_id"]
        != live_policy_root["admission_v2_live_policy_id"]
        or high_water["predecessor_b480_admission_candidate_result_id"]
        != refs["predecessor_b480_admission_candidate_result_id"]
        or high_water["predecessor_b480_admission_candidate_result_payload_sha256"]
        != refs["predecessor_b480_admission_candidate_result_payload_sha256"]
    ):
        _fail(FAIL_BINDING_V2, "A2 genesis/high-water lineage differs")
    if head_sequence == 0:
        if (
            not _exact_equal(high_water, genesis)
            or body["oob_high_water_checkpoint_id"] != body["genesis_checkpoint_id"]
            or body["oob_high_water_head_id"] != body["genesis_head_id"]
            or body["oob_high_water_checkpoint_payload_sha256"] != zero
            or body["oob_high_water_head_payload_sha256"] != zero
        ):
            _fail(FAIL_BINDING_V2, "initial A2 high-water is not exact genesis")
    elif (
        high_water["session_id"] == "GENESIS"
        or high_water["effective_state"] != "COMMITTED"
        or high_water["superseded"] is not False
        or high_water["revoked"] is not False
    ):
        _fail(FAIL_BINDING_V2, "non-genesis A2 high-water is not committed")
    _all_false(body["policy_alone_nonclaims"], "A2 policy nonclaims")
    _verify_signature_order(
        root,
        kind="admission_v2_a2_policy",
        body=body,
        signers=((
            "external", None, SIGN_A2_POLICY_PURPOSE_V2,
            _text(body["anchor_key_id"], "A2 key ID"),
            _integer(body["anchor_key_epoch"], "A2 key epoch", minimum=1),
            _text(body["anchor_public_key_hex"], "A2 public key"),
        ),),
    )
    return root, body


def _session_id(
    *, attempt_id: str, live_policy_id: str, a2_policy_id: str,
    online_key_id: str, session_sequence: int, challenge_nonce: str,
) -> str:
    subject = {
        "attempt_context_id": attempt_id,
        "admission_v2_live_policy_id": live_policy_id,
        "admission_v2_a2_policy_id": a2_policy_id,
        "online_witness_key_id": online_key_id,
        "session_sequence": session_sequence,
        "challenge_nonce": challenge_nonce,
    }
    return SESSION_ID_PREFIX_V2 + _SHA256(
        SESSION_ID_DOMAIN_V2 + _canonical(subject)
    ).hexdigest()


def _process_snapshot(value: object, *, expected_phase: str) -> dict[str, object]:
    try:
        return _overlay_process_snapshot(value, expected_phase=expected_phase)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2,
            f"overlay process-snapshot verifier rejected {expected_phase}",
        ) from exc


def _load_challenge(
    payload: bytes, *, context: dict[str, object], live_policy: Mapping[str, object],
    a2_policy: Mapping[str, object], source: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "pre_response_challenge")
    _candidate_header(
        body, ADMISSION_V2_PRE_RESPONSE_CHALLENGE_SCHEMA, "pre-response challenge"
    )
    nonce = _hex64(body["challenge_nonce"], "challenge nonce")
    sequence = _integer(body["session_sequence"], "session sequence", minimum=1)
    if sequence != a2_policy["oob_high_water_head_sequence"] + 1:
        _fail(FAIL_BINDING_V2, "challenge does not advance A2 high-water")
    session = _session_id(
        attempt_id=context["attempt_id"],
        live_policy_id=context["live_policy_id"],
        a2_policy_id=context["a2_policy_id"],
        online_key_id=live_policy["online_witness_key_id"],
        session_sequence=sequence,
        challenge_nonce=nonce,
    )
    context = dict(context)
    context["session_id"] = session
    context["session_sequence"] = sequence
    _common_session(body, context=context, label="pre-response challenge")
    expected = {
        "challenge_nonce_size_bytes": live_policy["challenge_nonce_size_bytes"],
        "challenger_role": "SOURCE_RUNTIME_CUSTODIAN",
        "challenger_key_id": source["key_id"],
        "challenger_key_epoch": source["key_epoch"],
        "expected_responder_key_id": live_policy["online_witness_key_id"],
        "expected_responder_key_epoch": live_policy["online_witness_key_epoch"],
        "maximum_round_trip_nanoseconds": live_policy["maximum_round_trip_nanoseconds"],
        "one_shot_required": True,
        "challenge_outcome": "ISSUED_BEFORE_W_RESPONSE",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "pre-response challenge mechanics differ")
    timestamp = _text(
        body["challenge_monotonic_nanoseconds_decimal"], "challenge monotonic timestamp"
    )
    if _DECIMAL_RE.fullmatch(timestamp) is None:
        _fail(FAIL_SCHEMA_V2, "challenge monotonic timestamp syntax differs")
    entry = _process_snapshot(body["entry_process_snapshot"], expected_phase="ENTRY")
    if body["entry_process_snapshot_payload_sha256"] != _sha(_canonical(entry)):
        _fail(FAIL_BINDING_V2, "challenge entry snapshot digest differs")
    previous = _mapping(body["expected_previous_a2_mapping"], "challenge previous A2 mapping")
    if not _exact_equal(previous, a2_policy["oob_high_water_mapping"]):
        _fail(FAIL_BINDING_V2, "challenge previous A2 mapping differs")
    _verify_signature_order(
        root,
        kind="pre_response_challenge",
        body=body,
        signers=(("authority", source, SIGN_SOURCE_PURPOSE_V2, "", 0, ""),),
    )
    return root, body, context


def _load_response(
    payload: bytes, *, context: Mapping[str, object],
    challenge_root: Mapping[str, object], challenge: Mapping[str, object],
    live_policy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "witness_response")
    _candidate_header(body, ADMISSION_V2_WITNESS_RESPONSE_SCHEMA, "W response")
    _common_session(body, context=context, label="W response")
    entry = challenge["entry_process_snapshot"]
    expected = {
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "challenge_nonce": challenge["challenge_nonce"],
        "challenge_nonce_size_bytes": live_policy["challenge_nonce_size_bytes"],
        "responder_key_id": live_policy["online_witness_key_id"],
        "responder_key_epoch": live_policy["online_witness_key_epoch"],
        "response_nonce_size_bytes": live_policy["response_nonce_size_bytes"],
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": challenge[
            "entry_process_snapshot_payload_sha256"
        ],
        "response_outcome": "W_RESPONDED_TO_PRE_RESPONSE_CHALLENGE",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "W response challenge/key binding differs")
    _hex64(body["response_nonce"], "response nonce")
    if body["response_nonce"] == body["challenge_nonce"]:
        _fail(FAIL_BINDING_V2, "challenge and response nonces alias")
    _verify_signature_order(
        root,
        kind="witness_response",
        body=body,
        signers=((
            "external", None, SIGN_WITNESS_RESPONSE_PURPOSE_V2,
            _text(live_policy["online_witness_key_id"], "online key ID"),
            _integer(live_policy["online_witness_key_epoch"], "online key epoch", minimum=1),
            _text(live_policy["online_witness_public_key_hex"], "online public key"),
        ),),
    )
    return root, body


def _load_report(
    payload: bytes, *, context: Mapping[str, object],
    challenge_root: Mapping[str, object], challenge: Mapping[str, object],
    response_root: Mapping[str, object], response: Mapping[str, object],
    live_policy: Mapping[str, object], source: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "witness_report")
    _candidate_header(body, ADMISSION_V2_WITNESS_REPORT_SCHEMA, "W report")
    _common_session(body, context=context, label="W report")
    expected = {
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(_canonical(dict(response_root))),
        "challenge_nonce": challenge["challenge_nonce"],
        "response_nonce": response["response_nonce"],
        "entry_process_snapshot_payload_sha256": challenge[
            "entry_process_snapshot_payload_sha256"
        ],
        "entry_exit_identity_fields_in_exact_order": list(
            ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2
        ),
        "maximum_round_trip_nanoseconds": live_policy[
            "maximum_round_trip_nanoseconds"
        ],
        "presented_entry_exit_identity_equal": True,
        "presented_round_trip_within_policy": True,
        "presented_exact_one_challenge_packet_sent": True,
        "presented_exact_one_response_packet_received": True,
        "response_message_flags": 0,
        "response_ancillary_item_count": 0,
        "presented_message_truncation_absent": True,
        "presented_control_truncation_absent": True,
        "presented_unexpected_ancillary_absent": True,
        "presented_so_peercred_check_passed": True,
        "presented_pidfd_identity_check_passed": True,
        "presented_procfs_identity_check_passed": True,
        "presented_service_binding_check_passed": True,
        "presented_runtime_closure_policy_binding_check_passed": True,
        "presented_runtime_closure_evidence_binding_check_passed": True,
        "presented_runtime_executable_binding_check_passed": True,
        "presented_distinct_uid_gid_check_passed": True,
        "presented_live_channel_observed_claim": True,
        "report_outcome": "PRESENTED_ENTRY_EXIT_EQUAL_CLAIM_ONLY",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "W report presented mechanics differ")
    entry = _process_snapshot(body["entry_process_snapshot"], expected_phase="ENTRY")
    exit_snapshot = _process_snapshot(
        body["exit_process_snapshot"], expected_phase="EXIT"
    )
    if (
        not _exact_equal(entry, challenge["entry_process_snapshot"])
        or body["exit_process_snapshot_payload_sha256"]
        != _sha(_canonical(exit_snapshot))
        or response["entry_process_snapshot_id"] != entry["process_snapshot_id"]
    ):
        _fail(FAIL_BINDING_V2, "W report process-snapshot linkage differs")
    if not all(
        _exact_equal(entry[field], exit_snapshot[field])
        for field in ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2
    ):
        _fail(FAIL_BINDING_V2, "presented entry/exit identity bytes differ")
    if (
        entry["peer_uid"] != live_policy["online_witness_service_uid"]
        or entry["peer_gid"] != live_policy["online_witness_service_gid"]
        or entry["user_namespace_inode"]
        != live_policy["online_witness_service_user_namespace_inode"]
        or entry["cgroup_sha256"]
        != live_policy["online_witness_expected_cgroup_sha256"]
    ):
        _fail(FAIL_BINDING_V2, "presented peer identity differs from live policy")
    sent_text = _text(
        body["challenge_sent_monotonic_nanoseconds_decimal"],
        "report challenge-sent monotonic timestamp",
    )
    received_text = _text(
        body["response_received_monotonic_nanoseconds_decimal"],
        "report response-received monotonic timestamp",
    )
    if (
        _DECIMAL_RE.fullmatch(sent_text) is None
        or _DECIMAL_RE.fullmatch(received_text) is None
    ):
        _fail(FAIL_SCHEMA_V2, "report monotonic timestamp syntax differs")
    sent = int(sent_text)
    received = int(received_text)
    round_trip = _integer(body["round_trip_nanoseconds"], "report round trip")
    if (
        sent_text != challenge["challenge_monotonic_nanoseconds_decimal"]
        or received < sent
        or round_trip != received - sent
        or round_trip > live_policy["maximum_round_trip_nanoseconds"]
    ):
        _fail(FAIL_BINDING_V2, "report monotonic arithmetic differs")
    packet_size = _integer(
        body["response_packet_size_bytes"], "response packet size", minimum=1
    )
    if (
        packet_size != len(_canonical(dict(response_root)))
        or packet_size > live_policy["maximum_packet_bytes"]
    ):
        _fail(FAIL_BINDING_V2, "response packet size differs")
    _verify_signature_order(
        root,
        kind="witness_report",
        body=body,
        signers=(("authority", source, SIGN_SOURCE_PURPOSE_V2, "", 0, ""),),
    )
    return root, body, entry, exit_snapshot


def _load_attestation(
    payload: bytes, *, context: Mapping[str, object],
    challenge_root: Mapping[str, object], challenge: Mapping[str, object],
    response_root: Mapping[str, object], response: Mapping[str, object],
    report_root: Mapping[str, object], report: Mapping[str, object],
    entry: Mapping[str, object], exit_snapshot: Mapping[str, object],
    custody_policy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "custody_attestation")
    _candidate_header(
        body, ADMISSION_V2_CUSTODY_ATTESTATION_SCHEMA, "custody attestation"
    )
    _common_session(body, context=context, label="custody attestation")
    expected = {
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(_canonical(dict(response_root))),
        "witness_report_id": report_root["witness_report_id"],
        "witness_report_payload_sha256": _sha(_canonical(dict(report_root))),
        "challenge_nonce": challenge["challenge_nonce"],
        "response_nonce": response["response_nonce"],
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": report[
            "entry_process_snapshot_payload_sha256"
        ],
        "exit_process_snapshot_id": exit_snapshot["process_snapshot_id"],
        "exit_process_snapshot_payload_sha256": report[
            "exit_process_snapshot_payload_sha256"
        ],
        "attestor_key_id": custody_policy["custody_attestor_key_id"],
        "attestor_key_epoch": custody_policy["custody_attestor_key_epoch"],
        "attestation_purpose": ATTEST_WITNESS_CUSTODY_PURPOSE_V2,
        "online_witness_key_id": custody_policy["online_witness_key_id"],
        "online_witness_key_epoch": custody_policy["online_witness_key_epoch"],
        "online_witness_public_key_hex": custody_policy[
            "online_witness_public_key_hex"
        ],
        "attestation_evidence_format": custody_policy["attestation_evidence_format"],
        "attested_challenge_nonce": challenge["challenge_nonce"],
        "attested_report_id": report_root["witness_report_id"],
        "attested_report_payload_sha256": _sha(_canonical(dict(report_root))),
        "attested_online_key_public_key_sha256": _sha(
            bytes.fromhex(custody_policy["online_witness_public_key_hex"])
        ),
        "presented_online_key_generated_in_attested_boundary_claim": True,
        "presented_online_key_nonexportable_claim": True,
        "presented_certificate_chain_check_passed": True,
        "presented_quote_check_passed": True,
        "presented_fresh_nonce_check_passed": True,
        "presented_report_binding_check_passed": True,
        "presented_online_key_binding_check_passed": True,
        "attestation_outcome": "PRESENTED_CUSTODY_EVIDENCE_CLAIM_ONLY",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "custody attestation evidence binding differs")
    if (
        _integer(body["attestation_sequence"], "attestation sequence", minimum=1)
        != context["session_sequence"]
    ):
        _fail(FAIL_BINDING_V2, "attestation sequence differs")
    _identifier(body["attestor_service_instance_id"], "attestor service instance")
    _identifier(body["attestor_control_domain_id"], "attestor control domain")
    opaque = _hex64(body["opaque_key_handle_sha256"], "opaque key-handle SHA")
    if body["attested_key_handle_sha256"] != opaque:
        _fail(FAIL_BINDING_V2, "attested key-handle digest differs")
    evidence = _text(body["attestation_evidence_hex"], "attestation evidence")
    if (
        not evidence
        or len(evidence) % 2
        or _LOWER_HEX_RE.fullmatch(evidence) is None
        or len(evidence) // 2 > custody_policy["maximum_attestation_evidence_bytes"]
    ):
        _fail(FAIL_SCHEMA_V2, "attestation evidence encoding/size differs")
    certificate_chain = _string_sequence(
        body["certificate_chain_der_hex_in_exact_order"],
        "certificate chain",
        nonempty=True,
    )
    if any(
        not item or len(item) % 2 or _LOWER_HEX_RE.fullmatch(item) is None
        for item in certificate_chain
    ):
        _fail(FAIL_SCHEMA_V2, "certificate DER encoding differs")
    _identifier(body["certificate_chain_root_id"], "certificate root ID")
    if body["certificate_chain_root_id"] not in custody_policy[
        "trusted_certificate_root_ids_in_exact_order"
    ]:
        _fail(FAIL_BINDING_V2, "certificate root is not policy-pinned")
    for field, label in (
        ("endorsement_material_hex_in_exact_order", "endorsement material"),
        ("revocation_material_hex_in_exact_order", "revocation material"),
    ):
        materials = _string_sequence(body[field], label, nonempty=True)
        if any(
            not item or len(item) % 2 or _LOWER_HEX_RE.fullmatch(item) is None
            for item in materials
        ):
            _fail(FAIL_SCHEMA_V2, f"{label} encoding differs")
    _verify_signature_order(
        root,
        kind="custody_attestation",
        body=body,
        signers=((
            "external", None, ATTEST_WITNESS_CUSTODY_PURPOSE_V2,
            _text(custody_policy["custody_attestor_key_id"], "custody key ID"),
            _integer(custody_policy["custody_attestor_key_epoch"], "custody key epoch", minimum=1),
            _text(custody_policy["custody_attestor_public_key_hex"], "custody public key"),
        ),),
    )
    return root, body


def _contains_exact_scalar(value: object, needles: frozenset[object]) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
        elif item in needles:
            return True
    return False


def _load_result(
    payload: bytes, *, context: Mapping[str, object],
    challenge_root: Mapping[str, object], response_root: Mapping[str, object],
    report_root: Mapping[str, object], attestation_root: Mapping[str, object],
    source: Mapping[str, object], pure: Mapping[str, object],
    prefix_objects: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "admission_v2_result")
    _candidate_header(
        body, ADMISSION_V2_CANDIDATE_RESULT_SCHEMA, "admission/2 result"
    )
    _common_session(body, context=context, label="admission/2 result")
    expected_refs = {
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(_canonical(dict(response_root))),
        "witness_report_id": report_root["witness_report_id"],
        "witness_report_payload_sha256": _sha(_canonical(dict(report_root))),
        "custody_attestation_id": attestation_root["custody_attestation_id"],
        "custody_attestation_payload_sha256": _sha(
            _canonical(dict(attestation_root))
        ),
    }
    if any(not _exact_equal(body[field], value) for field, value in expected_refs.items()):
        _fail(FAIL_BINDING_V2, "admission/2 result prefix references differ")
    expected_claims: dict[str, object] = {
        "verified_signature_count": PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2,
        "verified_signature_scope": (
            "CUSTODY_POLICY_1_LIVE_POLICY_2_PLUS_CHALLENGE_RESPONSE_REPORT_"
            "CUSTODY_AND_RESULT_2_"
            "EXCLUDES_AUTHORITY_GRAPH_B480_CAS_AND_A2"
        ),
        "verified_protocol_object_count": 7,
        "raw_authority_graph_verified": True,
        "frozen_b480_candidate_verified": True,
        "dependency_contract_constants_verified": True,
        "nineteen_expected_pins_verified": True,
        "key_sets_pairwise_disjoint": True,
        "no_same_session_forward_references_verified": True,
        "offline_static_image_graph_verified": True,
        "presented_response_report_chain_verified": True,
        "presented_custody_attestation_bytes_verified": True,
        "presented_only": True,
        "private_key_present": False,
        "filesystem_path_present": False,
        "socket_present": False,
        "decoded_mapping_input_present": False,
        "callback_present": False,
        "caller_nonce_present": False,
        "caller_clock_present": False,
        "actual_runtime_verified": False,
        "runtime_closure_verified": False,
        "live_channel_verified": False,
        "key_custody_verified": False,
        "online_key_nonexportable_verified": False,
        "distinct_live_uid_gid_verified": False,
        "external_cas_verified": False,
        "external_anchor_verified": False,
        "external_anti_rollback_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "malicious_root_resistance_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
        "first_failure_code": "NONE",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected_claims.items()):
        _fail(FAIL_CLAIM_BOUNDARY_V2, "admission/2 result claim boundary differs")
    _require_no_forbidden_fields(
        body,
        CAS_FORBIDDEN_FIELDS_V2 | A2_TRANSCRIPT_FORBIDDEN_IN_SIX_FIELDS_V2,
        "admission/2 result",
    )
    current_id = root["admission_v2_candidate_result_id"]
    current_sha = _sha(payload)
    if any(
        _contains_exact_scalar(item, frozenset((current_id, current_sha)))
        for item in prefix_objects
    ):
        _fail(
            FAIL_ACYCLIC_V2,
            "an object before admission/2 result names the same-session result",
        )
    _verify_signature_order(
        root,
        kind="admission_v2_result",
        body=body,
        signers=(
            ("authority", source, SIGN_SOURCE_PURPOSE_V2, "", 0, ""),
            ("authority", pure, SIGN_PURE_RESULT_PURPOSE_V2, "", 0, ""),
        ),
    )
    return root, body


def _cas_chain_id(
    *, attempt_id: str, session_id: str, session_sequence: int,
    result_id: str, result_sha: str,
) -> str:
    subject = {
        "attempt_context_id": attempt_id,
        "session_id": session_id,
        "session_sequence": session_sequence,
        "admission_v2_candidate_result_id": result_id,
        "admission_v2_candidate_result_payload_sha256": result_sha,
    }
    return CAS_CHAIN_ID_PREFIX_V2 + _SHA256(
        CAS_CHAIN_ID_DOMAIN_V2 + _canonical(subject)
    ).hexdigest()


def _verify_cas_session(
    *, stage_b_trust_lock_payload: bytes, authority_profile_payload: bytes,
    signer_registry_payload: bytes, oob_lock_id: bytes, oob_lock_digest: bytes,
    retention_plan_payload: bytes, request_payloads: tuple[bytes, ...],
    object_payloads: tuple[bytes, ...], signed_receipt_payloads: tuple[bytes, ...],
    checkpoint_payloads: tuple[bytes, ...], head_payloads: tuple[bytes, ...],
    response_payloads: tuple[bytes, ...], context: Mapping[str, object],
    object_roots: tuple[Mapping[str, object], ...],
    expected_object_payloads: tuple[bytes, ...],
    _verify=store_replay.verify_cas_store_complete_session_replay_core_v2,
    _require=store_replay.require_verified_cas_store_complete_session_replay_core_v2,
) -> tuple[object, dict[str, object]]:
    sequences = (
        request_payloads,
        object_payloads,
        signed_receipt_payloads,
        checkpoint_payloads,
        head_payloads,
        response_payloads,
    )
    if any(type(sequence) is not tuple or len(sequence) != 6 for sequence in sequences):
        _fail(FAIL_INPUT_V2, "CAS replay requires six exact six-item tuples")
    if any(type(item) is not bytes for sequence in sequences for item in sequence):
        _fail(FAIL_INPUT_V2, "CAS replay tuple elements must be exact bytes")
    if object_payloads != expected_object_payloads:
        _fail(FAIL_BINDING_V2, "CAS object tuple is not the exact six admission objects")
    plan = _object(
        _decode(retention_plan_payload, "CAS retention plan"),
        frozenset(overlay_v2.CAS_RETENTION_PLAN_FIELDS_V2),
        "CAS retention plan",
    )
    expected_chain = _cas_chain_id(
        attempt_id=context["attempt_id"],
        session_id=context["session_id"],
        session_sequence=context["session_sequence"],
        result_id=object_roots[-1]["admission_v2_candidate_result_id"],
        result_sha=_sha(expected_object_payloads[-1]),
    )
    if (
        plan["schema_version"]
        != "hegel-machine-phase2b-formal-wire-covert-step5-retention-plan/1"
        or plan["authority_profile_id"] != context["profile_id"]
        or plan["signer_registry_id"] != context["registry_id"]
        or plan["attempt_context_id"] != context["attempt_id"]
        or plan["retention_session_id"] != context["session_id"]
        or plan["chain_id"] != expected_chain
        or plan["plan_kind"] != CAS_PLAN_KIND_V2
        or _integer(plan["expected_object_count"], "CAS expected object count") != 6
        or plan["complete_head_required"] is not True
    ):
        _fail(FAIL_BINDING_V2, "CAS retention-plan summary differs")
    rows = _sequence(plan["objects_in_exact_order"], "CAS planned objects")
    if len(rows) != 6:
        _fail(FAIL_BINDING_V2, "CAS planned object count differs")
    for observed, expected in zip(rows, CAS_OBJECTS_IN_EXACT_ORDER_V2, strict=True):
        row = _object(
            observed,
            frozenset(overlay_v2.CAS_PLAN_OBJECT_FIELDS_V2),
            "CAS planned object",
        )
        observed_row = (
            _integer(row["ordinal"], "CAS planned ordinal"),
            row["artifact_kind"],
            _integer(row["maximum_bytes"], "CAS planned maximum bytes", minimum=1),
        )
        if observed_row != expected:
            _fail(FAIL_BINDING_V2, "CAS planned object row differs")
    id_fields = (
        "admission_v2_live_policy_id",
        "pre_response_challenge_id",
        "witness_response_id",
        "witness_report_id",
        "custody_attestation_id",
        "admission_v2_candidate_result_id",
    )
    for index, (request_payload, root, object_payload) in enumerate(
        zip(request_payloads, object_roots, expected_object_payloads, strict=True)
    ):
        request = _object(
            _decode(request_payload, f"CAS request {index}"),
            frozenset(overlay_v2.CAS_GENERIC_REQUEST_FIELDS_V2),
            f"CAS request {index}",
        )
        if (
            request["authority_profile_id"] != context["profile_id"]
            or request["signer_registry_id"] != context["registry_id"]
            or request["attempt_context_id"] != context["attempt_id"]
            or request["retention_session_id"] != context["session_id"]
            or request["chain_id"] != plan["chain_id"]
            or request["plan_id"] != plan["plan_id"]
            or _integer(request["ordinal"], f"CAS request {index} ordinal") != index
            or request["artifact_kind"] != CAS_OBJECTS_IN_EXACT_ORDER_V2[index][1]
            or request["bound_artifact_id"] != root[id_fields[index]]
            or request["bound_artifact_sha256"] != _store_sha(object_payload)
            or _integer(
                request["reported_size_bytes"],
                f"CAS request {index} reported size",
            ) != len(object_payload)
            or _integer(request["fd_count"], f"CAS request {index} FD count") != 1
        ):
            _fail(FAIL_BINDING_V2, f"CAS request {index} object binding differs")
    try:
        result = _verify(
            stage_b_lock_payload=stage_b_trust_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=oob_lock_id,
            out_of_band_expected_lock_payload_sha256_digest=oob_lock_digest,
            retention_plan_payload=retention_plan_payload,
            request_payloads_in_exact_order=request_payloads,
            object_payloads_in_exact_order=object_payloads,
            signed_receipt_payloads_in_exact_order=signed_receipt_payloads,
            checkpoint_payloads_in_exact_order=checkpoint_payloads,
            head_payloads_in_exact_order=head_payloads,
            response_payloads_in_exact_order=response_payloads,
        )
        result = _require(result)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2, "store replay-core rejected six-object session"
        ) from exc
    if (
        result.attempt_context_id != context["attempt_id"]
        or result.retention_session_id != context["session_id"]
        or result.chain_id != plan["chain_id"]
        or result.plan_id != plan["plan_id"]
        or result.plan_kind != CAS_PLAN_KIND_V2
        or result.expected_object_count != 6
        or result.canonical_bytes_verified is not True
        or result.content_addresses_verified is not True
        or result.ed25519_signatures_verified is not True
        or result.append_chain_verified is not True
        or result.if_absent_idempotent_complete_semantics_verified is not True
        or result.durability_verified is not False
        or result.authority_evidence is not False
        or result.execution_authorized is not False
    ):
        _fail(FAIL_CLAIM_BOUNDARY_V2, "store replay-core semantics differ")
    final_request = _object(
        _decode(request_payloads[-1], "final CAS request"),
        frozenset(overlay_v2.CAS_GENERIC_REQUEST_FIELDS_V2),
        "final CAS request",
    )
    final_response = _decode(response_payloads[-1], "final CAS response")
    if type(final_response) is not dict:
        _fail(FAIL_SCHEMA_V2, "final CAS response root differs")
    final_binding = result.object_bindings_in_exact_order[-1]
    if type(final_binding) is not tuple or len(final_binding) != 10:
        _fail(FAIL_BINDING_V2, "final CAS replay binding shape differs")
    if final_binding[2] != object_roots[-1]["admission_v2_candidate_result_id"]:
        _fail(FAIL_BINDING_V2, "final CAS object ID is not admission/2 result")
    complete = {
        "schema_version": ADMISSION_V2_CAS_COMPLETE_BINDING_SCHEMA,
        "authority_profile_id": result.authority_profile_id,
        "signer_registry_id": result.signer_registry_id,
        "attempt_context_id": result.attempt_context_id,
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
        "retention_plan_id": result.plan_id,
        "retention_plan_payload_sha256": _sha(retention_plan_payload),
        "retention_session_id": result.retention_session_id,
        "chain_id": result.chain_id,
        "plan_kind": result.plan_kind,
        "expected_object_count": result.expected_object_count,
        "artifact_kinds_in_exact_order": list(CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2),
        "final_ordinal": 5,
        "final_artifact_kind": CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2[-1],
        "admission_v2_candidate_result_id": object_roots[-1][
            "admission_v2_candidate_result_id"
        ],
        "admission_v2_candidate_result_payload_sha256": _sha(
            expected_object_payloads[-1]
        ),
        "final_store_request_id": final_request["request_id"],
        "final_store_request_payload_sha256": _store_sha(request_payloads[-1]),
        "final_object_id": final_binding[4],
        "final_object_sha256": _store_sha(expected_object_payloads[-1]),
        "final_object_size_bytes": len(expected_object_payloads[-1]),
        "final_signed_receipt_payload_sha256": final_binding[6],
        "final_checkpoint_payload_sha256": result.complete_checkpoint_sha256,
        "final_head_state_id": result.complete_head_state_id,
        "final_head_payload_sha256": result.complete_head_sha256,
        "final_response_payload_sha256": result.complete_response_sha256,
        "observed_object_count": 6,
        "complete": True,
        "complete_head_required": True,
        "presented_durable_before_return_claim": True,
        "presented_readback_verified_claim": True,
    }
    _object(
        complete,
        ADMISSION_V2_CAS_COMPLETE_BINDING_FIELDS,
        "derived CAS complete binding",
    )
    if (
        final_response.get("durable_before_return") is not True
        or final_response.get("readback_verified") is not True
    ):
        _fail(FAIL_BINDING_V2, "final CAS response lacks presented claims")
    return result, complete


def _committed_mapping(
    *, context: Mapping[str, object], complete: Mapping[str, object],
    anchor_sequence: int,
) -> dict[str, object]:
    return {
        "schema_version": ADMISSION_V2_A2_MAPPING_SCHEMA,
        "attempt_context_id": context["attempt_id"],
        "admission_v2_live_policy_id": context["live_policy_id"],
        "predecessor_b480_admission_candidate_result_id": context["b480_result_id"],
        "predecessor_b480_admission_candidate_result_payload_sha256": context[
            "b480_result_sha"
        ],
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
        "admission_v2_candidate_result_id": context["result_id"],
        "admission_v2_candidate_result_payload_sha256": context["result_sha"],
        "retention_plan_id": complete["retention_plan_id"],
        "retention_plan_payload_sha256": complete["retention_plan_payload_sha256"],
        "retention_session_id": complete["retention_session_id"],
        "chain_id": complete["chain_id"],
        "cas_complete_head_state_id": complete["final_head_state_id"],
        "cas_final_checkpoint_payload_sha256": complete[
            "final_checkpoint_payload_sha256"
        ],
        "cas_final_head_payload_sha256": complete["final_head_payload_sha256"],
        "cas_final_response_payload_sha256": complete[
            "final_response_payload_sha256"
        ],
        "anchor_sequence": anchor_sequence,
        "effective_state": "COMMITTED",
        "superseded": False,
        "revoked": False,
    }


def _load_a2_transcript(
    *, request_payload: bytes, checkpoint_payload: bytes, head_payload: bytes,
    receipt_payload: bytes, context: Mapping[str, object],
    a2_policy: Mapping[str, object], source: Mapping[str, object],
    object_roots: tuple[Mapping[str, object], ...],
    object_payloads: tuple[bytes, ...], complete: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    request_root, request = _load_root(request_payload, "admission_v2_a2_request")
    _candidate_header(request, ADMISSION_V2_A2_REQUEST_SCHEMA, "A2 request")
    _common_a2(request, context=context, label="A2 request")
    expected_refs = {
        "admission_v2_live_policy_payload_sha256": _sha(object_payloads[0]),
        "pre_response_challenge_id": object_roots[1]["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(object_payloads[1]),
        "witness_response_id": object_roots[2]["witness_response_id"],
        "witness_response_payload_sha256": _sha(object_payloads[2]),
        "witness_report_id": object_roots[3]["witness_report_id"],
        "witness_report_payload_sha256": _sha(object_payloads[3]),
        "custody_attestation_id": object_roots[4]["custody_attestation_id"],
        "custody_attestation_payload_sha256": _sha(object_payloads[4]),
        "expected_previous_a2_checkpoint_id": a2_policy[
            "oob_high_water_checkpoint_id"
        ],
        "expected_previous_a2_checkpoint_payload_sha256": a2_policy[
            "oob_high_water_checkpoint_payload_sha256"
        ],
        "expected_previous_a2_checkpoint_sequence": a2_policy[
            "oob_high_water_checkpoint_sequence"
        ],
        "expected_previous_a2_head_id": a2_policy["oob_high_water_head_id"],
        "expected_previous_a2_head_payload_sha256": a2_policy[
            "oob_high_water_head_payload_sha256"
        ],
        "expected_previous_a2_head_sequence": a2_policy[
            "oob_high_water_head_sequence"
        ],
        "compare_and_append_mode": a2_policy["compare_and_append_semantics"],
    }
    if any(not _exact_equal(request[field], value) for field, value in expected_refs.items()):
        _fail(FAIL_BINDING_V2, "A2 request artifact/high-water references differ")
    observed_complete = _object(
        request["cas_complete_binding"],
        ADMISSION_V2_CAS_COMPLETE_BINDING_FIELDS,
        "A2 request CAS complete binding",
    )
    if not _exact_equal(observed_complete, complete):
        _fail(FAIL_BINDING_V2, "A2 request CAS complete binding differs")
    previous = _mapping(request["expected_current_mapping"], "A2 expected mapping")
    if not _exact_equal(previous, a2_policy["oob_high_water_mapping"]):
        _fail(FAIL_BINDING_V2, "A2 expected mapping differs from OOB high-water")
    next_sequence = a2_policy["oob_high_water_head_sequence"] + 1
    committed = _committed_mapping(
        context=context, complete=complete, anchor_sequence=next_sequence
    )
    if not _exact_equal(
        _mapping(request["proposed_current_mapping"], "A2 proposed mapping"),
        committed,
    ):
        _fail(FAIL_BINDING_V2, "A2 proposed mapping differs")
    _verify_signature_order(
        request_root,
        kind="admission_v2_a2_request",
        body=request,
        signers=(("authority", source, SIGN_SOURCE_PURPOSE_V2, "", 0, ""),),
    )

    checkpoint_root, checkpoint = _load_root(
        checkpoint_payload, "admission_v2_a2_checkpoint"
    )
    _candidate_header(
        checkpoint, ADMISSION_V2_A2_CHECKPOINT_SCHEMA, "A2 checkpoint"
    )
    _common_a2(checkpoint, context=context, label="A2 checkpoint")
    if (
        checkpoint["admission_v2_a2_request_id"]
        != request_root["admission_v2_a2_request_id"]
        or checkpoint["admission_v2_a2_request_payload_sha256"]
        != _sha(request_payload)
        or not _exact_equal(
            _mapping(checkpoint["previous_mapping"], "A2 checkpoint previous mapping"),
            previous,
        )
        or not _exact_equal(
            _mapping(checkpoint["committed_mapping"], "A2 checkpoint committed mapping"),
            committed,
        )
        or not _exact_equal(checkpoint["cas_complete_binding"], complete)
        or checkpoint["cas_outcome"] != "APPENDED_PRESENTED_ONLY"
        or _integer(
            checkpoint["checkpoint_sequence"], "A2 checkpoint sequence", minimum=1
        ) != next_sequence
        or checkpoint["previous_admission_v2_a2_checkpoint_id"]
        != a2_policy["oob_high_water_checkpoint_id"]
    ):
        _fail(FAIL_BINDING_V2, "A2 checkpoint transition differs")
    _verify_signature_order(
        checkpoint_root,
        kind="admission_v2_a2_checkpoint",
        body=checkpoint,
        signers=((
            "external", None, SIGN_A2_CHECKPOINT_PURPOSE_V2,
            a2_policy["anchor_key_id"], a2_policy["anchor_key_epoch"],
            a2_policy["anchor_public_key_hex"],
        ),),
    )

    head_root, head = _load_root(head_payload, "admission_v2_a2_head")
    _candidate_header(head, ADMISSION_V2_A2_HEAD_SCHEMA, "A2 head")
    _common_a2(head, context=context, label="A2 head")
    if (
        head["admission_v2_a2_request_id"]
        != request_root["admission_v2_a2_request_id"]
        or head["admission_v2_a2_request_payload_sha256"] != _sha(request_payload)
        or head["admission_v2_a2_checkpoint_id"]
        != checkpoint_root["admission_v2_a2_checkpoint_id"]
        or head["admission_v2_a2_checkpoint_payload_sha256"]
        != _sha(checkpoint_payload)
        or not _exact_equal(
            _mapping(head["committed_mapping"], "A2 head committed mapping"),
            committed,
        )
        or not _exact_equal(head["cas_complete_binding"], complete)
        or head["cas_outcome"] != "APPENDED_PRESENTED_ONLY"
        or _integer(head["head_sequence"], "A2 head sequence", minimum=1)
        != next_sequence
        or head["previous_admission_v2_a2_head_id"]
        != a2_policy["oob_high_water_head_id"]
    ):
        _fail(FAIL_BINDING_V2, "A2 head transition differs")
    _verify_signature_order(
        head_root,
        kind="admission_v2_a2_head",
        body=head,
        signers=((
            "external", None, SIGN_A2_HEAD_PURPOSE_V2,
            a2_policy["anchor_key_id"], a2_policy["anchor_key_epoch"],
            a2_policy["anchor_public_key_hex"],
        ),),
    )

    receipt_root, receipt = _load_root(receipt_payload, "admission_v2_a2_receipt")
    _candidate_header(receipt, ADMISSION_V2_A2_RECEIPT_SCHEMA, "A2 receipt")
    _common_a2(receipt, context=context, label="A2 receipt")
    if (
        receipt["admission_v2_a2_request_id"]
        != request_root["admission_v2_a2_request_id"]
        or receipt["admission_v2_a2_request_payload_sha256"]
        != _sha(request_payload)
        or receipt["admission_v2_a2_checkpoint_id"]
        != checkpoint_root["admission_v2_a2_checkpoint_id"]
        or receipt["admission_v2_a2_checkpoint_payload_sha256"]
        != _sha(checkpoint_payload)
        or receipt["admission_v2_a2_head_id"]
        != head_root["admission_v2_a2_head_id"]
        or receipt["admission_v2_a2_head_payload_sha256"] != _sha(head_payload)
        or not _exact_equal(
            _mapping(receipt["previous_mapping"], "A2 receipt previous mapping"),
            previous,
        )
        or not _exact_equal(
            _mapping(receipt["committed_mapping"], "A2 receipt committed mapping"),
            committed,
        )
        or not _exact_equal(receipt["cas_complete_binding"], complete)
        or receipt["cas_outcome"] != "APPENDED_PRESENTED_ONLY"
        or receipt["compare_matched"] is not True
        or receipt["append_applied"] is not True
        or receipt["oob_high_water_checkpoint_id"]
        != a2_policy["oob_high_water_checkpoint_id"]
        or receipt["oob_high_water_head_id"] != a2_policy["oob_high_water_head_id"]
        or receipt["presented_durable_before_return_claim"] is not True
    ):
        _fail(FAIL_BINDING_V2, "A2 receipt transition differs")
    _verify_signature_order(
        receipt_root,
        kind="admission_v2_a2_receipt",
        body=receipt,
        signers=((
            "external", None, SIGN_A2_RECEIPT_PURPOSE_V2,
            a2_policy["anchor_key_id"], a2_policy["anchor_key_epoch"],
            a2_policy["anchor_public_key_hex"],
        ),),
    )
    return request_root, checkpoint_root, head_root, receipt_root


@dataclass(frozen=True, slots=True, init=False)
class VerifiedExternalWitnessAdmissionCandidateV2:
    """Privately issued result for the exact post-A2 presented transcript."""

    schema_version: str
    verification_status: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    predecessor_b480_admission_candidate_result_id: str
    predecessor_b480_admission_candidate_result_payload_sha256: str
    custody_attestation_policy_id: str
    admission_v2_live_policy_id: str
    admission_v2_a2_policy_id: str
    session_id: str
    session_sequence: int
    admission_v2_candidate_result_id: str
    admission_v2_candidate_result_payload_sha256: str
    six_object_payload_sha256_in_exact_order: tuple[str, ...]
    retention_plan_id: str
    complete_cas_head_state_id: str
    cas_complete_binding_payload_sha256: str
    admission_v2_a2_request_id: str
    admission_v2_a2_request_payload_sha256: str
    admission_v2_a2_checkpoint_id: str
    admission_v2_a2_checkpoint_payload_sha256: str
    admission_v2_a2_head_id: str
    admission_v2_a2_head_payload_sha256: str
    admission_v2_a2_receipt_id: str
    admission_v2_a2_receipt_payload_sha256: str
    verified_signature_count: int
    verified_signature_scope: str
    verified_protocol_object_count: int
    dependency_contract_constants_verified: bool
    raw_authority_graph_verified: bool
    frozen_b480_candidate_verified: bool
    offline_static_image_graph_verified: bool
    nineteen_expected_pins_verified: bool
    key_sets_pairwise_disjoint: bool
    pre_response_challenge_order_verified: bool
    no_same_session_forward_references_verified: bool
    response_report_custody_chain_verified: bool
    admission_v2_result_pre_cas_verified: bool
    exact_six_object_cas_replay_verified: bool
    presented_a2_compare_and_append_verified: bool
    post_a2_private_binding_verified: bool
    presented_only: bool
    private_key_present: bool
    filesystem_path_present: bool
    socket_present: bool
    decoded_mapping_input_present: bool
    callback_present: bool
    caller_nonce_present: bool
    caller_clock_present: bool
    actual_runtime_verified: bool
    runtime_closure_verified: bool
    live_channel_verified: bool
    key_custody_verified: bool
    online_key_nonexportable_verified: bool
    distinct_live_uid_gid_verified: bool
    external_cas_verified: bool
    external_anchor_verified: bool
    external_anti_rollback_verified: bool
    external_currentness_verified: bool
    durability_verified: bool
    non_equivocation_verified: bool
    global_anti_fork_verified: bool
    malicious_root_resistance_verified: bool
    authority_evidence: bool
    r01_satisfied: bool
    readiness_evidence: bool
    execution_authorized: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "VerifiedExternalWitnessAdmissionCandidateV2":
        raise TypeError("verified admission/2 results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified admission/2 results are privately issued")

    def __copy__(self) -> "VerifiedExternalWitnessAdmissionCandidateV2":
        raise TypeError("verified admission/2 results cannot be copied")

    def __deepcopy__(
        self, memo: object
    ) -> "VerifiedExternalWitnessAdmissionCandidateV2":
        raise TypeError("verified admission/2 results cannot be deep-copied")


def _build_private_result_lineage_v2():
    result_type = VerifiedExternalWitnessAdmissionCandidateV2
    fields = tuple(result_type.__dataclass_fields__)
    issued: dict[
        int, tuple[VerifiedExternalWitnessAdmissionCandidateV2, tuple[object, ...]]
    ] = {}
    lock = threading.Lock()
    error_type = Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2
    fail_code = FAIL_RESULT_V2
    expected_status = (
        "PRESENTED_RAW_ACYCLIC_EXTERNAL_WITNESS_ADMISSION_V2_VERIFIED_MECHANICS_ONLY"
    )

    def fail_exact(detail: str) -> NoReturn:
        raise error_type(fail_code, detail)

    def exact_equal(left: object, right: object) -> bool:
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            return frozenset(left) == frozenset(right) and all(
                exact_equal(left[key], right[key]) for key in left  # type: ignore[index]
            )
        if type(left) in (list, tuple):
            return len(left) == len(right) and all(  # type: ignore[arg-type]
                exact_equal(a, b)
                for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
            )
        return left == right

    def snapshot(
        value: VerifiedExternalWitnessAdmissionCandidateV2,
    ) -> tuple[object, ...]:
        try:
            return tuple(object.__getattribute__(value, field) for field in fields)
        except (AttributeError, TypeError) as exc:
            raise error_type(
                fail_code, "verified admission/2 result fields are missing"
            ) from exc

    def require(value: object) -> VerifiedExternalWitnessAdmissionCandidateV2:
        if type(value) is not result_type:
            fail_exact("verified admission/2 result exact type differs")
        with lock:
            record = issued.get(id(value))
        if (
            record is None
            or record[0] is not value
            or not exact_equal(snapshot(value), record[1])
        ):
            fail_exact("verified admission/2 result lacks intact private lineage")
        mechanics_true = (
            "dependency_contract_constants_verified",
            "raw_authority_graph_verified",
            "frozen_b480_candidate_verified",
            "offline_static_image_graph_verified",
            "nineteen_expected_pins_verified",
            "key_sets_pairwise_disjoint",
            "pre_response_challenge_order_verified",
            "no_same_session_forward_references_verified",
            "response_report_custody_chain_verified",
            "admission_v2_result_pre_cas_verified",
            "exact_six_object_cas_replay_verified",
            "presented_a2_compare_and_append_verified",
            "post_a2_private_binding_verified",
            "presented_only",
        )
        absent_inputs = (
            "private_key_present",
            "filesystem_path_present",
            "socket_present",
            "decoded_mapping_input_present",
            "callback_present",
            "caller_nonce_present",
            "caller_clock_present",
        )
        elevated_false = (
            "actual_runtime_verified",
            "runtime_closure_verified",
            "live_channel_verified",
            "key_custody_verified",
            "online_key_nonexportable_verified",
            "distinct_live_uid_gid_verified",
            "external_cas_verified",
            "external_anchor_verified",
            "external_anti_rollback_verified",
            "external_currentness_verified",
            "durability_verified",
            "non_equivocation_verified",
            "global_anti_fork_verified",
            "malicious_root_resistance_verified",
            "authority_evidence",
            "r01_satisfied",
            "readiness_evidence",
            "execution_authorized",
        )
        if (
            value.schema_version != SCHEMA_VERSION_V2
            or value.verification_status != expected_status
            or value.verified_signature_count
            != PRIVATE_RESULT_VERIFIED_SIGNATURE_COUNT_V2
            or value.verified_signature_scope
            != PRIVATE_RESULT_VERIFIED_SIGNATURE_SCOPE_V2
            or value.verified_protocol_object_count
            != VERIFIED_PROTOCOL_OBJECT_COUNT_V2
            or type(value.six_object_payload_sha256_in_exact_order) is not tuple
            or len(value.six_object_payload_sha256_in_exact_order) != 6
            or any(
                type(item) is not str or _HEX64_RE.fullmatch(item) is None
                for item in value.six_object_payload_sha256_in_exact_order
            )
            or any(getattr(value, field) is not True for field in mechanics_true)
            or any(getattr(value, field) is not False for field in absent_inputs)
            or any(getattr(value, field) is not False for field in elevated_false)
        ):
            fail_exact("verified admission/2 result semantics differ")
        return value

    def issue(**values: object) -> VerifiedExternalWitnessAdmissionCandidateV2:
        if frozenset(values) != frozenset(fields):
            fail_exact("private admission/2 result issuance fields differ")
        value = object.__new__(result_type)
        for field in fields:
            object.__setattr__(value, field, values[field])
        record = (value, tuple(values[field] for field in fields))
        with lock:
            if id(value) in issued:
                fail_exact("private admission/2 result identity reused")
            issued[id(value)] = record
        return require(value)

    return issue, require


(_issue_result_v2, _require_result_v2) = _build_private_result_lineage_v2()
del _build_private_result_lineage_v2


def require_verified_external_witness_admission_candidate_v2(
    value: object,
) -> VerifiedExternalWitnessAdmissionCandidateV2:
    return _require_result_v2(value)


def _verify_external_witness_admission_candidate_implementation_v2(
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
    predecessor_b480_admission_candidate_result_payload: bytes,
    runtime_closure_policy_payload: bytes,
    runtime_closure_evidence_payload: bytes,
    raw_oci_index_payload: bytes,
    raw_oci_manifest_payload: bytes,
    raw_oci_config_payload: bytes,
    ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
    ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
    custody_attestation_policy_payload: bytes,
    admission_v2_live_policy_payload: bytes,
    admission_v2_a2_policy_payload: bytes,
    pre_response_challenge_payload: bytes,
    witness_response_payload: bytes,
    witness_report_payload: bytes,
    witness_custody_attestation_payload: bytes,
    admission_v2_candidate_result_payload: bytes,
    admission_v2_cas_retention_plan_payload: bytes,
    admission_v2_cas_store_request_payloads_in_exact_order: tuple[bytes, ...],
    admission_v2_cas_object_payloads_in_exact_order: tuple[bytes, ...],
    admission_v2_cas_signed_receipt_payloads_in_exact_order: tuple[bytes, ...],
    admission_v2_cas_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    admission_v2_cas_head_payloads_in_exact_order: tuple[bytes, ...],
    admission_v2_cas_response_payloads_in_exact_order: tuple[bytes, ...],
    admission_v2_a2_request_payload: bytes,
    admission_v2_a2_checkpoint_payload: bytes,
    admission_v2_a2_head_payload: bytes,
    admission_v2_a2_receipt_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
    out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_online_witness_policy_id_bytes: bytes,
    out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
    out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
    out_of_band_expected_runtime_closure_policy_id_bytes: bytes,
    out_of_band_expected_runtime_closure_policy_payload_sha256_digest: bytes,
    out_of_band_expected_runtime_closure_evidence_id_bytes: bytes,
    out_of_band_expected_runtime_closure_evidence_payload_sha256_digest: bytes,
    out_of_band_expected_custody_attestation_policy_id_bytes: bytes,
    out_of_band_expected_custody_attestation_policy_payload_sha256_digest: bytes,
    out_of_band_expected_live_runtime_policy_id_bytes: bytes,
    out_of_band_expected_live_runtime_policy_payload_sha256_digest: bytes,
    out_of_band_expected_live_runtime_anchor_policy_id_bytes: bytes,
    out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest: bytes,
    expected_attempt_context_id_bytes: bytes,
    _verify_b480=admission_v1.verify_external_witness_admission_transcript_candidate_v1,
    _require_b480=admission_v1.require_verified_external_witness_admission_candidate_v1,
    _verify_closure=closure.verify_offline_step5_image_closure_mechanics_v1,
    _require_closure=closure.require_verified_offline_step5_image_closure_mechanics_v1,
    _issue_result=_issue_result_v2,
) -> VerifiedExternalWitnessAdmissionCandidateV2:
    """Verify the frozen admission/2 presented transcript from raw bytes only."""

    scalar_payloads = (
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
        predecessor_b480_admission_candidate_result_payload,
        runtime_closure_policy_payload,
        runtime_closure_evidence_payload,
        raw_oci_index_payload,
        raw_oci_manifest_payload,
        raw_oci_config_payload,
        custody_attestation_policy_payload,
        admission_v2_live_policy_payload,
        admission_v2_a2_policy_payload,
        pre_response_challenge_payload,
        witness_response_payload,
        witness_report_payload,
        witness_custody_attestation_payload,
        admission_v2_candidate_result_payload,
        admission_v2_cas_retention_plan_payload,
        admission_v2_a2_request_payload,
        admission_v2_a2_checkpoint_payload,
        admission_v2_a2_head_payload,
        admission_v2_a2_receipt_payload,
    )
    if any(type(payload) is not bytes for payload in scalar_payloads):
        _fail(FAIL_INPUT_V2, "all scalar transports must be exact bytes")
    for chunks, label in (
        (ordered_raw_layer_chunk_tuples, "ordered raw OCI layers"),
        (ordered_support_artifact_chunk_tuples, "ordered support artifacts"),
    ):
        if type(chunks) is not tuple or not chunks:
            _fail(FAIL_INPUT_V2, f"{label} must be a nonempty exact tuple")
        if any(
            type(row) is not tuple
            or not row
            or any(type(chunk) is not bytes for chunk in row)
            for row in chunks
        ):
            _fail(FAIL_INPUT_V2, f"{label} chunk matrix differs")
    _require_dependency_contract_constants()

    try:
        predecessor = _verify_b480(
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
            predecessor_b480_admission_candidate_result_payload,
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
        predecessor = _require_b480(predecessor)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2, "frozen b480 admission replay rejected"
        ) from exc

    try:
        auth = _overlay_load_authority_context(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2, "raw authority graph rejected"
        ) from exc
    bootstrap = _object(
        _decode(bootstrap_external_witness_policy_payload, "bootstrap policy"),
        _ADMISSION_BOOTSTRAP_POLICY_FIELDS_V1,
        "bootstrap policy",
    )
    online_root = _object(
        _decode(online_witness_policy_payload, "online policy"),
        _ADMISSION_ONLINE_POLICY_FIELDS_V1,
        "online policy",
    )
    online = _object(
        online_root["body"],
        _ADMISSION_ONLINE_POLICY_BODY_FIELDS_V1,
        "online policy body",
    )
    durable_root = _object(
        _decode(durable_anchor_policy_payload, "durable policy"),
        _ADMISSION_DURABLE_POLICY_FIELDS_V1,
        "durable policy",
    )
    durable = _object(
        durable_root["body"],
        _ADMISSION_DURABLE_POLICY_BODY_FIELDS_V1,
        "durable policy body",
    )
    predecessor_root = _object(
        _decode(
            predecessor_b480_admission_candidate_result_payload,
            "b480 admission candidate result",
        ),
        _ADMISSION_RESULT_FIELDS_V1,
        "b480 admission candidate result",
    )
    if (
        predecessor_root["candidate_result_id"]
        != predecessor.admission_candidate_result_id
    ):
        _fail(FAIL_BINDING_V2, "b480 private/raw result ID differs")
    if type(expected_attempt_context_id_bytes) is not bytes:
        _fail(FAIL_INPUT_V2, "attempt-context pin must be exact bytes")
    try:
        attempt_id = expected_attempt_context_id_bytes.decode("ascii")
    except UnicodeError as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_INPUT_V2, "attempt-context pin is not ASCII"
        ) from exc
    _identifier(attempt_id, "attempt context")
    if attempt_id != predecessor.attempt_context_id:
        _fail(FAIL_BINDING_V2, "attempt-context pin differs from b480 result")

    closure_policy_root = _decode(runtime_closure_policy_payload, "closure policy")
    closure_evidence_root = _decode(runtime_closure_evidence_payload, "closure evidence")
    if (
        type(closure_policy_root) is not dict
        or frozenset(closure_policy_root) != frozenset(("body", "id"))
        or type(closure_evidence_root) is not dict
        or frozenset(closure_evidence_root) != frozenset(("body", "id"))
    ):
        _fail(FAIL_SCHEMA_V2, "runtime-closure root fields differ")
    _require_raw_pin(
        out_of_band_expected_runtime_closure_policy_id_bytes,
        closure_policy_root["id"],
        "runtime closure policy ID",
    )
    _require_digest_pin(
        out_of_band_expected_runtime_closure_policy_payload_sha256_digest,
        runtime_closure_policy_payload,
        "runtime closure policy",
    )
    _require_raw_pin(
        out_of_band_expected_runtime_closure_evidence_id_bytes,
        closure_evidence_root["id"],
        "runtime closure evidence ID",
    )
    _require_digest_pin(
        out_of_band_expected_runtime_closure_evidence_payload_sha256_digest,
        runtime_closure_evidence_payload,
        "runtime closure evidence",
    )
    closure_policy_body = closure_policy_root["body"]
    if type(closure_policy_body) is not dict:
        _fail(FAIL_SCHEMA_V2, "runtime closure policy body differs")
    _pre_live_root, pre_live_body = _load_root(
        admission_v2_live_policy_payload, "admission_v2_live_policy"
    )
    service_name = pre_live_body["runtime_closure_service_name"]
    if service_name != "runner":
        _fail(FAIL_BINDING_V2, "live policy closure service must be runner")
    try:
        closure_result = _verify_closure(
            stage_b_trust_lock_payload=stage_b_trust_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
            service_name=service_name,
            runtime_closure_policy_payload=runtime_closure_policy_payload,
            image_closure_evidence_payload=runtime_closure_evidence_payload,
            raw_oci_index_payload=raw_oci_index_payload,
            raw_oci_manifest_payload=raw_oci_manifest_payload,
            raw_oci_config_payload=raw_oci_config_payload,
            ordered_raw_layer_chunk_tuples=ordered_raw_layer_chunk_tuples,
            ordered_support_artifact_chunk_tuples=ordered_support_artifact_chunk_tuples,
        )
        closure_result = _require_closure(closure_result)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2, "offline image-closure replay rejected"
        ) from exc
    if (
        closure_result.stage_b_trust_lock_id != auth["lock"].artifact_id
        or closure_result.authority_profile_id != auth["profile"].artifact_id
        or closure_result.signer_registry_id != auth["registry"].artifact_id
        or closure_result.service_binding_id
        != auth["services"]["runner"]["binding_id"]
        or closure_result.offline_static_byte_graph_mechanics_verified is not True
        or closure_result.mechanics_only is not True
        or closure_result.service_image_exists is not False
        or closure_result.service_image_pulled is not False
        or closure_result.service_image_executed is not False
        or closure_result.live_runtime_closure_observed is not False
        or closure_result.entry_exit_closure_equal is not False
        or closure_result.live_uid_gid_verified is not False
        or closure_result.live_control_domain_verified is not False
        or closure_result.key_custody_verified is not False
        or closure_result.runtime_closure_signature_verified is not False
        or closure_result.remote_attestation_verified is not False
        or closure_result.external_cas_verified is not False
        or closure_result.authority_evidence is not False
        or closure_result.readiness_evidence is not False
        or closure_result.r01_satisfied is not False
        or closure_result.execution_authorized is not False
    ):
        _fail(FAIL_CLAIM_BOUNDARY_V2, "offline closure claim boundary differs")

    refs: dict[str, object] = {
        "stage_b_trust_lock_id": auth["lock"].artifact_id,
        "stage_b_trust_lock_payload_sha256": _sha(stage_b_trust_lock_payload),
        "authority_profile_id": auth["profile"].artifact_id,
        "authority_profile_payload_sha256": _sha(authority_profile_payload),
        "signer_registry_id": auth["registry"].artifact_id,
        "signer_registry_payload_sha256": _sha(signer_registry_payload),
        "attempt_context_id": attempt_id,
        "admission_candidate_result_id": predecessor.admission_candidate_result_id,
        "admission_candidate_result_payload_sha256": _sha(
            predecessor_b480_admission_candidate_result_payload
        ),
        "predecessor_b480_admission_candidate_result_id": (
            predecessor.admission_candidate_result_id
        ),
        "predecessor_b480_admission_candidate_result_payload_sha256": _sha(
            predecessor_b480_admission_candidate_result_payload
        ),
        "bootstrap_external_witness_policy_id": bootstrap[
            "external_witness_policy_id"
        ],
        "bootstrap_external_witness_policy_payload_sha256": _sha(
            bootstrap_external_witness_policy_payload
        ),
        "online_witness_policy_id": online_root["online_witness_policy_id"],
        "online_witness_policy_payload_sha256": _sha(online_witness_policy_payload),
        "durable_anchor_policy_id": durable_root["durable_anchor_policy_id"],
        "durable_anchor_policy_payload_sha256": _sha(durable_anchor_policy_payload),
        "runtime_closure_policy_id": closure_policy_root["id"],
        "runtime_closure_policy_payload_sha256": _sha(runtime_closure_policy_payload),
        "runtime_closure_evidence_id": closure_evidence_root["id"],
        "runtime_closure_evidence_payload_sha256": _sha(
            runtime_closure_evidence_payload
        ),
    }
    try:
        custody_root, custody_policy = _overlay_load_custody_policy(
            custody_attestation_policy_payload,
            refs=refs,
            bootstrap=bootstrap,
            online=online,
            oob_id=out_of_band_expected_custody_attestation_policy_id_bytes,
            oob_digest=(
                out_of_band_expected_custody_attestation_policy_payload_sha256_digest
            ),
        )
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2(
            FAIL_DEPENDENCY_V2, "overlay custody-policy verifier rejected"
        ) from exc
    live_root, live_policy = _load_live_policy_v2(
        admission_v2_live_policy_payload,
        refs=refs,
        bootstrap=bootstrap,
        online=online,
        durable=durable,
        auth=auth,
        closure_result=closure_result,
        closure_policy_body=closure_policy_body,
        custody_policy_root=custody_root,
        custody_policy_payload=custody_attestation_policy_payload,
        oob_id=out_of_band_expected_live_runtime_policy_id_bytes,
        oob_digest=out_of_band_expected_live_runtime_policy_payload_sha256_digest,
    )
    a2_policy_root, a2_policy = _load_a2_policy(
        admission_v2_a2_policy_payload,
        refs=refs,
        live_policy_root=live_root,
        live_policy_payload=admission_v2_live_policy_payload,
        live_policy=live_policy,
        durable=durable,
        auth=auth,
        oob_id=out_of_band_expected_live_runtime_anchor_policy_id_bytes,
        oob_digest=(
            out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest
        ),
    )
    public_keys = [
        bootstrap["public_key_hex"],
        online["online_public_key_hex"],
        online["custody_attestor_public_key_hex"],
        a2_policy["anchor_public_key_hex"],
        *(auth["roles"][role]["public_key_hex"] for role in _AUTH_ROLE_ORDER_V2),
    ]
    key_ids = [
        bootstrap["witness_key_id"],
        online["online_key_id"],
        online["custody_attestor_key_id"],
        a2_policy["anchor_key_id"],
        *(auth["roles"][role]["key_id"] for role in _AUTH_ROLE_ORDER_V2),
    ]
    if len(set(public_keys)) != 12 or len(set(key_ids)) != 12:
        _fail(FAIL_BINDING_V2, "external and authority key sets are not disjoint")

    context: dict[str, object] = {
        "lock_id": refs["stage_b_trust_lock_id"],
        "profile_id": refs["authority_profile_id"],
        "registry_id": refs["signer_registry_id"],
        "attempt_id": attempt_id,
        "b480_result_id": refs[
            "predecessor_b480_admission_candidate_result_id"
        ],
        "b480_result_sha": refs[
            "predecessor_b480_admission_candidate_result_payload_sha256"
        ],
        "bootstrap_id": refs["bootstrap_external_witness_policy_id"],
        "online_id": refs["online_witness_policy_id"],
        "durable_id": refs["durable_anchor_policy_id"],
        "closure_policy_id": refs["runtime_closure_policy_id"],
        "closure_evidence_id": refs["runtime_closure_evidence_id"],
        "custody_policy_id": custody_root["custody_attestation_policy_id"],
        "live_policy_id": live_root["admission_v2_live_policy_id"],
        "a2_policy_id": a2_policy_root["admission_v2_a2_policy_id"],
    }
    source = auth["roles"]["SOURCE_RUNTIME_CUSTODIAN"]
    pure = auth["roles"]["PURE_REPLAYER"]
    challenge_root, challenge, context = _load_challenge(
        pre_response_challenge_payload,
        context=context,
        live_policy=live_policy,
        a2_policy=a2_policy,
        source=source,
    )
    response_root, response = _load_response(
        witness_response_payload,
        context=context,
        challenge_root=challenge_root,
        challenge=challenge,
        live_policy=live_policy,
    )
    report_root, report, entry, exit_snapshot = _load_report(
        witness_report_payload,
        context=context,
        challenge_root=challenge_root,
        challenge=challenge,
        response_root=response_root,
        response=response,
        live_policy=live_policy,
        source=source,
    )
    attestation_root, attestation = _load_attestation(
        witness_custody_attestation_payload,
        context=context,
        challenge_root=challenge_root,
        challenge=challenge,
        response_root=response_root,
        response=response,
        report_root=report_root,
        report=report,
        entry=entry,
        exit_snapshot=exit_snapshot,
        custody_policy=custody_policy,
    )
    prefix_bodies = (live_policy, challenge, response, report, attestation)
    if any(
        CURRENT_RESULT_FORBIDDEN_BEFORE_RESULT_FIELDS_V2 & frozenset(body)
        for body in prefix_bodies
    ):
        _fail(FAIL_ACYCLIC_V2, "an object before result has a direct result field")
    if any(
        (CAS_FORBIDDEN_FIELDS_V2 | A2_TRANSCRIPT_FORBIDDEN_IN_SIX_FIELDS_V2)
        & frozenset(body)
        for body in prefix_bodies
    ):
        _fail(FAIL_ACYCLIC_V2, "a pre-result body has a same-session suffix field")
    result_root, result_body = _load_result(
        admission_v2_candidate_result_payload,
        context=context,
        challenge_root=challenge_root,
        response_root=response_root,
        report_root=report_root,
        attestation_root=attestation_root,
        source=source,
        pure=pure,
        prefix_objects=prefix_bodies,
    )
    result_id = result_root["admission_v2_candidate_result_id"]
    result_sha = _sha(admission_v2_candidate_result_payload)
    if _contains_exact_scalar(a2_policy, frozenset((result_id, result_sha))):
        _fail(FAIL_ACYCLIC_V2, "A2 policy names the future admission/2 result")
    context = dict(context)
    context["result_id"] = result_id
    context["result_sha"] = result_sha
    object_roots = (
        live_root,
        challenge_root,
        response_root,
        report_root,
        attestation_root,
        result_root,
    )
    object_payloads = (
        admission_v2_live_policy_payload,
        pre_response_challenge_payload,
        witness_response_payload,
        witness_report_payload,
        witness_custody_attestation_payload,
        admission_v2_candidate_result_payload,
    )
    cas_result, complete = _verify_cas_session(
        stage_b_trust_lock_payload=stage_b_trust_lock_payload,
        authority_profile_payload=authority_profile_payload,
        signer_registry_payload=signer_registry_payload,
        oob_lock_id=out_of_band_expected_lock_id_bytes,
        oob_lock_digest=out_of_band_expected_lock_payload_sha256_digest,
        retention_plan_payload=admission_v2_cas_retention_plan_payload,
        request_payloads=admission_v2_cas_store_request_payloads_in_exact_order,
        object_payloads=admission_v2_cas_object_payloads_in_exact_order,
        signed_receipt_payloads=(
            admission_v2_cas_signed_receipt_payloads_in_exact_order
        ),
        checkpoint_payloads=admission_v2_cas_checkpoint_payloads_in_exact_order,
        head_payloads=admission_v2_cas_head_payloads_in_exact_order,
        response_payloads=admission_v2_cas_response_payloads_in_exact_order,
        context=context,
        object_roots=object_roots,
        expected_object_payloads=object_payloads,
    )
    request_root, checkpoint_root, head_root, receipt_root = _load_a2_transcript(
        request_payload=admission_v2_a2_request_payload,
        checkpoint_payload=admission_v2_a2_checkpoint_payload,
        head_payload=admission_v2_a2_head_payload,
        receipt_payload=admission_v2_a2_receipt_payload,
        context=context,
        a2_policy=a2_policy,
        source=source,
        object_roots=object_roots,
        object_payloads=object_payloads,
        complete=complete,
    )
    return _issue_result(
        schema_version=SCHEMA_VERSION_V2,
        verification_status=(
            "PRESENTED_RAW_ACYCLIC_EXTERNAL_WITNESS_ADMISSION_V2_"
            "VERIFIED_MECHANICS_ONLY"
        ),
        stage_b_trust_lock_id=context["lock_id"],
        authority_profile_id=context["profile_id"],
        signer_registry_id=context["registry_id"],
        attempt_context_id=context["attempt_id"],
        predecessor_b480_admission_candidate_result_id=context["b480_result_id"],
        predecessor_b480_admission_candidate_result_payload_sha256=context[
            "b480_result_sha"
        ],
        custody_attestation_policy_id=context["custody_policy_id"],
        admission_v2_live_policy_id=context["live_policy_id"],
        admission_v2_a2_policy_id=context["a2_policy_id"],
        session_id=context["session_id"],
        session_sequence=context["session_sequence"],
        admission_v2_candidate_result_id=context["result_id"],
        admission_v2_candidate_result_payload_sha256=context["result_sha"],
        six_object_payload_sha256_in_exact_order=tuple(
            _sha(payload) for payload in object_payloads
        ),
        retention_plan_id=cas_result.plan_id,
        complete_cas_head_state_id=cas_result.complete_head_state_id,
        cas_complete_binding_payload_sha256=_sha(_canonical(complete)),
        admission_v2_a2_request_id=request_root["admission_v2_a2_request_id"],
        admission_v2_a2_request_payload_sha256=_sha(
            admission_v2_a2_request_payload
        ),
        admission_v2_a2_checkpoint_id=checkpoint_root[
            "admission_v2_a2_checkpoint_id"
        ],
        admission_v2_a2_checkpoint_payload_sha256=_sha(
            admission_v2_a2_checkpoint_payload
        ),
        admission_v2_a2_head_id=head_root["admission_v2_a2_head_id"],
        admission_v2_a2_head_payload_sha256=_sha(admission_v2_a2_head_payload),
        admission_v2_a2_receipt_id=receipt_root["admission_v2_a2_receipt_id"],
        admission_v2_a2_receipt_payload_sha256=_sha(
            admission_v2_a2_receipt_payload
        ),
        verified_signature_count=PRIVATE_RESULT_VERIFIED_SIGNATURE_COUNT_V2,
        verified_signature_scope=PRIVATE_RESULT_VERIFIED_SIGNATURE_SCOPE_V2,
        verified_protocol_object_count=VERIFIED_PROTOCOL_OBJECT_COUNT_V2,
        dependency_contract_constants_verified=True,
        raw_authority_graph_verified=True,
        frozen_b480_candidate_verified=True,
        offline_static_image_graph_verified=True,
        nineteen_expected_pins_verified=True,
        key_sets_pairwise_disjoint=True,
        pre_response_challenge_order_verified=True,
        no_same_session_forward_references_verified=True,
        response_report_custody_chain_verified=True,
        admission_v2_result_pre_cas_verified=True,
        exact_six_object_cas_replay_verified=True,
        presented_a2_compare_and_append_verified=True,
        post_a2_private_binding_verified=True,
        presented_only=True,
        private_key_present=False,
        filesystem_path_present=False,
        socket_present=False,
        decoded_mapping_input_present=False,
        callback_present=False,
        caller_nonce_present=False,
        caller_clock_present=False,
        actual_runtime_verified=False,
        runtime_closure_verified=False,
        live_channel_verified=False,
        key_custody_verified=False,
        online_key_nonexportable_verified=False,
        distinct_live_uid_gid_verified=False,
        external_cas_verified=False,
        external_anchor_verified=False,
        external_anti_rollback_verified=False,
        external_currentness_verified=False,
        durability_verified=False,
        non_equivocation_verified=False,
        global_anti_fork_verified=False,
        malicious_root_resistance_verified=False,
        authority_evidence=False,
        r01_satisfied=False,
        readiness_evidence=False,
        execution_authorized=False,
    )


def _build_public_verifier_entry_v2(implementation):
    """Expose the exact raw-only ABI without dependency-injection keywords."""

    def verify_external_witness_admission_candidate_v2(
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
        predecessor_b480_admission_candidate_result_payload: bytes,
        runtime_closure_policy_payload: bytes,
        runtime_closure_evidence_payload: bytes,
        raw_oci_index_payload: bytes,
        raw_oci_manifest_payload: bytes,
        raw_oci_config_payload: bytes,
        ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
        ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
        custody_attestation_policy_payload: bytes,
        admission_v2_live_policy_payload: bytes,
        admission_v2_a2_policy_payload: bytes,
        pre_response_challenge_payload: bytes,
        witness_response_payload: bytes,
        witness_report_payload: bytes,
        witness_custody_attestation_payload: bytes,
        admission_v2_candidate_result_payload: bytes,
        admission_v2_cas_retention_plan_payload: bytes,
        admission_v2_cas_store_request_payloads_in_exact_order: tuple[bytes, ...],
        admission_v2_cas_object_payloads_in_exact_order: tuple[bytes, ...],
        admission_v2_cas_signed_receipt_payloads_in_exact_order: tuple[bytes, ...],
        admission_v2_cas_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        admission_v2_cas_head_payloads_in_exact_order: tuple[bytes, ...],
        admission_v2_cas_response_payloads_in_exact_order: tuple[bytes, ...],
        admission_v2_a2_request_payload: bytes,
        admission_v2_a2_checkpoint_payload: bytes,
        admission_v2_a2_head_payload: bytes,
        admission_v2_a2_receipt_payload: bytes,
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
        out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
        out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
        out_of_band_expected_online_witness_policy_id_bytes: bytes,
        out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
        out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
        out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
        out_of_band_expected_runtime_closure_policy_id_bytes: bytes,
        out_of_band_expected_runtime_closure_policy_payload_sha256_digest: bytes,
        out_of_band_expected_runtime_closure_evidence_id_bytes: bytes,
        out_of_band_expected_runtime_closure_evidence_payload_sha256_digest: bytes,
        out_of_band_expected_custody_attestation_policy_id_bytes: bytes,
        out_of_band_expected_custody_attestation_policy_payload_sha256_digest: bytes,
        out_of_band_expected_live_runtime_policy_id_bytes: bytes,
        out_of_band_expected_live_runtime_policy_payload_sha256_digest: bytes,
        out_of_band_expected_live_runtime_anchor_policy_id_bytes: bytes,
        out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest: bytes,
        expected_attempt_context_id_bytes: bytes,
    ) -> VerifiedExternalWitnessAdmissionCandidateV2:
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
            predecessor_b480_admission_candidate_result_payload,
            runtime_closure_policy_payload,
            runtime_closure_evidence_payload,
            raw_oci_index_payload,
            raw_oci_manifest_payload,
            raw_oci_config_payload,
            ordered_raw_layer_chunk_tuples,
            ordered_support_artifact_chunk_tuples,
            custody_attestation_policy_payload,
            admission_v2_live_policy_payload,
            admission_v2_a2_policy_payload,
            pre_response_challenge_payload,
            witness_response_payload,
            witness_report_payload,
            witness_custody_attestation_payload,
            admission_v2_candidate_result_payload,
            admission_v2_cas_retention_plan_payload,
            admission_v2_cas_store_request_payloads_in_exact_order,
            admission_v2_cas_object_payloads_in_exact_order,
            admission_v2_cas_signed_receipt_payloads_in_exact_order,
            admission_v2_cas_checkpoint_payloads_in_exact_order,
            admission_v2_cas_head_payloads_in_exact_order,
            admission_v2_cas_response_payloads_in_exact_order,
            admission_v2_a2_request_payload,
            admission_v2_a2_checkpoint_payload,
            admission_v2_a2_head_payload,
            admission_v2_a2_receipt_payload,
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
            out_of_band_expected_runtime_closure_policy_id_bytes=(
                out_of_band_expected_runtime_closure_policy_id_bytes
            ),
            out_of_band_expected_runtime_closure_policy_payload_sha256_digest=(
                out_of_band_expected_runtime_closure_policy_payload_sha256_digest
            ),
            out_of_band_expected_runtime_closure_evidence_id_bytes=(
                out_of_band_expected_runtime_closure_evidence_id_bytes
            ),
            out_of_band_expected_runtime_closure_evidence_payload_sha256_digest=(
                out_of_band_expected_runtime_closure_evidence_payload_sha256_digest
            ),
            out_of_band_expected_custody_attestation_policy_id_bytes=(
                out_of_band_expected_custody_attestation_policy_id_bytes
            ),
            out_of_band_expected_custody_attestation_policy_payload_sha256_digest=(
                out_of_band_expected_custody_attestation_policy_payload_sha256_digest
            ),
            out_of_band_expected_live_runtime_policy_id_bytes=(
                out_of_band_expected_live_runtime_policy_id_bytes
            ),
            out_of_band_expected_live_runtime_policy_payload_sha256_digest=(
                out_of_band_expected_live_runtime_policy_payload_sha256_digest
            ),
            out_of_band_expected_live_runtime_anchor_policy_id_bytes=(
                out_of_band_expected_live_runtime_anchor_policy_id_bytes
            ),
            out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest=(
                out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest
            ),
            expected_attempt_context_id_bytes=expected_attempt_context_id_bytes,
        )

    return verify_external_witness_admission_candidate_v2


def verify_real_external_witness_admission_v2(
    *args: object, **kwargs: object
) -> NoReturn:
    """Fail closed: no real admission actor/channel exists in this module."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V2,
        "real external-witness admission/2 is exactly NOT_IMPLEMENTED",
    )


def _definition_captured_function_graph_v2() -> dict[str, FunctionType]:
    """Clone local functions onto a private namespace and freeze defaults."""

    namespace = dict(globals())
    originals: dict[str, FunctionType] = {
        name: value
        for name, value in namespace.items()
        if type(value) is FunctionType
        and value.__module__ == __name__
        and name != "_definition_captured_function_graph_v2"
    }
    private_namespace = dict(namespace)
    provisional: dict[str, FunctionType] = {}
    provisional_by_id: dict[int, FunctionType] = {}
    for name, function in originals.items():
        clone = FunctionType(
            function.__code__, private_namespace, function.__name__,
            function.__defaults__, function.__closure__,
        )
        clone.__kwdefaults__ = function.__kwdefaults__
        provisional[name] = clone
        provisional_by_id[id(function)] = clone
    private_namespace.update(provisional)

    def captured_default(value: object) -> object:
        return provisional_by_id.get(id(value), value)

    captured: dict[str, FunctionType] = {}
    for name, function in originals.items():
        defaults = function.__defaults__
        clone = FunctionType(
            function.__code__, private_namespace, function.__name__,
            None if defaults is None else tuple(captured_default(item) for item in defaults),
            function.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if function.__kwdefaults__ is None
            else {
                key: captured_default(value)
                for key, value in function.__kwdefaults__.items()
            }
        )
        captured[name] = clone
    private_namespace.update(captured)
    return captured


_CAPTURED_V2 = _definition_captured_function_graph_v2()
verify_external_witness_admission_candidate_v2 = _build_public_verifier_entry_v2(
    _CAPTURED_V2["_verify_external_witness_admission_candidate_implementation_v2"]
)
require_verified_external_witness_admission_candidate_v2 = _CAPTURED_V2[
    "require_verified_external_witness_admission_candidate_v2"
]
verify_real_external_witness_admission_v2 = _CAPTURED_V2[
    "verify_real_external_witness_admission_v2"
]

del _CAPTURED_V2
del _verify_external_witness_admission_candidate_implementation_v2
del _issue_result_v2
del _require_result_v2
del _build_public_verifier_entry_v2
del _definition_captured_function_graph_v2
del _capture_dependency_function_graph


__all__ = [
    "SCHEMA_VERSION_V2",
    "STATUS_V2",
    "CLAIM_LEVEL_V2",
    "CANONICALIZATION_V2",
    "ALGORITHM_V2",
    "REAL_ADMISSION_STATUS_V2",
    "INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256",
    "INSPECTED_LIVE_RUNTIME_OVERLAY_V2_SOURCE_SHA256",
    "INSPECTED_AUTHORITY_V2_SOURCE_SHA256",
    "INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256",
    "INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256",
    "DEPENDENCY_SOURCE_SHA256_IN_EXACT_ORDER_V2",
    "ADMISSION_V2_LIVE_POLICY_SCHEMA",
    "ADMISSION_V2_PRE_RESPONSE_CHALLENGE_SCHEMA",
    "ADMISSION_V2_WITNESS_RESPONSE_SCHEMA",
    "ADMISSION_V2_WITNESS_REPORT_SCHEMA",
    "ADMISSION_V2_CUSTODY_ATTESTATION_SCHEMA",
    "ADMISSION_V2_CANDIDATE_RESULT_SCHEMA",
    "ADMISSION_V2_A2_MAPPING_SCHEMA",
    "ADMISSION_V2_A2_POLICY_SCHEMA",
    "ADMISSION_V2_A2_REQUEST_SCHEMA",
    "ADMISSION_V2_A2_CHECKPOINT_SCHEMA",
    "ADMISSION_V2_A2_HEAD_SCHEMA",
    "ADMISSION_V2_A2_RECEIPT_SCHEMA",
    "ADMISSION_V2_CAS_COMPLETE_BINDING_SCHEMA",
    "ADMISSION_V2_LIVE_POLICY_FIELDS",
    "ADMISSION_V2_LIVE_POLICY_BODY_FIELDS",
    "ADMISSION_V2_PRE_RESPONSE_CHALLENGE_FIELDS",
    "ADMISSION_V2_PRE_RESPONSE_CHALLENGE_BODY_FIELDS",
    "ADMISSION_V2_WITNESS_RESPONSE_FIELDS",
    "ADMISSION_V2_WITNESS_RESPONSE_BODY_FIELDS",
    "ADMISSION_V2_WITNESS_REPORT_FIELDS",
    "ADMISSION_V2_WITNESS_REPORT_BODY_FIELDS",
    "ADMISSION_V2_CUSTODY_ATTESTATION_FIELDS",
    "ADMISSION_V2_CUSTODY_ATTESTATION_BODY_FIELDS",
    "ADMISSION_V2_CANDIDATE_RESULT_FIELDS",
    "ADMISSION_V2_CANDIDATE_RESULT_BODY_FIELDS",
    "ADMISSION_V2_A2_MAPPING_FIELDS",
    "ADMISSION_V2_A2_POLICY_FIELDS",
    "ADMISSION_V2_A2_POLICY_BODY_FIELDS",
    "ADMISSION_V2_A2_REQUEST_FIELDS",
    "ADMISSION_V2_A2_REQUEST_BODY_FIELDS",
    "ADMISSION_V2_A2_CHECKPOINT_FIELDS",
    "ADMISSION_V2_A2_CHECKPOINT_BODY_FIELDS",
    "ADMISSION_V2_A2_HEAD_FIELDS",
    "ADMISSION_V2_A2_HEAD_BODY_FIELDS",
    "ADMISSION_V2_A2_RECEIPT_FIELDS",
    "ADMISSION_V2_A2_RECEIPT_BODY_FIELDS",
    "ADMISSION_V2_CAS_COMPLETE_BINDING_FIELDS",
    "EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2",
    "CAS_OBJECTS_IN_EXACT_ORDER_V2",
    "CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2",
    "PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2",
    "PRIVATE_RESULT_VERIFIED_SIGNATURE_COUNT_V2",
    "PRIVATE_RESULT_VERIFIED_SIGNATURE_SCOPE_V2",
    "VERIFIED_PROTOCOL_OBJECT_COUNT_V2",
    "SIGN_WITNESS_RESPONSE_PURPOSE_V2",
    "ATTEST_WITNESS_CUSTODY_PURPOSE_V2",
    "SIGN_A2_POLICY_PURPOSE_V2",
    "SIGN_A2_CHECKPOINT_PURPOSE_V2",
    "SIGN_A2_HEAD_PURPOSE_V2",
    "SIGN_A2_RECEIPT_PURPOSE_V2",
    "Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2",
    "VerifiedExternalWitnessAdmissionCandidateV2",
    "verify_external_witness_admission_candidate_v2",
    "require_verified_external_witness_admission_candidate_v2",
    "verify_real_external_witness_admission_v2",
]
