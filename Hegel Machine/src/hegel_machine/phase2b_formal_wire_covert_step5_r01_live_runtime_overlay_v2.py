"""Presented-only R01 live-runtime overlay/2 replay contract.

This module is deliberately a producer-free, raw-bytes verifier.  It composes
the frozen external-admission candidate, offline image-closure mechanics, a
six-object CAS replay, and a separately pinned A/2 compare-and-append
transcript.  Acceptance proves only exact presented mechanics.  It does not
observe a socket, open procfs or a pidfd, hold a private key, establish
hardware custody, make a filesystem durable, provide external currentness, or
authorize execution.  The real operational verifier is NOT_IMPLEMENTED.

This is necessarily a post-b480, non-admitted presented overlay: b480/1 Q1
already forward-references a runtime report, so binding the same Q1 to this
overlay's live challenge/report would create a content-address cycle.  Nothing
accepted here retroactively upgrades b480/1.  Production admission requires a
separate admission/2 pre-response challenge plus an independently durable
external A/2 service.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct
import threading
from types import FunctionType, MappingProxyType
from typing import Final, Mapping, NoReturn

from . import phase2b_formal_wire_covert_step5_authority_v2 as authority
from . import phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 as admission
from . import phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure
from . import phase2b_formal_wire_covert_step5_store_replay_core_v2 as store_replay
from .rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - deployment image dependent.
    _InvalidSignature = None
    _Ed25519PublicKey = None


SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-live-runtime-overlay/2"
)
STATUS_V2: Final = "PRESENTED_LIVE_RUNTIME_OVERLAY_MECHANICS_CANDIDATE"
CLAIM_LEVEL_V2: Final = (
    "RAW_PRESENTED_MECHANICS_ONLY_NOT_LIVE_NOT_CURRENT_NOT_DURABLE_"
    "NOT_NON_EQUIVOCATING_NOT_AUTHORITY_NOT_R01"
)
CANONICALIZATION_V2: Final = "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
ALGORITHM_V2: Final = "Ed25519"
KEY_EPOCH_V2: Final = 1

INSPECTED_AUTHORITY_V2_SOURCE_SHA256: Final = (
    "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
)
INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256: Final = (
    "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285"
)
INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256: Final = (
    "f836b3d00f332398214d3ccf7787109abb8f2de96b7b8281748bbb0ec695f2f9"
)
INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256: Final = (
    "e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02"
)

LIVE_RUNTIME_POLICY_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-policy/2"
)
CUSTODY_ATTESTATION_POLICY_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-custody-attestation-policy/2"
)
LIVE_RUNTIME_CHALLENGE_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-challenge/2"
)
LIVE_RUNTIME_RESPONSE_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-response/2"
)
LIVE_RUNTIME_REPORT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-report/2"
)
LIVE_RUNTIME_CUSTODY_ATTESTATION_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-custody-attestation/2"
)
LIVE_RUNTIME_RESULT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-result/2"
)
PROCESS_SNAPSHOT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-process-snapshot/2"
)
LIVE_RUNTIME_ANCHOR_MAPPING_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-anchor-mapping/2"
)
LIVE_RUNTIME_ANCHOR_POLICY_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-anchor-policy/2"
)
LIVE_RUNTIME_ANCHOR_REQUEST_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-anchor-request/2"
)
LIVE_RUNTIME_ANCHOR_CHECKPOINT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-anchor-checkpoint/2"
)
LIVE_RUNTIME_ANCHOR_HEAD_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-anchor-head/2"
)
LIVE_RUNTIME_ANCHOR_RECEIPT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-anchor-receipt/2"
)
CAS_COMPLETE_BINDING_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-step5-live-runtime-cas-complete-binding/2"
)

SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_PURPOSE_V2: Final = (
    "SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_V2"
)
ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2: Final = (
    "ATTEST_ONLINE_KEY_NONEXPORTABILITY_V2"
)
SIGN_LIVE_RUNTIME_ANCHOR_POLICY_PURPOSE_V2: Final = (
    "SIGN_LIVE_RUNTIME_ANCHOR_POLICY_V2"
)
SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_PURPOSE_V2: Final = (
    "SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_V2"
)
SIGN_LIVE_RUNTIME_ANCHOR_HEAD_PURPOSE_V2: Final = (
    "SIGN_LIVE_RUNTIME_ANCHOR_HEAD_V2"
)
SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_PURPOSE_V2: Final = (
    "SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_V2"
)
SIGN_RUNTIME_CLOSURE_PURPOSE_V2: Final = "SIGN_RUNTIME_CLOSURE"
SIGN_PURE_REPLAY_RESULT_PURPOSE_V2: Final = "SIGN_PURE_REPLAY_RESULT"
SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V2: Final = "SIGN_ONLINE_WITNESS_POLICY"

LIVE_OVERLAY_ONLINE_PURPOSES_IN_EXACT_ORDER_V2: Final = (
    *admission.ONLINE_PURPOSES_IN_EXACT_ORDER_V1,
    SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_PURPOSE_V2,
)
_AUTH_ROLE_ORDER_V2: Final = tuple(authority.ROLE_ORDER_V2)
_AUTH_SERVICE_ORDER_V2: Final = tuple(authority.SERVICE_ORDER_V2)
_AUTH_DETACHED_SIGNATURE_SCHEMA_V2: Final = authority.DETACHED_SIGNATURE_SCHEMA_V2
_ADMISSION_BOOTSTRAP_POLICY_FIELDS_V1: Final = frozenset(
    admission.BOOTSTRAP_EXTERNAL_WITNESS_POLICY_FIELDS_V1
)
_ADMISSION_ONLINE_POLICY_FIELDS_V1: Final = frozenset(
    admission.ONLINE_WITNESS_POLICY_FIELDS_V1
)
_ADMISSION_ONLINE_POLICY_BODY_FIELDS_V1: Final = frozenset(
    admission.ONLINE_WITNESS_POLICY_BODY_FIELDS_V1
)
_ADMISSION_DURABLE_POLICY_FIELDS_V1: Final = frozenset(
    admission.DURABLE_ANCHOR_POLICY_FIELDS_V1
)
_ADMISSION_DURABLE_POLICY_BODY_FIELDS_V1: Final = frozenset(
    admission.DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1
)
_ADMISSION_RESULT_FIELDS_V1: Final = frozenset(
    admission.ADMISSION_CANDIDATE_RESULT_FIELDS_V1
)
_ADMISSION_BOOTSTRAP_PURPOSES_V1: Final = tuple(
    admission.BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1
)
_ADMISSION_RESULT_PREFIX_V1: Final = (
    "phase2b_step5_external_admission_candidate_result_v1_"
)
_STORE_RETENTION_PLAN_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-retention-plan/1"
)
_STORE_PLAN_ID_PREFIX_V1: Final = "phase2b_step5_retention_plan_v1_"
_STORE_HEAD_STATE_ID_PREFIX_V2: Final = "phase2b_step5_cas_head_state_v2_"
CUSTODY_PURPOSES_IN_EXACT_ORDER_V2: Final = (
    ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2,
)
ANCHOR_PURPOSES_IN_EXACT_ORDER_V2: Final = (
    SIGN_LIVE_RUNTIME_ANCHOR_POLICY_PURPOSE_V2,
    SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_PURPOSE_V2,
    SIGN_LIVE_RUNTIME_ANCHOR_HEAD_PURPOSE_V2,
    SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_PURPOSE_V2,
)

PRE_CAS_VERIFIED_SIGNATURE_SCOPE_V2: Final = (
    "CUSTODY_POLICY_LIVE_POLICY_CHALLENGE_RESPONSE_REPORT_ATTESTATION_RESULT_"
    "ONLY_EXCLUDES_AUTHORITY_GRAPH_B480_CAS_AND_A2"
)
PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2: Final = 9
PRE_CAS_VERIFIED_PROTOCOL_OBJECT_COUNT_V2: Final = 7
PRIVATE_OVERLAY_VERIFIED_SIGNATURE_SCOPE_V2: Final = (
    "OVERLAY_CUSTODY_POLICY_SIX_LIVE_ARTIFACTS_AND_A2_POLICY_REQUEST_"
    "CHECKPOINT_HEAD_RECEIPT_ONLY_EXCLUDES_AUTHORITY_GRAPH_B480_AND_CAS_RECEIPTS"
)
PRIVATE_OVERLAY_VERIFIED_SIGNATURE_COUNT_V2: Final = 14
CAS_PLAN_KIND_V2: Final = (
    "PHASE2B_STEP5_LIVE_RUNTIME_OVERLAY_V2_SIX_OBJECTS"
)
CAS_OBJECTS_IN_EXACT_ORDER_V2: Final = (
    (0, "live_runtime_policy_v2", 16_777_216),
    (1, "live_runtime_challenge_v2", 16_777_216),
    (2, "live_runtime_response_v2", 16_777_216),
    (3, "live_runtime_report_v2", 16_777_216),
    (4, "live_runtime_custody_attestation_v2", 16_777_216),
    (5, "live_runtime_result_v2", 16_777_216),
)
CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2: Final = tuple(
    row[1] for row in CAS_OBJECTS_IN_EXACT_ORDER_V2
)

FAIL_INPUT_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_INPUT"
FAIL_CANONICAL_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_CANONICAL"
FAIL_SCHEMA_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_SCHEMA"
FAIL_CONTENT_ID_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_CONTENT_ID"
FAIL_BINDING_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_BINDING"
FAIL_SIGNATURE_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_SIGNATURE"
FAIL_DEPENDENCY_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_DEPENDENCY"
FAIL_CLAIM_BOUNDARY_V2: Final = (
    "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_CLAIM_BOUNDARY"
)
FAIL_RESULT_V2: Final = "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_RESULT"
FAIL_NOT_IMPLEMENTED_V2: Final = (
    "FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_NOT_IMPLEMENTED"
)

_MAXIMUM_CANONICAL_BYTES: Final = 16 * 1024 * 1024
_SAFE_INTEGER: Final = (1 << 53) - 1
_HEX64_RE: Final = re.compile(r"[0-9a-f]{64}")
_HEX128_RE: Final = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_CONTENT_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")
_STORE_SHA_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_DECIMAL_RE: Final = re.compile(r"0|[1-9][0-9]{0,63}")
_LOWER_HEX_RE: Final = re.compile(r"[0-9a-f]+")
_SHA256_V2: Final = hashlib.sha256
_STRUCT_PACK_V2: Final = struct.pack


class Phase2BStep5R01LiveRuntimeOverlayErrorV2(ValueError):
    """Stable fail-closed error for presented overlay/2 replay."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(code, detail)


def _fields(text: str) -> frozenset[str]:
    return frozenset(text.split())


EXTERNAL_SIGNATURE_FIELDS_V2: Final = _fields(
    "schema_version algorithm signer_key_id key_epoch purpose signature_hex"
)
AUTHORITY_DETACHED_SIGNATURE_FIELDS_V2: Final = authority.DETACHED_SIGNATURE_FIELDS_V2
EXTERNAL_SIGNATURE_PREIMAGE_FIELDS_V2: Final = _fields(
    "schema_version artifact_kind content_id body signer_key_id key_epoch purpose"
)
RUNTIME_IDENTITY_FIELDS_V2: Final = _fields(
    "implementation implementation_version language_version abi_tag cache_tag executable_artifact_role"
)
POLICY_ALONE_NONCLAIMS_FIELDS_V2: Final = _fields(
    "service_image_exists service_image_pulled service_image_executed "
    "live_runtime_closure_observed entry_exit_closure_equal live_uid_gid_verified "
    "live_control_domain_verified key_custody_verified online_key_nonexportable_verified "
    "runtime_closure_signature_verified remote_attestation_verified external_cas_verified "
    "external_anchor_verified external_currentness_verified durability_verified "
    "non_equivocation_verified global_anti_fork_verified malicious_root_resistance_verified "
    "authority_evidence readiness_evidence r01_satisfied execution_authorized"
)
PROCESS_SNAPSHOT_FIELDS_V2: Final = _fields(
    "schema_version phase observation_source address_family socket_type "
    "peer_credential_option peer_credential_struct_format peer_credential_struct_size_bytes "
    "so_peercred_payload_sha256 accepted_socket_inode server_socket_inode peer_pid peer_uid "
    "peer_gid host_boot_id process_start_ticks user_namespace_inode pid_namespace_inode "
    "mount_namespace_inode exe_device exe_inode exe_sha256 cgroup_sha256 pidfd_open_succeeded "
    "pidfd_alive procfs_read_succeeded observation_complete process_snapshot_id"
)
LIVE_RUNTIME_ANCHOR_MAPPING_FIELDS_V2: Final = _fields(
    "schema_version attempt_context_id session_id session_sequence "
    "admission_candidate_result_id admission_candidate_result_payload_sha256 "
    "live_runtime_result_id live_runtime_result_payload_sha256 retention_plan_id "
    "retention_plan_payload_sha256 retention_session_id chain_id cas_complete_head_state_id "
    "cas_final_checkpoint_payload_sha256 cas_final_head_payload_sha256 "
    "cas_final_response_payload_sha256 anchor_sequence effective_state superseded revoked"
)
LIVE_RUNTIME_CAS_COMPLETE_BINDING_FIELDS_V2: Final = _fields(
    "schema_version authority_profile_id signer_registry_id attempt_context_id session_id "
    "session_sequence retention_plan_id retention_plan_payload_sha256 retention_session_id "
    "chain_id plan_kind expected_object_count artifact_kinds_in_exact_order final_ordinal "
    "final_artifact_kind live_runtime_result_id live_runtime_result_payload_sha256 "
    "final_store_request_id final_store_request_payload_sha256 final_object_id "
    "final_object_sha256 final_object_size_bytes final_signed_receipt_payload_sha256 "
    "final_checkpoint_payload_sha256 final_head_state_id final_head_payload_sha256 "
    "final_response_payload_sha256 observed_object_count complete complete_head_required "
    "durable_before_return readback_verified"
)
CAS_PLAN_OBJECT_FIELDS_V2: Final = _fields("ordinal artifact_kind maximum_bytes")
CAS_RETENTION_PLAN_FIELDS_V2: Final = _fields(
    "schema_version authority_profile_id signer_registry_id attempt_context_id "
    "retention_session_id chain_id plan_kind expected_object_count objects_in_exact_order "
    "complete_head_required plan_id"
)
CAS_GENERIC_REQUEST_FIELDS_V2: Final = _fields(
    "schema_version authority_profile_id signer_registry_id attempt_context_id "
    "retention_session_id chain_id plan_id request_id ordinal artifact_kind "
    "bound_artifact_id bound_artifact_sha256 reported_size_bytes fd_count"
)

LIVE_RUNTIME_POLICY_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_policy_id signatures_in_exact_order"
)
CUSTODY_ATTESTATION_POLICY_FIELDS_V2: Final = _fields(
    "schema_version body custody_attestation_policy_id signatures_in_exact_order"
)
LIVE_RUNTIME_CHALLENGE_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_challenge_id signatures_in_exact_order"
)
LIVE_RUNTIME_RESPONSE_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_response_id signatures_in_exact_order"
)
LIVE_RUNTIME_REPORT_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_report_id signatures_in_exact_order"
)
LIVE_RUNTIME_CUSTODY_ATTESTATION_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_custody_attestation_id signatures_in_exact_order"
)
LIVE_RUNTIME_RESULT_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_result_id signatures_in_exact_order"
)
LIVE_RUNTIME_ANCHOR_POLICY_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_anchor_policy_id signatures_in_exact_order"
)
LIVE_RUNTIME_ANCHOR_REQUEST_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_anchor_request_id signatures_in_exact_order"
)
LIVE_RUNTIME_ANCHOR_CHECKPOINT_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_anchor_checkpoint_id signatures_in_exact_order"
)
LIVE_RUNTIME_ANCHOR_HEAD_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_anchor_head_id signatures_in_exact_order"
)
LIVE_RUNTIME_ANCHOR_RECEIPT_FIELDS_V2: Final = _fields(
    "schema_version body live_runtime_anchor_receipt_id signatures_in_exact_order"
)

LIVE_RUNTIME_POLICY_BODY_FIELDS_V2: Final = _fields(
    "schema_version status claim_level canonicalization policy_epoch stage_b_trust_lock_id "
    "stage_b_trust_lock_payload_sha256 authority_profile_id authority_profile_payload_sha256 "
    "signer_registry_id signer_registry_payload_sha256 attempt_context_id "
    "admission_candidate_result_id admission_candidate_result_payload_sha256 "
    "bootstrap_external_witness_policy_id bootstrap_external_witness_policy_payload_sha256 "
    "online_witness_policy_id online_witness_policy_payload_sha256 durable_anchor_policy_id "
    "durable_anchor_policy_payload_sha256 runtime_closure_policy_id "
    "runtime_closure_policy_payload_sha256 runtime_closure_evidence_id "
    "runtime_closure_evidence_payload_sha256 runtime_closure_service_name runtime_identity "
    "custody_attestation_policy_id custody_attestation_policy_payload_sha256 bootstrap_key_id "
    "bootstrap_key_epoch bootstrap_public_key_hex bootstrap_authorized_signature_purposes_in_exact_order "
    "online_witness_key_id online_witness_key_epoch online_witness_public_key_hex "
    "online_witness_authorized_signature_purposes_in_exact_order online_witness_service_binding_id "
    "online_witness_control_domain_id online_witness_service_uid online_witness_service_gid "
    "online_witness_service_user_namespace_inode online_witness_expected_cgroup_sha256 "
    "source_runtime_custodian_role source_runtime_custodian_control_domain "
    "source_runtime_custodian_service_binding_id source_runtime_custodian_service_uid "
    "source_runtime_custodian_service_gid source_runtime_custodian_key_id "
    "source_runtime_custodian_key_epoch source_runtime_custodian_public_key_hex "
    "source_runtime_custodian_authorized_signature_purposes_in_exact_order pure_replayer_role "
    "pure_replayer_control_domain pure_replayer_service_binding_id pure_replayer_service_uid "
    "pure_replayer_service_gid pure_replayer_key_id pure_replayer_key_epoch "
    "pure_replayer_public_key_hex pure_replayer_authorized_signature_purposes_in_exact_order "
    "cas_store_role cas_store_control_domain cas_store_service_binding_id cas_store_service_uid "
    "cas_store_service_gid cas_store_key_id cas_store_key_epoch cas_store_public_key_hex "
    "cas_store_authorized_signature_purposes_in_exact_order custody_attestor_key_id "
    "custody_attestor_key_epoch custody_attestor_public_key_hex "
    "custody_attestor_authorized_signature_purposes_in_exact_order custody_attestation_profile "
    "independent_custody_attestor_required online_key_nonexportable_required "
    "live_runtime_anchor_key_id live_runtime_anchor_key_epoch live_runtime_anchor_public_key_hex "
    "live_runtime_anchor_authorized_signature_purposes_in_exact_order "
    "live_runtime_anchor_control_domain_id transport_address_family transport_socket_type "
    "transport_peer_credential_option transport_peer_credential_struct_format "
    "transport_peer_credential_struct_size_bytes exactly_one_challenge_packet_required "
    "exactly_one_response_packet_required scm_rights_allowed maximum_packet_bytes pidfd_required "
    "procfs_identity_required entry_exit_identity_equal_required challenge_nonce_size_bytes "
    "response_nonce_size_bytes maximum_round_trip_nanoseconds monotonic_clock caller_nonce_forbidden "
    "caller_sequence_forbidden one_active_session_per_attempt_and_online_key_required "
    "cas_retention_plan_kind cas_chain_id_profile cas_expected_object_count "
    "cas_artifact_kinds_in_exact_order cas_complete_head_required "
    "cas_durable_before_return_required cas_readback_required anchor_slot_key_profile "
    "anchor_compare_and_append_semantics anchor_durable_before_return_required "
    "policy_alone_nonclaims"
)
CUSTODY_ATTESTATION_POLICY_BODY_FIELDS_V2: Final = _fields(
    "schema_version status claim_level canonicalization policy_epoch stage_b_trust_lock_id "
    "stage_b_trust_lock_payload_sha256 authority_profile_id authority_profile_payload_sha256 "
    "signer_registry_id signer_registry_payload_sha256 attempt_context_id "
    "bootstrap_external_witness_policy_id bootstrap_external_witness_policy_payload_sha256 "
    "online_witness_key_id online_witness_key_epoch online_witness_public_key_hex "
    "custody_attestor_key_id "
    "custody_attestor_key_epoch custody_attestor_public_key_hex custody_attestor_signature_purpose "
    "attestation_evidence_format trusted_certificate_root_ids_in_exact_order "
    "allowed_certificate_signature_algorithms_in_exact_order minimum_attestation_security_version "
    "challenge_nonce_size_bytes report_binding_required online_key_binding_required "
    "online_key_generated_in_attested_boundary_required online_key_nonexportable_required "
    "certificate_revocation_check_required maximum_attestation_evidence_bytes policy_alone_nonclaims"
)
_LIVE_SESSION_LINEAGE_FIELDS_V2: Final = _fields(
    "status claim_level stage_b_trust_lock_id authority_profile_id signer_registry_id "
    "attempt_context_id bootstrap_external_witness_policy_id online_witness_policy_id "
    "durable_anchor_policy_id runtime_closure_policy_id runtime_closure_evidence_id "
    "custody_attestation_policy_id live_runtime_policy_id live_runtime_anchor_policy_id "
    "admission_candidate_result_id admission_candidate_result_payload_sha256 session_id session_sequence"
)
LIVE_RUNTIME_CHALLENGE_BODY_FIELDS_V2: Final = _LIVE_SESSION_LINEAGE_FIELDS_V2 | _fields(
    "schema_version challenge_nonce challenge_nonce_size_bytes challenger_role challenger_key_id "
    "challenger_key_epoch expected_responder_key_id expected_responder_key_epoch "
    "entry_process_snapshot entry_process_snapshot_payload_sha256 expected_previous_anchor_mapping "
    "challenge_monotonic_nanoseconds_decimal maximum_round_trip_nanoseconds one_shot_required "
    "challenge_outcome"
)
LIVE_RUNTIME_RESPONSE_BODY_FIELDS_V2: Final = _LIVE_SESSION_LINEAGE_FIELDS_V2 | _fields(
    "schema_version live_runtime_challenge_id live_runtime_challenge_payload_sha256 challenge_nonce "
    "challenge_nonce_size_bytes responder_key_id responder_key_epoch response_nonce "
    "response_nonce_size_bytes entry_process_snapshot_id entry_process_snapshot_payload_sha256 "
    "response_outcome"
)
LIVE_RUNTIME_REPORT_BODY_FIELDS_V2: Final = _LIVE_SESSION_LINEAGE_FIELDS_V2 | _fields(
    "schema_version live_runtime_challenge_id live_runtime_challenge_payload_sha256 "
    "live_runtime_response_id live_runtime_response_payload_sha256 challenge_nonce response_nonce "
    "entry_process_snapshot entry_process_snapshot_payload_sha256 exit_process_snapshot "
    "exit_process_snapshot_payload_sha256 entry_exit_identity_fields_in_exact_order "
    "entry_exit_identity_equal challenge_sent_monotonic_nanoseconds_decimal "
    "response_received_monotonic_nanoseconds_decimal round_trip_nanoseconds "
    "maximum_round_trip_nanoseconds round_trip_within_policy exact_one_challenge_packet_sent "
    "exact_one_response_packet_received response_message_flags response_packet_size_bytes "
    "response_ancillary_item_count message_truncation_absent control_truncation_absent "
    "unexpected_ancillary_absent so_peercred_verified pidfd_identity_verified "
    "procfs_identity_verified service_binding_verified runtime_closure_policy_binding_verified "
    "runtime_closure_evidence_binding_verified runtime_executable_binding_verified "
    "distinct_live_uid_gid_verified live_channel_observed report_outcome"
)
LIVE_RUNTIME_CUSTODY_ATTESTATION_BODY_FIELDS_V2: Final = _LIVE_SESSION_LINEAGE_FIELDS_V2 | _fields(
    "schema_version live_runtime_challenge_id live_runtime_challenge_payload_sha256 "
    "live_runtime_response_id live_runtime_response_payload_sha256 live_runtime_report_id "
    "live_runtime_report_payload_sha256 challenge_nonce response_nonce entry_process_snapshot_id "
    "entry_process_snapshot_payload_sha256 exit_process_snapshot_id "
    "exit_process_snapshot_payload_sha256 attestation_sequence attestor_service_instance_id "
    "attestor_control_domain_id attestor_key_id attestor_key_epoch attestation_purpose "
    "online_witness_key_id online_witness_key_epoch online_witness_public_key_hex "
    "opaque_key_handle_sha256 attestation_evidence_format attestation_evidence_hex "
    "certificate_chain_der_hex_in_exact_order certificate_chain_root_id "
    "endorsement_material_hex_in_exact_order revocation_material_hex_in_exact_order "
    "attested_challenge_nonce attested_report_id attested_report_payload_sha256 "
    "attested_online_key_public_key_sha256 attested_key_handle_sha256 "
    "online_key_generated_in_attested_boundary online_key_nonexportable certificate_chain_verified "
    "quote_verified fresh_nonce_verified report_binding_verified online_key_binding_verified "
    "attestation_outcome"
)
LIVE_RUNTIME_RESULT_BODY_FIELDS_V2: Final = _LIVE_SESSION_LINEAGE_FIELDS_V2 | _fields(
    "schema_version live_runtime_challenge_id live_runtime_challenge_payload_sha256 "
    "live_runtime_response_id live_runtime_response_payload_sha256 live_runtime_report_id "
    "live_runtime_report_payload_sha256 live_runtime_custody_attestation_id "
    "live_runtime_custody_attestation_payload_sha256 verified_signature_count "
    "verified_signature_scope verified_protocol_object_count raw_authority_graph_verified "
    "frozen_b480_candidate_verified nineteen_expected_pins_verified key_sets_pairwise_disjoint "
    "offline_static_image_graph_verified service_image_exists service_image_pulled "
    "service_image_executed runtime_closure_verified entry_exit_process_identity_equal "
    "socket_seqpacket_verified so_peercred_verified pidfd_identity_verified procfs_identity_verified "
    "actor_nonce_freshness_verified actor_sequence_high_water_verified one_shot_verified "
    "round_trip_freshness_verified key_custody_verified online_key_nonexportable_verified "
    "distinct_live_uid_gid_verified live_channel_verified external_currentness_verified "
    "durability_verified non_equivocation_verified global_anti_fork_verified "
    "malicious_root_resistance_verified authority_evidence r01_satisfied readiness_evidence "
    "execution_authorized presented_only private_key_present spool_path_present socket_present "
    "decoded_mapping_input_present callback_present caller_nonce_present caller_clock_present "
    "first_failure_code"
)

LIVE_RUNTIME_ANCHOR_POLICY_BODY_FIELDS_V2: Final = _fields(
    "schema_version status claim_level canonicalization policy_epoch stage_b_trust_lock_id "
    "stage_b_trust_lock_payload_sha256 authority_profile_id authority_profile_payload_sha256 "
    "signer_registry_id signer_registry_payload_sha256 attempt_context_id "
    "admission_candidate_result_id admission_candidate_result_payload_sha256 "
    "bootstrap_external_witness_policy_id bootstrap_external_witness_policy_payload_sha256 "
    "online_witness_policy_id online_witness_policy_payload_sha256 durable_anchor_policy_id "
    "durable_anchor_policy_payload_sha256 live_runtime_policy_id live_runtime_policy_payload_sha256 "
    "anchor_namespace anchor_control_domain_id anchor_service_binding_id anchor_service_uid "
    "anchor_service_gid anchor_service_user_namespace_inode anchor_key_id anchor_key_epoch "
    "anchor_public_key_hex algorithm authorized_signature_purposes_in_exact_order requester_role "
    "requester_control_domain requester_key_id requester_key_epoch requester_public_key_hex "
    "requester_signature_purpose genesis_checkpoint_id genesis_head_id genesis_mapping "
    "oob_high_water_checkpoint_id oob_high_water_checkpoint_payload_sha256 "
    "oob_high_water_checkpoint_sequence oob_high_water_head_id "
    "oob_high_water_head_payload_sha256 oob_high_water_head_sequence oob_high_water_mapping "
    "admission_slot_key_profile compare_and_append_semantics durable_before_return_required "
    "cas_complete_head_required"
)
_ANCHOR_COMMON_FIELDS_V2: Final = _fields(
    "status claim_level stage_b_trust_lock_id authority_profile_id signer_registry_id "
    "attempt_context_id bootstrap_external_witness_policy_id online_witness_policy_id "
    "durable_anchor_policy_id live_runtime_policy_id live_runtime_anchor_policy_id "
    "admission_candidate_result_id admission_candidate_result_payload_sha256 session_id session_sequence"
)
LIVE_RUNTIME_ANCHOR_REQUEST_BODY_FIELDS_V2: Final = _ANCHOR_COMMON_FIELDS_V2 | _fields(
    "schema_version live_runtime_policy_payload_sha256 live_runtime_challenge_id "
    "live_runtime_challenge_payload_sha256 live_runtime_response_id live_runtime_response_payload_sha256 "
    "live_runtime_report_id live_runtime_report_payload_sha256 live_runtime_custody_attestation_id "
    "live_runtime_custody_attestation_payload_sha256 live_runtime_result_id "
    "live_runtime_result_payload_sha256 cas_complete_binding expected_previous_anchor_checkpoint_id "
    "expected_previous_anchor_checkpoint_payload_sha256 expected_previous_anchor_checkpoint_sequence "
    "expected_previous_anchor_head_id expected_previous_anchor_head_payload_sha256 "
    "expected_previous_anchor_head_sequence expected_current_mapping proposed_current_mapping "
    "compare_and_append_mode"
)
LIVE_RUNTIME_ANCHOR_CHECKPOINT_BODY_FIELDS_V2: Final = _ANCHOR_COMMON_FIELDS_V2 | _fields(
    "schema_version live_runtime_anchor_request_id live_runtime_anchor_request_payload_sha256 "
    "previous_mapping committed_mapping cas_complete_binding cas_outcome checkpoint_sequence "
    "previous_live_runtime_anchor_checkpoint_id"
)
LIVE_RUNTIME_ANCHOR_HEAD_BODY_FIELDS_V2: Final = _ANCHOR_COMMON_FIELDS_V2 | _fields(
    "schema_version live_runtime_anchor_request_id live_runtime_anchor_request_payload_sha256 "
    "live_runtime_anchor_checkpoint_id live_runtime_anchor_checkpoint_payload_sha256 "
    "committed_mapping cas_complete_binding cas_outcome head_sequence "
    "previous_live_runtime_anchor_head_id"
)
LIVE_RUNTIME_ANCHOR_RECEIPT_BODY_FIELDS_V2: Final = _ANCHOR_COMMON_FIELDS_V2 | _fields(
    "schema_version live_runtime_anchor_request_id live_runtime_anchor_request_payload_sha256 "
    "live_runtime_anchor_checkpoint_id live_runtime_anchor_checkpoint_payload_sha256 "
    "live_runtime_anchor_head_id live_runtime_anchor_head_payload_sha256 previous_mapping "
    "committed_mapping cas_complete_binding cas_outcome compare_matched append_applied "
    "oob_high_water_checkpoint_id oob_high_water_head_id durable_before_return"
)

CAS_FORBIDDEN_IN_ALL_SIX_LIVE_BODY_FIELDS_V2: Final = _fields(
    "retention_plan_id retention_plan_payload_sha256 retention_session_id cas_chain_id "
    "cas_request_id cas_object_id cas_signed_receipt_payload_sha256 cas_checkpoint_payload_sha256 "
    "cas_head_state_id cas_head_payload_sha256 cas_response_payload_sha256"
)
EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2: Final = (
    "out_of_band_expected_lock_id_bytes",
    "out_of_band_expected_lock_payload_sha256_digest",
    "out_of_band_expected_bootstrap_external_witness_policy_id_bytes",
    "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest",
    "out_of_band_expected_online_witness_policy_id_bytes",
    "out_of_band_expected_online_witness_policy_payload_sha256_digest",
    "out_of_band_expected_durable_anchor_policy_id_bytes",
    "out_of_band_expected_durable_anchor_policy_payload_sha256_digest",
    "out_of_band_expected_runtime_closure_policy_id_bytes",
    "out_of_band_expected_runtime_closure_policy_payload_sha256_digest",
    "out_of_band_expected_runtime_closure_evidence_id_bytes",
    "out_of_band_expected_runtime_closure_evidence_payload_sha256_digest",
    "out_of_band_expected_custody_attestation_policy_id_bytes",
    "out_of_band_expected_custody_attestation_policy_payload_sha256_digest",
    "out_of_band_expected_live_runtime_policy_id_bytes",
    "out_of_band_expected_live_runtime_policy_payload_sha256_digest",
    "out_of_band_expected_live_runtime_anchor_policy_id_bytes",
    "out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest",
    "expected_attempt_context_id_bytes",
)
EXPECTED_PIN_NAMES_V2: Final = frozenset(EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2)

_SCHEMA_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": LIVE_RUNTIME_POLICY_SCHEMA_V2,
        "custody_attestation_policy": CUSTODY_ATTESTATION_POLICY_SCHEMA_V2,
        "live_runtime_challenge": LIVE_RUNTIME_CHALLENGE_SCHEMA_V2,
        "live_runtime_response": LIVE_RUNTIME_RESPONSE_SCHEMA_V2,
        "live_runtime_report": LIVE_RUNTIME_REPORT_SCHEMA_V2,
        "live_runtime_custody_attestation": LIVE_RUNTIME_CUSTODY_ATTESTATION_SCHEMA_V2,
        "live_runtime_result": LIVE_RUNTIME_RESULT_SCHEMA_V2,
        "live_runtime_anchor_policy": LIVE_RUNTIME_ANCHOR_POLICY_SCHEMA_V2,
        "live_runtime_anchor_request": LIVE_RUNTIME_ANCHOR_REQUEST_SCHEMA_V2,
        "live_runtime_anchor_checkpoint": LIVE_RUNTIME_ANCHOR_CHECKPOINT_SCHEMA_V2,
        "live_runtime_anchor_head": LIVE_RUNTIME_ANCHOR_HEAD_SCHEMA_V2,
        "live_runtime_anchor_receipt": LIVE_RUNTIME_ANCHOR_RECEIPT_SCHEMA_V2,
        "process_snapshot": PROCESS_SNAPSHOT_SCHEMA_V2,
    }
)
_PREFIX_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": "phase2b_step5_live_runtime_policy_v2_",
        "custody_attestation_policy": "phase2b_step5_live_runtime_custody_attestation_policy_v2_",
        "live_runtime_challenge": "phase2b_step5_live_runtime_challenge_v2_",
        "live_runtime_response": "phase2b_step5_live_runtime_response_v2_",
        "live_runtime_report": "phase2b_step5_live_runtime_report_v2_",
        "live_runtime_custody_attestation": "phase2b_step5_live_runtime_custody_attestation_v2_",
        "live_runtime_result": "phase2b_step5_live_runtime_result_v2_",
        "live_runtime_anchor_policy": "phase2b_step5_live_runtime_anchor_policy_v2_",
        "live_runtime_anchor_request": "phase2b_step5_live_runtime_anchor_request_v2_",
        "live_runtime_anchor_checkpoint": "phase2b_step5_live_runtime_anchor_checkpoint_v2_",
        "live_runtime_anchor_head": "phase2b_step5_live_runtime_anchor_head_v2_",
        "live_runtime_anchor_receipt": "phase2b_step5_live_runtime_anchor_receipt_v2_",
        "process_snapshot": "phase2b_step5_live_runtime_process_snapshot_v2_",
    }
)
_CONTENT_ID_SEGMENT_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": "POLICY",
        "custody_attestation_policy": "CUSTODY_ATTESTATION_POLICY",
        "live_runtime_challenge": "CHALLENGE",
        "live_runtime_response": "RESPONSE",
        "live_runtime_report": "REPORT",
        "live_runtime_custody_attestation": "CUSTODY_ATTESTATION",
        "live_runtime_result": "RESULT",
        "live_runtime_anchor_policy": "ANCHOR_POLICY",
        "live_runtime_anchor_request": "ANCHOR_REQUEST",
        "live_runtime_anchor_checkpoint": "ANCHOR_CHECKPOINT",
        "live_runtime_anchor_head": "ANCHOR_HEAD",
        "live_runtime_anchor_receipt": "ANCHOR_RECEIPT",
        "process_snapshot": "PROCESS_SNAPSHOT",
    }
)
_CONTENT_ID_DOMAIN_BY_KIND: Final = MappingProxyType(
    {
        kind: (
            "HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/" + segment + "/ID/V2\0"
        ).encode("ascii")
        for kind, segment in _CONTENT_ID_SEGMENT_BY_KIND.items()
    }
)
SESSION_ID_PREFIX_V2: Final = "phase2b_step5_live_runtime_session_v2_"
SESSION_ID_DOMAIN_V2: Final = b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SESSION/ID/V2\0"
CAS_CHAIN_ID_PREFIX_V2: Final = "phase2b_step5_live_runtime_cas_chain_v2_"
CAS_CHAIN_ID_DOMAIN_V2: Final = b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/CAS_CHAIN/ID/V2\0"

_EXTERNAL_SIGNATURE_DOMAIN_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/POLICY/BOOTSTRAP/V2\0",
        "custody_attestation_policy": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/CUSTODY_ATTESTATION_POLICY/BOOTSTRAP/V2\0",
        "live_runtime_response": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/RESPONSE/ONLINE_WITNESS/V2\0",
        "live_runtime_custody_attestation": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/CUSTODY_ATTESTATION/ATTESTOR/V2\0",
        "live_runtime_anchor_policy": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/ANCHOR_POLICY/A/V2\0",
        "live_runtime_anchor_checkpoint": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/ANCHOR_CHECKPOINT/A/V2\0",
        "live_runtime_anchor_head": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/ANCHOR_HEAD/A/V2\0",
        "live_runtime_anchor_receipt": b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/SIGN/ANCHOR_RECEIPT/A/V2\0",
    }
)
_TOP_FIELDS_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": LIVE_RUNTIME_POLICY_FIELDS_V2,
        "custody_attestation_policy": CUSTODY_ATTESTATION_POLICY_FIELDS_V2,
        "live_runtime_challenge": LIVE_RUNTIME_CHALLENGE_FIELDS_V2,
        "live_runtime_response": LIVE_RUNTIME_RESPONSE_FIELDS_V2,
        "live_runtime_report": LIVE_RUNTIME_REPORT_FIELDS_V2,
        "live_runtime_custody_attestation": LIVE_RUNTIME_CUSTODY_ATTESTATION_FIELDS_V2,
        "live_runtime_result": LIVE_RUNTIME_RESULT_FIELDS_V2,
        "live_runtime_anchor_policy": LIVE_RUNTIME_ANCHOR_POLICY_FIELDS_V2,
        "live_runtime_anchor_request": LIVE_RUNTIME_ANCHOR_REQUEST_FIELDS_V2,
        "live_runtime_anchor_checkpoint": LIVE_RUNTIME_ANCHOR_CHECKPOINT_FIELDS_V2,
        "live_runtime_anchor_head": LIVE_RUNTIME_ANCHOR_HEAD_FIELDS_V2,
        "live_runtime_anchor_receipt": LIVE_RUNTIME_ANCHOR_RECEIPT_FIELDS_V2,
    }
)
_BODY_FIELDS_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": LIVE_RUNTIME_POLICY_BODY_FIELDS_V2,
        "custody_attestation_policy": CUSTODY_ATTESTATION_POLICY_BODY_FIELDS_V2,
        "live_runtime_challenge": LIVE_RUNTIME_CHALLENGE_BODY_FIELDS_V2,
        "live_runtime_response": LIVE_RUNTIME_RESPONSE_BODY_FIELDS_V2,
        "live_runtime_report": LIVE_RUNTIME_REPORT_BODY_FIELDS_V2,
        "live_runtime_custody_attestation": LIVE_RUNTIME_CUSTODY_ATTESTATION_BODY_FIELDS_V2,
        "live_runtime_result": LIVE_RUNTIME_RESULT_BODY_FIELDS_V2,
        "live_runtime_anchor_policy": LIVE_RUNTIME_ANCHOR_POLICY_BODY_FIELDS_V2,
        "live_runtime_anchor_request": LIVE_RUNTIME_ANCHOR_REQUEST_BODY_FIELDS_V2,
        "live_runtime_anchor_checkpoint": LIVE_RUNTIME_ANCHOR_CHECKPOINT_BODY_FIELDS_V2,
        "live_runtime_anchor_head": LIVE_RUNTIME_ANCHOR_HEAD_BODY_FIELDS_V2,
        "live_runtime_anchor_receipt": LIVE_RUNTIME_ANCHOR_RECEIPT_BODY_FIELDS_V2,
    }
)
_ID_FIELD_BY_KIND: Final = MappingProxyType(
    {
        "live_runtime_policy": "live_runtime_policy_id",
        "custody_attestation_policy": "custody_attestation_policy_id",
        "live_runtime_challenge": "live_runtime_challenge_id",
        "live_runtime_response": "live_runtime_response_id",
        "live_runtime_report": "live_runtime_report_id",
        "live_runtime_custody_attestation": "live_runtime_custody_attestation_id",
        "live_runtime_result": "live_runtime_result_id",
        "live_runtime_anchor_policy": "live_runtime_anchor_policy_id",
        "live_runtime_anchor_request": "live_runtime_anchor_request_id",
        "live_runtime_anchor_checkpoint": "live_runtime_anchor_checkpoint_id",
        "live_runtime_anchor_head": "live_runtime_anchor_head_id",
        "live_runtime_anchor_receipt": "live_runtime_anchor_receipt_id",
    }
)

ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2: Final = (
    "accepted_socket_inode",
    "server_socket_inode",
    "peer_pid",
    "peer_uid",
    "peer_gid",
    "host_boot_id",
    "process_start_ticks",
    "user_namespace_inode",
    "pid_namespace_inode",
    "mount_namespace_inode",
    "exe_device",
    "exe_inode",
    "exe_sha256",
    "cgroup_sha256",
)


def _canonical(value: object) -> bytes:
    try:
        return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
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
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
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
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return left == right


def _sha(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return _SHA256_V2(payload).hexdigest()


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


def _content_identifier(value: object, prefix: str, label: str) -> str:
    text = _text(value, label)
    if not text.startswith(prefix) or _HEX64_RE.fullmatch(text[len(prefix):]) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} content-ID syntax differs")
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


def _content_id(kind: str, body: Mapping[str, object]) -> str:
    return _PREFIX_BY_KIND[kind] + _SHA256_V2(
        _CONTENT_ID_DOMAIN_BY_KIND[kind] + _canonical(dict(body))
    ).hexdigest()


def _require_raw_pin(value: bytes, expected: str, label: str) -> None:
    if type(value) is not bytes:
        _fail(FAIL_INPUT_V2, f"{label} pin must be exact bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeError as exc:
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_INPUT_V2, f"{label} pin is not ASCII"
        ) from exc
    if decoded != expected:
        _fail(FAIL_BINDING_V2, f"{label} out-of-band pin differs")


def _require_digest_pin(value: bytes, payload: bytes, label: str) -> None:
    if type(value) is not bytes or len(value) != 32:
        _fail(FAIL_INPUT_V2, f"{label} digest pin must be 32 exact bytes")
    if value != _SHA256_V2(payload).digest():
        _fail(FAIL_BINDING_V2, f"{label} payload digest pin differs")


def _policy_header(body: Mapping[str, object], schema: str, label: str) -> None:
    if (
        body["schema_version"] != schema
        or body["status"] != STATUS_V2
        or body["claim_level"] != CLAIM_LEVEL_V2
        or body["canonicalization"] != CANONICALIZATION_V2
        or _integer(body["policy_epoch"], f"{label} epoch", minimum=1) != 1
    ):
        _fail(FAIL_SCHEMA_V2, f"{label} header differs")


def _candidate_header(body: Mapping[str, object], schema: str, label: str) -> None:
    if (
        body["schema_version"] != schema
        or body["status"] != STATUS_V2
        or body["claim_level"] != CLAIM_LEVEL_V2
    ):
        _fail(FAIL_SCHEMA_V2, f"{label} header differs")


def _load_root(payload: bytes, kind: str) -> tuple[dict[str, object], dict[str, object]]:
    root = _object(_decode(payload, kind), _TOP_FIELDS_BY_KIND[kind], kind)
    body = _object(root["body"], _BODY_FIELDS_BY_KIND[kind], f"{kind} body")
    if root["schema_version"] != _SCHEMA_BY_KIND[kind] or body["schema_version"] != root["schema_version"]:
        _fail(FAIL_SCHEMA_V2, f"{kind} schema differs")
    id_field = _ID_FIELD_BY_KIND[kind]
    expected_id = _content_id(kind, body)
    _content_identifier(root[id_field], _PREFIX_BY_KIND[kind], f"{kind} ID")
    if root[id_field] != expected_id:
        _fail(FAIL_CONTENT_ID_V2, f"{kind} content ID differs")
    signatures = root["signatures_in_exact_order"]
    if type(signatures) is not list:
        _fail(FAIL_SCHEMA_V2, f"{kind} signatures must be one exact array")
    return root, body


def _external_signature_preimage(
    *, kind: str, body: Mapping[str, object], content_id: str,
    signer_key_id: str, key_epoch: int, purpose: str,
) -> bytes:
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-live-runtime-external-signature-preimage/2",
        "artifact_kind": kind,
        "content_id": content_id,
        "body": dict(body),
        "signer_key_id": signer_key_id,
        "key_epoch": key_epoch,
        "purpose": purpose,
    }
    if frozenset(subject) != EXTERNAL_SIGNATURE_PREIMAGE_FIELDS_V2:
        _fail(FAIL_SCHEMA_V2, "external signature preimage fields differ")
    return _EXTERNAL_SIGNATURE_DOMAIN_BY_KIND[kind] + _canonical(subject)


def _verify_ed25519(public_key_hex: str, signature_hex: str, preimage: bytes, label: str) -> None:
    _hex64(public_key_hex, f"{label} public key")
    if _HEX128_RE.fullmatch(signature_hex) is None:
        _fail(FAIL_SIGNATURE_V2, f"{label} signature encoding differs")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V2, "Ed25519 verification backend is unavailable")
    try:
        _Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature_hex), preimage
        )
    except _InvalidSignature as exc:
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_SIGNATURE_V2, f"{label} Ed25519 signature rejected"
        ) from exc


def _verify_external_signature(
    signature_value: object, *, kind: str, body: Mapping[str, object],
    content_id: str, key_id: str, key_epoch: int, public_key_hex: str,
    purpose: str, label: str,
) -> None:
    signature = _object(signature_value, EXTERNAL_SIGNATURE_FIELDS_V2, label)
    if (
        signature["schema_version"] != "hegel-machine-phase2b-step5-live-runtime-external-signature/2"
        or signature["algorithm"] != ALGORITHM_V2
        or signature["signer_key_id"] != key_id
        or signature["key_epoch"] != key_epoch
        or signature["purpose"] != purpose
    ):
        _fail(FAIL_SIGNATURE_V2, f"{label} metadata differs")
    _verify_ed25519(
        public_key_hex,
        _text(signature["signature_hex"], f"{label} signature"),
        _external_signature_preimage(
            kind=kind, body=body, content_id=content_id,
            signer_key_id=key_id, key_epoch=key_epoch, purpose=purpose,
        ),
        label,
    )


def _verify_authority_signature(
    signature_value: object, *, kind: str, body: Mapping[str, object],
    role_row: Mapping[str, object], purpose: str, label: str,
    _preimage=authority.phase2b_step5_signature_preimage_v2,
) -> None:
    signature = _object(
        signature_value, AUTHORITY_DETACHED_SIGNATURE_FIELDS_V2, label
    )
    if (
        signature["schema_version"] != _AUTH_DETACHED_SIGNATURE_SCHEMA_V2
        or signature["algorithm"] != ALGORITHM_V2
        or signature["role"] != role_row["role"]
        or signature["purpose"] != purpose
        or signature["key_epoch"] != role_row["key_epoch"]
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
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_DEPENDENCY_V2, f"{label} authority preimage failed"
        ) from exc
    _verify_ed25519(
        _text(role_row["public_key_hex"], f"{label} public key"),
        _text(signature["signature_hex"], f"{label} signature"),
        preimage,
        label,
    )


def _sequence(value: object, label: str, *, nonempty: bool = False) -> tuple[object, ...]:
    if type(value) is not list or (nonempty and not value):
        _fail(FAIL_SCHEMA_V2, f"{label} must be one exact array")
    return tuple(value)


def _string_sequence(value: object, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    raw = _sequence(value, label, nonempty=nonempty)
    if any(type(item) is not str for item in raw):
        _fail(FAIL_SCHEMA_V2, f"{label} contains a non-string")
    return raw  # type: ignore[return-value]


def _all_false(value: object, label: str) -> dict[str, object]:
    body = _object(value, POLICY_ALONE_NONCLAIMS_FIELDS_V2, label)
    if any(item is not False for item in body.values()):
        _fail(FAIL_CLAIM_BOUNDARY_V2, f"{label} grants a claim")
    return body


def _upstream_object(payload: bytes, label: str) -> dict[str, object]:
    value = _decode(payload, label)
    if type(value) is not dict:
        _fail(FAIL_SCHEMA_V2, f"{label} root must be one exact object")
    return value


def _load_authority_context(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    _load_lock=authority.load_step5_stage_b_trust_lock_v2,
    _load_profile=authority.load_step5_authority_profile_v2,
    _load_registry=authority.load_step5_signer_registry_v2,
    _require_graph=authority.require_locked_step5_authority_graph_v2,
) -> dict[str, object]:
    try:
        lock = _load_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        profile = _load_profile(authority_profile_payload, lock)
        registry = _load_registry(signer_registry_payload, lock, profile)
        lock, profile, registry = _require_graph(lock, profile, registry)
    except Exception as exc:
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_DEPENDENCY_V2, "raw authority graph rejected"
        ) from exc
    rows_raw = registry.body["role_keys_in_exact_order"]
    services_raw = profile.body["service_static_bindings_in_exact_order"]
    if type(rows_raw) is not tuple or len(rows_raw) != len(_AUTH_ROLE_ORDER_V2):
        _fail(FAIL_BINDING_V2, "authority role-key order differs")
    if type(services_raw) is not tuple or len(services_raw) != len(_AUTH_SERVICE_ORDER_V2):
        _fail(FAIL_BINDING_V2, "authority service order differs")
    role_rows: dict[str, Mapping[str, object]] = {}
    for role, row in zip(_AUTH_ROLE_ORDER_V2, rows_raw, strict=True):
        if not isinstance(row, Mapping) or row.get("role") != role:
            _fail(FAIL_BINDING_V2, "authority role row differs")
        role_rows[role] = row
    service_rows: dict[str, Mapping[str, object]] = {}
    for name, row in zip(_AUTH_SERVICE_ORDER_V2, services_raw, strict=True):
        if not isinstance(row, Mapping) or row.get("service_name") != name:
            _fail(FAIL_BINDING_V2, "authority service row differs")
        service_rows[name] = row
    return {
        "lock": lock,
        "profile": profile,
        "registry": registry,
        "roles": MappingProxyType(role_rows),
        "services": MappingProxyType(service_rows),
    }


def _runtime_identity(value: object) -> dict[str, object]:
    body = _object(value, RUNTIME_IDENTITY_FIELDS_V2, "runtime identity")
    for field in RUNTIME_IDENTITY_FIELDS_V2:
        _text(body[field], f"runtime identity {field}")
    return body


def _mapping(value: object, label: str) -> dict[str, object]:
    body = _object(value, LIVE_RUNTIME_ANCHOR_MAPPING_FIELDS_V2, label)
    if body["schema_version"] != LIVE_RUNTIME_ANCHOR_MAPPING_SCHEMA_V2:
        _fail(FAIL_SCHEMA_V2, f"{label} schema differs")
    _identifier(body["attempt_context_id"], f"{label} attempt")
    _identifier(body["session_id"], f"{label} session")
    _integer(body["session_sequence"], f"{label} session sequence")
    _content_identifier(
        body["admission_candidate_result_id"],
        _ADMISSION_RESULT_PREFIX_V1,
        f"{label} admission result ID",
    )
    _hex64(body["admission_candidate_result_payload_sha256"], f"{label} admission result SHA")
    _content_identifier(body["live_runtime_result_id"], _PREFIX_BY_KIND["live_runtime_result"], f"{label} live result ID")
    _hex64(body["live_runtime_result_payload_sha256"], f"{label} live result SHA")
    _content_identifier(body["retention_plan_id"], _STORE_PLAN_ID_PREFIX_V1, f"{label} plan ID")
    _hex64(body["retention_plan_payload_sha256"], f"{label} plan SHA")
    _identifier(body["retention_session_id"], f"{label} retention session")
    _identifier(body["chain_id"], f"{label} chain")
    _content_identifier(body["cas_complete_head_state_id"], _STORE_HEAD_STATE_ID_PREFIX_V2, f"{label} CAS head-state ID")
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


def _process_snapshot(value: object, *, expected_phase: str) -> dict[str, object]:
    body = _object(value, PROCESS_SNAPSHOT_FIELDS_V2, f"{expected_phase} process snapshot")
    if (
        body["schema_version"] != PROCESS_SNAPSHOT_SCHEMA_V2
        or body["phase"] != expected_phase
        or body["observation_source"] != "KERNEL_SO_PEERCRED_PIDFD_PROCFS"
        or body["address_family"] != "AF_UNIX"
        or body["socket_type"] != "SOCK_SEQPACKET"
        or body["peer_credential_option"] != "SO_PEERCRED"
        or body["peer_credential_struct_format"] != "=iii"
        or body["peer_credential_struct_size_bytes"] != 12
    ):
        _fail(FAIL_SCHEMA_V2, f"{expected_phase} snapshot transport differs")
    _hex64(body["so_peercred_payload_sha256"], "SO_PEERCRED payload SHA")
    for field in (
        "accepted_socket_inode", "server_socket_inode", "peer_pid", "peer_uid",
        "peer_gid", "process_start_ticks", "user_namespace_inode",
        "pid_namespace_inode", "mount_namespace_inode", "exe_device", "exe_inode",
    ):
        _integer(body[field], f"snapshot {field}", minimum=1 if field == "peer_pid" else 0)
    if any(body[field] > 0x7FFF_FFFF for field in ("peer_pid", "peer_uid", "peer_gid")):
        _fail(FAIL_SCHEMA_V2, f"{expected_phase} SO_PEERCRED signed-int range differs")
    peer_credential_payload = _STRUCT_PACK_V2(
        "=iii", body["peer_pid"], body["peer_uid"], body["peer_gid"]
    )
    if body["so_peercred_payload_sha256"] != _SHA256_V2(
        peer_credential_payload
    ).hexdigest():
        _fail(FAIL_BINDING_V2, f"{expected_phase} SO_PEERCRED payload digest differs")
    _identifier(body["host_boot_id"], "snapshot host boot ID")
    _hex64(body["exe_sha256"], "snapshot executable SHA")
    _hex64(body["cgroup_sha256"], "snapshot cgroup SHA")
    for field in (
        "pidfd_open_succeeded", "pidfd_alive", "procfs_read_succeeded",
        "observation_complete",
    ):
        if _boolean(body[field], f"snapshot {field}") is not True:
            _fail(FAIL_BINDING_V2, f"snapshot {field} is not true")
    subject = dict(body)
    del subject["process_snapshot_id"]
    expected_id = _PREFIX_BY_KIND["process_snapshot"] + _SHA256_V2(
        _CONTENT_ID_DOMAIN_BY_KIND["process_snapshot"] + _canonical(subject)
    ).hexdigest()
    if body["process_snapshot_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V2, f"{expected_phase} snapshot ID differs")
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
        "bootstrap_external_witness_policy_id": context["bootstrap_id"],
        "online_witness_policy_id": context["online_id"],
        "durable_anchor_policy_id": context["durable_id"],
        "runtime_closure_policy_id": context["closure_policy_id"],
        "runtime_closure_evidence_id": context["closure_evidence_id"],
        "custody_attestation_policy_id": context["custody_policy_id"],
        "live_runtime_policy_id": context["live_policy_id"],
        "live_runtime_anchor_policy_id": context["anchor_policy_id"],
        "admission_candidate_result_id": context["admission_result_id"],
        "admission_candidate_result_payload_sha256": context["admission_result_sha"],
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, f"{label} session lineage differs")


def _verify_signature_order(
    root: Mapping[str, object], *, kind: str, body: Mapping[str, object],
    signers: tuple[tuple[str, Mapping[str, object] | None, str, str, int, str], ...],
) -> None:
    signatures = root["signatures_in_exact_order"]
    if type(signatures) is not list or len(signatures) != len(signers):
        _fail(FAIL_SIGNATURE_V2, f"{kind} signature order/cardinality differs")
    content_id = root[_ID_FIELD_BY_KIND[kind]]
    for index, (mode, role_row, purpose, key_id, key_epoch, public_key_hex) in enumerate(signers):
        label = f"{kind} signature {index}"
        if mode == "authority":
            if role_row is None:
                _fail(FAIL_SIGNATURE_V2, f"{label} authority row is absent")
            _verify_authority_signature(
                signatures[index], kind=kind, body=body,
                role_row=role_row, purpose=purpose, label=label,
            )
        elif mode == "external":
            _verify_external_signature(
                signatures[index], kind=kind, body=body,
                content_id=content_id, key_id=key_id, key_epoch=key_epoch,
                public_key_hex=public_key_hex, purpose=purpose, label=label,
            )
        else:
            _fail(FAIL_SIGNATURE_V2, f"{label} signer mode differs")


def _require_policy_refs(
    body: Mapping[str, object], *, refs: Mapping[str, object], label: str
) -> None:
    fields = (
        "stage_b_trust_lock_id", "stage_b_trust_lock_payload_sha256",
        "authority_profile_id", "authority_profile_payload_sha256",
        "signer_registry_id", "signer_registry_payload_sha256",
        "attempt_context_id", "admission_candidate_result_id",
        "admission_candidate_result_payload_sha256",
        "bootstrap_external_witness_policy_id",
        "bootstrap_external_witness_policy_payload_sha256",
        "online_witness_policy_id", "online_witness_policy_payload_sha256",
        "durable_anchor_policy_id", "durable_anchor_policy_payload_sha256",
    )
    for field in fields:
        if field in body and not _exact_equal(body[field], refs[field]):
            _fail(FAIL_BINDING_V2, f"{label} {field} differs")


def _load_custody_policy(
    payload: bytes, *, refs: Mapping[str, object], bootstrap: Mapping[str, object],
    online: Mapping[str, object], oob_id: bytes, oob_digest: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "custody_attestation_policy")
    _policy_header(body, CUSTODY_ATTESTATION_POLICY_SCHEMA_V2, "custody policy")
    _require_policy_refs(body, refs=refs, label="custody policy")
    _require_raw_pin(oob_id, root["custody_attestation_policy_id"], "custody policy ID")
    _require_digest_pin(oob_digest, payload, "custody policy")
    if (
        root["custody_attestation_policy_id"]
        != online["custody_attestation_policy_id"]
        or _sha(payload) != online["custody_attestation_policy_payload_sha256"]
    ):
        _fail(FAIL_BINDING_V2, "custody policy differs from online policy pins")
    expected = {
        "online_witness_key_id": online["online_key_id"],
        "online_witness_key_epoch": online["online_key_epoch"],
        "online_witness_public_key_hex": online["online_public_key_hex"],
        "custody_attestor_key_id": online["custody_attestor_key_id"],
        "custody_attestor_key_epoch": online["custody_attestor_key_epoch"],
        "custody_attestor_public_key_hex": online["custody_attestor_public_key_hex"],
        "custody_attestor_signature_purpose": ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2,
        "challenge_nonce_size_bytes": 32,
        "report_binding_required": True,
        "online_key_binding_required": True,
        "online_key_generated_in_attested_boundary_required": True,
        "online_key_nonexportable_required": True,
        "certificate_revocation_check_required": True,
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "custody policy key/binding requirements differ")
    _identifier(body["attestation_evidence_format"], "attestation evidence format")
    _string_sequence(body["trusted_certificate_root_ids_in_exact_order"], "trusted roots", nonempty=True)
    _string_sequence(body["allowed_certificate_signature_algorithms_in_exact_order"], "certificate algorithms", nonempty=True)
    _integer(body["minimum_attestation_security_version"], "attestation security version", minimum=1)
    _integer(body["maximum_attestation_evidence_bytes"], "attestation maximum bytes", minimum=1)
    _all_false(body["policy_alone_nonclaims"], "custody policy nonclaims")
    signatures = root["signatures_in_exact_order"]
    _verify_signature_order(
        root, kind="custody_attestation_policy", body=body,
        signers=((
            "external", None, SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V2,
            _text(bootstrap["witness_key_id"], "bootstrap key ID"),
            _integer(bootstrap["witness_key_epoch"], "bootstrap key epoch", minimum=1),
            _text(bootstrap["public_key_hex"], "bootstrap public key"),
        ),),
    )
    if len(signatures) != 1:
        _fail(FAIL_SIGNATURE_V2, "custody policy signature count differs")
    return root, body


def _require_registry_policy_fields(
    body: Mapping[str, object], *, auth: Mapping[str, object]
) -> None:
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
        "source_runtime_custodian_role": "SOURCE_RUNTIME_CUSTODIAN",
        "source_runtime_custodian_control_domain": source["control_domain"],
        "source_runtime_custodian_service_binding_id": runner["binding_id"],
        "source_runtime_custodian_service_uid": runner["uid"],
        "source_runtime_custodian_service_gid": runner["gid"],
        "source_runtime_custodian_key_id": source["key_id"],
        "source_runtime_custodian_key_epoch": source["key_epoch"],
        "source_runtime_custodian_public_key_hex": source["public_key_hex"],
        "source_runtime_custodian_authorized_signature_purposes_in_exact_order": list(source["authorized_signature_purposes"]),
        "pure_replayer_role": "PURE_REPLAYER",
        "pure_replayer_control_domain": pure["control_domain"],
        "pure_replayer_service_binding_id": pure_service["binding_id"],
        "pure_replayer_service_uid": pure_service["uid"],
        "pure_replayer_service_gid": pure_service["gid"],
        "pure_replayer_key_id": pure["key_id"],
        "pure_replayer_key_epoch": pure["key_epoch"],
        "pure_replayer_public_key_hex": pure["public_key_hex"],
        "pure_replayer_authorized_signature_purposes_in_exact_order": list(pure["authorized_signature_purposes"]),
        "cas_store_role": "CAS_STORE",
        "cas_store_control_domain": cas["control_domain"],
        "cas_store_service_binding_id": cas_service["binding_id"],
        "cas_store_service_uid": cas_service["uid"],
        "cas_store_service_gid": cas_service["gid"],
        "cas_store_key_id": cas["key_id"],
        "cas_store_key_epoch": cas["key_epoch"],
        "cas_store_public_key_hex": cas["public_key_hex"],
        "cas_store_authorized_signature_purposes_in_exact_order": list(cas["authorized_signature_purposes"]),
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "live policy authority role/service binding differs")


def _load_live_policy(
    payload: bytes, *, refs: Mapping[str, object], bootstrap: Mapping[str, object],
    online: Mapping[str, object], auth: Mapping[str, object],
    closure_result: object, closure_policy_body: Mapping[str, object],
    custody_policy_root: Mapping[str, object], custody_policy_payload: bytes,
    oob_id: bytes, oob_digest: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_policy")
    _policy_header(body, LIVE_RUNTIME_POLICY_SCHEMA_V2, "live runtime policy")
    _require_policy_refs(body, refs=refs, label="live runtime policy")
    _require_raw_pin(oob_id, root["live_runtime_policy_id"], "live runtime policy ID")
    _require_digest_pin(oob_digest, payload, "live runtime policy")
    if (
        body["runtime_closure_policy_id"] != closure_result.closure_policy_id
        or body["runtime_closure_policy_payload_sha256"]
        != closure_result.closure_policy_payload_sha256
        or body["runtime_closure_evidence_id"] != closure_result.evidence_id
        or body["runtime_closure_evidence_payload_sha256"] != refs["runtime_closure_evidence_payload_sha256"]
        or body["runtime_closure_service_name"] != closure_result.service_name
        or not _exact_equal(_runtime_identity(body["runtime_identity"]), closure_policy_body["runtime_identity"])
        or body["custody_attestation_policy_id"] != custody_policy_root["custody_attestation_policy_id"]
        or body["custody_attestation_policy_payload_sha256"] != _sha(custody_policy_payload)
    ):
        _fail(FAIL_BINDING_V2, "live policy closure/custody binding differs")
    expected = {
        "bootstrap_key_id": bootstrap["witness_key_id"],
        "bootstrap_key_epoch": bootstrap["witness_key_epoch"],
        "bootstrap_public_key_hex": bootstrap["public_key_hex"],
        "bootstrap_authorized_signature_purposes_in_exact_order": list(_ADMISSION_BOOTSTRAP_PURPOSES_V1),
        "online_witness_key_id": online["online_key_id"],
        "online_witness_key_epoch": online["online_key_epoch"],
        "online_witness_public_key_hex": online["online_public_key_hex"],
        "online_witness_authorized_signature_purposes_in_exact_order": list(LIVE_OVERLAY_ONLINE_PURPOSES_IN_EXACT_ORDER_V2),
        "online_witness_service_binding_id": online["service_binding_id"],
        "online_witness_control_domain_id": online["witness_control_domain_id"],
        "online_witness_service_uid": online["service_uid"],
        "online_witness_service_gid": online["service_gid"],
        "online_witness_service_user_namespace_inode": online["service_user_namespace_inode"],
        "custody_attestor_key_id": online["custody_attestor_key_id"],
        "custody_attestor_key_epoch": online["custody_attestor_key_epoch"],
        "custody_attestor_public_key_hex": online["custody_attestor_public_key_hex"],
        "custody_attestor_authorized_signature_purposes_in_exact_order": list(CUSTODY_PURPOSES_IN_EXACT_ORDER_V2),
        "online_key_nonexportable_required": True,
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
        "cas_artifact_kinds_in_exact_order": list(CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2),
        "cas_complete_head_required": True,
        "cas_durable_before_return_required": True,
        "cas_readback_required": True,
        "anchor_durable_before_return_required": True,
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "live policy exact mechanics/key binding differs")
    for field in (
        "independent_custody_attestor_required",
        "cas_complete_head_required",
        "cas_durable_before_return_required",
        "cas_readback_required",
        "anchor_durable_before_return_required",
    ):
        if body[field] is not True:
            _fail(FAIL_BINDING_V2, f"live policy {field} is not true")
    _hex64(body["online_witness_expected_cgroup_sha256"], "expected online cgroup SHA")
    _identifier(body["custody_attestation_profile"], "custody attestation profile")
    _identifier(body["live_runtime_anchor_control_domain_id"], "live anchor control domain")
    _identifier(body["cas_chain_id_profile"], "CAS chain profile")
    _identifier(body["anchor_slot_key_profile"], "anchor slot profile")
    _identifier(body["anchor_compare_and_append_semantics"], "anchor CAS semantics")
    _identifier(body["monotonic_clock"], "monotonic clock")
    _integer(body["maximum_packet_bytes"], "maximum packet bytes", minimum=1)
    _integer(body["maximum_round_trip_nanoseconds"], "maximum round trip", minimum=1)
    _all_false(body["policy_alone_nonclaims"], "live policy nonclaims")
    _require_registry_policy_fields(body, auth=auth)
    roles = auth["roles"]
    _verify_signature_order(
        root, kind="live_runtime_policy", body=body,
        signers=(
            (
                "external", None, SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V2,
                _text(bootstrap["witness_key_id"], "bootstrap key ID"),
                _integer(bootstrap["witness_key_epoch"], "bootstrap key epoch", minimum=1),
                _text(bootstrap["public_key_hex"], "bootstrap public key"),
            ),
            (
                "authority", roles["SOURCE_RUNTIME_CUSTODIAN"],
                SIGN_RUNTIME_CLOSURE_PURPOSE_V2, "", 0, "",
            ),
        ),
    )
    return root, body


def _genesis_mapping(
    *, attempt_id: str, admission_result_id: str, admission_result_sha: str
) -> dict[str, object]:
    zeros = "0" * 64
    return {
        "schema_version": LIVE_RUNTIME_ANCHOR_MAPPING_SCHEMA_V2,
        "attempt_context_id": attempt_id,
        "session_id": "GENESIS",
        "session_sequence": 0,
        "admission_candidate_result_id": admission_result_id,
        "admission_candidate_result_payload_sha256": admission_result_sha,
        "live_runtime_result_id": _PREFIX_BY_KIND["live_runtime_result"] + zeros,
        "live_runtime_result_payload_sha256": zeros,
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


def _load_anchor_policy(
    payload: bytes, *, refs: Mapping[str, object], live_policy_root: Mapping[str, object],
    live_policy_payload: bytes, durable: Mapping[str, object], auth: Mapping[str, object],
    oob_id: bytes, oob_digest: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_anchor_policy")
    _policy_header(body, LIVE_RUNTIME_ANCHOR_POLICY_SCHEMA_V2, "live anchor policy")
    _require_policy_refs(body, refs=refs, label="live anchor policy")
    _require_raw_pin(oob_id, root["live_runtime_anchor_policy_id"], "live anchor policy ID")
    _require_digest_pin(oob_digest, payload, "live anchor policy")
    source = auth["roles"]["SOURCE_RUNTIME_CUSTODIAN"]
    expected = {
        "live_runtime_policy_id": live_policy_root["live_runtime_policy_id"],
        "live_runtime_policy_payload_sha256": _sha(live_policy_payload),
        "anchor_namespace": durable["anchor_namespace"],
        "anchor_control_domain_id": durable["anchor_control_domain_id"],
        "anchor_key_id": durable["anchor_key_id"],
        "anchor_key_epoch": durable["anchor_key_epoch"],
        "anchor_public_key_hex": durable["anchor_public_key_hex"],
        "algorithm": ALGORITHM_V2,
        "authorized_signature_purposes_in_exact_order": list(ANCHOR_PURPOSES_IN_EXACT_ORDER_V2),
        "requester_role": "SOURCE_RUNTIME_CUSTODIAN",
        "requester_control_domain": source["control_domain"],
        "requester_key_id": source["key_id"],
        "requester_key_epoch": source["key_epoch"],
        "requester_public_key_hex": source["public_key_hex"],
        "requester_signature_purpose": SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
        "admission_slot_key_profile": durable["admission_slot_key_profile"],
        "compare_and_append_semantics": durable["compare_and_append_semantics"],
        "durable_before_return_required": True,
        "cas_complete_head_required": True,
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "live anchor policy key/ref binding differs")
    for field in (
        "anchor_namespace", "anchor_service_binding_id", "admission_slot_key_profile",
        "compare_and_append_semantics",
    ):
        _identifier(body[field], f"live anchor policy {field}")
    _content_identifier(
        body["genesis_checkpoint_id"], _PREFIX_BY_KIND["live_runtime_anchor_checkpoint"],
        "live anchor genesis checkpoint ID",
    )
    _content_identifier(
        body["genesis_head_id"], _PREFIX_BY_KIND["live_runtime_anchor_head"],
        "live anchor genesis head ID",
    )
    _content_identifier(
        body["oob_high_water_checkpoint_id"],
        _PREFIX_BY_KIND["live_runtime_anchor_checkpoint"],
        "live anchor OOB high-water checkpoint ID",
    )
    _content_identifier(
        body["oob_high_water_head_id"], _PREFIX_BY_KIND["live_runtime_anchor_head"],
        "live anchor OOB high-water head ID",
    )
    for field in ("anchor_service_uid", "anchor_service_gid", "anchor_service_user_namespace_inode"):
        _integer(body[field], f"live anchor policy {field}")
    for field in (
        "oob_high_water_checkpoint_payload_sha256",
        "oob_high_water_head_payload_sha256",
    ):
        _hex64(body[field], f"live anchor policy {field}")
    checkpoint_sequence = _integer(body["oob_high_water_checkpoint_sequence"], "A2 high-water checkpoint sequence")
    head_sequence = _integer(body["oob_high_water_head_sequence"], "A2 high-water head sequence")
    genesis = _genesis_mapping(
        attempt_id=refs["attempt_context_id"],
        admission_result_id=refs["admission_candidate_result_id"],
        admission_result_sha=refs["admission_candidate_result_payload_sha256"],
    )
    observed_genesis = _mapping(body["genesis_mapping"], "A2 genesis mapping")
    high_water = _mapping(body["oob_high_water_mapping"], "A2 high-water mapping")
    zero_digest = "0" * 64
    if (
        not _exact_equal(observed_genesis, genesis)
        or body["genesis_checkpoint_id"]
        != _PREFIX_BY_KIND["live_runtime_anchor_checkpoint"] + zero_digest
        or body["genesis_head_id"]
        != _PREFIX_BY_KIND["live_runtime_anchor_head"] + zero_digest
    ):
        _fail(FAIL_BINDING_V2, "A2 genesis mapping sentinel differs")
    if (
        checkpoint_sequence != head_sequence
        or high_water["anchor_sequence"] != head_sequence
        or high_water["session_sequence"] != head_sequence
        or high_water["attempt_context_id"] != refs["attempt_context_id"]
        or high_water["admission_candidate_result_id"]
        != refs["admission_candidate_result_id"]
        or high_water["admission_candidate_result_payload_sha256"]
        != refs["admission_candidate_result_payload_sha256"]
    ):
        _fail(FAIL_BINDING_V2, "A2 high-water sequences differ")
    if head_sequence == 0:
        if (
            not _exact_equal(high_water, genesis)
            or body["oob_high_water_checkpoint_id"] != body["genesis_checkpoint_id"]
            or body["oob_high_water_head_id"] != body["genesis_head_id"]
            or body["oob_high_water_checkpoint_payload_sha256"] != zero_digest
            or body["oob_high_water_head_payload_sha256"] != zero_digest
        ):
            _fail(FAIL_BINDING_V2, "initial A2 high-water state is not exact genesis")
    elif (
        high_water["session_id"] == "GENESIS"
        or high_water["effective_state"] != "COMMITTED"
        or high_water["superseded"] is not False
        or high_water["revoked"] is not False
    ):
        _fail(FAIL_BINDING_V2, "non-genesis A2 high-water mapping is not committed")
    _verify_signature_order(
        root, kind="live_runtime_anchor_policy", body=body,
        signers=((
            "external", None, SIGN_LIVE_RUNTIME_ANCHOR_POLICY_PURPOSE_V2,
            _text(body["anchor_key_id"], "A2 key ID"),
            _integer(body["anchor_key_epoch"], "A2 key epoch", minimum=1),
            _text(body["anchor_public_key_hex"], "A2 public key"),
        ),),
    )
    return root, body


def _session_id(
    *, attempt_context_id: str, online_witness_key_id: str,
    session_sequence: int, challenge_nonce: str,
) -> str:
    subject = {
        "attempt_context_id": attempt_context_id,
        "online_witness_key_id": online_witness_key_id,
        "session_sequence": session_sequence,
        "challenge_nonce": challenge_nonce,
    }
    return SESSION_ID_PREFIX_V2 + _SHA256_V2(
        SESSION_ID_DOMAIN_V2 + _canonical(subject)
    ).hexdigest()


def _load_challenge(
    payload: bytes, *, context: dict[str, object], live_policy: Mapping[str, object],
    anchor_policy: Mapping[str, object], source: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_challenge")
    _candidate_header(body, LIVE_RUNTIME_CHALLENGE_SCHEMA_V2, "live challenge")
    nonce = _hex64(body["challenge_nonce"], "challenge nonce")
    sequence = _integer(body["session_sequence"], "session sequence", minimum=1)
    if sequence != anchor_policy["oob_high_water_head_sequence"] + 1:
        _fail(FAIL_BINDING_V2, "live challenge session sequence does not advance A2 high-water")
    expected_session = _session_id(
        attempt_context_id=context["attempt_id"],
        online_witness_key_id=live_policy["online_witness_key_id"],
        session_sequence=sequence,
        challenge_nonce=nonce,
    )
    context = dict(context)
    context["session_id"] = expected_session
    context["session_sequence"] = sequence
    _common_session(body, context=context, label="live challenge")
    if (
        body["challenge_nonce_size_bytes"] != live_policy["challenge_nonce_size_bytes"]
        or body["challenger_role"] != "SOURCE_RUNTIME_CUSTODIAN"
        or body["challenger_key_id"] != source["key_id"]
        or body["challenger_key_epoch"] != source["key_epoch"]
        or body["expected_responder_key_id"] != live_policy["online_witness_key_id"]
        or body["expected_responder_key_epoch"] != live_policy["online_witness_key_epoch"]
        or body["maximum_round_trip_nanoseconds"] != live_policy["maximum_round_trip_nanoseconds"]
        or body["one_shot_required"] is not True
        or body["challenge_outcome"] != "ISSUED"
    ):
        _fail(FAIL_BINDING_V2, "live challenge key/freshness binding differs")
    timestamp = _text(body["challenge_monotonic_nanoseconds_decimal"], "challenge monotonic timestamp")
    if _DECIMAL_RE.fullmatch(timestamp) is None:
        _fail(FAIL_SCHEMA_V2, "challenge monotonic timestamp syntax differs")
    entry = _process_snapshot(body["entry_process_snapshot"], expected_phase="ENTRY")
    if body["entry_process_snapshot_payload_sha256"] != _sha(_canonical(entry)):
        _fail(FAIL_BINDING_V2, "challenge entry snapshot digest differs")
    expected_mapping = _mapping(body["expected_previous_anchor_mapping"], "challenge expected mapping")
    if not _exact_equal(expected_mapping, anchor_policy["oob_high_water_mapping"]):
        _fail(FAIL_BINDING_V2, "challenge expected A2 mapping differs")
    _verify_signature_order(
        root, kind="live_runtime_challenge", body=body,
        signers=(("authority", source, SIGN_RUNTIME_CLOSURE_PURPOSE_V2, "", 0, ""),),
    )
    return root, body, context


def _load_response(
    payload: bytes, *, context: Mapping[str, object], challenge_root: Mapping[str, object],
    challenge_body: Mapping[str, object], live_policy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_response")
    _candidate_header(body, LIVE_RUNTIME_RESPONSE_SCHEMA_V2, "live response")
    _common_session(body, context=context, label="live response")
    entry = challenge_body["entry_process_snapshot"]
    expected = {
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "challenge_nonce": challenge_body["challenge_nonce"],
        "challenge_nonce_size_bytes": live_policy["challenge_nonce_size_bytes"],
        "responder_key_id": live_policy["online_witness_key_id"],
        "responder_key_epoch": live_policy["online_witness_key_epoch"],
        "response_nonce_size_bytes": live_policy["response_nonce_size_bytes"],
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": challenge_body["entry_process_snapshot_payload_sha256"],
        "response_outcome": "RESPONDED",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "live response challenge/key binding differs")
    _hex64(body["response_nonce"], "response nonce")
    if body["response_nonce"] == body["challenge_nonce"]:
        _fail(FAIL_BINDING_V2, "challenge and response nonces alias")
    _verify_signature_order(
        root, kind="live_runtime_response", body=body,
        signers=((
            "external", None, SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_PURPOSE_V2,
            _text(live_policy["online_witness_key_id"], "online key ID"),
            _integer(live_policy["online_witness_key_epoch"], "online key epoch", minimum=1),
            _text(live_policy["online_witness_public_key_hex"], "online public key"),
        ),),
    )
    return root, body


def _load_report(
    payload: bytes, *, context: Mapping[str, object], challenge_root: Mapping[str, object],
    challenge_body: Mapping[str, object], response_root: Mapping[str, object],
    response_body: Mapping[str, object], live_policy: Mapping[str, object],
    source: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_report")
    _candidate_header(body, LIVE_RUNTIME_REPORT_SCHEMA_V2, "live report")
    _common_session(body, context=context, label="live report")
    expected = {
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "live_runtime_response_id": response_root["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(_canonical(dict(response_root))),
        "challenge_nonce": challenge_body["challenge_nonce"],
        "response_nonce": response_body["response_nonce"],
        "entry_process_snapshot_payload_sha256": challenge_body["entry_process_snapshot_payload_sha256"],
        "entry_exit_identity_fields_in_exact_order": list(ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2),
        "maximum_round_trip_nanoseconds": live_policy["maximum_round_trip_nanoseconds"],
        "exact_one_challenge_packet_sent": True,
        "exact_one_response_packet_received": True,
        "response_message_flags": 0,
        "response_ancillary_item_count": 0,
        "message_truncation_absent": True,
        "control_truncation_absent": True,
        "unexpected_ancillary_absent": True,
        "so_peercred_verified": True,
        "pidfd_identity_verified": True,
        "procfs_identity_verified": True,
        "service_binding_verified": True,
        "runtime_closure_policy_binding_verified": True,
        "runtime_closure_evidence_binding_verified": True,
        "runtime_executable_binding_verified": True,
        "distinct_live_uid_gid_verified": True,
        "live_channel_observed": True,
        "report_outcome": "OBSERVED_ENTRY_EXIT_EQUAL",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "live report protocol binding differs")
    entry = _process_snapshot(body["entry_process_snapshot"], expected_phase="ENTRY")
    exit_snapshot = _process_snapshot(body["exit_process_snapshot"], expected_phase="EXIT")
    if (
        not _exact_equal(entry, challenge_body["entry_process_snapshot"])
        or body["exit_process_snapshot_payload_sha256"] != _sha(_canonical(exit_snapshot))
        or response_body["entry_process_snapshot_id"] != entry["process_snapshot_id"]
    ):
        _fail(FAIL_BINDING_V2, "report process snapshot linkage differs")
    identity_equal = all(
        _exact_equal(entry[field], exit_snapshot[field])
        for field in ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2
    )
    if body["entry_exit_identity_equal"] is not identity_equal or not identity_equal:
        _fail(FAIL_BINDING_V2, "entry/exit process identity differs")
    if (
        entry["peer_uid"] != live_policy["online_witness_service_uid"]
        or entry["peer_gid"] != live_policy["online_witness_service_gid"]
        or entry["user_namespace_inode"] != live_policy["online_witness_service_user_namespace_inode"]
        or entry["cgroup_sha256"] != live_policy["online_witness_expected_cgroup_sha256"]
    ):
        _fail(FAIL_BINDING_V2, "observed peer differs from live policy")
    sent_text = _text(body["challenge_sent_monotonic_nanoseconds_decimal"], "report sent monotonic")
    received_text = _text(body["response_received_monotonic_nanoseconds_decimal"], "report received monotonic")
    if _DECIMAL_RE.fullmatch(sent_text) is None or _DECIMAL_RE.fullmatch(received_text) is None:
        _fail(FAIL_SCHEMA_V2, "report monotonic timestamp syntax differs")
    sent = int(sent_text)
    received = int(received_text)
    round_trip = _integer(body["round_trip_nanoseconds"], "report round trip")
    if (
        sent_text != challenge_body["challenge_monotonic_nanoseconds_decimal"]
        or received < sent
        or round_trip != received - sent
        or round_trip > live_policy["maximum_round_trip_nanoseconds"]
        or body["round_trip_within_policy"] is not True
    ):
        _fail(FAIL_BINDING_V2, "report monotonic freshness mechanics differ")
    packet_size = _integer(body["response_packet_size_bytes"], "response packet size", minimum=1)
    if packet_size != len(_canonical(dict(response_root))) or packet_size > live_policy["maximum_packet_bytes"]:
        _fail(FAIL_BINDING_V2, "response packet size differs")
    _verify_signature_order(
        root, kind="live_runtime_report", body=body,
        signers=(("authority", source, SIGN_RUNTIME_CLOSURE_PURPOSE_V2, "", 0, ""),),
    )
    return root, body, entry, exit_snapshot


def _load_attestation(
    payload: bytes, *, context: Mapping[str, object], challenge_root: Mapping[str, object],
    challenge_body: Mapping[str, object], response_root: Mapping[str, object],
    response_body: Mapping[str, object], report_root: Mapping[str, object],
    report_body: Mapping[str, object], entry: Mapping[str, object],
    exit_snapshot: Mapping[str, object], custody_policy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_custody_attestation")
    _candidate_header(body, LIVE_RUNTIME_CUSTODY_ATTESTATION_SCHEMA_V2, "custody attestation")
    _common_session(body, context=context, label="custody attestation")
    expected = {
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "live_runtime_response_id": response_root["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(_canonical(dict(response_root))),
        "live_runtime_report_id": report_root["live_runtime_report_id"],
        "live_runtime_report_payload_sha256": _sha(_canonical(dict(report_root))),
        "challenge_nonce": challenge_body["challenge_nonce"],
        "response_nonce": response_body["response_nonce"],
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": report_body["entry_process_snapshot_payload_sha256"],
        "exit_process_snapshot_id": exit_snapshot["process_snapshot_id"],
        "exit_process_snapshot_payload_sha256": report_body["exit_process_snapshot_payload_sha256"],
        "attestor_key_id": custody_policy["custody_attestor_key_id"],
        "attestor_key_epoch": custody_policy["custody_attestor_key_epoch"],
        "attestation_purpose": ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2,
        "online_witness_key_id": custody_policy["online_witness_key_id"],
        "online_witness_key_epoch": custody_policy["online_witness_key_epoch"],
        "online_witness_public_key_hex": custody_policy["online_witness_public_key_hex"],
        "attestation_evidence_format": custody_policy["attestation_evidence_format"],
        "attested_challenge_nonce": challenge_body["challenge_nonce"],
        "attested_report_id": report_root["live_runtime_report_id"],
        "attested_report_payload_sha256": _sha(_canonical(dict(report_root))),
        "attested_online_key_public_key_sha256": _sha(bytes.fromhex(custody_policy["online_witness_public_key_hex"])),
        "online_key_generated_in_attested_boundary": True,
        "online_key_nonexportable": True,
        "certificate_chain_verified": True,
        "quote_verified": True,
        "fresh_nonce_verified": True,
        "report_binding_verified": True,
        "online_key_binding_verified": True,
        "attestation_outcome": "ATTESTED_PRESENTED_EVIDENCE",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, "custody attestation report/key binding differs")
    if (
        _integer(body["attestation_sequence"], "attestation sequence", minimum=1)
        != context["session_sequence"]
    ):
        _fail(FAIL_BINDING_V2, "custody attestation sequence differs from live session")
    _identifier(body["attestor_service_instance_id"], "attestor service instance")
    _identifier(body["attestor_control_domain_id"], "attestor control domain")
    opaque = _hex64(body["opaque_key_handle_sha256"], "opaque key handle SHA")
    if body["attested_key_handle_sha256"] != opaque:
        _fail(FAIL_BINDING_V2, "attested key handle digest differs")
    evidence_hex = _text(body["attestation_evidence_hex"], "attestation evidence")
    if not evidence_hex or len(evidence_hex) % 2 or _LOWER_HEX_RE.fullmatch(evidence_hex) is None:
        _fail(FAIL_SCHEMA_V2, "attestation evidence hex differs")
    if len(evidence_hex) // 2 > custody_policy["maximum_attestation_evidence_bytes"]:
        _fail(FAIL_INPUT_V2, "attestation evidence exceeds policy")
    certificate_chain = _string_sequence(body["certificate_chain_der_hex_in_exact_order"], "certificate chain", nonempty=True)
    for item in certificate_chain:
        if not item or len(item) % 2 or _LOWER_HEX_RE.fullmatch(item) is None:
            _fail(FAIL_SCHEMA_V2, "certificate DER encoding differs")
    _identifier(body["certificate_chain_root_id"], "certificate root ID")
    if body["certificate_chain_root_id"] not in custody_policy["trusted_certificate_root_ids_in_exact_order"]:
        _fail(FAIL_BINDING_V2, "certificate root is not pinned")
    for field, label in (
        ("endorsement_material_hex_in_exact_order", "endorsement material"),
        ("revocation_material_hex_in_exact_order", "revocation material"),
    ):
        materials = _string_sequence(body[field], label, nonempty=True)
        if any(
            not item
            or len(item) % 2
            or _LOWER_HEX_RE.fullmatch(item) is None
            for item in materials
        ):
            _fail(FAIL_SCHEMA_V2, f"{label} hex encoding differs")
    _verify_signature_order(
        root, kind="live_runtime_custody_attestation", body=body,
        signers=((
            "external", None, ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2,
            _text(custody_policy["custody_attestor_key_id"], "attestor key ID"),
            _integer(custody_policy["custody_attestor_key_epoch"], "attestor key epoch", minimum=1),
            _text(custody_policy["custody_attestor_public_key_hex"], "attestor public key"),
        ),),
    )
    return root, body


def _load_live_result(
    payload: bytes, *, context: Mapping[str, object], challenge_root: Mapping[str, object],
    response_root: Mapping[str, object], report_root: Mapping[str, object],
    attestation_root: Mapping[str, object], source: Mapping[str, object],
    pure: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    root, body = _load_root(payload, "live_runtime_result")
    _candidate_header(body, LIVE_RUNTIME_RESULT_SCHEMA_V2, "live runtime result")
    _common_session(body, context=context, label="live runtime result")
    expected_refs = {
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(_canonical(dict(challenge_root))),
        "live_runtime_response_id": response_root["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(_canonical(dict(response_root))),
        "live_runtime_report_id": report_root["live_runtime_report_id"],
        "live_runtime_report_payload_sha256": _sha(_canonical(dict(report_root))),
        "live_runtime_custody_attestation_id": attestation_root["live_runtime_custody_attestation_id"],
        "live_runtime_custody_attestation_payload_sha256": _sha(_canonical(dict(attestation_root))),
    }
    if any(not _exact_equal(body[field], value) for field, value in expected_refs.items()):
        _fail(FAIL_BINDING_V2, "live result evidence references differ")
    expected_claims: dict[str, object] = {
        "verified_signature_count": PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2,
        "verified_signature_scope": PRE_CAS_VERIFIED_SIGNATURE_SCOPE_V2,
        "verified_protocol_object_count": PRE_CAS_VERIFIED_PROTOCOL_OBJECT_COUNT_V2,
        "raw_authority_graph_verified": True,
        "frozen_b480_candidate_verified": True,
        "nineteen_expected_pins_verified": True,
        "key_sets_pairwise_disjoint": True,
        "offline_static_image_graph_verified": True,
        "service_image_exists": False,
        "service_image_pulled": False,
        "service_image_executed": False,
        "runtime_closure_verified": False,
        "entry_exit_process_identity_equal": False,
        "socket_seqpacket_verified": False,
        "so_peercred_verified": False,
        "pidfd_identity_verified": False,
        "procfs_identity_verified": False,
        "actor_nonce_freshness_verified": False,
        "actor_sequence_high_water_verified": False,
        "one_shot_verified": False,
        "round_trip_freshness_verified": False,
        "key_custody_verified": False,
        "online_key_nonexportable_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "malicious_root_resistance_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
        "presented_only": True,
        "private_key_present": False,
        "spool_path_present": False,
        "socket_present": False,
        "decoded_mapping_input_present": False,
        "callback_present": False,
        "caller_nonce_present": False,
        "caller_clock_present": False,
        "first_failure_code": "NONE",
    }
    if any(not _exact_equal(body[field], value) for field, value in expected_claims.items()):
        _fail(FAIL_CLAIM_BOUNDARY_V2, "live result mechanics/claim boundary differs")
    _verify_signature_order(
        root, kind="live_runtime_result", body=body,
        signers=(
            ("authority", source, SIGN_RUNTIME_CLOSURE_PURPOSE_V2, "", 0, ""),
            ("authority", pure, SIGN_PURE_REPLAY_RESULT_PURPOSE_V2, "", 0, ""),
        ),
    )
    return root, body


def _cas_chain_id(
    *, attempt_context_id: str, session_id: str, session_sequence: int,
    live_runtime_result_id: str, live_runtime_result_payload_sha256: str,
) -> str:
    subject = {
        "attempt_context_id": attempt_context_id,
        "session_id": session_id,
        "session_sequence": session_sequence,
        "live_runtime_result_id": live_runtime_result_id,
        "live_runtime_result_payload_sha256": live_runtime_result_payload_sha256,
    }
    return CAS_CHAIN_ID_PREFIX_V2 + _SHA256_V2(
        CAS_CHAIN_ID_DOMAIN_V2 + _canonical(subject)
    ).hexdigest()


def _verify_cas_session(
    *, stage_b_trust_lock_payload: bytes, authority_profile_payload: bytes,
    signer_registry_payload: bytes, oob_lock_id: bytes, oob_lock_digest: bytes,
    retention_plan_payload: bytes, request_payloads: tuple[bytes, ...],
    object_payloads: tuple[bytes, ...], signed_receipt_payloads: tuple[bytes, ...],
    checkpoint_payloads: tuple[bytes, ...], head_payloads: tuple[bytes, ...],
    response_payloads: tuple[bytes, ...], context: Mapping[str, object],
    live_roots: tuple[Mapping[str, object], ...], live_payloads: tuple[bytes, ...],
    _verify=store_replay.verify_cas_store_complete_session_replay_core_v2,
    _require=store_replay.require_verified_cas_store_complete_session_replay_core_v2,
) -> tuple[object, dict[str, object]]:
    sequences = (
        request_payloads, object_payloads, signed_receipt_payloads,
        checkpoint_payloads, head_payloads, response_payloads,
    )
    if any(type(sequence) is not tuple or len(sequence) != 6 for sequence in sequences):
        _fail(FAIL_INPUT_V2, "CAS replay requires six exact six-item tuples")
    if any(type(item) is not bytes for sequence in sequences for item in sequence):
        _fail(FAIL_INPUT_V2, "CAS replay tuple elements must be exact bytes")
    if object_payloads != live_payloads:
        _fail(FAIL_BINDING_V2, "CAS object tuple is not exact six live payloads")
    plan = _object(_decode(retention_plan_payload, "CAS retention plan"), CAS_RETENTION_PLAN_FIELDS_V2, "CAS retention plan")
    if (
        plan["schema_version"] != _STORE_RETENTION_PLAN_SCHEMA_V1
        or plan["authority_profile_id"] != context["profile_id"]
        or plan["signer_registry_id"] != context["registry_id"]
        or plan["attempt_context_id"] != context["attempt_id"]
        or plan["retention_session_id"] != context["session_id"]
        or plan["chain_id"] != _cas_chain_id(
            attempt_context_id=context["attempt_id"],
            session_id=context["session_id"],
            session_sequence=context["session_sequence"],
            live_runtime_result_id=live_roots[-1]["live_runtime_result_id"],
            live_runtime_result_payload_sha256=_sha(live_payloads[-1]),
        )
        or plan["plan_kind"] != CAS_PLAN_KIND_V2
        or plan["expected_object_count"] != 6
        or plan["complete_head_required"] is not True
    ):
        _fail(FAIL_BINDING_V2, "CAS retention plan summary differs")
    rows = _sequence(plan["objects_in_exact_order"], "CAS planned objects")
    if len(rows) != 6:
        _fail(FAIL_BINDING_V2, "CAS planned object count differs")
    for observed, expected in zip(rows, CAS_OBJECTS_IN_EXACT_ORDER_V2, strict=True):
        row = _object(observed, CAS_PLAN_OBJECT_FIELDS_V2, "CAS planned object")
        if (row["ordinal"], row["artifact_kind"], row["maximum_bytes"]) != expected:
            _fail(FAIL_BINDING_V2, "CAS planned object row differs")
    for index, (request_payload, root, object_payload) in enumerate(
        zip(request_payloads, live_roots, live_payloads, strict=True)
    ):
        request = _object(_decode(request_payload, f"CAS request {index}"), CAS_GENERIC_REQUEST_FIELDS_V2, f"CAS request {index}")
        id_field = _ID_FIELD_BY_KIND[
            (
                "live_runtime_policy", "live_runtime_challenge", "live_runtime_response",
                "live_runtime_report", "live_runtime_custody_attestation", "live_runtime_result",
            )[index]
        ]
        if (
            request["authority_profile_id"] != context["profile_id"]
            or request["signer_registry_id"] != context["registry_id"]
            or request["attempt_context_id"] != context["attempt_id"]
            or request["retention_session_id"] != context["session_id"]
            or request["chain_id"] != plan["chain_id"]
            or request["plan_id"] != plan["plan_id"]
            or request["ordinal"] != index
            or request["artifact_kind"] != CAS_OBJECTS_IN_EXACT_ORDER_V2[index][1]
            or request["bound_artifact_id"] != root[id_field]
            or request["bound_artifact_sha256"] != _store_sha(object_payload)
            or request["reported_size_bytes"] != len(object_payload)
            or request["fd_count"] != 1
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
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
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
    final_request = _object(_decode(request_payloads[-1], "final CAS request"), CAS_GENERIC_REQUEST_FIELDS_V2, "final CAS request")
    final_response = _decode(response_payloads[-1], "final CAS response")
    if type(final_response) is not dict:
        _fail(FAIL_SCHEMA_V2, "final CAS response root differs")
    final_binding = result.object_bindings_in_exact_order[-1]
    if type(final_binding) is not tuple or len(final_binding) != 10:
        _fail(FAIL_BINDING_V2, "final CAS replay binding shape differs")
    complete = {
        "schema_version": CAS_COMPLETE_BINDING_SCHEMA_V2,
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
        "final_artifact_kind": "live_runtime_result_v2",
        "live_runtime_result_id": live_roots[-1]["live_runtime_result_id"],
        "live_runtime_result_payload_sha256": _sha(live_payloads[-1]),
        "final_store_request_id": final_request["request_id"],
        "final_store_request_payload_sha256": _store_sha(request_payloads[-1]),
        "final_object_id": final_binding[4],
        "final_object_sha256": _store_sha(live_payloads[-1]),
        "final_object_size_bytes": len(live_payloads[-1]),
        "final_signed_receipt_payload_sha256": final_binding[6],
        "final_checkpoint_payload_sha256": result.complete_checkpoint_sha256,
        "final_head_state_id": result.complete_head_state_id,
        "final_head_payload_sha256": result.complete_head_sha256,
        "final_response_payload_sha256": result.complete_response_sha256,
        "observed_object_count": 6,
        "complete": True,
        "complete_head_required": True,
        "durable_before_return": True,
        "readback_verified": True,
    }
    _object(complete, LIVE_RUNTIME_CAS_COMPLETE_BINDING_FIELDS_V2, "derived CAS complete binding")
    if final_response.get("durable_before_return") is not True or final_response.get("readback_verified") is not True:
        _fail(FAIL_BINDING_V2, "final CAS response lacks presented durable/readback claim")
    return result, complete


def _common_anchor(
    body: Mapping[str, object], *, context: Mapping[str, object], label: str
) -> None:
    expected = {
        "status": STATUS_V2,
        "claim_level": CLAIM_LEVEL_V2,
        "stage_b_trust_lock_id": context["lock_id"],
        "authority_profile_id": context["profile_id"],
        "signer_registry_id": context["registry_id"],
        "attempt_context_id": context["attempt_id"],
        "bootstrap_external_witness_policy_id": context["bootstrap_id"],
        "online_witness_policy_id": context["online_id"],
        "durable_anchor_policy_id": context["durable_id"],
        "live_runtime_policy_id": context["live_policy_id"],
        "live_runtime_anchor_policy_id": context["anchor_policy_id"],
        "admission_candidate_result_id": context["admission_result_id"],
        "admission_candidate_result_payload_sha256": context["admission_result_sha"],
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
    }
    if any(not _exact_equal(body[field], value) for field, value in expected.items()):
        _fail(FAIL_BINDING_V2, f"{label} A2 lineage differs")


def _committed_mapping(
    *, context: Mapping[str, object], result_root: Mapping[str, object],
    result_payload: bytes, complete: Mapping[str, object], anchor_sequence: int,
) -> dict[str, object]:
    return {
        "schema_version": LIVE_RUNTIME_ANCHOR_MAPPING_SCHEMA_V2,
        "attempt_context_id": context["attempt_id"],
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
        "admission_candidate_result_id": context["admission_result_id"],
        "admission_candidate_result_payload_sha256": context["admission_result_sha"],
        "live_runtime_result_id": result_root["live_runtime_result_id"],
        "live_runtime_result_payload_sha256": _sha(result_payload),
        "retention_plan_id": complete["retention_plan_id"],
        "retention_plan_payload_sha256": complete["retention_plan_payload_sha256"],
        "retention_session_id": complete["retention_session_id"],
        "chain_id": complete["chain_id"],
        "cas_complete_head_state_id": complete["final_head_state_id"],
        "cas_final_checkpoint_payload_sha256": complete["final_checkpoint_payload_sha256"],
        "cas_final_head_payload_sha256": complete["final_head_payload_sha256"],
        "cas_final_response_payload_sha256": complete["final_response_payload_sha256"],
        "anchor_sequence": anchor_sequence,
        "effective_state": "COMMITTED",
        "superseded": False,
        "revoked": False,
    }


def _load_anchor_transcript(
    *, request_payload: bytes, checkpoint_payload: bytes, head_payload: bytes,
    receipt_payload: bytes, context: Mapping[str, object],
    anchor_policy: Mapping[str, object], source: Mapping[str, object],
    live_roots: tuple[Mapping[str, object], ...], live_payloads: tuple[bytes, ...],
    complete: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    request_root, request = _load_root(request_payload, "live_runtime_anchor_request")
    _candidate_header(request, LIVE_RUNTIME_ANCHOR_REQUEST_SCHEMA_V2, "A2 request")
    _common_anchor(request, context=context, label="A2 request")
    expected_refs = {
        "live_runtime_policy_payload_sha256": _sha(live_payloads[0]),
        "live_runtime_challenge_id": live_roots[1]["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(live_payloads[1]),
        "live_runtime_response_id": live_roots[2]["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(live_payloads[2]),
        "live_runtime_report_id": live_roots[3]["live_runtime_report_id"],
        "live_runtime_report_payload_sha256": _sha(live_payloads[3]),
        "live_runtime_custody_attestation_id": live_roots[4]["live_runtime_custody_attestation_id"],
        "live_runtime_custody_attestation_payload_sha256": _sha(live_payloads[4]),
        "live_runtime_result_id": live_roots[5]["live_runtime_result_id"],
        "live_runtime_result_payload_sha256": _sha(live_payloads[5]),
        "expected_previous_anchor_checkpoint_id": anchor_policy["oob_high_water_checkpoint_id"],
        "expected_previous_anchor_checkpoint_payload_sha256": anchor_policy["oob_high_water_checkpoint_payload_sha256"],
        "expected_previous_anchor_checkpoint_sequence": anchor_policy["oob_high_water_checkpoint_sequence"],
        "expected_previous_anchor_head_id": anchor_policy["oob_high_water_head_id"],
        "expected_previous_anchor_head_payload_sha256": anchor_policy["oob_high_water_head_payload_sha256"],
        "expected_previous_anchor_head_sequence": anchor_policy["oob_high_water_head_sequence"],
        "compare_and_append_mode": anchor_policy["compare_and_append_semantics"],
    }
    if any(not _exact_equal(request[field], value) for field, value in expected_refs.items()):
        _fail(FAIL_BINDING_V2, "A2 request evidence/high-water refs differ")
    if not _exact_equal(
        _object(request["cas_complete_binding"], LIVE_RUNTIME_CAS_COMPLETE_BINDING_FIELDS_V2, "A2 request CAS binding"),
        complete,
    ):
        _fail(FAIL_BINDING_V2, "A2 request CAS complete binding differs")
    previous = _mapping(request["expected_current_mapping"], "A2 expected current mapping")
    if not _exact_equal(previous, anchor_policy["oob_high_water_mapping"]):
        _fail(FAIL_BINDING_V2, "A2 expected current mapping differs from OOB high-water")
    next_sequence = anchor_policy["oob_high_water_head_sequence"] + 1
    committed = _committed_mapping(
        context=context, result_root=live_roots[5], result_payload=live_payloads[5],
        complete=complete, anchor_sequence=next_sequence,
    )
    if not _exact_equal(_mapping(request["proposed_current_mapping"], "A2 proposed mapping"), committed):
        _fail(FAIL_BINDING_V2, "A2 proposed mapping differs")
    _verify_signature_order(
        request_root, kind="live_runtime_anchor_request", body=request,
        signers=(("authority", source, SIGN_RUNTIME_CLOSURE_PURPOSE_V2, "", 0, ""),),
    )

    checkpoint_root, checkpoint = _load_root(checkpoint_payload, "live_runtime_anchor_checkpoint")
    _candidate_header(checkpoint, LIVE_RUNTIME_ANCHOR_CHECKPOINT_SCHEMA_V2, "A2 checkpoint")
    _common_anchor(checkpoint, context=context, label="A2 checkpoint")
    if (
        checkpoint["live_runtime_anchor_request_id"] != request_root["live_runtime_anchor_request_id"]
        or checkpoint["live_runtime_anchor_request_payload_sha256"] != _sha(request_payload)
        or not _exact_equal(_mapping(checkpoint["previous_mapping"], "A2 checkpoint previous mapping"), previous)
        or not _exact_equal(_mapping(checkpoint["committed_mapping"], "A2 checkpoint committed mapping"), committed)
        or not _exact_equal(checkpoint["cas_complete_binding"], complete)
        or checkpoint["cas_outcome"] != "APPENDED"
        or checkpoint["checkpoint_sequence"] != next_sequence
        or checkpoint["previous_live_runtime_anchor_checkpoint_id"] != anchor_policy["oob_high_water_checkpoint_id"]
    ):
        _fail(FAIL_BINDING_V2, "A2 checkpoint transition differs")
    _verify_signature_order(
        checkpoint_root, kind="live_runtime_anchor_checkpoint", body=checkpoint,
        signers=((
            "external", None, SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_PURPOSE_V2,
            anchor_policy["anchor_key_id"], anchor_policy["anchor_key_epoch"],
            anchor_policy["anchor_public_key_hex"],
        ),),
    )

    head_root, head = _load_root(head_payload, "live_runtime_anchor_head")
    _candidate_header(head, LIVE_RUNTIME_ANCHOR_HEAD_SCHEMA_V2, "A2 head")
    _common_anchor(head, context=context, label="A2 head")
    if (
        head["live_runtime_anchor_request_id"] != request_root["live_runtime_anchor_request_id"]
        or head["live_runtime_anchor_request_payload_sha256"] != _sha(request_payload)
        or head["live_runtime_anchor_checkpoint_id"] != checkpoint_root["live_runtime_anchor_checkpoint_id"]
        or head["live_runtime_anchor_checkpoint_payload_sha256"] != _sha(checkpoint_payload)
        or not _exact_equal(_mapping(head["committed_mapping"], "A2 head mapping"), committed)
        or not _exact_equal(head["cas_complete_binding"], complete)
        or head["cas_outcome"] != "APPENDED"
        or head["head_sequence"] != next_sequence
        or head["previous_live_runtime_anchor_head_id"] != anchor_policy["oob_high_water_head_id"]
    ):
        _fail(FAIL_BINDING_V2, "A2 head transition differs")
    _verify_signature_order(
        head_root, kind="live_runtime_anchor_head", body=head,
        signers=((
            "external", None, SIGN_LIVE_RUNTIME_ANCHOR_HEAD_PURPOSE_V2,
            anchor_policy["anchor_key_id"], anchor_policy["anchor_key_epoch"],
            anchor_policy["anchor_public_key_hex"],
        ),),
    )

    receipt_root, receipt = _load_root(receipt_payload, "live_runtime_anchor_receipt")
    _candidate_header(receipt, LIVE_RUNTIME_ANCHOR_RECEIPT_SCHEMA_V2, "A2 receipt")
    _common_anchor(receipt, context=context, label="A2 receipt")
    if (
        receipt["live_runtime_anchor_request_id"] != request_root["live_runtime_anchor_request_id"]
        or receipt["live_runtime_anchor_request_payload_sha256"] != _sha(request_payload)
        or receipt["live_runtime_anchor_checkpoint_id"] != checkpoint_root["live_runtime_anchor_checkpoint_id"]
        or receipt["live_runtime_anchor_checkpoint_payload_sha256"] != _sha(checkpoint_payload)
        or receipt["live_runtime_anchor_head_id"] != head_root["live_runtime_anchor_head_id"]
        or receipt["live_runtime_anchor_head_payload_sha256"] != _sha(head_payload)
        or not _exact_equal(_mapping(receipt["previous_mapping"], "A2 receipt previous mapping"), previous)
        or not _exact_equal(_mapping(receipt["committed_mapping"], "A2 receipt committed mapping"), committed)
        or not _exact_equal(receipt["cas_complete_binding"], complete)
        or receipt["cas_outcome"] != "APPENDED"
        or receipt["compare_matched"] is not True
        or receipt["append_applied"] is not True
        or receipt["oob_high_water_checkpoint_id"] != anchor_policy["oob_high_water_checkpoint_id"]
        or receipt["oob_high_water_head_id"] != anchor_policy["oob_high_water_head_id"]
        or receipt["durable_before_return"] is not True
    ):
        _fail(FAIL_BINDING_V2, "A2 receipt transition differs")
    _verify_signature_order(
        receipt_root, kind="live_runtime_anchor_receipt", body=receipt,
        signers=((
            "external", None, SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_PURPOSE_V2,
            anchor_policy["anchor_key_id"], anchor_policy["anchor_key_epoch"],
            anchor_policy["anchor_public_key_hex"],
        ),),
    )
    return request_root, checkpoint_root, head_root, receipt_root


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPresentedLiveRuntimeOverlayV2:
    """Privately issued result for the exact presented overlay mechanics."""

    schema_version: str
    verification_status: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    admission_candidate_result_id: str
    runtime_closure_policy_id: str
    runtime_closure_evidence_id: str
    custody_attestation_policy_id: str
    live_runtime_policy_id: str
    live_runtime_anchor_policy_id: str
    session_id: str
    session_sequence: int
    live_runtime_result_id: str
    retention_plan_id: str
    complete_cas_head_state_id: str
    live_runtime_anchor_request_id: str
    live_runtime_anchor_checkpoint_id: str
    live_runtime_anchor_head_id: str
    live_runtime_anchor_receipt_id: str
    verified_signature_count: int
    verified_signature_scope: str
    raw_authority_graph_verified: bool
    frozen_b480_candidate_verified: bool
    offline_static_image_graph_verified: bool
    nineteen_expected_pins_verified: bool
    key_sets_pairwise_disjoint: bool
    six_live_artifact_chain_verified: bool
    presented_kernel_observation_report_verified: bool
    presented_custody_attestation_verified: bool
    six_object_cas_replay_verified: bool
    presented_a2_compare_and_append_verified: bool
    private_key_present: bool
    spool_path_present: bool
    socket_present: bool
    procfs_path_present: bool
    pidfd_present: bool
    decoded_mapping_input_present: bool
    callback_present: bool
    caller_nonce_present: bool
    caller_clock_present: bool
    presented_only: bool
    service_image_exists: bool
    service_image_pulled: bool
    service_image_executed: bool
    runtime_closure_signature_verified: bool
    runtime_closure_verified: bool
    live_runtime_closure_verified: bool
    remote_attestation_verified: bool
    entry_exit_process_identity_equal: bool
    socket_seqpacket_verified: bool
    so_peercred_verified: bool
    pidfd_identity_verified: bool
    procfs_identity_verified: bool
    actor_nonce_freshness_verified: bool
    actor_sequence_high_water_verified: bool
    one_shot_verified: bool
    round_trip_freshness_verified: bool
    distinct_live_uid_gid_verified: bool
    live_uid_gid_verified: bool
    live_control_domain_verified: bool
    key_custody_verified: bool
    online_key_nonexportable_verified: bool
    live_channel_verified: bool
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

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiedPresentedLiveRuntimeOverlayV2":
        raise TypeError("verified live-runtime overlay results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified live-runtime overlay results are privately issued")

    def __copy__(self) -> "VerifiedPresentedLiveRuntimeOverlayV2":
        raise TypeError("verified live-runtime overlay results cannot be copied")

    def __deepcopy__(self, memo: object) -> "VerifiedPresentedLiveRuntimeOverlayV2":
        raise TypeError("verified live-runtime overlay results cannot be deep-copied")


def _build_private_result_lineage_v2():
    result_type = VerifiedPresentedLiveRuntimeOverlayV2
    fields = tuple(result_type.__dataclass_fields__)
    issued: dict[int, tuple[VerifiedPresentedLiveRuntimeOverlayV2, tuple[object, ...]]] = {}
    lock = threading.Lock()
    error_type = Phase2BStep5R01LiveRuntimeOverlayErrorV2
    fail_code = FAIL_RESULT_V2
    schema_version = SCHEMA_VERSION_V2
    verification_status = (
        "PRESENTED_RAW_LIVE_RUNTIME_OVERLAY_V2_VERIFIED_MECHANICS_ONLY"
    )
    signature_count = PRIVATE_OVERLAY_VERIFIED_SIGNATURE_COUNT_V2
    signature_scope = PRIVATE_OVERLAY_VERIFIED_SIGNATURE_SCOPE_V2

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
                exact_equal(a, b) for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
            )
        return left == right

    def snapshot(value: VerifiedPresentedLiveRuntimeOverlayV2) -> tuple[object, ...]:
        try:
            return tuple(object.__getattribute__(value, field) for field in fields)
        except (AttributeError, TypeError) as exc:
            raise error_type(fail_code, "verified overlay result fields are missing") from exc

    def require(value: object) -> VerifiedPresentedLiveRuntimeOverlayV2:
        if type(value) is not result_type:
            fail_exact("verified overlay result exact type differs")
        with lock:
            record = issued.get(id(value))
        if record is None or record[0] is not value or not exact_equal(snapshot(value), record[1]):
            fail_exact("verified overlay result lacks intact private lineage")
        mechanics_true = (
            "raw_authority_graph_verified", "frozen_b480_candidate_verified",
            "offline_static_image_graph_verified", "nineteen_expected_pins_verified",
            "key_sets_pairwise_disjoint", "six_live_artifact_chain_verified",
            "presented_kernel_observation_report_verified",
            "presented_custody_attestation_verified", "six_object_cas_replay_verified",
            "presented_a2_compare_and_append_verified", "presented_only",
        )
        absent_inputs = (
            "private_key_present", "spool_path_present", "socket_present",
            "procfs_path_present", "pidfd_present", "decoded_mapping_input_present",
            "callback_present", "caller_nonce_present", "caller_clock_present",
        )
        elevated_false = (
            "service_image_exists", "service_image_pulled", "service_image_executed",
            "runtime_closure_signature_verified", "runtime_closure_verified",
            "live_runtime_closure_verified", "remote_attestation_verified",
            "entry_exit_process_identity_equal", "socket_seqpacket_verified",
            "so_peercred_verified", "pidfd_identity_verified", "procfs_identity_verified",
            "actor_nonce_freshness_verified", "actor_sequence_high_water_verified",
            "one_shot_verified", "round_trip_freshness_verified",
            "distinct_live_uid_gid_verified", "live_uid_gid_verified",
            "live_control_domain_verified", "key_custody_verified",
            "online_key_nonexportable_verified", "live_channel_verified",
            "external_cas_verified", "external_anchor_verified",
            "external_anti_rollback_verified",
            "external_currentness_verified", "durability_verified",
            "non_equivocation_verified", "global_anti_fork_verified",
            "malicious_root_resistance_verified", "authority_evidence", "r01_satisfied",
            "readiness_evidence", "execution_authorized",
        )
        if (
            value.schema_version != schema_version
            or value.verification_status != verification_status
            or value.verified_signature_count != signature_count
            or value.verified_signature_scope != signature_scope
            or any(getattr(value, field) is not True for field in mechanics_true)
            or any(getattr(value, field) is not False for field in absent_inputs)
            or any(getattr(value, field) is not False for field in elevated_false)
        ):
            fail_exact("verified overlay result semantics differ")
        return value

    def issue(**values: object) -> VerifiedPresentedLiveRuntimeOverlayV2:
        if frozenset(values) != frozenset(fields):
            fail_exact("private overlay result issuance fields differ")
        value = object.__new__(result_type)
        for field in fields:
            object.__setattr__(value, field, values[field])
        record = (value, tuple(values[field] for field in fields))
        with lock:
            if id(value) in issued:
                fail_exact("private overlay result identity reused")
            issued[id(value)] = record
        return require(value)

    return issue, require


(_issue_result_v2, _require_result_v2) = _build_private_result_lineage_v2()
del _build_private_result_lineage_v2


def require_verified_presented_live_runtime_overlay_v2(
    value: object,
) -> VerifiedPresentedLiveRuntimeOverlayV2:
    return _require_result_v2(value)


def _verify_presented_live_runtime_overlay_implementation_v2(
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
    runtime_closure_policy_payload: bytes,
    runtime_closure_evidence_payload: bytes,
    raw_oci_index_payload: bytes,
    raw_oci_manifest_payload: bytes,
    raw_oci_config_payload: bytes,
    ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
    ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
    custody_attestation_policy_payload: bytes,
    live_runtime_policy_payload: bytes,
    live_runtime_anchor_policy_payload: bytes,
    live_runtime_challenge_payload: bytes,
    live_runtime_response_payload: bytes,
    live_runtime_report_payload: bytes,
    live_runtime_custody_attestation_payload: bytes,
    live_runtime_result_payload: bytes,
    live_runtime_cas_retention_plan_payload: bytes,
    live_runtime_cas_store_request_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_cas_object_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_cas_signed_receipt_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_cas_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_cas_head_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_cas_response_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_anchor_request_payload: bytes,
    live_runtime_anchor_checkpoint_payload: bytes,
    live_runtime_anchor_head_payload: bytes,
    live_runtime_anchor_receipt_payload: bytes,
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
    _verify_admission=admission.verify_external_witness_admission_transcript_candidate_v1,
    _require_admission=admission.require_verified_external_witness_admission_candidate_v1,
    _verify_closure=closure.verify_offline_step5_image_closure_mechanics_v1,
    _require_closure=closure.require_verified_offline_step5_image_closure_mechanics_v1,
    _issue_result=_issue_result_v2,
) -> VerifiedPresentedLiveRuntimeOverlayV2:
    """Verify one exact presented overlay from raw bytes only."""

    scalar_payloads = (
        stage_b_trust_lock_payload, authority_profile_payload, signer_registry_payload,
        bootstrap_external_witness_policy_payload, online_witness_policy_payload,
        durable_anchor_policy_payload, candidate_anchor_payload, candidate_receipt_payload,
        candidate_ledger_payload, candidate_checkpoint_payload, candidate_head_payload,
        q1_prepare_request_payload, q1_prepare_response_payload,
        a_compare_and_append_request_payload, a_durable_checkpoint_payload,
        a_durable_head_payload, a_compare_and_append_receipt_payload,
        q2_confirm_request_payload, q2_confirm_response_payload, admission_transcript_payload,
        admission_candidate_result_payload, runtime_closure_policy_payload,
        runtime_closure_evidence_payload, raw_oci_index_payload, raw_oci_manifest_payload,
        raw_oci_config_payload, custody_attestation_policy_payload,
        live_runtime_policy_payload, live_runtime_anchor_policy_payload,
        live_runtime_challenge_payload, live_runtime_response_payload,
        live_runtime_report_payload, live_runtime_custody_attestation_payload,
        live_runtime_result_payload, live_runtime_cas_retention_plan_payload,
        live_runtime_anchor_request_payload, live_runtime_anchor_checkpoint_payload,
        live_runtime_anchor_head_payload, live_runtime_anchor_receipt_payload,
    )
    if any(type(payload) is not bytes for payload in scalar_payloads):
        _fail(FAIL_INPUT_V2, "all scalar transports must be exact bytes")
    for chunks, label in (
        (ordered_raw_layer_chunk_tuples, "ordered raw OCI layers"),
        (ordered_support_artifact_chunk_tuples, "ordered support artifacts"),
    ):
        if type(chunks) is not tuple or not chunks:
            _fail(FAIL_INPUT_V2, f"{label} must be one nonempty exact tuple")
        if any(type(row) is not tuple or not row or any(type(chunk) is not bytes for chunk in row) for row in chunks):
            _fail(FAIL_INPUT_V2, f"{label} chunk matrix differs")

    try:
        admission_result = _verify_admission(
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
            out_of_band_expected_lock_payload_sha256_digest=out_of_band_expected_lock_payload_sha256_digest,
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes=out_of_band_expected_bootstrap_external_witness_policy_id_bytes,
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest,
            out_of_band_expected_online_witness_policy_id_bytes=out_of_band_expected_online_witness_policy_id_bytes,
            out_of_band_expected_online_witness_policy_payload_sha256_digest=out_of_band_expected_online_witness_policy_payload_sha256_digest,
            out_of_band_expected_durable_anchor_policy_id_bytes=out_of_band_expected_durable_anchor_policy_id_bytes,
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest=out_of_band_expected_durable_anchor_policy_payload_sha256_digest,
            expected_attempt_context_id_bytes=expected_attempt_context_id_bytes,
        )
        admission_result = _require_admission(admission_result)
    except Exception as exc:
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_DEPENDENCY_V2, "frozen b480 admission replay rejected"
        ) from exc

    auth = _load_authority_context(
        stage_b_trust_lock_payload, authority_profile_payload, signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_lock_payload_sha256_digest=out_of_band_expected_lock_payload_sha256_digest,
    )
    bootstrap = _object(
        _decode(bootstrap_external_witness_policy_payload, "bootstrap witness policy"),
        _ADMISSION_BOOTSTRAP_POLICY_FIELDS_V1,
        "bootstrap witness policy",
    )
    online_root = _object(
        _decode(online_witness_policy_payload, "online witness policy"),
        _ADMISSION_ONLINE_POLICY_FIELDS_V1,
        "online witness policy",
    )
    online = _object(
        online_root["body"], _ADMISSION_ONLINE_POLICY_BODY_FIELDS_V1,
        "online witness policy body",
    )
    durable_root = _object(
        _decode(durable_anchor_policy_payload, "durable anchor policy"),
        _ADMISSION_DURABLE_POLICY_FIELDS_V1,
        "durable anchor policy",
    )
    durable = _object(
        durable_root["body"], _ADMISSION_DURABLE_POLICY_BODY_FIELDS_V1,
        "durable anchor policy body",
    )
    admission_result_root = _object(
        _decode(admission_candidate_result_payload, "admission candidate result"),
        _ADMISSION_RESULT_FIELDS_V1,
        "admission candidate result",
    )
    if admission_result_root["candidate_result_id"] != admission_result.admission_candidate_result_id:
        _fail(FAIL_BINDING_V2, "b480 private result/raw result ID differs")
    if type(expected_attempt_context_id_bytes) is not bytes:
        _fail(FAIL_INPUT_V2, "expected attempt context pin must be exact bytes")
    try:
        attempt_id = expected_attempt_context_id_bytes.decode("ascii")
    except UnicodeError as exc:
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_INPUT_V2, "expected attempt context is not ASCII"
        ) from exc
    if attempt_id != admission_result.attempt_context_id:
        _fail(FAIL_BINDING_V2, "expected attempt context differs")

    closure_policy_root = _upstream_object(runtime_closure_policy_payload, "runtime closure policy")
    closure_evidence_root = _upstream_object(runtime_closure_evidence_payload, "runtime closure evidence")
    if frozenset(closure_policy_root) != frozenset({"body", "id"}) or frozenset(closure_evidence_root) != frozenset({"body", "id"}):
        _fail(FAIL_SCHEMA_V2, "runtime closure root field set differs")
    _require_raw_pin(out_of_band_expected_runtime_closure_policy_id_bytes, closure_policy_root["id"], "runtime closure policy ID")
    _require_digest_pin(out_of_band_expected_runtime_closure_policy_payload_sha256_digest, runtime_closure_policy_payload, "runtime closure policy")
    _require_raw_pin(out_of_band_expected_runtime_closure_evidence_id_bytes, closure_evidence_root["id"], "runtime closure evidence ID")
    _require_digest_pin(out_of_band_expected_runtime_closure_evidence_payload_sha256_digest, runtime_closure_evidence_payload, "runtime closure evidence")
    closure_policy_body = closure_policy_root["body"]
    if type(closure_policy_body) is not dict:
        _fail(FAIL_SCHEMA_V2, "runtime closure policy body differs")
    _pre_live_policy_root, pre_live_policy_body = _load_root(
        live_runtime_policy_payload, "live_runtime_policy"
    )
    service_name = pre_live_policy_body["runtime_closure_service_name"]
    if type(service_name) is not str or service_name != "runner":
        _fail(
            FAIL_BINDING_V2,
            "live policy runtime closure service must be authority runner",
        )
    try:
        closure_result = _verify_closure(
            stage_b_trust_lock_payload=stage_b_trust_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=out_of_band_expected_lock_payload_sha256_digest,
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
        raise Phase2BStep5R01LiveRuntimeOverlayErrorV2(
            FAIL_DEPENDENCY_V2, "f836 offline image closure replay rejected"
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
        _fail(FAIL_CLAIM_BOUNDARY_V2, "f836 mechanics/claim boundary differs")

    refs: dict[str, object] = {
        "stage_b_trust_lock_id": auth["lock"].artifact_id,
        "stage_b_trust_lock_payload_sha256": _sha(stage_b_trust_lock_payload),
        "authority_profile_id": auth["profile"].artifact_id,
        "authority_profile_payload_sha256": _sha(authority_profile_payload),
        "signer_registry_id": auth["registry"].artifact_id,
        "signer_registry_payload_sha256": _sha(signer_registry_payload),
        "attempt_context_id": attempt_id,
        "admission_candidate_result_id": admission_result.admission_candidate_result_id,
        "admission_candidate_result_payload_sha256": _sha(admission_candidate_result_payload),
        "bootstrap_external_witness_policy_id": bootstrap["external_witness_policy_id"],
        "bootstrap_external_witness_policy_payload_sha256": _sha(bootstrap_external_witness_policy_payload),
        "online_witness_policy_id": online_root["online_witness_policy_id"],
        "online_witness_policy_payload_sha256": _sha(online_witness_policy_payload),
        "durable_anchor_policy_id": durable_root["durable_anchor_policy_id"],
        "durable_anchor_policy_payload_sha256": _sha(durable_anchor_policy_payload),
        "runtime_closure_policy_id": closure_policy_root["id"],
        "runtime_closure_policy_payload_sha256": _sha(runtime_closure_policy_payload),
        "runtime_closure_evidence_id": closure_evidence_root["id"],
        "runtime_closure_evidence_payload_sha256": _sha(runtime_closure_evidence_payload),
    }
    custody_policy_root, custody_policy = _load_custody_policy(
        custody_attestation_policy_payload, refs=refs, bootstrap=bootstrap,
        online=online,
        oob_id=out_of_band_expected_custody_attestation_policy_id_bytes,
        oob_digest=out_of_band_expected_custody_attestation_policy_payload_sha256_digest,
    )
    live_policy_root, live_policy = _load_live_policy(
        live_runtime_policy_payload, refs=refs, bootstrap=bootstrap, online=online,
        auth=auth, closure_result=closure_result,
        closure_policy_body=closure_policy_body,
        custody_policy_root=custody_policy_root,
        custody_policy_payload=custody_attestation_policy_payload,
        oob_id=out_of_band_expected_live_runtime_policy_id_bytes,
        oob_digest=out_of_band_expected_live_runtime_policy_payload_sha256_digest,
    )
    anchor_policy_root, anchor_policy = _load_anchor_policy(
        live_runtime_anchor_policy_payload, refs=refs,
        live_policy_root=live_policy_root, live_policy_payload=live_runtime_policy_payload,
        durable=durable, auth=auth,
        oob_id=out_of_band_expected_live_runtime_anchor_policy_id_bytes,
        oob_digest=out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest,
    )
    if (
        live_policy["live_runtime_anchor_key_id"] != anchor_policy["anchor_key_id"]
        or live_policy["live_runtime_anchor_key_epoch"] != anchor_policy["anchor_key_epoch"]
        or live_policy["live_runtime_anchor_public_key_hex"] != anchor_policy["anchor_public_key_hex"]
        or live_policy["live_runtime_anchor_authorized_signature_purposes_in_exact_order"]
        != anchor_policy["authorized_signature_purposes_in_exact_order"]
        or live_policy["live_runtime_anchor_control_domain_id"] != anchor_policy["anchor_control_domain_id"]
        or live_policy["anchor_slot_key_profile"]
        != anchor_policy["admission_slot_key_profile"]
        or live_policy["anchor_compare_and_append_semantics"]
        != anchor_policy["compare_and_append_semantics"]
    ):
        _fail(FAIL_BINDING_V2, "live policy/A2 policy key binding differs")
    public_keys = [
        bootstrap["public_key_hex"], online["online_public_key_hex"],
        online["custody_attestor_public_key_hex"], anchor_policy["anchor_public_key_hex"],
        *(auth["roles"][role]["public_key_hex"] for role in _AUTH_ROLE_ORDER_V2),
    ]
    key_ids = [
        bootstrap["witness_key_id"], online["online_key_id"],
        online["custody_attestor_key_id"], anchor_policy["anchor_key_id"],
        *(auth["roles"][role]["key_id"] for role in _AUTH_ROLE_ORDER_V2),
    ]
    if len(set(public_keys)) != 12 or len(set(key_ids)) != 12:
        _fail(FAIL_BINDING_V2, "external and authority key sets are not pairwise disjoint")

    context: dict[str, object] = {
        "lock_id": refs["stage_b_trust_lock_id"],
        "profile_id": refs["authority_profile_id"],
        "registry_id": refs["signer_registry_id"],
        "attempt_id": attempt_id,
        "bootstrap_id": refs["bootstrap_external_witness_policy_id"],
        "online_id": refs["online_witness_policy_id"],
        "durable_id": refs["durable_anchor_policy_id"],
        "closure_policy_id": refs["runtime_closure_policy_id"],
        "closure_evidence_id": refs["runtime_closure_evidence_id"],
        "custody_policy_id": custody_policy_root["custody_attestation_policy_id"],
        "live_policy_id": live_policy_root["live_runtime_policy_id"],
        "anchor_policy_id": anchor_policy_root["live_runtime_anchor_policy_id"],
        "admission_result_id": refs["admission_candidate_result_id"],
        "admission_result_sha": refs["admission_candidate_result_payload_sha256"],
    }
    source = auth["roles"]["SOURCE_RUNTIME_CUSTODIAN"]
    pure = auth["roles"]["PURE_REPLAYER"]
    challenge_root, challenge, context = _load_challenge(
        live_runtime_challenge_payload, context=context, live_policy=live_policy,
        anchor_policy=anchor_policy, source=source,
    )
    response_root, response = _load_response(
        live_runtime_response_payload, context=context,
        challenge_root=challenge_root, challenge_body=challenge,
        live_policy=live_policy,
    )
    report_root, report, entry, exit_snapshot = _load_report(
        live_runtime_report_payload, context=context,
        challenge_root=challenge_root, challenge_body=challenge,
        response_root=response_root, response_body=response,
        live_policy=live_policy, source=source,
    )
    attestation_root, attestation = _load_attestation(
        live_runtime_custody_attestation_payload, context=context,
        challenge_root=challenge_root, challenge_body=challenge,
        response_root=response_root, response_body=response,
        report_root=report_root, report_body=report,
        entry=entry, exit_snapshot=exit_snapshot, custody_policy=custody_policy,
    )
    result_root, result_body = _load_live_result(
        live_runtime_result_payload, context=context,
        challenge_root=challenge_root, response_root=response_root,
        report_root=report_root, attestation_root=attestation_root,
        source=source, pure=pure,
    )
    live_bodies = (live_policy, challenge, response, report, attestation, result_body)
    if any(CAS_FORBIDDEN_IN_ALL_SIX_LIVE_BODY_FIELDS_V2 & frozenset(body) for body in live_bodies):
        _fail(FAIL_SCHEMA_V2, "one live artifact illegally contains a same-session CAS reference")
    live_roots = (
        live_policy_root, challenge_root, response_root, report_root,
        attestation_root, result_root,
    )
    live_payloads = (
        live_runtime_policy_payload, live_runtime_challenge_payload,
        live_runtime_response_payload, live_runtime_report_payload,
        live_runtime_custody_attestation_payload, live_runtime_result_payload,
    )
    cas_result, complete = _verify_cas_session(
        stage_b_trust_lock_payload=stage_b_trust_lock_payload,
        authority_profile_payload=authority_profile_payload,
        signer_registry_payload=signer_registry_payload,
        oob_lock_id=out_of_band_expected_lock_id_bytes,
        oob_lock_digest=out_of_band_expected_lock_payload_sha256_digest,
        retention_plan_payload=live_runtime_cas_retention_plan_payload,
        request_payloads=live_runtime_cas_store_request_payloads_in_exact_order,
        object_payloads=live_runtime_cas_object_payloads_in_exact_order,
        signed_receipt_payloads=live_runtime_cas_signed_receipt_payloads_in_exact_order,
        checkpoint_payloads=live_runtime_cas_checkpoint_payloads_in_exact_order,
        head_payloads=live_runtime_cas_head_payloads_in_exact_order,
        response_payloads=live_runtime_cas_response_payloads_in_exact_order,
        context=context, live_roots=live_roots, live_payloads=live_payloads,
    )
    request_root, checkpoint_root, head_root, receipt_root = _load_anchor_transcript(
        request_payload=live_runtime_anchor_request_payload,
        checkpoint_payload=live_runtime_anchor_checkpoint_payload,
        head_payload=live_runtime_anchor_head_payload,
        receipt_payload=live_runtime_anchor_receipt_payload,
        context=context, anchor_policy=anchor_policy, source=source,
        live_roots=live_roots, live_payloads=live_payloads, complete=complete,
    )
    return _issue_result(
        schema_version=SCHEMA_VERSION_V2,
        verification_status="PRESENTED_RAW_LIVE_RUNTIME_OVERLAY_V2_VERIFIED_MECHANICS_ONLY",
        stage_b_trust_lock_id=context["lock_id"],
        authority_profile_id=context["profile_id"],
        signer_registry_id=context["registry_id"],
        attempt_context_id=context["attempt_id"],
        admission_candidate_result_id=context["admission_result_id"],
        runtime_closure_policy_id=context["closure_policy_id"],
        runtime_closure_evidence_id=context["closure_evidence_id"],
        custody_attestation_policy_id=context["custody_policy_id"],
        live_runtime_policy_id=context["live_policy_id"],
        live_runtime_anchor_policy_id=context["anchor_policy_id"],
        session_id=context["session_id"],
        session_sequence=context["session_sequence"],
        live_runtime_result_id=result_root["live_runtime_result_id"],
        retention_plan_id=cas_result.plan_id,
        complete_cas_head_state_id=cas_result.complete_head_state_id,
        live_runtime_anchor_request_id=request_root["live_runtime_anchor_request_id"],
        live_runtime_anchor_checkpoint_id=checkpoint_root["live_runtime_anchor_checkpoint_id"],
        live_runtime_anchor_head_id=head_root["live_runtime_anchor_head_id"],
        live_runtime_anchor_receipt_id=receipt_root["live_runtime_anchor_receipt_id"],
        verified_signature_count=PRIVATE_OVERLAY_VERIFIED_SIGNATURE_COUNT_V2,
        verified_signature_scope=PRIVATE_OVERLAY_VERIFIED_SIGNATURE_SCOPE_V2,
        raw_authority_graph_verified=True,
        frozen_b480_candidate_verified=True,
        offline_static_image_graph_verified=True,
        nineteen_expected_pins_verified=True,
        key_sets_pairwise_disjoint=True,
        six_live_artifact_chain_verified=True,
        presented_kernel_observation_report_verified=True,
        presented_custody_attestation_verified=True,
        six_object_cas_replay_verified=True,
        presented_a2_compare_and_append_verified=True,
        private_key_present=False,
        spool_path_present=False,
        socket_present=False,
        procfs_path_present=False,
        pidfd_present=False,
        decoded_mapping_input_present=False,
        callback_present=False,
        caller_nonce_present=False,
        caller_clock_present=False,
        presented_only=True,
        service_image_exists=False,
        service_image_pulled=False,
        service_image_executed=False,
        runtime_closure_signature_verified=False,
        runtime_closure_verified=False,
        live_runtime_closure_verified=False,
        remote_attestation_verified=False,
        entry_exit_process_identity_equal=False,
        socket_seqpacket_verified=False,
        so_peercred_verified=False,
        pidfd_identity_verified=False,
        procfs_identity_verified=False,
        actor_nonce_freshness_verified=False,
        actor_sequence_high_water_verified=False,
        one_shot_verified=False,
        round_trip_freshness_verified=False,
        distinct_live_uid_gid_verified=False,
        live_uid_gid_verified=False,
        live_control_domain_verified=False,
        key_custody_verified=False,
        online_key_nonexportable_verified=False,
        live_channel_verified=False,
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
    """Expose the exact raw-only signature without injection keywords."""

    def verify_presented_live_runtime_overlay_v2(
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
        runtime_closure_policy_payload: bytes,
        runtime_closure_evidence_payload: bytes,
        raw_oci_index_payload: bytes,
        raw_oci_manifest_payload: bytes,
        raw_oci_config_payload: bytes,
        ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
        ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
        custody_attestation_policy_payload: bytes,
        live_runtime_policy_payload: bytes,
        live_runtime_anchor_policy_payload: bytes,
        live_runtime_challenge_payload: bytes,
        live_runtime_response_payload: bytes,
        live_runtime_report_payload: bytes,
        live_runtime_custody_attestation_payload: bytes,
        live_runtime_result_payload: bytes,
        live_runtime_cas_retention_plan_payload: bytes,
        live_runtime_cas_store_request_payloads_in_exact_order: tuple[bytes, ...],
        live_runtime_cas_object_payloads_in_exact_order: tuple[bytes, ...],
        live_runtime_cas_signed_receipt_payloads_in_exact_order: tuple[bytes, ...],
        live_runtime_cas_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        live_runtime_cas_head_payloads_in_exact_order: tuple[bytes, ...],
        live_runtime_cas_response_payloads_in_exact_order: tuple[bytes, ...],
        live_runtime_anchor_request_payload: bytes,
        live_runtime_anchor_checkpoint_payload: bytes,
        live_runtime_anchor_head_payload: bytes,
        live_runtime_anchor_receipt_payload: bytes,
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
    ) -> VerifiedPresentedLiveRuntimeOverlayV2:
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
            runtime_closure_policy_payload,
            runtime_closure_evidence_payload,
            raw_oci_index_payload,
            raw_oci_manifest_payload,
            raw_oci_config_payload,
            ordered_raw_layer_chunk_tuples,
            ordered_support_artifact_chunk_tuples,
            custody_attestation_policy_payload,
            live_runtime_policy_payload,
            live_runtime_anchor_policy_payload,
            live_runtime_challenge_payload,
            live_runtime_response_payload,
            live_runtime_report_payload,
            live_runtime_custody_attestation_payload,
            live_runtime_result_payload,
            live_runtime_cas_retention_plan_payload,
            live_runtime_cas_store_request_payloads_in_exact_order,
            live_runtime_cas_object_payloads_in_exact_order,
            live_runtime_cas_signed_receipt_payloads_in_exact_order,
            live_runtime_cas_checkpoint_payloads_in_exact_order,
            live_runtime_cas_head_payloads_in_exact_order,
            live_runtime_cas_response_payloads_in_exact_order,
            live_runtime_anchor_request_payload,
            live_runtime_anchor_checkpoint_payload,
            live_runtime_anchor_head_payload,
            live_runtime_anchor_receipt_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=out_of_band_expected_lock_payload_sha256_digest,
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes=out_of_band_expected_bootstrap_external_witness_policy_id_bytes,
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest,
            out_of_band_expected_online_witness_policy_id_bytes=out_of_band_expected_online_witness_policy_id_bytes,
            out_of_band_expected_online_witness_policy_payload_sha256_digest=out_of_band_expected_online_witness_policy_payload_sha256_digest,
            out_of_band_expected_durable_anchor_policy_id_bytes=out_of_band_expected_durable_anchor_policy_id_bytes,
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest=out_of_band_expected_durable_anchor_policy_payload_sha256_digest,
            out_of_band_expected_runtime_closure_policy_id_bytes=out_of_band_expected_runtime_closure_policy_id_bytes,
            out_of_band_expected_runtime_closure_policy_payload_sha256_digest=out_of_band_expected_runtime_closure_policy_payload_sha256_digest,
            out_of_band_expected_runtime_closure_evidence_id_bytes=out_of_band_expected_runtime_closure_evidence_id_bytes,
            out_of_band_expected_runtime_closure_evidence_payload_sha256_digest=out_of_band_expected_runtime_closure_evidence_payload_sha256_digest,
            out_of_band_expected_custody_attestation_policy_id_bytes=out_of_band_expected_custody_attestation_policy_id_bytes,
            out_of_band_expected_custody_attestation_policy_payload_sha256_digest=out_of_band_expected_custody_attestation_policy_payload_sha256_digest,
            out_of_band_expected_live_runtime_policy_id_bytes=out_of_band_expected_live_runtime_policy_id_bytes,
            out_of_band_expected_live_runtime_policy_payload_sha256_digest=out_of_band_expected_live_runtime_policy_payload_sha256_digest,
            out_of_band_expected_live_runtime_anchor_policy_id_bytes=out_of_band_expected_live_runtime_anchor_policy_id_bytes,
            out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest=out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest,
            expected_attempt_context_id_bytes=expected_attempt_context_id_bytes,
        )

    return verify_presented_live_runtime_overlay_v2


def verify_real_live_runtime_overlay_v2(*args: object, **kwargs: object) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V2,
        "real live/current/durable/non-equivocating custody-backed overlay is NOT_IMPLEMENTED",
    )


def _definition_captured_function_graph_v2() -> dict[str, object]:
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
            None if function.__kwdefaults__ is None else {
                key: captured_default(value)
                for key, value in function.__kwdefaults__.items()
            }
        )
        captured[name] = clone
    private_namespace.update(captured)
    return captured


_CAPTURED_V2 = _definition_captured_function_graph_v2()
verify_presented_live_runtime_overlay_v2 = _build_public_verifier_entry_v2(
    _CAPTURED_V2["_verify_presented_live_runtime_overlay_implementation_v2"]
)
require_verified_presented_live_runtime_overlay_v2 = _CAPTURED_V2[
    "require_verified_presented_live_runtime_overlay_v2"
]
verify_real_live_runtime_overlay_v2 = _CAPTURED_V2[
    "verify_real_live_runtime_overlay_v2"
]

del _CAPTURED_V2
del _verify_presented_live_runtime_overlay_implementation_v2
del _issue_result_v2
del _require_result_v2
del _build_public_verifier_entry_v2
del _definition_captured_function_graph_v2


__all__ = [
    "SCHEMA_VERSION_V2",
    "STATUS_V2",
    "CLAIM_LEVEL_V2",
    "CANONICALIZATION_V2",
    "ALGORITHM_V2",
    "INSPECTED_AUTHORITY_V2_SOURCE_SHA256",
    "INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256",
    "INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256",
    "INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256",
    "LIVE_RUNTIME_POLICY_SCHEMA_V2",
    "CUSTODY_ATTESTATION_POLICY_SCHEMA_V2",
    "LIVE_RUNTIME_CHALLENGE_SCHEMA_V2",
    "LIVE_RUNTIME_RESPONSE_SCHEMA_V2",
    "LIVE_RUNTIME_REPORT_SCHEMA_V2",
    "LIVE_RUNTIME_CUSTODY_ATTESTATION_SCHEMA_V2",
    "LIVE_RUNTIME_RESULT_SCHEMA_V2",
    "PROCESS_SNAPSHOT_SCHEMA_V2",
    "LIVE_RUNTIME_ANCHOR_MAPPING_SCHEMA_V2",
    "LIVE_RUNTIME_ANCHOR_POLICY_SCHEMA_V2",
    "LIVE_RUNTIME_ANCHOR_REQUEST_SCHEMA_V2",
    "LIVE_RUNTIME_ANCHOR_CHECKPOINT_SCHEMA_V2",
    "LIVE_RUNTIME_ANCHOR_HEAD_SCHEMA_V2",
    "LIVE_RUNTIME_ANCHOR_RECEIPT_SCHEMA_V2",
    "CAS_COMPLETE_BINDING_SCHEMA_V2",
    "EXTERNAL_SIGNATURE_FIELDS_V2",
    "EXTERNAL_SIGNATURE_PREIMAGE_FIELDS_V2",
    "RUNTIME_IDENTITY_FIELDS_V2",
    "POLICY_ALONE_NONCLAIMS_FIELDS_V2",
    "PROCESS_SNAPSHOT_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_MAPPING_FIELDS_V2",
    "LIVE_RUNTIME_CAS_COMPLETE_BINDING_FIELDS_V2",
    "CAS_PLAN_OBJECT_FIELDS_V2",
    "CAS_RETENTION_PLAN_FIELDS_V2",
    "CAS_GENERIC_REQUEST_FIELDS_V2",
    "LIVE_RUNTIME_POLICY_FIELDS_V2",
    "LIVE_RUNTIME_POLICY_BODY_FIELDS_V2",
    "CUSTODY_ATTESTATION_POLICY_FIELDS_V2",
    "CUSTODY_ATTESTATION_POLICY_BODY_FIELDS_V2",
    "LIVE_RUNTIME_CHALLENGE_FIELDS_V2",
    "LIVE_RUNTIME_CHALLENGE_BODY_FIELDS_V2",
    "LIVE_RUNTIME_RESPONSE_FIELDS_V2",
    "LIVE_RUNTIME_RESPONSE_BODY_FIELDS_V2",
    "LIVE_RUNTIME_REPORT_FIELDS_V2",
    "LIVE_RUNTIME_REPORT_BODY_FIELDS_V2",
    "LIVE_RUNTIME_CUSTODY_ATTESTATION_FIELDS_V2",
    "LIVE_RUNTIME_CUSTODY_ATTESTATION_BODY_FIELDS_V2",
    "LIVE_RUNTIME_RESULT_FIELDS_V2",
    "LIVE_RUNTIME_RESULT_BODY_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_POLICY_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_POLICY_BODY_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_REQUEST_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_REQUEST_BODY_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_CHECKPOINT_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_CHECKPOINT_BODY_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_HEAD_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_HEAD_BODY_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_RECEIPT_FIELDS_V2",
    "LIVE_RUNTIME_ANCHOR_RECEIPT_BODY_FIELDS_V2",
    "EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2",
    "CAS_OBJECTS_IN_EXACT_ORDER_V2",
    "CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2",
    "ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2",
    "PRE_CAS_VERIFIED_SIGNATURE_SCOPE_V2",
    "PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2",
    "PRIVATE_OVERLAY_VERIFIED_SIGNATURE_SCOPE_V2",
    "PRIVATE_OVERLAY_VERIFIED_SIGNATURE_COUNT_V2",
    "SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_PURPOSE_V2",
    "ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2",
    "SIGN_LIVE_RUNTIME_ANCHOR_POLICY_PURPOSE_V2",
    "SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_PURPOSE_V2",
    "SIGN_LIVE_RUNTIME_ANCHOR_HEAD_PURPOSE_V2",
    "SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_PURPOSE_V2",
    "Phase2BStep5R01LiveRuntimeOverlayErrorV2",
    "VerifiedPresentedLiveRuntimeOverlayV2",
    "verify_presented_live_runtime_overlay_v2",
    "require_verified_presented_live_runtime_overlay_v2",
    "verify_real_live_runtime_overlay_v2",
]
