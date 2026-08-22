"""Verifier-only candidate R01 evidence contract for Phase-2B Step 5.

This module freezes the nine canonical ``/1`` wire objects needed by the
prospective R01 evidence graph.  The only accepting public composition entry
is deliberately limited to ``CANDIDATE_NONADMITTED`` mechanics: every field
that could assert live execution, external currentness, durability, authority,
R01, readiness, or execution authorization must remain false.

Inputs are exact canonical bytes.  The authority graph is reloaded from the
raw Stage-B lock/profile/registry bytes under two independently supplied lock
pins.  Ed25519 signatures are verified directly; there is no verifier callback.
The external witness policy is separately raw-pinned and its key/control domain
must be disjoint from all eight authority roles.  This pure module imports no
service, store, logical-time producer, owner-status producer, runtime producer,
filesystem, socket, or process-observation API.

Real admission and overlay issuance remain explicitly NOT_IMPLEMENTED.  Thus a
successful result is evidence only that a self-consistent non-admitted wire DAG
and its candidate signatures replayed; it is not a real R01 receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from types import MappingProxyType
from types import FunctionType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    DETACHED_SIGNATURE_FIELDS_V2 as _AUTH_SIGNATURE_FIELDS,
    DETACHED_SIGNATURE_SCHEMA_V2 as _AUTH_SIGNATURE_SCHEMA,
    READINESS_DAG_ID_V1 as _READINESS_DAG_ID,
    READINESS_NODE_ORDER_V1 as _AUTH_READINESS_NODES,
    ROLE_ORDER_V2 as _AUTH_ROLE_ORDER,
    SERVICE_ORDER_V2 as _AUTH_SERVICE_ORDER,
    Phase2BStep5AuthorityErrorV2 as _AuthorityError,
    load_step5_authority_profile_v2 as _load_authority_profile,
    load_step5_signer_registry_v2 as _load_signer_registry,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_lock,
    phase2b_step5_signature_preimage_v2 as _authority_signature_preimage,
    require_locked_step5_authority_graph_v2 as _require_authority_graph,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical,
)

try:
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - verifier image dependency.
    _InvalidSignature = None
    _Ed25519PublicKey = None


CURRENT_CUT_RESPONSE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-current-cut-response/1"
)
CURRENT_CUT_STATEMENT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-current-cut-statement/1"
)
LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-live-runtime-closure-report/1"
)
LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-live-runtime-closure-bundle/1"
)
SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-owner-status-anti-fork-anchor/1"
)
EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-owner-status-anchor-receipt/1"
)
R01_RECEIPT_BODY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-receipt-body/1"
)
DUAL_SIGNED_R01_RECEIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-dual-signed-r01-receipt/1"
)
R01_READINESS_OVERLAY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-readiness-overlay/1"
)
EXTERNAL_WITNESS_POLICY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-policy/1"
)

CANDIDATE_STATUS_V1: Final = "CANDIDATE_NONADMITTED"
CANDIDATE_CLAIM_LEVEL_V1: Final = (
    "SIGNED_WIRE_MECHANICS_CANDIDATE_NOT_LIVE_NOT_CURRENT_NOT_DURABLE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_CONTENT_ID"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_BINDING"
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_SIGNATURE"
FAIL_TRUST_ANCHOR_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_TRUST_ANCHOR"
FAIL_CLAIM_BOUNDARY_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_CLAIM_BOUNDARY"
FAIL_NOT_IMPLEMENTED_V1: Final = "FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_NOT_IMPLEMENTED"

CURRENT_CUT_RESPONSE_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "stage_b_trust_lock_id", "authority_profile_id",
    "signer_registry_id", "attempt_context_id", "governance_service_binding_id",
    "cut_request_nonce", "owner_status_ledger_id", "current_head_id",
    "current_head_payload_sha256", "complete_event_set_root", "projection_id",
    "projection_payload_sha256", "event_count", "latest_sequence",
    "latest_effective_status", "revocation_count", "terminal", "response_id",
})
CURRENT_CUT_STATEMENT_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "stage_b_trust_lock_id",
    "authority_profile_id", "signer_registry_id", "attempt_context_id",
    "current_cut_response_id", "current_cut_response_payload_sha256",
    "governance_peer_observation", "owner_status_binding",
    "pre_cut_logical_time_binding", "statement_id",
})
GOVERNANCE_PEER_OBSERVATION_FIELDS_V1: Final = frozenset({
    "service_name", "service_binding_id", "socket_endpoint_id", "request_nonce",
    "request_payload_sha256", "response_payload_sha256", "peer_pid", "peer_uid",
    "peer_gid", "runtime_instance_id",
})
OWNER_STATUS_BINDING_FIELDS_V1: Final = frozenset({
    "ledger_id", "current_head_id", "current_head_payload_sha256",
    "complete_event_set_root", "projection_id", "projection_payload_sha256",
    "event_count", "latest_sequence", "latest_effective_status",
    "revocation_count", "terminal",
})
PRE_CUT_LOGICAL_TIME_BINDING_FIELDS_V1: Final = frozenset({
    "ledger_id", "final_head_id", "final_head_payload_sha256", "last_token_id",
    "next_sequence",
})
SIGNED_WRAPPER_FIELDS_V1: Final = frozenset({"schema_version", "body", "report_id", "signature"})
LIVE_RUNTIME_CLOSURE_REPORT_BODY_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "stage_b_trust_lock_id",
    "authority_profile_id", "signer_registry_id", "attempt_context_id",
    "service_name", "service_binding_id", "role", "role_key_id", "key_epoch",
    "closure_policy_id", "closure_policy_payload_sha256",
    "image_closure_evidence_id", "image_closure_evidence_payload_sha256",
    "oci_manifest_digest", "runtime_instance_id", "challenge_nonce",
    "entry_observation", "exit_observation", "live_protocol_evidence_binding",
    "service_image_executed", "entry_exit_closure_equal", "uid_gid_verified",
    "control_domain_verified", "report_body_claim_boundary",
})
RUNTIME_OBSERVATION_FIELDS_V1: Final = frozenset({
    "pid", "process_start_ticks", "uid", "gid", "pid_namespace_inode",
    "user_namespace_inode", "mount_namespace_inode", "network_namespace_inode",
    "cgroup_v2_path", "rootfs_mount_id", "executable_sha256", "argv_sha256",
    "environment_sha256", "fd_table_sha256", "python_import_manifest_sha256",
    "native_mapping_manifest_sha256", "closure_projection_sha256",
})
LIVE_PROTOCOL_EVIDENCE_BINDING_FIELDS_V1: Final = frozenset({
    "artifact_kind", "artifact_id", "artifact_payload_sha256", "signature_purpose",
    "signer_key_id", "observed_runtime_instance_id",
})
LIVE_RUNTIME_CLOSURE_BUNDLE_FIELDS_V1: Final = frozenset({
    "schema_version", "body", "bundle_id", "signature",
})
LIVE_RUNTIME_CLOSURE_BUNDLE_BODY_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "stage_b_trust_lock_id",
    "authority_profile_id", "signer_registry_id", "attempt_context_id",
    "service_names_in_exact_order", "service_binding_ids_in_exact_order",
    "report_ids_in_exact_order", "report_payload_sha256s_in_exact_order",
    "runtime_instance_ids_in_exact_order", "uids_in_exact_order",
    "gids_in_exact_order", "control_domains_in_exact_order", "report_count",
    "all_images_executed", "all_entry_exit_closures_equal",
    "all_uids_gids_pairwise_distinct", "all_control_domains_pairwise_distinct",
    "all_live_protocol_evidence_bound", "bundle_claim_boundary",
})
SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_FIELDS_V1: Final = frozenset({
    "schema_version", "body", "anchor_id", "signature",
})
SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "stage_b_trust_lock_id",
    "authority_profile_id", "signer_registry_id", "attempt_context_id",
    "current_cut_statement_id", "current_cut_statement_payload_sha256",
    "current_cut_token_id", "current_cut_token_payload_sha256",
    "current_cut_checkpoint_id", "current_cut_checkpoint_payload_sha256",
    "current_cut_head_id", "current_cut_head_payload_sha256",
    "owner_status_ledger_id", "owner_current_head_id", "complete_event_set_root",
    "projection_id", "anchor_sequence", "previous_external_anchor_receipt_id",
    "external_witness_policy_id", "anchor_nonce",
})
EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1: Final = frozenset({
    "schema_version", "external_witness_policy_id", "witness_namespace",
    "witness_control_domain_id", "witness_key_id", "witness_key_epoch",
    "anchor_id", "anchor_payload_sha256", "witness_sequence",
    "previous_external_receipt_id", "leaf_index", "tree_size",
    "signed_tree_root_sha256", "inclusion_path_sha256_in_exact_order",
    "algorithm", "signature_hex", "receipt_id",
})
EXTERNAL_WITNESS_POLICY_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "external_witness_policy_id",
    "witness_namespace", "witness_control_domain_id", "witness_key_id",
    "witness_key_epoch", "algorithm", "public_key_hex",
})
R01_RECEIPT_BODY_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "stage_b_trust_lock_id",
    "authority_profile_id", "signer_registry_id", "readiness_dag_id",
    "attempt_context_id", "owner_authorization_binding", "owner_status_binding",
    "current_cut_binding", "anti_fork_binding", "runtime_closure_bundle_binding",
    "cas_complete_session_binding", "service_key_custody_evidence_in_exact_order",
    "readiness_effect", "r01_verdict", "overlay_retention_plan_id",
    "receipt_nonce", "receipt_body_id",
})
R01_VERDICT_FIELDS_V1: Final = frozenset({
    "raw_authority_graph_verified", "owner_status_chain_verified",
    "latest_status_active", "revocation_count_zero",
    "current_cut_logical_time_verified", "external_anti_fork_anchor_verified",
    "five_live_runtime_closures_verified", "five_distinct_uids_gids_verified",
    "five_distinct_control_domains_verified", "five_service_key_custodies_verified",
    "cas_complete_durable_session_verified", "producer_free_replay_verified",
    "r01_satisfied", "r02_or_later_satisfied", "execution_authorized",
})
OWNER_AUTHORIZATION_BINDING_FIELDS_V1: Final = frozenset({
    "authorization_schema", "authorization_id", "authorization_config_sha256",
    "authorization_source_sha256", "protocol_id", "exact_freeze_version",
    "exact_freeze_id", "authority_class", "authority_requirement_id",
    "actor_profile_content_id", "actor_profile_canonical_sha256",
    "logical_stage_wire_schema_id", "logical_stage_wire_policy_id",
    "stage_event_contract_manifest_id",
})
CURRENT_CUT_BINDING_FIELDS_V1: Final = frozenset({
    "current_cut_response_id", "current_cut_response_payload_sha256",
    "current_cut_statement_id", "current_cut_statement_payload_sha256",
    "current_cut_token_id", "current_cut_token_payload_sha256",
    "current_cut_checkpoint_id", "current_cut_checkpoint_payload_sha256",
    "current_cut_head_id", "current_cut_head_payload_sha256",
})
ANTI_FORK_BINDING_FIELDS_V1: Final = frozenset({
    "anchor_id", "anchor_payload_sha256", "external_witness_policy_id",
    "external_anchor_receipt_id", "external_anchor_receipt_payload_sha256",
    "witness_sequence", "signed_tree_root_sha256",
})
RUNTIME_CLOSURE_BUNDLE_BINDING_FIELDS_V1: Final = frozenset({
    "bundle_id", "bundle_payload_sha256", "report_ids_in_exact_order",
    "report_payload_sha256s_in_exact_order",
})
CAS_COMPLETE_SESSION_BINDING_FIELDS_V1: Final = frozenset({
    "session_id", "publication_plan_id", "publication_plan_payload_sha256",
    "request_ids_in_exact_order", "receipt_ids_in_exact_order",
    "checkpoint_id", "checkpoint_payload_sha256", "head_id", "head_payload_sha256",
    "response_id", "response_payload_sha256", "object_count",
})
SERVICE_KEY_CUSTODY_EVIDENCE_FIELDS_V1: Final = frozenset({
    "service_name", "service_binding_id", "role", "role_key_id", "key_epoch",
    "runtime_instance_id", "custody_artifact_id", "custody_artifact_payload_sha256",
    "custody_verified",
})
RECEIPT_READINESS_EFFECT_FIELDS_V1: Final = frozenset({
    "r01_satisfied", "r02_or_later_satisfied", "execution_authorized",
})
DUAL_SIGNED_R01_RECEIPT_FIELDS_V1: Final = frozenset({
    "schema_version", "body", "receipt_body_id", "custodian_signature",
    "pure_countersignature", "receipt_id",
})
PURE_COUNTERSIGN_SUBJECT_FIELDS_V1: Final = frozenset({
    "schema_version", "receipt_body_id", "receipt_body_payload_sha256",
    "custodian_signature_payload_sha256",
})
R01_READINESS_OVERLAY_FIELDS_V1: Final = frozenset({
    "schema_version", "status", "claim_level", "stage_b_trust_lock_id",
    "authority_profile_id", "signer_registry_id", "attempt_context_id",
    "readiness_effect", "r01_receipt_id", "r01_receipt_payload_sha256",
    "overlay_retention_plan_id", "overlay_id",
})
R01_READINESS_EFFECT_FIELDS_V1: Final = frozenset({
    "readiness_dag_id", "base_readiness_candidate_id", "overlay_sequence",
    "previous_overlay_id", "transitioned_node_id",
    "node_states_in_canonical_topological_order", "satisfied_node_count",
    "highest_contiguous_satisfied_node", "all_nodes_satisfied", "r01_satisfied",
    "r02_or_later_satisfied", "execution_authorized",
})

_SCHEMAS: Final = MappingProxyType({
    "current_cut_response": CURRENT_CUT_RESPONSE_SCHEMA_V1,
    "current_cut_statement": CURRENT_CUT_STATEMENT_SCHEMA_V1,
    "live_runtime_closure_report": LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1,
    "live_runtime_closure_bundle": LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1,
    "signed_owner_status_anti_fork_anchor": SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1,
    "external_owner_status_anchor_receipt": EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1,
    "r01_receipt_body": R01_RECEIPT_BODY_SCHEMA_V1,
    "dual_signed_r01_receipt": DUAL_SIGNED_R01_RECEIPT_SCHEMA_V1,
    "r01_readiness_overlay": R01_READINESS_OVERLAY_SCHEMA_V1,
})
_PREFIXES: Final = MappingProxyType({kind: f"phase2b_step5_{kind}_v1_" for kind in _SCHEMAS})
_DOMAINS: Final = MappingProxyType({
    kind: f"HEGEL/PHASE2B/STEP5/{kind.upper()}/V1\0".encode("ascii")
    for kind in _SCHEMAS
})
_EXTERNAL_POLICY_PREFIX: Final = "phase2b_step5_external_witness_policy_v1_"
_EXTERNAL_POLICY_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
_EXTERNAL_RECEIPT_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT/V1\0"
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX128 = re.compile(r"[0-9a-f]{128}")
_ID = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")
_MAX_BYTES: Final = 8 * 1024 * 1024
_SERVICE_ORDER: Final = tuple(_AUTH_SERVICE_ORDER)
_ROLE_ORDER: Final = tuple(_AUTH_ROLE_ORDER)
_READINESS_NODES: Final = tuple(_AUTH_READINESS_NODES)
_OWNER_LEDGER_PREFIX: Final = "phase2b_step5_owner_status_ledger_v1_"
_OWNER_HEAD_PREFIX: Final = "phase2b_step5_owner_status_head_v1_"
_OWNER_PROJECTION_PREFIX: Final = "phase2b_step5_owner_status_projection_v1_"
_LOGICAL_TIME_LEDGER_PREFIX: Final = "phase2b_step5_logical_time_ledger_v1_"
_LOGICAL_TIME_TOKEN_PREFIX: Final = "phase2b_step5_logical_time_token_v1_"
_LOGICAL_TIME_CHECKPOINT_PREFIX: Final = "phase2b_step5_logical_time_checkpoint_v1_"
_LOGICAL_TIME_HEAD_PREFIX: Final = "phase2b_step5_logical_time_head_v1_"
_MERKLE_LEAF_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/MERKLE_LEAF/V1\0"


class Phase2BStep5R01EvidenceContractErrorV1(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01EvidenceContractErrorV1(code, detail)


def _canonical(value: object, _canon=_canonicalize) -> bytes:
    try:
        return _canon(value)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_CANONICAL_V1, "RFC-8785 canonicalization failed"
        ) from exc


def _restore_wire_integers(value: object) -> object:
    if type(value) is float and math.isfinite(value) and value.is_integer():
        if abs(value) > (1 << 53) - 1:
            _fail(FAIL_SCHEMA_V1, "decoded integer exceeds the I-JSON safe range")
        return int(value)
    if type(value) is list:
        return [_restore_wire_integers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_wire_integers(item) for key, item in value.items()}
    return value


def _decode(payload: bytes, name: str, _decoder=_decode_canonical) -> dict[str, object]:
    if type(payload) is not bytes or not 1 <= len(payload) <= _MAX_BYTES or payload.endswith(b"\n"):
        _fail(FAIL_INPUT_V1, f"{name} must be bounded exact bytes without newline")
    try:
        decoded = _decoder(payload)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_CANONICAL_V1, f"{name} is not exact canonical RFC-8785"
        ) from exc
    value = _restore_wire_integers(decoded)
    if type(value) is not dict:
        _fail(FAIL_SCHEMA_V1, f"{name} must be an object")
    return value


def _object(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(FAIL_SCHEMA_V1, f"{name} keyset differs")
    return value


def _text(value: object, name: str, maximum: int = 4096) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        _fail(FAIL_SCHEMA_V1, f"{name} must be bounded nonempty text")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > (1 << 53) - 1:
        _fail(FAIL_SCHEMA_V1, f"{name} must be a safe integer >= {minimum}")
    return value


def _sha(value: object, name: str) -> str:
    text = _text(value, name, 64)
    if _HEX64.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V1, f"{name} must be lowercase SHA-256 hex")
    return text


def _sha_address(value: object, name: str) -> str:
    text = _text(value, name, 71)
    if not text.startswith("sha256:") or _HEX64.fullmatch(text[7:]) is None:
        _fail(FAIL_SCHEMA_V1, f"{name} must be sha256:<lowercase SHA-256 hex>")
    return text


def _identifier(value: object, name: str, prefix: str | None = None) -> str:
    text = _text(value, name, 320)
    if _ID.fullmatch(text) is None or (prefix is not None and not text.startswith(prefix)):
        _fail(FAIL_SCHEMA_V1, f"{name} identifier differs")
    return text


def _false(value: object, name: str) -> None:
    if value is not False:
        _fail(FAIL_CLAIM_BOUNDARY_V1, f"{name} must remain false in candidate mechanics")


def _candidate_header(body: Mapping[str, object], schema: str, name: str) -> None:
    if body.get("schema_version") != schema or body.get("status") != CANDIDATE_STATUS_V1:
        _fail(FAIL_SCHEMA_V1, f"{name} candidate header differs")
    if "claim_level" in body and body["claim_level"] != CANDIDATE_CLAIM_LEVEL_V1:
        _fail(FAIL_CLAIM_BOUNDARY_V1, f"{name} claim level overstates evidence")


def _id(kind: str, subject: object, _canon=_canonical) -> str:
    if type(kind) is not str:
        _fail(FAIL_INPUT_V1, "unknown R01 evidence content-ID kind")
    try:
        domain = _DOMAINS[kind]
        prefix = _PREFIXES[kind]
    except KeyError:
        _fail(FAIL_INPUT_V1, "unknown R01 evidence content-ID kind")
    return prefix + hashlib.sha256(domain + _canon(subject)).hexdigest()


def phase2b_step5_r01_evidence_content_id_v1(kind: str, subject: object) -> str:
    """Derive one of the nine frozen content IDs from its ID subject."""
    return _id(kind, subject)


def canonical_r01_evidence_contract_json_bytes_v1(value: object) -> bytes:
    return _canonical(value)


def phase2b_step5_r01_authority_signature_preimage_v1(
    *,
    artifact_kind: str,
    body: object,
    role: str,
    purpose: str,
    signer_key_id: str,
    key_epoch: int = 1,
    _preimage=_authority_signature_preimage,
) -> bytes:
    """Build the frozen authority-v2 role/purpose signature preimage."""
    try:
        return _preimage(
            artifact_kind=artifact_kind,
            body=body,
            role=role,
            purpose=purpose,
            signer_key_id=signer_key_id,
            key_epoch=key_epoch,
        )
    except _AuthorityError as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_SIGNATURE_V1, "authority-v2 signature preimage rejected"
        ) from exc


def phase2b_step5_external_anchor_receipt_signature_preimage_v1(
    receipt_without_signature_or_with_signature: object,
) -> bytes:
    """Build the external witness preimage, including the derived receipt ID."""
    if type(receipt_without_signature_or_with_signature) is not dict:
        _fail(FAIL_SCHEMA_V1, "external receipt preimage input must be an object")
    keys = frozenset(receipt_without_signature_or_with_signature)
    expected_unsigned = EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1 - {"signature_hex"}
    if keys == EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1:
        unsigned = _without(receipt_without_signature_or_with_signature, "signature_hex")
    elif keys == expected_unsigned:
        unsigned = dict(receipt_without_signature_or_with_signature)
    else:
        _fail(FAIL_SCHEMA_V1, "external receipt unsigned preimage keyset differs")
    return _EXTERNAL_RECEIPT_SIGNATURE_DOMAIN + _canonical(unsigned)


def phase2b_step5_r01_pure_countersign_subject_v1(
    *,
    receipt_body_id: str,
    receipt_body_payload: bytes,
    custodian_signature: object,
) -> dict[str, object]:
    """Build the exact four-field subject bound by the PURE countersignature."""
    _identifier(receipt_body_id, "R01 receipt body ID", _PREFIXES["r01_receipt_body"])
    if type(receipt_body_payload) is not bytes:
        _fail(FAIL_INPUT_V1, "R01 receipt-body payload must be exact bytes")
    signature = _object(custodian_signature, frozenset(_AUTH_SIGNATURE_FIELDS), "custodian signature")
    return {
        "schema_version": "hegel-machine-phase2b-formal-wire-covert-step5-r01-pure-countersign-subject/1",
        "receipt_body_id": receipt_body_id,
        "receipt_body_payload_sha256": _payload_sha(receipt_body_payload),
        "custodian_signature_payload_sha256": hashlib.sha256(_canonical(signature)).hexdigest(),
    }


def _without(mapping: Mapping[str, object], *names: str) -> dict[str, object]:
    return {key: value for key, value in mapping.items() if key not in names}


def _payload_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _same(value: object, expected: object, name: str) -> None:
    def equal(left: object, right: object) -> bool:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = tuple(left.keys())
            right_keys = tuple(right.keys())
            return (
                all(type(key) is str for key in left_keys + right_keys)
                and frozenset(left_keys) == frozenset(right_keys)
                and all(equal(left[key], right[key]) for key in left_keys)
            )
        if type(left) is list and type(right) in (list, tuple):
            return len(left) == len(right) and all(
                equal(item, other) for item, other in zip(left, right, strict=True)
            )
        if type(left) is tuple and type(right) in (list, tuple):
            return len(left) == len(right) and all(
                equal(item, other) for item, other in zip(left, right, strict=True)
            )
        return type(left) is type(right) and left == right

    if not equal(value, expected):
        _fail(FAIL_BINDING_V1, f"{name} differs")


def _registry_key(registry: object, role: str) -> Mapping[str, object]:
    try:
        rows = registry.body["role_keys_in_exact_order"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_TRUST_ANCHOR_V1, "verified registry projection unavailable"
        ) from exc
    if type(rows) is not tuple or len(rows) != len(_ROLE_ORDER):
        _fail(FAIL_TRUST_ANCHOR_V1, "verified registry role-key order differs")
    for expected_role, row in zip(_ROLE_ORDER, rows, strict=True):
        if type(row) is not MappingProxyType or row.get("role") != expected_role:
            _fail(FAIL_TRUST_ANCHOR_V1, "verified registry role-key shape differs")
        if expected_role == role:
            return row
    _fail(FAIL_TRUST_ANCHOR_V1, "required role absent from registry")


def _service_bindings(profile: object) -> tuple[Mapping[str, object], ...]:
    try:
        rows = profile.body["service_static_bindings_in_exact_order"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_TRUST_ANCHOR_V1, "verified profile service projection unavailable"
        ) from exc
    if type(rows) is not tuple or len(rows) != len(_SERVICE_ORDER):
        _fail(FAIL_TRUST_ANCHOR_V1, "verified service binding order differs")
    return rows


def _validate_detached_signature(
    value: object,
    *,
    registry: object,
    role: str,
    purpose: str,
    artifact_kind: str,
    body: object,
    _preimage=_authority_signature_preimage,
    _backend_loader=(
        None if _Ed25519PublicKey is None else _Ed25519PublicKey.from_public_bytes
    ),
    _invalid_signature=_InvalidSignature,
) -> dict[str, object]:
    signature = _object(value, frozenset(_AUTH_SIGNATURE_FIELDS), "detached signature")
    key = _registry_key(registry, role)
    if (
        signature["schema_version"] != _AUTH_SIGNATURE_SCHEMA
        or signature["algorithm"] != "Ed25519"
        or signature["role"] != role
        or signature["purpose"] != purpose
        or signature["key_epoch"] != key["key_epoch"]
        or signature["signer_key_id"] != key["key_id"]
        or purpose not in key["authorized_signature_purposes"]
    ):
        _fail(FAIL_SIGNATURE_V1, "detached signature authority metadata differs")
    signature_hex = _text(signature["signature_hex"], "signature_hex", 128)
    if _HEX128.fullmatch(signature_hex) is None:
        _fail(FAIL_SIGNATURE_V1, "detached Ed25519 signature encoding differs")
    if _backend_loader is None or _invalid_signature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        preimage = _preimage(
            artifact_kind=artifact_kind,
            body=body,
            role=role,
            purpose=purpose,
            signer_key_id=key["key_id"],
            key_epoch=key["key_epoch"],
        )
        public_key = _backend_loader(bytes.fromhex(key["public_key_hex"]))
        public_key.verify(bytes.fromhex(signature_hex), preimage)
    except Exception as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_SIGNATURE_V1, "detached Ed25519 signature rejected"
        ) from exc
    return signature


def _load_external_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_digest: bytes,
    registry: object,
) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAX_BYTES
        or payload.endswith(b"\n")
    ):
        _fail(
            FAIL_INPUT_V1,
            "external witness policy must be bounded exact bytes without newline",
        )
    if (
        type(expected_id_bytes) is not bytes
        or not 1 <= len(expected_id_bytes) <= 320
        or type(expected_payload_digest) is not bytes
    ):
        _fail(FAIL_INPUT_V1, "external policy OOB pins must be raw bytes")
    if (
        len(expected_payload_digest) != 32
        or hashlib.sha256(payload).digest() != expected_payload_digest
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "external policy raw SHA-256 pin differs")
    try:
        expected_id = expected_id_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_TRUST_ANCHOR_V1, "external policy OOB ID is not ASCII"
        ) from exc
    expected_id = _identifier(
        expected_id, "external witness policy OOB ID", _EXTERNAL_POLICY_PREFIX
    )
    policy = _object(
        _decode(payload, "external witness policy"),
        EXTERNAL_WITNESS_POLICY_FIELDS_V1,
        "external witness policy",
    )
    schema_version = _text(
        policy["schema_version"], "external witness policy schema", 256
    )
    status = _text(policy["status"], "external witness policy status", 256)
    claim_level = _text(
        policy["claim_level"], "external witness policy claim level", 512
    )
    algorithm = _text(
        policy["algorithm"], "external witness policy algorithm", 32
    )
    key_epoch = _integer(
        policy["witness_key_epoch"], "external witness key epoch", 1
    )
    if (
        schema_version != EXTERNAL_WITNESS_POLICY_SCHEMA_V1
        or status != CANDIDATE_STATUS_V1
        or claim_level != CANDIDATE_CLAIM_LEVEL_V1
        or algorithm != "Ed25519"
        or key_epoch != 1
    ):
        _fail(FAIL_SCHEMA_V1, "external witness policy header differs")
    public_hex = _text(policy["public_key_hex"], "external public key", 64)
    if _HEX64.fullmatch(public_hex) is None:
        _fail(FAIL_SCHEMA_V1, "external witness public key encoding differs")
    policy_id_value = _identifier(
        policy["external_witness_policy_id"],
        "external witness policy ID",
        _EXTERNAL_POLICY_PREFIX,
    )
    witness_key_id = _identifier(
        policy["witness_key_id"], "external witness key ID"
    )
    witness_control_domain_id = _identifier(
        policy["witness_control_domain_id"],
        "external witness control-domain ID",
    )
    witness_namespace = _text(
        policy["witness_namespace"], "external witness namespace", 256
    )
    subject = _without(policy, "external_witness_policy_id")
    policy_id = _EXTERNAL_POLICY_PREFIX + hashlib.sha256(
        _EXTERNAL_POLICY_DOMAIN + _canonical(subject)
    ).hexdigest()
    if policy_id_value != policy_id or expected_id != policy_id:
        _fail(FAIL_TRUST_ANCHOR_V1, "external witness policy content ID differs")
    role_keys = tuple(_registry_key(registry, role) for role in _ROLE_ORDER)
    if (
        public_hex in {row["public_key_hex"] for row in role_keys}
        or witness_key_id in {row["key_id"] for row in role_keys}
        or witness_control_domain_id
        in {row["control_domain"] for row in role_keys}
        or witness_namespace in set(_ROLE_ORDER)
        or witness_control_domain_id in set(_ROLE_ORDER)
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "external witness is not disjoint from eight authority roles")
    return policy


def _verify_external_receipt_signature(
    receipt: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    _backend_loader=(
        None if _Ed25519PublicKey is None else _Ed25519PublicKey.from_public_bytes
    ),
    _invalid_signature=_InvalidSignature,
) -> None:
    if _backend_loader is None or _invalid_signature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        public_key = _backend_loader(bytes.fromhex(policy["public_key_hex"]))
        public_key.verify(
            bytes.fromhex(receipt["signature_hex"]),
            phase2b_step5_external_anchor_receipt_signature_preimage_v1(dict(receipt)),
        )
    except Exception as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_SIGNATURE_V1, "external witness Ed25519 signature rejected"
        ) from exc


@dataclass(frozen=True, slots=True, init=False)
class VerifiedR01EvidenceContractCandidateV1:
    artifact_ids: Mapping[str, object]
    payload_sha256s: Mapping[str, object]
    claim_boundary: str
    candidate_signature_mechanics_verified: bool
    external_currentness_verified: bool
    live_runtime_closure_verified: bool
    external_anti_fork_anchor_verified: bool
    durable_cas_verified: bool
    authority_established: bool
    r01_satisfied: bool
    readiness_overlay_admitted: bool
    execution_authorized: bool

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiedR01EvidenceContractCandidateV1":
        raise TypeError("verified R01 candidate results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified R01 candidate results are privately issued")


def _build_result_lineage():
    result_type = VerifiedR01EvidenceContractCandidateV1
    issued: dict[int, tuple[object, Mapping[str, object], Mapping[str, object]]] = {}

    def issue(ids: Mapping[str, object], shas: Mapping[str, object]) -> VerifiedR01EvidenceContractCandidateV1:
        value = object.__new__(result_type)
        frozen_ids = MappingProxyType(dict(ids))
        frozen_shas = MappingProxyType(dict(shas))
        object.__setattr__(value, "artifact_ids", frozen_ids)
        object.__setattr__(value, "payload_sha256s", frozen_shas)
        object.__setattr__(value, "claim_boundary", CANDIDATE_CLAIM_LEVEL_V1)
        object.__setattr__(value, "candidate_signature_mechanics_verified", True)
        object.__setattr__(value, "external_currentness_verified", False)
        object.__setattr__(value, "live_runtime_closure_verified", False)
        object.__setattr__(value, "external_anti_fork_anchor_verified", False)
        object.__setattr__(value, "durable_cas_verified", False)
        object.__setattr__(value, "authority_established", False)
        object.__setattr__(value, "r01_satisfied", False)
        object.__setattr__(value, "readiness_overlay_admitted", False)
        object.__setattr__(value, "execution_authorized", False)
        issued[id(value)] = (value, frozen_ids, frozen_shas)
        return value

    def require(value: object) -> VerifiedR01EvidenceContractCandidateV1:
        if type(value) is not result_type:
            _fail(FAIL_INPUT_V1, "verified R01 candidate result required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            _fail(FAIL_INPUT_V1, "R01 candidate result lacks private lineage")
        if (
            value.artifact_ids is not record[1]
            or value.payload_sha256s is not record[2]
            or value.claim_boundary != CANDIDATE_CLAIM_LEVEL_V1
            or value.candidate_signature_mechanics_verified is not True
            or value.external_currentness_verified is not False
            or value.live_runtime_closure_verified is not False
            or value.external_anti_fork_anchor_verified is not False
            or value.durable_cas_verified is not False
            or value.authority_established is not False
            or value.r01_satisfied is not False
            or value.readiness_overlay_admitted is not False
            or value.execution_authorized is not False
        ):
            _fail(FAIL_INPUT_V1, "R01 candidate result lineage was mutated")
        return value

    return issue, require


_issue_result, _require_result = _build_result_lineage()
del _build_result_lineage


def _authority_ids(lock: object, profile: object, registry: object) -> tuple[str, str, str]:
    return lock.artifact_id, profile.artifact_id, registry.artifact_id


def _require_authority_refs(
    body: Mapping[str, object], ids: tuple[str, str, str], name: str
) -> None:
    fields = ("stage_b_trust_lock_id", "authority_profile_id", "signer_registry_id")
    if tuple(body[field] for field in fields) != ids:
        _fail(FAIL_BINDING_V1, f"{name} raw authority graph references differ")


def load_current_cut_response_v1(
    payload: bytes, *, authority_ids: tuple[str, str, str], expected_attempt_context_id: str
) -> dict[str, object]:
    body = _object(_decode(payload, "current-cut response"), CURRENT_CUT_RESPONSE_FIELDS_V1, "current-cut response")
    _candidate_header(body, CURRENT_CUT_RESPONSE_SCHEMA_V1, "current-cut response")
    _require_authority_refs(body, authority_ids, "current-cut response")
    if body["attempt_context_id"] != expected_attempt_context_id:
        _fail(FAIL_BINDING_V1, "current-cut response attempt differs")
    _identifier(body["governance_service_binding_id"], "current-cut governance binding")
    _identifier(body["owner_status_ledger_id"], "current-cut owner ledger", _OWNER_LEDGER_PREFIX)
    _identifier(body["current_head_id"], "current-cut owner head", _OWNER_HEAD_PREFIX)
    _identifier(body["projection_id"], "current-cut owner projection", _OWNER_PROJECTION_PREFIX)
    for field in ("current_head_payload_sha256", "complete_event_set_root", "projection_payload_sha256"):
        _sha_address(body[field], f"current-cut response {field}")
    _text(body["cut_request_nonce"], "current-cut request nonce", 256)
    event_count = _integer(body["event_count"], "event_count", 1)
    latest_sequence = _integer(body["latest_sequence"], "latest_sequence")
    revocations = _integer(body["revocation_count"], "revocation_count")
    if latest_sequence != event_count - 1 or body["latest_effective_status"] != "ACTIVE" or revocations != 0:
        _fail(FAIL_CLAIM_BOUNDARY_V1, "candidate current-cut response is not an ACTIVE zero-revocation projection")
    _false(body["terminal"], "current-cut response terminal")
    expected = _id("current_cut_response", _without(body, "response_id"))
    if body["response_id"] != expected:
        _fail(FAIL_CONTENT_ID_V1, "current-cut response ID differs")
    return body


def _owner_binding_from_response(response: Mapping[str, object]) -> dict[str, object]:
    return {
        "ledger_id": response["owner_status_ledger_id"],
        "current_head_id": response["current_head_id"],
        "current_head_payload_sha256": response["current_head_payload_sha256"],
        "complete_event_set_root": response["complete_event_set_root"],
        "projection_id": response["projection_id"],
        "projection_payload_sha256": response["projection_payload_sha256"],
        "event_count": response["event_count"],
        "latest_sequence": response["latest_sequence"],
        "latest_effective_status": response["latest_effective_status"],
        "revocation_count": response["revocation_count"],
        "terminal": response["terminal"],
    }


def load_current_cut_statement_v1(
    payload: bytes,
    *,
    authority_ids: tuple[str, str, str],
    expected_attempt_context_id: str,
    current_cut_response_payload: bytes,
    current_cut_response: Mapping[str, object],
) -> dict[str, object]:
    body = _object(_decode(payload, "current-cut statement"), CURRENT_CUT_STATEMENT_FIELDS_V1, "current-cut statement")
    _candidate_header(body, CURRENT_CUT_STATEMENT_SCHEMA_V1, "current-cut statement")
    _require_authority_refs(body, authority_ids, "current-cut statement")
    if (
        body["attempt_context_id"] != expected_attempt_context_id
        or body["current_cut_response_id"] != current_cut_response["response_id"]
        or body["current_cut_response_payload_sha256"] != _payload_sha(current_cut_response_payload)
    ):
        _fail(FAIL_BINDING_V1, "current-cut statement response lineage differs")
    peer = _object(body["governance_peer_observation"], GOVERNANCE_PEER_OBSERVATION_FIELDS_V1, "governance peer observation")
    if (
        peer["service_name"] != "governance_status"
        or peer["service_binding_id"] != current_cut_response["governance_service_binding_id"]
        or peer["request_nonce"] != current_cut_response["cut_request_nonce"]
        or peer["response_payload_sha256"] != _payload_sha(current_cut_response_payload)
    ):
        _fail(FAIL_BINDING_V1, "governance peer observation differs from response")
    for field in ("peer_pid", "peer_uid", "peer_gid"):
        _integer(peer[field], f"governance peer {field}", 1)
    for field in ("request_payload_sha256", "response_payload_sha256"):
        _sha(peer[field], f"governance peer {field}")
    for field in ("socket_endpoint_id", "runtime_instance_id"):
        _identifier(peer[field], f"governance peer {field}")
    owner = _object(body["owner_status_binding"], OWNER_STATUS_BINDING_FIELDS_V1, "owner-status binding")
    _same(owner, _owner_binding_from_response(current_cut_response), "current-cut owner-status projection")
    time_binding = _object(body["pre_cut_logical_time_binding"], PRE_CUT_LOGICAL_TIME_BINDING_FIELDS_V1, "pre-cut logical-time binding")
    _identifier(time_binding["ledger_id"], "pre-cut logical-time ledger", _LOGICAL_TIME_LEDGER_PREFIX)
    _identifier(time_binding["final_head_id"], "pre-cut logical-time head", _LOGICAL_TIME_HEAD_PREFIX)
    _identifier(time_binding["last_token_id"], "pre-cut logical-time token", _LOGICAL_TIME_TOKEN_PREFIX)
    _sha_address(time_binding["final_head_payload_sha256"], "pre-cut final head payload SHA")
    _integer(time_binding["next_sequence"], "pre-cut next sequence", 1)
    expected = _id("current_cut_statement", _without(body, "statement_id"))
    if body["statement_id"] != expected:
        _fail(FAIL_CONTENT_ID_V1, "current-cut statement ID differs")
    return body


def load_live_runtime_closure_report_v1(
    payload: bytes,
    *,
    authority_ids: tuple[str, str, str],
    expected_attempt_context_id: str,
    expected_service_binding: Mapping[str, object],
    registry: object,
) -> dict[str, object]:
    root = _object(_decode(payload, "live runtime-closure report"), SIGNED_WRAPPER_FIELDS_V1, "live runtime-closure report")
    if root["schema_version"] != LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1:
        _fail(FAIL_SCHEMA_V1, "live report wrapper schema differs")
    body = _object(root["body"], LIVE_RUNTIME_CLOSURE_REPORT_BODY_FIELDS_V1, "live runtime-closure report body")
    _candidate_header(body, LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1, "live runtime-closure report body")
    _require_authority_refs(body, authority_ids, "live report")
    if (
        body["attempt_context_id"] != expected_attempt_context_id
        or body["service_name"] != expected_service_binding["service_name"]
        or body["service_binding_id"] != expected_service_binding["binding_id"]
        or body["role"] != expected_service_binding["role"]
        or body["role_key_id"] != expected_service_binding["role_key_id"]
        or body["key_epoch"] != expected_service_binding["key_epoch"]
        or body["closure_policy_payload_sha256"] != expected_service_binding["static_runtime_closure_policy_sha256"]
        or body["oci_manifest_digest"] != expected_service_binding["oci_manifest_digest"]
    ):
        _fail(FAIL_BINDING_V1, "live report static service binding differs")
    _identifier(body["closure_policy_id"], "live report closure-policy ID")
    _sha(body["closure_policy_payload_sha256"], "live report closure-policy payload SHA")
    _identifier(body["image_closure_evidence_id"], "live report image-evidence ID")
    _sha(body["image_closure_evidence_payload_sha256"], "live report image-evidence payload SHA")
    _identifier(body["runtime_instance_id"], "live report runtime-instance ID")
    _text(body["challenge_nonce"], "live report challenge nonce", 256)
    for nested_name in ("entry_observation", "exit_observation"):
        observation = _object(body[nested_name], RUNTIME_OBSERVATION_FIELDS_V1, nested_name)
        for field in ("pid", "process_start_ticks", "uid", "gid", "pid_namespace_inode", "user_namespace_inode", "mount_namespace_inode", "network_namespace_inode"):
            _integer(observation[field], f"{nested_name}.{field}", 1)
        for field in ("executable_sha256", "argv_sha256", "environment_sha256", "fd_table_sha256", "python_import_manifest_sha256", "native_mapping_manifest_sha256", "closure_projection_sha256"):
            _sha(observation[field], f"{nested_name}.{field}")
        _text(observation["cgroup_v2_path"], f"{nested_name}.cgroup_v2_path", 4096)
        _text(observation["rootfs_mount_id"], f"{nested_name}.rootfs_mount_id", 512)
    protocol = _object(body["live_protocol_evidence_binding"], LIVE_PROTOCOL_EVIDENCE_BINDING_FIELDS_V1, "live protocol evidence binding")
    for field in ("artifact_id", "signer_key_id", "observed_runtime_instance_id"):
        _identifier(protocol[field], f"live protocol {field}")
    _sha(protocol["artifact_payload_sha256"], "live protocol payload SHA")
    _text(protocol["artifact_kind"], "live protocol artifact kind", 256)
    _text(protocol["signature_purpose"], "live protocol signature purpose", 256)
    if protocol["observed_runtime_instance_id"] != body["runtime_instance_id"]:
        _fail(FAIL_BINDING_V1, "live protocol runtime-instance binding differs")
    role_key = _registry_key(registry, expected_service_binding["role"])
    if (
        protocol["signer_key_id"] != role_key["key_id"]
        or protocol["signature_purpose"] not in role_key["authorized_signature_purposes"]
    ):
        _fail(FAIL_BINDING_V1, "live protocol role-key/purpose binding differs")
    for nested_name in ("entry_observation", "exit_observation"):
        observation = body[nested_name]
        if (
            observation["uid"] != expected_service_binding["uid"]
            or observation["gid"] != expected_service_binding["gid"]
        ):
            _fail(FAIL_BINDING_V1, "live report UID/GID differs from static service binding")
    for field in ("service_image_executed", "entry_exit_closure_equal", "uid_gid_verified", "control_domain_verified"):
        _false(body[field], f"live report {field}")
    if body["report_body_claim_boundary"] != CANDIDATE_CLAIM_LEVEL_V1:
        _fail(FAIL_CLAIM_BOUNDARY_V1, "live report body claim boundary differs")
    expected_id = _id("live_runtime_closure_report", body)
    if root["report_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "live report ID differs")
    _validate_detached_signature(
        root["signature"], registry=registry, role="SOURCE_RUNTIME_CUSTODIAN",
        purpose="SIGN_RUNTIME_CLOSURE", artifact_kind="live_runtime_closure_report", body=body,
    )
    return root


def load_live_runtime_closure_bundle_v1(
    payload: bytes,
    *,
    authority_ids: tuple[str, str, str],
    expected_attempt_context_id: str,
    report_payloads_in_exact_order: tuple[bytes, ...],
    reports_in_exact_order: tuple[Mapping[str, object], ...],
    service_bindings_in_exact_order: tuple[Mapping[str, object], ...],
    registry: object,
) -> dict[str, object]:
    root = _object(_decode(payload, "live runtime-closure bundle"), LIVE_RUNTIME_CLOSURE_BUNDLE_FIELDS_V1, "live runtime-closure bundle")
    if root["schema_version"] != LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1:
        _fail(FAIL_SCHEMA_V1, "live bundle wrapper schema differs")
    body = _object(root["body"], LIVE_RUNTIME_CLOSURE_BUNDLE_BODY_FIELDS_V1, "live runtime-closure bundle body")
    _candidate_header(body, LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1, "live runtime-closure bundle body")
    _require_authority_refs(body, authority_ids, "live bundle")
    if body["attempt_context_id"] != expected_attempt_context_id:
        _fail(FAIL_BINDING_V1, "live bundle attempt differs")
    if len(reports_in_exact_order) != 5 or len(report_payloads_in_exact_order) != 5:
        _fail(FAIL_SCHEMA_V1, "live bundle requires exactly five reports")
    report_bodies = tuple(report["body"] for report in reports_in_exact_order)
    runtime_ids = tuple(row["runtime_instance_id"] for row in report_bodies)
    uid_gid_pairs = tuple(
        (row["entry_observation"]["uid"], row["entry_observation"]["gid"])
        for row in report_bodies
    )
    control_domains = tuple(row["control_domain"] for row in service_bindings_in_exact_order)
    if (
        len(set(runtime_ids)) != 5
        or len(set(uid_gid_pairs)) != 5
        or len(set(control_domains)) != 5
    ):
        _fail(FAIL_BINDING_V1, "five-service runtime/UID-GID/control-domain identities collide")
    expected_lists = {
        "service_names_in_exact_order": list(_SERVICE_ORDER),
        "service_binding_ids_in_exact_order": [row["binding_id"] for row in service_bindings_in_exact_order],
        "report_ids_in_exact_order": [row["report_id"] for row in reports_in_exact_order],
        "report_payload_sha256s_in_exact_order": [_payload_sha(raw) for raw in report_payloads_in_exact_order],
        "runtime_instance_ids_in_exact_order": list(runtime_ids),
        "uids_in_exact_order": [row["entry_observation"]["uid"] for row in report_bodies],
        "gids_in_exact_order": [row["entry_observation"]["gid"] for row in report_bodies],
        "control_domains_in_exact_order": list(control_domains),
    }
    for field, expected in expected_lists.items():
        if type(body[field]) is not list:
            _fail(FAIL_SCHEMA_V1, f"live bundle {field} must be an exact list")
        _same(body[field], expected, f"live bundle {field}")
    if body["report_count"] != 5:
        _fail(FAIL_SCHEMA_V1, "live bundle report count differs")
    for field in (
        "all_images_executed", "all_entry_exit_closures_equal",
        "all_uids_gids_pairwise_distinct", "all_control_domains_pairwise_distinct",
        "all_live_protocol_evidence_bound",
    ):
        _false(body[field], f"live bundle {field}")
    if body["bundle_claim_boundary"] != CANDIDATE_CLAIM_LEVEL_V1:
        _fail(FAIL_CLAIM_BOUNDARY_V1, "live bundle claim boundary differs")
    expected_id = _id("live_runtime_closure_bundle", body)
    if root["bundle_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "live runtime-closure bundle ID differs")
    _validate_detached_signature(
        root["signature"], registry=registry, role="SOURCE_RUNTIME_CUSTODIAN",
        purpose="SIGN_RUNTIME_CLOSURE", artifact_kind="live_runtime_closure_bundle", body=body,
    )
    return root


def load_signed_owner_status_anti_fork_anchor_v1(
    payload: bytes,
    *,
    authority_ids: tuple[str, str, str],
    expected_attempt_context_id: str,
    statement_payload: bytes,
    statement: Mapping[str, object],
    external_policy_id: str,
    registry: object,
) -> dict[str, object]:
    root = _object(_decode(payload, "signed owner-status anti-fork anchor"), SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_FIELDS_V1, "signed owner-status anti-fork anchor")
    if root["schema_version"] != SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1:
        _fail(FAIL_SCHEMA_V1, "anti-fork anchor wrapper schema differs")
    body = _object(root["body"], SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1, "anti-fork anchor body")
    _candidate_header(body, SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1, "anti-fork anchor body")
    _require_authority_refs(body, authority_ids, "anti-fork anchor")
    owner = statement["owner_status_binding"]
    if (
        body["attempt_context_id"] != expected_attempt_context_id
        or body["current_cut_statement_id"] != statement["statement_id"]
        or body["current_cut_statement_payload_sha256"] != _payload_sha(statement_payload)
        or body["owner_status_ledger_id"] != owner["ledger_id"]
        or body["owner_current_head_id"] != owner["current_head_id"]
        or body["complete_event_set_root"] != owner["complete_event_set_root"]
        or body["projection_id"] != owner["projection_id"]
        or body["external_witness_policy_id"] != external_policy_id
    ):
        _fail(FAIL_BINDING_V1, "anti-fork anchor current-cut lineage differs")
    pre_cut_time = statement["pre_cut_logical_time_binding"]
    if (
        body["current_cut_token_id"] == pre_cut_time["last_token_id"]
        or body["current_cut_head_id"] == pre_cut_time["final_head_id"]
        or body["current_cut_head_payload_sha256"]
        == pre_cut_time["final_head_payload_sha256"]
    ):
        _fail(FAIL_BINDING_V1, "current-cut logical time reuses the pre-cut state")
    _identifier(body["current_cut_token_id"], "anti-fork current-cut token", _LOGICAL_TIME_TOKEN_PREFIX)
    _identifier(body["current_cut_checkpoint_id"], "anti-fork current-cut checkpoint", _LOGICAL_TIME_CHECKPOINT_PREFIX)
    _identifier(body["current_cut_head_id"], "anti-fork current-cut head", _LOGICAL_TIME_HEAD_PREFIX)
    _identifier(body["owner_status_ledger_id"], "anti-fork owner ledger", _OWNER_LEDGER_PREFIX)
    _identifier(body["owner_current_head_id"], "anti-fork owner head", _OWNER_HEAD_PREFIX)
    _identifier(body["projection_id"], "anti-fork owner projection", _OWNER_PROJECTION_PREFIX)
    for field in ("current_cut_token_payload_sha256", "current_cut_checkpoint_payload_sha256", "current_cut_head_payload_sha256", "complete_event_set_root"):
        _sha_address(body[field], f"anti-fork anchor {field}")
    if _integer(body["anchor_sequence"], "anchor sequence", 1) != 1:
        _fail(FAIL_BINDING_V1, "candidate R01 anchor must be the first witness sequence")
    if body["previous_external_anchor_receipt_id"] is not None:
        _fail(FAIL_BINDING_V1, "first candidate R01 anchor must not name a predecessor")
    _text(body["anchor_nonce"], "anchor nonce", 256)
    expected_id = _id("signed_owner_status_anti_fork_anchor", body)
    if root["anchor_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "anti-fork anchor ID differs")
    _validate_detached_signature(
        root["signature"], registry=registry, role="PURE_REPLAYER",
        purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
        artifact_kind="owner_status_anti_fork_anchor", body=body,
    )
    return root


def load_external_owner_status_anchor_receipt_v1(
    payload: bytes,
    *,
    anchor_payload: bytes,
    anchor: Mapping[str, object],
    external_policy: Mapping[str, object],
) -> dict[str, object]:
    receipt = _object(
        _decode(payload, "external owner-status anchor receipt"),
        EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1,
        "external owner-status anchor receipt",
    )
    schema_version = _text(
        receipt["schema_version"], "external receipt schema", 256
    )
    external_policy_id = _identifier(
        receipt["external_witness_policy_id"],
        "external receipt policy ID",
        _EXTERNAL_POLICY_PREFIX,
    )
    witness_namespace = _text(
        receipt["witness_namespace"], "external receipt witness namespace", 256
    )
    witness_control_domain_id = _identifier(
        receipt["witness_control_domain_id"],
        "external receipt witness control-domain ID",
    )
    witness_key_id = _identifier(
        receipt["witness_key_id"], "external receipt witness key ID"
    )
    witness_key_epoch = _integer(
        receipt["witness_key_epoch"], "external receipt witness key epoch", 1
    )
    policy_key_epoch = _integer(
        external_policy["witness_key_epoch"],
        "external policy witness key epoch",
        1,
    )
    algorithm = _text(
        receipt["algorithm"], "external receipt algorithm", 32
    )
    anchor_id = _identifier(
        receipt["anchor_id"],
        "external receipt anchor ID",
        _PREFIXES["signed_owner_status_anti_fork_anchor"],
    )
    anchor_payload_sha256 = _sha(
        receipt["anchor_payload_sha256"], "external receipt anchor payload SHA"
    )
    witness_sequence = _integer(
        receipt["witness_sequence"], "witness sequence", 1
    )
    previous_receipt_id = receipt["previous_external_receipt_id"]
    if previous_receipt_id is not None:
        previous_receipt_id = _identifier(
            previous_receipt_id,
            "previous external receipt ID",
            _PREFIXES["external_owner_status_anchor_receipt"],
        )
    leaf_index = _integer(receipt["leaf_index"], "external leaf index")
    tree_size = _integer(receipt["tree_size"], "external tree size", 1)
    signed_tree_root = _sha(
        receipt["signed_tree_root_sha256"], "external signed tree root"
    )
    path = receipt["inclusion_path_sha256_in_exact_order"]
    if type(path) is not list or len(path) > 64:
        _fail(FAIL_SCHEMA_V1, "external inclusion path shape differs")
    for index, digest in enumerate(path):
        _sha(digest, f"external inclusion path {index}")
    signature_hex = _text(
        receipt["signature_hex"], "external receipt signature", 128
    )
    if _HEX128.fullmatch(signature_hex) is None:
        _fail(FAIL_SIGNATURE_V1, "external receipt signature encoding differs")
    receipt_id = _identifier(
        receipt["receipt_id"],
        "external receipt ID",
        _PREFIXES["external_owner_status_anchor_receipt"],
    )
    if (
        schema_version != EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1
        or algorithm != "Ed25519"
        or witness_key_epoch != 1
        or policy_key_epoch != 1
    ):
        _fail(FAIL_SCHEMA_V1, "external anchor receipt header differs")
    if (
        external_policy_id != external_policy["external_witness_policy_id"]
        or witness_namespace != external_policy["witness_namespace"]
        or witness_control_domain_id
        != external_policy["witness_control_domain_id"]
        or witness_key_id != external_policy["witness_key_id"]
        or witness_key_epoch != policy_key_epoch
        or anchor_id != anchor["anchor_id"]
        or anchor_payload_sha256 != _payload_sha(anchor_payload)
        or previous_receipt_id
        != anchor["body"]["previous_external_anchor_receipt_id"]
    ):
        _fail(FAIL_BINDING_V1, "external anchor receipt lineage differs")
    if leaf_index >= tree_size or witness_sequence != anchor["body"]["anchor_sequence"]:
        _fail(FAIL_SCHEMA_V1, "external witness sequence/tree bounds differ")
    if leaf_index != 0 or tree_size != 1 or path:
        _fail(FAIL_BINDING_V1, "initial R01 witness receipt must be the one-leaf tree")
    expected_root = hashlib.sha256(_MERKLE_LEAF_DOMAIN + anchor_payload).hexdigest()
    if signed_tree_root != expected_root:
        _fail(FAIL_BINDING_V1, "external witness inclusion proof/root differs")
    expected_id = _id(
        "external_owner_status_anchor_receipt",
        _without(receipt, "signature_hex", "receipt_id"),
    )
    if receipt_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "external anchor receipt ID differs")
    _verify_external_receipt_signature(receipt, external_policy)
    return receipt


def load_r01_receipt_body_v1(
    payload: bytes,
    *,
    authority_ids: tuple[str, str, str],
    expected_attempt_context_id: str,
    statement_payload: bytes,
    statement: Mapping[str, object],
    response_payload: bytes,
    response: Mapping[str, object],
    anchor_payload: bytes,
    anchor: Mapping[str, object],
    external_receipt_payload: bytes,
    external_receipt: Mapping[str, object],
    bundle_payload: bytes,
    bundle: Mapping[str, object],
    service_bindings_in_exact_order: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    body = _object(_decode(payload, "R01 receipt body"), R01_RECEIPT_BODY_FIELDS_V1, "R01 receipt body")
    _candidate_header(body, R01_RECEIPT_BODY_SCHEMA_V1, "R01 receipt body")
    _require_authority_refs(body, authority_ids, "R01 receipt body")
    if body["readiness_dag_id"] != _READINESS_DAG_ID or body["attempt_context_id"] != expected_attempt_context_id:
        _fail(FAIL_BINDING_V1, "R01 receipt authority/attempt DAG binding differs")
    owner_auth = _object(body["owner_authorization_binding"], OWNER_AUTHORIZATION_BINDING_FIELDS_V1, "owner authorization binding")
    for field in ("authorization_config_sha256", "authorization_source_sha256", "actor_profile_canonical_sha256"):
        _sha(owner_auth[field], f"owner authorization {field}")
    for field in OWNER_AUTHORIZATION_BINDING_FIELDS_V1 - {
        "authorization_schema", "exact_freeze_version", "authority_class"
    } - {"authorization_config_sha256", "authorization_source_sha256", "actor_profile_canonical_sha256"}:
        _identifier(owner_auth[field], f"owner authorization {field}")
    _text(owner_auth["authorization_schema"], "owner authorization schema", 512)
    _text(owner_auth["exact_freeze_version"], "owner exact freeze version", 128)
    _text(owner_auth["authority_class"], "owner authority class", 256)
    owner_status = _object(body["owner_status_binding"], OWNER_STATUS_BINDING_FIELDS_V1, "R01 owner-status binding")
    _same(owner_status, statement["owner_status_binding"], "R01 owner-status binding")
    current_cut = _object(body["current_cut_binding"], CURRENT_CUT_BINDING_FIELDS_V1, "R01 current-cut binding")
    expected_current_cut = {
        "current_cut_response_id": response["response_id"],
        "current_cut_response_payload_sha256": _payload_sha(response_payload),
        "current_cut_statement_id": statement["statement_id"],
        "current_cut_statement_payload_sha256": _payload_sha(statement_payload),
        "current_cut_token_id": anchor["body"]["current_cut_token_id"],
        "current_cut_token_payload_sha256": anchor["body"]["current_cut_token_payload_sha256"],
        "current_cut_checkpoint_id": anchor["body"]["current_cut_checkpoint_id"],
        "current_cut_checkpoint_payload_sha256": anchor["body"]["current_cut_checkpoint_payload_sha256"],
        "current_cut_head_id": anchor["body"]["current_cut_head_id"],
        "current_cut_head_payload_sha256": anchor["body"]["current_cut_head_payload_sha256"],
    }
    _same(current_cut, expected_current_cut, "R01 current-cut lineage")
    anti_fork = _object(body["anti_fork_binding"], ANTI_FORK_BINDING_FIELDS_V1, "R01 anti-fork binding")
    expected_anti_fork = {
        "anchor_id": anchor["anchor_id"],
        "anchor_payload_sha256": _payload_sha(anchor_payload),
        "external_witness_policy_id": external_receipt["external_witness_policy_id"],
        "external_anchor_receipt_id": external_receipt["receipt_id"],
        "external_anchor_receipt_payload_sha256": _payload_sha(external_receipt_payload),
        "witness_sequence": external_receipt["witness_sequence"],
        "signed_tree_root_sha256": external_receipt["signed_tree_root_sha256"],
    }
    _same(anti_fork, expected_anti_fork, "R01 anti-fork lineage")
    runtime_binding = _object(body["runtime_closure_bundle_binding"], RUNTIME_CLOSURE_BUNDLE_BINDING_FIELDS_V1, "R01 runtime-closure bundle binding")
    expected_runtime = {
        "bundle_id": bundle["bundle_id"],
        "bundle_payload_sha256": _payload_sha(bundle_payload),
        "report_ids_in_exact_order": bundle["body"]["report_ids_in_exact_order"],
        "report_payload_sha256s_in_exact_order": bundle["body"]["report_payload_sha256s_in_exact_order"],
    }
    _same(runtime_binding, expected_runtime, "R01 runtime-closure bundle lineage")
    cas = _object(body["cas_complete_session_binding"], CAS_COMPLETE_SESSION_BINDING_FIELDS_V1, "R01 CAS complete-session binding")
    for field in ("session_id", "publication_plan_id", "checkpoint_id", "head_id", "response_id"):
        _identifier(cas[field], f"R01 CAS {field}")
    for field in ("publication_plan_payload_sha256", "checkpoint_payload_sha256", "head_payload_sha256", "response_payload_sha256"):
        _sha_address(cas[field], f"R01 CAS {field}")
    requests = cas["request_ids_in_exact_order"]
    receipts = cas["receipt_ids_in_exact_order"]
    count = _integer(cas["object_count"], "R01 CAS object count", 1)
    if type(requests) is not list or type(receipts) is not list or len(requests) != count or len(receipts) != count:
        _fail(FAIL_SCHEMA_V1, "R01 CAS exact request/receipt cardinality differs")
    expected_main_cas_count = 21 + 8 * owner_status["event_count"]
    if count != expected_main_cas_count:
        _fail(
            FAIL_BINDING_V1,
            "R01 main-CAS cardinality must be 21+8N and cannot reuse the pure-composer spool",
        )
    for index, value in enumerate(requests):
        _identifier(value, f"R01 CAS request {index}")
    for index, value in enumerate(receipts):
        _identifier(value, f"R01 CAS receipt {index}")
    custody = body["service_key_custody_evidence_in_exact_order"]
    if type(custody) is not list or len(custody) != 5:
        _fail(FAIL_SCHEMA_V1, "R01 needs exactly five service-key custody rows")
    runtime_ids = bundle["body"]["runtime_instance_ids_in_exact_order"]
    for index, (service_name, service, raw) in enumerate(zip(_SERVICE_ORDER, service_bindings_in_exact_order, custody, strict=True)):
        row = _object(raw, SERVICE_KEY_CUSTODY_EVIDENCE_FIELDS_V1, f"{service_name} custody row")
        if (
            row["service_name"] != service_name
            or row["service_binding_id"] != service["binding_id"]
            or row["role"] != service["role"]
            or row["role_key_id"] != service["role_key_id"]
            or row["key_epoch"] != service["key_epoch"]
            or row["runtime_instance_id"] != runtime_ids[index]
        ):
            _fail(FAIL_BINDING_V1, f"{service_name} custody static binding differs")
        _identifier(row["runtime_instance_id"], f"{service_name} custody runtime")
        _identifier(row["custody_artifact_id"], f"{service_name} custody artifact")
        _sha(row["custody_artifact_payload_sha256"], f"{service_name} custody artifact SHA")
        _false(row["custody_verified"], f"{service_name} custody_verified")
    receipt_effect = _object(body["readiness_effect"], RECEIPT_READINESS_EFFECT_FIELDS_V1, "R01 receipt readiness effect")
    for field in RECEIPT_READINESS_EFFECT_FIELDS_V1:
        _false(receipt_effect[field], f"R01 receipt readiness effect {field}")
    verdict = _object(body["r01_verdict"], R01_VERDICT_FIELDS_V1, "R01 verdict")
    for field in R01_VERDICT_FIELDS_V1:
        _false(verdict[field], f"R01 verdict {field}")
    _identifier(body["overlay_retention_plan_id"], "overlay retention plan ID")
    _text(body["receipt_nonce"], "R01 receipt nonce", 256)
    expected_id = _id("r01_receipt_body", _without(body, "receipt_body_id"))
    if body["receipt_body_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "R01 receipt body ID differs")
    return body


def load_dual_signed_r01_receipt_v1(
    payload: bytes,
    *,
    receipt_body_payload: bytes,
    receipt_body: Mapping[str, object],
    registry: object,
) -> dict[str, object]:
    root = _object(_decode(payload, "dual-signed R01 receipt"), DUAL_SIGNED_R01_RECEIPT_FIELDS_V1, "dual-signed R01 receipt")
    if root["schema_version"] != DUAL_SIGNED_R01_RECEIPT_SCHEMA_V1:
        _fail(FAIL_SCHEMA_V1, "dual-signed R01 wrapper schema differs")
    nested = _object(root["body"], R01_RECEIPT_BODY_FIELDS_V1, "nested R01 receipt body")
    _same(nested, receipt_body, "dual-signed nested receipt body")
    if root["receipt_body_id"] != receipt_body["receipt_body_id"]:
        _fail(FAIL_BINDING_V1, "dual-signed receipt body ID differs")
    custodian = _validate_detached_signature(
        root["custodian_signature"], registry=registry,
        role="SOURCE_RUNTIME_CUSTODIAN", purpose="SIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
        artifact_kind="r01_authorization_status_receipt", body=receipt_body,
    )
    countersign_subject = phase2b_step5_r01_pure_countersign_subject_v1(
        receipt_body_id=receipt_body["receipt_body_id"],
        receipt_body_payload=receipt_body_payload,
        custodian_signature=custodian,
    )
    if frozenset(countersign_subject) != PURE_COUNTERSIGN_SUBJECT_FIELDS_V1:
        _fail(FAIL_SCHEMA_V1, "internal pure countersign subject keyset differs")
    _validate_detached_signature(
        root["pure_countersignature"], registry=registry,
        role="PURE_REPLAYER", purpose="COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
        artifact_kind="r01_authorization_status_receipt_countersignature",
        body=countersign_subject,
    )
    expected_id = _id("dual_signed_r01_receipt", _without(root, "receipt_id"))
    if root["receipt_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "dual-signed R01 receipt ID differs")
    return root


def load_r01_readiness_overlay_v1(
    payload: bytes,
    *,
    authority_ids: tuple[str, str, str],
    expected_attempt_context_id: str,
    r01_receipt_payload: bytes,
    r01_receipt: Mapping[str, object],
    expected_overlay_retention_plan_id: str,
) -> dict[str, object]:
    overlay = _object(_decode(payload, "R01 readiness overlay"), R01_READINESS_OVERLAY_FIELDS_V1, "R01 readiness overlay")
    _candidate_header(overlay, R01_READINESS_OVERLAY_SCHEMA_V1, "R01 readiness overlay")
    _require_authority_refs(overlay, authority_ids, "R01 readiness overlay")
    if (
        overlay["attempt_context_id"] != expected_attempt_context_id
        or overlay["r01_receipt_id"] != r01_receipt["receipt_id"]
        or overlay["r01_receipt_payload_sha256"] != _payload_sha(r01_receipt_payload)
        or overlay["overlay_retention_plan_id"] != expected_overlay_retention_plan_id
    ):
        _fail(FAIL_BINDING_V1, "R01 readiness overlay receipt/retention lineage differs")
    effect = _object(overlay["readiness_effect"], R01_READINESS_EFFECT_FIELDS_V1, "R01 readiness overlay effect")
    rows = effect["node_states_in_canonical_topological_order"]
    if type(rows) is not list or len(rows) != len(_READINESS_NODES):
        _fail(FAIL_SCHEMA_V1, "candidate overlay needs all sixteen readiness rows")
    for node, raw in zip(_READINESS_NODES, rows, strict=True):
        row = _object(raw, frozenset({"node_id", "satisfied"}), "overlay readiness row")
        if row["node_id"] != node:
            _fail(FAIL_BINDING_V1, "overlay readiness node order differs")
        _false(row["satisfied"], f"overlay readiness {node}")
    if (
        effect["readiness_dag_id"] != _READINESS_DAG_ID
        or effect["overlay_sequence"] != 0
        or effect["previous_overlay_id"] is not None
        or effect["transitioned_node_id"] is not None
        or effect["satisfied_node_count"] != 0
        or effect["highest_contiguous_satisfied_node"] is not None
    ):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "candidate overlay must be sequence-zero with no transition")
    _identifier(effect["base_readiness_candidate_id"], "base readiness candidate ID")
    for field in ("all_nodes_satisfied", "r01_satisfied", "r02_or_later_satisfied", "execution_authorized"):
        _false(effect[field], f"overlay effect {field}")
    expected_id = _id("r01_readiness_overlay", _without(overlay, "overlay_id"))
    if overlay["overlay_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "R01 readiness overlay ID differs")
    return overlay


def _verify_r01_evidence_contract_candidate_implementation_v1(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    current_cut_response_payload: bytes,
    current_cut_statement_payload: bytes,
    live_runtime_closure_report_payloads_in_exact_order: tuple[bytes, ...],
    live_runtime_closure_bundle_payload: bytes,
    signed_owner_status_anti_fork_anchor_payload: bytes,
    external_owner_status_anchor_receipt_payload: bytes,
    r01_receipt_body_payload: bytes,
    dual_signed_r01_receipt_payload: bytes,
    r01_readiness_overlay_payload: bytes,
    external_witness_policy_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    out_of_band_expected_external_witness_policy_id_bytes: bytes,
    out_of_band_expected_external_witness_policy_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
) -> VerifiedR01EvidenceContractCandidateV1:
    """Replay the nine-object non-admitted R01 wire DAG.

    This verifies candidate signatures, including the external witness
    signature, solely as cryptographic mechanics.  The returned projection
    keeps the corresponding authority/currentness/R01 predicates false.
    """
    _identifier(expected_attempt_context_id, "expected attempt-context ID")
    if type(live_runtime_closure_report_payloads_in_exact_order) is not tuple:
        _fail(FAIL_INPUT_V1, "live report payloads must be one exact tuple")
    try:
        lock = _load_stage_b_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        profile = _load_authority_profile(authority_profile_payload, lock)
        registry = _load_signer_registry(
            signer_registry_payload, lock, profile
        )
        lock, profile, registry = _require_authority_graph(
            lock, profile, registry
        )
    except _AuthorityError as exc:
        raise Phase2BStep5R01EvidenceContractErrorV1(
            FAIL_TRUST_ANCHOR_V1, "raw authority-v2 graph rejected"
        ) from exc
    authority_ids = _authority_ids(lock, profile, registry)
    services = _service_bindings(profile)
    response = load_current_cut_response_v1(
        current_cut_response_payload,
        authority_ids=authority_ids,
        expected_attempt_context_id=expected_attempt_context_id,
    )
    governance_service = services[_SERVICE_ORDER.index("governance_status")]
    if response["governance_service_binding_id"] != governance_service["binding_id"]:
        _fail(FAIL_BINDING_V1, "current-cut response governance service binding differs")
    statement = load_current_cut_statement_v1(
        current_cut_statement_payload,
        authority_ids=authority_ids,
        expected_attempt_context_id=expected_attempt_context_id,
        current_cut_response_payload=current_cut_response_payload,
        current_cut_response=response,
    )
    if len(live_runtime_closure_report_payloads_in_exact_order) != 5:
        _fail(FAIL_SCHEMA_V1, "R01 evidence DAG needs exactly five live reports")
    reports = tuple(
        load_live_runtime_closure_report_v1(
            raw,
            authority_ids=authority_ids,
            expected_attempt_context_id=expected_attempt_context_id,
            expected_service_binding=service,
            registry=registry,
        )
        for raw, service in zip(
            live_runtime_closure_report_payloads_in_exact_order,
            services,
            strict=True,
        )
    )
    bundle = load_live_runtime_closure_bundle_v1(
        live_runtime_closure_bundle_payload,
        authority_ids=authority_ids,
        expected_attempt_context_id=expected_attempt_context_id,
        report_payloads_in_exact_order=live_runtime_closure_report_payloads_in_exact_order,
        reports_in_exact_order=reports,
        service_bindings_in_exact_order=services,
        registry=registry,
    )
    policy = _load_external_policy(
        external_witness_policy_payload,
        expected_id_bytes=out_of_band_expected_external_witness_policy_id_bytes,
        expected_payload_digest=(
            out_of_band_expected_external_witness_policy_payload_sha256_digest
        ),
        registry=registry,
    )
    anchor = load_signed_owner_status_anti_fork_anchor_v1(
        signed_owner_status_anti_fork_anchor_payload,
        authority_ids=authority_ids,
        expected_attempt_context_id=expected_attempt_context_id,
        statement_payload=current_cut_statement_payload,
        statement=statement,
        external_policy_id=policy["external_witness_policy_id"],
        registry=registry,
    )
    external_receipt = load_external_owner_status_anchor_receipt_v1(
        external_owner_status_anchor_receipt_payload,
        anchor_payload=signed_owner_status_anti_fork_anchor_payload,
        anchor=anchor,
        external_policy=policy,
    )
    receipt_body = load_r01_receipt_body_v1(
        r01_receipt_body_payload,
        authority_ids=authority_ids,
        expected_attempt_context_id=expected_attempt_context_id,
        statement_payload=current_cut_statement_payload,
        statement=statement,
        response_payload=current_cut_response_payload,
        response=response,
        anchor_payload=signed_owner_status_anti_fork_anchor_payload,
        anchor=anchor,
        external_receipt_payload=external_owner_status_anchor_receipt_payload,
        external_receipt=external_receipt,
        bundle_payload=live_runtime_closure_bundle_payload,
        bundle=bundle,
        service_bindings_in_exact_order=services,
    )
    _same(
        receipt_body["owner_authorization_binding"],
        profile.body["owner_authorization_binding"],
        "R01 frozen owner-authorization/profile lineage",
    )
    signed_receipt = load_dual_signed_r01_receipt_v1(
        dual_signed_r01_receipt_payload,
        receipt_body_payload=r01_receipt_body_payload,
        receipt_body=receipt_body,
        registry=registry,
    )
    overlay = load_r01_readiness_overlay_v1(
        r01_readiness_overlay_payload,
        authority_ids=authority_ids,
        expected_attempt_context_id=expected_attempt_context_id,
        r01_receipt_payload=dual_signed_r01_receipt_payload,
        r01_receipt=signed_receipt,
        expected_overlay_retention_plan_id=receipt_body["overlay_retention_plan_id"],
    )
    ids: dict[str, object] = {
        "stage_b_trust_lock_id": lock.artifact_id,
        "authority_profile_id": profile.artifact_id,
        "signer_registry_id": registry.artifact_id,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "current_cut_response_id": response["response_id"],
        "current_cut_statement_id": statement["statement_id"],
        "live_runtime_closure_report_ids_in_exact_order": tuple(
            report["report_id"] for report in reports
        ),
        "live_runtime_closure_bundle_id": bundle["bundle_id"],
        "signed_owner_status_anti_fork_anchor_id": anchor["anchor_id"],
        "external_owner_status_anchor_receipt_id": external_receipt["receipt_id"],
        "r01_receipt_body_id": receipt_body["receipt_body_id"],
        "dual_signed_r01_receipt_id": signed_receipt["receipt_id"],
        "r01_readiness_overlay_id": overlay["overlay_id"],
    }
    shas: dict[str, object] = {
        "stage_b_trust_lock_payload_sha256": _payload_sha(stage_b_trust_lock_payload),
        "authority_profile_payload_sha256": _payload_sha(authority_profile_payload),
        "signer_registry_payload_sha256": _payload_sha(signer_registry_payload),
        "external_witness_policy_payload_sha256": _payload_sha(external_witness_policy_payload),
        "current_cut_response_payload_sha256": _payload_sha(current_cut_response_payload),
        "current_cut_statement_payload_sha256": _payload_sha(current_cut_statement_payload),
        "live_runtime_closure_report_payload_sha256s_in_exact_order": tuple(
            _payload_sha(raw) for raw in live_runtime_closure_report_payloads_in_exact_order
        ),
        "live_runtime_closure_bundle_payload_sha256": _payload_sha(live_runtime_closure_bundle_payload),
        "signed_owner_status_anti_fork_anchor_payload_sha256": _payload_sha(signed_owner_status_anti_fork_anchor_payload),
        "external_owner_status_anchor_receipt_payload_sha256": _payload_sha(external_owner_status_anchor_receipt_payload),
        "r01_receipt_body_payload_sha256": _payload_sha(r01_receipt_body_payload),
        "dual_signed_r01_receipt_payload_sha256": _payload_sha(dual_signed_r01_receipt_payload),
        "r01_readiness_overlay_payload_sha256": _payload_sha(r01_readiness_overlay_payload),
    }
    return _issue_result(ids, shas)


def require_verified_r01_evidence_contract_candidate_v1(
    value: object,
) -> VerifiedR01EvidenceContractCandidateV1:
    return _require_result(value)


def verify_real_r01_admission_v1(*args: object, **kwargs: object) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "real currentness/durability/authority R01 admission is not implemented",
    )


def issue_admitted_r01_only_overlay_v1(*args: object, **kwargs: object) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "admitted sequence-one R01-only overlay issuance is not implemented",
    )


def _definition_captured_function_graph_v1() -> dict[str, object]:
    """Clone functions and function-valued defaults onto one private graph."""
    module_namespace = globals()
    namespace = dict(module_namespace)
    originals: dict[str, FunctionType] = {
        name: value
        for name, value in module_namespace.items()
        if type(value) is FunctionType
        and value.__module__ == __name__
        and name != "_definition_captured_function_graph_v1"
    }
    provisional: dict[str, FunctionType] = {}
    by_identity: dict[int, FunctionType] = {}
    for name, function in originals.items():
        clone = FunctionType(
            function.__code__, namespace, function.__name__,
            function.__defaults__, function.__closure__,
        )
        provisional[name] = clone
        by_identity[id(function)] = clone
    namespace.update(provisional)

    def captured_default(value: object) -> object:
        return by_identity.get(id(value), value)

    clones: dict[str, FunctionType] = {}
    for name, function in originals.items():
        defaults = function.__defaults__
        clone = FunctionType(
            function.__code__, namespace, function.__name__,
            None if defaults is None else tuple(captured_default(value) for value in defaults),
            function.__closure__,
        )
        clone.__kwdefaults__ = (
            None if function.__kwdefaults__ is None else {
                key: captured_default(value)
                for key, value in function.__kwdefaults__.items()
            }
        )
        clone.__annotations__ = dict(function.__annotations__)
        clone.__doc__ = function.__doc__
        clone.__module__ = function.__module__
        clone.__qualname__ = function.__qualname__
        clones[name] = clone
    final_by_identity = {id(originals[name]): clone for name, clone in clones.items()}

    def final_default(value: object) -> object:
        return final_by_identity.get(id(value), value)

    for name, function in originals.items():
        clone = clones[name]
        defaults = function.__defaults__
        clone.__defaults__ = (
            None if defaults is None else tuple(final_default(value) for value in defaults)
        )
        clone.__kwdefaults__ = (
            None if function.__kwdefaults__ is None else {
                key: final_default(value)
                for key, value in function.__kwdefaults__.items()
            }
        )
    namespace.update(clones)
    return clones


_CAPTURED = _definition_captured_function_graph_v1()
verify_r01_evidence_contract_candidate_v1 = _CAPTURED[
    "_verify_r01_evidence_contract_candidate_implementation_v1"
]
require_verified_r01_evidence_contract_candidate_v1 = _CAPTURED[
    "require_verified_r01_evidence_contract_candidate_v1"
]
verify_real_r01_admission_v1 = _CAPTURED["verify_real_r01_admission_v1"]
issue_admitted_r01_only_overlay_v1 = _CAPTURED["issue_admitted_r01_only_overlay_v1"]
phase2b_step5_r01_evidence_content_id_v1 = _CAPTURED[
    "phase2b_step5_r01_evidence_content_id_v1"
]
canonical_r01_evidence_contract_json_bytes_v1 = _CAPTURED[
    "canonical_r01_evidence_contract_json_bytes_v1"
]
phase2b_step5_r01_authority_signature_preimage_v1 = _CAPTURED[
    "phase2b_step5_r01_authority_signature_preimage_v1"
]
phase2b_step5_external_anchor_receipt_signature_preimage_v1 = _CAPTURED[
    "phase2b_step5_external_anchor_receipt_signature_preimage_v1"
]
phase2b_step5_r01_pure_countersign_subject_v1 = _CAPTURED[
    "phase2b_step5_r01_pure_countersign_subject_v1"
]
del _CAPTURED
del _definition_captured_function_graph_v1
del _verify_r01_evidence_contract_candidate_implementation_v1
del _issue_result
del load_current_cut_response_v1
del load_current_cut_statement_v1
del load_live_runtime_closure_report_v1
del load_live_runtime_closure_bundle_v1
del load_signed_owner_status_anti_fork_anchor_v1
del load_external_owner_status_anchor_receipt_v1
del load_r01_receipt_body_v1
del load_dual_signed_r01_receipt_v1
del load_r01_readiness_overlay_v1


__all__ = [
    "CANDIDATE_CLAIM_LEVEL_V1", "CANDIDATE_STATUS_V1",
    "CURRENT_CUT_RESPONSE_SCHEMA_V1", "CURRENT_CUT_RESPONSE_FIELDS_V1",
    "CURRENT_CUT_STATEMENT_SCHEMA_V1", "CURRENT_CUT_STATEMENT_FIELDS_V1",
    "GOVERNANCE_PEER_OBSERVATION_FIELDS_V1", "OWNER_STATUS_BINDING_FIELDS_V1",
    "PRE_CUT_LOGICAL_TIME_BINDING_FIELDS_V1",
    "LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1", "LIVE_RUNTIME_CLOSURE_REPORT_BODY_FIELDS_V1",
    "RUNTIME_OBSERVATION_FIELDS_V1", "LIVE_PROTOCOL_EVIDENCE_BINDING_FIELDS_V1",
    "LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1", "LIVE_RUNTIME_CLOSURE_BUNDLE_BODY_FIELDS_V1",
    "SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1",
    "SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1",
    "EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1",
    "EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1",
    "EXTERNAL_WITNESS_POLICY_SCHEMA_V1", "EXTERNAL_WITNESS_POLICY_FIELDS_V1",
    "R01_RECEIPT_BODY_SCHEMA_V1", "R01_RECEIPT_BODY_FIELDS_V1", "R01_VERDICT_FIELDS_V1",
    "DUAL_SIGNED_R01_RECEIPT_SCHEMA_V1", "DUAL_SIGNED_R01_RECEIPT_FIELDS_V1",
    "PURE_COUNTERSIGN_SUBJECT_FIELDS_V1", "R01_READINESS_OVERLAY_SCHEMA_V1",
    "R01_READINESS_OVERLAY_FIELDS_V1", "R01_READINESS_EFFECT_FIELDS_V1",
    "FAIL_INPUT_V1", "FAIL_CANONICAL_V1", "FAIL_SCHEMA_V1", "FAIL_CONTENT_ID_V1",
    "FAIL_BINDING_V1", "FAIL_SIGNATURE_V1", "FAIL_TRUST_ANCHOR_V1",
    "FAIL_CLAIM_BOUNDARY_V1", "FAIL_NOT_IMPLEMENTED_V1",
    "Phase2BStep5R01EvidenceContractErrorV1", "VerifiedR01EvidenceContractCandidateV1",
    "canonical_r01_evidence_contract_json_bytes_v1",
    "phase2b_step5_r01_evidence_content_id_v1",
    "phase2b_step5_r01_authority_signature_preimage_v1",
    "phase2b_step5_external_anchor_receipt_signature_preimage_v1",
    "phase2b_step5_r01_pure_countersign_subject_v1",
    "verify_r01_evidence_contract_candidate_v1",
    "require_verified_r01_evidence_contract_candidate_v1",
    "verify_real_r01_admission_v1", "issue_admitted_r01_only_overlay_v1",
]
