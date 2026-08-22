"""Producer-free mechanics composer for a future Step-5 R01 verifier.

This module composes only independently presented bytes through the pure
authority/profile V2 verifier, the verifier-only owner-status and logical-time
cores, the verifier-only generic CAS-store/2 core, and the offline static
runtime-closure verifier.  It has no producer, service, filesystem, clock,
socket, private key, or decoded-object entry point.

The generic CAS session is required to contain exactly ``13 + 8N`` objects,
where ``N`` is the one-event ACTIVE chain or the two-event ACTIVE->REVOKED
chain.  The fixed thirteen objects and each event's eight objects have one
frozen order.  Five caller-supplied canonical runtime-closure reports are
reconstructed from privately issued offline-verifier results and raw-input
digests; one canonical bundle binds their exact service order.  The reports
are mechanics summaries, not signed runtime attestations.

Success proves only that all *presented* mechanics are internally cross-bound.
It does not prove external currentness or anti-fork anchoring, CAS durability,
live runtime closure, UID/control-domain isolation, key custody, owner
signature authority, a real R01 receipt, readiness, or execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    SERVICE_ORDER_V2 as _AUTHORITY_SERVICE_ORDER_V2,
    Phase2BStep5AuthorityErrorV2 as _AuthorityErrorV2,
    load_step5_authority_profile_v2 as _load_authority_profile_v2,
    load_step5_signer_registry_v2 as _load_signer_registry_v2,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_trust_lock_v2,
    require_locked_step5_authority_graph_v2 as _require_authority_graph_v2,
)
from .phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 import (
    ExpectedLogicalEventCoreV1,
    Phase2BStep5LogicalTimeReplayCoreErrorV1 as _LogicalTimeErrorV1,
    require_verified_logical_time_replay_core_v1 as _require_time_core_v1,
    verify_logical_time_replay_core_v1 as _verify_time_core_v1,
)
from .phase2b_formal_wire_covert_step5_owner_status_replay_core_v1 import (
    Phase2BStep5OwnerStatusReplayCoreErrorV1 as _OwnerStatusErrorV1,
    require_verified_owner_status_replay_core_v1 as _require_owner_core_v1,
    verify_owner_status_replay_core_v1 as _verify_owner_core_v1,
)
from .phase2b_formal_wire_covert_step5_runtime_closure_v1 import (
    Phase2BStep5RuntimeClosureErrorV1 as _RuntimeClosureErrorV1,
    require_verified_offline_step5_image_closure_mechanics_v1 as _require_runtime_closure_v1,
    verify_offline_step5_image_closure_mechanics_v1 as _verify_runtime_closure_v1,
)
from .phase2b_formal_wire_covert_step5_store_replay_core_v2 import (
    CasStoreReplayCoreV2Error as _CasStoreReplayErrorV2,
    require_verified_cas_store_complete_session_replay_core_v2 as _require_store_core_v2,
    verify_cas_store_complete_session_replay_core_v2 as _verify_store_core_v2,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical_rfc8785_v1,
)


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-pure-replay-candidate/1"
)
RUNTIME_CLOSURE_REPORT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-runtime-closure-report/1"
)
RUNTIME_CLOSURE_REPORT_BUNDLE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-runtime-closure-report-bundle/1"
)
R01_PURE_REPLAY_CANDIDATE_PREFIX_V1: Final = (
    "phase2b_step5_r01_pure_replay_candidate_v1_"
)
RUNTIME_CLOSURE_REPORT_PREFIX_V1: Final = (
    "phase2b_step5_runtime_closure_report_v1_"
)
RUNTIME_CLOSURE_REPORT_BUNDLE_PREFIX_V1: Final = (
    "phase2b_step5_runtime_closure_report_bundle_v1_"
)

CLAIM_BOUNDARY_V1: Final = (
    "PRODUCER_FREE_PRESENTED_AUTHORITY_OWNER_TIME_CAS_AND_FIVE_OFFLINE_"
    "RUNTIME_CLOSURE_MECHANICS_EXACTLY_CROSS_BOUND_NOT_EXTERNAL_CURRENTNESS_"
    "NOT_ANTI_FORK_NOT_CAS_DURABILITY_NOT_LIVE_RUNTIME_NOT_UID_CONTROL_"
    "NOT_KEY_CUSTODY_NOT_OWNER_SIGNATURE_NOT_AUTHORITY_NOT_REAL_R01_NOT_"
    "READINESS_NOT_EXECUTION"
)
RUNTIME_REPORT_CLAIM_BOUNDARY_V1: Final = (
    "OFFLINE_STATIC_RUNTIME_CLOSURE_RAW_INPUT_DIGESTS_AND_PRIVATE_RESULT_"
    "MECHANICS_ONLY_NOT_SIGNED_ATTESTATION_NOT_DURABLY_RETAINED_NOT_LIVE_"
    "NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
RUNTIME_REPORT_BUNDLE_CLAIM_BOUNDARY_V1: Final = (
    "EXACT_FIVE_ORDERED_OFFLINE_RUNTIME_REPORTS_PRESENTED_MECHANICS_ONLY_"
    "NOT_SIGNED_NOT_DURABLE_NOT_LIVE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_"
    "NOT_EXECUTION"
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_SCHEMA"
FAIL_DEPENDENCY_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_DEPENDENCY"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_BINDING"
FAIL_CAS_SEQUENCE_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_CAS_SEQUENCE"
FAIL_RESULT_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_RESULT"
FAIL_CLAIM_BOUNDARY_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_CLAIM_BOUNDARY"
FAIL_NOT_IMPLEMENTED_V1: Final = "FAIL_STEP5_R01_PURE_REPLAY_V1_NOT_IMPLEMENTED"

_REPORT_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_REPORT/V1\x00"
)
_REPORT_BUNDLE_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_REPORT_BUNDLE/V1\x00"
)
_CANDIDATE_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/STEP5/R01_PURE_REPLAY_CANDIDATE/V1\x00"
)
_ORDERED_DIGEST_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_REPORT/ORDERED_DIGEST_ROOT/V1\x00"
)
_MAX_CANONICAL_BYTES: Final = 8 * 1024 * 1024

CAS_ARTIFACT_KINDS_BASE_V1: Final = (
    "stage_b_trust_lock_v2",
    "authority_profile_v2",
    "signer_registry_v2",
    "owner_authorization_v1",
    "runtime_closure_report_v1",
    "runtime_closure_report_v1",
    "runtime_closure_report_v1",
    "runtime_closure_report_v1",
    "runtime_closure_report_v1",
    "runtime_closure_report_bundle_v1",
    "owner_status_projection_v1",
    "owner_status_genesis_head_v1",
    "logical_time_genesis_head_v1",
)
CAS_ARTIFACT_KINDS_PER_EVENT_V1: Final = (
    "owner_status_decision_v1",
    "owner_status_signed_event_v1",
    "owner_status_checkpoint_v1",
    "owner_status_non_genesis_head_v1",
    "logical_time_request_v1",
    "logical_time_signed_token_v1",
    "logical_time_checkpoint_v1",
    "logical_time_non_genesis_head_v1",
)


class Phase2BStep5R01PureReplayErrorV1(ValueError):
    """Stable fail-closed error for the mechanics-only R01 composer."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _require_exact_bytes(value: object, label: str) -> None:
    if type(value) is not bytes:
        raise TypeError(f"{label} must be exact bytes")


def _require_exact_bytes_tuple(value: object, label: str) -> None:
    if type(value) is not tuple or any(type(item) is not bytes for item in value):
        raise TypeError(f"{label} must be an exact tuple of exact bytes")


def _require_exact_chunk_matrix(value: object, label: str) -> None:
    if (
        type(value) is not tuple
        or any(type(chunks) is not tuple for chunks in value)
        or any(type(chunk) is not bytes for chunks in value for chunk in chunks)
    ):
        raise TypeError(f"{label} must be an exact tuple of exact byte tuples")


@dataclass(frozen=True, slots=True)
class OwnerStatusRawReplayInputsV1:
    """Exact raw owner-status and its complete logical-time dependency chain."""

    owner_authorization_payload: bytes
    decision_payloads_in_exact_order: tuple[bytes, ...]
    signed_event_payloads_in_exact_order: tuple[bytes, ...]
    owner_checkpoint_payloads_in_exact_order: tuple[bytes, ...]
    owner_head_payloads_in_exact_order: tuple[bytes, ...]
    caller_supplied_current_head_payload: bytes
    owner_projection_payload: bytes
    logical_time_request_payloads_in_exact_order: tuple[bytes, ...]
    logical_time_signed_token_payloads_in_exact_order: tuple[bytes, ...]
    logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...]
    logical_time_head_payloads_in_exact_order: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_exact_bytes(self.owner_authorization_payload, "owner authorization")
        _require_exact_bytes(
            self.caller_supplied_current_head_payload, "caller current owner head"
        )
        _require_exact_bytes(self.owner_projection_payload, "owner projection")
        for name in (
            "decision_payloads_in_exact_order",
            "signed_event_payloads_in_exact_order",
            "owner_checkpoint_payloads_in_exact_order",
            "owner_head_payloads_in_exact_order",
            "logical_time_request_payloads_in_exact_order",
            "logical_time_signed_token_payloads_in_exact_order",
            "logical_time_checkpoint_payloads_in_exact_order",
            "logical_time_head_payloads_in_exact_order",
        ):
            _require_exact_bytes_tuple(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class CasStoreCompleteSessionRawInputsV2:
    """Exact raw generic CAS-store/2 complete-session transport."""

    retention_plan_payload: bytes
    request_payloads_in_exact_order: tuple[bytes, ...]
    object_payloads_in_exact_order: tuple[bytes, ...]
    signed_receipt_payloads_in_exact_order: tuple[bytes, ...]
    checkpoint_payloads_in_exact_order: tuple[bytes, ...]
    head_payloads_in_exact_order: tuple[bytes, ...]
    response_payloads_in_exact_order: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_exact_bytes(self.retention_plan_payload, "CAS retention plan")
        for name in (
            "request_payloads_in_exact_order",
            "object_payloads_in_exact_order",
            "signed_receipt_payloads_in_exact_order",
            "checkpoint_payloads_in_exact_order",
            "head_payloads_in_exact_order",
            "response_payloads_in_exact_order",
        ):
            _require_exact_bytes_tuple(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class OfflineRuntimeClosureRawInputsV1:
    """One service's raw offline closure graph plus its canonical report."""

    service_name: str
    runtime_closure_policy_payload: bytes
    image_closure_evidence_payload: bytes
    raw_oci_index_payload: bytes
    raw_oci_manifest_payload: bytes
    raw_oci_config_payload: bytes
    ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...]
    ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...]
    runtime_closure_report_payload: bytes

    def __post_init__(self) -> None:
        if type(self.service_name) is not str:
            raise TypeError("runtime-closure service name must be exact str")
        for name in (
            "runtime_closure_policy_payload",
            "image_closure_evidence_payload",
            "raw_oci_index_payload",
            "raw_oci_manifest_payload",
            "raw_oci_config_payload",
            "runtime_closure_report_payload",
        ):
            _require_exact_bytes(getattr(self, name), name)
        _require_exact_chunk_matrix(
            self.ordered_raw_layer_chunk_tuples, "ordered raw OCI layers"
        )
        _require_exact_chunk_matrix(
            self.ordered_support_artifact_chunk_tuples,
            "ordered support artifacts",
        )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedR01PureReplayCandidateV1:
    """Privately issued mechanics composition; never a real R01 result."""

    schema_version: str
    candidate_id: str
    claim_boundary: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    owner_authorization_id: str
    owner_status_ledger_id: str
    owner_current_head_id: str
    owner_current_head_payload_sha256: str
    owner_complete_event_set_root: str
    owner_latest_effective_status: str
    logical_time_ledger_id: str
    logical_time_final_head_id: str
    cas_retention_session_id: str
    cas_chain_id: str
    cas_complete_head_state_id: str
    cas_complete_checkpoint_sha256: str
    runtime_closure_report_ids_in_exact_service_order: tuple[str, ...]
    runtime_closure_report_payload_sha256s_in_exact_service_order: tuple[str, ...]
    runtime_closure_report_bundle_id: str
    runtime_closure_report_bundle_payload_sha256: str
    owner_event_count: int
    exact_cas_object_count: int
    raw_authority_graph_verified: bool
    owner_status_replay_core_verified: bool
    logical_time_replay_core_verified: bool
    cas_complete_session_replay_core_verified: bool
    five_offline_runtime_closure_mechanics_verified: bool
    runtime_reports_and_bundle_verified: bool
    exact_cas_13_plus_8n_sequence_verified: bool
    all_presented_mechanics_cross_bound: bool
    verifier_only: bool
    external_currentness_verified: bool
    external_anti_fork_anchor_verified: bool
    cas_durability_verified: bool
    live_cas_store_verified: bool
    runtime_closure_inputs_durably_retained: bool
    live_runtime_closure_verified: bool
    live_uid_gid_verified: bool
    live_control_domain_verified: bool
    key_custody_verified: bool
    owner_signature_authenticated: bool
    signed_runtime_attestations_verified: bool
    authority_evidence: bool
    real_r01_receipt_verified: bool
    r01_satisfied: bool
    readiness_overlay_issued: bool
    readiness_evidence: bool
    execution_authorized: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "VerifiedR01PureReplayCandidateV1":
        raise TypeError("R01 pure-replay candidate results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("R01 pure-replay candidate results are privately issued")


def _build_r01_pure_replay_v1():
    """Definition-capture the dependency graph and private result lineage."""

    error_type = Phase2BStep5R01PureReplayErrorV1
    owner_input_type = OwnerStatusRawReplayInputsV1
    cas_input_type = CasStoreCompleteSessionRawInputsV2
    runtime_input_type = OfflineRuntimeClosureRawInputsV1
    time_event_type = ExpectedLogicalEventCoreV1
    result_type = VerifiedR01PureReplayCandidateV1
    service_order = tuple(_AUTHORITY_SERVICE_ORDER_V2)
    base_kinds = tuple(CAS_ARTIFACT_KINDS_BASE_V1)
    per_event_kinds = tuple(CAS_ARTIFACT_KINDS_PER_EVENT_V1)
    schema = SCHEMA_VERSION
    report_schema = RUNTIME_CLOSURE_REPORT_SCHEMA_V1
    bundle_schema = RUNTIME_CLOSURE_REPORT_BUNDLE_SCHEMA_V1
    candidate_prefix = R01_PURE_REPLAY_CANDIDATE_PREFIX_V1
    report_prefix = RUNTIME_CLOSURE_REPORT_PREFIX_V1
    bundle_prefix = RUNTIME_CLOSURE_REPORT_BUNDLE_PREFIX_V1
    claim_boundary = CLAIM_BOUNDARY_V1
    report_claim_boundary = RUNTIME_REPORT_CLAIM_BOUNDARY_V1
    bundle_claim_boundary = RUNTIME_REPORT_BUNDLE_CLAIM_BOUNDARY_V1
    report_domain = bytes(_REPORT_ID_DOMAIN_V1)
    bundle_domain = bytes(_REPORT_BUNDLE_ID_DOMAIN_V1)
    candidate_domain = bytes(_CANDIDATE_ID_DOMAIN_V1)
    digest_root_domain = bytes(_ORDERED_DIGEST_ROOT_DOMAIN_V1)
    maximum_canonical_bytes = _MAX_CANONICAL_BYTES
    canonical = _canonicalize_rfc8785_v1
    decode_canonical = _decode_canonical_rfc8785_v1
    sha256 = hashlib.sha256
    load_lock = _load_stage_b_trust_lock_v2
    load_profile = _load_authority_profile_v2
    load_registry = _load_signer_registry_v2
    require_authority = _require_authority_graph_v2
    verify_owner = _verify_owner_core_v1
    require_owner = _require_owner_core_v1
    verify_time = _verify_time_core_v1
    require_time = _require_time_core_v1
    verify_store = _verify_store_core_v2
    require_store = _require_store_core_v2
    verify_runtime = _verify_runtime_closure_v1
    require_runtime = _require_runtime_closure_v1
    dependency_errors = (
        _AuthorityErrorV2,
        _OwnerStatusErrorV1,
        _LogicalTimeErrorV1,
        _CasStoreReplayErrorV2,
        _RuntimeClosureErrorV1,
        _RFC8785Error,
    )
    rfc8785_error = _RFC8785Error
    fail_input = FAIL_INPUT_V1
    fail_canonical = FAIL_CANONICAL_V1
    fail_schema = FAIL_SCHEMA_V1
    fail_dependency = FAIL_DEPENDENCY_V1
    fail_binding = FAIL_BINDING_V1
    fail_cas_sequence = FAIL_CAS_SEQUENCE_V1
    fail_result = FAIL_RESULT_V1
    fail_claim_boundary = FAIL_CLAIM_BOUNDARY_V1
    exact_type = type
    exact_bytes = bytes
    exact_str = str
    exact_tuple = tuple
    exact_int = int
    exact_bool = bool
    object_new = object.__new__
    object_setattr = object.__setattr__
    issued: dict[int, tuple[VerifiedR01PureReplayCandidateV1, tuple[object, ...]]] = {}
    result_fields = tuple(result_type.__dataclass_fields__)

    def fail(code: str, detail: str) -> NoReturn:
        raise error_type(code, detail)

    def sha_hex(payload: bytes) -> str:
        return sha256(payload).hexdigest()

    def sha_address(payload: bytes) -> str:
        return "sha256:" + sha_hex(payload)

    def content_id(prefix: str, domain: bytes, body: object) -> str:
        return prefix + sha256(domain + canonical(body)).hexdigest()

    def ordered_digest_root(values: tuple[str, ...]) -> str:
        return sha256(digest_root_domain + canonical(list(values))).hexdigest()

    def decode_exact_artifact(payload: bytes, label: str) -> dict[str, object]:
        if (
            exact_type(payload) is not exact_bytes
            or not payload
            or len(payload) > maximum_canonical_bytes
            or payload.endswith(b"\n")
        ):
            fail(fail_input, f"{label} must be exact non-newline canonical bytes")
        try:
            value = decode_canonical(payload)
        except rfc8785_error as exc:
            raise error_type(fail_canonical, f"{label} is not canonical") from exc
        if exact_type(value) is not dict or set(value) != {"body", "id"}:
            fail(fail_schema, f"{label} root keyset differs")
        if (
            exact_type(value["body"]) is not dict
            or exact_type(value["id"]) is not exact_str
        ):
            fail(fail_schema, f"{label} root field type differs")
        return value

    report_false_claims = {
        "runtime_closure_inputs_durably_retained": False,
        "signed_runtime_attestation_verified": False,
        "service_image_exists": False,
        "service_image_pulled": False,
        "service_image_executed": False,
        "live_runtime_closure_observed": False,
        "entry_exit_closure_equal": False,
        "live_uid_gid_verified": False,
        "live_control_domain_verified": False,
        "key_custody_verified": False,
        "runtime_closure_signature_verified": False,
        "remote_attestation_verified": False,
        "external_cas_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
    }
    bundle_false_claims = {
        "reports_signed": False,
        "reports_durably_retained": False,
        "live_runtime_closure_verified": False,
        "live_uid_gid_verified": False,
        "live_control_domain_verified": False,
        "key_custody_verified": False,
        "external_currentness_verified": False,
        "external_anti_fork_anchor_verified": False,
        "authority_evidence": False,
        "real_r01_receipt_verified": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
    }

    def validate_owner_input(value: object) -> OwnerStatusRawReplayInputsV1:
        if exact_type(value) is not owner_input_type:
            fail(fail_input, "owner inputs must use the exact raw DTO")
        singleton_names = (
            "owner_authorization_payload",
            "caller_supplied_current_head_payload",
            "owner_projection_payload",
        )
        tuple_names = (
            "decision_payloads_in_exact_order",
            "signed_event_payloads_in_exact_order",
            "owner_checkpoint_payloads_in_exact_order",
            "owner_head_payloads_in_exact_order",
            "logical_time_request_payloads_in_exact_order",
            "logical_time_signed_token_payloads_in_exact_order",
            "logical_time_checkpoint_payloads_in_exact_order",
            "logical_time_head_payloads_in_exact_order",
        )
        if any(
            exact_type(getattr(value, name)) is not exact_bytes
            for name in singleton_names
        ):
            fail(fail_input, "owner singleton transport differs")
        for name in tuple_names:
            sequence = getattr(value, name)
            if exact_type(sequence) is not exact_tuple or any(
                exact_type(item) is not exact_bytes for item in sequence
            ):
                fail(fail_input, f"owner raw tuple {name} differs")
        return value

    def validate_cas_input(value: object) -> CasStoreCompleteSessionRawInputsV2:
        if exact_type(value) is not cas_input_type:
            fail(fail_input, "CAS inputs must use the exact raw DTO")
        if exact_type(value.retention_plan_payload) is not exact_bytes:
            fail(fail_input, "CAS retention plan transport differs")
        for name in (
            "request_payloads_in_exact_order",
            "object_payloads_in_exact_order",
            "signed_receipt_payloads_in_exact_order",
            "checkpoint_payloads_in_exact_order",
            "head_payloads_in_exact_order",
            "response_payloads_in_exact_order",
        ):
            sequence = getattr(value, name)
            if exact_type(sequence) is not exact_tuple or any(
                exact_type(item) is not exact_bytes for item in sequence
            ):
                fail(fail_input, f"CAS raw tuple {name} differs")
        return value

    def validate_runtime_input(value: object) -> OfflineRuntimeClosureRawInputsV1:
        if exact_type(value) is not runtime_input_type:
            fail(fail_input, "runtime closure inputs must use the exact raw DTO")
        if exact_type(value.service_name) is not exact_str:
            fail(fail_input, "runtime closure service name differs")
        for name in (
            "runtime_closure_policy_payload",
            "image_closure_evidence_payload",
            "raw_oci_index_payload",
            "raw_oci_manifest_payload",
            "raw_oci_config_payload",
            "runtime_closure_report_payload",
        ):
            if exact_type(getattr(value, name)) is not exact_bytes:
                fail(fail_input, f"runtime closure raw field {name} differs")
        for name in (
            "ordered_raw_layer_chunk_tuples",
            "ordered_support_artifact_chunk_tuples",
        ):
            matrix = getattr(value, name)
            if exact_type(matrix) is not exact_tuple or any(
                exact_type(chunks) is not exact_tuple
                or any(exact_type(chunk) is not exact_bytes for chunk in chunks)
                for chunks in matrix
            ):
                fail(fail_input, f"runtime closure chunk matrix {name} differs")
        return value

    def validate_runtime_report(
        raw: OfflineRuntimeClosureRawInputsV1,
        result: object,
        *,
        ordinal: int,
        lock_id: str,
        profile_id: str,
        registry_id: str,
        attempt_context_id: str,
        expected_binding_id: str,
    ) -> tuple[str, str]:
        layer_digests = tuple(result.ordered_layer_digests)
        support_digests = tuple(result.support_artifact_digests)
        manifest_digests = tuple(result.static_manifest_digests)
        body = {
            "schema_version": report_schema,
            "claim_boundary": report_claim_boundary,
            "stage_b_trust_lock_id": lock_id,
            "authority_profile_id": profile_id,
            "signer_registry_id": registry_id,
            "attempt_context_id": attempt_context_id,
            "service_ordinal": ordinal,
            "service_name": raw.service_name,
            "service_binding_id": expected_binding_id,
            "runtime_closure_policy_id": result.closure_policy_id,
            "runtime_closure_policy_payload_sha256": sha_hex(
                raw.runtime_closure_policy_payload
            ),
            "image_closure_evidence_id": result.evidence_id,
            "image_closure_evidence_payload_sha256": sha_hex(
                raw.image_closure_evidence_payload
            ),
            "raw_oci_index_sha256": result.oci_index_digest,
            "raw_oci_manifest_sha256": result.oci_manifest_digest,
            "raw_oci_config_sha256": result.oci_config_digest,
            "ordered_layer_sha256s": list(layer_digests),
            "ordered_layer_sha256_root": ordered_digest_root(layer_digests),
            "ordered_support_artifact_sha256s": list(support_digests),
            "ordered_support_artifact_sha256_root": ordered_digest_root(
                support_digests
            ),
            "ordered_static_manifest_sha256s": list(manifest_digests),
            "ordered_static_manifest_sha256_root": ordered_digest_root(
                manifest_digests
            ),
            "offline_static_byte_graph_mechanics_verified": True,
            "claims": dict(report_false_claims),
        }
        expected_id = content_id(report_prefix, report_domain, body)
        expected = {"body": body, "id": expected_id}
        observed = decode_exact_artifact(
            raw.runtime_closure_report_payload,
            f"runtime closure report {ordinal}",
        )
        if (
            observed != expected
            or canonical(expected) != raw.runtime_closure_report_payload
        ):
            fail(fail_binding, f"runtime closure report {ordinal} differs")
        return expected_id, sha_address(raw.runtime_closure_report_payload)

    def require_result(value: object) -> VerifiedR01PureReplayCandidateV1:
        if exact_type(value) is not result_type:
            fail(fail_input, "privately issued R01 candidate result required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            fail(fail_result, "R01 candidate result lacks private lineage")
        try:
            intact = all(
                getattr(value, field) is expected
                for field, expected in zip(result_fields, record[1], strict=True)
            )
        except (AttributeError, TypeError):
            intact = False
        if not intact:
            fail(fail_result, "R01 candidate result lineage was mutated")
        for field in (
            "raw_authority_graph_verified",
            "owner_status_replay_core_verified",
            "logical_time_replay_core_verified",
            "cas_complete_session_replay_core_verified",
            "five_offline_runtime_closure_mechanics_verified",
            "runtime_reports_and_bundle_verified",
            "exact_cas_13_plus_8n_sequence_verified",
            "all_presented_mechanics_cross_bound",
            "verifier_only",
        ):
            if (
                exact_type(getattr(value, field)) is not exact_bool
                or getattr(value, field) is not True
            ):
                fail(fail_claim_boundary, f"result {field} differs")
        for field in (
            "external_currentness_verified",
            "external_anti_fork_anchor_verified",
            "cas_durability_verified",
            "live_cas_store_verified",
            "runtime_closure_inputs_durably_retained",
            "live_runtime_closure_verified",
            "live_uid_gid_verified",
            "live_control_domain_verified",
            "key_custody_verified",
            "owner_signature_authenticated",
            "signed_runtime_attestations_verified",
            "authority_evidence",
            "real_r01_receipt_verified",
            "r01_satisfied",
            "readiness_overlay_issued",
            "readiness_evidence",
            "execution_authorized",
        ):
            if (
                exact_type(getattr(value, field)) is not exact_bool
                or getattr(value, field) is not False
            ):
                fail(fail_claim_boundary, f"result {field} differs")
        return value

    def issue_result(values: dict[str, object]) -> VerifiedR01PureReplayCandidateV1:
        if set(values) != set(result_fields):
            fail(fail_result, "private result issuance field set differs")
        value = object_new(result_type)
        for field in result_fields:
            object_setattr(value, field, values[field])
        snapshot = tuple(getattr(value, field) for field in result_fields)
        issued[id(value)] = (value, snapshot)
        return require_result(value)

    def verify(
        *,
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
        expected_attempt_context_id: str,
        expected_logical_time_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
        owner_status_inputs: OwnerStatusRawReplayInputsV1,
        cas_store_inputs: CasStoreCompleteSessionRawInputsV2,
        runtime_closure_inputs_in_exact_service_order: tuple[
            OfflineRuntimeClosureRawInputsV1, ...
        ],
        runtime_closure_report_bundle_payload: bytes,
    ) -> VerifiedR01PureReplayCandidateV1:
        """Replay and exactly cross-bind one mechanics-only R01 candidate."""

        for payload in (
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest,
            runtime_closure_report_bundle_payload,
        ):
            if exact_type(payload) is not exact_bytes:
                fail(fail_input, "singleton transports must be exact bytes")
        if (
            len(out_of_band_expected_lock_payload_sha256_digest) != 32
            or exact_type(expected_attempt_context_id) is not exact_str
            or not expected_attempt_context_id
        ):
            fail(fail_input, "out-of-band digest or attempt context differs")
        if exact_type(expected_logical_time_event_plan) is not exact_tuple or any(
            exact_type(row) is not time_event_type
            for row in expected_logical_time_event_plan
        ):
            fail(fail_input, "logical-time event plan must use exact DTO rows")
        owner = validate_owner_input(owner_status_inputs)
        cas = validate_cas_input(cas_store_inputs)
        if (
            exact_type(runtime_closure_inputs_in_exact_service_order)
            is not exact_tuple
            or len(runtime_closure_inputs_in_exact_service_order) != len(service_order)
        ):
            fail(
                fail_input,
                "runtime closure tuple must contain exact five services",
            )
        runtime_inputs = tuple(
            validate_runtime_input(value)
            for value in runtime_closure_inputs_in_exact_service_order
        )
        if tuple(value.service_name for value in runtime_inputs) != service_order:
            fail(fail_binding, "runtime closure service order differs")

        try:
            lock = load_lock(
                stage_b_trust_lock_payload,
                out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
                out_of_band_expected_payload_sha256_digest=(
                    out_of_band_expected_lock_payload_sha256_digest
                ),
            )
            profile = load_profile(authority_profile_payload, lock)
            registry = load_registry(signer_registry_payload, lock, profile)
            lock, profile, registry = require_authority(lock, profile, registry)

            time_result = require_time(
                verify_time(
                    stage_b_trust_lock_payload,
                    authority_profile_payload,
                    signer_registry_payload,
                    owner.logical_time_request_payloads_in_exact_order,
                    owner.logical_time_signed_token_payloads_in_exact_order,
                    owner.logical_time_checkpoint_payloads_in_exact_order,
                    owner.logical_time_head_payloads_in_exact_order,
                    out_of_band_expected_lock_id_bytes=(
                        out_of_band_expected_lock_id_bytes
                    ),
                    out_of_band_expected_payload_sha256_digest=(
                        out_of_band_expected_lock_payload_sha256_digest
                    ),
                    expected_attempt_context_id=expected_attempt_context_id,
                    expected_event_plan=expected_logical_time_event_plan,
                )
            )
            owner_result = require_owner(
                verify_owner(
                    stage_b_trust_lock_payload,
                    authority_profile_payload,
                    signer_registry_payload,
                    owner.owner_authorization_payload,
                    owner.decision_payloads_in_exact_order,
                    owner.signed_event_payloads_in_exact_order,
                    owner.owner_checkpoint_payloads_in_exact_order,
                    owner.owner_head_payloads_in_exact_order,
                    owner.caller_supplied_current_head_payload,
                    owner.owner_projection_payload,
                    owner.logical_time_request_payloads_in_exact_order,
                    owner.logical_time_signed_token_payloads_in_exact_order,
                    owner.logical_time_checkpoint_payloads_in_exact_order,
                    owner.logical_time_head_payloads_in_exact_order,
                    out_of_band_expected_lock_id_bytes=(
                        out_of_band_expected_lock_id_bytes
                    ),
                    out_of_band_expected_payload_sha256_digest=(
                        out_of_band_expected_lock_payload_sha256_digest
                    ),
                    expected_attempt_context_id=expected_attempt_context_id,
                    expected_logical_time_event_plan=(
                        expected_logical_time_event_plan
                    ),
                )
            )
        except dependency_errors as exc:
            raise error_type(
                fail_dependency, "authority/owner/logical-time pure replay rejected"
            ) from exc

        graph_common = (
            lock.artifact_id,
            profile.artifact_id,
            registry.artifact_id,
            expected_attempt_context_id,
        )
        if not (
            time_result.stage_b_trust_lock_id == graph_common[0]
            and time_result.authority_profile_id == graph_common[1]
            and time_result.signer_registry_id == graph_common[2]
            and time_result.attempt_context_id == graph_common[3]
            and owner_result.stage_b_trust_lock_id == graph_common[0]
            and owner_result.authority_profile_id == graph_common[1]
            and owner_result.signer_registry_id == graph_common[2]
            and owner_result.attempt_context_id == graph_common[3]
            and owner_result.logical_time_ledger_id == time_result.ledger_id
            and owner_result.logical_time_final_head_id == time_result.final_head_id
            and owner_result.logical_time_request_ids == time_result.request_ids
            and owner_result.logical_time_token_ids == time_result.token_ids
            and owner_result.owner_authorization_payload_sha256
            == sha_address(owner.owner_authorization_payload)
            and owner_result.current_head_payload_sha256
            == sha_address(owner.caller_supplied_current_head_payload)
            and owner_result.projection_payload_sha256
            == sha_address(owner.owner_projection_payload)
            and time_result.final_head_payload_sha256
            == sha_address(owner.logical_time_head_payloads_in_exact_order[-1])
        ):
            fail(fail_binding, "owner/time/authority graph context differs")
        event_count = owner_result.verified_event_count
        if exact_type(event_count) is not exact_int or event_count not in (1, 2):
            fail(fail_binding, "owner event count must be exactly one or two")
        if event_count != len(expected_logical_time_event_plan):
            fail(fail_binding, "owner and logical-time event count differs")

        service_bindings = profile.body["service_static_bindings_in_exact_order"]
        if (
            exact_type(service_bindings) is not exact_tuple
            or len(service_bindings) != 5
        ):
            fail(fail_binding, "authority static service binding sequence differs")
        report_ids: list[str] = []
        report_shas: list[str] = []
        try:
            for ordinal, raw in enumerate(runtime_inputs):
                runtime_result = require_runtime(
                    verify_runtime(
                        stage_b_trust_lock_payload=stage_b_trust_lock_payload,
                        authority_profile_payload=authority_profile_payload,
                        signer_registry_payload=signer_registry_payload,
                        out_of_band_expected_lock_id_bytes=(
                            out_of_band_expected_lock_id_bytes
                        ),
                        out_of_band_expected_lock_payload_sha256_digest=(
                            out_of_band_expected_lock_payload_sha256_digest
                        ),
                        service_name=raw.service_name,
                        runtime_closure_policy_payload=(
                            raw.runtime_closure_policy_payload
                        ),
                        image_closure_evidence_payload=(
                            raw.image_closure_evidence_payload
                        ),
                        raw_oci_index_payload=raw.raw_oci_index_payload,
                        raw_oci_manifest_payload=raw.raw_oci_manifest_payload,
                        raw_oci_config_payload=raw.raw_oci_config_payload,
                        ordered_raw_layer_chunk_tuples=(
                            raw.ordered_raw_layer_chunk_tuples
                        ),
                        ordered_support_artifact_chunk_tuples=(
                            raw.ordered_support_artifact_chunk_tuples
                        ),
                    )
                )
                binding = service_bindings[ordinal]
                if not (
                    runtime_result.stage_b_trust_lock_id == graph_common[0]
                    and runtime_result.authority_profile_id == graph_common[1]
                    and runtime_result.signer_registry_id == graph_common[2]
                    and runtime_result.service_name == service_order[ordinal]
                    and runtime_result.service_binding_id == binding["binding_id"]
                    and runtime_result.offline_static_byte_graph_mechanics_verified
                    is True
                    and runtime_result.mechanics_only is True
                    and runtime_result.service_image_exists is False
                    and runtime_result.service_image_pulled is False
                    and runtime_result.service_image_executed is False
                    and runtime_result.live_runtime_closure_observed is False
                    and runtime_result.entry_exit_closure_equal is False
                    and runtime_result.live_uid_gid_verified is False
                    and runtime_result.live_control_domain_verified is False
                    and runtime_result.key_custody_verified is False
                    and runtime_result.runtime_closure_signature_verified is False
                    and runtime_result.remote_attestation_verified is False
                    and runtime_result.external_cas_verified is False
                    and runtime_result.authority_evidence is False
                    and runtime_result.readiness_evidence is False
                    and runtime_result.r01_satisfied is False
                    and runtime_result.execution_authorized is False
                ):
                    fail(fail_binding, f"runtime result {ordinal} boundary differs")
                report_id, report_sha = validate_runtime_report(
                    raw,
                    runtime_result,
                    ordinal=ordinal,
                    lock_id=graph_common[0],
                    profile_id=graph_common[1],
                    registry_id=graph_common[2],
                    attempt_context_id=graph_common[3],
                    expected_binding_id=binding["binding_id"],
                )
                report_ids.append(report_id)
                report_shas.append(report_sha)
        except dependency_errors as exc:
            raise error_type(
                fail_dependency, "offline runtime-closure replay rejected"
            ) from exc

        bundle_body = {
            "schema_version": bundle_schema,
            "claim_boundary": bundle_claim_boundary,
            "stage_b_trust_lock_id": graph_common[0],
            "authority_profile_id": graph_common[1],
            "signer_registry_id": graph_common[2],
            "attempt_context_id": graph_common[3],
            "service_names_in_exact_order": list(service_order),
            "runtime_closure_report_ids_in_exact_order": list(report_ids),
            "runtime_closure_report_payload_sha256s_in_exact_order": list(
                report_shas
            ),
            "exact_five_reports_presented": True,
            "claims": dict(bundle_false_claims),
        }
        bundle_id = content_id(bundle_prefix, bundle_domain, bundle_body)
        expected_bundle = {"body": bundle_body, "id": bundle_id}
        observed_bundle = decode_exact_artifact(
            runtime_closure_report_bundle_payload, "runtime closure report bundle"
        )
        if (
            observed_bundle != expected_bundle
            or canonical(expected_bundle) != runtime_closure_report_bundle_payload
        ):
            fail(fail_binding, "runtime closure report bundle differs")
        bundle_sha = sha_address(runtime_closure_report_bundle_payload)

        expected_objects: list[tuple[str, str, bytes]] = [
            (base_kinds[0], lock.artifact_id, stage_b_trust_lock_payload),
            (base_kinds[1], profile.artifact_id, authority_profile_payload),
            (base_kinds[2], registry.artifact_id, signer_registry_payload),
            (
                base_kinds[3],
                owner_result.owner_authorization_id,
                owner.owner_authorization_payload,
            ),
        ]
        expected_objects.extend(
            (
                base_kinds[4 + ordinal],
                report_ids[ordinal],
                runtime_inputs[ordinal].runtime_closure_report_payload,
            )
            for ordinal in range(5)
        )
        expected_objects.extend(
            (
                (base_kinds[9], bundle_id, runtime_closure_report_bundle_payload),
                (
                    base_kinds[10],
                    owner_result.projection_id,
                    owner.owner_projection_payload,
                ),
                (
                    base_kinds[11],
                    owner_result.head_ids[0],
                    owner.owner_head_payloads_in_exact_order[0],
                ),
                (
                    base_kinds[12],
                    time_result.head_ids[0],
                    owner.logical_time_head_payloads_in_exact_order[0],
                ),
            )
        )
        for index in range(event_count):
            expected_objects.extend(
                (
                    (
                        per_event_kinds[0],
                        owner_result.decision_ids[index],
                        owner.decision_payloads_in_exact_order[index],
                    ),
                    (
                        per_event_kinds[1],
                        owner_result.event_ids[index],
                        owner.signed_event_payloads_in_exact_order[index],
                    ),
                    (
                        per_event_kinds[2],
                        owner_result.checkpoint_ids[index],
                        owner.owner_checkpoint_payloads_in_exact_order[index],
                    ),
                    (
                        per_event_kinds[3],
                        owner_result.head_ids[index + 1],
                        owner.owner_head_payloads_in_exact_order[index + 1],
                    ),
                    (
                        per_event_kinds[4],
                        time_result.request_ids[index],
                        owner.logical_time_request_payloads_in_exact_order[index],
                    ),
                    (
                        per_event_kinds[5],
                        time_result.token_ids[index],
                        owner.logical_time_signed_token_payloads_in_exact_order[index],
                    ),
                    (
                        per_event_kinds[6],
                        time_result.checkpoint_ids[index],
                        owner.logical_time_checkpoint_payloads_in_exact_order[index],
                    ),
                    (
                        per_event_kinds[7],
                        time_result.head_ids[index + 1],
                        owner.logical_time_head_payloads_in_exact_order[index + 1],
                    ),
                )
            )
        exact_object_count = 13 + 8 * event_count
        if len(expected_objects) != exact_object_count:
            fail(fail_cas_sequence, "internal 13+8N object construction differs")
        expected_object_payloads = tuple(row[2] for row in expected_objects)
        if cas.object_payloads_in_exact_order != expected_object_payloads:
            fail(
                fail_cas_sequence,
                "CAS object bytes/order differ from exact 13+8N graph",
            )

        try:
            store_result = require_store(
                verify_store(
                    stage_b_lock_payload=stage_b_trust_lock_payload,
                    authority_profile_payload=authority_profile_payload,
                    signer_registry_payload=signer_registry_payload,
                    out_of_band_expected_lock_id_bytes=(
                        out_of_band_expected_lock_id_bytes
                    ),
                    out_of_band_expected_lock_payload_sha256_digest=(
                        out_of_band_expected_lock_payload_sha256_digest
                    ),
                    retention_plan_payload=cas.retention_plan_payload,
                    request_payloads_in_exact_order=cas.request_payloads_in_exact_order,
                    object_payloads_in_exact_order=cas.object_payloads_in_exact_order,
                    signed_receipt_payloads_in_exact_order=(
                        cas.signed_receipt_payloads_in_exact_order
                    ),
                    checkpoint_payloads_in_exact_order=(
                        cas.checkpoint_payloads_in_exact_order
                    ),
                    head_payloads_in_exact_order=cas.head_payloads_in_exact_order,
                    response_payloads_in_exact_order=(
                        cas.response_payloads_in_exact_order
                    ),
                )
            )
        except dependency_errors as exc:
            raise error_type(
                fail_dependency, "generic CAS/2 complete-session replay rejected"
            ) from exc
        if not (
            store_result.authority_profile_id == graph_common[1]
            and store_result.signer_registry_id == graph_common[2]
            and store_result.attempt_context_id == graph_common[3]
            and store_result.expected_object_count == exact_object_count
            and store_result.canonical_bytes_verified is True
            and store_result.content_addresses_verified is True
            and store_result.ed25519_signatures_verified is True
            and store_result.append_chain_verified is True
            and store_result.if_absent_idempotent_complete_semantics_verified is True
            and store_result.durability_verified is False
            and store_result.live_cas_store_verified is False
            and store_result.authority_evidence is False
            and store_result.readiness_evidence is False
            and store_result.execution_authorized is False
        ):
            fail(fail_binding, "CAS result context or claim boundary differs")
        object_bindings = store_result.object_bindings_in_exact_order
        if (
            exact_type(object_bindings) is not exact_tuple
            or len(object_bindings) != exact_object_count
        ):
            fail(fail_cas_sequence, "CAS object-binding count differs")
        for ordinal, ((kind, artifact_id, payload), binding) in enumerate(
            zip(expected_objects, object_bindings, strict=True)
        ):
            payload_sha = sha_address(payload)
            if not (
                exact_type(binding) is exact_tuple
                and len(binding) == 10
                and binding[0] == ordinal
                and binding[1] == kind
                and binding[2] == artifact_id
                and binding[3] == payload_sha
                and binding[4] == payload_sha
                and binding[5] == len(payload)
            ):
                fail(fail_cas_sequence, f"CAS binding {ordinal} differs")

        candidate_subject = {
            "schema_version": schema,
            "claim_boundary": claim_boundary,
            "stage_b_trust_lock_id": graph_common[0],
            "authority_profile_id": graph_common[1],
            "signer_registry_id": graph_common[2],
            "attempt_context_id": graph_common[3],
            "owner_current_head_id": owner_result.current_head_id,
            "owner_current_head_payload_sha256": (
                owner_result.current_head_payload_sha256
            ),
            "owner_complete_event_set_root": owner_result.complete_event_set_root,
            "logical_time_final_head_id": time_result.final_head_id,
            "cas_retention_session_id": store_result.retention_session_id,
            "cas_chain_id": store_result.chain_id,
            "cas_complete_head_state_id": store_result.complete_head_state_id,
            "runtime_closure_report_bundle_id": bundle_id,
            "owner_event_count": event_count,
            "exact_cas_object_count": exact_object_count,
        }
        candidate_id = content_id(candidate_prefix, candidate_domain, candidate_subject)
        return issue_result(
            {
                "schema_version": schema,
                "candidate_id": candidate_id,
                "claim_boundary": claim_boundary,
                "stage_b_trust_lock_id": graph_common[0],
                "authority_profile_id": graph_common[1],
                "signer_registry_id": graph_common[2],
                "attempt_context_id": graph_common[3],
                "owner_authorization_id": owner_result.owner_authorization_id,
                "owner_status_ledger_id": owner_result.ledger_id,
                "owner_current_head_id": owner_result.current_head_id,
                "owner_current_head_payload_sha256": (
                    owner_result.current_head_payload_sha256
                ),
                "owner_complete_event_set_root": owner_result.complete_event_set_root,
                "owner_latest_effective_status": owner_result.latest_effective_status,
                "logical_time_ledger_id": time_result.ledger_id,
                "logical_time_final_head_id": time_result.final_head_id,
                "cas_retention_session_id": store_result.retention_session_id,
                "cas_chain_id": store_result.chain_id,
                "cas_complete_head_state_id": store_result.complete_head_state_id,
                "cas_complete_checkpoint_sha256": (
                    store_result.complete_checkpoint_sha256
                ),
                "runtime_closure_report_ids_in_exact_service_order": tuple(report_ids),
                "runtime_closure_report_payload_sha256s_in_exact_service_order": tuple(
                    report_shas
                ),
                "runtime_closure_report_bundle_id": bundle_id,
                "runtime_closure_report_bundle_payload_sha256": bundle_sha,
                "owner_event_count": event_count,
                "exact_cas_object_count": exact_object_count,
                "raw_authority_graph_verified": True,
                "owner_status_replay_core_verified": True,
                "logical_time_replay_core_verified": True,
                "cas_complete_session_replay_core_verified": True,
                "five_offline_runtime_closure_mechanics_verified": True,
                "runtime_reports_and_bundle_verified": True,
                "exact_cas_13_plus_8n_sequence_verified": True,
                "all_presented_mechanics_cross_bound": True,
                "verifier_only": True,
                "external_currentness_verified": False,
                "external_anti_fork_anchor_verified": False,
                "cas_durability_verified": False,
                "live_cas_store_verified": False,
                "runtime_closure_inputs_durably_retained": False,
                "live_runtime_closure_verified": False,
                "live_uid_gid_verified": False,
                "live_control_domain_verified": False,
                "key_custody_verified": False,
                "owner_signature_authenticated": False,
                "signed_runtime_attestations_verified": False,
                "authority_evidence": False,
                "real_r01_receipt_verified": False,
                "r01_satisfied": False,
                "readiness_overlay_issued": False,
                "readiness_evidence": False,
                "execution_authorized": False,
            }
        )

    return verify, require_result


(
    verify_r01_pure_replay_candidate_v1,
    require_verified_r01_pure_replay_candidate_v1,
) = _build_r01_pure_replay_v1()
del _build_r01_pure_replay_v1


def _build_not_implemented_entry_points_v1():
    """Definition-capture the stable fail-closed public placeholder contract."""

    error_type = Phase2BStep5R01PureReplayErrorV1
    code = FAIL_NOT_IMPLEMENTED_V1

    def verify_real_r01(
        *args: object, **kwargs: object
    ) -> NoReturn:
        """Fail closed: this mechanics composer cannot verify a real R01."""

        raise error_type(
            code,
            "real R01 receipt verification requires external currentness and authority",
        )

    def derive_overlay(*args: object, **kwargs: object) -> NoReturn:
        """Fail closed: candidate mechanics cannot issue a readiness overlay."""

        raise error_type(
            code,
            "R01 readiness overlay issuance is not implemented",
        )

    return verify_real_r01, derive_overlay


(
    verify_real_r01_authorization_status_receipt_v1,
    derive_r01_readiness_overlay_v1,
) = _build_not_implemented_entry_points_v1()
del _build_not_implemented_entry_points_v1


__all__ = [
    "CAS_ARTIFACT_KINDS_BASE_V1",
    "CAS_ARTIFACT_KINDS_PER_EVENT_V1",
    "CLAIM_BOUNDARY_V1",
    "CasStoreCompleteSessionRawInputsV2",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CAS_SEQUENCE_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_DEPENDENCY_V1",
    "FAIL_INPUT_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "FAIL_RESULT_V1",
    "FAIL_SCHEMA_V1",
    "OfflineRuntimeClosureRawInputsV1",
    "OwnerStatusRawReplayInputsV1",
    "Phase2BStep5R01PureReplayErrorV1",
    "R01_PURE_REPLAY_CANDIDATE_PREFIX_V1",
    "RUNTIME_CLOSURE_REPORT_BUNDLE_PREFIX_V1",
    "RUNTIME_CLOSURE_REPORT_BUNDLE_SCHEMA_V1",
    "RUNTIME_CLOSURE_REPORT_PREFIX_V1",
    "RUNTIME_CLOSURE_REPORT_SCHEMA_V1",
    "RUNTIME_REPORT_BUNDLE_CLAIM_BOUNDARY_V1",
    "RUNTIME_REPORT_CLAIM_BOUNDARY_V1",
    "SCHEMA_VERSION",
    "VerifiedR01PureReplayCandidateV1",
    "derive_r01_readiness_overlay_v1",
    "require_verified_r01_pure_replay_candidate_v1",
    "verify_r01_pure_replay_candidate_v1",
    "verify_real_r01_authorization_status_receipt_v1",
]
