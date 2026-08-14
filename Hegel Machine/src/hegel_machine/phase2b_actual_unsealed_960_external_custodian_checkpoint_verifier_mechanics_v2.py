"""Candidate external-custodian checkpoint verification mechanics for Phase-2B.

The verifier checks one supplied terminal checkpoint, a four-role Ed25519
candidate registry, an IF_ABSENT store receipt, and a signed declared-time
token.  The registry and every identity are caller supplied.  Therefore a
successful result proves only the closed content/signature mechanics; it does
not establish an externally trusted signer registry, custodian independence,
store authority or rollback resistance, authoritative time, execution,
scoring, formal gates, effect, or C1 evidence.

The module performs no filesystem, clock, network, runner, scorer, key
generation, or signature generation operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
from typing import Final

from .hashing import canonical_json

try:  # Optional backend; absence is a fail-closed ordinary result.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - backend monkeypatch tests cover this.
    _Ed25519PublicKey = None


PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-external-custodian-checkpoint-verifier-mechanics/2"
)
PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_SUPPLIED_EXTERNAL_AUTHORITY_CANDIDATE_MECHANICS_ONLY"
)
PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE: Final = (
    "SUPPLIED_EXTERNAL_AUTHORITY_CANDIDATE_NOT_PRETRUSTED_SINGLE_TERMINAL_CHECKPOINT"
)
PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_ED25519_BACKEND_AVAILABLE: Final = (
    _Ed25519PublicKey is not None
)

_MAX_TEXT_BYTES: Final = 4_096
_MAX_OBJECT_KEY_BYTES: Final = 2_048
_MAX_U63: Final = (1 << 63) - 1
_REQUIRED_SIGNATURE_COUNT: Final = 4
_KEY_EPOCH: Final = 1
_CHECKPOINT_SEQUENCE: Final = 0
_PREVIOUS_GENERATION: Final = 0
_NEW_GENERATION: Final = 1
_CAS_CONDITION: Final = "IF_ABSENT"
_NO_PREDECESSOR: Final = None

_AUTHORITY_REQUIREMENT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/AUTHORITY_REQUIREMENT/V2\x00"
_AUTHORITY_REQUIREMENT_PREFIX: Final = "phase2b_actual_unsealed_960_external_authority_requirement_v2_"
PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_STATEMENTS: Final = (
    ("claim_boundary", "requirements_for_live_external_authority_not_evidence_that_authority_exists"),
    ("required_distinct_roles", ("GOVERNANCE_ANCHOR", "CUSTODIAN", "STORE_RECEIPT", "TIME_ATTESTOR")),
    ("registry_requirement", "pretrusted_out_of_band_registry_issuance_and_identity_ceremony_required"),
    ("custodian_requirement", "organizationally_independent_custodian_identity_evidence_required"),
    ("store_requirement", "external_compare_and_swap_receipt_and_rollback_witness_required"),
    ("time_requirement", "independently_pinned_authoritative_time_provider_evidence_required"),
)
PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_ID: Final = (
    _AUTHORITY_REQUIREMENT_PREFIX
    + hashlib.sha256(
        _AUTHORITY_REQUIREMENT_DOMAIN
        + canonical_json(PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_STATEMENTS).encode("utf-8")
    ).hexdigest()
)

_KEY_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/KEY/V2\x00"
_KEY_ROOT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/KEY_ROOT/V2\x00"
_PROFILE_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/PROFILE/V2\x00"
_SUBJECT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/SUBJECT/V2\x00"
_STORE_SIGNATURE_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/STORE_SIGNATURE/V2\x00"
_STORE_RECEIPT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/STORE_RECEIPT/V2\x00"
_TIME_SIGNATURE_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/TIME_SIGNATURE/V2\x00"
_TIME_TOKEN_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/TIME_TOKEN/V2\x00"
_CHECKPOINT_SIGNATURE_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/CHECKPOINT_SIGNATURE/V2\x00"
_CHECKPOINT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/CHECKPOINT/V2\x00"
_RESULT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/RESULT/V2\x00"
_SCHEMA_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/SCHEMA/V2\x00"
_POLICY_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/POLICY/V2\x00"

_KEY_PREFIX: Final = "phase2b_actual_unsealed_960_external_candidate_key_v2_"
_PROFILE_PREFIX: Final = "phase2b_actual_unsealed_960_external_candidate_profile_v2_"
_SUBJECT_PREFIX: Final = "phase2b_actual_unsealed_960_external_checkpoint_subject_v2_"
_STORE_SERVICE_PREFIX: Final = "phase2b_actual_unsealed_960_external_store_service_v2_"
_STORE_RECEIPT_PREFIX: Final = "phase2b_actual_unsealed_960_external_store_receipt_v2_"
_TIME_SOURCE_PREFIX: Final = "phase2b_actual_unsealed_960_external_time_source_v2_"
_TIME_TOKEN_PREFIX: Final = "phase2b_actual_unsealed_960_external_time_token_v2_"
_CHECKPOINT_PREFIX: Final = "phase2b_actual_unsealed_960_external_checkpoint_v2_"
_RESULT_PREFIX: Final = "phase2b_actual_unsealed_960_external_checkpoint_verifier_result_v2_"
_SCHEMA_PREFIX: Final = "phase2b_actual_unsealed_960_external_checkpoint_verifier_schema_v2_"
_POLICY_PREFIX: Final = "phase2b_actual_unsealed_960_external_checkpoint_verifier_policy_v2_"

_CAMPAIGN_PREFIX: Final = "phase2b_actual_unsealed_960_campaign_v2_"
_ATTEMPT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_v2_"
_SIGNED_CHAIN_PREFIX: Final = "phase2b_actual_unsealed_960_signed_attempt_chain_v2_"
_TERMINAL_EVENT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_event_v2_"
_LOCAL_LEDGER_PREFIX: Final = "phase2b_actual_unsealed_960_local_attempt_ledger_v2_"
_LOCAL_TERMINAL_PREFIX: Final = "phase2b_actual_unsealed_960_local_terminal_record_v2_"


class ExternalCustodianCandidateRoleV2(str, Enum):
    GOVERNANCE_ANCHOR = "GOVERNANCE_ANCHOR"
    CUSTODIAN = "CUSTODIAN"
    STORE_RECEIPT = "STORE_RECEIPT"
    TIME_ATTESTOR = "TIME_ATTESTOR"


_ROLE_ORDER: Final = tuple(ExternalCustodianCandidateRoleV2)


class ExternalCustodianSignaturePurposeV2(str, Enum):
    GOVERNANCE_CHECKPOINT = "GOVERNANCE_CHECKPOINT"
    CUSTODIAN_CHECKPOINT = "CUSTODIAN_CHECKPOINT"
    STORE_IF_ABSENT_RECEIPT = "STORE_IF_ABSENT_RECEIPT"
    DECLARED_TIME_TOKEN = "DECLARED_TIME_TOKEN"


_PURPOSE_BY_ROLE: Final = {
    ExternalCustodianCandidateRoleV2.GOVERNANCE_ANCHOR: ExternalCustodianSignaturePurposeV2.GOVERNANCE_CHECKPOINT,
    ExternalCustodianCandidateRoleV2.CUSTODIAN: ExternalCustodianSignaturePurposeV2.CUSTODIAN_CHECKPOINT,
    ExternalCustodianCandidateRoleV2.STORE_RECEIPT: ExternalCustodianSignaturePurposeV2.STORE_IF_ABSENT_RECEIPT,
    ExternalCustodianCandidateRoleV2.TIME_ATTESTOR: ExternalCustodianSignaturePurposeV2.DECLARED_TIME_TOKEN,
}


class ExternalCustodianCheckpointDispositionV2(str, Enum):
    CANDIDATE_EXTERNAL_CHECKPOINT_MECHANICS_VERIFIED_NOT_EXTERNAL_AUTHORITY = (
        "CANDIDATE_EXTERNAL_CHECKPOINT_MECHANICS_VERIFIED_NOT_EXTERNAL_AUTHORITY"
    )
    REJECTED = "REJECTED"


class ExternalCustodianCheckpointReasonV2(str, Enum):
    FOUR_ROLE_CANDIDATE_CHECKPOINT_VERIFIED_NOT_PRETRUSTED = (
        "FOUR_ROLE_CANDIDATE_CHECKPOINT_VERIFIED_NOT_PRETRUSTED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    CROSS_BINDING_INVALID = "CROSS_BINDING_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class _SignatureBackendUnavailable(Exception):
    pass


class _SignatureInvalid(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ExternalCustodianCandidateSignerV2:
    role: ExternalCustodianCandidateRoleV2
    key_epoch: int
    public_key: bytes
    key_id: str


@dataclass(frozen=True, slots=True)
class ExternalCustodianAuthorityCandidateProfileV2:
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    scope: str
    authority_requirement_id: str
    campaign_id: str
    attempt_id: str
    external_anchor_locator: str
    external_anchor_evidence_sha256: str
    signer_keys: tuple[ExternalCustodianCandidateSignerV2, ...]
    signer_key_ids_root_sha256: str
    required_signature_count: int
    profile_id: str


@dataclass(frozen=True, slots=True)
class ExternalCustodianDetachedSignatureV2:
    role: ExternalCustodianCandidateRoleV2
    purpose: ExternalCustodianSignaturePurposeV2
    key_epoch: int
    key_id: str
    signature: bytes


@dataclass(frozen=True, slots=True)
class ExternalCustodianStoreReceiptCandidateV2:
    version: str
    profile_id: str
    campaign_id: str
    attempt_id: str
    subject_id: str
    store_service_id: str
    object_key: str
    previous_generation: int
    new_generation: int
    cas_condition: str
    stored_content_sha256: str
    detached_signature: ExternalCustodianDetachedSignatureV2
    receipt_id: str


@dataclass(frozen=True, slots=True)
class ExternalCustodianTimeTokenCandidateV2:
    version: str
    profile_id: str
    campaign_id: str
    attempt_id: str
    subject_id: str
    time_source_id: str
    previous_declared_time_ns: int
    declared_time_ns: int
    detached_signature: ExternalCustodianDetachedSignatureV2
    time_token_id: str


@dataclass(frozen=True, slots=True)
class ExternalCustodianCheckpointCandidateV2:
    version: str
    profile_id: str
    campaign_id: str
    attempt_id: str
    checkpoint_sequence: int
    predecessor_checkpoint_id: None
    signed_attempt_chain_id: str
    signed_attempt_chain_root_sha256: str
    terminal_event_id: str
    local_ledger_id: str
    local_terminal_record_id: str
    subject_id: str
    store_receipt_id: str
    time_token_id: str
    checkpoint_signatures: tuple[ExternalCustodianDetachedSignatureV2, ...]
    checkpoint_id: str


_MECHANICS_CLAIMS: Final = (
    "candidate_profile_content_address_verified",
    "four_role_distinct_key_registry_verified",
    "role_epoch_purpose_domain_binding_verified",
    "terminal_subject_commitment_verified",
    "store_if_absent_generation_transition_verified",
    "store_receipt_signature_verified",
    "declared_time_monotonicity_verified",
    "time_token_signature_verified",
    "governance_checkpoint_signature_verified",
    "custodian_checkpoint_signature_verified",
    "checkpoint_cross_bindings_verified",
    "checkpoint_content_address_verified",
    "ordinary_exception_atomic_rejection_verified",
)

_NONCLAIMS: Final = (
    "external_trust_anchor_verified",
    "registry_issuance_authority_verified",
    "custodian_identity_verified",
    "signer_independence_verified",
    "qualified_cryptographic_backend_verified",
    "external_store_identity_verified",
    "external_store_authority_verified",
    "external_store_receipt_authority_verified",
    "external_store_rollback_resistance_verified",
    "external_store_durability_verified",
    "hostile_administrator_tamper_resistance_verified",
    "cross_host_global_one_shot_verified",
    "authoritative_time_source_verified",
    "authoritative_time_verified",
    "timeline_observed",
    "authoritative_timeline_order_verified",
    "signed_attempt_chain_authority_verified",
    "local_ledger_externalized_verified",
    "authoritative_attempt_denominator_verified",
    "attempt_registry_complete",
    "retry_authorization_verified",
    "rerun_policy_enforced",
    "one_shot_policy_enforced",
    "terminal_append_receipt_verified",
    "campaign_finalization_verified",
    "actual_evidence_inputs_accepted",
    "admission_evidence_contract_complete",
    "authoritative_evidence_verifier_implemented",
    "admission_ready",
    "execution_authorized",
    "evidence_supplied",
    "evidence_verified",
    "input_archive_authority_verified",
    "prediction_archive_authority_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    "runtime_attestation_authority_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "wilson_bounds_evaluated",
    "preservation_evaluated",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "baseline_outputs_verified",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)

_ISSUANCE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class ExternalCustodianCheckpointVerifierMechanicsV2:
    def __new__(cls, token: object = None) -> "ExternalCustodianCheckpointVerifierMechanicsV2":
        if token is not _ISSUANCE_TOKEN:
            raise TypeError("success results are privately issued")
        return object.__new__(cls)

    disposition: ExternalCustodianCheckpointDispositionV2
    reason: ExternalCustodianCheckpointReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    scope: str
    result_id: str
    profile_id: str
    subject_id: str
    store_receipt_id: str
    time_token_id: str
    checkpoint_id: str
    required_signature_count: int
    verified_signature_count: int
    candidate_profile_content_address_verified: bool
    four_role_distinct_key_registry_verified: bool
    role_epoch_purpose_domain_binding_verified: bool
    terminal_subject_commitment_verified: bool
    store_if_absent_generation_transition_verified: bool
    store_receipt_signature_verified: bool
    declared_time_monotonicity_verified: bool
    time_token_signature_verified: bool
    governance_checkpoint_signature_verified: bool
    custodian_checkpoint_signature_verified: bool
    checkpoint_cross_bindings_verified: bool
    checkpoint_content_address_verified: bool
    ordinary_exception_atomic_rejection_verified: bool
    external_trust_anchor_verified: bool
    registry_issuance_authority_verified: bool
    custodian_identity_verified: bool
    signer_independence_verified: bool
    qualified_cryptographic_backend_verified: bool
    external_store_identity_verified: bool
    external_store_authority_verified: bool
    external_store_receipt_authority_verified: bool
    external_store_rollback_resistance_verified: bool
    external_store_durability_verified: bool
    hostile_administrator_tamper_resistance_verified: bool
    cross_host_global_one_shot_verified: bool
    authoritative_time_source_verified: bool
    authoritative_time_verified: bool
    timeline_observed: bool
    authoritative_timeline_order_verified: bool
    signed_attempt_chain_authority_verified: bool
    local_ledger_externalized_verified: bool
    authoritative_attempt_denominator_verified: bool
    attempt_registry_complete: bool
    retry_authorization_verified: bool
    rerun_policy_enforced: bool
    one_shot_policy_enforced: bool
    terminal_append_receipt_verified: bool
    campaign_finalization_verified: bool
    actual_evidence_inputs_accepted: bool
    admission_evidence_contract_complete: bool
    authoritative_evidence_verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    evidence_supplied: bool
    evidence_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    runtime_attestation_authority_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    wilson_bounds_evaluated: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    baseline_outputs_verified: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool


@dataclass(frozen=True, slots=True, init=False)
class ExternalCustodianCheckpointVerifierRejectionV2:
    def __new__(cls, token: object = None) -> "ExternalCustodianCheckpointVerifierRejectionV2":
        if token is not _ISSUANCE_TOKEN:
            raise TypeError("rejection results are privately issued")
        return object.__new__(cls)

    disposition: ExternalCustodianCheckpointDispositionV2
    reason: ExternalCustodianCheckpointReasonV2
    version: None
    schema_id: None
    policy_id: None
    claim_level: None
    scope: None
    result_id: None
    profile_id: None
    subject_id: None
    store_receipt_id: None
    time_token_id: None
    checkpoint_id: None
    required_signature_count: int
    verified_signature_count: int
    partial_output_published: bool
    candidate_profile_content_address_verified: bool
    four_role_distinct_key_registry_verified: bool
    role_epoch_purpose_domain_binding_verified: bool
    terminal_subject_commitment_verified: bool
    store_if_absent_generation_transition_verified: bool
    store_receipt_signature_verified: bool
    declared_time_monotonicity_verified: bool
    time_token_signature_verified: bool
    governance_checkpoint_signature_verified: bool
    custodian_checkpoint_signature_verified: bool
    checkpoint_cross_bindings_verified: bool
    checkpoint_content_address_verified: bool
    ordinary_exception_atomic_rejection_verified: bool
    external_trust_anchor_verified: bool
    registry_issuance_authority_verified: bool
    custodian_identity_verified: bool
    signer_independence_verified: bool
    qualified_cryptographic_backend_verified: bool
    external_store_identity_verified: bool
    external_store_authority_verified: bool
    external_store_receipt_authority_verified: bool
    external_store_rollback_resistance_verified: bool
    external_store_durability_verified: bool
    hostile_administrator_tamper_resistance_verified: bool
    cross_host_global_one_shot_verified: bool
    authoritative_time_source_verified: bool
    authoritative_time_verified: bool
    timeline_observed: bool
    authoritative_timeline_order_verified: bool
    signed_attempt_chain_authority_verified: bool
    local_ledger_externalized_verified: bool
    authoritative_attempt_denominator_verified: bool
    attempt_registry_complete: bool
    retry_authorization_verified: bool
    rerun_policy_enforced: bool
    one_shot_policy_enforced: bool
    terminal_append_receipt_verified: bool
    campaign_finalization_verified: bool
    actual_evidence_inputs_accepted: bool
    admission_evidence_contract_complete: bool
    authoritative_evidence_verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    evidence_supplied: bool
    evidence_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    runtime_attestation_authority_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    wilson_bounds_evaluated: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    baseline_outputs_verified: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool


def _text(value: object, name: str, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if not 1 <= len(encoded) <= maximum:
        raise ValueError(f"{name} UTF-8 size drift")
    return value


def _sha(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _identifier(value: object, prefix: str, name: str) -> str:
    text = _text(value, name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _sha(text[len(prefix):], f"{name} suffix")
    return text


def _integer(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_U63:
        raise TypeError(f"{name} exact nonnegative integer drift")
    return value


def _raw(value: object, length: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        raise TypeError(f"{name} must be exact {length}-byte bytes")
    return bytes(value)


def _named_bytes(pairs: tuple[tuple[str, object], ...]) -> bytes:
    return canonical_json(pairs).encode("utf-8")


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + _named_bytes(pairs)).hexdigest()


def _sequence_digest(domain: bytes, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(values).to_bytes(4, "big"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return digest.hexdigest()


_SIGNER_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianCandidateSignerV2))
_PROFILE_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianAuthorityCandidateProfileV2))
_SIGNATURE_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianDetachedSignatureV2))
_STORE_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianStoreReceiptCandidateV2))
_TIME_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianTimeTokenCandidateV2))
_CHECKPOINT_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianCheckpointCandidateV2))
_RESULT_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianCheckpointVerifierMechanicsV2))
_REJECTION_FIELDS: Final = tuple(field.name for field in fields(ExternalCustodianCheckpointVerifierRejectionV2))

_FIELD_TYPE_MANIFESTS: Final = (
    ("ExternalCustodianCandidateSignerV2", (
        ("role", "exact ExternalCustodianCandidateRoleV2"),
        ("key_epoch", "exact int not bool; exact value 1"),
        ("public_key", "exact bytes length 32"),
        ("key_id", f"exact UTF-8 text {_KEY_PREFIX}<lowerhex64>"),
    )),
    ("ExternalCustodianAuthorityCandidateProfileV2", (
        ("version", "exact frozen version text"), ("schema_id", "exact frozen schema ID text"),
        ("policy_id", "exact frozen policy ID text"), ("claim_level", "exact frozen claim-level text"),
        ("scope", "exact frozen scope text"), ("authority_requirement_id", "exact frozen requirement ID text"),
        ("campaign_id", f"exact UTF-8 text {_CAMPAIGN_PREFIX}<lowerhex64>"),
        ("attempt_id", f"exact UTF-8 text {_ATTEMPT_PREFIX}<lowerhex64>"),
        ("external_anchor_locator", "exact nonempty UTF-8 text max4096 bytes; opaque candidate locator"),
        ("external_anchor_evidence_sha256", "exact lowercase hex64"),
        ("signer_keys", "exact tuple length4 of exact signer DTO in frozen role order"),
        ("signer_key_ids_root_sha256", "exact lowercase hex64"),
        ("required_signature_count", "exact int not bool; exact value4"),
        ("profile_id", f"exact UTF-8 text {_PROFILE_PREFIX}<lowerhex64>"),
    )),
    ("ExternalCustodianDetachedSignatureV2", (
        ("role", "exact ExternalCustodianCandidateRoleV2"),
        ("purpose", "exact ExternalCustodianSignaturePurposeV2"),
        ("key_epoch", "exact int not bool; exact value1"),
        ("key_id", f"exact UTF-8 text {_KEY_PREFIX}<lowerhex64>"),
        ("signature", "exact bytes length64"),
    )),
    ("ExternalCustodianStoreReceiptCandidateV2", (
        ("version", "exact frozen version text"), ("profile_id", f"exact {_PROFILE_PREFIX}<lowerhex64>"),
        ("campaign_id", f"exact {_CAMPAIGN_PREFIX}<lowerhex64>"), ("attempt_id", f"exact {_ATTEMPT_PREFIX}<lowerhex64>"),
        ("subject_id", f"exact {_SUBJECT_PREFIX}<lowerhex64>"), ("store_service_id", f"exact {_STORE_SERVICE_PREFIX}<lowerhex64>"),
        ("object_key", "exact derived UTF-8 text max2048 bytes"), ("previous_generation", "exact int not bool; exact value0"),
        ("new_generation", "exact int not bool; exact value1"), ("cas_condition", "exact text IF_ABSENT"),
        ("stored_content_sha256", "exact lowercase hex64"), ("detached_signature", "exact signature DTO; STORE_RECEIPT purpose"),
        ("receipt_id", f"exact {_STORE_RECEIPT_PREFIX}<lowerhex64>"),
    )),
    ("ExternalCustodianTimeTokenCandidateV2", (
        ("version", "exact frozen version text"), ("profile_id", f"exact {_PROFILE_PREFIX}<lowerhex64>"),
        ("campaign_id", f"exact {_CAMPAIGN_PREFIX}<lowerhex64>"), ("attempt_id", f"exact {_ATTEMPT_PREFIX}<lowerhex64>"),
        ("subject_id", f"exact {_SUBJECT_PREFIX}<lowerhex64>"), ("time_source_id", f"exact {_TIME_SOURCE_PREFIX}<lowerhex64>"),
        ("previous_declared_time_ns", "exact int not bool in [0,2^63-1]"),
        ("declared_time_ns", "exact int not bool in [0,2^63-1] and greater than previous"),
        ("detached_signature", "exact signature DTO; TIME_ATTESTOR purpose"),
        ("time_token_id", f"exact {_TIME_TOKEN_PREFIX}<lowerhex64>"),
    )),
    ("ExternalCustodianCheckpointCandidateV2", (
        ("version", "exact frozen version text"), ("profile_id", f"exact {_PROFILE_PREFIX}<lowerhex64>"),
        ("campaign_id", f"exact {_CAMPAIGN_PREFIX}<lowerhex64>"), ("attempt_id", f"exact {_ATTEMPT_PREFIX}<lowerhex64>"),
        ("checkpoint_sequence", "exact int not bool; exact value0"), ("predecessor_checkpoint_id", "exact None"),
        ("signed_attempt_chain_id", f"exact {_SIGNED_CHAIN_PREFIX}<lowerhex64>"),
        ("signed_attempt_chain_root_sha256", "exact lowercase hex64"),
        ("terminal_event_id", f"exact {_TERMINAL_EVENT_PREFIX}<lowerhex64>"),
        ("local_ledger_id", f"exact {_LOCAL_LEDGER_PREFIX}<lowerhex64>"),
        ("local_terminal_record_id", f"exact {_LOCAL_TERMINAL_PREFIX}<lowerhex64>"),
        ("subject_id", f"exact {_SUBJECT_PREFIX}<lowerhex64>"),
        ("store_receipt_id", f"exact {_STORE_RECEIPT_PREFIX}<lowerhex64>"),
        ("time_token_id", f"exact {_TIME_TOKEN_PREFIX}<lowerhex64>"),
        ("checkpoint_signatures", "exact tuple length2 in GOVERNANCE_ANCHOR,CUSTODIAN order"),
        ("checkpoint_id", f"exact {_CHECKPOINT_PREFIX}<lowerhex64>"),
    )),
)

_NONE_RULES: Final = (
    ("candidate_inputs", "only ExternalCustodianCheckpointCandidateV2.predecessor_checkpoint_id is None"),
    ("success_output", "no success field is None"),
    ("rejection_output", "all identity/content-address fields are None; counts zero; all claims false"),
)

_PUBLIC_SURFACE: Final = (
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_POLICY_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_ED25519_BACKEND_AVAILABLE",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_STATEMENTS",
    "ExternalCustodianCandidateRoleV2",
    "ExternalCustodianSignaturePurposeV2",
    "ExternalCustodianCheckpointDispositionV2",
    "ExternalCustodianCheckpointReasonV2",
    "ExternalCustodianCandidateSignerV2",
    "ExternalCustodianAuthorityCandidateProfileV2",
    "ExternalCustodianDetachedSignatureV2",
    "ExternalCustodianStoreReceiptCandidateV2",
    "ExternalCustodianTimeTokenCandidateV2",
    "ExternalCustodianCheckpointCandidateV2",
    "ExternalCustodianCheckpointVerifierMechanicsV2",
    "ExternalCustodianCheckpointVerifierRejectionV2",
    "verify_phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2",
)


_SCHEMA_PREIMAGE: Final = (
    ("version", PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION),
    ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL),
    ("scope", PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE),
    ("signer_fields", _SIGNER_FIELDS),
    ("profile_fields", _PROFILE_FIELDS),
    ("signature_fields", _SIGNATURE_FIELDS),
    ("store_receipt_fields", _STORE_FIELDS),
    ("time_token_fields", _TIME_FIELDS),
    ("checkpoint_fields", _CHECKPOINT_FIELDS),
    ("result_fields", _RESULT_FIELDS),
    ("rejection_fields", _REJECTION_FIELDS),
    ("public_surface", _PUBLIC_SURFACE),
    ("field_type_manifests", _FIELD_TYPE_MANIFESTS),
    ("none_rules", _NONE_RULES),
    ("roles", tuple(role.value for role in _ROLE_ORDER)),
    ("purposes", tuple(purpose.value for purpose in ExternalCustodianSignaturePurposeV2)),
    ("dispositions", tuple(value.value for value in ExternalCustodianCheckpointDispositionV2)),
    ("reasons", tuple(value.value for value in ExternalCustodianCheckpointReasonV2)),
    ("required_signature_count", _REQUIRED_SIGNATURE_COUNT),
    ("key_epoch", _KEY_EPOCH),
    ("checkpoint_sequence", _CHECKPOINT_SEQUENCE),
    ("text_cap_bytes", _MAX_TEXT_BYTES),
    ("object_key_cap_bytes", _MAX_OBJECT_KEY_BYTES),
)

PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID: Final = (
    _SCHEMA_PREFIX + _digest(_SCHEMA_DOMAIN, _SCHEMA_PREIMAGE)
)

_POLICY_PREIMAGE: Final = (
    ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID),
    ("authority_requirement_id", PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_ID),
    ("authority_requirement_statements", PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_STATEMENTS),
    ("profile_rule", "caller_supplied_candidate_profile_never_establishes_external_trust"),
    ("key_rule", "exact_four_distinct_role_ordered_ed25519_raw32_epoch1_keys"),
    ("signature_rule", "role_specific_purpose_and_domain_over_declared_order_named_pair_preimage"),
    ("subject_rule", "single_supplied_terminal_chain_and_local_terminal_record_exact_content_binding; opaque upstream IDs are bound but not replayed here"),
    ("store_rule", "signed_if_absent_generation_zero_to_one_receipt_candidate_only; IF_ABSENT means no previous object and therefore no previous content digest exists"),
    ("time_rule", "signed_declared_monotonic_time_candidate_only_no_clock_or_tsa_authority"),
    ("checkpoint_rule", "sequence_zero_no_predecessor_governance_and_custodian_signatures"),
    ("prehash_rule", "all_four_input_graphs_exact_type_format_cap_count_duplicate_and_version_closed_before_hash_or_signature"),
    ("atomic_rule", "ordinary_exception_fail_closed_BaseException_and_process_termination_out_of_scope"),
    ("canonical_wire_rule", "UTF8(canonical_json(tuple((declared_field_name,exact_closed_primitive_value) in declared order))); nested DTOs expanded to named primitive tuples; no dataclass reaches canonical_json"),
    ("content_address_formulas", (
        "key_id=KEY_PREFIX+sha256(KEY_DOMAIN||canonical_named_key_pairs)",
        "signer_key_ids_root_sha256=sha256(KEY_ROOT_DOMAIN||u32_count||repeated_u16_length_framed_ascii_key_id)",
        "profile_id=PROFILE_PREFIX+sha256(PROFILE_DOMAIN||canonical_named_profile_pairs_excluding_profile_id)",
        "subject_id=SUBJECT_PREFIX+sha256(SUBJECT_DOMAIN||canonical_named_terminal_subject_pairs)",
        "role_signature_message=ROLE_SPECIFIC_DOMAIN||canonical_named(role,purpose,key_epoch,key_id,unsigned_record_named_pairs)",
        "receipt_id=STORE_RECEIPT_PREFIX+sha256(STORE_RECEIPT_DOMAIN||canonical_named_unsigned_store_pairs_plus_signature_named_pairs)",
        "time_token_id=TIME_TOKEN_PREFIX+sha256(TIME_TOKEN_DOMAIN||canonical_named_unsigned_time_pairs_plus_signature_named_pairs)",
        "checkpoint_id=CHECKPOINT_PREFIX+sha256(CHECKPOINT_DOMAIN||canonical_named_unsigned_checkpoint_pairs_plus_ordered_signature_named_pairs)",
        "result_id=RESULT_PREFIX+sha256(RESULT_DOMAIN||canonical_named_all_success_fields_except_result_id)",
        "schema_id=SCHEMA_PREFIX+sha256(SCHEMA_DOMAIN||canonical_named_schema_preimage)",
        "policy_id=POLICY_PREFIX+sha256(POLICY_DOMAIN||canonical_named_policy_preimage)",
        "authority_requirement_id=AUTHORITY_REQUIREMENT_PREFIX+sha256(AUTHORITY_REQUIREMENT_DOMAIN||canonical_named_authority_requirement_preimage)",
    )),
    ("api_rule", "single keyword-only verify(profile,store_receipt,time_token,checkpoint); exact public DTO types; no verifier,clock,path,authority-bool hooks"),
    ("mechanics_claims", _MECHANICS_CLAIMS),
    ("nonclaims", _NONCLAIMS),
    ("domains_hex", tuple(domain.hex() for domain in (_AUTHORITY_REQUIREMENT_DOMAIN, _KEY_DOMAIN, _KEY_ROOT_DOMAIN, _PROFILE_DOMAIN, _SUBJECT_DOMAIN, _STORE_SIGNATURE_DOMAIN, _STORE_RECEIPT_DOMAIN, _TIME_SIGNATURE_DOMAIN, _TIME_TOKEN_DOMAIN, _CHECKPOINT_SIGNATURE_DOMAIN, _CHECKPOINT_DOMAIN, _RESULT_DOMAIN, _SCHEMA_DOMAIN, _POLICY_DOMAIN))),
    ("prefixes", (_AUTHORITY_REQUIREMENT_PREFIX, _KEY_PREFIX, _PROFILE_PREFIX, _SUBJECT_PREFIX, _STORE_SERVICE_PREFIX, _STORE_RECEIPT_PREFIX, _TIME_SOURCE_PREFIX, _TIME_TOKEN_PREFIX, _CHECKPOINT_PREFIX, _RESULT_PREFIX, _SCHEMA_PREFIX, _POLICY_PREFIX)),
    ("forbidden_operations", ("filesystem", "clock", "network", "runner", "scorer", "ledger_write", "key_generation", "signature_generation", "caller_verifier")),
)

PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_POLICY_ID: Final = (
    _POLICY_PREFIX + _digest(_POLICY_DOMAIN, _POLICY_PREIMAGE)
)


def _key_pairs(role: ExternalCustodianCandidateRoleV2, key_epoch: int, public_key: bytes) -> tuple[tuple[str, object], ...]:
    return (("role", role.value), ("key_epoch", key_epoch), ("algorithm", "Ed25519"), ("public_key_hex", public_key.hex()))


def _signer_pairs(snapshot: tuple[ExternalCustodianCandidateRoleV2, int, bytes, str]) -> tuple[tuple[str, object], ...]:
    role, epoch, public_key, key_id = snapshot
    return _key_pairs(role, epoch, public_key) + (("key_id", key_id),)


def _profile_pairs(snapshot: tuple[object, ...], signer_snapshots: tuple[tuple[ExternalCustodianCandidateRoleV2, int, bytes, str], ...]) -> tuple[tuple[str, object], ...]:
    return (
        ("version", snapshot[0]), ("schema_id", snapshot[1]), ("policy_id", snapshot[2]),
        ("claim_level", snapshot[3]), ("scope", snapshot[4]), ("authority_requirement_id", snapshot[5]),
        ("campaign_id", snapshot[6]), ("attempt_id", snapshot[7]), ("external_anchor_locator", snapshot[8]),
        ("external_anchor_evidence_sha256", snapshot[9]),
        ("signer_keys", tuple(tuple(_signer_pairs(item)) for item in signer_snapshots)),
        ("signer_key_ids_root_sha256", snapshot[11]), ("required_signature_count", snapshot[12]),
    )


def _signature_snapshot(value: object, name: str) -> tuple[ExternalCustodianCandidateRoleV2, ExternalCustodianSignaturePurposeV2, int, str, bytes]:
    if type(value) is not ExternalCustodianDetachedSignatureV2:
        raise TypeError(f"{name} exact type drift")
    raw = tuple(getattr(value, field) for field in _SIGNATURE_FIELDS)
    if type(raw[0]) is not ExternalCustodianCandidateRoleV2 or type(raw[1]) is not ExternalCustodianSignaturePurposeV2:
        raise TypeError(f"{name} enum type drift")
    epoch = _integer(raw[2], f"{name}.key_epoch")
    key_id = _identifier(raw[3], _KEY_PREFIX, f"{name}.key_id")
    signature = _raw(raw[4], 64, f"{name}.signature")
    return raw[0], raw[1], epoch, key_id, signature


def _snapshot_inputs(*, profile: object, store_receipt: object, time_token: object, checkpoint: object) -> tuple[tuple[object, ...], tuple[tuple[ExternalCustodianCandidateRoleV2, int, bytes, str], ...], tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    if type(profile) is not ExternalCustodianAuthorityCandidateProfileV2:
        raise TypeError("profile exact type drift")
    if type(store_receipt) is not ExternalCustodianStoreReceiptCandidateV2:
        raise TypeError("store_receipt exact type drift")
    if type(time_token) is not ExternalCustodianTimeTokenCandidateV2:
        raise TypeError("time_token exact type drift")
    if type(checkpoint) is not ExternalCustodianCheckpointCandidateV2:
        raise TypeError("checkpoint exact type drift")
    profile_raw = tuple(getattr(profile, field) for field in _PROFILE_FIELDS)
    store_raw = tuple(getattr(store_receipt, field) for field in _STORE_FIELDS)
    time_raw = tuple(getattr(time_token, field) for field in _TIME_FIELDS)
    checkpoint_raw = tuple(getattr(checkpoint, field) for field in _CHECKPOINT_FIELDS)
    if type(profile_raw[10]) is not tuple or len(profile_raw[10]) != _REQUIRED_SIGNATURE_COUNT:
        raise TypeError("profile.signer_keys tuple/count drift")
    signer_raw = tuple(tuple(getattr(item, field) for field in _SIGNER_FIELDS) if type(item) is ExternalCustodianCandidateSignerV2 else () for item in profile_raw[10])
    if any(not item for item in signer_raw):
        raise TypeError("profile signer exact type drift")
    if type(checkpoint_raw[14]) is not tuple or len(checkpoint_raw[14]) != 2:
        raise TypeError("checkpoint signatures tuple/count drift")
    store_signature = _signature_snapshot(store_raw[11], "store_receipt.detached_signature")
    time_signature = _signature_snapshot(time_raw[8], "time_token.detached_signature")
    checkpoint_signatures = tuple(_signature_snapshot(item, f"checkpoint.signature[{index}]") for index, item in enumerate(checkpoint_raw[14]))
    store_raw = store_raw[:11] + (store_signature,) + store_raw[12:]
    time_raw = time_raw[:8] + (time_signature,) + time_raw[9:]
    checkpoint_raw = checkpoint_raw[:14] + (checkpoint_signatures,) + checkpoint_raw[15:]
    return profile_raw, signer_raw, store_raw, time_raw, checkpoint_raw


def _close_snapshots(profile: tuple[object, ...], signers: tuple[tuple[object, ...], ...], store: tuple[object, ...], time: tuple[object, ...], checkpoint: tuple[object, ...]) -> None:
    for index, value in enumerate(profile[:10]):
        _text(value, f"profile[{index}]")
    _sha(profile[9], "profile.external_anchor_evidence_sha256")
    _sha(profile[11], "profile.signer_key_ids_root_sha256")
    _integer(profile[12], "profile.required_signature_count")
    _identifier(profile[13], _PROFILE_PREFIX, "profile.profile_id")
    for index, signer in enumerate(signers):
        if len(signer) != 4 or type(signer[0]) is not ExternalCustodianCandidateRoleV2:
            raise TypeError(f"signer[{index}] type drift")
        _integer(signer[1], f"signer[{index}].key_epoch")
        _raw(signer[2], 32, f"signer[{index}].public_key")
        _identifier(signer[3], _KEY_PREFIX, f"signer[{index}].key_id")
    for index, value in enumerate(store[:7]):
        _text(value, f"store[{index}]", maximum=_MAX_OBJECT_KEY_BYTES if index == 6 else _MAX_TEXT_BYTES)
    _integer(store[7], "store.previous_generation")
    _integer(store[8], "store.new_generation")
    _text(store[9], "store.cas_condition")
    _sha(store[10], "store.stored_content_sha256")
    _identifier(store[12], _STORE_RECEIPT_PREFIX, "store.receipt_id")
    for index, value in enumerate(time[:6]):
        _text(value, f"time[{index}]")
    _integer(time[6], "time.previous_declared_time_ns")
    _integer(time[7], "time.declared_time_ns")
    _identifier(time[9], _TIME_TOKEN_PREFIX, "time.time_token_id")
    for index, value in enumerate(checkpoint[:4]):
        _text(value, f"checkpoint[{index}]")
    _integer(checkpoint[4], "checkpoint.checkpoint_sequence")
    if checkpoint[5] is not None:
        raise TypeError("checkpoint.predecessor_checkpoint_id must be exact None")
    for index, value in enumerate(checkpoint[6:14], start=6):
        _text(value, f"checkpoint[{index}]")
    _identifier(checkpoint[15], _CHECKPOINT_PREFIX, "checkpoint.checkpoint_id")


def _validate_fixed(profile: tuple[object, ...], signers: tuple[tuple[object, ...], ...], store: tuple[object, ...], time: tuple[object, ...], checkpoint: tuple[object, ...]) -> None:
    expected_identity = (
        PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION,
        PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID,
        PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_POLICY_ID,
        PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL,
        PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE,
        PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_ID,
    )
    if profile[:6] != expected_identity:
        raise ValueError("profile frozen identity drift")
    _identifier(profile[6], _CAMPAIGN_PREFIX, "profile.campaign_id")
    _identifier(profile[7], _ATTEMPT_PREFIX, "profile.attempt_id")
    if profile[12] != _REQUIRED_SIGNATURE_COUNT:
        raise ValueError("required signature count drift")
    if tuple(item[0] for item in signers) != _ROLE_ORDER:
        raise ValueError("signer exact role/order drift")
    if any(item[1] != _KEY_EPOCH for item in signers):
        raise ValueError("signer epoch drift")
    if len({item[3] for item in signers}) != 4 or len({item[2] for item in signers}) != 4:
        raise ValueError("signer keys must be distinct")
    if store[0] != expected_identity[0] or time[0] != expected_identity[0] or checkpoint[0] != expected_identity[0]:
        raise ValueError("nested version drift")
    for nested, label in ((store, "store"), (time, "time"), (checkpoint, "checkpoint")):
        if nested[1] != profile[13] or nested[2] != profile[6] or nested[3] != profile[7]:
            raise ValueError(f"{label} profile/campaign/attempt binding drift")
    _identifier(store[5], _STORE_SERVICE_PREFIX, "store.service_id")
    _identifier(time[5], _TIME_SOURCE_PREFIX, "time.source_id")
    if store[7:10] != (_PREVIOUS_GENERATION, _NEW_GENERATION, _CAS_CONDITION):
        raise ValueError("IF_ABSENT generation transition drift")
    if not time[6] < time[7]:
        raise ValueError("declared time monotonicity drift")
    if checkpoint[4] != _CHECKPOINT_SEQUENCE or checkpoint[5] is not _NO_PREDECESSOR:
        raise ValueError("initial checkpoint sequence/predecessor drift")
    _identifier(checkpoint[6], _SIGNED_CHAIN_PREFIX, "checkpoint.signed_chain_id")
    _sha(checkpoint[7], "checkpoint.signed_chain_root_sha256")
    _identifier(checkpoint[8], _TERMINAL_EVENT_PREFIX, "checkpoint.terminal_event_id")
    _identifier(checkpoint[9], _LOCAL_LEDGER_PREFIX, "checkpoint.local_ledger_id")
    _identifier(checkpoint[10], _LOCAL_TERMINAL_PREFIX, "checkpoint.local_terminal_record_id")
    _identifier(checkpoint[11], _SUBJECT_PREFIX, "checkpoint.subject_id")
    _identifier(checkpoint[12], _STORE_RECEIPT_PREFIX, "checkpoint.store_receipt_id")
    _identifier(checkpoint[13], _TIME_TOKEN_PREFIX, "checkpoint.time_token_id")
    if store[4] != checkpoint[11] or time[4] != checkpoint[11]:
        raise ValueError("subject cross-binding drift")
    if checkpoint[12] != store[12] or checkpoint[13] != time[9]:
        raise ValueError("receipt/token cross-binding drift")
    if store[10] != checkpoint[10][len(_LOCAL_TERMINAL_PREFIX):]:
        raise ValueError("stored content digest must equal terminal record digest")
    expected_object_key = f"phase2b/actual-unsealed-960/{profile[6]}/{profile[7]}/terminal.json"
    if store[6] != expected_object_key:
        raise ValueError("store object key drift")
    store_signature = store[11]
    time_signature = time[8]
    checkpoint_signatures = checkpoint[14]
    expected_signatures = (
        (ExternalCustodianCandidateRoleV2.STORE_RECEIPT, ExternalCustodianSignaturePurposeV2.STORE_IF_ABSENT_RECEIPT),
        (ExternalCustodianCandidateRoleV2.TIME_ATTESTOR, ExternalCustodianSignaturePurposeV2.DECLARED_TIME_TOKEN),
    )
    if store_signature[:2] != expected_signatures[0] or time_signature[:2] != expected_signatures[1]:
        raise ValueError("receipt/time signature role-purpose drift")
    if tuple(item[:2] for item in checkpoint_signatures) != (
        (ExternalCustodianCandidateRoleV2.GOVERNANCE_ANCHOR, ExternalCustodianSignaturePurposeV2.GOVERNANCE_CHECKPOINT),
        (ExternalCustodianCandidateRoleV2.CUSTODIAN, ExternalCustodianSignaturePurposeV2.CUSTODIAN_CHECKPOINT),
    ):
        raise ValueError("checkpoint signature role-purpose/order drift")
    signer_by_role = {item[0]: item for item in signers}
    for signature in (store_signature, time_signature) + checkpoint_signatures:
        key = signer_by_role[signature[0]]
        if signature[2] != key[1] or signature[3] != key[3] or signature[1] is not _PURPOSE_BY_ROLE[signature[0]]:
            raise ValueError("signature key/epoch/purpose registry binding drift")


def _subject_pairs(checkpoint: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return (("campaign_id", checkpoint[2]), ("attempt_id", checkpoint[3]), ("signed_attempt_chain_id", checkpoint[6]), ("signed_attempt_chain_root_sha256", checkpoint[7]), ("terminal_event_id", checkpoint[8]), ("local_ledger_id", checkpoint[9]), ("local_terminal_record_id", checkpoint[10]))


def _store_unsigned_pairs(store: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return tuple((name, store[index]) for index, name in enumerate(_STORE_FIELDS[:11]))


def _time_unsigned_pairs(time: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return tuple((name, time[index]) for index, name in enumerate(_TIME_FIELDS[:8]))


def _checkpoint_unsigned_pairs(checkpoint: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return tuple((name, checkpoint[index]) for index, name in enumerate(_CHECKPOINT_FIELDS[:14]))


def _signature_pairs(signature: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return (("role", signature[0].value), ("purpose", signature[1].value), ("key_epoch", signature[2]), ("key_id", signature[3]), ("signature_hex", signature[4].hex()))


def _signature_message(domain: bytes, unsigned_pairs: tuple[tuple[str, object], ...], signature: tuple[object, ...]) -> bytes:
    framing = (("role", signature[0].value), ("purpose", signature[1].value), ("key_epoch", signature[2]), ("key_id", signature[3]), ("unsigned_record", unsigned_pairs))
    return domain + _named_bytes(framing)


def _verify_signature(public_key: bytes, signature: bytes, message: bytes) -> None:
    if _Ed25519PublicKey is None:
        raise _SignatureBackendUnavailable("Ed25519 backend unavailable")
    try:
        _Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except Exception as exc:
        raise _SignatureInvalid("Ed25519 signature verification failed") from exc


def _verify_content_and_signatures(profile: tuple[object, ...], signers: tuple[tuple[object, ...], ...], store: tuple[object, ...], time: tuple[object, ...], checkpoint: tuple[object, ...]) -> None:
    expected_key_ids = tuple(_KEY_PREFIX + _digest(_KEY_DOMAIN, _key_pairs(item[0], item[1], item[2])) for item in signers)
    if expected_key_ids != tuple(item[3] for item in signers):
        raise ValueError("signer key content address drift")
    if profile[11] != _sequence_digest(_KEY_ROOT_DOMAIN, expected_key_ids):
        raise ValueError("signer key root drift")
    if profile[13] != _PROFILE_PREFIX + _digest(_PROFILE_DOMAIN, _profile_pairs(profile, signers)):
        raise ValueError("candidate profile content address drift")
    subject_id = _SUBJECT_PREFIX + _digest(_SUBJECT_DOMAIN, _subject_pairs(checkpoint))
    if checkpoint[11] != subject_id:
        raise ValueError("subject content address drift")
    signer_by_role = {item[0]: item for item in signers}
    store_unsigned = _store_unsigned_pairs(store)
    store_signature = store[11]
    _verify_signature(signer_by_role[store_signature[0]][2], store_signature[4], _signature_message(_STORE_SIGNATURE_DOMAIN, store_unsigned, store_signature))
    store_pairs = store_unsigned + (("detached_signature", tuple(_signature_pairs(store_signature))),)
    if store[12] != _STORE_RECEIPT_PREFIX + _digest(_STORE_RECEIPT_DOMAIN, store_pairs):
        raise ValueError("store receipt content address drift")
    time_unsigned = _time_unsigned_pairs(time)
    time_signature = time[8]
    _verify_signature(signer_by_role[time_signature[0]][2], time_signature[4], _signature_message(_TIME_SIGNATURE_DOMAIN, time_unsigned, time_signature))
    time_pairs = time_unsigned + (("detached_signature", tuple(_signature_pairs(time_signature))),)
    if time[9] != _TIME_TOKEN_PREFIX + _digest(_TIME_TOKEN_DOMAIN, time_pairs):
        raise ValueError("time token content address drift")
    checkpoint_unsigned = _checkpoint_unsigned_pairs(checkpoint)
    for signature in checkpoint[14]:
        _verify_signature(signer_by_role[signature[0]][2], signature[4], _signature_message(_CHECKPOINT_SIGNATURE_DOMAIN, checkpoint_unsigned, signature))
    checkpoint_pairs = checkpoint_unsigned + (("checkpoint_signatures", tuple(tuple(_signature_pairs(item)) for item in checkpoint[14])),)
    if checkpoint[15] != _CHECKPOINT_PREFIX + _digest(_CHECKPOINT_DOMAIN, checkpoint_pairs):
        raise ValueError("checkpoint content address drift")


def _issue_result(profile: tuple[object, ...], store: tuple[object, ...], time: tuple[object, ...], checkpoint: tuple[object, ...]) -> ExternalCustodianCheckpointVerifierMechanicsV2:
    values: dict[str, object] = {
        "disposition": ExternalCustodianCheckpointDispositionV2.CANDIDATE_EXTERNAL_CHECKPOINT_MECHANICS_VERIFIED_NOT_EXTERNAL_AUTHORITY,
        "reason": ExternalCustodianCheckpointReasonV2.FOUR_ROLE_CANDIDATE_CHECKPOINT_VERIFIED_NOT_PRETRUSTED,
        "version": PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION,
        "schema_id": PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID,
        "policy_id": PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_POLICY_ID,
        "claim_level": PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL,
        "scope": PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE,
        "result_id": "",
        "profile_id": profile[13], "subject_id": checkpoint[11], "store_receipt_id": store[12],
        "time_token_id": time[9], "checkpoint_id": checkpoint[15],
        "required_signature_count": _REQUIRED_SIGNATURE_COUNT, "verified_signature_count": _REQUIRED_SIGNATURE_COUNT,
    }
    values.update({name: True for name in _MECHANICS_CLAIMS})
    values.update({name: False for name in _NONCLAIMS})
    result_pairs = tuple((field, values[field].value if isinstance(values[field], Enum) else values[field]) for field in _RESULT_FIELDS if field != "result_id")
    values["result_id"] = _RESULT_PREFIX + _digest(_RESULT_DOMAIN, result_pairs)
    result = object.__new__(ExternalCustodianCheckpointVerifierMechanicsV2)
    for field in _RESULT_FIELDS:
        object.__setattr__(result, field, values[field])
    return result


def _reject(reason: ExternalCustodianCheckpointReasonV2) -> ExternalCustodianCheckpointVerifierRejectionV2:
    values: dict[str, object] = {
        "disposition": ExternalCustodianCheckpointDispositionV2.REJECTED, "reason": reason,
        "version": None, "schema_id": None, "policy_id": None, "claim_level": None, "scope": None,
        "result_id": None, "profile_id": None, "subject_id": None, "store_receipt_id": None,
        "time_token_id": None, "checkpoint_id": None, "required_signature_count": 0,
        "verified_signature_count": 0, "partial_output_published": False,
    }
    values.update({name: False for name in _MECHANICS_CLAIMS})
    values.update({name: False for name in _NONCLAIMS})
    result = object.__new__(ExternalCustodianCheckpointVerifierRejectionV2)
    for field in _REJECTION_FIELDS:
        object.__setattr__(result, field, values[field])
    return result


def verify_phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2(
    *,
    profile: ExternalCustodianAuthorityCandidateProfileV2,
    store_receipt: ExternalCustodianStoreReceiptCandidateV2,
    time_token: ExternalCustodianTimeTokenCandidateV2,
    checkpoint: ExternalCustodianCheckpointCandidateV2,
) -> ExternalCustodianCheckpointVerifierMechanicsV2 | ExternalCustodianCheckpointVerifierRejectionV2:
    """Verify one supplied candidate checkpoint without granting authority."""

    try:
        snapshots = _snapshot_inputs(profile=profile, store_receipt=store_receipt, time_token=time_token, checkpoint=checkpoint)
    except Exception:
        return _reject(ExternalCustodianCheckpointReasonV2.WRONG_INPUT_TYPE)
    try:
        _close_snapshots(*snapshots)
        profile_snapshot, signer_snapshots, store_snapshot, time_snapshot, checkpoint_snapshot = snapshots
        versions = (profile_snapshot[0], store_snapshot[0], time_snapshot[0], checkpoint_snapshot[0])
        if any(value != PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION for value in versions):
            return _reject(ExternalCustodianCheckpointReasonV2.CROSS_VERSION_INPUT)
        _validate_fixed(*snapshots)
        if _Ed25519PublicKey is None:
            return _reject(ExternalCustodianCheckpointReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
        _verify_content_and_signatures(*snapshots)
        return _issue_result(profile_snapshot, store_snapshot, time_snapshot, checkpoint_snapshot)
    except TypeError:
        return _reject(ExternalCustodianCheckpointReasonV2.WRONG_INPUT_TYPE)
    except ValueError:
        return _reject(ExternalCustodianCheckpointReasonV2.CROSS_BINDING_INVALID)
    except _SignatureBackendUnavailable:
        return _reject(ExternalCustodianCheckpointReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
    except _SignatureInvalid:
        return _reject(ExternalCustodianCheckpointReasonV2.SIGNATURE_INVALID)
    except Exception:
        return _reject(ExternalCustodianCheckpointReasonV2.INTERNAL_ERROR)


if set(_MECHANICS_CLAIMS) & set(_NONCLAIMS):
    raise RuntimeError("external checkpoint claim polarity overlap")
if len(_MECHANICS_CLAIMS) != len(set(_MECHANICS_CLAIMS)) or len(_NONCLAIMS) != len(set(_NONCLAIMS)):
    raise RuntimeError("external checkpoint claim manifest duplicate")
if _REJECTION_FIELDS != _RESULT_FIELDS[:2] + ("version", "schema_id", "policy_id", "claim_level", "scope", "result_id", "profile_id", "subject_id", "store_receipt_id", "time_token_id", "checkpoint_id", "required_signature_count", "verified_signature_count", "partial_output_published") + _MECHANICS_CLAIMS + _NONCLAIMS:
    raise RuntimeError("external checkpoint rejection field manifest drift")


__all__ = list(_PUBLIC_SURFACE)
