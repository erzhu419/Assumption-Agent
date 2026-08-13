"""Test-anchor-only signed attempt-chain mechanics for Phase-2B.

This module verifies a caller-supplied, content-addressed ten-event attempt
chain and its Ed25519 signatures.  The only accepted trust-anchor profile is
explicitly TEST_ONLY_SUPPLIED_ANCHOR.  Cryptographic validity under that
caller-supplied test key does not establish custodian identity, external trust,
durable append-only storage, observed time, recognizer execution, scoring,
formal gates, effect, or C1 evidence.

The verifier performs no filesystem, clock, network, runner, scorer, ledger,
key-generation, or signature-generation operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
from typing import Final

from .hashing import canonical_json

try:  # Optional backend: absence is a fail-closed public result.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by backend monkeypatch tests.
    _Ed25519PublicKey = None


PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-signed-attempt-chain-mechanics/2"
)
PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_TEST_ONLY_SIGNED_ATTEMPT_CHAIN_MECHANICS"
)
PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2: Final = (
    "TEST_ONLY_SUPPLIED_ANCHOR"
)
PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_ED25519_BACKEND_AVAILABLE: Final = (
    _Ed25519PublicKey is not None
)

_FROZEN_PROTOCOL_ID: Final = (
    "phase2b_protocol_62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
_FROZEN_EXACT_FREEZE_ID: Final = (
    "phase2b_exact_freeze_ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
_MAX_TEXT_BYTES: Final = 4_096
_EVENT_COUNT: Final = 10

_KEY_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/KEY/V2\x00"
_KEY_ROOT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/KEY_ROOT/V2\x00"
_ATTEMPT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ATTEMPT/V2\x00"
_ANCHOR_ROOT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ANCHOR_ROOT/V2\x00"
_ANCHOR_ID_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ANCHOR_ID/V2\x00"
_EVENT_SIGNATURE_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/EVENT_SIGNATURE/V2\x00"
_EVENT_ID_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/EVENT_ID/V2\x00"
_CHAIN_ROOT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/CHAIN_ROOT/V2\x00"
_CHAIN_ID_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/CHAIN_ID/V2\x00"
_RESULT_ID_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/RESULT/V2\x00"
_SCHEMA_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/SCHEMA/V2\x00"
_POLICY_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/POLICY/V2\x00"

_KEY_PREFIX: Final = "phase2b_actual_unsealed_960_signer_key_v2_"
_ATTEMPT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_v2_"
_ANCHOR_PREFIX: Final = "phase2b_actual_unsealed_960_test_anchor_v2_"
_EVENT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_event_v2_"
_CHAIN_PREFIX: Final = "phase2b_actual_unsealed_960_signed_attempt_chain_v2_"
_RESULT_PREFIX: Final = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2_"
_SCHEMA_PREFIX: Final = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_schema_v2_"
_POLICY_PREFIX: Final = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_policy_v2_"


class SignedAttemptSignerRoleV2(str, Enum):
    CUSTODIAN = "CUSTODIAN"


class SignedAttemptStageV2(str, Enum):
    PREREGISTERED = "PREREGISTERED"
    RAW_INPUT_COMMITTED = "RAW_INPUT_COMMITTED"
    PACKAGE_COMMITMENTS_FROZEN = "PACKAGE_COMMITMENTS_FROZEN"
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    PREDICTIONS_COMMITTED = "PREDICTIONS_COMMITTED"
    REVEALED = "REVEALED"
    SCORING_STARTED = "SCORING_STARTED"
    REPORTS_COMMITTED = "REPORTS_COMMITTED"
    TERMINAL_CONSUMED = "TERMINAL_CONSUMED"


_STAGE_ORDER: Final = tuple(SignedAttemptStageV2)


class SignedAttemptChainDispositionV2(str, Enum):
    SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY = (
        "SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY"
    )
    REJECTED = "REJECTED"


class SignedAttemptChainReasonV2(str, Enum):
    TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED = (
        "TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    TEST_ANCHOR_INVALID = "TEST_ANCHOR_INVALID"
    CHAIN_INVALID = "CHAIN_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if not 1 <= len(encoded) <= _MAX_TEXT_BYTES:
        raise ValueError(f"{name} UTF-8 size drift")
    return value


def _sha(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _id(value: object, prefix: str, name: str) -> str:
    text = _text(value, name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _sha(text[len(prefix):], f"{name} suffix")
    return text


def _raw(value: object, length: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        raise TypeError(f"{name} must be exact {length}-byte bytes")
    return bytes(value)


def _integer(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} integer drift")
    return value


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


def _key_pairs(role: SignedAttemptSignerRoleV2, public_key: bytes) -> tuple[tuple[str, object], ...]:
    return (
        ("role", role.value),
        ("algorithm", "Ed25519"),
        ("public_key_hex", public_key.hex()),
    )


def _expected_key_id(role: SignedAttemptSignerRoleV2, public_key: bytes) -> str:
    return _KEY_PREFIX + _digest(_KEY_DOMAIN, _key_pairs(role, public_key))


@dataclass(frozen=True, slots=True)
class ActualUnsealed960SignerKeyV2:
    role: SignedAttemptSignerRoleV2
    public_key: bytes
    key_id: str

    def __post_init__(self) -> None:
        if type(self.role) is not SignedAttemptSignerRoleV2:
            raise TypeError("signer role exact type drift")
        public_key = _raw(self.public_key, 32, "Ed25519 public key")
        key_id = _id(self.key_id, _KEY_PREFIX, "signer key ID")
        if key_id != _expected_key_id(self.role, public_key):
            raise ValueError("signer key content address drift")


def _attempt_pairs(
    *,
    schema_id: str,
    policy_id: str,
    protocol_id: str,
    exact_freeze_id: str,
    execution_freeze_manifest_id: str,
    attempt_nonce_sha256: str,
) -> tuple[tuple[str, object], ...]:
    return (
        ("schema_id", schema_id),
        ("policy_id", policy_id),
        ("protocol_id", protocol_id),
        ("exact_freeze_id", exact_freeze_id),
        ("execution_freeze_manifest_id", execution_freeze_manifest_id),
        ("attempt_nonce_sha256", attempt_nonce_sha256),
    )


def _expected_attempt_id(**values: str) -> str:
    return _ATTEMPT_PREFIX + _digest(_ATTEMPT_DOMAIN, _attempt_pairs(**values))


def _anchor_pairs(
    *,
    version: str,
    claim_level: str,
    anchor_scope: str,
    schema_id: str,
    policy_id: str,
    protocol_id: str,
    exact_freeze_id: str,
    execution_freeze_manifest_id: str,
    attempt_nonce_sha256: str,
    attempt_id: str,
    signer_key_ids: tuple[str, ...],
    signer_key_ids_root: str,
) -> tuple[tuple[str, object], ...]:
    return (
        ("version", version),
        ("claim_level", claim_level),
        ("anchor_scope", anchor_scope),
        ("schema_id", schema_id),
        ("policy_id", policy_id),
        ("protocol_id", protocol_id),
        ("exact_freeze_id", exact_freeze_id),
        ("execution_freeze_manifest_id", execution_freeze_manifest_id),
        ("attempt_nonce_sha256", attempt_nonce_sha256),
        ("attempt_id", attempt_id),
        ("signer_key_ids", signer_key_ids),
        ("signer_key_ids_root", signer_key_ids_root),
    )


@dataclass(frozen=True, slots=True)
class ActualUnsealed960TestOnlyAnchorV2:
    version: str
    claim_level: str
    anchor_scope: str
    schema_id: str
    policy_id: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    attempt_nonce_sha256: str
    attempt_id: str
    signer_keys: tuple[ActualUnsealed960SignerKeyV2, ...]
    signer_key_ids_root: str
    anchor_root_sha256: str
    anchor_id: str

    def __post_init__(self) -> None:
        _validate_anchor(self)


def _event_unsigned_pairs(
    *,
    version: str,
    anchor_id: str,
    attempt_id: str,
    event_index: int,
    stage: SignedAttemptStageV2,
    predecessor_event_id: str | None,
    payload_root_sha256: str,
    signer_key_id: str,
) -> tuple[tuple[str, object], ...]:
    return (
        ("version", version),
        ("anchor_id", anchor_id),
        ("attempt_id", attempt_id),
        ("event_index", event_index),
        ("stage", stage.value),
        ("predecessor_event_id", predecessor_event_id),
        ("payload_root_sha256", payload_root_sha256),
        ("signer_key_id", signer_key_id),
    )


def _event_signature_preimage(pairs: tuple[tuple[str, object], ...]) -> bytes:
    return _EVENT_SIGNATURE_DOMAIN + _named_bytes(pairs)


def _expected_event_id(
    pairs: tuple[tuple[str, object], ...], signature: bytes
) -> str:
    id_pairs = (*pairs, ("signature_hex", signature.hex()))
    return _EVENT_PREFIX + _digest(_EVENT_ID_DOMAIN, id_pairs)


@dataclass(frozen=True, slots=True)
class ActualUnsealed960AttemptEventV2:
    version: str
    anchor_id: str
    attempt_id: str
    event_index: int
    stage: SignedAttemptStageV2
    predecessor_event_id: str | None
    payload_root_sha256: str
    signer_key_id: str
    signature: bytes
    event_id: str

    def __post_init__(self) -> None:
        _validate_event_shape(self)


def _chain_pairs(
    *,
    version: str,
    anchor_id: str,
    attempt_id: str,
    event_ids: tuple[str, ...],
    chain_root_sha256: str,
) -> tuple[tuple[str, object], ...]:
    return (
        ("version", version),
        ("anchor_id", anchor_id),
        ("attempt_id", attempt_id),
        ("event_ids", event_ids),
        ("chain_root_sha256", chain_root_sha256),
    )


@dataclass(frozen=True, slots=True)
class ActualUnsealed960SignedAttemptChainV2:
    version: str
    anchor_id: str
    attempt_id: str
    events: tuple[ActualUnsealed960AttemptEventV2, ...]
    chain_root_sha256: str
    chain_id: str

    def __post_init__(self) -> None:
        _validate_chain_shape(self)


_TRUE_CLAIMS: Final = (
    "test_only_anchor_scope_enforced",
    "test_only_anchor_content_addresses_verified",
    "test_only_signer_key_content_address_verified",
    "test_only_anchor_ed25519_event_signatures_verified",
    "signed_event_content_addresses_verified",
    "signed_test_chain_contiguous_event_indexes_verified",
    "signed_test_chain_predecessor_links_verified",
    "signed_test_chain_exact_ten_stage_order_verified",
    "signed_test_chain_terminal_consumed_label_verified",
    "signed_test_chain_root_and_id_verified",
    "atomic_fail_closed_rejection_verified",
)

_FALSE_CLAIMS: Final = (
    "external_trust_anchor_verified",
    "external_signature_authority_verified",
    "qualified_cryptographic_backend_verified",
    "signature_coverage_verified",
    "authoritative_evidence_chain_verified",
    "custodian_identity_verified",
    "pinned_signer_key_registry_verified",
    "signer_independence_verified",
    "durable_external_one_shot_ledger_verified",
    "append_only_storage_verified",
    "cross_process_fork_prevention_verified",
    "external_timestamp_authority_verified",
    "timeline_observed",
    "authoritative_timeline_order_verified",
    "authoritative_attempt_terminal_state_verified",
    "full_c1_timeline_schema_complete",
    "attempt_registry_complete",
    "terminal_append_receipt_verified",
    "campaign_finalization_verified",
    "retry_authorization_verified",
    "rerun_policy_enforced",
    "one_shot_policy_enforced",
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
    "answer_commitment_opening_verified",
    "gate_input_commitment_opening_verified",
    "pre_reveal_commitment_timing_verified",
    "runtime_attestation_authority_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "origin_authenticated",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "preservation_evaluated",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)


@dataclass(frozen=True, slots=True, init=False)
class ActualUnsealed960SignedAttemptChainMechanicsV2:
    disposition: SignedAttemptChainDispositionV2
    reason: SignedAttemptChainReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    result_id: str
    anchor_id: str
    attempt_id: str
    signer_key_id: str
    raw_input_committed_payload_root_sha256: str
    package_commitments_frozen_payload_root_sha256: str
    chain_root_sha256: str
    chain_id: str
    event_count: int
    terminal_stage: SignedAttemptStageV2
    test_only_anchor_scope_enforced: bool
    test_only_anchor_content_addresses_verified: bool
    test_only_signer_key_content_address_verified: bool
    test_only_anchor_ed25519_event_signatures_verified: bool
    signed_event_content_addresses_verified: bool
    signed_test_chain_contiguous_event_indexes_verified: bool
    signed_test_chain_predecessor_links_verified: bool
    signed_test_chain_exact_ten_stage_order_verified: bool
    signed_test_chain_terminal_consumed_label_verified: bool
    signed_test_chain_root_and_id_verified: bool
    atomic_fail_closed_rejection_verified: bool
    external_trust_anchor_verified: bool
    external_signature_authority_verified: bool
    qualified_cryptographic_backend_verified: bool
    signature_coverage_verified: bool
    authoritative_evidence_chain_verified: bool
    custodian_identity_verified: bool
    pinned_signer_key_registry_verified: bool
    signer_independence_verified: bool
    durable_external_one_shot_ledger_verified: bool
    append_only_storage_verified: bool
    cross_process_fork_prevention_verified: bool
    external_timestamp_authority_verified: bool
    timeline_observed: bool
    authoritative_timeline_order_verified: bool
    authoritative_attempt_terminal_state_verified: bool
    full_c1_timeline_schema_complete: bool
    attempt_registry_complete: bool
    terminal_append_receipt_verified: bool
    campaign_finalization_verified: bool
    retry_authorization_verified: bool
    rerun_policy_enforced: bool
    one_shot_policy_enforced: bool
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
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    pre_reveal_commitment_timing_verified: bool
    runtime_attestation_authority_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("signed-attempt-chain mechanics results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class ActualUnsealed960SignedAttemptChainRejectionV2:
    disposition: SignedAttemptChainDispositionV2
    reason: SignedAttemptChainReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    validation: None
    result_id: None
    anchor_id: None
    attempt_id: None
    signer_key_id: None
    raw_input_committed_payload_root_sha256: None
    package_commitments_frozen_payload_root_sha256: None
    chain_root_sha256: None
    chain_id: None
    event_count: int
    terminal_stage: None
    partial_output_published: bool
    test_only_anchor_scope_enforced: bool
    test_only_anchor_content_addresses_verified: bool
    test_only_signer_key_content_address_verified: bool
    test_only_anchor_ed25519_event_signatures_verified: bool
    signed_event_content_addresses_verified: bool
    signed_test_chain_contiguous_event_indexes_verified: bool
    signed_test_chain_predecessor_links_verified: bool
    signed_test_chain_exact_ten_stage_order_verified: bool
    signed_test_chain_terminal_consumed_label_verified: bool
    signed_test_chain_root_and_id_verified: bool
    atomic_fail_closed_rejection_verified: bool
    external_trust_anchor_verified: bool
    external_signature_authority_verified: bool
    qualified_cryptographic_backend_verified: bool
    signature_coverage_verified: bool
    authoritative_evidence_chain_verified: bool
    custodian_identity_verified: bool
    pinned_signer_key_registry_verified: bool
    signer_independence_verified: bool
    durable_external_one_shot_ledger_verified: bool
    append_only_storage_verified: bool
    cross_process_fork_prevention_verified: bool
    external_timestamp_authority_verified: bool
    timeline_observed: bool
    authoritative_timeline_order_verified: bool
    authoritative_attempt_terminal_state_verified: bool
    full_c1_timeline_schema_complete: bool
    attempt_registry_complete: bool
    terminal_append_receipt_verified: bool
    campaign_finalization_verified: bool
    retry_authorization_verified: bool
    rerun_policy_enforced: bool
    one_shot_policy_enforced: bool
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
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    pre_reveal_commitment_timing_verified: bool
    runtime_attestation_authority_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("signed-attempt-chain rejections are privately issued")


_KEY_FIELDS: Final = ("role", "public_key", "key_id")
_ANCHOR_FIELDS: Final = (
    "version", "claim_level", "anchor_scope", "schema_id", "policy_id",
    "protocol_id", "exact_freeze_id",
    "execution_freeze_manifest_id", "attempt_nonce_sha256", "attempt_id", "signer_keys",
    "signer_key_ids_root", "anchor_root_sha256", "anchor_id",
)
_EVENT_FIELDS: Final = (
    "version", "anchor_id", "attempt_id", "event_index", "stage",
    "predecessor_event_id", "payload_root_sha256", "signer_key_id",
    "signature", "event_id",
)
_CHAIN_FIELDS: Final = (
    "version", "anchor_id", "attempt_id", "events", "chain_root_sha256", "chain_id",
)
_FIELD_TYPE_MANIFESTS: Final = (
    (
        "ActualUnsealed960SignerKeyV2",
        (("role", "exact_SignedAttemptSignerRoleV2"),
         ("public_key", "exact_bytes_len_32"), ("key_id", "exact_prefixed_id")),
    ),
    (
        "ActualUnsealed960TestOnlyAnchorV2",
        tuple(
            (name, "exact_tuple_len_1") if name == "signer_keys"
            else (name, "exact_lowercase_sha256")
            if name in {"attempt_nonce_sha256", "signer_key_ids_root", "anchor_root_sha256"}
            else (name, "exact_prefixed_id")
            if name.endswith("_id")
            else (name, "exact_utf8_text")
            for name in _ANCHOR_FIELDS
        ),
    ),
    (
        "ActualUnsealed960AttemptEventV2",
        tuple(
            (name, "exact_int_0_through_9") if name == "event_index"
            else (name, "exact_SignedAttemptStageV2") if name == "stage"
            else (name, "exact_optional_prefixed_id") if name == "predecessor_event_id"
            else (name, "exact_bytes_len_64") if name == "signature"
            else (name, "exact_lowercase_sha256") if name == "payload_root_sha256"
            else (name, "exact_prefixed_id") if name.endswith("_id")
            else (name, "exact_utf8_text")
            for name in _EVENT_FIELDS
        ),
    ),
    (
        "ActualUnsealed960SignedAttemptChainV2",
        tuple(
            (name, "exact_tuple_len_10") if name == "events"
            else (name, "exact_lowercase_sha256") if name == "chain_root_sha256"
            else (name, "exact_prefixed_id") if name.endswith("_id")
            else (name, "exact_utf8_text")
            for name in _CHAIN_FIELDS
        ),
    ),
)


def _snapshot_key(value: object) -> dict[str, object]:
    if type(value) is not ActualUnsealed960SignerKeyV2:
        raise TypeError("signer key exact type drift")
    role = object.__getattribute__(value, "role")
    if type(role) is not SignedAttemptSignerRoleV2:
        raise TypeError("signer role exact enum drift")
    public_key = _raw(object.__getattribute__(value, "public_key"), 32, "public key")
    key_id = _id(object.__getattribute__(value, "key_id"), _KEY_PREFIX, "key ID")
    return {"role": role, "public_key": public_key, "key_id": key_id}


def _snapshot_anchor_shape(value: object) -> dict[str, object]:
    if type(value) is not ActualUnsealed960TestOnlyAnchorV2:
        raise TypeError("test anchor exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    for name in (
        "version", "claim_level", "anchor_scope", "schema_id", "policy_id",
        "protocol_id", "exact_freeze_id",
        "execution_freeze_manifest_id", "attempt_id", "anchor_id",
    ):
        if type(raw[name]) is not str:
            raise TypeError(f"anchor {name} exact string drift")
    for name in ("attempt_nonce_sha256", "signer_key_ids_root", "anchor_root_sha256"):
        if type(raw[name]) is not str:
            raise TypeError(f"anchor {name} exact string drift")
    if type(raw["signer_keys"]) is not tuple:
        raise TypeError("anchor signer keys exact tuple drift")
    version = _text(raw["version"], "anchor version")
    claim_level = _text(raw["claim_level"], "anchor claim level")
    anchor_scope = _text(raw["anchor_scope"], "anchor scope")
    schema_id = _id(raw["schema_id"], _SCHEMA_PREFIX, "anchor schema ID")
    policy_id = _id(raw["policy_id"], _POLICY_PREFIX, "anchor policy ID")
    protocol_id = _id(raw["protocol_id"], "phase2b_protocol_", "anchor protocol ID")
    exact_freeze_id = _id(raw["exact_freeze_id"], "phase2b_exact_freeze_", "anchor exact-freeze ID")
    execution_freeze_id = _id(raw["execution_freeze_manifest_id"], "phase2b_execution_freeze_", "execution freeze ID")
    attempt_nonce = _sha(raw["attempt_nonce_sha256"], "attempt nonce SHA")
    keys = tuple(_snapshot_key(item) for item in raw["signer_keys"])
    key_root = _sha(raw["signer_key_ids_root"], "signer key root")
    attempt_id = _id(raw["attempt_id"], _ATTEMPT_PREFIX, "attempt ID")
    anchor_root = _sha(raw["anchor_root_sha256"], "anchor root")
    anchor_id = _id(raw["anchor_id"], _ANCHOR_PREFIX, "anchor ID")
    return {
        "version": version, "claim_level": claim_level,
        "anchor_scope": anchor_scope, "schema_id": schema_id,
        "policy_id": policy_id, "protocol_id": protocol_id,
        "exact_freeze_id": exact_freeze_id,
        "execution_freeze_manifest_id": execution_freeze_id,
        "attempt_nonce_sha256": attempt_nonce, "attempt_id": attempt_id,
        "signer_keys": keys, "signer_key_ids_root": key_root,
        "anchor_root_sha256": anchor_root, "anchor_id": anchor_id,
    }


def _verify_anchor_content(raw: dict[str, object]) -> dict[str, object]:
    if raw["version"] != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION:
        raise ValueError("anchor version drift")
    if raw["claim_level"] != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL:
        raise ValueError("anchor claim level drift")
    if raw["anchor_scope"] != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2:
        raise ValueError("only the explicit test-only anchor scope is accepted")
    if (
        raw["schema_id"]
        != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID
        or raw["policy_id"]
        != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID
    ):
        raise ValueError("anchor schema/policy identity drift")
    if raw["protocol_id"] != _FROZEN_PROTOCOL_ID or raw["exact_freeze_id"] != _FROZEN_EXACT_FREEZE_ID:
        raise ValueError("anchor frozen identity drift")
    keys = raw["signer_keys"]
    if len(keys) != 1 or keys[0]["role"] is not SignedAttemptSignerRoleV2.CUSTODIAN:
        raise ValueError("test anchor requires exactly one custodian key")
    expected_key_id = _expected_key_id(keys[0]["role"], keys[0]["public_key"])
    if keys[0]["key_id"] != expected_key_id:
        raise ValueError("anchor signer key content address drift")
    key_ids = (expected_key_id,)
    key_root = _sequence_digest(_KEY_ROOT_DOMAIN, key_ids)
    if raw["signer_key_ids_root"] != key_root:
        raise ValueError("signer key root drift")
    attempt_values = {
        "schema_id": raw["schema_id"],
        "policy_id": raw["policy_id"],
        "protocol_id": raw["protocol_id"],
        "exact_freeze_id": raw["exact_freeze_id"],
        "execution_freeze_manifest_id": raw["execution_freeze_manifest_id"],
        "attempt_nonce_sha256": raw["attempt_nonce_sha256"],
    }
    if raw["attempt_id"] != _expected_attempt_id(**attempt_values):
        raise ValueError("attempt ID drift")
    pairs = _anchor_pairs(
        version=raw["version"], claim_level=raw["claim_level"],
        anchor_scope=raw["anchor_scope"], schema_id=raw["schema_id"],
        policy_id=raw["policy_id"], protocol_id=raw["protocol_id"],
        exact_freeze_id=raw["exact_freeze_id"],
        execution_freeze_manifest_id=raw["execution_freeze_manifest_id"],
        attempt_nonce_sha256=raw["attempt_nonce_sha256"],
        attempt_id=raw["attempt_id"],
        signer_key_ids=key_ids, signer_key_ids_root=key_root,
    )
    anchor_root = _digest(_ANCHOR_ROOT_DOMAIN, pairs)
    if raw["anchor_root_sha256"] != anchor_root:
        raise ValueError("anchor root drift")
    anchor_id = _ANCHOR_PREFIX + _digest(
        _ANCHOR_ID_DOMAIN, (*pairs, ("anchor_root_sha256", anchor_root))
    )
    if raw["anchor_id"] != anchor_id:
        raise ValueError("anchor ID drift")
    return {
        **raw, "signer_keys": keys, "signer_key_ids_root": key_root,
        "anchor_root_sha256": anchor_root, "anchor_id": anchor_id,
    }


def _validate_anchor(value: object) -> dict[str, object]:
    return _verify_anchor_content(_snapshot_anchor_shape(value))


def _snapshot_event_shape(value: object) -> dict[str, object]:
    if type(value) is not ActualUnsealed960AttemptEventV2:
        raise TypeError("attempt event exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    if type(raw["version"]) is not str or type(raw["anchor_id"]) is not str or type(raw["attempt_id"]) is not str:
        raise TypeError("event identity exact string drift")
    if type(raw["stage"]) is not SignedAttemptStageV2:
        raise TypeError("event stage exact enum drift")
    if raw["predecessor_event_id"] is not None and type(raw["predecessor_event_id"]) is not str:
        raise TypeError("event predecessor exact optional-string drift")
    index = _integer(raw["event_index"], "event index", maximum=_EVENT_COUNT - 1)
    version = _text(raw["version"], "event version")
    anchor_id = _id(raw["anchor_id"], _ANCHOR_PREFIX, "event anchor ID")
    attempt_id = _id(raw["attempt_id"], _ATTEMPT_PREFIX, "event attempt ID")
    predecessor = raw["predecessor_event_id"]
    if predecessor is not None:
        predecessor = _id(predecessor, _EVENT_PREFIX, "predecessor event ID")
    payload = _sha(raw["payload_root_sha256"], "event payload root")
    key_id = _id(raw["signer_key_id"], _KEY_PREFIX, "event signer key ID")
    signature = _raw(raw["signature"], 64, "event signature")
    event_id = _id(raw["event_id"], _EVENT_PREFIX, "event ID")
    return {
        "version": version, "anchor_id": anchor_id,
        "attempt_id": attempt_id, "event_index": index, "stage": raw["stage"],
        "predecessor_event_id": predecessor, "payload_root_sha256": payload,
        "signer_key_id": key_id, "signature": signature, "event_id": event_id,
    }


def _validate_event_shape(value: object) -> dict[str, object]:
    closed = _snapshot_event_shape(value)
    if closed["version"] != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION:
        raise ValueError("event version drift")
    return closed


def _snapshot_chain_shape(value: object) -> dict[str, object]:
    if type(value) is not ActualUnsealed960SignedAttemptChainV2:
        raise TypeError("signed attempt chain exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    if type(raw["version"]) is not str or type(raw["anchor_id"]) is not str or type(raw["attempt_id"]) is not str:
        raise TypeError("chain identity exact string drift")
    if type(raw["events"]) is not tuple:
        raise TypeError("chain events exact tuple drift")
    version = _text(raw["version"], "chain version")
    anchor_id = _id(raw["anchor_id"], _ANCHOR_PREFIX, "chain anchor ID")
    attempt_id = _id(raw["attempt_id"], _ATTEMPT_PREFIX, "chain attempt ID")
    if len(raw["events"]) != _EVENT_COUNT:
        raise ValueError("chain must contain exactly ten events")
    events = tuple(_snapshot_event_shape(item) for item in raw["events"])
    return {
        "version": version, "anchor_id": anchor_id,
        "attempt_id": attempt_id, "events": events,
        "chain_root_sha256": _sha(raw["chain_root_sha256"], "chain root"),
        "chain_id": _id(raw["chain_id"], _CHAIN_PREFIX, "chain ID"),
    }


def _validate_chain_shape(value: object) -> dict[str, object]:
    closed = _snapshot_chain_shape(value)
    if closed["version"] != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION:
        raise ValueError("chain version drift")
    if any(
        item["version"]
        != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION
        for item in closed["events"]
    ):
        raise ValueError("event version drift")
    return closed


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    if _Ed25519PublicKey is None:
        raise _BackendUnavailable
    try:
        _Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except Exception as exc:
        raise _SignatureInvalid from exc


class _BackendUnavailable(Exception):
    pass


class _SignatureInvalid(Exception):
    pass


def _verify_closed(anchor: dict[str, object], chain: dict[str, object]) -> dict[str, object]:
    if chain["anchor_id"] != anchor["anchor_id"] or chain["attempt_id"] != anchor["attempt_id"]:
        raise ValueError("chain/anchor identity mismatch")
    key = anchor["signer_keys"][0]
    prior: str | None = None
    event_ids: list[str] = []
    for expected_index, (expected_stage, event) in enumerate(zip(_STAGE_ORDER, chain["events"], strict=True)):
        if event["event_index"] != expected_index or event["stage"] is not expected_stage:
            raise ValueError("event index/stage order drift")
        if event["anchor_id"] != anchor["anchor_id"] or event["attempt_id"] != anchor["attempt_id"]:
            raise ValueError("event subject binding drift")
        if event["predecessor_event_id"] != prior:
            raise ValueError("event predecessor chain drift")
        if event["signer_key_id"] != key["key_id"]:
            raise ValueError("event signer key drift")
        if expected_stage is SignedAttemptStageV2.PREREGISTERED and event["payload_root_sha256"] != anchor["anchor_root_sha256"]:
            raise ValueError("preregistered event must bind anchor root")
        pairs = _event_unsigned_pairs(
            version=event["version"], anchor_id=event["anchor_id"],
            attempt_id=event["attempt_id"], event_index=event["event_index"],
            stage=event["stage"], predecessor_event_id=event["predecessor_event_id"],
            payload_root_sha256=event["payload_root_sha256"],
            signer_key_id=event["signer_key_id"],
        )
        expected_event_id = _expected_event_id(pairs, event["signature"])
        if event["event_id"] != expected_event_id:
            raise ValueError("event content address drift")
        _verify_ed25519(key["public_key"], event["signature"], _event_signature_preimage(pairs))
        prior = expected_event_id
        event_ids.append(expected_event_id)
    event_ids_tuple = tuple(event_ids)
    chain_root = _sequence_digest(_CHAIN_ROOT_DOMAIN, event_ids_tuple)
    if chain["chain_root_sha256"] != chain_root:
        raise ValueError("chain root drift")
    chain_pairs = _chain_pairs(
        version=chain["version"], anchor_id=chain["anchor_id"],
        attempt_id=chain["attempt_id"], event_ids=event_ids_tuple,
        chain_root_sha256=chain_root,
    )
    chain_id = _CHAIN_PREFIX + _digest(_CHAIN_ID_DOMAIN, chain_pairs)
    if chain["chain_id"] != chain_id:
        raise ValueError("chain ID drift")
    return {
        "key_id": key["key_id"],
        "raw_input_root": chain["events"][1]["payload_root_sha256"],
        "package_commitments_root": chain["events"][2]["payload_root_sha256"],
        "chain_root": chain_root,
        "chain_id": chain_id,
    }


_SUCCESS_FIELDS: Final = tuple(item.name for item in fields(ActualUnsealed960SignedAttemptChainMechanicsV2))
_REJECTION_FIELDS: Final = tuple(item.name for item in fields(ActualUnsealed960SignedAttemptChainRejectionV2))

_SCHEMA_PREIMAGE: Final = (
    ("version", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION),
    ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL),
    ("key_fields", _KEY_FIELDS), ("anchor_fields", _ANCHOR_FIELDS),
    ("event_fields", _EVENT_FIELDS), ("chain_fields", _CHAIN_FIELDS),
    ("success_fields", _SUCCESS_FIELDS), ("rejection_fields", _REJECTION_FIELDS),
    ("field_type_manifests", _FIELD_TYPE_MANIFESTS),
    ("signer_role_values", tuple(item.value for item in SignedAttemptSignerRoleV2)),
    ("stage_values", tuple(item.value for item in SignedAttemptStageV2)),
    ("disposition_values", tuple(item.value for item in SignedAttemptChainDispositionV2)),
    ("reason_values", tuple(item.value for item in SignedAttemptChainReasonV2)),
    ("caps", (("maximum_utf8_bytes", _MAX_TEXT_BYTES),
              ("exact_event_count", _EVENT_COUNT),
              ("raw_public_key_bytes", 32), ("raw_signature_bytes", 64))),
    ("public_operation_signature", (
        "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
        "keyword_only_anchor_exact_ActualUnsealed960TestOnlyAnchorV2",
        "keyword_only_chain_exact_ActualUnsealed960SignedAttemptChainV2",
    )),
)
PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID: Final = (
    _SCHEMA_PREFIX + _digest(_SCHEMA_DOMAIN, _SCHEMA_PREIMAGE)
)

_POLICY_PREIMAGE: Final = (
    ("version", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION),
    ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL),
    ("anchor_scope", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2),
    ("frozen_protocol_id", _FROZEN_PROTOCOL_ID), ("frozen_exact_freeze_id", _FROZEN_EXACT_FREEZE_ID),
    ("stage_order", tuple(item.value for item in _STAGE_ORDER)),
    ("true_claims", _TRUE_CLAIMS), ("false_claims", _FALSE_CLAIMS),
    ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID),
    ("domains", tuple(item.hex() for item in (
        _KEY_DOMAIN, _KEY_ROOT_DOMAIN, _ATTEMPT_DOMAIN, _ANCHOR_ROOT_DOMAIN,
        _ANCHOR_ID_DOMAIN, _EVENT_SIGNATURE_DOMAIN, _EVENT_ID_DOMAIN,
        _CHAIN_ROOT_DOMAIN, _CHAIN_ID_DOMAIN, _RESULT_ID_DOMAIN,
        _SCHEMA_DOMAIN, _POLICY_DOMAIN,
    ))),
    ("prefixes", (
        _KEY_PREFIX, _ATTEMPT_PREFIX, _ANCHOR_PREFIX, _EVENT_PREFIX,
        _CHAIN_PREFIX, _RESULT_PREFIX, _SCHEMA_PREFIX, _POLICY_PREFIX,
    )),
    ("id_prefix_rules", (
        ("key_id", _KEY_PREFIX), ("attempt_id", _ATTEMPT_PREFIX),
        ("anchor_id", _ANCHOR_PREFIX), ("event_id", _EVENT_PREFIX),
        ("predecessor_event_id", _EVENT_PREFIX), ("chain_id", _CHAIN_PREFIX),
        ("result_id", _RESULT_PREFIX), ("schema_id", _SCHEMA_PREFIX),
        ("policy_id", _POLICY_PREFIX),
        ("protocol_id", "phase2b_protocol_"),
        ("exact_freeze_id", "phase2b_exact_freeze_"),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_"),
    )),
    ("content_address_formula",
     "prefix || lowercase_hex_sha256(domain_bytes || utf8(canonical_json(tuple_of_declared_order_name_value_pairs)))"),
    ("key_id_preimage_fields", ("role", "algorithm=Ed25519", "public_key_hex")),
    ("attempt_id_preimage_fields", (
        "schema_id", "policy_id", "protocol_id", "exact_freeze_id",
        "execution_freeze_manifest_id", "attempt_nonce_sha256",
    )),
    ("anchor_root_preimage_fields", (
        "version", "claim_level", "anchor_scope", "schema_id", "policy_id",
        "protocol_id", "exact_freeze_id", "execution_freeze_manifest_id",
        "attempt_nonce_sha256", "attempt_id", "signer_key_ids",
        "signer_key_ids_root",
    )),
    ("anchor_id_additional_preimage_field", "anchor_root_sha256"),
    ("unsigned_event_preimage_fields", (
        "version", "anchor_id", "attempt_id", "event_index", "stage",
        "predecessor_event_id", "payload_root_sha256", "signer_key_id",
    )),
    ("event_id_additional_preimage_field", "signature_hex"),
    ("chain_id_preimage_fields", (
        "version", "anchor_id", "attempt_id", "event_ids",
        "chain_root_sha256",
    )),
    ("result_id_preimage_fields", (
        "version", "schema_id", "policy_id", "anchor_id", "attempt_id",
        "signer_key_id", "raw_input_committed_payload_root_sha256",
        "package_commitments_frozen_payload_root_sha256",
        "chain_root_sha256", "chain_id", "event_count", "terminal_stage",
    )),
    ("success_semantics", (
        SignedAttemptChainDispositionV2.SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY.value,
        SignedAttemptChainReasonV2.TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED.value,
        "all_true_claims_exact_true", "all_false_claims_exact_false",
    )),
    ("ordered_root_formula",
     "lowercase_hex_sha256(domain_bytes || u32_big_endian_count || repeated_u16_big_endian_length_plus_ascii_id)"),
    ("event_signature_formula",
     "Ed25519_sign(domain_bytes || utf8(canonical_json(exact_unsigned_event_named_pairs))); signature_and_event_id_excluded"),
    ("event_id_formula",
     "event_prefix || lowercase_hex_sha256(event_id_domain || utf8(canonical_json(unsigned_event_named_pairs_plus_signature_hex)))"),
    ("attempt_preregistration_formula",
     "attempt_id_binds_schema_policy_protocol_exact_freeze_execution_freeze_and_attempt_nonce_sha256_no_future_input_or_package_values"),
    ("payload_introduction_rules", (
        "PREREGISTERED_payload_equals_test_anchor_root",
        "RAW_INPUT_COMMITTED_payload_introduces_raw_input_commitment_root",
        "PACKAGE_COMMITMENTS_FROZEN_payload_introduces_package_commitments_root",
        "all_later_stage_payloads_are_signed_opaque_sha256_roots_only",
    )),
    ("rules", (
        "complete_anchor_and_ten_event_chain_exact_type_format_cap_snapshot_before_any_operation_hash_or_signature",
        "exact_one_test_only_custodian_key_raw32_and_ed25519_signature_raw64",
        "event_signature_excludes_signature_and_event_id",
        "event_id_binds_signature_hex",
        "exact_ten_contiguous_events_and_predecessor_chain",
        "no_future_value_placeholder_in_test_anchor_or_attempt_id",
        "cryptography_backend_unavailable_fails_closed",
        "atomic_rejection_covers_exact_input_and_ordinary_backend_failures_not_injected_BaseException_or_process_termination",
        "caller_cannot_supply_signature_verifier_or_signature_validity_boolean",
        "verification_performs_no_signature_or_key_generation",
        "no_external_authority_durability_time_execution_scoring_gate_effect_or_c1_claim",
    )),
)
PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID: Final = (
    _POLICY_PREFIX + _digest(_POLICY_DOMAIN, _POLICY_PREIMAGE)
)


def _issue_success(anchor: dict[str, object], verified: dict[str, object]) -> ActualUnsealed960SignedAttemptChainMechanicsV2:
    preimage = (
        ("version", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION),
        ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID),
        ("anchor_id", anchor["anchor_id"]), ("attempt_id", anchor["attempt_id"]),
        ("signer_key_id", verified["key_id"]),
        ("raw_input_committed_payload_root_sha256", verified["raw_input_root"]),
        ("package_commitments_frozen_payload_root_sha256", verified["package_commitments_root"]),
        ("chain_root_sha256", verified["chain_root"]),
        ("chain_id", verified["chain_id"]), ("event_count", _EVENT_COUNT),
        ("terminal_stage", SignedAttemptStageV2.TERMINAL_CONSUMED.value),
    )
    result_id = _RESULT_PREFIX + _digest(_RESULT_ID_DOMAIN, preimage)
    result = object.__new__(ActualUnsealed960SignedAttemptChainMechanicsV2)
    values: list[tuple[str, object]] = [
        ("disposition", SignedAttemptChainDispositionV2.SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY),
        ("reason", SignedAttemptChainReasonV2.TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED),
        ("version", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION),
        ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID),
        ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL),
        ("result_id", result_id), ("anchor_id", anchor["anchor_id"]),
        ("attempt_id", anchor["attempt_id"]), ("signer_key_id", verified["key_id"]),
        ("raw_input_committed_payload_root_sha256", verified["raw_input_root"]),
        ("package_commitments_frozen_payload_root_sha256", verified["package_commitments_root"]),
        ("chain_root_sha256", verified["chain_root"]), ("chain_id", verified["chain_id"]),
        ("event_count", _EVENT_COUNT), ("terminal_stage", SignedAttemptStageV2.TERMINAL_CONSUMED),
    ]
    values.extend((name, True) for name in _TRUE_CLAIMS)
    values.extend((name, False) for name in _FALSE_CLAIMS)
    if tuple(name for name, _ in values) != _SUCCESS_FIELDS:
        raise RuntimeError("signed-chain success issuance field drift")
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def _issue_rejection(reason: SignedAttemptChainReasonV2) -> ActualUnsealed960SignedAttemptChainRejectionV2:
    result = object.__new__(ActualUnsealed960SignedAttemptChainRejectionV2)
    values: list[tuple[str, object]] = [
        ("disposition", SignedAttemptChainDispositionV2.REJECTED), ("reason", reason),
        ("version", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION),
        ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID),
        ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL),
        ("validation", None), ("result_id", None), ("anchor_id", None),
        ("attempt_id", None), ("signer_key_id", None),
        ("raw_input_committed_payload_root_sha256", None),
        ("package_commitments_frozen_payload_root_sha256", None),
        ("chain_root_sha256", None),
        ("chain_id", None), ("event_count", 0), ("terminal_stage", None),
        ("partial_output_published", False),
    ]
    values.extend((name, False) for name in (*_TRUE_CLAIMS, *_FALSE_CLAIMS))
    if tuple(name for name, _ in values) != _REJECTION_FIELDS:
        raise RuntimeError("signed-chain rejection issuance field drift")
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2(
    *,
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    chain: ActualUnsealed960SignedAttemptChainV2,
) -> ActualUnsealed960SignedAttemptChainMechanicsV2 | ActualUnsealed960SignedAttemptChainRejectionV2:
    """Verify one test-anchor signed attempt chain without claiming authority."""

    if type(anchor) is not ActualUnsealed960TestOnlyAnchorV2 or type(chain) is not ActualUnsealed960SignedAttemptChainV2:
        return _issue_rejection(SignedAttemptChainReasonV2.WRONG_INPUT_TYPE)
    try:
        try:
            closed_anchor_shape = _snapshot_anchor_shape(anchor)
        except (TypeError, ValueError, OverflowError, UnicodeError):
            return _issue_rejection(SignedAttemptChainReasonV2.TEST_ANCHOR_INVALID)
        try:
            closed_chain = _snapshot_chain_shape(chain)
        except (TypeError, ValueError, OverflowError, UnicodeError):
            return _issue_rejection(SignedAttemptChainReasonV2.CHAIN_INVALID)
        versions = (
            closed_anchor_shape["version"],
            closed_chain["version"],
            *(item["version"] for item in closed_chain["events"]),
        )
        if any(
            value
            != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION
            for value in versions
        ):
            return _issue_rejection(SignedAttemptChainReasonV2.CROSS_VERSION_INPUT)
        try:
            closed_anchor = _verify_anchor_content(closed_anchor_shape)
        except (TypeError, ValueError, OverflowError, UnicodeError):
            return _issue_rejection(SignedAttemptChainReasonV2.TEST_ANCHOR_INVALID)
        try:
            verified = _verify_closed(closed_anchor, closed_chain)
        except _BackendUnavailable:
            return _issue_rejection(
                SignedAttemptChainReasonV2.SIGNATURE_BACKEND_UNAVAILABLE
            )
        except _SignatureInvalid:
            return _issue_rejection(SignedAttemptChainReasonV2.SIGNATURE_INVALID)
        except (TypeError, ValueError, OverflowError, UnicodeError):
            return _issue_rejection(SignedAttemptChainReasonV2.CHAIN_INVALID)
        return _issue_success(closed_anchor, verified)
    except _BackendUnavailable:
        return _issue_rejection(SignedAttemptChainReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
    except _SignatureInvalid:
        return _issue_rejection(SignedAttemptChainReasonV2.SIGNATURE_INVALID)
    except (TypeError, ValueError, OverflowError, UnicodeError):
        return _issue_rejection(SignedAttemptChainReasonV2.CHAIN_INVALID)
    except Exception:
        return _issue_rejection(SignedAttemptChainReasonV2.INTERNAL_ERROR)


for _type, _manifest in (
    (ActualUnsealed960SignerKeyV2, _KEY_FIELDS),
    (ActualUnsealed960TestOnlyAnchorV2, _ANCHOR_FIELDS),
    (ActualUnsealed960AttemptEventV2, _EVENT_FIELDS),
    (ActualUnsealed960SignedAttemptChainV2, _CHAIN_FIELDS),
):
    if tuple(item.name for item in fields(_type)) != _manifest:
        raise RuntimeError(f"signed-chain {_type.__name__} field drift")

if len(_STAGE_ORDER) != _EVENT_COUNT or len(set(_STAGE_ORDER)) != _EVENT_COUNT:
    raise RuntimeError("signed-chain stage catalog drift")
if len(set(_TRUE_CLAIMS)) != len(_TRUE_CLAIMS):
    raise RuntimeError("signed-chain true-claim duplicate")
if len(set(_FALSE_CLAIMS)) != len(_FALSE_CLAIMS):
    raise RuntimeError("signed-chain false-claim duplicate")
if set(_TRUE_CLAIMS) & set(_FALSE_CLAIMS):
    raise RuntimeError("signed-chain claim polarity overlap")
if not set((*_TRUE_CLAIMS, *_FALSE_CLAIMS)).issubset(_SUCCESS_FIELDS):
    raise RuntimeError("signed-chain success claim-field coverage drift")
if not set((*_TRUE_CLAIMS, *_FALSE_CLAIMS)).issubset(_REJECTION_FIELDS):
    raise RuntimeError("signed-chain rejection claim-field coverage drift")
_ALL_DOMAINS = (
    _KEY_DOMAIN, _KEY_ROOT_DOMAIN, _ATTEMPT_DOMAIN, _ANCHOR_ROOT_DOMAIN,
    _ANCHOR_ID_DOMAIN, _EVENT_SIGNATURE_DOMAIN, _EVENT_ID_DOMAIN,
    _CHAIN_ROOT_DOMAIN, _CHAIN_ID_DOMAIN, _RESULT_ID_DOMAIN,
    _SCHEMA_DOMAIN, _POLICY_DOMAIN,
)
if len(set(_ALL_DOMAINS)) != len(_ALL_DOMAINS):
    raise RuntimeError("signed-chain domain separation drift")
_ALL_PREFIXES = (
    _KEY_PREFIX, _ATTEMPT_PREFIX, _ANCHOR_PREFIX, _EVENT_PREFIX,
    _CHAIN_PREFIX, _RESULT_PREFIX, _SCHEMA_PREFIX, _POLICY_PREFIX,
)
if len(set(_ALL_PREFIXES)) != len(_ALL_PREFIXES):
    raise RuntimeError("signed-chain content-address prefix drift")
if {"input_archive_id", "input_archive_sha256", "package_commitments_root_sha256"} & set(_ANCHOR_FIELDS):
    raise RuntimeError("signed-chain test anchor contains future attempt values")


__all__ = (
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_ED25519_BACKEND_AVAILABLE",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID",
    "SignedAttemptSignerRoleV2",
    "SignedAttemptStageV2",
    "SignedAttemptChainDispositionV2",
    "SignedAttemptChainReasonV2",
    "ActualUnsealed960SignerKeyV2",
    "ActualUnsealed960TestOnlyAnchorV2",
    "ActualUnsealed960AttemptEventV2",
    "ActualUnsealed960SignedAttemptChainV2",
    "ActualUnsealed960SignedAttemptChainMechanicsV2",
    "ActualUnsealed960SignedAttemptChainRejectionV2",
    "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
)
