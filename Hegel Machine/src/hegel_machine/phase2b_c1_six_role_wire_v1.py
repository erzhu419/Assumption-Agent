"""Pure Phase-2B C1 six-role ceremony wire verification.

This module snapshots and verifies a caller-supplied graph consisting of one
ceremony context, an ordered six-key Ed25519 execution registry signed by the
declared governance key, a STORE ``IF_ABSENT`` generation ``0 -> 1`` receipt,
and a TIME logical-genesis token.  It deliberately performs no filesystem,
Docker, clock, key-generation, signature-generation, orchestration, actual-960,
scoring, gate, effect, or C1 operation.

The registry keys and all image/source identities are supplied by the caller.
Successful cryptographic verification under those keys proves only the
content-addressed wire mechanics and graph bindings.  It does not promote the
supplied registry, image digests, or source roots into an external authority or
an observation of a live ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final

try:  # Optional backend; absence produces a fail-closed public rejection.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by backend monkeypatch tests.
    _Ed25519PublicKey = None


PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION: Final = (
    "hegel-machine-phase2b-c1-six-role-wire/1"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_CLAIM_LEVEL: Final = (
    "CALLER_SUPPLIED_OWNER_ACCEPTED_CONTEXT_CRYPTOGRAPHIC_WIRE_MECHANICS_ONLY"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS: Final = (
    "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID: Final = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256: Final = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID: Final = (
    "phase2b_protocol_"
    "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID: Final = (
    "phase2b_exact_freeze_"
    "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256: Final = (
    "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_ED25519_BACKEND_AVAILABLE: Final = (
    _Ed25519PublicKey is not None
)
PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH: Final = 1
PHASE2B_C1_SIX_ROLE_WIRE_V1_STORE_TRANSITION: Final = (0, 1, "IF_ABSENT")
PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS: Final = (
    "SIGNED_MONOTONIC_LOGICAL_ORDER_NOT_UTC"
)

_LEGACY_SCORER_SOURCE_ROOT_SHA256: Final = (
    "3f28fcf1f5669859350d6297a8ab45e81740e5dea1a375f5257ea278f6ca0b6a"
)
_MAXIMUM_ASCII_TEXT_BYTES: Final = 4_096
_RAW_PUBLIC_KEY_BYTES: Final = 32
_RAW_SIGNATURE_BYTES: Final = 64

_ATTEMPT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/ATTEMPT/V1\x00"
_CONTEXT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/CONTEXT/V1\x00"
_KEY_DOMAIN: Final = b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/ROLE_KEY/V1\x00"
_GOVERNANCE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/GOVERNANCE_REGISTRY_SIGNATURE/V1\x00"
)
_REGISTRY_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/EXECUTION_REGISTRY_ID/V1\x00"
)
_STORE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/STORE_IF_ABSENT_SIGNATURE/V1\x00"
)
_STORE_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/STORE_IF_ABSENT_RECEIPT_ID/V1\x00"
)
_TIME_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/LOGICAL_GENESIS_SIGNATURE/V1\x00"
)
_TIME_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/LOGICAL_GENESIS_TOKEN_ID/V1\x00"
)
_RESULT_ID_DOMAIN: Final = b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/RESULT/V1\x00"

_ATTEMPT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_v2_"
_CONTEXT_PREFIX: Final = "phase2b_c1_six_role_ceremony_context_v1_"
_KEY_PREFIX: Final = "phase2b_c1_six_role_key_v1_"
_REGISTRY_PREFIX: Final = "phase2b_c1_owner_accepted_execution_registry_v1_"
_STORE_RECEIPT_PREFIX: Final = (
    "phase2b_c1_execution_registry_if_absent_receipt_v1_"
)
_TIME_TOKEN_PREFIX: Final = "phase2b_c1_signed_logical_genesis_token_v1_"
_RESULT_PREFIX: Final = "phase2b_c1_six_role_wire_verification_v1_"
_CAMPAIGN_PREFIX: Final = "phase2b_actual_unsealed_960_campaign_v2_"
_EXECUTION_FREEZE_PREFIX: Final = "phase2b_execution_freeze_"


class Phase2BC1SixRoleV1(str, Enum):
    GOVERNANCE_ANCHOR = "GOVERNANCE_ANCHOR"
    CUSTODIAN = "CUSTODIAN"
    STORE_RECEIPT = "STORE_RECEIPT"
    TIME_ATTESTOR = "TIME_ATTESTOR"
    RECOGNIZER = "RECOGNIZER"
    SCORER_EVALUATOR = "SCORER_EVALUATOR"


PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER: Final = tuple(Phase2BC1SixRoleV1)


class Phase2BC1SignaturePurposeV1(str, Enum):
    GOVERNANCE_CHECKPOINT = "GOVERNANCE_CHECKPOINT"
    STORE_IF_ABSENT_RECEIPT = "STORE_IF_ABSENT_RECEIPT"
    DECLARED_TIME_TOKEN = "DECLARED_TIME_TOKEN"


class Phase2BC1LogicalStageV1(str, Enum):
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


class Phase2BC1SixRoleWireDispositionV1(str, Enum):
    CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY = (
        "CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY"
    )
    REJECTED = "REJECTED"


class Phase2BC1SixRoleWireReasonV1(str, Enum):
    CALLER_SUPPLIED_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED = (
        "CALLER_SUPPLIED_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CONTEXT_INVALID = "CONTEXT_INVALID"
    REGISTRY_INVALID = "REGISTRY_INVALID"
    STORE_RECEIPT_INVALID = "STORE_RECEIPT_INVALID"
    LOGICAL_GENESIS_INVALID = "LOGICAL_GENESIS_INVALID"
    GRAPH_BINDING_INVALID = "GRAPH_BINDING_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def _ascii_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be ASCII") from exc
    if not 1 <= len(encoded) <= _MAXIMUM_ASCII_TEXT_BYTES:
        raise ValueError(f"{name} ASCII size drift")
    return value


def _sha256(value: object, name: str) -> str:
    text = _ascii_text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _identifier(value: object, prefix: str, name: str) -> str:
    text = _ascii_text(value, name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _sha256(text[len(prefix):], f"{name} suffix")
    return text


def _image_digest(value: object, name: str) -> str:
    text = _ascii_text(value, name)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be sha256:<lowercase-hex64>")
    return text


def _raw_bytes(value: object, length: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        raise TypeError(f"{name} must be exact {length}-byte bytes")
    return bytes(value)


def _exact_integer(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} integer drift")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _canonical_named_bytes(pairs: tuple[tuple[str, object], ...]) -> bytes:
    payload = json.dumps(
        pairs,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    if payload.endswith(b"\n"):
        raise RuntimeError("canonical named-pair wire unexpectedly ended in newline")
    return payload


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + _canonical_named_bytes(pairs)).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase2BC1CeremonyContextBodyV1:
    version: str
    authority_class: str
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    execution_freeze_manifest_sha256: str
    attempt_nonce_sha256: str
    attempt_id: str
    recognizer_image_digest: str
    recognizer_source_root_sha256: str
    scorer_image_digest: str
    scorer_source_root_sha256: str

    def __post_init__(self) -> None:
        _snapshot_context_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1CeremonyContextV1:
    body: Phase2BC1CeremonyContextBodyV1
    context_id: str

    def __post_init__(self) -> None:
        _snapshot_context(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1RoleKeyV1:
    purpose_id: int
    role: Phase2BC1SixRoleV1
    algorithm: str
    key_epoch: int
    purpose_private: bool
    public_key: bytes
    key_id: str

    def __post_init__(self) -> None:
        _snapshot_role_key(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1DetachedSignatureV1:
    purpose_id: int
    role: Phase2BC1SixRoleV1
    signature_purpose: Phase2BC1SignaturePurposeV1
    algorithm: str
    key_epoch: int
    signer_key_id: str
    signature: bytes

    def __post_init__(self) -> None:
        _snapshot_signature(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1ExecutionRegistryBodyV1:
    version: str
    context_id: str
    authority_class: str
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    execution_freeze_manifest_sha256: str
    attempt_nonce_sha256: str
    attempt_id: str
    recognizer_image_digest: str
    recognizer_source_root_sha256: str
    scorer_image_digest: str
    scorer_source_root_sha256: str
    role_keys: tuple[Phase2BC1RoleKeyV1, ...]

    def __post_init__(self) -> None:
        _snapshot_registry_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1ExecutionRegistryCandidateV1:
    body: Phase2BC1ExecutionRegistryBodyV1
    governance_signature: Phase2BC1DetachedSignatureV1
    execution_registry_id: str

    def __post_init__(self) -> None:
        _snapshot_registry(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1StoreIfAbsentReceiptBodyV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    object_key: str
    previous_generation: int
    new_generation: int
    condition: str
    previous_content_sha256_or_null: str | None
    stored_content_sha256: str

    def __post_init__(self) -> None:
        _snapshot_store_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1StoreIfAbsentReceiptCandidateV1:
    body: Phase2BC1StoreIfAbsentReceiptBodyV1
    store_signature: Phase2BC1DetachedSignatureV1
    store_receipt_id: str

    def __post_init__(self) -> None:
        _snapshot_store_receipt(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalGenesisTokenBodyV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    store_receipt_id: str
    time_semantics: str
    previous_logical_sequence_or_null: int | None
    logical_sequence: int
    stage: Phase2BC1LogicalStageV1
    stage_index: int

    def __post_init__(self) -> None:
        _snapshot_time_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalGenesisTokenCandidateV1:
    body: Phase2BC1LogicalGenesisTokenBodyV1
    time_signature: Phase2BC1DetachedSignatureV1
    logical_genesis_token_id: str

    def __post_init__(self) -> None:
        _snapshot_time_token(self)


_TRUE_CLAIMS: Final = (
    "complete_graph_snapshotted_before_hash_or_signature",
    "all_content_addresses_checked_before_signature_verification",
    "ceremony_context_content_address_verified",
    "ordered_six_role_key_registry_verified",
    "six_distinct_public_keys_verified",
    "declared_purpose_private_key_layout_verified",
    "governance_registry_signature_verified",
    "execution_registry_content_address_verified",
    "registry_image_source_context_binding_verified",
    "store_if_absent_zero_to_one_transition_verified",
    "store_receipt_signature_verified",
    "store_receipt_content_address_verified",
    "logical_genesis_shape_verified",
    "logical_genesis_signature_verified",
    "logical_genesis_content_address_verified",
    "graph_cross_bindings_verified",
    "verification_result_privately_issued",
)

_FALSE_CLAIMS: Final = (
    "actual_execution_claimed",
    "docker_invoked",
    "filesystem_read",
    "filesystem_written",
    "clock_read",
    "keys_generated",
    "signatures_generated",
    "fresh_keys_generated",
    "purpose_private_key_custody_verified",
    "caller_supplied_key_registry_authoritative",
    "governance_identity_verified",
    "registry_issuance_authority_verified",
    "image_runtime_identity_observed",
    "source_root_runtime_identity_observed",
    "actor_qualification_executed",
    "technical_role_independence_live_verified",
    "store_operation_executed",
    "store_durability_verified",
    "store_rollback_resistance_verified",
    "cross_process_no_replace_verified",
    "cross_host_global_one_shot_verified",
    "one_shot_policy_enforced",
    "time_operation_executed",
    "authoritative_time_verified",
    "utc_time_verified",
    "timeline_observed",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
    "organizational_independence_verified",
    "independent_human_actors_verified",
    "malicious_host_root_resistance",
    "malicious_docker_daemon_resistance",
    "owner_rollback_deletion_resistance",
    "remote_attestation_verified",
    "hardware_key_nonexportability_verified",
    "execution_authorized",
)


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1SixRoleWireVerificationV1:
    disposition: Phase2BC1SixRoleWireDispositionV1
    reason: Phase2BC1SixRoleWireReasonV1
    version: str
    claim_level: str
    result_id: str | None
    context_id: str | None
    attempt_id: str | None
    execution_registry_id: str | None
    store_receipt_id: str | None
    logical_genesis_token_id: str | None
    role_key_ids: tuple[str, ...]
    partial_output_published: bool
    complete_graph_snapshotted_before_hash_or_signature: bool
    all_content_addresses_checked_before_signature_verification: bool
    ceremony_context_content_address_verified: bool
    ordered_six_role_key_registry_verified: bool
    six_distinct_public_keys_verified: bool
    declared_purpose_private_key_layout_verified: bool
    governance_registry_signature_verified: bool
    execution_registry_content_address_verified: bool
    registry_image_source_context_binding_verified: bool
    store_if_absent_zero_to_one_transition_verified: bool
    store_receipt_signature_verified: bool
    store_receipt_content_address_verified: bool
    logical_genesis_shape_verified: bool
    logical_genesis_signature_verified: bool
    logical_genesis_content_address_verified: bool
    graph_cross_bindings_verified: bool
    verification_result_privately_issued: bool
    actual_execution_claimed: bool
    docker_invoked: bool
    filesystem_read: bool
    filesystem_written: bool
    clock_read: bool
    keys_generated: bool
    signatures_generated: bool
    fresh_keys_generated: bool
    purpose_private_key_custody_verified: bool
    caller_supplied_key_registry_authoritative: bool
    governance_identity_verified: bool
    registry_issuance_authority_verified: bool
    image_runtime_identity_observed: bool
    source_root_runtime_identity_observed: bool
    actor_qualification_executed: bool
    technical_role_independence_live_verified: bool
    store_operation_executed: bool
    store_durability_verified: bool
    store_rollback_resistance_verified: bool
    cross_process_no_replace_verified: bool
    cross_host_global_one_shot_verified: bool
    one_shot_policy_enforced: bool
    time_operation_executed: bool
    authoritative_time_verified: bool
    utc_time_verified: bool
    timeline_observed: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool
    organizational_independence_verified: bool
    independent_human_actors_verified: bool
    malicious_host_root_resistance: bool
    malicious_docker_daemon_resistance: bool
    owner_rollback_deletion_resistance: bool
    remote_attestation_verified: bool
    hardware_key_nonexportability_verified: bool
    execution_authorized: bool

    def __new__(
        cls, token: object = None
    ) -> "Phase2BC1SixRoleWireVerificationV1":
        if token is not _RESULT_ISSUANCE_TOKEN:
            raise TypeError("six-role wire verification results are privately issued")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _RESULT_ISSUANCE_TOKEN:
            raise TypeError("six-role wire verification results are privately issued")


_RESULT_ISSUANCE_TOKEN: Final = object()


def _snapshot_context_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1CeremonyContextBodyV1:
        raise TypeError("ceremony context body exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    return {
        "version": _ascii_text(raw["version"], "context version"),
        "authority_class": _ascii_text(raw["authority_class"], "authority class"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"],
            "phase2b_c1_container_actor_profile_v2_",
            "actor profile content ID",
        ),
        "actor_profile_canonical_sha256": _sha256(
            raw["actor_profile_canonical_sha256"], "actor profile canonical SHA"
        ),
        "campaign_id": _identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "campaign ID"),
        "protocol_id": _identifier(raw["protocol_id"], "phase2b_protocol_", "protocol ID"),
        "exact_freeze_id": _identifier(
            raw["exact_freeze_id"], "phase2b_exact_freeze_", "exact-freeze ID"
        ),
        "execution_freeze_manifest_id": _identifier(
            raw["execution_freeze_manifest_id"],
            _EXECUTION_FREEZE_PREFIX,
            "execution-freeze manifest ID",
        ),
        "execution_freeze_manifest_sha256": _sha256(
            raw["execution_freeze_manifest_sha256"],
            "execution-freeze manifest SHA",
        ),
        "attempt_nonce_sha256": _sha256(raw["attempt_nonce_sha256"], "attempt nonce SHA"),
        "attempt_id": _identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "attempt ID"),
        "recognizer_image_digest": _image_digest(
            raw["recognizer_image_digest"], "recognizer image digest"
        ),
        "recognizer_source_root_sha256": _sha256(
            raw["recognizer_source_root_sha256"], "recognizer source root"
        ),
        "scorer_image_digest": _image_digest(raw["scorer_image_digest"], "scorer image digest"),
        "scorer_source_root_sha256": _sha256(
            raw["scorer_source_root_sha256"], "scorer source root"
        ),
    }


def _snapshot_context(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1CeremonyContextV1:
        raise TypeError("ceremony context exact type drift")
    body = object.__getattribute__(value, "body")
    context_id = object.__getattribute__(value, "context_id")
    return {
        "body": _snapshot_context_body(body),
        "context_id": _identifier(context_id, _CONTEXT_PREFIX, "context ID"),
    }


def _snapshot_role_key(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1RoleKeyV1:
        raise TypeError("role key exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    if type(raw["role"]) is not Phase2BC1SixRoleV1:
        raise TypeError("role key role exact enum drift")
    return {
        "purpose_id": _exact_integer(raw["purpose_id"], "purpose ID", maximum=6),
        "role": raw["role"],
        "algorithm": _ascii_text(raw["algorithm"], "role key algorithm"),
        "key_epoch": _exact_integer(raw["key_epoch"], "key epoch", maximum=2**31 - 1),
        "purpose_private": _exact_bool(raw["purpose_private"], "purpose-private declaration"),
        "public_key": _raw_bytes(raw["public_key"], _RAW_PUBLIC_KEY_BYTES, "Ed25519 public key"),
        "key_id": _identifier(raw["key_id"], _KEY_PREFIX, "role key ID"),
    }


def _snapshot_signature(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1DetachedSignatureV1:
        raise TypeError("detached signature exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    if type(raw["role"]) is not Phase2BC1SixRoleV1:
        raise TypeError("signature role exact enum drift")
    if type(raw["signature_purpose"]) is not Phase2BC1SignaturePurposeV1:
        raise TypeError("signature purpose exact enum drift")
    return {
        "purpose_id": _exact_integer(raw["purpose_id"], "signature purpose ID", maximum=6),
        "role": raw["role"],
        "signature_purpose": raw["signature_purpose"],
        "algorithm": _ascii_text(raw["algorithm"], "signature algorithm"),
        "key_epoch": _exact_integer(raw["key_epoch"], "signature key epoch", maximum=2**31 - 1),
        "signer_key_id": _identifier(raw["signer_key_id"], _KEY_PREFIX, "signer key ID"),
        "signature": _raw_bytes(raw["signature"], _RAW_SIGNATURE_BYTES, "Ed25519 signature"),
    }


def _snapshot_registry_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1ExecutionRegistryBodyV1:
        raise TypeError("execution registry body exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    if type(raw["role_keys"]) is not tuple:
        raise TypeError("registry role keys must be an exact tuple")
    return {
        "version": _ascii_text(raw["version"], "registry version"),
        "context_id": _identifier(raw["context_id"], _CONTEXT_PREFIX, "registry context ID"),
        "authority_class": _ascii_text(raw["authority_class"], "registry authority class"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "registry authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"],
            "phase2b_c1_container_actor_profile_v2_",
            "registry actor profile content ID",
        ),
        "actor_profile_canonical_sha256": _sha256(
            raw["actor_profile_canonical_sha256"], "registry profile canonical SHA"
        ),
        "campaign_id": _identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "registry campaign ID"),
        "protocol_id": _identifier(raw["protocol_id"], "phase2b_protocol_", "registry protocol ID"),
        "exact_freeze_id": _identifier(
            raw["exact_freeze_id"], "phase2b_exact_freeze_", "registry exact-freeze ID"
        ),
        "execution_freeze_manifest_id": _identifier(
            raw["execution_freeze_manifest_id"],
            _EXECUTION_FREEZE_PREFIX,
            "registry execution-freeze ID",
        ),
        "execution_freeze_manifest_sha256": _sha256(
            raw["execution_freeze_manifest_sha256"], "registry execution-freeze SHA"
        ),
        "attempt_nonce_sha256": _sha256(raw["attempt_nonce_sha256"], "registry attempt nonce SHA"),
        "attempt_id": _identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "registry attempt ID"),
        "recognizer_image_digest": _image_digest(
            raw["recognizer_image_digest"], "registry recognizer image digest"
        ),
        "recognizer_source_root_sha256": _sha256(
            raw["recognizer_source_root_sha256"], "registry recognizer source root"
        ),
        "scorer_image_digest": _image_digest(
            raw["scorer_image_digest"], "registry scorer image digest"
        ),
        "scorer_source_root_sha256": _sha256(
            raw["scorer_source_root_sha256"], "registry scorer source root"
        ),
        "role_keys": tuple(_snapshot_role_key(item) for item in raw["role_keys"]),
    }


def _snapshot_registry(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1ExecutionRegistryCandidateV1:
        raise TypeError("execution registry candidate exact type drift")
    return {
        "body": _snapshot_registry_body(object.__getattribute__(value, "body")),
        "governance_signature": _snapshot_signature(
            object.__getattribute__(value, "governance_signature")
        ),
        "execution_registry_id": _identifier(
            object.__getattribute__(value, "execution_registry_id"),
            _REGISTRY_PREFIX,
            "execution registry ID",
        ),
    }


def _snapshot_store_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1StoreIfAbsentReceiptBodyV1:
        raise TypeError("STORE receipt body exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    previous = raw["previous_content_sha256_or_null"]
    if previous is not None:
        previous = _sha256(previous, "previous stored content SHA")
    return {
        "version": _ascii_text(raw["version"], "STORE receipt version"),
        "context_id": _identifier(raw["context_id"], _CONTEXT_PREFIX, "STORE context ID"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "STORE authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"],
            "phase2b_c1_container_actor_profile_v2_",
            "STORE actor profile content ID",
        ),
        "campaign_id": _identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "STORE campaign ID"),
        "attempt_id": _identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "STORE attempt ID"),
        "execution_registry_id": _identifier(
            raw["execution_registry_id"], _REGISTRY_PREFIX, "STORE execution registry ID"
        ),
        "object_key": _ascii_text(raw["object_key"], "STORE object key"),
        "previous_generation": _exact_integer(
            raw["previous_generation"], "previous generation", maximum=2**63 - 1
        ),
        "new_generation": _exact_integer(
            raw["new_generation"], "new generation", maximum=2**63 - 1
        ),
        "condition": _ascii_text(raw["condition"], "STORE condition"),
        "previous_content_sha256_or_null": previous,
        "stored_content_sha256": _sha256(raw["stored_content_sha256"], "stored content SHA"),
    }


def _snapshot_store_receipt(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1StoreIfAbsentReceiptCandidateV1:
        raise TypeError("STORE receipt candidate exact type drift")
    return {
        "body": _snapshot_store_body(object.__getattribute__(value, "body")),
        "store_signature": _snapshot_signature(object.__getattribute__(value, "store_signature")),
        "store_receipt_id": _identifier(
            object.__getattribute__(value, "store_receipt_id"),
            _STORE_RECEIPT_PREFIX,
            "STORE receipt ID",
        ),
    }


def _snapshot_time_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalGenesisTokenBodyV1:
        raise TypeError("logical-genesis body exact type drift")
    raw = {item.name: object.__getattribute__(value, item.name) for item in fields(type(value))}
    previous = raw["previous_logical_sequence_or_null"]
    if previous is not None:
        previous = _exact_integer(previous, "previous logical sequence", maximum=2**63 - 1)
    if type(raw["stage"]) is not Phase2BC1LogicalStageV1:
        raise TypeError("logical stage exact enum drift")
    return {
        "version": _ascii_text(raw["version"], "logical-genesis version"),
        "context_id": _identifier(raw["context_id"], _CONTEXT_PREFIX, "TIME context ID"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "TIME authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"],
            "phase2b_c1_container_actor_profile_v2_",
            "TIME actor profile content ID",
        ),
        "campaign_id": _identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "TIME campaign ID"),
        "attempt_id": _identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "TIME attempt ID"),
        "execution_registry_id": _identifier(
            raw["execution_registry_id"], _REGISTRY_PREFIX, "TIME execution registry ID"
        ),
        "store_receipt_id": _identifier(
            raw["store_receipt_id"], _STORE_RECEIPT_PREFIX, "TIME STORE receipt ID"
        ),
        "time_semantics": _ascii_text(raw["time_semantics"], "TIME semantics"),
        "previous_logical_sequence_or_null": previous,
        "logical_sequence": _exact_integer(
            raw["logical_sequence"], "logical sequence", maximum=2**63 - 1
        ),
        "stage": raw["stage"],
        "stage_index": _exact_integer(raw["stage_index"], "stage index", maximum=9),
    }


def _snapshot_time_token(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalGenesisTokenCandidateV1:
        raise TypeError("logical-genesis token candidate exact type drift")
    return {
        "body": _snapshot_time_body(object.__getattribute__(value, "body")),
        "time_signature": _snapshot_signature(object.__getattribute__(value, "time_signature")),
        "logical_genesis_token_id": _identifier(
            object.__getattribute__(value, "logical_genesis_token_id"),
            _TIME_TOKEN_PREFIX,
            "logical-genesis token ID",
        ),
    }


def _context_body_pairs(body: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple((name, body[name]) for name in _CONTEXT_BODY_FIELDS)


def _role_key_pairs(key: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return (
        ("purpose_id", key["purpose_id"]),
        ("role", key["role"].value),
        ("algorithm", key["algorithm"]),
        ("key_epoch", key["key_epoch"]),
        ("purpose_private", key["purpose_private"]),
        ("public_key_hex", key["public_key"].hex()),
        ("key_id", key["key_id"]),
    )


def _signature_pairs(signature: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return (
        ("purpose_id", signature["purpose_id"]),
        ("role", signature["role"].value),
        ("signature_purpose", signature["signature_purpose"].value),
        ("algorithm", signature["algorithm"]),
        ("key_epoch", signature["key_epoch"]),
        ("signer_key_id", signature["signer_key_id"]),
        ("signature_hex", signature["signature"].hex()),
    )


def _registry_body_pairs(body: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return (
        *((name, body[name]) for name in _REGISTRY_BODY_FIELDS[:-1]),
        ("role_keys", tuple(_role_key_pairs(key) for key in body["role_keys"])),
    )


def _store_body_pairs(body: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple((name, body[name]) for name in _STORE_BODY_FIELDS)


def _time_body_pairs(body: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, body[name].value if name == "stage" else body[name])
        for name in _TIME_BODY_FIELDS
    )


def _signature_header_pairs(
    *, purpose_id: int, role: Phase2BC1SixRoleV1,
    signature_purpose: Phase2BC1SignaturePurposeV1, signer_key_id: str,
) -> tuple[tuple[str, object], ...]:
    return (
        ("purpose_id", purpose_id),
        ("role", role.value),
        ("signature_purpose", signature_purpose.value),
        ("algorithm", "Ed25519"),
        ("key_epoch", PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH),
        ("signer_key_id", signer_key_id),
    )


def _attempt_id_from_values(values: dict[str, object]) -> str:
    pairs = tuple((name, values[name]) for name in _ATTEMPT_FIELDS)
    return _ATTEMPT_PREFIX + _digest(_ATTEMPT_DOMAIN, pairs)


def phase2b_c1_attempt_id_v1(
    *, authority_requirement_id: str, actor_profile_content_id: str,
    actor_profile_canonical_sha256: str, campaign_id: str, protocol_id: str,
    exact_freeze_id: str, execution_freeze_manifest_id: str,
    execution_freeze_manifest_sha256: str, attempt_nonce_sha256: str,
) -> str:
    """Content-address one attempt identity without performing an operation."""

    values = {
        "authority_requirement_id": _identifier(
            authority_requirement_id,
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            actor_profile_content_id,
            "phase2b_c1_container_actor_profile_v2_",
            "actor profile content ID",
        ),
        "actor_profile_canonical_sha256": _sha256(
            actor_profile_canonical_sha256, "actor profile canonical SHA"
        ),
        "campaign_id": _identifier(campaign_id, _CAMPAIGN_PREFIX, "campaign ID"),
        "protocol_id": _identifier(protocol_id, "phase2b_protocol_", "protocol ID"),
        "exact_freeze_id": _identifier(exact_freeze_id, "phase2b_exact_freeze_", "exact-freeze ID"),
        "execution_freeze_manifest_id": _identifier(
            execution_freeze_manifest_id, _EXECUTION_FREEZE_PREFIX, "execution-freeze ID"
        ),
        "execution_freeze_manifest_sha256": _sha256(
            execution_freeze_manifest_sha256, "execution-freeze SHA"
        ),
        "attempt_nonce_sha256": _sha256(attempt_nonce_sha256, "attempt nonce SHA"),
    }
    return _attempt_id_from_values(values)


def phase2b_c1_role_key_id_v1(
    *, purpose_id: int, role: Phase2BC1SixRoleV1, public_key: bytes
) -> str:
    """Content-address one declared purpose-private public key."""

    purpose = _exact_integer(purpose_id, "purpose ID", maximum=6)
    if type(role) is not Phase2BC1SixRoleV1:
        raise TypeError("role exact enum drift")
    raw = _raw_bytes(public_key, _RAW_PUBLIC_KEY_BYTES, "Ed25519 public key")
    pairs = (
        ("purpose_id", purpose),
        ("role", role.value),
        ("algorithm", "Ed25519"),
        ("key_epoch", PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH),
        ("purpose_private", True),
        ("public_key_hex", raw.hex()),
    )
    return _KEY_PREFIX + _digest(_KEY_DOMAIN, pairs)


def phase2b_c1_ceremony_context_id_v1(
    body: Phase2BC1CeremonyContextBodyV1,
) -> str:
    """Content-address a complete explicit ceremony context body."""

    closed = _snapshot_context_body(body)
    return _CONTEXT_PREFIX + _digest(_CONTEXT_DOMAIN, _context_body_pairs(closed))


def canonical_phase2b_c1_ceremony_context_bytes_v1(
    context: Phase2BC1CeremonyContextV1,
) -> bytes:
    """Return the exact ASCII wire for one context candidate."""

    closed = _snapshot_context(context)
    return _canonical_named_bytes((
        ("body", _context_body_pairs(closed["body"])),
        ("context_id", closed["context_id"]),
    ))


def _signature_preimage(
    domain: bytes, body_pairs: tuple[tuple[str, object], ...],
    *, purpose_id: int, role: Phase2BC1SixRoleV1,
    signature_purpose: Phase2BC1SignaturePurposeV1, signer_key_id: str,
) -> bytes:
    signer = _identifier(signer_key_id, _KEY_PREFIX, "signer key ID")
    return domain + _canonical_named_bytes((
        ("signature_header", _signature_header_pairs(
            purpose_id=purpose_id,
            role=role,
            signature_purpose=signature_purpose,
            signer_key_id=signer,
        )),
        ("body", body_pairs),
    ))


def phase2b_c1_governance_registry_signature_preimage_v1(
    body: Phase2BC1ExecutionRegistryBodyV1, *, signer_key_id: str
) -> bytes:
    """Return the governance-only registry signature preimage."""

    closed = _snapshot_registry_body(body)
    return _signature_preimage(
        _GOVERNANCE_SIGNATURE_DOMAIN,
        _registry_body_pairs(closed),
        purpose_id=1,
        role=Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signature_purpose=Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT,
        signer_key_id=signer_key_id,
    )


def phase2b_c1_execution_registry_id_v1(
    body: Phase2BC1ExecutionRegistryBodyV1,
    governance_signature: Phase2BC1DetachedSignatureV1,
) -> str:
    """Content-address a registry body and its detached governance signature."""

    closed_body = _snapshot_registry_body(body)
    closed_signature = _snapshot_signature(governance_signature)
    pairs = (
        ("body", _registry_body_pairs(closed_body)),
        ("governance_signature", _signature_pairs(closed_signature)),
    )
    return _REGISTRY_PREFIX + _digest(_REGISTRY_ID_DOMAIN, pairs)


def canonical_phase2b_c1_execution_registry_bytes_v1(
    candidate: Phase2BC1ExecutionRegistryCandidateV1,
) -> bytes:
    """Return exact ASCII bytes stored by the registry IF_ABSENT receipt."""

    closed = _snapshot_registry(candidate)
    return _canonical_registry_bytes(closed)


def _canonical_registry_bytes(closed: dict[str, object]) -> bytes:
    return _canonical_named_bytes((
        ("body", _registry_body_pairs(closed["body"])),
        ("governance_signature", _signature_pairs(closed["governance_signature"])),
        ("execution_registry_id", closed["execution_registry_id"]),
    ))


def phase2b_c1_store_receipt_signature_preimage_v1(
    body: Phase2BC1StoreIfAbsentReceiptBodyV1, *, signer_key_id: str
) -> bytes:
    """Return the STORE-purpose-only IF_ABSENT receipt signature preimage."""

    closed = _snapshot_store_body(body)
    return _signature_preimage(
        _STORE_SIGNATURE_DOMAIN,
        _store_body_pairs(closed),
        purpose_id=3,
        role=Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        signer_key_id=signer_key_id,
    )


def phase2b_c1_store_receipt_id_v1(
    body: Phase2BC1StoreIfAbsentReceiptBodyV1,
    store_signature: Phase2BC1DetachedSignatureV1,
) -> str:
    """Content-address a STORE body and its detached signature."""

    closed_body = _snapshot_store_body(body)
    closed_signature = _snapshot_signature(store_signature)
    pairs = (
        ("body", _store_body_pairs(closed_body)),
        ("store_signature", _signature_pairs(closed_signature)),
    )
    return _STORE_RECEIPT_PREFIX + _digest(_STORE_ID_DOMAIN, pairs)


def canonical_phase2b_c1_store_receipt_bytes_v1(
    candidate: Phase2BC1StoreIfAbsentReceiptCandidateV1,
) -> bytes:
    """Return exact ASCII bytes for one signed STORE receipt candidate."""

    closed = _snapshot_store_receipt(candidate)
    return _canonical_named_bytes((
        ("body", _store_body_pairs(closed["body"])),
        ("store_signature", _signature_pairs(closed["store_signature"])),
        ("store_receipt_id", closed["store_receipt_id"]),
    ))


def phase2b_c1_logical_genesis_signature_preimage_v1(
    body: Phase2BC1LogicalGenesisTokenBodyV1, *, signer_key_id: str
) -> bytes:
    """Return the TIME-purpose-only logical-genesis signature preimage."""

    closed = _snapshot_time_body(body)
    return _signature_preimage(
        _TIME_SIGNATURE_DOMAIN,
        _time_body_pairs(closed),
        purpose_id=4,
        role=Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        signer_key_id=signer_key_id,
    )


def phase2b_c1_logical_genesis_token_id_v1(
    body: Phase2BC1LogicalGenesisTokenBodyV1,
    time_signature: Phase2BC1DetachedSignatureV1,
) -> str:
    """Content-address a logical-genesis body and detached TIME signature."""

    closed_body = _snapshot_time_body(body)
    closed_signature = _snapshot_signature(time_signature)
    pairs = (
        ("body", _time_body_pairs(closed_body)),
        ("time_signature", _signature_pairs(closed_signature)),
    )
    return _TIME_TOKEN_PREFIX + _digest(_TIME_ID_DOMAIN, pairs)


def canonical_phase2b_c1_logical_genesis_token_bytes_v1(
    candidate: Phase2BC1LogicalGenesisTokenCandidateV1,
) -> bytes:
    """Return exact ASCII bytes for one signed logical-genesis candidate."""

    closed = _snapshot_time_token(candidate)
    return _canonical_named_bytes((
        ("body", _time_body_pairs(closed["body"])),
        ("time_signature", _signature_pairs(closed["time_signature"])),
        ("logical_genesis_token_id", closed["logical_genesis_token_id"]),
    ))


def _expected_signature_metadata(
    signature: dict[str, object], *, key: dict[str, object], purpose_id: int,
    role: Phase2BC1SixRoleV1,
    signature_purpose: Phase2BC1SignaturePurposeV1,
) -> None:
    if (
        signature["purpose_id"] != purpose_id
        or signature["role"] is not role
        or signature["signature_purpose"] is not signature_purpose
        or signature["algorithm"] != "Ed25519"
        or signature["key_epoch"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH
        or signature["signer_key_id"] != key["key_id"]
    ):
        raise ValueError("detached signature purpose/key metadata drift")


def _validate_context_content(closed: dict[str, object]) -> None:
    body = closed["body"]
    if (
        body["version"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION
        or body["authority_class"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS
        or body["authority_requirement_id"]
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
        or body["actor_profile_content_id"]
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
        or body["actor_profile_canonical_sha256"]
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
        or body["protocol_id"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID
        or body["exact_freeze_id"]
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID
    ):
        raise ValueError("ceremony context frozen identity drift")
    if body["scorer_source_root_sha256"] == _LEGACY_SCORER_SOURCE_ROOT_SHA256:
        raise ValueError("legacy scorer source root is explicitly rejected")
    if (
        body["scorer_source_root_sha256"]
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256
    ):
        raise ValueError("scorer source root differs from the final frozen root")
    expected_attempt = _attempt_id_from_values(body)
    if body["attempt_id"] != expected_attempt:
        raise ValueError("attempt content address drift")
    expected_context = _CONTEXT_PREFIX + _digest(
        _CONTEXT_DOMAIN, _context_body_pairs(body)
    )
    if closed["context_id"] != expected_context:
        raise ValueError("ceremony context content address drift")


def _validate_registry_content(
    context: dict[str, object], registry: dict[str, object]
) -> None:
    body = registry["body"]
    context_body = context["body"]
    if body["version"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION:
        raise ValueError("registry version drift")
    if body["context_id"] != context["context_id"]:
        raise ValueError("registry context binding drift")
    for name in _REGISTRY_CONTEXT_BINDING_FIELDS:
        if body[name] != context_body[name]:
            raise ValueError(f"registry explicit context binding drift: {name}")
    keys = body["role_keys"]
    if len(keys) != 6:
        raise ValueError("registry must contain exactly six role keys")
    for expected_purpose, (expected_role, key) in enumerate(
        zip(PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, keys, strict=True), start=1
    ):
        if (
            key["purpose_id"] != expected_purpose
            or key["role"] is not expected_role
            or key["algorithm"] != "Ed25519"
            or key["key_epoch"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH
            or key["purpose_private"] is not True
        ):
            raise ValueError("ordered purpose-private six-key layout drift")
        expected_key_id = phase2b_c1_role_key_id_v1(
            purpose_id=expected_purpose,
            role=expected_role,
            public_key=key["public_key"],
        )
        if key["key_id"] != expected_key_id:
            raise ValueError("role key content address drift")
    if len({key["public_key"] for key in keys}) != 6:
        raise ValueError("role public keys must be pairwise distinct")
    if len({key["key_id"] for key in keys}) != 6:
        raise ValueError("role key IDs must be pairwise distinct")
    _expected_signature_metadata(
        registry["governance_signature"],
        key=keys[0],
        purpose_id=1,
        role=Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signature_purpose=Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT,
    )
    expected_registry_id = _REGISTRY_PREFIX + _digest(
        _REGISTRY_ID_DOMAIN,
        (
            ("body", _registry_body_pairs(body)),
            ("governance_signature", _signature_pairs(registry["governance_signature"])),
        ),
    )
    if registry["execution_registry_id"] != expected_registry_id:
        raise ValueError("execution registry content address drift")


def _validate_store_content(
    context: dict[str, object], registry: dict[str, object],
    store_receipt: dict[str, object],
) -> None:
    body = store_receipt["body"]
    context_body = context["body"]
    expected_bindings = {
        "version": PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        "context_id": context["context_id"],
        "authority_requirement_id": context_body["authority_requirement_id"],
        "actor_profile_content_id": context_body["actor_profile_content_id"],
        "campaign_id": context_body["campaign_id"],
        "attempt_id": context_body["attempt_id"],
        "execution_registry_id": registry["execution_registry_id"],
    }
    if any(body[name] != value for name, value in expected_bindings.items()):
        raise ValueError("STORE receipt graph binding drift")
    expected_object_key = (
        "phase2b/c1/actual-unsealed-960/"
        f"{context_body['campaign_id']}/attempt-0/execution-registry"
    )
    if (
        body["object_key"] != expected_object_key
        or body["previous_generation"] != 0
        or body["new_generation"] != 1
        or body["condition"] != "IF_ABSENT"
        or body["previous_content_sha256_or_null"] is not None
    ):
        raise ValueError("STORE IF_ABSENT 0-to-1 transition drift")
    expected_stored_sha = hashlib.sha256(_canonical_registry_bytes(registry)).hexdigest()
    if body["stored_content_sha256"] != expected_stored_sha:
        raise ValueError("STORE receipt does not bind exact registry bytes")
    keys = registry["body"]["role_keys"]
    _expected_signature_metadata(
        store_receipt["store_signature"],
        key=keys[2],
        purpose_id=3,
        role=Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
    )
    expected_receipt_id = _STORE_RECEIPT_PREFIX + _digest(
        _STORE_ID_DOMAIN,
        (
            ("body", _store_body_pairs(body)),
            ("store_signature", _signature_pairs(store_receipt["store_signature"])),
        ),
    )
    if store_receipt["store_receipt_id"] != expected_receipt_id:
        raise ValueError("STORE receipt content address drift")


def _validate_time_content(
    context: dict[str, object], registry: dict[str, object],
    store_receipt: dict[str, object], time_token: dict[str, object],
) -> None:
    body = time_token["body"]
    context_body = context["body"]
    expected_bindings = {
        "version": PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        "context_id": context["context_id"],
        "authority_requirement_id": context_body["authority_requirement_id"],
        "actor_profile_content_id": context_body["actor_profile_content_id"],
        "campaign_id": context_body["campaign_id"],
        "attempt_id": context_body["attempt_id"],
        "execution_registry_id": registry["execution_registry_id"],
        "store_receipt_id": store_receipt["store_receipt_id"],
    }
    if any(body[name] != value for name, value in expected_bindings.items()):
        raise ValueError("logical-genesis graph binding drift")
    if (
        body["time_semantics"]
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS
        or body["previous_logical_sequence_or_null"] is not None
        or body["logical_sequence"] != 0
        or body["stage"] is not Phase2BC1LogicalStageV1.PREREGISTERED
        or body["stage_index"] != 0
    ):
        raise ValueError("signed logical-genesis shape drift")
    keys = registry["body"]["role_keys"]
    _expected_signature_metadata(
        time_token["time_signature"],
        key=keys[3],
        purpose_id=4,
        role=Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
    )
    expected_token_id = _TIME_TOKEN_PREFIX + _digest(
        _TIME_ID_DOMAIN,
        (
            ("body", _time_body_pairs(body)),
            ("time_signature", _signature_pairs(time_token["time_signature"])),
        ),
    )
    if time_token["logical_genesis_token_id"] != expected_token_id:
        raise ValueError("logical-genesis token content address drift")


class _BackendUnavailable(Exception):
    pass


class _SignatureInvalid(Exception):
    pass


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    if _Ed25519PublicKey is None:
        raise _BackendUnavailable
    try:
        _Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except Exception as exc:
        raise _SignatureInvalid from exc


def _issue_result(
    *, disposition: Phase2BC1SixRoleWireDispositionV1,
    reason: Phase2BC1SixRoleWireReasonV1,
    context: dict[str, object] | None = None,
    registry: dict[str, object] | None = None,
    store_receipt: dict[str, object] | None = None,
    time_token: dict[str, object] | None = None,
) -> Phase2BC1SixRoleWireVerificationV1:
    success = (
        disposition
        is Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
    )
    if success and any(
        item is None for item in (context, registry, store_receipt, time_token)
    ):
        raise RuntimeError("six-role success issuance lacks a complete closed graph")
    result_id: str | None = None
    role_key_ids: tuple[str, ...] = ()
    context_id = attempt_id = registry_id = store_id = time_id = None
    if success:
        assert context is not None and registry is not None
        assert store_receipt is not None and time_token is not None
        context_id = context["context_id"]
        attempt_id = context["body"]["attempt_id"]
        registry_id = registry["execution_registry_id"]
        store_id = store_receipt["store_receipt_id"]
        time_id = time_token["logical_genesis_token_id"]
        role_key_ids = tuple(
            key["key_id"] for key in registry["body"]["role_keys"]
        )
        result_pairs = (
            ("version", PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION),
            ("claim_level", PHASE2B_C1_SIX_ROLE_WIRE_V1_CLAIM_LEVEL),
            ("context_id", context_id),
            ("attempt_id", attempt_id),
            ("execution_registry_id", registry_id),
            ("store_receipt_id", store_id),
            ("logical_genesis_token_id", time_id),
            ("role_key_ids", role_key_ids),
        )
        result_id = _RESULT_PREFIX + _digest(_RESULT_ID_DOMAIN, result_pairs)
    result = Phase2BC1SixRoleWireVerificationV1(_RESULT_ISSUANCE_TOKEN)
    values: list[tuple[str, object]] = [
        ("disposition", disposition),
        ("reason", reason),
        ("version", PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION),
        ("claim_level", PHASE2B_C1_SIX_ROLE_WIRE_V1_CLAIM_LEVEL),
        ("result_id", result_id),
        ("context_id", context_id),
        ("attempt_id", attempt_id),
        ("execution_registry_id", registry_id),
        ("store_receipt_id", store_id),
        ("logical_genesis_token_id", time_id),
        ("role_key_ids", role_key_ids),
        ("partial_output_published", False),
    ]
    values.extend((name, success) for name in _TRUE_CLAIMS)
    values.extend((name, False) for name in _FALSE_CLAIMS)
    expected_fields = tuple(item.name for item in fields(Phase2BC1SixRoleWireVerificationV1))
    if tuple(name for name, _ in values) != expected_fields:
        raise RuntimeError("six-role result issuance field drift")
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def _reject(
    reason: Phase2BC1SixRoleWireReasonV1,
) -> Phase2BC1SixRoleWireVerificationV1:
    return _issue_result(
        disposition=Phase2BC1SixRoleWireDispositionV1.REJECTED,
        reason=reason,
    )


_ORDINARY_INPUT_ERRORS: Final = (
    AttributeError,
    TypeError,
    ValueError,
    OverflowError,
    UnicodeError,
)


def verify_phase2b_c1_six_role_wire_v1(
    *, context: Phase2BC1CeremonyContextV1,
    registry: Phase2BC1ExecutionRegistryCandidateV1,
    store_receipt: Phase2BC1StoreIfAbsentReceiptCandidateV1,
    logical_genesis_token: Phase2BC1LogicalGenesisTokenCandidateV1,
) -> Phase2BC1SixRoleWireVerificationV1:
    """Verify one caller-supplied graph without performing its live ceremony."""

    if (
        type(context) is not Phase2BC1CeremonyContextV1
        or type(registry) is not Phase2BC1ExecutionRegistryCandidateV1
        or type(store_receipt) is not Phase2BC1StoreIfAbsentReceiptCandidateV1
        or type(logical_genesis_token) is not Phase2BC1LogicalGenesisTokenCandidateV1
    ):
        return _reject(Phase2BC1SixRoleWireReasonV1.WRONG_INPUT_TYPE)
    try:
        # Copy the entire four-artifact graph into closed primitives before any
        # content hash or signature verification.  No input object is read again.
        try:
            closed_context = _snapshot_context(context)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.CONTEXT_INVALID)
        try:
            closed_registry = _snapshot_registry(registry)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.REGISTRY_INVALID)
        try:
            closed_store = _snapshot_store_receipt(store_receipt)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.STORE_RECEIPT_INVALID)
        try:
            closed_time = _snapshot_time_token(logical_genesis_token)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.LOGICAL_GENESIS_INVALID)

        # Validate every exact value, cross-binding, and content address before
        # invoking the cryptographic backend even once.
        try:
            _validate_context_content(closed_context)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.CONTEXT_INVALID)
        try:
            _validate_registry_content(closed_context, closed_registry)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.REGISTRY_INVALID)
        try:
            _validate_store_content(closed_context, closed_registry, closed_store)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.STORE_RECEIPT_INVALID)
        try:
            _validate_time_content(
                closed_context, closed_registry, closed_store, closed_time
            )
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1SixRoleWireReasonV1.LOGICAL_GENESIS_INVALID)

        keys = closed_registry["body"]["role_keys"]
        governance_preimage = _signature_preimage(
            _GOVERNANCE_SIGNATURE_DOMAIN,
            _registry_body_pairs(closed_registry["body"]),
            purpose_id=1,
            role=Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
            signature_purpose=Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT,
            signer_key_id=keys[0]["key_id"],
        )
        store_preimage = _signature_preimage(
            _STORE_SIGNATURE_DOMAIN,
            _store_body_pairs(closed_store["body"]),
            purpose_id=3,
            role=Phase2BC1SixRoleV1.STORE_RECEIPT,
            signature_purpose=Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
            signer_key_id=keys[2]["key_id"],
        )
        time_preimage = _signature_preimage(
            _TIME_SIGNATURE_DOMAIN,
            _time_body_pairs(closed_time["body"]),
            purpose_id=4,
            role=Phase2BC1SixRoleV1.TIME_ATTESTOR,
            signature_purpose=Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
            signer_key_id=keys[3]["key_id"],
        )
        try:
            _verify_ed25519(
                keys[0]["public_key"],
                closed_registry["governance_signature"]["signature"],
                governance_preimage,
            )
            _verify_ed25519(
                keys[2]["public_key"],
                closed_store["store_signature"]["signature"],
                store_preimage,
            )
            _verify_ed25519(
                keys[3]["public_key"],
                closed_time["time_signature"]["signature"],
                time_preimage,
            )
        except _BackendUnavailable:
            return _reject(
                Phase2BC1SixRoleWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE
            )
        except _SignatureInvalid:
            return _reject(Phase2BC1SixRoleWireReasonV1.SIGNATURE_INVALID)
        return _issue_result(
            disposition=(
                Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
            ),
            reason=(
                Phase2BC1SixRoleWireReasonV1.CALLER_SUPPLIED_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED
            ),
            context=closed_context,
            registry=closed_registry,
            store_receipt=closed_store,
            time_token=closed_time,
        )
    except _BackendUnavailable:
        return _reject(Phase2BC1SixRoleWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE)
    except _SignatureInvalid:
        return _reject(Phase2BC1SixRoleWireReasonV1.SIGNATURE_INVALID)
    except _ORDINARY_INPUT_ERRORS:
        return _reject(Phase2BC1SixRoleWireReasonV1.GRAPH_BINDING_INVALID)
    except Exception:
        return _reject(Phase2BC1SixRoleWireReasonV1.INTERNAL_ERROR)


_CONTEXT_BODY_FIELDS: Final = tuple(
    item.name for item in fields(Phase2BC1CeremonyContextBodyV1)
)
_REGISTRY_BODY_FIELDS: Final = tuple(
    item.name for item in fields(Phase2BC1ExecutionRegistryBodyV1)
)
_STORE_BODY_FIELDS: Final = tuple(
    item.name for item in fields(Phase2BC1StoreIfAbsentReceiptBodyV1)
)
_TIME_BODY_FIELDS: Final = tuple(
    item.name for item in fields(Phase2BC1LogicalGenesisTokenBodyV1)
)
_ATTEMPT_FIELDS: Final = (
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "execution_freeze_manifest_sha256",
    "attempt_nonce_sha256",
)
_REGISTRY_CONTEXT_BINDING_FIELDS: Final = (
    "authority_class",
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "execution_freeze_manifest_sha256",
    "attempt_nonce_sha256",
    "attempt_id",
    "recognizer_image_digest",
    "recognizer_source_root_sha256",
    "scorer_image_digest",
    "scorer_source_root_sha256",
)

_ALL_DOMAINS: Final = (
    _ATTEMPT_DOMAIN,
    _CONTEXT_DOMAIN,
    _KEY_DOMAIN,
    _GOVERNANCE_SIGNATURE_DOMAIN,
    _REGISTRY_ID_DOMAIN,
    _STORE_SIGNATURE_DOMAIN,
    _STORE_ID_DOMAIN,
    _TIME_SIGNATURE_DOMAIN,
    _TIME_ID_DOMAIN,
    _RESULT_ID_DOMAIN,
)
if len(set(_ALL_DOMAINS)) != len(_ALL_DOMAINS) or any(
    not domain.endswith(b"\x00") for domain in _ALL_DOMAINS
):
    raise RuntimeError("six-role wire domain separation drift")
if len(PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER) != 6 or len(
    set(PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER)
) != 6:
    raise RuntimeError("six-role wire role order drift")
if len(set(_TRUE_CLAIMS)) != len(_TRUE_CLAIMS):
    raise RuntimeError("six-role wire true-claim duplicate")
if len(set(_FALSE_CLAIMS)) != len(_FALSE_CLAIMS):
    raise RuntimeError("six-role wire false-claim duplicate")
if set(_TRUE_CLAIMS) & set(_FALSE_CLAIMS):
    raise RuntimeError("six-role wire claim polarity overlap")


__all__ = (
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_CLAIM_LEVEL",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ED25519_BACKEND_AVAILABLE",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_STORE_TRANSITION",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER",
    "Phase2BC1SixRoleV1",
    "Phase2BC1SignaturePurposeV1",
    "Phase2BC1LogicalStageV1",
    "Phase2BC1SixRoleWireDispositionV1",
    "Phase2BC1SixRoleWireReasonV1",
    "Phase2BC1CeremonyContextBodyV1",
    "Phase2BC1CeremonyContextV1",
    "Phase2BC1RoleKeyV1",
    "Phase2BC1DetachedSignatureV1",
    "Phase2BC1ExecutionRegistryBodyV1",
    "Phase2BC1ExecutionRegistryCandidateV1",
    "Phase2BC1StoreIfAbsentReceiptBodyV1",
    "Phase2BC1StoreIfAbsentReceiptCandidateV1",
    "Phase2BC1LogicalGenesisTokenBodyV1",
    "Phase2BC1LogicalGenesisTokenCandidateV1",
    "Phase2BC1SixRoleWireVerificationV1",
    "phase2b_c1_attempt_id_v1",
    "phase2b_c1_role_key_id_v1",
    "phase2b_c1_ceremony_context_id_v1",
    "canonical_phase2b_c1_ceremony_context_bytes_v1",
    "phase2b_c1_governance_registry_signature_preimage_v1",
    "phase2b_c1_execution_registry_id_v1",
    "canonical_phase2b_c1_execution_registry_bytes_v1",
    "phase2b_c1_store_receipt_signature_preimage_v1",
    "phase2b_c1_store_receipt_id_v1",
    "canonical_phase2b_c1_store_receipt_bytes_v1",
    "phase2b_c1_logical_genesis_signature_preimage_v1",
    "phase2b_c1_logical_genesis_token_id_v1",
    "canonical_phase2b_c1_logical_genesis_token_bytes_v1",
    "verify_phase2b_c1_six_role_wire_v1",
)
