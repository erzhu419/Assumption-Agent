"""Pure companion wire mechanics for Phase-2B C1 logical stages 1 through 4.

The frozen six-role wire owns stage 0 (``PREREGISTERED``).  This module first
re-verifies that complete caller-supplied graph and then verifies a contiguous
prefix ending at ``RUN_FINISHED``.  Every event projection and TIME token is
content addressed and every token is signed by registry purpose 4.

This is deliberately a conservative *companion*, not an amendment to the
frozen actor profile or execution registry.  Its explicit authorization fields
therefore remain false.  Successful verification proves only caller-supplied
cryptographic and content-address mechanics; it is not authority for an actual
campaign and observes no execution, filesystem, Docker, clock, one-shot,
scoring, gate, effect, or C1 fact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final, TypeAlias

from hegel_machine import phase2b_c1_six_role_wire_v1 as six

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by backend monkeypatch tests.
    _Ed25519PublicKey = None


PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION: Final = (
    "hegel-machine-phase2b-c1-logical-stage-wire/1"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_CLAIM_LEVEL: Final = (
    "CALLER_SUPPLIED_LOGICAL_STAGE_PREFIX_CRYPTOGRAPHIC_MECHANICS_ONLY"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_LEGACY_POLICY_AUTHORIZATION: Final = (
    "NOT_AUTHORIZED_BY_FROZEN_ACTOR_PROFILE_V2_OR_EXECUTION_REGISTRY_V1"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_ACTUAL_CAMPAIGN_ELIGIBLE: Final = False
PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_ED25519_BACKEND_AVAILABLE: Final = (
    _Ed25519PublicKey is not None
)

_MAXIMUM_ASCII_TEXT_BYTES: Final = 4_096
_RAW_SIGNATURE_BYTES: Final = 64

_STAGE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/STAGE_SIGNATURE/V1\x00"
)
_STAGE_TOKEN_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/STAGE_TOKEN_ID/V1\x00"
)
_RESULT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/VERIFICATION_RESULT/V1\x00"
)
_EVENT_DOMAINS: Final = {
    six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED: (
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/EVENT/RAW_INPUT_COMMITTED/V1\x00"
    ),
    six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN: (
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/EVENT/PACKAGE_COMMITMENTS_FROZEN/V1\x00"
    ),
    six.Phase2BC1LogicalStageV1.RUN_STARTED: (
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/EVENT/RUN_STARTED/V1\x00"
    ),
    six.Phase2BC1LogicalStageV1.RUN_FINISHED: (
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/EVENT/RUN_FINISHED/V1\x00"
    ),
}

_STAGE_TOKEN_PREFIX: Final = "phase2b_c1_signed_logical_stage_token_v1_"
_RESULT_PREFIX: Final = "phase2b_c1_logical_stage_wire_verification_v1_"
_CONTEXT_PREFIX: Final = "phase2b_c1_six_role_ceremony_context_v1_"
_AUTHORITY_REQUIREMENT_PREFIX: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
)
_PROFILE_PREFIX: Final = "phase2b_c1_container_actor_profile_v2_"
_CAMPAIGN_PREFIX: Final = "phase2b_actual_unsealed_960_campaign_v2_"
_ATTEMPT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_v2_"
_REGISTRY_PREFIX: Final = "phase2b_c1_owner_accepted_execution_registry_v1_"
_GENESIS_PREFIX: Final = "phase2b_c1_signed_logical_genesis_token_v1_"
_KEY_PREFIX: Final = "phase2b_c1_six_role_key_v1_"
_INPUT_ARCHIVE_PREFIX: Final = "phase2b_recognizer_input_archive_v2_"
_EXECUTION_FREEZE_PREFIX: Final = "phase2b_execution_freeze_"
_RUN_SPEC_PREFIX: Final = "phase2b_oci_run_spec_"
_RUNTIME_RECEIPT_PREFIX: Final = "phase2b_recognizer_runtime_receipt_v2_"
_LOCAL_EXECUTION_RECEIPT_PREFIX: Final = (
    "phase2b_recognizer_local_execution_receipt_v3_"
)
_PREDICTION_ARCHIVE_PREFIX: Final = "phase2b_recognizer_prediction_archive_v2_"
_RAW_COMMITMENT_RECEIPT_PREFIX: Final = (
    "phase2b_c1_raw_input_commitment_receipt_v1_"
)
_PUBLIC_PARTITION_SPEC_PREFIX: Final = "phase2b_c1_public_partition_spec_v1_"
_PACKAGE_MANIFEST_PREFIX: Final = "phase2b_c1_package_commitment_manifest_v1_"

_PROJECTION_PREFIXES: Final = {
    six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED: (
        "phase2b_c1_raw_input_committed_event_projection_v1_"
    ),
    six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN: (
        "phase2b_c1_package_commitments_frozen_event_projection_v1_"
    ),
    six.Phase2BC1LogicalStageV1.RUN_STARTED: (
        "phase2b_c1_run_started_event_projection_v1_"
    ),
    six.Phase2BC1LogicalStageV1.RUN_FINISHED: (
        "phase2b_c1_run_finished_event_projection_v1_"
    ),
}

_PREFIX_STAGES: Final = (
    six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED,
    six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN,
    six.Phase2BC1LogicalStageV1.RUN_STARTED,
    six.Phase2BC1LogicalStageV1.RUN_FINISHED,
)


class Phase2BC1LogicalStageWireDispositionV1(str, Enum):
    CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY = (
        "CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY"
    )
    REJECTED = "REJECTED"


class Phase2BC1LogicalStageWireReasonV1(str, Enum):
    CALLER_SUPPLIED_PREFIX_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED = (
        "CALLER_SUPPLIED_PREFIX_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    PREFIX_LENGTH_INVALID = "PREFIX_LENGTH_INVALID"
    GRAPH_SNAPSHOT_INVALID = "GRAPH_SNAPSHOT_INVALID"
    GENESIS_REVERIFICATION_REJECTED = "GENESIS_REVERIFICATION_REJECTED"
    EVENT_PROJECTION_INVALID = "EVENT_PROJECTION_INVALID"
    STAGE_TOKEN_INVALID = "STAGE_TOKEN_INVALID"
    GRAPH_BINDING_INVALID = "GRAPH_BINDING_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class Phase2BC1RawInputCommittedEventProjectionV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    input_archive_id: str
    input_archive_sha256: str
    input_archive_bytes: int
    raw_input_commitment_sha256: str
    raw_input_commitment_receipt_id: str

    def __post_init__(self) -> None:
        _snapshot_projection(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1PackageCommitmentsFrozenEventProjectionV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    raw_input_committed_token_id: str
    execution_freeze_manifest_id: str
    execution_freeze_manifest_sha256: str
    input_archive_id: str
    input_archive_sha256: str
    recognizer_run_spec_id: str
    answer_commitment_sha256: str
    gate_commitment_sha256: str
    public_partition_spec_id: str
    public_partition_spec_sha256: str
    package_commitment_manifest_id: str

    def __post_init__(self) -> None:
        _snapshot_projection(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1RunStartedEventProjectionV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    package_commitments_frozen_token_id: str
    recognizer_run_spec_id: str
    execution_freeze_manifest_id: str
    input_archive_id: str
    start_operation_nonce_sha256: str

    def __post_init__(self) -> None:
        _snapshot_projection(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1RunFinishedEventProjectionV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    run_started_token_id: str
    recognizer_run_spec_id: str
    execution_freeze_manifest_id: str
    execution_freeze_manifest_sha256: str
    input_archive_id: str
    input_archive_sha256: str
    runtime_receipt_id: str
    local_execution_receipt_id: str
    prediction_archive_id: str
    prediction_archive_sha256: str
    prediction_archive_bytes: int

    def __post_init__(self) -> None:
        _snapshot_projection(self)


Phase2BC1LogicalStageEventProjectionV1: TypeAlias = (
    Phase2BC1RawInputCommittedEventProjectionV1
    | Phase2BC1PackageCommitmentsFrozenEventProjectionV1
    | Phase2BC1RunStartedEventProjectionV1
    | Phase2BC1RunFinishedEventProjectionV1
)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageTokenBodyV1:
    version: str
    context_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    logical_genesis_token_id: str
    legacy_registry_policy_authorization: str
    actual_campaign_eligible: bool
    time_semantics: str
    predecessor_token_id: str
    previous_logical_sequence: int
    logical_sequence: int
    previous_stage: six.Phase2BC1LogicalStageV1
    stage: six.Phase2BC1LogicalStageV1
    stage_index: int
    event_projection_id: str
    event_root_sha256: str

    def __post_init__(self) -> None:
        _snapshot_token_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageTokenCandidateV1:
    body: Phase2BC1LogicalStageTokenBodyV1
    time_signature: six.Phase2BC1DetachedSignatureV1
    logical_stage_token_id: str

    def __post_init__(self) -> None:
        _snapshot_token(self)


_TRUE_CLAIMS: Final = (
    "complete_caller_graph_snapshotted_before_verification",
    "frozen_six_role_genesis_reverified",
    "exact_event_projection_types_verified",
    "event_projection_content_addresses_verified",
    "stage_token_content_addresses_verified",
    "purpose_4_time_attestor_signatures_verified",
    "sequence_stage_adjacency_verified",
    "campaign_attempt_profile_registry_bindings_verified",
    "content_address_dag_verified",
    "verification_result_privately_issued",
)

_FALSE_CLAIMS: Final = (
    "legacy_registry_policy_authorized",
    "actual_campaign_eligible",
    "actual_execution_claimed",
    "execution_authorized",
    "owner_acceptance_observed",
    "holdout_material_observed",
    "docker_invoked",
    "filesystem_read",
    "filesystem_written",
    "clock_read",
    "keys_generated",
    "signatures_generated",
    "time_operation_executed",
    "authoritative_time_verified",
    "utc_time_verified",
    "timeline_observed",
    "run_start_observed",
    "run_finish_observed",
    "recognizer_executed",
    "runtime_executed",
    "local_execution_receipt_is_purpose_5_receipt",
    "raw_input_commitment_authority_verified",
    "package_commitment_authority_verified",
    "store_operation_executed",
    "store_durability_verified",
    "one_shot_policy_enforced",
    "cross_host_global_one_shot_verified",
    "scoring_performed",
    "prediction_scored",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1LogicalStageWireVerificationV1:
    disposition: Phase2BC1LogicalStageWireDispositionV1
    reason: Phase2BC1LogicalStageWireReasonV1
    version: str
    claim_level: str
    result_id: str | None
    context_id: str | None
    attempt_id: str | None
    execution_registry_id: str | None
    logical_genesis_token_id: str | None
    terminal_stage: six.Phase2BC1LogicalStageV1 | None
    terminal_logical_sequence: int | None
    stage_token_ids: tuple[str, ...]
    event_projection_ids: tuple[str, ...]
    partial_output_published: bool
    complete_caller_graph_snapshotted_before_verification: bool
    frozen_six_role_genesis_reverified: bool
    exact_event_projection_types_verified: bool
    event_projection_content_addresses_verified: bool
    stage_token_content_addresses_verified: bool
    purpose_4_time_attestor_signatures_verified: bool
    sequence_stage_adjacency_verified: bool
    campaign_attempt_profile_registry_bindings_verified: bool
    content_address_dag_verified: bool
    verification_result_privately_issued: bool
    legacy_registry_policy_authorized: bool
    actual_campaign_eligible: bool
    actual_execution_claimed: bool
    execution_authorized: bool
    owner_acceptance_observed: bool
    holdout_material_observed: bool
    docker_invoked: bool
    filesystem_read: bool
    filesystem_written: bool
    clock_read: bool
    keys_generated: bool
    signatures_generated: bool
    time_operation_executed: bool
    authoritative_time_verified: bool
    utc_time_verified: bool
    timeline_observed: bool
    run_start_observed: bool
    run_finish_observed: bool
    recognizer_executed: bool
    runtime_executed: bool
    local_execution_receipt_is_purpose_5_receipt: bool
    raw_input_commitment_authority_verified: bool
    package_commitment_authority_verified: bool
    store_operation_executed: bool
    store_durability_verified: bool
    one_shot_policy_enforced: bool
    cross_host_global_one_shot_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __new__(
        cls, token: object = None
    ) -> "Phase2BC1LogicalStageWireVerificationV1":
        if token is not _RESULT_ISSUANCE_TOKEN:
            raise TypeError("logical-stage verification results are privately issued")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _RESULT_ISSUANCE_TOKEN:
            raise TypeError("logical-stage verification results are privately issued")


_RESULT_ISSUANCE_TOKEN: Final = object()


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


def _exact_integer(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} integer drift")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _canonical_named_bytes(pairs: tuple[tuple[str, object], ...]) -> bytes:
    return json.dumps(
        pairs, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + _canonical_named_bytes(pairs)).hexdigest()


def _common_projection_snapshot(raw: dict[str, object]) -> dict[str, object]:
    return {
        "version": _ascii_text(raw["version"], "projection version"),
        "context_id": _identifier(raw["context_id"], _CONTEXT_PREFIX, "context ID"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            _AUTHORITY_REQUIREMENT_PREFIX,
            "authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"], _PROFILE_PREFIX, "actor profile ID"
        ),
        "campaign_id": _identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "campaign ID"),
        "attempt_id": _identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "attempt ID"),
        "execution_registry_id": _identifier(
            raw["execution_registry_id"], _REGISTRY_PREFIX, "execution registry ID"
        ),
    }


def _raw_fields(value: object) -> dict[str, object]:
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(type(value))
    }


def _snapshot_projection(
    value: object,
) -> tuple[six.Phase2BC1LogicalStageV1, dict[str, object]]:
    exact_type = type(value)
    raw = _raw_fields(value)
    closed = _common_projection_snapshot(raw)
    if exact_type is Phase2BC1RawInputCommittedEventProjectionV1:
        stage = six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED
        closed.update({
            "input_archive_id": _identifier(raw["input_archive_id"], _INPUT_ARCHIVE_PREFIX, "input archive ID"),
            "input_archive_sha256": _sha256(raw["input_archive_sha256"], "input archive SHA"),
            "input_archive_bytes": _exact_integer(raw["input_archive_bytes"], "input archive bytes", maximum=2**63 - 1),
            "raw_input_commitment_sha256": _sha256(raw["raw_input_commitment_sha256"], "raw-input commitment SHA"),
            "raw_input_commitment_receipt_id": _identifier(raw["raw_input_commitment_receipt_id"], _RAW_COMMITMENT_RECEIPT_PREFIX, "raw-input commitment receipt ID"),
        })
    elif exact_type is Phase2BC1PackageCommitmentsFrozenEventProjectionV1:
        stage = six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN
        closed.update({
            "raw_input_committed_token_id": _identifier(raw["raw_input_committed_token_id"], _STAGE_TOKEN_PREFIX, "raw-input token ID"),
            "execution_freeze_manifest_id": _identifier(raw["execution_freeze_manifest_id"], _EXECUTION_FREEZE_PREFIX, "execution-freeze ID"),
            "execution_freeze_manifest_sha256": _sha256(raw["execution_freeze_manifest_sha256"], "execution-freeze SHA"),
            "input_archive_id": _identifier(raw["input_archive_id"], _INPUT_ARCHIVE_PREFIX, "input archive ID"),
            "input_archive_sha256": _sha256(raw["input_archive_sha256"], "input archive SHA"),
            "recognizer_run_spec_id": _identifier(raw["recognizer_run_spec_id"], _RUN_SPEC_PREFIX, "recognizer run-spec ID"),
            "answer_commitment_sha256": _sha256(raw["answer_commitment_sha256"], "answer commitment SHA"),
            "gate_commitment_sha256": _sha256(raw["gate_commitment_sha256"], "gate commitment SHA"),
            "public_partition_spec_id": _identifier(raw["public_partition_spec_id"], _PUBLIC_PARTITION_SPEC_PREFIX, "public partition spec ID"),
            "public_partition_spec_sha256": _sha256(raw["public_partition_spec_sha256"], "public partition spec SHA"),
            "package_commitment_manifest_id": _identifier(raw["package_commitment_manifest_id"], _PACKAGE_MANIFEST_PREFIX, "package commitment manifest ID"),
        })
    elif exact_type is Phase2BC1RunStartedEventProjectionV1:
        stage = six.Phase2BC1LogicalStageV1.RUN_STARTED
        closed.update({
            "package_commitments_frozen_token_id": _identifier(raw["package_commitments_frozen_token_id"], _STAGE_TOKEN_PREFIX, "package token ID"),
            "recognizer_run_spec_id": _identifier(raw["recognizer_run_spec_id"], _RUN_SPEC_PREFIX, "recognizer run-spec ID"),
            "execution_freeze_manifest_id": _identifier(raw["execution_freeze_manifest_id"], _EXECUTION_FREEZE_PREFIX, "execution-freeze ID"),
            "input_archive_id": _identifier(raw["input_archive_id"], _INPUT_ARCHIVE_PREFIX, "input archive ID"),
            "start_operation_nonce_sha256": _sha256(raw["start_operation_nonce_sha256"], "start-operation nonce SHA"),
        })
    elif exact_type is Phase2BC1RunFinishedEventProjectionV1:
        stage = six.Phase2BC1LogicalStageV1.RUN_FINISHED
        closed.update({
            "run_started_token_id": _identifier(raw["run_started_token_id"], _STAGE_TOKEN_PREFIX, "run-started token ID"),
            "recognizer_run_spec_id": _identifier(raw["recognizer_run_spec_id"], _RUN_SPEC_PREFIX, "recognizer run-spec ID"),
            "execution_freeze_manifest_id": _identifier(raw["execution_freeze_manifest_id"], _EXECUTION_FREEZE_PREFIX, "execution-freeze ID"),
            "execution_freeze_manifest_sha256": _sha256(raw["execution_freeze_manifest_sha256"], "execution-freeze SHA"),
            "input_archive_id": _identifier(raw["input_archive_id"], _INPUT_ARCHIVE_PREFIX, "input archive ID"),
            "input_archive_sha256": _sha256(raw["input_archive_sha256"], "input archive SHA"),
            "runtime_receipt_id": _identifier(raw["runtime_receipt_id"], _RUNTIME_RECEIPT_PREFIX, "runtime receipt ID"),
            "local_execution_receipt_id": _identifier(raw["local_execution_receipt_id"], _LOCAL_EXECUTION_RECEIPT_PREFIX, "local execution receipt ID"),
            "prediction_archive_id": _identifier(raw["prediction_archive_id"], _PREDICTION_ARCHIVE_PREFIX, "prediction archive ID"),
            "prediction_archive_sha256": _sha256(raw["prediction_archive_sha256"], "prediction archive SHA"),
            "prediction_archive_bytes": _exact_integer(raw["prediction_archive_bytes"], "prediction archive bytes", maximum=2**63 - 1),
        })
    else:
        raise TypeError("event projection exact type drift")
    return stage, closed


def _projection_pairs(value: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple((name, item) for name, item in value.items() if name != "event_projection_id")


def phase2b_c1_logical_stage_event_projection_id_v1(
    projection: Phase2BC1LogicalStageEventProjectionV1,
) -> str:
    stage, closed = _snapshot_projection(projection)
    return _PROJECTION_PREFIXES[stage] + _digest(
        _EVENT_DOMAINS[stage], _projection_pairs(closed)
    )


def phase2b_c1_logical_stage_event_root_sha256_v1(
    projection: Phase2BC1LogicalStageEventProjectionV1,
) -> str:
    stage, closed = _snapshot_projection(projection)
    return _digest(_EVENT_DOMAINS[stage], _projection_pairs(closed))


def canonical_phase2b_c1_logical_stage_event_projection_bytes_v1(
    projection: Phase2BC1LogicalStageEventProjectionV1,
) -> bytes:
    stage, closed = _snapshot_projection(projection)
    projection_id, _ = _projection_id_and_root(stage, closed)
    return _canonical_named_bytes((
        ("stage", stage.value),
        ("body", _projection_pairs(closed)),
        ("event_projection_id", projection_id),
    ))


def _snapshot_token_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageTokenBodyV1:
        raise TypeError("logical-stage body exact type drift")
    raw = _raw_fields(value)
    if type(raw["previous_stage"]) is not six.Phase2BC1LogicalStageV1:
        raise TypeError("previous stage exact enum drift")
    if type(raw["stage"]) is not six.Phase2BC1LogicalStageV1:
        raise TypeError("stage exact enum drift")
    predecessor = _ascii_text(raw["predecessor_token_id"], "predecessor token ID")
    if not (
        predecessor.startswith(_GENESIS_PREFIX)
        or predecessor.startswith(_STAGE_TOKEN_PREFIX)
    ):
        raise ValueError("predecessor token ID prefix drift")
    _sha256(predecessor.rsplit("_", 1)[-1], "predecessor token ID suffix")
    projection_id = _ascii_text(raw["event_projection_id"], "event projection ID")
    if not any(projection_id.startswith(prefix) for prefix in _PROJECTION_PREFIXES.values()):
        raise ValueError("event projection ID prefix drift")
    _sha256(projection_id.rsplit("_", 1)[-1], "event projection ID suffix")
    return {
        "version": _ascii_text(raw["version"], "logical-stage version"),
        "context_id": _identifier(raw["context_id"], _CONTEXT_PREFIX, "TIME context ID"),
        "authority_requirement_id": _identifier(raw["authority_requirement_id"], _AUTHORITY_REQUIREMENT_PREFIX, "TIME authority requirement ID"),
        "actor_profile_content_id": _identifier(raw["actor_profile_content_id"], _PROFILE_PREFIX, "TIME actor profile ID"),
        "actor_profile_canonical_sha256": _sha256(raw["actor_profile_canonical_sha256"], "TIME actor profile SHA"),
        "campaign_id": _identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "TIME campaign ID"),
        "attempt_id": _identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "TIME attempt ID"),
        "execution_registry_id": _identifier(raw["execution_registry_id"], _REGISTRY_PREFIX, "TIME execution registry ID"),
        "logical_genesis_token_id": _identifier(raw["logical_genesis_token_id"], _GENESIS_PREFIX, "logical genesis token ID"),
        "legacy_registry_policy_authorization": _ascii_text(raw["legacy_registry_policy_authorization"], "legacy registry policy authorization"),
        "actual_campaign_eligible": _exact_bool(raw["actual_campaign_eligible"], "actual campaign eligibility"),
        "time_semantics": _ascii_text(raw["time_semantics"], "TIME semantics"),
        "predecessor_token_id": predecessor,
        "previous_logical_sequence": _exact_integer(raw["previous_logical_sequence"], "previous logical sequence", maximum=9),
        "logical_sequence": _exact_integer(raw["logical_sequence"], "logical sequence", maximum=9),
        "previous_stage": raw["previous_stage"],
        "stage": raw["stage"],
        "stage_index": _exact_integer(raw["stage_index"], "stage index", maximum=9),
        "event_projection_id": projection_id,
        "event_root_sha256": _sha256(raw["event_root_sha256"], "event root SHA"),
    }


def _snapshot_signature(value: object) -> dict[str, object]:
    if type(value) is not six.Phase2BC1DetachedSignatureV1:
        raise TypeError("TIME signature exact type drift")
    raw = _raw_fields(value)
    if type(raw["role"]) is not six.Phase2BC1SixRoleV1:
        raise TypeError("signature role exact enum drift")
    if type(raw["signature_purpose"]) is not six.Phase2BC1SignaturePurposeV1:
        raise TypeError("signature purpose exact enum drift")
    signature = raw["signature"]
    if type(signature) is not bytes or len(signature) != _RAW_SIGNATURE_BYTES:
        raise TypeError("TIME signature must be exact 64-byte bytes")
    return {
        "purpose_id": _exact_integer(raw["purpose_id"], "signature purpose ID", maximum=6),
        "role": raw["role"],
        "signature_purpose": raw["signature_purpose"],
        "algorithm": _ascii_text(raw["algorithm"], "signature algorithm"),
        "key_epoch": _exact_integer(raw["key_epoch"], "signature key epoch", maximum=2**31 - 1),
        "signer_key_id": _identifier(raw["signer_key_id"], _KEY_PREFIX, "signer key ID"),
        "signature": bytes(signature),
    }


def _snapshot_token(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageTokenCandidateV1:
        raise TypeError("logical-stage token exact type drift")
    return {
        "body": _snapshot_token_body(object.__getattribute__(value, "body")),
        "time_signature": _snapshot_signature(object.__getattribute__(value, "time_signature")),
        "logical_stage_token_id": _identifier(object.__getattribute__(value, "logical_stage_token_id"), _STAGE_TOKEN_PREFIX, "logical-stage token ID"),
    }


def _token_body_pairs(body: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, value.value if name in ("previous_stage", "stage") else value)
        for name, value in body.items()
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


def _signature_header_pairs(signer_key_id: str) -> tuple[tuple[str, object], ...]:
    return (
        ("purpose_id", 4),
        ("role", six.Phase2BC1SixRoleV1.TIME_ATTESTOR.value),
        ("signature_purpose", six.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN.value),
        ("algorithm", "Ed25519"),
        ("key_epoch", six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH),
        ("signer_key_id", signer_key_id),
    )


def phase2b_c1_logical_stage_signature_preimage_v1(
    body: Phase2BC1LogicalStageTokenBodyV1, *, signer_key_id: str
) -> bytes:
    closed = _snapshot_token_body(body)
    signer = _identifier(signer_key_id, _KEY_PREFIX, "signer key ID")
    return _STAGE_SIGNATURE_DOMAIN + _canonical_named_bytes((
        ("signature_header", _signature_header_pairs(signer)),
        ("body", _token_body_pairs(closed)),
    ))


def phase2b_c1_logical_stage_token_id_v1(
    body: Phase2BC1LogicalStageTokenBodyV1,
    time_signature: six.Phase2BC1DetachedSignatureV1,
) -> str:
    closed_body = _snapshot_token_body(body)
    closed_signature = _snapshot_signature(time_signature)
    return _STAGE_TOKEN_PREFIX + _digest(_STAGE_TOKEN_ID_DOMAIN, (
        ("body", _token_body_pairs(closed_body)),
        ("time_signature", _signature_pairs(closed_signature)),
    ))


def canonical_phase2b_c1_logical_stage_token_bytes_v1(
    candidate: Phase2BC1LogicalStageTokenCandidateV1,
) -> bytes:
    closed = _snapshot_token(candidate)
    return _canonical_named_bytes((
        ("body", _token_body_pairs(closed["body"])),
        ("time_signature", _signature_pairs(closed["time_signature"])),
        ("logical_stage_token_id", closed["logical_stage_token_id"]),
    ))


def _projection_id_and_root(
    stage: six.Phase2BC1LogicalStageV1, projection: dict[str, object]
) -> tuple[str, str]:
    digest = _digest(_EVENT_DOMAINS[stage], _projection_pairs(projection))
    return _PROJECTION_PREFIXES[stage] + digest, digest


def _validate_projection_bindings(
    *, index: int, stage: six.Phase2BC1LogicalStageV1,
    projection: dict[str, object], context: six.Phase2BC1CeremonyContextV1,
    registry: six.Phase2BC1ExecutionRegistryCandidateV1,
    prior_tokens: tuple[dict[str, object], ...],
    prior_projections: tuple[dict[str, object], ...],
) -> None:
    body = context.body
    expected_common = {
        "version": PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION,
        "context_id": context.context_id,
        "authority_requirement_id": body.authority_requirement_id,
        "actor_profile_content_id": body.actor_profile_content_id,
        "campaign_id": body.campaign_id,
        "attempt_id": body.attempt_id,
        "execution_registry_id": registry.execution_registry_id,
    }
    if any(projection[name] != expected for name, expected in expected_common.items()):
        raise ValueError("event projection common graph binding drift")
    if index == 1:
        return
    previous_projection = prior_projections[-1]
    previous_token_id = prior_tokens[-1]["logical_stage_token_id"]
    if index == 2:
        if (
            projection["raw_input_committed_token_id"] != previous_token_id
            or projection["execution_freeze_manifest_id"] != body.execution_freeze_manifest_id
            or projection["execution_freeze_manifest_sha256"] != body.execution_freeze_manifest_sha256
            or projection["input_archive_id"] != previous_projection["input_archive_id"]
            or projection["input_archive_sha256"] != previous_projection["input_archive_sha256"]
        ):
            raise ValueError("package projection predecessor binding drift")
    elif index == 3:
        package = previous_projection
        if (
            projection["package_commitments_frozen_token_id"] != previous_token_id
            or projection["recognizer_run_spec_id"] != package["recognizer_run_spec_id"]
            or projection["execution_freeze_manifest_id"] != package["execution_freeze_manifest_id"]
            or projection["input_archive_id"] != package["input_archive_id"]
        ):
            raise ValueError("run-start projection predecessor binding drift")
    elif index == 4:
        started = previous_projection
        raw = prior_projections[0]
        package = prior_projections[1]
        if (
            projection["run_started_token_id"] != previous_token_id
            or projection["recognizer_run_spec_id"] != started["recognizer_run_spec_id"]
            or projection["execution_freeze_manifest_id"] != body.execution_freeze_manifest_id
            or projection["execution_freeze_manifest_sha256"] != body.execution_freeze_manifest_sha256
            or projection["input_archive_id"] != raw["input_archive_id"]
            or projection["input_archive_sha256"] != raw["input_archive_sha256"]
            or projection["recognizer_run_spec_id"] != package["recognizer_run_spec_id"]
        ):
            raise ValueError("run-finish projection predecessor binding drift")


def _validate_token_bindings(
    *, index: int, stage: six.Phase2BC1LogicalStageV1,
    token: dict[str, object], projection: dict[str, object],
    context: six.Phase2BC1CeremonyContextV1,
    registry: six.Phase2BC1ExecutionRegistryCandidateV1,
    genesis: six.Phase2BC1LogicalGenesisTokenCandidateV1,
    prior_tokens: tuple[dict[str, object], ...],
) -> None:
    body = token["body"]
    context_body = context.body
    previous_stage = (
        six.Phase2BC1LogicalStageV1.PREREGISTERED
        if index == 1 else _PREFIX_STAGES[index - 2]
    )
    predecessor_id = (
        genesis.logical_genesis_token_id
        if index == 1 else prior_tokens[-1]["logical_stage_token_id"]
    )
    expected = {
        "version": PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION,
        "context_id": context.context_id,
        "authority_requirement_id": context_body.authority_requirement_id,
        "actor_profile_content_id": context_body.actor_profile_content_id,
        "actor_profile_canonical_sha256": context_body.actor_profile_canonical_sha256,
        "campaign_id": context_body.campaign_id,
        "attempt_id": context_body.attempt_id,
        "execution_registry_id": registry.execution_registry_id,
        "logical_genesis_token_id": genesis.logical_genesis_token_id,
        "legacy_registry_policy_authorization": PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_LEGACY_POLICY_AUTHORIZATION,
        "actual_campaign_eligible": False,
        "time_semantics": six.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        "predecessor_token_id": predecessor_id,
        "previous_logical_sequence": index - 1,
        "logical_sequence": index,
        "previous_stage": previous_stage,
        "stage": stage,
        "stage_index": index,
        "event_projection_id": projection["event_projection_id"],
    }
    if any(body[name] != value for name, value in expected.items()):
        raise ValueError("logical-stage token graph or adjacency drift")
    expected_projection_id, expected_root = _projection_id_and_root(stage, projection)
    if (
        projection["event_projection_id"] != expected_projection_id
        or body["event_root_sha256"] != expected_root
    ):
        raise ValueError("logical-stage event content address drift")
    time_key = registry.body.role_keys[3]
    signature = token["time_signature"]
    if (
        signature["purpose_id"] != 4
        or signature["role"] is not six.Phase2BC1SixRoleV1.TIME_ATTESTOR
        or signature["signature_purpose"] is not six.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN
        or signature["algorithm"] != "Ed25519"
        or signature["key_epoch"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH
        or signature["signer_key_id"] != time_key.key_id
    ):
        raise ValueError("logical-stage TIME signature metadata drift")
    expected_token_id = _STAGE_TOKEN_PREFIX + _digest(_STAGE_TOKEN_ID_DOMAIN, (
        ("body", _token_body_pairs(body)),
        ("time_signature", _signature_pairs(signature)),
    ))
    if token["logical_stage_token_id"] != expected_token_id:
        raise ValueError("logical-stage token content address drift")


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
    *, disposition: Phase2BC1LogicalStageWireDispositionV1,
    reason: Phase2BC1LogicalStageWireReasonV1,
    context: six.Phase2BC1CeremonyContextV1 | None = None,
    registry: six.Phase2BC1ExecutionRegistryCandidateV1 | None = None,
    genesis: six.Phase2BC1LogicalGenesisTokenCandidateV1 | None = None,
    projections: tuple[dict[str, object], ...] = (),
    tokens: tuple[dict[str, object], ...] = (),
) -> Phase2BC1LogicalStageWireVerificationV1:
    success = disposition is Phase2BC1LogicalStageWireDispositionV1.CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY
    result_id = context_id = attempt_id = registry_id = genesis_id = None
    terminal_stage = None
    terminal_sequence = None
    token_ids: tuple[str, ...] = ()
    projection_ids: tuple[str, ...] = ()
    if success:
        if context is None or registry is None or genesis is None or not tokens:
            raise RuntimeError("logical-stage success lacks a complete prefix")
        context_id = context.context_id
        attempt_id = context.body.attempt_id
        registry_id = registry.execution_registry_id
        genesis_id = genesis.logical_genesis_token_id
        terminal_stage = tokens[-1]["body"]["stage"]
        terminal_sequence = tokens[-1]["body"]["logical_sequence"]
        token_ids = tuple(item["logical_stage_token_id"] for item in tokens)
        projection_ids = tuple(item["event_projection_id"] for item in projections)
        result_id = _RESULT_PREFIX + _digest(_RESULT_ID_DOMAIN, (
            ("version", PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION),
            ("claim_level", PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_CLAIM_LEVEL),
            ("context_id", context_id),
            ("attempt_id", attempt_id),
            ("execution_registry_id", registry_id),
            ("logical_genesis_token_id", genesis_id),
            ("stage_token_ids", token_ids),
            ("event_projection_ids", projection_ids),
        ))
    result = Phase2BC1LogicalStageWireVerificationV1(_RESULT_ISSUANCE_TOKEN)
    values: list[tuple[str, object]] = [
        ("disposition", disposition), ("reason", reason),
        ("version", PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION),
        ("claim_level", PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_CLAIM_LEVEL),
        ("result_id", result_id), ("context_id", context_id),
        ("attempt_id", attempt_id), ("execution_registry_id", registry_id),
        ("logical_genesis_token_id", genesis_id), ("terminal_stage", terminal_stage),
        ("terminal_logical_sequence", terminal_sequence), ("stage_token_ids", token_ids),
        ("event_projection_ids", projection_ids), ("partial_output_published", False),
    ]
    values.extend((name, success) for name in _TRUE_CLAIMS)
    values.extend((name, False) for name in _FALSE_CLAIMS)
    if tuple(name for name, _ in values) != tuple(item.name for item in fields(type(result))):
        raise RuntimeError("logical-stage result field drift")
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def _reject(reason: Phase2BC1LogicalStageWireReasonV1) -> Phase2BC1LogicalStageWireVerificationV1:
    return _issue_result(disposition=Phase2BC1LogicalStageWireDispositionV1.REJECTED, reason=reason)


_ORDINARY_INPUT_ERRORS: Final = (
    AttributeError, TypeError, ValueError, OverflowError, UnicodeError,
)


def verify_phase2b_c1_logical_stage_prefix_v1(
    *,
    context: six.Phase2BC1CeremonyContextV1,
    registry: six.Phase2BC1ExecutionRegistryCandidateV1,
    store_receipt: six.Phase2BC1StoreIfAbsentReceiptCandidateV1,
    logical_genesis_token: six.Phase2BC1LogicalGenesisTokenCandidateV1,
    event_projections: tuple[Phase2BC1LogicalStageEventProjectionV1, ...],
    stage_tokens: tuple[Phase2BC1LogicalStageTokenCandidateV1, ...],
) -> Phase2BC1LogicalStageWireVerificationV1:
    """Verify genesis plus one exact contiguous stage prefix (length 1..4)."""

    if (
        type(context) is not six.Phase2BC1CeremonyContextV1
        or type(registry) is not six.Phase2BC1ExecutionRegistryCandidateV1
        or type(store_receipt) is not six.Phase2BC1StoreIfAbsentReceiptCandidateV1
        or type(logical_genesis_token) is not six.Phase2BC1LogicalGenesisTokenCandidateV1
        or type(event_projections) is not tuple
        or type(stage_tokens) is not tuple
    ):
        return _reject(Phase2BC1LogicalStageWireReasonV1.WRONG_INPUT_TYPE)
    if not (1 <= len(event_projections) == len(stage_tokens) <= 4):
        return _reject(Phase2BC1LogicalStageWireReasonV1.PREFIX_LENGTH_INVALID)
    if any(type(item) not in (
        Phase2BC1RawInputCommittedEventProjectionV1,
        Phase2BC1PackageCommitmentsFrozenEventProjectionV1,
        Phase2BC1RunStartedEventProjectionV1,
        Phase2BC1RunFinishedEventProjectionV1,
    ) for item in event_projections) or any(
        type(item) is not Phase2BC1LogicalStageTokenCandidateV1 for item in stage_tokens
    ):
        return _reject(Phase2BC1LogicalStageWireReasonV1.WRONG_INPUT_TYPE)
    try:
        # Freeze every caller object together.  Original inputs are never read again.
        try:
            (
                closed_context, closed_registry, closed_store, closed_genesis,
                frozen_projections, frozen_tokens,
            ) = copy.deepcopy((
                context, registry, store_receipt, logical_genesis_token,
                event_projections, stage_tokens,
            ))
        except Exception:
            return _reject(Phase2BC1LogicalStageWireReasonV1.GRAPH_SNAPSHOT_INVALID)

        projection_snapshots: list[tuple[six.Phase2BC1LogicalStageV1, dict[str, object]]] = []
        try:
            projection_snapshots = [
                _snapshot_projection(item) for item in frozen_projections
            ]
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1LogicalStageWireReasonV1.EVENT_PROJECTION_INVALID)
        try:
            token_snapshots = tuple(_snapshot_token(item) for item in frozen_tokens)
        except _ORDINARY_INPUT_ERRORS:
            return _reject(Phase2BC1LogicalStageWireReasonV1.STAGE_TOKEN_INVALID)

        for stage, projection in projection_snapshots:
            projection["event_projection_id"] = _projection_id_and_root(
                stage, projection
            )[0]
        closed_projection_values = tuple(item[1] for item in projection_snapshots)
        for offset, ((stage, projection), token) in enumerate(
            zip(projection_snapshots, token_snapshots, strict=True), start=1
        ):
            if stage is not _PREFIX_STAGES[offset - 1]:
                return _reject(Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID)
            try:
                _validate_projection_bindings(
                    index=offset, stage=stage, projection=projection,
                    context=closed_context, registry=closed_registry,
                    prior_tokens=token_snapshots[:offset - 1],
                    prior_projections=closed_projection_values[:offset - 1],
                )
                _validate_token_bindings(
                    index=offset, stage=stage, token=token, projection=projection,
                    context=closed_context, registry=closed_registry,
                    genesis=closed_genesis, prior_tokens=token_snapshots[:offset - 1],
                )
            except _ORDINARY_INPUT_ERRORS:
                return _reject(Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID)

        genesis_result = six.verify_phase2b_c1_six_role_wire_v1(
            context=closed_context, registry=closed_registry,
            store_receipt=closed_store, logical_genesis_token=closed_genesis,
        )
        if genesis_result.disposition is not six.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY:
            if genesis_result.reason is six.Phase2BC1SixRoleWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE:
                return _reject(Phase2BC1LogicalStageWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE)
            return _reject(Phase2BC1LogicalStageWireReasonV1.GENESIS_REVERIFICATION_REJECTED)

        time_key = closed_registry.body.role_keys[3]
        try:
            for token in token_snapshots:
                body_pairs = _token_body_pairs(token["body"])
                preimage = _STAGE_SIGNATURE_DOMAIN + _canonical_named_bytes((
                    ("signature_header", _signature_header_pairs(time_key.key_id)),
                    ("body", body_pairs),
                ))
                _verify_ed25519(
                    time_key.public_key,
                    token["time_signature"]["signature"],
                    preimage,
                )
        except _BackendUnavailable:
            return _reject(Phase2BC1LogicalStageWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE)
        except _SignatureInvalid:
            return _reject(Phase2BC1LogicalStageWireReasonV1.SIGNATURE_INVALID)
        return _issue_result(
            disposition=Phase2BC1LogicalStageWireDispositionV1.CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY,
            reason=Phase2BC1LogicalStageWireReasonV1.CALLER_SUPPLIED_PREFIX_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED,
            context=closed_context, registry=closed_registry, genesis=closed_genesis,
            projections=closed_projection_values, tokens=token_snapshots,
        )
    except _ORDINARY_INPUT_ERRORS:
        return _reject(Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID)
    except Exception:
        return _reject(Phase2BC1LogicalStageWireReasonV1.INTERNAL_ERROR)


_ALL_DOMAINS: Final = (
    _STAGE_SIGNATURE_DOMAIN, _STAGE_TOKEN_ID_DOMAIN, _RESULT_ID_DOMAIN,
    *_EVENT_DOMAINS.values(),
)
if len(set(_ALL_DOMAINS)) != len(_ALL_DOMAINS) or any(not item.endswith(b"\x00") for item in _ALL_DOMAINS):
    raise RuntimeError("logical-stage domain separation drift")
if len(set(_TRUE_CLAIMS)) != len(_TRUE_CLAIMS) or len(set(_FALSE_CLAIMS)) != len(_FALSE_CLAIMS):
    raise RuntimeError("logical-stage claim duplicate")
if set(_TRUE_CLAIMS) & set(_FALSE_CLAIMS):
    raise RuntimeError("logical-stage claim polarity overlap")


__all__ = (
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_CLAIM_LEVEL",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_LEGACY_POLICY_AUTHORIZATION",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_ACTUAL_CAMPAIGN_ELIGIBLE",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_ED25519_BACKEND_AVAILABLE",
    "Phase2BC1LogicalStageWireDispositionV1",
    "Phase2BC1LogicalStageWireReasonV1",
    "Phase2BC1RawInputCommittedEventProjectionV1",
    "Phase2BC1PackageCommitmentsFrozenEventProjectionV1",
    "Phase2BC1RunStartedEventProjectionV1",
    "Phase2BC1RunFinishedEventProjectionV1",
    "Phase2BC1LogicalStageEventProjectionV1",
    "Phase2BC1LogicalStageTokenBodyV1",
    "Phase2BC1LogicalStageTokenCandidateV1",
    "Phase2BC1LogicalStageWireVerificationV1",
    "phase2b_c1_logical_stage_event_projection_id_v1",
    "phase2b_c1_logical_stage_event_root_sha256_v1",
    "canonical_phase2b_c1_logical_stage_event_projection_bytes_v1",
    "phase2b_c1_logical_stage_signature_preimage_v1",
    "phase2b_c1_logical_stage_token_id_v1",
    "canonical_phase2b_c1_logical_stage_token_bytes_v1",
    "verify_phase2b_c1_logical_stage_prefix_v1",
)
