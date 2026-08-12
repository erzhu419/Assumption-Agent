"""Non-authoritative, contract-only Phase-2B V2 scoring mechanics.

This module freezes an evaluator-side answer-manifest schema, a supplied
salted-opening check, nine metric definitions, and references to the already
frozen gate thresholds.  It deliberately does not read predictions, compute a
metric result, execute a gate, run a recognizer, or establish answer authority
or reveal timing.  A successful return therefore means only
``CONTRACT_BINDING_COMPLETE_NOT_SCORED``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
from typing import Final
from uuid import UUID

from .hashing import canonical_json, stable_hash
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from .phase2b_protocol import (
    ONE_SIDED_CONFIDENCE,
    Phase2BCaseType,
    frozen_phase2b_protocol,
    salted_answer_commitment_sha256 as _salted_answer_commitment_sha256,
)
from .phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from . import phase2b_strict_recognizer_cli_v2 as _strict_v2
from .phase2b_strict_recognizer_cli_v2 import (
    StrictRecognizerCliDispositionV2,
    StrictRecognizerStructuralReceiptV2,
)
from . import phase2b_unsealed_prediction_evaluator_v2 as _evaluator_v2
from .phase2b_unsealed_prediction_evaluator_v2 import (
    UnsealedPredictionEvaluationDispositionV2,
    UnsealedPredictionPartitionManifestV2,
    UnsealedPredictionStructuralEvaluationV2,
)
from .phase2b_wire import RoleBinding


FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION: Final = (
    "hegel-machine-phase2b-formal-unsealed-prediction-scoring-contract/2"
)
FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_CONTRACT_ONLY"
)

_ANSWER_MANIFEST_SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-formal-unsealed-answer-manifest/2"
)
_ANSWER_ROW_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW/V2\x00"
)
_ANSWER_ROW_IDS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00"
)
_ANSWER_MANIFEST_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_MANIFEST/V2\x00"
)
_METRIC_DEFINITION_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/METRIC_DEFINITION/V2\x00"
)
_SCORING_CONTRACT_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/SCORING_CONTRACT/V2\x00"
)
_MAIN_ROWS_DOMAIN_V2: Final = b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00"
_SEMANTIC_CONFLICT_ROWS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/UNSEALED/SEMANTIC_CONFLICT_ROWS/V2\x00"
)
_PARTITION_UNION_ROWS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/UNSEALED/PARTITION_UNION_ROWS/V2\x00"
)
_ORDERED_ARCHIVE_INPUT_ROWS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00"
)
_PARTITION_MANIFEST_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/UNSEALED/PARTITION_MANIFEST/V2\x00"
)

_ANSWER_ROW_ID_PREFIX_V2: Final = "phase2b_formal_unsealed_answer_row_v2_"
_ANSWER_ROW_IDS_ROOT_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_answer_rows_v2_"
)
_ANSWER_MANIFEST_ID_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_answer_manifest_v2_"
)
_METRIC_DEFINITION_ID_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_metric_definition_v2_"
)
_SCORING_CONTRACT_ID_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
)

_MAIN_COUNT_V2: Final = 720
_SEMANTIC_CONFLICT_COUNT_V2: Final = 240
_TOTAL_COUNT_V2: Final = 960
_MAXIMUM_TEXT_BYTES_V2: Final = 4_096
_MAXIMUM_SALT_BYTES_V2: Final = 4_096
_MAXIMUM_BINDINGS_V2: Final = 64
_MAXIMUM_SCALES_V2: Final = 4_096

_RECEIPT_SUCCESS_REASON_V2: Final = (
    "strict_v2_structural_input_output_binding_complete"
)
_EVALUATION_SUCCESS_REASON_V2: Final = (
    "sorted_disjoint_exhaustive_720_240_same_v2_archive_row_set_and_ordered_root"
)


class FormalUnsealedMetricKindV2(str, Enum):
    COUNT = "COUNT"
    BINARY_ACCURACY = "BINARY_ACCURACY"


class FormalUnsealedPredictionScoringContractDispositionV2(str, Enum):
    CONTRACT_BINDING_COMPLETE_NOT_SCORED = (
        "CONTRACT_BINDING_COMPLETE_NOT_SCORED"
    )
    REJECTED = "REJECTED"


class FormalUnsealedPredictionScoringContractReasonV2(str, Enum):
    CONTRACT_BINDING_VERIFIED = "CONTRACT_BINDING_VERIFIED"
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    STRUCTURAL_EVALUATION_NOT_COMPLETE = (
        "STRUCTURAL_EVALUATION_NOT_COMPLETE"
    )
    PARTITION_MANIFEST_INVALID = "PARTITION_MANIFEST_INVALID"
    ANSWER_MANIFEST_INVALID = "ANSWER_MANIFEST_INVALID"
    ANSWER_COMMITMENT_OPENING_INVALID = (
        "ANSWER_COMMITMENT_OPENING_INVALID"
    )
    ROW_COVERAGE_MISMATCH = "ROW_COVERAGE_MISMATCH"
    CASE_TYPE_QUOTA_MISMATCH = "CASE_TYPE_QUOTA_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_TRUE_VALIDATION_CLAIMS_V2: Final = (
    "contract_identity_verified",
    "structural_receipt_binding_verified",
    "structural_evaluation_binding_verified",
    "partition_manifest_binding_verified",
    "evaluator_side_answer_schema_verified",
    "supplied_answer_commitment_opening_verified",
    "exact_main_answer_row_coverage_verified",
    "frozen_case_type_quota_verified",
    "nine_metric_definition_mechanics_verified",
    "challenge_excluded_from_main_denominator",
)

_FALSE_VALIDATION_CLAIMS_V2: Final = (
    "challenge_in_main_denominator",
    "answer_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "formal_gate_evaluation_performed",
    "metric_results_materialized",
    "scored_rows_materialized",
    "control_rejection_metrics_implemented",
    "slice_gate_metrics_implemented",
    "challenge_scoring_performed",
    "effect_evidence",
    "c1_exit_evidence",
)

_RECEIPT_TRUE_CLAIMS_V2: Final = (
    "structural_input_archive_verified",
    "structural_prediction_archive_verified",
    "cross_archive_context_binding_verified",
    "ordered_row_identity_verified",
    "seven_input_root_columns_positionally_verified",
)
_RECEIPT_FALSE_CLAIMS_V2: Final = (
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)
_EVALUATION_FALSE_CLAIMS_V2: Final = (
    "challenge_in_main_denominator",
    *_RECEIPT_FALSE_CLAIMS_V2,
)

_ANSWER_ROW_FIELDS_V2: Final = (
    "input_row_id",
    "case_type",
    "expected_decision",
    "canonical_family_id",
    "binding",
    "admissible_scale_ids",
    "answer_row_id",
)
_ANSWER_MANIFEST_FIELDS_V2: Final = (
    "schema_version",
    "schema_id",
    "policy_id",
    "claim_level",
    "exact_freeze_id",
    "phase2b_protocol_id",
    "execution_freeze_manifest_id",
    "input_archive_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_archive_policy_id",
    "batch_id",
    "batch_policy_id",
    "ordered_archive_input_row_ids_root",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "main_answer_rows",
    "main_answer_row_ids_root",
    "answer_manifest_sha256",
    "answer_manifest_id",
)
_METRIC_DEFINITION_FIELDS_V2: Final = (
    "metric_name",
    "metric_kind",
    "denominator_case_types",
    "expected_denominator",
    "success_rule",
    "separately_reported",
    "metric_definition_id",
)
_SCORING_CONTRACT_FIELDS_V2: Final = (
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "answer_row_schema_id",
    "answer_manifest_schema_id",
    "required_structural_receipt_type",
    "required_structural_evaluation_type",
    "required_partition_manifest_type",
    "main_row_count",
    "semantic_conflict_row_count",
    "case_type_counts",
    "metric_definitions",
    "set_valued_joint_rule",
    "commitment_opening_formula",
    "challenge_denominator_policy",
    "overall_gate_definitions",
    "slice_gate_definitions",
    "scale_regret_gate_definition",
    "bootstrap_reference",
    "bootstrap_evaluated",
    "overall_gate_metric_mapping",
    "wilson_method",
    "wilson_semantics",
    "wilson_confidence",
    "gate_inputs_implemented",
    "gate_results",
    "gates_executed",
    "contract_id",
)
_VALIDATION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "prediction_archive_id",
    "partition_manifest_id",
    "structural_receipt_id",
    "answer_manifest_id",
    "answer_manifest_sha256",
    "salted_answer_commitment_sha256",
    "main_row_count",
    "semantic_conflict_row_count",
    "answerable_row_count",
    "main_answer_row_ids_root",
    "ordered_archive_input_row_ids_root",
    *_TRUE_VALIDATION_CLAIMS_V2,
    *_FALSE_VALIDATION_CLAIMS_V2,
    "metric_definitions",
    "metric_results",
    "scored_rows",
)
_REJECTION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "validation",
    "metric_definitions",
    "metric_results",
    "scored_rows",
    "partial_output_published",
    *_TRUE_VALIDATION_CLAIMS_V2,
    *_FALSE_VALIDATION_CLAIMS_V2,
)

_ANSWER_MANIFEST_PREIMAGE_FIELDS_V2: Final = tuple(
    name
    for name in _ANSWER_MANIFEST_FIELDS_V2
    if name not in {"answer_manifest_sha256", "answer_manifest_id"}
)


def _exact_text_v2(
    value: object,
    *,
    name: str,
    maximum_bytes: int = _MAXIMUM_TEXT_BYTES_V2,
    ascii_only: bool = True,
) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must use nonempty exact text")
    try:
        encoded = value.encode("ascii" if ascii_only else "utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} encoding drift") from exc
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{name} exceeds the frozen byte cap")
    return value


def _hex64_v2(value: object, *, name: str) -> str:
    text = _exact_text_v2(value, name=name)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return text


def _digest_v2(value: object, *, prefix: str, name: str) -> str:
    text = _exact_text_v2(value, name=name)
    if len(text) != len(prefix) + 64 or text[: len(prefix)] != prefix:
        raise ValueError(f"{name} prefix drift")
    _hex64_v2(text[len(prefix) :], name=name)
    return text


def _uuid4_v2(value: object, *, name: str) -> str:
    text = _exact_text_v2(value, name=name, maximum_bytes=36)
    try:
        parsed = UUID(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != text:
        raise ValueError(f"{name} must be canonical lowercase UUIDv4")
    return text


def _exact_bool_v2(value: object, *, expected: bool, name: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{name} exact boolean drift")
    return value


def _exact_empty_tuple_v2(value: object, *, name: str) -> tuple[()]:
    if type(value) is not tuple or value != ():
        raise ValueError(f"{name} must remain exactly empty")
    return ()


def _stable_primitive_id_v2(
    value: dict[str, object],
    *,
    domain: bytes,
    prefix: str,
) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return prefix + hashlib.sha256(domain + encoded).hexdigest()


def _lexically_equal_v2(left: object, right: object) -> bool:
    """Compare values already exact-closed without invoking subclass hooks."""

    return type(left) is type(right) and left == right


def _row_sequence_root_v2(
    values: tuple[str, ...],
    *,
    expected_count: int,
    domain: bytes,
    prefix: str,
    sorted_unique: bool,
    name: str,
) -> str:
    if type(values) is not tuple or len(values) != expected_count:
        raise ValueError(f"{name} requires the exact frozen count")
    encoded_values: list[bytes] = []
    for value in values:
        _digest_v2(
            value,
            prefix="phase2b_recognizer_input_row_v2_",
            name=f"{name} row ID",
        )
        encoded_values.append(value.encode("ascii"))
    if len(set(values)) != expected_count:
        raise ValueError(f"{name} row IDs are not unique")
    if sorted_unique and values != tuple(sorted(values)):
        raise ValueError(f"{name} row IDs are not sorted")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(expected_count.to_bytes(4, "big"))
    for encoded in encoded_values:
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return prefix + digest.hexdigest()


def _main_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=_MAIN_COUNT_V2,
        domain=_MAIN_ROWS_DOMAIN_V2,
        prefix="phase2b_unsealed_main_rows_v2_",
        sorted_unique=True,
        name="formal V2 main partition",
    )


def _semantic_conflict_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=_SEMANTIC_CONFLICT_COUNT_V2,
        domain=_SEMANTIC_CONFLICT_ROWS_DOMAIN_V2,
        prefix="phase2b_unsealed_semantic_conflict_rows_v2_",
        sorted_unique=True,
        name="formal V2 semantic-conflict partition",
    )


def _partition_union_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=_TOTAL_COUNT_V2,
        domain=_PARTITION_UNION_ROWS_DOMAIN_V2,
        prefix="phase2b_unsealed_partition_union_rows_v2_",
        sorted_unique=True,
        name="formal V2 sorted partition union",
    )


def _ordered_archive_input_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=_TOTAL_COUNT_V2,
        domain=_ORDERED_ARCHIVE_INPUT_ROWS_DOMAIN_V2,
        prefix="phase2b_prediction_input_rows_v2_",
        sorted_unique=False,
        name="formal V2 ordered archive rows",
    )


def _answer_row_ids_root_v2(values: tuple[str, ...]) -> str:
    if type(values) is not tuple or len(values) != _MAIN_COUNT_V2:
        raise ValueError("formal V2 answer row IDs require exact 720 count")
    digest = hashlib.sha256()
    digest.update(_ANSWER_ROW_IDS_DOMAIN_V2)
    digest.update(_MAIN_COUNT_V2.to_bytes(4, "big"))
    seen: set[str] = set()
    for value in values:
        _digest_v2(value, prefix=_ANSWER_ROW_ID_PREFIX_V2, name="answer row ID")
        if value in seen:
            raise ValueError("formal V2 answer row IDs are not unique")
        seen.add(value)
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return _ANSWER_ROW_IDS_ROOT_PREFIX_V2 + digest.hexdigest()


def _partition_manifest_id_v2(
    *,
    prediction_archive_id: str,
    prediction_archive_schema_version: str,
    prediction_archive_policy_id: str,
    exact_freeze_id: str,
    evaluator_policy_id: str,
    main_row_ids_root: str,
    semantic_conflict_row_ids_root: str,
    partition_union_row_ids_root: str,
    ordered_archive_input_row_ids_root: str,
) -> str:
    values = (
        prediction_archive_id,
        prediction_archive_schema_version,
        prediction_archive_policy_id,
        exact_freeze_id,
        evaluator_policy_id,
        main_row_ids_root,
        semantic_conflict_row_ids_root,
        partition_union_row_ids_root,
        ordered_archive_input_row_ids_root,
    )
    digest = hashlib.sha256()
    digest.update(_PARTITION_MANIFEST_DOMAIN_V2)
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    digest.update(_MAIN_COUNT_V2.to_bytes(4, "big"))
    digest.update(_SEMANTIC_CONFLICT_COUNT_V2.to_bytes(4, "big"))
    digest.update(_TOTAL_COUNT_V2.to_bytes(4, "big"))
    return "phase2b_unsealed_prediction_partition_v2_" + digest.hexdigest()


def _binding_mapping_v2(value: tuple[RoleBinding, ...]) -> list[dict[str, str]]:
    return [
        {"role_id": item.role_id, "entity_id": item.entity_id}
        for item in value
    ]


def _answer_row_preimage_v2(
    *,
    input_row_id: str,
    case_type: Phase2BCaseType,
    expected_decision: PredictionDecisionV2,
    canonical_family_id: CanonicalFamilyId | None,
    binding: tuple[RoleBinding, ...],
    admissible_scale_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "input_row_id": input_row_id,
        "case_type": case_type.value,
        "expected_decision": expected_decision.value,
        "canonical_family_id": (
            None if canonical_family_id is None else canonical_family_id.value
        ),
        "binding": _binding_mapping_v2(binding),
        "admissible_scale_ids": list(admissible_scale_ids),
    }


def _answer_row_id_v2(
    *,
    input_row_id: str,
    case_type: Phase2BCaseType,
    expected_decision: PredictionDecisionV2,
    canonical_family_id: CanonicalFamilyId | None,
    binding: tuple[RoleBinding, ...],
    admissible_scale_ids: tuple[str, ...],
) -> str:
    return _stable_primitive_id_v2(
        _answer_row_preimage_v2(
            input_row_id=input_row_id,
            case_type=case_type,
            expected_decision=expected_decision,
            canonical_family_id=canonical_family_id,
            binding=binding,
            admissible_scale_ids=admissible_scale_ids,
        ),
        domain=_ANSWER_ROW_DOMAIN_V2,
        prefix=_ANSWER_ROW_ID_PREFIX_V2,
    )


@dataclass(frozen=True, slots=True)
class FormalUnsealedAnswerRowV2:
    """Evaluator-side transport row; the builder privately issues an empty ID."""

    input_row_id: str
    case_type: Phase2BCaseType
    expected_decision: PredictionDecisionV2
    canonical_family_id: CanonicalFamilyId | None
    binding: tuple[RoleBinding, ...]
    admissible_scale_ids: tuple[str, ...]
    answer_row_id: str = ""


@dataclass(frozen=True, slots=True, init=False)
class FormalUnsealedAnswerManifestV2:
    schema_version: str
    schema_id: str
    policy_id: str
    claim_level: str
    exact_freeze_id: str
    phase2b_protocol_id: str
    execution_freeze_manifest_id: str
    input_archive_id: str
    input_archive_sha256: str
    input_archive_version: str
    input_archive_policy_id: str
    batch_id: str
    batch_policy_id: str
    ordered_archive_input_row_ids_root: str
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    partition_union_row_ids_root: str
    main_answer_rows: tuple[FormalUnsealedAnswerRowV2, ...]
    main_answer_row_ids_root: str
    answer_manifest_sha256: str
    answer_manifest_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("formal V2 answer manifests are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class FormalUnsealedMetricDefinitionV2:
    metric_name: str
    metric_kind: FormalUnsealedMetricKindV2
    denominator_case_types: tuple[Phase2BCaseType, ...]
    expected_denominator: int
    success_rule: str
    separately_reported: bool
    metric_definition_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("formal V2 metric definitions are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class FormalUnsealedPredictionScoringContractV2:
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    answer_row_schema_id: str
    answer_manifest_schema_id: str
    required_structural_receipt_type: str
    required_structural_evaluation_type: str
    required_partition_manifest_type: str
    main_row_count: int
    semantic_conflict_row_count: int
    case_type_counts: tuple[tuple[Phase2BCaseType, int], ...]
    metric_definitions: tuple[FormalUnsealedMetricDefinitionV2, ...]
    set_valued_joint_rule: str
    commitment_opening_formula: str
    challenge_denominator_policy: str
    overall_gate_definitions: tuple[tuple[object, ...], ...]
    slice_gate_definitions: tuple[tuple[object, ...], ...]
    scale_regret_gate_definition: tuple[object, ...]
    bootstrap_reference: tuple[object, ...]
    bootstrap_evaluated: bool
    overall_gate_metric_mapping: tuple[tuple[str, str | None], ...]
    wilson_method: str
    wilson_semantics: str
    wilson_confidence: float
    gate_inputs_implemented: bool
    gate_results: tuple[()]
    gates_executed: bool
    contract_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("formal V2 scoring contracts are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class FormalUnsealedPredictionScoringContractValidationV2:
    disposition: FormalUnsealedPredictionScoringContractDispositionV2
    reason: FormalUnsealedPredictionScoringContractReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    prediction_archive_id: str
    partition_manifest_id: str
    structural_receipt_id: str
    answer_manifest_id: str
    answer_manifest_sha256: str
    salted_answer_commitment_sha256: str
    main_row_count: int
    semantic_conflict_row_count: int
    answerable_row_count: int
    main_answer_row_ids_root: str
    ordered_archive_input_row_ids_root: str
    contract_identity_verified: bool
    structural_receipt_binding_verified: bool
    structural_evaluation_binding_verified: bool
    partition_manifest_binding_verified: bool
    evaluator_side_answer_schema_verified: bool
    supplied_answer_commitment_opening_verified: bool
    exact_main_answer_row_coverage_verified: bool
    frozen_case_type_quota_verified: bool
    nine_metric_definition_mechanics_verified: bool
    challenge_excluded_from_main_denominator: bool
    challenge_in_main_denominator: bool
    answer_manifest_authority_verified: bool
    answer_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    execution_manifest_authority_verified: bool
    partition_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    scoring_performed: bool
    prediction_scored: bool
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
    control_rejection_metrics_implemented: bool
    slice_gate_metrics_implemented: bool
    challenge_scoring_performed: bool
    effect_evidence: bool
    c1_exit_evidence: bool
    metric_definitions: tuple[FormalUnsealedMetricDefinitionV2, ...]
    metric_results: tuple[()]
    scored_rows: tuple[()]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("formal V2 validations are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class FormalUnsealedPredictionScoringContractRejectionV2:
    disposition: FormalUnsealedPredictionScoringContractDispositionV2
    reason: FormalUnsealedPredictionScoringContractReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    validation: None
    metric_definitions: tuple[()]
    metric_results: tuple[()]
    scored_rows: tuple[()]
    partial_output_published: bool
    contract_identity_verified: bool
    structural_receipt_binding_verified: bool
    structural_evaluation_binding_verified: bool
    partition_manifest_binding_verified: bool
    evaluator_side_answer_schema_verified: bool
    supplied_answer_commitment_opening_verified: bool
    exact_main_answer_row_coverage_verified: bool
    frozen_case_type_quota_verified: bool
    nine_metric_definition_mechanics_verified: bool
    challenge_excluded_from_main_denominator: bool
    challenge_in_main_denominator: bool
    answer_manifest_authority_verified: bool
    answer_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    execution_manifest_authority_verified: bool
    partition_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    scoring_performed: bool
    prediction_scored: bool
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
    control_rejection_metrics_implemented: bool
    slice_gate_metrics_implemented: bool
    challenge_scoring_performed: bool
    effect_evidence: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("formal V2 rejections are privately issued")


def _field_manifest_v2(value_type: type[object]) -> tuple[str, ...]:
    return tuple(item.name for item in fields(value_type))


def _assert_field_manifests_v2() -> None:
    expected = (
        (FormalUnsealedAnswerRowV2, _ANSWER_ROW_FIELDS_V2),
        (FormalUnsealedAnswerManifestV2, _ANSWER_MANIFEST_FIELDS_V2),
        (FormalUnsealedMetricDefinitionV2, _METRIC_DEFINITION_FIELDS_V2),
        (
            FormalUnsealedPredictionScoringContractV2,
            _SCORING_CONTRACT_FIELDS_V2,
        ),
        (
            FormalUnsealedPredictionScoringContractValidationV2,
            _VALIDATION_FIELDS_V2,
        ),
        (
            FormalUnsealedPredictionScoringContractRejectionV2,
            _REJECTION_FIELDS_V2,
        ),
    )
    for value_type, manifest in expected:
        if _field_manifest_v2(value_type) != manifest:
            raise RuntimeError(f"{value_type.__name__} field manifest drift")


_assert_field_manifests_v2()


_EXACT_FREEZE_V2: Final = frozen_phase2b_exact_freeze()
_FROZEN_PROTOCOL_V2: Final = frozen_phase2b_protocol()
_EXACT_FREEZE_ID_V2: Final = _digest_v2(
    _EXACT_FREEZE_V2.freeze_id,
    prefix="phase2b_exact_freeze_",
    name="formal V2 exact freeze ID",
)
_PHASE2B_PROTOCOL_ID_V2: Final = _digest_v2(
    _FROZEN_PROTOCOL_V2.protocol_id,
    prefix="phase2b_protocol_",
    name="formal V2 protocol ID",
)

_INPUT_ARCHIVE_VERSION_V2: Final = _exact_text_v2(
    _strict_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
    name="formal V2 input archive version",
)
_INPUT_ARCHIVE_POLICY_ID_V2: Final = _digest_v2(
    _strict_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    prefix="phase2b_recognizer_input_archive_policy_v2_",
    name="formal V2 input archive policy",
)
_PREDICTION_ARCHIVE_VERSION_V2: Final = _exact_text_v2(
    _strict_v2.RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
    name="formal V2 prediction archive version",
)
_PREDICTION_ARCHIVE_POLICY_ID_V2: Final = _digest_v2(
    _strict_v2.RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
    prefix="phase2b_recognizer_prediction_archive_policy_v2_",
    name="formal V2 prediction archive policy",
)
_BATCH_POLICY_ID_V2: Final = _digest_v2(
    _strict_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    prefix="phase2b_trusted_wire_batch_v2_policy_",
    name="formal V2 batch policy",
)
_STRICT_SCHEMA_VERSION_V2: Final = _exact_text_v2(
    _strict_v2.STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION,
    name="formal V2 strict receipt schema version",
)
_STRICT_SCHEMA_ID_V2: Final = _digest_v2(
    _strict_v2.STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID,
    prefix="phase2b_strict_recognizer_cli_schema_v2_",
    name="formal V2 strict receipt schema ID",
)
_STRICT_POLICY_ID_V2: Final = _digest_v2(
    _strict_v2.STRICT_RECOGNIZER_CLI_V2_POLICY_ID,
    prefix="phase2b_strict_recognizer_cli_policy_v2_",
    name="formal V2 strict receipt policy",
)
_EVALUATOR_VERSION_V2: Final = _exact_text_v2(
    _evaluator_v2.UNSEALED_PREDICTION_EVALUATOR_V2_VERSION,
    name="formal V2 evaluator version",
)
_EVALUATOR_POLICY_ID_V2: Final = _digest_v2(
    _evaluator_v2.UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2,
    prefix="phase2b_unsealed_prediction_evaluator_policy_v2_",
    name="formal V2 evaluator policy",
)
_MECHANICS_CLAIM_LEVEL_V2: Final = _exact_text_v2(
    _strict_v2.NON_AUTHORITATIVE_CLAIM_LEVEL,
    name="formal V2 upstream claim level",
)


_CASE_TYPE_COUNTS_V2: Final = (
    (Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE, 228),
    (Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE, 12),
    (Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE, 96),
    (Phase2BCaseType.BINDING_COUNTERFACTUAL, 96),
    (Phase2BCaseType.SCALE_COUNTERFACTUAL, 96),
    (Phase2BCaseType.SIGN_OR_INVARIANT_BREAK, 96),
    (Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE, 96),
)
_ANSWERABLE_TYPES_V2: Final = (
    Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
)

_METRIC_SPECS_V2: Final = (
    (
        "answerable_count",
        FormalUnsealedMetricKindV2.COUNT,
        _ANSWERABLE_TYPES_V2,
        240,
        "eligible_case_count_equals_exact_denominator",
        False,
    ),
    (
        "family_exact_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        _ANSWERABLE_TYPES_V2,
        240,
        "positive_decision_and_canonical_family_exact",
        False,
    ),
    (
        "binding_exact_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        _ANSWERABLE_TYPES_V2,
        240,
        "positive_decision_and_canonical_role_binding_tuple_exact",
        False,
    ),
    (
        "scale_set_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        _ANSWERABLE_TYPES_V2,
        240,
        "positive_decision_and_canonical_admissible_scale_tuple_exact",
        False,
    ),
    (
        "unique_scale_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        (Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,),
        228,
        "unique_case_requires_ANSWER_and_exact_singleton_scale",
        False,
    ),
    (
        "joint_exact_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        _ANSWERABLE_TYPES_V2,
        240,
        "expected_decision_family_binding_and_scale_all_exact_set_requires_ANSWER_SET",
        False,
    ),
    (
        "abstention_specificity",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        (Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,),
        228,
        "unique_case_decision_is_not_ABSTAIN",
        False,
    ),
    (
        "nonidentifiability_abstention_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        (Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,),
        96,
        "insufficient_or_nonidentifiable_requires_ABSTAIN_reason_not_scored",
        False,
    ),
    (
        "set_valued_answer_accuracy",
        FormalUnsealedMetricKindV2.BINARY_ACCURACY,
        (Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,),
        12,
        "set_case_requires_ANSWER_SET_family_binding_and_full_scale_set_exact",
        True,
    ),
)

_SET_VALUED_JOINT_RULE_V2: Final = (
    "family_exact_and_binding_exact_and_scale_set_exact_and_ANSWER_SET"
)
_COMMITMENT_OPENING_FORMULA_V2: Final = (
    "sha256(utf8_exact_bounded_salt||ascii_colon||lowercase_manifest_sha256)"
)
_CHALLENGE_DENOMINATOR_POLICY_V2: Final = (
    "semantic_conflict_240_excluded_from_all_main_metric_denominators_and_threshold_tuning"
)

_OVERALL_GATES_V2: Final = (
    ("family_exact", 0.90, 0.86),
    ("binding_exact", 0.90, 0.86),
    ("scale_set_accuracy", 0.87, 0.82),
    ("joint_exact", 0.85, 0.80),
    ("hard_negative_rejection", 0.95, 0.90),
    ("binding_counterfactual_rejection", 0.95, 0.90),
    ("scale_counterfactual_rejection", 0.93, 0.88),
    ("sign_or_invariant_break_rejection", 0.95, 0.90),
    ("abstention_specificity", 0.95, 0.90),
    ("fail_closed_rate", 1.0, None),
    ("preservation_consistency", 0.97, 0.94),
    ("nonidentifiable_scale_abstention", 0.95, 0.90),
)
_SLICE_GATES_V2: Final = (
    ("answerable_joint_exact", 0.80, 0.70, "family"),
    ("all_control_rejection", 0.88, 0.78, "family"),
    ("abstention_specificity", 0.85, 0.75, "family"),
    ("answerable_joint_exact", 0.80, 0.70, "scale"),
    ("all_control_rejection", 0.88, 0.78, "scale"),
    ("abstention_specificity", 0.85, 0.75, "scale"),
)
_SCALE_REGRET_GATE_V2: Final = (
    "normalized_scale_decision_regret",
    0.05,
    0.08,
)
_BOOTSTRAP_REFERENCE_V2: Final = (
    "paired_cluster_bootstrap",
    10_000,
    411_876_909_552_964_556,
    "sha256_domain_separated_uint64_be_first32_v1",
    2_611_585_425,
    "latent_base_case",
    "one_sided_95_percent_percentile",
)
_OVERALL_GATE_METRIC_MAPPING_V2: Final = (
    ("family_exact", "family_exact_accuracy"),
    ("binding_exact", "binding_exact_accuracy"),
    ("scale_set_accuracy", "scale_set_accuracy"),
    ("joint_exact", "joint_exact_accuracy"),
    ("hard_negative_rejection", None),
    ("binding_counterfactual_rejection", None),
    ("scale_counterfactual_rejection", None),
    ("sign_or_invariant_break_rejection", None),
    ("abstention_specificity", "abstention_specificity"),
    ("fail_closed_rate", None),
    ("preservation_consistency", None),
    (
        "nonidentifiable_scale_abstention",
        "nonidentifiability_abstention_accuracy",
    ),
)
_WILSON_METHOD_V2: Final = "one_sided_wilson_lower_confidence_bound"
_WILSON_SEMANTICS_V2: Final = (
    "binary_success_count_over_exact_frozen_denominator_using_"
    "NormalDist_inv_cdf_confidence_no_gate_execution"
)


def _validate_frozen_references_v2() -> None:
    if sum(count for _, count in _CASE_TYPE_COUNTS_V2) != _MAIN_COUNT_V2:
        raise RuntimeError("formal V2 case quota total drift")
    frozen_denominators = tuple(
        (
            item.metric,
            tuple(item.included_case_types),
            item.expected_count,
            item.separately_reported,
        )
        for item in _EXACT_FREEZE_V2.holdout.metric_denominators
    )
    expected_denominators = tuple(
        (
            name,
            tuple(case_type.value for case_type in cases),
            count,
            separate,
        )
        for name, _kind, cases, count, _rule, separate in _METRIC_SPECS_V2
    )
    if frozen_denominators != expected_denominators:
        raise RuntimeError("formal V2 metric denominator freeze drift")
    if _EXACT_FREEZE_V2.holdout.set_valued_joint_rule != _SET_VALUED_JOINT_RULE_V2:
        raise RuntimeError("formal V2 set-valued joint rule drift")
    overall = tuple(
        (
            item.metric,
            item.minimum_point_estimate,
            item.minimum_one_sided_wilson_lcb,
        )
        for item in _FROZEN_PROTOCOL_V2.overall_gates
    )
    slices = tuple(
        (
            item.metric,
            item.minimum_point_estimate,
            item.minimum_one_sided_wilson_lcb,
            item.scope,
        )
        for item in _FROZEN_PROTOCOL_V2.slice_gates
    )
    regret = _FROZEN_PROTOCOL_V2.scale_regret_gate
    regret_tuple = (
        regret.metric,
        regret.maximum_point_estimate,
        regret.maximum_bootstrap_upper_bound,
    )
    bootstrap = _EXACT_FREEZE_V2.bootstrap
    bootstrap_tuple = (
        bootstrap.method,
        bootstrap.replicates,
        bootstrap.seed,
        bootstrap.seed_derivation_id,
        bootstrap.derived_uint32_seed,
        bootstrap.resampling_unit,
        bootstrap.interval,
    )
    if (
        overall != _OVERALL_GATES_V2
        or slices != _SLICE_GATES_V2
        or regret_tuple != _SCALE_REGRET_GATE_V2
        or bootstrap_tuple != _BOOTSTRAP_REFERENCE_V2
        or bootstrap.cluster_members
        != (
            "original_case",
            "all_preservation_variants",
            "all_baseline_predictions",
        )
        or type(ONE_SIDED_CONFIDENCE) is not float
        or ONE_SIDED_CONFIDENCE != 0.95
    ):
        raise RuntimeError("formal V2 referenced gate freeze drift")


_validate_frozen_references_v2()


_ANSWER_ROW_SCHEMA_ID_V2: Final = stable_hash(
    {
        "version": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
        "fields": _ANSWER_ROW_FIELDS_V2,
        "domain_hex": _ANSWER_ROW_DOMAIN_V2.hex(),
        "id_formula": "prefix+sha256(domain||canonical_json(exact_primitive_preimage))",
        "case_semantics": (
            "unique_requires_ANSWER_family_binding_singleton_scale",
            "set_requires_ANSWER_SET_family_binding_sorted_unique_multi_scale",
            "five_control_types_require_ABSTAIN_null_family_empty_binding_scales",
            "prediction_reason_not_frozen_or_scored",
            "empty_answer_row_id_is_privately_content_addressed_by_manifest_builder",
        ),
    },
    prefix="phase2b_formal_unsealed_answer_row_schema_v2_",
)
_ANSWER_MANIFEST_SCHEMA_ID_V2: Final = stable_hash(
    {
        "version": _ANSWER_MANIFEST_SCHEMA_VERSION_V2,
        "fields": _ANSWER_MANIFEST_FIELDS_V2,
        "preimage_fields": _ANSWER_MANIFEST_PREIMAGE_FIELDS_V2,
        "domain_hex": _ANSWER_MANIFEST_DOMAIN_V2.hex(),
        "sha_formula": (
            "sha256(answer_manifest_domain||canonical_json(exact_primitive_preimage))"
        ),
        "id_formula": "answer_manifest_prefix+answer_manifest_sha256",
        "answer_row_schema_id": _ANSWER_ROW_SCHEMA_ID_V2,
        "postprediction_fields_forbidden": True,
    },
    prefix="phase2b_formal_unsealed_answer_manifest_schema_v2_",
)
_ANSWER_MANIFEST_POLICY_ID_V2: Final = stable_hash(
    {
        "schema_id": _ANSWER_MANIFEST_SCHEMA_ID_V2,
        "claim_level": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        "counts": (_MAIN_COUNT_V2, _SEMANTIC_CONFLICT_COUNT_V2, _TOTAL_COUNT_V2),
        "case_type_counts": tuple(
            (case_type.value, count) for case_type, count in _CASE_TYPE_COUNTS_V2
        ),
        "roots": {
            "answer_rows_domain": _ANSWER_ROW_IDS_DOMAIN_V2.hex(),
            "main_domain": _MAIN_ROWS_DOMAIN_V2.hex(),
            "semantic_conflict_domain": _SEMANTIC_CONFLICT_ROWS_DOMAIN_V2.hex(),
            "union_domain": _PARTITION_UNION_ROWS_DOMAIN_V2.hex(),
            "ordered_domain": _ORDERED_ARCHIVE_INPUT_ROWS_DOMAIN_V2.hex(),
        },
        "precommit": (
            "bind_upstream_input_batch_protocol_exact_and_execution_freezes",
            "bind_partition_roots_without_prediction_or_partition_ids",
            "no_prediction_metric_or_scored_row_fields",
        ),
        "scoring_mechanics": {
            "metric_specs": tuple(
                (
                    name,
                    kind.value,
                    tuple(case_type.value for case_type in cases),
                    count,
                    rule,
                    separate,
                )
                for name, kind, cases, count, rule, separate in _METRIC_SPECS_V2
            ),
            "set_valued_joint_rule": _SET_VALUED_JOINT_RULE_V2,
            "overall_gates_referenced_not_executed": _OVERALL_GATES_V2,
            "slice_gates_referenced_not_executed": _SLICE_GATES_V2,
            "scale_regret_gate_referenced_not_executed": _SCALE_REGRET_GATE_V2,
            "bootstrap_referenced_not_executed": _BOOTSTRAP_REFERENCE_V2,
            "overall_gate_metric_mapping": _OVERALL_GATE_METRIC_MAPPING_V2,
            "wilson_method": _WILSON_METHOD_V2,
            "wilson_semantics": _WILSON_SEMANTICS_V2,
            "wilson_confidence": ONE_SIDED_CONFIDENCE,
            "gate_inputs_implemented": False,
            "gate_results": (),
            "gates_executed": False,
            "bootstrap_evaluated": False,
        },
    },
    prefix="phase2b_formal_unsealed_answer_manifest_policy_v2_",
)

FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID: Final = stable_hash(
    {
        "version": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
        "claim_level": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        "answer_row_fields": _ANSWER_ROW_FIELDS_V2,
        "answer_manifest_fields": _ANSWER_MANIFEST_FIELDS_V2,
        "metric_definition_fields": _METRIC_DEFINITION_FIELDS_V2,
        "contract_fields": _SCORING_CONTRACT_FIELDS_V2,
        "validation_fields": _VALIDATION_FIELDS_V2,
        "rejection_fields": _REJECTION_FIELDS_V2,
        "success_disposition": "CONTRACT_BINDING_COMPLETE_NOT_SCORED",
        "success_reason": "CONTRACT_BINDING_VERIFIED",
        "true_claims": _TRUE_VALIDATION_CLAIMS_V2,
        "false_claims": _FALSE_VALIDATION_CLAIMS_V2,
        "empty_outputs": ("metric_results", "scored_rows", "gate_results"),
    },
    prefix="phase2b_formal_unsealed_prediction_scoring_contract_schema_v2_",
)

FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID: Final = stable_hash(
    {
        "version": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
        "schema_id": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID,
        "answer_row_schema_id": _ANSWER_ROW_SCHEMA_ID_V2,
        "answer_manifest_schema_id": _ANSWER_MANIFEST_SCHEMA_ID_V2,
        "answer_manifest_policy_id": _ANSWER_MANIFEST_POLICY_ID_V2,
        "dependencies": {
            "strict_schema_version": _STRICT_SCHEMA_VERSION_V2,
            "strict_schema_id": _STRICT_SCHEMA_ID_V2,
            "strict_policy_id": _STRICT_POLICY_ID_V2,
            "evaluator_version": _EVALUATOR_VERSION_V2,
            "evaluator_policy_id": _EVALUATOR_POLICY_ID_V2,
            "input_archive_version": _INPUT_ARCHIVE_VERSION_V2,
            "input_archive_policy_id": _INPUT_ARCHIVE_POLICY_ID_V2,
            "prediction_archive_version": _PREDICTION_ARCHIVE_VERSION_V2,
            "prediction_archive_policy_id": _PREDICTION_ARCHIVE_POLICY_ID_V2,
            "batch_policy_id": _BATCH_POLICY_ID_V2,
            "exact_freeze_id": _EXACT_FREEZE_ID_V2,
            "protocol_id": _PHASE2B_PROTOCOL_ID_V2,
        },
        "metric_specs": tuple(
            (
                name,
                kind.value,
                tuple(case_type.value for case_type in cases),
                count,
                rule,
                separate,
            )
            for name, kind, cases, count, rule, separate in _METRIC_SPECS_V2
        ),
        "set_valued_joint_rule": _SET_VALUED_JOINT_RULE_V2,
        "commitment_opening_formula": _COMMITMENT_OPENING_FORMULA_V2,
        "challenge_denominator_policy": _CHALLENGE_DENOMINATOR_POLICY_V2,
        "referenced_not_executed": {
            "overall_gates": _OVERALL_GATES_V2,
            "slice_gates": _SLICE_GATES_V2,
            "scale_regret_gate": _SCALE_REGRET_GATE_V2,
            "bootstrap": _BOOTSTRAP_REFERENCE_V2,
            "overall_metric_mapping": _OVERALL_GATE_METRIC_MAPPING_V2,
            "wilson_method": _WILSON_METHOD_V2,
            "wilson_semantics": _WILSON_SEMANTICS_V2,
            "wilson_confidence": ONE_SIDED_CONFIDENCE,
            "gate_inputs_implemented": False,
            "gate_results": (),
            "gates_executed": False,
            "bootstrap_evaluated": False,
        },
        "validation_order": (
            "exact_top_level_types",
            "exact_scalar_tuple_enum_bool_and_empty_output_closure",
            "answer_row_semantic_quota_and_partition_shape_closure",
            "local_public_root_and_manifest_recalculation",
            "cross_object_builtin_value_binding",
            "answer_manifest_nonself_hash_and_supplied_opening",
            "private_atomic_success_or_all_false_rejection",
        ),
        "forbidden": (
            "raw_prediction_archive_decode",
            "prediction_reading_or_metric_computation",
            "gate_or_bootstrap_execution",
            "answer_authority_or_reveal_timing_claim",
            "runtime_capacity_effect_or_c1_claim",
        ),
        "true_claims": _TRUE_VALIDATION_CLAIMS_V2,
        "false_claims": _FALSE_VALIDATION_CLAIMS_V2,
    },
    prefix="phase2b_formal_unsealed_prediction_scoring_contract_policy_v2_",
)


_METRIC_ISSUE_TOKEN_V2: Final = object()
_MANIFEST_ISSUE_TOKEN_V2: Final = object()
_CONTRACT_ISSUE_TOKEN_V2: Final = object()
_VALIDATION_ISSUE_TOKEN_V2: Final = object()
_REJECTION_ISSUE_TOKEN_V2: Final = object()


def _preflight_answer_row_v2(
    value: FormalUnsealedAnswerRowV2,
    *,
    allow_unissued_id: bool = False,
) -> tuple[
    str,
    Phase2BCaseType,
    PredictionDecisionV2,
    CanonicalFamilyId | None,
    tuple[RoleBinding, ...],
    tuple[str, ...],
    str,
]:
    if type(value) is not FormalUnsealedAnswerRowV2:
        raise TypeError("formal V2 answer row exact type drift")
    try:
        input_row_id = object.__getattribute__(value, "input_row_id")
        case_type = object.__getattribute__(value, "case_type")
        expected_decision = object.__getattribute__(value, "expected_decision")
        family = object.__getattribute__(value, "canonical_family_id")
        binding = object.__getattribute__(value, "binding")
        scales = object.__getattribute__(value, "admissible_scale_ids")
        answer_row_id = object.__getattribute__(value, "answer_row_id")
    except AttributeError as exc:
        raise ValueError("formal V2 answer row slot missing") from exc
    _digest_v2(
        input_row_id,
        prefix="phase2b_recognizer_input_row_v2_",
        name="formal V2 answer input row ID",
    )
    if type(case_type) is not Phase2BCaseType:
        raise TypeError("formal V2 answer case type exact enum drift")
    if type(expected_decision) is not PredictionDecisionV2:
        raise TypeError("formal V2 expected decision exact enum drift")
    if family is not None and type(family) is not CanonicalFamilyId:
        raise TypeError("formal V2 canonical family exact enum drift")
    if type(binding) is not tuple or len(binding) > _MAXIMUM_BINDINGS_V2:
        raise TypeError("formal V2 binding exact bounded tuple drift")
    binding_keys: list[tuple[str, str]] = []
    for item in binding:
        if type(item) is not RoleBinding:
            raise TypeError("formal V2 answer binding exact item drift")
        role_id = object.__getattribute__(item, "role_id")
        entity_id = object.__getattribute__(item, "entity_id")
        _uuid4_v2(role_id, name="formal V2 answer role ID")
        _uuid4_v2(entity_id, name="formal V2 answer entity ID")
        binding_keys.append((role_id, entity_id))
    if (
        binding_keys != sorted(binding_keys)
        or len({role for role, _ in binding_keys}) != len(binding_keys)
        or len({entity for _, entity in binding_keys}) != len(binding_keys)
    ):
        raise ValueError("formal V2 answer binding is not canonical injective")
    if type(scales) is not tuple or len(scales) > _MAXIMUM_SCALES_V2:
        raise TypeError("formal V2 answer scales exact bounded tuple drift")
    for item in scales:
        _uuid4_v2(item, name="formal V2 answer scale ID")
    if scales != tuple(sorted(set(scales))):
        raise ValueError("formal V2 answer scales are not sorted unique")
    if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
        coherent = (
            expected_decision is PredictionDecisionV2.ANSWER
            and type(family) is CanonicalFamilyId
            and bool(binding)
            and len(scales) == 1
        )
    elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
        coherent = (
            expected_decision is PredictionDecisionV2.ANSWER_SET
            and type(family) is CanonicalFamilyId
            and bool(binding)
            and len(scales) >= 2
        )
    else:
        coherent = (
            expected_decision is PredictionDecisionV2.ABSTAIN
            and family is None
            and binding == ()
            and scales == ()
        )
    if not coherent:
        raise ValueError("formal V2 answer row case/decision payload drift")
    if type(answer_row_id) is not str:
        raise TypeError("formal V2 answer row ID exact text drift")
    if allow_unissued_id and answer_row_id == "":
        pass
    else:
        _digest_v2(
            answer_row_id,
            prefix=_ANSWER_ROW_ID_PREFIX_V2,
            name="formal V2 answer row ID",
        )
    return (
        input_row_id,
        case_type,
        expected_decision,
        family,
        binding,
        scales,
        answer_row_id,
    )


def _finalize_answer_row_v2(
    preflight: tuple[
        str,
        Phase2BCaseType,
        PredictionDecisionV2,
        CanonicalFamilyId | None,
        tuple[RoleBinding, ...],
        tuple[str, ...],
        str,
    ],
) -> tuple[dict[str, object], FormalUnsealedAnswerRowV2]:
    (
        input_row_id,
        case_type,
        expected_decision,
        family,
        binding,
        scales,
        answer_row_id,
    ) = preflight
    expected_id = _answer_row_id_v2(
        input_row_id=input_row_id,
        case_type=case_type,
        expected_decision=expected_decision,
        canonical_family_id=family,
        binding=binding,
        admissible_scale_ids=scales,
    )
    if answer_row_id == "":
        answer_row_id = expected_id
    elif not _lexically_equal_v2(answer_row_id, expected_id):
        raise ValueError("formal V2 answer row content root drift")
    canonical_row = FormalUnsealedAnswerRowV2(
        input_row_id=input_row_id,
        case_type=case_type,
        expected_decision=expected_decision,
        canonical_family_id=family,
        binding=tuple(
            RoleBinding(role_id=item.role_id, entity_id=item.entity_id)
            for item in binding
        ),
        admissible_scale_ids=tuple(scales),
        answer_row_id=answer_row_id,
    )
    return ({
        **_answer_row_preimage_v2(
            input_row_id=input_row_id,
            case_type=case_type,
            expected_decision=expected_decision,
            canonical_family_id=family,
            binding=binding,
            admissible_scale_ids=scales,
        ),
        "answer_row_id": answer_row_id,
    }, canonical_row)


_AnswerRowPreflightV2 = tuple[
    str,
    Phase2BCaseType,
    PredictionDecisionV2,
    CanonicalFamilyId | None,
    tuple[RoleBinding, ...],
    tuple[str, ...],
    str,
]


def _preflight_answer_rows_v2(
    values: object,
    *,
    allow_unissued_ids: bool = False,
) -> tuple[_AnswerRowPreflightV2, ...]:
    if type(values) is not tuple or len(values) != _MAIN_COUNT_V2:
        raise ValueError("formal V2 answer manifest requires exact 720 rows")
    preflights: list[_AnswerRowPreflightV2] = []
    for value in values:
        preflights.append(
            _preflight_answer_row_v2(
                value,
                allow_unissued_id=allow_unissued_ids,
            )
        )
    input_ids = tuple(item[0] for item in preflights)
    if input_ids != tuple(sorted(input_ids)) or len(set(input_ids)) != _MAIN_COUNT_V2:
        raise ValueError("formal V2 answer rows are not sorted unique by input row")
    observed = tuple(
        (case_type, sum(item[1] is case_type for item in preflights))
        for case_type, _count in _CASE_TYPE_COUNTS_V2
    )
    if observed != _CASE_TYPE_COUNTS_V2:
        raise ValueError("formal V2 answer row case quota drift")
    return tuple(preflights)


def _finalize_answer_rows_v2(
    preflights: tuple[_AnswerRowPreflightV2, ...],
) -> tuple[tuple[FormalUnsealedAnswerRowV2, ...], tuple[dict[str, object], ...]]:
    mappings: list[dict[str, object]] = []
    rows: list[FormalUnsealedAnswerRowV2] = []
    for preflight in preflights:
        mapping, canonical_row = _finalize_answer_row_v2(preflight)
        mappings.append(mapping)
        rows.append(canonical_row)
    return tuple(rows), tuple(mappings)


def _validate_answer_rows_v2(
    values: object,
    *,
    allow_unissued_ids: bool = False,
) -> tuple[tuple[FormalUnsealedAnswerRowV2, ...], tuple[dict[str, object], ...]]:
    return _finalize_answer_rows_v2(
        _preflight_answer_rows_v2(
            values,
            allow_unissued_ids=allow_unissued_ids,
        )
    )


def _answer_manifest_preimage_v2(
    value: FormalUnsealedAnswerManifestV2,
    *,
    row_mappings: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "schema_id": value.schema_id,
        "policy_id": value.policy_id,
        "claim_level": value.claim_level,
        "exact_freeze_id": value.exact_freeze_id,
        "phase2b_protocol_id": value.phase2b_protocol_id,
        "execution_freeze_manifest_id": value.execution_freeze_manifest_id,
        "input_archive_id": value.input_archive_id,
        "input_archive_sha256": value.input_archive_sha256,
        "input_archive_version": value.input_archive_version,
        "input_archive_policy_id": value.input_archive_policy_id,
        "batch_id": value.batch_id,
        "batch_policy_id": value.batch_policy_id,
        "ordered_archive_input_row_ids_root": (
            value.ordered_archive_input_row_ids_root
        ),
        "main_row_ids_root": value.main_row_ids_root,
        "semantic_conflict_row_ids_root": value.semantic_conflict_row_ids_root,
        "partition_union_row_ids_root": value.partition_union_row_ids_root,
        "main_answer_rows": list(row_mappings),
        "main_answer_row_ids_root": value.main_answer_row_ids_root,
    }


def _answer_manifest_sha_v2(preimage: dict[str, object]) -> str:
    return hashlib.sha256(
        _ANSWER_MANIFEST_DOMAIN_V2
        + canonical_json(preimage).encode("utf-8")
    ).hexdigest()


def _preflight_answer_manifest_v2(
    value: FormalUnsealedAnswerManifestV2,
) -> tuple[_AnswerRowPreflightV2, ...]:
    if type(value) is not FormalUnsealedAnswerManifestV2:
        raise TypeError("formal V2 answer manifest exact type drift")
    try:
        schema_version = object.__getattribute__(value, "schema_version")
        schema_id = object.__getattribute__(value, "schema_id")
        policy_id = object.__getattribute__(value, "policy_id")
        claim_level = object.__getattribute__(value, "claim_level")
        exact_freeze_id = object.__getattribute__(value, "exact_freeze_id")
        protocol_id = object.__getattribute__(value, "phase2b_protocol_id")
        execution_freeze_id = object.__getattribute__(
            value, "execution_freeze_manifest_id"
        )
        input_archive_id = object.__getattribute__(value, "input_archive_id")
        input_sha = object.__getattribute__(value, "input_archive_sha256")
        input_version = object.__getattribute__(value, "input_archive_version")
        input_policy = object.__getattribute__(value, "input_archive_policy_id")
        batch_id = object.__getattribute__(value, "batch_id")
        batch_policy = object.__getattribute__(value, "batch_policy_id")
        ordered_root = object.__getattribute__(
            value, "ordered_archive_input_row_ids_root"
        )
        main_root = object.__getattribute__(value, "main_row_ids_root")
        conflict_root = object.__getattribute__(
            value, "semantic_conflict_row_ids_root"
        )
        union_root = object.__getattribute__(value, "partition_union_row_ids_root")
        raw_rows = object.__getattribute__(value, "main_answer_rows")
        answer_rows_root = object.__getattribute__(
            value, "main_answer_row_ids_root"
        )
        stored_sha = object.__getattribute__(value, "answer_manifest_sha256")
        manifest_id = object.__getattribute__(value, "answer_manifest_id")
    except AttributeError as exc:
        raise ValueError("formal V2 answer manifest slot missing") from exc
    for item, name in (
        (schema_version, "answer schema version"),
        (input_version, "input archive version"),
        (claim_level, "answer claim level"),
    ):
        _exact_text_v2(item, name=f"formal V2 {name}")
    for item, prefix, name in (
        (schema_id, "phase2b_formal_unsealed_answer_manifest_schema_v2_", "schema ID"),
        (policy_id, "phase2b_formal_unsealed_answer_manifest_policy_v2_", "policy ID"),
        (exact_freeze_id, "phase2b_exact_freeze_", "exact freeze ID"),
        (protocol_id, "phase2b_protocol_", "protocol ID"),
        (execution_freeze_id, "phase2b_execution_freeze_", "execution freeze ID"),
        (input_archive_id, "phase2b_recognizer_input_archive_v2_", "input archive ID"),
        (input_policy, "phase2b_recognizer_input_archive_policy_v2_", "input policy ID"),
        (batch_id, "phase2b_trusted_wire_batch_v2_", "batch ID"),
        (batch_policy, "phase2b_trusted_wire_batch_v2_policy_", "batch policy ID"),
        (ordered_root, "phase2b_prediction_input_rows_v2_", "ordered row root"),
        (main_root, "phase2b_unsealed_main_rows_v2_", "main row root"),
        (conflict_root, "phase2b_unsealed_semantic_conflict_rows_v2_", "challenge root"),
        (union_root, "phase2b_unsealed_partition_union_rows_v2_", "union root"),
        (answer_rows_root, _ANSWER_ROW_IDS_ROOT_PREFIX_V2, "answer row root"),
        (manifest_id, _ANSWER_MANIFEST_ID_PREFIX_V2, "answer manifest ID"),
    ):
        _digest_v2(item, prefix=prefix, name=f"formal V2 {name}")
    _hex64_v2(input_sha, name="formal V2 input archive SHA")
    _hex64_v2(stored_sha, name="formal V2 answer manifest SHA")
    if (
        schema_version != _ANSWER_MANIFEST_SCHEMA_VERSION_V2
        or schema_id != _ANSWER_MANIFEST_SCHEMA_ID_V2
        or policy_id != _ANSWER_MANIFEST_POLICY_ID_V2
        or claim_level != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL
        or exact_freeze_id != _EXACT_FREEZE_ID_V2
        or protocol_id != _PHASE2B_PROTOCOL_ID_V2
        or input_version != _INPUT_ARCHIVE_VERSION_V2
        or input_policy != _INPUT_ARCHIVE_POLICY_ID_V2
        or batch_policy != _BATCH_POLICY_ID_V2
    ):
        raise ValueError("formal V2 answer manifest frozen identity drift")
    return _preflight_answer_rows_v2(raw_rows)


def _finalize_answer_manifest_v2(
    value: FormalUnsealedAnswerManifestV2,
    row_preflights: tuple[_AnswerRowPreflightV2, ...],
) -> tuple[dict[str, object], tuple[str, ...]]:
    rows, row_mappings = _finalize_answer_rows_v2(row_preflights)
    input_ids = tuple(item.input_row_id for item in rows)
    answer_ids = tuple(item.answer_row_id for item in rows)
    expected_main_root = _main_row_ids_root_v2(input_ids)
    expected_answer_root = _answer_row_ids_root_v2(answer_ids)
    if (
        not _lexically_equal_v2(value.main_row_ids_root, expected_main_root)
        or not _lexically_equal_v2(
            value.main_answer_row_ids_root,
            expected_answer_root,
        )
    ):
        raise ValueError("formal V2 answer manifest row root drift")
    preimage = _answer_manifest_preimage_v2(value, row_mappings=row_mappings)
    expected_sha = _answer_manifest_sha_v2(preimage)
    if (
        not _lexically_equal_v2(value.answer_manifest_sha256, expected_sha)
        or not _lexically_equal_v2(
            value.answer_manifest_id,
            _ANSWER_MANIFEST_ID_PREFIX_V2 + expected_sha,
        )
    ):
        raise ValueError("formal V2 answer manifest content root drift")
    return preimage, input_ids


def _validate_answer_manifest_v2(
    value: FormalUnsealedAnswerManifestV2,
) -> tuple[dict[str, object], tuple[str, ...]]:
    return _finalize_answer_manifest_v2(
        value,
        _preflight_answer_manifest_v2(value),
    )


def _issue_metric_definition_v2(
    spec: tuple[
        str,
        FormalUnsealedMetricKindV2,
        tuple[Phase2BCaseType, ...],
        int,
        str,
        bool,
    ],
) -> FormalUnsealedMetricDefinitionV2:
    name, kind, case_types, denominator, success_rule, separate = spec
    _exact_text_v2(name, name="formal V2 metric name")
    if type(kind) is not FormalUnsealedMetricKindV2:
        raise TypeError("formal V2 metric kind drift")
    if type(case_types) is not tuple or not case_types or any(
        type(item) is not Phase2BCaseType for item in case_types
    ):
        raise TypeError("formal V2 metric denominator case types drift")
    if type(denominator) is not int or denominator <= 0:
        raise TypeError("formal V2 metric denominator drift")
    _exact_text_v2(success_rule, name="formal V2 metric success rule")
    if type(separate) is not bool:
        raise TypeError("formal V2 metric separate flag drift")
    preimage = {
        "metric_name": name,
        "metric_kind": kind.value,
        "denominator_case_types": [item.value for item in case_types],
        "expected_denominator": denominator,
        "success_rule": success_rule,
        "separately_reported": separate,
    }
    value = object.__new__(FormalUnsealedMetricDefinitionV2)
    for field_name, item in (
        ("metric_name", name),
        ("metric_kind", kind),
        ("denominator_case_types", case_types),
        ("expected_denominator", denominator),
        ("success_rule", success_rule),
        ("separately_reported", separate),
        (
            "metric_definition_id",
            _stable_primitive_id_v2(
                preimage,
                domain=_METRIC_DEFINITION_DOMAIN_V2,
                prefix=_METRIC_DEFINITION_ID_PREFIX_V2,
            ),
        ),
    ):
        object.__setattr__(value, field_name, item)
    return value


def _fresh_metric_definitions_v2() -> tuple[FormalUnsealedMetricDefinitionV2, ...]:
    """Issue an independent set so caller pollution cannot cross invocations."""

    return tuple(_issue_metric_definition_v2(spec) for spec in _METRIC_SPECS_V2)


_METRIC_DEFINITION_IDS_V2: Final = tuple(
    item.metric_definition_id for item in _fresh_metric_definitions_v2()
)


def _contract_preimage_v2() -> dict[str, object]:
    return {
        "version": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
        "schema_id": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID,
        "policy_id": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID,
        "claim_level": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        "answer_row_schema_id": _ANSWER_ROW_SCHEMA_ID_V2,
        "answer_manifest_schema_id": _ANSWER_MANIFEST_SCHEMA_ID_V2,
        "required_structural_receipt_type": "StrictRecognizerStructuralReceiptV2",
        "required_structural_evaluation_type": "UnsealedPredictionStructuralEvaluationV2",
        "required_partition_manifest_type": "UnsealedPredictionPartitionManifestV2",
        "main_row_count": _MAIN_COUNT_V2,
        "semantic_conflict_row_count": _SEMANTIC_CONFLICT_COUNT_V2,
        "case_type_counts": [
            [case_type.value, count] for case_type, count in _CASE_TYPE_COUNTS_V2
        ],
        "metric_definition_ids": list(_METRIC_DEFINITION_IDS_V2),
        "set_valued_joint_rule": _SET_VALUED_JOINT_RULE_V2,
        "commitment_opening_formula": _COMMITMENT_OPENING_FORMULA_V2,
        "challenge_denominator_policy": _CHALLENGE_DENOMINATOR_POLICY_V2,
        "overall_gate_definitions": _OVERALL_GATES_V2,
        "slice_gate_definitions": _SLICE_GATES_V2,
        "scale_regret_gate_definition": _SCALE_REGRET_GATE_V2,
        "bootstrap_reference": _BOOTSTRAP_REFERENCE_V2,
        "bootstrap_evaluated": False,
        "overall_gate_metric_mapping": _OVERALL_GATE_METRIC_MAPPING_V2,
        "wilson_method": _WILSON_METHOD_V2,
        "wilson_semantics": _WILSON_SEMANTICS_V2,
        "wilson_confidence": ONE_SIDED_CONFIDENCE,
        "gate_inputs_implemented": False,
        "gate_results": (),
        "gates_executed": False,
    }


def _issue_contract_v2() -> FormalUnsealedPredictionScoringContractV2:
    value = object.__new__(FormalUnsealedPredictionScoringContractV2)
    metric_definitions = _fresh_metric_definitions_v2()
    frozen = (
        ("version", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION),
        ("schema_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID),
        ("policy_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID),
        ("claim_level", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL),
        ("answer_row_schema_id", _ANSWER_ROW_SCHEMA_ID_V2),
        ("answer_manifest_schema_id", _ANSWER_MANIFEST_SCHEMA_ID_V2),
        ("required_structural_receipt_type", "StrictRecognizerStructuralReceiptV2"),
        (
            "required_structural_evaluation_type",
            "UnsealedPredictionStructuralEvaluationV2",
        ),
        ("required_partition_manifest_type", "UnsealedPredictionPartitionManifestV2"),
        ("main_row_count", _MAIN_COUNT_V2),
        ("semantic_conflict_row_count", _SEMANTIC_CONFLICT_COUNT_V2),
        ("case_type_counts", _CASE_TYPE_COUNTS_V2),
        ("metric_definitions", metric_definitions),
        ("set_valued_joint_rule", _SET_VALUED_JOINT_RULE_V2),
        ("commitment_opening_formula", _COMMITMENT_OPENING_FORMULA_V2),
        ("challenge_denominator_policy", _CHALLENGE_DENOMINATOR_POLICY_V2),
        ("overall_gate_definitions", _OVERALL_GATES_V2),
        ("slice_gate_definitions", _SLICE_GATES_V2),
        ("scale_regret_gate_definition", _SCALE_REGRET_GATE_V2),
        ("bootstrap_reference", _BOOTSTRAP_REFERENCE_V2),
        ("bootstrap_evaluated", False),
        ("overall_gate_metric_mapping", _OVERALL_GATE_METRIC_MAPPING_V2),
        ("wilson_method", _WILSON_METHOD_V2),
        ("wilson_semantics", _WILSON_SEMANTICS_V2),
        ("wilson_confidence", ONE_SIDED_CONFIDENCE),
        ("gate_inputs_implemented", False),
        ("gate_results", ()),
        ("gates_executed", False),
        (
            "contract_id",
            _stable_primitive_id_v2(
                _contract_preimage_v2(),
                domain=_SCORING_CONTRACT_DOMAIN_V2,
                prefix=_SCORING_CONTRACT_ID_PREFIX_V2,
            ),
        ),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    return value


def build_formal_unsealed_answer_manifest_v2(
    *,
    input_archive_id: str,
    input_archive_sha256: str,
    input_archive_version: str,
    input_archive_policy_id: str,
    batch_id: str,
    batch_policy_id: str,
    exact_freeze_id: str,
    phase2b_protocol_id: str,
    execution_freeze_manifest_id: str,
    ordered_archive_input_row_ids_root: str,
    main_row_ids_root: str,
    semantic_conflict_row_ids_root: str,
    partition_union_row_ids_root: str,
    main_answer_rows: tuple[FormalUnsealedAnswerRowV2, ...],
) -> FormalUnsealedAnswerManifestV2:
    """Build an evaluator-side manifest structurally intended for precommitment."""

    _digest_v2(
        input_archive_id,
        prefix="phase2b_recognizer_input_archive_v2_",
        name="formal V2 input archive ID",
    )
    _hex64_v2(input_archive_sha256, name="formal V2 input archive SHA")
    _exact_text_v2(input_archive_version, name="formal V2 input archive version")
    _digest_v2(
        input_archive_policy_id,
        prefix="phase2b_recognizer_input_archive_policy_v2_",
        name="formal V2 input archive policy ID",
    )
    _digest_v2(batch_id, prefix="phase2b_trusted_wire_batch_v2_", name="formal V2 batch ID")
    _digest_v2(
        batch_policy_id,
        prefix="phase2b_trusted_wire_batch_v2_policy_",
        name="formal V2 batch policy ID",
    )
    _digest_v2(exact_freeze_id, prefix="phase2b_exact_freeze_", name="formal V2 exact freeze ID")
    _digest_v2(phase2b_protocol_id, prefix="phase2b_protocol_", name="formal V2 protocol ID")
    _digest_v2(
        execution_freeze_manifest_id,
        prefix="phase2b_execution_freeze_",
        name="formal V2 execution freeze ID",
    )
    for item, prefix, name in (
        (ordered_archive_input_row_ids_root, "phase2b_prediction_input_rows_v2_", "ordered root"),
        (main_row_ids_root, "phase2b_unsealed_main_rows_v2_", "main root"),
        (semantic_conflict_row_ids_root, "phase2b_unsealed_semantic_conflict_rows_v2_", "challenge root"),
        (partition_union_row_ids_root, "phase2b_unsealed_partition_union_rows_v2_", "union root"),
    ):
        _digest_v2(item, prefix=prefix, name=f"formal V2 {name}")
    if (
        input_archive_version != _INPUT_ARCHIVE_VERSION_V2
        or input_archive_policy_id != _INPUT_ARCHIVE_POLICY_ID_V2
        or batch_policy_id != _BATCH_POLICY_ID_V2
        or exact_freeze_id != _EXACT_FREEZE_ID_V2
        or phase2b_protocol_id != _PHASE2B_PROTOCOL_ID_V2
    ):
        raise ValueError("formal V2 answer builder frozen identity drift")
    rows, row_mappings = _validate_answer_rows_v2(
        main_answer_rows,
        allow_unissued_ids=True,
    )
    input_ids = tuple(item.input_row_id for item in rows)
    if _main_row_ids_root_v2(input_ids) != main_row_ids_root:
        raise ValueError("formal V2 answer rows do not match main partition root")
    answer_root = _answer_row_ids_root_v2(
        tuple(item.answer_row_id for item in rows)
    )
    value = object.__new__(FormalUnsealedAnswerManifestV2)
    frozen = (
        ("schema_version", _ANSWER_MANIFEST_SCHEMA_VERSION_V2),
        ("schema_id", _ANSWER_MANIFEST_SCHEMA_ID_V2),
        ("policy_id", _ANSWER_MANIFEST_POLICY_ID_V2),
        ("claim_level", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL),
        ("exact_freeze_id", exact_freeze_id),
        ("phase2b_protocol_id", phase2b_protocol_id),
        ("execution_freeze_manifest_id", execution_freeze_manifest_id),
        ("input_archive_id", input_archive_id),
        ("input_archive_sha256", input_archive_sha256),
        ("input_archive_version", input_archive_version),
        ("input_archive_policy_id", input_archive_policy_id),
        ("batch_id", batch_id),
        ("batch_policy_id", batch_policy_id),
        ("ordered_archive_input_row_ids_root", ordered_archive_input_row_ids_root),
        ("main_row_ids_root", main_row_ids_root),
        ("semantic_conflict_row_ids_root", semantic_conflict_row_ids_root),
        ("partition_union_row_ids_root", partition_union_row_ids_root),
        ("main_answer_rows", rows),
        ("main_answer_row_ids_root", answer_root),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    preimage = _answer_manifest_preimage_v2(value, row_mappings=row_mappings)
    manifest_sha = _answer_manifest_sha_v2(preimage)
    object.__setattr__(value, "answer_manifest_sha256", manifest_sha)
    object.__setattr__(value, "answer_manifest_id", _ANSWER_MANIFEST_ID_PREFIX_V2 + manifest_sha)
    _validate_answer_manifest_v2(value)
    return value


def frozen_formal_unsealed_prediction_scoring_contract_v2(
) -> FormalUnsealedPredictionScoringContractV2:
    """Return the immutable definitions; no score or gate is evaluated."""

    return _issue_contract_v2()


def _preflight_receipt_v2(value: StrictRecognizerStructuralReceiptV2) -> None:
    if type(value) is not StrictRecognizerStructuralReceiptV2:
        raise TypeError("formal V2 validator needs exact strict receipt")
    disposition = object.__getattribute__(value, "disposition")
    if type(disposition) is not StrictRecognizerCliDispositionV2:
        raise TypeError("formal V2 receipt disposition exact enum drift")
    scalar_specs = (
        ("reason", None, None),
        ("schema_version", None, None),
        ("policy_id", "phase2b_strict_recognizer_cli_policy_v2_", None),
        ("claim_level", None, None),
        ("receipt_id", "phase2b_strict_recognizer_receipt_v2_", None),
        ("input_archive_id", "phase2b_recognizer_input_archive_v2_", None),
        ("input_archive_sha256", None, "hex"),
        ("input_archive_version", None, None),
        ("input_archive_policy_id", "phase2b_recognizer_input_archive_policy_v2_", None),
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_", None),
        ("prediction_archive_sha256", None, "hex"),
        ("prediction_archive_version", None, None),
        ("prediction_archive_policy_id", "phase2b_recognizer_prediction_archive_policy_v2_", None),
        ("batch_id", "phase2b_trusted_wire_batch_v2_", None),
        ("batch_policy_id", "phase2b_trusted_wire_batch_v2_policy_", None),
        ("run_context_id", "phase2b_public_prediction_run_context_v2_", None),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_", None),
        ("protocol_id", "phase2b_protocol_", None),
    )
    for name, prefix, kind in scalar_specs:
        item = object.__getattribute__(value, name)
        if kind == "hex":
            _hex64_v2(item, name=f"formal V2 receipt {name}")
        elif prefix is not None:
            _digest_v2(item, prefix=prefix, name=f"formal V2 receipt {name}")
        else:
            _exact_text_v2(item, name=f"formal V2 receipt {name}")
    case_count = object.__getattribute__(value, "case_count")
    if type(case_count) is not int or case_count != _TOTAL_COUNT_V2:
        raise ValueError("formal V2 receipt exact count drift")
    for name in _RECEIPT_TRUE_CLAIMS_V2:
        _exact_bool_v2(object.__getattribute__(value, name), expected=True, name=name)
    for name in _RECEIPT_FALSE_CLAIMS_V2:
        _exact_bool_v2(object.__getattribute__(value, name), expected=False, name=name)
    _exact_empty_tuple_v2(object.__getattribute__(value, "metric_results"), name="receipt metric results")
    _exact_empty_tuple_v2(object.__getattribute__(value, "scored_rows"), name="receipt scored rows")
    if (
        disposition is not StrictRecognizerCliDispositionV2.COMPLETE
        or value.reason != _RECEIPT_SUCCESS_REASON_V2
        or value.schema_version != _STRICT_SCHEMA_VERSION_V2
        or value.policy_id != _STRICT_POLICY_ID_V2
        or value.claim_level != _MECHANICS_CLAIM_LEVEL_V2
        or value.input_archive_version != _INPUT_ARCHIVE_VERSION_V2
        or value.input_archive_policy_id != _INPUT_ARCHIVE_POLICY_ID_V2
        or value.prediction_archive_version != _PREDICTION_ARCHIVE_VERSION_V2
        or value.prediction_archive_policy_id != _PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.batch_policy_id != _BATCH_POLICY_ID_V2
    ):
        raise ValueError("formal V2 receipt fixed identity drift")


def _finalize_receipt_v2(value: StrictRecognizerStructuralReceiptV2) -> None:
    mapping = value.to_mapping()
    if type(mapping) is not dict or set(mapping) != {
        item.name for item in fields(StrictRecognizerStructuralReceiptV2)
    }:
        raise ValueError("formal V2 receipt public mapping drift")


def _preflight_partition_v2(
    value: UnsealedPredictionPartitionManifestV2,
) -> tuple[str, ...]:
    if type(value) is not UnsealedPredictionPartitionManifestV2:
        raise TypeError("formal V2 validator needs exact partition manifest")
    main = object.__getattribute__(value, "main_row_ids")
    conflict = object.__getattribute__(value, "semantic_conflict_row_ids")
    if type(main) is not tuple or type(conflict) is not tuple:
        raise TypeError("formal V2 partition rows exact tuple drift")
    if len(main) != _MAIN_COUNT_V2 or len(conflict) != _SEMANTIC_CONFLICT_COUNT_V2:
        raise ValueError("formal V2 partition exact count drift")
    for name, values in (("main", main), ("semantic-conflict", conflict)):
        for item in values:
            _digest_v2(
                item,
                prefix="phase2b_recognizer_input_row_v2_",
                name=f"formal V2 partition {name} row ID",
            )
        if values != tuple(sorted(values)) or len(set(values)) != len(values):
            raise ValueError(f"formal V2 partition {name} rows are not sorted unique")
    if set(main) & set(conflict):
        raise ValueError("formal V2 partition overlap")
    union = tuple(sorted((*main, *conflict)))
    for name, prefix in (
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_"),
        ("prediction_archive_policy_id", "phase2b_recognizer_prediction_archive_policy_v2_"),
        ("exact_freeze_id", "phase2b_exact_freeze_"),
        ("evaluator_policy_id", "phase2b_unsealed_prediction_evaluator_policy_v2_"),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_"),
        ("semantic_conflict_row_ids_root", "phase2b_unsealed_semantic_conflict_rows_v2_"),
        ("partition_union_row_ids_root", "phase2b_unsealed_partition_union_rows_v2_"),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_"),
        ("manifest_id", "phase2b_unsealed_prediction_partition_v2_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix=prefix, name=f"formal V2 partition {name}")
    _exact_text_v2(value.prediction_archive_schema_version, name="formal V2 partition archive version")
    if (
        value.prediction_archive_schema_version != _PREDICTION_ARCHIVE_VERSION_V2
        or value.prediction_archive_policy_id != _PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.exact_freeze_id != _EXACT_FREEZE_ID_V2
        or value.evaluator_policy_id != _EVALUATOR_POLICY_ID_V2
    ):
        raise ValueError("formal V2 partition frozen identity drift")
    return union


def _finalize_partition_v2(
    value: UnsealedPredictionPartitionManifestV2,
    union: tuple[str, ...],
) -> None:
    main = value.main_row_ids
    conflict = value.semantic_conflict_row_ids
    main_root = _main_row_ids_root_v2(main)
    conflict_root = _semantic_conflict_row_ids_root_v2(conflict)
    union_root = _partition_union_row_ids_root_v2(union)
    ordered_root = value.ordered_archive_input_row_ids_root
    expected_manifest = _partition_manifest_id_v2(
        prediction_archive_id=value.prediction_archive_id,
        prediction_archive_schema_version=value.prediction_archive_schema_version,
        prediction_archive_policy_id=value.prediction_archive_policy_id,
        exact_freeze_id=value.exact_freeze_id,
        evaluator_policy_id=value.evaluator_policy_id,
        main_row_ids_root=main_root,
        semantic_conflict_row_ids_root=conflict_root,
        partition_union_row_ids_root=union_root,
        ordered_archive_input_row_ids_root=ordered_root,
    )
    if (
        value.main_row_ids_root != main_root
        or value.semantic_conflict_row_ids_root != conflict_root
        or value.partition_union_row_ids_root != union_root
        or value.manifest_id != expected_manifest
    ):
        raise ValueError("formal V2 partition identity or root drift")


def _validate_partition_v2(
    value: UnsealedPredictionPartitionManifestV2,
) -> tuple[str, ...]:
    union = _preflight_partition_v2(value)
    _finalize_partition_v2(value, union)
    return union


def _validate_evaluation_v2(value: UnsealedPredictionStructuralEvaluationV2) -> None:
    if type(value) is not UnsealedPredictionStructuralEvaluationV2:
        raise TypeError("formal V2 validator needs exact structural evaluation")
    disposition = object.__getattribute__(value, "disposition")
    if type(disposition) is not UnsealedPredictionEvaluationDispositionV2:
        raise TypeError("formal V2 evaluation disposition exact enum drift")
    for name, prefix in (
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_"),
        ("prediction_archive_policy_id", "phase2b_recognizer_prediction_archive_policy_v2_"),
        ("partition_manifest_id", "phase2b_unsealed_prediction_partition_v2_"),
        ("exact_freeze_id", "phase2b_exact_freeze_"),
        ("evaluator_policy_id", "phase2b_unsealed_prediction_evaluator_policy_v2_"),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_"),
        ("semantic_conflict_row_ids_root", "phase2b_unsealed_semantic_conflict_rows_v2_"),
        ("partition_union_row_ids_root", "phase2b_unsealed_partition_union_rows_v2_"),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix=prefix, name=f"formal V2 evaluation {name}")
    for name in ("reason", "prediction_archive_schema_version", "claim_level"):
        _exact_text_v2(object.__getattribute__(value, name), name=f"formal V2 evaluation {name}")
    for name, expected in (("main_count", 720), ("semantic_conflict_count", 240), ("total_count", 960)):
        item = object.__getattribute__(value, name)
        if type(item) is not int or item != expected:
            raise ValueError(f"formal V2 evaluation {name} drift")
    _exact_bool_v2(value.structural_completeness_verified, expected=True, name="structural completeness")
    for name in _EVALUATION_FALSE_CLAIMS_V2:
        _exact_bool_v2(object.__getattribute__(value, name), expected=False, name=name)
    _exact_empty_tuple_v2(value.metric_results, name="evaluation metric results")
    _exact_empty_tuple_v2(value.scored_rows, name="evaluation scored rows")
    if (
        disposition is not UnsealedPredictionEvaluationDispositionV2.STRUCTURALLY_COMPLETE_NOT_SCORED
        or value.reason != _EVALUATION_SUCCESS_REASON_V2
        or value.prediction_archive_schema_version != _PREDICTION_ARCHIVE_VERSION_V2
        or value.prediction_archive_policy_id != _PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.exact_freeze_id != _EXACT_FREEZE_ID_V2
        or value.evaluator_policy_id != _EVALUATOR_POLICY_ID_V2
        or value.claim_level != _MECHANICS_CLAIM_LEVEL_V2
    ):
        raise ValueError("formal V2 structural evaluation identity drift")


def _preflight_opening_v2(
    *,
    revealed_answer_manifest_sha256: object,
    answer_commitment_salt: object,
    salted_answer_commitment_sha256: object,
) -> tuple[str, str, str]:
    revealed = _hex64_v2(
        revealed_answer_manifest_sha256,
        name="formal V2 revealed answer manifest SHA",
    )
    supplied_commitment = _hex64_v2(
        salted_answer_commitment_sha256,
        name="formal V2 supplied answer commitment",
    )
    salt = _exact_text_v2(
        answer_commitment_salt,
        name="formal V2 answer commitment salt",
        maximum_bytes=_MAXIMUM_SALT_BYTES_V2,
        ascii_only=False,
    )
    if len(salt.encode("utf-8")) < 32:
        raise ValueError("formal V2 answer salt is shorter than 32 bytes")
    return revealed, salt, supplied_commitment


def _issue_validation_v2(
    *,
    receipt: StrictRecognizerStructuralReceiptV2,
    evaluation: UnsealedPredictionStructuralEvaluationV2,
    partition: UnsealedPredictionPartitionManifestV2,
    answer: FormalUnsealedAnswerManifestV2,
    commitment: str,
) -> FormalUnsealedPredictionScoringContractValidationV2:
    value = object.__new__(FormalUnsealedPredictionScoringContractValidationV2)
    frozen = (
        ("disposition", FormalUnsealedPredictionScoringContractDispositionV2.CONTRACT_BINDING_COMPLETE_NOT_SCORED),
        ("reason", FormalUnsealedPredictionScoringContractReasonV2.CONTRACT_BINDING_VERIFIED),
        ("version", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION),
        ("schema_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID),
        ("policy_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID),
        ("claim_level", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL),
        ("prediction_archive_id", evaluation.prediction_archive_id),
        ("partition_manifest_id", partition.manifest_id),
        ("structural_receipt_id", receipt.receipt_id),
        ("answer_manifest_id", answer.answer_manifest_id),
        ("answer_manifest_sha256", answer.answer_manifest_sha256),
        ("salted_answer_commitment_sha256", commitment),
        ("main_row_count", _MAIN_COUNT_V2),
        ("semantic_conflict_row_count", _SEMANTIC_CONFLICT_COUNT_V2),
        ("answerable_row_count", 240),
        ("main_answer_row_ids_root", answer.main_answer_row_ids_root),
        ("ordered_archive_input_row_ids_root", answer.ordered_archive_input_row_ids_root),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    for name in _TRUE_VALIDATION_CLAIMS_V2:
        object.__setattr__(value, name, True)
    for name in _FALSE_VALIDATION_CLAIMS_V2:
        object.__setattr__(value, name, False)
    object.__setattr__(value, "metric_definitions", _fresh_metric_definitions_v2())
    object.__setattr__(value, "metric_results", ())
    object.__setattr__(value, "scored_rows", ())
    return value


def _issue_rejection_v2(
    reason: FormalUnsealedPredictionScoringContractReasonV2,
) -> FormalUnsealedPredictionScoringContractRejectionV2:
    value = object.__new__(FormalUnsealedPredictionScoringContractRejectionV2)
    frozen = (
        ("disposition", FormalUnsealedPredictionScoringContractDispositionV2.REJECTED),
        ("reason", reason),
        ("version", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION),
        ("schema_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID),
        ("policy_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID),
        ("claim_level", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL),
        ("validation", None),
        ("metric_definitions", ()),
        ("metric_results", ()),
        ("scored_rows", ()),
        ("partial_output_published", False),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    for name in (*_TRUE_VALIDATION_CLAIMS_V2, *_FALSE_VALIDATION_CLAIMS_V2):
        object.__setattr__(value, name, False)
    return value


class _ValidationRejectedV2(Exception):
    def __init__(self, reason: FormalUnsealedPredictionScoringContractReasonV2) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _require_cross_bindings_v2(
    *,
    receipt: StrictRecognizerStructuralReceiptV2,
    evaluation: UnsealedPredictionStructuralEvaluationV2,
    partition: UnsealedPredictionPartitionManifestV2,
    answer: FormalUnsealedAnswerManifestV2,
    answer_input_ids: tuple[str, ...],
) -> None:
    groups = (
        (
            receipt.prediction_archive_id,
            evaluation.prediction_archive_id,
            partition.prediction_archive_id,
        ),
        (
            receipt.prediction_archive_version,
            evaluation.prediction_archive_schema_version,
            partition.prediction_archive_schema_version,
        ),
        (
            receipt.prediction_archive_policy_id,
            evaluation.prediction_archive_policy_id,
            partition.prediction_archive_policy_id,
        ),
        (evaluation.partition_manifest_id, partition.manifest_id),
        (
            evaluation.exact_freeze_id,
            partition.exact_freeze_id,
            answer.exact_freeze_id,
        ),
        (evaluation.evaluator_policy_id, partition.evaluator_policy_id),
        (
            evaluation.main_row_ids_root,
            partition.main_row_ids_root,
            answer.main_row_ids_root,
        ),
        (
            evaluation.semantic_conflict_row_ids_root,
            partition.semantic_conflict_row_ids_root,
            answer.semantic_conflict_row_ids_root,
        ),
        (
            evaluation.partition_union_row_ids_root,
            partition.partition_union_row_ids_root,
            answer.partition_union_row_ids_root,
        ),
        (
            evaluation.ordered_archive_input_row_ids_root,
            partition.ordered_archive_input_row_ids_root,
            answer.ordered_archive_input_row_ids_root,
        ),
        (answer.input_archive_id, receipt.input_archive_id),
        (answer.input_archive_sha256, receipt.input_archive_sha256),
        (answer.input_archive_version, receipt.input_archive_version),
        (answer.input_archive_policy_id, receipt.input_archive_policy_id),
        (answer.batch_id, receipt.batch_id),
        (answer.batch_policy_id, receipt.batch_policy_id),
        (
            answer.execution_freeze_manifest_id,
            receipt.execution_freeze_manifest_id,
        ),
        (answer.phase2b_protocol_id, receipt.protocol_id),
    )
    if any(
        any(not _lexically_equal_v2(group[0], item) for item in group[1:])
        for group in groups
    ) or not (
        type(receipt.case_count) is int
        and type(evaluation.total_count) is int
        and receipt.case_count == evaluation.total_count == _TOTAL_COUNT_V2
    ):
        raise _ValidationRejectedV2(
            FormalUnsealedPredictionScoringContractReasonV2.IDENTITY_MISMATCH
        )
    if not _lexically_equal_v2(answer_input_ids, partition.main_row_ids):
        raise _ValidationRejectedV2(
            FormalUnsealedPredictionScoringContractReasonV2.ROW_COVERAGE_MISMATCH
        )


def validate_formal_unsealed_prediction_scoring_contract_v2(
    *,
    structural_receipt: StrictRecognizerStructuralReceiptV2,
    structural_evaluation: UnsealedPredictionStructuralEvaluationV2,
    partition_manifest: UnsealedPredictionPartitionManifestV2,
    answer_manifest: FormalUnsealedAnswerManifestV2,
    revealed_answer_manifest_sha256: str,
    answer_commitment_salt: str,
    salted_answer_commitment_sha256: str,
) -> (
    FormalUnsealedPredictionScoringContractValidationV2
    | FormalUnsealedPredictionScoringContractRejectionV2
):
    """Validate contract bindings without inspecting or scoring predictions."""

    if (
        type(structural_receipt) is not StrictRecognizerStructuralReceiptV2
        or type(structural_evaluation) is not UnsealedPredictionStructuralEvaluationV2
        or type(partition_manifest) is not UnsealedPredictionPartitionManifestV2
        or type(answer_manifest) is not FormalUnsealedAnswerManifestV2
    ):
        return _issue_rejection_v2(
            FormalUnsealedPredictionScoringContractReasonV2.WRONG_INPUT_TYPE
        )
    try:
        partition_union: tuple[str, ...]
        answer_row_preflights: tuple[_AnswerRowPreflightV2, ...]
        revealed: str
        salt: str
        supplied_commitment: str
        try:
            _preflight_receipt_v2(structural_receipt)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.CROSS_VERSION_INPUT
            ) from exc
        try:
            _validate_evaluation_v2(structural_evaluation)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.STRUCTURAL_EVALUATION_NOT_COMPLETE
            ) from exc
        try:
            partition_union = _preflight_partition_v2(partition_manifest)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.PARTITION_MANIFEST_INVALID
            ) from exc
        try:
            answer_row_preflights = _preflight_answer_manifest_v2(answer_manifest)
        except (TypeError, ValueError, AttributeError) as exc:
            reason = (
                FormalUnsealedPredictionScoringContractReasonV2.CASE_TYPE_QUOTA_MISMATCH
                if "quota" in str(exc)
                else FormalUnsealedPredictionScoringContractReasonV2.ANSWER_MANIFEST_INVALID
            )
            raise _ValidationRejectedV2(reason) from exc
        try:
            revealed, salt, supplied_commitment = _preflight_opening_v2(
                revealed_answer_manifest_sha256=revealed_answer_manifest_sha256,
                answer_commitment_salt=answer_commitment_salt,
                salted_answer_commitment_sha256=salted_answer_commitment_sha256,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.ANSWER_COMMITMENT_OPENING_INVALID
            ) from exc

        try:
            _finalize_receipt_v2(structural_receipt)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.CROSS_VERSION_INPUT
            ) from exc
        try:
            _finalize_partition_v2(partition_manifest, partition_union)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.PARTITION_MANIFEST_INVALID
            ) from exc
        try:
            _preimage, answer_input_ids = _finalize_answer_manifest_v2(
                answer_manifest,
                answer_row_preflights,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.ANSWER_MANIFEST_INVALID
            ) from exc
        _require_cross_bindings_v2(
            receipt=structural_receipt,
            evaluation=structural_evaluation,
            partition=partition_manifest,
            answer=answer_manifest,
            answer_input_ids=answer_input_ids,
        )
        try:
            if not _lexically_equal_v2(
                revealed,
                answer_manifest.answer_manifest_sha256,
            ):
                raise ValueError("formal V2 revealed answer SHA drift")
            expected_commitment = _salted_answer_commitment_sha256(
                revealed,
                salt,
            )
            if not _lexically_equal_v2(supplied_commitment, expected_commitment):
                raise ValueError("formal V2 supplied answer opening drift")
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ValidationRejectedV2(
                FormalUnsealedPredictionScoringContractReasonV2.ANSWER_COMMITMENT_OPENING_INVALID
            ) from exc
        return _issue_validation_v2(
            receipt=structural_receipt,
            evaluation=structural_evaluation,
            partition=partition_manifest,
            answer=answer_manifest,
            commitment=supplied_commitment,
        )
    except _ValidationRejectedV2 as exc:
        return _issue_rejection_v2(exc.reason)
    except Exception:
        return _issue_rejection_v2(
            FormalUnsealedPredictionScoringContractReasonV2.INTERNAL_ERROR
        )


__all__ = [
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION",
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL",
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID",
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID",
    "FormalUnsealedMetricKindV2",
    "FormalUnsealedPredictionScoringContractDispositionV2",
    "FormalUnsealedPredictionScoringContractReasonV2",
    "FormalUnsealedAnswerRowV2",
    "FormalUnsealedAnswerManifestV2",
    "FormalUnsealedMetricDefinitionV2",
    "FormalUnsealedPredictionScoringContractV2",
    "FormalUnsealedPredictionScoringContractValidationV2",
    "FormalUnsealedPredictionScoringContractRejectionV2",
    "build_formal_unsealed_answer_manifest_v2",
    "frozen_formal_unsealed_prediction_scoring_contract_v2",
    "validate_formal_unsealed_prediction_scoring_contract_v2",
]
