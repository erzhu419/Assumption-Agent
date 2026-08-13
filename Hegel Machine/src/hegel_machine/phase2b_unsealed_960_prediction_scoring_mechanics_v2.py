"""Non-authoritative nine-metric scoring mechanics for supplied V2 bytes.

This module replays the public formal scoring-contract validator and the public
prediction-archive decoder.  It materializes nine frozen metric results and
720 main-row comparison records.  It does not execute a recognizer, score the
240 semantic-conflict challenge rows, evaluate a formal gate, establish any
authority or reveal timing, or provide actual-run/effect/C1 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
from typing import Final
from uuid import UUID

from .hashing import stable_hash
from .phase2b_formal_unsealed_prediction_scoring_contract_v2 import (
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID,
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID,
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
    FormalUnsealedAnswerManifestV2,
    FormalUnsealedAnswerRowV2,
    FormalUnsealedMetricDefinitionV2,
    FormalUnsealedMetricKindV2,
    FormalUnsealedPredictionScoringContractV2,
    FormalUnsealedPredictionScoringContractDispositionV2,
    FormalUnsealedPredictionScoringContractReasonV2,
    FormalUnsealedPredictionScoringContractValidationV2,
    frozen_formal_unsealed_prediction_scoring_contract_v2,
    validate_formal_unsealed_prediction_scoring_contract_v2,
)
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from .phase2b_protocol import Phase2BCaseType, frozen_phase2b_protocol
from .phase2b_recognizer_prediction_archive_v2 import (
    FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2,
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID,
    PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION,
    PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID,
    PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION,
    RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
    RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
    DecodedRecognizerPredictionArchiveV2,
    PredictionArchiveDispositionV2,
    PublicPredictionRunContextV2,
    PublicRecognizerPredictionRecordV2,
    decode_public_recognizer_prediction_archive_v2,
)
from .phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from .phase2b_recognizer_input_archive_v2 import (
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
)
from .phase2b_strict_recognizer_cli_v2 import (
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    STRICT_RECOGNIZER_CLI_V2_POLICY_ID,
    STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID,
    STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION,
    StrictRecognizerCliDispositionV2,
    StrictRecognizerStructuralReceiptV2,
)
from .phase2b_unsealed_prediction_evaluator_v2 import (
    UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2,
    UNSEALED_PREDICTION_EVALUATOR_V2_VERSION,
    UnsealedPredictionEvaluationDispositionV2,
    UnsealedPredictionPartitionManifestV2,
    UnsealedPredictionStructuralEvaluationV2,
)
from .phase2b_trusted_wire_batch_v2 import TRUSTED_WIRE_BATCH_V2_POLICY_ID
from .phase2b_wire import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBundle,
    PredictionDisposition,
    PredictionReason,
    RoleBinding,
)
from .phase2b_trusted_wire_v1 import encode_phase2b_jcs_profile_v1


UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION: Final = (
    "hegel-machine-phase2b-unsealed-960-prediction-scoring-mechanics/2"
)
UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_SCORING_MECHANICS_ONLY"
)
_ANSWER_MANIFEST_SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-formal-unsealed-answer-manifest/2"
)
_ANSWER_MANIFEST_SCHEMA_ID_V2: Final = (
    "phase2b_formal_unsealed_answer_manifest_schema_v2_"
    "3f427810029665a54854751b7d021a77c4d5f874b7df1992d50434b7108d32f0"
)
_ANSWER_MANIFEST_POLICY_ID_V2: Final = (
    "phase2b_formal_unsealed_answer_manifest_policy_v2_"
    "be684716aadb4bb6cced67348233d0c6ca78d7e0c98c6df2542bcc1787c50f1e"
)

_MAIN_COUNT_V2: Final = 720
_METRIC_ELIGIBLE_COUNT_V2: Final = 336
_CONTROL_WITHOUT_METRIC_COUNT_V2: Final = 384
_SEMANTIC_CONFLICT_COUNT_V2: Final = 240
_TOTAL_COUNT_V2: Final = 960
_METRIC_COUNT_V2: Final = 9
_MAXIMUM_TEXT_BYTES_V2: Final = 4_096
_MAXIMUM_BINDINGS_V2: Final = 64
_MAXIMUM_SCALES_V2: Final = 4_096

_METRIC_ROW_OUTCOME_ID_PREFIX_V2: Final = (
    "phase2b_unsealed_960_metric_row_outcome_v2_"
)
_MAIN_ROW_RESULT_ID_PREFIX_V2: Final = (
    "phase2b_unsealed_960_main_row_result_v2_"
)
_METRIC_RESULT_ID_PREFIX_V2: Final = (
    "phase2b_unsealed_960_metric_result_v2_"
)
_RESULT_ID_PREFIX_V2: Final = (
    "phase2b_unsealed_960_prediction_scoring_mechanics_v2_"
)
_INPUT_ROW_DOMAIN_V2: Final = b"HEGEL/PHASE2B/RECOGNIZER_INPUT_ROW/V2\x00"
_INPUT_ROWS_DOMAIN_V2: Final = b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00"
_RECORD_DOMAIN_V2: Final = b"HEGEL/PHASE2B/RECOGNIZER_PREDICTION_RECORD/V2\x00"
_ARCHIVE_DOMAIN_V2: Final = b"HEGEL/PHASE2B/RECOGNIZER_PREDICTION_ARCHIVE/V2\x00"
_BRIDGE_FAMILY_BY_KIND_V2: Final = dict(FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2)
_FROZEN_EXACT_FREEZE_V2: Final = frozen_phase2b_exact_freeze()
_FROZEN_PROTOCOL_V2: Final = frozen_phase2b_protocol()
_FROZEN_FORMAL_CONTRACT_V2: Final = (
    frozen_formal_unsealed_prediction_scoring_contract_v2()
)
_KIND_BY_CANONICAL_FAMILY_V2: Final = {
    family: kind for kind, family in _FROZEN_EXACT_FREEZE_V2.family_mapping
}
for _dependency_name_v2, _dependency_value_v2, _dependency_prefix_v2 in (
    (
        "exact freeze ID",
        _FROZEN_EXACT_FREEZE_V2.freeze_id,
        "phase2b_exact_freeze_",
    ),
    ("protocol ID", _FROZEN_PROTOCOL_V2.protocol_id, "phase2b_protocol_"),
    (
        "formal contract ID",
        _FROZEN_FORMAL_CONTRACT_V2.contract_id,
        "phase2b_formal_unsealed_prediction_scoring_contract_v2_",
    ),
):
    if (
        type(_dependency_value_v2) is not str
        or not _dependency_value_v2.startswith(_dependency_prefix_v2)
        or len(_dependency_value_v2.removeprefix(_dependency_prefix_v2)) != 64
        or any(
            item not in "0123456789abcdef"
            for item in _dependency_value_v2.removeprefix(_dependency_prefix_v2)
        )
    ):
        raise RuntimeError(f"mechanics V2 {_dependency_name_v2} identity drift")
_ALLOWED_ABSTAIN_REASONS_V2: Final = frozenset(
    {
        PredictionReason.NO_PASSING_CANDIDATE,
        PredictionReason.MULTIPLE_STRUCTURAL_MATCHES,
        PredictionReason.NONIDENTIFIABLE_SCALE,
        PredictionReason.INSUFFICIENT_EVIDENCE,
        PredictionReason.INSUFFICIENT_MARGIN,
        PredictionReason.INCOMPLETE_CANDIDATE_COVERAGE,
        PredictionReason.VERIFIER_ERROR,
        PredictionReason.RESOURCE_LIMIT,
    }
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
_DECODED_TRUE_CLAIMS_V2: Final = (
    "structural_archive_verified",
    "canonical_record_framing_verified",
    "record_schema_verified",
    "row_root_coverage_verified",
)
_DECODED_FALSE_CLAIMS_V2: Final = (
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)
_FORMAL_TRUE_CLAIMS_V2: Final = (
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
_FORMAL_FALSE_CLAIMS_V2: Final = (
    "challenge_in_main_denominator",
    "answer_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    *_RECEIPT_FALSE_CLAIMS_V2,
    "formal_gate_evaluation_performed",
    "metric_results_materialized",
    "scored_rows_materialized",
    "control_rejection_metrics_implemented",
    "slice_gate_metrics_implemented",
    "challenge_scoring_performed",
)

_ANSWERABLE_CASE_TYPES_V2: Final = frozenset(
    {
        Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
        Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
    }
)
_CONTROL_CASE_TYPES_V2: Final = frozenset(
    {
        Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE,
        Phase2BCaseType.BINDING_COUNTERFACTUAL,
        Phase2BCaseType.SCALE_COUNTERFACTUAL,
        Phase2BCaseType.SIGN_OR_INVARIANT_BREAK,
    }
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


class Unsealed960PredictionScoringDispositionV2(str, Enum):
    MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION = (
        "MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION"
    )
    REJECTED = "REJECTED"


class Unsealed960PredictionScoringReasonV2(str, Enum):
    CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE = (
        "CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    FORMAL_CONTRACT_REJECTED = "FORMAL_CONTRACT_REJECTED"
    PREDICTION_ARCHIVE_INVALID = "PREDICTION_ARCHIVE_INVALID"
    PREDICTION_ARCHIVE_BINDING_MISMATCH = (
        "PREDICTION_ARCHIVE_BINDING_MISMATCH"
    )
    MAIN_ROW_JOIN_MISMATCH = "MAIN_ROW_JOIN_MISMATCH"
    METRIC_DENOMINATOR_MISMATCH = "METRIC_DENOMINATOR_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_TRUE_RESULT_CLAIMS_V2: Final = (
    "canonical_prediction_archive_replay_verified",
    "formal_contract_validation_replayed",
    "supplied_answer_commitment_opening_verified",
    "prediction_archive_context_cross_binding_verified",
    "exact_main_720_row_join_verified",
    "semantic_conflict_240_excluded_from_metrics",
    "nine_metric_results_materialized",
    "exact_720_main_row_results_materialized",
    "supplied_archive_nine_metric_mechanics_performed",
)

_FALSE_RESULT_CLAIMS_V2: Final = (
    "challenge_in_main_denominator",
    "challenge_scoring_performed",
    "control_rejection_metrics_implemented",
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_metrics_implemented",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
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
    "actual_prediction_scoring_evidence",
    "effect_evidence",
    "c1_exit_evidence",
)


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960MetricRowOutcomeV2:
    metric_definition_id: str
    metric_name: str
    eligible: bool
    success: bool | None
    metric_row_outcome_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed 960 V2 metric row outcomes are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960MainRowResultV2:
    input_row_id: str
    prediction_record_id: str
    prediction_content_id: str
    answer_row_id: str
    case_type: Phase2BCaseType
    predicted_decision: PredictionDecisionV2
    expected_decision: PredictionDecisionV2
    predicted_canonical_family_id: CanonicalFamilyId | None
    expected_canonical_family_id: CanonicalFamilyId | None
    predicted_binding: tuple[RoleBinding, ...]
    expected_binding: tuple[RoleBinding, ...]
    predicted_admissible_scale_ids: tuple[str, ...]
    expected_admissible_scale_ids: tuple[str, ...]
    decision_exact: bool
    family_exact: bool | None
    binding_exact: bool | None
    scale_set_exact: bool | None
    joint_exact: bool | None
    metric_eligible: bool
    metric_outcomes: tuple[Unsealed960MetricRowOutcomeV2, ...]
    row_result_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed 960 V2 main-row results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960MetricResultV2:
    metric_definition_id: str
    metric_name: str
    metric_kind: FormalUnsealedMetricKindV2
    denominator_case_types: tuple[Phase2BCaseType, ...]
    expected_denominator: int
    observed_denominator: int
    success_count: int | None
    count_value: int | None
    success_rule: str
    separately_reported: bool
    metric_result_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed 960 V2 metric results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960PredictionScoringMechanicsV2:
    disposition: Unsealed960PredictionScoringDispositionV2
    reason: Unsealed960PredictionScoringReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    result_id: str
    prediction_archive_id: str
    prediction_archive_sha256: str
    prediction_archive_version: str
    prediction_archive_policy_id: str
    run_context_id: str
    input_archive_id: str
    input_archive_sha256: str
    batch_id: str
    execution_freeze_manifest_id: str
    protocol_id: str
    structural_receipt_id: str
    partition_manifest_id: str
    answer_manifest_id: str
    answer_manifest_sha256: str
    salted_answer_commitment_sha256: str
    formal_scoring_contract_id: str
    ordered_archive_input_row_ids_root: str
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    partition_union_row_ids_root: str
    main_answer_row_ids_root: str
    total_prediction_count: int
    main_row_result_count: int
    metric_eligible_main_row_count: int
    control_row_without_frozen_metric_count: int
    semantic_conflict_excluded_count: int
    canonical_prediction_archive_replay_verified: bool
    formal_contract_validation_replayed: bool
    supplied_answer_commitment_opening_verified: bool
    prediction_archive_context_cross_binding_verified: bool
    exact_main_720_row_join_verified: bool
    semantic_conflict_240_excluded_from_metrics: bool
    nine_metric_results_materialized: bool
    exact_720_main_row_results_materialized: bool
    supplied_archive_nine_metric_mechanics_performed: bool
    challenge_in_main_denominator: bool
    challenge_scoring_performed: bool
    control_rejection_metrics_implemented: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_metrics_implemented: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
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
    actual_prediction_scoring_evidence: bool
    effect_evidence: bool
    c1_exit_evidence: bool
    metric_results: tuple[Unsealed960MetricResultV2, ...]
    main_row_results: tuple[Unsealed960MainRowResultV2, ...]
    gate_results: tuple[()]
    scale_regret_result: None
    bootstrap_result: None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed 960 V2 mechanics results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960PredictionScoringRejectionV2:
    disposition: Unsealed960PredictionScoringDispositionV2
    reason: Unsealed960PredictionScoringReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    result: None
    metric_results: tuple[()]
    main_row_results: tuple[()]
    gate_results: tuple[()]
    scale_regret_result: None
    bootstrap_result: None
    partial_output_published: bool
    canonical_prediction_archive_replay_verified: bool
    formal_contract_validation_replayed: bool
    supplied_answer_commitment_opening_verified: bool
    prediction_archive_context_cross_binding_verified: bool
    exact_main_720_row_join_verified: bool
    semantic_conflict_240_excluded_from_metrics: bool
    nine_metric_results_materialized: bool
    exact_720_main_row_results_materialized: bool
    supplied_archive_nine_metric_mechanics_performed: bool
    challenge_in_main_denominator: bool
    challenge_scoring_performed: bool
    control_rejection_metrics_implemented: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_metrics_implemented: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
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
    actual_prediction_scoring_evidence: bool
    effect_evidence: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed 960 V2 rejections are privately issued")


_METRIC_ROW_OUTCOME_FIELDS_V2: Final = (
    "metric_definition_id",
    "metric_name",
    "eligible",
    "success",
    "metric_row_outcome_id",
)
_MAIN_ROW_RESULT_FIELDS_V2: Final = (
    "input_row_id",
    "prediction_record_id",
    "prediction_content_id",
    "answer_row_id",
    "case_type",
    "predicted_decision",
    "expected_decision",
    "predicted_canonical_family_id",
    "expected_canonical_family_id",
    "predicted_binding",
    "expected_binding",
    "predicted_admissible_scale_ids",
    "expected_admissible_scale_ids",
    "decision_exact",
    "family_exact",
    "binding_exact",
    "scale_set_exact",
    "joint_exact",
    "metric_eligible",
    "metric_outcomes",
    "row_result_id",
)
_METRIC_RESULT_FIELDS_V2: Final = (
    "metric_definition_id",
    "metric_name",
    "metric_kind",
    "denominator_case_types",
    "expected_denominator",
    "observed_denominator",
    "success_count",
    "count_value",
    "success_rule",
    "separately_reported",
    "metric_result_id",
)
_SUCCESS_IDENTITY_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "result_id",
    "prediction_archive_id",
    "prediction_archive_sha256",
    "prediction_archive_version",
    "prediction_archive_policy_id",
    "run_context_id",
    "input_archive_id",
    "input_archive_sha256",
    "batch_id",
    "execution_freeze_manifest_id",
    "protocol_id",
    "structural_receipt_id",
    "partition_manifest_id",
    "answer_manifest_id",
    "answer_manifest_sha256",
    "salted_answer_commitment_sha256",
    "formal_scoring_contract_id",
    "ordered_archive_input_row_ids_root",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "main_answer_row_ids_root",
    "total_prediction_count",
    "main_row_result_count",
    "metric_eligible_main_row_count",
    "control_row_without_frozen_metric_count",
    "semantic_conflict_excluded_count",
)
_SUCCESS_OUTPUT_FIELDS_V2: Final = (
    "metric_results",
    "main_row_results",
    "gate_results",
    "scale_regret_result",
    "bootstrap_result",
)
_SUCCESS_FIELDS_V2: Final = (
    *_SUCCESS_IDENTITY_FIELDS_V2,
    *_TRUE_RESULT_CLAIMS_V2,
    *_FALSE_RESULT_CLAIMS_V2,
    *_SUCCESS_OUTPUT_FIELDS_V2,
)
_REJECTION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "result",
    "metric_results",
    "main_row_results",
    "gate_results",
    "scale_regret_result",
    "bootstrap_result",
    "partial_output_published",
    *_TRUE_RESULT_CLAIMS_V2,
    *_FALSE_RESULT_CLAIMS_V2,
)

UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID: Final = stable_hash(
    {
        "version": UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION,
        "claim_level": UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL,
        "metric_row_outcome_fields": _METRIC_ROW_OUTCOME_FIELDS_V2,
        "main_row_result_fields": _MAIN_ROW_RESULT_FIELDS_V2,
        "metric_result_fields": _METRIC_RESULT_FIELDS_V2,
        "success_fields": _SUCCESS_FIELDS_V2,
        "rejection_fields": _REJECTION_FIELDS_V2,
    },
    prefix="phase2b_unsealed_960_prediction_scoring_mechanics_schema_v2_",
)
UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID: Final = stable_hash(
    {
        "version": UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION,
        "schema_id": UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID,
        "claim_level": UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL,
        "formal_contract": {
            "version": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
            "schema_id": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID,
            "policy_id": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID,
            "claim_level": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
            "contract_id": _FROZEN_FORMAL_CONTRACT_V2.contract_id,
            "fresh_public_validator_calls": 1,
        },
        "upstream_structural_dependencies": {
            "exact_freeze_id": _FROZEN_EXACT_FREEZE_V2.freeze_id,
            "phase2b_protocol_id": _FROZEN_PROTOCOL_V2.protocol_id,
            "strict_receipt_schema_version": STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION,
            "strict_receipt_schema_id": STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID,
            "strict_receipt_policy_id": STRICT_RECOGNIZER_CLI_V2_POLICY_ID,
            "evaluator_version": UNSEALED_PREDICTION_EVALUATOR_V2_VERSION,
            "evaluator_policy_id": UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2,
            "input_archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
            "input_archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
            "trusted_batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
            "prediction_context_schema_version": PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION,
            "prediction_context_schema_id": PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID,
            "prediction_record_schema_version": PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION,
            "prediction_record_schema_id": PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID,
        },
        "prediction_archive": {
            "version": RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
            "policy_id": RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
            "maximum_bytes": MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
            "fresh_public_decoder_calls": 1,
        },
        "independent_content_roots": {
            "input_row": {
                "domain_hex": _INPUT_ROW_DOMAIN_V2.hex(),
                "formula": "sha256(domain||accepted_JCS_exact_seven_root_mapping)",
                "prefix": "phase2b_recognizer_input_row_v2_",
            },
            "ordered_input_rows": {
                "domain_hex": _INPUT_ROWS_DOMAIN_V2.hex(),
                "formula": "sha256(domain||u32_count||repeated(u16_ascii_length||row_id))",
                "prefix": "phase2b_prediction_input_rows_v2_",
            },
            "prediction_content": {
                "formula": "stable_hash(exact_closed_prediction_bundle_mapping)",
                "prefix": "phase2b_prediction_",
            },
            "prediction_record": {
                "domain_hex": _RECORD_DOMAIN_V2.hex(),
                "formula": "sha256(domain||accepted_JCS_exact_closed_record_preimage)",
                "prefix": "phase2b_recognizer_prediction_record_v2_",
            },
            "prediction_context": {
                "formula": "stable_hash(exact_closed_context_preimage)",
                "prefix": "phase2b_public_prediction_run_context_v2_",
            },
            "prediction_archive": {
                "domain_hex": _ARCHIVE_DOMAIN_V2.hex(),
                "formula": "sha256(domain||exact_raw_archive_bytes)",
                "prefix": "phase2b_recognizer_prediction_archive_v2_",
            },
        },
        "validation_order": (
            "exact_top_level_types_and_raw_byte_cap",
            "complete_receipt_evaluation_partition_answer_opening_preflight",
            "one_fresh_public_formal_contract_validation",
            "complete_fresh_formal_validation_and_metric_definition_parity",
            "formal_validation_to_supplied_inputs_parity_before_decode",
            "one_fresh_public_prediction_archive_decode",
            "complete_context_all_960_records_and_three_columns_preflight",
            "independent_context_row_content_record_and_archive_root_recalculation",
            "cross_object_identity_root_and_set_binding",
            "input_row_id_lookup_then_exact_main_partition_order",
            "nine_integer_only_metric_results_and_720_main_row_results",
            "single_atomic_success_publication_or_all_false_rejection",
        ),
        "record_semantics": {
            "binding": "sorted_unique_role_ids_and_unique_entity_ids",
            "preflight": "ABSTAIN_RESOURCE_LIMIT_with_null_bridge_roots",
            "derived_run": "nonnull_bridge_compilation_and_decision_roots",
            "positive": "canonical_family_bridge_family_and_scale_cardinality_coherent",
            "abstain_reasons": tuple(
                sorted(item.value for item in _ALLOWED_ABSTAIN_REASONS_V2)
            ),
        },
        "counts": {
            "total": _TOTAL_COUNT_V2,
            "main_row_results": _MAIN_COUNT_V2,
            "metric_eligible": _METRIC_ELIGIBLE_COUNT_V2,
            "controls_without_frozen_metric": _CONTROL_WITHOUT_METRIC_COUNT_V2,
            "semantic_conflict_excluded": _SEMANTIC_CONFLICT_COUNT_V2,
            "metric_results": _METRIC_COUNT_V2,
        },
        "row_join": "exact_input_row_id_lookup_then_main_partition_order",
        "challenge": "semantic_conflict_240_excluded_from_all_metrics",
        "gates": "empty_referenced_not_evaluated",
        "scale_regret": "missing_not_evaluated",
        "bootstrap": "missing_not_evaluated",
        "result_identifiers": {
            "metric_row_outcome": "stable_hash(exact_public_fields_without_ID)",
            "main_row_result": "stable_hash(exact_public_fields_without_ID)",
            "metric_result": "stable_hash(exact_public_fields_without_ID)",
            "success": "stable_hash(exact_public_fields_without_result_ID)",
        },
        "true_claims": _TRUE_RESULT_CLAIMS_V2,
        "false_claims": _FALSE_RESULT_CLAIMS_V2,
    },
    prefix="phase2b_unsealed_960_prediction_scoring_mechanics_policy_v2_",
)


def _issue_rejection_v2(
    reason: Unsealed960PredictionScoringReasonV2,
) -> Unsealed960PredictionScoringRejectionV2:
    if type(reason) is not Unsealed960PredictionScoringReasonV2:
        reason = Unsealed960PredictionScoringReasonV2.INTERNAL_ERROR
    value = object.__new__(Unsealed960PredictionScoringRejectionV2)
    frozen = (
        ("disposition", Unsealed960PredictionScoringDispositionV2.REJECTED),
        ("reason", reason),
        ("version", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION),
        ("schema_id", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID),
        ("claim_level", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL),
        ("result", None),
        ("metric_results", ()),
        ("main_row_results", ()),
        ("gate_results", ()),
        ("scale_regret_result", None),
        ("bootstrap_result", None),
        ("partial_output_published", False),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    for name in (*_TRUE_RESULT_CLAIMS_V2, *_FALSE_RESULT_CLAIMS_V2):
        object.__setattr__(value, name, False)
    return value


def _assert_field_manifests_v2() -> None:
    actual = tuple(
        tuple(item.name for item in fields(value_type))
        for value_type in (
            Unsealed960MetricRowOutcomeV2,
            Unsealed960MainRowResultV2,
            Unsealed960MetricResultV2,
            Unsealed960PredictionScoringMechanicsV2,
            Unsealed960PredictionScoringRejectionV2,
        )
    )
    expected = (
        _METRIC_ROW_OUTCOME_FIELDS_V2,
        _MAIN_ROW_RESULT_FIELDS_V2,
        _METRIC_RESULT_FIELDS_V2,
        _SUCCESS_FIELDS_V2,
        _REJECTION_FIELDS_V2,
    )
    if actual != expected:
        raise RuntimeError("unsealed 960 V2 public field manifest drift")


_assert_field_manifests_v2()


class _ScoringRejectedV2(Exception):
    def __init__(self, reason: Unsealed960PredictionScoringReasonV2) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _reject_v2(reason: Unsealed960PredictionScoringReasonV2) -> None:
    raise _ScoringRejectedV2(reason)


def _text_v2(value: object, name: str, *, maximum: int = _MAXIMUM_TEXT_BYTES_V2) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exact bounded text drift")
    return value


def _ascii_v2(value: object, name: str) -> str:
    text = _text_v2(value, name)
    if not text.isascii():
        raise ValueError(f"{name} ASCII drift")
    return text


def _digest_v2(value: object, prefix: str, name: str) -> str:
    text = _ascii_v2(value, name)
    suffix = text.removeprefix(prefix)
    if not text.startswith(prefix) or len(suffix) != 64 or any(
        item not in "0123456789abcdef" for item in suffix
    ):
        raise ValueError(f"{name} digest drift")
    return text


def _hex_v2(value: object, name: str) -> str:
    text = _ascii_v2(value, name)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise ValueError(f"{name} SHA drift")
    return text


def _uuid_v2(value: object, name: str) -> str:
    text = _ascii_v2(value, name)
    parsed = UUID(text)
    if parsed.version != 4 or str(parsed) != text:
        raise ValueError(f"{name} UUIDv4 drift")
    return text


def _bool_claims_v2(value: object, names: tuple[str, ...], expected: bool) -> None:
    for name in names:
        item = object.__getattribute__(value, name)
        if type(item) is not bool or item is not expected:
            raise ValueError(f"{name} claim drift")


def _empty_tuple_v2(value: object, name: str) -> None:
    if type(value) is not tuple or value != ():
        raise ValueError(f"{name} must be exact empty tuple")


def _binding_v2(value: object, name: str) -> tuple[RoleBinding, ...]:
    if type(value) is not tuple or len(value) > _MAXIMUM_BINDINGS_V2:
        raise TypeError(f"{name} bounded tuple drift")
    keys: list[tuple[str, str]] = []
    for item in value:
        if type(item) is not RoleBinding:
            raise TypeError(f"{name} item type drift")
        keys.append(
            (
                _uuid_v2(object.__getattribute__(item, "role_id"), f"{name} role"),
                _uuid_v2(object.__getattribute__(item, "entity_id"), f"{name} entity"),
            )
        )
    role_ids = tuple(item[0] for item in keys)
    entity_ids = tuple(item[1] for item in keys)
    if (
        keys != sorted(keys)
        or len(set(keys)) != len(keys)
        or len(set(role_ids)) != len(role_ids)
        or len(set(entity_ids)) != len(entity_ids)
    ):
        raise ValueError(f"{name} canonical order drift")
    return value


def _scales_v2(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) > _MAXIMUM_SCALES_V2:
        raise TypeError(f"{name} bounded tuple drift")
    result = tuple(_uuid_v2(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} canonical order drift")
    return result


def _preflight_receipt_v2(value: object) -> StrictRecognizerStructuralReceiptV2:
    if type(value) is not StrictRecognizerStructuralReceiptV2:
        raise TypeError("mechanics V2 receipt exact type drift")
    if type(value.disposition) is not StrictRecognizerCliDispositionV2:
        raise TypeError("mechanics V2 receipt disposition type drift")
    for name, prefix in (
        ("policy_id", "phase2b_strict_recognizer_cli_policy_v2_"),
        ("receipt_id", "phase2b_strict_recognizer_receipt_v2_"),
        ("input_archive_id", "phase2b_recognizer_input_archive_v2_"),
        ("input_archive_policy_id", "phase2b_recognizer_input_archive_policy_v2_"),
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_"),
        ("prediction_archive_policy_id", "phase2b_recognizer_prediction_archive_policy_v2_"),
        ("batch_id", "phase2b_trusted_wire_batch_v2_"),
        ("batch_policy_id", "phase2b_trusted_wire_batch_v2_policy_"),
        ("run_context_id", "phase2b_public_prediction_run_context_v2_"),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_"),
        ("protocol_id", "phase2b_protocol_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix, f"receipt {name}")
    _hex_v2(value.input_archive_sha256, "receipt input archive SHA")
    _hex_v2(value.prediction_archive_sha256, "receipt prediction archive SHA")
    for name in ("reason", "schema_version", "claim_level", "input_archive_version", "prediction_archive_version"):
        _ascii_v2(object.__getattribute__(value, name), f"receipt {name}")
    if type(value.case_count) is not int or value.case_count != _TOTAL_COUNT_V2:
        raise ValueError("mechanics V2 receipt count drift")
    _bool_claims_v2(value, _RECEIPT_TRUE_CLAIMS_V2, True)
    _bool_claims_v2(value, _RECEIPT_FALSE_CLAIMS_V2, False)
    _empty_tuple_v2(value.metric_results, "receipt metric results")
    _empty_tuple_v2(value.scored_rows, "receipt scored rows")
    if (
        value.disposition is not StrictRecognizerCliDispositionV2.COMPLETE
        or value.reason != "strict_v2_structural_input_output_binding_complete"
        or value.schema_version != STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION
        or value.policy_id != STRICT_RECOGNIZER_CLI_V2_POLICY_ID
        or value.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        or value.input_archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or value.input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or value.prediction_archive_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or value.prediction_archive_policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.protocol_id != _FROZEN_PROTOCOL_V2.protocol_id
        or STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID.removeprefix("phase2b_strict_recognizer_cli_schema_v2_") == ""
    ):
        raise ValueError("mechanics V2 receipt identity drift")
    return value


def _row_ids_v2(value: object, count: int, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != count:
        raise TypeError(f"{name} tuple/count drift")
    result = tuple(
        _digest_v2(item, "phase2b_recognizer_input_row_v2_", name)
        for item in value
    )
    if len(set(result)) != count:
        raise ValueError(f"{name} repeats rows")
    return result


def _preflight_evaluation_v2(value: object) -> UnsealedPredictionStructuralEvaluationV2:
    if type(value) is not UnsealedPredictionStructuralEvaluationV2:
        raise TypeError("mechanics V2 evaluation exact type drift")
    if type(value.disposition) is not UnsealedPredictionEvaluationDispositionV2:
        raise TypeError("mechanics V2 evaluation disposition type drift")
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
        _digest_v2(object.__getattribute__(value, name), prefix, f"evaluation {name}")
    for name in ("reason", "prediction_archive_schema_version", "claim_level"):
        _ascii_v2(object.__getattribute__(value, name), f"evaluation {name}")
    for name, expected in (("main_count", 720), ("semantic_conflict_count", 240), ("total_count", 960)):
        item = object.__getattribute__(value, name)
        if type(item) is not int or item != expected:
            raise ValueError(f"evaluation {name} drift")
    if type(value.structural_completeness_verified) is not bool or not value.structural_completeness_verified:
        raise ValueError("evaluation structural completeness drift")
    _bool_claims_v2(value, _EVALUATION_FALSE_CLAIMS_V2, False)
    _empty_tuple_v2(value.metric_results, "evaluation metric results")
    _empty_tuple_v2(value.scored_rows, "evaluation scored rows")
    if (
        value.disposition is not UnsealedPredictionEvaluationDispositionV2.STRUCTURALLY_COMPLETE_NOT_SCORED
        or value.reason != "sorted_disjoint_exhaustive_720_240_same_v2_archive_row_set_and_ordered_root"
        or value.prediction_archive_schema_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or value.prediction_archive_policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.evaluator_policy_id != UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2
        or value.exact_freeze_id != _FROZEN_EXACT_FREEZE_V2.freeze_id
        or value.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        or UNSEALED_PREDICTION_EVALUATOR_V2_VERSION != "hegel-machine-phase2b-unsealed-prediction-evaluator/2"
    ):
        raise ValueError("evaluation identity drift")
    return value


def _preflight_partition_v2(value: object) -> UnsealedPredictionPartitionManifestV2:
    if type(value) is not UnsealedPredictionPartitionManifestV2:
        raise TypeError("mechanics V2 partition exact type drift")
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
        _digest_v2(object.__getattribute__(value, name), prefix, f"partition {name}")
    _ascii_v2(value.prediction_archive_schema_version, "partition archive version")
    main = _row_ids_v2(value.main_row_ids, _MAIN_COUNT_V2, "partition main rows")
    challenge = _row_ids_v2(
        value.semantic_conflict_row_ids,
        _SEMANTIC_CONFLICT_COUNT_V2,
        "partition challenge rows",
    )
    if main != tuple(sorted(main)) or challenge != tuple(sorted(challenge)) or set(main) & set(challenge):
        raise ValueError("partition order/disjointness drift")
    if (
        value.prediction_archive_schema_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or value.prediction_archive_policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.evaluator_policy_id != UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2
        or value.exact_freeze_id != _FROZEN_EXACT_FREEZE_V2.freeze_id
    ):
        raise ValueError("partition identity drift")
    return value


def _preflight_answer_row_v2(value: object) -> None:
    if type(value) is not FormalUnsealedAnswerRowV2:
        raise TypeError("mechanics V2 answer row exact type drift")
    _digest_v2(object.__getattribute__(value, "input_row_id"), "phase2b_recognizer_input_row_v2_", "answer row input ID")
    case_type = object.__getattribute__(value, "case_type")
    decision = object.__getattribute__(value, "expected_decision")
    family = object.__getattribute__(value, "canonical_family_id")
    binding = _binding_v2(object.__getattribute__(value, "binding"), "answer binding")
    scales = _scales_v2(object.__getattribute__(value, "admissible_scale_ids"), "answer scales")
    _digest_v2(object.__getattribute__(value, "answer_row_id"), "phase2b_formal_unsealed_answer_row_v2_", "answer row ID")
    if type(case_type) is not Phase2BCaseType or type(decision) is not PredictionDecisionV2:
        raise TypeError("answer row enum type drift")
    if family is not None and type(family) is not CanonicalFamilyId:
        raise TypeError("answer row family type drift")
    if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
        coherent = decision is PredictionDecisionV2.ANSWER and type(family) is CanonicalFamilyId and bool(binding) and len(scales) == 1
    elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
        coherent = decision is PredictionDecisionV2.ANSWER_SET and type(family) is CanonicalFamilyId and bool(binding) and len(scales) >= 2
    else:
        coherent = decision is PredictionDecisionV2.ABSTAIN and family is None and binding == () and scales == ()
    if not coherent:
        raise ValueError("answer row semantic drift")


def _preflight_answer_v2(value: object) -> FormalUnsealedAnswerManifestV2:
    if type(value) is not FormalUnsealedAnswerManifestV2:
        raise TypeError("mechanics V2 answer manifest exact type drift")
    for name, prefix in (
        ("schema_id", "phase2b_formal_unsealed_answer_manifest_schema_v2_"),
        ("policy_id", "phase2b_formal_unsealed_answer_manifest_policy_v2_"),
        ("exact_freeze_id", "phase2b_exact_freeze_"),
        ("phase2b_protocol_id", "phase2b_protocol_"),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_"),
        ("input_archive_id", "phase2b_recognizer_input_archive_v2_"),
        ("input_archive_policy_id", "phase2b_recognizer_input_archive_policy_v2_"),
        ("batch_id", "phase2b_trusted_wire_batch_v2_"),
        ("batch_policy_id", "phase2b_trusted_wire_batch_v2_policy_"),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_"),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_"),
        ("semantic_conflict_row_ids_root", "phase2b_unsealed_semantic_conflict_rows_v2_"),
        ("partition_union_row_ids_root", "phase2b_unsealed_partition_union_rows_v2_"),
        ("main_answer_row_ids_root", "phase2b_formal_unsealed_answer_rows_v2_"),
        ("answer_manifest_id", "phase2b_formal_unsealed_answer_manifest_v2_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix, f"answer {name}")
    _hex_v2(value.input_archive_sha256, "answer input SHA")
    _hex_v2(value.answer_manifest_sha256, "answer manifest SHA")
    for name in ("schema_version", "claim_level", "input_archive_version"):
        _ascii_v2(object.__getattribute__(value, name), f"answer {name}")
    rows = object.__getattribute__(value, "main_answer_rows")
    if type(rows) is not tuple or len(rows) != _MAIN_COUNT_V2:
        raise TypeError("answer rows tuple/count drift")
    for row in rows:
        _preflight_answer_row_v2(row)
    row_ids = tuple(object.__getattribute__(row, "input_row_id") for row in rows)
    if row_ids != tuple(sorted(set(row_ids))):
        raise ValueError("answer row order/uniqueness drift")
    observed_counts = tuple(
        (
            case_type,
            sum(
                object.__getattribute__(row, "case_type") is case_type
                for row in rows
            ),
        )
        for case_type, _expected in _CASE_TYPE_COUNTS_V2
    )
    if observed_counts != _CASE_TYPE_COUNTS_V2:
        raise ValueError("answer row case quota drift")
    if (
        value.schema_version != _ANSWER_MANIFEST_SCHEMA_VERSION_V2
        or value.schema_id != _ANSWER_MANIFEST_SCHEMA_ID_V2
        or value.policy_id != _ANSWER_MANIFEST_POLICY_ID_V2
    ):
        raise ValueError("answer manifest schema/policy identity drift")
    if (
        value.claim_level != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL
        or value.exact_freeze_id != _FROZEN_EXACT_FREEZE_V2.freeze_id
        or value.input_archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or value.input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or value.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or value.phase2b_protocol_id != _FROZEN_PROTOCOL_V2.protocol_id
    ):
        raise ValueError("answer manifest frozen identity drift")
    return value


def _preflight_opening_v2(revealed: object, salt: object, commitment: object) -> None:
    _hex_v2(revealed, "revealed answer manifest SHA")
    text = _text_v2(salt, "answer commitment salt")
    if len(text.encode("utf-8")) < 32:
        raise ValueError("answer commitment salt too short")
    _hex_v2(commitment, "salted answer commitment SHA")


def _preflight_prediction_bundle_v2(value: object) -> PredictionBundle:
    if type(value) is not PredictionBundle:
        raise TypeError("prediction bundle exact type drift")
    if type(value.disposition) is not PredictionDisposition or type(value.reason) is not PredictionReason:
        raise TypeError("prediction bundle enum type drift")
    for name in ("schema_version", "bundle_id", "input_root_sha256", "protocol_sha256", "freeze_manifest_sha256"):
        item = object.__getattribute__(value, name)
        if name == "bundle_id":
            _uuid_v2(item, "prediction bundle ID")
        elif name.endswith("sha256"):
            _hex_v2(item, f"prediction {name}")
        else:
            _ascii_v2(item, f"prediction {name}")
    family = object.__getattribute__(value, "family_id")
    if family is not None:
        _uuid_v2(family, "prediction family ID")
    binding = _binding_v2(object.__getattribute__(value, "binding"), "prediction binding")
    scales = _scales_v2(object.__getattribute__(value, "admissible_scale_ids"), "prediction scales")
    if value.schema_version != PREDICTION_SCHEMA_VERSION:
        raise ValueError("prediction schema drift")
    if value.disposition is PredictionDisposition.UNIQUE_MATCH:
        if family is None or not binding or not scales or value.reason is not PredictionReason.UNIQUE_STRUCTURAL_MATCH:
            raise ValueError("positive prediction payload drift")
    elif (
        family is not None
        or binding
        or scales
        or value.reason not in _ALLOWED_ABSTAIN_REASONS_V2
    ):
        raise ValueError("abstention prediction payload drift")
    return value


def _prediction_mapping_v2(value: PredictionBundle) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "bundle_id": value.bundle_id,
        "input_root_sha256": value.input_root_sha256,
        "protocol_sha256": value.protocol_sha256,
        "freeze_manifest_sha256": value.freeze_manifest_sha256,
        "disposition": value.disposition.value,
        "reason": value.reason.value,
        "family_id": value.family_id,
        "binding": [
            {"role_id": item.role_id, "entity_id": item.entity_id}
            for item in value.binding
        ],
        "admissible_scale_ids": list(value.admissible_scale_ids),
    }


def _record_mapping_v2(value: PublicRecognizerPredictionRecordV2) -> dict[str, object]:
    return {
        "bridge_compilation_id": value.bridge_compilation_id,
        "bridge_decision_id": value.bridge_decision_id,
        "bridge_outcome_id": value.bridge_outcome_id,
        "canonical_family_id": None if value.canonical_family_id is None else value.canonical_family_id.value,
        "decision": value.decision.value,
        "input_authority_content_id": value.input_authority_content_id,
        "input_envelope_id": value.input_envelope_id,
        "input_namespace_audit_id": value.input_namespace_audit_id,
        "input_padding_sha256": value.input_padding_sha256,
        "input_payload_sha256": value.input_payload_sha256,
        "input_public_registry_id": value.input_public_registry_id,
        "input_row_id": value.input_row_id,
        "input_transform_result_id": value.input_transform_result_id,
        "prediction": _prediction_mapping_v2(value.prediction),
        "prediction_content_id": value.prediction_content_id,
        "run_context_id": value.run_context_id,
        "schema_version": value.schema_version,
    }


def _finalize_record_roots_v2(value: PublicRecognizerPredictionRecordV2) -> None:
    prediction_mapping = _prediction_mapping_v2(value.prediction)
    expected_content = stable_hash(prediction_mapping, prefix="phase2b_prediction_")
    if value.prediction_content_id != expected_content:
        raise ValueError("record prediction content root drift")
    payload = encode_phase2b_jcs_profile_v1(_record_mapping_v2(value))
    expected_record = "phase2b_recognizer_prediction_record_v2_" + hashlib.sha256(
        _RECORD_DOMAIN_V2 + payload
    ).hexdigest()
    if value.record_id != expected_record:
        raise ValueError("prediction record content root drift")


def _preflight_record_v2(value: object) -> PublicRecognizerPredictionRecordV2:
    if type(value) is not PublicRecognizerPredictionRecordV2:
        raise TypeError("prediction record exact type drift")
    if type(value.decision) is not PredictionDecisionV2:
        raise TypeError("prediction record decision type drift")
    if value.canonical_family_id is not None and type(value.canonical_family_id) is not CanonicalFamilyId:
        raise TypeError("prediction record family type drift")
    for name, prefix in (
        ("input_authority_content_id", "phase2b_public_transform_evidence_"),
        ("input_envelope_id", "phase2b_trusted_envelope_v2_"),
        ("input_namespace_audit_id", "phase2b_namespace_audit_v2_"),
        ("input_public_registry_id", "phase2b_public_recognizer_registry_v2_"),
        ("input_row_id", "phase2b_recognizer_input_row_v2_"),
        ("input_transform_result_id", "phase2b_exact_transform_result_"),
        ("prediction_content_id", "phase2b_prediction_"),
        ("record_id", "phase2b_recognizer_prediction_record_v2_"),
        ("run_context_id", "phase2b_public_prediction_run_context_v2_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix, f"record {name}")
    bridge_outcome = _ascii_v2(value.bridge_outcome_id, "record bridge outcome ID")
    if not (
        bridge_outcome.startswith("phase2b_exact_derived_preflight_v2_")
        or bridge_outcome.startswith("phase2b_exact_derived_run_")
    ):
        raise ValueError("record bridge outcome prefix drift")
    suffix = bridge_outcome.rsplit("_", 1)[-1]
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError("record bridge outcome digest drift")
    for name in ("input_padding_sha256", "input_payload_sha256"):
        _hex_v2(object.__getattribute__(value, name), f"record {name}")
    compilation = object.__getattribute__(value, "bridge_compilation_id")
    bridge_decision = object.__getattribute__(value, "bridge_decision_id")
    if (compilation is None) is not (bridge_decision is None):
        raise ValueError("record bridge root pair drift")
    if compilation is not None:
        _digest_v2(compilation, "phase2b_exact_derived_bridge_result_", "record bridge compilation")
        _digest_v2(bridge_decision, "phase2b_exact_derived_decision_", "record bridge decision")
    is_preflight = bridge_outcome.startswith("phase2b_exact_derived_preflight_v2_")
    if is_preflight is not (compilation is None):
        raise ValueError("record bridge stage/root drift")
    _ascii_v2(value.schema_version, "record schema version")
    if value.schema_version != PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION:
        raise ValueError("record schema identity drift")
    prediction = _preflight_prediction_bundle_v2(value.prediction)
    if is_preflight and prediction.reason is not PredictionReason.RESOURCE_LIMIT:
        raise ValueError("record preflight public reason drift")
    if prediction.input_root_sha256 != value.input_payload_sha256:
        raise ValueError("record input/content parity drift")
    if value.decision is PredictionDecisionV2.ABSTAIN:
        coherent = value.canonical_family_id is None and prediction.disposition is PredictionDisposition.ABSTAIN
    else:
        law_kind = _KIND_BY_CANONICAL_FAMILY_V2.get(value.canonical_family_id)
        coherent = (
            type(value.canonical_family_id) is CanonicalFamilyId
            and prediction.disposition is PredictionDisposition.UNIQUE_MATCH
            and law_kind is not None
            and prediction.family_id == _BRIDGE_FAMILY_BY_KIND_V2[law_kind]
            and bool(prediction.binding)
            and (
                (value.decision is PredictionDecisionV2.ANSWER and len(prediction.admissible_scale_ids) == 1)
                or (value.decision is PredictionDecisionV2.ANSWER_SET and len(prediction.admissible_scale_ids) > 1)
            )
        )
    if not coherent:
        raise ValueError("record decision/prediction coherence drift")
    return value


def _finalize_record_row_root_v2(value: PublicRecognizerPredictionRecordV2) -> None:
    root_mapping = {
        "authority_content_id": value.input_authority_content_id,
        "envelope_id": value.input_envelope_id,
        "namespace_audit_id": value.input_namespace_audit_id,
        "padding_sha256": value.input_padding_sha256,
        "payload_sha256": value.input_payload_sha256,
        "public_registry_id": value.input_public_registry_id,
        "transform_result_id": value.input_transform_result_id,
    }
    expected_row_id = "phase2b_recognizer_input_row_v2_" + hashlib.sha256(
        _INPUT_ROW_DOMAIN_V2 + encode_phase2b_jcs_profile_v1(root_mapping)
    ).hexdigest()
    if value.input_row_id != expected_row_id:
        raise ValueError("record input row root drift")


def _preflight_decoded_v2(value: object, raw: bytes) -> DecodedRecognizerPredictionArchiveV2:
    if type(value) is not DecodedRecognizerPredictionArchiveV2:
        raise TypeError("public prediction decoder result exact type drift")
    if type(value.disposition) is not PredictionArchiveDispositionV2 or value.disposition is not PredictionArchiveDispositionV2.COMPLETE:
        raise ValueError("decoded prediction disposition drift")
    if type(value.archive) is not bytes or value.archive != raw:
        raise ValueError("decoded archive bytes parity drift")
    _digest_v2(value.archive_id, "phase2b_recognizer_prediction_archive_v2_", "decoded archive ID")
    _digest_v2(value.policy_id, "phase2b_recognizer_prediction_archive_policy_v2_", "decoded policy ID")
    _ascii_v2(value.schema_version, "decoded archive version")
    _ascii_v2(value.claim_level, "decoded claim level")
    _bool_claims_v2(value, _DECODED_TRUE_CLAIMS_V2, True)
    _bool_claims_v2(value, _DECODED_FALSE_CLAIMS_V2, False)
    if (
        value.schema_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or value.policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or value.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("decoded prediction identity drift")
    _preflight_context_v2(value.context)
    if type(value.records) is not tuple or len(value.records) != _TOTAL_COUNT_V2:
        raise TypeError("decoded record tuple/count drift")
    for record in value.records:
        _preflight_record_v2(record)
    input_ids = _row_ids_v2(value.input_row_ids, _TOTAL_COUNT_V2, "decoded input row IDs")
    if type(value.prediction_record_ids) is not tuple or len(value.prediction_record_ids) != _TOTAL_COUNT_V2:
        raise TypeError("decoded record ID column drift")
    if type(value.prediction_content_ids) is not tuple or len(value.prediction_content_ids) != _TOTAL_COUNT_V2:
        raise TypeError("decoded content ID column drift")
    for item in value.prediction_record_ids:
        _digest_v2(item, "phase2b_recognizer_prediction_record_v2_", "decoded record ID")
    for item in value.prediction_content_ids:
        _digest_v2(item, "phase2b_prediction_", "decoded content ID")
    expected_columns = (
        tuple(record.input_row_id for record in value.records),
        tuple(record.record_id for record in value.records),
        tuple(record.prediction_content_id for record in value.records),
    )
    actual_columns = (input_ids, value.prediction_record_ids, value.prediction_content_ids)
    for actual, expected in zip(actual_columns, expected_columns, strict=True):
        if len(actual) != len(expected) or any(type(a) is not type(b) or a != b for a, b in zip(actual, expected, strict=True)):
            raise ValueError("decoded root column parity drift")
    context = value.context
    expected_protocol_sha = context.protocol_id.rsplit("_", 1)[1]
    expected_freeze_sha = context.execution_freeze_manifest_id.rsplit("_", 1)[1]
    for record in value.records:
        if (
            record.run_context_id != context.context_id
            or record.prediction.protocol_sha256 != expected_protocol_sha
            or record.prediction.freeze_manifest_sha256 != expected_freeze_sha
        ):
            raise ValueError("decoded record/context binding drift")
    row_root_digest = hashlib.sha256()
    row_root_digest.update(_INPUT_ROWS_DOMAIN_V2)
    row_root_digest.update(len(input_ids).to_bytes(4, "big"))
    for row_id in input_ids:
        encoded = row_id.encode("ascii")
        row_root_digest.update(len(encoded).to_bytes(2, "big"))
        row_root_digest.update(encoded)
    expected_input_root = "phase2b_prediction_input_rows_v2_" + row_root_digest.hexdigest()
    if context.input_row_ids_root != expected_input_root:
        raise ValueError("decoded context input-row root drift")
    context_preimage = {
        "batch_id": context.batch_id,
        "batch_policy_id": context.batch_policy_id,
        "claim_level": context.claim_level,
        "execution_freeze_manifest_id": context.execution_freeze_manifest_id,
        "expected_prediction_count": context.expected_prediction_count,
        "input_archive_id": context.input_archive_id,
        "input_archive_policy_id": context.input_archive_policy_id,
        "input_archive_sha256": context.input_archive_sha256,
        "input_archive_version": context.input_archive_version,
        "input_row_ids_root": context.input_row_ids_root,
        "protocol_id": context.protocol_id,
        "schema_version": context.schema_version,
    }
    expected_context_id = stable_hash(
        context_preimage,
        prefix="phase2b_public_prediction_run_context_v2_",
    )
    if context.context_id != expected_context_id:
        raise ValueError("decoded context content root drift")
    expected_archive_id = "phase2b_recognizer_prediction_archive_v2_" + hashlib.sha256(
        _ARCHIVE_DOMAIN_V2 + raw
    ).hexdigest()
    if value.archive_id != expected_archive_id:
        raise ValueError("decoded archive content root drift")
    for record in value.records:
        _finalize_record_row_root_v2(record)
        _finalize_record_roots_v2(record)
    return value


def _preflight_formal_validation_v2(value: object) -> FormalUnsealedPredictionScoringContractValidationV2:
    if type(value) is not FormalUnsealedPredictionScoringContractValidationV2:
        raise TypeError("formal validation exact success type drift")
    if type(value.disposition) is not FormalUnsealedPredictionScoringContractDispositionV2:
        raise TypeError("formal validation disposition type drift")
    if type(value.reason) is not FormalUnsealedPredictionScoringContractReasonV2:
        raise TypeError("formal validation reason type drift")
    for name, prefix in (
        ("schema_id", "phase2b_formal_unsealed_prediction_scoring_contract_schema_v2_"),
        ("policy_id", "phase2b_formal_unsealed_prediction_scoring_contract_policy_v2_"),
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_"),
        ("partition_manifest_id", "phase2b_unsealed_prediction_partition_v2_"),
        ("structural_receipt_id", "phase2b_strict_recognizer_receipt_v2_"),
        ("answer_manifest_id", "phase2b_formal_unsealed_answer_manifest_v2_"),
        ("main_answer_row_ids_root", "phase2b_formal_unsealed_answer_rows_v2_"),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix, f"formal validation {name}")
    _hex_v2(value.answer_manifest_sha256, "formal validation answer SHA")
    _hex_v2(value.salted_answer_commitment_sha256, "formal validation commitment")
    for name in ("version", "claim_level"):
        _ascii_v2(object.__getattribute__(value, name), f"formal validation {name}")
    for name, expected in (
        ("main_row_count", _MAIN_COUNT_V2),
        ("semantic_conflict_row_count", _SEMANTIC_CONFLICT_COUNT_V2),
        ("answerable_row_count", 240),
    ):
        item = object.__getattribute__(value, name)
        if type(item) is not int or item != expected:
            raise ValueError(f"formal validation {name} drift")
    _bool_claims_v2(value, _FORMAL_TRUE_CLAIMS_V2, True)
    _bool_claims_v2(value, _FORMAL_FALSE_CLAIMS_V2, False)
    _empty_tuple_v2(value.metric_results, "formal metric results")
    _empty_tuple_v2(value.scored_rows, "formal scored rows")
    if type(value.metric_definitions) is not tuple or len(value.metric_definitions) != _METRIC_COUNT_V2:
        raise TypeError("formal metric definitions tuple/count drift")
    for metric in value.metric_definitions:
        if type(metric) is not FormalUnsealedMetricDefinitionV2:
            raise TypeError("formal metric definition exact type drift")
        _digest_v2(metric.metric_definition_id, "phase2b_formal_unsealed_metric_definition_v2_", "metric definition ID")
        _ascii_v2(metric.metric_name, "metric name")
        _ascii_v2(metric.success_rule, "metric success rule")
        if type(metric.metric_kind) is not FormalUnsealedMetricKindV2:
            raise TypeError("metric kind type drift")
        if type(metric.denominator_case_types) is not tuple or any(type(item) is not Phase2BCaseType for item in metric.denominator_case_types):
            raise TypeError("metric denominator cases drift")
        if type(metric.expected_denominator) is not int or metric.expected_denominator <= 0:
            raise ValueError("metric denominator drift")
        if type(metric.separately_reported) is not bool:
            raise TypeError("metric separately-reported flag drift")
    contract = frozen_formal_unsealed_prediction_scoring_contract_v2()
    if type(contract) is not FormalUnsealedPredictionScoringContractV2:
        raise TypeError("formal frozen contract exact type drift")
    canonical_definitions = contract.metric_definitions
    if type(canonical_definitions) is not tuple or len(canonical_definitions) != _METRIC_COUNT_V2:
        raise TypeError("formal frozen metric tuple drift")
    for actual, expected in zip(value.metric_definitions, canonical_definitions, strict=True):
        actual_fields = (
            actual.metric_name,
            actual.metric_kind,
            actual.denominator_case_types,
            actual.expected_denominator,
            actual.success_rule,
            actual.separately_reported,
            actual.metric_definition_id,
        )
        expected_fields = (
            expected.metric_name,
            expected.metric_kind,
            expected.denominator_case_types,
            expected.expected_denominator,
            expected.success_rule,
            expected.separately_reported,
            expected.metric_definition_id,
        )
        if any(type(a) is not type(b) or a != b for a, b in zip(actual_fields, expected_fields, strict=True)):
            raise ValueError("formal metric definition canonical parity drift")
    if (
        value.disposition is not FormalUnsealedPredictionScoringContractDispositionV2.CONTRACT_BINDING_COMPLETE_NOT_SCORED
        or value.reason is not FormalUnsealedPredictionScoringContractReasonV2.CONTRACT_BINDING_VERIFIED
        or value.version != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION
        or value.schema_id != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID
        or value.policy_id != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID
        or value.claim_level != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL
    ):
        raise ValueError("formal validation identity drift")
    return value


def _preflight_context_v2(value: object) -> PublicPredictionRunContextV2:
    if type(value) is not PublicPredictionRunContextV2:
        raise TypeError("decoded context exact type drift")
    for name, prefix in (
        ("batch_id", "phase2b_trusted_wire_batch_v2_"),
        ("batch_policy_id", "phase2b_trusted_wire_batch_v2_policy_"),
        ("context_id", "phase2b_public_prediction_run_context_v2_"),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_"),
        ("input_archive_id", "phase2b_recognizer_input_archive_v2_"),
        ("input_archive_policy_id", "phase2b_recognizer_input_archive_policy_v2_"),
        ("input_row_ids_root", "phase2b_prediction_input_rows_v2_"),
        ("protocol_id", "phase2b_protocol_"),
    ):
        _digest_v2(object.__getattribute__(value, name), prefix, f"context {name}")
    _hex_v2(value.input_archive_sha256, "context input archive SHA")
    for name in ("claim_level", "input_archive_version", "schema_version"):
        _ascii_v2(object.__getattribute__(value, name), f"context {name}")
    if type(value.expected_prediction_count) is not int or value.expected_prediction_count != _TOTAL_COUNT_V2:
        raise ValueError("context count drift")
    if (
        value.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        or value.schema_version != PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION
        or value.protocol_id != _FROZEN_PROTOCOL_V2.protocol_id
        or value.input_archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or value.input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or value.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
    ):
        raise ValueError("context upstream identity drift")
    return value


def _primitive_v2(value: object) -> object:
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is tuple:
        return [_primitive_v2(item) for item in value]
    if type(value) in (
        Phase2BCaseType,
        PredictionDecisionV2,
        CanonicalFamilyId,
        FormalUnsealedMetricKindV2,
        Unsealed960PredictionScoringDispositionV2,
        Unsealed960PredictionScoringReasonV2,
    ):
        return value.value
    if type(value) is RoleBinding:
        return {"role_id": value.role_id, "entity_id": value.entity_id}
    if type(value) is Unsealed960MetricRowOutcomeV2:
        return {name: _primitive_v2(object.__getattribute__(value, name)) for name in _METRIC_ROW_OUTCOME_FIELDS_V2}
    if type(value) is Unsealed960MainRowResultV2:
        return {name: _primitive_v2(object.__getattribute__(value, name)) for name in _MAIN_ROW_RESULT_FIELDS_V2}
    if type(value) is Unsealed960MetricResultV2:
        return {name: _primitive_v2(object.__getattribute__(value, name)) for name in _METRIC_RESULT_FIELDS_V2}
    raise TypeError("mechanics V2 primitive mapping type drift")


def _fresh_binding_v2(value: tuple[RoleBinding, ...]) -> tuple[RoleBinding, ...]:
    return tuple(RoleBinding(role_id=item.role_id, entity_id=item.entity_id) for item in value)


def _issue_metric_outcome_v2(
    definition: FormalUnsealedMetricDefinitionV2,
    *,
    eligible: bool,
    success: bool | None,
) -> Unsealed960MetricRowOutcomeV2:
    if type(eligible) is not bool or (success is not None and type(success) is not bool):
        raise TypeError("metric row outcome flag drift")
    value = object.__new__(Unsealed960MetricRowOutcomeV2)
    frozen = (
        ("metric_definition_id", definition.metric_definition_id),
        ("metric_name", definition.metric_name),
        ("eligible", eligible),
        ("success", success),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "metric_row_outcome_id",
        stable_hash(
            {name: _primitive_v2(object.__getattribute__(value, name)) for name in _METRIC_ROW_OUTCOME_FIELDS_V2[:-1]},
            prefix=_METRIC_ROW_OUTCOME_ID_PREFIX_V2,
        ),
    )
    return value


def _metric_success_v2(
    *,
    name: str,
    case_type: Phase2BCaseType,
    decision: PredictionDecisionV2,
    expected_decision: PredictionDecisionV2,
    family_exact: bool,
    binding_exact: bool,
    scale_exact: bool,
) -> tuple[bool, bool | None]:
    answerable = case_type in _ANSWERABLE_CASE_TYPES_V2
    if name == "answerable_count":
        return answerable, None
    if name == "family_exact_accuracy":
        return answerable, family_exact if answerable else None
    if name == "binding_exact_accuracy":
        return answerable, binding_exact if answerable else None
    if name == "scale_set_accuracy":
        return answerable, scale_exact if answerable else None
    if name == "unique_scale_accuracy":
        eligible = case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
        return eligible, (decision is PredictionDecisionV2.ANSWER and scale_exact) if eligible else None
    if name == "joint_exact_accuracy":
        return answerable, (decision is expected_decision and family_exact and binding_exact and scale_exact) if answerable else None
    if name == "abstention_specificity":
        eligible = case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
        return eligible, (decision is not PredictionDecisionV2.ABSTAIN) if eligible else None
    if name == "nonidentifiability_abstention_accuracy":
        eligible = case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
        return eligible, (decision is PredictionDecisionV2.ABSTAIN) if eligible else None
    if name == "set_valued_answer_accuracy":
        eligible = case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE
        return eligible, (decision is PredictionDecisionV2.ANSWER_SET and family_exact and binding_exact and scale_exact) if eligible else None
    raise ValueError("unknown frozen metric definition")


def _issue_main_row_v2(
    *,
    record: PublicRecognizerPredictionRecordV2,
    answer: FormalUnsealedAnswerRowV2,
    definitions: tuple[FormalUnsealedMetricDefinitionV2, ...],
) -> Unsealed960MainRowResultV2:
    positive = record.decision in (PredictionDecisionV2.ANSWER, PredictionDecisionV2.ANSWER_SET)
    family_exact_raw = positive and record.canonical_family_id is answer.canonical_family_id
    binding_exact_raw = positive and record.prediction.binding == answer.binding
    scale_exact_raw = positive and record.prediction.admissible_scale_ids == answer.admissible_scale_ids
    metric_eligible = answer.case_type in _ANSWERABLE_CASE_TYPES_V2 or answer.case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
    outcomes = tuple(
        _issue_metric_outcome_v2(
            definition,
            eligible=eligible,
            success=success,
        )
        for definition in definitions
        for eligible, success in (
            _metric_success_v2(
                name=definition.metric_name,
                case_type=answer.case_type,
                decision=record.decision,
                expected_decision=answer.expected_decision,
                family_exact=family_exact_raw,
                binding_exact=binding_exact_raw,
                scale_exact=scale_exact_raw,
            ),
        )
    )
    answerable = answer.case_type in _ANSWERABLE_CASE_TYPES_V2
    value = object.__new__(Unsealed960MainRowResultV2)
    frozen = (
        ("input_row_id", answer.input_row_id),
        ("prediction_record_id", record.record_id),
        ("prediction_content_id", record.prediction_content_id),
        ("answer_row_id", answer.answer_row_id),
        ("case_type", answer.case_type),
        ("predicted_decision", record.decision),
        ("expected_decision", answer.expected_decision),
        ("predicted_canonical_family_id", record.canonical_family_id),
        ("expected_canonical_family_id", answer.canonical_family_id),
        ("predicted_binding", _fresh_binding_v2(record.prediction.binding)),
        ("expected_binding", _fresh_binding_v2(answer.binding)),
        ("predicted_admissible_scale_ids", tuple(record.prediction.admissible_scale_ids)),
        ("expected_admissible_scale_ids", tuple(answer.admissible_scale_ids)),
        ("decision_exact", record.decision is answer.expected_decision),
        ("family_exact", family_exact_raw if answerable else None),
        ("binding_exact", binding_exact_raw if answerable else None),
        ("scale_set_exact", scale_exact_raw if answerable else None),
        ("joint_exact", (record.decision is answer.expected_decision and family_exact_raw and binding_exact_raw and scale_exact_raw) if answerable else None),
        ("metric_eligible", metric_eligible),
        ("metric_outcomes", outcomes),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "row_result_id",
        stable_hash(
            {name: _primitive_v2(object.__getattribute__(value, name)) for name in _MAIN_ROW_RESULT_FIELDS_V2[:-1]},
            prefix=_MAIN_ROW_RESULT_ID_PREFIX_V2,
        ),
    )
    return value


def _issue_metric_result_v2(
    definition: FormalUnsealedMetricDefinitionV2,
    rows: tuple[Unsealed960MainRowResultV2, ...],
) -> Unsealed960MetricResultV2:
    matching = tuple(
        outcome
        for row in rows
        for outcome in row.metric_outcomes
        if outcome.metric_definition_id == definition.metric_definition_id
    )
    if len(matching) != _MAIN_COUNT_V2:
        raise ValueError("metric row outcome coverage drift")
    eligible = tuple(item for item in matching if item.eligible)
    observed = len(eligible)
    if observed != definition.expected_denominator:
        _reject_v2(Unsealed960PredictionScoringReasonV2.METRIC_DENOMINATOR_MISMATCH)
    if definition.metric_kind is FormalUnsealedMetricKindV2.COUNT:
        success_count: int | None = None
        count_value: int | None = observed
        if any(item.success is not None for item in eligible):
            raise ValueError("count metric row success must be None")
    else:
        if any(type(item.success) is not bool for item in eligible):
            raise ValueError("accuracy metric row success type drift")
        success_count = sum(bool(item.success) for item in eligible)
        count_value = None
    value = object.__new__(Unsealed960MetricResultV2)
    frozen = (
        ("metric_definition_id", definition.metric_definition_id),
        ("metric_name", definition.metric_name),
        ("metric_kind", definition.metric_kind),
        ("denominator_case_types", tuple(definition.denominator_case_types)),
        ("expected_denominator", definition.expected_denominator),
        ("observed_denominator", observed),
        ("success_count", success_count),
        ("count_value", count_value),
        ("success_rule", definition.success_rule),
        ("separately_reported", definition.separately_reported),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "metric_result_id",
        stable_hash(
            {name: _primitive_v2(object.__getattribute__(value, name)) for name in _METRIC_RESULT_FIELDS_V2[:-1]},
            prefix=_METRIC_RESULT_ID_PREFIX_V2,
        ),
    )
    return value


def _require_cross_bindings_v2(
    *,
    raw_sha: str,
    decoded: DecodedRecognizerPredictionArchiveV2,
    receipt: StrictRecognizerStructuralReceiptV2,
    evaluation: UnsealedPredictionStructuralEvaluationV2,
    partition: UnsealedPredictionPartitionManifestV2,
    answer: FormalUnsealedAnswerManifestV2,
    formal: FormalUnsealedPredictionScoringContractValidationV2,
) -> None:
    context = decoded.context
    archive_ids = (
        decoded.archive_id,
        receipt.prediction_archive_id,
        evaluation.prediction_archive_id,
        partition.prediction_archive_id,
        formal.prediction_archive_id,
    )
    if any(type(item) is not str or item != archive_ids[0] for item in archive_ids):
        _reject_v2(Unsealed960PredictionScoringReasonV2.PREDICTION_ARCHIVE_BINDING_MISMATCH)
    scalar_groups = (
        (raw_sha, receipt.prediction_archive_sha256),
        (decoded.schema_version, receipt.prediction_archive_version, evaluation.prediction_archive_schema_version, partition.prediction_archive_schema_version),
        (decoded.policy_id, receipt.prediction_archive_policy_id, evaluation.prediction_archive_policy_id, partition.prediction_archive_policy_id),
        (context.context_id, receipt.run_context_id),
        (context.input_archive_id, receipt.input_archive_id, answer.input_archive_id),
        (context.input_archive_sha256, receipt.input_archive_sha256, answer.input_archive_sha256),
        (context.input_archive_version, receipt.input_archive_version, answer.input_archive_version),
        (context.input_archive_policy_id, receipt.input_archive_policy_id, answer.input_archive_policy_id),
        (context.batch_id, receipt.batch_id, answer.batch_id),
        (context.batch_policy_id, receipt.batch_policy_id, answer.batch_policy_id),
        (context.execution_freeze_manifest_id, receipt.execution_freeze_manifest_id, answer.execution_freeze_manifest_id),
        (context.protocol_id, receipt.protocol_id, answer.phase2b_protocol_id),
        (evaluation.exact_freeze_id, partition.exact_freeze_id, answer.exact_freeze_id),
        (partition.manifest_id, evaluation.partition_manifest_id, formal.partition_manifest_id),
        (partition.main_row_ids_root, evaluation.main_row_ids_root, answer.main_row_ids_root),
        (partition.semantic_conflict_row_ids_root, evaluation.semantic_conflict_row_ids_root, answer.semantic_conflict_row_ids_root),
        (partition.partition_union_row_ids_root, evaluation.partition_union_row_ids_root, answer.partition_union_row_ids_root),
        (partition.ordered_archive_input_row_ids_root, evaluation.ordered_archive_input_row_ids_root, answer.ordered_archive_input_row_ids_root, formal.ordered_archive_input_row_ids_root, context.input_row_ids_root),
        (answer.answer_manifest_id, formal.answer_manifest_id),
        (answer.answer_manifest_sha256, formal.answer_manifest_sha256),
        (answer.main_answer_row_ids_root, formal.main_answer_row_ids_root),
        (receipt.receipt_id, formal.structural_receipt_id),
    )
    for group in scalar_groups:
        first = group[0]
        if any(type(item) is not type(first) or item != first for item in group[1:]):
            _reject_v2(Unsealed960PredictionScoringReasonV2.PREDICTION_ARCHIVE_BINDING_MISMATCH)
    union = set(partition.main_row_ids) | set(partition.semantic_conflict_row_ids)
    if (
        len(union) != _TOTAL_COUNT_V2
        or union != set(decoded.input_row_ids)
        or tuple(row.input_row_id for row in answer.main_answer_rows) != partition.main_row_ids
    ):
        _reject_v2(Unsealed960PredictionScoringReasonV2.MAIN_ROW_JOIN_MISMATCH)


def _issue_success_v2(
    *,
    raw_sha: str,
    decoded: DecodedRecognizerPredictionArchiveV2,
    receipt: StrictRecognizerStructuralReceiptV2,
    partition: UnsealedPredictionPartitionManifestV2,
    answer: FormalUnsealedAnswerManifestV2,
    formal: FormalUnsealedPredictionScoringContractValidationV2,
    metrics: tuple[Unsealed960MetricResultV2, ...],
    rows: tuple[Unsealed960MainRowResultV2, ...],
) -> Unsealed960PredictionScoringMechanicsV2:
    contract = frozen_formal_unsealed_prediction_scoring_contract_v2()
    if type(contract) is not FormalUnsealedPredictionScoringContractV2:
        raise TypeError("formal frozen contract exact type drift")
    value = object.__new__(Unsealed960PredictionScoringMechanicsV2)
    context = decoded.context
    frozen = (
        ("disposition", Unsealed960PredictionScoringDispositionV2.MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION),
        ("reason", Unsealed960PredictionScoringReasonV2.CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE),
        ("version", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION),
        ("schema_id", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID),
        ("claim_level", UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL),
        ("prediction_archive_id", decoded.archive_id),
        ("prediction_archive_sha256", raw_sha),
        ("prediction_archive_version", decoded.schema_version),
        ("prediction_archive_policy_id", decoded.policy_id),
        ("run_context_id", context.context_id),
        ("input_archive_id", context.input_archive_id),
        ("input_archive_sha256", context.input_archive_sha256),
        ("batch_id", context.batch_id),
        ("execution_freeze_manifest_id", context.execution_freeze_manifest_id),
        ("protocol_id", context.protocol_id),
        ("structural_receipt_id", receipt.receipt_id),
        ("partition_manifest_id", partition.manifest_id),
        ("answer_manifest_id", answer.answer_manifest_id),
        ("answer_manifest_sha256", answer.answer_manifest_sha256),
        ("salted_answer_commitment_sha256", formal.salted_answer_commitment_sha256),
        ("formal_scoring_contract_id", contract.contract_id),
        ("ordered_archive_input_row_ids_root", partition.ordered_archive_input_row_ids_root),
        ("main_row_ids_root", partition.main_row_ids_root),
        ("semantic_conflict_row_ids_root", partition.semantic_conflict_row_ids_root),
        ("partition_union_row_ids_root", partition.partition_union_row_ids_root),
        ("main_answer_row_ids_root", answer.main_answer_row_ids_root),
        ("total_prediction_count", _TOTAL_COUNT_V2),
        ("main_row_result_count", _MAIN_COUNT_V2),
        ("metric_eligible_main_row_count", _METRIC_ELIGIBLE_COUNT_V2),
        ("control_row_without_frozen_metric_count", _CONTROL_WITHOUT_METRIC_COUNT_V2),
        ("semantic_conflict_excluded_count", _SEMANTIC_CONFLICT_COUNT_V2),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    for name in _TRUE_RESULT_CLAIMS_V2:
        object.__setattr__(value, name, True)
    for name in _FALSE_RESULT_CLAIMS_V2:
        object.__setattr__(value, name, False)
    object.__setattr__(value, "metric_results", metrics)
    object.__setattr__(value, "main_row_results", rows)
    object.__setattr__(value, "gate_results", ())
    object.__setattr__(value, "scale_regret_result", None)
    object.__setattr__(value, "bootstrap_result", None)
    object.__setattr__(
        value,
        "result_id",
        stable_hash(
            {name: _primitive_v2(object.__getattribute__(value, name)) for name in _SUCCESS_FIELDS_V2 if name != "result_id"},
            prefix=_RESULT_ID_PREFIX_V2,
        ),
    )
    return value


def score_unsealed_960_prediction_scoring_mechanics_v2(
    *,
    prediction_archive: bytes,
    structural_receipt: StrictRecognizerStructuralReceiptV2,
    structural_evaluation: UnsealedPredictionStructuralEvaluationV2,
    partition_manifest: UnsealedPredictionPartitionManifestV2,
    answer_manifest: FormalUnsealedAnswerManifestV2,
    revealed_answer_manifest_sha256: str,
    answer_commitment_salt: str,
    salted_answer_commitment_sha256: str,
) -> Unsealed960PredictionScoringMechanicsV2 | Unsealed960PredictionScoringRejectionV2:
    """Replay and score supplied V2 mechanics; never establish actual evidence."""

    if (
        type(prediction_archive) is not bytes
        or type(structural_receipt) is not StrictRecognizerStructuralReceiptV2
        or type(structural_evaluation) is not UnsealedPredictionStructuralEvaluationV2
        or type(partition_manifest) is not UnsealedPredictionPartitionManifestV2
        or type(answer_manifest) is not FormalUnsealedAnswerManifestV2
        or type(revealed_answer_manifest_sha256) is not str
        or type(answer_commitment_salt) is not str
        or type(salted_answer_commitment_sha256) is not str
    ):
        return _issue_rejection_v2(Unsealed960PredictionScoringReasonV2.WRONG_INPUT_TYPE)
    try:
        if not (
            PREDICTION_ARCHIVE_HEADER_BYTES_V2
            <= len(prediction_archive)
            <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
        ):
            _reject_v2(Unsealed960PredictionScoringReasonV2.PREDICTION_ARCHIVE_INVALID)
        try:
            receipt = _preflight_receipt_v2(structural_receipt)
            evaluation = _preflight_evaluation_v2(structural_evaluation)
            partition = _preflight_partition_v2(partition_manifest)
            answer = _preflight_answer_v2(answer_manifest)
            _preflight_opening_v2(
                revealed_answer_manifest_sha256,
                answer_commitment_salt,
                salted_answer_commitment_sha256,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise _ScoringRejectedV2(
                Unsealed960PredictionScoringReasonV2.FORMAL_CONTRACT_REJECTED
            ) from exc

        formal_candidate = validate_formal_unsealed_prediction_scoring_contract_v2(
            structural_receipt=receipt,
            structural_evaluation=evaluation,
            partition_manifest=partition,
            answer_manifest=answer,
            revealed_answer_manifest_sha256=revealed_answer_manifest_sha256,
            answer_commitment_salt=answer_commitment_salt,
            salted_answer_commitment_sha256=salted_answer_commitment_sha256,
        )
        try:
            formal = _preflight_formal_validation_v2(formal_candidate)
        except (AttributeError, TypeError, ValueError) as exc:
            raise _ScoringRejectedV2(
                Unsealed960PredictionScoringReasonV2.FORMAL_CONTRACT_REJECTED
            ) from exc
        formal_input_groups = (
            (formal.prediction_archive_id, evaluation.prediction_archive_id),
            (formal.partition_manifest_id, partition.manifest_id),
            (formal.structural_receipt_id, receipt.receipt_id),
            (formal.answer_manifest_id, answer.answer_manifest_id),
            (formal.answer_manifest_sha256, answer.answer_manifest_sha256, revealed_answer_manifest_sha256),
            (formal.salted_answer_commitment_sha256, salted_answer_commitment_sha256),
            (formal.main_answer_row_ids_root, answer.main_answer_row_ids_root),
            (formal.ordered_archive_input_row_ids_root, answer.ordered_archive_input_row_ids_root),
        )
        if any(
            any(type(item) is not type(group[0]) or item != group[0] for item in group[1:])
            for group in formal_input_groups
        ):
            _reject_v2(Unsealed960PredictionScoringReasonV2.FORMAL_CONTRACT_REJECTED)

        try:
            decoded_candidate = decode_public_recognizer_prediction_archive_v2(
                prediction_archive
            )
            decoded = _preflight_decoded_v2(decoded_candidate, prediction_archive)
        except (AttributeError, KeyError, OverflowError, RecursionError, RuntimeError, TypeError, ValueError) as exc:
            raise _ScoringRejectedV2(
                Unsealed960PredictionScoringReasonV2.PREDICTION_ARCHIVE_INVALID
            ) from exc
        raw_sha = hashlib.sha256(prediction_archive).hexdigest()
        _require_cross_bindings_v2(
            raw_sha=raw_sha,
            decoded=decoded,
            receipt=receipt,
            evaluation=evaluation,
            partition=partition,
            answer=answer,
            formal=formal,
        )

        by_row_id = {record.input_row_id: record for record in decoded.records}
        if len(by_row_id) != _TOTAL_COUNT_V2:
            _reject_v2(Unsealed960PredictionScoringReasonV2.MAIN_ROW_JOIN_MISMATCH)
        definitions = tuple(formal.metric_definitions)
        rows = tuple(
            _issue_main_row_v2(
                record=by_row_id[answer_row.input_row_id],
                answer=answer_row,
                definitions=definitions,
            )
            for answer_row in answer.main_answer_rows
        )
        if (
            len(rows) != _MAIN_COUNT_V2
            or tuple(item.input_row_id for item in rows) != partition.main_row_ids
            or sum(item.metric_eligible for item in rows) != _METRIC_ELIGIBLE_COUNT_V2
            or sum(item.case_type in _CONTROL_CASE_TYPES_V2 for item in rows)
            != _CONTROL_WITHOUT_METRIC_COUNT_V2
        ):
            _reject_v2(Unsealed960PredictionScoringReasonV2.MAIN_ROW_JOIN_MISMATCH)
        metrics = tuple(
            _issue_metric_result_v2(definition, rows) for definition in definitions
        )
        if len(metrics) != _METRIC_COUNT_V2:
            _reject_v2(Unsealed960PredictionScoringReasonV2.METRIC_DENOMINATOR_MISMATCH)
        return _issue_success_v2(
            raw_sha=raw_sha,
            decoded=decoded,
            receipt=receipt,
            partition=partition,
            answer=answer,
            formal=formal,
            metrics=metrics,
            rows=rows,
        )
    except _ScoringRejectedV2 as exc:
        return _issue_rejection_v2(exc.reason)
    except Exception:
        return _issue_rejection_v2(Unsealed960PredictionScoringReasonV2.INTERNAL_ERROR)


__all__ = (
    "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION",
    "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL",
    "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID",
    "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID",
    "Unsealed960PredictionScoringDispositionV2",
    "Unsealed960PredictionScoringReasonV2",
    "Unsealed960MetricRowOutcomeV2",
    "Unsealed960MainRowResultV2",
    "Unsealed960MetricResultV2",
    "Unsealed960PredictionScoringMechanicsV2",
    "Unsealed960PredictionScoringRejectionV2",
    "score_unsealed_960_prediction_scoring_mechanics_v2",
)
