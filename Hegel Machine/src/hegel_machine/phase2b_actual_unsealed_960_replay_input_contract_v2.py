"""Evaluator-side gate-input package contract for a future Phase-2B replay.

This module freezes and validates supplied, content-addressed label-package
mechanics only.  It does not establish that the package was committed before a
run, authenticate an evaluator or custodian, execute a recognizer, score a
prediction, evaluate a gate, or constitute actual-960/effect/C1 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
from typing import Final
from uuid import UUID

from .hashing import canonical_json, stable_hash
from .phase2b_formal_unsealed_prediction_scoring_contract_v2 import (
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
    FormalUnsealedAnswerManifestV2,
    FormalUnsealedAnswerRowV2,
    frozen_formal_unsealed_prediction_scoring_contract_v2,
)
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from .phase2b_protocol import (
    MarginStratum,
    Phase2BCaseType,
    frozen_phase2b_protocol,
)
from .phase2b_recognizer_input_archive_v2 import (
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
)
from .phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from .phase2b_trusted_wire_batch_v2 import TRUSTED_WIRE_BATCH_V2_POLICY_ID
from .phase2b_wire import RoleBinding


ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-replay-input-contract/2"
)
ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_ACTUAL_REPLAY_INPUT_CONTRACT_ONLY"
)
FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-unsealed-gate-input-manifest/2"
)

_MAXIMUM_TEXT_BYTES_V2: Final = 4096
_MAXIMUM_SALT_BYTES_V2: Final = 4096
_MAXIMUM_BINDINGS_V2: Final = 64
_MAXIMUM_SCALES_V2: Final = 4096
_MAIN_COUNT_V2: Final = 720
_SEMANTIC_CONFLICT_COUNT_V2: Final = 240
_TOTAL_COUNT_V2: Final = 960
_CELL_COUNT_V2: Final = 12
_CELL_SIZE_V2: Final = 60
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
_EXACT_FREEZE_ID_V2: Final = frozen_phase2b_exact_freeze().freeze_id
_PHASE2B_PROTOCOL_ID_V2: Final = frozen_phase2b_protocol().protocol_id
_FORMAL_SCORING_CONTRACT_ID_V2: Final = (
    frozen_formal_unsealed_prediction_scoring_contract_v2().contract_id
)

_GATE_ROW_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW/V2\x00"
_GATE_ROW_IDS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW_IDS/V2\x00"
)
_GATE_MANIFEST_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_MANIFEST/V2\x00"
)
_GATE_COMMITMENT_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_COMMITMENT/V2\x00"
)
_GATE_DEFINITION_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_DEFINITION/V2\x00"
)
_REQUIRED_EVIDENCE_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/ACTUAL_REPLAY/REQUIRED_EVIDENCE/V2\x00"
)
_RESULT_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/RESULT/V2\x00"
_ANSWER_ROW_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW/V2\x00"
)
_ANSWER_ROW_IDS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00"
)
_ANSWER_MANIFEST_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_MANIFEST/V2\x00"
)
_GATE_ROW_ID_PREFIX_V2: Final = "phase2b_actual_replay_gate_input_row_v2_"
_GATE_ROW_IDS_ROOT_PREFIX_V2: Final = (
    "phase2b_actual_replay_gate_input_rows_v2_"
)
_GATE_MANIFEST_ID_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_gate_input_manifest_v2_"
)
_RESULT_ID_PREFIX_V2: Final = (
    "phase2b_actual_unsealed_960_replay_input_contract_v2_"
)
_GATE_DEFINITION_ID_PREFIX_V2: Final = (
    "phase2b_actual_replay_gate_input_definition_v2_"
)
_REQUIRED_EVIDENCE_ID_PREFIX_V2: Final = (
    "phase2b_actual_replay_required_evidence_v2_"
)
_ANSWER_ROW_ID_PREFIX_V2: Final = "phase2b_formal_unsealed_answer_row_v2_"
_ANSWER_ROW_IDS_ROOT_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_answer_rows_v2_"
)
_ANSWER_MANIFEST_ID_PREFIX_V2: Final = (
    "phase2b_formal_unsealed_answer_manifest_v2_"
)


class FormalUnsealedScaleSliceIdV2(str, Enum):
    """Opaque evaluator slice IDs, not admissible-scale identifiers."""

    S01 = "S01"
    S02 = "S02"


class ActualUnsealed960ReplayInputDispositionV2(str, Enum):
    ACTUAL_REPLAY_CONTRACT_COMPLETE_NOT_EXECUTED = (
        "ACTUAL_REPLAY_CONTRACT_COMPLETE_NOT_EXECUTED"
    )
    REJECTED = "REJECTED"


class ActualUnsealed960ReplayInputReasonV2(str, Enum):
    SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED = (
        "SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    ANSWER_MANIFEST_INVALID = "ANSWER_MANIFEST_INVALID"
    GATE_INPUT_MANIFEST_INVALID = "GATE_INPUT_MANIFEST_INVALID"
    GATE_INPUT_OPENING_INVALID = "GATE_INPUT_OPENING_INVALID"
    ROW_COVERAGE_MISMATCH = "ROW_COVERAGE_MISMATCH"
    QUOTA_MISMATCH = "QUOTA_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class _ContractRejectedV2(Exception):
    def __init__(self, reason: ActualUnsealed960ReplayInputReasonV2) -> None:
        super().__init__(reason.value)
        self.reason = reason


_GATE_INPUT_ROW_FIELDS_V2: Final = (
    "input_row_id",
    "answer_row_id",
    "case_type",
    "margin_stratum",
    "canonical_family_id",
    "scale_slice_id",
    "latent_base_case_id",
    "gate_input_row_id",
)
_ANSWER_ROW_FIELDS_V2: Final = (
    "input_row_id", "case_type", "expected_decision", "canonical_family_id",
    "binding", "admissible_scale_ids", "answer_row_id",
)
_ANSWER_MANIFEST_FIELDS_V2: Final = (
    "schema_version", "schema_id", "policy_id", "claim_level",
    "exact_freeze_id", "phase2b_protocol_id", "execution_freeze_manifest_id",
    "input_archive_id", "input_archive_sha256", "input_archive_version",
    "input_archive_policy_id", "batch_id", "batch_policy_id",
    "ordered_archive_input_row_ids_root", "main_row_ids_root",
    "semantic_conflict_row_ids_root", "partition_union_row_ids_root",
    "main_answer_rows", "main_answer_row_ids_root", "answer_manifest_sha256",
    "answer_manifest_id",
)
_GATE_INPUT_DEFINITION_FIELDS_V2: Final = (
    "gate_name",
    "scope",
    "expected_denominator",
    "success_rule",
    "input_available",
    "missing_input_reason",
    "definition_id",
)
_REQUIRED_EVIDENCE_FIELDS_V2: Final = (
    "evidence_name",
    "purpose",
    "supplied_by_this_contract",
    "verifier_implemented",
    "requirement_id",
)
_GATE_INPUT_MANIFEST_FIELDS_V2: Final = (
    "schema_version",
    "schema_id",
    "policy_id",
    "claim_level",
    "exact_freeze_id",
    "phase2b_protocol_id",
    "formal_scoring_contract_id",
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
    "answer_manifest_id",
    "answer_manifest_sha256",
    "main_answer_row_ids_root",
    "main_gate_input_rows",
    "main_gate_input_row_ids_root",
    "required_evidence_inventory",
    "gate_input_manifest_sha256",
    "gate_input_manifest_id",
)

_TRUE_RESULT_CLAIMS_V2: Final = (
    "exact_contract_identity_verified",
    "answer_gate_manifest_cross_binding_verified",
    "supplied_gate_input_commitment_opening_verified",
    "exact_main_gate_row_coverage_verified",
    "exact_family_scale_cell_quota_verified",
    "exact_case_type_per_cell_quota_verified",
    "exact_margin_per_cell_quota_verified",
    "exact_nonunique_margin_case_composition_verified",
    "supplied_family_slice_labels_complete",
    "supplied_scale_slice_labels_complete",
    "unique_latent_base_case_ids_verified",
    "downstream_prediction_identifier_fields_absent_from_schema_verified",
    "semantic_conflict_root_bound_and_exclusion_contract_frozen",
    "control_gate_input_semantics_frozen",
    "slice_gate_input_semantics_frozen",
    "required_unsupplied_evidence_inventory_frozen",
)
_FALSE_RESULT_CLAIMS_V2: Final = (
    "challenge_in_main_denominator",
    "margin_stratum_authority_verified",
    "family_slice_label_authority_verified",
    "scale_slice_semantics_authority_verified",
    "latent_case_independence_verified",
    "one_shot_policy_enforced",
    "durable_attempt_ledger_verified",
    "raw_input_archive_replayed",
    "raw_prediction_archive_replayed",
    "prediction_commit_before_reveal_verified",
    "wilson_bounds_evaluated",
    "preservation_evaluated",
    "challenge_descriptor_rows_implemented",
    "challenge_scoring_performed",
    "fail_closed_gate_inputs_contract_complete",
    "preservation_gate_inputs_contract_complete",
    "scale_regret_inputs_contract_complete",
    "bootstrap_inputs_contract_complete",
    "answer_manifest_authority_verified",
    "gate_input_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
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
    "formal_gate_evaluation_performed",
    "metric_results_materialized",
    "scored_rows_materialized",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "effect_evidence",
    "c1_exit_evidence",
)

_CASE_QUOTA_PER_CELL_V2: Final = (
    (Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE, 19),
    (Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE, 1),
    (Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE, 8),
    (Phase2BCaseType.BINDING_COUNTERFACTUAL, 8),
    (Phase2BCaseType.SCALE_COUNTERFACTUAL, 8),
    (Phase2BCaseType.SIGN_OR_INVARIANT_BREAK, 8),
    (Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE, 8),
)
_MARGIN_QUOTA_PER_CELL_V2: Final = (
    (MarginStratum.CLEAR_INTERIOR, 21),
    (MarginStratum.MODERATE, 18),
    (MarginStratum.NEAR_BOUNDARY_IDENTIFIABLE, 12),
    (MarginStratum.NONUNIQUE_OR_INSUFFICIENT, 9),
)
_ANSWERABLE_CASE_TYPES_V2: Final = (
    Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
)
_CONTROL_CASE_TYPES_V2: Final = (
    Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE,
    Phase2BCaseType.BINDING_COUNTERFACTUAL,
    Phase2BCaseType.SCALE_COUNTERFACTUAL,
    Phase2BCaseType.SIGN_OR_INVARIANT_BREAK,
)

# name, scope, denominator-or-None, success rule, available, missing reason
_AVAILABLE_OVERALL_GATE_INPUT_SPECS_V2: Final = (
    ("family_exact", "overall", 240, "positive_decision_and_canonical_family_exact", True, None),
    ("binding_exact", "overall", 240, "positive_decision_and_canonical_binding_tuple_exact", True, None),
    ("scale_set_accuracy", "overall", 240, "positive_decision_and_canonical_scale_tuple_exact", True, None),
    ("joint_exact", "overall", 240, "expected_decision_family_binding_and_scale_all_exact_set_rows_require_answer_set", True, None),
    ("hard_negative_rejection", "overall", 96, "predicted_decision_is_abstain_reason_unscored", True, None),
    ("binding_counterfactual_rejection", "overall", 96, "predicted_decision_is_abstain_reason_unscored", True, None),
    ("scale_counterfactual_rejection", "overall", 96, "predicted_decision_is_abstain_reason_unscored", True, None),
    ("sign_or_invariant_break_rejection", "overall", 96, "predicted_decision_is_abstain_reason_unscored", True, None),
    ("abstention_specificity", "overall", 228, "unique_answerable_decision_is_not_abstain", True, None),
    ("nonidentifiable_scale_abstention", "overall", 96, "insufficient_or_nonidentifiable_decision_is_abstain_reason_unscored", True, None),
)
_UNAVAILABLE_OVERALL_GATE_INPUT_SPECS_V2: Final = (
    ("fail_closed_rate", "overall", None, "requires_attempt_event_denominator_not_abstention", False, "durable_attempt_event_manifest_not_supplied"),
    ("preservation_consistency", "overall", None, "requires_detached_496_legal_and_76_invalid_pair_results", False, "preservation_pair_result_manifest_not_supplied"),
)
_SLICE_GATE_INPUT_SPECS_V2: Final = (
    ("answerable_joint_exact", "family", 40, "expected_decision_family_binding_and_scale_all_exact", True, None),
    ("all_control_rejection", "family", 64, "four_control_case_types_predict_abstain_reason_unscored", True, None),
    ("abstention_specificity", "family", 38, "unique_answerable_decision_is_not_abstain", True, None),
    ("answerable_joint_exact", "scale", 120, "expected_decision_family_binding_and_scale_all_exact", True, None),
    ("all_control_rejection", "scale", 192, "four_control_case_types_predict_abstain_reason_unscored", True, None),
    ("abstention_specificity", "scale", 114, "unique_answerable_decision_is_not_abstain", True, None),
)

_REQUIRED_EVIDENCE_SPECS_V2: Final = (
    ("durable_signed_custodian_ledger_and_signer_chain", "authenticate_one_shot_append_only_custody"),
    ("prediction_reveal_scoring_chronology", "prove_package_commitment_precedes_run_and_predictions_precede_reveal"),
    ("raw_input_and_prediction_archive_cas", "bind_authoritative_input_and_output_bytes"),
    ("canonical_archive_replay_and_decoder_transcript", "prove_public_canonical_input_and_prediction_replay"),
    ("source_image_sbom_config_runtime_execution_attestations", "authenticate_frozen_executable_environment"),
    ("actual_recognizer_execution_capacity_and_output_attestation", "prove_recognizer_execution_capacity_and_output_origin"),
    ("allocation_source_batch_partition_answer_authorities", "authenticate_membership_and_evaluator_inputs"),
    ("independent_720_240_generation_and_source_disjointness_receipt", "prove_post_validation_generation_source_public_disjointness_and_latent_case_independence"),
    ("audit_archive_formal_uuid_and_covert_receipts", "bind_formal_audit_outputs"),
    ("three_baseline_prediction_outputs", "support_embedding_nearest_prototype_frozen_llm_semantic_only_flat_learned_typed_comparisons"),
    ("durable_fail_closed_attempt_event_denominator", "define_fail_closed_rate_over_all_attempt_events"),
    ("preservation_496_legal_76_invalid_pair_results", "bind_original_variant_links_predictions_and_evaluator_results_for_496_legal_and_76_invalid_pairs"),
    ("semantic_conflict_240_descriptor_audit_package", "support_separately_reported_challenge_audit_or_scoring_excluded_from_main_thresholds_and_tuning"),
    ("scale_regret_oracle_loss_and_normalizer", "define_normalized_per_case_regret_point_max_0_05_and_bootstrap_upper_bound_0_08"),
    ("clustered_bootstrap_inputs_and_statistic", "define_10000_replicate_paired_latent_base_case_cluster_bootstrap_master_seed_411876909552964556_uint32_2611585425_original_and_preservation_variants_one_sided_95_percentile"),
    ("metric_wilson_overall_and_slice_gate_report", "materialize_metric_counts_frozen_thresholds_one_sided_95_percent_wilson_and_gate_results"),
    ("attempt_rerun_permanent_record", "enforce_attempt_policy_and_permanent_attempt_history"),
    ("formal_c1_consumed_sealed_report", "prove_all_overall_family_scale_gates_and_consumed_ledger_state"),
)

_FORBIDDEN_DOWNSTREAM_FIELDS_V2: Final = frozenset(
    {
        "prediction_archive_id", "prediction_archive_sha256",
        "prediction_record_id", "prediction_content_id", "run_context_id",
        "partition_manifest_id", "structural_receipt_id",
        "structural_evaluation_id", "runtime_receipt_id", "score_id",
        "metric_result_id", "scored_row_id", "gate_result_id", "effect_id",
        "c1_result_id", "timestamp", "run_started_at", "predictions_committed_at",
        "answer_revealed_at", "attempt_status", "attempt_index",
    }
)
if (
    set(_GATE_INPUT_ROW_FIELDS_V2) & _FORBIDDEN_DOWNSTREAM_FIELDS_V2
    or set(_GATE_INPUT_MANIFEST_FIELDS_V2) & _FORBIDDEN_DOWNSTREAM_FIELDS_V2
):
    raise RuntimeError("gate-input V2 schema contains downstream fields")


@dataclass(frozen=True, slots=True)
class FormalUnsealedGateInputRowV2:
    input_row_id: str
    answer_row_id: str
    case_type: Phase2BCaseType
    margin_stratum: MarginStratum
    canonical_family_id: CanonicalFamilyId
    scale_slice_id: FormalUnsealedScaleSliceIdV2
    latent_base_case_id: str
    gate_input_row_id: str = ""


@dataclass(frozen=True, slots=True, init=False)
class ActualReplayGateInputDefinitionV2:
    gate_name: str
    scope: str
    expected_denominator: int | None
    success_rule: str
    input_available: bool
    missing_input_reason: str | None
    definition_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("actual replay V2 gate definitions are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class ActualReplayRequiredEvidenceV2:
    evidence_name: str
    purpose: str
    supplied_by_this_contract: bool
    verifier_implemented: bool
    requirement_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("actual replay V2 evidence requirements are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class FormalUnsealedGateInputManifestV2:
    schema_version: str
    schema_id: str
    policy_id: str
    claim_level: str
    exact_freeze_id: str
    phase2b_protocol_id: str
    formal_scoring_contract_id: str
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
    answer_manifest_id: str
    answer_manifest_sha256: str
    main_answer_row_ids_root: str
    main_gate_input_rows: tuple[FormalUnsealedGateInputRowV2, ...]
    main_gate_input_row_ids_root: str
    required_evidence_inventory: tuple[ActualReplayRequiredEvidenceV2, ...]
    gate_input_manifest_sha256: str
    gate_input_manifest_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("formal V2 gate-input manifests are privately issued")


_RESULT_FIELDS_V2: Final = (
    "disposition", "reason", "version", "schema_id", "policy_id",
    "claim_level", "result_id", "gate_input_manifest_id",
    "gate_input_manifest_sha256", "salted_gate_input_commitment_sha256",
    "gate_input_manifest_schema_id", "gate_input_manifest_policy_id",
    "answer_manifest_id", "answer_manifest_sha256",
    "execution_freeze_manifest_id", "input_archive_id", "input_archive_sha256",
    "input_archive_version", "input_archive_policy_id", "batch_id",
    "batch_policy_id",
    "exact_freeze_id", "protocol_id", "formal_scoring_contract_id",
    "ordered_archive_input_row_ids_root", "main_row_ids_root",
    "semantic_conflict_row_ids_root", "partition_union_row_ids_root",
    "main_answer_row_ids_root", "main_gate_input_row_ids_root",
    "main_row_count", "semantic_conflict_expected_row_count",
    "total_expected_prediction_count", "unique_latent_base_case_id_count",
    "family_scale_cell_count", *_TRUE_RESULT_CLAIMS_V2,
    *_FALSE_RESULT_CLAIMS_V2, "required_evidence_inventory",
    "available_overall_gate_input_definitions",
    "unavailable_overall_gate_input_definitions", "slice_gate_input_definitions",
    "metric_results", "scored_rows", "gate_results", "scale_regret_result",
    "bootstrap_result",
)


@dataclass(frozen=True, slots=True, init=False)
class ActualUnsealed960ReplayInputContractV2:
    disposition: ActualUnsealed960ReplayInputDispositionV2
    reason: ActualUnsealed960ReplayInputReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    result_id: str
    gate_input_manifest_id: str
    gate_input_manifest_sha256: str
    salted_gate_input_commitment_sha256: str
    gate_input_manifest_schema_id: str
    gate_input_manifest_policy_id: str
    answer_manifest_id: str
    answer_manifest_sha256: str
    execution_freeze_manifest_id: str
    input_archive_id: str
    input_archive_sha256: str
    input_archive_version: str
    input_archive_policy_id: str
    batch_id: str
    batch_policy_id: str
    exact_freeze_id: str
    protocol_id: str
    formal_scoring_contract_id: str
    ordered_archive_input_row_ids_root: str
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    partition_union_row_ids_root: str
    main_answer_row_ids_root: str
    main_gate_input_row_ids_root: str
    main_row_count: int
    semantic_conflict_expected_row_count: int
    total_expected_prediction_count: int
    unique_latent_base_case_id_count: int
    family_scale_cell_count: int
    exact_contract_identity_verified: bool
    answer_gate_manifest_cross_binding_verified: bool
    supplied_gate_input_commitment_opening_verified: bool
    exact_main_gate_row_coverage_verified: bool
    exact_family_scale_cell_quota_verified: bool
    exact_case_type_per_cell_quota_verified: bool
    exact_margin_per_cell_quota_verified: bool
    exact_nonunique_margin_case_composition_verified: bool
    supplied_family_slice_labels_complete: bool
    supplied_scale_slice_labels_complete: bool
    unique_latent_base_case_ids_verified: bool
    downstream_prediction_identifier_fields_absent_from_schema_verified: bool
    semantic_conflict_root_bound_and_exclusion_contract_frozen: bool
    control_gate_input_semantics_frozen: bool
    slice_gate_input_semantics_frozen: bool
    required_unsupplied_evidence_inventory_frozen: bool
    challenge_in_main_denominator: bool
    margin_stratum_authority_verified: bool
    family_slice_label_authority_verified: bool
    scale_slice_semantics_authority_verified: bool
    latent_case_independence_verified: bool
    one_shot_policy_enforced: bool
    durable_attempt_ledger_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    prediction_commit_before_reveal_verified: bool
    wilson_bounds_evaluated: bool
    preservation_evaluated: bool
    challenge_descriptor_rows_implemented: bool
    challenge_scoring_performed: bool
    fail_closed_gate_inputs_contract_complete: bool
    preservation_gate_inputs_contract_complete: bool
    scale_regret_inputs_contract_complete: bool
    bootstrap_inputs_contract_complete: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
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
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    effect_evidence: bool
    c1_exit_evidence: bool
    required_evidence_inventory: tuple[ActualReplayRequiredEvidenceV2, ...]
    available_overall_gate_input_definitions: tuple[ActualReplayGateInputDefinitionV2, ...]
    unavailable_overall_gate_input_definitions: tuple[ActualReplayGateInputDefinitionV2, ...]
    slice_gate_input_definitions: tuple[ActualReplayGateInputDefinitionV2, ...]
    metric_results: tuple[()]
    scored_rows: tuple[()]
    gate_results: tuple[()]
    scale_regret_result: None
    bootstrap_result: None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("actual replay V2 contract results are privately issued")


_REJECTION_FIELDS_V2: Final = (
    "disposition", "reason", "version", "schema_id", "policy_id",
    "claim_level", "validation", "required_evidence_inventory",
    "available_overall_gate_input_definitions",
    "unavailable_overall_gate_input_definitions", "slice_gate_input_definitions",
    "metric_results", "scored_rows", "gate_results", "scale_regret_result",
    "bootstrap_result",
    "partial_output_published", *_TRUE_RESULT_CLAIMS_V2, *_FALSE_RESULT_CLAIMS_V2,
)


@dataclass(frozen=True, slots=True, init=False)
class ActualUnsealed960ReplayInputContractRejectionV2:
    disposition: ActualUnsealed960ReplayInputDispositionV2
    reason: ActualUnsealed960ReplayInputReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    validation: None
    required_evidence_inventory: tuple[()]
    available_overall_gate_input_definitions: tuple[()]
    unavailable_overall_gate_input_definitions: tuple[()]
    slice_gate_input_definitions: tuple[()]
    metric_results: tuple[()]
    scored_rows: tuple[()]
    gate_results: tuple[()]
    scale_regret_result: None
    bootstrap_result: None
    partial_output_published: bool
    exact_contract_identity_verified: bool
    answer_gate_manifest_cross_binding_verified: bool
    supplied_gate_input_commitment_opening_verified: bool
    exact_main_gate_row_coverage_verified: bool
    exact_family_scale_cell_quota_verified: bool
    exact_case_type_per_cell_quota_verified: bool
    exact_margin_per_cell_quota_verified: bool
    exact_nonunique_margin_case_composition_verified: bool
    supplied_family_slice_labels_complete: bool
    supplied_scale_slice_labels_complete: bool
    unique_latent_base_case_ids_verified: bool
    downstream_prediction_identifier_fields_absent_from_schema_verified: bool
    semantic_conflict_root_bound_and_exclusion_contract_frozen: bool
    control_gate_input_semantics_frozen: bool
    slice_gate_input_semantics_frozen: bool
    required_unsupplied_evidence_inventory_frozen: bool
    challenge_in_main_denominator: bool
    margin_stratum_authority_verified: bool
    family_slice_label_authority_verified: bool
    scale_slice_semantics_authority_verified: bool
    latent_case_independence_verified: bool
    one_shot_policy_enforced: bool
    durable_attempt_ledger_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    prediction_commit_before_reveal_verified: bool
    wilson_bounds_evaluated: bool
    preservation_evaluated: bool
    challenge_descriptor_rows_implemented: bool
    challenge_scoring_performed: bool
    fail_closed_gate_inputs_contract_complete: bool
    preservation_gate_inputs_contract_complete: bool
    scale_regret_inputs_contract_complete: bool
    bootstrap_inputs_contract_complete: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
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
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    effect_evidence: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("actual replay V2 rejections are privately issued")


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
        raise ValueError(f"{name} exceeds its frozen byte cap")
    return value


def _hex64_v2(value: object, *, name: str) -> str:
    text = _exact_text_v2(value, name=name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _digest_v2(value: object, *, prefix: str, name: str) -> str:
    text = _exact_text_v2(value, name=name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _hex64_v2(text.removeprefix(prefix), name=f"{name} suffix")
    return text


def _uuid_v2(value: object, *, name: str) -> str:
    text = _exact_text_v2(value, name=name, maximum_bytes=36)
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{name} UUID drift") from exc
    if parsed.version != 4 or str(parsed) != text:
        raise ValueError(f"{name} must be canonical lowercase UUIDv4")
    return text


def _primitive_id_v2(
    mapping: dict[str, object],
    *,
    domain: bytes,
    prefix: str,
) -> str:
    return prefix + hashlib.sha256(
        domain + canonical_json(mapping).encode("utf-8")
    ).hexdigest()


def _sequence_root_v2(
    values: tuple[str, ...],
    *,
    domain: bytes,
    prefix: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(values).to_bytes(4, "big"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return prefix + digest.hexdigest()


def _binding_mapping_v2(value: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"role_id": item[0], "entity_id": item[1]} for item in value]


def _preflight_answer_row_v2(value: object) -> tuple[object, ...]:
    if type(value) is not FormalUnsealedAnswerRowV2:
        raise TypeError("gate-input V2 needs exact answer rows")
    input_row_id = _digest_v2(
        object.__getattribute__(value, "input_row_id"),
        prefix="phase2b_recognizer_input_row_v2_",
        name="gate-input V2 answer input row ID",
    )
    case_type = object.__getattribute__(value, "case_type")
    decision = object.__getattribute__(value, "expected_decision")
    family = object.__getattribute__(value, "canonical_family_id")
    binding = object.__getattribute__(value, "binding")
    scales = object.__getattribute__(value, "admissible_scale_ids")
    answer_row_id = _digest_v2(
        object.__getattribute__(value, "answer_row_id"),
        prefix=_ANSWER_ROW_ID_PREFIX_V2,
        name="gate-input V2 answer row ID",
    )
    if type(case_type) is not Phase2BCaseType:
        raise TypeError("gate-input V2 answer case type drift")
    if type(decision) is not PredictionDecisionV2:
        raise TypeError("gate-input V2 answer decision drift")
    if family is not None and type(family) is not CanonicalFamilyId:
        raise TypeError("gate-input V2 answer family drift")
    if (
        type(binding) is not tuple
        or len(binding) > _MAXIMUM_BINDINGS_V2
        or type(scales) is not tuple
        or len(scales) > _MAXIMUM_SCALES_V2
    ):
        raise TypeError("gate-input V2 answer tuple drift")
    binding_keys: list[tuple[str, str]] = []
    for item in binding:
        if type(item) is not RoleBinding:
            raise TypeError("gate-input V2 answer binding item drift")
        role_id = _uuid_v2(object.__getattribute__(item, "role_id"), name="answer role ID")
        entity_id = _uuid_v2(object.__getattribute__(item, "entity_id"), name="answer entity ID")
        binding_keys.append((role_id, entity_id))
    if (
        binding_keys != sorted(binding_keys)
        or len({item[0] for item in binding_keys}) != len(binding_keys)
        or len({item[1] for item in binding_keys}) != len(binding_keys)
    ):
        raise ValueError("gate-input V2 answer binding not canonical injective")
    for item in scales:
        _uuid_v2(item, name="answer scale ID")
    if scales != tuple(sorted(set(scales))):
        raise ValueError("gate-input V2 answer scales not sorted unique")
    if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
        coherent = (
            decision is PredictionDecisionV2.ANSWER
            and type(family) is CanonicalFamilyId
            and bool(binding)
            and len(scales) == 1
        )
    elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
        coherent = (
            decision is PredictionDecisionV2.ANSWER_SET
            and type(family) is CanonicalFamilyId
            and bool(binding)
            and len(scales) >= 2
        )
    else:
        coherent = (
            decision is PredictionDecisionV2.ABSTAIN
            and family is None
            and binding == ()
            and scales == ()
        )
    if not coherent:
        raise ValueError("gate-input V2 answer semantic drift")
    return (
        input_row_id, case_type, decision, family, tuple(binding_keys),
        tuple(scales), answer_row_id,
    )


def _preflight_answer_manifest_v2(
    value: object,
) -> tuple[dict[str, str], tuple[tuple[object, ...], ...]]:
    if type(value) is not FormalUnsealedAnswerManifestV2:
        raise TypeError("gate-input V2 needs exact answer manifest")
    text_specs = (
        ("schema_version", None, None),
        ("schema_id", "phase2b_formal_unsealed_answer_manifest_schema_v2_", None),
        ("policy_id", "phase2b_formal_unsealed_answer_manifest_policy_v2_", None),
        ("claim_level", None, None),
        ("exact_freeze_id", "phase2b_exact_freeze_", None),
        ("phase2b_protocol_id", "phase2b_protocol_", None),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_", None),
        ("input_archive_id", "phase2b_recognizer_input_archive_v2_", None),
        ("input_archive_sha256", None, "hex"),
        ("input_archive_version", None, None),
        ("input_archive_policy_id", "phase2b_recognizer_input_archive_policy_v2_", None),
        ("batch_id", "phase2b_trusted_wire_batch_v2_", None),
        ("batch_policy_id", "phase2b_trusted_wire_batch_v2_policy_", None),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_", None),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_", None),
        ("semantic_conflict_row_ids_root", "phase2b_unsealed_semantic_conflict_rows_v2_", None),
        ("partition_union_row_ids_root", "phase2b_unsealed_partition_union_rows_v2_", None),
        ("main_answer_row_ids_root", _ANSWER_ROW_IDS_ROOT_PREFIX_V2, None),
        ("answer_manifest_sha256", None, "hex"),
        ("answer_manifest_id", _ANSWER_MANIFEST_ID_PREFIX_V2, None),
    )
    closed: dict[str, str] = {}
    for name, prefix, kind in text_specs:
        item = object.__getattribute__(value, name)
        if kind == "hex":
            closed[name] = _hex64_v2(item, name=f"gate-input V2 answer {name}")
        elif prefix is None:
            closed[name] = _exact_text_v2(item, name=f"gate-input V2 answer {name}")
        else:
            closed[name] = _digest_v2(item, prefix=prefix, name=f"gate-input V2 answer {name}")
    if (
        closed["schema_version"] != _ANSWER_MANIFEST_SCHEMA_VERSION_V2
        or closed["schema_id"] != _ANSWER_MANIFEST_SCHEMA_ID_V2
        or closed["policy_id"] != _ANSWER_MANIFEST_POLICY_ID_V2
        or closed["claim_level"] != FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL
        or closed["exact_freeze_id"] != _EXACT_FREEZE_ID_V2
        or closed["phase2b_protocol_id"] != _PHASE2B_PROTOCOL_ID_V2
    ):
        raise ValueError("gate-input V2 answer frozen identity drift")
    if (
        closed["input_archive_version"] != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or closed["input_archive_policy_id"] != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or closed["batch_policy_id"] != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or type(_FORMAL_SCORING_CONTRACT_ID_V2) is not str
    ):
        raise ValueError("gate-input V2 answer dependency identity drift")
    raw_rows = object.__getattribute__(value, "main_answer_rows")
    if type(raw_rows) is not tuple or len(raw_rows) != _MAIN_COUNT_V2:
        raise ValueError("gate-input V2 answer requires exact 720 rows")
    rows = tuple(_preflight_answer_row_v2(item) for item in raw_rows)
    input_ids = tuple(item[0] for item in rows)
    answer_ids = tuple(item[6] for item in rows)
    if (
        input_ids != tuple(sorted(input_ids))
        or len(set(input_ids)) != _MAIN_COUNT_V2
        or len(set(answer_ids)) != _MAIN_COUNT_V2
    ):
        raise ValueError("gate-input V2 answer input rows not sorted unique")
    observed = tuple(
        (case_type, sum(item[1] is case_type for item in rows))
        for case_type, count in _CASE_QUOTA_PER_CELL_V2
    )
    expected = tuple((case_type, count * _CELL_COUNT_V2) for case_type, count in _CASE_QUOTA_PER_CELL_V2)
    if observed != expected:
        raise ValueError("gate-input V2 answer case quota drift")
    return closed, rows


def _preflight_gate_rows_v2(
    values: object,
    answer_rows: tuple[tuple[object, ...], ...],
    *,
    allow_unissued_ids: bool,
) -> tuple[tuple[object, ...], ...]:
    if type(values) is not tuple or len(values) != _MAIN_COUNT_V2:
        raise ValueError("gate-input V2 requires exact 720 label rows")
    closed: list[tuple[object, ...]] = []
    for index, (value, answer) in enumerate(zip(values, answer_rows, strict=True)):
        if type(value) is not FormalUnsealedGateInputRowV2:
            raise TypeError("gate-input V2 row exact type drift")
        input_id = _digest_v2(object.__getattribute__(value, "input_row_id"), prefix="phase2b_recognizer_input_row_v2_", name="gate row input ID")
        answer_id = _digest_v2(object.__getattribute__(value, "answer_row_id"), prefix=_ANSWER_ROW_ID_PREFIX_V2, name="gate row answer ID")
        case_type = object.__getattribute__(value, "case_type")
        margin = object.__getattribute__(value, "margin_stratum")
        family = object.__getattribute__(value, "canonical_family_id")
        scale = object.__getattribute__(value, "scale_slice_id")
        latent = _digest_v2(object.__getattribute__(value, "latent_base_case_id"), prefix="phase2b_latent_base_case_v2_", name="gate row latent ID")
        row_id = object.__getattribute__(value, "gate_input_row_id")
        if type(case_type) is not Phase2BCaseType or type(margin) is not MarginStratum:
            raise TypeError("gate-input V2 case or margin enum drift")
        if type(family) is not CanonicalFamilyId or type(scale) is not FormalUnsealedScaleSliceIdV2:
            raise TypeError("gate-input V2 family or scale enum drift")
        if type(row_id) is not str:
            raise TypeError("gate-input V2 row ID text drift")
        if row_id == "" and allow_unissued_ids:
            pass
        else:
            _digest_v2(row_id, prefix=_GATE_ROW_ID_PREFIX_V2, name="gate-input V2 row ID")
        if (input_id, answer_id, case_type) != (answer[0], answer[6], answer[1]):
            raise ValueError(f"gate-input V2 row {index} answer parity drift")
        if case_type in _ANSWERABLE_CASE_TYPES_V2:
            if family is not answer[3]:
                raise ValueError("gate-input V2 answerable family label drift")
        elif answer[3] is not None:
            raise ValueError("gate-input V2 control answer family must remain null")
        nonunique = margin is MarginStratum.NONUNIQUE_OR_INSUFFICIENT
        should_be_nonunique = case_type in (
            Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
            Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,
        )
        if nonunique is not should_be_nonunique:
            raise ValueError("gate-input V2 margin/case composition drift")
        closed.append((input_id, answer_id, case_type, margin, family, scale, latent, row_id))
    result = tuple(closed)
    input_ids = tuple(item[0] for item in result)
    latent_ids = tuple(item[6] for item in result)
    if input_ids != tuple(sorted(input_ids)) or len(set(input_ids)) != _MAIN_COUNT_V2:
        raise ValueError("gate-input V2 rows not sorted unique")
    if len(set(latent_ids)) != _MAIN_COUNT_V2:
        raise ValueError("gate-input V2 latent IDs not unique")
    cells = tuple((family, scale) for family in CanonicalFamilyId for scale in FormalUnsealedScaleSliceIdV2)
    if len(cells) != _CELL_COUNT_V2:
        raise RuntimeError("gate-input V2 cell registry drift")
    for cell in cells:
        members = tuple(item for item in result if (item[4], item[5]) == cell)
        if len(members) != _CELL_SIZE_V2:
            raise ValueError("gate-input V2 family-scale cell quota drift")
        for case_type, count in _CASE_QUOTA_PER_CELL_V2:
            if sum(item[2] is case_type for item in members) != count:
                raise ValueError("gate-input V2 per-cell case quota drift")
        for margin, count in _MARGIN_QUOTA_PER_CELL_V2:
            if sum(item[3] is margin for item in members) != count:
                raise ValueError("gate-input V2 per-cell margin quota drift")
        nonunique = tuple(item[2] for item in members if item[3] is MarginStratum.NONUNIQUE_OR_INSUFFICIENT)
        if (
            nonunique.count(Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE) != 1
            or nonunique.count(Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE) != 8
            or len(nonunique) != 9
        ):
            raise ValueError("gate-input V2 nonunique composition drift")
    return result


def _gate_row_mapping_v2(item: tuple[object, ...]) -> dict[str, object]:
    return {
        "input_row_id": item[0], "answer_row_id": item[1],
        "case_type": item[2].value, "margin_stratum": item[3].value,
        "canonical_family_id": item[4].value, "scale_slice_id": item[5].value,
        "latent_base_case_id": item[6],
    }


def _finalize_gate_rows_v2(
    preflights: tuple[tuple[object, ...], ...],
) -> tuple[tuple[FormalUnsealedGateInputRowV2, ...], tuple[dict[str, object], ...]]:
    rows: list[FormalUnsealedGateInputRowV2] = []
    mappings: list[dict[str, object]] = []
    for item in preflights:
        preimage = _gate_row_mapping_v2(item)
        expected_id = _primitive_id_v2(preimage, domain=_GATE_ROW_DOMAIN_V2, prefix=_GATE_ROW_ID_PREFIX_V2)
        stored_id = item[7]
        if stored_id not in ("", expected_id):
            raise ValueError("gate-input V2 row content root drift")
        row = FormalUnsealedGateInputRowV2(
            input_row_id=item[0], answer_row_id=item[1], case_type=item[2],
            margin_stratum=item[3], canonical_family_id=item[4], scale_slice_id=item[5],
            latent_base_case_id=item[6], gate_input_row_id=expected_id,
        )
        rows.append(row)
        mappings.append({**preimage, "gate_input_row_id": expected_id})
    return tuple(rows), tuple(mappings)


def _answer_row_mapping_from_preflight_v2(item: tuple[object, ...]) -> dict[str, object]:
    return {
        "input_row_id": item[0],
        "case_type": item[1].value,
        "expected_decision": item[2].value,
        "canonical_family_id": None if item[3] is None else item[3].value,
        "binding": _binding_mapping_v2(item[4]),
        "admissible_scale_ids": list(item[5]),
        "answer_row_id": item[6],
    }


def _finalize_answer_manifest_v2(
    closed: dict[str, str],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    row_mappings: list[dict[str, object]] = []
    for item in rows:
        preimage = _answer_row_mapping_from_preflight_v2(item)
        stored_id = preimage.pop("answer_row_id")
        expected_id = _primitive_id_v2(
            preimage,
            domain=_ANSWER_ROW_DOMAIN_V2,
            prefix=_ANSWER_ROW_ID_PREFIX_V2,
        )
        if stored_id != expected_id:
            raise ValueError("gate-input V2 answer row content root drift")
        row_mappings.append({**preimage, "answer_row_id": stored_id})
    input_ids = tuple(item[0] for item in rows)
    answer_ids = tuple(item[6] for item in rows)
    main_root = _sequence_root_v2(
        input_ids,
        domain=b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00",
        prefix="phase2b_unsealed_main_rows_v2_",
    )
    answer_root = _sequence_root_v2(
        answer_ids, domain=_ANSWER_ROW_IDS_DOMAIN_V2,
        prefix=_ANSWER_ROW_IDS_ROOT_PREFIX_V2,
    )
    if main_root != closed["main_row_ids_root"] or answer_root != closed["main_answer_row_ids_root"]:
        raise ValueError("gate-input V2 answer row root drift")
    preimage = {
        "schema_version": closed["schema_version"], "schema_id": closed["schema_id"],
        "policy_id": closed["policy_id"], "claim_level": closed["claim_level"],
        "exact_freeze_id": closed["exact_freeze_id"],
        "phase2b_protocol_id": closed["phase2b_protocol_id"],
        "execution_freeze_manifest_id": closed["execution_freeze_manifest_id"],
        "input_archive_id": closed["input_archive_id"],
        "input_archive_sha256": closed["input_archive_sha256"],
        "input_archive_version": closed["input_archive_version"],
        "input_archive_policy_id": closed["input_archive_policy_id"],
        "batch_id": closed["batch_id"], "batch_policy_id": closed["batch_policy_id"],
        "ordered_archive_input_row_ids_root": closed["ordered_archive_input_row_ids_root"],
        "main_row_ids_root": closed["main_row_ids_root"],
        "semantic_conflict_row_ids_root": closed["semantic_conflict_row_ids_root"],
        "partition_union_row_ids_root": closed["partition_union_row_ids_root"],
        "main_answer_rows": row_mappings,
        "main_answer_row_ids_root": closed["main_answer_row_ids_root"],
    }
    expected_sha = hashlib.sha256(
        _ANSWER_MANIFEST_DOMAIN_V2 + canonical_json(preimage).encode("utf-8")
    ).hexdigest()
    if (
        closed["answer_manifest_sha256"] != expected_sha
        or closed["answer_manifest_id"] != _ANSWER_MANIFEST_ID_PREFIX_V2 + expected_sha
    ):
        raise ValueError("gate-input V2 answer manifest content root drift")


def _issue_gate_definition_v2(spec: tuple[object, ...]) -> ActualReplayGateInputDefinitionV2:
    name, scope, denominator, rule, available, missing = spec
    value = object.__new__(ActualReplayGateInputDefinitionV2)
    preimage = {
        "gate_name": name, "scope": scope, "expected_denominator": denominator,
        "success_rule": rule, "input_available": available,
        "missing_input_reason": missing,
    }
    for field_name, item in preimage.items():
        object.__setattr__(value, field_name, item)
    object.__setattr__(value, "definition_id", _primitive_id_v2(
        preimage, domain=_GATE_DEFINITION_DOMAIN_V2,
        prefix=_GATE_DEFINITION_ID_PREFIX_V2,
    ))
    return value


def _issue_evidence_v2(spec: tuple[str, str]) -> ActualReplayRequiredEvidenceV2:
    name, purpose = spec
    value = object.__new__(ActualReplayRequiredEvidenceV2)
    preimage = {
        "evidence_name": name, "purpose": purpose,
        "supplied_by_this_contract": False, "verifier_implemented": False,
    }
    for field_name, item in preimage.items():
        object.__setattr__(value, field_name, item)
    object.__setattr__(value, "requirement_id", _primitive_id_v2(
        preimage, domain=_REQUIRED_EVIDENCE_DOMAIN_V2,
        prefix=_REQUIRED_EVIDENCE_ID_PREFIX_V2,
    ))
    return value


def _fresh_definitions_v2(specs: tuple[tuple[object, ...], ...]) -> tuple[ActualReplayGateInputDefinitionV2, ...]:
    return tuple(_issue_gate_definition_v2(item) for item in specs)


def _fresh_evidence_v2() -> tuple[ActualReplayRequiredEvidenceV2, ...]:
    return tuple(_issue_evidence_v2(item) for item in _REQUIRED_EVIDENCE_SPECS_V2)


def _manifest_preimage_v2(
    value: FormalUnsealedGateInputManifestV2,
    *,
    row_mappings: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "schema_version": value.schema_version, "schema_id": value.schema_id,
        "policy_id": value.policy_id, "claim_level": value.claim_level,
        "exact_freeze_id": value.exact_freeze_id,
        "phase2b_protocol_id": value.phase2b_protocol_id,
        "formal_scoring_contract_id": value.formal_scoring_contract_id,
        "execution_freeze_manifest_id": value.execution_freeze_manifest_id,
        "input_archive_id": value.input_archive_id,
        "input_archive_sha256": value.input_archive_sha256,
        "input_archive_version": value.input_archive_version,
        "input_archive_policy_id": value.input_archive_policy_id,
        "batch_id": value.batch_id, "batch_policy_id": value.batch_policy_id,
        "ordered_archive_input_row_ids_root": value.ordered_archive_input_row_ids_root,
        "main_row_ids_root": value.main_row_ids_root,
        "semantic_conflict_row_ids_root": value.semantic_conflict_row_ids_root,
        "partition_union_row_ids_root": value.partition_union_row_ids_root,
        "answer_manifest_id": value.answer_manifest_id,
        "answer_manifest_sha256": value.answer_manifest_sha256,
        "main_answer_row_ids_root": value.main_answer_row_ids_root,
        "main_gate_input_rows": list(row_mappings),
        "main_gate_input_row_ids_root": value.main_gate_input_row_ids_root,
        "required_evidence_inventory": [
            {
                "evidence_name": item.evidence_name, "purpose": item.purpose,
                "supplied_by_this_contract": item.supplied_by_this_contract,
                "verifier_implemented": item.verifier_implemented,
                "requirement_id": item.requirement_id,
            }
            for item in value.required_evidence_inventory
        ],
    }


def _preflight_manifest_v2(
    value: object,
) -> tuple[
    dict[str, str],
    tuple[FormalUnsealedGateInputRowV2, ...],
    tuple[tuple[str, str, bool, bool, str], ...],
]:
    if type(value) is not FormalUnsealedGateInputManifestV2:
        raise TypeError("gate-input V2 manifest exact type drift")
    specs = (
        ("schema_version", None, None),
        ("schema_id", "phase2b_formal_unsealed_gate_input_manifest_schema_v2_", None),
        ("policy_id", "phase2b_formal_unsealed_gate_input_manifest_policy_v2_", None),
        ("claim_level", None, None),
        ("exact_freeze_id", "phase2b_exact_freeze_", None),
        ("phase2b_protocol_id", "phase2b_protocol_", None),
        ("formal_scoring_contract_id", "phase2b_formal_unsealed_prediction_scoring_contract_v2_", None),
        ("execution_freeze_manifest_id", "phase2b_execution_freeze_", None),
        ("input_archive_id", "phase2b_recognizer_input_archive_v2_", None),
        ("input_archive_sha256", None, "hex"),
        ("input_archive_version", None, None),
        ("input_archive_policy_id", "phase2b_recognizer_input_archive_policy_v2_", None),
        ("batch_id", "phase2b_trusted_wire_batch_v2_", None),
        ("batch_policy_id", "phase2b_trusted_wire_batch_v2_policy_", None),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_", None),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_", None),
        ("semantic_conflict_row_ids_root", "phase2b_unsealed_semantic_conflict_rows_v2_", None),
        ("partition_union_row_ids_root", "phase2b_unsealed_partition_union_rows_v2_", None),
        ("answer_manifest_id", _ANSWER_MANIFEST_ID_PREFIX_V2, None),
        ("answer_manifest_sha256", None, "hex"),
        ("main_answer_row_ids_root", _ANSWER_ROW_IDS_ROOT_PREFIX_V2, None),
        ("main_gate_input_row_ids_root", _GATE_ROW_IDS_ROOT_PREFIX_V2, None),
        ("gate_input_manifest_sha256", None, "hex"),
        ("gate_input_manifest_id", _GATE_MANIFEST_ID_PREFIX_V2, None),
    )
    closed: dict[str, str] = {}
    for name, prefix, kind in specs:
        item = object.__getattribute__(value, name)
        if kind == "hex":
            closed[name] = _hex64_v2(item, name=f"gate-input V2 manifest {name}")
        elif prefix is None:
            closed[name] = _exact_text_v2(item, name=f"gate-input V2 manifest {name}")
        else:
            closed[name] = _digest_v2(item, prefix=prefix, name=f"gate-input V2 manifest {name}")
    if (
        closed["schema_version"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION
        or closed["schema_id"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID
        or closed["policy_id"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID
        or closed["claim_level"] != ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
        or closed["exact_freeze_id"] != _EXACT_FREEZE_ID_V2
        or closed["phase2b_protocol_id"] != _PHASE2B_PROTOCOL_ID_V2
        or closed["formal_scoring_contract_id"] != _FORMAL_SCORING_CONTRACT_ID_V2
        or closed["input_archive_version"] != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or closed["input_archive_policy_id"] != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or closed["batch_policy_id"] != TRUSTED_WIRE_BATCH_V2_POLICY_ID
    ):
        raise ValueError("gate-input V2 manifest frozen identity drift")
    raw_rows = object.__getattribute__(value, "main_gate_input_rows")
    if type(raw_rows) is not tuple or len(raw_rows) != _MAIN_COUNT_V2:
        raise ValueError("gate-input V2 manifest row tuple drift")
    raw_inventory = object.__getattribute__(value, "required_evidence_inventory")
    if type(raw_inventory) is not tuple or len(raw_inventory) != len(_REQUIRED_EVIDENCE_SPECS_V2):
        raise ValueError("gate-input V2 evidence inventory count drift")
    inventory_preflights: list[tuple[str, str, bool, bool, str]] = []
    for actual, expected_spec in zip(raw_inventory, _REQUIRED_EVIDENCE_SPECS_V2, strict=True):
        if type(actual) is not ActualReplayRequiredEvidenceV2:
            raise TypeError("gate-input V2 evidence item type drift")
        name = _exact_text_v2(object.__getattribute__(actual, "evidence_name"), name="evidence name")
        purpose = _exact_text_v2(object.__getattribute__(actual, "purpose"), name="evidence purpose")
        supplied = object.__getattribute__(actual, "supplied_by_this_contract")
        verifier = object.__getattribute__(actual, "verifier_implemented")
        requirement_id = _digest_v2(object.__getattribute__(actual, "requirement_id"), prefix=_REQUIRED_EVIDENCE_ID_PREFIX_V2, name="evidence requirement ID")
        if type(supplied) is not bool or supplied or type(verifier) is not bool or verifier:
            raise ValueError("gate-input V2 evidence availability drift")
        if (name, purpose) != expected_spec:
            raise ValueError("gate-input V2 evidence registry drift")
        inventory_preflights.append((name, purpose, supplied, verifier, requirement_id))
    return closed, raw_rows, tuple(inventory_preflights)


def _finalize_evidence_v2(
    values: tuple[tuple[str, str, bool, bool, str], ...],
) -> tuple[ActualReplayRequiredEvidenceV2, ...]:
    fresh: list[ActualReplayRequiredEvidenceV2] = []
    for actual, expected_spec in zip(values, _REQUIRED_EVIDENCE_SPECS_V2, strict=True):
        expected = _issue_evidence_v2(expected_spec)
        if (
            actual[0] != expected.evidence_name
            or actual[1] != expected.purpose
            or actual[2] is not False
            or actual[3] is not False
            or actual[4] != expected.requirement_id
        ):
            raise ValueError("gate-input V2 evidence content root drift")
        fresh.append(expected)
    return tuple(fresh)


def _answer_manifest_bindings_v2(closed: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return (
        ("exact_freeze_id", closed["exact_freeze_id"]),
        ("phase2b_protocol_id", closed["phase2b_protocol_id"]),
        ("execution_freeze_manifest_id", closed["execution_freeze_manifest_id"]),
        ("input_archive_id", closed["input_archive_id"]),
        ("input_archive_sha256", closed["input_archive_sha256"]),
        ("input_archive_version", closed["input_archive_version"]),
        ("input_archive_policy_id", closed["input_archive_policy_id"]),
        ("batch_id", closed["batch_id"]),
        ("batch_policy_id", closed["batch_policy_id"]),
        ("ordered_archive_input_row_ids_root", closed["ordered_archive_input_row_ids_root"]),
        ("main_row_ids_root", closed["main_row_ids_root"]),
        ("semantic_conflict_row_ids_root", closed["semantic_conflict_row_ids_root"]),
        ("partition_union_row_ids_root", closed["partition_union_row_ids_root"]),
        ("answer_manifest_id", closed["answer_manifest_id"]),
        ("answer_manifest_sha256", closed["answer_manifest_sha256"]),
        ("main_answer_row_ids_root", closed["main_answer_row_ids_root"]),
    )


def _require_manifest_answer_bindings_v2(
    manifest: dict[str, str],
    answer: dict[str, str],
) -> None:
    if manifest["formal_scoring_contract_id"] != _FORMAL_SCORING_CONTRACT_ID_V2:
        raise ValueError("gate-input V2 formal contract binding drift")
    for name, expected in _answer_manifest_bindings_v2(answer):
        if manifest[name] != expected:
            raise ValueError(f"gate-input V2 answer cross-binding drift: {name}")


def _finalize_manifest_v2(
    closed: dict[str, str],
    preflights: tuple[tuple[object, ...], ...],
    inventory_preflights: tuple[tuple[str, str, bool, bool, str], ...],
) -> FormalUnsealedGateInputManifestV2:
    rows, mappings = _finalize_gate_rows_v2(preflights)
    row_ids = tuple(item.gate_input_row_id for item in rows)
    expected_root = _sequence_root_v2(
        row_ids,
        domain=_GATE_ROW_IDS_DOMAIN_V2,
        prefix=_GATE_ROW_IDS_ROOT_PREFIX_V2,
    )
    if closed["main_gate_input_row_ids_root"] != expected_root:
        raise ValueError("gate-input V2 label-row root drift")
    inventory = _finalize_evidence_v2(inventory_preflights)
    canonical = object.__new__(FormalUnsealedGateInputManifestV2)
    for name in _GATE_INPUT_MANIFEST_FIELDS_V2:
        if name in {"main_gate_input_rows", "required_evidence_inventory"}:
            continue
        if name in {"gate_input_manifest_sha256", "gate_input_manifest_id"}:
            object.__setattr__(canonical, name, closed[name])
        else:
            object.__setattr__(canonical, name, closed[name])
    object.__setattr__(canonical, "main_gate_input_rows", rows)
    object.__setattr__(canonical, "required_evidence_inventory", inventory)
    preimage = _manifest_preimage_v2(canonical, row_mappings=mappings)
    expected_sha = hashlib.sha256(
        _GATE_MANIFEST_DOMAIN_V2 + canonical_json(preimage).encode("utf-8")
    ).hexdigest()
    if (
        closed["gate_input_manifest_sha256"] != expected_sha
        or closed["gate_input_manifest_id"] != _GATE_MANIFEST_ID_PREFIX_V2 + expected_sha
    ):
        raise ValueError("gate-input V2 manifest content root drift")
    return canonical


def build_formal_unsealed_gate_input_manifest_v2(
    *,
    answer_manifest: FormalUnsealedAnswerManifestV2,
    main_gate_input_rows: tuple[FormalUnsealedGateInputRowV2, ...],
) -> FormalUnsealedGateInputManifestV2:
    """Build a package structurally intended for external precommitment."""

    answer_closed, answer_rows = _preflight_answer_manifest_v2(answer_manifest)
    gate_preflights = _preflight_gate_rows_v2(
        main_gate_input_rows,
        answer_rows,
        allow_unissued_ids=True,
    )
    # All 720 supplied answer and label rows are closed before the first
    # content hash or public commitment operation.
    _finalize_answer_manifest_v2(answer_closed, answer_rows)
    rows, row_mappings = _finalize_gate_rows_v2(gate_preflights)
    row_root = _sequence_root_v2(
        tuple(item.gate_input_row_id for item in rows),
        domain=_GATE_ROW_IDS_DOMAIN_V2,
        prefix=_GATE_ROW_IDS_ROOT_PREFIX_V2,
    )
    inventory = _fresh_evidence_v2()
    value = object.__new__(FormalUnsealedGateInputManifestV2)
    frozen = (
        ("schema_version", FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION),
        ("schema_id", FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID),
        ("policy_id", FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID),
        ("claim_level", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL),
        ("exact_freeze_id", answer_closed["exact_freeze_id"]),
        ("phase2b_protocol_id", answer_closed["phase2b_protocol_id"]),
        ("formal_scoring_contract_id", _FORMAL_SCORING_CONTRACT_ID_V2),
        ("execution_freeze_manifest_id", answer_closed["execution_freeze_manifest_id"]),
        ("input_archive_id", answer_closed["input_archive_id"]),
        ("input_archive_sha256", answer_closed["input_archive_sha256"]),
        ("input_archive_version", answer_closed["input_archive_version"]),
        ("input_archive_policy_id", answer_closed["input_archive_policy_id"]),
        ("batch_id", answer_closed["batch_id"]),
        ("batch_policy_id", answer_closed["batch_policy_id"]),
        ("ordered_archive_input_row_ids_root", answer_closed["ordered_archive_input_row_ids_root"]),
        ("main_row_ids_root", answer_closed["main_row_ids_root"]),
        ("semantic_conflict_row_ids_root", answer_closed["semantic_conflict_row_ids_root"]),
        ("partition_union_row_ids_root", answer_closed["partition_union_row_ids_root"]),
        ("answer_manifest_id", answer_closed["answer_manifest_id"]),
        ("answer_manifest_sha256", answer_closed["answer_manifest_sha256"]),
        ("main_answer_row_ids_root", answer_closed["main_answer_row_ids_root"]),
        ("main_gate_input_rows", rows),
        ("main_gate_input_row_ids_root", row_root),
        ("required_evidence_inventory", inventory),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    preimage = _manifest_preimage_v2(value, row_mappings=row_mappings)
    manifest_sha = hashlib.sha256(
        _GATE_MANIFEST_DOMAIN_V2 + canonical_json(preimage).encode("utf-8")
    ).hexdigest()
    object.__setattr__(value, "gate_input_manifest_sha256", manifest_sha)
    object.__setattr__(value, "gate_input_manifest_id", _GATE_MANIFEST_ID_PREFIX_V2 + manifest_sha)
    return value


def salted_gate_input_commitment_sha256_v2(
    gate_input_manifest_sha256: str,
    salt: str,
) -> str:
    """Domain-separated commitment helper; it proves no timing or authority."""

    if type(gate_input_manifest_sha256) is not str or re.fullmatch(
        r"[0-9a-f]{64}", gate_input_manifest_sha256
    ) is None:
        raise ValueError("gate-input V2 manifest SHA must be lowercase SHA-256")
    if type(salt) is not str:
        raise TypeError("gate-input V2 commitment salt must use exact text")
    encoded = salt.encode("utf-8")
    if len(encoded) < 32 or len(encoded) > _MAXIMUM_SALT_BYTES_V2:
        raise ValueError("gate-input V2 commitment salt length drift")
    digest = hashlib.sha256()
    digest.update(_GATE_COMMITMENT_DOMAIN_V2)
    digest.update(len(encoded).to_bytes(4, "big"))
    digest.update(encoded)
    digest.update(bytes.fromhex(gate_input_manifest_sha256))
    return digest.hexdigest()


def _issue_rejection_v2(
    reason: ActualUnsealed960ReplayInputReasonV2,
) -> ActualUnsealed960ReplayInputContractRejectionV2:
    value = object.__new__(ActualUnsealed960ReplayInputContractRejectionV2)
    frozen: list[tuple[str, object]] = [
        ("disposition", ActualUnsealed960ReplayInputDispositionV2.REJECTED),
        ("reason", reason),
        ("version", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION),
        ("schema_id", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID),
        ("policy_id", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID),
        ("claim_level", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL),
        ("validation", None),
        ("required_evidence_inventory", ()),
        ("available_overall_gate_input_definitions", ()),
        ("unavailable_overall_gate_input_definitions", ()),
        ("slice_gate_input_definitions", ()),
        ("metric_results", ()), ("scored_rows", ()), ("gate_results", ()),
        ("scale_regret_result", None), ("bootstrap_result", None),
        ("partial_output_published", False),
    ]
    frozen.extend((name, False) for name in (*_TRUE_RESULT_CLAIMS_V2, *_FALSE_RESULT_CLAIMS_V2))
    for name, item in frozen:
        object.__setattr__(value, name, item)
    return value


def _result_preimage_v2(
    value: ActualUnsealed960ReplayInputContractV2,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in fields(value):
        if item.name == "result_id":
            continue
        raw = object.__getattribute__(value, item.name)
        if isinstance(raw, Enum):
            result[item.name] = raw.value
        elif type(raw) is tuple:
            encoded: list[object] = []
            for nested in raw:
                if type(nested) is ActualReplayRequiredEvidenceV2:
                    encoded.append({name: object.__getattribute__(nested, name) for name in _REQUIRED_EVIDENCE_FIELDS_V2})
                elif type(nested) is ActualReplayGateInputDefinitionV2:
                    encoded.append({name: (object.__getattribute__(nested, name).value if isinstance(object.__getattribute__(nested, name), Enum) else object.__getattribute__(nested, name)) for name in _GATE_INPUT_DEFINITION_FIELDS_V2})
                else:
                    encoded.append(nested)
            result[item.name] = encoded
        else:
            result[item.name] = raw
    return result


def _issue_result_v2(
    manifest: FormalUnsealedGateInputManifestV2,
    commitment: str,
) -> ActualUnsealed960ReplayInputContractV2:
    value = object.__new__(ActualUnsealed960ReplayInputContractV2)
    frozen: list[tuple[str, object]] = [
        ("disposition", ActualUnsealed960ReplayInputDispositionV2.ACTUAL_REPLAY_CONTRACT_COMPLETE_NOT_EXECUTED),
        ("reason", ActualUnsealed960ReplayInputReasonV2.SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED),
        ("version", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION),
        ("schema_id", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID),
        ("policy_id", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID),
        ("claim_level", ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL),
        ("result_id", ""),
        ("gate_input_manifest_id", manifest.gate_input_manifest_id),
        ("gate_input_manifest_sha256", manifest.gate_input_manifest_sha256),
        ("salted_gate_input_commitment_sha256", commitment),
        ("gate_input_manifest_schema_id", manifest.schema_id),
        ("gate_input_manifest_policy_id", manifest.policy_id),
        ("answer_manifest_id", manifest.answer_manifest_id),
        ("answer_manifest_sha256", manifest.answer_manifest_sha256),
        ("execution_freeze_manifest_id", manifest.execution_freeze_manifest_id),
        ("input_archive_id", manifest.input_archive_id),
        ("input_archive_sha256", manifest.input_archive_sha256),
        ("input_archive_version", manifest.input_archive_version),
        ("input_archive_policy_id", manifest.input_archive_policy_id),
        ("batch_id", manifest.batch_id), ("batch_policy_id", manifest.batch_policy_id),
        ("exact_freeze_id", manifest.exact_freeze_id),
        ("protocol_id", manifest.phase2b_protocol_id),
        ("formal_scoring_contract_id", manifest.formal_scoring_contract_id),
        ("ordered_archive_input_row_ids_root", manifest.ordered_archive_input_row_ids_root),
        ("main_row_ids_root", manifest.main_row_ids_root),
        ("semantic_conflict_row_ids_root", manifest.semantic_conflict_row_ids_root),
        ("partition_union_row_ids_root", manifest.partition_union_row_ids_root),
        ("main_answer_row_ids_root", manifest.main_answer_row_ids_root),
        ("main_gate_input_row_ids_root", manifest.main_gate_input_row_ids_root),
        ("main_row_count", _MAIN_COUNT_V2),
        ("semantic_conflict_expected_row_count", _SEMANTIC_CONFLICT_COUNT_V2),
        ("total_expected_prediction_count", _TOTAL_COUNT_V2),
        ("unique_latent_base_case_id_count", _MAIN_COUNT_V2),
        ("family_scale_cell_count", _CELL_COUNT_V2),
    ]
    frozen.extend((name, True) for name in _TRUE_RESULT_CLAIMS_V2)
    frozen.extend((name, False) for name in _FALSE_RESULT_CLAIMS_V2)
    frozen.extend((
        ("required_evidence_inventory", _fresh_evidence_v2()),
        ("available_overall_gate_input_definitions", _fresh_definitions_v2(_AVAILABLE_OVERALL_GATE_INPUT_SPECS_V2)),
        ("unavailable_overall_gate_input_definitions", _fresh_definitions_v2(_UNAVAILABLE_OVERALL_GATE_INPUT_SPECS_V2)),
        ("slice_gate_input_definitions", _fresh_definitions_v2(_SLICE_GATE_INPUT_SPECS_V2)),
        ("metric_results", ()), ("scored_rows", ()), ("gate_results", ()),
        ("scale_regret_result", None), ("bootstrap_result", None),
    ))
    for name, item in frozen:
        object.__setattr__(value, name, item)
    object.__setattr__(value, "result_id", _primitive_id_v2(
        _result_preimage_v2(value), domain=_RESULT_DOMAIN_V2, prefix=_RESULT_ID_PREFIX_V2,
    ))
    return value


def validate_actual_unsealed_960_replay_input_contract_v2(
    *,
    gate_input_manifest: FormalUnsealedGateInputManifestV2,
    answer_manifest: FormalUnsealedAnswerManifestV2,
    revealed_gate_input_manifest_sha256: str,
    gate_input_commitment_salt: str,
    salted_gate_input_commitment_sha256: str,
) -> ActualUnsealed960ReplayInputContractV2 | ActualUnsealed960ReplayInputContractRejectionV2:
    """Validate supplied package mechanics without executing an actual replay."""

    if (
        type(gate_input_manifest) is not FormalUnsealedGateInputManifestV2
        or type(answer_manifest) is not FormalUnsealedAnswerManifestV2
    ):
        return _issue_rejection_v2(ActualUnsealed960ReplayInputReasonV2.WRONG_INPUT_TYPE)
    try:
        try:
            answer_version = _exact_text_v2(
                object.__getattribute__(answer_manifest, "schema_version"),
                name="gate-input V2 answer schema version",
            )
            manifest_version = _exact_text_v2(
                object.__getattribute__(gate_input_manifest, "schema_version"),
                name="gate-input V2 manifest schema version",
            )
            if (
                answer_version != _ANSWER_MANIFEST_SCHEMA_VERSION_V2
                or manifest_version
                != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION
            ):
                raise ValueError("gate-input V2 cross-version input")
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.CROSS_VERSION_INPUT
            ) from exc
        try:
            answer_closed, answer_rows = _preflight_answer_manifest_v2(answer_manifest)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.ANSWER_MANIFEST_INVALID
            ) from exc
        try:
            manifest_closed, manifest_rows, inventory = _preflight_manifest_v2(gate_input_manifest)
            gate_preflights = _preflight_gate_rows_v2(
                manifest_rows, answer_rows, allow_unissued_ids=False
            )
            _require_manifest_answer_bindings_v2(manifest_closed, answer_closed)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_MANIFEST_INVALID
            ) from exc
        try:
            revealed = _hex64_v2(
                revealed_gate_input_manifest_sha256,
                name="gate-input V2 revealed manifest SHA",
            )
            supplied_commitment = _hex64_v2(
                salted_gate_input_commitment_sha256,
                name="gate-input V2 supplied commitment",
            )
            salt = _exact_text_v2(
                gate_input_commitment_salt,
                name="gate-input V2 commitment salt",
                maximum_bytes=_MAXIMUM_SALT_BYTES_V2,
                ascii_only=False,
            )
            encoded_salt = salt.encode("utf-8")
            if len(encoded_salt) < 32 or revealed != manifest_closed["gate_input_manifest_sha256"]:
                raise ValueError("gate-input V2 opening preflight drift")
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_OPENING_INVALID
            ) from exc

        # All supplied inputs now closed.  Only this phase computes content IDs.
        try:
            _finalize_answer_manifest_v2(answer_closed, answer_rows)
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.ANSWER_MANIFEST_INVALID
            ) from exc
        try:
            canonical_manifest = _finalize_manifest_v2(
                manifest_closed, gate_preflights, inventory
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_MANIFEST_INVALID
            ) from exc
        try:
            expected_commitment = salted_gate_input_commitment_sha256_v2(revealed, salt)
            if supplied_commitment != expected_commitment:
                raise ValueError("gate-input V2 commitment opening drift")
        except (TypeError, ValueError, AttributeError) as exc:
            raise _ContractRejectedV2(
                ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_OPENING_INVALID
            ) from exc
        return _issue_result_v2(canonical_manifest, supplied_commitment)
    except _ContractRejectedV2 as exc:
        return _issue_rejection_v2(exc.reason)
    except Exception:
        return _issue_rejection_v2(ActualUnsealed960ReplayInputReasonV2.INTERNAL_ERROR)


_PUBLIC_VALUE_ACCEPTANCE_SEMANTICS_V2: Final = {
    "default_text": {
        "exact_builtin_type": "str",
        "nonempty": True,
        "encoding": "ascii",
        "maximum_encoded_bytes": _MAXIMUM_TEXT_BYTES_V2,
    },
    "lowercase_sha256": {
        "exact_hex_character_count": 64,
        "alphabet": "0123456789abcdef",
    },
    "uuid": {
        "exact_canonical_text_bytes": 36,
        "canonical_lowercase": True,
        "version": 4,
    },
    "answer_binding": {
        "exact_container_type": "tuple",
        "maximum_items": _MAXIMUM_BINDINGS_V2,
        "exact_item_type": "RoleBinding",
        "item_identifiers": "canonical_lowercase_uuidv4",
        "sort_order": "lexicographic_role_id_then_entity_id",
        "role_ids_injective": True,
        "entity_ids_injective": True,
    },
    "answer_admissible_scale_ids": {
        "exact_container_type": "tuple",
        "maximum_items": _MAXIMUM_SCALES_V2,
        "item_identifiers": "canonical_lowercase_uuidv4",
        "sort_order": "lexicographic",
        "unique": True,
    },
    "answer_payload_cardinality": {
        "unique_scale_answerable_binding_minimum": 1,
        "unique_scale_answerable_scale_count": 1,
        "admissible_scale_set_answerable_binding_minimum": 1,
        "admissible_scale_set_answerable_scale_minimum": 2,
        "nonanswerable_binding_count": 0,
        "nonanswerable_scale_count": 0,
    },
    "exact_public_collection_counts": {
        "main_answer_rows": _MAIN_COUNT_V2,
        "main_gate_input_rows": _MAIN_COUNT_V2,
        "semantic_conflict_expected_rows": _SEMANTIC_CONFLICT_COUNT_V2,
        "total_expected_prediction_rows": _TOTAL_COUNT_V2,
        "family_scale_cells": _CELL_COUNT_V2,
        "rows_per_family_scale_cell": _CELL_SIZE_V2,
        "unique_main_input_row_ids": _MAIN_COUNT_V2,
        "unique_main_answer_row_ids": _MAIN_COUNT_V2,
        "unique_latent_base_case_ids": _MAIN_COUNT_V2,
        "required_evidence_items": len(_REQUIRED_EVIDENCE_SPECS_V2),
        "available_overall_gate_input_definitions": len(
            _AVAILABLE_OVERALL_GATE_INPUT_SPECS_V2
        ),
        "unavailable_overall_gate_input_definitions": len(
            _UNAVAILABLE_OVERALL_GATE_INPUT_SPECS_V2
        ),
        "slice_gate_input_definitions": len(_SLICE_GATE_INPUT_SPECS_V2),
    },
    "sequence_root_framing": {
        "collection_count_unsigned_big_endian_bytes": 4,
        "ascii_item_length_unsigned_big_endian_bytes": 2,
    },
    "canonical_json": {
        "serializer": "hegel_machine.hashing.canonical_json",
        "encoded_as": "utf-8",
        "independent_maximum_encoded_bytes": None,
        "bounded_transitively_by_exact_collection_and_leaf_caps": True,
        "text_leaf_maximum_encoded_bytes": _MAXIMUM_TEXT_BYTES_V2,
        "binding_item_maximum": _MAXIMUM_BINDINGS_V2,
        "admissible_scale_item_maximum": _MAXIMUM_SCALES_V2,
    },
}
_COMMITMENT_ACCEPTANCE_SEMANTICS_V2: Final = {
    "salt_exact_builtin_type": "str",
    "salt_encoding": "utf-8",
    "salt_minimum_encoded_bytes": 32,
    "salt_maximum_encoded_bytes": _MAXIMUM_SALT_BYTES_V2,
    "salt_length_unsigned_big_endian_bytes": 4,
    "manifest_digest_input": "exact_64_character_lowercase_sha256_hex",
    "manifest_digest_hash_input": "raw_32_bytes",
    "commitment_output": "exact_64_character_lowercase_sha256_hex",
}


_schema_preimage_v2 = {
    "version": ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION,
    "row_fields": _GATE_INPUT_ROW_FIELDS_V2,
    "definition_fields": _GATE_INPUT_DEFINITION_FIELDS_V2,
    "required_evidence_fields": _REQUIRED_EVIDENCE_FIELDS_V2,
    "manifest_fields": _GATE_INPUT_MANIFEST_FIELDS_V2,
    "result_fields": _RESULT_FIELDS_V2,
    "rejection_fields": _REJECTION_FIELDS_V2,
    "scale_slice_values": tuple(item.value for item in FormalUnsealedScaleSliceIdV2),
    "disposition_values": tuple(
        item.value for item in ActualUnsealed960ReplayInputDispositionV2
    ),
    "reason_values": tuple(item.value for item in ActualUnsealed960ReplayInputReasonV2),
    "success_reason": (
        ActualUnsealed960ReplayInputReasonV2
        .SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED.value
    ),
    "empty_outputs": (
        "metric_results",
        "scored_rows",
        "gate_results",
        "scale_regret_result",
        "bootstrap_result",
    ),
    "public_value_acceptance_semantics": _PUBLIC_VALUE_ACCEPTANCE_SEMANTICS_V2,
    "commitment_acceptance_semantics": _COMMITMENT_ACCEPTANCE_SEMANTICS_V2,
}
ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID: Final = stable_hash(
    _schema_preimage_v2,
    prefix="phase2b_actual_unsealed_960_replay_input_contract_schema_v2_",
)
FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID: Final = stable_hash(
    {
        "version": FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION,
        "row_fields": _GATE_INPUT_ROW_FIELDS_V2,
        "required_evidence_fields": _REQUIRED_EVIDENCE_FIELDS_V2,
        "manifest_fields": _GATE_INPUT_MANIFEST_FIELDS_V2,
        "scale_slice_values": tuple(
            item.value for item in FormalUnsealedScaleSliceIdV2
        ),
        "public_value_acceptance_semantics": (
            _PUBLIC_VALUE_ACCEPTANCE_SEMANTICS_V2
        ),
    },
    prefix="phase2b_formal_unsealed_gate_input_manifest_schema_v2_",
)
_primitive_gate_specs_v2 = tuple(
    (
        item[0], item[1], item[2], item[3], item[4], item[5]
    )
    for item in (
        *_AVAILABLE_OVERALL_GATE_INPUT_SPECS_V2,
        *_UNAVAILABLE_OVERALL_GATE_INPUT_SPECS_V2,
        *_SLICE_GATE_INPUT_SPECS_V2,
    )
)
_manifest_policy_preimage_v2 = {
    "schema_version": FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION,
    "schema_id": FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID,
    "claim_level": ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL,
    "answer_dependency": {
        "schema_version": _ANSWER_MANIFEST_SCHEMA_VERSION_V2,
        "schema_id": _ANSWER_MANIFEST_SCHEMA_ID_V2,
        "policy_id": _ANSWER_MANIFEST_POLICY_ID_V2,
        "claim_level": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        "exact_freeze_id": _EXACT_FREEZE_ID_V2,
        "phase2b_protocol_id": _PHASE2B_PROTOCOL_ID_V2,
        "formal_scoring_contract_id": _FORMAL_SCORING_CONTRACT_ID_V2,
        "input_archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "input_archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    },
    "field_manifests": {
        "answer_row": _ANSWER_ROW_FIELDS_V2,
        "answer_manifest": _ANSWER_MANIFEST_FIELDS_V2,
        "gate_row": _GATE_INPUT_ROW_FIELDS_V2,
        "gate_definition": _GATE_INPUT_DEFINITION_FIELDS_V2,
        "required_evidence": _REQUIRED_EVIDENCE_FIELDS_V2,
        "gate_manifest": _GATE_INPUT_MANIFEST_FIELDS_V2,
    },
    "domains_and_prefixes": (
        (_ANSWER_ROW_DOMAIN_V2.decode("ascii"), _ANSWER_ROW_ID_PREFIX_V2),
        (_ANSWER_ROW_IDS_DOMAIN_V2.decode("ascii"), _ANSWER_ROW_IDS_ROOT_PREFIX_V2),
        (_ANSWER_MANIFEST_DOMAIN_V2.decode("ascii"), _ANSWER_MANIFEST_ID_PREFIX_V2),
        (_GATE_ROW_DOMAIN_V2.decode("ascii"), _GATE_ROW_ID_PREFIX_V2),
        (_GATE_ROW_IDS_DOMAIN_V2.decode("ascii"), _GATE_ROW_IDS_ROOT_PREFIX_V2),
        (_GATE_MANIFEST_DOMAIN_V2.decode("ascii"), _GATE_MANIFEST_ID_PREFIX_V2),
        (_GATE_COMMITMENT_DOMAIN_V2.decode("ascii"), "lowercase_sha256"),
        (_GATE_DEFINITION_DOMAIN_V2.decode("ascii"), _GATE_DEFINITION_ID_PREFIX_V2),
        (_REQUIRED_EVIDENCE_DOMAIN_V2.decode("ascii"), _REQUIRED_EVIDENCE_ID_PREFIX_V2),
        (_RESULT_DOMAIN_V2.decode("ascii"), _RESULT_ID_PREFIX_V2),
    ),
    "content_address_formulas": {
        "primitive": "prefix || sha256(domain || canonical_json(preimage))",
        "sequence": "prefix || sha256(domain || u32be(count) || repeated(u16be(ascii_id_len) || ascii_id))",
        "manifest": "sha256(domain || canonical_json(preimage)); id = prefix || sha256",
        "result": "prefix || sha256(domain || canonical_json(all_public_fields_except_result_id))",
    },
    "commitment_formula": (
        "sha256(domain || u32be(utf8_salt_len) || utf8_salt || raw_manifest_sha256)"
    ),
    "public_value_acceptance_semantics": _PUBLIC_VALUE_ACCEPTANCE_SEMANTICS_V2,
    "commitment_acceptance_semantics": _COMMITMENT_ACCEPTANCE_SEMANTICS_V2,
    "counts": (_MAIN_COUNT_V2, _SEMANTIC_CONFLICT_COUNT_V2, _TOTAL_COUNT_V2, _CELL_COUNT_V2, _CELL_SIZE_V2),
    "scale_slices": tuple(item.value for item in FormalUnsealedScaleSliceIdV2),
    "case_quota_per_cell": tuple((item.value, count) for item, count in _CASE_QUOTA_PER_CELL_V2),
    "margin_quota_per_cell": tuple((item.value, count) for item, count in _MARGIN_QUOTA_PER_CELL_V2),
    "nonunique_composition": (
        (Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE.value, 1),
        (Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE.value, 8),
    ),
    "gate_input_semantics": _primitive_gate_specs_v2,
    "input_available_meaning": "evaluator_label_side_input_available_not_numerator_metric_or_gate_execution",
    "required_unsupplied_evidence": _REQUIRED_EVIDENCE_SPECS_V2,
    "forbidden_downstream_fields": tuple(sorted(_FORBIDDEN_DOWNSTREAM_FIELDS_V2)),
    "validation_order": (
        "all_exact_types_slots_scalars_caps_enums_tuples",
        "all_720_answer_rows_and_gate_rows",
        "all_nested_evidence_items_crosslinks_and_quotas",
        "answer_row_and_manifest_recalculation",
        "gate_row_root_manifest_and_commitment_recalculation",
    ),
    "no_actions": (
        "no_prediction_decoder", "no_scoring_mechanics", "no_gate_or_wilson",
        "no_bootstrap_rng", "no_runner_files_network_or_ledger_transition",
    ),
}
FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID: Final = stable_hash(
    _manifest_policy_preimage_v2,
    prefix="phase2b_formal_unsealed_gate_input_manifest_policy_v2_",
)
_contract_policy_preimage_v2 = {
    "version": ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION,
    "schema_id": ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID,
    "manifest_policy_id": FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID,
    "claim_level": ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL,
    "success_disposition": (
        ActualUnsealed960ReplayInputDispositionV2
        .ACTUAL_REPLAY_CONTRACT_COMPLETE_NOT_EXECUTED.value
    ),
    "success_reason": (
        ActualUnsealed960ReplayInputReasonV2
        .SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED.value
    ),
    "true_claims": _TRUE_RESULT_CLAIMS_V2,
    "false_claims": _FALSE_RESULT_CLAIMS_V2,
    "empty_outputs": ("metric_results", "scored_rows", "gate_results"),
    "missing_outputs": ("scale_regret_result", "bootstrap_result"),
    "atomic_rejection": (
        "no_ids_rows_inventory_definitions_or_partial_output_all_claims_false"
    ),
    "result_id": "content_address_all_public_result_fields_except_result_id",
    "commitment_acceptance_semantics": _COMMITMENT_ACCEPTANCE_SEMANTICS_V2,
}
ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID: Final = stable_hash(
    _contract_policy_preimage_v2,
    prefix="phase2b_actual_unsealed_960_replay_input_contract_policy_v2_",
)


for _type_v2, _field_manifest_v2 in (
    (FormalUnsealedGateInputRowV2, _GATE_INPUT_ROW_FIELDS_V2),
    (ActualReplayGateInputDefinitionV2, _GATE_INPUT_DEFINITION_FIELDS_V2),
    (ActualReplayRequiredEvidenceV2, _REQUIRED_EVIDENCE_FIELDS_V2),
    (FormalUnsealedGateInputManifestV2, _GATE_INPUT_MANIFEST_FIELDS_V2),
    (ActualUnsealed960ReplayInputContractV2, _RESULT_FIELDS_V2),
    (ActualUnsealed960ReplayInputContractRejectionV2, _REJECTION_FIELDS_V2),
):
    if tuple(item.name for item in fields(_type_v2)) != _field_manifest_v2:
        raise RuntimeError(f"actual replay V2 {_type_v2.__name__} field drift")


__all__ = [
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION",
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL",
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID",
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID",
    "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION",
    "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID",
    "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID",
    "FormalUnsealedScaleSliceIdV2",
    "ActualUnsealed960ReplayInputDispositionV2",
    "ActualUnsealed960ReplayInputReasonV2",
    "FormalUnsealedGateInputRowV2",
    "ActualReplayGateInputDefinitionV2",
    "ActualReplayRequiredEvidenceV2",
    "FormalUnsealedGateInputManifestV2",
    "ActualUnsealed960ReplayInputContractV2",
    "ActualUnsealed960ReplayInputContractRejectionV2",
    "build_formal_unsealed_gate_input_manifest_v2",
    "salted_gate_input_commitment_sha256_v2",
    "validate_actual_unsealed_960_replay_input_contract_v2",
]
