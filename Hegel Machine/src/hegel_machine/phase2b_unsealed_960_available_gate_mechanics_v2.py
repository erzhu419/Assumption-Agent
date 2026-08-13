"""Pure composition of the available Phase-2B binary-gate mechanics.

This module accepts only already-materialized, non-authoritative mechanics
graphs.  It performs no archive decoding, scoring, execution, custody, or
formal gate evaluation.  Its positive result is deliberately narrower than a
scientific or operational Phase-2B result.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import math
from typing import Final
from uuid import UUID

from .hashing import canonical_json, stable_hash
from .phase2b_actual_unsealed_960_replay_input_contract_v2 import (
    ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL,
    ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID,
    ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID,
    ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION,
    FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID,
    FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID,
    FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION,
    ActualReplayGateInputDefinitionV2,
    ActualReplayRequiredEvidenceV2,
    ActualUnsealed960ReplayInputContractV2,
    ActualUnsealed960ReplayInputDispositionV2,
    ActualUnsealed960ReplayInputReasonV2,
    FormalUnsealedGateInputManifestV2,
    FormalUnsealedGateInputRowV2,
    FormalUnsealedScaleSliceIdV2,
)
from .phase2b_freeze_v1 import CanonicalFamilyId
from .phase2b_formal_unsealed_prediction_scoring_contract_v2 import (
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID,
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID,
    FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
    FormalUnsealedMetricKindV2,
)
from .phase2b_protocol import MarginStratum, Phase2BCaseType
from .phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from .phase2b_unsealed_960_prediction_scoring_mechanics_v2 import (
    UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL,
    UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID,
    UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID,
    UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION,
    Unsealed960MainRowResultV2,
    Unsealed960MetricResultV2,
    Unsealed960MetricRowOutcomeV2,
    Unsealed960PredictionScoringDispositionV2,
    Unsealed960PredictionScoringMechanicsV2,
    Unsealed960PredictionScoringReasonV2,
)
from .phase2b_wire import RoleBinding


UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION: Final = (
    "hegel-machine-phase2b-unsealed-960-available-gate-mechanics/2"
)
UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_AVAILABLE_GATE_MECHANICS_ONLY"
)

_MAIN_COUNT: Final = 720
_CHALLENGE_COUNT: Final = 240
_OVERALL_COUNT: Final = 10
_SLICE_COUNT: Final = 24
_UNAVAILABLE_COUNT: Final = 2
_TEXT_CAP: Final = 4_096
_BINDING_CAP: Final = 64
_SCALE_CAP: Final = 4_096
_ONE_SIDED_CONFIDENCE: Final = 0.95
_ONE_SIDED_Z_HEX: Final = "0x1.a515209676ab8p+0"

_SCORING_SOURCE_SHA256: Final = (
    "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
)
_REPLAY_INPUT_SOURCE_SHA256: Final = (
    "35da6f01163835ca90c24cdbd4ad85a1f7f0b2ef78ceb6e70fe80de2534814f2"
)
_FROZEN_PROTOCOL_ID: Final = (
    "phase2b_protocol_62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
_FROZEN_EXACT_FREEZE_ID: Final = (
    "phase2b_exact_freeze_ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
_FROZEN_FORMAL_SCORING_CONTRACT_ID: Final = (
    "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
    "37fce52fac6287a16d1925e76424d4d5b4e05fdcc552bc093d75d60a601d183e"
)
_FROZEN_INPUT_ARCHIVE_VERSION: Final = (
    "hegel-machine-phase2b-trusted-recognizer-input-archive/2"
)
_FROZEN_INPUT_ARCHIVE_POLICY_ID: Final = (
    "phase2b_recognizer_input_archive_policy_v2_"
    "529a91fdf2e8b5d545dd94002eabb4199685ead0577e3d9f803d24963324fc12"
)
_FROZEN_BATCH_POLICY_ID: Final = (
    "phase2b_trusted_wire_batch_v2_policy_"
    "be9672f8efb5867075b27b0342818c9caa97fe434f3bb76f84c612194da5b0e8"
)
_FROZEN_PREDICTION_ARCHIVE_VERSION: Final = (
    "hegel-machine-phase2b-recognizer-prediction-archive/2"
)
_FROZEN_PREDICTION_ARCHIVE_POLICY_ID: Final = (
    "phase2b_recognizer_prediction_archive_policy_v2_"
    "925a7e62d285ae8ea58b6c2f4ddea5111fa7482ec7957b82476b1341b41b905b"
)
_FROZEN_RUN_CONTEXT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-prediction-run-context/2"
)
_FROZEN_PREDICTION_RECORD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-recognizer-prediction-record/2"
)

_ANSWER_MANIFEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-unsealed-answer-manifest/2"
)
_ANSWER_MANIFEST_SCHEMA_ID: Final = (
    "phase2b_formal_unsealed_answer_manifest_schema_v2_"
    "3f427810029665a54854751b7d021a77c4d5f874b7df1992d50434b7108d32f0"
)
_ANSWER_MANIFEST_POLICY_ID: Final = (
    "phase2b_formal_unsealed_answer_manifest_policy_v2_"
    "be684716aadb4bb6cced67348233d0c6ca78d7e0c98c6df2542bcc1787c50f1e"
)
_ANSWER_ROW_DOMAIN: Final = b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW/V2\x00"
_ANSWER_MANIFEST_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_MANIFEST/V2\x00"
)
_ANSWER_ROW_PREFIX: Final = "phase2b_formal_unsealed_answer_row_v2_"
_ANSWER_MANIFEST_PREFIX: Final = "phase2b_formal_unsealed_answer_manifest_v2_"

_ANSWERABLE_CASES: Final = (
    Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
)
_CONTROL_CASES: Final = (
    Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE,
    Phase2BCaseType.BINDING_COUNTERFACTUAL,
    Phase2BCaseType.SCALE_COUNTERFACTUAL,
    Phase2BCaseType.SIGN_OR_INVARIANT_BREAK,
)

# name, denominator, point numerator/denominator, Wilson numerator/denominator
_OVERALL_GATE_SPECS: Final = (
    ("family_exact", 240, 9, 10, 86, 100),
    ("binding_exact", 240, 9, 10, 86, 100),
    ("scale_set_accuracy", 240, 87, 100, 82, 100),
    ("joint_exact", 240, 85, 100, 8, 10),
    ("hard_negative_rejection", 96, 95, 100, 9, 10),
    ("binding_counterfactual_rejection", 96, 95, 100, 9, 10),
    ("scale_counterfactual_rejection", 96, 93, 100, 88, 100),
    ("sign_or_invariant_break_rejection", 96, 95, 100, 9, 10),
    ("abstention_specificity", 228, 95, 100, 9, 10),
    ("nonidentifiable_scale_abstention", 96, 95, 100, 9, 10),
)
_SLICE_GATE_SPECS: Final = (
    ("answerable_joint_exact", 40, 4, 5, 7, 10),
    ("all_control_rejection", 64, 22, 25, 78, 100),
    ("abstention_specificity", 38, 17, 20, 3, 4),
)
_SCALE_SLICE_GATE_SPECS: Final = (
    ("answerable_joint_exact", 120, 4, 5, 7, 10),
    ("all_control_rejection", 192, 22, 25, 78, 100),
    ("abstention_specificity", 114, 17, 20, 3, 4),
)
_UNAVAILABLE_GATE_SPECS: Final = (
    (
        "fail_closed_rate", 1, 1, None, None,
        "durable_attempt_event_manifest_not_supplied",
    ),
    (
        "preservation_consistency", 97, 100, 94, 100,
        "preservation_pair_result_manifest_not_supplied",
    ),
)

_GATE_DEFINITION_SPECS: Final = (
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
    ("fail_closed_rate", "overall", None, "requires_attempt_event_denominator_not_abstention", False, "durable_attempt_event_manifest_not_supplied"),
    ("preservation_consistency", "overall", None, "requires_detached_496_legal_and_76_invalid_pair_results", False, "preservation_pair_result_manifest_not_supplied"),
    ("answerable_joint_exact", "family", 40, "expected_decision_family_binding_and_scale_all_exact", True, None),
    ("all_control_rejection", "family", 64, "four_control_case_types_predict_abstain_reason_unscored", True, None),
    ("abstention_specificity", "family", 38, "unique_answerable_decision_is_not_abstain", True, None),
    ("answerable_joint_exact", "scale", 120, "expected_decision_family_binding_and_scale_all_exact", True, None),
    ("all_control_rejection", "scale", 192, "four_control_case_types_predict_abstain_reason_unscored", True, None),
    ("abstention_specificity", "scale", 114, "unique_answerable_decision_is_not_abstain", True, None),
)

_METRIC_SPECS: Final = (
    ("answerable_count", "phase2b_formal_unsealed_metric_definition_v2_f842790d0f3f3573684195793fee5d6f773c697444f64f57309f55a0bbdd4586", "COUNT", ("unique_scale_answerable", "admissible_scale_set_answerable"), 240, "eligible_case_count_equals_exact_denominator", False),
    ("family_exact_accuracy", "phase2b_formal_unsealed_metric_definition_v2_d41dd3729f03c96f9f7956614172811e9e3917bb3a090b4f23746c432dffa3de", "BINARY_ACCURACY", ("unique_scale_answerable", "admissible_scale_set_answerable"), 240, "positive_decision_and_canonical_family_exact", False),
    ("binding_exact_accuracy", "phase2b_formal_unsealed_metric_definition_v2_063ff69910066d684da2798e036ee34dfbdda871cc36d2368f2841c9abfd77de", "BINARY_ACCURACY", ("unique_scale_answerable", "admissible_scale_set_answerable"), 240, "positive_decision_and_canonical_role_binding_tuple_exact", False),
    ("scale_set_accuracy", "phase2b_formal_unsealed_metric_definition_v2_87962cb124288ba52bd3c9fa7e447ec6cca7b1413bb6dcad9e51fa8901556ba9", "BINARY_ACCURACY", ("unique_scale_answerable", "admissible_scale_set_answerable"), 240, "positive_decision_and_canonical_admissible_scale_tuple_exact", False),
    ("unique_scale_accuracy", "phase2b_formal_unsealed_metric_definition_v2_975490f67189772d5a1817d1b823bf301ffef583fb33160f4ec0ff3f0ecaa888", "BINARY_ACCURACY", ("unique_scale_answerable",), 228, "unique_case_requires_ANSWER_and_exact_singleton_scale", False),
    ("joint_exact_accuracy", "phase2b_formal_unsealed_metric_definition_v2_5705b13d9379ff85568c2b4ae6afc9540c11bf13b7e260f95e562f14a66e3140", "BINARY_ACCURACY", ("unique_scale_answerable", "admissible_scale_set_answerable"), 240, "expected_decision_family_binding_and_scale_all_exact_set_requires_ANSWER_SET", False),
    ("abstention_specificity", "phase2b_formal_unsealed_metric_definition_v2_c310d7bec48084263e166d48fab530e303f02f53279e25805006a73a5b55dd60", "BINARY_ACCURACY", ("unique_scale_answerable",), 228, "unique_case_decision_is_not_ABSTAIN", False),
    ("nonidentifiability_abstention_accuracy", "phase2b_formal_unsealed_metric_definition_v2_75deb9137785a81af4a6cfb861120001527adf961c3ffa7041f7c8c6d905b48a", "BINARY_ACCURACY", ("insufficient_or_nonidentifiable",), 96, "insufficient_or_nonidentifiable_requires_ABSTAIN_reason_not_scored", False),
    ("set_valued_answer_accuracy", "phase2b_formal_unsealed_metric_definition_v2_f0e22be97f9fa046d5b2762bf519ef3e3df12ddc6938f18a1b8e0d2c9a71c418", "BINARY_ACCURACY", ("admissible_scale_set_answerable",), 12, "set_case_requires_ANSWER_SET_family_binding_and_full_scale_set_exact", True),
)

_EVIDENCE_SPECS: Final = (
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

_SCORING_TRUE_CLAIMS: Final = (
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
_SCORING_FALSE_CLAIMS: Final = (
    "challenge_in_main_denominator", "challenge_scoring_performed",
    "control_rejection_metrics_implemented", "formal_gate_evaluation_performed",
    "overall_gate_results_materialized", "slice_gate_metrics_implemented",
    "scale_regret_evaluated", "bootstrap_evaluated",
    "answer_manifest_authority_verified", "answer_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified", "input_archive_membership_verified",
    "batch_policy_membership_verified", "source_registry_projection_verified",
    "source_public_disjoint_verified", "single_live_allocation_verified",
    "secret_custodian_replay_verified", "execution_manifest_authority_verified",
    "partition_manifest_authority_verified", "derived_mapping_verified",
    "recognizer_executed", "runtime_executed", "actual_960_case_run_verified",
    "recognizer_capacity_evidence", "origin_authenticated", "formal_uuid_audit",
    "formal_covert_audit", "sealed_holdout_eligible", "scoring_performed",
    "prediction_scored", "actual_prediction_scoring_evidence", "effect_evidence",
    "c1_exit_evidence",
)

_SCORING_ADDRESS_PREFIXES: Final = {
    "schema_id": "phase2b_unsealed_960_prediction_scoring_mechanics_schema_v2_",
    "policy_id": "phase2b_unsealed_960_prediction_scoring_mechanics_policy_v2_",
    "result_id": "phase2b_unsealed_960_prediction_scoring_mechanics_v2_",
    "prediction_archive_id": "phase2b_recognizer_prediction_archive_v2_",
    "prediction_archive_policy_id": "phase2b_recognizer_prediction_archive_policy_v2_",
    "run_context_id": "phase2b_public_prediction_run_context_v2_",
    "input_archive_id": "phase2b_recognizer_input_archive_v2_",
    "batch_id": "phase2b_trusted_wire_batch_v2_",
    "execution_freeze_manifest_id": "phase2b_execution_freeze_",
    "protocol_id": "phase2b_protocol_",
    "structural_receipt_id": "phase2b_strict_recognizer_receipt_v2_",
    "partition_manifest_id": "phase2b_unsealed_prediction_partition_v2_",
    "answer_manifest_id": "phase2b_formal_unsealed_answer_manifest_v2_",
    "formal_scoring_contract_id": "phase2b_formal_unsealed_prediction_scoring_contract_v2_",
    "ordered_archive_input_row_ids_root": "phase2b_prediction_input_rows_v2_",
    "main_row_ids_root": "phase2b_unsealed_main_rows_v2_",
    "semantic_conflict_row_ids_root": "phase2b_unsealed_semantic_conflict_rows_v2_",
    "partition_union_row_ids_root": "phase2b_unsealed_partition_union_rows_v2_",
    "main_answer_row_ids_root": "phase2b_formal_unsealed_answer_rows_v2_",
}
_REPLAY_ADDRESS_PREFIXES: Final = {
    "schema_id": "phase2b_actual_unsealed_960_replay_input_contract_schema_v2_",
    "policy_id": "phase2b_actual_unsealed_960_replay_input_contract_policy_v2_",
    "result_id": "phase2b_actual_unsealed_960_replay_input_contract_v2_",
    "gate_input_manifest_id": "phase2b_formal_unsealed_gate_input_manifest_v2_",
    "gate_input_manifest_schema_id": "phase2b_formal_unsealed_gate_input_manifest_schema_v2_",
    "gate_input_manifest_policy_id": "phase2b_formal_unsealed_gate_input_manifest_policy_v2_",
    "answer_manifest_id": "phase2b_formal_unsealed_answer_manifest_v2_",
    "execution_freeze_manifest_id": "phase2b_execution_freeze_",
    "input_archive_id": "phase2b_recognizer_input_archive_v2_",
    "input_archive_policy_id": "phase2b_recognizer_input_archive_policy_v2_",
    "batch_id": "phase2b_trusted_wire_batch_v2_",
    "batch_policy_id": "phase2b_trusted_wire_batch_v2_policy_",
    "exact_freeze_id": "phase2b_exact_freeze_",
    "protocol_id": "phase2b_protocol_",
    "formal_scoring_contract_id": "phase2b_formal_unsealed_prediction_scoring_contract_v2_",
    "ordered_archive_input_row_ids_root": "phase2b_prediction_input_rows_v2_",
    "main_row_ids_root": "phase2b_unsealed_main_rows_v2_",
    "semantic_conflict_row_ids_root": "phase2b_unsealed_semantic_conflict_rows_v2_",
    "partition_union_row_ids_root": "phase2b_unsealed_partition_union_rows_v2_",
    "main_answer_row_ids_root": "phase2b_formal_unsealed_answer_rows_v2_",
    "main_gate_input_row_ids_root": "phase2b_actual_replay_gate_input_rows_v2_",
}
_MANIFEST_ADDRESS_PREFIXES: Final = {
    "schema_id": "phase2b_formal_unsealed_gate_input_manifest_schema_v2_",
    "policy_id": "phase2b_formal_unsealed_gate_input_manifest_policy_v2_",
    "exact_freeze_id": "phase2b_exact_freeze_",
    "phase2b_protocol_id": "phase2b_protocol_",
    "formal_scoring_contract_id": "phase2b_formal_unsealed_prediction_scoring_contract_v2_",
    "execution_freeze_manifest_id": "phase2b_execution_freeze_",
    "input_archive_id": "phase2b_recognizer_input_archive_v2_",
    "input_archive_policy_id": "phase2b_recognizer_input_archive_policy_v2_",
    "batch_id": "phase2b_trusted_wire_batch_v2_",
    "batch_policy_id": "phase2b_trusted_wire_batch_v2_policy_",
    "ordered_archive_input_row_ids_root": "phase2b_prediction_input_rows_v2_",
    "main_row_ids_root": "phase2b_unsealed_main_rows_v2_",
    "semantic_conflict_row_ids_root": "phase2b_unsealed_semantic_conflict_rows_v2_",
    "partition_union_row_ids_root": "phase2b_unsealed_partition_union_rows_v2_",
    "answer_manifest_id": "phase2b_formal_unsealed_answer_manifest_v2_",
    "main_answer_row_ids_root": "phase2b_formal_unsealed_answer_rows_v2_",
    "main_gate_input_row_ids_root": "phase2b_actual_replay_gate_input_rows_v2_",
    "gate_input_manifest_id": "phase2b_formal_unsealed_gate_input_manifest_v2_",
}
_SCORING_TOP_ENUM_FIELDS: Final = (
    ("disposition", Unsealed960PredictionScoringDispositionV2),
    ("reason", Unsealed960PredictionScoringReasonV2),
)
_SCORING_TOP_STR_FIELDS: Final = (
    "version", "schema_id", "policy_id", "claim_level", "result_id",
    "prediction_archive_id", "prediction_archive_sha256",
    "prediction_archive_version", "prediction_archive_policy_id",
    "run_context_id", "input_archive_id", "input_archive_sha256", "batch_id",
    "execution_freeze_manifest_id", "protocol_id", "structural_receipt_id",
    "partition_manifest_id", "answer_manifest_id", "answer_manifest_sha256",
    "salted_answer_commitment_sha256", "formal_scoring_contract_id",
    "ordered_archive_input_row_ids_root", "main_row_ids_root",
    "semantic_conflict_row_ids_root", "partition_union_row_ids_root",
    "main_answer_row_ids_root",
)
_SCORING_TOP_INT_FIELDS: Final = (
    "total_prediction_count", "main_row_result_count",
    "metric_eligible_main_row_count", "control_row_without_frozen_metric_count",
    "semantic_conflict_excluded_count",
)
_SCORING_TOP_BOOL_FIELDS: Final = (*_SCORING_TRUE_CLAIMS, *_SCORING_FALSE_CLAIMS)
_SCORING_TOP_TUPLE_FIELDS: Final = (
    "metric_results", "main_row_results", "gate_results",
)
_SCORING_TOP_NONE_FIELDS: Final = ("scale_regret_result", "bootstrap_result")

_REPLAY_TOP_ENUM_FIELDS: Final = (
    ("disposition", ActualUnsealed960ReplayInputDispositionV2),
    ("reason", ActualUnsealed960ReplayInputReasonV2),
)
_REPLAY_TOP_STR_FIELDS: Final = (
    "version", "schema_id", "policy_id", "claim_level", "result_id",
    "gate_input_manifest_id", "gate_input_manifest_sha256",
    "salted_gate_input_commitment_sha256", "gate_input_manifest_schema_id",
    "gate_input_manifest_policy_id", "answer_manifest_id",
    "answer_manifest_sha256", "execution_freeze_manifest_id",
    "input_archive_id", "input_archive_sha256", "input_archive_version",
    "input_archive_policy_id", "batch_id", "batch_policy_id", "exact_freeze_id",
    "protocol_id", "formal_scoring_contract_id",
    "ordered_archive_input_row_ids_root", "main_row_ids_root",
    "semantic_conflict_row_ids_root", "partition_union_row_ids_root",
    "main_answer_row_ids_root", "main_gate_input_row_ids_root",
)
_REPLAY_TOP_INT_FIELDS: Final = (
    "main_row_count", "semantic_conflict_expected_row_count",
    "total_expected_prediction_count", "unique_latent_base_case_id_count",
    "family_scale_cell_count",
)
_REPLAY_TOP_TUPLE_FIELDS: Final = (
    "required_evidence_inventory", "available_overall_gate_input_definitions",
    "unavailable_overall_gate_input_definitions", "slice_gate_input_definitions",
    "metric_results", "scored_rows", "gate_results",
)
_REPLAY_TOP_NONE_FIELDS: Final = ("scale_regret_result", "bootstrap_result")
_REPLAY_TOP_BOOL_FIELDS: Final = tuple(
    item.name
    for item in fields(ActualUnsealed960ReplayInputContractV2)
    if item.name not in (
        {name for name, _enum_type in _REPLAY_TOP_ENUM_FIELDS}
        | set(_REPLAY_TOP_STR_FIELDS)
        | set(_REPLAY_TOP_INT_FIELDS)
        | set(_REPLAY_TOP_TUPLE_FIELDS)
        | set(_REPLAY_TOP_NONE_FIELDS)
    )
)

_MANIFEST_TOP_TUPLE_FIELDS: Final = (
    "main_gate_input_rows", "required_evidence_inventory",
)
_MANIFEST_TOP_STR_FIELDS: Final = tuple(
    item.name
    for item in fields(FormalUnsealedGateInputManifestV2)
    if item.name not in _MANIFEST_TOP_TUPLE_FIELDS
)
_TOP_LEVEL_EXACT_TYPE_MANIFEST: Final = (
    ("scoring", _SCORING_TOP_ENUM_FIELDS, _SCORING_TOP_STR_FIELDS,
     _SCORING_TOP_INT_FIELDS, _SCORING_TOP_BOOL_FIELDS,
     _SCORING_TOP_TUPLE_FIELDS, _SCORING_TOP_NONE_FIELDS),
    ("replay", _REPLAY_TOP_ENUM_FIELDS, _REPLAY_TOP_STR_FIELDS,
     _REPLAY_TOP_INT_FIELDS, _REPLAY_TOP_BOOL_FIELDS,
     _REPLAY_TOP_TUPLE_FIELDS, _REPLAY_TOP_NONE_FIELDS),
    ("gate_manifest", (), _MANIFEST_TOP_STR_FIELDS, (), (),
     _MANIFEST_TOP_TUPLE_FIELDS, ()),
)
_TOP_LEVEL_EXACT_TYPE_POLICY_MANIFEST: Final = tuple(
    (
        scope,
        tuple((name, enum_type.__name__) for name, enum_type in enum_fields),
        tuple(str_fields), tuple(int_fields), tuple(bool_fields),
        tuple(tuple_fields), tuple(none_fields),
    )
    for (
        scope, enum_fields, str_fields, int_fields, bool_fields,
        tuple_fields, none_fields,
    ) in _TOP_LEVEL_EXACT_TYPE_MANIFEST
)
for _top_type, _enum_fields, _str_fields, _int_fields, _bool_fields, _tuple_fields, _none_fields in (
    (Unsealed960PredictionScoringMechanicsV2, *_TOP_LEVEL_EXACT_TYPE_MANIFEST[0][1:]),
    (ActualUnsealed960ReplayInputContractV2, *_TOP_LEVEL_EXACT_TYPE_MANIFEST[1][1:]),
    (FormalUnsealedGateInputManifestV2, *_TOP_LEVEL_EXACT_TYPE_MANIFEST[2][1:]),
):
    _partition = (
        {name for name, _enum_type in _enum_fields}
        | set(_str_fields)
        | set(_int_fields)
        | set(_bool_fields)
        | set(_tuple_fields)
        | set(_none_fields)
    )
    if _partition != {item.name for item in fields(_top_type)}:
        raise RuntimeError(f"available-gate V2 {_top_type.__name__} type partition drift")
    _flat_partition = (
        tuple(name for name, _enum_type in _enum_fields)
        + tuple(_str_fields) + tuple(_int_fields) + tuple(_bool_fields)
        + tuple(_tuple_fields) + tuple(_none_fields)
    )
    if len(_flat_partition) != len(set(_flat_partition)):
        raise RuntimeError(f"available-gate V2 {_top_type.__name__} overlapping type partition")
_CROSS_BINDING_MANIFEST: Final = (
    ("input_archive_id", "scoring.input_archive_id", "replay.input_archive_id", "manifest.input_archive_id"),
    ("input_archive_sha256", "scoring.input_archive_sha256", "replay.input_archive_sha256", "manifest.input_archive_sha256"),
    ("batch_id", "scoring.batch_id", "replay.batch_id", "manifest.batch_id"),
    ("execution_freeze_manifest_id", "scoring.execution_freeze_manifest_id", "replay.execution_freeze_manifest_id", "manifest.execution_freeze_manifest_id"),
    ("ordered_archive_input_row_ids_root", "scoring.ordered_archive_input_row_ids_root", "replay.ordered_archive_input_row_ids_root", "manifest.ordered_archive_input_row_ids_root"),
    ("partition_union_row_ids_root", "scoring.partition_union_row_ids_root", "replay.partition_union_row_ids_root", "manifest.partition_union_row_ids_root"),
    ("answer_manifest_id", "scoring.answer_manifest_id", "replay.answer_manifest_id", "manifest.answer_manifest_id"),
    ("answer_manifest_sha256", "scoring.answer_manifest_sha256", "replay.answer_manifest_sha256", "manifest.answer_manifest_sha256"),
    ("main_row_ids_root", "scoring.main_row_ids_root", "replay.main_row_ids_root", "manifest.main_row_ids_root"),
    ("semantic_conflict_row_ids_root", "scoring.semantic_conflict_row_ids_root", "replay.semantic_conflict_row_ids_root", "manifest.semantic_conflict_row_ids_root"),
    ("main_answer_row_ids_root", "scoring.main_answer_row_ids_root", "replay.main_answer_row_ids_root", "manifest.main_answer_row_ids_root"),
    ("protocol_id", "scoring.protocol_id", "replay.protocol_id", "manifest.phase2b_protocol_id"),
    ("formal_scoring_contract_id", "scoring.formal_scoring_contract_id", "replay.formal_scoring_contract_id", "manifest.formal_scoring_contract_id"),
    ("gate_input_manifest_id", "replay.gate_input_manifest_id", "manifest.gate_input_manifest_id"),
    ("gate_input_manifest_sha256", "replay.gate_input_manifest_sha256", "manifest.gate_input_manifest_sha256"),
    ("main_gate_input_row_ids_root", "replay.main_gate_input_row_ids_root", "manifest.main_gate_input_row_ids_root"),
    ("input_archive_version", "replay.input_archive_version", "manifest.input_archive_version"),
    ("input_archive_policy_id", "replay.input_archive_policy_id", "manifest.input_archive_policy_id"),
    ("batch_policy_id", "replay.batch_policy_id", "manifest.batch_policy_id"),
    ("exact_freeze_id", "replay.exact_freeze_id", "manifest.exact_freeze_id"),
    ("gate_input_manifest_schema_id", "replay.gate_input_manifest_schema_id", "manifest.schema_id"),
    ("gate_input_manifest_policy_id", "replay.gate_input_manifest_policy_id", "manifest.policy_id"),
)
_REPLAY_TRUE_CLAIMS: Final = (
    "exact_contract_identity_verified", "answer_gate_manifest_cross_binding_verified",
    "supplied_gate_input_commitment_opening_verified", "exact_main_gate_row_coverage_verified",
    "exact_family_scale_cell_quota_verified", "exact_case_type_per_cell_quota_verified",
    "exact_margin_per_cell_quota_verified", "exact_nonunique_margin_case_composition_verified",
    "supplied_family_slice_labels_complete", "supplied_scale_slice_labels_complete",
    "unique_latent_base_case_ids_verified",
    "downstream_prediction_identifier_fields_absent_from_schema_verified",
    "semantic_conflict_root_bound_and_exclusion_contract_frozen",
    "control_gate_input_semantics_frozen", "slice_gate_input_semantics_frozen",
    "required_unsupplied_evidence_inventory_frozen",
)
_REPLAY_FALSE_CLAIMS: Final = (
    "challenge_in_main_denominator", "margin_stratum_authority_verified",
    "family_slice_label_authority_verified", "scale_slice_semantics_authority_verified",
    "latent_case_independence_verified", "one_shot_policy_enforced",
    "durable_attempt_ledger_verified", "raw_input_archive_replayed",
    "raw_prediction_archive_replayed", "prediction_commit_before_reveal_verified",
    "wilson_bounds_evaluated", "preservation_evaluated",
    "challenge_descriptor_rows_implemented", "challenge_scoring_performed",
    "fail_closed_gate_inputs_contract_complete", "preservation_gate_inputs_contract_complete",
    "scale_regret_inputs_contract_complete", "bootstrap_inputs_contract_complete",
    "answer_manifest_authority_verified", "gate_input_manifest_authority_verified",
    "answer_commitment_authority_verified", "gate_input_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified", "input_archive_membership_verified",
    "batch_policy_membership_verified", "source_registry_projection_verified",
    "source_public_disjoint_verified", "single_live_allocation_verified",
    "secret_custodian_replay_verified", "execution_manifest_authority_verified",
    "partition_manifest_authority_verified", "derived_mapping_verified",
    "recognizer_executed", "runtime_executed", "actual_960_case_run_verified",
    "recognizer_capacity_evidence", "origin_authenticated", "formal_uuid_audit",
    "formal_covert_audit", "sealed_holdout_eligible", "scoring_performed",
    "prediction_scored", "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed", "metric_results_materialized",
    "scored_rows_materialized", "overall_gate_results_materialized",
    "slice_gate_results_materialized", "scale_regret_evaluated",
    "bootstrap_evaluated", "effect_evidence", "c1_exit_evidence",
)
if _REPLAY_TOP_BOOL_FIELDS != (*_REPLAY_TRUE_CLAIMS, *_REPLAY_FALSE_CLAIMS):
    raise RuntimeError("available-gate V2 replay boolean field/claim drift")


class Unsealed960AvailableGateMechanicsDispositionV2(str, Enum):
    AVAILABLE_GATE_MECHANICS_COMPLETE_NOT_FORMAL_GATE_EVALUATION = (
        "AVAILABLE_GATE_MECHANICS_COMPLETE_NOT_FORMAL_GATE_EVALUATION"
    )
    REJECTED = "REJECTED"


class Unsealed960AvailableGateMechanicsReasonV2(str, Enum):
    TEN_OVERALL_AND_TWENTY_FOUR_SLICE_MECHANICS_COMPLETE = (
        "TEN_OVERALL_AND_TWENTY_FOUR_SLICE_MECHANICS_COMPLETE"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    SCORING_MECHANICS_INVALID = "SCORING_MECHANICS_INVALID"
    REPLAY_INPUT_CONTRACT_INVALID = "REPLAY_INPUT_CONTRACT_INVALID"
    GATE_INPUT_MANIFEST_INVALID = "GATE_INPUT_MANIFEST_INVALID"
    CROSS_BINDING_MISMATCH = "CROSS_BINDING_MISMATCH"
    ROW_JOIN_OR_QUOTA_MISMATCH = "ROW_JOIN_OR_QUOTA_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class _AvailableGateRejected(Exception):
    def __init__(self, reason: Unsealed960AvailableGateMechanicsReasonV2) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _CrossVersionInput(Exception):
    pass


class _RowJoinOrQuota(Exception):
    pass


_AVAILABLE_RESULT_FIELDS: Final = (
    "metric_name", "scope", "slice_id", "gate_input_definition_id",
    "successes", "total",
    "expected_denominator", "minimum_point_estimate_ratio",
    "minimum_wilson_lcb_ratio", "point_estimate_ratio",
    "point_estimate_hex", "one_sided_wilson_lcb_hex",
    "point_threshold_passed", "wilson_threshold_passed",
    "available_gate_passed", "result_id",
)
_UNAVAILABLE_RESULT_FIELDS: Final = (
    "metric_name", "scope", "gate_input_definition_id",
    "minimum_point_estimate_ratio",
    "minimum_wilson_lcb_ratio", "expected_denominator", "successes",
    "total", "point_estimate_ratio", "point_estimate_hex",
    "one_sided_wilson_lcb_hex", "point_threshold_passed",
    "wilson_threshold_passed", "available_gate_passed",
    "missing_input_reason", "unavailable_id",
)

_TRUE_CLAIMS: Final = (
    "supplied_scoring_mechanics_graph_independently_verified",
    "supplied_replay_input_contract_graph_independently_verified",
    "supplied_gate_input_manifest_graph_independently_verified",
    "three_supplied_graphs_cross_bound",
    "exact_main_720_row_join_verified",
    "ten_available_overall_gate_mechanics_results_materialized",
    "twenty_four_available_slice_gate_mechanics_results_materialized",
    "frozen_threshold_identity_verified",
    "one_sided_95_percent_wilson_mechanics_evaluated",
    "semantic_conflict_240_excluded_from_available_mechanics",
    "two_unavailable_gate_inputs_retained",
    "atomic_fail_closed_rejection_verified",
)
_FALSE_CLAIMS: Final = (
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "metric_results_materialized",
    "scored_rows_materialized",
    "formal_wilson_gate_bounds_evaluated",
    "upstream_scoring_control_rejection_metrics_implemented",
    "upstream_scoring_slice_gate_metrics_implemented",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "challenge_in_main_denominator",
    "challenge_scoring_performed",
    "challenge_descriptor_rows_implemented",
    "fail_closed_rate_evaluated",
    "preservation_consistency_evaluated",
    "preservation_evaluated",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "fail_closed_gate_inputs_contract_complete",
    "preservation_gate_inputs_contract_complete",
    "scale_regret_inputs_contract_complete",
    "bootstrap_inputs_contract_complete",
    "baseline_outputs_verified",
    "margin_stratum_authority_verified",
    "family_slice_label_authority_verified",
    "scale_slice_semantics_authority_verified",
    "latent_case_independence_verified",
    "raw_input_archive_replayed",
    "raw_prediction_archive_replayed",
    "answer_commitment_opening_verified",
    "gate_input_commitment_opening_verified",
    "prediction_commit_before_reveal_verified",
    "evidence_supplied",
    "evidence_verified",
    "answer_manifest_authority_verified",
    "gate_input_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    "one_shot_policy_enforced",
    "durable_attempt_ledger_verified",
    "secret_custodian_replay_verified",
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
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
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)

_SUCCESS_IDENTITY_FIELDS: Final = (
    "disposition", "reason", "version", "schema_id", "policy_id",
    "claim_level", "result_id", "scoring_mechanics_result_id",
    "replay_input_contract_result_id", "gate_input_manifest_id",
    "gate_input_manifest_sha256", "gate_input_manifest_schema_id",
    "gate_input_manifest_policy_id", "prediction_archive_id",
    "prediction_archive_sha256", "answer_manifest_id",
    "answer_manifest_sha256", "scoring_mechanics_schema_id",
    "scoring_mechanics_policy_id", "scoring_mechanics_version",
    "scoring_mechanics_claim_level", "replay_input_contract_schema_id",
    "replay_input_contract_policy_id", "replay_input_contract_version",
    "replay_input_contract_claim_level", "gate_input_manifest_schema_version",
    "gate_input_manifest_claim_level", "protocol_id",
    "formal_scoring_contract_id", "formal_scoring_contract_schema_id",
    "formal_scoring_contract_policy_id", "formal_scoring_contract_version",
    "formal_scoring_contract_claim_level",
    "main_row_ids_root", "semantic_conflict_row_ids_root",
    "main_answer_row_ids_root", "main_gate_input_row_ids_root",
    "main_row_count", "semantic_conflict_excluded_count",
    "overall_result_count", "slice_result_count", "unavailable_result_count",
)
_SUCCESS_FIELDS: Final = (
    *_SUCCESS_IDENTITY_FIELDS, *_TRUE_CLAIMS, *_FALSE_CLAIMS,
    "available_overall_gate_mechanics_results",
    "available_slice_gate_mechanics_results",
    "unavailable_gate_mechanics",
)
_REJECTION_FIELDS: Final = (
    "disposition", "reason", "version", "schema_id", "policy_id",
    "claim_level", "validation", "available_overall_gate_mechanics_results",
    "available_slice_gate_mechanics_results", "unavailable_gate_mechanics",
    "partial_output_published", *_TRUE_CLAIMS, *_FALSE_CLAIMS,
)


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960AvailableGateMechanicsResultV2:
    metric_name: str
    scope: str
    slice_id: str | None
    gate_input_definition_id: str
    successes: int
    total: int
    expected_denominator: int
    minimum_point_estimate_ratio: str
    minimum_wilson_lcb_ratio: str
    point_estimate_ratio: str
    point_estimate_hex: str
    one_sided_wilson_lcb_hex: str
    point_threshold_passed: bool
    wilson_threshold_passed: bool
    available_gate_passed: bool
    result_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("available-gate V2 results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960UnavailableGateMechanicsV2:
    metric_name: str
    scope: str
    gate_input_definition_id: str
    minimum_point_estimate_ratio: str
    minimum_wilson_lcb_ratio: str | None
    expected_denominator: None
    successes: None
    total: None
    point_estimate_ratio: None
    point_estimate_hex: None
    one_sided_wilson_lcb_hex: None
    point_threshold_passed: None
    wilson_threshold_passed: None
    available_gate_passed: None
    missing_input_reason: str
    unavailable_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unavailable-gate V2 records are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960AvailableGateMechanicsV2:
    disposition: Unsealed960AvailableGateMechanicsDispositionV2
    reason: Unsealed960AvailableGateMechanicsReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    result_id: str
    scoring_mechanics_result_id: str
    replay_input_contract_result_id: str
    gate_input_manifest_id: str
    gate_input_manifest_sha256: str
    gate_input_manifest_schema_id: str
    gate_input_manifest_policy_id: str
    prediction_archive_id: str
    prediction_archive_sha256: str
    answer_manifest_id: str
    answer_manifest_sha256: str
    scoring_mechanics_schema_id: str
    scoring_mechanics_policy_id: str
    scoring_mechanics_version: str
    scoring_mechanics_claim_level: str
    replay_input_contract_schema_id: str
    replay_input_contract_policy_id: str
    replay_input_contract_version: str
    replay_input_contract_claim_level: str
    gate_input_manifest_schema_version: str
    gate_input_manifest_claim_level: str
    protocol_id: str
    formal_scoring_contract_id: str
    formal_scoring_contract_schema_id: str
    formal_scoring_contract_policy_id: str
    formal_scoring_contract_version: str
    formal_scoring_contract_claim_level: str
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    main_answer_row_ids_root: str
    main_gate_input_row_ids_root: str
    main_row_count: int
    semantic_conflict_excluded_count: int
    overall_result_count: int
    slice_result_count: int
    unavailable_result_count: int
    supplied_scoring_mechanics_graph_independently_verified: bool
    supplied_replay_input_contract_graph_independently_verified: bool
    supplied_gate_input_manifest_graph_independently_verified: bool
    three_supplied_graphs_cross_bound: bool
    exact_main_720_row_join_verified: bool
    ten_available_overall_gate_mechanics_results_materialized: bool
    twenty_four_available_slice_gate_mechanics_results_materialized: bool
    frozen_threshold_identity_verified: bool
    one_sided_95_percent_wilson_mechanics_evaluated: bool
    semantic_conflict_240_excluded_from_available_mechanics: bool
    two_unavailable_gate_inputs_retained: bool
    atomic_fail_closed_rejection_verified: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
    formal_wilson_gate_bounds_evaluated: bool
    upstream_scoring_control_rejection_metrics_implemented: bool
    upstream_scoring_slice_gate_metrics_implemented: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    challenge_in_main_denominator: bool
    challenge_scoring_performed: bool
    challenge_descriptor_rows_implemented: bool
    fail_closed_rate_evaluated: bool
    preservation_consistency_evaluated: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    fail_closed_gate_inputs_contract_complete: bool
    preservation_gate_inputs_contract_complete: bool
    scale_regret_inputs_contract_complete: bool
    bootstrap_inputs_contract_complete: bool
    baseline_outputs_verified: bool
    margin_stratum_authority_verified: bool
    family_slice_label_authority_verified: bool
    scale_slice_semantics_authority_verified: bool
    latent_case_independence_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    prediction_commit_before_reveal_verified: bool
    evidence_supplied: bool
    evidence_verified: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    one_shot_policy_enforced: bool
    durable_attempt_ledger_verified: bool
    secret_custodian_replay_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
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
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool
    available_overall_gate_mechanics_results: tuple[Unsealed960AvailableGateMechanicsResultV2, ...]
    available_slice_gate_mechanics_results: tuple[Unsealed960AvailableGateMechanicsResultV2, ...]
    unavailable_gate_mechanics: tuple[Unsealed960UnavailableGateMechanicsV2, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("available-gate V2 mechanics are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class Unsealed960AvailableGateMechanicsRejectionV2:
    disposition: Unsealed960AvailableGateMechanicsDispositionV2
    reason: Unsealed960AvailableGateMechanicsReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    validation: None
    available_overall_gate_mechanics_results: tuple[()]
    available_slice_gate_mechanics_results: tuple[()]
    unavailable_gate_mechanics: tuple[()]
    partial_output_published: bool
    supplied_scoring_mechanics_graph_independently_verified: bool
    supplied_replay_input_contract_graph_independently_verified: bool
    supplied_gate_input_manifest_graph_independently_verified: bool
    three_supplied_graphs_cross_bound: bool
    exact_main_720_row_join_verified: bool
    ten_available_overall_gate_mechanics_results_materialized: bool
    twenty_four_available_slice_gate_mechanics_results_materialized: bool
    frozen_threshold_identity_verified: bool
    one_sided_95_percent_wilson_mechanics_evaluated: bool
    semantic_conflict_240_excluded_from_available_mechanics: bool
    two_unavailable_gate_inputs_retained: bool
    atomic_fail_closed_rejection_verified: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
    formal_wilson_gate_bounds_evaluated: bool
    upstream_scoring_control_rejection_metrics_implemented: bool
    upstream_scoring_slice_gate_metrics_implemented: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    challenge_in_main_denominator: bool
    challenge_scoring_performed: bool
    challenge_descriptor_rows_implemented: bool
    fail_closed_rate_evaluated: bool
    preservation_consistency_evaluated: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    fail_closed_gate_inputs_contract_complete: bool
    preservation_gate_inputs_contract_complete: bool
    scale_regret_inputs_contract_complete: bool
    bootstrap_inputs_contract_complete: bool
    baseline_outputs_verified: bool
    margin_stratum_authority_verified: bool
    family_slice_label_authority_verified: bool
    scale_slice_semantics_authority_verified: bool
    latent_case_independence_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    prediction_commit_before_reveal_verified: bool
    evidence_supplied: bool
    evidence_verified: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    one_shot_policy_enforced: bool
    durable_attempt_ledger_verified: bool
    secret_custodian_replay_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
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
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("available-gate V2 rejections are privately issued")


_RESULT_DOMAIN: Final = b"HEGEL/PHASE2B/AVAILABLE_GATE/RESULT/V2\x00"
_UNAVAILABLE_DOMAIN: Final = b"HEGEL/PHASE2B/AVAILABLE_GATE/UNAVAILABLE/V2\x00"
_SUCCESS_DOMAIN: Final = b"HEGEL/PHASE2B/AVAILABLE_GATE/MECHANICS/V2\x00"
_RESULT_PREFIX: Final = "phase2b_unsealed_960_available_gate_result_v2_"
_UNAVAILABLE_PREFIX: Final = "phase2b_unsealed_960_unavailable_gate_v2_"
_SUCCESS_PREFIX: Final = "phase2b_unsealed_960_available_gate_mechanics_v2_"


def _stable_id(mapping: object, *, domain: bytes, prefix: str) -> str:
    return prefix + hashlib.sha256(
        domain + canonical_json(mapping).encode("utf-8")
    ).hexdigest()


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    if not 1 <= len(encoded) <= _TEXT_CAP:
        raise ValueError(f"{name} text cap drift")
    return value


def _sha(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _id(value: object, prefix: str, name: str) -> str:
    text = _text(value, name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _sha(text[len(prefix):], f"{name} suffix")
    return text


def _uuid(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text.encode("utf-8")) != 36:
        raise ValueError(f"{name} UUID length drift")
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{name} UUID drift") from exc
    if parsed.version != 4 or str(parsed) != text:
        raise ValueError(f"{name} must be canonical lowercase UUIDv4")
    return text


def _integer(value: object, name: str, *, maximum: int = 10_000) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} integer drift")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} boolean drift")
    return value


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator < 0:
        raise ValueError("ratio outside frozen domain")
    divisor = math.gcd(numerator, denominator)
    return f"{numerator // divisor}/{denominator // divisor}"


def _sequence_root(values: tuple[str, ...], *, domain: bytes, prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(values).to_bytes(4, "big"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return prefix + digest.hexdigest()


def _binding_snapshot(value: object, name: str) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or len(value) > _BINDING_CAP:
        raise TypeError(f"{name} binding tuple drift")
    result: list[tuple[str, str]] = []
    for item in value:
        if type(item) is not RoleBinding:
            raise TypeError(f"{name} binding item drift")
        role = _uuid(object.__getattribute__(item, "role_id"), f"{name} role")
        entity = _uuid(object.__getattribute__(item, "entity_id"), f"{name} entity")
        result.append((role, entity))
    closed = tuple(result)
    if closed != tuple(sorted(closed)):
        raise ValueError(f"{name} binding order drift")
    if len({item[0] for item in closed}) != len(closed) or len({item[1] for item in closed}) != len(closed):
        raise ValueError(f"{name} binding injectivity drift")
    return closed


def _binding_primitive(value: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"role_id": role, "entity_id": entity} for role, entity in value]


def _outcome_snapshot(value: object) -> dict[str, object]:
    if type(value) is not Unsealed960MetricRowOutcomeV2:
        raise TypeError("metric outcome exact type drift")
    result = {
        "metric_definition_id": _id(object.__getattribute__(value, "metric_definition_id"), "phase2b_formal_unsealed_metric_definition_v2_", "metric definition ID"),
        "metric_name": _text(object.__getattribute__(value, "metric_name"), "metric name"),
        "eligible": _boolean(object.__getattribute__(value, "eligible"), "metric eligibility"),
        "success": object.__getattribute__(value, "success"),
        "metric_row_outcome_id": _id(object.__getattribute__(value, "metric_row_outcome_id"), "phase2b_unsealed_960_metric_row_outcome_v2_", "metric outcome ID"),
    }
    if result["success"] is not None and type(result["success"]) is not bool:
        raise TypeError("metric outcome success drift")
    return result


def _row_snapshot(value: object) -> dict[str, object]:
    if type(value) is not Unsealed960MainRowResultV2:
        raise TypeError("main row exact type drift")
    case_type = object.__getattribute__(value, "case_type")
    predicted_decision = object.__getattribute__(value, "predicted_decision")
    expected_decision = object.__getattribute__(value, "expected_decision")
    predicted_family = object.__getattribute__(value, "predicted_canonical_family_id")
    expected_family = object.__getattribute__(value, "expected_canonical_family_id")
    if type(case_type) is not Phase2BCaseType:
        raise TypeError("main row case type drift")
    if type(predicted_decision) is not PredictionDecisionV2 or type(expected_decision) is not PredictionDecisionV2:
        raise TypeError("main row decision drift")
    if predicted_family is not None and type(predicted_family) is not CanonicalFamilyId:
        raise TypeError("predicted family drift")
    if expected_family is not None and type(expected_family) is not CanonicalFamilyId:
        raise TypeError("expected family drift")
    predicted_scales = object.__getattribute__(value, "predicted_admissible_scale_ids")
    expected_scales = object.__getattribute__(value, "expected_admissible_scale_ids")
    if type(predicted_scales) is not tuple or type(expected_scales) is not tuple or len(predicted_scales) > _SCALE_CAP or len(expected_scales) > _SCALE_CAP:
        raise TypeError("scale tuple drift")
    predicted_scales = tuple(_uuid(item, "predicted scale ID") for item in predicted_scales)
    expected_scales = tuple(_uuid(item, "expected scale ID") for item in expected_scales)
    if predicted_scales != tuple(sorted(set(predicted_scales))) or expected_scales != tuple(sorted(set(expected_scales))):
        raise ValueError("scale tuple canonical order drift")
    outcomes_raw = object.__getattribute__(value, "metric_outcomes")
    if type(outcomes_raw) is not tuple or len(outcomes_raw) != 9:
        raise ValueError("main row needs exact nine outcomes")
    flags: dict[str, object] = {}
    for name in ("decision_exact", "family_exact", "binding_exact", "scale_set_exact", "joint_exact", "metric_eligible"):
        item = object.__getattribute__(value, name)
        if item is not None and type(item) is not bool:
            raise TypeError(f"main row {name} drift")
        flags[name] = item
    return {
        "input_row_id": _id(object.__getattribute__(value, "input_row_id"), "phase2b_recognizer_input_row_v2_", "main input row ID"),
        "prediction_record_id": _id(object.__getattribute__(value, "prediction_record_id"), "phase2b_recognizer_prediction_record_v2_", "prediction record ID"),
        "prediction_content_id": _id(object.__getattribute__(value, "prediction_content_id"), "phase2b_prediction_", "prediction content ID"),
        "answer_row_id": _id(object.__getattribute__(value, "answer_row_id"), "phase2b_formal_unsealed_answer_row_v2_", "answer row ID"),
        "case_type": case_type,
        "predicted_decision": predicted_decision,
        "expected_decision": expected_decision,
        "predicted_canonical_family_id": predicted_family,
        "expected_canonical_family_id": expected_family,
        "predicted_binding": _binding_snapshot(object.__getattribute__(value, "predicted_binding"), "predicted"),
        "expected_binding": _binding_snapshot(object.__getattribute__(value, "expected_binding"), "expected"),
        "predicted_admissible_scale_ids": predicted_scales,
        "expected_admissible_scale_ids": expected_scales,
        **flags,
        "metric_outcomes": tuple(_outcome_snapshot(item) for item in outcomes_raw),
        "row_result_id": _id(object.__getattribute__(value, "row_result_id"), "phase2b_unsealed_960_main_row_result_v2_", "main row result ID"),
    }


def _metric_snapshot(value: object) -> dict[str, object]:
    if type(value) is not Unsealed960MetricResultV2:
        raise TypeError("metric result exact type drift")
    kind = object.__getattribute__(value, "metric_kind")
    case_types = object.__getattribute__(value, "denominator_case_types")
    if type(kind) is not FormalUnsealedMetricKindV2 or type(case_types) is not tuple or len(case_types) > 7:
        raise TypeError("metric result kind/cases drift")
    if any(type(item) is not Phase2BCaseType for item in case_types):
        raise TypeError("metric denominator case enum drift")
    success = object.__getattribute__(value, "success_count")
    count = object.__getattribute__(value, "count_value")
    if success is not None:
        success = _integer(success, "metric success", maximum=_MAIN_COUNT)
    if count is not None:
        count = _integer(count, "metric count", maximum=_MAIN_COUNT)
    return {
        "metric_definition_id": _id(object.__getattribute__(value, "metric_definition_id"), "phase2b_formal_unsealed_metric_definition_v2_", "metric result definition ID"),
        "metric_name": _text(object.__getattribute__(value, "metric_name"), "metric result name"),
        "metric_kind": kind,
        "denominator_case_types": tuple(case_types),
        "expected_denominator": _integer(object.__getattribute__(value, "expected_denominator"), "expected denominator", maximum=_MAIN_COUNT),
        "observed_denominator": _integer(object.__getattribute__(value, "observed_denominator"), "observed denominator", maximum=_MAIN_COUNT),
        "success_count": success,
        "count_value": count,
        "success_rule": _text(object.__getattribute__(value, "success_rule"), "metric success rule"),
        "separately_reported": _boolean(object.__getattribute__(value, "separately_reported"), "separately reported"),
        "metric_result_id": _id(object.__getattribute__(value, "metric_result_id"), "phase2b_unsealed_960_metric_result_v2_", "metric result ID"),
    }


def _snapshot_top(value: object, expected_type: type[object]) -> dict[str, object]:
    if type(value) is not expected_type:
        raise TypeError(f"{expected_type.__name__} exact type drift")
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(expected_type)
    }


def _definition_snapshot(value: object) -> dict[str, object]:
    if type(value) is not ActualReplayGateInputDefinitionV2:
        raise TypeError("gate definition exact type drift")
    denominator = object.__getattribute__(value, "expected_denominator")
    if denominator is not None:
        denominator = _integer(denominator, "gate definition denominator", maximum=_MAIN_COUNT)
    missing = object.__getattribute__(value, "missing_input_reason")
    if missing is not None:
        missing = _text(missing, "gate missing reason")
    return {
        "gate_name": _text(object.__getattribute__(value, "gate_name"), "gate name"),
        "scope": _text(object.__getattribute__(value, "scope"), "gate scope"),
        "expected_denominator": denominator,
        "success_rule": _text(object.__getattribute__(value, "success_rule"), "gate success rule"),
        "input_available": _boolean(object.__getattribute__(value, "input_available"), "gate input available"),
        "missing_input_reason": missing,
        "definition_id": _id(object.__getattribute__(value, "definition_id"), "phase2b_actual_replay_gate_input_definition_v2_", "gate definition ID"),
    }


def _evidence_snapshot(value: object) -> dict[str, object]:
    if type(value) is not ActualReplayRequiredEvidenceV2:
        raise TypeError("required evidence exact type drift")
    return {
        "evidence_name": _text(object.__getattribute__(value, "evidence_name"), "evidence name"),
        "purpose": _text(object.__getattribute__(value, "purpose"), "evidence purpose"),
        "supplied_by_this_contract": _boolean(object.__getattribute__(value, "supplied_by_this_contract"), "evidence supplied"),
        "verifier_implemented": _boolean(object.__getattribute__(value, "verifier_implemented"), "evidence verifier"),
        "requirement_id": _id(object.__getattribute__(value, "requirement_id"), "phase2b_actual_replay_required_evidence_v2_", "evidence requirement ID"),
    }


def _gate_row_snapshot(value: object) -> dict[str, object]:
    if type(value) is not FormalUnsealedGateInputRowV2:
        raise TypeError("gate row exact type drift")
    case_type = object.__getattribute__(value, "case_type")
    margin = object.__getattribute__(value, "margin_stratum")
    family = object.__getattribute__(value, "canonical_family_id")
    scale = object.__getattribute__(value, "scale_slice_id")
    if type(case_type) is not Phase2BCaseType or type(margin) is not MarginStratum:
        raise TypeError("gate row case/margin drift")
    if type(family) is not CanonicalFamilyId or type(scale) is not FormalUnsealedScaleSliceIdV2:
        raise TypeError("gate row slice label drift")
    return {
        "input_row_id": _id(object.__getattribute__(value, "input_row_id"), "phase2b_recognizer_input_row_v2_", "gate input row ID"),
        "answer_row_id": _id(object.__getattribute__(value, "answer_row_id"), "phase2b_formal_unsealed_answer_row_v2_", "gate answer row ID"),
        "case_type": case_type,
        "margin_stratum": margin,
        "canonical_family_id": family,
        "scale_slice_id": scale,
        "latent_base_case_id": _id(object.__getattribute__(value, "latent_base_case_id"), "phase2b_latent_base_case_v2_", "latent base case ID"),
        "gate_input_row_id": _id(object.__getattribute__(value, "gate_input_row_id"), "phase2b_actual_replay_gate_input_row_v2_", "gate row content ID"),
    }


def _snapshot_scoring(value: object) -> dict[str, object]:
    raw = _snapshot_top(value, Unsealed960PredictionScoringMechanicsV2)
    for name, enum_type in _SCORING_TOP_ENUM_FIELDS:
        if type(raw[name]) is not enum_type:
            raise TypeError(f"scoring {name} exact enum type drift")
    for name in _SCORING_TOP_STR_FIELDS:
        if type(raw[name]) is not str:
            raise TypeError(f"scoring {name} exact string type drift")
    for name in _SCORING_TOP_INT_FIELDS:
        if type(raw[name]) is not int:
            raise TypeError(f"scoring {name} exact integer type drift")
    for name in _SCORING_TOP_BOOL_FIELDS:
        if type(raw[name]) is not bool:
            raise TypeError(f"scoring {name} exact boolean type drift")
    for name in _SCORING_TOP_TUPLE_FIELDS:
        if type(raw[name]) is not tuple:
            raise TypeError(f"scoring {name} exact tuple type drift")
    for name in _SCORING_TOP_NONE_FIELDS:
        if raw[name] is not None:
            raise TypeError(f"scoring {name} exact None drift")
    if raw["version"] != UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION:
        raise _CrossVersionInput("scoring mechanics version drift")
    rows_raw = raw["main_row_results"]
    metrics_raw = raw["metric_results"]
    if type(rows_raw) is not tuple or len(rows_raw) != _MAIN_COUNT:
        raise ValueError("scoring graph row count drift")
    if type(metrics_raw) is not tuple or len(metrics_raw) != 9:
        raise ValueError("scoring graph metric count drift")
    for name in ("gate_results",):
        if type(raw[name]) is not tuple or raw[name]:
            raise ValueError(f"scoring {name} must remain empty")
    if raw["scale_regret_result"] is not None or raw["bootstrap_result"] is not None:
        raise ValueError("scoring auxiliary result unexpectedly present")
    if (
        raw["disposition"] is not Unsealed960PredictionScoringDispositionV2.MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION
        or raw["reason"] is not Unsealed960PredictionScoringReasonV2.CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE
        or raw["schema_id"] != UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID
        or raw["policy_id"] != UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID
        or raw["claim_level"] != UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL
        or raw["prediction_archive_version"] != _FROZEN_PREDICTION_ARCHIVE_VERSION
        or raw["prediction_archive_policy_id"] != _FROZEN_PREDICTION_ARCHIVE_POLICY_ID
        or raw["protocol_id"] != _FROZEN_PROTOCOL_ID
        or raw["formal_scoring_contract_id"] != _FROZEN_FORMAL_SCORING_CONTRACT_ID
    ):
        raise ValueError("scoring mechanics identity drift")
    for name in _SCORING_TRUE_CLAIMS:
        if raw[name] is not True:
            raise ValueError(f"scoring true claim drift: {name}")
    for name in _SCORING_FALSE_CLAIMS:
        if raw[name] is not False:
            raise ValueError(f"scoring false claim drift: {name}")
    exact_counts = {
        "total_prediction_count": 960,
        "main_row_result_count": 720,
        "metric_eligible_main_row_count": 336,
        "control_row_without_frozen_metric_count": 384,
        "semantic_conflict_excluded_count": 240,
    }
    for name, expected in exact_counts.items():
        if _integer(raw[name], f"scoring {name}", maximum=960) != expected:
            raise ValueError(f"scoring count drift: {name}")
    for name, prefix in (
        ("result_id", "phase2b_unsealed_960_prediction_scoring_mechanics_v2_"),
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_"),
        ("answer_manifest_id", "phase2b_formal_unsealed_answer_manifest_v2_"),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_"),
        ("semantic_conflict_row_ids_root", "phase2b_unsealed_semantic_conflict_rows_v2_"),
        ("main_answer_row_ids_root", "phase2b_formal_unsealed_answer_rows_v2_"),
    ):
        _id(raw[name], prefix, f"scoring {name}")
    for name in ("prediction_archive_sha256", "input_archive_sha256", "answer_manifest_sha256", "salted_answer_commitment_sha256"):
        _sha(raw[name], f"scoring {name}")
    for item in fields(Unsealed960PredictionScoringMechanicsV2):
        if item.name in {"metric_results", "main_row_results", "gate_results", "scale_regret_result", "bootstrap_result"}:
            continue
        candidate = raw[item.name]
        if type(candidate) is str:
            if item.name.endswith("sha256"):
                _sha(candidate, f"scoring {item.name}")
            elif item.name.endswith("_id") or item.name.endswith("_root"):
                prefix = _SCORING_ADDRESS_PREFIXES.get(item.name)
                if prefix is None:
                    raise ValueError(f"unregistered scoring content address: {item.name}")
                _id(candidate, prefix, f"scoring {item.name}")
            else:
                _text(candidate, f"scoring {item.name}")
        elif type(candidate) is int:
            _integer(candidate, f"scoring {item.name}", maximum=10_000)
        elif type(candidate) is bool or type(candidate) in (
            Unsealed960PredictionScoringDispositionV2,
            Unsealed960PredictionScoringReasonV2,
        ):
            pass
        else:
            raise TypeError(f"scoring scalar type drift: {item.name}")
    raw["main_row_results"] = tuple(_row_snapshot(item) for item in rows_raw)
    raw["metric_results"] = tuple(_metric_snapshot(item) for item in metrics_raw)
    return raw


def _snapshot_replay(value: object) -> dict[str, object]:
    raw = _snapshot_top(value, ActualUnsealed960ReplayInputContractV2)
    for name, enum_type in _REPLAY_TOP_ENUM_FIELDS:
        if type(raw[name]) is not enum_type:
            raise TypeError(f"replay {name} exact enum type drift")
    for name in _REPLAY_TOP_STR_FIELDS:
        if type(raw[name]) is not str:
            raise TypeError(f"replay {name} exact string type drift")
    for name in _REPLAY_TOP_INT_FIELDS:
        if type(raw[name]) is not int:
            raise TypeError(f"replay {name} exact integer type drift")
    for name in _REPLAY_TOP_BOOL_FIELDS:
        if type(raw[name]) is not bool:
            raise TypeError(f"replay {name} exact boolean type drift")
    for name in _REPLAY_TOP_TUPLE_FIELDS:
        if type(raw[name]) is not tuple:
            raise TypeError(f"replay {name} exact tuple type drift")
    for name in _REPLAY_TOP_NONE_FIELDS:
        if raw[name] is not None:
            raise TypeError(f"replay {name} exact None drift")
    if raw["version"] != ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION:
        raise _CrossVersionInput("replay-input contract version drift")
    if (
        raw["disposition"] is not ActualUnsealed960ReplayInputDispositionV2.ACTUAL_REPLAY_CONTRACT_COMPLETE_NOT_EXECUTED
        or raw["reason"] is not ActualUnsealed960ReplayInputReasonV2.SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED
        or raw["schema_id"] != ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID
        or raw["policy_id"] != ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID
        or raw["claim_level"] != ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
        or raw["gate_input_manifest_schema_id"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID
        or raw["gate_input_manifest_policy_id"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID
        or raw["input_archive_version"] != _FROZEN_INPUT_ARCHIVE_VERSION
        or raw["input_archive_policy_id"] != _FROZEN_INPUT_ARCHIVE_POLICY_ID
        or raw["batch_policy_id"] != _FROZEN_BATCH_POLICY_ID
        or raw["exact_freeze_id"] != _FROZEN_EXACT_FREEZE_ID
        or raw["protocol_id"] != _FROZEN_PROTOCOL_ID
        or raw["formal_scoring_contract_id"] != _FROZEN_FORMAL_SCORING_CONTRACT_ID
    ):
        raise ValueError("replay-input identity drift")
    for name in _REPLAY_TRUE_CLAIMS:
        if raw[name] is not True:
            raise ValueError(f"replay-input true claim drift: {name}")
    for name in _REPLAY_FALSE_CLAIMS:
        if raw[name] is not False:
            raise ValueError(f"replay-input false claim drift: {name}")
    for name, expected in (
        ("main_row_count", 720), ("semantic_conflict_expected_row_count", 240),
        ("total_expected_prediction_count", 960),
        ("unique_latent_base_case_id_count", 720), ("family_scale_cell_count", 12),
    ):
        if _integer(raw[name], f"replay {name}", maximum=960) != expected:
            raise ValueError(f"replay count drift: {name}")
    for name in ("metric_results", "scored_rows", "gate_results"):
        if type(raw[name]) is not tuple or raw[name]:
            raise ValueError(f"replay {name} must remain empty")
    if raw["scale_regret_result"] is not None or raw["bootstrap_result"] is not None:
        raise ValueError("replay auxiliary result unexpectedly present")
    inventory = raw["required_evidence_inventory"]
    available = raw["available_overall_gate_input_definitions"]
    unavailable = raw["unavailable_overall_gate_input_definitions"]
    slices = raw["slice_gate_input_definitions"]
    if type(inventory) is not tuple or len(inventory) != 18 or type(available) is not tuple or len(available) != 10 or type(unavailable) is not tuple or len(unavailable) != 2 or type(slices) is not tuple or len(slices) != 6:
        raise ValueError("replay nested catalog count drift")
    raw["required_evidence_inventory"] = tuple(_evidence_snapshot(item) for item in inventory)
    raw["available_overall_gate_input_definitions"] = tuple(_definition_snapshot(item) for item in available)
    raw["unavailable_overall_gate_input_definitions"] = tuple(_definition_snapshot(item) for item in unavailable)
    raw["slice_gate_input_definitions"] = tuple(_definition_snapshot(item) for item in slices)
    for item in fields(ActualUnsealed960ReplayInputContractV2):
        if item.name in {"required_evidence_inventory", "available_overall_gate_input_definitions", "unavailable_overall_gate_input_definitions", "slice_gate_input_definitions", "metric_results", "scored_rows", "gate_results", "scale_regret_result", "bootstrap_result"}:
            continue
        candidate = raw[item.name]
        if type(candidate) is str:
            if item.name.endswith("sha256"):
                _sha(candidate, f"replay {item.name}")
            elif item.name.endswith("_id") or item.name.endswith("_root"):
                prefix = _REPLAY_ADDRESS_PREFIXES.get(item.name)
                if prefix is None:
                    raise ValueError(f"unregistered replay content address: {item.name}")
                _id(candidate, prefix, f"replay {item.name}")
            else:
                _text(candidate, f"replay {item.name}")
        elif type(candidate) is int:
            _integer(candidate, f"replay {item.name}", maximum=10_000)
        elif type(candidate) is bool or type(candidate) in (
            ActualUnsealed960ReplayInputDispositionV2,
            ActualUnsealed960ReplayInputReasonV2,
        ):
            pass
        else:
            raise TypeError(f"replay scalar type drift: {item.name}")
    return raw


def _snapshot_manifest(value: object) -> dict[str, object]:
    raw = _snapshot_top(value, FormalUnsealedGateInputManifestV2)
    for name in _MANIFEST_TOP_STR_FIELDS:
        if type(raw[name]) is not str:
            raise TypeError(f"gate manifest {name} exact string type drift")
    for name in _MANIFEST_TOP_TUPLE_FIELDS:
        if type(raw[name]) is not tuple:
            raise TypeError(f"gate manifest {name} exact tuple type drift")
    if raw["schema_version"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION:
        raise _CrossVersionInput("gate manifest version drift")
    if (
        raw["schema_id"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID
        or raw["policy_id"] != FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID
        or raw["claim_level"] != ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
        or raw["input_archive_version"] != _FROZEN_INPUT_ARCHIVE_VERSION
        or raw["input_archive_policy_id"] != _FROZEN_INPUT_ARCHIVE_POLICY_ID
        or raw["batch_policy_id"] != _FROZEN_BATCH_POLICY_ID
        or raw["exact_freeze_id"] != _FROZEN_EXACT_FREEZE_ID
        or raw["phase2b_protocol_id"] != _FROZEN_PROTOCOL_ID
        or raw["formal_scoring_contract_id"] != _FROZEN_FORMAL_SCORING_CONTRACT_ID
    ):
        raise ValueError("gate manifest identity drift")
    rows = raw["main_gate_input_rows"]
    inventory = raw["required_evidence_inventory"]
    if type(rows) is not tuple or len(rows) != _MAIN_COUNT or type(inventory) is not tuple or len(inventory) != 18:
        raise ValueError("gate manifest nested count drift")
    raw["main_gate_input_rows"] = tuple(_gate_row_snapshot(item) for item in rows)
    raw["required_evidence_inventory"] = tuple(_evidence_snapshot(item) for item in inventory)
    for item in fields(FormalUnsealedGateInputManifestV2):
        if item.name in {"main_gate_input_rows", "required_evidence_inventory"}:
            continue
        if item.name.endswith("sha256"):
            _sha(raw[item.name], f"gate manifest {item.name}")
        elif item.name.endswith("_id") or item.name.endswith("_root"):
            prefix = _MANIFEST_ADDRESS_PREFIXES.get(item.name)
            if prefix is None:
                raise ValueError(f"unregistered manifest content address: {item.name}")
            _id(raw[item.name], prefix, f"gate manifest {item.name}")
        else:
            _text(raw[item.name], f"gate manifest {item.name}")
    return raw


def _expected_outcome(
    metric_name: str,
    *,
    case_type: Phase2BCaseType,
    decision: PredictionDecisionV2,
    expected_decision: PredictionDecisionV2,
    family_exact: bool,
    binding_exact: bool,
    scale_exact: bool,
) -> tuple[bool, bool | None]:
    answerable = case_type in _ANSWERABLE_CASES
    if metric_name == "answerable_count":
        return answerable, None
    if metric_name == "family_exact_accuracy":
        return answerable, family_exact if answerable else None
    if metric_name == "binding_exact_accuracy":
        return answerable, binding_exact if answerable else None
    if metric_name == "scale_set_accuracy":
        return answerable, scale_exact if answerable else None
    if metric_name == "unique_scale_accuracy":
        eligible = case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
        return eligible, (decision is PredictionDecisionV2.ANSWER and scale_exact) if eligible else None
    if metric_name == "joint_exact_accuracy":
        return answerable, (decision is expected_decision and family_exact and binding_exact and scale_exact) if answerable else None
    if metric_name == "abstention_specificity":
        eligible = case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
        return eligible, (decision is not PredictionDecisionV2.ABSTAIN) if eligible else None
    if metric_name == "nonidentifiability_abstention_accuracy":
        eligible = case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
        return eligible, (decision is PredictionDecisionV2.ABSTAIN) if eligible else None
    if metric_name == "set_valued_answer_accuracy":
        eligible = case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE
        return eligible, (decision is PredictionDecisionV2.ANSWER_SET and family_exact and binding_exact and scale_exact) if eligible else None
    raise ValueError("unknown metric outcome")


def _validate_semantics(
    scoring: dict[str, object],
    replay: dict[str, object],
    manifest: dict[str, object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], dict[str, str]]:
    available = replay["available_overall_gate_input_definitions"]
    unavailable = replay["unavailable_overall_gate_input_definitions"]
    slice_defs = replay["slice_gate_input_definitions"]
    definitions = (*available, *unavailable, *slice_defs)
    observed_specs = tuple(
        (
            item["gate_name"], item["scope"], item["expected_denominator"],
            item["success_rule"], item["input_available"], item["missing_input_reason"],
        )
        for item in definitions
    )
    if observed_specs != _GATE_DEFINITION_SPECS:
        raise ValueError("frozen gate definition catalog drift")
    evidence = replay["required_evidence_inventory"]
    if tuple((item["evidence_name"], item["purpose"]) for item in evidence) != _EVIDENCE_SPECS:
        raise ValueError("frozen evidence inventory drift")
    if any(item["supplied_by_this_contract"] or item["verifier_implemented"] for item in evidence):
        raise ValueError("unsupplied evidence status drift")
    if evidence != manifest["required_evidence_inventory"]:
        raise ValueError("manifest/replay evidence inventory mismatch")

    rows = scoring["main_row_results"]
    gates = manifest["main_gate_input_rows"]
    row_ids = tuple(item["input_row_id"] for item in rows)
    gate_ids = tuple(item["input_row_id"] for item in gates)
    answer_ids = tuple(item["answer_row_id"] for item in rows)
    gate_answer_ids = tuple(item["answer_row_id"] for item in gates)
    prediction_record_ids = tuple(item["prediction_record_id"] for item in rows)
    if len(set(row_ids)) != _MAIN_COUNT or len(set(gate_ids)) != _MAIN_COUNT:
        raise _RowJoinOrQuota("row ID uniqueness drift")
    if (
        len(set(answer_ids)) != _MAIN_COUNT
        or len(set(gate_answer_ids)) != _MAIN_COUNT
        or len(set(prediction_record_ids)) != _MAIN_COUNT
        or answer_ids != gate_answer_ids
    ):
        raise _RowJoinOrQuota("answer/prediction record ID uniqueness or parity drift")
    if gate_ids != tuple(sorted(gate_ids)) or row_ids != gate_ids:
        raise _RowJoinOrQuota("gate/scoring row coverage drift")
    by_row = {item["input_row_id"]: item for item in rows}
    joined = tuple((by_row[item["input_row_id"]], item) for item in gates)
    if any(
        row["answer_row_id"] != gate["answer_row_id"]
        or row["case_type"] is not gate["case_type"]
        or (
            row["case_type"] in _ANSWERABLE_CASES
            and row["expected_canonical_family_id"] is not gate["canonical_family_id"]
        )
        or (
            row["case_type"] not in _ANSWERABLE_CASES
            and row["expected_canonical_family_id"] is not None
        )
        for row, gate in joined
    ):
        raise _RowJoinOrQuota("720 row answer/family parity drift")

    expected_case_counts = {
        Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE: 228,
        Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE: 12,
        Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE: 96,
        Phase2BCaseType.BINDING_COUNTERFACTUAL: 96,
        Phase2BCaseType.SCALE_COUNTERFACTUAL: 96,
        Phase2BCaseType.SIGN_OR_INVARIANT_BREAK: 96,
        Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE: 96,
    }
    if {case: sum(row["case_type"] is case for row in rows) for case in Phase2BCaseType} != expected_case_counts:
        raise _RowJoinOrQuota("main row case quota drift")
    latent_ids = tuple(item["latent_base_case_id"] for item in gates)
    if len(set(latent_ids)) != _MAIN_COUNT:
        raise _RowJoinOrQuota("latent ID uniqueness drift")
    for family in CanonicalFamilyId:
        for scale in FormalUnsealedScaleSliceIdV2:
            members = tuple(item for item in gates if item["canonical_family_id"] is family and item["scale_slice_id"] is scale)
            if len(members) != 60:
                raise _RowJoinOrQuota("family/scale cell quota drift")
            for case, count in (
                (Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE, 19),
                (Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE, 1),
                (Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE, 8),
                (Phase2BCaseType.BINDING_COUNTERFACTUAL, 8),
                (Phase2BCaseType.SCALE_COUNTERFACTUAL, 8),
                (Phase2BCaseType.SIGN_OR_INVARIANT_BREAK, 8),
                (Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE, 8),
            ):
                if sum(item["case_type"] is case for item in members) != count:
                    raise _RowJoinOrQuota("cell case quota drift")
            for margin, count in (
                (MarginStratum.CLEAR_INTERIOR, 21),
                (MarginStratum.MODERATE, 18),
                (MarginStratum.NEAR_BOUNDARY_IDENTIFIABLE, 12),
                (MarginStratum.NONUNIQUE_OR_INSUFFICIENT, 9),
            ):
                if sum(item["margin_stratum"] is margin for item in members) != count:
                    raise _RowJoinOrQuota("cell margin quota drift")
            if any(
                (item["margin_stratum"] is MarginStratum.NONUNIQUE_OR_INSUFFICIENT)
                is not (
                    item["case_type"] in (
                        Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
                        Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,
                    )
                )
                for item in members
            ):
                raise _RowJoinOrQuota("cell case/margin composition drift")

    metric_by_name: dict[str, dict[str, object]] = {}
    for row in rows:
        expected_decision_by_case = {
            Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE: PredictionDecisionV2.ANSWER,
            Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE: PredictionDecisionV2.ANSWER_SET,
            Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE: PredictionDecisionV2.ABSTAIN,
            Phase2BCaseType.BINDING_COUNTERFACTUAL: PredictionDecisionV2.ABSTAIN,
            Phase2BCaseType.SCALE_COUNTERFACTUAL: PredictionDecisionV2.ABSTAIN,
            Phase2BCaseType.SIGN_OR_INVARIANT_BREAK: PredictionDecisionV2.ABSTAIN,
            Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE: PredictionDecisionV2.ABSTAIN,
        }
        if row["expected_decision"] is not expected_decision_by_case[row["case_type"]]:
            raise _RowJoinOrQuota("answer expected-decision/case coherence drift")
        if row["case_type"] is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
            answer_coherent = (
                row["expected_canonical_family_id"] is not None
                and bool(row["expected_binding"])
                and len(row["expected_admissible_scale_ids"]) == 1
            )
        elif row["case_type"] is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
            answer_coherent = (
                row["expected_canonical_family_id"] is not None
                and bool(row["expected_binding"])
                and len(row["expected_admissible_scale_ids"]) >= 2
            )
        else:
            answer_coherent = (
                row["expected_canonical_family_id"] is None
                and row["expected_binding"] == ()
                and row["expected_admissible_scale_ids"] == ()
            )
        if not answer_coherent:
            raise _RowJoinOrQuota("answer family/binding/scale coherence drift")
        if row["predicted_decision"] is PredictionDecisionV2.ABSTAIN:
            prediction_coherent = (
                row["predicted_canonical_family_id"] is None
                and row["predicted_binding"] == ()
                and row["predicted_admissible_scale_ids"] == ()
            )
        elif row["predicted_decision"] is PredictionDecisionV2.ANSWER:
            prediction_coherent = (
                row["predicted_canonical_family_id"] is not None
                and bool(row["predicted_binding"])
                and len(row["predicted_admissible_scale_ids"]) == 1
            )
        else:
            prediction_coherent = (
                row["predicted_canonical_family_id"] is not None
                and bool(row["predicted_binding"])
                and len(row["predicted_admissible_scale_ids"]) >= 2
            )
        if not prediction_coherent:
            raise _RowJoinOrQuota("prediction decision/payload coherence drift")
        positive = row["predicted_decision"] in (PredictionDecisionV2.ANSWER, PredictionDecisionV2.ANSWER_SET)
        answerable = row["case_type"] in _ANSWERABLE_CASES
        family_exact = bool(positive and row["predicted_canonical_family_id"] is row["expected_canonical_family_id"])
        binding_exact = bool(positive and row["predicted_binding"] == row["expected_binding"])
        scale_exact = bool(positive and row["predicted_admissible_scale_ids"] == row["expected_admissible_scale_ids"])
        expected_flags = {
            "decision_exact": row["predicted_decision"] is row["expected_decision"],
            "family_exact": family_exact if answerable else None,
            "binding_exact": binding_exact if answerable else None,
            "scale_set_exact": scale_exact if answerable else None,
            "joint_exact": (row["predicted_decision"] is row["expected_decision"] and family_exact and binding_exact and scale_exact) if answerable else None,
            "metric_eligible": answerable or row["case_type"] is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,
        }
        if any(row[name] != value for name, value in expected_flags.items()):
            raise _RowJoinOrQuota("main row derived flag drift")
        for outcome, spec in zip(row["metric_outcomes"], _METRIC_SPECS, strict=True):
            if outcome["metric_name"] != spec[0] or outcome["metric_definition_id"] != spec[1]:
                raise _RowJoinOrQuota("metric row outcome catalog drift")
            eligible, success = _expected_outcome(
                spec[0], case_type=row["case_type"], decision=row["predicted_decision"],
                expected_decision=row["expected_decision"], family_exact=family_exact,
                binding_exact=binding_exact, scale_exact=scale_exact,
            )
            if outcome["eligible"] is not eligible or outcome["success"] is not success:
                raise _RowJoinOrQuota("metric row outcome semantics drift")

    metrics = scoring["metric_results"]
    for metric, spec in zip(metrics, _METRIC_SPECS, strict=True):
        expected_projection = (
            spec[1], spec[0], spec[2], spec[3], spec[4], spec[5], spec[6]
        )
        observed_projection = (
            metric["metric_definition_id"], metric["metric_name"],
            metric["metric_kind"].value,
            tuple(item.value for item in metric["denominator_case_types"]),
            metric["expected_denominator"], metric["success_rule"],
            metric["separately_reported"],
        )
        if observed_projection != expected_projection:
            raise _RowJoinOrQuota("metric result catalog drift")
        outcomes = tuple(row["metric_outcomes"][_METRIC_SPECS.index(spec)] for row in rows)
        eligible = tuple(item for item in outcomes if item["eligible"])
        if metric["observed_denominator"] != len(eligible) or len(eligible) != spec[4]:
            raise _RowJoinOrQuota("metric denominator drift")
        if spec[2] == "COUNT":
            if metric["success_count"] is not None or metric["count_value"] != len(eligible):
                raise _RowJoinOrQuota("count metric value drift")
        else:
            successes = sum(item["success"] is True for item in eligible)
            if metric["success_count"] != successes or metric["count_value"] is not None:
                raise _RowJoinOrQuota("binary metric value drift")
        metric_by_name[spec[0]] = metric

    metric_parity = {
        "family_exact": metric_by_name["family_exact_accuracy"]["success_count"],
        "binding_exact": metric_by_name["binding_exact_accuracy"]["success_count"],
        "scale_set_accuracy": metric_by_name["scale_set_accuracy"]["success_count"],
        "joint_exact": metric_by_name["joint_exact_accuracy"]["success_count"],
        "abstention_specificity": metric_by_name["abstention_specificity"]["success_count"],
        "nonidentifiable_scale_abstention": metric_by_name["nonidentifiability_abstention_accuracy"]["success_count"],
    }
    if metric_by_name["answerable_count"]["count_value"] != 240:
        raise _RowJoinOrQuota("answerable count metric parity drift")
    for name, expected_successes in metric_parity.items():
        eligible = tuple(row for row in rows if _eligible_for_metric(name, row))
        if expected_successes != sum(_success_for_metric(name, row) for row in eligible):
            raise _RowJoinOrQuota(f"available/scoring metric parity drift: {name}")

    identity_pairs = (
        (scoring["input_archive_id"], replay["input_archive_id"], manifest["input_archive_id"]),
        (scoring["input_archive_sha256"], replay["input_archive_sha256"], manifest["input_archive_sha256"]),
        (scoring["batch_id"], replay["batch_id"], manifest["batch_id"]),
        (scoring["execution_freeze_manifest_id"], replay["execution_freeze_manifest_id"], manifest["execution_freeze_manifest_id"]),
        (scoring["ordered_archive_input_row_ids_root"], replay["ordered_archive_input_row_ids_root"], manifest["ordered_archive_input_row_ids_root"]),
        (scoring["partition_union_row_ids_root"], replay["partition_union_row_ids_root"], manifest["partition_union_row_ids_root"]),
        (scoring["answer_manifest_id"], replay["answer_manifest_id"], manifest["answer_manifest_id"]),
        (scoring["answer_manifest_sha256"], replay["answer_manifest_sha256"], manifest["answer_manifest_sha256"]),
        (scoring["main_row_ids_root"], replay["main_row_ids_root"], manifest["main_row_ids_root"]),
        (scoring["semantic_conflict_row_ids_root"], replay["semantic_conflict_row_ids_root"], manifest["semantic_conflict_row_ids_root"]),
        (scoring["main_answer_row_ids_root"], replay["main_answer_row_ids_root"], manifest["main_answer_row_ids_root"]),
        (scoring["protocol_id"], replay["protocol_id"], manifest["phase2b_protocol_id"]),
        (scoring["formal_scoring_contract_id"], replay["formal_scoring_contract_id"], manifest["formal_scoring_contract_id"]),
        (replay["gate_input_manifest_id"], manifest["gate_input_manifest_id"]),
        (replay["gate_input_manifest_sha256"], manifest["gate_input_manifest_sha256"]),
        (replay["main_gate_input_row_ids_root"], manifest["main_gate_input_row_ids_root"]),
        (replay["input_archive_version"], manifest["input_archive_version"]),
        (replay["input_archive_policy_id"], manifest["input_archive_policy_id"]),
        (replay["batch_policy_id"], manifest["batch_policy_id"]),
        (replay["exact_freeze_id"], manifest["exact_freeze_id"]),
        (replay["gate_input_manifest_schema_id"], manifest["schema_id"]),
        (replay["gate_input_manifest_policy_id"], manifest["policy_id"]),
    )
    if any(any(item != group[0] for item in group[1:]) for group in identity_pairs):
        raise ValueError("three-graph identity cross-binding drift")
    if (
        scoring["protocol_id"] != _FROZEN_PROTOCOL_ID
        or scoring["formal_scoring_contract_id"]
        != _FROZEN_FORMAL_SCORING_CONTRACT_ID
    ):
        raise ValueError("frozen protocol or formal scoring contract identity drift")
    definition_ids = {
        f"{item['scope']}:{item['gate_name']}": item["definition_id"]
        for item in definitions
    }
    return rows, gates, definition_ids


def _outcome_primitive(item: dict[str, object]) -> dict[str, object]:
    return {name: item[name] for name in (
        "metric_definition_id", "metric_name", "eligible", "success",
        "metric_row_outcome_id",
    )}


def _row_primitive(item: dict[str, object]) -> dict[str, object]:
    return {
        "input_row_id": item["input_row_id"],
        "prediction_record_id": item["prediction_record_id"],
        "prediction_content_id": item["prediction_content_id"],
        "answer_row_id": item["answer_row_id"],
        "case_type": item["case_type"].value,
        "predicted_decision": item["predicted_decision"].value,
        "expected_decision": item["expected_decision"].value,
        "predicted_canonical_family_id": None if item["predicted_canonical_family_id"] is None else item["predicted_canonical_family_id"].value,
        "expected_canonical_family_id": None if item["expected_canonical_family_id"] is None else item["expected_canonical_family_id"].value,
        "predicted_binding": _binding_primitive(item["predicted_binding"]),
        "expected_binding": _binding_primitive(item["expected_binding"]),
        "predicted_admissible_scale_ids": list(item["predicted_admissible_scale_ids"]),
        "expected_admissible_scale_ids": list(item["expected_admissible_scale_ids"]),
        "decision_exact": item["decision_exact"],
        "family_exact": item["family_exact"],
        "binding_exact": item["binding_exact"],
        "scale_set_exact": item["scale_set_exact"],
        "joint_exact": item["joint_exact"],
        "metric_eligible": item["metric_eligible"],
        "metric_outcomes": [_outcome_primitive(value) for value in item["metric_outcomes"]],
        "row_result_id": item["row_result_id"],
    }


def _metric_primitive(item: dict[str, object]) -> dict[str, object]:
    return {
        "metric_definition_id": item["metric_definition_id"],
        "metric_name": item["metric_name"],
        "metric_kind": item["metric_kind"].value,
        "denominator_case_types": [value.value for value in item["denominator_case_types"]],
        "expected_denominator": item["expected_denominator"],
        "observed_denominator": item["observed_denominator"],
        "success_count": item["success_count"],
        "count_value": item["count_value"],
        "success_rule": item["success_rule"],
        "separately_reported": item["separately_reported"],
        "metric_result_id": item["metric_result_id"],
    }


def _generic_primitive(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) in (
        Phase2BCaseType, PredictionDecisionV2, CanonicalFamilyId,
        FormalUnsealedMetricKindV2, MarginStratum, FormalUnsealedScaleSliceIdV2,
        Unsealed960PredictionScoringDispositionV2,
        Unsealed960PredictionScoringReasonV2,
        ActualUnsealed960ReplayInputDispositionV2,
        ActualUnsealed960ReplayInputReasonV2,
        Unsealed960AvailableGateMechanicsDispositionV2,
        Unsealed960AvailableGateMechanicsReasonV2,
    ):
        return value.value
    if type(value) is tuple:
        return [_generic_primitive(item) for item in value]
    if type(value) is dict:
        return {str(name): _generic_primitive(item) for name, item in value.items()}
    raise TypeError("unsupported closed primitive")


def _verify_content_addresses(
    scoring: dict[str, object],
    replay: dict[str, object],
    manifest: dict[str, object],
) -> None:
    answer_row_mappings: list[dict[str, object]] = []
    for row in scoring["main_row_results"]:
        answer_preimage = {
            "input_row_id": row["input_row_id"],
            "case_type": row["case_type"].value,
            "expected_decision": row["expected_decision"].value,
            "canonical_family_id": (
                None
                if row["expected_canonical_family_id"] is None
                else row["expected_canonical_family_id"].value
            ),
            "binding": _binding_primitive(row["expected_binding"]),
            "admissible_scale_ids": list(row["expected_admissible_scale_ids"]),
        }
        expected_answer_id = _stable_id(
            answer_preimage,
            domain=_ANSWER_ROW_DOMAIN,
            prefix=_ANSWER_ROW_PREFIX,
        )
        if row["answer_row_id"] != expected_answer_id:
            raise ValueError("formal answer row content address drift")
        answer_row_mappings.append(
            {**answer_preimage, "answer_row_id": expected_answer_id}
        )
        for outcome in row["metric_outcomes"]:
            preimage = _outcome_primitive(outcome)
            stored = preimage.pop("metric_row_outcome_id")
            expected = stable_hash(preimage, prefix="phase2b_unsealed_960_metric_row_outcome_v2_")
            if stored != expected:
                raise ValueError("metric outcome content address drift")
        preimage = _row_primitive(row)
        stored = preimage.pop("row_result_id")
        expected = stable_hash(preimage, prefix="phase2b_unsealed_960_main_row_result_v2_")
        if stored != expected:
            raise ValueError("main row result content address drift")

    expected_answer_root = _sequence_root(
        tuple(item["answer_row_id"] for item in answer_row_mappings),
        domain=b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00",
        prefix="phase2b_formal_unsealed_answer_rows_v2_",
    )
    if scoring["main_answer_row_ids_root"] != expected_answer_root:
        raise ValueError("formal answer row root drift")
    answer_manifest_preimage = {
        "schema_version": _ANSWER_MANIFEST_SCHEMA_VERSION,
        "schema_id": _ANSWER_MANIFEST_SCHEMA_ID,
        "policy_id": _ANSWER_MANIFEST_POLICY_ID,
        "claim_level": FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        "exact_freeze_id": replay["exact_freeze_id"],
        "phase2b_protocol_id": replay["protocol_id"],
        "execution_freeze_manifest_id": replay["execution_freeze_manifest_id"],
        "input_archive_id": replay["input_archive_id"],
        "input_archive_sha256": replay["input_archive_sha256"],
        "input_archive_version": replay["input_archive_version"],
        "input_archive_policy_id": replay["input_archive_policy_id"],
        "batch_id": replay["batch_id"],
        "batch_policy_id": replay["batch_policy_id"],
        "ordered_archive_input_row_ids_root": replay["ordered_archive_input_row_ids_root"],
        "main_row_ids_root": replay["main_row_ids_root"],
        "semantic_conflict_row_ids_root": replay["semantic_conflict_row_ids_root"],
        "partition_union_row_ids_root": replay["partition_union_row_ids_root"],
        "main_answer_rows": answer_row_mappings,
        "main_answer_row_ids_root": expected_answer_root,
    }
    expected_answer_sha = hashlib.sha256(
        _ANSWER_MANIFEST_DOMAIN
        + canonical_json(answer_manifest_preimage).encode("utf-8")
    ).hexdigest()
    if (
        scoring["answer_manifest_sha256"] != expected_answer_sha
        or scoring["answer_manifest_id"]
        != _ANSWER_MANIFEST_PREFIX + expected_answer_sha
    ):
        raise ValueError("formal answer manifest content address drift")

    for metric in scoring["metric_results"]:
        preimage = _metric_primitive(metric)
        stored = preimage.pop("metric_result_id")
        expected = stable_hash(preimage, prefix="phase2b_unsealed_960_metric_result_v2_")
        if stored != expected:
            raise ValueError("metric result content address drift")
    scoring_preimage: dict[str, object] = {}
    for name, value in scoring.items():
        if name == "result_id":
            continue
        if name == "main_row_results":
            scoring_preimage[name] = [_row_primitive(item) for item in value]
        elif name == "metric_results":
            scoring_preimage[name] = [_metric_primitive(item) for item in value]
        else:
            scoring_preimage[name] = _generic_primitive(value)
    if scoring["result_id"] != stable_hash(
        scoring_preimage,
        prefix="phase2b_unsealed_960_prediction_scoring_mechanics_v2_",
    ):
        raise ValueError("scoring result content address drift")

    definition_groups = (
        replay["available_overall_gate_input_definitions"],
        replay["unavailable_overall_gate_input_definitions"],
        replay["slice_gate_input_definitions"],
    )
    for group in definition_groups:
        for item in group:
            preimage = {name: item[name] for name in (
                "gate_name", "scope", "expected_denominator", "success_rule",
                "input_available", "missing_input_reason",
            )}
            expected = _stable_id(
                preimage,
                domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_DEFINITION/V2\x00",
                prefix="phase2b_actual_replay_gate_input_definition_v2_",
            )
            if item["definition_id"] != expected:
                raise ValueError("gate definition content address drift")
    for item in replay["required_evidence_inventory"]:
        preimage = {name: item[name] for name in (
            "evidence_name", "purpose", "supplied_by_this_contract",
            "verifier_implemented",
        )}
        expected = _stable_id(
            preimage,
            domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/REQUIRED_EVIDENCE/V2\x00",
            prefix="phase2b_actual_replay_required_evidence_v2_",
        )
        if item["requirement_id"] != expected:
            raise ValueError("required evidence content address drift")
    replay_preimage: dict[str, object] = {}
    for name, value in replay.items():
        if name == "result_id":
            continue
        replay_preimage[name] = _generic_primitive(value)
    expected_replay = _stable_id(
        replay_preimage,
        domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/RESULT/V2\x00",
        prefix="phase2b_actual_unsealed_960_replay_input_contract_v2_",
    )
    if replay["result_id"] != expected_replay:
        raise ValueError("replay-input result content address drift")

    gate_row_mappings: list[dict[str, object]] = []
    for item in manifest["main_gate_input_rows"]:
        preimage = {
            "input_row_id": item["input_row_id"],
            "answer_row_id": item["answer_row_id"],
            "case_type": item["case_type"].value,
            "margin_stratum": item["margin_stratum"].value,
            "canonical_family_id": item["canonical_family_id"].value,
            "scale_slice_id": item["scale_slice_id"].value,
            "latent_base_case_id": item["latent_base_case_id"],
        }
        expected = _stable_id(
            preimage,
            domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW/V2\x00",
            prefix="phase2b_actual_replay_gate_input_row_v2_",
        )
        if item["gate_input_row_id"] != expected:
            raise ValueError("gate row content address drift")
        gate_row_mappings.append({**preimage, "gate_input_row_id": expected})
    expected_root = _sequence_root(
        tuple(item["gate_input_row_id"] for item in manifest["main_gate_input_rows"]),
        domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW_IDS/V2\x00",
        prefix="phase2b_actual_replay_gate_input_rows_v2_",
    )
    if manifest["main_gate_input_row_ids_root"] != expected_root:
        raise ValueError("gate row root drift")
    expected_main_root = _sequence_root(
        tuple(item["input_row_id"] for item in manifest["main_gate_input_rows"]),
        domain=b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00",
        prefix="phase2b_unsealed_main_rows_v2_",
    )
    expected_answer_root = _sequence_root(
        tuple(item["answer_row_id"] for item in manifest["main_gate_input_rows"]),
        domain=b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00",
        prefix="phase2b_formal_unsealed_answer_rows_v2_",
    )
    if manifest["main_row_ids_root"] != expected_main_root or manifest["main_answer_row_ids_root"] != expected_answer_root:
        raise ValueError("gate manifest main/answer row root drift")
    manifest_preimage = {
        "schema_version": manifest["schema_version"],
        "schema_id": manifest["schema_id"],
        "policy_id": manifest["policy_id"],
        "claim_level": manifest["claim_level"],
        "exact_freeze_id": manifest["exact_freeze_id"],
        "phase2b_protocol_id": manifest["phase2b_protocol_id"],
        "formal_scoring_contract_id": manifest["formal_scoring_contract_id"],
        "execution_freeze_manifest_id": manifest["execution_freeze_manifest_id"],
        "input_archive_id": manifest["input_archive_id"],
        "input_archive_sha256": manifest["input_archive_sha256"],
        "input_archive_version": manifest["input_archive_version"],
        "input_archive_policy_id": manifest["input_archive_policy_id"],
        "batch_id": manifest["batch_id"],
        "batch_policy_id": manifest["batch_policy_id"],
        "ordered_archive_input_row_ids_root": manifest["ordered_archive_input_row_ids_root"],
        "main_row_ids_root": manifest["main_row_ids_root"],
        "semantic_conflict_row_ids_root": manifest["semantic_conflict_row_ids_root"],
        "partition_union_row_ids_root": manifest["partition_union_row_ids_root"],
        "answer_manifest_id": manifest["answer_manifest_id"],
        "answer_manifest_sha256": manifest["answer_manifest_sha256"],
        "main_answer_row_ids_root": manifest["main_answer_row_ids_root"],
        "main_gate_input_rows": gate_row_mappings,
        "main_gate_input_row_ids_root": manifest["main_gate_input_row_ids_root"],
        "required_evidence_inventory": list(manifest["required_evidence_inventory"]),
    }
    expected_sha = hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_MANIFEST/V2\x00"
        + canonical_json(manifest_preimage).encode("utf-8")
    ).hexdigest()
    if manifest["gate_input_manifest_sha256"] != expected_sha or manifest["gate_input_manifest_id"] != "phase2b_formal_unsealed_gate_input_manifest_v2_" + expected_sha:
        raise ValueError("gate manifest content address drift")


def _wilson_lower_bound(successes: int, total: int) -> float:
    proportion = successes / total
    z = float.fromhex(_ONE_SIDED_Z_HEX)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_squared / (4.0 * total * total)
    )
    return max(0.0, (center - radius) / denominator)


def _available_result_mapping(value: Unsealed960AvailableGateMechanicsResultV2) -> dict[str, object]:
    return {name: object.__getattribute__(value, name) for name in _AVAILABLE_RESULT_FIELDS}


def _unavailable_result_mapping(value: Unsealed960UnavailableGateMechanicsV2) -> dict[str, object]:
    return {name: object.__getattribute__(value, name) for name in _UNAVAILABLE_RESULT_FIELDS}


def _issue_available_result(
    *,
    metric_name: str,
    scope: str,
    slice_id: str | None,
    definition_id: str,
    successes: int,
    total: int,
    point_numerator: int,
    point_denominator: int,
    wilson_numerator: int,
    wilson_denominator: int,
) -> Unsealed960AvailableGateMechanicsResultV2:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("available gate counts drift")
    point = successes / total
    lower = _wilson_lower_bound(successes, total)
    point_pass = successes * point_denominator >= total * point_numerator
    lower_numerator, lower_denominator = lower.as_integer_ratio()
    wilson_pass = (
        lower_numerator * wilson_denominator
        >= wilson_numerator * lower_denominator
    )
    value = object.__new__(Unsealed960AvailableGateMechanicsResultV2)
    frozen = (
        ("metric_name", metric_name), ("scope", scope), ("slice_id", slice_id),
        ("gate_input_definition_id", definition_id), ("successes", successes),
        ("total", total), ("expected_denominator", total),
        ("minimum_point_estimate_ratio", _ratio(point_numerator, point_denominator)),
        ("minimum_wilson_lcb_ratio", _ratio(wilson_numerator, wilson_denominator)),
        ("point_estimate_ratio", _ratio(successes, total)),
        ("point_estimate_hex", point.hex()),
        ("one_sided_wilson_lcb_hex", lower.hex()),
        ("point_threshold_passed", point_pass),
        ("wilson_threshold_passed", wilson_pass),
        ("available_gate_passed", point_pass and wilson_pass),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    preimage = tuple((name, object.__getattribute__(value, name)) for name in _AVAILABLE_RESULT_FIELDS[:-1])
    object.__setattr__(value, "result_id", _stable_id(preimage, domain=_RESULT_DOMAIN, prefix=_RESULT_PREFIX))
    return value


def _issue_unavailable_result(
    *,
    metric_name: str,
    definition_id: str,
    point_numerator: int,
    point_denominator: int,
    wilson_numerator: int | None,
    wilson_denominator: int | None,
    missing_reason: str,
) -> Unsealed960UnavailableGateMechanicsV2:
    value = object.__new__(Unsealed960UnavailableGateMechanicsV2)
    frozen = (
        ("metric_name", metric_name), ("scope", "overall"),
        ("gate_input_definition_id", definition_id),
        ("minimum_point_estimate_ratio", _ratio(point_numerator, point_denominator)),
        ("minimum_wilson_lcb_ratio", None if wilson_numerator is None else _ratio(wilson_numerator, wilson_denominator)),
        ("expected_denominator", None), ("successes", None), ("total", None),
        ("point_estimate_ratio", None), ("point_estimate_hex", None),
        ("one_sided_wilson_lcb_hex", None), ("point_threshold_passed", None),
        ("wilson_threshold_passed", None), ("available_gate_passed", None),
        ("missing_input_reason", missing_reason),
    )
    for name, item in frozen:
        object.__setattr__(value, name, item)
    preimage = tuple((name, object.__getattribute__(value, name)) for name in _UNAVAILABLE_RESULT_FIELDS[:-1])
    object.__setattr__(value, "unavailable_id", _stable_id(preimage, domain=_UNAVAILABLE_DOMAIN, prefix=_UNAVAILABLE_PREFIX))
    return value


def _success_for_metric(metric_name: str, row: dict[str, object]) -> bool:
    case_type = row["case_type"]
    if metric_name == "family_exact":
        return case_type in _ANSWERABLE_CASES and row["family_exact"] is True
    if metric_name == "binding_exact":
        return case_type in _ANSWERABLE_CASES and row["binding_exact"] is True
    if metric_name == "scale_set_accuracy":
        return case_type in _ANSWERABLE_CASES and row["scale_set_exact"] is True
    if metric_name in ("joint_exact", "answerable_joint_exact"):
        return case_type in _ANSWERABLE_CASES and row["joint_exact"] is True
    if metric_name == "hard_negative_rejection":
        return case_type is Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE and row["predicted_decision"] is PredictionDecisionV2.ABSTAIN
    if metric_name == "binding_counterfactual_rejection":
        return case_type is Phase2BCaseType.BINDING_COUNTERFACTUAL and row["predicted_decision"] is PredictionDecisionV2.ABSTAIN
    if metric_name == "scale_counterfactual_rejection":
        return case_type is Phase2BCaseType.SCALE_COUNTERFACTUAL and row["predicted_decision"] is PredictionDecisionV2.ABSTAIN
    if metric_name == "sign_or_invariant_break_rejection":
        return case_type is Phase2BCaseType.SIGN_OR_INVARIANT_BREAK and row["predicted_decision"] is PredictionDecisionV2.ABSTAIN
    if metric_name == "all_control_rejection":
        return case_type in _CONTROL_CASES and row["predicted_decision"] is PredictionDecisionV2.ABSTAIN
    if metric_name == "abstention_specificity":
        return case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE and row["predicted_decision"] is not PredictionDecisionV2.ABSTAIN
    if metric_name == "nonidentifiable_scale_abstention":
        return case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE and row["predicted_decision"] is PredictionDecisionV2.ABSTAIN
    raise ValueError("unknown available gate metric")


def _eligible_for_metric(metric_name: str, row: dict[str, object]) -> bool:
    case_type = row["case_type"]
    if metric_name in ("family_exact", "binding_exact", "scale_set_accuracy", "joint_exact", "answerable_joint_exact"):
        return case_type in _ANSWERABLE_CASES
    if metric_name == "hard_negative_rejection":
        return case_type is Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE
    if metric_name == "binding_counterfactual_rejection":
        return case_type is Phase2BCaseType.BINDING_COUNTERFACTUAL
    if metric_name == "scale_counterfactual_rejection":
        return case_type is Phase2BCaseType.SCALE_COUNTERFACTUAL
    if metric_name == "sign_or_invariant_break_rejection":
        return case_type is Phase2BCaseType.SIGN_OR_INVARIANT_BREAK
    if metric_name == "all_control_rejection":
        return case_type in _CONTROL_CASES
    if metric_name == "abstention_specificity":
        return case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
    if metric_name == "nonidentifiable_scale_abstention":
        return case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
    raise ValueError("unknown available gate eligibility")


def _calculate_results(
    rows: tuple[dict[str, object], ...],
    gates: tuple[dict[str, object], ...],
    definition_ids: dict[str, str],
) -> tuple[
    tuple[Unsealed960AvailableGateMechanicsResultV2, ...],
    tuple[Unsealed960AvailableGateMechanicsResultV2, ...],
    tuple[Unsealed960UnavailableGateMechanicsV2, ...],
]:
    overall: list[Unsealed960AvailableGateMechanicsResultV2] = []
    for name, denominator, pn, pd, wn, wd in _OVERALL_GATE_SPECS:
        eligible = tuple(row for row in rows if _eligible_for_metric(name, row))
        if len(eligible) != denominator:
            raise ValueError("overall available denominator drift")
        overall.append(_issue_available_result(
            metric_name=name, scope="overall", slice_id=None,
            definition_id=definition_ids[f"overall:{name}"],
            successes=sum(_success_for_metric(name, row) for row in eligible),
            total=denominator, point_numerator=pn, point_denominator=pd,
            wilson_numerator=wn, wilson_denominator=wd,
        ))
    by_input = {row["input_row_id"]: row for row in rows}
    joined = tuple((by_input[gate["input_row_id"]], gate) for gate in gates)
    sliced: list[Unsealed960AvailableGateMechanicsResultV2] = []
    for family in CanonicalFamilyId:
        members = tuple(row for row, gate in joined if gate["canonical_family_id"] is family)
        for name, denominator, pn, pd, wn, wd in _SLICE_GATE_SPECS:
            eligible = tuple(row for row in members if _eligible_for_metric(name, row))
            if len(eligible) != denominator:
                raise ValueError("family slice denominator drift")
            sliced.append(_issue_available_result(
                metric_name=name, scope="family", slice_id=family.value,
                definition_id=definition_ids[f"family:{name}"],
                successes=sum(_success_for_metric(name, row) for row in eligible),
                total=denominator, point_numerator=pn, point_denominator=pd,
                wilson_numerator=wn, wilson_denominator=wd,
            ))
    for scale in FormalUnsealedScaleSliceIdV2:
        members = tuple(row for row, gate in joined if gate["scale_slice_id"] is scale)
        for name, denominator, pn, pd, wn, wd in _SCALE_SLICE_GATE_SPECS:
            eligible = tuple(row for row in members if _eligible_for_metric(name, row))
            if len(eligible) != denominator:
                raise ValueError("scale slice denominator drift")
            sliced.append(_issue_available_result(
                metric_name=name, scope="scale", slice_id=scale.value,
                definition_id=definition_ids[f"scale:{name}"],
                successes=sum(_success_for_metric(name, row) for row in eligible),
                total=denominator, point_numerator=pn, point_denominator=pd,
                wilson_numerator=wn, wilson_denominator=wd,
            ))
    missing = tuple(
        _issue_unavailable_result(
            metric_name=name, definition_id=definition_ids[f"overall:{name}"],
            point_numerator=pn, point_denominator=pd,
            wilson_numerator=wn, wilson_denominator=wd,
            missing_reason=reason,
        )
        for name, pn, pd, wn, wd, reason in _UNAVAILABLE_GATE_SPECS
    )
    if len(overall) != _OVERALL_COUNT or len(sliced) != _SLICE_COUNT or len(missing) != _UNAVAILABLE_COUNT:
        raise RuntimeError("available gate result cardinality drift")
    return tuple(overall), tuple(sliced), missing


_EXPECTED_OVERALL_THRESHOLD_PROJECTION: Final = (
    ("family_exact", "0x1.ccccccccccccdp-1", "0x1.b851eb851eb85p-1", "overall"),
    ("binding_exact", "0x1.ccccccccccccdp-1", "0x1.b851eb851eb85p-1", "overall"),
    ("scale_set_accuracy", "0x1.bd70a3d70a3d7p-1", "0x1.a3d70a3d70a3dp-1", "overall"),
    ("joint_exact", "0x1.b333333333333p-1", "0x1.999999999999ap-1", "overall"),
    ("hard_negative_rejection", "0x1.e666666666666p-1", "0x1.ccccccccccccdp-1", "overall"),
    ("binding_counterfactual_rejection", "0x1.e666666666666p-1", "0x1.ccccccccccccdp-1", "overall"),
    ("scale_counterfactual_rejection", "0x1.dc28f5c28f5c3p-1", "0x1.c28f5c28f5c29p-1", "overall"),
    ("sign_or_invariant_break_rejection", "0x1.e666666666666p-1", "0x1.ccccccccccccdp-1", "overall"),
    ("abstention_specificity", "0x1.e666666666666p-1", "0x1.ccccccccccccdp-1", "overall"),
    ("fail_closed_rate", "0x1.0000000000000p+0", None, "overall"),
    ("preservation_consistency", "0x1.f0a3d70a3d70ap-1", "0x1.e147ae147ae14p-1", "overall"),
    ("nonidentifiable_scale_abstention", "0x1.e666666666666p-1", "0x1.ccccccccccccdp-1", "overall"),
)
_EXPECTED_SLICE_THRESHOLD_PROJECTION: Final = (
    ("answerable_joint_exact", "0x1.999999999999ap-1", "0x1.6666666666666p-1", "family"),
    ("all_control_rejection", "0x1.c28f5c28f5c29p-1", "0x1.8f5c28f5c28f6p-1", "family"),
    ("abstention_specificity", "0x1.b333333333333p-1", "0x1.8000000000000p-1", "family"),
    ("answerable_joint_exact", "0x1.999999999999ap-1", "0x1.6666666666666p-1", "scale"),
    ("all_control_rejection", "0x1.c28f5c28f5c29p-1", "0x1.8f5c28f5c28f6p-1", "scale"),
    ("abstention_specificity", "0x1.b333333333333p-1", "0x1.8000000000000p-1", "scale"),
)
_FIELD_TYPE_MANIFEST: Final = tuple(
    (
        value_type.__name__,
        tuple((name, str(annotation)) for name, annotation in value_type.__annotations__.items()),
    )
    for value_type in (
        Unsealed960AvailableGateMechanicsResultV2,
        Unsealed960UnavailableGateMechanicsV2,
        Unsealed960AvailableGateMechanicsV2,
        Unsealed960AvailableGateMechanicsRejectionV2,
    )
)
_UPSTREAM_INPUT_FIELD_TYPE_MANIFEST: Final = tuple(
    (
        value_type.__name__,
        tuple(
            (item.name, str(value_type.__annotations__[item.name]))
            for item in fields(value_type)
        ),
    )
    for value_type in (
        Unsealed960PredictionScoringMechanicsV2,
        ActualUnsealed960ReplayInputContractV2,
        FormalUnsealedGateInputManifestV2,
        Unsealed960MainRowResultV2,
        Unsealed960MetricRowOutcomeV2,
        Unsealed960MetricResultV2,
        ActualReplayGateInputDefinitionV2,
        ActualReplayRequiredEvidenceV2,
        FormalUnsealedGateInputRowV2,
    )
)
_UPSTREAM_NONE_EMPTY_RULES: Final = (
    (
        "scoring_mechanics",
        ("scale_regret_result", "bootstrap_result"),
        ("gate_results",),
        "all_other_top_level_fields_non_none_exact_declared_type",
    ),
    (
        "replay_input_contract",
        ("scale_regret_result", "bootstrap_result"),
        ("metric_results", "scored_rows", "gate_results"),
        "all_other_top_level_fields_non_none_exact_declared_type",
    ),
    (
        "gate_input_manifest",
        (),
        (),
        "all_top_level_fields_non_none_exact_declared_type",
    ),
)
_UPSTREAM_ADDRESS_PREFIX_MANIFEST: Final = (
    ("scoring", tuple(_SCORING_ADDRESS_PREFIXES.items())),
    ("replay", tuple(_REPLAY_ADDRESS_PREFIXES.items())),
    ("gate_manifest", tuple(_MANIFEST_ADDRESS_PREFIXES.items())),
)
_CONTENT_ADDRESS_PARAMETERS: Final = (
    ("available_result", _RESULT_DOMAIN.hex(), _RESULT_PREFIX),
    ("unavailable_result", _UNAVAILABLE_DOMAIN.hex(), _UNAVAILABLE_PREFIX),
    ("success", _SUCCESS_DOMAIN.hex(), _SUCCESS_PREFIX),
    ("schema", b"HEGEL/PHASE2B/AVAILABLE_GATE/SCHEMA/V2\x00".hex(), "phase2b_unsealed_960_available_gate_mechanics_schema_v2_"),
    ("policy", b"HEGEL/PHASE2B/AVAILABLE_GATE/POLICY/V2\x00".hex(), "phase2b_unsealed_960_available_gate_mechanics_policy_v2_"),
    ("formal_answer_row", _ANSWER_ROW_DOMAIN.hex(), _ANSWER_ROW_PREFIX),
    ("formal_answer_manifest", _ANSWER_MANIFEST_DOMAIN.hex(), _ANSWER_MANIFEST_PREFIX),
)
_UPSTREAM_RECOMPUTATION_PARAMETERS: Final = (
    ("scoring_metric_outcome", "stable_hash_canonical_json_without_domain", "phase2b_unsealed_960_metric_row_outcome_v2_"),
    ("scoring_main_row", "stable_hash_canonical_json_without_domain", "phase2b_unsealed_960_main_row_result_v2_"),
    ("scoring_metric_result", "stable_hash_canonical_json_without_domain", "phase2b_unsealed_960_metric_result_v2_"),
    ("scoring_result", "stable_hash_canonical_json_without_domain", "phase2b_unsealed_960_prediction_scoring_mechanics_v2_"),
    ("replay_gate_definition", b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_DEFINITION/V2\x00".hex(), "phase2b_actual_replay_gate_input_definition_v2_"),
    ("replay_required_evidence", b"HEGEL/PHASE2B/ACTUAL_REPLAY/REQUIRED_EVIDENCE/V2\x00".hex(), "phase2b_actual_replay_required_evidence_v2_"),
    ("replay_result", b"HEGEL/PHASE2B/ACTUAL_REPLAY/RESULT/V2\x00".hex(), "phase2b_actual_unsealed_960_replay_input_contract_v2_"),
    ("gate_input_row", b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW/V2\x00".hex(), "phase2b_actual_replay_gate_input_row_v2_"),
    ("gate_input_row_ids", b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW_IDS/V2\x00".hex(), "phase2b_actual_replay_gate_input_rows_v2_"),
    ("gate_input_manifest", b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_MANIFEST/V2\x00".hex(), "phase2b_formal_unsealed_gate_input_manifest_v2_"),
    ("main_input_row_ids", b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00".hex(), "phase2b_unsealed_main_rows_v2_"),
    ("formal_answer_row_ids", b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00".hex(), "phase2b_formal_unsealed_answer_rows_v2_"),
    ("formal_answer_row", _ANSWER_ROW_DOMAIN.hex(), _ANSWER_ROW_PREFIX),
    ("formal_answer_manifest", _ANSWER_MANIFEST_DOMAIN.hex(), _ANSWER_MANIFEST_PREFIX),
)
_UPSTREAM_PREIMAGE_FIELD_MANIFEST: Final = (
    ("scoring_metric_outcome", ("metric_definition_id", "metric_name", "eligible", "success")),
    ("scoring_main_row", (
        "input_row_id", "prediction_record_id", "prediction_content_id",
        "answer_row_id", "case_type", "predicted_decision",
        "expected_decision", "predicted_canonical_family_id",
        "expected_canonical_family_id", "predicted_binding",
        "expected_binding", "predicted_admissible_scale_ids",
        "expected_admissible_scale_ids", "decision_exact", "family_exact",
        "binding_exact", "scale_set_exact", "joint_exact", "metric_eligible",
        "metric_outcomes",
    )),
    ("scoring_metric_result", (
        "metric_definition_id", "metric_name", "metric_kind",
        "denominator_case_types", "expected_denominator", "observed_denominator",
        "success_count", "count_value", "success_rule", "separately_reported",
    )),
    ("scoring_result", tuple(
        item.name for item in fields(Unsealed960PredictionScoringMechanicsV2)
        if item.name != "result_id"
    )),
    ("replay_gate_definition", (
        "gate_name", "scope", "expected_denominator", "success_rule",
        "input_available", "missing_input_reason",
    )),
    ("replay_required_evidence", (
        "evidence_name", "purpose", "supplied_by_this_contract",
        "verifier_implemented",
    )),
    ("replay_result", tuple(
        item.name for item in fields(ActualUnsealed960ReplayInputContractV2)
        if item.name != "result_id"
    )),
    ("gate_input_row", (
        "input_row_id", "answer_row_id", "case_type", "margin_stratum",
        "canonical_family_id", "scale_slice_id", "latent_base_case_id",
    )),
    ("gate_input_manifest", (
        "schema_version", "schema_id", "policy_id", "claim_level",
        "exact_freeze_id", "phase2b_protocol_id", "formal_scoring_contract_id",
        "execution_freeze_manifest_id", "input_archive_id", "input_archive_sha256",
        "input_archive_version", "input_archive_policy_id", "batch_id",
        "batch_policy_id", "ordered_archive_input_row_ids_root", "main_row_ids_root",
        "semantic_conflict_row_ids_root", "partition_union_row_ids_root",
        "answer_manifest_id", "answer_manifest_sha256", "main_answer_row_ids_root",
        "main_gate_input_rows", "main_gate_input_row_ids_root",
        "required_evidence_inventory",
    )),
    ("formal_answer_row", (
        "input_row_id", "case_type", "expected_decision", "canonical_family_id",
        "binding", "admissible_scale_ids",
    )),
    ("formal_answer_manifest", (
        "schema_version", "schema_id", "policy_id", "claim_level",
        "exact_freeze_id", "phase2b_protocol_id", "execution_freeze_manifest_id",
        "input_archive_id", "input_archive_sha256", "input_archive_version",
        "input_archive_policy_id", "batch_id", "batch_policy_id",
        "ordered_archive_input_row_ids_root", "main_row_ids_root",
        "semantic_conflict_row_ids_root", "partition_union_row_ids_root",
        "main_answer_rows", "main_answer_row_ids_root",
    )),
)


_SCHEMA_PREIMAGE: Final = {
    "version": UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION,
    "claim_level": UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL,
    "available_result_fields": _AVAILABLE_RESULT_FIELDS,
    "unavailable_result_fields": _UNAVAILABLE_RESULT_FIELDS,
    "success_fields": _SUCCESS_FIELDS,
    "rejection_fields": _REJECTION_FIELDS,
    "dispositions": tuple(item.value for item in Unsealed960AvailableGateMechanicsDispositionV2),
    "reasons": tuple(item.value for item in Unsealed960AvailableGateMechanicsReasonV2),
    "counts": (_MAIN_COUNT, _CHALLENGE_COUNT, _OVERALL_COUNT, _SLICE_COUNT, _UNAVAILABLE_COUNT),
    "caps": (_TEXT_CAP, _BINDING_CAP, _SCALE_CAP),
    "field_types_and_none_rules": _FIELD_TYPE_MANIFEST,
    "upstream_input_field_types": _UPSTREAM_INPUT_FIELD_TYPE_MANIFEST,
    "upstream_exact_none_and_empty_rules": _UPSTREAM_NONE_EMPTY_RULES,
    "top_level_exact_runtime_type_partition": _TOP_LEVEL_EXACT_TYPE_POLICY_MANIFEST,
    "upstream_content_address_prefixes": _UPSTREAM_ADDRESS_PREFIX_MANIFEST,
    "upstream_recomputation_parameters": _UPSTREAM_RECOMPUTATION_PARAMETERS,
    "upstream_preimage_fields": _UPSTREAM_PREIMAGE_FIELD_MANIFEST,
    "content_address_parameters": _CONTENT_ADDRESS_PARAMETERS,
    "ratio_grammar": "reduced_nonnegative_integer_numerator_slash_positive_integer_denominator",
    "binary64_grammar": "exact_builtin_float_hex_lowercase_finite",
    "one_sided_confidence_and_z_binary64": (
        _ONE_SIDED_CONFIDENCE.hex(),
        _ONE_SIDED_Z_HEX,
    ),
    "tuple_caps": (
        ("main_row_results", 720), ("metric_outcomes_per_row", 9),
        ("metric_results", 9), ("evidence_inventory", 18),
        ("available_definitions", 10), ("unavailable_definitions", 2),
        ("slice_definitions", 6), ("overall_outputs", 10),
        ("slice_outputs", 24), ("unavailable_outputs", 2),
    ),
}
UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID: Final = _stable_id(
    _SCHEMA_PREIMAGE,
    domain=b"HEGEL/PHASE2B/AVAILABLE_GATE/SCHEMA/V2\x00",
    prefix="phase2b_unsealed_960_available_gate_mechanics_schema_v2_",
)


def _success_mapping(
    value: Unsealed960AvailableGateMechanicsV2,
    *,
    include_id: bool,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in _SUCCESS_FIELDS:
        if name == "result_id" and not include_id:
            continue
        item = object.__getattribute__(value, name)
        if type(item) in (
            Unsealed960AvailableGateMechanicsDispositionV2,
            Unsealed960AvailableGateMechanicsReasonV2,
        ):
            result[name] = item.value
        elif name in ("available_overall_gate_mechanics_results", "available_slice_gate_mechanics_results"):
            result[name] = [_available_result_mapping(nested) for nested in item]
        elif name == "unavailable_gate_mechanics":
            result[name] = [_unavailable_result_mapping(nested) for nested in item]
        else:
            result[name] = item
    return result


def _issue_success(
    *,
    scoring: dict[str, object],
    replay: dict[str, object],
    manifest: dict[str, object],
    overall: tuple[Unsealed960AvailableGateMechanicsResultV2, ...],
    sliced: tuple[Unsealed960AvailableGateMechanicsResultV2, ...],
    unavailable: tuple[Unsealed960UnavailableGateMechanicsV2, ...],
) -> Unsealed960AvailableGateMechanicsV2:
    value = object.__new__(Unsealed960AvailableGateMechanicsV2)
    frozen: list[tuple[str, object]] = [
        ("disposition", Unsealed960AvailableGateMechanicsDispositionV2.AVAILABLE_GATE_MECHANICS_COMPLETE_NOT_FORMAL_GATE_EVALUATION),
        ("reason", Unsealed960AvailableGateMechanicsReasonV2.TEN_OVERALL_AND_TWENTY_FOUR_SLICE_MECHANICS_COMPLETE),
        ("version", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION),
        ("schema_id", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_POLICY_ID),
        ("claim_level", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL),
        ("result_id", ""),
        ("scoring_mechanics_result_id", scoring["result_id"]),
        ("replay_input_contract_result_id", replay["result_id"]),
        ("gate_input_manifest_id", manifest["gate_input_manifest_id"]),
        ("gate_input_manifest_sha256", manifest["gate_input_manifest_sha256"]),
        ("gate_input_manifest_schema_id", manifest["schema_id"]),
        ("gate_input_manifest_policy_id", manifest["policy_id"]),
        ("prediction_archive_id", scoring["prediction_archive_id"]),
        ("prediction_archive_sha256", scoring["prediction_archive_sha256"]),
        ("answer_manifest_id", scoring["answer_manifest_id"]),
        ("answer_manifest_sha256", scoring["answer_manifest_sha256"]),
        ("scoring_mechanics_schema_id", scoring["schema_id"]),
        ("scoring_mechanics_policy_id", scoring["policy_id"]),
        ("scoring_mechanics_version", scoring["version"]),
        ("scoring_mechanics_claim_level", scoring["claim_level"]),
        ("replay_input_contract_schema_id", replay["schema_id"]),
        ("replay_input_contract_policy_id", replay["policy_id"]),
        ("replay_input_contract_version", replay["version"]),
        ("replay_input_contract_claim_level", replay["claim_level"]),
        ("gate_input_manifest_schema_version", manifest["schema_version"]),
        ("gate_input_manifest_claim_level", manifest["claim_level"]),
        ("protocol_id", scoring["protocol_id"]),
        ("formal_scoring_contract_id", scoring["formal_scoring_contract_id"]),
        ("formal_scoring_contract_schema_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID),
        ("formal_scoring_contract_policy_id", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID),
        ("formal_scoring_contract_version", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION),
        ("formal_scoring_contract_claim_level", FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL),
        ("main_row_ids_root", scoring["main_row_ids_root"]),
        ("semantic_conflict_row_ids_root", scoring["semantic_conflict_row_ids_root"]),
        ("main_answer_row_ids_root", scoring["main_answer_row_ids_root"]),
        ("main_gate_input_row_ids_root", replay["main_gate_input_row_ids_root"]),
        ("main_row_count", _MAIN_COUNT),
        ("semantic_conflict_excluded_count", _CHALLENGE_COUNT),
        ("overall_result_count", _OVERALL_COUNT),
        ("slice_result_count", _SLICE_COUNT),
        ("unavailable_result_count", _UNAVAILABLE_COUNT),
    ]
    frozen.extend((name, True) for name in _TRUE_CLAIMS)
    frozen.extend((name, False) for name in _FALSE_CLAIMS)
    frozen.extend((
        ("available_overall_gate_mechanics_results", overall),
        ("available_slice_gate_mechanics_results", sliced),
        ("unavailable_gate_mechanics", unavailable),
    ))
    for name, item in frozen:
        object.__setattr__(value, name, item)
    object.__setattr__(value, "result_id", _stable_id(
        tuple(_success_mapping(value, include_id=False).items()),
        domain=_SUCCESS_DOMAIN,
        prefix=_SUCCESS_PREFIX,
    ))
    return value


def evaluate_unsealed_960_available_gate_mechanics_v2(
    *,
    scoring_mechanics: Unsealed960PredictionScoringMechanicsV2,
    replay_input_contract: ActualUnsealed960ReplayInputContractV2,
    gate_input_manifest: FormalUnsealedGateInputManifestV2,
) -> Unsealed960AvailableGateMechanicsV2 | Unsealed960AvailableGateMechanicsRejectionV2:
    """Compose supplied mechanics; never perform an actual/formal evaluation."""

    if (
        type(scoring_mechanics) is not Unsealed960PredictionScoringMechanicsV2
        or type(replay_input_contract) is not ActualUnsealed960ReplayInputContractV2
        or type(gate_input_manifest) is not FormalUnsealedGateInputManifestV2
    ):
        return _issue_rejection(Unsealed960AvailableGateMechanicsReasonV2.WRONG_INPUT_TYPE)
    try:
        # Every caller-controlled slot and nested slot is copied into primitive
        # closed graphs before the first content hash or Wilson operation.
        try:
            scoring = _snapshot_scoring(scoring_mechanics)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _AvailableGateRejected(Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID) from exc
        try:
            replay = _snapshot_replay(replay_input_contract)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _AvailableGateRejected(Unsealed960AvailableGateMechanicsReasonV2.REPLAY_INPUT_CONTRACT_INVALID) from exc
        try:
            manifest = _snapshot_manifest(gate_input_manifest)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _AvailableGateRejected(Unsealed960AvailableGateMechanicsReasonV2.GATE_INPUT_MANIFEST_INVALID) from exc
        try:
            rows, gates, definition_ids = _validate_semantics(scoring, replay, manifest)
        except _RowJoinOrQuota as exc:
            raise _AvailableGateRejected(
                Unsealed960AvailableGateMechanicsReasonV2.ROW_JOIN_OR_QUOTA_MISMATCH
            ) from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _AvailableGateRejected(Unsealed960AvailableGateMechanicsReasonV2.CROSS_BINDING_MISMATCH) from exc
        try:
            _verify_content_addresses(scoring, replay, manifest)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _AvailableGateRejected(Unsealed960AvailableGateMechanicsReasonV2.CROSS_BINDING_MISMATCH) from exc
        try:
            overall, sliced, unavailable = _calculate_results(rows, gates, definition_ids)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _AvailableGateRejected(Unsealed960AvailableGateMechanicsReasonV2.ROW_JOIN_OR_QUOTA_MISMATCH) from exc
        return _issue_success(
            scoring=scoring, replay=replay, manifest=manifest,
            overall=overall, sliced=sliced, unavailable=unavailable,
        )
    except _AvailableGateRejected as exc:
        return _issue_rejection(exc.reason)
    except _CrossVersionInput:
        return _issue_rejection(Unsealed960AvailableGateMechanicsReasonV2.CROSS_VERSION_INPUT)
    except Exception:
        return _issue_rejection(Unsealed960AvailableGateMechanicsReasonV2.INTERNAL_ERROR)


_POLICY_PREIMAGE: Final = {
    "version": UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION,
    "schema_id": UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID,
    "claim_level": UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL,
    "source_sha256": (_SCORING_SOURCE_SHA256, _REPLAY_INPUT_SOURCE_SHA256),
    "protocol_id": _FROZEN_PROTOCOL_ID,
    "exact_freeze_id": _FROZEN_EXACT_FREEZE_ID,
    "formal_scoring_contract_id": _FROZEN_FORMAL_SCORING_CONTRACT_ID,
    "api_signature": (
        "evaluate_unsealed_960_available_gate_mechanics_v2",
        "keyword_only",
        (
            ("scoring_mechanics", "exact_Unsealed960PredictionScoringMechanicsV2"),
            ("replay_input_contract", "exact_ActualUnsealed960ReplayInputContractV2"),
            ("gate_input_manifest", "exact_FormalUnsealedGateInputManifestV2"),
        ),
        "fresh_success_or_atomic_rejection_union",
    ),
    "upstream_identities": (
        UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION,
        UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID,
        UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID,
        UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL,
        ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION,
        ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID,
        ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID,
        ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL,
        FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION,
        FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID,
        FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID,
        FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION,
        FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID,
        FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID,
        FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        _FROZEN_FORMAL_SCORING_CONTRACT_ID,
        _FROZEN_PREDICTION_ARCHIVE_VERSION,
        _FROZEN_PREDICTION_ARCHIVE_POLICY_ID,
        _FROZEN_RUN_CONTEXT_SCHEMA_VERSION,
        _FROZEN_PREDICTION_RECORD_SCHEMA_VERSION,
        _FROZEN_INPUT_ARCHIVE_VERSION,
        _FROZEN_INPUT_ARCHIVE_POLICY_ID,
        _FROZEN_BATCH_POLICY_ID,
        _ANSWER_MANIFEST_SCHEMA_VERSION,
        _ANSWER_MANIFEST_SCHEMA_ID,
        _ANSWER_MANIFEST_POLICY_ID,
    ),
    "upstream_input_field_types": _UPSTREAM_INPUT_FIELD_TYPE_MANIFEST,
    "upstream_exact_none_and_empty_rules": _UPSTREAM_NONE_EMPTY_RULES,
    "top_level_exact_runtime_type_partition": _TOP_LEVEL_EXACT_TYPE_POLICY_MANIFEST,
    "upstream_content_address_prefixes": _UPSTREAM_ADDRESS_PREFIX_MANIFEST,
    "upstream_recomputation_parameters": _UPSTREAM_RECOMPUTATION_PARAMETERS,
    "upstream_preimage_fields": _UPSTREAM_PREIMAGE_FIELD_MANIFEST,
    "three_graph_cross_binding_manifest": _CROSS_BINDING_MANIFEST,
    "true_claims": _TRUE_CLAIMS,
    "false_claims": _FALSE_CLAIMS,
    "available_overall_gate_specs": _OVERALL_GATE_SPECS,
    "available_family_gate_specs": _SLICE_GATE_SPECS,
    "available_scale_gate_specs": _SCALE_SLICE_GATE_SPECS,
    "unavailable_gate_specs": _UNAVAILABLE_GATE_SPECS,
    "gate_input_definition_specs": _GATE_DEFINITION_SPECS,
    "required_evidence_specs": _EVIDENCE_SPECS,
    "nine_metric_specs": _METRIC_SPECS,
    "one_sided_confidence_and_z_binary64": (
        _ONE_SIDED_CONFIDENCE.hex(),
        _ONE_SIDED_Z_HEX,
    ),
    "overall_threshold_projection": _EXPECTED_OVERALL_THRESHOLD_PROJECTION,
    "slice_threshold_projection": _EXPECTED_SLICE_THRESHOLD_PROJECTION,
    "slice_expansion_order": (
        tuple((family.value, metric[0]) for family in CanonicalFamilyId for metric in _SLICE_GATE_SPECS),
        tuple((scale.value, metric[0]) for scale in FormalUnsealedScaleSliceIdV2 for metric in _SCALE_SLICE_GATE_SPECS),
    ),
    "case_semantics": (
        ("answerable", tuple(item.value for item in _ANSWERABLE_CASES)),
        ("controls", tuple(item.value for item in _CONTROL_CASES)),
        ("specificity", Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE.value),
        ("nonidentifiable", Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE.value),
        ("control_success", PredictionDecisionV2.ABSTAIN.value),
    ),
    "join_and_quotas": (
        "input_row_id_and_answer_row_id_exact_bijection_not_position",
        "answerable_family_equals_expected_answer_family_controls_expected_family_none",
        "six_family_by_two_opaque_scale_cells_each_exact_60",
        "per_cell_cases_19_1_8_8_8_8_8",
        "per_cell_margins_21_18_12_9",
        "unique_720_latent_ids_do_not_prove_independence",
        "opaque_S01_S02_never_compared_to_admissible_scale_UUIDs",
        "semantic_conflict_240_root_bound_and_excluded_from_all_34_denominators",
        "cross_bind_input_archive_id_sha_batch_execution_freeze_ordered_archive_partition_union_answer_main_challenge_protocol_formal_contract_and_gate_manifest_identities",
    ),
    "success_disposition": (
        Unsealed960AvailableGateMechanicsDispositionV2
        .AVAILABLE_GATE_MECHANICS_COMPLETE_NOT_FORMAL_GATE_EVALUATION.value
    ),
    "formulas": (
        "global_exact_snapshot_and_semantic_preflight_before_any_hash_or_wilson",
        "independent_upstream_nested_content_address_recalculation",
        "formal_answer_row_id_equals_prefix_plus_sha256(answer_row_domain_plus_canonical_json_exact_named_mapping)",
        "formal_answer_manifest_sha_equals_sha256(answer_manifest_domain_plus_canonical_json_exact_named_mapping_with_720_recomputed_answer_rows)",
        "formal_answer_manifest_id_equals_answer_manifest_prefix_plus_recomputed_sha",
        "exact_720_input_row_id_answer_row_id_bijective_join",
        "one_sided_95_percent_binary64_wilson_lower_bound",
        "available_gate_passed_equals_point_threshold_passed_and_wilson_threshold_passed",
        "missing_gate_counts_estimates_bounds_and_passes_are_none_not_zero",
        "z_is_frozen_binary64_projection_of_statistics_NormalDist_inv_cdf_0_95",
        "z_binary64_hex_equals_0x1_a515209676ab8p_plus_0",
        "wilson_equals_max_0_of_center_minus_radius_over_1_plus_z_squared_over_n",
        "center_equals_p_plus_z_squared_over_2n",
        "radius_equals_z_times_sqrt_p_times_1_minus_p_over_n_plus_z_squared_over_4n_squared",
        "point_pass_uses_exact_integer_cross_multiplication",
        "wilson_pass_uses_binary64_lower_bound_greater_equal_exact_rational_threshold",
        "wilson_pass_cross_multiplies_float_as_integer_ratio_against_exact_threshold_rational",
        "result_preimages_are_declared_order_tuple_of_field_name_and_exact_closed_value_excluding_own_id_with_float_hex_not_json_float",
    ),
    "validation_order": (
        "exact_three_top_level_types",
        "exact_upstream_version_fields_route_to_cross_version_before_any_hash",
        "single_read_complete_scoring_replay_manifest_nested_snapshot",
        "all_type_none_utf8_cap_enum_count_duplicate_catalog_claim_quota_join_checks",
        "all_upstream_nested_answer_row_answer_manifest_and_top_content_address_recomputation",
        "available_count_and_wilson_mechanics",
        "fresh_single_success_or_atomic_all_false_no_id_no_result_rejection",
    ),
    "forbidden_actions": (
        "no_decoder_scorer_evaluator_or_upstream_validator_call",
        "no_filesystem_network_clock_random_subprocess_container_runner_ledger_or_signature_call",
        "no_actual_formal_scientific_effect_or_c1_claim",
    ),
}
UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_POLICY_ID: Final = _stable_id(
    _POLICY_PREIMAGE,
    domain=b"HEGEL/PHASE2B/AVAILABLE_GATE/POLICY/V2\x00",
    prefix="phase2b_unsealed_960_available_gate_mechanics_policy_v2_",
)


def _issue_rejection(
    reason: Unsealed960AvailableGateMechanicsReasonV2,
) -> Unsealed960AvailableGateMechanicsRejectionV2:
    value = object.__new__(Unsealed960AvailableGateMechanicsRejectionV2)
    frozen: list[tuple[str, object]] = [
        ("disposition", Unsealed960AvailableGateMechanicsDispositionV2.REJECTED),
        ("reason", reason),
        ("version", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION),
        ("schema_id", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_POLICY_ID),
        ("claim_level", UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL),
        ("validation", None),
        ("available_overall_gate_mechanics_results", ()),
        ("available_slice_gate_mechanics_results", ()),
        ("unavailable_gate_mechanics", ()),
        ("partial_output_published", False),
    ]
    frozen.extend((name, False) for name in (*_TRUE_CLAIMS, *_FALSE_CLAIMS))
    for name, item in frozen:
        object.__setattr__(value, name, item)
    return value


for _type, _manifest in (
    (Unsealed960AvailableGateMechanicsResultV2, _AVAILABLE_RESULT_FIELDS),
    (Unsealed960UnavailableGateMechanicsV2, _UNAVAILABLE_RESULT_FIELDS),
    (Unsealed960AvailableGateMechanicsV2, _SUCCESS_FIELDS),
    (Unsealed960AvailableGateMechanicsRejectionV2, _REJECTION_FIELDS),
):
    if tuple(item.name for item in fields(_type)) != _manifest:
        raise RuntimeError(f"available-gate V2 {_type.__name__} field drift")


__all__ = (
    "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION",
    "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL",
    "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID",
    "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_POLICY_ID",
    "Unsealed960AvailableGateMechanicsDispositionV2",
    "Unsealed960AvailableGateMechanicsReasonV2",
    "Unsealed960AvailableGateMechanicsResultV2",
    "Unsealed960UnavailableGateMechanicsV2",
    "Unsealed960AvailableGateMechanicsV2",
    "Unsealed960AvailableGateMechanicsRejectionV2",
    "evaluate_unsealed_960_available_gate_mechanics_v2",
)
