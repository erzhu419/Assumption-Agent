"""Formal Phase-2B field-level covert-channel auditor and variant runner.

The authoritative entry point accepts public fixed envelopes, a private audit
authority transcript, and executable wire/recognizer adapters.  It never
accepts caller-produced invariant receipts, recognizer outputs, thresholds, or
permutation seeds.  Baseline plus 32 global UUID renamings, 32 case-order
permutations, and 16 within-case observation-order permutations are constructed
and executed inside this module.

This implementation can qualify an implementation on synthetic/development
transcripts.  An authority transcript root proves only which private labels
were used; it does not prove their origin, independence, or actual-corpus
status.  Those claims remain the responsibility of the later custody ceremony.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import (
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from enum import Enum
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import struct
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Callable, Final, TypeAlias

from .hashing import stable_hash
from .phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)


SCHEMA_VERSION: Final = "hegel-machine-phase2b-formal-covert-auditor/5"
SEMANTICS_VERSION: Final = (
    "hegel-machine-phase2b-formal-covert-statistics/9-cgroup-memory-max-transient-overage-erratum"
)
CLAIM_LEVEL: Final = (
    "AUDIT_TRANSCRIPT_ONLY_NOT_AUTHORITY_ORIGIN_NOT_INDEPENDENCE_NOT_ACTUAL_CORPUS"
)
POSITIVE_DECISION_CLAIM: Final = (
    "NO_MATERIAL_COVERT_CHANNEL_DETECTED_AT_TARGET_FAMILY_FWER_0.01"
)
TRANSCRIPT_AUTHENTICATION: Final = (
    "COOPERATIVE_SAME_PROCESS_OBJECT_REGISTRY_NOT_UNFORGEABILITY_"
    "NOT_ISSUER_SIGNATURE_NOT_CROSS_PROCESS_AUTHORITY"
)
RANDOMIZATION_ASSUMPTION: Final = (
    "AUDITOR_OS_CSPRNG_SHA256_PRF_WITH_REPLACEMENT_COMPUTATIONAL_RANDOMIZATION_"
    "NOT_AUTHENTICATED_INDEPENDENCE_OR_UNCONDITIONAL_MONTE_CARLO_EXACTNESS"
)
ENVELOPE_BYTES: Final = 65_536
LABEL_PERMUTATIONS: Final = 10_000
EXCHANGEABLE_SAMPLE_SIZE: Final = LABEL_PERMUTATIONS + 1
GLOBAL_RENAMINGS: Final = 32
GLOBAL_CASE_ORDER_PERMUTATIONS: Final = 32
OBSERVATION_ORDER_PERMUTATIONS: Final = 16
MAXIMUM_ROWS: Final = 1_024
MAXIMUM_FEATURES: Final = 32_768
MAXIMUM_VALUES_PER_FEATURE: Final = 1_024
MAXIMUM_LABEL_CLASSES: Final = 256
MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS: Final = 20_000_000_000
MAXIMUM_ESTIMATED_RSS_BYTES: Final = 12_884_901_888
MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES: Final = (1 << 63) - 1
MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS: Final = 50_000_000_000
MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES: Final = 2_147_483_648
MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES: Final = 268_435_456
MAXIMUM_METRIC_CACHE_ENTRIES: Final = 2_048
MAXIMUM_METRIC_CACHE_CELLS: Final = 64
PARALLEL_STATISTICS_WORK_THRESHOLD: Final = 250_000_000
HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES: Final = 2_147_483_648
STATISTICS_WORKER_TIMEOUT_SECONDS: Final = 3_600
PARENT_POSTPROCESS_DEADLINE_SCHEMA: Final = (
    "ABSOLUTE_MONOTONIC_COOPERATIVE_PARENT_POSTPROCESS_CHECKS_V1"
)
PARTITION_PREPARATION_TIMEOUT_SECONDS: Final = 600
INVARIANCE_WORKER_TIMEOUT_SECONDS: Final = 7_200
INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND: Final = 1_048_576
INVARIANCE_WORKER_ARTIFACT_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-covert-invariant-artifact/1"
)
STATISTICS_WORKER_ARTIFACT_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-covert-statistics-artifact/1"
)
ARTIFACT_TEMP_ROOT: Final = "/tmp"
INVARIANCE_EXECUTION_CONTAINMENT: Final = (
    "LINUX_SETSID_DIRECT_CHILD_PROCESS_GROUP_EMPTY_AT_EXIT_NOT_CONTINUOUS_"
    "DESCENDANT_PROOF_COOPERATIVE_NO_DAEMONIZE_OR_NEW_SESSION_CONTRACT_"
    "STDIO_FDS_0_1_2_INHERITED_AS_TCB_"
    "OUTER_ISOLATED_QUALIFICATION_CGROUP_REQUIRED_FOR_EXTERNAL_EXECUTION_SCOPE_"
    "PHYSICAL_PEAK_CLAIM"
)
PARENT_SIDE_ADAPTER_TCB: Final = (
    "COOPERATIVE_PARENT_IMPLEMENTATION_ID_SCHEMA_RETAINED_BOUND_ALIGNMENT_"
    "BIND_PRIVATE_AUTHORITY_DECODE_ENVELOPE_TCB_NOT_SANDBOXED"
)
WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE: Final = 512
WORKER_RESULT_ARTIFACT_FIXED_BYTES: Final = 1_048_576
TARGET_MAX_VECTOR_BYTES_PER_PERMUTATION: Final = 64
MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS: Final = 8
MAXIMUM_RECOGNIZER_BINDINGS_PER_ROW: Final = 256
MAXIMUM_RECOGNIZER_SCALES_PER_ROW: Final = 256
MAXIMUM_RECOGNIZER_TEXT_UTF8_BYTES: Final = 4_096
MAXIMUM_RECOGNIZER_RESIDUAL_BYTES_PER_ROW: Final = 1_048_576
MAXIMUM_RECOGNIZER_RESIDUAL_BYTES_PER_BATCH: Final = 67_108_864
RECOGNIZER_OUTPUT_METADATA_DEEP_BYTES_PER_ROW: Final = 262_144
RECOGNIZER_OUTPUT_SIMULTANEOUS_GRAPH_MULTIPLIER: Final = 3
WIRE_V3_ATOMIC_SUITE_UNIQUE_BATCH_EQUIVALENTS: Final = 49
ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE: Final = 512
ATOMIC_VARIANT_SUITE_FIXED_BYTES_BOUND: Final = 268_435_456
DEEP_SIZE_SAFETY_MULTIPLIER: Final = 4
MAXIMUM_DECODE_GRAPH_BYTES_PER_ROW: Final = 4 * 1_048_576
MAXIMUM_SHUFFLE_REJECTION_DRAWS: Final = 4_096
PREEXTRACTION_FEATURE_CELL_BYTES_BOUND: Final = 256
PREEXTRACTION_FEATURE_ALIAS_FIXED_BYTES_BOUND: Final = 1_024
PREEXTRACTION_PRESENT_VALUE_SAFETY_MULTIPLIER: Final = 4
PREEXTRACTION_DESCRIPTOR_UTF8_EXPANSION_MULTIPLIER: Final = 512
PARENT_RESULT_WORKING_BYTES_PER_HYPOTHESIS: Final = 8_192
PARENT_RESULT_WORKING_FIXED_BYTES_PER_TARGET: Final = 1_048_576
BINARY64_SUPERACCUMULATOR_FRACTION_BITS: Final = 1_074
BINARY64_SUPERACCUMULATOR_WORDS: Final = (
    BINARY64_SUPERACCUMULATOR_FRACTION_BITS + 63
) // 64
NMI_LIMIT: Final = 0.02
BALANCED_ACCURACY_ADVANTAGE_LIMIT: Final = 0.05
FAMILY_WISE_ALPHA: Final = 0.01
FAMILY_WISE_ALPHA_NUMERATOR: Final = 1
FAMILY_WISE_ALPHA_DENOMINATOR: Final = 100
TARGET_FAMILY_HOLM_SIZE: Final = 5

WIRE_V3_SOURCE_SHA256: Final = (
    "c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765"
)
WIRE_V3_TEST_SHA256: Final = (
    "253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0"
)
WIRE_V3_CONFIG_SHA256: Final = (
    "4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784"
)
WIRE_V3_DOC_SHA256: Final = (
    "704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a"
)
EXACT_TRANSFORM_CORE_SOURCE_SHA256: Final = (
    "e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f"
)
EXACT_TRANSFORM_CORE_TEST_SHA256: Final = (
    "15d0d6084743d21c2b7a79ec0d4f2f07c84ecfef641fb3d47f5253c95e5a3920"
)
RFC8785_SOURCE_SHA256: Final = (
    "16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654"
)
WIRE_V3_FIELD_CLASSIFICATION_MANIFEST_ID: Final = (
    "phase2b_formal_wire_field_manifest_v3_"
    "12916eefe10770f54c3d6c71f4b8b2c14ebbd671642e96ff4082dd298e5088dc"
)
WIRE_V3_GRAPH_COMPLETENESS_POLICY_ID: Final = (
    "phase2b_formal_case_graph_completeness_policy_v3_"
    "7585a9f9e58823d766906296905efcd24b1e350330ca5c3f64d75361e9ad90e5"
)

TARGETS: Final = (
    "family",
    "binding",
    "scale",
    "answerable_vs_abstain",
    "joint_decision_class",
)
if len(TARGETS) != TARGET_FAMILY_HOLM_SIZE:
    raise RuntimeError("target-family Holm size drift")
RENAMING_NAMESPACES: Final = (
    "bundle_id",
    "observation_id",
    "entity_id",
    "role_candidate_id",
    "quantity_id",
    "context_id",
    "task_id",
    "scale_id",
    "aggregate_map_id",
    "transform_id",
)
MUST_BE_ABSENT_FIELDS: Final = (
    "answer",
    "case_sequence_index",
    "challenge_label",
    "filename",
    "latent",
    "local_id",
    "order_label",
    "overlap_stratum",
    "private_answer",
    "private_truth",
    "public_order",
    "row_order",
    "source_case_id",
    "source_id",
    "source_index",
    "source_order",
    "transform_catalog",
    "unused_transform_ids",
)
UNIQUE_ID_FEATURE_FAMILY: Final = (
    "128_individual_bits",
    "first_8_16_32_bits",
    "last_8_16_32_bits",
    "hamming_weight",
    "integer_mod_3_5_7_11_13",
    "hex_character_histogram",
)
CHANNEL_GROUPS: Final = (
    "uuid_leaf",
    "missingness",
    "provenance",
    "payload_length",
    "padding",
    "record_order",
    "slot_order",
    "eligibility_mask",
    "control_metadata",
    "descriptor_metadata",
    "byte_prefix_suffix",
)
CHANNEL_ONLY_OWNER_GROUPS: Final = (
    "uuid_leaf",
    "missingness",
    "provenance",
    "record_order",
    "slot_order",
    "eligibility_mask",
    "control_metadata",
)

_SEMANTICS_DESCRIPTOR: Final = (
    "private_authority=labels_and_strata_derived_only_from_transcript",
    "private_authority_adapter_isolation=primitive_snapshot_pass_copy_post_call_replay_then_fresh_unexposed_execution_copy",
    "private_source_binding=each_label_row_commitment_exactly_matches_wire_issued_source_case_commitment_in_source_order",
    "variants=auditor_generated_baseline_plus_32_renaming_plus_32_case_order_plus_16_observation_order",
    "invariance_execution=whole_81_call_atomic_suite_decode_compare_root_pipeline_in_one_raw_fork_where_7200_seconds_triggers_termination_followed_by_at_most_5_seconds_term_grace_then_kill_and_reap_with_parent_canonical_replay_content_sha256_binding",
    "renaming=domain_separated_global_namespace_bijections",
    "invariants=decision_family_binding_scale_set_residual_bit_exact_after_inverse_mapping",
    "id_features=128_bits_plus_prefix_aliases_plus_hamming_plus_moduli_plus_hex_histogram",
    "prefix_aliases=first_last_8_16_32_are_bit_decomposition_aliases_deduplicated_by_semantic_column",
    "other_features=missingness_provenance_payload_length_padding_record_slot_order_eligibility_control_descriptor_representation_metadata_envelope_prefix_suffix",
    "descriptor_unicode_profile=ucd_13_0_0_derived_core_sha256_a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248_prop_list_sha256_485b5a3ed25dbf1f94dfa5a9b69d8b4550ffd0c33045ccc55ccfd7c80b2a40cf_default_ignorable_4173_range_sha256_629f6c5f9f20e8322371ff59b61ce34246a9c4afb8fd299756540058dc8a553e",
    "descriptor_validation=exact_closed_nonempty_nfc_rfc8785_utf8_bounded_reject_categories_Cc_Cf_Cs_Co_Cn_Zl_Zp_default_ignorables_noncharacters_uuid_sha256_and_raw_64hex_tokens",
    "descriptor_metadata=kind_segment_count_position_lengths_spaces_boundaries_exact_allowed_whitespace_position_codepoint_sequences_and_runs_full_unicode_category_and_ascii_class_sequences_ordered_lexical_token_lengths_nfkc_change_deltas_and_counts",
    "descriptor_composite=ordered_canonical_representation_metadata_categorical_without_lexical_or_punctuation_symbol_exact_identity",
    "wire_v3_stop=source_c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765_test_253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0_config_4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784_doc_704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a_git_commit_not_a_runtime_authority_exact_candidate_bytes_only",
    "wire_v3_public_decode=raw_envelope_only_constructor_all_derived_fields_init_false_one_union_validation_preserves_header_payload_padding_bounds_digests_one_canonical_parse_restored_integer_canonical_replay_normalization_context_graph_leaf_classification_channel_projection_envelope_id_and_false_claims",
    "wire_v3_variant_suite=one_atomic_exact_80_schedule_build_success_replay_then_strict_public_decode_of_every_returned_envelope_before_recognizer",
    "wire_v3_variant_suite_adapter_bound_state=one_deep_build_success_variant_context_private_alignment_and_envelope_binding_replay_per_exact_80_suite_while_every_plan_and_schedule_is_independently_replayed",
    "wire_v3_policy_manifest=graph_7585a9f9e58823d766906296905efcd24b1e350330ca5c3f64d75361e9ad90e5_field_12916eefe10770f54c3d6c71f4b8b2c14ebbd671642e96ff4082dd298e5088dc",
    "wire_v3_dependencies=exact_core_source_e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f_test_15d0d6084743d21c2b7a79ec0d4f2f07c84ecfef641fb3d47f5253c95e5a3920_rfc8785_source_16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654",
    "wire_v3_fresh_build_verification=caller_held_run_id_keys_source_cases_catalog_and_binding_rebuilt_exactly_external_complete_status_ignored",
    "audited_bytes=padding_prefix16_suffix16_envelope_prefix16_suffix16_provenance_digest_prefix16_suffix16",
    "byte_primary=raw_byte_categorical_is_retained_as_its_own_holm_hypothesis",
    "byte_supplements=8_bits_plus_all_28_within_byte_pair_xors_plus_high_low_nibbles_plus_byte_hamming_weight",
    "byte_non_equivalence=supplements_are_not_claimed_equivalent_to_raw_byte_categorical_test",
    "padding_summaries=length_total_hamming_and_all_256_full_padding_byte_histogram_counts",
    "variable_shape=each_union_uuid_channel_and_provenance_path_has_an_explicit_presence_hypothesis",
    "resource_preallocation=decoded_union_paths_project_exact_feature_count_before_columns_then_incremental_32768_cap_early_and_branch_converged_fresh_parent_only_feature_gates_use_max_current_or_envelope_decode_plus_projected_feature_result_and_worker_artifact_then_partition_parent_only_gate_uses_max_current_or_feature_plus_envelope_decode_plus_projected_partition_and_result",
    "resource_feature_projection=256_bytes_per_row_feature_cell_plus_1024_bytes_per_semantic_alias_plus_path_name_repetition_present_channel_value_and_512x_descriptor_utf8_expansion_bounds",
    "resource_phase_order=decode_then_feature_cardinality_projection_and_early_parent_only_host_gate_then_eligibility_quota_and_sufficiency_then_invariant_plan_for_all_inputs_sufficient_executes_invariance_insufficient_skips_it_both_converge_on_fresh_parent_only_feature_gate_then_feature_partition_and_statistics_plan_then_only_admissible_insufficient_receipt_or_sufficient_statistics",
    "resource_invariance_plan=max_parent_current_or_envelope_decode_plus_adapter_plus_64MiB_times_2_plus_variant_resident_recognizer_live_and_artifact_excludes_phase_exclusive_feature_and_partition_graphs",
    "resource_statistics_plan=max_parent_current_or_masks_plus_partition_plus_adapter_plus_64MiB_times_parent_plus_workers_plus_exact_worker_and_parent_tails_excludes_released_phase_exclusive_feature_and_envelope_decode_graphs",
    "resource_partition_projection=per_concrete_feature_cardinality_checked_at_1024_then_python_integer_tuple_partition_allocation_projected_before_materialization_and_one_producer_owned_deep_graph_measurement_inside_statistics_resource_plan_no_caller_scalar",
    "resource_partition_deadline=600_seconds_absolute_from_cardinality_projection_through_single_partition_materialization_and_resource_plan_measurement_separate_from_statistics_clock",
    "resource_final_fork_preflight=after_invariance_snapshot_or_statistics_masks_artifact_and_worker_ownership_inputs_resample_exact_parent_current_rss_recompute_max_current_or_phase_static_projection_and_bind_current_projection_static_multiplier_tail_plan_root_cap_and_margin_into_runtime_evidence_statistics_raw_worker_requires_plan_and_empty_evidence_slot_with_no_optional_bypass",
    "resource_plan_cross_binding=private_invariant_plan_root_copied_into_public_statistics_composite_plan_each_phase_runtime_evidence_replays_projection_from_its_observed_current_and_receipt_reconstructs_only_static_multiplier_tail_cap_margin_and_exact_plan_root_from_the_composite_plan_current_and_projected_values_are_runtime_evidence_not_plan_reconstruction",
    "resource_admissibility_priority=feature_projection_and_invariant_plan_resource_failures_precede_both_branches_then_feature_partition_and_statistics_plan_resource_failures_precede_even_an_insufficient_receipt_no_resource_inadmissible_receipt_is_issued",
    "resource_post_invariance_failure=feature_partition_or_statistics_plan_failure_after_completed_invariance_issues_no_receipt_restores_reservation_and_cleans_auditor_artifacts_but_does_not_claim_zero_recognizer_callbacks_or_malicious_callback_rollback",
    "resource_runtime_observation=phase_preflight_is_exact_fork_boundary_current_projection_parent_whole_lifetime_ru_maxrss_and_wait4_worker_peaks_are_diagnostics_only_bounded_to_signed_64bit_bytes_while_one_exact_unified_cgroup_v2_attempt_leaf_memory_peak_is_the_independent_12GiB_physical_cap_authority",
    "resource_cgroup_counter_semantics=finite_structural_snapshot_requires_peak_ge_current_but_does_not_assume_current_or_peak_le_memory_max_because_memory_max_may_be_temporarily_exceeded_target_runtime_dtos_and_explicit_gates_still_require_peak_le_12GiB_physical_cap",
    "resource_parent_results=8192_bytes_per_target_feature_hypothesis_plus_1MiB_per_target_bounds_live_artifact_deserialization_diagnostics_probabilities_and_final_dtos",
    "uuid_absence_encoding=absent_is_internal_all_zero_128_bit_value_outside_canonical_uuidv4_domain",
    "provenance_absence_encoding=absent_is_internal_all_zero_32_byte_digest",
    "generic_channel_absence_encoding=tagged_present_or_absent_with_separate_presence_hypothesis",
    "channel_partition=every_wire_declared_channel_only_concrete_leaf_has_exactly_one_registered_owner",
    "channel_feature_projection=integrity_hashes_replay_raw_paths_but_statistics_use_closed_order_slot_eligibility_tokens_not_full_evidence_records",
    "eligibility_signature=presence_and_count_for_observation_role_candidate_and_scale_namespaces_order_independent_without_id_bytes",
    "eligibility_quota=for_family_scale_answerability_each_signature_by_class_cell_times_N_equals_signature_total_times_class_total_using_exact_integers",
    "eligibility_selection_conditioning=family_scale_answerability_permutations_append_exact_structural_signature_to_each_preexisting_stratum_so_every_draw_preserves_quota_cells",
    "absence=answer_case_sequence_filename_latent_local_private_answer_source_case_source_id_source_index_per_case_transform_catalog_and_unused_transform_ids_must_be_absent",
    "nmi=max_zero_raw_nmi_minus_fully_symmetric_mean_of_observed_plus_10000_within_strata_permuted_nmi",
    "symmetric_center_arithmetic=order_independent_nonnegative_binary64_integer_superaccumulator_at_2^-1074_then_exact_fraction_divide_10001_then_pinned_runtime_correct_round",
    "balanced_accuracy=exact_leave_one_out_single_categorical_feature_mode_minus_one_over_class_count",
    "omnibus=max(bias_corrected_nmi_over_0.02,positive_ba_advantage_over_0.05)",
    "randomization_seed=auditor_os_csprng_32_bytes_drawn_before_reading_authority_or_envelope_content_caller_seed_forbidden_commit_then_reveal",
    "randomization_schedule=sha256_prf_with_replacement_target_stratum_uniform_shuffle_stream",
    "randomization_claim=computational_assumption_not_authenticated_seed_independence_or_unconditional_exact_monte_carlo",
    "null_assumption=private_target_labels_exchangeable_within_every_preregistered_stratum",
    "exchangeability=observed_plus_10000_permutations_use_one_symmetric_center_and_each_target_permutation_is_shared_across_all_features",
    "target_max_t=per_permutation_maximum_omnibus_over_every_concrete_feature_with_semantic_aliases_retained_in_manifest",
    "target_family_p=exact_rational_(1+count_permuted_target_max_ge_observed_target_max)/10001_with_ties_in_tail",
    "feature_max_t_localization_p=exact_rational_(1+count_permuted_target_max_ge_observed_feature_omnibus)/10001_with_ties_in_tail",
    "multiple_testing=holm_bonferroni_over_exactly_5_target_family_p_values_under_arbitrary_cross_target_dependence",
    "inferential_scope=five_complete_target_nulls_each_asserting_all_registered_features_null_for_that_target_under_its_quota_conditioned_strata_no_partial_null_or_feature_localization_strong_fwer_claim",
    "decision=fail_iff_same_feature_has_observed_omnibus_strictly_above_1_and_its_target_family_holm_rejects_strictly_below_0.01_and_its_feature_max_t_p_is_strictly_below_that_target_holm_critical_value_otherwise_pass",
    "positive_claim=NO_MATERIAL_COVERT_CHANNEL_DETECTED_AT_TARGET_FAMILY_FWER_0.01",
    "feature_localization=max_t_witness_only_not_partial_null_strong_fwer_inference",
    "pairwise_p=(1+count_permuted_feature_omnibus_ge_observed_feature_omnibus)/10001",
    "pairwise_holm=resolution_limited_replay_diagnostic_decision_eligible_false",
    "threshold_equality=pass",
    "insufficient=atomic_reject_without_partial_statistics",
    "statistics=feature_and_label_value_bitsets_with_python_int_popcount",
    "statistics_intersection_pass=observed_plus_one_ordered_10000_metric_tuple_materialization_then_scalar_pairwise_and_shared_target_max_t_scan",
    "intersection_work=unique_partition_value_count_times_label_class_count_times_10001",
    "permutation_metric_tuple_bound=256_bytes_per_permutation_per_parallel_worker",
    "target_max_vector_bound=64_bytes_per_permutation_per_parallel_worker",
    "worker_result_transport=one_or_five_raw_wait4_children_private_retained_dirfd_closed_rfc8785_artifacts_parent_canonical_replay_and_content_sha256_binding_no_object_or_pickle_ipc",
    "worker_result_artifact_bound=512_bytes_per_concrete_feature_plus_1MiB_per_worker",
    "memory_accounting=runtime_deep_object_graph_size_times_4_with_fixed_envelope_feature_floors_4MiB_per_row_decode_gate_phase_specific_lifetimes_and_current_rss_covering_retained_allocator_or_outer_aliases",
    "host_capacity=min_linux_memavailable_and_same_level_cgroup_ancestor_remaining_cpu_capacity_min_os_affinity_and_tightest_cgroup_ancestor_quota",
    "affinity_read_failure=conservative_one_worker_capacity_never_parallel_five",
    "shuffle_rejection=at_most_4096_sha256_u64_draws_per_randbelow_then_atomic_fail",
    "statistics_deadline=3600_seconds_one_exact_absolute_monotonic_nanosecond_deadline_created_before_statistics_preparation_and_passed_unchanged_through_raw_statistics_waits_parent_result_and_receipt_finalization_the_waiter_checks_deadline_before_accepting_observed_exit_timeout_uses_FAIL_STATISTICAL_RUNTIME_containment_uses_FAIL_STATISTICAL_CONTAINMENT_and_both_identify_raw_statistics_child_then_terminate_followed_by_at_most_5_seconds_term_grace_then_kill_and_reap_invariance_preserves_its_distinct_7200_second_wait_boundary_deadline_created_after_pidfd_ownership_immediately_before_wait_timeout_retains_FAIL_INVARIANCE_RUNTIME_and_containment_retains_FAIL_INVARIANCE_CONTAINMENT_parent_result_and_receipt_replay_use_cooperative_absolute_monotonic_checks_not_preemptive_only_external_supervisor_has_hard_end_to_end_deadline",
    "qualification_supervisor=fresh_isolated_outer_cgroup_pids_memory_and_hard_total_deadline_required_for_external_execution_claim",
    "fork_integrity_boundary=raw_fork_provides_liveness_resource_and_copy_on_write_state_isolation_not_malicious_callback_integrity_wire_v3_stop_bytes_are_module_hash_pinned_while_recognizer_and_other_adapter_ids_are_only_recorded_cooperatively_stable_tcb_external_cas_or_signature_must_pin_their_exact_bytes",
    "fork_sigchld_boundary=python_sigchld_ignore_is_rejected_selected_lifecycle_and_cancellation_signals_are_blocked_then_sigchld_is_temporarily_default_through_owned_wait4_and_original_disposition_is_restored_before_unblocking_other_parent_signal_handlers_are_cooperative_tcb_external_qualification_requires_fresh_default_signal_dispositions_and_cgroup_pids_containment",
    "artifact_cleanup_boundary=ordinary_paths_remove_private_artifacts_path_replacement_or_unexpected_directory_fails_closed_without_following_or_recursive_deletion_and_may_leave_one_bounded_forensic_temp_directory_for_external_namespace_cleanup",
    "resource_caps=1024_rows_32768_features_20000000000_intersections_50000000000_total_statistical_work_units_12884901888_static_phase_plan_estimate_bytes_and_12884901888_unified_cgroup_v2_attempt_leaf_memory_peak_physical_cap_bytes",
    "receipt_execution_coverage=nonempty_invariant_and_permutation_roots_81_runs_rows_times_80_times_5_comparisons",
    "receipt_finalization=unified_issuance_evidence_for_sufficient_and_insufficient_one_exact_attempt_leaf_cgroup_v2_memory_peak_prefinal_sample_one_construct_full_replay_insert_exact_pending_entry_one_producer_pending_full_replay_capture_validated_id_then_under_one_registry_lock_recheck_exact_pending_entry_id_scope_identity_absolute_deadline_and_monotonic_physical_peak_at_or_below_12GiB_immediately_before_issued_no_post_commit_work_else_direct_exact_rollback_no_retry_parent_and_wait4_peaks_diagnostic_only",
    "receipt_authentication=cooperative_same_process_factory_object_identity_only_not_unforgeability_signature_or_authority",
    "receipt_registry=maximum_8_live_receipts_mutable_exact_entry_pending_to_issued_fsm_public_validate_and_release_require_issued_preallocated_reservation_slot_O1_rollback_no_silent_eviction_atomic_reservation_before_execution",
)
SEMANTICS_ID: Final = stable_hash(
    {"version": SEMANTICS_VERSION, "descriptor": _SEMANTICS_DESCRIPTOR},
    prefix="phase2b_formal_covert_semantics_",
)

_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UUID_SHAPED = re.compile(
    r"^(?:urn:uuid:)?(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}|\{[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\})$",
    flags=re.IGNORECASE,
)
_DESCRIPTOR_UUID_TOKEN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_DESCRIPTOR_SHA256_TOKEN = re.compile(r"sha256:[0-9a-fA-F]{64}")
_DESCRIPTOR_HEX_DIGEST_TOKEN = re.compile(
    r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])"
)
DESCRIPTOR_UNICODE_DATA_VERSION: Final = "13.0.0"
DESCRIPTOR_UCD13_DERIVED_CORE_PROPERTIES_SHA256: Final = (
    "sha256:a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248"
)
DESCRIPTOR_UCD13_PROP_LIST_SHA256: Final = (
    "sha256:485b5a3ed25dbf1f94dfa5a9b69d8b4550ffd0c33045ccc55ccfd7c80b2a40cf"
)
DESCRIPTOR_DEFAULT_IGNORABLE_CODEPOINT_COUNT: Final = 4_173
DESCRIPTOR_DEFAULT_IGNORABLE_RANGE_TABLE_SHA256: Final = (
    "sha256:629f6c5f9f20e8322371ff59b61ce34246a9c4afb8fd299756540058dc8a553e"
)
_DESCRIPTOR_DEFAULT_IGNORABLE_RANGES: Final = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
if unicodedata.unidata_version != DESCRIPTOR_UNICODE_DATA_VERSION:
    raise RuntimeError("formal covert descriptor Unicode database version drift")
if sum(
    upper - lower + 1
    for lower, upper in _DESCRIPTOR_DEFAULT_IGNORABLE_RANGES
) != DESCRIPTOR_DEFAULT_IGNORABLE_CODEPOINT_COUNT:
    raise RuntimeError("formal covert descriptor default-ignorable count drift")
if (
    "sha256:"
    + hashlib.sha256(
        canonicalize_rfc8785_v1(
            [list(value) for value in _DESCRIPTOR_DEFAULT_IGNORABLE_RANGES]
        )
    ).hexdigest()
    != DESCRIPTOR_DEFAULT_IGNORABLE_RANGE_TABLE_SHA256
):
    raise RuntimeError("formal covert descriptor default-ignorable table drift")
_VARIANT_DOMAIN: Final = b"HEGEL/PHASE2B/FORMAL_COVERT/VARIANT/V1\x00"
_PERMUTATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_COVERT/TARGET_MAX_T_LABEL_PERMUTATION/V2\x00"
)
_DEVELOPMENT_LABEL_LINK_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_COVERT/DEVELOPMENT_LABEL_LINK/V1\x00"
)
_PRIVATE_LABEL_PROJECTION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_COVERT/PRIVATE_LABEL_PROJECTION/V1\x00"
)

JsonPathToken: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathToken, ...]


class FormalCovertDisposition(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REJECT_INSUFFICIENT = "REJECT_INSUFFICIENT"


class AnswerabilityLabelV1(str, Enum):
    ANSWERABLE = "answerable"
    ABSTAIN = "abstain"


class FormalCovertAuditRejectionV1(RuntimeError):
    """Atomic structural, adapter, resource, or invariance rejection."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _reject(code: str, detail: str) -> None:
    raise FormalCovertAuditRejectionV1(code, detail)


def _append_exception_note(error: BaseException, note: str) -> None:
    """Attach cleanup context on pinned Python 3.10 and newer runtimes."""

    if hasattr(error, "add_note"):
        error.add_note(note)  # type: ignore[attr-defined]
        return
    prior = getattr(error, "__notes__", None)
    if prior is None:
        setattr(error, "__notes__", [note])
    elif type(prior) is list:
        prior.append(note)
    else:
        setattr(error, "__notes__", [*tuple(prior), note])


def _record_cleanup_failure_without_replacing_primary(
    primary: BaseException,
    cleanup_error: BaseException,
    operation: str,
) -> None:
    """Record a cleanup failure while preserving the original BaseException."""

    try:
        _append_exception_note(
            primary,
            f"{operation} cleanup failed with "
            f"{type(cleanup_error).__name__}: {cleanup_error}",
        )
    except BaseException:
        # The primary exception is the caller-visible failure identity.  Even a
        # hostile exception formatter or note hook may not replace it.
        pass


def _exact_nonempty_str(value: object, name: str, maximum: int = 512) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be an exact non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds the UTF-8 budget")
    return value


def _sha_id(domain: bytes, value: object, prefix: str) -> str:
    digest = stable_hash(value).encode("ascii")
    return prefix + hashlib.sha256(domain + digest).hexdigest()


def _bytes_commit(value: bytes) -> dict[str, object]:
    return {"length": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _path_text(path: JsonPath) -> str:
    pieces = []
    for token in path:
        raw = str(token).replace("~", "~0").replace("/", "~1")
        pieces.append(raw)
    return "/" + "/".join(pieces)


def _validate_json_tree(value: object) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > 65_536 or depth > 64:
            raise ValueError("decoded public JSON exceeds the resource budget")
        exact = type(current)
        if exact in (type(None), bool, int, str):
            continue
        if exact is float:
            if not math.isfinite(current):
                raise ValueError("decoded public JSON contains a non-finite number")
            continue
        if exact is list:
            if len(current) > 4_096:
                raise ValueError("decoded public JSON array exceeds the resource budget")
            stack.extend((item, depth + 1) for item in current)
            continue
        if exact is dict:
            if len(current) > 4_096 or any(type(key) is not str for key in current):
                raise TypeError("decoded public JSON object keys must be exact strings")
            stack.extend((item, depth + 1) for item in current.values())
            continue
        raise TypeError("decoded public value is outside the JSON data model")


def _resolve(root: object, path: JsonPath) -> object:
    current = root
    for token in path:
        if type(token) is str and type(current) is dict and token in current:
            current = current[token]
        elif (
            type(token) is int
            and type(current) is list
            and 0 <= token < len(current)
        ):
            current = current[token]
        else:
            raise ValueError(f"JSON path does not resolve: {_path_text(path)}")
    return current


def _replace(root: object, path: JsonPath, value: object) -> None:
    if not path:
        raise ValueError("root replacement is forbidden")
    parent = _resolve(root, path[:-1])
    token = path[-1]
    if type(token) is str and type(parent) is dict and token in parent:
        parent[token] = value
        return
    if type(token) is int and type(parent) is list and 0 <= token < len(parent):
        parent[token] = value
        return
    raise ValueError(f"JSON replacement path does not resolve: {_path_text(path)}")


@dataclass(frozen=True, slots=True)
class NamespaceLeafV1:
    namespace: str
    path: JsonPath
    public_uuid: str

    def __post_init__(self) -> None:
        if type(self) is not NamespaceLeafV1:
            raise TypeError("namespace leaf must use the exact type")
        if self.namespace not in RENAMING_NAMESPACES:
            raise ValueError("namespace leaf uses an unregistered namespace")
        if type(self.path) is not tuple or not self.path:
            raise TypeError("namespace leaf path must be a non-empty exact tuple")
        if any(type(item) not in (str, int) for item in self.path):
            raise TypeError("namespace leaf path contains an invalid token")
        if type(self.public_uuid) is not str or _UUID4.fullmatch(self.public_uuid) is None:
            raise ValueError("namespace leaf must contain a canonical lowercase UUIDv4")


@dataclass(frozen=True, slots=True)
class ChannelLeafV1:
    group: str
    path: JsonPath
    canonical_value: str
    feature_value: str | None = None

    def __post_init__(self) -> None:
        if type(self) is not ChannelLeafV1:
            raise TypeError("channel leaf must use the exact type")
        if self.group not in (
            "missingness",
            "provenance",
            "record_order",
            "slot_order",
            "eligibility_mask",
            "control_metadata",
            "descriptor_metadata",
        ):
            raise ValueError("channel leaf group is not registered")
        if type(self.path) is not tuple or not self.path:
            raise TypeError("channel leaf path must be a non-empty exact tuple")
        _exact_nonempty_str(self.canonical_value, "channel canonical value", 8_192)
        if self.feature_value is None:
            object.__setattr__(self, "feature_value", self.canonical_value)
        else:
            _exact_nonempty_str(self.feature_value, "channel feature value", 8_192)


@dataclass(frozen=True, slots=True)
class ChannelOnlyLeafOwnerV1:
    path: JsonPath
    owner_group: str

    def __post_init__(self) -> None:
        if type(self) is not ChannelOnlyLeafOwnerV1:
            raise TypeError("channel-only owner must use the exact type")
        if type(self.path) is not tuple or not self.path:
            raise TypeError("channel-only owner path must be a non-empty tuple")
        if any(type(item) not in (str, int) for item in self.path):
            raise TypeError("channel-only owner path contains an invalid token")
        if self.owner_group not in CHANNEL_ONLY_OWNER_GROUPS:
            raise ValueError("channel-only leaf owner group is not registered")


@dataclass(frozen=True, slots=True)
class DecodedFormalEnvelopeV1:
    payload: object
    payload_bytes: bytes
    padding_bytes: bytes
    namespace_leaves: tuple[NamespaceLeafV1, ...]
    observation_array_paths: tuple[JsonPath, ...]
    channel_leaves: tuple[ChannelLeafV1, ...]
    channel_only_leaf_owners: tuple[ChannelOnlyLeafOwnerV1, ...]

    def __post_init__(self) -> None:
        if type(self) is not DecodedFormalEnvelopeV1:
            raise TypeError("decoded envelope must use the exact adapter DTO")
        _validate_json_tree(self.payload)
        if type(self.payload_bytes) is not bytes or not self.payload_bytes:
            raise TypeError("decoded payload bytes must be exact non-empty bytes")
        if type(self.padding_bytes) is not bytes:
            raise TypeError("decoded padding bytes must be exact bytes")
        if len(self.payload_bytes) + len(self.padding_bytes) >= ENVELOPE_BYTES:
            raise ValueError("decoded payload and padding leave no fixed header")
        if type(self.namespace_leaves) is not tuple or any(
            type(item) is not NamespaceLeafV1 for item in self.namespace_leaves
        ):
            raise TypeError("decoded namespace leaves must use an exact tuple")
        if type(self.channel_leaves) is not tuple or any(
            type(item) is not ChannelLeafV1 for item in self.channel_leaves
        ):
            raise TypeError("decoded channel leaves must use an exact tuple")
        if type(self.channel_only_leaf_owners) is not tuple or any(
            type(item) is not ChannelOnlyLeafOwnerV1
            for item in self.channel_only_leaf_owners
        ):
            raise TypeError("decoded channel-only leaf owners must use an exact tuple")
        if type(self.observation_array_paths) is not tuple or any(
            type(path) is not tuple or not path for path in self.observation_array_paths
        ):
            raise TypeError("observation array paths must use exact tuples")
        ordered = tuple(
            sorted(
                self.namespace_leaves,
                key=lambda item: (item.namespace, _path_text(item.path), item.public_uuid),
            )
        )
        if self.namespace_leaves != ordered or len(
            {(item.namespace, item.path) for item in self.namespace_leaves}
        ) != len(self.namespace_leaves):
            raise ValueError("namespace leaves are not canonical and unique")
        if self.channel_leaves != tuple(
            sorted(self.channel_leaves, key=lambda item: (item.group, _path_text(item.path)))
        ):
            raise ValueError("channel leaves are not canonical")
        if self.channel_only_leaf_owners != tuple(
            sorted(
                self.channel_only_leaf_owners,
                key=lambda item: (_path_text(item.path), item.owner_group),
            )
        ) or len({item.path for item in self.channel_only_leaf_owners}) != len(
            self.channel_only_leaf_owners
        ):
            raise ValueError(
                "channel-only concrete leaves must have one canonical owner each"
            )
        for item in self.namespace_leaves:
            if _resolve(self.payload, item.path) != item.public_uuid:
                raise ValueError("namespace leaf does not match decoded payload")
        for path in self.observation_array_paths:
            if type(_resolve(self.payload, path)) is not list:
                raise ValueError("observation array path does not resolve to a list")
        for item in self.channel_only_leaf_owners:
            _resolve(self.payload, item.path)


class FormalWireAdapterV1(ABC):
    """Frozen adapter boundary for a trusted formal-wire implementation."""

    @property
    @abstractmethod
    def implementation_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def schema_id(self) -> str:
        raise NotImplementedError

    @property
    def retained_graph_bytes_bound(self) -> int:
        """Frozen parent/child resident allowance for development adapters."""

        return MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES

    @property
    def variant_suite_resident_bytes_per_row_bound(self) -> int:
        # Development adapters produce/consume one batch at a time.
        return ENVELOPE_BYTES + ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE

    @property
    def variant_suite_resident_fixed_bytes_bound(self) -> int:
        return INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND

    @abstractmethod
    def decode_envelope(self, envelope: bytes) -> DecodedFormalEnvelopeV1:
        raise NotImplementedError

    @abstractmethod
    def canonical_payload_bytes(self, payload: object) -> bytes:
        raise NotImplementedError

    def normalize_variant_payload(self, payload: object) -> object:
        """Development-only raw mutation hook used by the generic adapter path."""

        raise NotImplementedError

    @property
    @abstractmethod
    def private_alignment(self) -> "FormalPrivateBatchAlignmentV1":
        raise NotImplementedError

    def bind_private_authority(
        self,
        authority: "FormalAuditAuthorityV1",
        envelopes: tuple[bytes, ...],
    ) -> "FormalPrivateAuthorityBindingV1":
        """Bind labels to the adapter-owned alignment.

        The default is deliberately development-only: it validates the local
        row/link commitments but cannot claim wire-builder issuance or UUID
        projection.  A production adapter must override it.
        """

        alignment = self.private_alignment
        _validate_authority_links(envelopes, authority, alignment)
        projected = tuple(
            authority.rows[link.source_row_index].canonical_source_label_bytes
            for link in alignment.links_in_public_order
        )
        return FormalPrivateAuthorityBindingV1(
            alignment=alignment,
            public_label_projection_root=_private_label_projection_root(projected),
            alignment_issuance_verified=False,
            label_row_binding_verified=True,
        )

    def make_variant_envelopes(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple["DecodedFormalEnvelopeV1", ...],
        plan: "FormalVariantPlanV1",
    ) -> tuple[bytes, ...]:
        """Execute an explicit variant plan.

        This generic implementation exists for synthetic/development adapters.
        Trusted Wire V3 overrides it with its full-BuildSuccess-only surface.
        """

        _validate_variant_plan(plan, envelopes, decoded)
        rename = {
            (row.namespace, row.source_uuid): row.target_uuid
            for row in plan.uuid_renamings
        }
        orders_by_row: dict[int, list[FormalObservationOrderPlanV1]] = defaultdict(list)
        for row in plan.observation_orders:
            orders_by_row[row.canonical_row_index].append(row)
        variants: list[bytes] = []
        for canonical_row_index in plan.case_order_indices:
            item = decoded[canonical_row_index]
            payload = deepcopy(item.payload)
            for leaf in item.namespace_leaves:
                target = rename.get((leaf.namespace, leaf.public_uuid))
                if target is not None:
                    _replace(payload, leaf.path, target)
            for order in orders_by_row.get(canonical_row_index, ()):
                array = _resolve(payload, order.path)
                _replace(payload, order.path, [array[index] for index in order.order_indices])
            normalized = self.normalize_variant_payload(payload)
            _validate_json_tree(normalized)
            envelope = self.reencode_envelope(normalized, item.padding_bytes)
            if type(envelope) is not bytes or len(envelope) != ENVELOPE_BYTES:
                _reject("FAIL_VARIANT_ENVELOPE", "wire adapter returned a non-fixed envelope")
            replay = self.decode_envelope(envelope)
            if type(replay) is not DecodedFormalEnvelopeV1:
                _reject("FAIL_VARIANT_DECODE", "wire adapter returned a non-exact decoded DTO")
            expected = self.canonical_payload_bytes(normalized)
            if (
                replay.payload_bytes != expected
                or replay.payload != normalized
                or replay.padding_bytes != item.padding_bytes
            ):
                _reject(
                    "FAIL_VARIANT_WIRE_REPLAY",
                    "variant envelope failed exact decode/reencode replay",
                )
            _validate_decoded(replay, self)
            variants.append(envelope)
        return tuple(variants)

    @property
    def supports_atomic_variant_suite(self) -> bool:
        """Whether this adapter can atomically issue all 80 frozen schedules."""

        return False

    def make_variant_suite_envelopes(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple["DecodedFormalEnvelopeV1", ...],
        plans: tuple["FormalVariantPlanV1", ...],
    ) -> "_FormalVariantSuiteExecutionV1":
        raise NotImplementedError(
            "this wire adapter has no atomic variant-suite issuance surface"
        )

    def reencode_envelope(self, payload: object, padding_bytes: bytes) -> bytes:
        """Development-only raw mutation hook used by the generic path."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True, order=True)
class FormalBindingV1:
    role_candidate_id: str
    entity_id: str

    def __post_init__(self) -> None:
        if _UUID4.fullmatch(self.role_candidate_id) is None or _UUID4.fullmatch(
            self.entity_id
        ) is None:
            raise ValueError("formal binding IDs must be canonical UUIDv4")


@dataclass(frozen=True, slots=True)
class FormalRecognizerDecisionV1:
    decision: str
    family: str
    binding: tuple[FormalBindingV1, ...]
    scale_set: tuple[str, ...]
    candidate_residual_vector: bytes

    def __post_init__(self) -> None:
        if type(self) is not FormalRecognizerDecisionV1:
            raise TypeError("recognizer decision must use the exact DTO")
        _exact_nonempty_str(self.decision, "recognizer decision")
        _exact_nonempty_str(self.family, "recognizer family")
        if (
            len(self.decision.encode("utf-8"))
            > MAXIMUM_RECOGNIZER_TEXT_UTF8_BYTES
            or len(self.family.encode("utf-8"))
            > MAXIMUM_RECOGNIZER_TEXT_UTF8_BYTES
        ):
            raise ValueError("recognizer decision/family text exceeds its byte cap")
        if type(self.binding) is not tuple or any(
            type(item) is not FormalBindingV1 for item in self.binding
        ):
            raise TypeError("recognizer binding must use an exact tuple")
        if (
            len(self.binding) > MAXIMUM_RECOGNIZER_BINDINGS_PER_ROW
            or self.binding != tuple(sorted(set(self.binding)))
        ):
            raise ValueError("recognizer binding must be bounded, unique, and canonical")
        for item in self.binding:
            item.__post_init__()
        if type(self.scale_set) is not tuple or any(
            type(item) is not str or _UUID4.fullmatch(item) is None
            for item in self.scale_set
        ):
            raise TypeError("recognizer scale set must contain canonical UUIDv4 values")
        if (
            len(self.scale_set) > MAXIMUM_RECOGNIZER_SCALES_PER_ROW
            or self.scale_set != tuple(sorted(set(self.scale_set)))
        ):
            raise ValueError("recognizer scale set must be bounded and canonical")
        if type(self.candidate_residual_vector) is not bytes:
            raise TypeError("candidate residual vector must be exact bytes")
        if (
            len(self.candidate_residual_vector)
            > MAXIMUM_RECOGNIZER_RESIDUAL_BYTES_PER_ROW
        ):
            raise ValueError("candidate residual vector exceeds the row budget")

    def commitment(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "family": self.family,
            "binding": tuple(
                (item.role_candidate_id, item.entity_id) for item in self.binding
            ),
            "scale_set": self.scale_set,
            "residual": _bytes_commit(self.candidate_residual_vector),
        }


class FormalRecognizerAdapterV1(ABC):
    """Batch recognizer executable; the auditor invokes it 81 times."""

    @property
    @abstractmethod
    def implementation_id(self) -> str:
        raise NotImplementedError

    @property
    def retained_graph_bytes_bound(self) -> int:
        """Frozen resident allowance; production runs also enforce RLIMIT_AS."""

        return MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES

    @abstractmethod
    def run_batch(
        self, envelopes: tuple[bytes, ...]
    ) -> tuple[FormalRecognizerDecisionV1, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FormalAuthorityLabelInputV1:
    """Private semantic label before a wire source/link binding is attached."""

    row_index: int
    family: str
    binding: str
    scale: str
    answerability: AnswerabilityLabelV1
    joint_decision_class: str
    case_type: str

    def __post_init__(self) -> None:
        if type(self) is not FormalAuthorityLabelInputV1:
            raise TypeError("authority label input must use the exact private type")
        if type(self.row_index) is not int or not 0 <= self.row_index < MAXIMUM_ROWS:
            raise ValueError("authority row index is outside the frozen range")
        for name in (
            "family",
            "binding",
            "scale",
            "joint_decision_class",
            "case_type",
        ):
            _exact_nonempty_str(getattr(self, name), f"private authority {name}")
        if type(self.answerability) is not AnswerabilityLabelV1:
            raise TypeError("private answerability must use the exact enum")

    def canonical_source_label_bytes(self) -> bytes:
        from .rfc8785_v1 import canonicalize_rfc8785_v1

        return canonicalize_rfc8785_v1(
            {
                "answerability": self.answerability.value,
                "binding": self.binding,
                "case_type": self.case_type,
                "family": self.family,
                "joint_decision_class": self.joint_decision_class,
                "row_index": self.row_index,
                "scale": self.scale,
            }
        )


@dataclass(frozen=True, slots=True)
class FormalAuthorityLabelRowV1:
    """Private label plus its exact source/link/content binding."""

    row_index: int
    family: str
    binding: str
    scale: str
    answerability: AnswerabilityLabelV1
    joint_decision_class: str
    case_type: str
    source_case_commitment: str
    label_binding_id: str
    row_binding: str
    canonical_source_label_bytes: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not FormalAuthorityLabelRowV1:
            raise TypeError("authority label row must use the exact private type")
        label = FormalAuthorityLabelInputV1(
            self.row_index,
            self.family,
            self.binding,
            self.scale,
            self.answerability,
            self.joint_decision_class,
            self.case_type,
        )
        if (
            type(self.source_case_commitment) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_case_commitment)
            is None
        ):
            raise ValueError("private authority row needs an exact source-case commitment")
        for value, name in (
            (self.label_binding_id, "private authority label binding ID"),
            (self.row_binding, "private authority row binding"),
        ):
            if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be an exact SHA-256 digest")
        object.__setattr__(
            self,
            "canonical_source_label_bytes",
            label.canonical_source_label_bytes(),
        )


@dataclass(frozen=True, slots=True)
class FormalAuditAuthorityV1:
    rows: tuple[FormalAuthorityLabelRowV1, ...]
    transcript_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not FormalAuditAuthorityV1:
            raise TypeError("formal audit authority must use the exact private type")
        if type(self.rows) is not tuple or not 1 <= len(self.rows) <= MAXIMUM_ROWS:
            raise ValueError("formal audit authority row count is outside the frozen range")
        if any(type(item) is not FormalAuthorityLabelRowV1 for item in self.rows):
            raise TypeError("formal audit authority contains a non-exact row")
        if tuple(item.row_index for item in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("formal audit authority rows must be contiguous and ordered")
        object.__setattr__(
            self,
            "transcript_root",
            stable_hash(
                {
                    "schema": SCHEMA_VERSION,
                    "private_rows": tuple(
                        {
                            "row_index": item.row_index,
                            "family": item.family,
                            "binding": item.binding,
                            "scale": item.scale,
                            "answerability": item.answerability.value,
                            "joint_decision_class": item.joint_decision_class,
                            "case_type": item.case_type,
                            "source_case_commitment": item.source_case_commitment,
                            "label_binding_id": item.label_binding_id,
                            "row_binding": item.row_binding,
                            "canonical_source_label": _bytes_commit(
                                item.canonical_source_label_bytes
                            ),
                        }
                        for item in self.rows
                    ),
                },
                prefix="phase2b_formal_audit_authority_transcript_",
            ),
        )


_AuthorityPrimitiveRowV1: TypeAlias = tuple[
    int,
    str,
    str,
    str,
    AnswerabilityLabelV1,
    str,
    str,
    str,
    str,
    str,
]


def _authority_from_primitive_rows(
    rows: tuple[_AuthorityPrimitiveRowV1, ...],
) -> FormalAuditAuthorityV1:
    return FormalAuditAuthorityV1(
        tuple(FormalAuthorityLabelRowV1(*row) for row in rows)
    )


def _capture_authority_primitive_snapshot(
    authority: FormalAuditAuthorityV1,
) -> tuple[tuple[_AuthorityPrimitiveRowV1, ...], FormalAuditAuthorityV1]:
    if type(authority) is not FormalAuditAuthorityV1:
        _reject("FAIL_PRIVATE_AUTHORITY", "authority must use the exact private DTO")
    try:
        primitives = tuple(
            (
                row.row_index,
                row.family,
                row.binding,
                row.scale,
                row.answerability,
                row.joint_decision_class,
                row.case_type,
                row.source_case_commitment,
                row.label_binding_id,
                row.row_binding,
            )
            for row in authority.rows
        )
        snapshot = _authority_from_primitive_rows(primitives)
    except (AttributeError, TypeError, ValueError) as exc:
        _reject(
            "FAIL_PRIVATE_AUTHORITY_ROOT",
            f"private authority primitive snapshot failed: {exc}",
        )
    if (
        snapshot.transcript_root != authority.transcript_root
        or tuple(row.canonical_source_label_bytes for row in snapshot.rows)
        != tuple(row.canonical_source_label_bytes for row in authority.rows)
    ):
        _reject(
            "FAIL_PRIVATE_AUTHORITY_ROOT",
            "private authority differs from its primitive/canonical transcript snapshot",
        )
    return primitives, snapshot


@dataclass(frozen=True, slots=True, order=True)
class FormalPrivateUuidMappingV1:
    namespace: str
    source_uuid: str
    public_uuid: str

    def __post_init__(self) -> None:
        if type(self) is not FormalPrivateUuidMappingV1:
            raise TypeError("private UUID mapping must use the exact DTO")
        if self.namespace not in RENAMING_NAMESPACES:
            raise ValueError("private UUID mapping namespace is outside exact-10")
        if _UUID4.fullmatch(self.source_uuid) is None or _UUID4.fullmatch(
            self.public_uuid
        ) is None:
            raise ValueError("private UUID mapping values must be canonical UUIDv4")


@dataclass(frozen=True, slots=True)
class FormalPrivateAlignmentLinkV1:
    public_envelope_sha256: str
    source_row_index: int
    source_case_commitment: str
    source_to_public_uuid_rows: tuple[FormalPrivateUuidMappingV1, ...]
    label_binding_id: str

    def __post_init__(self) -> None:
        if type(self) is not FormalPrivateAlignmentLinkV1:
            raise TypeError("private alignment link must use the exact DTO")
        for value, name in (
            (self.public_envelope_sha256, "public envelope digest"),
            (self.source_case_commitment, "source case commitment"),
            (self.label_binding_id, "label binding ID"),
        ):
            if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError(f"private alignment {name} must be an exact digest")
        if type(self.source_row_index) is not int or not 0 <= self.source_row_index < MAXIMUM_ROWS:
            raise ValueError("private alignment source row index is outside the row cap")
        if type(self.source_to_public_uuid_rows) is not tuple or any(
            type(row) is not FormalPrivateUuidMappingV1
            for row in self.source_to_public_uuid_rows
        ):
            raise TypeError("private alignment UUID mappings must use an exact tuple")
        if self.source_to_public_uuid_rows != tuple(sorted(self.source_to_public_uuid_rows)):
            raise ValueError("private alignment UUID mappings are not canonical")
        if (
            len({(row.namespace, row.source_uuid) for row in self.source_to_public_uuid_rows})
            != len(self.source_to_public_uuid_rows)
            or len({row.public_uuid for row in self.source_to_public_uuid_rows})
            != len(self.source_to_public_uuid_rows)
        ):
            raise ValueError("private alignment UUID mapping is not injective")


@dataclass(frozen=True, slots=True)
class FormalPrivateBatchAlignmentV1:
    """Adapter-owned private links from each public row to one source label."""

    links_in_public_order: tuple[FormalPrivateAlignmentLinkV1, ...]
    wire_issued_alignment_id: str | None
    batch_binding_root: str = field(init=False)
    alignment_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not FormalPrivateBatchAlignmentV1:
            raise TypeError("private batch alignment must use the exact type")
        if (
            type(self.links_in_public_order) is not tuple
            or not 1 <= len(self.links_in_public_order) <= MAXIMUM_ROWS
            or any(
                type(link) is not FormalPrivateAlignmentLinkV1
                for link in self.links_in_public_order
            )
        ):
            raise ValueError("private alignment links are outside the row cap")
        count = len(self.links_in_public_order)
        if tuple(sorted(link.source_row_index for link in self.links_in_public_order)) != tuple(range(count)):
            raise ValueError("private alignment source indices must be an exact bijection")
        if any(
            len({getattr(link, field_name) for link in self.links_in_public_order}) != count
            for field_name in ("source_case_commitment", "label_binding_id")
        ) or len(set(self.links_in_public_order)) != count:
            raise ValueError("private alignment source/label links must be unique")
        if self.wire_issued_alignment_id is not None:
            _exact_nonempty_str(
                self.wire_issued_alignment_id,
                "wire-issued private alignment ID",
            )
        batch_root = stable_hash(
            {
                "public_envelope_sha256_in_order": tuple(
                    link.public_envelope_sha256 for link in self.links_in_public_order
                )
            },
            prefix="phase2b_formal_private_batch_binding_",
        )
        object.__setattr__(self, "batch_binding_root", batch_root)
        object.__setattr__(
            self,
            "alignment_root",
            stable_hash(
                {
                    "schema": SCHEMA_VERSION,
                    "batch_binding_root": batch_root,
                    "links_in_public_order": tuple(
                        {
                            "public_envelope_sha256": link.public_envelope_sha256,
                            "source_row_index": link.source_row_index,
                            "source_case_commitment": link.source_case_commitment,
                            "source_to_public_uuid_rows": tuple(
                                (row.namespace, row.source_uuid, row.public_uuid)
                                for row in link.source_to_public_uuid_rows
                            ),
                            "label_binding_id": link.label_binding_id,
                        }
                        for link in self.links_in_public_order
                    ),
                    "wire_issued_alignment_id": self.wire_issued_alignment_id,
                },
                prefix="phase2b_formal_private_batch_alignment_",
            ),
        )

    @property
    def public_envelope_sha256_in_order(self) -> tuple[str, ...]:
        return tuple(link.public_envelope_sha256 for link in self.links_in_public_order)

    @property
    def source_row_index_by_public_row(self) -> tuple[int, ...]:
        return tuple(link.source_row_index for link in self.links_in_public_order)

    @property
    def source_case_commitments_in_source_order(self) -> tuple[str, ...]:
        return tuple(
            link.source_case_commitment
            for link in sorted(
                self.links_in_public_order,
                key=lambda item: item.source_row_index,
            )
        )


@dataclass(frozen=True, slots=True)
class FormalPrivateAuthorityBindingV1:
    alignment: FormalPrivateBatchAlignmentV1
    public_label_projection_root: str
    alignment_issuance_verified: bool
    label_row_binding_verified: bool

    def __post_init__(self) -> None:
        if type(self) is not FormalPrivateAuthorityBindingV1:
            raise TypeError("private authority binding must use the exact DTO")
        if type(self.alignment) is not FormalPrivateBatchAlignmentV1:
            raise TypeError("private authority binding needs an exact alignment")
        if (
            type(self.public_label_projection_root) is not str
            or not self.public_label_projection_root.startswith(
                "phase2b_formal_private_label_projection_"
            )
        ):
            raise ValueError("private authority public projection root is invalid")
        if type(self.alignment_issuance_verified) is not bool or type(
            self.label_row_binding_verified
        ) is not bool:
            raise TypeError("private authority binding verdicts must be exact booleans")


@dataclass(frozen=True, slots=True)
class _AdapterExecutionPinsV1:
    wire_implementation_id: str
    wire_schema_id: str
    recognizer_implementation_id: str
    private_alignment_root: str
    wire_issued_alignment_id: str | None
    supports_atomic_variant_suite: bool
    stop_binding_id: str | None
    fresh_build_verification_root: str | None
    wire_retained_graph_bytes_bound: int
    recognizer_retained_graph_bytes_bound: int
    variant_suite_resident_bytes_per_row_bound: int
    variant_suite_resident_fixed_bytes_bound: int
    pin_root: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "wire_implementation_id",
            "wire_schema_id",
            "recognizer_implementation_id",
            "private_alignment_root",
        ):
            _exact_nonempty_str(getattr(self, name), f"adapter pin {name}", 8_192)
        for name in (
            "wire_issued_alignment_id",
            "stop_binding_id",
            "fresh_build_verification_root",
        ):
            value = getattr(self, name)
            if value is not None:
                _exact_nonempty_str(value, f"adapter pin {name}", 8_192)
        if type(self.supports_atomic_variant_suite) is not bool:
            raise TypeError("adapter atomic-suite support pin must be exact bool")
        if (
            type(self.wire_retained_graph_bytes_bound) is not int
            or not 0 < self.wire_retained_graph_bytes_bound
            <= MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES
            or type(self.recognizer_retained_graph_bytes_bound) is not int
            or not 0 < self.recognizer_retained_graph_bytes_bound
            <= MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES
            or type(self.variant_suite_resident_bytes_per_row_bound) is not int
            or self.variant_suite_resident_bytes_per_row_bound <= 0
            or type(self.variant_suite_resident_fixed_bytes_bound) is not int
            or self.variant_suite_resident_fixed_bytes_bound <= 0
        ):
            raise ValueError("adapter retained-graph pin exceeds its frozen cap")
        object.__setattr__(
            self,
            "pin_root",
            stable_hash(
                tuple(
                    (row.name, getattr(self, row.name))
                    for row in dataclass_fields(self)
                    if row.name != "pin_root"
                ),
                prefix="phase2b_formal_covert_adapter_execution_pins_",
            ),
        )

    @property
    def retained_graph_bytes_bound(self) -> int:
        return (
            self.wire_retained_graph_bytes_bound
            + self.recognizer_retained_graph_bytes_bound
        )


def _private_label_projection_root(rows: tuple[bytes, ...]) -> str:
    if type(rows) is not tuple or any(type(row) is not bytes for row in rows):
        raise TypeError("private label projection needs exact canonical byte rows")
    return "phase2b_formal_private_label_projection_" + hashlib.sha256(
        _PRIVATE_LABEL_PROJECTION_DOMAIN
        + b"".join(len(row).to_bytes(8, "big") + row for row in rows)
    ).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class FormalUuidRenamingPlanV1:
    namespace: str
    source_uuid: str
    target_uuid: str

    def __post_init__(self) -> None:
        if type(self) is not FormalUuidRenamingPlanV1:
            raise TypeError("variant UUID renaming must use the exact DTO")
        if self.namespace not in RENAMING_NAMESPACES:
            raise ValueError("variant UUID renaming namespace is outside exact-10")
        if _UUID4.fullmatch(self.source_uuid) is None or _UUID4.fullmatch(
            self.target_uuid
        ) is None:
            raise ValueError("variant UUID renaming values must be canonical UUIDv4")


@dataclass(frozen=True, slots=True)
class FormalObservationOrderPlanV1:
    canonical_row_index: int
    path: JsonPath
    order_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self) is not FormalObservationOrderPlanV1:
            raise TypeError("observation-order plan must use the exact DTO")
        if type(self.canonical_row_index) is not int or self.canonical_row_index < 0:
            raise ValueError("observation-order row index must be nonnegative")
        if type(self.path) is not tuple or not self.path:
            raise ValueError("observation-order plan needs an exact nonempty path")
        if type(self.order_indices) is not tuple or not self.order_indices or any(
            type(value) is not int for value in self.order_indices
        ):
            raise TypeError("observation order must use exact integer indices")
        if tuple(sorted(self.order_indices)) != tuple(range(len(self.order_indices))):
            raise ValueError("observation order is not an exact permutation")


@dataclass(frozen=True, slots=True)
class FormalVariantPlanV1:
    case_order_indices: tuple[int, ...]
    uuid_renamings: tuple[FormalUuidRenamingPlanV1, ...]
    observation_orders: tuple[FormalObservationOrderPlanV1, ...]

    def __post_init__(self) -> None:
        if type(self) is not FormalVariantPlanV1:
            raise TypeError("formal variant plan must use the exact DTO")
        if type(self.case_order_indices) is not tuple or not self.case_order_indices or any(
            type(value) is not int for value in self.case_order_indices
        ):
            raise TypeError("variant case order must use exact integer indices")
        if tuple(sorted(self.case_order_indices)) != tuple(
            range(len(self.case_order_indices))
        ):
            raise ValueError("variant case order is not an exact permutation")
        if type(self.uuid_renamings) is not tuple or any(
            type(row) is not FormalUuidRenamingPlanV1 for row in self.uuid_renamings
        ):
            raise TypeError("variant UUID renamings must use exact immutable rows")
        if self.uuid_renamings != tuple(sorted(self.uuid_renamings)):
            raise ValueError("variant UUID renamings are not canonical")
        if type(self.observation_orders) is not tuple or any(
            type(row) is not FormalObservationOrderPlanV1
            for row in self.observation_orders
        ):
            raise TypeError("variant observation orders must use exact immutable rows")
        if self.observation_orders != tuple(
            sorted(
                self.observation_orders,
                key=lambda row: (row.canonical_row_index, _path_text(row.path)),
            )
        ):
            raise ValueError("variant observation orders are not canonical")


@dataclass(frozen=True, slots=True)
class _FormalVariantSuiteExecutionV1:
    batches: tuple[tuple[bytes, ...], ...]
    suite_root: str
    consumer_decode_count: int
    unique_envelope_object_count: int
    resident_object_graph_bytes_observed: int

    def __post_init__(self) -> None:
        if (
            type(self.batches) is not tuple
            or len(self.batches)
            != GLOBAL_RENAMINGS
            + GLOBAL_CASE_ORDER_PERMUTATIONS
            + OBSERVATION_ORDER_PERMUTATIONS
            or any(
                type(batch) is not tuple
                or not batch
                or len(batch) > MAXIMUM_ROWS
                or any(
                    type(envelope) is not bytes or len(envelope) != ENVELOPE_BYTES
                    for envelope in batch
                )
                for batch in self.batches
            )
        ):
            raise ValueError("atomic variant suite does not exactly cover 80 fixed batches")
        if (
            type(self.suite_root) is not str
            or not self.suite_root.startswith("phase2b_formal_variant_suite_v3_")
        ):
            raise ValueError("atomic variant suite root drift")
        if (
            type(self.consumer_decode_count) is not int
            or self.consumer_decode_count
            != sum(len(batch) for batch in self.batches)
        ):
            raise ValueError("atomic variant suite consumer decode coverage drift")
        unique_count = len(
            {id(envelope) for batch in self.batches for envelope in batch}
        )
        if (
            type(self.unique_envelope_object_count) is not int
            or self.unique_envelope_object_count != unique_count
            or type(self.resident_object_graph_bytes_observed) is not int
            or self.resident_object_graph_bytes_observed
            != _deep_object_graph_bytes(self.batches)
        ):
            raise ValueError("atomic variant suite resident-object evidence drift")


@dataclass(frozen=True, slots=True)
class _InvariantExecutionResultV1:
    fork_input_snapshot_root: str
    adapter_execution_pin_root: str
    invariant_execution_root: str
    exact_invariant_comparisons: int
    invariant_run_count: int
    variant_suite_root: str
    consumer_decode_count: int
    variant_suite_unique_envelope_object_count: int
    variant_suite_resident_object_graph_bytes_observed: int
    recognizer_output_root: str

    def __post_init__(self) -> None:
        if (
            type(self.fork_input_snapshot_root) is not str
            or not self.fork_input_snapshot_root.startswith(
                "phase2b_formal_covert_invariance_input_snapshot_"
            )
            or type(self.adapter_execution_pin_root) is not str
            or not self.adapter_execution_pin_root.startswith(
                "phase2b_formal_covert_adapter_execution_pins_"
            )
            or type(self.invariant_execution_root) is not str
            or not self.invariant_execution_root.startswith(
                "phase2b_formal_covert_invariant_execution_"
            )
            or type(self.variant_suite_root) is not str
            or not self.variant_suite_root.startswith(
                "phase2b_formal_variant_suite_"
            )
            or type(self.recognizer_output_root) is not str
            or not self.recognizer_output_root.startswith(
                "phase2b_formal_covert_recognizer_output_"
            )
        ):
            raise ValueError("invariant execution result root drift")
        for name in (
            "exact_invariant_comparisons",
            "invariant_run_count",
            "consumer_decode_count",
            "variant_suite_unique_envelope_object_count",
            "variant_suite_resident_object_graph_bytes_observed",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("invariant execution counts must be exact nonnegative integers")
        if self.invariant_run_count != (
            1
            + GLOBAL_RENAMINGS
            + GLOBAL_CASE_ORDER_PERMUTATIONS
            + OBSERVATION_ORDER_PERMUTATIONS
        ):
            raise ValueError("invariant recognizer run count drift")
        if self.consumer_decode_count % (
            GLOBAL_RENAMINGS
            + GLOBAL_CASE_ORDER_PERMUTATIONS
            + OBSERVATION_ORDER_PERMUTATIONS
        ):
            raise ValueError("invariant consumer-decode count is not batch-aligned")
        if (
            self.variant_suite_unique_envelope_object_count
            > self.consumer_decode_count
            or self.variant_suite_resident_object_graph_bytes_observed < 0
        ):
            raise ValueError("invariant variant-suite resident evidence drift")
        if self.variant_suite_root.startswith("phase2b_formal_variant_suite_v3_") and (
            self.variant_suite_unique_envelope_object_count <= 0
            or self.variant_suite_resident_object_graph_bytes_observed <= 0
            or self.variant_suite_unique_envelope_object_count
            > (self.consumer_decode_count // 80)
            * WIRE_V3_ATOMIC_SUITE_UNIQUE_BATCH_EQUIVALENTS
        ):
            raise ValueError("Wire V3 suite reuse/resident evidence drift")
        rows = self.consumer_decode_count // (
            GLOBAL_RENAMINGS
            + GLOBAL_CASE_ORDER_PERMUTATIONS
            + OBSERVATION_ORDER_PERMUTATIONS
        )
        if self.exact_invariant_comparisons != rows * 80 * 5:
            raise ValueError("invariant comparison/decode coverage drift")


def _validate_variant_plan(
    plan: FormalVariantPlanV1,
    envelopes: tuple[bytes, ...],
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
) -> None:
    if type(plan) is not FormalVariantPlanV1:
        _reject("FAIL_VARIANT_PLAN", "variant plan must use the exact DTO")
    try:
        plan.__post_init__()
    except (TypeError, ValueError) as exc:
        _reject("FAIL_VARIANT_PLAN", f"variant plan replay failed: {exc}")
    if len(plan.case_order_indices) != len(envelopes) or len(decoded) != len(envelopes):
        _reject("FAIL_VARIANT_PLAN", "variant plan does not cover the exact batch")
    source_domain = {
        (leaf.namespace, leaf.public_uuid)
        for item in decoded
        for leaf in item.namespace_leaves
    }
    if plan.uuid_renamings:
        observed = {(row.namespace, row.source_uuid) for row in plan.uuid_renamings}
        if observed != source_domain:
            _reject("FAIL_VARIANT_PLAN", "UUID renaming does not cover the exact batch domain")
        targets = tuple(row.target_uuid for row in plan.uuid_renamings)
        if len(set(targets)) != len(targets):
            _reject("FAIL_VARIANT_PLAN", "UUID renaming targets are not globally injective")
    seen_orders: set[tuple[int, JsonPath]] = set()
    for row in plan.observation_orders:
        if not 0 <= row.canonical_row_index < len(decoded):
            _reject("FAIL_VARIANT_PLAN", "observation-order row is outside the batch")
        key = (row.canonical_row_index, row.path)
        if key in seen_orders:
            _reject("FAIL_VARIANT_PLAN", "observation-order plan repeats a row/path")
        seen_orders.add(key)
        item = decoded[row.canonical_row_index]
        if row.path not in item.observation_array_paths:
            _reject("FAIL_VARIANT_PLAN", "observation-order path is not wire-declared")
        array = _resolve(item.payload, row.path)
        if len(row.order_indices) != len(array):
            _reject("FAIL_VARIANT_PLAN", "observation-order length changed")


def _json_pointer_to_path(root: object, pointer: str) -> JsonPath:
    if type(pointer) is not str or not pointer.startswith("/") or pointer == "/":
        raise ValueError("formal JSON pointer must be an absolute non-root pointer")
    current = root
    result: list[JsonPathToken] = []
    for encoded in pointer[1:].split("/"):
        token_text = encoded.replace("~1", "/").replace("~0", "~")
        if type(current) is list:
            if not token_text.isdigit():
                raise ValueError("formal JSON pointer has a nonnumeric array token")
            token: JsonPathToken = int(token_text)
        else:
            token = token_text
        result.append(token)
        current = _resolve(root, tuple(result))
    return tuple(result)


def _wire_v3_full_pointer(pointer: str) -> str:
    if pointer.startswith("/public_case/graph/"):
        return pointer
    if pointer.startswith("/"):
        return "/public_case/graph" + pointer
    raise ValueError("wire V3 namespace pointer is not absolute")


def _wire_v3_channel_owner(
    pointer: str,
    value: object,
    namespace_pointers: set[str],
) -> str:
    last = pointer.rsplit("/", 1)[-1]
    if "/eligibility_mask/" in pointer:
        return "eligibility_mask"
    if last == "public_provenance_commitment":
        return "provenance"
    if last == "available" or value is None:
        return "missingness"
    if last == "component_index":
        return "slot_order"
    if pointer in namespace_pointers:
        if re.search(
            r"/public_case/graph/(?:entities|quantities|contexts|observations|scales|aggregate_maps|transforms)/[0-9]+/"
            r"(?:entity_id|quantity_id|context_id|observation_id|scale_id|aggregate_map_id|transform_id)$",
            pointer,
        ):
            return "record_order"
        return "uuid_leaf"
    if not pointer.startswith("/public_case/graph/") or last in {
        "field_classification_manifest_id",
        "graph_completeness_policy_id",
        "id_collision_retry_count_decimal",
        "max_candidate_count",
        "run_id_commitment",
        "schema_version",
    }:
        return "control_metadata"
    raise ValueError(
        "wire V3 CHANNEL_ONLY leaf has no frozen auditor owner: " + pointer
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_wire_v3_stop_surface(wire_v3: object) -> str:
    """Bind the adapter to the reviewed Wire/Core/RFC bytes and identifiers."""

    from . import phase2b_exact_transform_core_v1 as exact_core
    from . import rfc8785_v1

    wire_path = Path(wire_v3.__file__).resolve()  # type: ignore[attr-defined]
    project = wire_path.parents[2]
    paths = {
        "wire_source": wire_path,
        "wire_test": project / "tests" / "test_phase2b_formal_trusted_wire_v3.py",
        "wire_config": project / "config" / "phase2b_formal_trusted_wire_v3.json",
        "wire_doc": project / "docs" / "Hegel_Machine_Phase2B_Formal_Trusted_Wire_V3.md",
        "core_source": Path(exact_core.__file__).resolve(),
        "core_test": project / "tests" / "test_phase2b_exact_transform_core_v1.py",
        "rfc8785_source": Path(rfc8785_v1.__file__).resolve(),
    }
    expected_hashes = {
        "wire_source": WIRE_V3_SOURCE_SHA256,
        "wire_test": WIRE_V3_TEST_SHA256,
        "wire_config": WIRE_V3_CONFIG_SHA256,
        "wire_doc": WIRE_V3_DOC_SHA256,
        "core_source": EXACT_TRANSFORM_CORE_SOURCE_SHA256,
        "core_test": EXACT_TRANSFORM_CORE_TEST_SHA256,
        "rfc8785_source": RFC8785_SOURCE_SHA256,
    }
    try:
        observed_hashes = {name: _sha256_path(path) for name, path in paths.items()}
    except OSError as exc:
        raise RuntimeError(f"formal Wire V3 STOP material is unavailable: {exc}") from exc
    if observed_hashes != expected_hashes:
        drift = tuple(
            name
            for name in sorted(expected_hashes)
            if observed_hashes.get(name) != expected_hashes[name]
        )
        raise RuntimeError(
            "formal Wire V3 STOP byte binding drift: " + ",".join(drift)
        )
    observed_surface = (
        wire_v3.FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,  # type: ignore[attr-defined]
        wire_v3.FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,  # type: ignore[attr-defined]
        wire_v3.FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION,  # type: ignore[attr-defined]
        wire_v3.FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION,  # type: ignore[attr-defined]
        wire_v3.FIELD_CLASSIFICATION_MANIFEST_ID_V3,  # type: ignore[attr-defined]
        wire_v3.FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3,  # type: ignore[attr-defined]
        wire_v3.EXACT_TRANSFORM_CORE_SOURCE_SHA256_V3,  # type: ignore[attr-defined]
        exact_core.EXACT_TRANSFORM_CORE_VERSION_V1,
        wire_v3.UNICODE_DATA_VERSION_V3,  # type: ignore[attr-defined]
        wire_v3.UCD13_DERIVED_CORE_PROPERTIES_SHA256_V3,  # type: ignore[attr-defined]
        wire_v3.UCD13_PROP_LIST_SHA256_V3,  # type: ignore[attr-defined]
        wire_v3.DEFAULT_IGNORABLE_CODEPOINT_COUNT_V3,  # type: ignore[attr-defined]
        wire_v3.DEFAULT_IGNORABLE_RANGE_TABLE_SHA256_V3,  # type: ignore[attr-defined]
    )
    expected_surface = (
        "hegel-machine-phase2b-formal-trusted-wire/3",
        "hegel-machine-phase2b-formal-trusted-wire-payload/3",
        "hegel-machine-phase2b-formal-case-graph/3",
        "hegel-machine-phase2b-formal-public-case/3",
        WIRE_V3_FIELD_CLASSIFICATION_MANIFEST_ID,
        WIRE_V3_GRAPH_COMPLETENESS_POLICY_ID,
        "sha256:" + EXACT_TRANSFORM_CORE_SOURCE_SHA256,
        "hegel-machine-phase2b-pure-exact-transform-core/1",
        DESCRIPTOR_UNICODE_DATA_VERSION,
        DESCRIPTOR_UCD13_DERIVED_CORE_PROPERTIES_SHA256,
        DESCRIPTOR_UCD13_PROP_LIST_SHA256,
        DESCRIPTOR_DEFAULT_IGNORABLE_CODEPOINT_COUNT,
        DESCRIPTOR_DEFAULT_IGNORABLE_RANGE_TABLE_SHA256,
    )
    if observed_surface != expected_surface:
        raise RuntimeError("formal Wire V3 policy/manifest/dependency surface drift")
    return stable_hash(
        {
            "hashes": expected_hashes,
            "surface": expected_surface,
        },
        prefix="phase2b_formal_wire_v3_stop_binding_",
    )


class FormalWireV3AdapterV1(FormalWireAdapterV1):
    """Batch-bound adapter for the rich exact-10 trusted Wire V3.

    Construction accepts only a wire-owned build-success wrapper.  The public
    batch never exposes its private alignment or private variant context; both
    remain inside the build-success-bound adapter.  Every Wire re-encoding call
    accepts the full build success rather than a standalone context object.
    """

    __slots__ = (
        "_build_success",
        "_batch",
        "_wire_alignment",
        "_variant_context",
        "_envelopes",
        "_private_alignment",
        "_stop_binding_id",
        "_fresh_build_verification_root",
        "_implementation_id",
        "_schema_id",
        "_retained_graph_bytes_bound",
    )

    def __init__(
        self,
        build_success: object,
        *,
        run_id: bytes,
        key_sources: object,
        source_cases: tuple[object, ...],
        global_transform_catalog: object,
        transform_catalog_binding_receipt: object,
    ) -> None:
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        stop_binding_id = _validate_wire_v3_stop_surface(wire_v3)
        if (
            tuple(wire_v3.NAMESPACE_REGISTRY_V3) != RENAMING_NAMESPACES
            or tuple(wire_v3.PUBLIC_PAYLOAD_MUST_BE_ABSENT_FIELDS_V3)
            != MUST_BE_ABSENT_FIELDS
            or wire_v3.ENVELOPE_BYTES_V3 != ENVELOPE_BYTES
            or tuple(wire_v3.PUBLIC_OBSERVATION_ARRAY_PATHS_V3)
            != ("/public_case/graph/observations",)
        ):
            raise RuntimeError("formal Wire V3 surface drift from the auditor closure")
        if type(build_success) is not wire_v3.FormalTrustedWireBuildSuccessV3:
            raise TypeError(
                "FormalWireV3AdapterV1 needs an exact trusted Wire V3 build success"
            )
        build_success._validate_issued_shape()
        batch = build_success.public_batch
        if batch.disposition is not wire_v3.FormalTrustedWireDispositionV3.COMPLETE:
            raise ValueError("formal Wire V3 batch is not complete")
        if any(
            value is not False
            for value in (
                batch.key_source_transcript_verified,
                batch.key_source_independence_verified,
                batch.origin_authenticated,
                batch.formal_covert_audit,
                batch.capacity_1024_qualified,
                batch.actual_execution_claimed,
                build_success.origin_authenticated,
            )
        ):
            raise ValueError("formal Wire V3 public/build claim boundary drift")
        try:
            fresh_verified = wire_v3.verify_formal_trusted_wire_build_success_v3(
                build_success=build_success,
                run_id=run_id,
                key_sources=key_sources,
                source_cases=source_cases,
                global_transform_catalog=global_transform_catalog,
                transform_catalog_binding_receipt=(
                    transform_catalog_binding_receipt
                ),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"formal Wire V3 external fresh build verification failed: {exc}"
            ) from exc
        if fresh_verified is not True:
            raise ValueError(
                "formal Wire V3 external fresh build verification did not return exact true"
            )
        wire_alignment = build_success.private_audit_alignment
        variant_context = build_success.variant_context
        if type(wire_alignment) is not wire_v3.FormalPrivateAuditAlignmentV3:
            raise TypeError("formal Wire V3 build lacks its exact private alignment")
        if not wire_v3.verify_private_audit_alignment_v3(
            batch=batch,
            alignment=wire_alignment,
        ):
            raise ValueError("formal Wire V3 private alignment did not verify")
        variant_context._validate_issued_shape()
        if (
            variant_context.batch_id != batch.batch_id
            or variant_context.private_alignment_id != wire_alignment.alignment_id
            or variant_context.transform_catalog_root
            != batch.batch_context.transform_catalog_root
            or variant_context.origin_authenticated is not False
        ):
            raise ValueError("formal Wire V3 variant context is stale from batch/alignment")
        envelopes = tuple(item.envelope for item in batch.envelopes)
        private_alignment = FormalPrivateBatchAlignmentV1(
            tuple(
                FormalPrivateAlignmentLinkV1(
                    public_envelope_sha256=link.public_envelope_sha256,
                    source_row_index=link.source_row_index,
                    source_case_commitment=link.source_case_commitment,
                    source_to_public_uuid_rows=tuple(
                        FormalPrivateUuidMappingV1(
                            row.namespace,
                            row.source_uuid,
                            row.public_uuid,
                        )
                        for row in link.source_to_public_uuid_rows
                    ),
                    label_binding_id=link.label_binding_id,
                )
                for link in wire_alignment.links_in_public_order
            ),
            wire_alignment.alignment_id,
        )
        object.__setattr__(self, "_build_success", build_success)
        object.__setattr__(self, "_batch", batch)
        object.__setattr__(self, "_wire_alignment", wire_alignment)
        object.__setattr__(self, "_variant_context", variant_context)
        object.__setattr__(self, "_envelopes", envelopes)
        object.__setattr__(self, "_private_alignment", private_alignment)
        object.__setattr__(self, "_stop_binding_id", stop_binding_id)
        object.__setattr__(
            self,
            "_fresh_build_verification_root",
            stable_hash(
                {
                    "success_id": build_success.success_id,
                    "batch_id": batch.batch_id,
                    "key_source_receipt_id": build_success.key_source_receipt.receipt_id,
                    "private_alignment_id": wire_alignment.alignment_id,
                    "variant_context_id": variant_context.context_id,
                    "run_id_commitment": batch.batch_context.run_id_commitment,
                    "transform_catalog_root": batch.batch_context.transform_catalog_root,
                    "source_case_commitments_in_source_order": (
                        wire_alignment.source_case_commitments_in_source_order
                    ),
                    "stop_binding_id": stop_binding_id,
                },
                prefix="phase2b_formal_wire_v3_fresh_build_verification_",
            ),
        )
        object.__setattr__(
            self,
            "_implementation_id",
            stable_hash(
                {
                    "adapter": "hegel-machine-phase2b-formal-wire-v3-adapter/1",
                    "wire_schema": wire_v3.FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,
                    "payload_schema": wire_v3.FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,
                    "field_manifest": wire_v3.FIELD_CLASSIFICATION_MANIFEST_ID_V3,
                    "graph_policy": wire_v3.FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3,
                    "stop_binding_id": stop_binding_id,
                    "fresh_external_build_verification_required": True,
                },
                prefix="phase2b_formal_wire_v3_adapter_",
            ),
        )
        object.__setattr__(
            self,
            "_schema_id",
            wire_v3.FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,
        )
        retained_graph_bytes_bound = max(
            INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND,
            DEEP_SIZE_SAFETY_MULTIPLIER
            * _deep_object_graph_bytes(
                (
                    build_success,
                    batch,
                    wire_alignment,
                    variant_context,
                    envelopes,
                    private_alignment,
                )
            ),
        )
        if retained_graph_bytes_bound > MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES:
            raise MemoryError("formal Wire V3 retained graph exceeds the frozen cap")
        object.__setattr__(
            self,
            "_retained_graph_bytes_bound",
            retained_graph_bytes_bound,
        )

    @property
    def implementation_id(self) -> str:
        return self._implementation_id

    @property
    def schema_id(self) -> str:
        return self._schema_id

    @property
    def retained_graph_bytes_bound(self) -> int:
        return self._retained_graph_bytes_bound

    @property
    def variant_suite_resident_bytes_per_row_bound(self) -> int:
        return (
            WIRE_V3_ATOMIC_SUITE_UNIQUE_BATCH_EQUIVALENTS * ENVELOPE_BYTES
            + 80 * ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE
        )

    @property
    def variant_suite_resident_fixed_bytes_bound(self) -> int:
        return ATOMIC_VARIANT_SUITE_FIXED_BYTES_BOUND

    @property
    def envelopes(self) -> tuple[bytes, ...]:
        return self._envelopes

    @property
    def stop_binding_id(self) -> str:
        return self._stop_binding_id

    @property
    def fresh_build_verification_root(self) -> str:
        return self._fresh_build_verification_root

    @property
    def private_alignment(self) -> FormalPrivateBatchAlignmentV1:
        return self._private_alignment

    def canonical_payload_bytes(self, payload: object) -> bytes:
        from .rfc8785_v1 import canonicalize_rfc8785_v1

        return canonicalize_rfc8785_v1(payload)

    def bind_private_authority(
        self,
        authority: FormalAuditAuthorityV1,
        envelopes: tuple[bytes, ...],
    ) -> FormalPrivateAuthorityBindingV1:
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        try:
            self._build_success._validate_issued_shape()
            if envelopes != self._envelopes:
                raise ValueError("private authority was presented against another batch")
            if (
                self._private_alignment.wire_issued_alignment_id
                != self._wire_alignment.alignment_id
                or len(self._private_alignment.links_in_public_order)
                != len(self._wire_alignment.links_in_public_order)
            ):
                raise ValueError("auditor private alignment projection is stale")
            for projected_link, wire_link in zip(
                self._private_alignment.links_in_public_order,
                self._wire_alignment.links_in_public_order,
                strict=True,
            ):
                if (
                    projected_link.public_envelope_sha256
                    != wire_link.public_envelope_sha256
                    or projected_link.source_row_index != wire_link.source_row_index
                    or projected_link.source_case_commitment
                    != wire_link.source_case_commitment
                    or projected_link.label_binding_id != wire_link.label_binding_id
                    or tuple(
                        (row.namespace, row.source_uuid, row.public_uuid)
                        for row in projected_link.source_to_public_uuid_rows
                    )
                    != tuple(
                        (row.namespace, row.source_uuid, row.public_uuid)
                        for row in wire_link.source_to_public_uuid_rows
                    )
                ):
                    raise ValueError("auditor private alignment link/mapping is stale")
            _validate_authority_links(envelopes, authority, self._private_alignment)
            source_bytes = tuple(
                row.canonical_source_label_bytes for row in authority.rows
            )
            projected = wire_v3.project_private_source_labels_to_public_order_v3(
                build_success=self._build_success,
                canonical_source_label_bytes_in_source_order=source_bytes,
            )
            issued_public = wire_v3.issue_projected_private_audit_rows_v3(
                build_success=self._build_success,
                canonical_source_label_bytes_in_source_order=source_bytes,
            )
            if len(projected) != len(issued_public):
                raise ValueError("private authority projected row coverage drift")
            links = self._wire_alignment.links_in_public_order
            for public_index, (payload, issued, link) in enumerate(
                zip(projected, issued_public, links, strict=True)
            ):
                issued._validate_issued_shape()
                source = authority.rows[link.source_row_index]
                if (
                    issued.private_payload_bytes != payload
                    or issued.source_case_commitment != link.source_case_commitment
                    or issued.label_binding_id != link.label_binding_id
                    or source.source_case_commitment != issued.source_case_commitment
                    or source.label_binding_id != issued.label_binding_id
                    or source.row_binding != issued.row_binding
                    or issued.semantic_truth_verified is not False
                    or issued.origin_authenticated is not False
                ):
                    raise ValueError(
                        f"private authority source/link/content binding drift at public row {public_index}"
                    )
            if not wire_v3.verify_private_audit_alignment_v3(
                batch=self._batch,
                alignment=self._wire_alignment,
            ):
                raise ValueError("private alignment replay returned false")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _reject(
                "FAIL_PRIVATE_AUTHORITY_WIRE_BINDING",
                f"Wire V3 private label/link verification failed: {exc}",
            )
        return FormalPrivateAuthorityBindingV1(
            alignment=self._private_alignment,
            public_label_projection_root=_private_label_projection_root(projected),
            alignment_issuance_verified=True,
            label_row_binding_verified=True,
        )

    def make_variant_envelopes(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple[DecodedFormalEnvelopeV1, ...],
        plan: FormalVariantPlanV1,
    ) -> tuple[bytes, ...]:
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        try:
            schedule = self._wire_v3_schedule(
                envelopes=envelopes,
                decoded=decoded,
                plan=plan,
            )
            result = wire_v3.reencode_formal_variant_batch_v3(
                build_success=self._build_success,
                schedule=schedule,
            )
            if type(result) is not tuple or len(result) != len(envelopes):
                raise ValueError("Wire V3 variant batch did not exactly cover the input")
            for envelope in result:
                replay = self.decode_envelope(envelope)
                _validate_decoded(replay, self)
            return result
        except FormalCovertAuditRejectionV1:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _reject("FAIL_VARIANT_CAPABILITY", f"Wire V3 variant issuance failed: {exc}")

    @property
    def supports_atomic_variant_suite(self) -> bool:
        return True

    def _wire_v3_schedule(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple[DecodedFormalEnvelopeV1, ...],
        plan: FormalVariantPlanV1,
    ) -> object:
        self._validate_wire_v3_variant_bound_state(
            envelopes=envelopes,
            decoded=decoded,
        )
        return self._wire_v3_schedule_from_validated_bound_state(
            envelopes=envelopes,
            decoded=decoded,
            plan=plan,
        )

    def _validate_wire_v3_variant_bound_state(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple[DecodedFormalEnvelopeV1, ...],
    ) -> None:
        """Replay the immutable full-build binding once per issuance call.

        The single-batch surface invokes this for every call.  The atomic
        80-schedule surface invokes it once before constructing any schedule;
        its per-plan validator below remains mandatory for all 80 plans.
        """
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        if type(envelopes) is not tuple or type(decoded) is not tuple:
            raise TypeError("variant bound state needs exact immutable tuples")
        if envelopes != self._envelopes:
            raise ValueError("variant input is outside the bound batch")
        if len(decoded) != len(envelopes):
            raise ValueError("variant decoded input does not cover the bound batch")
        self._build_success._validate_issued_shape()
        if (
            self._build_success.public_batch is not self._batch
            or self._build_success.private_audit_alignment
            is not self._wire_alignment
            or self._build_success.variant_context is not self._variant_context
            or tuple(item.envelope for item in self._batch.envelopes)
            != self._envelopes
            or self._variant_context.private_alignment_id
            != self._wire_alignment.alignment_id
            or self._variant_context.batch_id != self._batch.batch_id
            or self._variant_context.transform_catalog_root
            != self._batch.batch_context.transform_catalog_root
            or type(self._wire_alignment)
            is not wire_v3.FormalPrivateAuditAlignmentV3
        ):
            raise ValueError("variant build/alignment/context/envelope binding drift")

    def _wire_v3_schedule_from_validated_bound_state(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple[DecodedFormalEnvelopeV1, ...],
        plan: FormalVariantPlanV1,
    ) -> object:
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        _validate_variant_plan(plan, envelopes, decoded)
        observation_by_row: dict[int, FormalObservationOrderPlanV1] = {}
        for row in plan.observation_orders:
            if row.canonical_row_index in observation_by_row:
                raise ValueError("Wire V3 permits one declared observation array per row")
            observation_by_row[row.canonical_row_index] = row
        observation_rows = []
        for row_index, (trusted, item) in enumerate(
            zip(self._batch.envelopes, decoded, strict=True)
        ):
            paths = item.observation_array_paths
            if len(paths) != 1:
                raise ValueError("Wire V3 observation array closure drift")
            array = _resolve(item.payload, paths[0])
            if type(array) is not list:
                raise ValueError("Wire V3 observation array is not a list")
            order = observation_by_row.get(row_index)
            indices = tuple(range(len(array))) if order is None else order.order_indices
            observation_ids = tuple(
                array[index]["observation_id"] for index in indices
            )
            observation_rows.append(
                wire_v3.FormalObservationOrderV3(
                    trusted.envelope_id,
                    observation_ids,
                )
            )
        schedule = wire_v3.FormalVariantScheduleV3(
            case_order_envelope_ids=tuple(
                self._batch.envelopes[index].envelope_id
                for index in plan.case_order_indices
            ),
            uuid_renamings=tuple(
                wire_v3.FormalUuidRenamingV3(
                    row.namespace,
                    row.source_uuid,
                    row.target_uuid,
                )
                for row in plan.uuid_renamings
            ),
            observation_orders=tuple(
                sorted(observation_rows, key=lambda row: row.envelope_id)
            ),
        )
        schedule.__post_init__()
        return schedule

    def make_variant_suite_envelopes(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple[DecodedFormalEnvelopeV1, ...],
        plans: tuple[FormalVariantPlanV1, ...],
    ) -> _FormalVariantSuiteExecutionV1:
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        if (
            type(plans) is not tuple
            or len(plans)
            != GLOBAL_RENAMINGS
            + GLOBAL_CASE_ORDER_PERMUTATIONS
            + OBSERVATION_ORDER_PERMUTATIONS
            or any(type(plan) is not FormalVariantPlanV1 for plan in plans)
        ):
            _reject(
                "FAIL_VARIANT_SUITE_COVERAGE",
                "Wire V3 atomic suite requires the exact frozen 80 plans",
            )
        try:
            self._validate_wire_v3_variant_bound_state(
                envelopes=envelopes,
                decoded=decoded,
            )
            schedules = tuple(
                self._wire_v3_schedule_from_validated_bound_state(
                    envelopes=envelopes,
                    decoded=decoded,
                    plan=plan,
                )
                for plan in plans
            )
            suite = wire_v3.reencode_formal_variant_suite_v3(
                build_success=self._build_success,
                schedules=schedules,
            )
            if type(suite) is not wire_v3.FormalVariantSuiteV3:
                raise TypeError("Wire V3 returned a non-exact variant suite")
            suite.__post_init__()
            if (
                suite.source_batch_id != self._batch.batch_id
                or suite.build_success_id != self._build_success.success_id
                or len(suite.batches) != len(plans)
                or any(
                    value is not False
                    for value in (
                        suite.key_source_transcript_verified,
                        suite.catalog_membership_verified,
                        suite.origin_authenticated,
                        suite.formal_covert_audit,
                        suite.actual_execution_claimed,
                    )
                )
            ):
                raise ValueError("Wire V3 variant suite binding/claim boundary drift")
            batches: list[tuple[bytes, ...]] = []
            for plan, schedule, batch in zip(
                plans, schedules, suite.batches, strict=True
            ):
                batch.__post_init__()
                expected_base_ids = tuple(
                    self._batch.envelopes[index].envelope_id
                    for index in plan.case_order_indices
                )
                if (
                    batch.schedule != schedule
                    or batch.base_envelope_ids_in_order != expected_base_ids
                    or len(batch.envelopes) != len(envelopes)
                ):
                    raise ValueError("Wire V3 suite batch schedule/order coverage drift")
                for envelope in batch.envelopes:
                    replay = self.decode_envelope(envelope)
                    _validate_decoded(replay, self)
                batches.append(batch.envelopes)
            frozen_batches = tuple(batches)
            resident_observed = (
                _deep_object_graph_bytes(frozen_batches)
            )
            resident_bound = (
                self.variant_suite_resident_fixed_bytes_bound
                + len(envelopes)
                * self.variant_suite_resident_bytes_per_row_bound
            )
            if resident_observed > resident_bound:
                raise MemoryError("Wire V3 suite resident graph exceeds its pinned bound")
            return _FormalVariantSuiteExecutionV1(
                frozen_batches,
                suite.suite_root,
                sum(len(batch) for batch in batches),
                len(
                    {
                        id(envelope)
                        for batch in frozen_batches
                        for envelope in batch
                    }
                ),
                resident_observed,
            )
        except FormalCovertAuditRejectionV1:
            raise
        except (AttributeError, KeyError, MemoryError, TypeError, ValueError) as exc:
            _reject(
                "FAIL_VARIANT_SUITE_CAPABILITY",
                f"Wire V3 atomic variant suite issuance failed: {exc}",
            )

    def decode_envelope(self, envelope: bytes) -> DecodedFormalEnvelopeV1:
        from . import phase2b_formal_trusted_wire_v3 as wire_v3

        decoded = wire_v3.decode_and_audit_formal_envelope_v3(envelope)
        payload = decoded.payload
        namespace_rows: list[NamespaceLeafV1] = []
        namespace_pointer_texts: set[str] = set()
        for item in decoded.channel_projection.namespace_uuid_channels:
            pointer = _wire_v3_full_pointer(item.json_pointer)
            namespace_pointer_texts.add(pointer)
            namespace_rows.append(
                NamespaceLeafV1(
                    namespace=item.namespace,
                    path=_json_pointer_to_path(payload, pointer),
                    public_uuid=item.public_uuid,
                )
            )
        namespace_rows.sort(
            key=lambda item: (item.namespace, _path_text(item.path), item.public_uuid)
        )

        channel_rows: dict[tuple[str, JsonPath], ChannelLeafV1] = {}

        def add_channel(
            group: str,
            path: JsonPath,
            *,
            feature_projection: object | None = None,
        ) -> None:
            value = _resolve(payload, path)
            channel_rows[(group, path)] = ChannelLeafV1(
                group=group,
                path=path,
                canonical_value=stable_hash(
                    value,
                    prefix="phase2b_channel_leaf_",
                ),
                feature_value=stable_hash(
                    value if feature_projection is None else feature_projection,
                    prefix="phase2b_channel_feature_",
                ),
            )

        owner_rows: list[ChannelOnlyLeafOwnerV1] = []
        declared_channel_pointers = {
            item.json_pointer
            for item in decoded.classified_leaves
            if item.classification is wire_v3.PublicLeafClassV3.CHANNEL_ONLY
        }
        for item in decoded.classified_leaves:
            if item.classification is not wire_v3.PublicLeafClassV3.CHANNEL_ONLY:
                continue
            path = _json_pointer_to_path(payload, item.json_pointer)
            owner = _wire_v3_channel_owner(
                item.json_pointer,
                item.value,
                namespace_pointer_texts,
            )
            owner_rows.append(ChannelOnlyLeafOwnerV1(path, owner))
            if item.json_pointer not in namespace_pointer_texts:
                add_channel(owner, path)
        if len(owner_rows) != len(declared_channel_pointers):
            raise ValueError("wire V3 channel-only partition is not one-to-one")

        graph_path: JsonPath = ("public_case", "graph")
        record_identity_fields = (
            ("entities", "entity_id"),
            ("quantities", "quantity_id"),
            ("contexts", "context_id"),
            ("observations", "observation_id"),
            ("scales", "scale_id"),
            ("aggregate_maps", "aggregate_map_id"),
            ("transforms", "transform_id"),
        )
        record_tokens: list[str] = []
        for name, identity_field in record_identity_fields:
            path = (*graph_path, name)
            records = _resolve(payload, path)
            if type(records) is not list or any(type(item) is not dict for item in records):
                raise ValueError("wire V3 record-order projection is not an object array")
            identities = tuple(item[identity_field] for item in records)
            record_tokens.extend(f"{name}:{item}" for item in identities)
            add_channel(
                "record_order",
                path,
                feature_projection=identities,
            )
        if tuple(record_tokens) != decoded.channel_projection.record_order_tokens:
            raise ValueError("wire V3 record-order token replay drift")
        observations = _resolve(payload, (*graph_path, "observations"))
        if type(observations) is not list:
            raise ValueError("wire V3 observation records are not an array")
        component_tokens: list[str] = []
        for index in range(len(observations)):
            observation = observations[index]
            if type(observation) is not dict:
                raise ValueError("wire V3 observation record is not an object")
            components = observation["components"]
            if type(components) is not list or any(
                type(component) is not dict
                or set(component) != {"label", "semantic_axis", "value_role"}
                for component in components
            ):
                raise ValueError("wire V3 component-slot projection is not closed")
            slot_tokens = tuple(
                component["label"]
                + "/"
                + component["semantic_axis"]
                + "/"
                + component["value_role"]
                for component in components
            )
            component_tokens.append(
                observation["observation_id"] + ":" + ",".join(slot_tokens)
            )
            add_channel(
                "slot_order",
                (*graph_path, "observations", index, "components"),
                feature_projection=slot_tokens,
            )
        if tuple(component_tokens) != decoded.channel_projection.component_slot_order_tokens:
            raise ValueError("wire V3 component-slot token replay drift")
        eligibility_path = (*graph_path, "eligibility_mask")
        eligibility = _resolve(payload, eligibility_path)
        if type(eligibility) is not dict:
            raise ValueError("wire V3 eligibility mask is not an object")
        eligibility_tokens = tuple(
            [
                "observation_id:" + value
                for value in eligibility["observation_ids"]
            ]
            + [
                "role_candidate_id:" + value
                for value in eligibility["role_candidate_ids"]
            ]
            + ["scale_id:" + value for value in eligibility["scale_ids"]]
        )
        if eligibility_tokens != decoded.channel_projection.eligibility_mask_tokens:
            raise ValueError("wire V3 eligibility-mask token replay drift")
        add_channel(
            "eligibility_mask",
            eligibility_path,
            feature_projection=eligibility_tokens,
        )
        descriptor_path = (*graph_path, "semantic_descriptor")
        descriptor = _resolve(payload, descriptor_path)
        _validate_descriptor_object(descriptor)
        # canonical_value is an integrity replay only.  Feature extraction
        # explicitly decomposes hash-free representation metadata and never
        # uses this whole-object hash as a statistical category.
        add_channel(
            "descriptor_metadata",
            descriptor_path,
            feature_projection="REPRESENTATION_METADATA_ONLY_V1",
        )

        observation_paths = tuple(
            _json_pointer_to_path(payload, pointer)
            for pointer in decoded.channel_projection.observation_array_paths
        )
        return DecodedFormalEnvelopeV1(
            payload=payload,
            payload_bytes=decoded.payload_bytes,
            padding_bytes=decoded.padding_bytes,
            namespace_leaves=tuple(namespace_rows),
            observation_array_paths=observation_paths,
            channel_leaves=tuple(
                sorted(
                    channel_rows.values(),
                    key=lambda item: (item.group, _path_text(item.path)),
                )
            ),
            channel_only_leaf_owners=tuple(
                sorted(
                    owner_rows,
                    key=lambda item: (_path_text(item.path), item.owner_group),
                )
            ),
        )


def _capture_adapter_surface_without_alignment(
    wire: FormalWireAdapterV1,
    recognizer: FormalRecognizerAdapterV1,
) -> tuple[
    str,
    str,
    str,
    bool,
    str | None,
    str | None,
    int,
    int,
    int,
    int,
]:
    return (
        _exact_nonempty_str(wire.implementation_id, "wire implementation ID", 8_192),
        _exact_nonempty_str(wire.schema_id, "wire schema ID", 8_192),
        _exact_nonempty_str(
            recognizer.implementation_id,
            "recognizer implementation ID",
            8_192,
        ),
        wire.supports_atomic_variant_suite,
        getattr(wire, "stop_binding_id", None),
        getattr(wire, "fresh_build_verification_root", None),
        wire.retained_graph_bytes_bound,
        recognizer.retained_graph_bytes_bound,
        wire.variant_suite_resident_bytes_per_row_bound,
        wire.variant_suite_resident_fixed_bytes_bound,
    )


def _capture_adapter_execution_pins(
    wire: FormalWireAdapterV1,
    recognizer: FormalRecognizerAdapterV1,
) -> tuple[_AdapterExecutionPinsV1, FormalPrivateBatchAlignmentV1]:
    try:
        surface = _capture_adapter_surface_without_alignment(wire, recognizer)
        alignment = wire.private_alignment
        if type(alignment) is not FormalPrivateBatchAlignmentV1:
            raise TypeError("wire private alignment is not the exact DTO")
        alignment.__post_init__()
        pins = _AdapterExecutionPinsV1(
            wire_implementation_id=surface[0],
            wire_schema_id=surface[1],
            recognizer_implementation_id=surface[2],
            private_alignment_root=alignment.alignment_root,
            wire_issued_alignment_id=alignment.wire_issued_alignment_id,
            supports_atomic_variant_suite=surface[3],
            stop_binding_id=surface[4],
            fresh_build_verification_root=surface[5],
            wire_retained_graph_bytes_bound=surface[6],
            recognizer_retained_graph_bytes_bound=surface[7],
            variant_suite_resident_bytes_per_row_bound=surface[8],
            variant_suite_resident_fixed_bytes_bound=surface[9],
        )
        if isinstance(wire, FormalWireV3AdapterV1) and (
            pins.supports_atomic_variant_suite is not True
            or pins.stop_binding_id is None
            or pins.fresh_build_verification_root is None
            or pins.wire_issued_alignment_id is None
        ):
            raise ValueError("trusted Wire V3 adapter execution pins are incomplete")
        return pins, alignment
    except FormalCovertAuditRejectionV1:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        _reject(
            "FAIL_ADAPTER_EXECUTION_PINS",
            f"adapter primitive pin capture failed: {exc}",
        )


def _verify_adapter_execution_pins(
    wire: FormalWireAdapterV1,
    recognizer: FormalRecognizerAdapterV1,
    expected: _AdapterExecutionPinsV1,
) -> FormalPrivateBatchAlignmentV1:
    observed, alignment = _capture_adapter_execution_pins(wire, recognizer)
    if observed != expected or observed.pin_root != expected.pin_root:
        _reject(
            "FAIL_ADAPTER_EXECUTION_PIN_DRIFT",
            "wire/recognizer/alignment/suite-support properties changed after pinning",
        )
    return alignment


def issue_formal_wire_v3_audit_authority_v1(
    build_success: object,
    labels_in_source_order: tuple[FormalAuthorityLabelInputV1, ...],
) -> FormalAuditAuthorityV1:
    """Bind private semantic labels to one Wire V3 build-success transcript.

    This proves exact in-process linkage only.  Neither Wire nor this auditor
    authenticates label origin or asserts semantic truth.
    """

    from . import phase2b_formal_trusted_wire_v3 as wire_v3

    if type(build_success) is not wire_v3.FormalTrustedWireBuildSuccessV3:
        raise TypeError("authority issuance needs an exact Wire V3 build success")
    build_success._validate_issued_shape()
    if type(labels_in_source_order) is not tuple or any(
        type(row) is not FormalAuthorityLabelInputV1
        for row in labels_in_source_order
    ):
        raise TypeError("authority issuance needs exact semantic label inputs")
    if tuple(row.row_index for row in labels_in_source_order) != tuple(
        range(len(labels_in_source_order))
    ):
        raise ValueError("authority semantic labels must be contiguous in source order")
    source_bytes = tuple(
        row.canonical_source_label_bytes() for row in labels_in_source_order
    )
    issued_public = wire_v3.issue_projected_private_audit_rows_v3(
        build_success=build_success,
        canonical_source_label_bytes_in_source_order=source_bytes,
    )
    alignment = build_success.private_audit_alignment
    if len(issued_public) != len(labels_in_source_order):
        raise ValueError("Wire V3 private row issuance did not exactly cover labels")
    issued_by_source = {
        link.source_row_index: issued
        for link, issued in zip(
            alignment.links_in_public_order,
            issued_public,
            strict=True,
        )
    }
    rows = []
    for label in labels_in_source_order:
        issued = issued_by_source[label.row_index]
        issued._validate_issued_shape()
        rows.append(
            FormalAuthorityLabelRowV1(
                row_index=label.row_index,
                family=label.family,
                binding=label.binding,
                scale=label.scale,
                answerability=label.answerability,
                joint_decision_class=label.joint_decision_class,
                case_type=label.case_type,
                source_case_commitment=issued.source_case_commitment,
                label_binding_id=issued.label_binding_id,
                row_binding=issued.row_binding,
            )
        )
    return FormalAuditAuthorityV1(tuple(rows))


def issue_development_formal_audit_authority_v1(
    labels_in_source_order: tuple[FormalAuthorityLabelInputV1, ...],
    source_case_commitments_in_source_order: tuple[str, ...],
) -> FormalAuditAuthorityV1:
    """Issue a deterministic non-authoritative development transcript."""

    if type(labels_in_source_order) is not tuple or type(
        source_case_commitments_in_source_order
    ) is not tuple or len(labels_in_source_order) != len(
        source_case_commitments_in_source_order
    ):
        raise TypeError("development authority inputs must be exact aligned tuples")
    if tuple(row.row_index for row in labels_in_source_order) != tuple(
        range(len(labels_in_source_order))
    ):
        raise ValueError("development authority labels are not contiguous")
    rows = []
    for label, commitment in zip(
        labels_in_source_order,
        source_case_commitments_in_source_order,
        strict=True,
    ):
        if type(label) is not FormalAuthorityLabelInputV1:
            raise TypeError("development authority contains a non-exact label")
        if type(commitment) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", commitment) is None:
            raise ValueError("development source commitment is not an exact digest")
        payload = label.canonical_source_label_bytes()
        label_binding_id = "sha256:" + hashlib.sha256(
            _DEVELOPMENT_LABEL_LINK_DOMAIN
            + b"label"
            + commitment.encode("ascii")
            + label.row_index.to_bytes(4, "big")
            + payload
        ).hexdigest()
        row_binding = "sha256:" + hashlib.sha256(
            _DEVELOPMENT_LABEL_LINK_DOMAIN
            + b"row"
            + commitment.encode("ascii")
            + label_binding_id.encode("ascii")
            + len(payload).to_bytes(8, "big")
            + payload
        ).hexdigest()
        rows.append(
            FormalAuthorityLabelRowV1(
                label.row_index,
                label.family,
                label.binding,
                label.scale,
                label.answerability,
                label.joint_decision_class,
                label.case_type,
                commitment,
                label_binding_id,
                row_binding,
            )
        )
    return FormalAuditAuthorityV1(tuple(rows))


@dataclass(frozen=True, slots=True)
class FeatureAliasCoverageV1:
    concrete_feature_id: str
    semantic_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureManifestV1:
    concrete_feature_ids: tuple[str, ...]
    alias_coverage: tuple[FeatureAliasCoverageV1, ...]
    channel_groups: tuple[str, ...]
    unique_id_feature_family: tuple[str, ...]
    uuid_leaf_count: int
    channel_leaf_count: int
    raw_semantic_feature_count: int
    deduplicated_feature_count: int
    manifest_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not FeatureManifestV1:
            raise TypeError("feature manifest must use the exact DTO")
        if (
            type(self.concrete_feature_ids) is not tuple
            or type(self.alias_coverage) is not tuple
            or any(type(item) is not FeatureAliasCoverageV1 for item in self.alias_coverage)
        ):
            raise TypeError("feature manifest IDs and alias coverage must be exact tuples")
        if self.channel_groups != CHANNEL_GROUPS or self.unique_id_feature_family != UNIQUE_ID_FEATURE_FAMILY:
            raise ValueError("feature manifest family drift")
        if (
            type(self.uuid_leaf_count) is not int
            or type(self.channel_leaf_count) is not int
            or self.uuid_leaf_count < 1
            or self.channel_leaf_count < 1
        ):
            raise ValueError("feature manifest must record positive UUID/channel path counts")
        if (
            type(self.raw_semantic_feature_count) is not int
            or type(self.deduplicated_feature_count) is not int
            or self.deduplicated_feature_count < 1
            or self.raw_semantic_feature_count < self.deduplicated_feature_count
            or len(self.concrete_feature_ids) != self.deduplicated_feature_count
        ):
            raise ValueError("feature manifest count drift")
        if len(set(self.concrete_feature_ids)) != len(self.concrete_feature_ids):
            raise ValueError("feature manifest contains duplicate concrete IDs")
        if tuple(item.concrete_feature_id for item in self.alias_coverage) != self.concrete_feature_ids:
            raise ValueError("feature alias coverage is not complete and ordered")
        if any(not item.semantic_aliases for item in self.alias_coverage):
            raise ValueError("concrete feature lacks a semantic alias")
        for item in self.alias_coverage:
            if (
                type(item.semantic_aliases) is not tuple
                or item.semantic_aliases != tuple(sorted(set(item.semantic_aliases)))
                or item.concrete_feature_id
                != stable_hash(
                    {"aliases": item.semantic_aliases},
                    prefix="phase2b_formal_covert_feature_",
                )
            ):
                raise ValueError("feature alias set/concrete ID replay drift")
        if self.concrete_feature_ids != tuple(sorted(self.concrete_feature_ids)):
            raise ValueError("feature manifest concrete ID order drift")
        if self.raw_semantic_feature_count != sum(
            len(item.semantic_aliases) for item in self.alias_coverage
        ):
            raise ValueError("feature manifest raw semantic count drift")
        aliases = {alias for item in self.alias_coverage for alias in item.semantic_aliases}
        for group in CHANNEL_GROUPS:
            if not any(alias == group or alias.startswith(group + "/") for alias in aliases):
                raise ValueError(f"channel feature group is uncovered: {group}")
        for member in UNIQUE_ID_FEATURE_FAMILY:
            if self.uuid_leaf_count and not any(member in alias for alias in aliases):
                raise ValueError(f"unique-ID feature family member is uncovered: {member}")
        object.__setattr__(
            self,
            "manifest_root",
            stable_hash(
                {
                    "concrete": self.concrete_feature_ids,
                    "aliases": tuple(
                        (item.concrete_feature_id, item.semantic_aliases)
                        for item in self.alias_coverage
                    ),
                    "groups": self.channel_groups,
                    "uuid_leaf_count": self.uuid_leaf_count,
                    "channel_leaf_count": self.channel_leaf_count,
                    "raw_count": self.raw_semantic_feature_count,
                },
                prefix="phase2b_formal_covert_feature_manifest_",
            ),
        )


@dataclass(frozen=True, slots=True)
class _InvariantResourcePlanV1:
    rows: int
    envelope_and_decode_bytes_bound: int
    adapter_retained_graph_bytes_bound: int
    parent_current_rss_bytes_at_plan: int
    invariance_variant_suite_output_bytes_bound: int
    variant_suite_resident_bytes_per_row_bound: int
    variant_suite_resident_fixed_bytes_bound: int
    invariance_variant_suite_resident_bytes_bound: int
    invariance_recognizer_live_output_bytes_bound: int
    invariance_worker_artifact_bytes_bound: int
    estimated_invariance_peak_rss_bytes: int
    maximum_invariance_estimated_rss_bytes: int
    invariance_rss_formula: str
    host_available_memory_safety_margin_bytes: int
    admissible: bool
    plan_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not _InvariantResourcePlanV1:
            raise TypeError("invariant resource plan must use the exact type")
        integer_fields = (
            "rows",
            "envelope_and_decode_bytes_bound",
            "adapter_retained_graph_bytes_bound",
            "parent_current_rss_bytes_at_plan",
            "invariance_variant_suite_output_bytes_bound",
            "variant_suite_resident_bytes_per_row_bound",
            "variant_suite_resident_fixed_bytes_bound",
            "invariance_variant_suite_resident_bytes_bound",
            "invariance_recognizer_live_output_bytes_bound",
            "invariance_worker_artifact_bytes_bound",
            "estimated_invariance_peak_rss_bytes",
            "maximum_invariance_estimated_rss_bytes",
            "host_available_memory_safety_margin_bytes",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise TypeError("invariant resource-plan scalars must be exact integers")
        if not 1 <= self.rows <= MAXIMUM_ROWS:
            raise ValueError("invariant resource-plan row count is outside the frozen cap")
        if (
            self.envelope_and_decode_bytes_bound < self.rows * ENVELOPE_BYTES * 2
            or not 0 < self.adapter_retained_graph_bytes_bound
            <= MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES
            + MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES
            or self.parent_current_rss_bytes_at_plan <= 0
            or self.invariance_variant_suite_output_bytes_bound
            != self.rows * 80 * ENVELOPE_BYTES
            or self.variant_suite_resident_bytes_per_row_bound <= 0
            or self.variant_suite_resident_fixed_bytes_bound <= 0
            or self.invariance_variant_suite_resident_bytes_bound
            != self.rows * self.variant_suite_resident_bytes_per_row_bound
            + self.variant_suite_resident_fixed_bytes_bound
            or self.invariance_recognizer_live_output_bytes_bound
            != RECOGNIZER_OUTPUT_SIMULTANEOUS_GRAPH_MULTIPLIER
            * _recognizer_output_graph_bytes_bound(self.rows)
            or self.invariance_worker_artifact_bytes_bound
            != INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
            or self.maximum_invariance_estimated_rss_bytes
            != MAXIMUM_ESTIMATED_RSS_BYTES
            or self.host_available_memory_safety_margin_bytes
            != HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ):
            raise ValueError("invariant resource-plan bound/frozen-cap drift")
        if self.invariance_rss_formula != (
            "max(parent_current_rss,envelope_and_decode_bytes+"
            "adapter_retained_graph_bytes+64MiB)*2+variant_suite_resident_bytes+"
            "recognizer_live_output_bytes+invariance_worker_artifact_bytes"
        ):
            raise ValueError("invariant resource-plan RSS formula drift")
        expected_rss = (
            max(
                self.parent_current_rss_bytes_at_plan,
                self.envelope_and_decode_bytes_bound
                + self.adapter_retained_graph_bytes_bound
                + 64 * 1_048_576,
            )
            * 2
            + self.invariance_variant_suite_resident_bytes_bound
            + self.invariance_recognizer_live_output_bytes_bound
            + self.invariance_worker_artifact_bytes_bound
        )
        if self.estimated_invariance_peak_rss_bytes != expected_rss:
            raise ValueError("invariant resource-plan RSS estimate drift")
        expected_admissible = expected_rss <= MAXIMUM_ESTIMATED_RSS_BYTES
        if type(self.admissible) is not bool or self.admissible is not expected_admissible:
            raise ValueError("invariant resource-plan admissibility drift")
        object.__setattr__(
            self,
            "plan_root",
            stable_hash(
                tuple(
                    (name, getattr(self, name))
                    for name in _InvariantResourcePlanV1.__slots__
                    if name != "plan_root"
                ),
                prefix="phase2b_formal_covert_invariant_resource_plan_",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourcePlanV1:
    rows: int
    concrete_features: int
    unique_statistical_partitions: int
    hypotheses: int
    maximum_feature_values: int
    maximum_label_classes: int
    bitset_words_per_mask: int
    permutation_passes: int
    label_permutations: int
    exchangeable_sample_size: int
    intersections_per_exchangeable_draw: int
    estimated_intersection_work_units: int
    bias_center_canonicalization_work_units: int
    scalar_omnibus_scan_work_units: int
    estimated_total_statistical_work_units: int
    maximum_label_shuffle_randbelow_calls: int
    maximum_shuffle_rejection_draws: int
    maximum_intersection_work_units: int
    maximum_total_statistical_work_units: int
    scalar_row_feature_permutation_work_avoided: int
    permutation_label_mask_bytes_bound: int
    metric_cache_bytes_bound: int
    permutation_metric_tuple_bytes_bound: int
    target_max_vector_bytes_bound: int
    worker_result_artifact_bytes_bound: int
    preextraction_feature_matrix_bytes_bound: int
    feature_matrix_bytes_bound: int
    envelope_and_decode_bytes_bound: int
    partition_allocation_bytes_bound: int
    partition_matrix_bytes_bound: int
    parent_result_working_bytes_bound: int
    adapter_retained_graph_bytes_bound: int
    parent_current_rss_bytes_at_plan: int
    invariance_parent_current_rss_bytes_at_plan: int
    invariance_resource_plan_root: str
    invariance_variant_suite_output_bytes_bound: int
    variant_suite_resident_bytes_per_row_bound: int
    variant_suite_resident_fixed_bytes_bound: int
    invariance_variant_suite_resident_bytes_bound: int
    invariance_recognizer_live_output_bytes_bound: int
    invariance_worker_artifact_bytes_bound: int
    estimated_invariance_peak_rss_bytes: int
    maximum_invariance_estimated_rss_bytes: int
    invariance_rss_formula: str
    estimated_peak_rss_bytes: int
    maximum_estimated_rss_bytes: int
    aggregate_rss_formula: str
    host_available_memory_safety_margin_bytes: int
    partition_preparation_timeout_seconds: int
    invariance_worker_timeout_seconds: int
    statistics_worker_timeout_seconds: int
    statistics_execution_mode: str
    effective_cpu_capacity_at_preflight: int
    maximum_parallel_workers: int
    admissible: bool
    resource_plan_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not ResourcePlanV1:
            raise TypeError("resource plan must use the exact type")
        integer_fields = (
            "rows",
            "concrete_features",
            "unique_statistical_partitions",
            "hypotheses",
            "maximum_feature_values",
            "maximum_label_classes",
            "bitset_words_per_mask",
            "permutation_passes",
            "label_permutations",
            "exchangeable_sample_size",
            "intersections_per_exchangeable_draw",
            "estimated_intersection_work_units",
            "bias_center_canonicalization_work_units",
            "scalar_omnibus_scan_work_units",
            "estimated_total_statistical_work_units",
            "maximum_label_shuffle_randbelow_calls",
            "maximum_shuffle_rejection_draws",
            "maximum_intersection_work_units",
            "maximum_total_statistical_work_units",
            "scalar_row_feature_permutation_work_avoided",
            "permutation_label_mask_bytes_bound",
            "metric_cache_bytes_bound",
            "permutation_metric_tuple_bytes_bound",
            "target_max_vector_bytes_bound",
            "worker_result_artifact_bytes_bound",
            "preextraction_feature_matrix_bytes_bound",
            "feature_matrix_bytes_bound",
            "envelope_and_decode_bytes_bound",
            "partition_allocation_bytes_bound",
            "partition_matrix_bytes_bound",
            "parent_result_working_bytes_bound",
            "adapter_retained_graph_bytes_bound",
            "parent_current_rss_bytes_at_plan",
            "invariance_parent_current_rss_bytes_at_plan",
            "invariance_variant_suite_output_bytes_bound",
            "variant_suite_resident_bytes_per_row_bound",
            "variant_suite_resident_fixed_bytes_bound",
            "invariance_variant_suite_resident_bytes_bound",
            "invariance_recognizer_live_output_bytes_bound",
            "invariance_worker_artifact_bytes_bound",
            "estimated_invariance_peak_rss_bytes",
            "maximum_invariance_estimated_rss_bytes",
            "estimated_peak_rss_bytes",
            "maximum_estimated_rss_bytes",
            "host_available_memory_safety_margin_bytes",
            "partition_preparation_timeout_seconds",
            "invariance_worker_timeout_seconds",
            "statistics_worker_timeout_seconds",
            "effective_cpu_capacity_at_preflight",
            "maximum_parallel_workers",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise TypeError("resource-plan scalar fields must be exact integers")
        if not 1 <= self.rows <= MAXIMUM_ROWS:
            raise ValueError("resource-plan row count is outside the frozen cap")
        if self.concrete_features < 1:
            raise ValueError("resource-plan feature count must be positive")
        if not 1 <= self.unique_statistical_partitions <= self.concrete_features:
            raise ValueError("resource-plan statistical partition count is invalid")
        if self.hypotheses != len(TARGETS) * self.concrete_features:
            raise ValueError("resource-plan hypothesis count drift")
        if self.maximum_feature_values < 1 or self.maximum_label_classes < 1:
            raise ValueError("resource-plan feature/label cardinality must be positive")
        if self.bitset_words_per_mask != (self.rows + 63) // 64:
            raise ValueError("resource-plan bitset word count drift")
        if (
            self.permutation_passes != 1
            or self.label_permutations != LABEL_PERMUTATIONS
            or self.exchangeable_sample_size != EXCHANGEABLE_SAMPLE_SIZE
        ):
            raise ValueError("resource-plan permutation coverage drift")
        if (
            self.intersections_per_exchangeable_draw <= 0
            or self.estimated_intersection_work_units
            != self.exchangeable_sample_size
            * self.intersections_per_exchangeable_draw
        ):
            raise ValueError("resource-plan B+1 intersection-work formula drift")
        if (
            self.maximum_intersection_work_units
            != MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS
            or self.maximum_total_statistical_work_units
            != MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS
            or self.maximum_estimated_rss_bytes != MAXIMUM_ESTIMATED_RSS_BYTES
            or self.maximum_invariance_estimated_rss_bytes
            != MAXIMUM_ESTIMATED_RSS_BYTES
            or self.host_available_memory_safety_margin_bytes
            != HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
            or self.partition_preparation_timeout_seconds
            != PARTITION_PREPARATION_TIMEOUT_SECONDS
            or self.invariance_worker_timeout_seconds
            != INVARIANCE_WORKER_TIMEOUT_SECONDS
            or self.statistics_worker_timeout_seconds
            != STATISTICS_WORKER_TIMEOUT_SECONDS
        ):
            raise ValueError("resource-plan frozen cap drift")
        if self.aggregate_rss_formula != (
            "max(parent_current_rss,mask_bytes+partition_matrix_bytes+"
            "adapter_retained_graph_bytes+64MiB)"
            "*(1+workers)+metric_cache_bytes*workers+permutation_metric_tuple_bytes"
            "+target_max_vector_bytes+worker_result_artifact_bytes+"
            "parent_result_working_bytes"
        ):
            raise ValueError("resource-plan aggregate RSS formula drift")
        if self.invariance_rss_formula != (
            "max(parent_current_rss,envelope_and_decode_bytes+"
            "adapter_retained_graph_bytes+64MiB)*2+variant_suite_resident_bytes+"
            "recognizer_live_output_bytes+invariance_worker_artifact_bytes"
        ):
            raise ValueError("resource-plan invariant RSS formula drift")
        if any(
            getattr(self, name) < 0
            for name in (
                "estimated_intersection_work_units",
                "bias_center_canonicalization_work_units",
                "scalar_omnibus_scan_work_units",
                "estimated_total_statistical_work_units",
                "maximum_label_shuffle_randbelow_calls",
                "maximum_shuffle_rejection_draws",
                "scalar_row_feature_permutation_work_avoided",
                "permutation_label_mask_bytes_bound",
                "metric_cache_bytes_bound",
                "permutation_metric_tuple_bytes_bound",
                "target_max_vector_bytes_bound",
                "worker_result_artifact_bytes_bound",
                "preextraction_feature_matrix_bytes_bound",
                "feature_matrix_bytes_bound",
                "envelope_and_decode_bytes_bound",
                "partition_allocation_bytes_bound",
                "partition_matrix_bytes_bound",
                "parent_result_working_bytes_bound",
                "adapter_retained_graph_bytes_bound",
                "parent_current_rss_bytes_at_plan",
                "invariance_parent_current_rss_bytes_at_plan",
                "invariance_variant_suite_output_bytes_bound",
                "variant_suite_resident_bytes_per_row_bound",
                "variant_suite_resident_fixed_bytes_bound",
                "invariance_variant_suite_resident_bytes_bound",
                "invariance_recognizer_live_output_bytes_bound",
                "invariance_worker_artifact_bytes_bound",
                "estimated_invariance_peak_rss_bytes",
                "estimated_peak_rss_bytes",
            )
        ):
            raise ValueError("resource-plan byte/work bounds cannot be negative")
        if (
            self.feature_matrix_bytes_bound
            < self.rows * self.concrete_features * 72
            or self.preextraction_feature_matrix_bytes_bound
            < self.feature_matrix_bytes_bound
            or self.envelope_and_decode_bytes_bound
            < self.rows * ENVELOPE_BYTES * 2
            or self.partition_matrix_bytes_bound <= 0
            or self.partition_allocation_bytes_bound
            < self.partition_matrix_bytes_bound
            or self.parent_result_working_bytes_bound
            != _parent_result_working_bytes_bound(self.concrete_features)
            or not 0 < self.adapter_retained_graph_bytes_bound
            <= MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES
            + MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES
            or self.parent_current_rss_bytes_at_plan <= 0
            or self.invariance_parent_current_rss_bytes_at_plan <= 0
            or self.invariance_variant_suite_output_bytes_bound
            != self.rows * 80 * ENVELOPE_BYTES
            or self.variant_suite_resident_bytes_per_row_bound <= 0
            or self.variant_suite_resident_fixed_bytes_bound <= 0
            or self.invariance_variant_suite_resident_bytes_bound
            != self.rows * self.variant_suite_resident_bytes_per_row_bound
            + self.variant_suite_resident_fixed_bytes_bound
            or self.invariance_recognizer_live_output_bytes_bound
            != RECOGNIZER_OUTPUT_SIMULTANEOUS_GRAPH_MULTIPLIER
            * _recognizer_output_graph_bytes_bound(self.rows)
            or self.invariance_worker_artifact_bytes_bound
            != INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
        ):
            raise ValueError("resource-plan object-graph/preallocation bound is below its floor")
        invariant_plan = _InvariantResourcePlanV1(
            rows=self.rows,
            envelope_and_decode_bytes_bound=self.envelope_and_decode_bytes_bound,
            adapter_retained_graph_bytes_bound=self.adapter_retained_graph_bytes_bound,
            parent_current_rss_bytes_at_plan=(
                self.invariance_parent_current_rss_bytes_at_plan
            ),
            invariance_variant_suite_output_bytes_bound=(
                self.invariance_variant_suite_output_bytes_bound
            ),
            variant_suite_resident_bytes_per_row_bound=(
                self.variant_suite_resident_bytes_per_row_bound
            ),
            variant_suite_resident_fixed_bytes_bound=(
                self.variant_suite_resident_fixed_bytes_bound
            ),
            invariance_variant_suite_resident_bytes_bound=(
                self.invariance_variant_suite_resident_bytes_bound
            ),
            invariance_recognizer_live_output_bytes_bound=(
                self.invariance_recognizer_live_output_bytes_bound
            ),
            invariance_worker_artifact_bytes_bound=(
                self.invariance_worker_artifact_bytes_bound
            ),
            estimated_invariance_peak_rss_bytes=(
                self.estimated_invariance_peak_rss_bytes
            ),
            maximum_invariance_estimated_rss_bytes=(
                self.maximum_invariance_estimated_rss_bytes
            ),
            invariance_rss_formula=self.invariance_rss_formula,
            host_available_memory_safety_margin_bytes=(
                self.host_available_memory_safety_margin_bytes
            ),
            admissible=(
                self.estimated_invariance_peak_rss_bytes
                <= self.maximum_invariance_estimated_rss_bytes
            ),
        )
        if self.invariance_resource_plan_root != invariant_plan.plan_root:
            raise ValueError("resource-plan invariant phase root drift")
        if self.effective_cpu_capacity_at_preflight < 1:
            raise ValueError("resource-plan effective CPU capacity must be positive")
        resident_parent_bound = max(
            self.parent_current_rss_bytes_at_plan,
            self.permutation_label_mask_bytes_bound
            + self.partition_matrix_bytes_bound
            + self.adapter_retained_graph_bytes_bound
            + 64 * 1_048_576,
        )
        five_worker_rss = (
            resident_parent_bound * (1 + len(TARGETS))
            + self.metric_cache_bytes_bound * len(TARGETS)
            + LABEL_PERMUTATIONS * 256 * len(TARGETS)
            + LABEL_PERMUTATIONS
            * TARGET_MAX_VECTOR_BYTES_PER_PERMUTATION
            * len(TARGETS)
            + self.worker_result_artifact_bytes_bound
            + self.parent_result_working_bytes_bound
        )
        expected_workers = (
            len(TARGETS)
            if self.estimated_intersection_work_units
            >= PARALLEL_STATISTICS_WORK_THRESHOLD
            and self.effective_cpu_capacity_at_preflight >= len(TARGETS)
            and five_worker_rss <= MAXIMUM_ESTIMATED_RSS_BYTES
            else 1
        )
        expected_mode = "LINUX_RAW_FORK_WAIT4_ONE_OR_FIVE_TARGET_WORKERS"
        if (
            self.maximum_parallel_workers != expected_workers
            or self.statistics_execution_mode != expected_mode
        ):
            raise ValueError("resource-plan execution mode drift")
        expected_center_work = (
            self.unique_statistical_partitions
            * len(TARGETS)
            * (
                EXCHANGEABLE_SAMPLE_SIZE * BINARY64_SUPERACCUMULATOR_WORDS
                + BINARY64_SUPERACCUMULATOR_WORDS**2
            )
        )
        expected_shuffle_calls = (
            LABEL_PERMUTATIONS * len(TARGETS) * max(0, self.rows - 1)
        )
        if (
            self.bias_center_canonicalization_work_units
            != expected_center_work
            or self.scalar_omnibus_scan_work_units != (
            LABEL_PERMUTATIONS
            * self.unique_statistical_partitions
            * len(TARGETS)
            )
        ) or self.permutation_metric_tuple_bytes_bound != (
            LABEL_PERMUTATIONS * 256 * expected_workers
        ) or self.target_max_vector_bytes_bound != (
            LABEL_PERMUTATIONS
            * TARGET_MAX_VECTOR_BYTES_PER_PERMUTATION
            * expected_workers
        ) or self.worker_result_artifact_bytes_bound != (
            len(TARGETS)
            * (
                self.concrete_features
                * WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE
                + WORKER_RESULT_ARTIFACT_FIXED_BYTES
            )
        ):
            raise ValueError("resource-plan one-pass scalar/tuple bound drift")
        if (
            self.maximum_label_shuffle_randbelow_calls != expected_shuffle_calls
            or self.maximum_shuffle_rejection_draws
            != expected_shuffle_calls * MAXIMUM_SHUFFLE_REJECTION_DRAWS
        ):
            raise ValueError("resource-plan bounded shuffle work drift")
        expected_total_work = (
            self.estimated_intersection_work_units
            + self.bias_center_canonicalization_work_units
            + self.scalar_omnibus_scan_work_units
            + self.maximum_label_shuffle_randbelow_calls
        )
        if self.estimated_total_statistical_work_units != expected_total_work:
            raise ValueError("resource-plan total statistical work formula drift")
        expected_admissible = (
            self.concrete_features <= MAXIMUM_FEATURES
            and self.maximum_feature_values <= MAXIMUM_VALUES_PER_FEATURE
            and self.maximum_label_classes <= MAXIMUM_LABEL_CLASSES
            and self.estimated_intersection_work_units
            <= MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS
            and self.estimated_total_statistical_work_units
            <= MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS
            and self.estimated_peak_rss_bytes <= MAXIMUM_ESTIMATED_RSS_BYTES
            and invariant_plan.admissible
        )
        if type(self.admissible) is not bool or self.admissible is not expected_admissible:
            raise ValueError("resource-plan admissibility drift")
        object.__setattr__(
            self,
            "resource_plan_root",
            stable_hash(
                tuple(
                    (name, getattr(self, name))
                    for name in ResourcePlanV1.__slots__
                    if name != "resource_plan_root"
                ),
                prefix="phase2b_formal_covert_statistics_resource_plan_",
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetSufficiencyV1:
    target: str
    label_class_count: int
    minimum_class_count: int
    stratum_count: int
    minimum_stratum_size: int
    mutable_stratum_count: int
    sufficient: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self) is not TargetSufficiencyV1 or self.target not in TARGETS:
            raise TypeError("target sufficiency uses an invalid exact type or target")
        for name in (
            "label_class_count",
            "minimum_class_count",
            "stratum_count",
            "minimum_stratum_size",
            "mutable_stratum_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("target sufficiency counts must be nonnegative exact integers")
        if (
            self.label_class_count > MAXIMUM_LABEL_CLASSES
            or self.mutable_stratum_count > self.stratum_count
        ):
            raise ValueError("target sufficiency count exceeds a frozen bound")
        reasons = {
            "TARGET_HAS_FEWER_THAN_TWO_CLASSES",
            "LOO_CLASS_HAS_FEWER_THAN_TWO_ROWS",
            "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_ROWS",
            "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_TARGET_CLASSES",
            "TARGET_CLASS_IS_IMMUTABLE_WITHIN_STRATA",
            "TARGET_CLASS_BUDGET_EXCEEDED",
            "SUFFICIENT",
        }
        if self.reason not in reasons or type(self.sufficient) is not bool:
            raise ValueError("target sufficiency reason/type drift")
        if self.sufficient is not (self.reason == "SUFFICIENT"):
            raise ValueError("target sufficiency disposition drift")


@dataclass(frozen=True, slots=True)
class EligibilityMaskTargetQuotaV1:
    target: str
    class_count: int
    structural_signature_count: int
    checked_integer_cells: int
    maximum_absolute_cross_product_difference: int
    exact_quota_matched: bool
    private_contingency_root: str

    def __post_init__(self) -> None:
        if self.target not in _ELIGIBILITY_QUOTA_TARGETS:
            raise ValueError("eligibility quota target drift")
        for name in (
            "class_count",
            "structural_signature_count",
            "checked_integer_cells",
            "maximum_absolute_cross_product_difference",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("eligibility quota counts must be nonnegative exact integers")
        if (
            self.checked_integer_cells
            != self.class_count * self.structural_signature_count
        ):
            raise ValueError("eligibility quota cell-count drift")
        if type(self.exact_quota_matched) is not bool or self.exact_quota_matched is not (
            self.maximum_absolute_cross_product_difference == 0
        ):
            raise ValueError("eligibility quota exact verdict drift")
        _exact_nonempty_str(
            self.private_contingency_root,
            "eligibility private contingency root",
        )


@dataclass(frozen=True, slots=True)
class EligibilityMaskQuotaReceiptV1:
    signature_schema: str
    public_signature_root: str
    targets: tuple[EligibilityMaskTargetQuotaV1, ...]
    exact_quota_matched: bool
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.signature_schema != (
            "observation_role_candidate_scale_presence_count_order_independent_v1"
        ):
            raise ValueError("eligibility-mask signature schema drift")
        if tuple(item.target for item in self.targets) != (
            "family",
            "scale",
            "answerable_vs_abstain",
        ):
            raise ValueError("eligibility-mask quota target family drift")
        observed = all(item.exact_quota_matched for item in self.targets)
        if self.exact_quota_matched is not observed:
            raise ValueError("eligibility-mask quota disposition drift")
        object.__setattr__(
            self,
            "receipt_id",
            stable_hash(
                {
                    "signature_schema": self.signature_schema,
                    "public_signature_root": self.public_signature_root,
                    "targets": self.targets,
                    "exact_quota_matched": self.exact_quota_matched,
                },
                prefix="phase2b_formal_eligibility_mask_quota_",
            ),
        )


@dataclass(frozen=True, slots=True)
class _PhaseResourcePreflightEvidenceV1:
    phase: str
    resource_plan_root: str
    parent_current_rss_bytes_at_preflight: int
    static_parent_bytes_bound: int
    fork_parent_rss_multiplier: int
    additional_bytes_bound: int
    projected_aggregate_rss_bytes_at_preflight: int
    maximum_estimated_rss_bytes: int
    host_available_memory_safety_margin_bytes: int

    def __post_init__(self) -> None:
        if self.phase not in {"INVARIANCE", "STATISTICS"}:
            raise ValueError("phase resource-preflight kind drift")
        expected_root_prefix = (
            "phase2b_formal_covert_invariant_resource_plan_"
            if self.phase == "INVARIANCE"
            else "phase2b_formal_covert_statistics_resource_plan_"
        )
        if (
            type(self.resource_plan_root) is not str
            or re.fullmatch(re.escape(expected_root_prefix) + r"[0-9a-f]{64}", self.resource_plan_root)
            is None
        ):
            raise ValueError("phase resource-preflight plan root drift")
        for name in (
            "parent_current_rss_bytes_at_preflight",
            "static_parent_bytes_bound",
            "fork_parent_rss_multiplier",
            "additional_bytes_bound",
            "projected_aggregate_rss_bytes_at_preflight",
            "maximum_estimated_rss_bytes",
            "host_available_memory_safety_margin_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("phase resource-preflight scalars must be exact nonnegative integers")
        expected_multiplier = 2 if self.phase == "INVARIANCE" else None
        if (
            self.parent_current_rss_bytes_at_preflight <= 0
            or self.static_parent_bytes_bound <= 0
            or self.fork_parent_rss_multiplier < 2
            or expected_multiplier is not None
            and self.fork_parent_rss_multiplier != expected_multiplier
            or self.maximum_estimated_rss_bytes != MAXIMUM_ESTIMATED_RSS_BYTES
            or self.host_available_memory_safety_margin_bytes
            != HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ):
            raise ValueError("phase resource-preflight frozen scalar drift")
        expected_projection = (
            max(
                self.parent_current_rss_bytes_at_preflight,
                self.static_parent_bytes_bound,
            )
            * self.fork_parent_rss_multiplier
            + self.additional_bytes_bound
        )
        if self.projected_aggregate_rss_bytes_at_preflight != expected_projection:
            raise ValueError("phase resource-preflight projection formula drift")
        if (
            self.projected_aggregate_rss_bytes_at_preflight
            > self.maximum_estimated_rss_bytes
        ):
            raise ValueError("phase resource-preflight projection exceeds frozen cap")


@dataclass(frozen=True, slots=True)
class UnifiedCgroupMemorySnapshotV1:
    unified_membership: str
    memory_scope_root: str
    leaf_device: int
    leaf_inode: int
    memory_current_bytes: int
    memory_peak_bytes: int
    memory_max_bytes: int

    def __post_init__(self) -> None:
        if type(self) is not UnifiedCgroupMemorySnapshotV1:
            raise TypeError("unified cgroup memory snapshot must use the exact type")
        if (
            type(self.unified_membership) is not str
            or not 1 < len(self.unified_membership) <= 4_096
            or re.fullmatch(
                r"/(?:[A-Za-z0-9_.@:-]+/)*[A-Za-z0-9_.@:-]+",
                self.unified_membership,
            )
            is None
            or any(
                component in {"", ".", ".."}
                for component in self.unified_membership.split("/")[1:]
            )
        ):
            raise ValueError("unified cgroup membership is not one exact non-root leaf")
        for name in (
            "leaf_device",
            "leaf_inode",
            "memory_current_bytes",
            "memory_peak_bytes",
            "memory_max_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("unified cgroup memory counters must be exact integers")
            if getattr(self, name) > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES:
                raise ValueError("unified cgroup memory counters exceed the finite bound")
        if (
            self.leaf_device <= 0
            or self.leaf_inode <= 0
            or self.memory_max_bytes <= 0
            or self.memory_peak_bytes < self.memory_current_bytes
        ):
            raise ValueError("unified cgroup memory counter/leaf semantics drift")
        expected_scope = stable_hash(
            {
                "leaf_device": self.leaf_device,
                "leaf_inode": self.leaf_inode,
                "unified_membership": self.unified_membership,
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        )
        if self.memory_scope_root != expected_scope:
            raise ValueError("unified cgroup memory scope root drift")


@dataclass(frozen=True, slots=True)
class _StatisticalRuntimeEvidenceInputsV1:
    resource_plan_root: str
    resource_preflight: _PhaseResourcePreflightEvidenceV1
    system_available_memory_bytes_at_preflight: int
    cgroup_remaining_memory_bytes_at_preflight: int | None
    host_available_memory_bytes_at_preflight: int
    worker_peak_rss_bytes: tuple[int, ...]
    worker_result_artifact_bytes: tuple[int, ...]
    worker_result_artifact_sha256: tuple[str, ...]
    parent_result_object_graph_bytes_observed: int
    worker_timeout_seconds: int
    parent_postprocess_deadline_schema: str
    parent_postprocess_deadline_preemptive: bool
    external_supervisor_required: bool


@dataclass(frozen=True, slots=True)
class StatisticalRuntimeEvidenceV1:
    resource_plan_root: str
    resource_preflight: _PhaseResourcePreflightEvidenceV1
    system_available_memory_bytes_at_preflight: int
    cgroup_remaining_memory_bytes_at_preflight: int | None
    host_available_memory_bytes_at_preflight: int
    parent_historical_peak_rss_bytes_before_receipt_finalization: int
    worker_peak_rss_bytes: tuple[int, ...]
    worker_result_artifact_bytes: tuple[int, ...]
    worker_result_artifact_sha256: tuple[str, ...]
    parent_result_object_graph_bytes_observed: int
    diagnostic_parent_plus_worker_historical_peak_sum_bytes: int
    unified_cgroup_membership: str
    unified_cgroup_memory_scope_root: str
    unified_cgroup_leaf_device: int
    unified_cgroup_leaf_inode: int
    physical_memory_current_bytes_before_receipt_finalization: int
    physical_memory_peak_bytes_before_receipt_finalization: int
    physical_memory_max_bytes: int
    physical_memory_cap_bytes: int
    statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds: int
    worker_timeout_seconds: int
    parent_postprocess_deadline_schema: str
    parent_postprocess_deadline_preemptive: bool
    external_supervisor_required: bool
    receipt_finalization_complete_pending_validation_count: int
    receipt_finalization_physical_peak_within_cap: bool
    within_frozen_physical_peak_cap: bool

    def __post_init__(self) -> None:
        if type(self) is not StatisticalRuntimeEvidenceV1:
            raise TypeError("statistical runtime evidence must use the exact type")
        if (
            type(self.resource_plan_root) is not str
            or not self.resource_plan_root.startswith(
                "phase2b_formal_covert_statistics_resource_plan_"
            )
        ):
            raise ValueError("statistical runtime resource-plan root drift")
        if (
            type(self.resource_preflight) is not _PhaseResourcePreflightEvidenceV1
            or self.resource_preflight.phase != "STATISTICS"
            or self.resource_preflight.resource_plan_root != self.resource_plan_root
        ):
            raise ValueError("statistical runtime phase-preflight evidence drift")
        self.resource_preflight.__post_init__()
        for name in (
            "system_available_memory_bytes_at_preflight",
            "host_available_memory_bytes_at_preflight",
            "parent_historical_peak_rss_bytes_before_receipt_finalization",
            "parent_result_object_graph_bytes_observed",
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes",
            "unified_cgroup_leaf_device",
            "unified_cgroup_leaf_inode",
            "physical_memory_current_bytes_before_receipt_finalization",
            "physical_memory_peak_bytes_before_receipt_finalization",
            "physical_memory_max_bytes",
            "physical_memory_cap_bytes",
            "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds",
            "worker_timeout_seconds",
            "receipt_finalization_complete_pending_validation_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("statistical runtime evidence scalars must be exact integers")
        if type(self.worker_peak_rss_bytes) is not tuple or any(
            type(item) is not int or item < 0 for item in self.worker_peak_rss_bytes
        ):
            raise TypeError("worker RSS evidence must be an exact integer tuple")
        if (
            self.parent_historical_peak_rss_bytes_before_receipt_finalization
            > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
            or self.diagnostic_parent_plus_worker_historical_peak_sum_bytes
            > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
            or any(
                item > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
                for item in self.worker_peak_rss_bytes
            )
        ):
            raise ValueError("statistical historical RSS diagnostics exceed the finite bound")
        if self.cgroup_remaining_memory_bytes_at_preflight is not None and (
            type(self.cgroup_remaining_memory_bytes_at_preflight) is not int
            or self.cgroup_remaining_memory_bytes_at_preflight < 0
        ):
            raise TypeError("cgroup memory evidence must be absent or a nonnegative integer")
        expected_available = min(
            self.system_available_memory_bytes_at_preflight,
            self.cgroup_remaining_memory_bytes_at_preflight
            if self.cgroup_remaining_memory_bytes_at_preflight is not None
            else self.system_available_memory_bytes_at_preflight,
        )
        if self.host_available_memory_bytes_at_preflight != expected_available:
            raise ValueError("effective host/cgroup memory evidence drift")
        if self.host_available_memory_bytes_at_preflight < (
            self.resource_preflight.projected_aggregate_rss_bytes_at_preflight
            + self.resource_preflight.host_available_memory_safety_margin_bytes
        ):
            raise ValueError("statistical runtime host preflight did not cover final projection")
        if type(self.worker_result_artifact_bytes) is not tuple or any(
            type(item) is not int or item <= 0
            for item in self.worker_result_artifact_bytes
        ):
            raise TypeError("worker artifact byte evidence must be a positive integer tuple")
        if type(self.worker_result_artifact_sha256) is not tuple or any(
            type(item) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None
            for item in self.worker_result_artifact_sha256
        ):
            raise TypeError("worker artifact digest evidence must be canonical SHA-256")
        if len(self.worker_result_artifact_bytes) != len(
            self.worker_result_artifact_sha256
        ):
            raise ValueError("worker artifact byte/digest evidence cardinality drift")
        worker_peak_sum = sum(self.worker_peak_rss_bytes)
        if (
            self.diagnostic_parent_plus_worker_historical_peak_sum_bytes
            != self.parent_historical_peak_rss_bytes_before_receipt_finalization
            + worker_peak_sum
        ):
            raise ValueError("statistical historical RSS diagnostic sum drift")
        if (
            type(self.unified_cgroup_membership) is not str
            or type(self.unified_cgroup_memory_scope_root) is not str
            or self.physical_memory_cap_bytes != MAXIMUM_ESTIMATED_RSS_BYTES
            or self.physical_memory_peak_bytes_before_receipt_finalization
            < self.physical_memory_current_bytes_before_receipt_finalization
            or self.physical_memory_peak_bytes_before_receipt_finalization
            > self.physical_memory_cap_bytes
        ):
            raise ValueError("statistical physical memory evidence drift")
        UnifiedCgroupMemorySnapshotV1(
            self.unified_cgroup_membership,
            self.unified_cgroup_memory_scope_root,
            self.unified_cgroup_leaf_device,
            self.unified_cgroup_leaf_inode,
            self.physical_memory_current_bytes_before_receipt_finalization,
            self.physical_memory_peak_bytes_before_receipt_finalization,
            self.physical_memory_max_bytes,
        )
        if self.worker_timeout_seconds != STATISTICS_WORKER_TIMEOUT_SECONDS:
            raise ValueError("statistical runtime timeout drift")
        if self.statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds > (
            STATISTICS_WORKER_TIMEOUT_SECONDS * 1_000_000_000
        ):
            raise ValueError("statistical runtime exceeded the absolute deadline")
        if (
            self.parent_postprocess_deadline_schema
            != PARENT_POSTPROCESS_DEADLINE_SCHEMA
            or self.parent_postprocess_deadline_preemptive is not False
            or self.external_supervisor_required is not True
        ):
            raise ValueError("statistical parent-postprocess deadline boundary drift")
        if (
            self.receipt_finalization_complete_pending_validation_count != 1
            or type(
                self.receipt_finalization_physical_peak_within_cap
            )
            is not bool
            or self.receipt_finalization_physical_peak_within_cap is not True
        ):
            raise ValueError("statistical receipt finalization evidence drift")
        if (
            type(self.within_frozen_physical_peak_cap) is not bool
            or self.within_frozen_physical_peak_cap is not True
        ):
            raise ValueError("statistical runtime physical-peak-cap verdict drift")


@dataclass(frozen=True, slots=True)
class ReceiptIssuanceRuntimeEvidenceV1:
    absolute_deadline_kind: str
    absolute_deadline_monotonic_nanoseconds: int
    absolute_deadline_timeout_seconds: int
    diagnostic_worker_historical_peak_rss_bytes_sum: int
    diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization: int
    diagnostic_parent_plus_worker_historical_peak_sum_bytes: int
    unified_cgroup_membership: str
    unified_cgroup_memory_scope_root: str
    unified_cgroup_leaf_device: int
    unified_cgroup_leaf_inode: int
    physical_memory_current_bytes_before_receipt_finalization: int
    physical_memory_peak_bytes_before_receipt_finalization: int
    physical_memory_max_bytes: int
    physical_memory_cap_bytes: int
    receipt_finalization_complete_pending_validation_count: int
    receipt_finalization_locked_deadline_check_count: int
    receipt_finalization_locked_deadline_within_absolute_limit: bool
    receipt_finalization_locked_physical_peak_within_cap: bool

    def __post_init__(self) -> None:
        if type(self) is not ReceiptIssuanceRuntimeEvidenceV1:
            raise TypeError("receipt issuance evidence must use the exact type")
        if type(self.absolute_deadline_kind) is not str or (
            self.absolute_deadline_kind
            not in {
                "STATISTICS",
                "PARTITION_PREPARATION",
            }
        ):
            raise ValueError("receipt issuance absolute-deadline kind drift")
        for name in (
            "absolute_deadline_monotonic_nanoseconds",
            "absolute_deadline_timeout_seconds",
            "diagnostic_worker_historical_peak_rss_bytes_sum",
            "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization",
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes",
            "unified_cgroup_leaf_device",
            "unified_cgroup_leaf_inode",
            "physical_memory_current_bytes_before_receipt_finalization",
            "physical_memory_peak_bytes_before_receipt_finalization",
            "physical_memory_max_bytes",
            "physical_memory_cap_bytes",
            "receipt_finalization_complete_pending_validation_count",
            "receipt_finalization_locked_deadline_check_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError(
                    "receipt issuance evidence scalars must be exact nonnegative integers"
                )
        expected_timeout = (
            STATISTICS_WORKER_TIMEOUT_SECONDS
            if self.absolute_deadline_kind == "STATISTICS"
            else PARTITION_PREPARATION_TIMEOUT_SECONDS
        )
        if (
            self.absolute_deadline_monotonic_nanoseconds <= 0
            or self.absolute_deadline_timeout_seconds != expected_timeout
        ):
            raise ValueError("receipt issuance deadline evidence drift")
        if (
            self.diagnostic_parent_plus_worker_historical_peak_sum_bytes
            != self.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization
            + self.diagnostic_worker_historical_peak_rss_bytes_sum
        ):
            raise ValueError("receipt issuance historical RSS diagnostic drift")
        if any(
            value > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
            for value in (
                self.diagnostic_worker_historical_peak_rss_bytes_sum,
                self.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization,
                self.diagnostic_parent_plus_worker_historical_peak_sum_bytes,
            )
        ):
            raise ValueError("receipt issuance historical RSS diagnostics exceed the finite bound")
        snapshot = UnifiedCgroupMemorySnapshotV1(
            self.unified_cgroup_membership,
            self.unified_cgroup_memory_scope_root,
            self.unified_cgroup_leaf_device,
            self.unified_cgroup_leaf_inode,
            self.physical_memory_current_bytes_before_receipt_finalization,
            self.physical_memory_peak_bytes_before_receipt_finalization,
            self.physical_memory_max_bytes,
        )
        if (
            self.physical_memory_cap_bytes != MAXIMUM_ESTIMATED_RSS_BYTES
            or snapshot.memory_peak_bytes > self.physical_memory_cap_bytes
        ):
            raise ValueError("receipt issuance physical memory cap drift")
        if (
            self.receipt_finalization_complete_pending_validation_count != 1
            or self.receipt_finalization_locked_deadline_check_count != 1
            or type(
                self.receipt_finalization_locked_deadline_within_absolute_limit
            )
            is not bool
            or not self.receipt_finalization_locked_deadline_within_absolute_limit
            or type(
                self.receipt_finalization_locked_physical_peak_within_cap
            )
            is not bool
            or not self.receipt_finalization_locked_physical_peak_within_cap
        ):
            raise ValueError("receipt issuance locked-finalization evidence drift")


@dataclass(frozen=True, slots=True)
class InvariantRuntimeEvidenceV1:
    invariance_resource_plan_root: str
    resource_preflight: _PhaseResourcePreflightEvidenceV1
    system_available_memory_bytes_at_preflight: int
    cgroup_remaining_memory_bytes_at_preflight: int | None
    host_available_memory_bytes_at_preflight: int
    parent_current_rss_bytes_at_start: int
    parent_peak_current_rss_bytes_observed: int
    parent_current_rss_bytes_at_completion: int
    parent_historical_peak_rss_bytes: int
    worker_peak_rss_bytes: int
    worker_result_artifact_bytes: int
    worker_result_artifact_sha256: str
    diagnostic_parent_plus_worker_historical_peak_sum_bytes: int
    unified_cgroup_membership: str
    unified_cgroup_memory_scope_root: str
    unified_cgroup_leaf_device: int
    unified_cgroup_leaf_inode: int
    physical_memory_current_bytes_at_completion: int
    physical_memory_peak_bytes_at_completion: int
    physical_memory_max_bytes: int
    physical_memory_cap_bytes: int
    invariance_wall_time_nanoseconds: int
    worker_timeout_seconds: int
    invariant_run_count: int
    exact_invariant_comparisons: int
    fork_input_snapshot_root: str
    adapter_execution_pin_root: str
    variant_suite_root: str
    consumer_decode_count: int
    variant_suite_unique_envelope_object_count: int
    variant_suite_resident_object_graph_bytes_observed: int
    recognizer_output_root: str
    execution_containment: str
    parent_side_adapter_tcb: str
    parent_side_adapter_sandboxed: bool
    process_group_empty_at_exit: bool
    child_address_space_limit_bytes: int
    within_frozen_physical_peak_cap: bool

    def __post_init__(self) -> None:
        if type(self) is not InvariantRuntimeEvidenceV1:
            raise TypeError("invariant runtime evidence must use the exact type")
        if (
            type(self.invariance_resource_plan_root) is not str
            or not self.invariance_resource_plan_root.startswith(
                "phase2b_formal_covert_invariant_resource_plan_"
            )
        ):
            raise ValueError("invariant runtime resource-plan root drift")
        if (
            type(self.resource_preflight) is not _PhaseResourcePreflightEvidenceV1
            or self.resource_preflight.phase != "INVARIANCE"
            or self.resource_preflight.resource_plan_root
            != self.invariance_resource_plan_root
        ):
            raise ValueError("invariant runtime phase-preflight evidence drift")
        self.resource_preflight.__post_init__()
        integer_fields = (
            "system_available_memory_bytes_at_preflight",
            "host_available_memory_bytes_at_preflight",
            "parent_current_rss_bytes_at_start",
            "parent_peak_current_rss_bytes_observed",
            "parent_current_rss_bytes_at_completion",
            "parent_historical_peak_rss_bytes",
            "worker_peak_rss_bytes",
            "worker_result_artifact_bytes",
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes",
            "unified_cgroup_leaf_device",
            "unified_cgroup_leaf_inode",
            "physical_memory_current_bytes_at_completion",
            "physical_memory_peak_bytes_at_completion",
            "physical_memory_max_bytes",
            "physical_memory_cap_bytes",
            "invariance_wall_time_nanoseconds",
            "worker_timeout_seconds",
            "invariant_run_count",
            "exact_invariant_comparisons",
            "consumer_decode_count",
            "variant_suite_unique_envelope_object_count",
            "variant_suite_resident_object_graph_bytes_observed",
            "child_address_space_limit_bytes",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in integer_fields
        ):
            raise TypeError("invariant runtime evidence scalars must be nonnegative integers")
        if self.cgroup_remaining_memory_bytes_at_preflight is not None and (
            type(self.cgroup_remaining_memory_bytes_at_preflight) is not int
            or self.cgroup_remaining_memory_bytes_at_preflight < 0
        ):
            raise TypeError("invariant cgroup memory evidence drift")
        expected_available = min(
            self.system_available_memory_bytes_at_preflight,
            self.cgroup_remaining_memory_bytes_at_preflight
            if self.cgroup_remaining_memory_bytes_at_preflight is not None
            else self.system_available_memory_bytes_at_preflight,
        )
        if self.host_available_memory_bytes_at_preflight != expected_available:
            raise ValueError("invariant effective host-memory evidence drift")
        if self.host_available_memory_bytes_at_preflight < (
            self.resource_preflight.projected_aggregate_rss_bytes_at_preflight
            + self.resource_preflight.host_available_memory_safety_margin_bytes
        ):
            raise ValueError("invariant runtime host preflight did not cover final projection")
        if self.diagnostic_parent_plus_worker_historical_peak_sum_bytes != (
            self.parent_historical_peak_rss_bytes + self.worker_peak_rss_bytes
        ):
            raise ValueError("invariant historical RSS diagnostic sum drift")
        if any(
            value > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
            for value in (
                self.parent_historical_peak_rss_bytes,
                self.worker_peak_rss_bytes,
                self.diagnostic_parent_plus_worker_historical_peak_sum_bytes,
            )
        ):
            raise ValueError("invariant historical RSS diagnostics exceed the finite bound")
        maximum_sampled_current = max(
            self.parent_current_rss_bytes_at_start,
            self.parent_peak_current_rss_bytes_observed,
            self.parent_current_rss_bytes_at_completion,
        )
        if (
            self.parent_current_rss_bytes_at_start
            != self.resource_preflight.parent_current_rss_bytes_at_preflight
            or self.parent_historical_peak_rss_bytes < maximum_sampled_current
        ):
            raise ValueError("invariant parent current/historical RSS evidence drift")
        snapshot = UnifiedCgroupMemorySnapshotV1(
            self.unified_cgroup_membership,
            self.unified_cgroup_memory_scope_root,
            self.unified_cgroup_leaf_device,
            self.unified_cgroup_leaf_inode,
            self.physical_memory_current_bytes_at_completion,
            self.physical_memory_peak_bytes_at_completion,
            self.physical_memory_max_bytes,
        )
        if (
            self.physical_memory_cap_bytes != MAXIMUM_ESTIMATED_RSS_BYTES
            or snapshot.memory_peak_bytes > self.physical_memory_cap_bytes
        ):
            raise ValueError("invariant physical memory cap evidence drift")
        if (
            self.worker_timeout_seconds != INVARIANCE_WORKER_TIMEOUT_SECONDS
            or self.invariance_wall_time_nanoseconds
            > INVARIANCE_WORKER_TIMEOUT_SECONDS * 1_000_000_000
        ):
            raise ValueError("invariant runtime deadline evidence drift")
        if (
            type(self.worker_result_artifact_sha256) is not str
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", self.worker_result_artifact_sha256
            )
            is None
            or not 0 < self.worker_result_artifact_bytes
            <= INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
        ):
            raise ValueError("invariant artifact evidence drift")
        if (
            not self.fork_input_snapshot_root.startswith(
                "phase2b_formal_covert_invariance_input_snapshot_"
            )
            or not self.adapter_execution_pin_root.startswith(
                "phase2b_formal_covert_adapter_execution_pins_"
            )
            or not self.variant_suite_root.startswith("phase2b_formal_variant_suite_")
            or not self.recognizer_output_root.startswith(
                "phase2b_formal_covert_recognizer_output_"
            )
            or self.invariant_run_count != 81
            or self.consumer_decode_count % 80
            or self.exact_invariant_comparisons
            != (self.consumer_decode_count // 80) * 80 * 5
            or self.variant_suite_unique_envelope_object_count
            > self.consumer_decode_count
        ):
            raise ValueError("invariant runtime execution coverage/root drift")
        if (
            self.execution_containment != INVARIANCE_EXECUTION_CONTAINMENT
            or self.parent_side_adapter_tcb != PARENT_SIDE_ADAPTER_TCB
            or self.parent_side_adapter_sandboxed is not False
            or self.process_group_empty_at_exit is not True
            or not 0 < self.child_address_space_limit_bytes
            <= MAXIMUM_ESTIMATED_RSS_BYTES
        ):
            raise ValueError("invariant runtime containment evidence drift")
        if (
            type(self.within_frozen_physical_peak_cap) is not bool
            or self.within_frozen_physical_peak_cap is not True
        ):
            raise ValueError("invariant runtime physical-cap verdict drift")


@dataclass(frozen=True, slots=True)
class ExactProbabilityV1:
    """Canonical exact rational in the closed unit interval."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self) is not ExactProbabilityV1:
            raise TypeError("probability must use the exact DTO")
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("probability numerator and denominator must be exact integers")
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise ValueError("probability is outside the canonical unit interval")
        if math.gcd(self.numerator, self.denominator) != 1:
            raise ValueError("probability fraction must be reduced")

    @property
    def as_float(self) -> float:
        return self.numerator / self.denominator

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


def _exact_probability(value: Fraction | int) -> ExactProbabilityV1:
    fraction = Fraction(value)
    if not 0 <= fraction <= 1:
        raise ValueError("probability construction is outside the unit interval")
    return ExactProbabilityV1(fraction.numerator, fraction.denominator)


def _probability_projection(value: ExactProbabilityV1) -> dict[str, int]:
    value.__post_init__()
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


@dataclass(frozen=True, slots=True)
class TargetFamilyMaxTResultV1:
    target: str
    observed_target_max_omnibus: float
    exceedance_count: int
    raw_p: ExactProbabilityV1
    holm_adjusted_p: ExactProbabilityV1
    holm_rank: int
    holm_critical_value: ExactProbabilityV1
    holm_rejected: bool

    def __post_init__(self) -> None:
        if type(self) is not TargetFamilyMaxTResultV1 or self.target not in TARGETS:
            raise TypeError("target-family maxT result uses an invalid exact type or target")
        if (
            type(self.observed_target_max_omnibus) is not float
            or not math.isfinite(self.observed_target_max_omnibus)
            or self.observed_target_max_omnibus < 0.0
        ):
            raise ValueError("target-family observed maxT statistic is invalid")
        if type(self.exceedance_count) is not int or not (
            0 <= self.exceedance_count <= LABEL_PERMUTATIONS
        ):
            raise ValueError("target-family exceedance count is outside the schedule")
        if type(self.raw_p) is not ExactProbabilityV1 or type(
            self.holm_adjusted_p
        ) is not ExactProbabilityV1:
            raise TypeError("target-family p-values must use exact rational DTOs")
        self.raw_p.__post_init__()
        self.holm_adjusted_p.__post_init__()
        expected_raw = Fraction(self.exceedance_count + 1, EXCHANGEABLE_SAMPLE_SIZE)
        if self.raw_p.as_fraction() != expected_raw:
            raise ValueError("target-family plus-one p-value drift")
        if type(self.holm_rank) is not int or not (
            1 <= self.holm_rank <= TARGET_FAMILY_HOLM_SIZE
        ):
            raise ValueError("target-family Holm rank drift")
        if type(self.holm_critical_value) is not ExactProbabilityV1:
            raise TypeError("target-family Holm critical value must be exact rational")
        self.holm_critical_value.__post_init__()
        expected_critical = Fraction(
            FAMILY_WISE_ALPHA_NUMERATOR,
            FAMILY_WISE_ALPHA_DENOMINATOR
            * (TARGET_FAMILY_HOLM_SIZE - self.holm_rank + 1),
        )
        if self.holm_critical_value.as_fraction() != expected_critical:
            raise ValueError("target-family Holm critical value drift")
        expected_rejected = self.holm_adjusted_p.as_fraction() < Fraction(
            FAMILY_WISE_ALPHA_NUMERATOR,
            FAMILY_WISE_ALPHA_DENOMINATOR,
        )
        if type(self.holm_rejected) is not bool or self.holm_rejected is not expected_rejected:
            raise ValueError("target-family Holm decision drift")


@dataclass(frozen=True, slots=True)
class TargetFeatureResultV1:
    target: str
    feature_id: str
    raw_nmi: float
    permutation_mean_nmi: float
    bias_corrected_nmi: float
    balanced_accuracy_advantage: float
    observed_omnibus: float
    pairwise_exceedance_count: int
    pairwise_raw_p: ExactProbabilityV1
    pairwise_holm_adjusted_p: ExactProbabilityV1
    feature_max_t_exceedance_count: int
    feature_max_t_p: ExactProbabilityV1
    target_holm_critical_value: ExactProbabilityV1
    target_family_holm_rejected: bool
    material_effect: bool
    feature_localization_rejected: bool
    failure_witness: bool
    passed: bool

    def __post_init__(self) -> None:
        if type(self) is not TargetFeatureResultV1 or self.target not in TARGETS:
            raise TypeError("target-feature result uses an invalid exact type or target")
        _exact_nonempty_str(self.feature_id, "target-feature ID")
        float_fields = (
            "raw_nmi",
            "permutation_mean_nmi",
            "bias_corrected_nmi",
            "balanced_accuracy_advantage",
            "observed_omnibus",
        )
        if any(
            type(getattr(self, name)) is not float
            or not math.isfinite(getattr(self, name))
            for name in float_fields
        ):
            raise TypeError("target-feature metrics must be finite exact floats")
        if not (
            0.0 <= self.raw_nmi <= 1.0
            and 0.0 <= self.permutation_mean_nmi <= 1.0
            and 0.0 <= self.bias_corrected_nmi <= 1.0
            and -1.0 <= self.balanced_accuracy_advantage <= 1.0
            and self.observed_omnibus >= 0.0
        ):
            raise ValueError("target-feature metric is outside its closed range")
        for name in (
            "pairwise_exceedance_count",
            "feature_max_t_exceedance_count",
        ):
            if type(getattr(self, name)) is not int or not (
                0 <= getattr(self, name) <= LABEL_PERMUTATIONS
            ):
                raise ValueError("target-feature exceedance count is outside the schedule")
        for name in (
            "pairwise_raw_p",
            "pairwise_holm_adjusted_p",
            "feature_max_t_p",
            "target_holm_critical_value",
        ):
            value = getattr(self, name)
            if type(value) is not ExactProbabilityV1:
                raise TypeError("target-feature p-values must use exact rational DTOs")
            value.__post_init__()
        if self.pairwise_raw_p.as_fraction() != Fraction(
            self.pairwise_exceedance_count + 1,
            EXCHANGEABLE_SAMPLE_SIZE,
        ):
            raise ValueError("target-feature pairwise plus-one p-value drift")
        if self.feature_max_t_p.as_fraction() != Fraction(
            self.feature_max_t_exceedance_count + 1,
            EXCHANGEABLE_SAMPLE_SIZE,
        ):
            raise ValueError("target-feature maxT localization p-value drift")
        corrected = max(0.0, self.raw_nmi - self.permutation_mean_nmi)
        omnibus = max(
            corrected / NMI_LIMIT,
            max(0.0, self.balanced_accuracy_advantage)
            / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
        )
        expected_material = omnibus > 1.0
        expected_localization = (
            self.target_family_holm_rejected
            and self.feature_max_t_p.as_fraction()
            < self.target_holm_critical_value.as_fraction()
        )
        expected_witness = expected_material and expected_localization
        if (
            self.bias_corrected_nmi != corrected
            or self.observed_omnibus != omnibus
            or type(self.target_family_holm_rejected) is not bool
            or type(self.material_effect) is not bool
            or self.material_effect is not expected_material
            or type(self.feature_localization_rejected) is not bool
            or self.feature_localization_rejected is not expected_localization
            or type(self.failure_witness) is not bool
            or self.failure_witness is not expected_witness
            or type(self.passed) is not bool
            or self.passed is not (not expected_witness)
        ):
            raise ValueError("target-feature derived metric or decision-witness drift")

    @property
    def exceedance_count(self) -> int:
        """Compatibility view of the resolution-limited pairwise diagnostic."""

        return self.pairwise_exceedance_count

    @property
    def raw_permutation_p(self) -> float:
        """Compatibility float view; the receipt binds the exact rational."""

        return self.pairwise_raw_p.as_float

    @property
    def holm_adjusted_p(self) -> float:
        """Compatibility float view of the non-decision pairwise diagnostic."""

        return self.pairwise_holm_adjusted_p.as_float


def _target_family_metric_projection(
    value: TargetFamilyMaxTResultV1,
) -> dict[str, object]:
    value.__post_init__()
    return {
        "target": value.target,
        "observed_target_max_omnibus_hex": value.observed_target_max_omnibus.hex(),
        "exceedance_count": value.exceedance_count,
        "raw_p": _probability_projection(value.raw_p),
        "holm_adjusted_p": _probability_projection(value.holm_adjusted_p),
        "holm_rank": value.holm_rank,
        "holm_critical_value": _probability_projection(value.holm_critical_value),
        "holm_rejected": value.holm_rejected,
    }


def _target_feature_metric_projection(
    value: TargetFeatureResultV1,
) -> dict[str, object]:
    value.__post_init__()
    return {
        "target": value.target,
        "feature_id": value.feature_id,
        "raw_nmi_hex": value.raw_nmi.hex(),
        "symmetric_permutation_mean_nmi_hex": value.permutation_mean_nmi.hex(),
        "bias_corrected_nmi_hex": value.bias_corrected_nmi.hex(),
        "balanced_accuracy_advantage_hex": value.balanced_accuracy_advantage.hex(),
        "observed_omnibus_hex": value.observed_omnibus.hex(),
        "pairwise_exceedance_count": value.pairwise_exceedance_count,
        "pairwise_raw_p": _probability_projection(value.pairwise_raw_p),
        "pairwise_holm_adjusted_p": _probability_projection(
            value.pairwise_holm_adjusted_p
        ),
        "feature_max_t_exceedance_count": value.feature_max_t_exceedance_count,
        "feature_max_t_p": _probability_projection(value.feature_max_t_p),
        "target_holm_critical_value": _probability_projection(
            value.target_holm_critical_value
        ),
        "target_family_holm_rejected": value.target_family_holm_rejected,
        "material_effect": value.material_effect,
        "feature_localization_rejected": value.feature_localization_rejected,
        "failure_witness": value.failure_witness,
        "passed": value.passed,
    }


@dataclass(frozen=True, slots=True, init=False)
class FormalCovertAuditReceiptV1:
    disposition: FormalCovertDisposition
    authority_transcript_root: str
    private_alignment_root: str
    wire_issued_alignment_id: str | None
    wire_alignment_issuance_verified: bool
    private_label_projection_root: str
    private_label_row_binding_verified: bool
    audited_row_count: int
    wire_implementation_id: str
    wire_schema_id: str
    recognizer_implementation_id: str
    adapter_execution_pin_root: str
    input_envelope_root: str
    feature_manifest: FeatureManifestV1
    resource_plan: ResourcePlanV1
    sufficiency: tuple[TargetSufficiencyV1, ...]
    eligibility_mask_quota: EligibilityMaskQuotaReceiptV1
    invariant_runtime_evidence: InvariantRuntimeEvidenceV1 | None
    statistics_runtime_evidence: StatisticalRuntimeEvidenceV1 | None
    receipt_issuance_runtime_evidence: ReceiptIssuanceRuntimeEvidenceV1
    results: tuple[TargetFeatureResultV1, ...]
    target_family_results: tuple[TargetFamilyMaxTResultV1, ...]
    invariant_execution_root: str | None
    invariant_run_count: int
    exact_invariant_comparisons: int
    permutation_schedule_root: str | None
    permutation_seed_commitment: str | None
    permutation_seed_reveal_hex: str | None
    holm_family_size: int
    pairwise_holm_family_size: int
    pairwise_holm_decision_eligible: bool
    pairwise_holm_resolution_capable: bool
    absence_fields_proved: tuple[str, ...]
    authority_origin_authenticated: bool = False
    authority_independence_verified: bool = False
    actual_corpus_audit: bool = False
    claim_level: str = CLAIM_LEVEL
    positive_decision_claim: str | None = None
    randomization_assumption: str = RANDOMIZATION_ASSUMPTION
    transcript_authentication: str = TRANSCRIPT_AUTHENTICATION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not FormalCovertAuditReceiptV1:
            raise TypeError("formal covert audit receipt must use the exact type")
        for name in (
            "authority_transcript_root",
            "private_alignment_root",
            "private_label_projection_root",
            "wire_implementation_id",
            "wire_schema_id",
            "recognizer_implementation_id",
            "adapter_execution_pin_root",
            "input_envelope_root",
        ):
            _exact_nonempty_str(getattr(self, name), f"receipt {name}", 8_192)
        if not self.adapter_execution_pin_root.startswith(
            "phase2b_formal_covert_adapter_execution_pins_"
        ):
            raise ValueError("receipt adapter execution pin root drift")
        if self.wire_issued_alignment_id is not None:
            _exact_nonempty_str(
                self.wire_issued_alignment_id,
                "receipt wire-issued alignment ID",
            )
        if type(self.audited_row_count) is not int or not (
            1 <= self.audited_row_count <= MAXIMUM_ROWS
        ):
            raise ValueError("receipt audited row count is outside the frozen cap")
        if type(self.feature_manifest) is not FeatureManifestV1:
            raise TypeError("receipt feature manifest must use the exact type")
        self.feature_manifest.__post_init__()
        if type(self.resource_plan) is not ResourcePlanV1:
            raise TypeError("receipt resource plan must use the exact type")
        self.resource_plan.__post_init__()
        if (
            self.resource_plan.rows != self.audited_row_count
            or self.resource_plan.concrete_features
            != self.feature_manifest.deduplicated_feature_count
        ):
            raise ValueError("receipt resource-plan/manifest coverage drift")
        if type(self.sufficiency) is not tuple or any(
            type(item) is not TargetSufficiencyV1 for item in self.sufficiency
        ):
            raise TypeError("receipt sufficiency rows must use exact DTOs")
        if tuple(item.target for item in self.sufficiency) != TARGETS:
            raise ValueError("receipt sufficiency target order drift")
        for item in self.sufficiency:
            item.__post_init__()
        claim_bools = (
            self.authority_origin_authenticated,
            self.authority_independence_verified,
            self.actual_corpus_audit,
        )
        if (
            self.claim_level != CLAIM_LEVEL
            or self.randomization_assumption != RANDOMIZATION_ASSUMPTION
            or self.transcript_authentication != TRANSCRIPT_AUTHENTICATION
            or any(type(item) is not bool for item in claim_bools)
            or any(claim_bools)
        ):
            raise ValueError("auditor receipt exceeds its transcript-only claim boundary")
        if self.absence_fields_proved != MUST_BE_ABSENT_FIELDS:
            raise ValueError("MUST_BE_ABSENT proof set drift")
        if (
            type(self.eligibility_mask_quota) is not EligibilityMaskQuotaReceiptV1
            or not self.eligibility_mask_quota.exact_quota_matched
        ):
            raise ValueError("receipt lacks the exact eligibility-mask quota gate")
        self.eligibility_mask_quota.__post_init__()
        if not self.private_label_projection_root.startswith(
            "phase2b_formal_private_label_projection_"
        ):
            raise ValueError("receipt private-label projection root drift")
        if type(self.wire_alignment_issuance_verified) is not bool or type(
            self.private_label_row_binding_verified
        ) is not bool:
            raise TypeError("wire/private-label binding flags must be exact bools")
        if not self.private_label_row_binding_verified:
            raise ValueError("executed transcript lacks exact private-label row binding")
        if (
            self.wire_issued_alignment_id is None
            and self.wire_alignment_issuance_verified
        ):
                raise ValueError("an absent wire alignment ID cannot be verified")
        if type(self.disposition) is not FormalCovertDisposition:
            raise TypeError("receipt disposition must use the exact enum")
        if type(self.results) is not tuple or type(self.target_family_results) is not tuple:
            raise TypeError("receipt statistical results must use exact tuples")
        if type(self.absence_fields_proved) is not tuple:
            raise TypeError("receipt absence proof must use an exact tuple")
        if not self.resource_plan.admissible:
            raise ValueError("an executed transcript cannot use an inadmissible resource plan")
        if (
            type(self.receipt_issuance_runtime_evidence)
            is not ReceiptIssuanceRuntimeEvidenceV1
        ):
            raise TypeError("receipt lacks exact branch-neutral issuance evidence")
        self.receipt_issuance_runtime_evidence.__post_init__()
        for name in (
            "invariant_run_count",
            "exact_invariant_comparisons",
            "holm_family_size",
            "pairwise_holm_family_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("receipt coverage counts must be nonnegative exact integers")
        expected_pairwise_family = (
            len(TARGETS) * self.feature_manifest.deduplicated_feature_count
        )
        expected_pairwise_resolution = (
            FAMILY_WISE_ALPHA_DENOMINATOR * expected_pairwise_family
            < EXCHANGEABLE_SAMPLE_SIZE
        )
        if type(self.pairwise_holm_decision_eligible) is not bool or (
            self.pairwise_holm_decision_eligible
        ):
            raise ValueError("pairwise 10k Holm is diagnostic-only and never decision-eligible")
        if type(self.pairwise_holm_resolution_capable) is not bool:
            raise TypeError("pairwise Holm resolution flag must be an exact bool")
        if self.disposition is FormalCovertDisposition.REJECT_INSUFFICIENT:
            if all(item.sufficient for item in self.sufficiency):
                raise ValueError("insufficient receipt has no insufficient target")
            if (
                self.results
                or self.target_family_results
                or self.invariant_run_count
                or self.exact_invariant_comparisons
                or self.invariant_execution_root is not None
            ):
                raise ValueError("insufficient rejection cannot expose partial execution")
            if (
                self.holm_family_size != 0
                or self.pairwise_holm_family_size != 0
                or self.pairwise_holm_resolution_capable
                or self.permutation_schedule_root is not None
                or self.permutation_seed_commitment is not None
                or self.permutation_seed_reveal_hex is not None
            ):
                raise ValueError("insufficient rejection cannot expose partial statistics")
            if (
                self.invariant_runtime_evidence is not None
                or self.statistics_runtime_evidence is not None
            ):
                raise ValueError("insufficient rejection cannot expose runtime evidence")
            if self.positive_decision_claim is not None:
                raise ValueError("insufficient rejection cannot carry a positive claim")
            if (
                self.receipt_issuance_runtime_evidence.absolute_deadline_kind
                != "PARTITION_PREPARATION"
                or self.receipt_issuance_runtime_evidence.diagnostic_worker_historical_peak_rss_bytes_sum
                != 0
                or self.receipt_issuance_runtime_evidence.physical_memory_cap_bytes
                != MAXIMUM_ESTIMATED_RSS_BYTES
            ):
                raise ValueError(
                    "insufficient receipt issuance evidence must use W=0 and its partition deadline"
                )
        else:
            if not all(item.sufficient for item in self.sufficiency):
                raise ValueError("executed receipt contains an insufficient target")
            if self.invariant_run_count != (
                1
                + GLOBAL_RENAMINGS
                + GLOBAL_CASE_ORDER_PERMUTATIONS
                + OBSERVATION_ORDER_PERMUTATIONS
            ):
                raise ValueError("invariant run coverage drift")
            expected_comparisons = self.audited_row_count * (
                GLOBAL_RENAMINGS
                + GLOBAL_CASE_ORDER_PERMUTATIONS
                + OBSERVATION_ORDER_PERMUTATIONS
            ) * 5
            if self.exact_invariant_comparisons != expected_comparisons:
                raise ValueError("exact invariant comparison coverage drift")
            if (
                type(self.invariant_execution_root) is not str
                or not self.invariant_execution_root.startswith(
                    "phase2b_formal_covert_invariant_execution_"
                )
            ):
                raise ValueError("executed transcript lacks the exact invariant root")
            if type(self.invariant_runtime_evidence) is not InvariantRuntimeEvidenceV1:
                raise TypeError("executed transcript lacks exact invariant runtime evidence")
            self.invariant_runtime_evidence.__post_init__()
            _validate_invariant_phase_preflight_against_resource_plan(
                self.resource_plan,
                self.invariant_runtime_evidence.resource_preflight,
            )
            if (
                self.invariant_runtime_evidence.invariant_run_count
                != self.invariant_run_count
                or self.invariant_runtime_evidence.exact_invariant_comparisons
                != self.exact_invariant_comparisons
                or self.invariant_runtime_evidence.consumer_decode_count
                != self.audited_row_count * 80
                or self.invariant_runtime_evidence.adapter_execution_pin_root
                != self.adapter_execution_pin_root
                or self.invariant_runtime_evidence.invariance_resource_plan_root
                != self.resource_plan.invariance_resource_plan_root
                or self.invariant_runtime_evidence.resource_preflight.static_parent_bytes_bound
                != self.resource_plan.envelope_and_decode_bytes_bound
                + self.resource_plan.adapter_retained_graph_bytes_bound
                + 64 * 1_048_576
                or self.invariant_runtime_evidence.resource_preflight.fork_parent_rss_multiplier
                != 2
                or self.invariant_runtime_evidence.resource_preflight.additional_bytes_bound
                != self.resource_plan.invariance_variant_suite_resident_bytes_bound
                + self.resource_plan.invariance_recognizer_live_output_bytes_bound
                + self.resource_plan.invariance_worker_artifact_bytes_bound
                or self.invariant_runtime_evidence.resource_preflight.maximum_estimated_rss_bytes
                != self.resource_plan.maximum_invariance_estimated_rss_bytes
                or self.invariant_runtime_evidence.resource_preflight.host_available_memory_safety_margin_bytes
                != self.resource_plan.host_available_memory_safety_margin_bytes
                or self.invariant_runtime_evidence.variant_suite_resident_object_graph_bytes_observed
                > self.resource_plan.invariance_variant_suite_resident_bytes_bound
                or not self.invariant_runtime_evidence.within_frozen_physical_peak_cap
                or self.invariant_runtime_evidence.host_available_memory_bytes_at_preflight
                < self.invariant_runtime_evidence.resource_preflight.projected_aggregate_rss_bytes_at_preflight
                + self.resource_plan.host_available_memory_safety_margin_bytes
            ):
                raise ValueError("invariant runtime/resource receipt linkage drift")
            if (
                type(self.permutation_schedule_root) is not str
                or not self.permutation_schedule_root.startswith(
                    "phase2b_formal_covert_permutation_schedule_"
                )
            ):
                raise ValueError("executed transcript lacks the exact permutation root")
            if (
                type(self.permutation_seed_commitment) is not str
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.permutation_seed_commitment)
                or type(self.permutation_seed_reveal_hex) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", self.permutation_seed_reveal_hex)
                or _permutation_seed_commitment(
                    bytes.fromhex(self.permutation_seed_reveal_hex)
                )
                != self.permutation_seed_commitment
            ):
                raise ValueError("executed transcript permutation seed commit/reveal drift")
            if (
                self.holm_family_size != TARGET_FAMILY_HOLM_SIZE
                or len(self.target_family_results) != TARGET_FAMILY_HOLM_SIZE
            ):
                raise ValueError("decision Holm family must contain exactly five targets")
            if (
                self.pairwise_holm_family_size != expected_pairwise_family
                or len(self.results) != expected_pairwise_family
                or self.pairwise_holm_resolution_capable
                is not expected_pairwise_resolution
            ):
                raise ValueError("pairwise diagnostic Holm coverage/resolution drift")
            expected_order = tuple(
                (target, feature_id)
                for target in TARGETS
                for feature_id in self.feature_manifest.concrete_feature_ids
            )
            if tuple((item.target, item.feature_id) for item in self.results) != expected_order:
                raise ValueError("pairwise diagnostic target-feature order drift")
            if any(type(item) is not TargetFeatureResultV1 for item in self.results):
                raise TypeError("receipt target-feature rows must use exact DTOs")
            for item in self.results:
                item.__post_init__()
            if tuple(item.target for item in self.target_family_results) != TARGETS:
                raise ValueError("target-family maxT result order drift")
            if any(
                type(item) is not TargetFamilyMaxTResultV1
                for item in self.target_family_results
            ):
                raise TypeError("target-family rows must use exact DTOs")
            for item in self.target_family_results:
                item.__post_init__()
            target_adjusted = _holm_adjust_exact(
                tuple(item.raw_p for item in self.target_family_results)
            )
            target_ranks = _holm_ranks_exact(
                tuple(item.raw_p for item in self.target_family_results)
            )
            if tuple(
                (item.holm_adjusted_p, item.holm_rank)
                for item in self.target_family_results
            ) != tuple(zip(target_adjusted, target_ranks, strict=True)):
                raise ValueError("receipt five-target Holm replay drift")
            pairwise_adjusted = _holm_adjust_exact(
                tuple(item.pairwise_raw_p for item in self.results)
            )
            if tuple(
                item.pairwise_holm_adjusted_p for item in self.results
            ) != pairwise_adjusted:
                raise ValueError("receipt diagnostic pairwise Holm replay drift")
            target_by_name = {
                item.target: item for item in self.target_family_results
            }
            for target in TARGETS:
                target_row = target_by_name[target]
                feature_rows = tuple(
                    item for item in self.results if item.target == target
                )
                if target_row.observed_target_max_omnibus != max(
                    item.observed_omnibus for item in feature_rows
                ):
                    raise ValueError("target maxT observed statistic replay drift")
                for item in feature_rows:
                    if (
                        item.target_holm_critical_value
                        != target_row.holm_critical_value
                        or item.target_family_holm_rejected
                        is not target_row.holm_rejected
                        or item.feature_max_t_p.as_fraction()
                        < target_row.raw_p.as_fraction()
                    ):
                        raise ValueError("feature maxT/target-family linkage drift")
            should_pass = not any(item.failure_witness for item in self.results)
            if self.disposition is not (
                FormalCovertDisposition.PASS
                if should_pass
                else FormalCovertDisposition.FAIL
            ):
                raise ValueError("receipt disposition/statistical results drift")
            expected_positive_claim = (
                POSITIVE_DECISION_CLAIM if should_pass else None
            )
            if self.positive_decision_claim != expected_positive_claim:
                raise ValueError("receipt positive claim/disposition drift")
            if type(self.statistics_runtime_evidence) is not StatisticalRuntimeEvidenceV1:
                raise TypeError("executed receipt lacks exact statistical runtime evidence")
            self.statistics_runtime_evidence.__post_init__()
            _validate_statistics_phase_preflight_against_resource_plan(
                self.resource_plan,
                self.statistics_runtime_evidence.resource_preflight,
            )
            expected_worker_count = self.resource_plan.maximum_parallel_workers
            if len(self.statistics_runtime_evidence.worker_peak_rss_bytes) != expected_worker_count:
                raise ValueError("statistical runtime worker-count evidence drift")
            if len(
                self.statistics_runtime_evidence.worker_result_artifact_bytes
            ) != len(TARGETS):
                raise ValueError("statistical runtime target-artifact coverage drift")
            if sum(
                self.statistics_runtime_evidence.worker_result_artifact_bytes
            ) > self.resource_plan.worker_result_artifact_bytes_bound:
                raise ValueError("statistical worker artifact exceeded its resource bound")
            if (
                self.statistics_runtime_evidence.resource_plan_root
                != self.resource_plan.resource_plan_root
                or self.statistics_runtime_evidence.resource_preflight.static_parent_bytes_bound
                != self.resource_plan.permutation_label_mask_bytes_bound
                + self.resource_plan.partition_matrix_bytes_bound
                + self.resource_plan.adapter_retained_graph_bytes_bound
                + 64 * 1_048_576
                or self.statistics_runtime_evidence.resource_preflight.fork_parent_rss_multiplier
                != 1 + self.resource_plan.maximum_parallel_workers
                or self.statistics_runtime_evidence.resource_preflight.additional_bytes_bound
                != self.resource_plan.metric_cache_bytes_bound
                * self.resource_plan.maximum_parallel_workers
                + self.resource_plan.permutation_metric_tuple_bytes_bound
                + self.resource_plan.target_max_vector_bytes_bound
                + self.resource_plan.worker_result_artifact_bytes_bound
                + self.resource_plan.parent_result_working_bytes_bound
                or self.statistics_runtime_evidence.resource_preflight.maximum_estimated_rss_bytes
                != self.resource_plan.maximum_estimated_rss_bytes
                or self.statistics_runtime_evidence.resource_preflight.host_available_memory_safety_margin_bytes
                != self.resource_plan.host_available_memory_safety_margin_bytes
                or
                self.statistics_runtime_evidence.parent_result_object_graph_bytes_observed
                > self.resource_plan.parent_result_working_bytes_bound
            ):
                raise ValueError("parent statistical result graph exceeded its resource bound")
            if not self.statistics_runtime_evidence.within_frozen_physical_peak_cap:
                raise ValueError("statistical runtime evidence exceeds the physical peak cap")
            if (
                self.statistics_runtime_evidence.parent_postprocess_deadline_preemptive
                or not self.statistics_runtime_evidence.external_supervisor_required
                or self.statistics_runtime_evidence.receipt_finalization_complete_pending_validation_count
                != 1
                or self.statistics_runtime_evidence.receipt_finalization_physical_peak_within_cap
                is not True
            ):
                raise ValueError(
                    "executed receipt parent-postprocess boundary drift"
                )
            issuance = self.receipt_issuance_runtime_evidence
            worker_peak_sum = sum(
                self.statistics_runtime_evidence.worker_peak_rss_bytes
            )
            if (
                issuance.absolute_deadline_kind != "STATISTICS"
                or issuance.diagnostic_worker_historical_peak_rss_bytes_sum
                != worker_peak_sum
                or issuance.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization
                != self.statistics_runtime_evidence.parent_historical_peak_rss_bytes_before_receipt_finalization
                or issuance.diagnostic_parent_plus_worker_historical_peak_sum_bytes
                != self.statistics_runtime_evidence.diagnostic_parent_plus_worker_historical_peak_sum_bytes
                or issuance.unified_cgroup_membership
                != self.statistics_runtime_evidence.unified_cgroup_membership
                or issuance.unified_cgroup_memory_scope_root
                != self.statistics_runtime_evidence.unified_cgroup_memory_scope_root
                or issuance.unified_cgroup_leaf_device
                != self.statistics_runtime_evidence.unified_cgroup_leaf_device
                or issuance.unified_cgroup_leaf_inode
                != self.statistics_runtime_evidence.unified_cgroup_leaf_inode
                or issuance.physical_memory_current_bytes_before_receipt_finalization
                != self.statistics_runtime_evidence.physical_memory_current_bytes_before_receipt_finalization
                or issuance.physical_memory_peak_bytes_before_receipt_finalization
                != self.statistics_runtime_evidence.physical_memory_peak_bytes_before_receipt_finalization
                or issuance.physical_memory_max_bytes
                != self.statistics_runtime_evidence.physical_memory_max_bytes
                or issuance.physical_memory_cap_bytes
                != self.statistics_runtime_evidence.physical_memory_cap_bytes
                or issuance.receipt_finalization_complete_pending_validation_count
                != self.statistics_runtime_evidence.receipt_finalization_complete_pending_validation_count
                or issuance.receipt_finalization_locked_physical_peak_within_cap
                is not self.statistics_runtime_evidence.receipt_finalization_physical_peak_within_cap
            ):
                raise ValueError(
                    "statistical and branch-neutral issuance evidence cross-binding drift"
                )
            invariant_scope = self.invariant_runtime_evidence
            if (
                invariant_scope.unified_cgroup_membership
                != issuance.unified_cgroup_membership
                or invariant_scope.unified_cgroup_memory_scope_root
                != issuance.unified_cgroup_memory_scope_root
                or invariant_scope.unified_cgroup_leaf_device
                != issuance.unified_cgroup_leaf_device
                or invariant_scope.unified_cgroup_leaf_inode
                != issuance.unified_cgroup_leaf_inode
                or invariant_scope.physical_memory_max_bytes
                != issuance.physical_memory_max_bytes
                or invariant_scope.physical_memory_peak_bytes_at_completion
                > issuance.physical_memory_peak_bytes_before_receipt_finalization
            ):
                raise ValueError("invariant/statistics physical memory scope binding drift")
            if (
                self.statistics_runtime_evidence.host_available_memory_bytes_at_preflight
                < self.statistics_runtime_evidence.resource_preflight.projected_aggregate_rss_bytes_at_preflight
                + self.resource_plan.host_available_memory_safety_margin_bytes
            ):
                raise ValueError("statistical runtime host-memory preflight evidence drift")
        object.__setattr__(
            self,
            "receipt_id",
            stable_hash(
                {
                    "schema": SCHEMA_VERSION,
                    "semantics_id": SEMANTICS_ID,
                    "disposition": self.disposition.value,
                    "authority_root": self.authority_transcript_root,
                    "private_alignment_root": self.private_alignment_root,
                    "wire_issued_alignment_id": self.wire_issued_alignment_id,
                    "wire_alignment_issuance_verified": self.wire_alignment_issuance_verified,
                    "private_label_projection_root": self.private_label_projection_root,
                    "private_label_row_binding_verified": self.private_label_row_binding_verified,
                    "rows": self.audited_row_count,
                    "wire": (self.wire_implementation_id, self.wire_schema_id),
                    "recognizer": self.recognizer_implementation_id,
                    "adapter_execution_pin_root": self.adapter_execution_pin_root,
                    "input_root": self.input_envelope_root,
                    "feature_manifest_root": self.feature_manifest.manifest_root,
                    "resource_plan": self.resource_plan,
                    "sufficiency": self.sufficiency,
                    "eligibility_mask_quota": self.eligibility_mask_quota,
                    "invariant_runtime_evidence": self.invariant_runtime_evidence,
                    "statistics_runtime_evidence": self.statistics_runtime_evidence,
                    "receipt_issuance_runtime_evidence": (
                        self.receipt_issuance_runtime_evidence
                    ),
                    "results_hex": tuple(
                        _target_feature_metric_projection(item)
                        for item in self.results
                    ),
                    "target_family_results_hex": tuple(
                        _target_family_metric_projection(item)
                        for item in self.target_family_results
                    ),
                    "invariant_execution_root": self.invariant_execution_root,
                    "invariant_run_count": self.invariant_run_count,
                    "comparisons": self.exact_invariant_comparisons,
                    "permutation_schedule_root": self.permutation_schedule_root,
                    "permutation_seed_commitment": self.permutation_seed_commitment,
                    "permutation_seed_reveal_hex": self.permutation_seed_reveal_hex,
                    "holm_family_size": self.holm_family_size,
                    "pairwise_holm_family_size": self.pairwise_holm_family_size,
                    "pairwise_holm_decision_eligible": self.pairwise_holm_decision_eligible,
                    "pairwise_holm_resolution_capable": self.pairwise_holm_resolution_capable,
                    "absence": self.absence_fields_proved,
                    "claim_level": self.claim_level,
                    "positive_decision_claim": self.positive_decision_claim,
                    "randomization_assumption": self.randomization_assumption,
                    "transcript_authentication": self.transcript_authentication,
                    "float_policy": {
                        "nmi_limit_hex": NMI_LIMIT.hex(),
                        "balanced_accuracy_advantage_limit_hex": BALANCED_ACCURACY_ADVANTAGE_LIMIT.hex(),
                        "family_wise_alpha_hex": FAMILY_WISE_ALPHA.hex(),
                        "runtime_scope": "PINNED_PYTHON_RUNTIME_NO_CROSS_RUNTIME_CLAIM",
                    },
                },
                prefix="phase2b_formal_covert_audit_transcript_",
            ),
        )


_RECEIPT_EXECUTION_FACTORY_TOKEN: Final = object()


class _InProcessReceiptRegistryStateV1(Enum):
    PENDING = "PENDING"
    ISSUED = "ISSUED"


@dataclass(slots=True, eq=False)
class _InProcessReceiptReservationV1:
    entry_identity: int | None = None


@dataclass(slots=True, eq=False)
class _InProcessReceiptRegistryEntryV1:
    receipt: FormalCovertAuditReceiptV1
    receipt_id: str
    reservation: _InProcessReceiptReservationV1
    state: _InProcessReceiptRegistryStateV1


_IN_PROCESS_EXECUTED_RECEIPTS: dict[
    int, _InProcessReceiptRegistryEntryV1
] = {}
_IN_PROCESS_RECEIPT_RESERVATIONS: set[
    _InProcessReceiptReservationV1
] = set()
_IN_PROCESS_RECEIPT_REGISTRY_LOCK: Final = threading.Lock()


def _reserve_in_process_receipt_slot() -> _InProcessReceiptReservationV1:
    reservation = _InProcessReceiptReservationV1()
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        if (
            len(_IN_PROCESS_EXECUTED_RECEIPTS)
            + len(_IN_PROCESS_RECEIPT_RESERVATIONS)
            >= MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS
        ):
            _reject(
                "FAIL_RECEIPT_REGISTRY_CAPACITY",
                "the cooperative in-process receipt registry is full; explicitly "
                "release an issued receipt before starting another audit",
            )
        _IN_PROCESS_RECEIPT_RESERVATIONS.add(reservation)
    return reservation


def _cancel_in_process_receipt_reservation(
    reservation: _InProcessReceiptReservationV1,
) -> None:
    if type(reservation) is not _InProcessReceiptReservationV1:
        raise TypeError("receipt reservation must use the exact private slot type")
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        if reservation.entry_identity is not None:
            raise RuntimeError("a receipt-bearing reservation cannot be cancelled")
        _IN_PROCESS_RECEIPT_RESERVATIONS.discard(reservation)


def _rollback_in_process_receipt_issuance(
    reservation: _InProcessReceiptReservationV1,
) -> None:
    """O(1) fallback removal for one receipt that was never returned."""

    if type(reservation) is not _InProcessReceiptReservationV1:
        raise TypeError("receipt rollback requires the exact reservation slot")
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        identity = reservation.entry_identity
        if identity is None:
            return
        entry = _IN_PROCESS_EXECUTED_RECEIPTS.get(identity)
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.reservation is not reservation
            or id(entry.receipt) != identity
        ):
            raise RuntimeError("receipt reservation/registry rollback linkage drift")
        del _IN_PROCESS_EXECUTED_RECEIPTS[identity]
        reservation.entry_identity = None


def _rollback_pending_in_process_receipt_issuance(
    receipt: FormalCovertAuditReceiptV1,
    reservation: _InProcessReceiptReservationV1,
) -> None:
    """Directly remove one unreturned exact candidate without a registry scan."""

    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("pending receipt rollback requires the exact receipt DTO")
    if type(reservation) is not _InProcessReceiptReservationV1:
        raise TypeError("pending receipt rollback requires the exact reservation slot")
    identity = id(receipt)
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        entry = _IN_PROCESS_EXECUTED_RECEIPTS.get(identity)
        if entry is None and reservation.entry_identity is None:
            return
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.receipt is not receipt
            or entry.reservation is not reservation
            or (
                entry.state is not _InProcessReceiptRegistryStateV1.PENDING
                and entry.state is not _InProcessReceiptRegistryStateV1.ISSUED
            )
            or reservation.entry_identity != identity
        ):
            raise RuntimeError("receipt rollback exact-entry linkage drift")
        del _IN_PROCESS_EXECUTED_RECEIPTS[identity]
        reservation.entry_identity = None


def _register_preconstructed_formal_covert_audit_receipt_v1(
    *,
    _factory_token: object,
    _reservation: _InProcessReceiptReservationV1,
    receipt: FormalCovertAuditReceiptV1,
) -> FormalCovertAuditReceiptV1:
    """Insert one already replayed exact receipt as PENDING."""

    if _factory_token is not _RECEIPT_EXECUTION_FACTORY_TOKEN:
        raise PermissionError("formal covert transcript factory token mismatch")
    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("formal covert transcript registration requires the exact DTO")
    if type(_reservation) is not _InProcessReceiptReservationV1:
        raise TypeError("formal covert transcript registration requires an exact reservation")
    identity = id(receipt)
    entry = _InProcessReceiptRegistryEntryV1(
        receipt=receipt,
        receipt_id=receipt.receipt_id,
        reservation=_reservation,
        state=_InProcessReceiptRegistryStateV1.PENDING,
    )
    # Retain the exact issued object as well as its content ID.  Keeping only
    # ``id(receipt)`` would admit a different object after address reuse.
    # Underscore names and the module token are cooperative provenance markers,
    # not a security boundary or unforgeability mechanism.
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        if (
            _reservation not in _IN_PROCESS_RECEIPT_RESERVATIONS
            or _reservation.entry_identity is not None
        ):
            raise PermissionError("formal covert transcript reservation is absent or stale")
        if len(_IN_PROCESS_EXECUTED_RECEIPTS) >= MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS:
            raise RuntimeError("formal covert transcript registry capacity drift")
        if identity in _IN_PROCESS_EXECUTED_RECEIPTS:
            raise RuntimeError("formal covert transcript identity collision")
        try:
            _IN_PROCESS_EXECUTED_RECEIPTS[identity] = entry
            _reservation.entry_identity = identity
            _IN_PROCESS_RECEIPT_RESERVATIONS.remove(_reservation)
        except BaseException:
            if _IN_PROCESS_EXECUTED_RECEIPTS.get(identity) is entry:
                del _IN_PROCESS_EXECUTED_RECEIPTS[identity]
            _reservation.entry_identity = None
            raise
    return receipt


def _replay_formal_covert_audit_receipt_content_v1(
    receipt: FormalCovertAuditReceiptV1,
    expected_receipt_id: str,
) -> str:
    """Fully replay exact receipt content and preserve its registered ID."""

    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("receipt content replay requires the exact receipt DTO")
    if type(expected_receipt_id) is not str:
        raise TypeError("receipt content replay requires an exact registered ID")
    observed_id = receipt.receipt_id
    receipt.__post_init__()
    if (
        observed_id != expected_receipt_id
        or receipt.receipt_id != observed_id
        or receipt.receipt_id != expected_receipt_id
    ):
        raise ValueError("formal covert audit transcript content changed after construction")
    return receipt.receipt_id


def _validate_pending_formal_covert_audit_receipt_v1(
    *,
    _factory_token: object,
    _reservation: _InProcessReceiptReservationV1,
    receipt: FormalCovertAuditReceiptV1,
) -> str:
    """Producer-only full replay of one exact PENDING registry entry."""

    if _factory_token is not _RECEIPT_EXECUTION_FACTORY_TOKEN:
        raise PermissionError("formal covert pending-validator factory token mismatch")
    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("pending receipt validator requires the exact receipt DTO")
    if type(_reservation) is not _InProcessReceiptReservationV1:
        raise TypeError("pending receipt validator requires the exact reservation slot")
    identity = id(receipt)
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        entry = _IN_PROCESS_EXECUTED_RECEIPTS.get(identity)
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.receipt is not receipt
            or entry.reservation is not _reservation
            or entry.state is not _InProcessReceiptRegistryStateV1.PENDING
            or type(entry.receipt_id) is not str
            or _reservation.entry_identity != identity
        ):
            raise ValueError("producer pending receipt registry entry drift")
        registered_id = entry.receipt_id
    replayed_id = _replay_formal_covert_audit_receipt_content_v1(
        receipt,
        registered_id,
    )
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        if (
            _IN_PROCESS_EXECUTED_RECEIPTS.get(identity) is not entry
            or entry.receipt is not receipt
            or entry.receipt_id != replayed_id
            or type(entry.receipt_id) is not str
            or entry.reservation is not _reservation
            or entry.state is not _InProcessReceiptRegistryStateV1.PENDING
            or _reservation.entry_identity != identity
        ):
            raise ValueError("producer pending receipt changed during full replay")
    return replayed_id


def _commit_pending_formal_covert_audit_receipt_v1(
    *,
    _factory_token: object,
    _reservation: _InProcessReceiptReservationV1,
    receipt: FormalCovertAuditReceiptV1,
    validated_id: str,
    absolute_deadline_kind: str,
    absolute_deadline_ns: int,
) -> FormalCovertAuditReceiptV1:
    """Linearize one exact PENDING receipt after its locked final gates."""

    if _factory_token is not _RECEIPT_EXECUTION_FACTORY_TOKEN:
        raise PermissionError("formal covert pending-commit factory token mismatch")
    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("pending receipt commit requires the exact receipt DTO")
    if type(_reservation) is not _InProcessReceiptReservationV1:
        raise TypeError("pending receipt commit requires the exact reservation slot")
    if type(validated_id) is not str:
        raise TypeError("pending receipt commit requires an exact validated ID")
    if (
        type(absolute_deadline_kind) is not str
        or absolute_deadline_kind
        not in {"STATISTICS", "PARTITION_PREPARATION"}
        or type(absolute_deadline_ns) is not int
        or absolute_deadline_ns <= 0
    ):
        raise TypeError("pending receipt commit deadline arguments are invalid")
    identity = id(receipt)
    issuance = receipt.receipt_issuance_runtime_evidence
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        entry = _IN_PROCESS_EXECUTED_RECEIPTS.get(identity)
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.receipt is not receipt
            or entry.reservation is not _reservation
            or entry.state is not _InProcessReceiptRegistryStateV1.PENDING
            or type(entry.receipt_id) is not str
            or entry.receipt_id != validated_id
            or validated_id != receipt.receipt_id
            or _reservation.entry_identity != identity
            or type(issuance) is not ReceiptIssuanceRuntimeEvidenceV1
            or issuance.absolute_deadline_kind != absolute_deadline_kind
            or issuance.absolute_deadline_monotonic_nanoseconds
            != absolute_deadline_ns
        ):
            raise ValueError("pending receipt locked commit identity/evidence drift")
        if absolute_deadline_kind == "STATISTICS":
            _remaining_statistics_deadline_seconds(absolute_deadline_ns)
        else:
            _remaining_partition_preparation_deadline_seconds(
                absolute_deadline_ns
            )
        expected_snapshot = UnifiedCgroupMemorySnapshotV1(
            issuance.unified_cgroup_membership,
            issuance.unified_cgroup_memory_scope_root,
            issuance.unified_cgroup_leaf_device,
            issuance.unified_cgroup_leaf_inode,
            issuance.physical_memory_current_bytes_before_receipt_finalization,
            issuance.physical_memory_peak_bytes_before_receipt_finalization,
            issuance.physical_memory_max_bytes,
        )
        locked_parent_historical_peak_rss_bytes = _linux_peak_rss_bytes()
        locked_snapshot = _capture_unified_cgroup_memory_snapshot_v1()
        if locked_snapshot.memory_peak_bytes < expected_snapshot.memory_peak_bytes:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                "unified cgroup memory.peak decreased before locked issuance",
            )
        _require_unified_cgroup_physical_peak_v1(
            failure_code="FAIL_RECEIPT_ISSUANCE_RUNTIME_RSS",
            stage="LOCKED_POST_VALIDATION",
            snapshot=locked_snapshot,
            parent_historical_peak_rss_bytes=(
                locked_parent_historical_peak_rss_bytes
            ),
            worker_historical_peak_rss_bytes_sum=(
                issuance.diagnostic_worker_historical_peak_rss_bytes_sum
            ),
            expected_scope=expected_snapshot,
        )
        entry.state = _InProcessReceiptRegistryStateV1.ISSUED
    return receipt


def _issue_formal_covert_audit_receipt_v1(
    *,
    _factory_token: object,
    _reservation: _InProcessReceiptReservationV1,
    _absolute_deadline_kind: str,
    _absolute_deadline_ns: int,
    **values: object,
) -> FormalCovertAuditReceiptV1:
    """Construct, replay, and atomically issue a non-statistical transcript."""

    if _factory_token is not _RECEIPT_EXECUTION_FACTORY_TOKEN:
        raise PermissionError("formal covert transcript factory token mismatch")
    receipt = _construct_formal_covert_audit_receipt_v1(**values)
    _register_preconstructed_formal_covert_audit_receipt_v1(
        _factory_token=_factory_token,
        _reservation=_reservation,
        receipt=receipt,
    )
    try:
        validated_id = _validate_pending_formal_covert_audit_receipt_v1(
            _factory_token=_factory_token,
            _reservation=_reservation,
            receipt=receipt,
        )
        _commit_pending_formal_covert_audit_receipt_v1(
            _factory_token=_factory_token,
            _reservation=_reservation,
            receipt=receipt,
            validated_id=validated_id,
            absolute_deadline_kind=_absolute_deadline_kind,
            absolute_deadline_ns=_absolute_deadline_ns,
        )
    except BaseException as primary:
        try:
            _rollback_pending_in_process_receipt_issuance(receipt, _reservation)
        except BaseException as cleanup_error:
            _record_cleanup_failure_without_replacing_primary(
                primary,
                cleanup_error,
                "non-statistical direct receipt rollback",
            )
        raise
    return receipt


def _construct_formal_covert_audit_receipt_v1(
    **values: object,
) -> FormalCovertAuditReceiptV1:
    """Construct and fully replay one unissued receipt candidate."""

    defaults: dict[str, object] = {
        "authority_origin_authenticated": False,
        "authority_independence_verified": False,
        "actual_corpus_audit": False,
        "claim_level": CLAIM_LEVEL,
        "positive_decision_claim": None,
        "randomization_assumption": RANDOMIZATION_ASSUMPTION,
        "transcript_authentication": TRANSCRIPT_AUTHENTICATION,
    }
    expected = set(FormalCovertAuditReceiptV1.__dataclass_fields__) - {"receipt_id"}
    observed = set(values) | set(defaults)
    unknown = set(values) - expected
    missing = expected - observed
    if unknown or missing:
        raise TypeError(
            "formal covert transcript factory field mismatch: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    receipt = object.__new__(FormalCovertAuditReceiptV1)
    for name in expected:
        object.__setattr__(
            receipt,
            name,
            values[name] if name in values else defaults[name],
        )
    receipt.__post_init__()
    return receipt


def release_formal_covert_audit_receipt_v1(
    receipt: FormalCovertAuditReceiptV1,
) -> str:
    """Explicitly end process-local validation for one exact issued object."""

    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("receipt release requires the exact receipt DTO")
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        identity = id(receipt)
        entry = _IN_PROCESS_EXECUTED_RECEIPTS.get(identity)
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.receipt is not receipt
        ):
            raise ValueError("receipt is not live in the cooperative in-process registry")
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.state is not _InProcessReceiptRegistryStateV1.ISSUED
            or type(entry.receipt_id) is not str
            or type(entry.reservation) is not _InProcessReceiptReservationV1
            or entry.reservation.entry_identity != identity
        ):
            raise ValueError("only an exact ISSUED receipt can be released")
        receipt_id = entry.receipt_id
        # The registry is the issuance authority and capacity account.  Delete
        # it first; an asynchronous failure while clearing the now-unreachable
        # backlink cannot leave a live/publicly valid entry or consume capacity.
        del _IN_PROCESS_EXECUTED_RECEIPTS[identity]
        entry.reservation.entry_identity = None
        return receipt_id


@dataclass(frozen=True, slots=True)
class _FeatureColumn:
    aliases: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Partition:
    masks: tuple[int, ...]
    counts: tuple[int, ...]


def _deep_object_graph_bytes(value: object) -> int:
    """Pinned-runtime deep-size observation with cycle/shared-object handling."""

    total = 0
    seen: set[int] = set()
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            stack.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, row.name) for row in dataclass_fields(item))
    return total


def _measured_feature_matrix_bytes_bound(
    rows: int,
    columns: tuple[_FeatureColumn, ...],
) -> int:
    floor = rows * len(columns) * 72
    return max(
        floor,
        DEEP_SIZE_SAFETY_MULTIPLIER * _deep_object_graph_bytes(columns),
    )


def _measured_envelope_decode_bytes_bound(
    rows: int,
    envelopes: tuple[bytes, ...],
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
) -> int:
    floor = rows * ENVELOPE_BYTES * 2
    return max(
        floor,
        DEEP_SIZE_SAFETY_MULTIPLIER
        * _deep_object_graph_bytes((envelopes, decoded)),
    )


class _Sha256Stream:
    __slots__ = ("seed", "counter", "buffer")

    def __init__(self, seed: bytes) -> None:
        self.seed = seed
        self.counter = 0
        self.buffer = b""

    def _u64(self) -> int:
        if len(self.buffer) < 8:
            self.buffer += hashlib.sha256(
                self.seed + self.counter.to_bytes(8, "big")
            ).digest()
            self.counter += 1
        value = int.from_bytes(self.buffer[:8], "big")
        self.buffer = self.buffer[8:]
        return value

    def randbelow(self, bound: int) -> int:
        if type(bound) is not int or bound <= 0:
            raise ValueError("random bound must be a positive exact integer")
        limit = (1 << 64) - ((1 << 64) % bound)
        for _ in range(MAXIMUM_SHUFFLE_REJECTION_DRAWS):
            value = self._u64()
            if value < limit:
                return value % bound
        _reject(
            "FAIL_SHUFFLE_REJECTION_BOUND",
            "SHA-256 shuffle stream exceeded 4096 rejection draws",
        )


def _shuffle(values: list[object], seed: bytes) -> None:
    stream = _Sha256Stream(seed)
    for index in range(len(values) - 1, 0, -1):
        swap = stream.randbelow(index + 1)
        values[index], values[swap] = values[swap], values[index]


def _fresh_permutation_seed() -> bytes:
    """Draw the auditor-owned schedule seed; it is never a public argument."""

    seed = os.urandom(32)
    if type(seed) is not bytes or len(seed) != 32:
        raise RuntimeError("operating-system random source returned an invalid seed")
    return seed


def _permutation_seed_commitment(seed: bytes) -> str:
    if type(seed) is not bytes or len(seed) != 32:
        raise TypeError("permutation seed must be exact 32-byte auditor-owned bytes")
    return "sha256:" + hashlib.sha256(_PERMUTATION_DOMAIN + seed).hexdigest()


def _target_label(row: FormalAuthorityLabelRowV1, target: str) -> str:
    if target == "family":
        return row.family
    if target == "binding":
        return row.binding
    if target == "scale":
        return row.scale
    if target == "answerable_vs_abstain":
        return row.answerability.value
    if target == "joint_decision_class":
        return row.joint_decision_class
    raise RuntimeError("unknown frozen target")


def _target_stratum(
    row: FormalAuthorityLabelRowV1,
    target: str,
    eligibility_signature: tuple[int, int, int, int, int, int],
) -> tuple[str, ...]:
    if target == "family":
        base = (row.case_type, row.scale)
    elif target == "scale":
        base = (row.case_type, row.family)
    elif target == "binding":
        base = (row.case_type, row.family, row.scale)
    elif target == "answerable_vs_abstain":
        base = (row.family, row.scale)
    elif target == "joint_decision_class":
        base = (row.case_type,)
    else:
        raise RuntimeError("unknown frozen target")
    if (
        type(eligibility_signature) is not tuple
        or len(eligibility_signature) != 6
        or any(type(value) is not int or value < 0 for value in eligibility_signature)
    ):
        raise TypeError("target stratum eligibility signature drift")
    if target in _ELIGIBILITY_QUOTA_TARGETS:
        token = "eligibility:" + ",".join(
            str(value) for value in eligibility_signature
        )
        return (*base, token)
    return base


def _sufficiency(
    authority: FormalAuditAuthorityV1,
    target: str,
    eligibility_signatures: tuple[
        tuple[int, int, int, int, int, int], ...
    ],
) -> TargetSufficiencyV1:
    labels = tuple(_target_label(row, target) for row in authority.rows)
    if len(eligibility_signatures) != len(authority.rows):
        raise ValueError("sufficiency eligibility signature count drift")
    strata = tuple(
        _target_stratum(row, target, signature)
        for row, signature in zip(
            authority.rows,
            eligibility_signatures,
            strict=True,
        )
    )
    label_counts = Counter(labels)
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for label, stratum in zip(labels, strata, strict=True):
        groups[stratum].append(label)
    minimum_class = min(label_counts.values()) if label_counts else 0
    minimum_stratum = min(map(len, groups.values())) if groups else 0
    mutable = sum(len(set(group)) >= 2 for group in groups.values())
    movable_classes = {
        label
        for group in groups.values()
        if len(set(group)) >= 2
        for label in group
    }
    if len(label_counts) < 2:
        reason = "TARGET_HAS_FEWER_THAN_TWO_CLASSES"
    elif minimum_class < 2:
        reason = "LOO_CLASS_HAS_FEWER_THAN_TWO_ROWS"
    elif minimum_stratum < 2:
        reason = "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_ROWS"
    elif mutable != len(groups):
        reason = "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_TARGET_CLASSES"
    elif movable_classes != set(label_counts):
        reason = "TARGET_CLASS_IS_IMMUTABLE_WITHIN_STRATA"
    elif len(label_counts) > MAXIMUM_LABEL_CLASSES:
        reason = "TARGET_CLASS_BUDGET_EXCEEDED"
    else:
        reason = "SUFFICIENT"
    return TargetSufficiencyV1(
        target=target,
        label_class_count=len(label_counts),
        minimum_class_count=minimum_class,
        stratum_count=len(groups),
        minimum_stratum_size=minimum_stratum,
        mutable_stratum_count=mutable,
        sufficient=reason == "SUFFICIENT",
        reason=reason,
    )


_ELIGIBILITY_SIGNATURE_SCHEMA: Final = (
    "observation_role_candidate_scale_presence_count_order_independent_v1"
)
_ELIGIBILITY_KEYS: Final = (
    "observation_ids",
    "role_candidate_ids",
    "scale_ids",
)
_ELIGIBILITY_QUOTA_TARGETS: Final = (
    "family",
    "scale",
    "answerable_vs_abstain",
)


def _eligibility_structural_signature(
    decoded: DecodedFormalEnvelopeV1,
) -> tuple[int, int, int, int, int, int]:
    candidates: list[dict[str, object]] = []
    for leaf in decoded.channel_leaves:
        if leaf.group != "eligibility_mask":
            continue
        value = _resolve(decoded.payload, leaf.path)
        if type(value) is dict and set(value).issubset(_ELIGIBILITY_KEYS):
            candidates.append(value)
    if len(candidates) != 1:
        _reject(
            "FAIL_ELIGIBILITY_MASK_SEMANTICS",
            "each row must project exactly one structural eligibility-mask object",
        )
    mask = candidates[0]
    signature: list[int] = []
    for name in _ELIGIBILITY_KEYS:
        if name not in mask:
            signature.extend((0, 0))
            continue
        values = mask[name]
        if (
            type(values) is not list
            or len(values) > 4_096
            or any(type(item) is not str or not item for item in values)
            or len(set(values)) != len(values)
        ):
            _reject(
                "FAIL_ELIGIBILITY_MASK_SEMANTICS",
                f"eligibility {name} must be a bounded unique string array",
            )
        signature.extend((1, len(values)))
    if not any(signature[::2]):
        _reject(
            "FAIL_ELIGIBILITY_MASK_SEMANTICS",
            "eligibility mask cannot omit all three frozen namespaces",
        )
    return tuple(signature)  # type: ignore[return-value]


def _eligibility_mask_quota(
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
    authority: FormalAuditAuthorityV1,
    signatures: tuple[tuple[int, int, int, int, int, int], ...],
) -> EligibilityMaskQuotaReceiptV1:
    if signatures != tuple(
        _eligibility_structural_signature(item) for item in decoded
    ):
        _reject(
            "FAIL_ELIGIBILITY_MASK_SEMANTICS",
            "eligibility signature snapshot changed before quota replay",
        )
    signature_order = tuple(sorted(set(signatures)))
    signature_totals = Counter(signatures)
    total = len(signatures)
    target_rows: list[EligibilityMaskTargetQuotaV1] = []
    for target in _ELIGIBILITY_QUOTA_TARGETS:
        labels = tuple(_target_label(row, target) for row in authority.rows)
        label_order = tuple(sorted(set(labels), key=lambda item: item.encode("utf-8")))
        label_totals = Counter(labels)
        cells = Counter(zip(signatures, labels, strict=True))
        differences = tuple(
            abs(cells[(signature, label)] * total - signature_totals[signature] * label_totals[label])
            for signature in signature_order
            for label in label_order
        )
        maximum = max(differences, default=0)
        target_rows.append(
            EligibilityMaskTargetQuotaV1(
                target=target,
                class_count=len(label_order),
                structural_signature_count=len(signature_order),
                checked_integer_cells=len(differences),
                maximum_absolute_cross_product_difference=maximum,
                exact_quota_matched=maximum == 0,
                private_contingency_root=stable_hash(
                    {
                        "target": target,
                        "signature_order": signature_order,
                        "label_order": label_order,
                        "signature_totals": tuple(
                            (item, signature_totals[item]) for item in signature_order
                        ),
                        "label_totals": tuple(
                            (item, label_totals[item]) for item in label_order
                        ),
                        "cells": tuple(
                            (signature, label, cells[(signature, label)])
                            for signature in signature_order
                            for label in label_order
                        ),
                    },
                    prefix="phase2b_formal_eligibility_private_contingency_",
                ),
            )
        )
    receipt = EligibilityMaskQuotaReceiptV1(
        signature_schema=_ELIGIBILITY_SIGNATURE_SCHEMA,
        public_signature_root=stable_hash(
            {"schema": _ELIGIBILITY_SIGNATURE_SCHEMA, "signatures": signatures},
            prefix="phase2b_formal_eligibility_public_signatures_",
        ),
        targets=tuple(target_rows),
        exact_quota_matched=all(item.exact_quota_matched for item in target_rows),
    )
    if not receipt.exact_quota_matched:
        _reject(
            "FAIL_ELIGIBILITY_MASK_QUOTA",
            "exact integer class-by-signature quota equality failed; receipt="
            + receipt.receipt_id,
        )
    return receipt


def _scan_absent_fields(payload: object) -> None:
    stack = [payload]
    forbidden = set(MUST_BE_ABSENT_FIELDS)
    while stack:
        value = stack.pop()
        if type(value) is dict:
            overlap = forbidden.intersection(value)
            if overlap:
                _reject(
                    "FAIL_MUST_BE_ABSENT",
                    "forbidden public fields present: " + ",".join(sorted(overlap)),
                )
            stack.extend(value.values())
        elif type(value) is list:
            stack.extend(value)


def _scan_uuid_paths(payload: object) -> set[JsonPath]:
    found: set[JsonPath] = set()
    stack: list[tuple[object, JsonPath]] = [(payload, ())]
    while stack:
        value, path = stack.pop()
        if type(value) is dict:
            stack.extend((item, (*path, key)) for key, item in value.items())
        elif type(value) is list:
            stack.extend((item, (*path, index)) for index, item in enumerate(value))
        elif type(value) is str and _UUID_SHAPED.fullmatch(value) is not None:
            if _UUID4.fullmatch(value) is None:
                _reject("FAIL_UUID_CANONICAL", f"noncanonical UUID-shaped leaf at {_path_text(path)}")
            found.add(path)
    return found


def _validate_decoded(decoded: DecodedFormalEnvelopeV1, wire: FormalWireAdapterV1) -> None:
    _scan_absent_fields(decoded.payload)
    canonical = wire.canonical_payload_bytes(decoded.payload)
    if type(canonical) is not bytes or canonical != decoded.payload_bytes:
        _reject("FAIL_WIRE_CANONICAL_REPLAY", "decoded payload is not its exact canonical byte encoding")
    manifested = {item.path for item in decoded.namespace_leaves}
    if _scan_uuid_paths(decoded.payload) != manifested:
        _reject("FAIL_UUID_NAMESPACE_CLOSURE", "UUID-shaped leaves and namespace manifest differ")
    namespace_by_uuid: dict[str, str] = {}
    for item in decoded.namespace_leaves:
        prior = namespace_by_uuid.setdefault(item.public_uuid, item.namespace)
        if prior != item.namespace:
            _reject("FAIL_UUID_NAMESPACE_ALIAS", "one UUID aliases multiple namespaces")
    seen_channel_paths: set[tuple[str, JsonPath]] = set()
    for item in decoded.channel_leaves:
        key = (item.group, item.path)
        if key in seen_channel_paths:
            _reject("FAIL_CHANNEL_FIELD_DUPLICATE", "channel projection repeats a field")
        seen_channel_paths.add(key)
        observed = stable_hash(
            _resolve(decoded.payload, item.path),
            prefix="phase2b_channel_leaf_",
        )
        if item.canonical_value != observed:
            _reject(
                "FAIL_CHANNEL_FIELD_REPLAY",
                f"channel projection value drift at {_path_text(item.path)}",
            )
    namespace_paths = {item.path for item in decoded.namespace_leaves}
    channel_paths = {item.path for item in decoded.channel_leaves}
    owned_paths = {item.path for item in decoded.channel_only_leaf_owners}
    if not namespace_paths.issubset(owned_paths):
        _reject(
            "FAIL_CHANNEL_CLASSIFICATION_COVERAGE",
            "a UUID channel leaf lacks an exactly owned channel-only declaration",
        )
    for owner in decoded.channel_only_leaf_owners:
        if owner.path not in namespace_paths and owner.path not in channel_paths:
            _reject(
                "FAIL_CHANNEL_CLASSIFICATION_COVERAGE",
                "a declared channel-only concrete leaf lacks a field-level projection",
            )
        if owner.owner_group == "uuid_leaf" and owner.path not in namespace_paths:
            _reject(
                "FAIL_CHANNEL_CLASSIFICATION_OWNER",
                "uuid_leaf owner does not resolve through the namespace manifest",
            )
    required_groups = {
        "missingness",
        "provenance",
        "record_order",
        "slot_order",
        "eligibility_mask",
        "control_metadata",
        "descriptor_metadata",
    }
    present_groups = {item.group for item in decoded.channel_leaves}
    if not required_groups.issubset(present_groups):
        _reject(
            "FAIL_CHANNEL_FIELD_COVERAGE",
            "decoded channel projection omits a mandatory field-level group",
        )


def _uuid_bytes(value: str) -> bytes:
    return bytes.fromhex(value.replace("-", ""))


def _uuid_raw_features(
    namespace: str,
    path: JsonPath,
    values: tuple[str | None, ...],
) -> list[tuple[str, tuple[str, ...]]]:
    path_id = _path_text(path)
    # The zero block is an internal missing-value encoding, never a public
    # UUID.  Canonical UUIDv4 validation makes that domain separation exact.
    blobs = tuple(
        b"\x00" * 16 if value is None else _uuid_bytes(value)
        for value in values
    )
    rows = []
    for bit_index in range(128):
        semantic = f"uuid_leaf/{namespace}/{path_id}/128_individual_bits/bit_{bit_index:03d}"
        aliases = [semantic]
        if bit_index < 8:
            aliases.append(f"uuid_leaf/{namespace}/{path_id}/first_8_16_32_bits/first_8/bit_{bit_index:03d}")
        if bit_index < 16:
            aliases.append(f"uuid_leaf/{namespace}/{path_id}/first_8_16_32_bits/first_16/bit_{bit_index:03d}")
        if bit_index < 32:
            aliases.append(f"uuid_leaf/{namespace}/{path_id}/first_8_16_32_bits/first_32/bit_{bit_index:03d}")
        if bit_index >= 120:
            aliases.append(f"uuid_leaf/{namespace}/{path_id}/last_8_16_32_bits/last_8/bit_{bit_index:03d}")
        if bit_index >= 112:
            aliases.append(f"uuid_leaf/{namespace}/{path_id}/last_8_16_32_bits/last_16/bit_{bit_index:03d}")
        if bit_index >= 96:
            aliases.append(f"uuid_leaf/{namespace}/{path_id}/last_8_16_32_bits/last_32/bit_{bit_index:03d}")
        byte_index, within = divmod(bit_index, 8)
        column = tuple(str((blob[byte_index] >> (7 - within)) & 1) for blob in blobs)
        rows.append(("\x1f".join(aliases), column))
    integers = tuple(int.from_bytes(blob, "big") for blob in blobs)
    rows.append((f"uuid_leaf/{namespace}/{path_id}/hamming_weight", tuple(str(value.bit_count()) for value in integers)))
    for modulus in (3, 5, 7, 11, 13):
        rows.append((f"uuid_leaf/{namespace}/{path_id}/integer_mod_3_5_7_11_13/mod_{modulus}", tuple(str(value % modulus) for value in integers)))
    for nibble in range(16):
        rows.append(
            (
                f"uuid_leaf/{namespace}/{path_id}/hex_character_histogram/bin_{nibble:x}",
                tuple(
                    str(sum(character == f"{nibble:x}" for character in blob.hex()))
                    for blob in blobs
                ),
            )
        )
    return rows


def _bounded_byte_features(
    group: str,
    blobs: tuple[bytes, ...],
) -> list[tuple[str, tuple[str, ...]]]:
    if not blobs or any(len(blob) != 32 for blob in blobs):
        raise ValueError("bounded byte audit needs aligned 32-byte prefix/suffix blocks")
    rows: list[tuple[str, tuple[str, ...]]] = []
    for offset in range(32):
        side = "prefix" if offset < 16 else "suffix"
        side_offset = offset if offset < 16 else offset - 16
        values = tuple(blob[offset] for blob in blobs)
        base = f"{group}/{side}/byte_{side_offset:02d}"
        # Raw categorical is the primary byte hypothesis.  Supplemental
        # decompositions are separate Holm hypotheses and are not claimed to
        # be statistically equivalent to the raw-byte test.
        rows.append((base + "/raw_byte_categorical", tuple(str(value) for value in values)))
        bit_columns = tuple(
            tuple((value >> (7 - bit)) & 1 for value in values)
            for bit in range(8)
        )
        for bit, column in enumerate(bit_columns):
            rows.append((base + f"/individual_bit_{bit}", tuple(str(value) for value in column)))
        for first in range(8):
            for second in range(first + 1, 8):
                rows.append(
                    (
                        base + f"/pairwise_xor_{first}_{second}",
                        tuple(
                            str(left ^ right)
                            for left, right in zip(
                                bit_columns[first],
                                bit_columns[second],
                                strict=True,
                            )
                        ),
                    )
                )
        rows.append((base + "/high_nibble_categorical", tuple(str(value >> 4) for value in values)))
        rows.append((base + "/low_nibble_categorical", tuple(str(value & 0x0F) for value in values)))
        rows.append((base + "/byte_hamming_weight", tuple(str(value.bit_count()) for value in values)))
    return rows


def _provenance_digest(value: object) -> bytes:
    if (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    ):
        return bytes.fromhex(value[7:])
    return hashlib.sha256(stable_hash(value).encode("utf-8")).digest()


_DESCRIPTOR_KINDS: Final = ("natural_language", "symbolic", "mixed")
_DESCRIPTOR_ASCII_CLASSES: Final = (
    "upper",
    "lower",
    "digit",
    "space",
    "tab",
    "linebreak",
    "punctuation",
    "other_ascii",
    "non_ascii",
)
_DESCRIPTOR_UNICODE_CATEGORIES: Final = tuple(
    major + minor
    for major, minors in (
        ("L", "ultmo"),
        ("M", "nce"),
        ("N", "dlo"),
        ("P", "cdseifo"),
        ("S", "mcko"),
        ("Z", "slp"),
        ("C", "cfson"),
    )
    for minor in minors
)
_DESCRIPTOR_TOKEN_SHAPES: Final = (
    "upper_word",
    "lower_word",
    "title_word",
    "mixed_word",
    "digits",
    "alnum",
    "punctuation",
    "symbol",
    "other",
)
_DESCRIPTOR_TOKEN_LENGTH_BUCKETS: Final = (
    "1",
    "2",
    "3",
    "4",
    "5_8",
    "9_16",
    "17_32",
    "33_plus",
)
_UUID_CONCRETE_FEATURES_PER_PATH: Final = 151
_UUID_RAW_SEMANTIC_ALIASES_PER_PATH: Final = 263
_BOUNDED_BYTE_FEATURES_PER_BLOCK: Final = 32 * (1 + 8 + 28 + 2 + 1)
_DESCRIPTOR_FEATURES_PER_SEGMENT: Final = (
    22
    + len(_DESCRIPTOR_ASCII_CLASSES)
    + len(_DESCRIPTOR_UNICODE_CATEGORIES)
    + len(_DESCRIPTOR_TOKEN_SHAPES)
    + len(_DESCRIPTOR_TOKEN_LENGTH_BUCKETS)
)
_FIXED_NON_PATH_FEATURES: Final = 3 + 256 + 2 * _BOUNDED_BYTE_FEATURES_PER_BLOCK


@dataclass(frozen=True, slots=True)
class _FeatureCardinalityProjectionV1:
    rows: int
    uuid_union_path_count: int
    channel_union_path_count: int
    generic_channel_union_path_count: int
    provenance_union_path_count: int
    descriptor_union_path_count: int
    descriptor_segment_position_count: int
    eligibility_absent_feature_count: int
    descriptor_source_utf8_bytes: int
    present_generic_channel_feature_utf8_bytes: int
    semantic_path_repetition_utf8_bytes: int
    projected_concrete_feature_count: int
    projected_raw_semantic_feature_count: int
    projected_feature_matrix_bytes_bound: int
    projection_root: str = field(init=False)

    def __post_init__(self) -> None:
        integer_fields = tuple(
            row.name
            for row in dataclass_fields(self)
            if row.name != "projection_root"
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise TypeError("feature-cardinality projection fields must be exact integers")
        if not 1 <= self.rows <= MAXIMUM_ROWS or any(
            getattr(self, name) < 0 for name in integer_fields if name != "rows"
        ):
            raise ValueError("feature-cardinality projection contains an invalid count")
        expected_features = (
            self.uuid_union_path_count * _UUID_CONCRETE_FEATURES_PER_PATH
            + self.generic_channel_union_path_count * 2
            + self.provenance_union_path_count
            * (1 + _BOUNDED_BYTE_FEATURES_PER_BLOCK)
            + self.descriptor_union_path_count * 4
            + self.descriptor_segment_position_count
            * _DESCRIPTOR_FEATURES_PER_SEGMENT
            + self.eligibility_absent_feature_count
            + _FIXED_NON_PATH_FEATURES
        )
        expected_raw = expected_features + self.uuid_union_path_count * (
            _UUID_RAW_SEMANTIC_ALIASES_PER_PATH
            - _UUID_CONCRETE_FEATURES_PER_PATH
        )
        expected_bytes = (
            self.rows
            * expected_features
            * PREEXTRACTION_FEATURE_CELL_BYTES_BOUND
            + expected_raw * PREEXTRACTION_FEATURE_ALIAS_FIXED_BYTES_BOUND
            + self.semantic_path_repetition_utf8_bytes
            * DEEP_SIZE_SAFETY_MULTIPLIER
            + self.present_generic_channel_feature_utf8_bytes
            * PREEXTRACTION_PRESENT_VALUE_SAFETY_MULTIPLIER
            + self.descriptor_source_utf8_bytes
            * PREEXTRACTION_DESCRIPTOR_UTF8_EXPANSION_MULTIPLIER
        )
        if (
            self.projected_concrete_feature_count != expected_features
            or self.projected_raw_semantic_feature_count != expected_raw
            or self.projected_feature_matrix_bytes_bound != expected_bytes
        ):
            raise ValueError("feature-cardinality projection formula drift")
        object.__setattr__(
            self,
            "projection_root",
            stable_hash(
                tuple(
                    (name, getattr(self, name))
                    for name in integer_fields
                ),
                prefix="phase2b_formal_covert_feature_projection_",
            ),
        )


def _parent_result_working_bytes_bound(features: int) -> int:
    if type(features) is not int or features < 1:
        raise ValueError("parent-result projection needs a positive feature count")
    return (
        features
        * len(TARGETS)
        * PARENT_RESULT_WORKING_BYTES_PER_HYPOTHESIS
        + len(TARGETS) * PARENT_RESULT_WORKING_FIXED_BYTES_PER_TARGET
    )


def _feature_cardinality_projection(
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
) -> _FeatureCardinalityProjectionV1:
    if type(decoded) is not tuple or not decoded:
        raise TypeError("feature projection needs a nonempty exact decoded tuple")
    uuid_keys: set[tuple[str, tuple[str | int, ...]]] = set()
    channel_keys: set[tuple[str, tuple[str | int, ...]]] = set()
    generic_keys: set[tuple[str, tuple[str | int, ...]]] = set()
    provenance_paths: set[tuple[str | int, ...]] = set()
    maximum_segments: dict[tuple[str | int, ...], int] = {}
    descriptor_source_utf8_bytes = 0
    present_generic_channel_feature_utf8_bytes = 0
    eligibility_present = False

    def reject_if_over_feature_cap(*, include_absent_eligibility: bool = False) -> None:
        projected = (
            len(uuid_keys) * _UUID_CONCRETE_FEATURES_PER_PATH
            + len(generic_keys) * 2
            + len(provenance_paths) * (1 + _BOUNDED_BYTE_FEATURES_PER_BLOCK)
            + len(maximum_segments) * 4
            + sum(maximum_segments.values()) * _DESCRIPTOR_FEATURES_PER_SEGMENT
            + int(include_absent_eligibility and not eligibility_present)
            + _FIXED_NON_PATH_FEATURES
        )
        if projected > MAXIMUM_FEATURES:
            _reject(
                "FAIL_FEATURE_CARDINALITY_PROJECTION",
                "decoded union paths project more than 32,768 concrete features",
            )

    for item in decoded:
        for leaf in item.namespace_leaves:
            key = (leaf.namespace, leaf.path)
            if key not in uuid_keys:
                uuid_keys.add(key)
                reject_if_over_feature_cap()
        for leaf in item.channel_leaves:
            channel_keys.add((leaf.group, leaf.path))
            if leaf.group == "descriptor_metadata":
                if leaf.path not in maximum_segments:
                    maximum_segments[leaf.path] = 0
                    reject_if_over_feature_cap()
                try:
                    descriptor = _validate_descriptor_object(
                        _resolve(item.payload, leaf.path)
                    )
                except (TypeError, ValueError) as exc:
                    _reject(
                        "FAIL_DESCRIPTOR_METADATA_SHAPE",
                        f"descriptor metadata at {_path_text(leaf.path)} failed: {exc}",
                    )
                segments = descriptor["segments"]
                if type(segments) is not list:
                    _reject(
                        "FAIL_DESCRIPTOR_METADATA_SHAPE",
                        "validated descriptor segments changed exact type",
                    )
                previous_maximum = maximum_segments[leaf.path]
                if len(segments) > previous_maximum:
                    maximum_segments[leaf.path] = len(segments)
                    reject_if_over_feature_cap()
                descriptor_source_utf8_bytes += sum(
                    len(segment.encode("utf-8")) for segment in segments
                )
            elif leaf.group == "provenance":
                if leaf.path not in provenance_paths:
                    provenance_paths.add(leaf.path)
                    reject_if_over_feature_cap()
            else:
                key = (leaf.group, leaf.path)
                if key not in generic_keys:
                    generic_keys.add(key)
                    if leaf.group == "eligibility_mask":
                        eligibility_present = True
                    reject_if_over_feature_cap()
                if type(leaf.feature_value) is not str:
                    _reject(
                        "FAIL_FEATURE_COLUMN_SHAPE",
                        "validated generic channel feature value changed type",
                    )
                present_generic_channel_feature_utf8_bytes += len(
                    leaf.feature_value.encode("utf-8")
                )

    descriptor_segment_positions = sum(maximum_segments.values())
    descriptor_paths = set(maximum_segments)
    eligibility_absent = int(not eligibility_present)
    reject_if_over_feature_cap(include_absent_eligibility=True)
    projected_features = (
        len(uuid_keys) * _UUID_CONCRETE_FEATURES_PER_PATH
        + len(generic_keys) * 2
        + len(provenance_paths) * (1 + _BOUNDED_BYTE_FEATURES_PER_BLOCK)
        + len(descriptor_paths) * 4
        + descriptor_segment_positions * _DESCRIPTOR_FEATURES_PER_SEGMENT
        + eligibility_absent
        + _FIXED_NON_PATH_FEATURES
    )
    projected_raw = projected_features + len(uuid_keys) * (
        _UUID_RAW_SEMANTIC_ALIASES_PER_PATH
        - _UUID_CONCRETE_FEATURES_PER_PATH
    )
    path_repetitions = 0
    for namespace, path in uuid_keys:
        path_repetitions += (
            len(namespace.encode("utf-8")) + len(_path_text(path).encode("utf-8"))
        ) * _UUID_RAW_SEMANTIC_ALIASES_PER_PATH
    for group, path in generic_keys:
        path_repetitions += (
            len(group.encode("utf-8")) + len(_path_text(path).encode("utf-8"))
        ) * 2
    for path in provenance_paths:
        path_repetitions += len(_path_text(path).encode("utf-8")) * (
            1 + _BOUNDED_BYTE_FEATURES_PER_BLOCK
        )
    for path in descriptor_paths:
        path_repetitions += len(_path_text(path).encode("utf-8")) * (
            4
            + maximum_segments[path] * _DESCRIPTOR_FEATURES_PER_SEGMENT
        )
    projected_bytes = (
        len(decoded)
        * projected_features
        * PREEXTRACTION_FEATURE_CELL_BYTES_BOUND
        + projected_raw * PREEXTRACTION_FEATURE_ALIAS_FIXED_BYTES_BOUND
        + path_repetitions * DEEP_SIZE_SAFETY_MULTIPLIER
        + present_generic_channel_feature_utf8_bytes
        * PREEXTRACTION_PRESENT_VALUE_SAFETY_MULTIPLIER
        + descriptor_source_utf8_bytes
        * PREEXTRACTION_DESCRIPTOR_UTF8_EXPANSION_MULTIPLIER
    )
    projection = _FeatureCardinalityProjectionV1(
        rows=len(decoded),
        uuid_union_path_count=len(uuid_keys),
        channel_union_path_count=len(channel_keys),
        generic_channel_union_path_count=len(generic_keys),
        provenance_union_path_count=len(provenance_paths),
        descriptor_union_path_count=len(descriptor_paths),
        descriptor_segment_position_count=descriptor_segment_positions,
        eligibility_absent_feature_count=eligibility_absent,
        descriptor_source_utf8_bytes=descriptor_source_utf8_bytes,
        present_generic_channel_feature_utf8_bytes=(
            present_generic_channel_feature_utf8_bytes
        ),
        semantic_path_repetition_utf8_bytes=path_repetitions,
        projected_concrete_feature_count=projected_features,
        projected_raw_semantic_feature_count=projected_raw,
        projected_feature_matrix_bytes_bound=projected_bytes,
    )
    if projection.projected_concrete_feature_count > MAXIMUM_FEATURES:
        _reject(
            "FAIL_FEATURE_CARDINALITY_PROJECTION",
            "decoded union paths project more than 32,768 concrete features",
        )
    return projection


def _validate_descriptor_object(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"kind", "segments"}:
        raise ValueError("semantic descriptor must be the exact closed object")
    kind = value["kind"]
    segments = value["segments"]
    if type(kind) is not str or kind not in _DESCRIPTOR_KINDS:
        raise ValueError("semantic descriptor kind is outside the frozen enum")
    if type(segments) is not list or not 1 <= len(segments) <= 16:
        raise ValueError("semantic descriptor segments are outside the frozen shape")
    total = 0
    for segment in segments:
        if type(segment) is not str or not segment:
            raise ValueError("semantic descriptor segment must be an exact nonempty string")
        try:
            encoded = segment.encode("utf-8", errors="strict")
            canonicalize_rfc8785_v1(segment)
        except (UnicodeError, RFC8785Error) as exc:
            raise ValueError(
                "semantic descriptor segment is outside the RFC-8785 string domain"
            ) from exc
        if unicodedata.normalize("NFC", segment) != segment:
            raise ValueError("semantic descriptor segment must already be NFC")
        if any(
            unicodedata.category(character)
            in {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
            for character in segment
        ):
            raise ValueError(
                "semantic descriptor segment contains an invisible/private/undefined code point"
            )
        if any(_descriptor_is_default_ignorable(character) for character in segment):
            raise ValueError(
                "semantic descriptor segment contains a frozen default-ignorable code point"
            )
        if any(_descriptor_is_noncharacter(character) for character in segment):
            raise ValueError("semantic descriptor segment contains a Unicode noncharacter")
        if (
            _DESCRIPTOR_UUID_TOKEN.search(segment)
            or _DESCRIPTOR_SHA256_TOKEN.search(segment)
            or _DESCRIPTOR_HEX_DIGEST_TOKEN.search(segment)
        ):
            raise ValueError("semantic descriptor segment contains an identifier/digest token")
        if len(encoded) > 1_024:
            raise ValueError("semantic descriptor segment exceeds its UTF-8 byte bound")
        total += len(encoded)
    if total > 4_096:
        raise ValueError("semantic descriptor exceeds its total UTF-8 byte bound")
    return value


def _descriptor_is_default_ignorable(character: str) -> bool:
    point = ord(character)
    return any(
        lower <= point <= upper
        for lower, upper in _DESCRIPTOR_DEFAULT_IGNORABLE_RANGES
    )


def _descriptor_is_noncharacter(character: str) -> bool:
    point = ord(character)
    return 0xFDD0 <= point <= 0xFDEF or point & 0xFFFF in (0xFFFE, 0xFFFF)


def _descriptor_ascii_class(character: str) -> str:
    if ord(character) > 127:
        return "non_ascii"
    if character.isupper():
        return "upper"
    if character.islower():
        return "lower"
    if character.isdigit():
        return "digit"
    if character == " ":
        return "space"
    if character == "\t":
        return "tab"
    if character in "\r\n":
        return "linebreak"
    if unicodedata.category(character).startswith("P"):
        return "punctuation"
    return "other_ascii"


def _descriptor_boundary_shape(character: str | None) -> str:
    if character is None:
        return "EMPTY"
    return _descriptor_ascii_class(character) + ":" + unicodedata.category(character)


def _descriptor_token_shape(token: str) -> str:
    categories = {unicodedata.category(character)[0] for character in token}
    if token.isalpha():
        if token.isupper():
            return "upper_word"
        if token.islower():
            return "lower_word"
        if token.istitle():
            return "title_word"
        return "mixed_word"
    if token.isdigit():
        return "digits"
    if token.isalnum():
        return "alnum"
    if categories == {"P"}:
        return "punctuation"
    if categories == {"S"}:
        return "symbol"
    return "other"


def _descriptor_token_length_bucket(length: int) -> str:
    if length <= 4:
        return str(length)
    if length <= 8:
        return "5_8"
    if length <= 16:
        return "9_16"
    if length <= 32:
        return "17_32"
    return "33_plus"


def _descriptor_lexical_tokens(text: str) -> tuple[str, ...]:
    """Split on representation boundaries without retaining lexical identity."""

    result: list[str] = []
    current: list[str] = []
    for character in text:
        if unicodedata.category(character)[0] in {"L", "M", "N"}:
            current.append(character)
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return tuple(result)


def _descriptor_allowed_whitespace_projection(
    text: str,
) -> tuple[list[list[int]], list[list[object]]]:
    position_codepoints = [
        [position, ord(character)]
        for position, character in enumerate(text)
        if character.isspace()
    ]
    runs: list[list[object]] = []
    position = 0
    while position < len(text):
        if not text[position].isspace():
            position += 1
            continue
        start = position
        codepoints: list[int] = []
        while position < len(text) and text[position].isspace():
            codepoints.append(ord(text[position]))
            position += 1
        runs.append([start, len(codepoints), codepoints])
    return position_codepoints, runs


def _descriptor_segment_representation(text: str) -> dict[str, object]:
    """Return only frozen representation metadata, never descriptor text.

    Exact code points are retained only for allowed whitespace.  Letter,
    number, punctuation, and symbol identity is deliberately absent; their
    ordered classes/categories and lexical-token lengths remain observable.
    """

    ascii_classes = [_descriptor_ascii_class(character) for character in text]
    categories = [unicodedata.category(character) for character in text]
    tokens = _descriptor_lexical_tokens(text)
    token_shapes = [_descriptor_token_shape(token) for token in tokens]
    token_codepoint_lengths = [len(token) for token in tokens]
    token_utf8_lengths = [len(token.encode("utf-8")) for token in tokens]
    whitespace_positions, whitespace_runs = (
        _descriptor_allowed_whitespace_projection(text)
    )
    nfkc = unicodedata.normalize("NFKC", text)
    return {
        "utf8_length": len(text.encode("utf-8")),
        "codepoint_length": len(text),
        "leading_ascii_spaces": len(text) - len(text.lstrip(" ")),
        "trailing_ascii_spaces": len(text) - len(text.rstrip(" ")),
        "first_boundary_shape": _descriptor_boundary_shape(text[0]),
        "last_boundary_shape": _descriptor_boundary_shape(text[-1]),
        "allowed_whitespace_position_codepoint_sequence": whitespace_positions,
        "allowed_whitespace_run_signature": whitespace_runs,
        "allowed_whitespace_count": len(whitespace_positions),
        "allowed_whitespace_run_count": len(whitespace_runs),
        "unicode_category_sequence": categories,
        "ascii_class_sequence": ascii_classes,
        "lexical_token_count": len(tokens),
        "token_shape_sequence": token_shapes,
        "token_codepoint_length_vector": token_codepoint_lengths,
        "token_utf8_length_vector": token_utf8_lengths,
        "separator_run_count": len(whitespace_runs),
        "ascii_class_counts": {
            name: ascii_classes.count(name) for name in _DESCRIPTOR_ASCII_CLASSES
        },
        "unicode_category_counts": {
            name: categories.count(name) for name in _DESCRIPTOR_UNICODE_CATEGORIES
        },
        "token_shape_counts": {
            name: token_shapes.count(name) for name in _DESCRIPTOR_TOKEN_SHAPES
        },
        "token_utf8_length_bucket_counts": {
            name: sum(
                _descriptor_token_length_bucket(length) == name
                for length in token_utf8_lengths
            )
            for name in _DESCRIPTOR_TOKEN_LENGTH_BUCKETS
        },
        "nfkc_changed": nfkc != text,
        "nfkc_codepoint_length_delta": len(nfkc) - len(text),
        "nfkc_utf8_length_delta": len(nfkc.encode("utf-8"))
        - len(text.encode("utf-8")),
    }


def _descriptor_representation_categorical(value: object) -> str:
    return canonicalize_rfc8785_v1(value).decode("utf-8")


def _append_feature_row_bounded(
    raw: list[tuple[str, tuple[str, ...]]],
    row: tuple[str, tuple[str, ...]],
    expected_rows: int,
) -> None:
    if len(raw) >= MAXIMUM_FEATURES:
        _reject(
            "FAIL_FEATURE_INCREMENTAL_CAP",
            "feature extraction reached the frozen 32,768-column cap",
        )
    aliases, values = row
    if (
        type(aliases) is not str
        or not aliases
        or type(values) is not tuple
        or len(values) != expected_rows
        or any(type(value) is not str for value in values)
    ):
        _reject("FAIL_FEATURE_COLUMN_SHAPE", "feature column shape drift")
    raw.append(row)


def _extend_feature_rows_bounded(
    raw: list[tuple[str, tuple[str, ...]]],
    rows: list[tuple[str, tuple[str, ...]]]
    | tuple[tuple[str, tuple[str, ...]], ...],
    expected_rows: int,
) -> None:
    for row in rows:
        _append_feature_row_bounded(raw, row, expected_rows)


def _descriptor_representation_features(
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
) -> list[tuple[str, tuple[str, ...]]]:
    raw: list[tuple[str, tuple[str, ...]]] = []
    paths = tuple(
        sorted(
            {
                leaf.path
                for item in decoded
                for leaf in item.channel_leaves
                if leaf.group == "descriptor_metadata"
            },
            key=_path_text,
        )
    )
    if not paths:
        _reject(
            "FAIL_DESCRIPTOR_METADATA_COVERAGE",
            "no semantic descriptor representation projection was registered",
        )
    by_row = [
        {
            leaf.path: _resolve(item.payload, leaf.path)
            for leaf in item.channel_leaves
            if leaf.group == "descriptor_metadata"
        }
        for item in decoded
    ]
    for path in paths:
        descriptors: list[dict[str, object] | None] = []
        for mapping in by_row:
            value = mapping.get(path)
            if value is None:
                descriptors.append(None)
            else:
                try:
                    descriptors.append(_validate_descriptor_object(value))
                except (TypeError, ValueError) as exc:
                    _reject(
                        "FAIL_DESCRIPTOR_METADATA_SHAPE",
                        f"descriptor metadata at {_path_text(path)} failed: {exc}",
                    )
        prefix = f"descriptor_metadata/{_path_text(path)}"
        _append_feature_row_bounded(raw,
            (
                prefix + "/presence",
                tuple("1" if value is not None else "0" for value in descriptors),
            ), len(decoded)
        )
        _append_feature_row_bounded(raw,
            (
                prefix + "/kind",
                tuple(
                    "ABSENT" if value is None else "PRESENT:" + str(value["kind"])
                    for value in descriptors
                ),
            ), len(decoded)
        )
        segment_counts = tuple(
            None if value is None else len(value["segments"])  # type: ignore[arg-type]
            for value in descriptors
        )
        _append_feature_row_bounded(raw,
            (
                prefix + "/segment_count",
                tuple("ABSENT" if value is None else str(value) for value in segment_counts),
            ), len(decoded)
        )
        projections = tuple(
            None
            if descriptor is None
            else tuple(
                _descriptor_segment_representation(segment)
                for segment in descriptor["segments"]  # type: ignore[union-attr]
            )
            for descriptor in descriptors
        )
        _append_feature_row_bounded(raw,
            (
                prefix + "/representation_composite",
                tuple(
                    "ABSENT"
                    if descriptor is None or rows is None
                    else "PRESENT:"
                    + _descriptor_representation_categorical(
                        {
                            "kind": descriptor["kind"],
                            "segment_count": len(rows),
                            "segments_in_order": list(rows),
                        }
                    )
                    for descriptor, rows in zip(descriptors, projections, strict=True)
                ),
            ), len(decoded)
        )
        maximum_segments = max((value or 0 for value in segment_counts), default=0)
        for position in range(maximum_segments):
            segment_projections = tuple(
                None
                if rows is None or position >= len(rows)
                else rows[position]
                for rows in projections
            )
            segment_prefix = prefix + f"/segment_position_{position:02d}"

            def values_of(name: str) -> tuple[str, ...]:
                return tuple(
                    "ABSENT" if segment is None else str(segment[name])
                    for segment in segment_projections
                )

            def categorical_values_of(name: str) -> tuple[str, ...]:
                return tuple(
                    "ABSENT"
                    if segment is None
                    else _descriptor_representation_categorical(segment[name])
                    for segment in segment_projections
                )

            _extend_feature_rows_bounded(raw,
                (
                    (
                        segment_prefix + "/presence",
                        tuple(
                            "1" if segment is not None else "0"
                            for segment in segment_projections
                        ),
                    ),
                    (
                        segment_prefix + "/order_position",
                        tuple(
                            "ABSENT" if segment is None else str(position)
                            for segment in segment_projections
                        ),
                    ),
                    (segment_prefix + "/utf8_length", values_of("utf8_length")),
                    (
                        segment_prefix + "/codepoint_length",
                        values_of("codepoint_length"),
                    ),
                    (
                        segment_prefix + "/leading_ascii_spaces",
                        values_of("leading_ascii_spaces"),
                    ),
                    (
                        segment_prefix + "/trailing_ascii_spaces",
                        values_of("trailing_ascii_spaces"),
                    ),
                    (
                        segment_prefix + "/first_boundary_shape",
                        values_of("first_boundary_shape"),
                    ),
                    (
                        segment_prefix + "/last_boundary_shape",
                        values_of("last_boundary_shape"),
                    ),
                    (
                        segment_prefix
                        + "/allowed_whitespace_position_codepoint_sequence",
                        categorical_values_of(
                            "allowed_whitespace_position_codepoint_sequence"
                        ),
                    ),
                    (
                        segment_prefix + "/allowed_whitespace_run_signature",
                        categorical_values_of("allowed_whitespace_run_signature"),
                    ),
                    (
                        segment_prefix + "/allowed_whitespace_count",
                        values_of("allowed_whitespace_count"),
                    ),
                    (
                        segment_prefix + "/allowed_whitespace_run_count",
                        values_of("allowed_whitespace_run_count"),
                    ),
                    (
                        segment_prefix + "/unicode_category_sequence",
                        categorical_values_of("unicode_category_sequence"),
                    ),
                    (
                        segment_prefix + "/ascii_class_sequence",
                        categorical_values_of("ascii_class_sequence"),
                    ),
                    (
                        segment_prefix + "/token_shape_sequence",
                        categorical_values_of("token_shape_sequence"),
                    ),
                    (
                        segment_prefix + "/token_codepoint_length_vector",
                        categorical_values_of("token_codepoint_length_vector"),
                    ),
                    (
                        segment_prefix + "/token_utf8_length_vector",
                        categorical_values_of("token_utf8_length_vector"),
                    ),
                    (
                        segment_prefix + "/nfkc_changed",
                        tuple(
                            "ABSENT"
                            if segment is None
                            else "1" if segment["nfkc_changed"] else "0"
                            for segment in segment_projections
                        ),
                    ),
                    (
                        segment_prefix + "/nfkc_codepoint_length_delta",
                        values_of("nfkc_codepoint_length_delta"),
                    ),
                    (
                        segment_prefix + "/nfkc_utf8_length_delta",
                        values_of("nfkc_utf8_length_delta"),
                    ),
                ), len(decoded)
            )
            _extend_feature_rows_bounded(raw,
                (
                    (
                        segment_prefix + "/lexical_token_count",
                        values_of("lexical_token_count"),
                    ),
                    (
                        segment_prefix + "/separator_run_count",
                        values_of("separator_run_count"),
                    ),
                ), len(decoded)
            )
            for class_name in _DESCRIPTOR_ASCII_CLASSES:
                _append_feature_row_bounded(raw,
                    (
                        segment_prefix + "/ascii_class_count/" + class_name,
                        tuple(
                            "ABSENT"
                            if segment is None
                            else str(segment["ascii_class_counts"][class_name])  # type: ignore[index]
                            for segment in segment_projections
                        ),
                    ), len(decoded)
                )
            for category in _DESCRIPTOR_UNICODE_CATEGORIES:
                _append_feature_row_bounded(raw,
                    (
                        segment_prefix + "/unicode_category_count/" + category,
                        tuple(
                            "ABSENT"
                            if segment is None
                            else str(segment["unicode_category_counts"][category])  # type: ignore[index]
                            for segment in segment_projections
                        ),
                    ), len(decoded)
                )
            for shape in _DESCRIPTOR_TOKEN_SHAPES:
                _append_feature_row_bounded(raw,
                    (
                        segment_prefix + "/token_shape_count/" + shape,
                        tuple(
                            "ABSENT"
                            if segment is None
                            else str(segment["token_shape_counts"][shape])  # type: ignore[index]
                            for segment in segment_projections
                        ),
                    ), len(decoded)
                )
            for bucket in _DESCRIPTOR_TOKEN_LENGTH_BUCKETS:
                _append_feature_row_bounded(raw,
                    (
                        segment_prefix + "/token_utf8_length_bucket/" + bucket,
                        tuple(
                            "ABSENT"
                            if segment is None
                            else str(
                                segment["token_utf8_length_bucket_counts"][bucket]  # type: ignore[index]
                            )
                            for segment in segment_projections
                        ),
                    ), len(decoded)
                )
    return raw


def _extract_feature_columns(
    envelopes: tuple[bytes, ...],
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
    projection: _FeatureCardinalityProjectionV1,
) -> tuple[FeatureManifestV1, tuple[_FeatureColumn, ...]]:
    projection.__post_init__()
    if projection.rows != len(decoded):
        _reject("FAIL_FEATURE_PROJECTION_REPLAY", "feature projection row count drift")
    raw: list[tuple[str, tuple[str, ...]]] = []
    uuid_keys = sorted(
        {(leaf.namespace, leaf.path) for item in decoded for leaf in item.namespace_leaves},
        key=lambda item: (item[0], _path_text(item[1])),
    )
    by_row_uuid = [
        {(leaf.namespace, leaf.path): leaf.public_uuid for leaf in item.namespace_leaves}
        for item in decoded
    ]
    for namespace, path in uuid_keys:
        values = tuple(mapping.get((namespace, path)) for mapping in by_row_uuid)
        _append_feature_row_bounded(raw,
            (
                f"uuid_leaf/{namespace}/{_path_text(path)}/presence",
                tuple("1" if value is not None else "0" for value in values),
            ), len(decoded)
        )
        _extend_feature_rows_bounded(
            raw,
            _uuid_raw_features(namespace, path, values),
            len(decoded),
        )

    channel_keys = sorted(
        {(leaf.group, leaf.path) for item in decoded for leaf in item.channel_leaves},
        key=lambda item: (item[0], _path_text(item[1])),
    )
    by_row_channel = [
        {(leaf.group, leaf.path): leaf.feature_value for leaf in item.channel_leaves}
        for item in decoded
    ]
    for group, path in channel_keys:
        if group in {"provenance", "descriptor_metadata"}:
            continue
        values = tuple(mapping.get((group, path)) for mapping in by_row_channel)
        _append_feature_row_bounded(raw,
            (
                f"{group}/{_path_text(path)}/presence",
                tuple("1" if value is not None else "0" for value in values),
            ), len(decoded)
        )
        _append_feature_row_bounded(raw,
            (
                f"{group}/{_path_text(path)}/tagged_value",
                tuple(
                    "ABSENT" if value is None else "PRESENT:" + value
                    for value in values
                ),
            ), len(decoded)
        )
    _extend_feature_rows_bounded(
        raw,
        _descriptor_representation_features(decoded),
        len(decoded),
    )
    provenance_paths = tuple(path for group, path in channel_keys if group == "provenance")
    for path in provenance_paths:
        present = tuple(
            ("provenance", path) in mapping for mapping in by_row_channel
        )
        _append_feature_row_bounded(raw,
            (
                f"provenance/{_path_text(path)}/presence",
                tuple("1" if value else "0" for value in present),
            ), len(decoded)
        )
        blobs = tuple(
            _provenance_digest(_resolve(item.payload, path))
            if is_present
            else b"\x00" * 32
            for item, is_present in zip(decoded, present, strict=True)
        )
        _extend_feature_rows_bounded(raw,
            _bounded_byte_features(
                f"provenance/{_path_text(path)}",
                blobs,
            ),
            len(decoded),
        )
    if not any(group == "eligibility_mask" for group, _ in channel_keys):
        _append_feature_row_bounded(raw,
            (
                "eligibility_mask/MUST_BE_ABSENT",
                tuple("MUST_BE_ABSENT" for _ in decoded),
            ), len(decoded)
        )

    _append_feature_row_bounded(raw, ("payload_length", tuple(str(len(item.payload_bytes)) for item in decoded)), len(decoded))
    _append_feature_row_bounded(raw, ("padding/length", tuple(str(len(item.padding_bytes)) for item in decoded)), len(decoded))
    _append_feature_row_bounded(raw, ("padding/hamming_weight", tuple(str(sum(byte.bit_count() for byte in item.padding_bytes)) for item in decoded)), len(decoded))
    for byte_value in range(256):
        _append_feature_row_bounded(raw,
            (
                f"padding/byte_histogram/count_{byte_value:03d}",
                tuple(str(item.padding_bytes.count(byte_value)) for item in decoded),
            ), len(decoded)
        )
    _extend_feature_rows_bounded(raw,
        _bounded_byte_features(
            "padding",
            tuple(item.padding_bytes[:16] + item.padding_bytes[-16:] for item in decoded),
        ),
        len(decoded),
    )
    _extend_feature_rows_bounded(raw,
        _bounded_byte_features(
            "byte_prefix_suffix/envelope",
            tuple(envelope[:16] + envelope[-16:] for envelope in envelopes),
        ),
        len(decoded),
    )

    if (
        len(raw) != projection.projected_concrete_feature_count
        or sum(len(set(name.split("\x1f"))) for name, _ in raw)
        != projection.projected_raw_semantic_feature_count
    ):
        _reject(
            "FAIL_FEATURE_PROJECTION_REPLAY",
            "materialized feature count/alias coverage differs from the preallocation projection",
        )

    # Only aliases explicitly placed on the same raw row are deduplicated.  In
    # particular, first/last 8/16/32 bit-decomposition names alias their exact
    # 128-bit positions.  Two different semantic fields remain two hypotheses
    # even when their observed columns happen to be byte-identical.
    aliases: list[tuple[str, ...]] = []
    columns: list[tuple[str, ...]] = []
    for encoded_aliases, values in raw:
        aliases.append(tuple(sorted(set(encoded_aliases.split("\x1f")))))
        columns.append(values)
    canonical_aliases = tuple(aliases)
    concrete_ids = tuple(
        stable_hash({"aliases": items}, prefix="phase2b_formal_covert_feature_")
        for items in canonical_aliases
    )
    paired = sorted(
        zip(concrete_ids, canonical_aliases, columns, strict=True),
        key=lambda item: item[0],
    )
    manifest = FeatureManifestV1(
        concrete_feature_ids=tuple(item[0] for item in paired),
        alias_coverage=tuple(FeatureAliasCoverageV1(item[0], item[1]) for item in paired),
        channel_groups=CHANNEL_GROUPS,
        unique_id_feature_family=UNIQUE_ID_FEATURE_FAMILY,
        uuid_leaf_count=len(uuid_keys),
        channel_leaf_count=len(channel_keys),
        raw_semantic_feature_count=sum(len(item) for item in canonical_aliases),
        deduplicated_feature_count=len(paired),
    )
    return manifest, tuple(_FeatureColumn(item[1], item[2]) for item in paired)


def _partition(values: tuple[str, ...]) -> _Partition:
    masks: dict[str, int] = {}
    for index, value in enumerate(values):
        if value not in masks:
            if len(masks) >= MAXIMUM_VALUES_PER_FEATURE:
                _reject(
                    "FAIL_FEATURE_VALUE_CARDINALITY",
                    "one feature exceeds the frozen 1,024-value cap",
                )
            masks[value] = 0
        masks[value] |= 1 << index
    ordered = tuple(masks[key] for key in sorted(masks, key=lambda value: value.encode("utf-8")))
    return _Partition(ordered, tuple(mask.bit_count() for mask in ordered))


@dataclass(frozen=True, slots=True)
class _PartitionAllocationProjectionV1:
    rows: int
    cardinalities: tuple[int, ...]
    projected_partition_matrix_bytes_bound: int
    projection_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.rows) is not int or not 1 <= self.rows <= MAXIMUM_ROWS:
            raise ValueError("partition projection row count drift")
        if type(self.cardinalities) is not tuple or not self.cardinalities:
            raise TypeError("partition projection needs exact feature cardinalities")
        if any(
            type(value) is not int
            or not 1 <= value <= min(self.rows, MAXIMUM_VALUES_PER_FEATURE)
            for value in self.cardinalities
        ):
            raise ValueError("partition projection feature cardinality drift")
        maximum_mask_bytes = sys.getsizeof((1 << self.rows) - 1)
        maximum_count_bytes = sys.getsizeof(self.rows)
        partition_shell_bytes = sys.getsizeof(_Partition((), ()))
        projected = sys.getsizeof((None,) * len(self.cardinalities))
        for cardinality in self.cardinalities:
            projected += (
                partition_shell_bytes
                + 2 * sys.getsizeof((None,) * cardinality)
                + cardinality * (maximum_mask_bytes + maximum_count_bytes)
            )
        projected *= DEEP_SIZE_SAFETY_MULTIPLIER
        if self.projected_partition_matrix_bytes_bound != projected:
            raise ValueError("partition allocation projection formula drift")
        object.__setattr__(
            self,
            "projection_root",
            stable_hash(
                {
                    "rows": self.rows,
                    "cardinalities": self.cardinalities,
                    "bytes_bound": projected,
                },
                prefix="phase2b_formal_covert_partition_projection_",
            ),
        )


def _partition_allocation_projection(
    rows: int,
    columns: tuple[_FeatureColumn, ...],
    deadline_ns: int,
) -> _PartitionAllocationProjectionV1:
    cardinalities: list[int] = []
    for column in columns:
        _remaining_partition_preparation_deadline_seconds(deadline_ns)
        observed: set[str] = set()
        for value in column.values:
            observed.add(value)
            if len(observed) > MAXIMUM_VALUES_PER_FEATURE:
                _reject(
                    "FAIL_FEATURE_VALUE_CARDINALITY",
                    "one feature exceeds the frozen 1,024-value cap before partition allocation",
                )
        cardinalities.append(len(observed))
    values = tuple(cardinalities)
    maximum_mask_bytes = sys.getsizeof((1 << rows) - 1)
    maximum_count_bytes = sys.getsizeof(rows)
    partition_shell_bytes = sys.getsizeof(_Partition((), ()))
    projected = sys.getsizeof((None,) * len(values)) + sum(
        partition_shell_bytes
        + 2 * sys.getsizeof((None,) * cardinality)
        + cardinality * (maximum_mask_bytes + maximum_count_bytes)
        for cardinality in values
    )
    return _PartitionAllocationProjectionV1(
        rows,
        values,
        projected * DEEP_SIZE_SAFETY_MULTIPLIER,
    )


def _materialize_partitions_bounded(
    columns: tuple[_FeatureColumn, ...],
    projection: _PartitionAllocationProjectionV1,
    deadline_ns: int,
) -> tuple[_Partition, ...]:
    projection.__post_init__()
    if len(columns) != len(projection.cardinalities):
        _reject("FAIL_PARTITION_PROJECTION_REPLAY", "partition feature count drift")
    rows: list[_Partition] = []
    for column, expected_cardinality in zip(
        columns, projection.cardinalities, strict=True
    ):
        _remaining_partition_preparation_deadline_seconds(deadline_ns)
        partition = _partition(column.values)
        if len(partition.masks) != expected_cardinality:
            _reject(
                "FAIL_PARTITION_PROJECTION_REPLAY",
                "partition cardinality changed after preallocation projection",
            )
        rows.append(partition)
    _remaining_partition_preparation_deadline_seconds(deadline_ns)
    return tuple(rows)


def _label_masks(labels: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    order = tuple(sorted(set(labels), key=lambda item: item.encode("utf-8")))
    masks = {label: 0 for label in order}
    for index, label in enumerate(labels):
        masks[label] |= 1 << index
    return order, tuple(masks[label] for label in order)


def _metrics_from_masks(
    partition: _Partition,
    label_masks: tuple[int, ...],
    cache: dict[tuple[int, ...], tuple[float, float]],
) -> tuple[float, float]:
    key = tuple((feature_mask & label_mask).bit_count() for feature_mask in partition.masks for label_mask in label_masks)
    cached = cache.get(key)
    if cached is not None:
        return cached
    label_counts = tuple(mask.bit_count() for mask in label_masks)
    total = sum(label_counts)
    if len(partition.masks) < 2 or len(label_masks) < 2:
        nmi = 0.0
    else:
        cells = iter(key)
        mi_terms = []
        for feature_count in partition.counts:
            for label_count in label_counts:
                count = next(cells)
                if count:
                    mi_terms.append((count / total) * math.log2(count * total / (feature_count * label_count)))
        mi = math.fsum(mi_terms)
        hx = -math.fsum((count / total) * math.log2(count / total) for count in partition.counts)
        hy = -math.fsum((count / total) * math.log2(count / total) for count in label_counts)
        nmi = 0.0 if hx == 0.0 or hy == 0.0 else min(1.0, max(0.0, mi / math.sqrt(hx * hy)))

    class_count = len(label_masks)
    correct = [0] * class_count
    cursor = 0
    global_counts = label_counts
    for feature_count in partition.counts:
        local = key[cursor : cursor + class_count]
        cursor += class_count
        for true_class, row_count in enumerate(local):
            if not row_count:
                continue
            source = local if feature_count > 1 else global_counts
            predicted = min(
                range(class_count),
                key=lambda candidate: (
                    -(source[candidate] - (1 if candidate == true_class else 0)),
                    candidate,
                ),
            )
            if predicted == true_class:
                correct[true_class] += row_count
    balanced = math.fsum(correct[index] / global_counts[index] for index in range(class_count)) / class_count
    advantage = balanced - 1.0 / class_count
    if (
        len(key) <= MAXIMUM_METRIC_CACHE_CELLS
        and len(cache) < MAXIMUM_METRIC_CACHE_ENTRIES
    ):
        cache[key] = (nmi, advantage)
    return nmi, advantage


def _permuted_labels(
    labels: tuple[str, ...],
    strata: tuple[tuple[str, ...], ...],
    target: str,
    permutation_index: int,
    master_seed: bytes,
    schedule_hasher: object | None,
) -> tuple[str, ...]:
    if type(master_seed) is not bytes or len(master_seed) != 32:
        raise TypeError("label permutation requires an exact auditor-owned seed")
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row_index, stratum in enumerate(strata):
        groups[stratum].append(row_index)
    output = list(labels)
    for stratum in sorted(groups, key=lambda item: tuple(part.encode("utf-8") for part in item)):
        positions = groups[stratum]
        donors: list[object] = list(positions)
        seed = hashlib.sha256(
            _PERMUTATION_DOMAIN
            + master_seed
            + target.encode("ascii")
            + permutation_index.to_bytes(8, "big")
            + repr(stratum).encode("utf-8")
        ).digest()
        _shuffle(donors, seed)
        for position, donor_object in zip(positions, donors, strict=True):
            donor = int(donor_object)
            output[position] = labels[donor]
            if schedule_hasher is not None:
                schedule_hasher.update(position.to_bytes(4, "big"))
                schedule_hasher.update(donor.to_bytes(4, "big"))
    return tuple(output)


def _holm_adjust_exact(
    values: tuple[ExactProbabilityV1, ...],
) -> tuple[ExactProbabilityV1, ...]:
    if not values:
        return ()
    for value in values:
        if type(value) is not ExactProbabilityV1:
            raise TypeError("Holm input must use exact probability DTOs")
        value.__post_init__()
    order = sorted(
        range(len(values)),
        key=lambda index: (values[index].as_fraction(), index),
    )
    result: list[ExactProbabilityV1 | None] = [None] * len(values)
    running = Fraction(0, 1)
    for rank, index in enumerate(order):
        candidate = min(
            Fraction(1, 1),
            (len(values) - rank) * values[index].as_fraction(),
        )
        running = max(running, candidate)
        result[index] = _exact_probability(running)
    if any(item is None for item in result):
        raise RuntimeError("Holm exact adjustment left an unassigned result")
    return tuple(item for item in result if item is not None)


def _holm_ranks_exact(
    values: tuple[ExactProbabilityV1, ...],
) -> tuple[int, ...]:
    order = sorted(
        range(len(values)),
        key=lambda index: (values[index].as_fraction(), index),
    )
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return tuple(ranks)


def _holm_adjust(values: tuple[float, ...]) -> tuple[float, ...]:
    """Compatibility oracle view; executed receipts use exact rationals."""

    exact = tuple(_exact_probability(Fraction(value)) for value in values)
    return tuple(item.as_float for item in _holm_adjust_exact(exact))


def _self_cgroup_membership_path(controller: str | None) -> Path | None:
    """Return one normalized hierarchy-absolute membership path."""

    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"cgroup membership inventory is unreadable: {exc}",
        )
    for row in rows:
        fields = row.split(":", 2)
        if len(fields) != 3:
            continue
        controllers = fields[1].split(",") if fields[1] else []
        if (controller is None and fields[1] == "") or (
            controller is not None and controller in controllers
        ):
            parts = tuple(part for part in fields[2].split("/") if part)
            if any(part in {".", ".."} for part in parts):
                _reject(
                    "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                    "cgroup membership contains traversal components",
                )
            return Path("/", *parts)
    return None


def _cgroup_membership_declared(controller: str | None) -> bool:
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"cgroup membership inventory is unreadable: {exc}",
        )
    for row in rows:
        fields = row.split(":", 2)
        if len(fields) != 3:
            continue
        controllers = fields[1].split(",") if fields[1] else []
        if (controller is None and fields[1] == "") or (
            controller is not None and controller in controllers
        ):
            return True
    return False


def _decode_mountinfo_path(value: str) -> Path:
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    path = Path(value)
    if not path.is_absolute():
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "cgroup mountinfo contained a non-absolute mount point",
        )
    return path


def _parse_cgroup_mount_roots(
    mountinfo_text: str,
    controller: str | None,
) -> tuple[tuple[Path, Path], ...]:
    roots: list[tuple[Path, Path]] = []
    for row in mountinfo_text.splitlines():
        halves = row.split(" - ", 1)
        if len(halves) != 2:
            continue
        before = halves[0].split()
        after = halves[1].split()
        if len(before) < 5 or len(after) < 3:
            continue
        filesystem = after[0]
        options = set(after[2].split(","))
        if controller is None and filesystem == "cgroup2":
            roots.append(
                (
                    _decode_mountinfo_path(before[4]),
                    _decode_mountinfo_path(before[3]),
                )
            )
        elif (
            controller is not None
            and filesystem == "cgroup"
            and controller in options
        ):
            roots.append(
                (
                    _decode_mountinfo_path(before[4]),
                    _decode_mountinfo_path(before[3]),
                )
            )
    return tuple(dict.fromkeys(roots))


def _read_bounded_ascii_cgroup_file(
    path: Path,
    *,
    name: str,
    maximum_bytes: int = 65_536,
) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{name} is unreadable: {type(exc).__name__}",
        )
    if len(payload) > maximum_bytes:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{name} exceeds its bounded read",
        )
    try:
        return payload.decode("ascii", errors="strict")
    except UnicodeError:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{name} is not exact ASCII",
        )


def _parse_exact_cgroup_decimal(value: str, *, name: str, positive: bool) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)\n?", value) is None:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{name} is not one canonical finite decimal",
        )
    parsed = int(value.removesuffix("\n"))
    if parsed > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{name} exceeds the finite resource-counter bound",
        )
    if positive and parsed <= 0:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{name} is not positive",
        )
    return parsed


def _read_unified_cgroup_memory_snapshot_v1(
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    proc_mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> UnifiedCgroupMemorySnapshotV1:
    """Read one exact visible non-root cgroup-v2 leaf without mutating it."""

    membership_text = _read_bounded_ascii_cgroup_file(
        proc_cgroup_path,
        name="unified cgroup membership",
    )
    membership_rows = membership_text.splitlines()
    if len(membership_rows) != 1:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup membership is not one exact row",
        )
    match = re.fullmatch(
        r"0::(/(?:[A-Za-z0-9_.@:-]+/)*[A-Za-z0-9_.@:-]+)",
        membership_rows[0],
    )
    if match is None:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup membership is not one exact non-root leaf",
        )
    membership_text_value = match.group(1)
    if any(
        component in {"", ".", ".."}
        for component in membership_text_value.split("/")[1:]
    ):
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup membership contains a noncanonical path component",
        )
    membership = Path(membership_text_value)
    mountinfo_text = _read_bounded_ascii_cgroup_file(
        proc_mountinfo_path,
        name="cgroup mount inventory",
        maximum_bytes=1_048_576,
    )
    mounts: list[tuple[Path, Path]] = []
    for row in mountinfo_text.splitlines():
        halves = row.split(" - ", 1)
        if len(halves) != 2:
            if "cgroup2" in row:
                _reject(
                    "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                    "cgroup-v2 mount row is malformed",
                )
            continue
        before = halves[0].split()
        after = halves[1].split()
        if not after or after[0] != "cgroup2":
            continue
        if len(before) < 6 or len(after) < 3:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                "cgroup-v2 mount row is malformed",
            )
        mounts.append(
            (
                _decode_mountinfo_path(before[4]),
                _decode_mountinfo_path(before[3]),
            )
        )
    visible: list[Path] = []
    for mount_point, filesystem_root in tuple(dict.fromkeys(mounts)):
        try:
            relative = membership.relative_to(filesystem_root)
        except ValueError:
            continue
        if relative == Path("."):
            continue
        visible.append(mount_point / relative)
    visible = list(dict.fromkeys(visible))
    if len(visible) != 1:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup membership lacks one unique visible leaf",
        )
    leaf = visible[0]
    try:
        before_stat = leaf.stat(follow_symlinks=False)
        resolved_leaf = leaf.resolve(strict=True)
    except OSError as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"unified cgroup leaf is unreadable: {type(exc).__name__}",
        )
    if not stat.S_ISDIR(before_stat.st_mode) or resolved_leaf != leaf:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup leaf is not one exact real directory",
        )
    values = {
        name: _parse_exact_cgroup_decimal(
            _read_bounded_ascii_cgroup_file(
                leaf / name,
                name=name,
                maximum_bytes=4_096,
            ),
            name=name,
            positive=name == "memory.max",
        )
        for name in ("memory.current", "memory.peak", "memory.max")
    }
    try:
        after_stat = leaf.stat(follow_symlinks=False)
    except OSError as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"unified cgroup leaf identity disappeared: {type(exc).__name__}",
        )
    if (
        before_stat.st_dev,
        before_stat.st_ino,
    ) != (
        after_stat.st_dev,
        after_stat.st_ino,
    ) or _read_bounded_ascii_cgroup_file(
        proc_cgroup_path,
        name="unified cgroup membership replay",
    ) != membership_text:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup membership or leaf identity drifted during snapshot",
        )
    scope_root = stable_hash(
        {
            "leaf_device": before_stat.st_dev,
            "leaf_inode": before_stat.st_ino,
            "unified_membership": membership_text_value,
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )
    try:
        snapshot = UnifiedCgroupMemorySnapshotV1(
            unified_membership=membership_text_value,
            memory_scope_root=scope_root,
            leaf_device=before_stat.st_dev,
            leaf_inode=before_stat.st_ino,
            memory_current_bytes=values["memory.current"],
            memory_peak_bytes=values["memory.peak"],
            memory_max_bytes=values["memory.max"],
        )
        snapshot.__post_init__()
    except (TypeError, ValueError) as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"unified cgroup memory snapshot is malformed: {exc}",
        )
    return snapshot


_UNIFIED_CGROUP_MEMORY_SNAPSHOT_PROVIDER_V1: Callable[
    [], UnifiedCgroupMemorySnapshotV1
] = _read_unified_cgroup_memory_snapshot_v1


def _capture_unified_cgroup_memory_snapshot_v1() -> UnifiedCgroupMemorySnapshotV1:
    snapshot = _UNIFIED_CGROUP_MEMORY_SNAPSHOT_PROVIDER_V1()
    if type(snapshot) is not UnifiedCgroupMemorySnapshotV1:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup memory snapshot provider returned a non-exact DTO",
        )
    snapshot.__post_init__()
    return snapshot


def _same_unified_cgroup_memory_scope_v1(
    left: UnifiedCgroupMemorySnapshotV1,
    right: UnifiedCgroupMemorySnapshotV1,
) -> bool:
    left.__post_init__()
    right.__post_init__()
    return (
        left.unified_membership,
        left.memory_scope_root,
        left.leaf_device,
        left.leaf_inode,
        left.memory_max_bytes,
    ) == (
        right.unified_membership,
        right.memory_scope_root,
        right.leaf_device,
        right.leaf_inode,
        right.memory_max_bytes,
    )


def _physical_memory_failure_detail_v1(
    *,
    stage: str,
    snapshot: UnifiedCgroupMemorySnapshotV1,
    parent_historical_peak_rss_bytes: int,
    worker_historical_peak_rss_bytes_sum: int,
) -> str:
    if (
        type(stage) is not str
        or re.fullmatch(r"[A-Z_]{1,64}", stage) is None
        or type(parent_historical_peak_rss_bytes) is not int
        or parent_historical_peak_rss_bytes < 0
        or parent_historical_peak_rss_bytes
        > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
        or type(worker_historical_peak_rss_bytes_sum) is not int
        or worker_historical_peak_rss_bytes_sum < 0
        or worker_historical_peak_rss_bytes_sum
        > MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
    ):
        raise TypeError("physical memory failure diagnostic arguments are invalid")
    snapshot.__post_init__()
    overage = max(
        0,
        snapshot.memory_peak_bytes - MAXIMUM_ESTIMATED_RSS_BYTES,
    )
    detail = (
        f"stage={stage};scope={snapshot.memory_scope_root};"
        f"physical_current_bytes={snapshot.memory_current_bytes};"
        f"physical_peak_bytes={snapshot.memory_peak_bytes};"
        f"physical_memory_max_bytes={snapshot.memory_max_bytes};"
        f"physical_cap_bytes={MAXIMUM_ESTIMATED_RSS_BYTES};"
        f"parent_historical_peak_rss_bytes={parent_historical_peak_rss_bytes};"
        f"worker_historical_peak_rss_bytes_sum={worker_historical_peak_rss_bytes_sum};"
        f"overage_bytes={overage}"
    )
    if "\n" in detail or "\r" in detail or len(detail.encode("ascii")) > 1_024:
        raise RuntimeError("physical memory failure diagnostic is not bounded one-line ASCII")
    return detail


def _require_unified_cgroup_physical_peak_v1(
    *,
    failure_code: str,
    stage: str,
    snapshot: UnifiedCgroupMemorySnapshotV1,
    parent_historical_peak_rss_bytes: int,
    worker_historical_peak_rss_bytes_sum: int,
    expected_scope: UnifiedCgroupMemorySnapshotV1 | None = None,
) -> None:
    snapshot.__post_init__()
    if expected_scope is not None and not _same_unified_cgroup_memory_scope_v1(
        expected_scope,
        snapshot,
    ):
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup memory scope identity drifted between receipt samples",
        )
    if snapshot.memory_peak_bytes > MAXIMUM_ESTIMATED_RSS_BYTES:
        _reject(
            failure_code,
            _physical_memory_failure_detail_v1(
                stage=stage,
                snapshot=snapshot,
                parent_historical_peak_rss_bytes=(
                    parent_historical_peak_rss_bytes
                ),
                worker_historical_peak_rss_bytes_sum=(
                    worker_historical_peak_rss_bytes_sum
                ),
            ),
        )


def _cgroup_mount_roots(
    controller: str | None,
) -> tuple[tuple[Path, Path], ...]:
    try:
        text = Path("/proc/self/mountinfo").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"cgroup mount inventory is unreadable: {exc}",
        )
    roots = _parse_cgroup_mount_roots(text, controller)
    if not roots and _cgroup_membership_declared(controller):
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "declared cgroup membership has no exact controller mount",
        )
    return roots


def _cgroup_ancestor_directories(controller: str | None) -> tuple[Path, ...]:
    """Return exact leaf-to-mount-root directories without escaping a root."""

    membership = _self_cgroup_membership_path(controller)
    if membership is None:
        return ()
    mounts = _cgroup_mount_roots(controller)
    directories: list[Path] = []
    for mount_point, filesystem_root in mounts:
        try:
            visible_relative = membership.relative_to(filesystem_root)
        except ValueError:
            continue
        leaf = mount_point / visible_relative
        cursor = leaf
        lineage: list[Path] = []
        while True:
            lineage.append(cursor)
            if cursor == mount_point:
                break
            parent = cursor.parent
            if (
                parent == cursor
                or mount_point not in parent.parents
                and parent != mount_point
            ):
                lineage = []
                break
            cursor = parent
        directories.extend(lineage)
    if not directories:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "cgroup membership is outside every visible controller mount root",
        )
    return tuple(dict.fromkeys(directories))


def _read_cgroup_control_pair(
    directory: Path,
    first_name: str,
    second_name: str,
) -> tuple[str, str] | None:
    """Read two controls from one hierarchy level or fail on a partial pair."""

    first = directory / first_name
    second = directory / second_name
    try:
        first_text = first.read_text(encoding="ascii").strip()
        second_text = second.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        if first.exists() or second.exists():
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"partial cgroup control pair at {directory}",
            )
        return None
    except (OSError, UnicodeError) as exc:
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"unreadable cgroup control pair at {directory}: {exc}",
        )
    return first_text, second_text


def _linux_cgroup_cpu_quota_capacity() -> int | None:
    capacities: list[int] = []
    v2_seen = False
    for directory in _cgroup_ancestor_directories(None):
        path = directory / "cpu.max"
        try:
            value = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"unreadable cgroup v2 cpu.max at {directory}: {exc}",
            )
        v2_seen = True
        fields = value.split()
        if len(fields) != 2:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"malformed cgroup v2 cpu.max at {directory}",
            )
        try:
            period = int(fields[1])
            quota = None if fields[0] == "max" else int(fields[0])
        except ValueError:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"non-integer cgroup v2 cpu.max at {directory}",
            )
        if period <= 0 or quota is not None and quota <= 0:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"nonpositive cgroup v2 cpu.max at {directory}",
            )
        if quota is not None:
            capacities.append(max(1, quota // period))
    if v2_seen:
        return min(capacities) if capacities else None

    for directory in _cgroup_ancestor_directories("cpu"):
        pair = _read_cgroup_control_pair(
            directory,
            "cpu.cfs_quota_us",
            "cpu.cfs_period_us",
        )
        if pair is None:
            continue
        try:
            quota, period = (int(item) for item in pair)
        except ValueError:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"malformed cgroup v1 CPU pair at {directory}",
            )
        if quota == -1:
            if period <= 0:
                _reject(
                    "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                    f"nonpositive cgroup v1 CPU period at {directory}",
                )
            continue
        if quota <= 0 or period <= 0:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"nonpositive cgroup v1 CPU pair at {directory}",
            )
        capacities.append(max(1, quota // period))
    return min(capacities) if capacities else None


def _linux_effective_cpu_capacity() -> int:
    capacities = [max(1, os.cpu_count() or 1)]
    if hasattr(os, "sched_getaffinity"):
        try:
            capacities.append(max(1, len(os.sched_getaffinity(0))))
        except OSError:
            capacities.append(1)
    quota_capacity = _linux_cgroup_cpu_quota_capacity()
    if quota_capacity is not None:
        capacities.append(quota_capacity)
    return min(capacities)


def _invariance_resource_plan(
    rows: int,
    *,
    envelope_and_decode_bytes_bound: int,
    adapter_retained_graph_bytes_bound: int,
    variant_suite_resident_bytes_per_row_bound: int,
    variant_suite_resident_fixed_bytes_bound: int,
) -> _InvariantResourcePlanV1:
    if type(rows) is not int or not 1 <= rows <= MAXIMUM_ROWS:
        raise ValueError("invariant resource-plan row count is outside the frozen cap")
    if (
        type(envelope_and_decode_bytes_bound) is not int
        or envelope_and_decode_bytes_bound < rows * ENVELOPE_BYTES * 2
    ):
        raise ValueError("invariant envelope/decode bound is below its frozen floor")
    if (
        type(adapter_retained_graph_bytes_bound) is not int
        or not 0 < adapter_retained_graph_bytes_bound
        <= MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES
        + MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES
    ):
        raise ValueError("invariant adapter retained-graph bound is absent or invalid")
    if (
        type(variant_suite_resident_bytes_per_row_bound) is not int
        or variant_suite_resident_bytes_per_row_bound <= 0
        or type(variant_suite_resident_fixed_bytes_bound) is not int
        or variant_suite_resident_fixed_bytes_bound <= 0
    ):
        raise ValueError("invariant variant-suite resident bound is absent or invalid")
    variant_suite_output = rows * 80 * ENVELOPE_BYTES
    variant_suite_resident = (
        rows * variant_suite_resident_bytes_per_row_bound
        + variant_suite_resident_fixed_bytes_bound
    )
    recognizer_live = (
        RECOGNIZER_OUTPUT_SIMULTANEOUS_GRAPH_MULTIPLIER
        * _recognizer_output_graph_bytes_bound(rows)
    )
    # This is the sole parent baseline for the invariant phase.  The final
    # preflight resamples immediately before fork and charges any later growth.
    parent_current_rss = _linux_current_rss_bytes()
    estimated_rss = (
        max(
            parent_current_rss,
            envelope_and_decode_bytes_bound
            + adapter_retained_graph_bytes_bound
            + 64 * 1_048_576,
        )
        * 2
        + variant_suite_resident
        + recognizer_live
        + INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
    )
    return _InvariantResourcePlanV1(
        rows=rows,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=adapter_retained_graph_bytes_bound,
        parent_current_rss_bytes_at_plan=parent_current_rss,
        invariance_variant_suite_output_bytes_bound=variant_suite_output,
        variant_suite_resident_bytes_per_row_bound=(
            variant_suite_resident_bytes_per_row_bound
        ),
        variant_suite_resident_fixed_bytes_bound=(
            variant_suite_resident_fixed_bytes_bound
        ),
        invariance_variant_suite_resident_bytes_bound=variant_suite_resident,
        invariance_recognizer_live_output_bytes_bound=recognizer_live,
        invariance_worker_artifact_bytes_bound=INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND,
        estimated_invariance_peak_rss_bytes=estimated_rss,
        maximum_invariance_estimated_rss_bytes=MAXIMUM_ESTIMATED_RSS_BYTES,
        invariance_rss_formula=(
            "max(parent_current_rss,envelope_and_decode_bytes+"
            "adapter_retained_graph_bytes+64MiB)*2+variant_suite_resident_bytes+"
            "recognizer_live_output_bytes+invariance_worker_artifact_bytes"
        ),
        host_available_memory_safety_margin_bytes=(
            HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ),
        admissible=estimated_rss <= MAXIMUM_ESTIMATED_RSS_BYTES,
    )


def _resource_plan(
    rows: int,
    partitions: tuple[_Partition, ...],
    label_class_counts: tuple[int, ...],
    *,
    preextraction_feature_matrix_bytes_bound: int | None = None,
    feature_matrix_bytes_bound: int | None = None,
    envelope_and_decode_bytes_bound: int | None = None,
    adapter_retained_graph_bytes_bound: int | None = None,
    partition_projection: _PartitionAllocationProjectionV1,
    partition_preparation_deadline_ns: int,
    invariance_plan: _InvariantResourcePlanV1,
) -> ResourcePlanV1:
    if type(partition_projection) is not _PartitionAllocationProjectionV1:
        raise TypeError("resource plan requires the exact partition projection")
    partition_projection.__post_init__()
    if (
        type(partition_preparation_deadline_ns) is not int
        or partition_projection.rows != rows
        or len(partition_projection.cardinalities) != len(partitions)
        or tuple(len(item.masks) for item in partitions)
        != partition_projection.cardinalities
    ):
        _reject(
            "FAIL_PARTITION_PROJECTION_REPLAY",
            "resource-plan partition projection/deadline drift",
        )
    _remaining_partition_preparation_deadline_seconds(
        partition_preparation_deadline_ns
    )
    features = len(partitions)
    unique_partitions = tuple(
        {partition.masks: partition for partition in partitions}.values()
    )
    maximum_values = max((len(item.masks) for item in partitions), default=0)
    maximum_classes = max(label_class_counts, default=0)
    # Each contingency metric is computed once and retained in frozen
    # permutation order.  The second operation is only a scalar omnibus scan.
    intersections_per_all_targets = sum(
        sum(
            len(partition.masks) * class_count
            for partition in unique_partitions
        )
        for class_count in label_class_counts
    )
    # The observed member is a full contingency evaluation too.  Resource
    # accounting therefore uses B+1 even though only B label masks are stored.
    estimated = EXCHANGEABLE_SAMPLE_SIZE * intersections_per_all_targets
    bias_center_work = (
        len(unique_partitions)
        * len(TARGETS)
        * (
            EXCHANGEABLE_SAMPLE_SIZE * BINARY64_SUPERACCUMULATOR_WORDS
            + BINARY64_SUPERACCUMULATOR_WORDS**2
        )
    )
    scalar_omnibus_scans = (
        LABEL_PERMUTATIONS * len(unique_partitions) * len(TARGETS)
    )
    maximum_label_shuffle_calls = (
        LABEL_PERMUTATIONS * len(TARGETS) * max(0, rows - 1)
    )
    estimated_total_work = (
        estimated
        + bias_center_work
        + scalar_omnibus_scans
        + maximum_label_shuffle_calls
    )
    avoided = rows * features * len(TARGETS) * LABEL_PERMUTATIONS
    mask_bytes = LABEL_PERMUTATIONS * sum(label_class_counts) * (
        64 + ((rows + 7) // 8)
    )
    maximum_cache_cells = min(
        MAXIMUM_METRIC_CACHE_CELLS,
        maximum_values * maximum_classes,
    )
    cache_bytes = MAXIMUM_METRIC_CACHE_ENTRIES * (
        256 + 64 * maximum_cache_cells
    )
    feature_matrix_floor = rows * features * 72
    envelope_and_decode_floor = rows * ENVELOPE_BYTES * 2
    # This is the sole deep traversal of the materialized partition graph.
    # The producer cannot substitute a caller-provided scalar for this value.
    measured_partition_floor = (
        DEEP_SIZE_SAFETY_MULTIPLIER * _deep_object_graph_bytes(partitions)
    )
    _remaining_partition_preparation_deadline_seconds(
        partition_preparation_deadline_ns
    )
    if (
        measured_partition_floor
        > partition_projection.projected_partition_matrix_bytes_bound
    ):
        _reject(
            "FAIL_PARTITION_ALLOCATION_BOUND",
            "materialized partition graph exceeds its conservative preallocation projection",
        )
    if feature_matrix_bytes_bound is not None and (
        type(feature_matrix_bytes_bound) is not int
        or feature_matrix_bytes_bound < feature_matrix_floor
    ):
        raise ValueError("measured feature-matrix bound is below its frozen floor")
    if envelope_and_decode_bytes_bound is not None and (
        type(envelope_and_decode_bytes_bound) is not int
        or envelope_and_decode_bytes_bound < envelope_and_decode_floor
    ):
        raise ValueError("measured envelope/decode bound is below its frozen floor")
    feature_matrix_bytes = (
        feature_matrix_bytes_bound
        if feature_matrix_bytes_bound is not None
        else feature_matrix_floor
    )
    envelope_and_decode_bytes = (
        envelope_and_decode_bytes_bound
        if envelope_and_decode_bytes_bound is not None
        else envelope_and_decode_floor
    )
    if preextraction_feature_matrix_bytes_bound is not None and (
        type(preextraction_feature_matrix_bytes_bound) is not int
        or preextraction_feature_matrix_bytes_bound < feature_matrix_bytes
    ):
        raise ValueError("pre-extraction feature bound is below the measured feature graph")
    preextraction_feature_bytes = (
        preextraction_feature_matrix_bytes_bound
        if preextraction_feature_matrix_bytes_bound is not None
        else feature_matrix_bytes
    )
    partition_matrix_bytes = measured_partition_floor
    partition_allocation_bytes = (
        partition_projection.projected_partition_matrix_bytes_bound
    )
    effective_cpu_capacity = _linux_effective_cpu_capacity()
    execution_mode = "LINUX_RAW_FORK_WAIT4_ONE_OR_FIVE_TARGET_WORKERS"
    worker_result_artifact_bytes = len(TARGETS) * (
        features * WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE
        + WORKER_RESULT_ARTIFACT_FIXED_BYTES
    )
    parent_result_working_bytes = _parent_result_working_bytes_bound(features)
    if (
        type(adapter_retained_graph_bytes_bound) is not int
        or not 0 < adapter_retained_graph_bytes_bound
        <= MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES
        + MAXIMUM_RECOGNIZER_ADAPTER_RETAINED_GRAPH_BYTES
    ):
        raise ValueError("adapter retained-graph resource bound is absent or invalid")
    if type(invariance_plan) is not _InvariantResourcePlanV1:
        raise TypeError("statistics plan requires the exact invariant phase plan")
    invariance_plan.__post_init__()
    if (
        invariance_plan.rows != rows
        or invariance_plan.envelope_and_decode_bytes_bound
        != envelope_and_decode_bytes
        or invariance_plan.adapter_retained_graph_bytes_bound
        != adapter_retained_graph_bytes_bound
    ):
        raise ValueError("statistics/invariant phase resource inputs drifted")
    # The statistics parent baseline is sampled only after the sole partition
    # measurement and all plan-side temporary computations above.
    parent_current_rss_bytes_at_plan = _linux_current_rss_bytes()
    resident_parent_bound = max(
        parent_current_rss_bytes_at_plan,
        mask_bytes
        + partition_matrix_bytes
        + adapter_retained_graph_bytes_bound
        + 64 * 1024 * 1024,
    )

    def statistics_rss_for_workers(workers: int) -> int:
        return (
            resident_parent_bound * (1 + workers)
            + cache_bytes * workers
            + LABEL_PERMUTATIONS * 256 * workers
            + LABEL_PERMUTATIONS
            * TARGET_MAX_VECTOR_BYTES_PER_PERMUTATION
            * workers
            + worker_result_artifact_bytes
            + parent_result_working_bytes
        )

    five_worker_eligible = (
        estimated >= PARALLEL_STATISTICS_WORK_THRESHOLD
        and effective_cpu_capacity >= len(TARGETS)
        and statistics_rss_for_workers(len(TARGETS))
        <= MAXIMUM_ESTIMATED_RSS_BYTES
    )
    parallel_workers = len(TARGETS) if five_worker_eligible else 1
    permutation_metric_tuple_bytes = (
        LABEL_PERMUTATIONS * 256 * parallel_workers
    )
    target_max_vector_bytes = (
        LABEL_PERMUTATIONS
        * TARGET_MAX_VECTOR_BYTES_PER_PERMUTATION
        * parallel_workers
    )
    # The public cap is deliberately expressed as aggregate SELF+CHILD RSS,
    # so shared fork pages are conservatively counted once for the parent and
    # once for every worker even though their physical PSS is normally shared.
    estimated_rss = statistics_rss_for_workers(parallel_workers)
    admissible = (
        1 <= rows <= MAXIMUM_ROWS
        and 1 <= features <= MAXIMUM_FEATURES
        and maximum_values <= MAXIMUM_VALUES_PER_FEATURE
        and maximum_classes <= MAXIMUM_LABEL_CLASSES
        and estimated <= MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS
        and estimated_total_work <= MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS
        and estimated_rss <= MAXIMUM_ESTIMATED_RSS_BYTES
        and invariance_plan.admissible
    )
    return ResourcePlanV1(
        rows=rows,
        concrete_features=features,
        unique_statistical_partitions=len(unique_partitions),
        hypotheses=features * len(TARGETS),
        maximum_feature_values=maximum_values,
        maximum_label_classes=maximum_classes,
        bitset_words_per_mask=(rows + 63) // 64,
        permutation_passes=1,
        label_permutations=LABEL_PERMUTATIONS,
        exchangeable_sample_size=EXCHANGEABLE_SAMPLE_SIZE,
        intersections_per_exchangeable_draw=intersections_per_all_targets,
        estimated_intersection_work_units=estimated,
        bias_center_canonicalization_work_units=bias_center_work,
        scalar_omnibus_scan_work_units=scalar_omnibus_scans,
        estimated_total_statistical_work_units=estimated_total_work,
        maximum_label_shuffle_randbelow_calls=(
            maximum_label_shuffle_calls
        ),
        maximum_shuffle_rejection_draws=(
            LABEL_PERMUTATIONS
            * len(TARGETS)
            * max(0, rows - 1)
            * MAXIMUM_SHUFFLE_REJECTION_DRAWS
        ),
        maximum_intersection_work_units=MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS,
        maximum_total_statistical_work_units=(
            MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS
        ),
        scalar_row_feature_permutation_work_avoided=avoided,
        permutation_label_mask_bytes_bound=mask_bytes,
        metric_cache_bytes_bound=cache_bytes,
        permutation_metric_tuple_bytes_bound=permutation_metric_tuple_bytes,
        target_max_vector_bytes_bound=target_max_vector_bytes,
        worker_result_artifact_bytes_bound=worker_result_artifact_bytes,
        preextraction_feature_matrix_bytes_bound=preextraction_feature_bytes,
        feature_matrix_bytes_bound=feature_matrix_bytes,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes,
        partition_allocation_bytes_bound=partition_allocation_bytes,
        partition_matrix_bytes_bound=partition_matrix_bytes,
        parent_result_working_bytes_bound=parent_result_working_bytes,
        adapter_retained_graph_bytes_bound=adapter_retained_graph_bytes_bound,
        parent_current_rss_bytes_at_plan=parent_current_rss_bytes_at_plan,
        invariance_parent_current_rss_bytes_at_plan=(
            invariance_plan.parent_current_rss_bytes_at_plan
        ),
        invariance_resource_plan_root=invariance_plan.plan_root,
        invariance_variant_suite_output_bytes_bound=(
            invariance_plan.invariance_variant_suite_output_bytes_bound
        ),
        variant_suite_resident_bytes_per_row_bound=(
            invariance_plan.variant_suite_resident_bytes_per_row_bound
        ),
        variant_suite_resident_fixed_bytes_bound=(
            invariance_plan.variant_suite_resident_fixed_bytes_bound
        ),
        invariance_variant_suite_resident_bytes_bound=(
            invariance_plan.invariance_variant_suite_resident_bytes_bound
        ),
        invariance_recognizer_live_output_bytes_bound=(
            invariance_plan.invariance_recognizer_live_output_bytes_bound
        ),
        invariance_worker_artifact_bytes_bound=(
            invariance_plan.invariance_worker_artifact_bytes_bound
        ),
        estimated_invariance_peak_rss_bytes=(
            invariance_plan.estimated_invariance_peak_rss_bytes
        ),
        maximum_invariance_estimated_rss_bytes=(
            invariance_plan.maximum_invariance_estimated_rss_bytes
        ),
        invariance_rss_formula=invariance_plan.invariance_rss_formula,
        estimated_peak_rss_bytes=estimated_rss,
        maximum_estimated_rss_bytes=MAXIMUM_ESTIMATED_RSS_BYTES,
        aggregate_rss_formula=(
            "max(parent_current_rss,mask_bytes+partition_matrix_bytes+"
            "adapter_retained_graph_bytes+64MiB)"
            "*(1+workers)+metric_cache_bytes*workers+permutation_metric_tuple_bytes"
            "+target_max_vector_bytes+worker_result_artifact_bytes+"
            "parent_result_working_bytes"
        ),
        host_available_memory_safety_margin_bytes=(
            HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ),
        partition_preparation_timeout_seconds=(
            PARTITION_PREPARATION_TIMEOUT_SECONDS
        ),
        invariance_worker_timeout_seconds=INVARIANCE_WORKER_TIMEOUT_SECONDS,
        statistics_worker_timeout_seconds=STATISTICS_WORKER_TIMEOUT_SECONDS,
        statistics_execution_mode=execution_mode,
        effective_cpu_capacity_at_preflight=effective_cpu_capacity,
        maximum_parallel_workers=parallel_workers,
        admissible=admissible,
    )


def _symmetric_nmi_center(
    observed_nmi: float,
    permuted_nmis: tuple[float, ...],
) -> float:
    """Permutation-invariant binary64 center over the complete B+1 multiset."""

    if type(observed_nmi) is not float or not math.isfinite(observed_nmi):
        raise TypeError("symmetric NMI center needs one finite observed float")
    if type(permuted_nmis) is not tuple or len(permuted_nmis) != LABEL_PERMUTATIONS:
        raise ValueError("symmetric NMI center needs exactly 10,000 permuted values")
    if any(type(value) is not float or not math.isfinite(value) for value in permuted_nmis):
        raise TypeError("symmetric NMI center values must be finite exact floats")
    # Every nonnegative binary64 value is an exact integer multiple of 2^-1074.
    # Accumulating those integers is associative and independent of which of
    # the 10,001 exchangeable members was designated observed.  One exact
    # Fraction division and pinned-runtime correctly-rounded conversion occurs
    # only after the complete multiset has been accumulated.
    scaled_cache: dict[float, int] = {}

    def scaled_integer(value: float) -> int:
        if not 0.0 <= value <= 1.0:
            raise ValueError("symmetric NMI center value is outside [0,1]")
        if value == 0.0:
            return 0
        cached = scaled_cache.get(value)
        if cached is not None:
            return cached
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        if bits >> 63:
            raise ValueError("symmetric NMI center cannot accumulate a negative value")
        exponent = (bits >> 52) & 0x7FF
        fraction = bits & ((1 << 52) - 1)
        if exponent == 0:
            significand = fraction
            shift = 0
        else:
            significand = (1 << 52) | fraction
            shift = exponent - 1
        scaled = significand << shift
        if len(scaled_cache) < MAXIMUM_METRIC_CACHE_ENTRIES:
            scaled_cache[value] = scaled
        return scaled

    scaled_sum = scaled_integer(observed_nmi)
    for value in permuted_nmis:
        scaled_sum += scaled_integer(value)
    exact_center = Fraction(
        scaled_sum,
        EXCHANGEABLE_SAMPLE_SIZE
        << BINARY64_SUPERACCUMULATOR_FRACTION_BITS,
    )
    return float(exact_center)


def _evaluate_target_statistics(
    partitions: tuple[_Partition, ...],
    observed_masks: tuple[int, ...],
    permutation_masks: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[tuple[float, float, float, int, int], ...],
    float,
    int,
]:
    if len(permutation_masks) != LABEL_PERMUTATIONS:
        raise ValueError("target maxT evaluation requires the exact permutation count")
    partition_statistics: dict[
        tuple[int, ...], tuple[float, float, float, int, float]
    ] = {}
    permutation_target_max = [0.0] * LABEL_PERMUTATIONS
    result_keys: list[tuple[int, ...]] = []
    for partition in partitions:
        result_keys.append(partition.masks)
        if partition.masks in partition_statistics:
            continue
        cache: dict[tuple[int, ...], tuple[float, float]] = {}
        raw_nmi, advantage = _metrics_from_masks(
            partition,
            observed_masks,
            cache,
        )
        permutation_metrics = tuple(
            _metrics_from_masks(partition, masks, cache)
            for masks in permutation_masks
        )
        # The observed statistic and every permuted statistic are treated
        # symmetrically.  This B+1 mean is the same fixed center for all
        # exchangeable members of the target-feature vector.
        mean_nmi = _symmetric_nmi_center(
            raw_nmi,
            tuple(nmi for nmi, _ in permutation_metrics),
        )
        corrected_nmi = max(0.0, raw_nmi - mean_nmi)
        omnibus = max(
            corrected_nmi / NMI_LIMIT,
            max(0.0, advantage) / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
        )
        count = 0
        for permutation_index, (nmi, permuted_advantage) in enumerate(
            permutation_metrics
        ):
            statistic = max(
                max(0.0, nmi - mean_nmi) / NMI_LIMIT,
                max(0.0, permuted_advantage)
                / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
            )
            if statistic >= omnibus:
                count += 1
            if statistic > permutation_target_max[permutation_index]:
                permutation_target_max[permutation_index] = statistic
        row = (raw_nmi, advantage, mean_nmi, count, omnibus)
        partition_statistics[partition.masks] = row
    observed_target_max = max(
        (row[4] for row in partition_statistics.values()),
        default=0.0,
    )
    sorted_target_max = sorted(permutation_target_max)
    target_exceedance_count = LABEL_PERMUTATIONS - bisect_left(
        sorted_target_max,
        observed_target_max,
    )
    feature_localization_counts = {
        masks: LABEL_PERMUTATIONS - bisect_left(sorted_target_max, row[4])
        for masks, row in partition_statistics.items()
    }
    result = tuple(
        (
            partition_statistics[masks][0],
            partition_statistics[masks][1],
            partition_statistics[masks][2],
            partition_statistics[masks][3],
            feature_localization_counts[masks],
        )
        for masks in result_keys
    )
    return result, observed_target_max, target_exceedance_count


def _linux_peak_rss_bytes() -> int:
    if os.name != "posix" or not hasattr(os, "uname") or os.uname().sysname != "Linux":
        _reject(
            "FAIL_STATISTICAL_RUNTIME",
            "frozen RSS evidence requires Linux ru_maxrss byte semantics",
        )
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1_024


def _linux_current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) == 3 and fields[2] == "kB":
                        value = int(fields[1]) * 1_024
                        if value > 0:
                            return value
    except (OSError, UnicodeError, ValueError):
        pass
    _reject(
        "FAIL_PLATFORM_RESOURCE_SEMANTICS",
        "Linux current VmRSS could not be read for invariant accounting",
    )


def _linux_current_virtual_memory_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    fields = line.split()
                    if len(fields) == 3 and fields[2] == "kB":
                        value = int(fields[1]) * 1_024
                        if value > 0:
                            return value
    except (OSError, UnicodeError, ValueError):
        pass
    _reject(
        "FAIL_PLATFORM_RESOURCE_SEMANTICS",
        "Linux current VmSize could not be read for invariant containment",
    )


def _linux_system_available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    if len(fields) == 3 and fields[2] == "kB":
                        return int(fields[1]) * 1_024
    except (OSError, UnicodeError, ValueError):
        pass
    _reject(
        "FAIL_STATISTICAL_HOST_MEMORY",
        "Linux MemAvailable could not be read for the frozen preflight",
    )


def _linux_cgroup_memory_remaining_bytes() -> int | None:
    remaining: list[int] = []
    v2_seen = False
    for directory in _cgroup_ancestor_directories(None):
        pair = _read_cgroup_control_pair(
            directory,
            "memory.max",
            "memory.current",
        )
        if pair is None:
            continue
        v2_seen = True
        maximum_text, current_text = pair
        try:
            current = int(current_text)
            maximum = None if maximum_text == "max" else int(maximum_text)
        except ValueError:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"malformed cgroup v2 memory pair at {directory}",
            )
        if current < 0 or maximum is not None and maximum < 0:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"negative cgroup v2 memory pair at {directory}",
            )
        if maximum is not None:
            remaining.append(max(0, maximum - current))
    if v2_seen:
        return min(remaining) if remaining else None

    for directory in _cgroup_ancestor_directories("memory"):
        pair = _read_cgroup_control_pair(
            directory,
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
        )
        if pair is None:
            continue
        try:
            maximum, current = (int(item) for item in pair)
        except ValueError:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"malformed cgroup v1 memory pair at {directory}",
            )
        if maximum < 0 or current < 0:
            _reject(
                "FAIL_PLATFORM_RESOURCE_SEMANTICS",
                f"negative cgroup v1 memory pair at {directory}",
            )
        # Common v1 unlimited sentinels are close to LONG_MAX.
        if maximum >= (1 << 60):
            continue
        remaining.append(max(0, maximum - current))
    return min(remaining) if remaining else None


@dataclass(frozen=True, slots=True)
class _HostMemoryPreflightV1:
    system_available_bytes: int
    cgroup_remaining_bytes: int | None
    effective_available_bytes: int

    def __post_init__(self) -> None:
        if type(self.system_available_bytes) is not int or self.system_available_bytes < 0:
            raise TypeError("system available-memory preflight value is invalid")
        if self.cgroup_remaining_bytes is not None and (
            type(self.cgroup_remaining_bytes) is not int
            or self.cgroup_remaining_bytes < 0
        ):
            raise TypeError("cgroup remaining-memory preflight value is invalid")
        expected = min(
            self.system_available_bytes,
            self.cgroup_remaining_bytes
            if self.cgroup_remaining_bytes is not None
            else self.system_available_bytes,
        )
        if self.effective_available_bytes != expected:
            raise ValueError("effective host-memory preflight value drift")


@dataclass(frozen=True, slots=True)
class _PhaseHostMemoryPreflightV1:
    system_available_bytes: int
    cgroup_remaining_bytes: int | None
    effective_available_bytes: int
    resource_preflight: _PhaseResourcePreflightEvidenceV1

    def __post_init__(self) -> None:
        _HostMemoryPreflightV1(
            self.system_available_bytes,
            self.cgroup_remaining_bytes,
            self.effective_available_bytes,
        ).__post_init__()
        if type(self.resource_preflight) is not _PhaseResourcePreflightEvidenceV1:
            raise TypeError("phase host preflight lacks exact resource evidence")
        self.resource_preflight.__post_init__()


def _host_memory_snapshot() -> _HostMemoryPreflightV1:
    system_available = _linux_system_available_memory_bytes()
    cgroup_remaining = _linux_cgroup_memory_remaining_bytes()
    available = min(
        system_available,
        cgroup_remaining if cgroup_remaining is not None else system_available,
    )
    return _HostMemoryPreflightV1(
        system_available,
        cgroup_remaining,
        available,
    )


def _preflight_decode_memory(rows: int) -> _HostMemoryPreflightV1:
    evidence = _host_memory_snapshot()
    required = (
        rows * MAXIMUM_DECODE_GRAPH_BYTES_PER_ROW
        + HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
    )
    if evidence.effective_available_bytes < required:
        _reject(
            "FAIL_DECODE_HOST_MEMORY",
            "effective MemAvailable/cgroup remaining is below the bounded "
            "per-row decode projection plus safety margin",
        )
    return evidence


def _preflight_feature_extraction_memory(
    projection: _FeatureCardinalityProjectionV1,
    envelope_and_decode_bytes_bound: int,
) -> _HostMemoryPreflightV1:
    projection.__post_init__()
    if (
        type(envelope_and_decode_bytes_bound) is not int
        or envelope_and_decode_bytes_bound < projection.rows * ENVELOPE_BYTES * 2
    ):
        raise ValueError("feature preflight envelope/decode bound is below its floor")
    result_bound = _parent_result_working_bytes_bound(
        projection.projected_concrete_feature_count
    )
    artifact_bound = len(TARGETS) * (
        projection.projected_concrete_feature_count
        * WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE
        + WORKER_RESULT_ARTIFACT_FIXED_BYTES
    )
    evidence = _host_memory_snapshot()
    current_rss = _linux_current_rss_bytes()
    required_without_margin = (
        max(current_rss, envelope_and_decode_bytes_bound)
        + projection.projected_feature_matrix_bytes_bound
        + result_bound
        + artifact_bound
    )
    if required_without_margin > MAXIMUM_ESTIMATED_RSS_BYTES:
        _reject(
            "FAIL_FEATURE_PROJECTION_RESOURCE_CAP",
            "pre-extraction feature/result projection exceeds the frozen aggregate cap",
        )
    if evidence.effective_available_bytes < (
        required_without_margin + HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
    ):
        _reject(
            "FAIL_FEATURE_PROJECTION_HOST_MEMORY",
            "effective MemAvailable/cgroup remaining is below the projected "
            "feature/result allocation plus safety margin",
        )
    return evidence


def _preflight_partition_allocation_memory(
    *,
    feature_matrix_bytes_bound: int,
    envelope_and_decode_bytes_bound: int,
    partition_projection: _PartitionAllocationProjectionV1,
    concrete_features: int,
) -> _HostMemoryPreflightV1:
    partition_projection.__post_init__()
    if (
        type(feature_matrix_bytes_bound) is not int
        or feature_matrix_bytes_bound < 0
        or type(envelope_and_decode_bytes_bound) is not int
        or envelope_and_decode_bytes_bound
        < partition_projection.rows * ENVELOPE_BYTES * 2
    ):
        raise ValueError("partition preflight resident bound is invalid")
    result_bound = _parent_result_working_bytes_bound(concrete_features)
    evidence = _host_memory_snapshot()
    current_rss = _linux_current_rss_bytes()
    required_without_margin = (
        max(
            current_rss,
            feature_matrix_bytes_bound + envelope_and_decode_bytes_bound,
        )
        + partition_projection.projected_partition_matrix_bytes_bound
        + result_bound
    )
    if required_without_margin > MAXIMUM_ESTIMATED_RSS_BYTES:
        _reject(
            "FAIL_PARTITION_PROJECTION_RESOURCE_CAP",
            "projected parent partition/result allocation exceeds the frozen aggregate cap",
        )
    if evidence.effective_available_bytes < (
        required_without_margin + HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
    ):
        _reject(
            "FAIL_PARTITION_PROJECTION_HOST_MEMORY",
            "effective MemAvailable/cgroup remaining is below the projected "
            "partition/result allocation plus safety margin",
        )
    return evidence


def _preflight_host_memory(plan: ResourcePlanV1) -> _PhaseHostMemoryPreflightV1:
    plan.__post_init__()
    evidence = _host_memory_snapshot()
    available = evidence.effective_available_bytes
    current_rss = _linux_current_rss_bytes()
    static_parent = (
        plan.permutation_label_mask_bytes_bound
        + plan.partition_matrix_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    multiplier = 1 + plan.maximum_parallel_workers
    additional = (
        plan.metric_cache_bytes_bound * plan.maximum_parallel_workers
        + plan.permutation_metric_tuple_bytes_bound
        + plan.target_max_vector_bytes_bound
        + plan.worker_result_artifact_bytes_bound
        + plan.parent_result_working_bytes_bound
    )
    runtime_projection = max(current_rss, static_parent) * multiplier + additional
    required = (
        runtime_projection
        + plan.host_available_memory_safety_margin_bytes
    )
    if runtime_projection > plan.maximum_estimated_rss_bytes:
        _reject(
            "FAIL_STATISTICAL_RESOURCE_PLAN",
            "projected aggregate SELF+CHILD RSS exceeds the frozen cap",
        )
    phase_evidence = _PhaseResourcePreflightEvidenceV1(
        phase="STATISTICS",
        resource_plan_root=plan.resource_plan_root,
        parent_current_rss_bytes_at_preflight=current_rss,
        static_parent_bytes_bound=static_parent,
        fork_parent_rss_multiplier=multiplier,
        additional_bytes_bound=additional,
        projected_aggregate_rss_bytes_at_preflight=runtime_projection,
        maximum_estimated_rss_bytes=plan.maximum_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    if available < required:
        _reject(
            "FAIL_STATISTICAL_HOST_MEMORY",
            "MemAvailable is below projected aggregate RSS plus the frozen safety margin",
        )
    return _PhaseHostMemoryPreflightV1(
        evidence.system_available_bytes,
        evidence.cgroup_remaining_bytes,
        evidence.effective_available_bytes,
        phase_evidence,
    )


def _preflight_invariance_host_memory(
    plan: _InvariantResourcePlanV1,
) -> _PhaseHostMemoryPreflightV1:
    plan.__post_init__()
    evidence = _host_memory_snapshot()
    current_rss = _linux_current_rss_bytes()
    static_parent = (
        plan.envelope_and_decode_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    additional = (
        plan.invariance_variant_suite_resident_bytes_bound
        + plan.invariance_recognizer_live_output_bytes_bound
        + plan.invariance_worker_artifact_bytes_bound
    )
    runtime_projection = max(current_rss, static_parent) * 2 + additional
    if runtime_projection > plan.maximum_invariance_estimated_rss_bytes:
        _reject(
            "FAIL_INVARIANCE_RESOURCE_PLAN",
            "projected invariant parent+worker RSS exceeds the frozen cap",
        )
    phase_evidence = _PhaseResourcePreflightEvidenceV1(
        phase="INVARIANCE",
        resource_plan_root=plan.plan_root,
        parent_current_rss_bytes_at_preflight=current_rss,
        static_parent_bytes_bound=static_parent,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=additional,
        projected_aggregate_rss_bytes_at_preflight=runtime_projection,
        maximum_estimated_rss_bytes=(
            plan.maximum_invariance_estimated_rss_bytes
        ),
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    if evidence.effective_available_bytes < (
        runtime_projection
        + plan.host_available_memory_safety_margin_bytes
    ):
        _reject(
            "FAIL_INVARIANCE_HOST_MEMORY",
            "MemAvailable is below invariant parent+worker projection plus margin",
        )
    return _PhaseHostMemoryPreflightV1(
        evidence.system_available_bytes,
        evidence.cgroup_remaining_bytes,
        evidence.effective_available_bytes,
        phase_evidence,
    )


def _validate_invariant_phase_preflight_against_resource_plan(
    plan: ResourcePlanV1,
    evidence: _PhaseResourcePreflightEvidenceV1,
) -> None:
    plan.__post_init__()
    evidence.__post_init__()
    expected = (
        "INVARIANCE",
        plan.invariance_resource_plan_root,
        plan.envelope_and_decode_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576,
        2,
        plan.invariance_variant_suite_resident_bytes_bound
        + plan.invariance_recognizer_live_output_bytes_bound
        + plan.invariance_worker_artifact_bytes_bound,
        plan.maximum_invariance_estimated_rss_bytes,
        plan.host_available_memory_safety_margin_bytes,
    )
    observed = (
        evidence.phase,
        evidence.resource_plan_root,
        evidence.static_parent_bytes_bound,
        evidence.fork_parent_rss_multiplier,
        evidence.additional_bytes_bound,
        evidence.maximum_estimated_rss_bytes,
        evidence.host_available_memory_safety_margin_bytes,
    )
    if observed != expected:
        raise ValueError("invariant phase-preflight/resource-plan splice drift")


def _validate_statistics_phase_preflight_against_resource_plan(
    plan: ResourcePlanV1,
    evidence: _PhaseResourcePreflightEvidenceV1,
) -> None:
    plan.__post_init__()
    evidence.__post_init__()
    expected = (
        "STATISTICS",
        plan.resource_plan_root,
        plan.permutation_label_mask_bytes_bound
        + plan.partition_matrix_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576,
        1 + plan.maximum_parallel_workers,
        plan.metric_cache_bytes_bound * plan.maximum_parallel_workers
        + plan.permutation_metric_tuple_bytes_bound
        + plan.target_max_vector_bytes_bound
        + plan.worker_result_artifact_bytes_bound
        + plan.parent_result_working_bytes_bound,
        plan.maximum_estimated_rss_bytes,
        plan.host_available_memory_safety_margin_bytes,
    )
    observed = (
        evidence.phase,
        evidence.resource_plan_root,
        evidence.static_parent_bytes_bound,
        evidence.fork_parent_rss_multiplier,
        evidence.additional_bytes_bound,
        evidence.maximum_estimated_rss_bytes,
        evidence.host_available_memory_safety_margin_bytes,
    )
    if observed != expected:
        raise ValueError("statistics phase-preflight/resource-plan splice drift")


@dataclass(frozen=True, slots=True)
class _StatisticsWorkerArtifactV1:
    target_index: int
    worker_pid: int
    path: str
    serialized_bytes: int
    sha256: str
    worker_peak_rss_bytes: int

    def __post_init__(self) -> None:
        if type(self.target_index) is not int or not 0 <= self.target_index < len(TARGETS):
            raise ValueError("statistics worker artifact target index drift")
        if type(self.worker_pid) is not int or self.worker_pid <= 0:
            raise ValueError("statistics worker artifact PID drift")
        _exact_nonempty_str(self.path, "statistics worker artifact path")
        if type(self.serialized_bytes) is not int or self.serialized_bytes <= 0:
            raise ValueError("statistics worker artifact size must be positive")
        if type(self.sha256) is not str or re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.sha256
        ) is None:
            raise ValueError("statistics worker artifact digest drift")
        if type(self.worker_peak_rss_bytes) is not int or self.worker_peak_rss_bytes <= 0:
            raise ValueError("statistics worker artifact RSS evidence drift")


@dataclass(frozen=True, slots=True)
class _InvariantWorkerArtifactV1:
    worker_pid: int
    path: str
    serialized_bytes: int
    sha256: str
    worker_peak_rss_bytes: int

    def __post_init__(self) -> None:
        if type(self.worker_pid) is not int or self.worker_pid <= 0:
            raise ValueError("invariant worker artifact PID drift")
        _exact_nonempty_str(self.path, "invariant worker artifact path")
        if (
            type(self.serialized_bytes) is not int
            or not 0 < self.serialized_bytes
            <= INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
            or type(self.sha256) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.sha256) is None
            or type(self.worker_peak_rss_bytes) is not int
            or self.worker_peak_rss_bytes <= 0
        ):
            raise ValueError("invariant worker artifact evidence drift")


def _write_statistics_worker_artifact(
    target_index: int,
    rows: tuple[tuple[tuple[float, float, float, int, int], ...], float, int],
    artifact_directory: str,
    artifact_directory_fd: int,
    maximum_bytes: int,
) -> None:
    validated = _validate_target_statistics_payload(rows, len(rows[0]))
    body = {
        "artifact_schema": STATISTICS_WORKER_ARTIFACT_SCHEMA,
        "observed_target_max_hex": validated[1].hex(),
        "target_exceedance_count_decimal": str(validated[2]),
        "target_index_decimal": str(target_index),
        "rows": [
            [
                row[0].hex(),
                row[1].hex(),
                row[2].hex(),
                str(row[3]),
                str(row[4]),
            ]
            for row in validated[0]
        ],
    }
    payload = canonicalize_rfc8785_v1(body)
    if len(payload) > maximum_bytes:
        raise MemoryError(
            "statistics worker result serialization exceeded its frozen per-worker bound"
        )
    temporary_name = f".target-{target_index}-{os.getpid()}.tmp"
    final_name = f"target-{target_index}.json"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        temporary_fd = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=artifact_directory_fd,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(temporary_fd, payload[offset:])
                if written <= 0:
                    raise OSError("statistics artifact write made no progress")
                offset += written
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        os.rename(
            temporary_name,
            final_name,
            src_dir_fd=artifact_directory_fd,
            dst_dir_fd=artifact_directory_fd,
        )
        os.fsync(artifact_directory_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=artifact_directory_fd)
        except FileNotFoundError:
            pass
    del payload


def _validate_target_statistics_payload(
    value: object,
    expected_feature_count: int,
) -> tuple[tuple[tuple[float, float, float, int, int], ...], float, int]:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("statistics worker artifact payload has an invalid outer shape")
    rows, observed_target_max, exceedance_count = value
    if type(rows) is not tuple or len(rows) != expected_feature_count:
        raise ValueError("statistics worker artifact feature count drift")
    for row in rows:
        if type(row) is not tuple or len(row) != 5:
            raise TypeError("statistics worker artifact row shape drift")
        if any(type(item) is not float or not math.isfinite(item) for item in row[:3]):
            raise TypeError("statistics worker artifact metric type drift")
        if any(type(item) is not int or not 0 <= item <= LABEL_PERMUTATIONS for item in row[3:]):
            raise ValueError("statistics worker artifact exceedance count drift")
    if type(observed_target_max) is not float or not math.isfinite(observed_target_max):
        raise TypeError("statistics worker artifact target maximum drift")
    if type(exceedance_count) is not int or not 0 <= exceedance_count <= LABEL_PERMUTATIONS:
        raise ValueError("statistics worker artifact target exceedance drift")
    return value


def _read_statistics_worker_artifact(
    target_index: int,
    artifact_directory: str,
    artifact_directory_fd: int,
    artifact_directory_device: int,
    artifact_directory_inode: int,
    maximum_bytes: int,
    expected_feature_count: int,
    worker_pid: int,
    worker_peak_rss_bytes: int,
) -> tuple[
    tuple[tuple[tuple[float, float, float, int, int], ...], float, int],
    _StatisticsWorkerArtifactV1,
]:
    if type(target_index) is not int or not 0 <= target_index < len(TARGETS):
        raise ValueError("statistics worker artifact target index drift")
    expected_name = f"target-{target_index}.json"
    retained = os.fstat(artifact_directory_fd)
    if (
        not stat.S_ISDIR(retained.st_mode)
        or retained.st_dev != artifact_directory_device
        or retained.st_ino != artifact_directory_inode
    ):
        raise ValueError("retained statistics artifact directory identity drift")
    path_stat = os.lstat(artifact_directory)
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or path_stat.st_dev != artifact_directory_device
        or path_stat.st_ino != artifact_directory_inode
    ):
        raise ValueError("statistics artifact directory path was replaced")
    before = os.stat(
        expected_name,
        dir_fd=artifact_directory_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise ValueError("statistics worker artifact byte/type/link drift")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    artifact_fd = os.open(expected_name, flags, dir_fd=artifact_directory_fd)
    try:
        after = os.fstat(artifact_fd)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_nlink != 1
            or after.st_size != before.st_size
        ):
            raise ValueError("statistics artifact changed during no-follow open")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(artifact_fd, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("statistics artifact exceeded its byte bound")
        payload = b"".join(chunks)
    finally:
        os.close(artifact_fd)
    if len(payload) != before.st_size:
        raise ValueError("statistics worker artifact read-size drift")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    try:
        body = decode_and_require_canonical_rfc8785_v1(payload)
    except (UnicodeError, RFC8785Error, TypeError, ValueError) as exc:
        raise ValueError("statistics artifact is not canonical closed JSON") from exc
    del payload
    if type(body) is not dict or set(body) != {
        "artifact_schema",
        "observed_target_max_hex",
        "rows",
        "target_exceedance_count_decimal",
        "target_index_decimal",
    }:
        raise ValueError("statistics artifact closed schema drift")
    if body["artifact_schema"] != STATISTICS_WORKER_ARTIFACT_SCHEMA:
        raise ValueError("statistics artifact schema identifier drift")
    if body["target_index_decimal"] != str(target_index):
        raise ValueError("statistics artifact target index binding drift")
    rows_body = body["rows"]
    if type(rows_body) is not list or len(rows_body) != expected_feature_count:
        raise ValueError("statistics artifact feature count drift")
    decoded_rows: list[tuple[float, float, float, int, int]] = []
    for row in rows_body:
        if (
            type(row) is not list
            or len(row) != 5
            or any(type(value) is not str for value in row)
            or any(
                len(value) > 32
                or re.fullmatch(
                    r"-?0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+",
                    value,
                )
                is None
                for value in row[:3]
            )
            or re.fullmatch(r"0|[1-9][0-9]{0,4}", row[3]) is None
            or re.fullmatch(r"0|[1-9][0-9]{0,4}", row[4]) is None
        ):
            raise ValueError("statistics artifact row encoding drift")
        try:
            metrics = tuple(float.fromhex(value) for value in row[:3])
        except ValueError as exc:
            raise ValueError("statistics artifact float.hex encoding drift") from exc
        if any(value.hex() != encoded for value, encoded in zip(metrics, row[:3])):
            raise ValueError("statistics artifact float.hex canonicalization drift")
        decoded_rows.append(
            (metrics[0], metrics[1], metrics[2], int(row[3]), int(row[4]))
        )
    observed_hex = body["observed_target_max_hex"]
    exceedance_decimal = body["target_exceedance_count_decimal"]
    if (
        type(observed_hex) is not str
        or type(exceedance_decimal) is not str
        or len(observed_hex) > 32
        or re.fullmatch(
            r"-?0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+",
            observed_hex,
        )
        is None
        or re.fullmatch(r"0|[1-9][0-9]{0,4}", exceedance_decimal) is None
    ):
        raise ValueError("statistics artifact target summary encoding drift")
    try:
        observed_target_max = float.fromhex(observed_hex)
    except ValueError as exc:
        raise ValueError("statistics artifact target float.hex encoding drift") from exc
    if observed_target_max.hex() != observed_hex:
        raise ValueError("statistics artifact target float.hex canonicalization drift")
    decoded = _validate_target_statistics_payload(
        (
            tuple(decoded_rows),
            observed_target_max,
            int(exceedance_decimal),
        ),
        expected_feature_count,
    )
    artifact = _StatisticsWorkerArtifactV1(
        target_index=target_index,
        worker_pid=worker_pid,
        path=str(Path(artifact_directory) / expected_name),
        serialized_bytes=before.st_size,
        sha256=digest,
        worker_peak_rss_bytes=worker_peak_rss_bytes,
    )
    artifact.__post_init__()
    return decoded, artifact


def _write_invariant_child_artifact(
    result: _InvariantExecutionResultV1,
    final_path: str,
    maximum_bytes: int,
) -> None:
    result.__post_init__()
    final = Path(final_path)
    if final.name != "invariant-result.json" or not final.parent.is_absolute():
        raise ValueError("invariant child artifact path is not exact and predeclared")
    body = {
        "adapter_execution_pin_root": result.adapter_execution_pin_root,
        "artifact_schema": INVARIANCE_WORKER_ARTIFACT_SCHEMA,
        "consumer_decode_count_decimal": str(result.consumer_decode_count),
        "exact_invariant_comparisons_decimal": str(
            result.exact_invariant_comparisons
        ),
        "fork_input_snapshot_root": result.fork_input_snapshot_root,
        "invariant_execution_root": result.invariant_execution_root,
        "invariant_run_count_decimal": str(result.invariant_run_count),
        "recognizer_output_root": result.recognizer_output_root,
        "variant_suite_resident_object_graph_bytes_observed_decimal": str(
            result.variant_suite_resident_object_graph_bytes_observed
        ),
        "variant_suite_root": result.variant_suite_root,
        "variant_suite_unique_envelope_object_count_decimal": str(
            result.variant_suite_unique_envelope_object_count
        ),
    }
    payload = canonicalize_rfc8785_v1(body)
    if not 0 < len(payload) <= maximum_bytes:
        raise MemoryError("invariant result serialization exceeded its frozen bound")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(str(final.parent), directory_flags)
    temporary_name = ".invariant-result.tmp"
    try:
        try:
            os.stat(final.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("predeclared invariant result already exists")
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        temporary_fd = os.open(
            temporary_name,
            file_flags,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(temporary_fd, payload[offset:])
                if written <= 0:
                    raise OSError("invariant artifact write made no progress")
                offset += written
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        os.rename(
            temporary_name,
            final.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _read_invariant_worker_artifact(
    artifact_name: str,
    artifact_directory: str,
    artifact_directory_fd: int,
    artifact_directory_device: int,
    artifact_directory_inode: int,
    maximum_bytes: int,
    worker_pid: int,
    worker_peak_rss_bytes: int,
) -> tuple[_InvariantExecutionResultV1, _InvariantWorkerArtifactV1]:
    if artifact_name != "invariant-result.json":
        raise ValueError("invariant worker artifact basename drift")
    directory_stat = os.fstat(artifact_directory_fd)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_dev != artifact_directory_device
        or directory_stat.st_ino != artifact_directory_inode
    ):
        raise ValueError("retained invariant artifact directory identity drift")
    path_stat = os.lstat(artifact_directory)
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or path_stat.st_dev != artifact_directory_device
        or path_stat.st_ino != artifact_directory_inode
    ):
        raise ValueError("invariant artifact directory path was replaced")
    before = os.stat(
        artifact_name,
        dir_fd=artifact_directory_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise ValueError("invariant worker artifact is not one bounded regular link")
    open_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    artifact_fd = os.open(
        artifact_name,
        open_flags,
        dir_fd=artifact_directory_fd,
    )
    try:
        after = os.fstat(artifact_fd)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_nlink != 1
            or after.st_size != before.st_size
        ):
            raise ValueError("invariant artifact changed during no-follow open")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(artifact_fd, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("invariant worker artifact exceeded its byte bound")
        payload = b"".join(chunks)
    finally:
        os.close(artifact_fd)
    if len(payload) != before.st_size:
        raise ValueError("invariant worker artifact read-size drift")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    try:
        body = decode_and_require_canonical_rfc8785_v1(payload)
    except (UnicodeError, RFC8785Error, TypeError, ValueError) as exc:
        raise ValueError("invariant artifact is not canonical closed JSON") from exc
    if type(body) is not dict or set(body) != {
        "adapter_execution_pin_root",
        "artifact_schema",
        "consumer_decode_count_decimal",
        "exact_invariant_comparisons_decimal",
        "fork_input_snapshot_root",
        "invariant_execution_root",
        "invariant_run_count_decimal",
        "recognizer_output_root",
        "variant_suite_resident_object_graph_bytes_observed_decimal",
        "variant_suite_root",
        "variant_suite_unique_envelope_object_count_decimal",
    }:
        raise ValueError("invariant artifact closed schema/canonical bytes drift")
    if body["artifact_schema"] != INVARIANCE_WORKER_ARTIFACT_SCHEMA:
        raise ValueError("invariant artifact schema identifier drift")
    integer_keys = (
        "consumer_decode_count_decimal",
        "exact_invariant_comparisons_decimal",
        "invariant_run_count_decimal",
        "variant_suite_resident_object_graph_bytes_observed_decimal",
        "variant_suite_unique_envelope_object_count_decimal",
    )
    if any(
        type(body[name]) is not str
        or re.fullmatch(r"0|[1-9][0-9]{0,19}", body[name]) is None
        for name in integer_keys
    ):
        raise ValueError("invariant artifact decimal integer encoding drift")
    del payload
    result = _InvariantExecutionResultV1(
        fork_input_snapshot_root=body["fork_input_snapshot_root"],
        adapter_execution_pin_root=body["adapter_execution_pin_root"],
        invariant_execution_root=body["invariant_execution_root"],
        exact_invariant_comparisons=int(
            body["exact_invariant_comparisons_decimal"]
        ),
        invariant_run_count=int(body["invariant_run_count_decimal"]),
        variant_suite_root=body["variant_suite_root"],
        consumer_decode_count=int(body["consumer_decode_count_decimal"]),
        variant_suite_unique_envelope_object_count=int(
            body["variant_suite_unique_envelope_object_count_decimal"]
        ),
        variant_suite_resident_object_graph_bytes_observed=int(
            body["variant_suite_resident_object_graph_bytes_observed_decimal"]
        ),
        recognizer_output_root=body["recognizer_output_root"],
    )
    result.__post_init__()
    artifact = _InvariantWorkerArtifactV1(
        worker_pid=worker_pid,
        path=str(Path(artifact_directory) / artifact_name),
        serialized_bytes=before.st_size,
        sha256=digest,
        worker_peak_rss_bytes=worker_peak_rss_bytes,
    )
    artifact.__post_init__()
    return result, artifact


def _cleanup_invariant_artifact_directory(
    artifact_directory: str,
    artifact_directory_fd: int,
    artifact_directory_device: int,
    artifact_directory_inode: int,
) -> None:
    """Clean only the retained directory inode; never traverse a replaced path."""

    directory = Path(artifact_directory)
    if not directory.name.startswith("hegel-formal-covert-invariant-"):
        raise ValueError("refusing to clean an unrecognized invariant artifact directory")
    retained = os.fstat(artifact_directory_fd)
    if (
        not stat.S_ISDIR(retained.st_mode)
        or retained.st_dev != artifact_directory_device
        or retained.st_ino != artifact_directory_inode
    ):
        raise ValueError("retained invariant artifact directory identity drift")
    unexpected_directories: list[str] = []
    for name in _retained_directory_entry_names(artifact_directory_fd):
        if type(name) is not str or name in {"", ".", ".."} or "/" in name:
            raise ValueError("invariant artifact directory entry name drift")
        entry = os.stat(name, dir_fd=artifact_directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry.st_mode):
            unexpected_directories.append(name)
            continue
        os.unlink(name, dir_fd=artifact_directory_fd)
    if unexpected_directories:
        raise ValueError(
            "invariant artifact directory contains unexpected subdirectories: "
            + repr(tuple(sorted(unexpected_directories)))
        )
    try:
        path_stat = os.lstat(artifact_directory)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(path_stat.st_mode):
        os.unlink(artifact_directory)
        raise ValueError("invariant artifact directory path was replaced by a symlink")
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or path_stat.st_dev != artifact_directory_device
        or path_stat.st_ino != artifact_directory_inode
    ):
        raise ValueError("invariant artifact directory path identity was replaced")
    os.rmdir(artifact_directory)


def _acquire_private_artifact_directory(
    *,
    prefix: str,
) -> tuple[str, int, int, int]:
    """Acquire one retained private directory or clean every partial resource."""

    if prefix not in {
        "hegel-formal-covert-invariant-",
        "hegel-formal-covert-stats-",
    }:
        raise ValueError("private artifact directory prefix drift")
    raw_path: str | None = None
    directory_fd: int | None = None
    initial_stat: os.stat_result | None = None
    try:
        raw_path = tempfile.mkdtemp(prefix=prefix, dir=ARTIFACT_TEMP_ROOT)
        if (
            type(raw_path) is not str
            or not os.path.isabs(raw_path)
            or not Path(raw_path).name.startswith(prefix)
        ):
            raise ValueError("private artifact directory path drift")
        initial_stat = os.lstat(raw_path)
        if not stat.S_ISDIR(initial_stat.st_mode):
            raise ValueError("private artifact path is not a directory")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(raw_path, directory_flags)
        retained = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(retained.st_mode)
            or retained.st_dev != initial_stat.st_dev
            or retained.st_ino != initial_stat.st_ino
        ):
            raise ValueError("private artifact directory identity drift during acquisition")
        return raw_path, directory_fd, retained.st_dev, retained.st_ino
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if raw_path is not None:
            try:
                current = os.lstat(raw_path)
                if stat.S_ISLNK(current.st_mode):
                    os.unlink(raw_path)
                elif (
                    stat.S_ISDIR(current.st_mode)
                    and (
                        initial_stat is None
                        or (
                            current.st_dev == initial_stat.st_dev
                            and current.st_ino == initial_stat.st_ino
                        )
                    )
                ):
                    # With no initial stat, rmdir remains non-destructive: it
                    # succeeds only for the exact empty basename just minted.
                    os.rmdir(raw_path)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            _append_exception_note(
                primary,
                "private artifact acquisition cleanup also failed: "
                + "; ".join(
                    f"{type(item).__name__}: {item}" for item in cleanup_errors
                )
            )
        raise


def _retained_directory_entry_names(directory_fd: int) -> tuple[str, ...]:
    """List through a fresh open-file description rooted at a retained dirfd."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    scan_fd = os.open(".", flags, dir_fd=directory_fd)
    try:
        return tuple(os.listdir(scan_fd))
    finally:
        os.close(scan_fd)


def _cleanup_statistics_artifact_directory(
    artifact_directory: str,
    artifact_directory_fd: int,
    artifact_directory_device: int,
    artifact_directory_inode: int,
) -> None:
    directory = Path(artifact_directory)
    if not directory.name.startswith("hegel-formal-covert-stats-"):
        raise ValueError("refusing to clean an unrecognized statistics artifact directory")
    retained = os.fstat(artifact_directory_fd)
    if (
        not stat.S_ISDIR(retained.st_mode)
        or retained.st_dev != artifact_directory_device
        or retained.st_ino != artifact_directory_inode
    ):
        raise ValueError("retained statistics artifact directory identity drift")
    unexpected_directories: list[str] = []
    for name in _retained_directory_entry_names(artifact_directory_fd):
        if type(name) is not str or name in {"", ".", ".."} or "/" in name:
            raise ValueError("statistics artifact directory entry name drift")
        entry = os.stat(name, dir_fd=artifact_directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry.st_mode):
            unexpected_directories.append(name)
            continue
        os.unlink(name, dir_fd=artifact_directory_fd)
    if unexpected_directories:
        raise ValueError(
            "statistics artifact directory contains unexpected subdirectories: "
            + repr(tuple(sorted(unexpected_directories)))
        )
    try:
        path_stat = os.lstat(artifact_directory)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(path_stat.st_mode):
        os.unlink(artifact_directory)
        raise ValueError("statistics artifact directory path was replaced by a symlink")
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or path_stat.st_dev != artifact_directory_device
        or path_stat.st_ino != artifact_directory_inode
    ):
        raise ValueError("statistics artifact directory path identity was replaced")
    os.rmdir(artifact_directory)


def _run_raw_statistics_workers(
    partitions: tuple[_Partition, ...],
    observed_masks: tuple[tuple[int, ...], ...],
    permutation_masks: tuple[tuple[tuple[int, ...], ...], ...],
    artifact_directory: str,
    artifact_directory_fd: int,
    maximum_artifact_bytes: int,
    maximum_workers: int,
    deadline_ns: int,
    *,
    preflight_plan: ResourcePlanV1,
    preflight_evidence_slot: list[_PhaseHostMemoryPreflightV1 | None],
) -> tuple[tuple[int, int, int], ...]:
    """Run five target jobs in one or five raw children with wait4 evidence."""

    if maximum_workers not in {1, len(TARGETS)}:
        raise ValueError("raw statistics worker count must be exactly one or five")
    if type(preflight_plan) is not ResourcePlanV1:
        raise TypeError("raw statistics require the exact final-preflight plan")
    if (
        type(preflight_evidence_slot) is not list
        or len(preflight_evidence_slot) != 1
        or preflight_evidence_slot[0] is not None
    ):
        raise ValueError("raw statistics final-preflight evidence slot drift")
    preflight_plan.__post_init__()
    if preflight_plan.maximum_parallel_workers != maximum_workers:
        raise ValueError("raw statistics final-preflight worker-count drift")
    original_sigchld_disposition = signal.getsignal(signal.SIGCHLD)
    if original_sigchld_disposition == signal.SIG_IGN:
        _reject(
            "FAIL_STATISTICAL_FORK_SAFETY",
            "raw statistics reject SIGCHLD ignore/automatic child reaping",
        )
    native_task_ids = _linux_process_task_ids()
    if (
        threading.active_count() != 1
        or len(native_task_ids) != 1
        or native_task_ids[0] != threading.get_native_id()
    ):
        _reject(
            "FAIL_STATISTICAL_FORK_SAFETY",
            "raw statistics require one exact Python/native Linux task",
        )
    groups = (
        (tuple(range(len(TARGETS))),)
        if maximum_workers == 1
        else tuple((index,) for index in range(len(TARGETS)))
    )
    child_address_space_limit = _invariance_child_address_space_limit_bytes()
    blocked_signals = {
        signal.SIGALRM,
        signal.SIGCHLD,
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTERM,
    }
    # Allocate every ownership slot and its reap marker before the first fork.
    # The first parent-side instruction after a successful fork only replaces
    # the pre-existing ``pid`` value, so even a later MemoryError/pidfd failure
    # leaves the child reachable by the unconditional cleanup path.
    active: list[dict[str, object]] = [
        {
            "group": group,
            "pid": -1,
            "pidfd": None,
            "marker": [False],
        }
        for group in groups
    ]
    target_execution: dict[int, tuple[int, int]] = {}
    original_signal_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        blocked_signals,
    )
    parent_signal_mask_restored = False
    sigchld_disposition_overridden = False
    sigchld_disposition_restored = False
    try:
        if signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN:
            _reject(
                "FAIL_STATISTICAL_FORK_SAFETY",
                "SIGCHLD changed to ignore before the blocked fork boundary",
            )
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        sigchld_disposition_overridden = True
        # Masks, target groups, ownership records, signal state, address
        # limit, artifact dirfd and every raw-fork input are now resident.
        # Only bounded evidence bookkeeping follows the phase evaluator;
        # its fixed 64 MiB parent term covers that small producer overhead.
        preflight_evidence_slot[0] = _preflight_host_memory(preflight_plan)
        for record in active:
            group = record["group"]
            if type(group) is not tuple:
                raise RuntimeError("raw statistics target group drift")
            pid = os.fork()
            if pid == 0:
                try:
                    os.setsid()
                    _close_inherited_fds_for_invariance_child(
                        keep_fds=(artifact_directory_fd,)
                    )
                    _set_invariance_child_address_space_limit(
                        child_address_space_limit
                    )
                    signal.pthread_sigmask(
                        signal.SIG_SETMASK,
                        original_signal_mask,
                    )
                    for target_index in group:
                        rows = _evaluate_target_statistics(
                            partitions,
                            observed_masks[target_index],
                            permutation_masks[target_index],
                        )
                        _write_statistics_worker_artifact(
                            target_index,
                            rows,
                            artifact_directory,
                            artifact_directory_fd,
                            maximum_artifact_bytes,
                        )
                except BaseException:
                    os._exit(70)
                os._exit(0)
            record["pid"] = pid
            record["pidfd"] = os.pidfd_open(pid, 0)

        for record in active:
            pid = record["pid"]
            pidfd = record["pidfd"]
            marker = record["marker"]
            if type(pid) is not int or type(pidfd) is not int or type(marker) is not list:
                raise RuntimeError("raw statistics worker ownership record drift")
            group = record["group"]
            if type(group) is not tuple:
                raise RuntimeError("raw statistics target group drift")
            group_diagnostic = "target_group=" + ",".join(
                TARGETS[target_index] for target_index in group
            )
            status, peak, _ = _wait_raw_child(
                pid,
                deadline_ns,
                pidfd,
                marker,
                phase="STATISTICS",
                diagnostic=group_diagnostic,
            )
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                raise RuntimeError(
                    "raw statistics child exited without all assigned artifacts"
                )
            for target_index in group:
                target_execution[target_index] = (pid, peak)
        signal.signal(signal.SIGCHLD, original_sigchld_disposition)
        sigchld_disposition_restored = True
        signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
        parent_signal_mask_restored = True
    finally:
        cleanup_errors: list[BaseException] = []
        for record in active:
            pid = record["pid"]
            pidfd = record["pidfd"]
            marker = record["marker"]
            if type(pid) is int and pid > 0 and type(marker) is list and not marker[0]:
                try:
                    _terminate_and_reap_raw_child(
                        pid,
                        pidfd if type(pidfd) is int else None,
                        marker,
                        child_label="raw statistics child",
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if type(pidfd) is int:
                try:
                    os.close(pidfd)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if sigchld_disposition_overridden and not sigchld_disposition_restored:
            try:
                signal.signal(signal.SIGCHLD, original_sigchld_disposition)
                sigchld_disposition_restored = True
            except BaseException as exc:
                cleanup_errors.append(exc)
        if not parent_signal_mask_restored:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(
                f"{type(item).__name__}: {item}" for item in cleanup_errors
            )
            active_exception = sys.exc_info()[1]
            if active_exception is not None:
                _append_exception_note(
                    active_exception,
                    "raw statistics cleanup also failed closed: " + detail
                )
            else:
                raise RuntimeError(
                    "raw statistics cleanup failed closed: " + detail
                )
    if tuple(sorted(target_execution)) != tuple(range(len(TARGETS))):
        raise RuntimeError("raw statistics target execution coverage drift")
    return tuple(
        (target, *target_execution[target]) for target in range(len(TARGETS))
    )


def _remaining_statistics_deadline_seconds(deadline_ns: int) -> float:
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        _reject(
            "FAIL_STATISTICAL_RUNTIME",
            "absolute formal covert statistics deadline expired",
        )
    return remaining_ns / 1_000_000_000


def _remaining_partition_preparation_deadline_seconds(deadline_ns: int) -> float:
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        _reject(
            "FAIL_PARTITION_PREPARATION_RUNTIME",
            "absolute formal covert partition-preparation deadline expired",
        )
    return remaining_ns / 1_000_000_000


def _statistics(
    authority: FormalAuditAuthorityV1,
    eligibility_signatures: tuple[
        tuple[int, int, int, int, int, int], ...
    ],
    manifest: FeatureManifestV1,
    partitions: tuple[_Partition, ...],
    permutation_seed: bytes,
    preextraction_feature_matrix_bytes_bound: int,
    feature_matrix_bytes_bound: int,
    envelope_and_decode_bytes_bound: int,
    partition_allocation_bytes_bound: int,
    partition_matrix_bytes_bound: int,
    resource_plan: ResourcePlanV1,
    statistics_started_ns: int,
    statistics_deadline_ns: int,
) -> tuple[
    tuple[TargetFeatureResultV1, ...],
    tuple[TargetFamilyMaxTResultV1, ...],
    str,
    ResourcePlanV1,
    _StatisticalRuntimeEvidenceInputsV1,
]:
    if type(permutation_seed) is not bytes or len(permutation_seed) != 32:
        _reject(
            "FAIL_PERMUTATION_RANDOMIZATION",
            "statistics require an exact auditor-owned 32-byte seed",
        )
    if (
        type(statistics_started_ns) is not int
        or type(statistics_deadline_ns) is not int
        or statistics_deadline_ns
        != statistics_started_ns
        + STATISTICS_WORKER_TIMEOUT_SECONDS * 1_000_000_000
    ):
        _reject("FAIL_STATISTICAL_RUNTIME", "statistics absolute deadline drift")
    _remaining_statistics_deadline_seconds(statistics_deadline_ns)
    if len(eligibility_signatures) != len(authority.rows):
        _reject(
            "FAIL_ELIGIBILITY_MASK_SEMANTICS",
            "statistics eligibility signature count drift",
        )
    target_labels = tuple(tuple(_target_label(row, target) for row in authority.rows) for target in TARGETS)
    target_strata = tuple(
        tuple(
            _target_stratum(row, target, signature)
            for row, signature in zip(
                authority.rows,
                eligibility_signatures,
                strict=True,
            )
        )
        for target in TARGETS
    )
    label_class_counts = tuple(len(set(labels)) for labels in target_labels)
    if type(resource_plan) is not ResourcePlanV1:
        _reject("FAIL_STATISTICAL_RESOURCE_PLAN", "statistics lack the exact resource plan")
    resource_plan.__post_init__()
    plan = resource_plan
    if (
        plan.rows != len(authority.rows)
        or plan.concrete_features != len(partitions)
        or plan.maximum_label_classes != max(label_class_counts, default=0)
        or plan.preextraction_feature_matrix_bytes_bound
        != preextraction_feature_matrix_bytes_bound
        or plan.feature_matrix_bytes_bound != feature_matrix_bytes_bound
        or plan.envelope_and_decode_bytes_bound != envelope_and_decode_bytes_bound
        or plan.partition_allocation_bytes_bound
        != partition_allocation_bytes_bound
        or plan.partition_matrix_bytes_bound != partition_matrix_bytes_bound
    ):
        _reject("FAIL_STATISTICAL_RESOURCE_PLAN", "statistics/resource inputs drifted")
    if not plan.admissible:
        _reject("FAIL_STATISTICAL_RESOURCE_PLAN", "bitset statistical resource plan exceeds a frozen cap")

    schedule_hasher = hashlib.sha256(
        _PERMUTATION_DOMAIN
        + bytes.fromhex(SEMANTICS_ID.rsplit("_", 1)[-1])
        + bytes.fromhex(_permutation_seed_commitment(permutation_seed)[7:])
    )
    all_observed_masks: list[tuple[int, ...]] = []
    all_permutation_masks: list[tuple[tuple[int, ...], ...]] = []
    for target, labels, strata in zip(TARGETS, target_labels, target_strata, strict=True):
        _, observed_masks = _label_masks(labels)
        all_observed_masks.append(observed_masks)
        # Materialize only the target label masks, not row x feature x
        # permutation values.  A 1024-row mask is one bounded Python integer.
        permutation_masks: list[tuple[int, ...]] = []
        for permutation_index in range(LABEL_PERMUTATIONS):
            _remaining_statistics_deadline_seconds(statistics_deadline_ns)
            permuted = _permuted_labels(
                labels,
                strata,
                target,
                permutation_index,
                permutation_seed,
                schedule_hasher,
            )
            _, masks = _label_masks(permuted)
            permutation_masks.append(masks)
        all_permutation_masks.append(tuple(permutation_masks))

    if (
        os.name != "posix"
        or not hasattr(os, "fork")
        or not hasattr(os, "pidfd_open")
        or not hasattr(os, "wait4")
        or not hasattr(os, "waitid")
    ):
        _reject(
            "FAIL_STATISTICAL_RUNTIME",
            "all frozen statistics require Linux fork/pidfd/waitid/wait4 semantics",
        )
    artifact_directory: str | None = None
    artifact_directory_fd: int | None = None
    artifact_directory_device: int | None = None
    artifact_directory_inode: int | None = None
    try:
        (
            artifact_directory,
            artifact_directory_fd,
            artifact_directory_device,
            artifact_directory_inode,
        ) = _acquire_private_artifact_directory(
            prefix="hegel-formal-covert-stats-"
        )
        per_worker_artifact_bound = (
            len(partitions) * WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE
            + WORKER_RESULT_ARTIFACT_FIXED_BYTES
        )
        preflight_evidence_slot: list[_PhaseHostMemoryPreflightV1 | None] = [None]
        execution_rows = _run_raw_statistics_workers(
            partitions,
            tuple(all_observed_masks),
            tuple(all_permutation_masks),
            artifact_directory,
            artifact_directory_fd,
            per_worker_artifact_bound,
            plan.maximum_parallel_workers,
            statistics_deadline_ns,
            preflight_plan=plan,
            preflight_evidence_slot=preflight_evidence_slot,
        )
        host_memory = preflight_evidence_slot[0]
        if type(host_memory) is not _PhaseHostMemoryPreflightV1:
            raise RuntimeError("statistics fork boundary lacks exact host preflight")
        expected_names = {
            f"target-{index}.json" for index in range(len(TARGETS))
        }
        if set(_retained_directory_entry_names(artifact_directory_fd)) != expected_names:
            raise ValueError("statistics artifact directory exact coverage drift")
        replay_rows = tuple(
            _read_statistics_worker_artifact(
                target_index,
                artifact_directory,
                artifact_directory_fd,
                artifact_directory_device,
                artifact_directory_inode,
                per_worker_artifact_bound,
                len(partitions),
                worker_pid,
                worker_peak,
            )
            for target_index, worker_pid, worker_peak in execution_rows
        )
        target_statistics = tuple(row[0] for row in replay_rows)
        artifacts = tuple(row[1] for row in replay_rows)
        peak_by_pid: dict[int, int] = {}
        for item in artifacts:
            peak_by_pid[item.worker_pid] = max(
                peak_by_pid.get(item.worker_pid, 0),
                item.worker_peak_rss_bytes,
            )
        if len(peak_by_pid) != plan.maximum_parallel_workers:
            raise RuntimeError("raw statistics worker cardinality drift")
        worker_peak_rss = tuple(
            peak_by_pid[pid] for pid in sorted(peak_by_pid)
        )
        worker_artifact_bytes = tuple(
            item.serialized_bytes for item in artifacts
        )
        worker_artifact_sha256 = tuple(item.sha256 for item in artifacts)
    except FormalCovertAuditRejectionV1 as exc:
        if exc.code in {
            "FAIL_STATISTICAL_RUNTIME",
            "FAIL_STATISTICAL_CONTAINMENT",
        }:
            raise
        _reject(
            "FAIL_STATISTICAL_RUNTIME",
            f"raw statistics worker failed closed: {exc}",
        )
    except BaseException as exc:
        _reject(
            "FAIL_STATISTICAL_RUNTIME",
            "worker artifact replay failed closed: "
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        active_exception = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        if (
            artifact_directory is not None
            and artifact_directory_fd is not None
            and artifact_directory_device is not None
            and artifact_directory_inode is not None
        ):
            try:
                _cleanup_statistics_artifact_directory(
                    artifact_directory,
                    artifact_directory_fd,
                    artifact_directory_device,
                    artifact_directory_inode,
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                os.close(artifact_directory_fd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(
                f"{type(item).__name__}: {item}" for item in cleanup_errors
            )
            if active_exception is not None:
                _append_exception_note(
                    active_exception,
                    "statistics artifact cleanup also failed closed: " + detail
                )
            else:
                _reject(
                    "FAIL_STATISTICAL_ARTIFACT_CLEANUP",
                    "statistics artifact cleanup failed closed: " + detail,
                )

    target_raw_p_values = tuple(
        _exact_probability(Fraction(statistics[2] + 1, EXCHANGEABLE_SAMPLE_SIZE))
        for statistics in target_statistics
    )
    target_adjusted = _holm_adjust_exact(target_raw_p_values)
    target_ranks = _holm_ranks_exact(target_raw_p_values)
    target_family_results = tuple(
        TargetFamilyMaxTResultV1(
            target=target,
            observed_target_max_omnibus=statistics[1],
            exceedance_count=statistics[2],
            raw_p=raw_p,
            holm_adjusted_p=adjusted_p,
            holm_rank=rank,
            holm_critical_value=_exact_probability(
                Fraction(
                    FAMILY_WISE_ALPHA_NUMERATOR,
                    FAMILY_WISE_ALPHA_DENOMINATOR
                    * (TARGET_FAMILY_HOLM_SIZE - rank + 1),
                )
            ),
            holm_rejected=(
                adjusted_p.as_fraction()
                < Fraction(
                    FAMILY_WISE_ALPHA_NUMERATOR,
                    FAMILY_WISE_ALPHA_DENOMINATOR,
                )
            ),
        )
        for target, statistics, raw_p, adjusted_p, rank in zip(
            TARGETS,
            target_statistics,
            target_raw_p_values,
            target_adjusted,
            target_ranks,
            strict=True,
        )
    )
    target_family_by_target = {
        item.target: item for item in target_family_results
    }

    observed_rows: list[tuple[str, str, float, float, float, int, int]] = []
    pairwise_raw_p_values: list[ExactProbabilityV1] = []
    for target, statistics in zip(TARGETS, target_statistics, strict=True):
        for feature_id, (
            raw_nmi,
            advantage,
            mean_nmi,
            pairwise_count,
            feature_max_t_count,
        ) in zip(
            manifest.concrete_feature_ids,
            statistics[0],
            strict=True,
        ):
            observed_rows.append(
                (
                    target,
                    feature_id,
                    raw_nmi,
                    advantage,
                    mean_nmi,
                    pairwise_count,
                    feature_max_t_count,
                )
            )
            pairwise_raw_p_values.append(
                _exact_probability(
                    Fraction(pairwise_count + 1, EXCHANGEABLE_SAMPLE_SIZE)
                )
            )

    pairwise_adjusted = _holm_adjust_exact(tuple(pairwise_raw_p_values))
    _remaining_statistics_deadline_seconds(statistics_deadline_ns)
    results = tuple(
        TargetFeatureResultV1(
            target=target,
            feature_id=feature_id,
            raw_nmi=raw_nmi,
            permutation_mean_nmi=mean_nmi,
            bias_corrected_nmi=max(0.0, raw_nmi - mean_nmi),
            balanced_accuracy_advantage=advantage,
            observed_omnibus=max(
                max(0.0, raw_nmi - mean_nmi) / NMI_LIMIT,
                max(0.0, advantage) / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
            ),
            pairwise_exceedance_count=pairwise_count,
            pairwise_raw_p=pairwise_raw_p,
            pairwise_holm_adjusted_p=pairwise_adjusted_p,
            feature_max_t_exceedance_count=feature_max_t_count,
            feature_max_t_p=_exact_probability(
                Fraction(feature_max_t_count + 1, EXCHANGEABLE_SAMPLE_SIZE)
            ),
            target_holm_critical_value=(
                target_family_by_target[target].holm_critical_value
            ),
            target_family_holm_rejected=(
                target_family_by_target[target].holm_rejected
            ),
            material_effect=(
                max(
                    max(0.0, raw_nmi - mean_nmi) / NMI_LIMIT,
                    max(0.0, advantage)
                    / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
                )
                > 1.0
            ),
            feature_localization_rejected=(
                target_family_by_target[target].holm_rejected
                and Fraction(feature_max_t_count + 1, EXCHANGEABLE_SAMPLE_SIZE)
                < target_family_by_target[target].holm_critical_value.as_fraction()
            ),
            failure_witness=(
                max(
                    max(0.0, raw_nmi - mean_nmi) / NMI_LIMIT,
                    max(0.0, advantage)
                    / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
                )
                > 1.0
                and target_family_by_target[target].holm_rejected
                and Fraction(feature_max_t_count + 1, EXCHANGEABLE_SAMPLE_SIZE)
                < target_family_by_target[target].holm_critical_value.as_fraction()
            ),
            passed=not (
                max(
                    max(0.0, raw_nmi - mean_nmi) / NMI_LIMIT,
                    max(0.0, advantage)
                    / BALANCED_ACCURACY_ADVANTAGE_LIMIT,
                )
                > 1.0
                and target_family_by_target[target].holm_rejected
                and Fraction(feature_max_t_count + 1, EXCHANGEABLE_SAMPLE_SIZE)
                < target_family_by_target[target].holm_critical_value.as_fraction()
            ),
        )
        for (
            (
                target,
                feature_id,
                raw_nmi,
                advantage,
                mean_nmi,
                pairwise_count,
                feature_max_t_count,
            ),
            pairwise_raw_p,
            pairwise_adjusted_p,
        ) in zip(
            observed_rows,
            pairwise_raw_p_values,
            pairwise_adjusted,
            strict=True,
        )
    )
    _remaining_statistics_deadline_seconds(statistics_deadline_ns)
    parent_result_object_graph_bytes_observed = (
        DEEP_SIZE_SAFETY_MULTIPLIER
        * _deep_object_graph_bytes(
            (
                target_statistics,
                target_raw_p_values,
                target_adjusted,
                target_ranks,
                target_family_results,
                observed_rows,
                pairwise_raw_p_values,
                pairwise_adjusted,
                results,
            )
        )
    )
    if (
        parent_result_object_graph_bytes_observed
        > plan.parent_result_working_bytes_bound
    ):
        _reject(
            "FAIL_PARENT_RESULT_GRAPH_BOUND",
            "live artifact replay/result DTO graph exceeds its conservative bound",
        )
    worker_peak_sum = sum(worker_peak_rss)
    runtime_inputs = _StatisticalRuntimeEvidenceInputsV1(
        resource_plan_root=plan.resource_plan_root,
        resource_preflight=host_memory.resource_preflight,
        system_available_memory_bytes_at_preflight=(
            host_memory.system_available_bytes
        ),
        cgroup_remaining_memory_bytes_at_preflight=(
            host_memory.cgroup_remaining_bytes
        ),
        host_available_memory_bytes_at_preflight=(
            host_memory.effective_available_bytes
        ),
        worker_peak_rss_bytes=worker_peak_rss,
        worker_result_artifact_bytes=worker_artifact_bytes,
        worker_result_artifact_sha256=worker_artifact_sha256,
        parent_result_object_graph_bytes_observed=(
            parent_result_object_graph_bytes_observed
        ),
        worker_timeout_seconds=STATISTICS_WORKER_TIMEOUT_SECONDS,
        parent_postprocess_deadline_schema=(
            PARENT_POSTPROCESS_DEADLINE_SCHEMA
        ),
        parent_postprocess_deadline_preemptive=False,
        external_supervisor_required=True,
    )
    return (
        results,
        target_family_results,
        "phase2b_formal_covert_permutation_schedule_" + schedule_hasher.hexdigest(),
        plan,
        runtime_inputs,
    )


def _input_root(envelopes: tuple[bytes, ...]) -> str:
    return stable_hash(
        {"envelopes": tuple(_bytes_commit(item) for item in envelopes)},
        prefix="phase2b_formal_covert_input_",
    )


def _validate_authority_links(
    envelopes: tuple[bytes, ...],
    authority: FormalAuditAuthorityV1,
    alignment: FormalPrivateBatchAlignmentV1,
) -> None:
    if type(alignment) is not FormalPrivateBatchAlignmentV1:
        _reject("FAIL_PRIVATE_ALIGNMENT", "private batch alignment must use the exact type")
    try:
        replay = FormalPrivateBatchAlignmentV1(
            alignment.links_in_public_order,
            alignment.wire_issued_alignment_id,
        )
    except (TypeError, ValueError) as exc:
        _reject("FAIL_PRIVATE_ALIGNMENT", f"private alignment replay failed: {exc}")
    if (
        replay.batch_binding_root != alignment.batch_binding_root
        or replay.alignment_root != alignment.alignment_root
    ):
        _reject("FAIL_PRIVATE_ALIGNMENT_ROOT", "private alignment cached root drift")
    observed = tuple("sha256:" + hashlib.sha256(item).hexdigest() for item in envelopes)
    if observed != alignment.public_envelope_sha256_in_order:
        _reject(
            "FAIL_PRIVATE_ALIGNMENT_BATCH_BINDING",
            "private alignment is stale, spliced, or bound to another envelope order",
        )
    if len(authority.rows) != len(alignment.source_row_index_by_public_row):
        _reject("FAIL_PRIVATE_ALIGNMENT_COVERAGE", "private alignment does not cover authority rows")
    for public_index, link in enumerate(alignment.links_in_public_order):
        source = authority.rows[link.source_row_index]
        if (
            source.source_case_commitment != link.source_case_commitment
            or source.label_binding_id != link.label_binding_id
        ):
            _reject(
                "FAIL_PRIVATE_AUTHORITY_SOURCE_BINDING",
                "private label row is stale, swapped, or spliced against its exact wire link "
                f"at public row {public_index}",
            )


def _align_private_authority(
    envelopes: tuple[bytes, ...],
    authority: FormalAuditAuthorityV1,
    alignment: FormalPrivateBatchAlignmentV1,
) -> FormalAuditAuthorityV1:
    _validate_authority_links(envelopes, authority, alignment)
    rows = []
    for public_index, link in enumerate(alignment.links_in_public_order):
        source = authority.rows[link.source_row_index]
        rows.append(
            FormalAuthorityLabelRowV1(
                row_index=public_index,
                family=source.family,
                binding=source.binding,
                scale=source.scale,
                answerability=source.answerability,
                joint_decision_class=source.joint_decision_class,
                case_type=source.case_type,
                source_case_commitment=source.source_case_commitment,
                label_binding_id=source.label_binding_id,
                row_binding=source.row_binding,
            )
        )
    return FormalAuditAuthorityV1(tuple(rows))


def _recognizer_output_graph_bytes_bound(rows: int) -> int:
    if type(rows) is not int or not 1 <= rows <= MAXIMUM_ROWS:
        raise ValueError("recognizer output bound needs a frozen row count")
    return (
        MAXIMUM_RECOGNIZER_RESIDUAL_BYTES_PER_BATCH
        + rows * RECOGNIZER_OUTPUT_METADATA_DEEP_BYTES_PER_ROW
    )


def _run_recognizer(
    recognizer: FormalRecognizerAdapterV1,
    envelopes: tuple[bytes, ...],
) -> tuple[FormalRecognizerDecisionV1, ...]:
    output = recognizer.run_batch(envelopes)
    if type(output) is not tuple or len(output) != len(envelopes):
        _reject("FAIL_RECOGNIZER_OUTPUT_SHAPE", "recognizer output is not an exact aligned tuple")
    if any(type(item) is not FormalRecognizerDecisionV1 for item in output):
        _reject("FAIL_RECOGNIZER_OUTPUT_TYPE", "recognizer output contains a non-exact decision")
    replayed: list[FormalRecognizerDecisionV1] = []
    try:
        for item in output:
            item.__post_init__()
            replayed.append(
                FormalRecognizerDecisionV1(
                    decision=item.decision,
                    family=item.family,
                    binding=tuple(
                        FormalBindingV1(
                            row.role_candidate_id,
                            row.entity_id,
                        )
                        for row in item.binding
                    ),
                    scale_set=tuple(item.scale_set),
                    candidate_residual_vector=bytes(
                        item.candidate_residual_vector
                    ),
                )
            )
    except (AttributeError, TypeError, ValueError) as exc:
        _reject(
            "FAIL_RECOGNIZER_OUTPUT_REPLAY",
            f"recognizer output failed exact fresh DTO replay: {exc}",
        )
    result = tuple(replayed)
    if sum(len(item.candidate_residual_vector) for item in result) > (
        MAXIMUM_RECOGNIZER_RESIDUAL_BYTES_PER_BATCH
    ):
        _reject("FAIL_RECOGNIZER_RESIDUAL_BUDGET", "recognizer output exceeds the residual budget")
    measured = DEEP_SIZE_SAFETY_MULTIPLIER * _deep_object_graph_bytes(result)
    if measured > _recognizer_output_graph_bytes_bound(len(envelopes)):
        _reject(
            "FAIL_RECOGNIZER_OUTPUT_GRAPH_BOUND",
            "recognizer output exceeds the frozen conservative object-graph bound",
        )
    return result


def _uuid_from_digest(digest: bytes) -> str:
    data = bytearray(digest[:16])
    data[6] = (data[6] & 0x0F) | 0x40
    data[8] = (data[8] & 0x3F) | 0x80
    raw = data.hex()
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def _renaming_maps(
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
    input_root: str,
    trial: int,
) -> dict[str, dict[str, str]]:
    old_by_namespace: dict[str, set[str]] = defaultdict(set)
    for item in decoded:
        for leaf in item.namespace_leaves:
            old_by_namespace[leaf.namespace].add(leaf.public_uuid)
    result: dict[str, dict[str, str]] = {}
    all_old = {value for values in old_by_namespace.values() for value in values}
    all_new: set[str] = set()
    for namespace in RENAMING_NAMESPACES:
        mapping: dict[str, str] = {}
        for old in sorted(old_by_namespace.get(namespace, ())):
            counter = 0
            while True:
                digest = hashlib.sha256(
                    _VARIANT_DOMAIN
                    + input_root.encode("ascii")
                    + trial.to_bytes(2, "big")
                    + namespace.encode("ascii")
                    + old.encode("ascii")
                    + counter.to_bytes(2, "big")
                ).digest()
                candidate = _uuid_from_digest(digest)
                if candidate not in all_old and candidate not in all_new:
                    break
                counter += 1
                if counter > 10:
                    _reject("FAIL_RENAMING_COLLISION", "global renaming collision retry budget exhausted")
            mapping[old] = candidate
            all_new.add(candidate)
        result[namespace] = mapping
    return result


def _inverse_decision(
    decision: FormalRecognizerDecisionV1,
    maps: dict[str, dict[str, str]],
) -> FormalRecognizerDecisionV1:
    inverses = {namespace: {new: old for old, new in mapping.items()} for namespace, mapping in maps.items()}
    try:
        binding = tuple(
            sorted(
                FormalBindingV1(
                    inverses["role_candidate_id"][item.role_candidate_id],
                    inverses["entity_id"][item.entity_id],
                )
                for item in decision.binding
            )
        )
        scales = tuple(sorted(inverses["scale_id"][item] for item in decision.scale_set))
    except KeyError as exc:
        _reject("FAIL_RENAMING_OUTPUT_DECODE", f"recognizer emitted an unmapped renamed ID: {exc}")
    return FormalRecognizerDecisionV1(
        decision=decision.decision,
        family=decision.family,
        binding=binding,
        scale_set=scales,
        candidate_residual_vector=decision.candidate_residual_vector,
    )


def _compare_decisions(
    baseline: tuple[FormalRecognizerDecisionV1, ...],
    candidate: tuple[FormalRecognizerDecisionV1, ...],
    kind: str,
    trial: int,
) -> None:
    if len(baseline) != len(candidate):
        _reject("FAIL_INVARIANT_ROW_COVERAGE", f"{kind} trial {trial} changed row count")
    for row_index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        fields = (
            ("decision", left.decision, right.decision),
            ("family", left.family, right.family),
            ("binding", left.binding, right.binding),
            ("scale_set", left.scale_set, right.scale_set),
            ("candidate_residual_vector", left.candidate_residual_vector, right.candidate_residual_vector),
        )
        for name, expected, observed in fields:
            if expected != observed:
                _reject("FAIL_RECOGNIZER_INVARIANCE", f"{kind} trial {trial} row {row_index} changed {name}")


def _invariance_input_snapshot_root(
    envelopes: tuple[bytes, ...],
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
    wire: FormalWireAdapterV1,
    recognizer: FormalRecognizerAdapterV1,
    input_root: str,
    adapter_pins: _AdapterExecutionPinsV1,
) -> str:
    adapter_pins.__post_init__()
    return stable_hash(
        {
            "input_root": input_root,
            "envelope_sha256": tuple(
                "sha256:" + hashlib.sha256(row).hexdigest() for row in envelopes
            ),
            "decoded": tuple(
                {
                    "payload_sha256": "sha256:"
                    + hashlib.sha256(row.payload_bytes).hexdigest(),
                    "padding_sha256": "sha256:"
                    + hashlib.sha256(row.padding_bytes).hexdigest(),
                    "namespace": tuple(
                        (leaf.namespace, leaf.path, leaf.public_uuid)
                        for leaf in row.namespace_leaves
                    ),
                    "observation_paths": row.observation_array_paths,
                    "channels": tuple(
                        (
                            leaf.group,
                            leaf.path,
                            leaf.canonical_value,
                            leaf.feature_value,
                        )
                        for leaf in row.channel_leaves
                    ),
                    "owners": tuple(
                        (leaf.path, leaf.owner_group)
                        for leaf in row.channel_only_leaf_owners
                    ),
                }
                for row in decoded
            ),
            "adapter_execution_pin_root": adapter_pins.pin_root,
            "wire": (
                wire.__class__.__module__,
                wire.__class__.__qualname__,
                adapter_pins.wire_implementation_id,
                adapter_pins.wire_schema_id,
                adapter_pins.private_alignment_root,
                adapter_pins.wire_issued_alignment_id,
                adapter_pins.supports_atomic_variant_suite,
                adapter_pins.stop_binding_id,
                adapter_pins.fresh_build_verification_root,
            ),
            "recognizer": (
                recognizer.__class__.__module__,
                recognizer.__class__.__qualname__,
                adapter_pins.recognizer_implementation_id,
            ),
        },
        prefix="phase2b_formal_covert_invariance_input_snapshot_",
    )


def _run_invariance_core(
    envelopes: tuple[bytes, ...],
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
    wire: FormalWireAdapterV1,
    recognizer: FormalRecognizerAdapterV1,
    input_root: str,
    fork_input_snapshot_root: str,
    adapter_pins: _AdapterExecutionPinsV1,
) -> _InvariantExecutionResultV1:
    _verify_adapter_execution_pins(wire, recognizer, adapter_pins)
    if _invariance_input_snapshot_root(
        envelopes, decoded, wire, recognizer, input_root, adapter_pins
    ) != fork_input_snapshot_root:
        _reject(
            "FAIL_INVARIANCE_INPUT_SNAPSHOT",
            "forked invariant inputs/adapters differ from the parent snapshot",
        )
    baseline = _run_recognizer(recognizer, envelopes)
    execution = hashlib.sha256(_VARIANT_DOMAIN)
    recognizer_outputs = hashlib.sha256(_VARIANT_DOMAIN + b"recognizer-output")
    baseline_root = stable_hash(tuple(item.commitment() for item in baseline))
    execution.update(baseline_root.encode("ascii"))
    recognizer_outputs.update(baseline_root.encode("ascii"))
    map_roots: set[str] = set()
    identity_case_order = tuple(range(len(envelopes)))
    renaming_trials: list[
        tuple[dict[str, dict[str, str]], str, FormalVariantPlanV1]
    ] = []
    for trial in range(GLOBAL_RENAMINGS):
        maps = _renaming_maps(decoded, input_root, trial)
        map_root = stable_hash(maps, prefix="phase2b_formal_covert_renaming_")
        if map_root in map_roots:
            _reject("FAIL_RENAMING_INDEPENDENCE", "two global renaming trials are identical")
        map_roots.add(map_root)
        renaming_rows = tuple(
            sorted(
                FormalUuidRenamingPlanV1(namespace, source, target)
                for namespace, mapping in maps.items()
                for source, target in mapping.items()
            )
        )
        plan = FormalVariantPlanV1(identity_case_order, renaming_rows, ())
        renaming_trials.append((maps, map_root, plan))

    case_trials: list[tuple[tuple[int, ...], str, FormalVariantPlanV1]] = []
    case_order_roots: set[str] = set()
    for trial in range(GLOBAL_CASE_ORDER_PERMUTATIONS):
        indices: tuple[int, ...] | None = None
        order_root: str | None = None
        for retry in range(1_024):
            order: list[object] = list(range(len(envelopes)))
            _shuffle(
                order,
                hashlib.sha256(
                    _VARIANT_DOMAIN
                    + input_root.encode("ascii")
                    + b"case"
                    + trial.to_bytes(2, "big")
                    + retry.to_bytes(2, "big")
                ).digest(),
            )
            candidate = tuple(int(item) for item in order)
            candidate_root = stable_hash(
                candidate,
                prefix="phase2b_formal_covert_case_order_",
            )
            if candidate != identity_case_order and candidate_root not in case_order_roots:
                indices, order_root = candidate, candidate_root
                break
        if indices is None or order_root is None:
            _reject(
                "FAIL_CASE_ORDER_INDEPENDENCE",
                "could not construct 32 distinct non-identity case-order schedules",
            )
        case_order_roots.add(order_root)
        case_trials.append(
            (indices, order_root, FormalVariantPlanV1(indices, (), ()))
        )
    if len(case_order_roots) != GLOBAL_CASE_ORDER_PERMUTATIONS:
        _reject("FAIL_CASE_ORDER_INDEPENDENCE", "case-order schedule coverage drift")

    reorderable = any(
        len(_resolve(item.payload, path)) >= 2
        for item in decoded
        for path in item.observation_array_paths
    )
    if not reorderable:
        _reject("FAIL_OBSERVATION_ORDER_COVERAGE", "no observation array has at least two records")
    observation_roots: set[str] = set()
    observation_trials: list[tuple[str, FormalVariantPlanV1]] = []
    for trial in range(OBSERVATION_ORDER_PERMUTATIONS):
        order_rows: tuple[FormalObservationOrderPlanV1, ...] | None = None
        order_root: str | None = None
        for retry in range(1_024):
            candidate_rows: list[FormalObservationOrderPlanV1] = []
            changed = False
            for row_index, item in enumerate(decoded):
                for path in item.observation_array_paths:
                    array = _resolve(item.payload, path)
                    order = list(range(len(array)))
                    _shuffle(
                        order,
                        hashlib.sha256(
                            _VARIANT_DOMAIN
                            + input_root.encode("ascii")
                            + b"observation"
                            + trial.to_bytes(2, "big")
                            + retry.to_bytes(2, "big")
                            + row_index.to_bytes(4, "big")
                            + _path_text(path).encode("utf-8")
                        ).digest(),
                    )
                    indices = tuple(int(value) for value in order)
                    changed = changed or indices != tuple(range(len(array)))
                    candidate_rows.append(
                        FormalObservationOrderPlanV1(row_index, path, indices)
                    )
            candidate_tuple = tuple(
                sorted(
                    candidate_rows,
                    key=lambda row: (row.canonical_row_index, _path_text(row.path)),
                )
            )
            candidate_root = stable_hash(
                tuple(
                    (row.canonical_row_index, row.path, row.order_indices)
                    for row in candidate_tuple
                ),
                prefix="phase2b_formal_covert_observation_order_",
            )
            if changed and candidate_root not in observation_roots:
                order_rows, order_root = candidate_tuple, candidate_root
                break
        if order_rows is None or order_root is None:
            _reject(
                "FAIL_OBSERVATION_ORDER_INDEPENDENCE",
                "could not construct 16 distinct non-identity observation schedules",
            )
        observation_roots.add(order_root)
        observation_trials.append(
            (
                order_root,
                FormalVariantPlanV1(identity_case_order, (), order_rows),
            )
        )

    plans = tuple(
        [row[2] for row in renaming_trials]
        + [row[2] for row in case_trials]
        + [row[1] for row in observation_trials]
    )
    expected_plan_count = (
        GLOBAL_RENAMINGS
        + GLOBAL_CASE_ORDER_PERMUTATIONS
        + OBSERVATION_ORDER_PERMUTATIONS
    )
    if len(plans) != expected_plan_count:
        _reject("FAIL_VARIANT_SUITE_COVERAGE", "invariance plan count drift")
    supports_suite = adapter_pins.supports_atomic_variant_suite
    if type(supports_suite) is not bool:
        _reject("FAIL_VARIANT_SUITE_CAPABILITY", "wire atomic-suite support flag drift")
    suite_batches: list[tuple[bytes, ...] | None] | None = None
    consumer_decode_count = 0
    suite_unique_envelope_object_count = 0
    suite_resident_object_graph_bytes_observed = 0
    if supports_suite:
        suite = wire.make_variant_suite_envelopes(
            envelopes=envelopes,
            decoded=decoded,
            plans=plans,
        )
        if type(suite) is not _FormalVariantSuiteExecutionV1:
            _reject("FAIL_VARIANT_SUITE_TYPE", "wire returned a non-exact suite DTO")
        suite.__post_init__()
        if any(len(batch) != len(envelopes) for batch in suite.batches):
            _reject("FAIL_VARIANT_SUITE_COVERAGE", "suite batch row-count drift")
        variant_suite_root = suite.suite_root
        consumer_decode_count = suite.consumer_decode_count
        suite_unique_envelope_object_count = suite.unique_envelope_object_count
        suite_resident_object_graph_bytes_observed = (
            suite.resident_object_graph_bytes_observed
        )
        suite_batches = list(suite.batches)
        del suite
    else:
        variant_suite_root = stable_hash(
            {
                "adapter": wire.implementation_id,
                "plans": plans,
            },
            prefix="phase2b_formal_variant_suite_development_",
        )
    execution.update(variant_suite_root.encode("ascii"))

    def variant_envelopes_for(
        plan_index: int,
        plan: FormalVariantPlanV1,
    ) -> tuple[bytes, ...]:
        nonlocal consumer_decode_count
        if suite_batches is None:
            batch = wire.make_variant_envelopes(
                envelopes=envelopes,
                decoded=decoded,
                plan=plan,
            )
            consumer_decode_count += len(batch)
            return batch
        batch = suite_batches[plan_index]
        if batch is None:
            _reject("FAIL_VARIANT_SUITE_REUSE", "variant suite batch was reused")
        suite_batches[plan_index] = None
        return batch

    cursor = 0
    for trial, (maps, map_root, plan) in enumerate(renaming_trials):
        variant_envelopes = variant_envelopes_for(cursor, plan)
        cursor += 1
        output = _run_recognizer(recognizer, variant_envelopes)
        normalized = tuple(_inverse_decision(item, maps) for item in output)
        _compare_decisions(baseline, normalized, "global_renaming", trial)
        execution.update(map_root.encode("ascii"))
        output_root = stable_hash(tuple(item.commitment() for item in normalized))
        execution.update(output_root.encode("ascii"))
        recognizer_outputs.update(output_root.encode("ascii"))

    for trial, (indices, order_root, plan) in enumerate(case_trials):
        variant_envelopes = variant_envelopes_for(cursor, plan)
        cursor += 1
        output = _run_recognizer(recognizer, variant_envelopes)
        restored: list[FormalRecognizerDecisionV1 | None] = [None] * len(output)
        for position, canonical_index in enumerate(indices):
            restored[canonical_index] = output[position]
        if any(item is None for item in restored):
            _reject("FAIL_CASE_ORDER_COVERAGE", "case-order inverse lost a row")
        canonical = tuple(item for item in restored if item is not None)
        _compare_decisions(baseline, canonical, "case_order", trial)
        execution.update(order_root.encode("ascii"))
        output_root = stable_hash(tuple(item.commitment() for item in canonical))
        execution.update(output_root.encode("ascii"))
        recognizer_outputs.update(output_root.encode("ascii"))
    for trial, (order_root, plan) in enumerate(observation_trials):
        variant_envelopes = variant_envelopes_for(cursor, plan)
        cursor += 1
        output = _run_recognizer(recognizer, variant_envelopes)
        _compare_decisions(baseline, output, "observation_order", trial)
        execution.update(order_root.encode("ascii"))
        output_root = stable_hash(tuple(item.commitment() for item in output))
        execution.update(output_root.encode("ascii"))
        recognizer_outputs.update(output_root.encode("ascii"))

    if cursor != expected_plan_count or (
        suite_batches is not None and any(batch is not None for batch in suite_batches)
    ):
        _reject("FAIL_VARIANT_SUITE_COVERAGE", "not every frozen variant batch was consumed")
    _verify_adapter_execution_pins(wire, recognizer, adapter_pins)

    comparisons = len(envelopes) * (GLOBAL_RENAMINGS + GLOBAL_CASE_ORDER_PERMUTATIONS + OBSERVATION_ORDER_PERMUTATIONS) * 5
    result = _InvariantExecutionResultV1(
        fork_input_snapshot_root=fork_input_snapshot_root,
        adapter_execution_pin_root=adapter_pins.pin_root,
        invariant_execution_root=(
            "phase2b_formal_covert_invariant_execution_" + execution.hexdigest()
        ),
        exact_invariant_comparisons=comparisons,
        invariant_run_count=(
            1
            + GLOBAL_RENAMINGS
            + GLOBAL_CASE_ORDER_PERMUTATIONS
            + OBSERVATION_ORDER_PERMUTATIONS
        ),
        variant_suite_root=variant_suite_root,
        consumer_decode_count=consumer_decode_count,
        variant_suite_unique_envelope_object_count=(
            suite_unique_envelope_object_count
        ),
        variant_suite_resident_object_graph_bytes_observed=(
            suite_resident_object_graph_bytes_observed
        ),
        recognizer_output_root=(
            "phase2b_formal_covert_recognizer_output_"
            + recognizer_outputs.hexdigest()
        ),
    )
    result.__post_init__()
    return result


def _raw_child_rusage_peak_bytes(
    usage: resource.struct_rusage,
    *,
    child_label: str = "raw child",
) -> int:
    if type(child_label) is not str or child_label not in {
        "raw child",
        "raw invariant child",
        "raw statistics child",
    }:
        raise ValueError("raw child label drift")
    if not sys.platform.startswith("linux"):
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            f"{child_label} RSS evidence requires Linux wait4 semantics",
        )
    peak = int(usage.ru_maxrss) * 1_024
    if peak <= 0:
        raise ValueError(f"{child_label} wait4 RSS observation is not positive")
    return peak


def _linux_process_group_id(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    close = value.rfind(")")
    if close < 0:
        raise ValueError("Linux process stat has no closing command delimiter")
    fields = value[close + 2 :].split()
    if len(fields) < 3:
        raise ValueError("Linux process stat lacks a process-group field")
    return int(fields[2])


def _linux_process_group_members(process_group_id: int) -> tuple[int, ...]:
    if type(process_group_id) is not int or process_group_id <= 0:
        raise ValueError("process-group ID must be a positive exact integer")
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isascii() or not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if _linux_process_group_id(pid) == process_group_id:
                members.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return tuple(sorted(members))


def _signal_raw_child_scope(
    pid: int,
    signal_number: int,
    pidfd: int | None,
) -> None:
    """Signal a verified child session/process-group, falling back to its PID."""

    try:
        process_group_id = _linux_process_group_id(pid)
    except (FileNotFoundError, ProcessLookupError):
        process_group_id = None
    if process_group_id == pid:
        try:
            os.killpg(pid, signal_number)
            return
        except ProcessLookupError:
            pass
    if pidfd is not None:
        try:
            signal.pidfd_send_signal(pidfd, signal_number)
            return
        except ProcessLookupError:
            # A pidfd is immune to PID reuse.  ESRCH means that exact process
            # is gone; never fall back to a now-reusable numeric PID.
            return
    try:
        os.kill(pid, signal_number)
    except ProcessLookupError:
        pass


def _terminate_and_reap_raw_child(
    pid: int,
    pidfd: int | None = None,
    reap_marker: list[bool] | None = None,
    *,
    child_label: str = "raw child",
) -> tuple[int, int]:
    """Kill the owned session, clean descendants, then exactly once wait4."""

    if type(child_label) is not str or child_label not in {
        "raw child",
        "raw invariant child",
        "raw statistics child",
    }:
        raise ValueError("raw child label drift")
    _signal_raw_child_scope(pid, signal.SIGTERM, pidfd)
    grace_deadline = time.monotonic() + 5.0
    exited = False
    while time.monotonic() < grace_deadline:
        try:
            info = os.waitid(
                os.P_PID,
                pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError as exc:
            raise RuntimeError(f"{child_label} was already reaped") from exc
        if info is not None and info.si_pid == pid:
            exited = True
            break
        time.sleep(0.05)
    if not exited:
        _signal_raw_child_scope(pid, signal.SIGKILL, pidfd)
        try:
            info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        except ChildProcessError as exc:
            raise RuntimeError(f"{child_label} disappeared before reap") from exc
        if info is None or info.si_pid != pid:
            raise RuntimeError(
                f"{child_label} SIGKILL did not produce an owned zombie"
            )

    # Keep the direct child as an unreaped zombie while acting on its numeric
    # process-group ID.  This prevents PID/PGID reuse from redirecting cleanup.
    containment_error: str | None = None
    process_group_id = _linux_process_group_id(pid)
    if process_group_id != pid:
        containment_error = f"{child_label} did not own its private session"
    else:
        descendants = tuple(
            member for member in _linux_process_group_members(pid) if member != pid
        )
        if descendants:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            disappearance_deadline = time.monotonic() + 5.0
            while time.monotonic() < disappearance_deadline:
                remaining = tuple(
                    member
                    for member in _linux_process_group_members(pid)
                    if member != pid
                )
                if not remaining:
                    break
                time.sleep(0.05)
            if tuple(
                member
                for member in _linux_process_group_members(pid)
                if member != pid
            ):
                containment_error = (
                    f"{child_label} descendant process group survived SIGKILL"
                )
    try:
        waited, status, usage = os.wait4(pid, 0)
    except ChildProcessError as exc:
        raise RuntimeError(
            f"{child_label} disappeared before final wait4"
        ) from exc
    if waited != pid:
        raise RuntimeError(f"blocking wait4 did not reap the {child_label}")
    if reap_marker is not None:
        reap_marker[0] = True
    if containment_error is not None:
        raise RuntimeError(containment_error)
    return status, _raw_child_rusage_peak_bytes(
        usage,
        child_label=child_label,
    )


def _wait_raw_child(
    pid: int,
    absolute_deadline_ns: int,
    pidfd: int | None = None,
    reap_marker: list[bool] | None = None,
    *,
    phase: str,
    diagnostic: str = "",
) -> tuple[int, int, int]:
    if type(absolute_deadline_ns) is not int or absolute_deadline_ns <= 0:
        raise ValueError("raw child absolute deadline must be a positive integer")
    if phase == "INVARIANCE":
        runtime_failure_code = "FAIL_INVARIANCE_RUNTIME"
        containment_failure_code = "FAIL_INVARIANCE_CONTAINMENT"
        child_label = "raw invariant child"
    elif phase == "STATISTICS":
        runtime_failure_code = "FAIL_STATISTICAL_RUNTIME"
        containment_failure_code = "FAIL_STATISTICAL_CONTAINMENT"
        child_label = "raw statistics child"
    else:
        raise ValueError("raw child phase must be INVARIANCE or STATISTICS")
    if (
        type(diagnostic) is not str
        or len(diagnostic.encode("utf-8")) > 256
        or any(character in diagnostic for character in "\r\n\x00")
    ):
        raise ValueError("raw child diagnostic must be a bounded single line")
    diagnostic_suffix = f" ({diagnostic})" if diagnostic else ""
    reaped = False
    parent_peak_current_rss = _linux_current_rss_bytes()
    try:
        while True:
            parent_peak_current_rss = max(
                parent_peak_current_rss,
                _linux_current_rss_bytes(),
            )
            remaining_ns = absolute_deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                try:
                    _terminate_and_reap_raw_child(
                        pid,
                        pidfd,
                        reap_marker,
                        child_label=child_label,
                    )
                except BaseException as exc:
                    reaped = bool(reap_marker is not None and reap_marker[0])
                    _reject(
                        runtime_failure_code,
                        f"{child_label} deadline cleanup failed closed: "
                        f"{type(exc).__name__}: {exc}{diagnostic_suffix}",
                    )
                reaped = True
                _reject(
                    runtime_failure_code,
                    f"{child_label} exceeded its absolute deadline"
                    + diagnostic_suffix,
                )
            info = os.waitid(
                os.P_PID,
                pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
            if info is not None and info.si_pid == pid:
                process_group_id = _linux_process_group_id(pid)
                descendants = (
                    tuple(
                        member
                        for member in _linux_process_group_members(pid)
                        if member != pid
                    )
                    if process_group_id == pid
                    else ()
                )
                if process_group_id != pid or descendants:
                    try:
                        _terminate_and_reap_raw_child(
                            pid,
                            pidfd,
                            reap_marker,
                            child_label=child_label,
                        )
                    except BaseException as exc:
                        reaped = bool(reap_marker is not None and reap_marker[0])
                        _reject(
                            containment_failure_code,
                            f"{child_label} containment cleanup failed closed: "
                            f"{type(exc).__name__}: {exc}{diagnostic_suffix}",
                        )
                    reaped = True
                    reason = (
                        f"{child_label} did not retain its private session"
                        if process_group_id != pid
                        else f"{child_label} left descendant processes"
                    )
                    _reject(containment_failure_code, reason + diagnostic_suffix)
                waited, status, usage = os.wait4(pid, 0)
                reaped = True
                if reap_marker is not None:
                    reap_marker[0] = True
                if waited != pid:
                    raise RuntimeError(
                        f"{child_label} success wait4 reaped the wrong PID"
                    )
                return (
                    status,
                    _raw_child_rusage_peak_bytes(
                        usage,
                        child_label=child_label,
                    ),
                    parent_peak_current_rss,
                )
            time.sleep(min(0.05, remaining_ns / 1_000_000_000))
    except BaseException:
        if not reaped and not (reap_marker is not None and reap_marker[0]):
            _terminate_and_reap_raw_child(
                pid,
                pidfd,
                reap_marker,
                child_label=child_label,
            )
        raise


def _close_inherited_fds_for_invariance_child(
    *,
    keep_fds: tuple[int, ...] = (),
) -> None:
    """Close every inherited descriptor except an exact internal allowlist."""

    if type(keep_fds) is not tuple or any(
        type(value) is not int or value <= 2 for value in keep_fds
    ) or len(set(keep_fds)) != len(keep_fds):
        raise ValueError("child FD allowlist must be exact, unique, and nonstdio")
    keep = set(keep_fds)

    descriptors: list[int] = []
    for name in os.listdir("/proc/self/fd"):
        if name.isascii() and name.isdigit():
            descriptor = int(name)
            if descriptor > 2 and descriptor not in keep:
                descriptors.append(descriptor)
    for descriptor in sorted(set(descriptors), reverse=True):
        try:
            os.close(descriptor)
        except OSError as exc:
            if exc.errno != 9:  # EBADF: /proc listing's own transient descriptor.
                raise


def _invariance_child_address_space_limit_bytes() -> int:
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = MAXIMUM_ESTIMATED_RSS_BYTES
    if hard != resource.RLIM_INFINITY:
        target = min(target, int(hard))
    if target <= _linux_current_virtual_memory_bytes() + 64 * 1_048_576:
        _reject(
            "FAIL_INVARIANCE_ADDRESS_SPACE_PREFLIGHT",
            "existing virtual address space leaves no frozen child allocation margin",
        )
    return target


def _set_invariance_child_address_space_limit(limit_bytes: int) -> None:
    if type(limit_bytes) is not int or not 0 < limit_bytes <= (
        MAXIMUM_ESTIMATED_RSS_BYTES
    ):
        raise ValueError("invariant child address-space limit drift")
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and limit_bytes > hard:
        raise ValueError("invariant child address-space limit exceeds hard RLIMIT_AS")
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    observed_soft, observed_hard = resource.getrlimit(resource.RLIMIT_AS)
    if observed_soft != limit_bytes or observed_hard != limit_bytes:
        raise RuntimeError("invariant child RLIMIT_AS did not replay exactly")


def _linux_process_task_ids() -> tuple[int, ...]:
    try:
        tasks = tuple(
            sorted(
                int(entry.name)
                for entry in Path("/proc/self/task").iterdir()
                if entry.name.isascii() and entry.name.isdigit()
            )
        )
    except (OSError, ValueError) as exc:
        _reject(
            "FAIL_INVARIANCE_FORK_SAFETY",
            f"Linux native task inventory failed: {exc}",
        )
    if not tasks:
        _reject(
            "FAIL_INVARIANCE_FORK_SAFETY",
            "Linux native task inventory was empty",
        )
    return tasks


def _run_invariance(
    envelopes: tuple[bytes, ...],
    decoded: tuple[DecodedFormalEnvelopeV1, ...],
    wire: FormalWireAdapterV1,
    recognizer: FormalRecognizerAdapterV1,
    input_root: str,
    plan: _InvariantResourcePlanV1,
    adapter_pins: _AdapterExecutionPinsV1,
) -> tuple[_InvariantExecutionResultV1, InvariantRuntimeEvidenceV1]:
    """Isolate the 81 suite/recognizer callbacks; parent adapter surfaces are TCB."""

    if (
        os.name != "posix"
        or not hasattr(os, "fork")
        or not hasattr(os, "pidfd_open")
        or not hasattr(os, "wait4")
        or not hasattr(os, "waitid")
    ):
        _reject(
            "FAIL_INVARIANCE_FORK_SAFETY",
            "invariance requires Linux fork/pidfd/waitid/wait4 semantics",
        )
    original_sigchld_disposition = signal.getsignal(signal.SIGCHLD)
    if original_sigchld_disposition == signal.SIG_IGN:
        _reject(
            "FAIL_INVARIANCE_FORK_SAFETY",
            "raw invariant execution rejects SIGCHLD ignore/automatic child reaping",
        )
    native_tasks = _linux_process_task_ids()
    if (
        threading.active_count() != 1
        or len(native_tasks) != 1
        or native_tasks[0] != threading.get_native_id()
    ):
        _reject(
            "FAIL_INVARIANCE_FORK_SAFETY",
            "raw invariant execution requires one exact Python/native Linux task",
        )
    child_address_space_limit = _invariance_child_address_space_limit_bytes()
    fork_input_snapshot_root = _invariance_input_snapshot_root(
        envelopes,
        decoded,
        wire,
        recognizer,
        input_root,
        adapter_pins,
    )
    started_ns = time.monotonic_ns()
    artifact_name = "invariant-result.json"
    child_exit = os._exit
    invariant_core = _run_invariance_core
    artifact_writer = _write_invariant_child_artifact
    artifact: _InvariantWorkerArtifactV1 | None = None
    result: _InvariantExecutionResultV1 | None = None
    cleanup_errors: list[BaseException] = []
    child_pid: int | None = None
    child_pidfd: int | None = None
    child_reaped = [False]
    artifact_directory: str | None = None
    artifact_directory_fd: int | None = None
    artifact_directory_device: int | None = None
    artifact_directory_inode: int | None = None
    blocked_signals = {
        signal.SIGALRM,
        signal.SIGCHLD,
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTERM,
    }
    original_signal_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        blocked_signals,
    )
    parent_signal_mask_restored = False
    sigchld_disposition_overridden = False
    sigchld_disposition_restored = False
    try:
        if signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN:
            _reject(
                "FAIL_INVARIANCE_FORK_SAFETY",
                "SIGCHLD changed to ignore before the blocked fork boundary",
            )
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        sigchld_disposition_overridden = True
        (
            artifact_directory,
            artifact_directory_fd,
            artifact_directory_device,
            artifact_directory_inode,
        ) = _acquire_private_artifact_directory(
            prefix="hegel-formal-covert-invariant-"
        )
        artifact_path = str(Path(artifact_directory) / artifact_name)
        # The snapshot root, signal state, child limit, artifact path, and all
        # other fork inputs already exist.  Only bounded evidence bookkeeping
        # follows this phase-exact sample; the fixed 64 MiB term covers it.
        host_memory = _preflight_invariance_host_memory(plan)
        parent_current_rss_at_start = (
            host_memory.resource_preflight.parent_current_rss_bytes_at_preflight
        )
        child_pid = os.fork()
        if child_pid == 0:
            try:
                os.setsid()
                _close_inherited_fds_for_invariance_child()
                _set_invariance_child_address_space_limit(
                    child_address_space_limit
                )
                signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
                child_result = invariant_core(
                    envelopes,
                    decoded,
                    wire,
                    recognizer,
                    input_root,
                    fork_input_snapshot_root,
                    adapter_pins,
                )
                artifact_writer(
                    child_result,
                    artifact_path,
                    INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND,
                )
            except BaseException:
                child_exit(70)
            child_exit(0)
        child_pidfd = os.pidfd_open(child_pid, 0)
        invariance_deadline_ns = (
            time.monotonic_ns()
            + INVARIANCE_WORKER_TIMEOUT_SECONDS * 1_000_000_000
        )
        # Keep the selected lifecycle/cancellation handlers blocked while the
        # direct child is owned.  Other parent signal handlers are cooperative
        # TCB; external qualification requires fresh default dispositions.
        (
            status,
            worker_peak_rss_bytes,
            parent_peak_current_rss_observed,
        ) = _wait_raw_child(
            child_pid,
            invariance_deadline_ns,
            child_pidfd,
            child_reaped,
            phase="INVARIANCE",
        )
        signal.signal(signal.SIGCHLD, original_sigchld_disposition)
        sigchld_disposition_restored = True
        signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
        parent_signal_mask_restored = True
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            _reject(
                "FAIL_INVARIANCE_RUNTIME",
                "raw invariant child exited without a successful closed artifact",
            )
        result, artifact = _read_invariant_worker_artifact(
            artifact_name,
            artifact_directory,
            artifact_directory_fd,
            artifact_directory_device,
            artifact_directory_inode,
            INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND,
            child_pid,
            worker_peak_rss_bytes,
        )
        if (
            result.fork_input_snapshot_root != fork_input_snapshot_root
            or _invariance_input_snapshot_root(
                envelopes,
                decoded,
                wire,
                recognizer,
                input_root,
                adapter_pins,
            )
            != fork_input_snapshot_root
            or result.adapter_execution_pin_root != adapter_pins.pin_root
        ):
            raise ValueError("invariant parent/worker input snapshot root drift")
        _verify_adapter_execution_pins(wire, recognizer, adapter_pins)
    except FormalCovertAuditRejectionV1:
        raise
    except BaseException as exc:
        _reject(
            "FAIL_INVARIANCE_RUNTIME",
            "invariant artifact replay failed closed: "
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if child_pid is not None and child_pid > 0 and not child_reaped[0]:
            try:
                _terminate_and_reap_raw_child(
                    child_pid,
                    child_pidfd,
                    child_reaped,
                    child_label="raw invariant child",
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        if sigchld_disposition_overridden and not sigchld_disposition_restored:
            try:
                signal.signal(signal.SIGCHLD, original_sigchld_disposition)
                sigchld_disposition_restored = True
            except BaseException as exc:
                cleanup_errors.append(exc)
        if not parent_signal_mask_restored:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
                parent_signal_mask_restored = True
            except BaseException as exc:
                cleanup_errors.append(exc)
        if child_pidfd is not None:
            try:
                os.close(child_pidfd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if (
            artifact_directory is not None
            and artifact_directory_fd is not None
            and artifact_directory_device is not None
            and artifact_directory_inode is not None
        ):
            try:
                _cleanup_invariant_artifact_directory(
                    artifact_directory,
                    artifact_directory_fd,
                    artifact_directory_device,
                    artifact_directory_inode,
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                os.close(artifact_directory_fd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            active = sys.exc_info()[1]
            detail = "; ".join(
                f"{type(item).__name__}: {item}" for item in cleanup_errors
            )
            if active is not None:
                _append_exception_note(
                    active,
                    "invariant child/artifact cleanup also failed closed: " + detail
                )
            else:
                _reject(
                    "FAIL_INVARIANCE_ARTIFACT_CLEANUP",
                    "invariant child/artifact cleanup failed closed: " + detail,
                )
    if artifact is None or result is None:
        _reject(
            "FAIL_INVARIANCE_RUNTIME",
            "invariant worker completed without an exact artifact/result",
        )
    parent_current_rss_at_completion = _linux_current_rss_bytes()
    parent_historical_peak_rss = _linux_peak_rss_bytes()
    diagnostic_historical_sum = (
        parent_historical_peak_rss + artifact.worker_peak_rss_bytes
    )
    physical_snapshot = _capture_unified_cgroup_memory_snapshot_v1()
    _require_unified_cgroup_physical_peak_v1(
        failure_code="FAIL_INVARIANCE_RUNTIME_RSS",
        stage="INVARIANCE_COMPLETION",
        snapshot=physical_snapshot,
        parent_historical_peak_rss_bytes=parent_historical_peak_rss,
        worker_historical_peak_rss_bytes_sum=artifact.worker_peak_rss_bytes,
    )
    evidence = InvariantRuntimeEvidenceV1(
        invariance_resource_plan_root=plan.plan_root,
        resource_preflight=host_memory.resource_preflight,
        system_available_memory_bytes_at_preflight=(
            host_memory.system_available_bytes
        ),
        cgroup_remaining_memory_bytes_at_preflight=(
            host_memory.cgroup_remaining_bytes
        ),
        host_available_memory_bytes_at_preflight=(
            host_memory.effective_available_bytes
        ),
        parent_current_rss_bytes_at_start=parent_current_rss_at_start,
        parent_peak_current_rss_bytes_observed=(
            parent_peak_current_rss_observed
        ),
        parent_current_rss_bytes_at_completion=(
            parent_current_rss_at_completion
        ),
        parent_historical_peak_rss_bytes=parent_historical_peak_rss,
        worker_peak_rss_bytes=artifact.worker_peak_rss_bytes,
        worker_result_artifact_bytes=artifact.serialized_bytes,
        worker_result_artifact_sha256=artifact.sha256,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=(
            diagnostic_historical_sum
        ),
        unified_cgroup_membership=physical_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=physical_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=physical_snapshot.leaf_device,
        unified_cgroup_leaf_inode=physical_snapshot.leaf_inode,
        physical_memory_current_bytes_at_completion=(
            physical_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_at_completion=(
            physical_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=physical_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=MAXIMUM_ESTIMATED_RSS_BYTES,
        invariance_wall_time_nanoseconds=time.monotonic_ns() - started_ns,
        worker_timeout_seconds=INVARIANCE_WORKER_TIMEOUT_SECONDS,
        invariant_run_count=result.invariant_run_count,
        exact_invariant_comparisons=result.exact_invariant_comparisons,
        fork_input_snapshot_root=result.fork_input_snapshot_root,
        adapter_execution_pin_root=result.adapter_execution_pin_root,
        variant_suite_root=result.variant_suite_root,
        consumer_decode_count=result.consumer_decode_count,
        variant_suite_unique_envelope_object_count=(
            result.variant_suite_unique_envelope_object_count
        ),
        variant_suite_resident_object_graph_bytes_observed=(
            result.variant_suite_resident_object_graph_bytes_observed
        ),
        recognizer_output_root=result.recognizer_output_root,
        execution_containment=INVARIANCE_EXECUTION_CONTAINMENT,
        parent_side_adapter_tcb=PARENT_SIDE_ADAPTER_TCB,
        parent_side_adapter_sandboxed=False,
        process_group_empty_at_exit=True,
        child_address_space_limit_bytes=child_address_space_limit,
        within_frozen_physical_peak_cap=True,
    )
    evidence.__post_init__()
    return result, evidence


def _run_formal_covert_audit_with_reservation_v1(
    envelopes: tuple[bytes, ...],
    authority: FormalAuditAuthorityV1,
    wire_adapter: FormalWireAdapterV1,
    recognizer_adapter: FormalRecognizerAdapterV1,
    *,
    _receipt_reservation: _InProcessReceiptReservationV1,
) -> FormalCovertAuditReceiptV1:
    """Execute the closed field audit; never accept caller-produced receipts."""

    if type(envelopes) is not tuple or not 1 <= len(envelopes) <= MAXIMUM_ROWS:
        _reject("FAIL_INPUT_ROWS", "envelopes must be an exact bounded non-empty tuple")
    if any(type(item) is not bytes or len(item) != ENVELOPE_BYTES for item in envelopes):
        _reject("FAIL_FIXED_ENVELOPE", "every input must be exact 65,536-byte envelope bytes")
    _preflight_decode_memory(len(envelopes))
    # Production never accepts a caller seed.  Draw before reading authority
    # or envelope contents; the transcript discloses that this is only a
    # computational randomization assumption, not authenticated independence.
    permutation_seed = _fresh_permutation_seed()
    if type(authority) is not FormalAuditAuthorityV1 or len(authority.rows) != len(envelopes):
        _reject("FAIL_PRIVATE_AUTHORITY", "private authority must exactly cover the input rows")
    authority_primitives, adapter_authority = (
        _capture_authority_primitive_snapshot(authority)
    )
    if not isinstance(wire_adapter, FormalWireAdapterV1) or not isinstance(
        recognizer_adapter, FormalRecognizerAdapterV1
    ):
        _reject("FAIL_ADAPTER_TYPE", "wire and recognizer must implement the frozen adapters")
    try:
        pre_bind_adapter_surface = _capture_adapter_surface_without_alignment(
            wire_adapter,
            recognizer_adapter,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        _reject(
            "FAIL_ADAPTER_EXECUTION_PINS",
            f"pre-bind adapter primitive pin capture failed: {exc}",
        )
    pre_bind_v3_pins: _AdapterExecutionPinsV1 | None = None
    pre_bind_v3_alignment: FormalPrivateBatchAlignmentV1 | None = None
    if isinstance(wire_adapter, FormalWireV3AdapterV1):
        pre_bind_v3_pins, pre_bind_v3_alignment = (
            _capture_adapter_execution_pins(
                wire_adapter,
                recognizer_adapter,
            )
        )
    authority_binding = wire_adapter.bind_private_authority(
        adapter_authority,
        envelopes,
    )
    adapter_pins, pinned_private_alignment = _capture_adapter_execution_pins(
        wire_adapter,
        recognizer_adapter,
    )
    if (
        _capture_adapter_surface_without_alignment(
            wire_adapter,
            recognizer_adapter,
        )
        != pre_bind_adapter_surface
        or (
            pre_bind_v3_pins is not None
            and (
                adapter_pins != pre_bind_v3_pins
                or pinned_private_alignment != pre_bind_v3_alignment
            )
        )
    ):
        _reject(
            "FAIL_ADAPTER_EXECUTION_PIN_DRIFT",
            "adapter properties changed during private-authority binding",
        )
    wire_id = adapter_pins.wire_implementation_id
    schema_id = adapter_pins.wire_schema_id
    recognizer_id = adapter_pins.recognizer_implementation_id
    post_adapter_primitives, _ = _capture_authority_primitive_snapshot(
        adapter_authority
    )
    if post_adapter_primitives != authority_primitives:
        _reject(
            "FAIL_PRIVATE_AUTHORITY_ADAPTER_MUTATION",
            "wire adapter mutated the isolated authority snapshot",
        )
    # All downstream labels come from a fresh reconstruction that was never
    # exposed to the adapter, not from the caller object or adapter snapshot.
    execution_authority = _authority_from_primitive_rows(authority_primitives)
    if type(authority_binding) is not FormalPrivateAuthorityBindingV1:
        _reject(
            "FAIL_PRIVATE_AUTHORITY_WIRE_BINDING",
            "wire adapter returned a non-exact private authority binding",
        )
    authority_binding.__post_init__()
    private_alignment = authority_binding.alignment
    if type(private_alignment) is not FormalPrivateBatchAlignmentV1:
        _reject("FAIL_PRIVATE_ALIGNMENT", "authority binding lacks an exact alignment")
    post_bind_alignment = _verify_adapter_execution_pins(
        wire_adapter,
        recognizer_adapter,
        adapter_pins,
    )
    if (
        pinned_private_alignment != private_alignment
        or post_bind_alignment != private_alignment
    ):
        _reject(
            "FAIL_PRIVATE_AUTHORITY_WIRE_BINDING",
            "wire adapter bound labels against a different private alignment",
        )
    issuance_verified = authority_binding.alignment_issuance_verified
    if issuance_verified and private_alignment.wire_issued_alignment_id is None:
        _reject(
            "FAIL_PRIVATE_ALIGNMENT_ISSUANCE",
            "wire adapter verified an alignment with no issued content ID",
        )
    if isinstance(wire_adapter, FormalWireV3AdapterV1) and (
        not issuance_verified or not authority_binding.label_row_binding_verified
    ):
        _reject(
            "FAIL_PRIVATE_AUTHORITY_WIRE_BINDING",
            "trusted Wire V3 audits require builder-issued alignment and label rows",
        )
    aligned_authority = _align_private_authority(
        envelopes,
        execution_authority,
        private_alignment,
    )
    input_root = _input_root(envelopes)
    decoded_list = []
    for envelope in envelopes:
        item = wire_adapter.decode_envelope(envelope)
        if type(item) is not DecodedFormalEnvelopeV1:
            _reject("FAIL_DECODED_TYPE", "wire adapter returned a non-exact decoded DTO")
        _validate_decoded(item, wire_adapter)
        if (
            DEEP_SIZE_SAFETY_MULTIPLIER * _deep_object_graph_bytes(item)
            > MAXIMUM_DECODE_GRAPH_BYTES_PER_ROW
        ):
            _reject(
                "FAIL_DECODE_GRAPH_BOUND",
                "one decoded row exceeds the frozen 4 MiB conservative object-graph bound",
            )
        decoded_list.append(item)
    decoded = tuple(decoded_list)
    _verify_adapter_execution_pins(
        wire_adapter,
        recognizer_adapter,
        adapter_pins,
    )
    envelope_and_decode_bytes_bound = _measured_envelope_decode_bytes_bound(
        len(envelopes),
        envelopes,
        decoded,
    )
    feature_projection = _feature_cardinality_projection(decoded)
    _preflight_feature_extraction_memory(
        feature_projection,
        envelope_and_decode_bytes_bound,
    )
    eligibility_signatures = tuple(
        _eligibility_structural_signature(item) for item in decoded
    )
    eligibility_quota = _eligibility_mask_quota(
        decoded,
        aligned_authority,
        eligibility_signatures,
    )
    sufficiency = tuple(
        _sufficiency(
            aligned_authority,
            target,
            eligibility_signatures,
        )
        for target in TARGETS
    )
    label_counts = tuple(item.label_class_count for item in sufficiency)
    invariant_plan = _invariance_resource_plan(
        len(envelopes),
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=(
            adapter_pins.retained_graph_bytes_bound
        ),
        variant_suite_resident_bytes_per_row_bound=(
            adapter_pins.variant_suite_resident_bytes_per_row_bound
        ),
        variant_suite_resident_fixed_bytes_bound=(
            adapter_pins.variant_suite_resident_fixed_bytes_bound
        ),
    )
    if not invariant_plan.admissible:
        _reject(
            "FAIL_INVARIANCE_RESOURCE_PLAN",
            "input exceeds the frozen invariant phase resource plan",
        )
    invariant_result: _InvariantExecutionResultV1 | None = None
    invariant_runtime_evidence: InvariantRuntimeEvidenceV1 | None = None
    if all(item.sufficient for item in sufficiency):
        invariant_result, invariant_runtime_evidence = _run_invariance(
            envelopes,
            decoded,
            wire_adapter,
            recognizer_adapter,
            input_root,
            invariant_plan,
            adapter_pins,
        )
    # Invariance can consume substantial time and the insufficient branch
    # skips it entirely.  Both branches converge on one fresh feature host
    # gate immediately before the projected column allocation.
    _preflight_feature_extraction_memory(
        feature_projection,
        envelope_and_decode_bytes_bound,
    )
    manifest, columns = _extract_feature_columns(
        envelopes,
        decoded,
        feature_projection,
    )
    feature_matrix_bytes_bound = _measured_feature_matrix_bytes_bound(
        len(envelopes),
        columns,
    )
    if (
        feature_matrix_bytes_bound
        > feature_projection.projected_feature_matrix_bytes_bound
    ):
        _reject(
            "FAIL_FEATURE_ALLOCATION_BOUND",
            "materialized feature graph exceeds its conservative pre-extraction projection",
        )
    partition_preparation_started_ns = time.monotonic_ns()
    partition_preparation_deadline_ns = (
        partition_preparation_started_ns
        + PARTITION_PREPARATION_TIMEOUT_SECONDS * 1_000_000_000
    )
    partition_projection = _partition_allocation_projection(
        len(envelopes),
        columns,
        partition_preparation_deadline_ns,
    )
    _preflight_partition_allocation_memory(
        feature_matrix_bytes_bound=feature_matrix_bytes_bound,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        partition_projection=partition_projection,
        concrete_features=manifest.deduplicated_feature_count,
    )
    partitions = _materialize_partitions_bounded(
        columns,
        partition_projection,
        partition_preparation_deadline_ns,
    )
    # Statistics has no semantic dependency on decoded rows, feature columns,
    # or adapter callbacks.  Drop the auditor-owned aliases before its sole
    # partition measurement and parent-RSS plan sample.  Retained outer-frame
    # aliases remain conservatively covered by current RSS and the adapter
    # retained-graph static term.
    del decoded_list, decoded, columns, wire_adapter, recognizer_adapter
    plan = _resource_plan(
        len(envelopes),
        partitions,
        label_counts,
        preextraction_feature_matrix_bytes_bound=(
            feature_projection.projected_feature_matrix_bytes_bound
        ),
        feature_matrix_bytes_bound=feature_matrix_bytes_bound,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=(
            adapter_pins.retained_graph_bytes_bound
        ),
        partition_projection=partition_projection,
        partition_preparation_deadline_ns=(
            partition_preparation_deadline_ns
        ),
        invariance_plan=invariant_plan,
    )
    if not plan.admissible:
        _reject("FAIL_STATISTICAL_RESOURCE_PLAN", "input exceeds the frozen bitset resource plan")
    if not all(item.sufficient for item in sufficiency):
        _remaining_partition_preparation_deadline_seconds(
            partition_preparation_deadline_ns
        )
        parent_historical_peak_before_finalization = _linux_peak_rss_bytes()
        physical_snapshot = _capture_unified_cgroup_memory_snapshot_v1()
        _require_unified_cgroup_physical_peak_v1(
            failure_code="FAIL_RECEIPT_ISSUANCE_RUNTIME_RSS",
            stage="INSUFFICIENT_PRE_FINALIZATION",
            snapshot=physical_snapshot,
            parent_historical_peak_rss_bytes=(
                parent_historical_peak_before_finalization
            ),
            worker_historical_peak_rss_bytes_sum=0,
        )
        issuance_evidence = ReceiptIssuanceRuntimeEvidenceV1(
            absolute_deadline_kind="PARTITION_PREPARATION",
            absolute_deadline_monotonic_nanoseconds=(
                partition_preparation_deadline_ns
            ),
            absolute_deadline_timeout_seconds=(
                PARTITION_PREPARATION_TIMEOUT_SECONDS
            ),
            diagnostic_worker_historical_peak_rss_bytes_sum=0,
            diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=(
                parent_historical_peak_before_finalization
            ),
            diagnostic_parent_plus_worker_historical_peak_sum_bytes=(
                parent_historical_peak_before_finalization
            ),
            unified_cgroup_membership=physical_snapshot.unified_membership,
            unified_cgroup_memory_scope_root=physical_snapshot.memory_scope_root,
            unified_cgroup_leaf_device=physical_snapshot.leaf_device,
            unified_cgroup_leaf_inode=physical_snapshot.leaf_inode,
            physical_memory_current_bytes_before_receipt_finalization=(
                physical_snapshot.memory_current_bytes
            ),
            physical_memory_peak_bytes_before_receipt_finalization=(
                physical_snapshot.memory_peak_bytes
            ),
            physical_memory_max_bytes=physical_snapshot.memory_max_bytes,
            physical_memory_cap_bytes=MAXIMUM_ESTIMATED_RSS_BYTES,
            receipt_finalization_complete_pending_validation_count=1,
            receipt_finalization_locked_deadline_check_count=1,
            receipt_finalization_locked_deadline_within_absolute_limit=True,
            receipt_finalization_locked_physical_peak_within_cap=True,
        )
        receipt = _issue_formal_covert_audit_receipt_v1(
            _factory_token=_RECEIPT_EXECUTION_FACTORY_TOKEN,
            _reservation=_receipt_reservation,
            _absolute_deadline_kind="PARTITION_PREPARATION",
            _absolute_deadline_ns=partition_preparation_deadline_ns,
            disposition=FormalCovertDisposition.REJECT_INSUFFICIENT,
            authority_transcript_root=execution_authority.transcript_root,
            private_alignment_root=private_alignment.alignment_root,
            wire_issued_alignment_id=private_alignment.wire_issued_alignment_id,
            wire_alignment_issuance_verified=issuance_verified,
            private_label_projection_root=authority_binding.public_label_projection_root,
            private_label_row_binding_verified=authority_binding.label_row_binding_verified,
            audited_row_count=len(envelopes),
            wire_implementation_id=wire_id,
            wire_schema_id=schema_id,
            recognizer_implementation_id=recognizer_id,
            adapter_execution_pin_root=adapter_pins.pin_root,
            input_envelope_root=input_root,
            feature_manifest=manifest,
            resource_plan=plan,
            sufficiency=sufficiency,
            eligibility_mask_quota=eligibility_quota,
            invariant_runtime_evidence=None,
            statistics_runtime_evidence=None,
            receipt_issuance_runtime_evidence=issuance_evidence,
            results=(),
            target_family_results=(),
            invariant_execution_root=None,
            invariant_run_count=0,
            exact_invariant_comparisons=0,
            permutation_schedule_root=None,
            permutation_seed_commitment=None,
            permutation_seed_reveal_hex=None,
            holm_family_size=0,
            pairwise_holm_family_size=0,
            pairwise_holm_decision_eligible=False,
            pairwise_holm_resolution_capable=False,
            absence_fields_proved=MUST_BE_ABSENT_FIELDS,
        )
        return receipt
    if invariant_result is None or invariant_runtime_evidence is None:
        _reject(
            "FAIL_INVARIANCE_RUNTIME",
            "sufficient execution lacks completed invariant phase evidence",
        )
    statistics_started_ns = time.monotonic_ns()
    statistics_deadline_ns = (
        statistics_started_ns
        + STATISTICS_WORKER_TIMEOUT_SECONDS * 1_000_000_000
    )
    (
        results,
        target_family_results,
        schedule_root,
        statistics_plan,
        runtime_inputs,
    ) = _statistics(
        aligned_authority,
        eligibility_signatures,
        manifest,
        partitions,
        permutation_seed,
        feature_projection.projected_feature_matrix_bytes_bound,
        feature_matrix_bytes_bound,
        envelope_and_decode_bytes_bound,
        plan.partition_allocation_bytes_bound,
        plan.partition_matrix_bytes_bound,
        plan,
        statistics_started_ns,
        statistics_deadline_ns,
    )
    if statistics_plan != plan:
        _reject("FAIL_RESOURCE_PLAN_REPLAY", "statistical resource plan changed during execution")
    disposition = (
        FormalCovertDisposition.PASS
        if all(item.passed for item in results)
        else FormalCovertDisposition.FAIL
    )
    receipt_values: dict[str, object] = dict(
        disposition=disposition,
        authority_transcript_root=execution_authority.transcript_root,
        private_alignment_root=private_alignment.alignment_root,
        wire_issued_alignment_id=private_alignment.wire_issued_alignment_id,
        wire_alignment_issuance_verified=issuance_verified,
        private_label_projection_root=authority_binding.public_label_projection_root,
        private_label_row_binding_verified=authority_binding.label_row_binding_verified,
        audited_row_count=len(envelopes),
        wire_implementation_id=wire_id,
        wire_schema_id=schema_id,
        recognizer_implementation_id=recognizer_id,
        adapter_execution_pin_root=adapter_pins.pin_root,
        input_envelope_root=input_root,
        feature_manifest=manifest,
        resource_plan=plan,
        sufficiency=sufficiency,
        eligibility_mask_quota=eligibility_quota,
        invariant_runtime_evidence=invariant_runtime_evidence,
        statistics_runtime_evidence=None,
        results=results,
        target_family_results=target_family_results,
        invariant_execution_root=invariant_result.invariant_execution_root,
        invariant_run_count=invariant_result.invariant_run_count,
        exact_invariant_comparisons=(
            invariant_result.exact_invariant_comparisons
        ),
        permutation_schedule_root=schedule_root,
        permutation_seed_commitment=_permutation_seed_commitment(
            permutation_seed
        ),
        permutation_seed_reveal_hex=permutation_seed.hex(),
        holm_family_size=TARGET_FAMILY_HOLM_SIZE,
        pairwise_holm_family_size=len(results),
        pairwise_holm_decision_eligible=False,
        pairwise_holm_resolution_capable=(
            FAMILY_WISE_ALPHA_DENOMINATOR * len(results)
            < EXCHANGEABLE_SAMPLE_SIZE
        ),
        absence_fields_proved=MUST_BE_ABSENT_FIELDS,
        positive_decision_claim=(
            POSITIVE_DECISION_CLAIM
            if disposition is FormalCovertDisposition.PASS
            else None
        ),
    )
    if type(runtime_inputs) is not _StatisticalRuntimeEvidenceInputsV1:
        raise RuntimeError("statistics returned malformed runtime inputs")
    worker_peak_sum = sum(runtime_inputs.worker_peak_rss_bytes)
    _remaining_statistics_deadline_seconds(statistics_deadline_ns)
    parent_historical_peak_before_finalization = _linux_peak_rss_bytes()
    diagnostic_historical_sum = (
        parent_historical_peak_before_finalization + worker_peak_sum
    )
    invariant_physical_snapshot = UnifiedCgroupMemorySnapshotV1(
        invariant_runtime_evidence.unified_cgroup_membership,
        invariant_runtime_evidence.unified_cgroup_memory_scope_root,
        invariant_runtime_evidence.unified_cgroup_leaf_device,
        invariant_runtime_evidence.unified_cgroup_leaf_inode,
        invariant_runtime_evidence.physical_memory_current_bytes_at_completion,
        invariant_runtime_evidence.physical_memory_peak_bytes_at_completion,
        invariant_runtime_evidence.physical_memory_max_bytes,
    )
    physical_snapshot = _capture_unified_cgroup_memory_snapshot_v1()
    if (
        physical_snapshot.memory_peak_bytes
        < invariant_physical_snapshot.memory_peak_bytes
    ):
        _reject(
            "FAIL_PLATFORM_RESOURCE_SEMANTICS",
            "unified cgroup memory.peak decreased after invariance",
        )
    _require_unified_cgroup_physical_peak_v1(
        failure_code="FAIL_STATISTICAL_RUNTIME_RSS",
        stage="STATISTICS_PRE_FINALIZATION",
        snapshot=physical_snapshot,
        parent_historical_peak_rss_bytes=(
            parent_historical_peak_before_finalization
        ),
        worker_historical_peak_rss_bytes_sum=worker_peak_sum,
        expected_scope=invariant_physical_snapshot,
    )
    runtime_evidence = StatisticalRuntimeEvidenceV1(
        resource_plan_root=runtime_inputs.resource_plan_root,
        resource_preflight=runtime_inputs.resource_preflight,
        system_available_memory_bytes_at_preflight=(
            runtime_inputs.system_available_memory_bytes_at_preflight
        ),
        cgroup_remaining_memory_bytes_at_preflight=(
            runtime_inputs.cgroup_remaining_memory_bytes_at_preflight
        ),
        host_available_memory_bytes_at_preflight=(
            runtime_inputs.host_available_memory_bytes_at_preflight
        ),
        parent_historical_peak_rss_bytes_before_receipt_finalization=(
            parent_historical_peak_before_finalization
        ),
        worker_peak_rss_bytes=runtime_inputs.worker_peak_rss_bytes,
        worker_result_artifact_bytes=runtime_inputs.worker_result_artifact_bytes,
        worker_result_artifact_sha256=(
            runtime_inputs.worker_result_artifact_sha256
        ),
        parent_result_object_graph_bytes_observed=(
            runtime_inputs.parent_result_object_graph_bytes_observed
        ),
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=(
            diagnostic_historical_sum
        ),
        unified_cgroup_membership=physical_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=physical_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=physical_snapshot.leaf_device,
        unified_cgroup_leaf_inode=physical_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            physical_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            physical_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=physical_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=MAXIMUM_ESTIMATED_RSS_BYTES,
        statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds=(
            time.monotonic_ns() - statistics_started_ns
        ),
        worker_timeout_seconds=runtime_inputs.worker_timeout_seconds,
        parent_postprocess_deadline_schema=(
            runtime_inputs.parent_postprocess_deadline_schema
        ),
        parent_postprocess_deadline_preemptive=(
            runtime_inputs.parent_postprocess_deadline_preemptive
        ),
        external_supervisor_required=runtime_inputs.external_supervisor_required,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_physical_peak_within_cap=True,
        within_frozen_physical_peak_cap=True,
    )
    issuance_evidence = ReceiptIssuanceRuntimeEvidenceV1(
        absolute_deadline_kind="STATISTICS",
        absolute_deadline_monotonic_nanoseconds=statistics_deadline_ns,
        absolute_deadline_timeout_seconds=STATISTICS_WORKER_TIMEOUT_SECONDS,
        diagnostic_worker_historical_peak_rss_bytes_sum=worker_peak_sum,
        diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=(
            parent_historical_peak_before_finalization
        ),
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=(
            diagnostic_historical_sum
        ),
        unified_cgroup_membership=physical_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=physical_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=physical_snapshot.leaf_device,
        unified_cgroup_leaf_inode=physical_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            physical_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            physical_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=physical_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=MAXIMUM_ESTIMATED_RSS_BYTES,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_locked_deadline_check_count=1,
        receipt_finalization_locked_deadline_within_absolute_limit=True,
        receipt_finalization_locked_physical_peak_within_cap=True,
    )
    receipt_values["statistics_runtime_evidence"] = runtime_evidence
    receipt_values["receipt_issuance_runtime_evidence"] = issuance_evidence
    candidate = _construct_formal_covert_audit_receipt_v1(**receipt_values)
    _remaining_statistics_deadline_seconds(statistics_deadline_ns)
    _register_preconstructed_formal_covert_audit_receipt_v1(
        _factory_token=_RECEIPT_EXECUTION_FACTORY_TOKEN,
        _reservation=_receipt_reservation,
        receipt=candidate,
    )
    try:
        validated_id = _validate_pending_formal_covert_audit_receipt_v1(
            _factory_token=_RECEIPT_EXECUTION_FACTORY_TOKEN,
            _reservation=_receipt_reservation,
            receipt=candidate,
        )
        _commit_pending_formal_covert_audit_receipt_v1(
            _factory_token=_RECEIPT_EXECUTION_FACTORY_TOKEN,
            _reservation=_receipt_reservation,
            receipt=candidate,
            validated_id=validated_id,
            absolute_deadline_kind="STATISTICS",
            absolute_deadline_ns=statistics_deadline_ns,
        )
    except BaseException as primary:
        try:
            _rollback_pending_in_process_receipt_issuance(
                candidate,
                _receipt_reservation,
            )
        except BaseException as cleanup_error:
            _record_cleanup_failure_without_replacing_primary(
                primary,
                cleanup_error,
                "statistical direct receipt rollback",
            )
        raise
    return candidate


def run_formal_covert_audit_v1(
    envelopes: tuple[bytes, ...],
    authority: FormalAuditAuthorityV1,
    wire_adapter: FormalWireAdapterV1,
    recognizer_adapter: FormalRecognizerAdapterV1,
) -> FormalCovertAuditReceiptV1:
    """Execute one bounded audit under a reserved process-local receipt slot."""

    reservation = _reserve_in_process_receipt_slot()
    try:
        return _run_formal_covert_audit_with_reservation_v1(
            envelopes,
            authority,
            wire_adapter,
            recognizer_adapter,
            _receipt_reservation=reservation,
        )
    except BaseException as primary:
        try:
            _rollback_in_process_receipt_issuance(reservation)
        except BaseException as cleanup_error:
            _record_cleanup_failure_without_replacing_primary(
                primary,
                cleanup_error,
                "outer receipt rollback",
            )
        try:
            _cancel_in_process_receipt_reservation(reservation)
        except BaseException as cleanup_error:
            _record_cleanup_failure_without_replacing_primary(
                primary,
                cleanup_error,
                "outer receipt reservation cancellation",
            )
        raise


def validate_formal_covert_audit_receipt_v1(
    receipt: FormalCovertAuditReceiptV1,
) -> str:
    """Validate a transcript issued by this process's execution factory.

    The returned content hash is not a signature or cross-process authority.
    Durable qualification must bind it in an external signed/CAS artifact.
    """

    if type(receipt) is not FormalCovertAuditReceiptV1:
        raise TypeError("receipt validator requires the exact receipt DTO")
    identity = id(receipt)
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        entry = _IN_PROCESS_EXECUTED_RECEIPTS.get(identity)
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.receipt is not receipt
        ):
            raise ValueError(
                "publicly constructed transcript was not issued by the in-process execution factory"
            )
        if (
            type(entry) is not _InProcessReceiptRegistryEntryV1
            or entry.state is not _InProcessReceiptRegistryStateV1.ISSUED
            or type(entry.receipt_id) is not str
            or type(entry.reservation) is not _InProcessReceiptReservationV1
            or entry.reservation.entry_identity != identity
        ):
            raise ValueError("a PENDING receipt is not publicly issued")
        registered_id = entry.receipt_id
    replayed_id = _replay_formal_covert_audit_receipt_content_v1(
        receipt,
        registered_id,
    )
    with _IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        if (
            _IN_PROCESS_EXECUTED_RECEIPTS.get(identity) is not entry
            or entry.receipt is not receipt
            or entry.receipt_id != replayed_id
            or type(entry.receipt_id) is not str
            or entry.state is not _InProcessReceiptRegistryStateV1.ISSUED
            or type(entry.reservation) is not _InProcessReceiptReservationV1
            or entry.reservation.entry_identity != identity
        ):
            raise ValueError("issued receipt registry entry changed during full replay")
    return replayed_id


def validate_frozen_policy_binding_v1() -> str:
    """Replay all scalar/list bindings against the exact freeze."""

    frozen = frozen_phase2b_exact_freeze().covert_channel_audit
    observed = (
        frozen.envelope_bytes,
        frozen.label_permutations,
        frozen.global_consistent_renamings,
        frozen.global_case_order_permutations,
        frozen.within_case_observation_order_permutations,
        frozen.maximum_normalized_mutual_information,
        frozen.maximum_balanced_accuracy_advantage,
        frozen.family_wise_alpha,
        frozen.unique_id_feature_family,
        frozen.renaming_namespaces,
        frozen.renaming_invariants,
    )
    expected = (
        ENVELOPE_BYTES,
        LABEL_PERMUTATIONS,
        GLOBAL_RENAMINGS,
        GLOBAL_CASE_ORDER_PERMUTATIONS,
        OBSERVATION_ORDER_PERMUTATIONS,
        NMI_LIMIT,
        BALANCED_ACCURACY_ADVANTAGE_LIMIT,
        FAMILY_WISE_ALPHA,
        UNIQUE_ID_FEATURE_FAMILY,
        RENAMING_NAMESPACES,
        (
            "decision_identical",
            "family_identical",
            "binding_identical",
            "scale_set_identical",
            "candidate_residual_vector_bit_identical",
        ),
    )
    if observed != expected:
        raise RuntimeError("formal covert auditor policy drift from exact freeze")
    return stable_hash(
        {"semantics_id": SEMANTICS_ID, "binding": expected},
        prefix="phase2b_formal_covert_policy_binding_",
    )


POLICY_BINDING_ID: Final = validate_frozen_policy_binding_v1()


__all__ = [
    "AnswerabilityLabelV1",
    "ChannelLeafV1",
    "ChannelOnlyLeafOwnerV1",
    "DecodedFormalEnvelopeV1",
    "EligibilityMaskQuotaReceiptV1",
    "EligibilityMaskTargetQuotaV1",
    "ExactProbabilityV1",
    "FeatureAliasCoverageV1",
    "FeatureManifestV1",
    "FormalAuditAuthorityV1",
    "FormalAuthorityLabelInputV1",
    "FormalAuthorityLabelRowV1",
    "FormalBindingV1",
    "FormalCovertAuditReceiptV1",
    "FormalCovertAuditRejectionV1",
    "FormalCovertDisposition",
    "FormalObservationOrderPlanV1",
    "FormalPrivateAlignmentLinkV1",
    "FormalPrivateAuthorityBindingV1",
    "FormalPrivateBatchAlignmentV1",
    "FormalPrivateUuidMappingV1",
    "FormalRecognizerAdapterV1",
    "FormalRecognizerDecisionV1",
    "FormalWireAdapterV1",
    "FormalWireV3AdapterV1",
    "FormalUuidRenamingPlanV1",
    "FormalVariantPlanV1",
    "GLOBAL_CASE_ORDER_PERMUTATIONS",
    "GLOBAL_RENAMINGS",
    "LABEL_PERMUTATIONS",
    "MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS",
    "MUST_BE_ABSENT_FIELDS",
    "NamespaceLeafV1",
    "OBSERVATION_ORDER_PERMUTATIONS",
    "POLICY_BINDING_ID",
    "POSITIVE_DECISION_CLAIM",
    "ResourcePlanV1",
    "ReceiptIssuanceRuntimeEvidenceV1",
    "SEMANTICS_ID",
    "StatisticalRuntimeEvidenceV1",
    "TargetFamilyMaxTResultV1",
    "TargetFeatureResultV1",
    "TargetSufficiencyV1",
    "UNIQUE_ID_FEATURE_FAMILY",
    "TRANSCRIPT_AUTHENTICATION",
    "UnifiedCgroupMemorySnapshotV1",
    "issue_development_formal_audit_authority_v1",
    "issue_formal_wire_v3_audit_authority_v1",
    "run_formal_covert_audit_v1",
    "release_formal_covert_audit_receipt_v1",
    "validate_formal_covert_audit_receipt_v1",
    "validate_frozen_policy_binding_v1",
]
