"""Independent tests for the actual-replay admission gap inventory V2.

The positive object proves only that a content-addressed catalogue of missing
evidence and declarative ordering gaps is frozen.  These tests never provide
evidence, run a recognizer, verify a ledger or signature, score predictions,
evaluate gates, or establish effect/C1 evidence.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable

import pytest

import hegel_machine.phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2 as gap_v2


EXPECTED_PUBLIC_SURFACE = (
    "ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION",
    "ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL",
    "ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_SCHEMA_ID",
    "ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_POLICY_ID",
    "AdmissionGapInventoryDispositionV2",
    "AdmissionGapInventoryReasonV2",
    "MissingEvidenceRequirementV2",
    "RequiredOrderingStatementV2",
    "AdmissionGapInventoryContractV2",
    "AdmissionGapInventoryValidationV2",
    "AdmissionGapInventoryRejectionV2",
    "frozen_actual_unsealed_960_replay_admission_gap_inventory_v2",
    "validate_actual_unsealed_960_replay_admission_gap_inventory_v2",
)
EXPECTED_VERSION = (
    "hegel-machine-phase2b-actual-unsealed-960-replay-admission-gap-inventory/2"
)
EXPECTED_CLAIM_LEVEL = (
    "NON_AUTHORITATIVE_ACTUAL_REPLAY_ADMISSION_GAP_INVENTORY_ONLY"
)
EXPECTED_DISPOSITIONS = (
    "GAP_INVENTORY_FROZEN_NOT_ADMITTED",
    "REJECTED",
)
EXPECTED_REASONS = (
    "MISSING_EVIDENCE_AND_ORDERING_GAPS_INVENTORIED_NOT_VERIFIED",
    "WRONG_INPUT_TYPE",
    "CROSS_VERSION_INPUT",
    "CONTRACT_INVALID",
    "INTERNAL_ERROR",
)

IDENTITY_FIELDS = (
    "dependency_name",
    "identity_kind",
    "identity_value",
)
MISSING_FIELDS = (
    "ordinal",
    "evidence_name",
    "purpose",
    "status",
    "missing",
    "supplied_by_this_contract",
    "verified",
    "verifier_implemented",
    "requirement_id",
)
ORDERING_FIELDS = (
    "ordinal",
    "predecessor_stage",
    "successor_stage",
    "statement",
    "status",
    "event_schema_supplied",
    "signature_schema_supplied",
    "ledger_schema_supplied",
    "timing_verifier_implemented",
    "ordering_statement_id",
)
TRUE_CLAIMS = (
    "gap_inventory_frozen",
    "required_ordering_statements_frozen",
    "upstream_dependency_identities_frozen",
    "no_evidence_input_schema_accepted",
    "content_addressed_gap_catalog_validated",
    "atomic_catalog_validation_verified",
)
FALSE_CLAIMS = (
    "admission_evidence_contract_complete",
    "cryptographic_record_schemas_complete",
    "signature_payload_profiles_complete",
    "pinned_signer_registry_schema_complete",
    "attempt_registry_schema_complete",
    "rerun_policy_executable",
    "timeline_event_schemas_complete",
    "timeline_verifier_implemented",
    "evidence_role_lifecycle_fully_bound",
    "actual_evidence_inputs_accepted",
    "verifier_implemented",
    "admission_ready",
    "execution_authorized",
    "one_shot_policy_enforced",
    "signature_coverage_verified",
    "evidence_chain_verified",
    "attempt_terminal_state_verified",
    "pre_reveal_commitment_timing_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
    "answer_commitment_opening_verified",
    "gate_input_commitment_opening_verified",
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "partition_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_capacity_evidence",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "preservation_prediction_commitment_verified",
    "evidence_supplied",
    "evidence_verified",
    "custodian_identity_verified",
    "signer_chain_verified",
    "signatures_verified",
    "pinned_signer_key_registry_verified",
    "durable_attempt_ledger_verified",
    "attempt_policy_enforced",
    "retry_authorization_verified",
    "timeline_observed",
    "timeline_order_verified",
    "package_commitment_before_run_verified",
    "prediction_before_reveal_verified",
    "answer_manifest_authority_verified",
    "gate_input_manifest_authority_verified",
    "input_archive_authority_verified",
    "prediction_archive_authority_verified",
    "allocation_authority_verified",
    "evaluator_input_authority_verified",
    "execution_freeze_authority_verified",
    "runtime_attestation_authority_verified",
    "raw_input_archive_replayed",
    "raw_prediction_archive_replayed",
    "canonical_archive_replay_verified",
    "decoder_transcript_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "origin_authenticated",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "metric_results_materialized",
    "scored_rows_materialized",
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

CONTRACT_FIELDS = (
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "upstream_dependency_identities",
    "upstream_dependency_identity_ids",
    "upstream_dependency_identity_ids_root",
    "missing_evidence_requirements",
    "missing_evidence_requirement_ids_root",
    "required_ordering_statements",
    "required_ordering_statement_ids_root",
    "upstream_dependency_identity_count",
    "missing_evidence_requirement_count",
    "required_ordering_statement_count",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
    "contract_sha256",
    "contract_id",
)
VALIDATION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "validation_id",
    "contract_id",
    "contract_sha256",
    "upstream_dependency_identity_ids_root",
    "missing_evidence_requirement_ids_root",
    "required_ordering_statement_ids_root",
    "upstream_dependency_identity_count",
    "missing_evidence_requirement_count",
    "required_ordering_statement_count",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
    "upstream_dependency_identities",
    "missing_evidence_requirements",
    "required_ordering_statements",
)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "claim_level",
    "schema_id",
    "policy_id",
    "validation_id",
    "contract_id",
    "contract_sha256",
    "upstream_dependency_identity_ids_root",
    "missing_evidence_requirement_ids_root",
    "required_ordering_statement_ids_root",
    "upstream_dependency_identity_count",
    "missing_evidence_requirement_count",
    "required_ordering_statement_count",
    "upstream_dependency_identities",
    "missing_evidence_requirements",
    "required_ordering_statements",
    "partial_output_published",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
)

EXPECTED_UPSTREAM_IDENTITIES = (
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "version",
        "hegel-machine-phase2b-actual-unsealed-960-replay-input-contract/2",
    ),
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "schema_id",
        "phase2b_actual_unsealed_960_replay_input_contract_schema_v2_"
        "a4f61ddfb07643e23ac404616062127e2ae6ca02f13b29c265062d6a1f660f4a",
    ),
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "policy_id",
        "phase2b_actual_unsealed_960_replay_input_contract_policy_v2_"
        "a12ca51dd6f17f29a28a7229f4108c32f438d4775a367a7ad3e5a6275557b531",
    ),
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "claim_level",
        "NON_AUTHORITATIVE_ACTUAL_REPLAY_INPUT_CONTRACT_ONLY",
    ),
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "source_sha256",
        "35da6f01163835ca90c24cdbd4ad85a1f7f0b2ef78ceb6e70fe80de2534814f2",
    ),
)

EXPECTED_GAP_ROLES = (
    "durable_signed_custodian_ledger_and_signer_chain",
    "prediction_reveal_scoring_chronology",
    "raw_input_and_prediction_archive_cas",
    "canonical_archive_replay_and_decoder_transcript",
    "source_image_sbom_config_runtime_execution_attestations",
    "actual_recognizer_execution_capacity_and_output_attestation",
    "allocation_source_batch_partition_answer_authorities",
    "independent_720_240_generation_and_source_disjointness_receipt",
    "audit_archive_formal_uuid_and_covert_receipts",
    "three_baseline_prediction_outputs",
    "durable_fail_closed_attempt_event_denominator",
    "preservation_496_legal_76_invalid_pair_results",
    "semantic_conflict_240_descriptor_audit_package",
    "scale_regret_oracle_loss_and_normalizer",
    "clustered_bootstrap_inputs_and_statistic",
    "metric_wilson_overall_and_slice_gate_report",
    "attempt_rerun_permanent_record",
    "formal_c1_consumed_sealed_report",
)
EXPECTED_GAP_PURPOSES = (
    "authenticate_one_shot_append_only_custody; unresolved_schema_gaps=authoritative_custodian_identity_pinned_keys_detached_signatures_append_only_ledger_and_external_trust_anchor",
    "prove_package_commitment_precedes_run_and_predictions_precede_reveal; unresolved_schema_gaps=signed_event_records_chain_root_external_timestamps_and_ordering_verifier",
    "bind_authoritative_input_and_output_bytes; unresolved_schema_gaps=canonical_archive_bytes_content_addresses_authorities_and_membership_verifier",
    "prove_public_canonical_input_and_prediction_replay; unresolved_schema_gaps=decoder_version_transcript_format_replay_rules_and_verifier",
    "authenticate_frozen_executable_environment; unresolved_schema_gaps=authoritative_image_SBOM_configuration_OCI_spec_runtime_attestation_and_signature_schemas",
    "prove_recognizer_execution_capacity_and_output_origin; unresolved_schema_gaps=run_identity_capacity_isolation_exit_output_origin_and_attestation_verifier",
    "authenticate_membership_and_evaluator_inputs; unresolved_schema_gaps=allocation_source_batch_partition_answer_gate_authority_records_and_resolvers",
    "prove_post_validation_generation_source_public_disjointness_and_latent_case_independence; unresolved_schema_gaps=generator_custody_source_projection_disjointness_receipt_and_independent_audit_verifier",
    "bind_formal_audit_outputs; unresolved_schema_gaps=audit_archive_UUID_covert_receipt_content_schemas_authorities_and_verifiers",
    "support_embedding_nearest_prototype_frozen_llm_semantic_only_flat_learned_typed_comparisons; unresolved_schema_gaps=baseline_registration_implementation_prediction_archive_and_comparison_report_schemas",
    "define_fail_closed_rate_over_all_attempt_events; unresolved_schema_gaps=durable_attempt_open_terminal_registry_denominator_finalization_and_no_cherry_pick_verifier",
    "bind_original_variant_links_predictions_and_evaluator_results_for_496_legal_and_76_invalid_pairs; unresolved_schema_gaps=pair_manifest_pre_reveal_predictions_post_reveal_results_and_consistency_verifier",
    "support_separately_reported_challenge_audit_or_scoring_excluded_from_main_thresholds_and_tuning; unresolved_schema_gaps=descriptor_package_authority_exclusion_receipt_and_audit_verifier",
    "define_normalized_per_case_regret_point_max_0_05_and_bootstrap_upper_bound_0_08; unresolved_schema_gaps=oracle_loss_normalizer_per_case_regret_report_and_verifier",
    "define_10000_replicate_paired_latent_base_case_cluster_bootstrap_master_seed_411876909552964556_uint32_2611585425_original_and_preservation_variants_one_sided_95_percentile; unresolved_schema_gaps=cluster_inputs_statistic_replicate_report_and_recomputation_verifier",
    "materialize_metric_counts_frozen_thresholds_one_sided_95_percent_wilson_and_gate_results; unresolved_schema_gaps=scored_rows_metric_counts_Wilson_bounds_gate_report_and_independent_verifier",
    "enforce_attempt_policy_and_permanent_attempt_history; unresolved_schema_gaps=reason_action_authorization_retry_linkage_permanent_attempt_record_and_policy_verifier",
    "prove_all_overall_family_scale_gates_and_consumed_ledger_state; unresolved_schema_gaps=post_terminal_denominator_score_gate_consumption_C1_report_and_authority_verifier",
)
EXPECTED_MISSING_SPECS = tuple(
    (ordinal, name, purpose)
    for ordinal, (name, purpose) in enumerate(
        zip(EXPECTED_GAP_ROLES, EXPECTED_GAP_PURPOSES)
    )
)

EXPECTED_ORDERING_TRIPLES = (
    ("preregister_before_input", "attempt_preregistered", "raw_input_committed"),
    (
        "input_before_package_freeze",
        "raw_input_committed",
        "answer_gate_generation_disjointness_commitments_frozen",
    ),
    (
        "package_freeze_before_run",
        "answer_gate_generation_disjointness_commitments_frozen",
        "recognizer_run_started",
    ),
    ("start_before_finish", "recognizer_run_started", "recognizer_run_finished"),
    (
        "finish_before_prediction_commitment",
        "recognizer_run_finished",
        "prediction_audit_baseline_preservation_outputs_committed",
    ),
    (
        "predictions_before_reveal",
        "prediction_audit_baseline_preservation_outputs_committed",
        "answer_and_gate_packages_revealed",
    ),
    (
        "prediction_commitment_before_optional_upload_failure",
        "prediction_audit_baseline_preservation_outputs_committed",
        "committed_output_upload_failed",
    ),
    (
        "upload_failure_before_reupload",
        "committed_output_upload_failed",
        "committed_output_reuploaded",
    ),
    (
        "reupload_before_reveal",
        "committed_output_reuploaded",
        "answer_and_gate_packages_revealed",
    ),
    (
        "reveal_before_scoring",
        "answer_and_gate_packages_revealed",
        "scoring_started",
    ),
    (
        "scoring_before_reports",
        "scoring_started",
        "score_and_auxiliary_reports_committed",
    ),
    (
        "reports_before_consumed_terminal",
        "score_and_auxiliary_reports_committed",
        "attempt_consumed_terminal",
    ),
    (
        "terminal_before_append_receipt",
        "attempt_terminal",
        "terminal_append_receipt",
    ),
    (
        "append_receipt_before_retry_authorization",
        "terminal_append_receipt",
        "retry_authorization_if_eligible",
    ),
    (
        "retry_authorization_before_next_open",
        "retry_authorization_if_eligible",
        "next_attempt_preregistered",
    ),
    (
        "last_append_receipt_before_campaign_finalization",
        "last_terminal_append_receipt",
        "campaign_finalization",
    ),
)
EXPECTED_ORDERING_SPECS = tuple(
    (
        ordinal,
        predecessor,
        successor,
        name
        + ": high_level_declarative_gap_only; required_order="
        + predecessor
        + "<"
        + successor
        + "; event_schema_signature_schema_ledger_schema_and_timing_verifier_are_not_supplied",
    )
    for ordinal, (name, predecessor, successor) in enumerate(
        EXPECTED_ORDERING_TRIPLES
    )
)

MISSING_STATUS = "MISSING_UNSUPPLIED_UNVERIFIED_NO_VERIFIER"
ORDERING_STATUS = "DECLARATIVE_ONLY_UNVERIFIED_NO_VERIFIER"
TYPE_AND_CAP_MANIFEST = (
    ("text", "exact_str_nonempty_valid_UTF8_maximum_4096_bytes"),
    ("sha256", "exact_lowercase_64_hex"),
    ("content_id", "exact_declared_prefix_plus_lowercase_64_hex"),
    ("integer", "exact_int_not_bool_nonnegative_maximum_9223372036854775807"),
    ("boolean", "exact_bool"),
    ("tuple", "exact_tuple_declared_length_maximum_64_snapshot_once"),
    ("nested_DTO", "exact_type_exact_field_manifest_snapshot_once"),
    (
        "prehash",
        "close_entire_graph_types_formats_caps_counts_and_duplicates_before_any_hash",
    ),
)
NON_ORDERING_GAP_STATEMENTS = (
    "each_attempt_OPEN_requires_exactly_one_terminal_closure_but_no_record_schema_or_verifier_is_supplied",
    "no_attempt_event_may_follow_that_attempt_terminal_but_no_event_schema_or_verifier_is_supplied",
    "attempt_indices_must_be_contiguous_0_through_at_most_2_but_no_registry_schema_or_verifier_is_supplied",
    "REUPLOAD_COMMITTED_OUTPUT_must_not_create_a_next_attempt_but_no_retry_schema_or_verifier_is_supplied",
    "campaign_finalization_must_follow_all_terminal_append_receipts_but_no_finalization_schema_or_verifier_is_supplied",
)
CONTENT_ADDRESS_FORMULAS = (
    (
        "primitive",
        "prefix || sha256(domain || canonical_json(tuple((field_name,exact_closed_value) in declared_order)))",
    ),
    (
        "root",
        "prefix || sha256(domain || u32be(count) || repeated(u16be(ascii_identifier_length) || ascii_identifier))",
    ),
    (
        "contract",
        "contract_prefix || sha256(contract_domain || canonical_json(exact_ordered_contract_preimage; upstream_dependency_identities_are_tuple_of_declared_order_named_pairs; missing_and_ordering_DTOs_are_tuple_of_declared_order_named_pairs))",
    ),
    (
        "validation",
        "validation_prefix || sha256(validation_domain || canonical_json(exact_ordered_validation_preimage_excluding_validation_id; upstream_dependency_identities_are_tuple_of_declared_order_named_pairs; missing_and_ordering_DTOs_are_tuple_of_declared_order_named_pairs))",
    ),
    (
        "schema",
        "schema_prefix || sha256(schema_domain || canonical_json(exact_ordered_SCHEMA_PREIMAGE_V2))",
    ),
    (
        "policy",
        "policy_prefix || sha256(policy_domain || canonical_json(exact_ordered_POLICY_PREIMAGE_V2))",
    ),
)
VALIDATION_ORDER = (
    "exact_top_level_type",
    "snapshot_each_top_and_nested_field_once",
    "global_type_format_cap_count_and_duplicate_preflight",
    "cross_version_identity_check",
    "independent_primitive_content_ID_recalculation",
    "independent_length_framed_root_recalculation",
    "independent_contract_SHA_and_ID_recalculation",
    "exact_frozen_primitive_snapshot_comparison",
    "fresh_success_graph_or_atomic_all_false_no_catalog_no_ID_rejection",
)
GLOBAL_PREFLIGHT_SENTINEL = (
    "snapshot_every_top_and_nested_field_exactly_once",
    "close_exact_types_UTF8_caps_integer_caps_SHA64_and_prefixed_ID_formats",
    "close_exact_tuple_counts_declared_order_and_no_duplicates",
    "compare_closed_upstream_missing_evidence_ordering_claim_and_count_catalogs_to_frozen_specs",
    "only_after_complete_global_closure_recompute_primitive_IDs_length_framed_roots_contract_SHA_and_contract_ID",
    "atomic_rejection_exposes_no_catalog_no_ID_and_all_claims_false",
)
API_BOUNDARY = (
    "getter_has_no_arguments",
    "validator_has_keyword_only_contract_argument",
    "no_evidence_bytes_paths_timestamps_signatures_ledgers_attestations_or_booleans_accepted",
    "catalogue_validation_never_admits_or_authorizes_execution",
)
FORBIDDEN_OPERATIONS = (
    "no_runner_decoder_scorer_formal_validator_ledger_signature_or_runtime_verifier",
    "no_filesystem_network_clock_random_subprocess_container_or_Q_access",
)

DOMAINS_AND_PREFIXES = {
    "identity": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/UPSTREAM_IDENTITY/V2\x00",
        "phase2b_actual_replay_admission_gap_upstream_identity_v2_",
    ),
    "missing": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/MISSING_EVIDENCE/V2\x00",
        "phase2b_actual_replay_admission_gap_missing_evidence_v2_",
    ),
    "ordering": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/ORDERING/V2\x00",
        "phase2b_actual_replay_admission_gap_ordering_v2_",
    ),
    "identity_root": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/UPSTREAM_IDENTITY_ROOT/V2\x00",
        "phase2b_actual_replay_admission_gap_upstream_identities_v2_",
    ),
    "missing_root": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/MISSING_EVIDENCE_ROOT/V2\x00",
        "phase2b_actual_replay_admission_gap_missing_evidence_requirements_v2_",
    ),
    "ordering_root": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/ORDERING_ROOT/V2\x00",
        "phase2b_actual_replay_admission_gap_ordering_statements_v2_",
    ),
    "contract": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/CONTRACT/V2\x00",
        "phase2b_actual_unsealed_960_replay_admission_gap_inventory_v2_",
    ),
    "validation": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/VALIDATION/V2\x00",
        "phase2b_actual_unsealed_960_replay_admission_gap_inventory_validation_v2_",
    ),
    "schema": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/SCHEMA/V2\x00",
        "phase2b_actual_unsealed_960_replay_admission_gap_inventory_schema_v2_",
    ),
    "policy": (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/POLICY/V2\x00",
        "phase2b_actual_unsealed_960_replay_admission_gap_inventory_policy_v2_",
    ),
}
CONTENT_ADDRESS_BINDING_KEYS = (
    ("upstream_dependency_identity_id", "identity"),
    ("missing_evidence_requirement_id", "missing"),
    ("required_ordering_statement_id", "ordering"),
    ("upstream_dependency_identity_ids_root", "identity_root"),
    ("missing_evidence_requirement_ids_root", "missing_root"),
    ("required_ordering_statement_ids_root", "ordering_root"),
    ("contract_id", "contract"),
    ("validation_id", "validation"),
    ("schema_id", "schema"),
    ("policy_id", "policy"),
)
CONTENT_ADDRESS_DOMAIN_PREFIX_BINDINGS = tuple(
    (
        public_name,
        DOMAINS_AND_PREFIXES[kind][0].hex(),
        DOMAINS_AND_PREFIXES[kind][1],
    )
    for public_name, kind in CONTENT_ADDRESS_BINDING_KEYS
)
EXPECTED_SCHEMA_PREIMAGE = (
    ("version", EXPECTED_VERSION),
    ("claim_level", EXPECTED_CLAIM_LEVEL),
    (
        "field_manifests",
        (
            ("upstream_dependency_identity", IDENTITY_FIELDS),
            ("missing_evidence_requirement", MISSING_FIELDS),
            ("required_ordering_statement", ORDERING_FIELDS),
            ("contract", CONTRACT_FIELDS),
            ("validation", VALIDATION_FIELDS),
            ("rejection", REJECTION_FIELDS),
        ),
    ),
    ("disposition_values", EXPECTED_DISPOSITIONS),
    ("reason_values", EXPECTED_REASONS),
    ("type_and_cap_manifest", TYPE_AND_CAP_MANIFEST),
)

EXPECTED_SCHEMA_ID = (
    "phase2b_actual_unsealed_960_replay_admission_gap_inventory_schema_v2_"
    "f7c827897ec9719516f612cf1b072efedd8d2814fea8d6eb8e544ca06fb20866"
)
EXPECTED_POLICY_ID = (
    "phase2b_actual_unsealed_960_replay_admission_gap_inventory_policy_v2_"
    "fbb1b97f5dfcd794c0dca6c50a0e5ef8f2a8965a277150ea3027b0985ae1e0f1"
)
EXPECTED_CONTRACT_SHA = "5cb13b80972410f0434fc26614b87b6105086ff0d5ced2c19ea2b592ba58d455"
EXPECTED_CONTRACT_ID = DOMAINS_AND_PREFIXES["contract"][1] + EXPECTED_CONTRACT_SHA
EXPECTED_VALIDATION_ID = (
    DOMAINS_AND_PREFIXES["validation"][1]
    + "11ecd75cc6d982ff57a801330702af9b90b43bf88ce7bc3b4c2c646fcfe75098"
)


def _plain(value: object) -> object:
    """Independent primitive encoder matching the frozen canonical JSON rules."""

    if isinstance(value, Enum):
        return value.value
    if type(value) in {tuple, list}:
        return [_plain(item) for item in value]  # type: ignore[union-attr]
    if type(value) is dict:
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))  # type: ignore[union-attr]
        }
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"non-primitive oracle value: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(domain: bytes, preimage: object) -> str:
    return hashlib.sha256(domain + _canonical_json(preimage).encode("utf-8")).hexdigest()


def _independent_schema_id() -> str:
    domain, prefix = DOMAINS_AND_PREFIXES["schema"]
    return prefix + _digest(domain, EXPECTED_SCHEMA_PREIMAGE)


def _independent_policy_preimage(schema_id: str) -> tuple[tuple[str, object], ...]:
    return (
        ("version", EXPECTED_VERSION),
        ("schema_id", schema_id),
        ("claim_level", EXPECTED_CLAIM_LEVEL),
        ("success_disposition", EXPECTED_DISPOSITIONS[0]),
        ("success_reason", EXPECTED_REASONS[0]),
        ("upstream_dependency_identities", EXPECTED_UPSTREAM_IDENTITIES),
        ("missing_evidence_specs", EXPECTED_MISSING_SPECS),
        ("ordering_statement_specs", EXPECTED_ORDERING_SPECS),
        ("missing_status", MISSING_STATUS),
        ("ordering_status", ORDERING_STATUS),
        ("non_ordering_gap_statements", NON_ORDERING_GAP_STATEMENTS),
        ("true_claims", TRUE_CLAIMS),
        ("false_claims", FALSE_CLAIMS),
        (
            "content_address_domain_prefix_bindings",
            (
                "domain_bytes_are_exactly_lowercase_hex_decoded_values_in_declared_order",
                CONTENT_ADDRESS_DOMAIN_PREFIX_BINDINGS,
            ),
        ),
        ("content_address_formulas", CONTENT_ADDRESS_FORMULAS),
        ("validation_order", VALIDATION_ORDER),
        ("global_preflight_before_hash_sentinel", GLOBAL_PREFLIGHT_SENTINEL),
        ("api_boundary", API_BOUNDARY),
        ("forbidden_operations", FORBIDDEN_OPERATIONS),
    )


def _independent_policy_id(schema_id: str) -> str:
    domain, prefix = DOMAINS_AND_PREFIXES["policy"]
    return prefix + _digest(domain, _independent_policy_preimage(schema_id))


def _content_id(kind: str, names: tuple[str, ...], values: tuple[object, ...]) -> str:
    domain, prefix = DOMAINS_AND_PREFIXES[kind]
    return prefix + _digest(domain, tuple(zip(names, values)))


def _sequence_root(kind: str, identifiers: tuple[str, ...]) -> str:
    domain, prefix = DOMAINS_AND_PREFIXES[kind]
    framed = bytearray(domain)
    framed.extend(len(identifiers).to_bytes(4, "big"))
    for identifier in identifiers:
        raw = identifier.encode("ascii")
        framed.extend(len(raw).to_bytes(2, "big"))
        framed.extend(raw)
    return prefix + hashlib.sha256(bytes(framed)).hexdigest()


def _pairs(value: object, names: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
    return tuple((name, getattr(value, name)) for name in names)


def _contract_preimage(
    contract: gap_v2.AdmissionGapInventoryContractV2,
) -> tuple[tuple[str, object], ...]:
    result: list[tuple[str, object]] = []
    for name in CONTRACT_FIELDS[:-2]:
        value = getattr(contract, name)
        if name == "upstream_dependency_identities":
            value = tuple(tuple(zip(IDENTITY_FIELDS, item)) for item in value)
        elif name == "missing_evidence_requirements":
            value = tuple(_pairs(item, MISSING_FIELDS) for item in value)
        elif name == "required_ordering_statements":
            value = tuple(_pairs(item, ORDERING_FIELDS) for item in value)
        result.append((name, value))
    return tuple(result)


def _validation_preimage(
    result: gap_v2.AdmissionGapInventoryValidationV2,
) -> tuple[tuple[str, object], ...]:
    preimage: list[tuple[str, object]] = []
    for name in VALIDATION_FIELDS:
        if name == "validation_id":
            continue
        value = getattr(result, name)
        if name == "upstream_dependency_identities":
            value = tuple(tuple(zip(IDENTITY_FIELDS, item)) for item in value)
        elif name == "missing_evidence_requirements":
            value = tuple(_pairs(item, MISSING_FIELDS) for item in value)
        elif name == "required_ordering_statements":
            value = tuple(_pairs(item, ORDERING_FIELDS) for item in value)
        preimage.append((name, value))
    return tuple(preimage)


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


def _replace_digest(identifier: str) -> str:
    return identifier[:-1] + ("0" if identifier[-1] != "0" else "1")


class _PrehashBoundaryReached(BaseException):
    """Must escape the validator's ordinary internal-error normalizer."""


def _assert_atomic_rejection(
    value: object,
    reason: gap_v2.AdmissionGapInventoryReasonV2,
) -> None:
    assert type(value) is gap_v2.AdmissionGapInventoryRejectionV2
    assert value.disposition is gap_v2.AdmissionGapInventoryDispositionV2.REJECTED
    assert value.reason is reason
    for name in (
        "schema_id",
        "policy_id",
        "validation_id",
        "contract_id",
        "contract_sha256",
        "upstream_dependency_identity_ids_root",
        "missing_evidence_requirement_ids_root",
        "required_ordering_statement_ids_root",
    ):
        assert getattr(value, name) is None
    assert value.upstream_dependency_identity_count == 0
    assert value.missing_evidence_requirement_count == 0
    assert value.required_ordering_statement_count == 0
    assert value.upstream_dependency_identities == ()
    assert value.missing_evidence_requirements == ()
    assert value.required_ordering_statements == ()
    assert value.partial_output_published is False
    assert all(getattr(value, name) is False for name in (*TRUE_CLAIMS, *FALSE_CLAIMS))


def _mutated_contract(case: str) -> gap_v2.AdmissionGapInventoryContractV2:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    missing = contract.missing_evidence_requirements
    ordering = contract.required_ordering_statements
    identities = contract.upstream_dependency_identities

    if case == "identity_value":
        changed = (*identities[:-1], (identities[-1][0], identities[-1][1], "0" * 64))
        return _unchecked_copy(contract, upstream_dependency_identities=changed)  # type: ignore[return-value]
    if case == "identity_reorder":
        return _unchecked_copy(
            contract,
            upstream_dependency_identities=(identities[1], identities[0], *identities[2:]),
        )  # type: ignore[return-value]
    if case == "identity_id_duplicate":
        return _unchecked_copy(
            contract,
            upstream_dependency_identity_ids=(
                *contract.upstream_dependency_identity_ids[:-1],
                contract.upstream_dependency_identity_ids[0],
            ),
        )  # type: ignore[return-value]
    if case == "missing_purpose":
        changed = _unchecked_copy(missing[-1], purpose="changed-but-still-well-formed")
        return _unchecked_copy(contract, missing_evidence_requirements=(*missing[:-1], changed))  # type: ignore[return-value]
    if case == "missing_reorder":
        return _unchecked_copy(
            contract,
            missing_evidence_requirements=(missing[1], missing[0], *missing[2:]),
        )  # type: ignore[return-value]
    if case == "missing_id_duplicate":
        changed = _unchecked_copy(missing[-1], requirement_id=missing[0].requirement_id)
        return _unchecked_copy(contract, missing_evidence_requirements=(*missing[:-1], changed))  # type: ignore[return-value]
    if case == "missing_id_splice":
        changed = _unchecked_copy(missing[-1], requirement_id=_replace_digest(missing[-1].requirement_id))
        return _unchecked_copy(contract, missing_evidence_requirements=(*missing[:-1], changed))  # type: ignore[return-value]
    if case == "ordering_statement":
        changed = _unchecked_copy(ordering[-1], statement="changed-but-still-well-formed")
        return _unchecked_copy(contract, required_ordering_statements=(*ordering[:-1], changed))  # type: ignore[return-value]
    if case == "ordering_reorder":
        return _unchecked_copy(
            contract,
            required_ordering_statements=(ordering[1], ordering[0], *ordering[2:]),
        )  # type: ignore[return-value]
    if case == "ordering_id_duplicate":
        changed = _unchecked_copy(ordering[-1], ordering_statement_id=ordering[0].ordering_statement_id)
        return _unchecked_copy(contract, required_ordering_statements=(*ordering[:-1], changed))  # type: ignore[return-value]
    if case == "ordering_id_splice":
        changed = _unchecked_copy(ordering[-1], ordering_statement_id=_replace_digest(ordering[-1].ordering_statement_id))
        return _unchecked_copy(contract, required_ordering_statements=(*ordering[:-1], changed))  # type: ignore[return-value]
    if case == "identity_root":
        return _unchecked_copy(
            contract,
            upstream_dependency_identity_ids_root=_replace_digest(contract.upstream_dependency_identity_ids_root),
        )  # type: ignore[return-value]
    if case == "missing_root":
        return _unchecked_copy(
            contract,
            missing_evidence_requirement_ids_root=_replace_digest(contract.missing_evidence_requirement_ids_root),
        )  # type: ignore[return-value]
    if case == "ordering_root":
        return _unchecked_copy(
            contract,
            required_ordering_statement_ids_root=_replace_digest(contract.required_ordering_statement_ids_root),
        )  # type: ignore[return-value]
    if case == "root_cross_splice":
        return _unchecked_copy(
            contract,
            missing_evidence_requirement_ids_root=contract.required_ordering_statement_ids_root,
        )  # type: ignore[return-value]
    if case == "count":
        return _unchecked_copy(contract, missing_evidence_requirement_count=17)  # type: ignore[return-value]
    if case == "true_claim":
        return _unchecked_copy(contract, gap_inventory_frozen=False)  # type: ignore[return-value]
    if case == "false_claim":
        return _unchecked_copy(contract, admission_ready=True)  # type: ignore[return-value]
    if case == "contract_sha":
        return _unchecked_copy(contract, contract_sha256=_replace_digest(contract.contract_sha256))  # type: ignore[return-value]
    if case == "contract_id":
        return _unchecked_copy(contract, contract_id=_replace_digest(contract.contract_id))  # type: ignore[return-value]
    raise AssertionError(case)


def _malformed_prehash_contract(case: str) -> gap_v2.AdmissionGapInventoryContractV2:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    missing = contract.missing_evidence_requirements
    ordering = contract.required_ordering_statements
    if case == "top_level_list":
        return _unchecked_copy(contract, upstream_dependency_identity_ids=list(contract.upstream_dependency_identity_ids))  # type: ignore[return-value]
    if case == "bool_as_count":
        return _unchecked_copy(contract, required_ordering_statement_count=True)  # type: ignore[return-value]
    if case == "uppercase_sha":
        return _unchecked_copy(contract, contract_sha256="A" * 64)  # type: ignore[return-value]
    if case == "oversize_final_text":
        changed = _unchecked_copy(missing[-1], purpose="x" * 4097)
        return _unchecked_copy(contract, missing_evidence_requirements=(*missing[:-1], changed))  # type: ignore[return-value]
    if case == "invalid_utf8_final_text":
        changed = _unchecked_copy(missing[-1], purpose="\ud800")
        return _unchecked_copy(contract, missing_evidence_requirements=(*missing[:-1], changed))  # type: ignore[return-value]
    if case == "non_bool_final_nested":
        changed = _unchecked_copy(ordering[-1], timing_verifier_implemented=0)
        return _unchecked_copy(contract, required_ordering_statements=(*ordering[:-1], changed))  # type: ignore[return-value]
    if case == "duplicate_final_nested_id":
        changed = _unchecked_copy(missing[-1], requirement_id=missing[0].requirement_id)
        return _unchecked_copy(contract, missing_evidence_requirements=(*missing[:-1], changed))  # type: ignore[return-value]
    if case == "wrong_nested_type":
        return _unchecked_copy(contract, required_ordering_statements=(*ordering[:-1], object()))  # type: ignore[return-value]
    raise AssertionError(case)


def _ast_imports_and_calls(source: str) -> tuple[frozenset[str], frozenset[str]]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imported.add(item.name)
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = f"{module}.{item.name}".strip(".")

    def qualified(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Call):
            return qualified(node.func)
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            return node.attr if parent is None else f"{parent}.{node.attr}"
        return None

    calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (qualified(node.func),)
        if name is not None
    }
    return frozenset(imported), frozenset(calls)


def test_public_surface_signatures_and_no_evidence_input_channel() -> None:
    assert tuple(gap_v2.__all__) == EXPECTED_PUBLIC_SURFACE
    assert inspect.signature(
        gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2
    ).parameters == {}
    signature = inspect.signature(
        gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2
    )
    assert tuple(signature.parameters) == ("contract",)
    assert signature.parameters["contract"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
            gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
        )


def test_exact_dataclass_field_manifests_and_enum_values() -> None:
    manifests = (
        (gap_v2.MissingEvidenceRequirementV2, MISSING_FIELDS),
        (gap_v2.RequiredOrderingStatementV2, ORDERING_FIELDS),
        (gap_v2.AdmissionGapInventoryContractV2, CONTRACT_FIELDS),
        (gap_v2.AdmissionGapInventoryValidationV2, VALIDATION_FIELDS),
        (gap_v2.AdmissionGapInventoryRejectionV2, REJECTION_FIELDS),
    )
    for cls, expected in manifests:
        assert tuple(item.name for item in fields(cls)) == expected
    assert tuple(item.value for item in gap_v2.AdmissionGapInventoryDispositionV2) == EXPECTED_DISPOSITIONS
    assert tuple(item.value for item in gap_v2.AdmissionGapInventoryReasonV2) == EXPECTED_REASONS
    assert len(TRUE_CLAIMS) == 6
    assert len(FALSE_CLAIMS) == 80
    assert not set(TRUE_CLAIMS).intersection(FALSE_CLAIMS)


def test_exact_version_schema_policy_and_non_authoritative_claim_level() -> None:
    assert gap_v2.ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION == EXPECTED_VERSION
    assert gap_v2.ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL == EXPECTED_CLAIM_LEVEL
    for kind, (domain, prefix) in DOMAINS_AND_PREFIXES.items():
        assert domain.endswith(b"\x00"), kind
        assert prefix.endswith("_"), kind
    assert len({domain for domain, _ in DOMAINS_AND_PREFIXES.values()}) == 10
    assert len({prefix for _, prefix in DOMAINS_AND_PREFIXES.values()}) == 10
    assert len(CONTENT_ADDRESS_DOMAIN_PREFIX_BINDINGS) == 10

    independently_rebuilt_schema_id = _independent_schema_id()
    independently_rebuilt_policy_id = _independent_policy_id(
        independently_rebuilt_schema_id
    )
    assert independently_rebuilt_schema_id == EXPECTED_SCHEMA_ID
    assert independently_rebuilt_policy_id == EXPECTED_POLICY_ID
    assert gap_v2.ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_SCHEMA_ID == independently_rebuilt_schema_id
    assert gap_v2.ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_POLICY_ID == independently_rebuilt_policy_id


def test_private_output_constructors_and_immutability() -> None:
    for cls in (
        gap_v2.MissingEvidenceRequirementV2,
        gap_v2.RequiredOrderingStatementV2,
        gap_v2.AdmissionGapInventoryContractV2,
        gap_v2.AdmissionGapInventoryValidationV2,
        gap_v2.AdmissionGapInventoryRejectionV2,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            cls()  # type: ignore[call-arg]
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    with pytest.raises(FrozenInstanceError):
        contract.admission_ready = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.missing_evidence_requirements[0].verified = True  # type: ignore[misc]


def test_exact_upstream_identity_and_gap_role_catalogue() -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    assert contract.upstream_dependency_identities == EXPECTED_UPSTREAM_IDENTITIES
    assert contract.upstream_dependency_identity_count == 5
    assert contract.missing_evidence_requirement_count == 18
    assert tuple(item.ordinal for item in contract.missing_evidence_requirements) == tuple(range(18))
    assert tuple(item.evidence_name for item in contract.missing_evidence_requirements) == EXPECTED_GAP_ROLES
    assert tuple(
        (item.ordinal, item.evidence_name, item.purpose)
        for item in contract.missing_evidence_requirements
    ) == EXPECTED_MISSING_SPECS
    assert len({item.evidence_name for item in contract.missing_evidence_requirements}) == 18
    assert all(item.status == MISSING_STATUS for item in contract.missing_evidence_requirements)
    assert all("unresolved_schema_gaps=" in item.purpose for item in contract.missing_evidence_requirements)
    assert all(item.missing is True for item in contract.missing_evidence_requirements)
    assert all(item.supplied_by_this_contract is False for item in contract.missing_evidence_requirements)
    assert all(item.verified is False for item in contract.missing_evidence_requirements)
    assert all(item.verifier_implemented is False for item in contract.missing_evidence_requirements)


def test_exact_declarative_ordering_catalogue_is_not_timeline_evidence() -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    statements = contract.required_ordering_statements
    assert contract.required_ordering_statement_count == 16
    assert tuple(item.ordinal for item in statements) == tuple(range(16))
    assert tuple((item.predecessor_stage, item.successor_stage) for item in statements) == tuple(
        (predecessor, successor)
        for _name, predecessor, successor in EXPECTED_ORDERING_TRIPLES
    )
    assert tuple(
        (item.ordinal, item.predecessor_stage, item.successor_stage, item.statement)
        for item in statements
    ) == EXPECTED_ORDERING_SPECS
    assert len({(item.predecessor_stage, item.successor_stage) for item in statements}) == 16
    assert all(item.status == ORDERING_STATUS for item in statements)
    assert all(item.event_schema_supplied is False for item in statements)
    assert all(item.signature_schema_supplied is False for item in statements)
    assert all(item.ledger_schema_supplied is False for item in statements)
    assert all(item.timing_verifier_implemented is False for item in statements)


def test_independent_descriptor_and_length_framed_root_oracles() -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    identity_ids = tuple(
        _content_id("identity", IDENTITY_FIELDS, identity)
        for identity in EXPECTED_UPSTREAM_IDENTITIES
    )
    missing_ids = tuple(
        _content_id(
            "missing",
            MISSING_FIELDS[:-1],
            tuple(getattr(item, name) for name in MISSING_FIELDS[:-1]),
        )
        for item in contract.missing_evidence_requirements
    )
    ordering_ids = tuple(
        _content_id(
            "ordering",
            ORDERING_FIELDS[:-1],
            tuple(getattr(item, name) for name in ORDERING_FIELDS[:-1]),
        )
        for item in contract.required_ordering_statements
    )
    assert contract.upstream_dependency_identity_ids == identity_ids
    assert tuple(item.requirement_id for item in contract.missing_evidence_requirements) == missing_ids
    assert tuple(item.ordering_statement_id for item in contract.required_ordering_statements) == ordering_ids
    assert contract.upstream_dependency_identity_ids_root == _sequence_root("identity_root", identity_ids)
    assert contract.missing_evidence_requirement_ids_root == _sequence_root("missing_root", missing_ids)
    assert contract.required_ordering_statement_ids_root == _sequence_root("ordering_root", ordering_ids)
    assert contract.upstream_dependency_identity_ids_root.endswith(
        "0c7792c650de0f3d08f08a798786a7c5fc880ee3fa677275c241df50767f1eba"
    )
    assert contract.missing_evidence_requirement_ids_root.endswith(
        "5e06ce843e1ec97cd991126a489c239b0833dc91c80191b417a5bf407f0689fa"
    )
    assert contract.required_ordering_statement_ids_root.endswith(
        "a65b5987a1b0a2c354c385a977fd8d7869e16caaf5786fcd0938236ae064c5d4"
    )
    assert _sequence_root("missing_root", tuple(reversed(missing_ids))) != contract.missing_evidence_requirement_ids_root


def test_independent_contract_and_validation_content_address_oracles() -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    contract_sha = _digest(DOMAINS_AND_PREFIXES["contract"][0], _contract_preimage(contract))
    assert contract_sha == EXPECTED_CONTRACT_SHA
    assert contract.contract_sha256 == contract_sha
    assert contract.contract_id == DOMAINS_AND_PREFIXES["contract"][1] + contract_sha
    assert contract.contract_id == EXPECTED_CONTRACT_ID

    result = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=contract
    )
    assert type(result) is gap_v2.AdmissionGapInventoryValidationV2
    validation_sha = _digest(
        DOMAINS_AND_PREFIXES["validation"][0],
        _validation_preimage(result),
    )
    assert result.validation_id == DOMAINS_AND_PREFIXES["validation"][1] + validation_sha
    assert result.validation_id == EXPECTED_VALIDATION_ID


def test_success_is_catalogue_validation_not_admission_or_evidence() -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    result = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=contract
    )
    assert type(result) is gap_v2.AdmissionGapInventoryValidationV2
    assert result.disposition is gap_v2.AdmissionGapInventoryDispositionV2.GAP_INVENTORY_FROZEN_NOT_ADMITTED
    assert result.reason is gap_v2.AdmissionGapInventoryReasonV2.MISSING_EVIDENCE_AND_ORDERING_GAPS_INVENTORIED_NOT_VERIFIED
    for value in (contract, result):
        assert all(getattr(value, name) is True for name in TRUE_CLAIMS)
        assert all(getattr(value, name) is False for name in FALSE_CLAIMS)
        assert value.admission_ready is False
        assert value.execution_authorized is False
        assert value.evidence_supplied is False
        assert value.evidence_verified is False
        assert value.recognizer_executed is False
        assert value.scoring_performed is False
        assert value.effect_evidence is False
        assert value.c1_exit_evidence is False


def test_getter_and_validator_return_deep_fresh_graphs() -> None:
    first = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    second = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    assert first == second
    assert first is not second
    assert first.missing_evidence_requirements is not second.missing_evidence_requirements
    assert first.required_ordering_statements is not second.required_ordering_statements
    assert all(a is not b for a, b in zip(first.missing_evidence_requirements, second.missing_evidence_requirements))
    assert all(a is not b for a, b in zip(first.required_ordering_statements, second.required_ordering_statements))

    one = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(contract=first)
    two = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(contract=first)
    assert type(one) is type(two) is gap_v2.AdmissionGapInventoryValidationV2
    assert one == two
    assert one is not two
    assert one.missing_evidence_requirements is not first.missing_evidence_requirements
    assert one.required_ordering_statements is not first.required_ordering_statements
    assert all(a is not b for a, b in zip(one.missing_evidence_requirements, first.missing_evidence_requirements))
    assert all(a is not b for a, b in zip(one.required_ordering_statements, first.required_ordering_statements))


@pytest.mark.parametrize("wrong", [None, object(), (), {}, 0, False, "contract"])
def test_wrong_top_level_types_reject_atomically(wrong: object) -> None:
    value = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=wrong
    )
    _assert_atomic_rejection(value, gap_v2.AdmissionGapInventoryReasonV2.WRONG_INPUT_TYPE)


def test_subclass_is_not_accepted_as_exact_contract_type() -> None:
    class ContractSubclass(gap_v2.AdmissionGapInventoryContractV2):
        pass

    impostor = object.__new__(ContractSubclass)
    value = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=impostor
    )
    _assert_atomic_rejection(value, gap_v2.AdmissionGapInventoryReasonV2.WRONG_INPUT_TYPE)


@pytest.mark.parametrize("field", ["version", "schema_id", "policy_id", "claim_level"])
def test_cross_version_identity_rejects_atomically(field: str) -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    if field in {"schema_id", "policy_id"}:
        changed = _replace_digest(getattr(contract, field))
    else:
        changed = getattr(contract, field) + "-cross-version"
    value = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=_unchecked_copy(contract, **{field: changed})
    )
    _assert_atomic_rejection(value, gap_v2.AdmissionGapInventoryReasonV2.CROSS_VERSION_INPUT)


@pytest.mark.parametrize(
    "case",
    [
        "identity_value",
        "identity_reorder",
        "identity_id_duplicate",
        "missing_purpose",
        "missing_reorder",
        "missing_id_duplicate",
        "missing_id_splice",
        "ordering_statement",
        "ordering_reorder",
        "ordering_id_duplicate",
        "ordering_id_splice",
        "identity_root",
        "missing_root",
        "ordering_root",
        "root_cross_splice",
        "count",
        "true_claim",
        "false_claim",
        "contract_sha",
        "contract_id",
    ],
)
def test_mutation_reorder_duplicate_id_root_and_claim_splices_reject(case: str) -> None:
    value = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=_mutated_contract(case)
    )
    _assert_atomic_rejection(value, gap_v2.AdmissionGapInventoryReasonV2.CONTRACT_INVALID)


@pytest.mark.parametrize(
    "case",
    [
        "top_level_list",
        "bool_as_count",
        "uppercase_sha",
        "oversize_final_text",
        "invalid_utf8_final_text",
        "non_bool_final_nested",
        "duplicate_final_nested_id",
        "wrong_nested_type",
    ],
)
def test_global_type_format_cap_and_duplicate_preflight_precedes_all_hashes(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = _malformed_prehash_contract(case)

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached(
            f"{case} reached a content hash before global closure"
        )

    monkeypatch.setattr(gap_v2, "_sha_v2", forbidden_hash)
    monkeypatch.setattr(gap_v2, "canonical_json", forbidden_hash)
    monkeypatch.setattr(gap_v2.hashlib, "sha256", forbidden_hash)
    value = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=malformed
    )
    _assert_atomic_rejection(value, gap_v2.AdmissionGapInventoryReasonV2.CONTRACT_INVALID)


def test_prehash_boundary_sentinel_escapes_internal_error_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()

    def reached_hash(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("expected sentinel escape")

    assert issubclass(_PrehashBoundaryReached, BaseException)
    assert not issubclass(_PrehashBoundaryReached, Exception)
    monkeypatch.setattr(gap_v2, "_sha_v2", reached_hash)
    monkeypatch.setattr(gap_v2, "canonical_json", reached_hash)
    monkeypatch.setattr(gap_v2.hashlib, "sha256", reached_hash)
    with pytest.raises(_PrehashBoundaryReached, match="expected sentinel escape"):
        gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
            contract=contract
        )


def test_validator_uses_closed_snapshot_under_hash_time_caller_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()
    caller_missing = tuple(_unchecked_copy(item) for item in baseline.missing_evidence_requirements)
    caller_ordering = tuple(_unchecked_copy(item) for item in baseline.required_ordering_statements)
    caller = _unchecked_copy(
        baseline,
        upstream_dependency_identities=tuple(tuple(item) for item in baseline.upstream_dependency_identities),
        upstream_dependency_identity_ids=tuple(baseline.upstream_dependency_identity_ids),
        missing_evidence_requirements=caller_missing,
        required_ordering_statements=caller_ordering,
    )
    original_canonical_json: Callable[[object], str] = gap_v2.canonical_json
    calls = 0

    def mutate_caller_on_first_hash(value: object) -> str:
        nonlocal calls
        if calls == 0:
            object.__setattr__(caller, "missing_evidence_requirements", ())
            object.__setattr__(caller, "required_ordering_statements", ())
            object.__setattr__(caller, "contract_sha256", "0" * 64)
            object.__setattr__(caller_missing[-1], "purpose", "mutated-after-snapshot")
            object.__setattr__(caller_ordering[-1], "statement", "mutated-after-snapshot")
        calls += 1
        return original_canonical_json(value)

    monkeypatch.setattr(gap_v2, "canonical_json", mutate_caller_on_first_hash)
    result = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=caller
    )
    assert calls > 0
    assert type(result) is gap_v2.AdmissionGapInventoryValidationV2
    assert result.contract_id == EXPECTED_CONTRACT_ID
    assert result.validation_id == EXPECTED_VALIDATION_ID
    assert len(result.missing_evidence_requirements) == 18
    assert len(result.required_ordering_statements) == 16
    assert caller.missing_evidence_requirements == ()
    assert caller.required_ordering_statements == ()


def test_internal_exception_is_atomic_all_false_no_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = gap_v2.frozen_actual_unsealed_960_replay_admission_gap_inventory_v2()

    def explode(_value: object) -> None:
        raise RuntimeError("synthetic internal failure")

    monkeypatch.setattr(gap_v2, "_verify_contract_v2", explode)
    value = gap_v2.validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
        contract=contract
    )
    _assert_atomic_rejection(value, gap_v2.AdmissionGapInventoryReasonV2.INTERNAL_ERROR)


def test_source_ast_has_no_operational_import_or_call_boundary() -> None:
    source = Path(gap_v2.__file__).read_text(encoding="utf-8")
    imported, calls = _ast_imports_and_calls(source)
    forbidden_import_roots = {
        "asyncio",
        "concurrent",
        "datetime",
        "docker",
        "http",
        "httpx",
        "multiprocessing",
        "numpy",
        "os",
        "pathlib",
        "queue",
        "random",
        "requests",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "tempfile",
        "time",
        "urllib",
    }
    forbidden_project_fragments = (
        "runner",
        "decoder",
        "scoring",
        "evaluator",
        "ledger",
        "runtime",
        "signature",
        "docker",
    )
    for name in imported:
        assert name.split(".", 1)[0] not in forbidden_import_roots, name
        assert not any(fragment in name.casefold() for fragment in forbidden_project_fragments), name
    forbidden_call_roots = forbidden_import_roots | {"open"}
    forbidden_call_terminals = {
        "exec",
        "eval",
        "open",
        "read_bytes",
        "read_text",
        "write_bytes",
        "write_text",
        "urlopen",
        "uuid4",
        "sleep",
        "run",
        "popen",
        "system",
    }
    for name in calls:
        assert name.split(".", 1)[0] not in forbidden_call_roots, name
        assert name.rsplit(".", 1)[-1].casefold() not in forbidden_call_terminals, name


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("import subprocess as sp\nsp.run([])", "subprocess.run"),
        ("from urllib import request as r\nr.urlopen('x')", "urllib.request.urlopen"),
        ("from pathlib import Path\nPath('x').read_bytes()", "pathlib.Path.read_bytes"),
    ),
)
def test_ast_guard_resolves_aliases_and_dotted_operational_calls(
    source: str,
    expected: str,
) -> None:
    _imports, calls = _ast_imports_and_calls(source)
    assert expected in calls
