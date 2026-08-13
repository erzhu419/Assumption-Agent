"""Non-authoritative gap inventory for a future actual-unsealed-960 replay.

This module inventories missing evidence and declarative ordering requirements.
It accepts no evidence and supplies no operational, cryptographic, timeline,
runtime, scoring, gate, effect, or C1 verifier.  A successful catalogue
validation is therefore explicitly not an admission decision.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
from typing import Final

from .hashing import canonical_json


ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-replay-admission-gap-inventory/2"
)
ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_ACTUAL_REPLAY_ADMISSION_GAP_INVENTORY_ONLY"
)


class AdmissionGapInventoryDispositionV2(str, Enum):
    GAP_INVENTORY_FROZEN_NOT_ADMITTED = "GAP_INVENTORY_FROZEN_NOT_ADMITTED"
    REJECTED = "REJECTED"


class AdmissionGapInventoryReasonV2(str, Enum):
    MISSING_EVIDENCE_AND_ORDERING_GAPS_INVENTORIED_NOT_VERIFIED = (
        "MISSING_EVIDENCE_AND_ORDERING_GAPS_INVENTORIED_NOT_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True, init=False)
class MissingEvidenceRequirementV2:
    ordinal: int
    evidence_name: str
    purpose: str
    status: str
    missing: bool
    supplied_by_this_contract: bool
    verified: bool
    verifier_implemented: bool
    requirement_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("gap-inventory requirements are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class RequiredOrderingStatementV2:
    ordinal: int
    predecessor_stage: str
    successor_stage: str
    statement: str
    status: str
    event_schema_supplied: bool
    signature_schema_supplied: bool
    ledger_schema_supplied: bool
    timing_verifier_implemented: bool
    ordering_statement_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("gap-inventory ordering statements are privately issued")


_TRUE_CLAIMS_V2: Final = (
    "gap_inventory_frozen",
    "required_ordering_statements_frozen",
    "upstream_dependency_identities_frozen",
    "no_evidence_input_schema_accepted",
    "content_addressed_gap_catalog_validated",
    "atomic_catalog_validation_verified",
)

_FALSE_CLAIMS_V2: Final = (
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


@dataclass(frozen=True, slots=True, init=False)
class AdmissionGapInventoryContractV2:
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    upstream_dependency_identities: tuple[tuple[str, str, str], ...]
    upstream_dependency_identity_ids: tuple[str, ...]
    upstream_dependency_identity_ids_root: str
    missing_evidence_requirements: tuple[MissingEvidenceRequirementV2, ...]
    missing_evidence_requirement_ids_root: str
    required_ordering_statements: tuple[RequiredOrderingStatementV2, ...]
    required_ordering_statement_ids_root: str
    upstream_dependency_identity_count: int
    missing_evidence_requirement_count: int
    required_ordering_statement_count: int
    gap_inventory_frozen: bool
    required_ordering_statements_frozen: bool
    upstream_dependency_identities_frozen: bool
    no_evidence_input_schema_accepted: bool
    content_addressed_gap_catalog_validated: bool
    atomic_catalog_validation_verified: bool
    admission_evidence_contract_complete: bool
    cryptographic_record_schemas_complete: bool
    signature_payload_profiles_complete: bool
    pinned_signer_registry_schema_complete: bool
    attempt_registry_schema_complete: bool
    rerun_policy_executable: bool
    timeline_event_schemas_complete: bool
    timeline_verifier_implemented: bool
    evidence_role_lifecycle_fully_bound: bool
    actual_evidence_inputs_accepted: bool
    verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    one_shot_policy_enforced: bool
    signature_coverage_verified: bool
    evidence_chain_verified: bool
    attempt_terminal_state_verified: bool
    pre_reveal_commitment_timing_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    partition_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_capacity_evidence: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    preservation_prediction_commitment_verified: bool
    evidence_supplied: bool
    evidence_verified: bool
    custodian_identity_verified: bool
    signer_chain_verified: bool
    signatures_verified: bool
    pinned_signer_key_registry_verified: bool
    durable_attempt_ledger_verified: bool
    attempt_policy_enforced: bool
    retry_authorization_verified: bool
    timeline_observed: bool
    timeline_order_verified: bool
    package_commitment_before_run_verified: bool
    prediction_before_reveal_verified: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    allocation_authority_verified: bool
    evaluator_input_authority_verified: bool
    execution_freeze_authority_verified: bool
    runtime_attestation_authority_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    canonical_archive_replay_verified: bool
    decoder_transcript_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
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
    contract_sha256: str
    contract_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("gap-inventory contracts are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class AdmissionGapInventoryValidationV2:
    disposition: AdmissionGapInventoryDispositionV2
    reason: AdmissionGapInventoryReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    validation_id: str
    contract_id: str
    contract_sha256: str
    upstream_dependency_identity_ids_root: str
    missing_evidence_requirement_ids_root: str
    required_ordering_statement_ids_root: str
    upstream_dependency_identity_count: int
    missing_evidence_requirement_count: int
    required_ordering_statement_count: int
    gap_inventory_frozen: bool
    required_ordering_statements_frozen: bool
    upstream_dependency_identities_frozen: bool
    no_evidence_input_schema_accepted: bool
    content_addressed_gap_catalog_validated: bool
    atomic_catalog_validation_verified: bool
    admission_evidence_contract_complete: bool
    cryptographic_record_schemas_complete: bool
    signature_payload_profiles_complete: bool
    pinned_signer_registry_schema_complete: bool
    attempt_registry_schema_complete: bool
    rerun_policy_executable: bool
    timeline_event_schemas_complete: bool
    timeline_verifier_implemented: bool
    evidence_role_lifecycle_fully_bound: bool
    actual_evidence_inputs_accepted: bool
    verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    one_shot_policy_enforced: bool
    signature_coverage_verified: bool
    evidence_chain_verified: bool
    attempt_terminal_state_verified: bool
    pre_reveal_commitment_timing_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    partition_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_capacity_evidence: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    preservation_prediction_commitment_verified: bool
    evidence_supplied: bool
    evidence_verified: bool
    custodian_identity_verified: bool
    signer_chain_verified: bool
    signatures_verified: bool
    pinned_signer_key_registry_verified: bool
    durable_attempt_ledger_verified: bool
    attempt_policy_enforced: bool
    retry_authorization_verified: bool
    timeline_observed: bool
    timeline_order_verified: bool
    package_commitment_before_run_verified: bool
    prediction_before_reveal_verified: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    allocation_authority_verified: bool
    evaluator_input_authority_verified: bool
    execution_freeze_authority_verified: bool
    runtime_attestation_authority_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    canonical_archive_replay_verified: bool
    decoder_transcript_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
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
    upstream_dependency_identities: tuple[tuple[str, str, str], ...]
    missing_evidence_requirements: tuple[MissingEvidenceRequirementV2, ...]
    required_ordering_statements: tuple[RequiredOrderingStatementV2, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("gap-inventory validations are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class AdmissionGapInventoryRejectionV2:
    disposition: AdmissionGapInventoryDispositionV2
    reason: AdmissionGapInventoryReasonV2
    version: str
    claim_level: str
    schema_id: None
    policy_id: None
    validation_id: None
    contract_id: None
    contract_sha256: None
    upstream_dependency_identity_ids_root: None
    missing_evidence_requirement_ids_root: None
    required_ordering_statement_ids_root: None
    upstream_dependency_identity_count: int
    missing_evidence_requirement_count: int
    required_ordering_statement_count: int
    upstream_dependency_identities: tuple[()]
    missing_evidence_requirements: tuple[()]
    required_ordering_statements: tuple[()]
    partial_output_published: bool
    gap_inventory_frozen: bool
    required_ordering_statements_frozen: bool
    upstream_dependency_identities_frozen: bool
    no_evidence_input_schema_accepted: bool
    content_addressed_gap_catalog_validated: bool
    atomic_catalog_validation_verified: bool
    admission_evidence_contract_complete: bool
    cryptographic_record_schemas_complete: bool
    signature_payload_profiles_complete: bool
    pinned_signer_registry_schema_complete: bool
    attempt_registry_schema_complete: bool
    rerun_policy_executable: bool
    timeline_event_schemas_complete: bool
    timeline_verifier_implemented: bool
    evidence_role_lifecycle_fully_bound: bool
    actual_evidence_inputs_accepted: bool
    verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    one_shot_policy_enforced: bool
    signature_coverage_verified: bool
    evidence_chain_verified: bool
    attempt_terminal_state_verified: bool
    pre_reveal_commitment_timing_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    answer_commitment_opening_verified: bool
    gate_input_commitment_opening_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    partition_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_capacity_evidence: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    preservation_prediction_commitment_verified: bool
    evidence_supplied: bool
    evidence_verified: bool
    custodian_identity_verified: bool
    signer_chain_verified: bool
    signatures_verified: bool
    pinned_signer_key_registry_verified: bool
    durable_attempt_ledger_verified: bool
    attempt_policy_enforced: bool
    retry_authorization_verified: bool
    timeline_observed: bool
    timeline_order_verified: bool
    package_commitment_before_run_verified: bool
    prediction_before_reveal_verified: bool
    answer_manifest_authority_verified: bool
    gate_input_manifest_authority_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    allocation_authority_verified: bool
    evaluator_input_authority_verified: bool
    execution_freeze_authority_verified: bool
    runtime_attestation_authority_verified: bool
    raw_input_archive_replayed: bool
    raw_prediction_archive_replayed: bool
    canonical_archive_replay_verified: bool
    decoder_transcript_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    metric_results_materialized: bool
    scored_rows_materialized: bool
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

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("gap-inventory rejections are privately issued")


_UPSTREAM_DEPENDENCY_IDENTITY_FIELDS_V2: Final = (
    "dependency_name",
    "identity_kind",
    "identity_value",
)
_MISSING_EVIDENCE_FIELDS_V2: Final = (
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
_ORDERING_STATEMENT_FIELDS_V2: Final = (
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
_CONTRACT_FIELDS_V2: Final = (
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
    *_TRUE_CLAIMS_V2,
    *_FALSE_CLAIMS_V2,
    "contract_sha256",
    "contract_id",
)
_VALIDATION_FIELDS_V2: Final = (
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
    *_TRUE_CLAIMS_V2,
    *_FALSE_CLAIMS_V2,
    "upstream_dependency_identities",
    "missing_evidence_requirements",
    "required_ordering_statements",
)
_REJECTION_FIELDS_V2: Final = (
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
    *_TRUE_CLAIMS_V2,
    *_FALSE_CLAIMS_V2,
)


_MISSING_STATUS_V2: Final = "MISSING_UNSUPPLIED_UNVERIFIED_NO_VERIFIER"
_ORDERING_STATUS_V2: Final = "DECLARATIVE_ONLY_UNVERIFIED_NO_VERIFIER"

_UPSTREAM_DEPENDENCY_IDENTITIES_V2: Final = (
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "version",
        "hegel-machine-phase2b-actual-unsealed-960-replay-input-contract/2",
    ),
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "schema_id",
        "phase2b_actual_unsealed_960_replay_input_contract_schema_v2_a4f61ddfb07643e23ac404616062127e2ae6ca02f13b29c265062d6a1f660f4a",
    ),
    (
        "actual_unsealed_960_replay_input_contract_v2",
        "policy_id",
        "phase2b_actual_unsealed_960_replay_input_contract_policy_v2_a12ca51dd6f17f29a28a7229f4108c32f438d4775a367a7ad3e5a6275557b531",
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

_MISSING_EVIDENCE_SPECS_V2: Final = (
    (0, "durable_signed_custodian_ledger_and_signer_chain", "authenticate_one_shot_append_only_custody; unresolved_schema_gaps=authoritative_custodian_identity_pinned_keys_detached_signatures_append_only_ledger_and_external_trust_anchor"),
    (1, "prediction_reveal_scoring_chronology", "prove_package_commitment_precedes_run_and_predictions_precede_reveal; unresolved_schema_gaps=signed_event_records_chain_root_external_timestamps_and_ordering_verifier"),
    (2, "raw_input_and_prediction_archive_cas", "bind_authoritative_input_and_output_bytes; unresolved_schema_gaps=canonical_archive_bytes_content_addresses_authorities_and_membership_verifier"),
    (3, "canonical_archive_replay_and_decoder_transcript", "prove_public_canonical_input_and_prediction_replay; unresolved_schema_gaps=decoder_version_transcript_format_replay_rules_and_verifier"),
    (4, "source_image_sbom_config_runtime_execution_attestations", "authenticate_frozen_executable_environment; unresolved_schema_gaps=authoritative_image_SBOM_configuration_OCI_spec_runtime_attestation_and_signature_schemas"),
    (5, "actual_recognizer_execution_capacity_and_output_attestation", "prove_recognizer_execution_capacity_and_output_origin; unresolved_schema_gaps=run_identity_capacity_isolation_exit_output_origin_and_attestation_verifier"),
    (6, "allocation_source_batch_partition_answer_authorities", "authenticate_membership_and_evaluator_inputs; unresolved_schema_gaps=allocation_source_batch_partition_answer_gate_authority_records_and_resolvers"),
    (7, "independent_720_240_generation_and_source_disjointness_receipt", "prove_post_validation_generation_source_public_disjointness_and_latent_case_independence; unresolved_schema_gaps=generator_custody_source_projection_disjointness_receipt_and_independent_audit_verifier"),
    (8, "audit_archive_formal_uuid_and_covert_receipts", "bind_formal_audit_outputs; unresolved_schema_gaps=audit_archive_UUID_covert_receipt_content_schemas_authorities_and_verifiers"),
    (9, "three_baseline_prediction_outputs", "support_embedding_nearest_prototype_frozen_llm_semantic_only_flat_learned_typed_comparisons; unresolved_schema_gaps=baseline_registration_implementation_prediction_archive_and_comparison_report_schemas"),
    (10, "durable_fail_closed_attempt_event_denominator", "define_fail_closed_rate_over_all_attempt_events; unresolved_schema_gaps=durable_attempt_open_terminal_registry_denominator_finalization_and_no_cherry_pick_verifier"),
    (11, "preservation_496_legal_76_invalid_pair_results", "bind_original_variant_links_predictions_and_evaluator_results_for_496_legal_and_76_invalid_pairs; unresolved_schema_gaps=pair_manifest_pre_reveal_predictions_post_reveal_results_and_consistency_verifier"),
    (12, "semantic_conflict_240_descriptor_audit_package", "support_separately_reported_challenge_audit_or_scoring_excluded_from_main_thresholds_and_tuning; unresolved_schema_gaps=descriptor_package_authority_exclusion_receipt_and_audit_verifier"),
    (13, "scale_regret_oracle_loss_and_normalizer", "define_normalized_per_case_regret_point_max_0_05_and_bootstrap_upper_bound_0_08; unresolved_schema_gaps=oracle_loss_normalizer_per_case_regret_report_and_verifier"),
    (14, "clustered_bootstrap_inputs_and_statistic", "define_10000_replicate_paired_latent_base_case_cluster_bootstrap_master_seed_411876909552964556_uint32_2611585425_original_and_preservation_variants_one_sided_95_percentile; unresolved_schema_gaps=cluster_inputs_statistic_replicate_report_and_recomputation_verifier"),
    (15, "metric_wilson_overall_and_slice_gate_report", "materialize_metric_counts_frozen_thresholds_one_sided_95_percent_wilson_and_gate_results; unresolved_schema_gaps=scored_rows_metric_counts_Wilson_bounds_gate_report_and_independent_verifier"),
    (16, "attempt_rerun_permanent_record", "enforce_attempt_policy_and_permanent_attempt_history; unresolved_schema_gaps=reason_action_authorization_retry_linkage_permanent_attempt_record_and_policy_verifier"),
    (17, "formal_c1_consumed_sealed_report", "prove_all_overall_family_scale_gates_and_consumed_ledger_state; unresolved_schema_gaps=post_terminal_denominator_score_gate_consumption_C1_report_and_authority_verifier"),
)

_ORDERING_TRIPLES_V2: Final = (
    ("preregister_before_input", "attempt_preregistered", "raw_input_committed"),
    ("input_before_package_freeze", "raw_input_committed", "answer_gate_generation_disjointness_commitments_frozen"),
    ("package_freeze_before_run", "answer_gate_generation_disjointness_commitments_frozen", "recognizer_run_started"),
    ("start_before_finish", "recognizer_run_started", "recognizer_run_finished"),
    ("finish_before_prediction_commitment", "recognizer_run_finished", "prediction_audit_baseline_preservation_outputs_committed"),
    ("predictions_before_reveal", "prediction_audit_baseline_preservation_outputs_committed", "answer_and_gate_packages_revealed"),
    ("prediction_commitment_before_optional_upload_failure", "prediction_audit_baseline_preservation_outputs_committed", "committed_output_upload_failed"),
    ("upload_failure_before_reupload", "committed_output_upload_failed", "committed_output_reuploaded"),
    ("reupload_before_reveal", "committed_output_reuploaded", "answer_and_gate_packages_revealed"),
    ("reveal_before_scoring", "answer_and_gate_packages_revealed", "scoring_started"),
    ("scoring_before_reports", "scoring_started", "score_and_auxiliary_reports_committed"),
    ("reports_before_consumed_terminal", "score_and_auxiliary_reports_committed", "attempt_consumed_terminal"),
    ("terminal_before_append_receipt", "attempt_terminal", "terminal_append_receipt"),
    ("append_receipt_before_retry_authorization", "terminal_append_receipt", "retry_authorization_if_eligible"),
    ("retry_authorization_before_next_open", "retry_authorization_if_eligible", "next_attempt_preregistered"),
    ("last_append_receipt_before_campaign_finalization", "last_terminal_append_receipt", "campaign_finalization"),
)
_ORDERING_STATEMENT_SPECS_V2: Final = tuple(
    (
        ordinal,
        predecessor_stage,
        successor_stage,
        statement_name
        + ": high_level_declarative_gap_only; required_order="
        + predecessor_stage
        + "<"
        + successor_stage
        + "; event_schema_signature_schema_ledger_schema_and_timing_verifier_are_not_supplied",
    )
    for ordinal, (statement_name, predecessor_stage, successor_stage) in enumerate(
        _ORDERING_TRIPLES_V2
    )
)


_MAX_TEXT_UTF8_BYTES_V2: Final = 4096
_MAX_TUPLE_ITEMS_V2: Final = 64
_MAX_INTEGER_V2: Final = 2**63 - 1
_LOWER_HEX = frozenset("0123456789abcdef")

_IDENTITY_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/UPSTREAM_IDENTITY/V2\x00"
_MISSING_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/MISSING_EVIDENCE/V2\x00"
_ORDERING_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/ORDERING/V2\x00"
_IDENTITY_ROOT_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/UPSTREAM_IDENTITY_ROOT/V2\x00"
_MISSING_ROOT_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/MISSING_EVIDENCE_ROOT/V2\x00"
_ORDERING_ROOT_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/ORDERING_ROOT/V2\x00"
_CONTRACT_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/CONTRACT/V2\x00"
_VALIDATION_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/VALIDATION/V2\x00"
_SCHEMA_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/SCHEMA/V2\x00"
_POLICY_DOMAIN_V2: Final = b"HEGEL/PHASE2B/ACTUAL_REPLAY/ADMISSION/GAP/POLICY/V2\x00"

_IDENTITY_ID_PREFIX_V2: Final = "phase2b_actual_replay_admission_gap_upstream_identity_v2_"
_MISSING_ID_PREFIX_V2: Final = "phase2b_actual_replay_admission_gap_missing_evidence_v2_"
_ORDERING_ID_PREFIX_V2: Final = "phase2b_actual_replay_admission_gap_ordering_v2_"
_IDENTITY_ROOT_PREFIX_V2: Final = "phase2b_actual_replay_admission_gap_upstream_identities_v2_"
_MISSING_ROOT_PREFIX_V2: Final = "phase2b_actual_replay_admission_gap_missing_evidence_requirements_v2_"
_ORDERING_ROOT_PREFIX_V2: Final = "phase2b_actual_replay_admission_gap_ordering_statements_v2_"
_CONTRACT_ID_PREFIX_V2: Final = "phase2b_actual_unsealed_960_replay_admission_gap_inventory_v2_"
_VALIDATION_ID_PREFIX_V2: Final = "phase2b_actual_unsealed_960_replay_admission_gap_inventory_validation_v2_"
_SCHEMA_PREFIX_V2: Final = "phase2b_actual_unsealed_960_replay_admission_gap_inventory_schema_v2_"
_POLICY_PREFIX_V2: Final = "phase2b_actual_unsealed_960_replay_admission_gap_inventory_policy_v2_"

_CONTENT_ADDRESS_DOMAIN_PREFIX_BINDINGS_V2: Final = (
    ("upstream_dependency_identity_id", _IDENTITY_DOMAIN_V2.hex(), _IDENTITY_ID_PREFIX_V2),
    ("missing_evidence_requirement_id", _MISSING_DOMAIN_V2.hex(), _MISSING_ID_PREFIX_V2),
    ("required_ordering_statement_id", _ORDERING_DOMAIN_V2.hex(), _ORDERING_ID_PREFIX_V2),
    ("upstream_dependency_identity_ids_root", _IDENTITY_ROOT_DOMAIN_V2.hex(), _IDENTITY_ROOT_PREFIX_V2),
    ("missing_evidence_requirement_ids_root", _MISSING_ROOT_DOMAIN_V2.hex(), _MISSING_ROOT_PREFIX_V2),
    ("required_ordering_statement_ids_root", _ORDERING_ROOT_DOMAIN_V2.hex(), _ORDERING_ROOT_PREFIX_V2),
    ("contract_id", _CONTRACT_DOMAIN_V2.hex(), _CONTRACT_ID_PREFIX_V2),
    ("validation_id", _VALIDATION_DOMAIN_V2.hex(), _VALIDATION_ID_PREFIX_V2),
    ("schema_id", _SCHEMA_DOMAIN_V2.hex(), _SCHEMA_PREFIX_V2),
    ("policy_id", _POLICY_DOMAIN_V2.hex(), _POLICY_PREFIX_V2),
)

_GLOBAL_PREFLIGHT_BEFORE_HASH_SENTINEL_V2: Final = (
    "snapshot_every_top_and_nested_field_exactly_once",
    "close_exact_types_UTF8_caps_integer_caps_SHA64_and_prefixed_ID_formats",
    "close_exact_tuple_counts_declared_order_and_no_duplicates",
    "compare_closed_upstream_missing_evidence_ordering_claim_and_count_catalogs_to_frozen_specs",
    "only_after_complete_global_closure_recompute_primitive_IDs_length_framed_roots_contract_SHA_and_contract_ID",
    "atomic_rejection_exposes_no_catalog_no_ID_and_all_claims_false",
)


def _ordered_pairs_v2(names: tuple[str, ...], values: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    if len(names) != len(values):
        raise RuntimeError("ordered preimage field drift")
    return tuple(zip(names, values))


def _sha_v2(domain: bytes, preimage: object) -> str:
    return hashlib.sha256(domain + canonical_json(preimage).encode("utf-8")).hexdigest()


def _content_id_v2(domain: bytes, prefix: str, names: tuple[str, ...], values: tuple[object, ...]) -> str:
    return prefix + _sha_v2(domain, _ordered_pairs_v2(names, values))


def _sequence_root_v2(domain: bytes, prefix: str, identifiers: tuple[str, ...]) -> str:
    framed = bytearray(domain)
    framed.extend(len(identifiers).to_bytes(4, "big"))
    for identifier in identifiers:
        raw = identifier.encode("ascii")
        if len(raw) > 65_535:
            raise ValueError("identifier exceeds u16 frame")
        framed.extend(len(raw).to_bytes(2, "big"))
        framed.extend(raw)
    return prefix + hashlib.sha256(bytes(framed)).hexdigest()


def _issue_v2(cls: type[object], values: tuple[tuple[str, object], ...]) -> object:
    if tuple(name for name, _ in values) != tuple(item.name for item in fields(cls)):
        raise RuntimeError(f"{cls.__name__} issuance field drift")
    result = object.__new__(cls)
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


_SCHEMA_PREIMAGE_V2: Final = (
    ("version", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION),
    ("claim_level", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL),
    ("field_manifests", (
        ("upstream_dependency_identity", _UPSTREAM_DEPENDENCY_IDENTITY_FIELDS_V2),
        ("missing_evidence_requirement", _MISSING_EVIDENCE_FIELDS_V2),
        ("required_ordering_statement", _ORDERING_STATEMENT_FIELDS_V2),
        ("contract", _CONTRACT_FIELDS_V2),
        ("validation", _VALIDATION_FIELDS_V2),
        ("rejection", _REJECTION_FIELDS_V2),
    )),
    ("disposition_values", tuple(item.value for item in AdmissionGapInventoryDispositionV2)),
    ("reason_values", tuple(item.value for item in AdmissionGapInventoryReasonV2)),
    ("type_and_cap_manifest", (
        ("text", "exact_str_nonempty_valid_UTF8_maximum_4096_bytes"),
        ("sha256", "exact_lowercase_64_hex"),
        ("content_id", "exact_declared_prefix_plus_lowercase_64_hex"),
        ("integer", "exact_int_not_bool_nonnegative_maximum_9223372036854775807"),
        ("boolean", "exact_bool"),
        ("tuple", "exact_tuple_declared_length_maximum_64_snapshot_once"),
        ("nested_DTO", "exact_type_exact_field_manifest_snapshot_once"),
        ("prehash", "close_entire_graph_types_formats_caps_counts_and_duplicates_before_any_hash"),
    )),
)
ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_SCHEMA_ID: Final = (
    _SCHEMA_PREFIX_V2 + _sha_v2(_SCHEMA_DOMAIN_V2, _SCHEMA_PREIMAGE_V2)
)

_POLICY_PREIMAGE_V2: Final = (
    ("version", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION),
    ("schema_id", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_SCHEMA_ID),
    ("claim_level", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL),
    ("success_disposition", AdmissionGapInventoryDispositionV2.GAP_INVENTORY_FROZEN_NOT_ADMITTED.value),
    ("success_reason", AdmissionGapInventoryReasonV2.MISSING_EVIDENCE_AND_ORDERING_GAPS_INVENTORIED_NOT_VERIFIED.value),
    ("upstream_dependency_identities", _UPSTREAM_DEPENDENCY_IDENTITIES_V2),
    ("missing_evidence_specs", _MISSING_EVIDENCE_SPECS_V2),
    ("ordering_statement_specs", _ORDERING_STATEMENT_SPECS_V2),
    ("missing_status", _MISSING_STATUS_V2),
    ("ordering_status", _ORDERING_STATUS_V2),
    ("non_ordering_gap_statements", (
        "each_attempt_OPEN_requires_exactly_one_terminal_closure_but_no_record_schema_or_verifier_is_supplied",
        "no_attempt_event_may_follow_that_attempt_terminal_but_no_event_schema_or_verifier_is_supplied",
        "attempt_indices_must_be_contiguous_0_through_at_most_2_but_no_registry_schema_or_verifier_is_supplied",
        "REUPLOAD_COMMITTED_OUTPUT_must_not_create_a_next_attempt_but_no_retry_schema_or_verifier_is_supplied",
        "campaign_finalization_must_follow_all_terminal_append_receipts_but_no_finalization_schema_or_verifier_is_supplied",
    )),
    ("true_claims", _TRUE_CLAIMS_V2),
    ("false_claims", _FALSE_CLAIMS_V2),
    ("content_address_domain_prefix_bindings", (
        "domain_bytes_are_exactly_lowercase_hex_decoded_values_in_declared_order",
        _CONTENT_ADDRESS_DOMAIN_PREFIX_BINDINGS_V2,
    )),
    ("content_address_formulas", (
        ("primitive", "prefix || sha256(domain || canonical_json(tuple((field_name,exact_closed_value) in declared_order)))"),
        ("root", "prefix || sha256(domain || u32be(count) || repeated(u16be(ascii_identifier_length) || ascii_identifier))"),
        ("contract", "contract_prefix || sha256(contract_domain || canonical_json(exact_ordered_contract_preimage; upstream_dependency_identities_are_tuple_of_declared_order_named_pairs; missing_and_ordering_DTOs_are_tuple_of_declared_order_named_pairs))"),
        ("validation", "validation_prefix || sha256(validation_domain || canonical_json(exact_ordered_validation_preimage_excluding_validation_id; upstream_dependency_identities_are_tuple_of_declared_order_named_pairs; missing_and_ordering_DTOs_are_tuple_of_declared_order_named_pairs))"),
        ("schema", "schema_prefix || sha256(schema_domain || canonical_json(exact_ordered_SCHEMA_PREIMAGE_V2))"),
        ("policy", "policy_prefix || sha256(policy_domain || canonical_json(exact_ordered_POLICY_PREIMAGE_V2))"),
    )),
    ("validation_order", (
        "exact_top_level_type",
        "snapshot_each_top_and_nested_field_once",
        "global_type_format_cap_count_and_duplicate_preflight",
        "cross_version_identity_check",
        "independent_primitive_content_ID_recalculation",
        "independent_length_framed_root_recalculation",
        "independent_contract_SHA_and_ID_recalculation",
        "exact_frozen_primitive_snapshot_comparison",
        "fresh_success_graph_or_atomic_all_false_no_catalog_no_ID_rejection",
    )),
    ("global_preflight_before_hash_sentinel", _GLOBAL_PREFLIGHT_BEFORE_HASH_SENTINEL_V2),
    ("api_boundary", (
        "getter_has_no_arguments",
        "validator_has_keyword_only_contract_argument",
        "no_evidence_bytes_paths_timestamps_signatures_ledgers_attestations_or_booleans_accepted",
        "catalogue_validation_never_admits_or_authorizes_execution",
    )),
    ("forbidden_operations", (
        "no_runner_decoder_scorer_formal_validator_ledger_signature_or_runtime_verifier",
        "no_filesystem_network_clock_random_subprocess_container_or_Q_access",
    )),
)
ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_POLICY_ID: Final = (
    _POLICY_PREFIX_V2 + _sha_v2(_POLICY_DOMAIN_V2, _POLICY_PREIMAGE_V2)
)


def _identity_id_v2(identity: tuple[str, str, str]) -> str:
    return _content_id_v2(
        _IDENTITY_DOMAIN_V2,
        _IDENTITY_ID_PREFIX_V2,
        _UPSTREAM_DEPENDENCY_IDENTITY_FIELDS_V2,
        identity,
    )


def _new_missing_v2(spec: tuple[int, str, str]) -> MissingEvidenceRequirementV2:
    values: tuple[object, ...] = (
        spec[0], spec[1], spec[2], _MISSING_STATUS_V2, True, False, False, False,
    )
    requirement_id = _content_id_v2(
        _MISSING_DOMAIN_V2,
        _MISSING_ID_PREFIX_V2,
        _MISSING_EVIDENCE_FIELDS_V2[:-1],
        values,
    )
    return _issue_v2(
        MissingEvidenceRequirementV2,
        _ordered_pairs_v2(_MISSING_EVIDENCE_FIELDS_V2, values + (requirement_id,)),
    )  # type: ignore[return-value]


def _new_ordering_v2(spec: tuple[int, str, str, str]) -> RequiredOrderingStatementV2:
    values: tuple[object, ...] = (
        spec[0], spec[1], spec[2], spec[3], _ORDERING_STATUS_V2,
        False, False, False, False,
    )
    statement_id = _content_id_v2(
        _ORDERING_DOMAIN_V2,
        _ORDERING_ID_PREFIX_V2,
        _ORDERING_STATEMENT_FIELDS_V2[:-1],
        values,
    )
    return _issue_v2(
        RequiredOrderingStatementV2,
        _ordered_pairs_v2(_ORDERING_STATEMENT_FIELDS_V2, values + (statement_id,)),
    )  # type: ignore[return-value]


def _explicit_missing_v2(value: MissingEvidenceRequirementV2) -> tuple[tuple[str, object], ...]:
    return _ordered_pairs_v2(
        _MISSING_EVIDENCE_FIELDS_V2,
        tuple(getattr(value, name) for name in _MISSING_EVIDENCE_FIELDS_V2),
    )


def _explicit_identity_v2(
    value: tuple[str, str, str],
) -> tuple[tuple[str, object], ...]:
    return _ordered_pairs_v2(
        _UPSTREAM_DEPENDENCY_IDENTITY_FIELDS_V2,
        (value[0], value[1], value[2]),
    )


def _explicit_ordering_v2(value: RequiredOrderingStatementV2) -> tuple[tuple[str, object], ...]:
    return _ordered_pairs_v2(
        _ORDERING_STATEMENT_FIELDS_V2,
        tuple(getattr(value, name) for name in _ORDERING_STATEMENT_FIELDS_V2),
    )


def _contract_preimage_v2(values: tuple[tuple[str, object], ...]) -> tuple[tuple[str, object], ...]:
    converted: list[tuple[str, object]] = []
    for name, value in values:
        if name == "upstream_dependency_identities":
            value = tuple(_explicit_identity_v2(item) for item in value)  # type: ignore[union-attr]
        elif name == "missing_evidence_requirements":
            value = tuple(_explicit_missing_v2(item) for item in value)  # type: ignore[union-attr]
        elif name == "required_ordering_statements":
            value = tuple(_explicit_ordering_v2(item) for item in value)  # type: ignore[union-attr]
        converted.append((name, value))
    return tuple(converted)


def _new_contract_v2() -> AdmissionGapInventoryContractV2:
    identities = tuple((item[0], item[1], item[2]) for item in _UPSTREAM_DEPENDENCY_IDENTITIES_V2)
    identity_ids = tuple(_identity_id_v2(item) for item in identities)
    missing = tuple(_new_missing_v2(item) for item in _MISSING_EVIDENCE_SPECS_V2)
    ordering = tuple(_new_ordering_v2(item) for item in _ORDERING_STATEMENT_SPECS_V2)
    base_values: tuple[tuple[str, object], ...] = (
        ("version", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION),
        ("schema_id", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_SCHEMA_ID),
        ("policy_id", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_POLICY_ID),
        ("claim_level", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL),
        ("upstream_dependency_identities", identities),
        ("upstream_dependency_identity_ids", identity_ids),
        ("upstream_dependency_identity_ids_root", _sequence_root_v2(_IDENTITY_ROOT_DOMAIN_V2, _IDENTITY_ROOT_PREFIX_V2, identity_ids)),
        ("missing_evidence_requirements", missing),
        ("missing_evidence_requirement_ids_root", _sequence_root_v2(_MISSING_ROOT_DOMAIN_V2, _MISSING_ROOT_PREFIX_V2, tuple(item.requirement_id for item in missing))),
        ("required_ordering_statements", ordering),
        ("required_ordering_statement_ids_root", _sequence_root_v2(_ORDERING_ROOT_DOMAIN_V2, _ORDERING_ROOT_PREFIX_V2, tuple(item.ordering_statement_id for item in ordering))),
        ("upstream_dependency_identity_count", len(identities)),
        ("missing_evidence_requirement_count", len(missing)),
        ("required_ordering_statement_count", len(ordering)),
        *((name, True) for name in _TRUE_CLAIMS_V2),
        *((name, False) for name in _FALSE_CLAIMS_V2),
    )
    contract_sha = _sha_v2(_CONTRACT_DOMAIN_V2, _contract_preimage_v2(base_values))
    return _issue_v2(
        AdmissionGapInventoryContractV2,
        base_values + (
            ("contract_sha256", contract_sha),
            ("contract_id", _CONTRACT_ID_PREFIX_V2 + contract_sha),
        ),
    )  # type: ignore[return-value]


def frozen_actual_unsealed_960_replay_admission_gap_inventory_v2() -> AdmissionGapInventoryContractV2:
    """Return a fresh gap inventory; this is not an admission decision."""

    return _new_contract_v2()


class _RejectedV2(Exception):
    def __init__(self, reason: AdmissionGapInventoryReasonV2) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _reject_v2(reason: AdmissionGapInventoryReasonV2) -> None:
    raise _RejectedV2(reason)


def _require_text_v2(value: object) -> str:
    if type(value) is not str or not value:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    try:
        raw = value.encode("utf-8")
    except UnicodeError:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    if len(raw) > _MAX_TEXT_UTF8_BYTES_V2:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return value


def _require_bool_v2(value: object) -> bool:
    if type(value) is not bool:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return value


def _require_int_v2(value: object, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > min(maximum, _MAX_INTEGER_V2):
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return value


def _require_sha_v2(value: object) -> str:
    text = _require_text_v2(value)
    if len(text) != 64 or any(char not in _LOWER_HEX for char in text):
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return text


def _require_prefixed_id_v2(value: object, prefix: str) -> str:
    text = _require_text_v2(value)
    if not text.startswith(prefix):
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    _require_sha_v2(text[len(prefix):])
    return text


def _require_generic_content_id_v2(value: object) -> str:
    text = _require_text_v2(value)
    marker = text.rfind("_")
    if marker < 1:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    _require_sha_v2(text[marker + 1:])
    return text


def _require_tuple_v2(value: object, length: int) -> tuple[object, ...]:
    if type(value) is not tuple or len(value) != length or length > _MAX_TUPLE_ITEMS_V2:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return value


def _close_identity_v2(value: object, expected: tuple[str, str, str]) -> tuple[str, str, str]:
    raw = _require_tuple_v2(value, 3)
    name = _require_text_v2(raw[0])
    kind = _require_text_v2(raw[1])
    identity_value = _require_text_v2(raw[2])
    if kind == "source_sha256":
        _require_sha_v2(identity_value)
    elif kind in {"schema_id", "policy_id"}:
        _require_generic_content_id_v2(identity_value)
    elif kind not in {"version", "claim_level"}:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    closed = (name, kind, identity_value)
    if closed != expected:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return closed


def _close_missing_v2(value: object) -> MissingEvidenceRequirementV2:
    if type(value) is not MissingEvidenceRequirementV2:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    raw = tuple(getattr(value, name) for name in _MISSING_EVIDENCE_FIELDS_V2)
    closed: tuple[object, ...] = (
        _require_int_v2(raw[0], len(_MISSING_EVIDENCE_SPECS_V2) - 1),
        _require_text_v2(raw[1]),
        _require_text_v2(raw[2]),
        _require_text_v2(raw[3]),
        _require_bool_v2(raw[4]),
        _require_bool_v2(raw[5]),
        _require_bool_v2(raw[6]),
        _require_bool_v2(raw[7]),
        _require_prefixed_id_v2(raw[8], _MISSING_ID_PREFIX_V2),
    )
    return _issue_v2(MissingEvidenceRequirementV2, _ordered_pairs_v2(_MISSING_EVIDENCE_FIELDS_V2, closed))  # type: ignore[return-value]


def _close_ordering_v2(value: object) -> RequiredOrderingStatementV2:
    if type(value) is not RequiredOrderingStatementV2:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    raw = tuple(getattr(value, name) for name in _ORDERING_STATEMENT_FIELDS_V2)
    closed: tuple[object, ...] = (
        _require_int_v2(raw[0], len(_ORDERING_STATEMENT_SPECS_V2) - 1),
        _require_text_v2(raw[1]),
        _require_text_v2(raw[2]),
        _require_text_v2(raw[3]),
        _require_text_v2(raw[4]),
        _require_bool_v2(raw[5]),
        _require_bool_v2(raw[6]),
        _require_bool_v2(raw[7]),
        _require_bool_v2(raw[8]),
        _require_prefixed_id_v2(raw[9], _ORDERING_ID_PREFIX_V2),
    )
    return _issue_v2(RequiredOrderingStatementV2, _ordered_pairs_v2(_ORDERING_STATEMENT_FIELDS_V2, closed))  # type: ignore[return-value]


def _preflight_contract_v2(value: AdmissionGapInventoryContractV2) -> AdmissionGapInventoryContractV2:
    raw = {name: getattr(value, name) for name in _CONTRACT_FIELDS_V2}
    version = _require_text_v2(raw["version"])
    schema_id = _require_prefixed_id_v2(raw["schema_id"], _SCHEMA_PREFIX_V2)
    policy_id = _require_prefixed_id_v2(raw["policy_id"], _POLICY_PREFIX_V2)
    claim_level = _require_text_v2(raw["claim_level"])

    raw_identities = _require_tuple_v2(raw["upstream_dependency_identities"], len(_UPSTREAM_DEPENDENCY_IDENTITIES_V2))
    identities = tuple(_close_identity_v2(item, expected) for item, expected in zip(raw_identities, _UPSTREAM_DEPENDENCY_IDENTITIES_V2))
    raw_identity_ids = _require_tuple_v2(raw["upstream_dependency_identity_ids"], len(_UPSTREAM_DEPENDENCY_IDENTITIES_V2))
    identity_ids = tuple(_require_prefixed_id_v2(item, _IDENTITY_ID_PREFIX_V2) for item in raw_identity_ids)
    identity_root = _require_prefixed_id_v2(raw["upstream_dependency_identity_ids_root"], _IDENTITY_ROOT_PREFIX_V2)

    raw_missing = _require_tuple_v2(raw["missing_evidence_requirements"], len(_MISSING_EVIDENCE_SPECS_V2))
    missing = tuple(_close_missing_v2(item) for item in raw_missing)
    missing_root = _require_prefixed_id_v2(raw["missing_evidence_requirement_ids_root"], _MISSING_ROOT_PREFIX_V2)
    raw_ordering = _require_tuple_v2(raw["required_ordering_statements"], len(_ORDERING_STATEMENT_SPECS_V2))
    ordering = tuple(_close_ordering_v2(item) for item in raw_ordering)
    ordering_root = _require_prefixed_id_v2(raw["required_ordering_statement_ids_root"], _ORDERING_ROOT_PREFIX_V2)

    identity_count = _require_int_v2(raw["upstream_dependency_identity_count"], len(_UPSTREAM_DEPENDENCY_IDENTITIES_V2))
    missing_count = _require_int_v2(raw["missing_evidence_requirement_count"], len(_MISSING_EVIDENCE_SPECS_V2))
    ordering_count = _require_int_v2(raw["required_ordering_statement_count"], len(_ORDERING_STATEMENT_SPECS_V2))
    true_values = tuple((name, _require_bool_v2(raw[name])) for name in _TRUE_CLAIMS_V2)
    false_values = tuple((name, _require_bool_v2(raw[name])) for name in _FALSE_CLAIMS_V2)
    contract_sha = _require_sha_v2(raw["contract_sha256"])
    contract_id = _require_prefixed_id_v2(raw["contract_id"], _CONTRACT_ID_PREFIX_V2)
    # Exact catalogue closure is deliberately complete before any content hash.
    # IDs are format-closed above and cryptographically recomputed only later.
    if (
        identities != _UPSTREAM_DEPENDENCY_IDENTITIES_V2
        or len(set(identities)) != len(identities)
        or len(set(identity_ids)) != len(identity_ids)
        or tuple((item.ordinal, item.evidence_name, item.purpose) for item in missing)
        != _MISSING_EVIDENCE_SPECS_V2
        or any(
            item.status != _MISSING_STATUS_V2
            or item.missing is not True
            or item.supplied_by_this_contract is not False
            or item.verified is not False
            or item.verifier_implemented is not False
            for item in missing
        )
        or len({item.evidence_name for item in missing}) != len(missing)
        or len({item.requirement_id for item in missing}) != len(missing)
        or tuple(
            (
                item.ordinal,
                item.predecessor_stage,
                item.successor_stage,
                item.statement,
            )
            for item in ordering
        )
        != _ORDERING_STATEMENT_SPECS_V2
        or any(
            item.status != _ORDERING_STATUS_V2
            or item.event_schema_supplied is not False
            or item.signature_schema_supplied is not False
            or item.ledger_schema_supplied is not False
            or item.timing_verifier_implemented is not False
            for item in ordering
        )
        or len(
            {(item.predecessor_stage, item.successor_stage) for item in ordering}
        )
        != len(ordering)
        or len({item.ordering_statement_id for item in ordering}) != len(ordering)
        or identity_count != len(_UPSTREAM_DEPENDENCY_IDENTITIES_V2)
        or missing_count != len(_MISSING_EVIDENCE_SPECS_V2)
        or ordering_count != len(_ORDERING_STATEMENT_SPECS_V2)
        or any(value is not True for _, value in true_values)
        or any(value is not False for _, value in false_values)
    ):
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    return _issue_v2(
        AdmissionGapInventoryContractV2,
        (
            ("version", version), ("schema_id", schema_id), ("policy_id", policy_id), ("claim_level", claim_level),
            ("upstream_dependency_identities", identities), ("upstream_dependency_identity_ids", identity_ids),
            ("upstream_dependency_identity_ids_root", identity_root),
            ("missing_evidence_requirements", missing), ("missing_evidence_requirement_ids_root", missing_root),
            ("required_ordering_statements", ordering), ("required_ordering_statement_ids_root", ordering_root),
            ("upstream_dependency_identity_count", identity_count), ("missing_evidence_requirement_count", missing_count),
            ("required_ordering_statement_count", ordering_count), *true_values, *false_values,
            ("contract_sha256", contract_sha), ("contract_id", contract_id),
        ),
    )  # type: ignore[return-value]


def _primitive_snapshot_v2(value: AdmissionGapInventoryContractV2) -> tuple[object, ...]:
    return tuple(
        tuple(_explicit_missing_v2(item) for item in field_value)
        if name == "missing_evidence_requirements"
        else tuple(_explicit_ordering_v2(item) for item in field_value)
        if name == "required_ordering_statements"
        else field_value
        for name in _CONTRACT_FIELDS_V2
        for field_value in (getattr(value, name),)
    )


def _verify_contract_v2(value: AdmissionGapInventoryContractV2) -> None:
    expected_identity_ids = tuple(_identity_id_v2(item) for item in value.upstream_dependency_identities)
    for item in value.missing_evidence_requirements:
        expected_id = _content_id_v2(
            _MISSING_DOMAIN_V2, _MISSING_ID_PREFIX_V2,
            _MISSING_EVIDENCE_FIELDS_V2[:-1],
            tuple(getattr(item, name) for name in _MISSING_EVIDENCE_FIELDS_V2[:-1]),
        )
        if item.requirement_id != expected_id:
            _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    for item in value.required_ordering_statements:
        expected_id = _content_id_v2(
            _ORDERING_DOMAIN_V2, _ORDERING_ID_PREFIX_V2,
            _ORDERING_STATEMENT_FIELDS_V2[:-1],
            tuple(getattr(item, name) for name in _ORDERING_STATEMENT_FIELDS_V2[:-1]),
        )
        if item.ordering_statement_id != expected_id:
            _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    if (
        value.upstream_dependency_identity_ids != expected_identity_ids
        or value.upstream_dependency_identity_ids_root != _sequence_root_v2(_IDENTITY_ROOT_DOMAIN_V2, _IDENTITY_ROOT_PREFIX_V2, expected_identity_ids)
        or value.missing_evidence_requirement_ids_root != _sequence_root_v2(_MISSING_ROOT_DOMAIN_V2, _MISSING_ROOT_PREFIX_V2, tuple(item.requirement_id for item in value.missing_evidence_requirements))
        or value.required_ordering_statement_ids_root != _sequence_root_v2(_ORDERING_ROOT_DOMAIN_V2, _ORDERING_ROOT_PREFIX_V2, tuple(item.ordering_statement_id for item in value.required_ordering_statements))
    ):
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
    base = tuple((name, getattr(value, name)) for name in _CONTRACT_FIELDS_V2[:-2])
    expected_sha = _sha_v2(_CONTRACT_DOMAIN_V2, _contract_preimage_v2(base))
    if value.contract_sha256 != expected_sha or value.contract_id != _CONTRACT_ID_PREFIX_V2 + expected_sha:
        _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)


def _new_validation_v2(contract: AdmissionGapInventoryContractV2) -> AdmissionGapInventoryValidationV2:
    values: tuple[tuple[str, object], ...] = (
        ("disposition", AdmissionGapInventoryDispositionV2.GAP_INVENTORY_FROZEN_NOT_ADMITTED),
        ("reason", AdmissionGapInventoryReasonV2.MISSING_EVIDENCE_AND_ORDERING_GAPS_INVENTORIED_NOT_VERIFIED),
        ("version", contract.version), ("schema_id", contract.schema_id), ("policy_id", contract.policy_id),
        ("claim_level", contract.claim_level), ("validation_id", "pending"),
        ("contract_id", contract.contract_id), ("contract_sha256", contract.contract_sha256),
        ("upstream_dependency_identity_ids_root", contract.upstream_dependency_identity_ids_root),
        ("missing_evidence_requirement_ids_root", contract.missing_evidence_requirement_ids_root),
        ("required_ordering_statement_ids_root", contract.required_ordering_statement_ids_root),
        ("upstream_dependency_identity_count", contract.upstream_dependency_identity_count),
        ("missing_evidence_requirement_count", contract.missing_evidence_requirement_count),
        ("required_ordering_statement_count", contract.required_ordering_statement_count),
        *((name, True) for name in _TRUE_CLAIMS_V2),
        *((name, False) for name in _FALSE_CLAIMS_V2),
        ("upstream_dependency_identities", tuple((item[0], item[1], item[2]) for item in contract.upstream_dependency_identities)),
        ("missing_evidence_requirements", tuple(_new_missing_v2(item) for item in _MISSING_EVIDENCE_SPECS_V2)),
        ("required_ordering_statements", tuple(_new_ordering_v2(item) for item in _ORDERING_STATEMENT_SPECS_V2)),
    )
    preimage = tuple(
        (
            name,
            tuple(_explicit_identity_v2(item) for item in value)
            if name == "upstream_dependency_identities"
            else tuple(_explicit_missing_v2(item) for item in value)
            if name == "missing_evidence_requirements"
            else tuple(_explicit_ordering_v2(item) for item in value)
            if name == "required_ordering_statements"
            else value,
        )
        for name, value in values if name != "validation_id"
    )
    validation_id = _VALIDATION_ID_PREFIX_V2 + _sha_v2(_VALIDATION_DOMAIN_V2, preimage)
    return _issue_v2(
        AdmissionGapInventoryValidationV2,
        tuple((name, validation_id if name == "validation_id" else value) for name, value in values),
    )  # type: ignore[return-value]


def _new_rejection_v2(reason: AdmissionGapInventoryReasonV2) -> AdmissionGapInventoryRejectionV2:
    values: tuple[tuple[str, object], ...] = (
        ("disposition", AdmissionGapInventoryDispositionV2.REJECTED), ("reason", reason),
        ("version", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION),
        ("claim_level", ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL),
        ("schema_id", None), ("policy_id", None), ("validation_id", None),
        ("contract_id", None), ("contract_sha256", None),
        ("upstream_dependency_identity_ids_root", None), ("missing_evidence_requirement_ids_root", None),
        ("required_ordering_statement_ids_root", None),
        ("upstream_dependency_identity_count", 0), ("missing_evidence_requirement_count", 0),
        ("required_ordering_statement_count", 0), ("upstream_dependency_identities", ()),
        ("missing_evidence_requirements", ()), ("required_ordering_statements", ()),
        ("partial_output_published", False),
        *((name, False) for name in _TRUE_CLAIMS_V2),
        *((name, False) for name in _FALSE_CLAIMS_V2),
    )
    return _issue_v2(AdmissionGapInventoryRejectionV2, values)  # type: ignore[return-value]


def validate_actual_unsealed_960_replay_admission_gap_inventory_v2(
    *, contract: object,
) -> AdmissionGapInventoryValidationV2 | AdmissionGapInventoryRejectionV2:
    """Validate only the frozen inventory and never accept evidence."""

    try:
        if type(contract) is not AdmissionGapInventoryContractV2:
            _reject_v2(AdmissionGapInventoryReasonV2.WRONG_INPUT_TYPE)
        closed = _preflight_contract_v2(contract)
        if (
            closed.version != ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_VERSION
            or closed.schema_id != ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_SCHEMA_ID
            or closed.policy_id != ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_POLICY_ID
            or closed.claim_level != ACTUAL_UNSEALED_960_REPLAY_ADMISSION_GAP_INVENTORY_V2_CLAIM_LEVEL
        ):
            _reject_v2(AdmissionGapInventoryReasonV2.CROSS_VERSION_INPUT)
        _verify_contract_v2(closed)
        expected = _new_contract_v2()
        if _primitive_snapshot_v2(closed) != _primitive_snapshot_v2(expected):
            _reject_v2(AdmissionGapInventoryReasonV2.CONTRACT_INVALID)
        return _new_validation_v2(_new_contract_v2())
    except _RejectedV2 as error:
        return _new_rejection_v2(error.reason)
    except Exception:
        return _new_rejection_v2(AdmissionGapInventoryReasonV2.INTERNAL_ERROR)


for _cls_v2, _manifest_v2 in (
    (MissingEvidenceRequirementV2, _MISSING_EVIDENCE_FIELDS_V2),
    (RequiredOrderingStatementV2, _ORDERING_STATEMENT_FIELDS_V2),
    (AdmissionGapInventoryContractV2, _CONTRACT_FIELDS_V2),
    (AdmissionGapInventoryValidationV2, _VALIDATION_FIELDS_V2),
    (AdmissionGapInventoryRejectionV2, _REJECTION_FIELDS_V2),
):
    if tuple(item.name for item in fields(_cls_v2)) != _manifest_v2:
        raise RuntimeError(f"{_cls_v2.__name__} field manifest drift")
if (
    len(_MISSING_EVIDENCE_SPECS_V2) != 18
    or tuple(item[0] for item in _MISSING_EVIDENCE_SPECS_V2) != tuple(range(18))
    or len({item[1] for item in _MISSING_EVIDENCE_SPECS_V2}) != 18
    or len(_ORDERING_STATEMENT_SPECS_V2) != 16
    or tuple(item[0] for item in _ORDERING_STATEMENT_SPECS_V2) != tuple(range(16))
    or tuple((item[1], item[2]) for item in _ORDERING_STATEMENT_SPECS_V2)
    != tuple((item[1], item[2]) for item in _ORDERING_TRIPLES_V2)
):
    raise RuntimeError("gap-inventory catalogue drift")
if set(_TRUE_CLAIMS_V2).intersection(_FALSE_CLAIMS_V2):
    raise RuntimeError("gap-inventory claim polarity drift")
if _GLOBAL_PREFLIGHT_BEFORE_HASH_SENTINEL_V2 != (
    "snapshot_every_top_and_nested_field_exactly_once",
    "close_exact_types_UTF8_caps_integer_caps_SHA64_and_prefixed_ID_formats",
    "close_exact_tuple_counts_declared_order_and_no_duplicates",
    "compare_closed_upstream_missing_evidence_ordering_claim_and_count_catalogs_to_frozen_specs",
    "only_after_complete_global_closure_recompute_primitive_IDs_length_framed_roots_contract_SHA_and_contract_ID",
    "atomic_rejection_exposes_no_catalog_no_ID_and_all_claims_false",
):
    raise RuntimeError("gap-inventory global preflight sentinel drift")


__all__ = (
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
