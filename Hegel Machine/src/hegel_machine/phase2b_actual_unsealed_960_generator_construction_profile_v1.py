"""Static Phase-2B generator-construction profile verification only.

This module verifies one exact canonical profile supplied as caller-held bytes
and returns immutable quota/oracle contract data.  It cannot generate a case,
materialize a corpus, allocate an identifier, use entropy, issue a receipt or
commitment, invoke a runtime, or score a prediction.  A successful result is a
cooperative-computation DTO: it is forgeable and is not execution evidence.
Every consumer must freshly verify the original profile bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from .rfc8785_v1 import RFC8785Error, canonicalize_rfc8785_v1


PROFILE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-generator-construction-profile/1"
)
PROFILE_CLAIM_LEVEL_V1: Final = (
    "STATIC_CONSTRUCTION_PROFILE_ONLY_NOT_GENERATOR_NOT_GENERATED"
)
PROFILE_SUCCESS_DISPOSITION_V1: Final = (
    "PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED"
)
PROFILE_CANONICALIZATION_V1: Final = "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
PROFILE_ID_V1: Final = (
    "phase2b_actual_unsealed_960_generator_construction_profile_v1_"
    "3d5175cbfb3749ce7f225a55e3d3d045156adf53ef7aea726ea592d4aa82d531"
)
PROFILE_CANONICAL_SHA256_V1: Final = (
    "8d14629f1e449d9e31bdc9a65b38375e3a994cd6c4a0b8e80006248d0d32b4aa"
)

_PROFILE_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/GENERATOR_CONSTRUCTION_PROFILE/V1\x00"
)
_PROFILE_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_generator_construction_profile_v1_"
)
_MAXIMUM_PROFILE_BYTES_V1: Final = 262_144
_MAXIMUM_SAFE_INTEGER_V1: Final = (1 << 53) - 1
_MAXIMUM_JSON_DEPTH_V1: Final = 64
_MAXIMUM_JSON_NODES_V1: Final = 100_000

_OWNER_AUTHORIZATION_ID_V1: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)
_OWNER_AUTHORIZATION_CONFIG_SHA256_V1: Final = (
    "26ff84c3fe239158b1e8db24b5634ccfe8b3f3995d54bdc1f728c49a70f8fca1"
)
_OWNER_AUTHORIZATION_SOURCE_SHA256_V1: Final = (
    "4c96d50706e520f1c67b2d2c31fce9ed605feacbf4c1177ddf9a7a2a9c1d9a73"
)
_FRESH_SECRET_PURPOSES_V1: Final = (
    "latent_generator_seed",
    "answer_seed",
    "K_shuffle",
    "K_id",
    "K_padding",
)

_FAMILIES_V1: Final = (
    "F01_symmetry_equivariance",
    "F02_monotonicity_order",
    "F03_conservation_balance",
    "F04_additivity_complementarity",
    "F05_locality_composition",
    "F06_negative_feedback_stability",
)
_SCALES_V1: Final = ("S0_FINE_LOCAL", "S1_COARSE_AGGREGATE")
_MARGINS_V1: Final = (
    "clear_interior",
    "moderate",
    "near_boundary_identifiable",
    "nonunique_or_insufficient",
)
_CASE_MARGIN_ROWS_V1: Final = (
    (
        "unique_scale_answerable",
        11,
        8,
        0,
        0,
        19,
        "EXACT_ONE_PASSING_FAMILY_BINDING_SCALE_GROUP_WITH_M_SELECT",
    ),
    (
        "admissible_scale_set_answerable",
        0,
        0,
        0,
        1,
        1,
        "EXACT_ONE_FAMILY_BINDING_WITH_EQUAL_PASSING_S0_S1_ANSWER_SET",
    ),
    (
        "wrong_family_hard_negative",
        3,
        2,
        3,
        0,
        8,
        "ALL_FAIL_UNIQUE_CLOSEST_CYCLIC_NEXT_FAMILY_ABSTAIN",
    ),
    (
        "binding_counterfactual",
        3,
        2,
        3,
        0,
        8,
        "ALL_FAIL_UNIQUE_CLOSEST_SUCCESSOR_BINDING_ABSTAIN",
    ),
    (
        "scale_counterfactual",
        2,
        3,
        3,
        0,
        8,
        "ALL_FAIL_UNIQUE_CLOSEST_OPPOSITE_SCALE_ABSTAIN",
    ),
    (
        "sign_or_invariant_break",
        2,
        3,
        3,
        0,
        8,
        "ALL_FAIL_ONE_EXACT_SIGN_OR_INVARIANT_CLAUSE_BROKEN_ABSTAIN",
    ),
    (
        "insufficient_or_nonidentifiable",
        0,
        0,
        0,
        8,
        8,
        "EXACT_TWO_EQUAL_PASSING_GROUPS_DIFFERENT_FAMILY_OR_BINDING_ABSTAIN",
    ),
)
_PRESERVATION_ROWS_V1: Final = (
    (
        "entity_alpha_renaming",
        _FAMILIES_V1,
        6,
        0,
        2,
        72,
        12,
        "PRIVATE_EXACT_UUID_BIJECTION_WITNESS_NO_FALSE_WIRE_OPERATION",
    ),
    (
        "observation_reorder",
        (_FAMILIES_V1[0], _FAMILIES_V1[2], _FAMILIES_V1[3], _FAMILIES_V1[4]),
        6,
        0,
        2,
        48,
        8,
        "PRIVATE_EXACT_SEQUENCE_PERMUTATION_WITNESS",
    ),
    (
        "irrelevant_entity_augmentation",
        _FAMILIES_V1,
        6,
        0,
        2,
        72,
        12,
        "PRIVATE_EXACT_TASK_DEPENDENCY_UNREACHABILITY_WITNESS",
    ),
    (
        "unit_conversion",
        _FAMILIES_V1,
        8,
        0,
        2,
        96,
        12,
        "WIRE_UNIT_CONVERSION_POSITIVE_EXACT_FACTOR",
    ),
    (
        "coordinate_affine_transform",
        (_FAMILIES_V1[0], _FAMILIES_V1[1], _FAMILIES_V1[4], _FAMILIES_V1[5]),
        8,
        0,
        2,
        64,
        8,
        "WIRE_COORDINATE_AFFINE_EXACT_RATIONAL_INVERTIBLE_MAP",
    ),
    (
        "equivalent_aggregation_split_merge",
        (_FAMILIES_V1[2], _FAMILIES_V1[3], _FAMILIES_V1[4]),
        8,
        0,
        2,
        48,
        6,
        "WIRE_EQUIVALENT_SPLIT_MERGE_EXACT_CERTIFICATE",
    ),
    (
        "nontrivial_scale_map",
        _FAMILIES_V1,
        0,
        10,
        2,
        60,
        12,
        "WIRE_TEMPORAL_SPATIAL_SAMPLING_COARSE_OPERATIONS_3_3_2_2_PER_FAMILY",
    ),
    (
        "sign_convention_reparameterization",
        (_FAMILIES_V1[1], _FAMILIES_V1[2], _FAMILIES_V1[5]),
        6,
        0,
        2,
        36,
        6,
        "WIRE_ONE_DIMENSION_COORDINATE_AFFINE_A_MINUS_ONE_B_ZERO_PLUS_PRIVATE_SIGN_WITNESS",
    ),
)
_UUID_NAMESPACES_V1: Final = (
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
_NEUTRAL_TOKENS_V1: Final = (
    "amber",
    "cedar",
    "cobalt",
    "driftwood",
    "ember",
    "flint",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kestrel",
    "lantern",
    "marble",
    "nectar",
    "opal",
    "pebble",
    "quartz",
    "raven",
    "saffron",
    "topaz",
    "umber",
    "velvet",
    "willow",
    "zephyr",
)
_FAMILY_LEXICONS_V1: Final = (
    ("symmetry", "equivariance", "invariance", "permutation", "reflection", "rotation"),
    ("monotonicity", "ordering", "increasing", "decreasing", "isotonic", "rank"),
    ("conservation", "balance", "inflow", "outflow", "storage", "continuity"),
    ("additivity", "complementarity", "synergy", "interaction", "union", "combined"),
    ("locality", "markov", "blanket", "conditional", "neighborhood", "separation"),
    ("feedback", "stability", "disturbance", "response", "control", "mitigation"),
)
_INVALID_MUTATIONS_V1: Final = (
    (
        "entity_alpha_renaming",
        "CHANGE_ONE_ESSENTIAL_OBSERVATION_ENTITY_ASSOCIATION_REFERENCES_REMAIN_CLOSED",
    ),
    (
        "observation_reorder",
        "CHANGE_TEMPORAL_SUPPORT_OR_DIRECTED_ORDER_NOT_STORAGE_ARRAY_ONLY",
    ),
    (
        "irrelevant_entity_augmentation",
        "NEW_ENTITY_REFERENCED_BY_TASK_OR_ESSENTIAL_OBSERVATION",
    ),
    (
        "unit_conversion",
        "MULTIPLY_ONE_ESSENTIAL_VALUE_BY_TWO_WITHOUT_UNIT_OR_CONTRACT_UPDATE",
    ),
    (
        "coordinate_affine_transform",
        "APPLY_X_PRIME_EQUALS_TWO_X_PLUS_ONE_WITHOUT_SUPPORT_OR_ORIENTATION_UPDATE",
    ),
    (
        "equivalent_aggregation_split_merge",
        "PERTURB_ONE_EXACT_COEFFICIENT_BY_ONE_SIXTEENTH",
    ),
    (
        "nontrivial_scale_map",
        "PERTURB_ONE_TRANSFORMED_COARSE_VALUE_BY_ONE_SIXTEENTH",
    ),
    (
        "sign_convention_reparameterization",
        "FLIP_EXACTLY_ONE_CONJUGATE_TERM_NOT_THE_FULL_CONVENTION",
    ),
)
_FALSE_CLAIMS_V1: Final = {
    "actual_bytes_materialized": False,
    "actual_corpus_audited": False,
    "actual_rows_generated": False,
    "authority_verified": False,
    "answer_seed_commitment_issued": False,
    "commitment_issued": False,
    "commitment_set_consumed": False,
    "per_purpose_commitments_consumed": False,
    "docker_invoked": False,
    "entropy_used": False,
    "filesystem_accessed": False,
    "generation_receipt_issued": False,
    "final_receipt_root_issued": False,
    "generator_seed_commitment_issued": False,
    "generator_implemented": False,
    "generator_qualified": False,
    "identifiers_allocated": False,
    "keys_generated": False,
    "network_accessed": False,
    "origin_authenticated": False,
    "public_variant_uuid_disjointness_verified": False,
    "r03_satisfied": False,
    "r04_satisfied": False,
    "r05_verified": False,
    "r06_satisfied": False,
    "r07_satisfied": False,
    "r08_satisfied": False,
    "r14_satisfied": False,
    "r15_satisfied": False,
    "r16_satisfied": False,
    "r06_qualification_executed": False,
    "r06_qualification_cache_issued": False,
    "qualification_cache_admission_evidence": False,
    "r06_qualification_receipt_issued": False,
    "scoring_performed": False,
    "signatures_generated": False,
    "signature_key_registry_verified": False,
    "durable_store_transaction_executed": False,
    "technical_role_independence_verified": False,
    "custody_verified": False,
    "deterministic_replay_executed": False,
    "wire_replay_executed": False,
}


def _case_margin_rows_json_v1() -> list[dict[str, object]]:
    return [
        {
            "case_type": row[0],
            "clear_interior": row[1],
            "moderate": row[2],
            "near_boundary_identifiable": row[3],
            "nonunique_or_insufficient": row[4],
            "oracle_contract": row[6],
            "total_per_family_scale_cell": row[5],
        }
        for row in _CASE_MARGIN_ROWS_V1
    ]


def _preservation_rows_json_v1() -> list[dict[str, object]]:
    return [
        {
            "applicable_families_in_order": list(row[1]),
            "invalid_controls_per_applicable_family": row[4],
            "invalid_total": row[6],
            "legal_pairs_per_family": row[3],
            "legal_pairs_per_family_scale": row[2],
            "legal_total": row[5],
            "operation_oracle_contract": row[7],
            "transform": row[0],
        }
        for row in _PRESERVATION_ROWS_V1
    ]


def _expected_corpus_contract_v1() -> dict[str, object]:
    return {
        "case_margin_rows": _case_margin_rows_json_v1(),
        "case_type_totals_main": [
            ["unique_scale_answerable", 228],
            ["admissible_scale_set_answerable", 12],
            ["wrong_family_hard_negative", 96],
            ["binding_counterfactual", 96],
            ["scale_counterfactual", 96],
            ["sign_or_invariant_break", 96],
            ["insufficient_or_nonidentifiable", 96],
        ],
        "combined_primary_and_preservation_count": 1532,
        "decision_status_case_and_margin_marginals": "DERIVED_FROM_EXISTING_FREEZE",
        "decision_status_joint_matrix": "NEW_PROSPECTIVE_COMPANION_DECISION",
        "families_in_order": list(_FAMILIES_V1),
        "family_scale_cell_count": 12,
        "invalid_preservation_control_count": 76,
        "legal_preservation_variant_count": 496,
        "main_case_count": 720,
        "main_cases_per_family_scale_cell": 60,
        "margin_totals_per_family_scale_cell": [
            ["clear_interior", 21],
            ["moderate", 18],
            ["near_boundary_identifiable", 12],
            ["nonunique_or_insufficient", 9],
        ],
        "preservation_manifest_count": 572,
        "primary_case_count": 960,
        "primary_decision_multiplicities": [
            ["ANSWER", 348],
            ["ANSWER_SET", 12],
            ["ABSTAIN", 600],
        ],
        "primary_eligibility_signature": {
            "available_observations": 12,
            "eligible_role_candidates": 5,
            "eligible_scales": 2,
            "entities": 6,
            "missing_eligibility_fields": 0,
            "role_candidates": 5,
            "scales": 2,
        },
        "primary_target_multiplicities": {
            "abstain": 600,
            "answerable": 360,
            "binding_each_of_four": 240,
            "family_each_of_six": 160,
            "scale_each_of_two": 480,
        },
        "private_scale_classes_in_order": list(_SCALES_V1),
        "semantic_conflict_case_count": 240,
        "variant_eligibility_signature": {
            "available_observations": 12,
            "eligible_role_candidates": 5,
            "eligible_scales": 2,
            "entities": 7,
            "missing_eligibility_fields": 0,
            "role_candidates": 5,
            "scales": 2,
        },
        "wire_batches_in_order": [
            {"batch": "PRIMARY_960", "count": 960, "within_1024_capacity": True},
            {
                "batch": "PRESERVATION_VARIANTS_572",
                "count": 572,
                "within_1024_capacity": True,
            },
        ],
        "wire_single_batch_1532_forbidden": True,
    }


def _expected_claim_boundary_v1() -> dict[str, object]:
    return {
        "claim_level": PROFILE_CLAIM_LEVEL_V1,
        "false_claims": dict(_FALSE_CLAIMS_V1),
        "fresh_original_bytes_verification_required": True,
        "is_execution_evidence": False,
        "result_object_unforgeable": False,
        "success_disposition": PROFILE_SUCCESS_DISPOSITION_V1,
    }


def _expected_dag_contract_v1() -> dict[str, object]:
    return {
        "actual_materialization_node": "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",
        "actual_materialization_requires_in_order": [
            "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
            "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
        ],
        "actual_materialization_satisfied": False,
        "inference_admission_node": "R16_INFERENCE_ADMISSION_UNLOCKED",
        "inference_admission_satisfied": False,
        "r05_evidence_verified_by_profile": False,
        "r05_required_predecessor": "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
        "r05_parallel_successors_in_order": [
            "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
            "R07_BASELINE_REGISTRATIONS_FROZEN",
        ],
        "r06_satisfied_by_profile_verification": False,
        "r06_target": "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
        "r07_parallel_node": "R07_BASELINE_REGISTRATIONS_FROZEN",
        "r08_successor": "R08_STATISTICAL_VALIDATION_PASSED",
        "r08_successor_dependencies_in_order": [
            "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
            "R07_BASELINE_REGISTRATIONS_FROZEN",
        ],
        "r15_actual_corpus_audit_node": "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED",
        "r15_actual_corpus_audit_satisfied": False,
    }


def _expected_decision_provenance_v1() -> dict[str, object]:
    return {
        "derived_from_existing_freeze_in_order": [
            "MAIN_720_EQUAL_6_FAMILY_X_2_SCALE_X_60",
            "SEMANTIC_CONFLICT_240_EQUAL_12_CELLS_X_10_LOW_X_10_HIGH",
            "LEGAL_496_INVALID_76_PRESERVATION_COUNTS_AND_APPLICABILITY",
            "CASE_MARGINALS_19_1_8_8_8_8_8",
            "MARGIN_MARGINALS_21_18_12_9",
            "ABSOLUTE_BOUND_ONLY_STANDARD_ERROR_WHOLE_BATCH_REJECT",
            "SCALE_REGRET_POINT_0_05_BOOTSTRAP_UPPER_0_08",
        ],
        "new_prospective_companion_decisions_in_order": [
            "FULL_CASE_TYPE_X_MARGIN_JOINT_MATRIX",
            "FIVE_CONTROL_EXACT_ORACLES",
            "TWO_PRIVATE_SCALE_CLASS_SEMANTICS",
            "FIXED_SEMANTIC_DESCRIPTOR_TOKEN_SLOT_SHAPE",
            "PRESERVATION_BASE_SELECTION_NO_REUSE_AND_OPERATION_MAPPING",
            "CANONICAL_LATENT_EXACT_AND_STRUCTURAL_DISJOINTNESS",
            "CHALLENGE_DESCRIPTOR_AND_R08_EVALUATION_BINDING",
            "CONSTANT_ELIGIBILITY_MULTIPLICITIES",
            "VALIDATION_AND_FUTURE_RECEIPT_INTERFACES",
            "FIVE_FRESH_SECRET_COMMITMENT_AND_CUSTODY_CONTRACT",
            "PRIVATE_PRIMARY_PUBLIC_VARIANT_UUID_THREE_DOMAIN_DISJOINTNESS",
            "SYNTHETIC_DEV_ONLY_R06_QUALIFICATION_INTERFACE",
            "CLOSED_TYPED_R14_GENERATION_RECEIPT_AND_CROSS_ATTEMPT_REPLAY_REJECTION",
            "ATOMIC_SEVEN_KEY_DURABLE_STORE_TRANSACTION_AND_FINAL_ROOT_SIGNATURES",
            "FIVE_PER_PURPOSE_PLUS_FULL_SET_SINGLE_WINNER_CONSUMPTION_CAS",
            "PARTIAL_SECRET_REUSE_CARDINALITY_ONE_THROUGH_FIVE_REJECTION",
            "PURPOSE_KEY_REGISTRY_EPOCH_PUBLIC_KEY_REVOCATION_AND_DISTINCTNESS",
            "R06_QUALIFICATION_CACHE_NON_EVIDENCE_AND_MANDATORY_FRESH_RERUN",
        ],
        "section_provenance_in_order": [
            ["authorization_and_fresh_secret_contract", "OWNER_AUTHORIZATION_DERIVED_PLUS_NEW_EXECUTION_COMPANION"],
            ["canonicalization", "NEW_PROFILE_TRANSPORT_MECHANIC"],
            ["challenge_descriptor", "EXISTING_240_QUOTA_PLUS_NEW_DESCRIPTOR_R08_BINDING"],
            ["claim_boundary", "NEW_CONSERVATIVE_PROFILE_CLAIM_BOUNDARY"],
            ["corpus_contract", "EXISTING_QUOTAS_PLUS_NEW_JOINT_MATRIX_AND_ELIGIBILITY"],
            ["dag_contract", "EXISTING_READINESS_DAG_BINDING"],
            ["disjointness_contract", "NEW_PROSPECTIVE_COMPANION_DECISION"],
            [
                "generation_receipt_schema",
                "NEW_FUTURE_TWO_STAGE_ATOMIC_SEVEN_KEY_CAS_FINAL_ROOT_AND_KEY_REGISTRY_SCHEMA",
            ],
            ["oracle_contracts", "EXISTING_MARGIN_AND_REGRET_LIMITS_PLUS_NEW_CONSTRUCTIVE_ORACLES"],
            ["preservation_manifest", "EXISTING_APPLICABILITY_TOTALS_PLUS_NEW_MAPPING_AND_BASE_SCHEDULE"],
            ["private_uuid_projection", "WIRE_V3_MECHANIC_PLUS_NEW_GENERATOR_PREREQUISITE"],
            [
                "r06_qualification_contract",
                "NEW_SYNTHETIC_DEV_NON_EVIDENCE_CACHE_AND_MANDATORY_FRESH_RERUN_SCHEMA",
            ],
            ["semantic_descriptor", "NEW_PROSPECTIVE_COMPANION_DECISION"],
            ["uuid_domain_contract", "NEW_PROSPECTIVE_RULE_AND_REQUIRED_WIRE_DEPENDENCY"],
            ["version_bindings", "EXISTING_FROZEN_CONTENT_IDENTITIES"],
            [
                "wire_binding",
                "FINAL_WIRE_DEPENDENCY_PINS_VERIFIED_STATIC_GO_REPORTED_NOT_R03_R04_R05_R06",
            ],
        ],
        "wire_v3_implemented_mechanics_in_order": [
            "FORMAL_SEMANTIC_DESCRIPTOR_V3_REQUIRED_ON_FORMAL_CASE_GRAPH_V3",
            "EXACT_TEN_NAMESPACE_PRIVATE_SOURCE_TO_PUBLIC_UUID_MAPPING",
            "CANONICAL_PRIVATE_LABEL_SOURCE_ORDER_PROJECTION_APIS",
            "NO_TRUTH_OR_ORIGIN_CLAIM_FROM_PROJECTION",
        ],
    }


def _expected_semantic_descriptor_v1() -> dict[str, object]:
    return {
        "decision_status": "NEW_PROSPECTIVE_COMPANION_DECISION",
        "descriptor_kind": "natural_language",
        "family_lexicons_in_order": [
            {"family": family, "tokens_in_order": list(tokens)}
            for family, tokens in zip(_FAMILIES_V1, _FAMILY_LEXICONS_V1, strict=True)
        ],
        "fixed_neutral_segments_in_order": list(_NEUTRAL_TOKENS_V1[:6]),
        "forbidden_metadata_in_descriptor": [
            "binding",
            "case_type",
            "challenge_label",
            "family",
            "filename",
            "margin",
            "order",
            "scale",
            "stratum",
            "truth",
        ],
        "neutral_token_bank_in_order": list(_NEUTRAL_TOKENS_V1),
        "overlap_formula": "CARDINALITY_SET_TOKENS_INTERSECTION_FAMILY_LEXICON_DIVIDED_BY_6",
        "padding_codepoint": "U+0020",
        "padding_only_u0020": True,
        "right_trim_policy": "RIGHT_STRIP_U+0020_ONLY",
        "segment_count": 6,
        "segment_payload_utf8_bytes": 144,
        "segment_token_pattern": "LOWERCASE_ASCII_A_TO_Z_PLUS",
        "segment_utf8_bytes": 24,
    }


def _expected_challenge_descriptor_v1() -> dict[str, object]:
    return {
        "case_count": 240,
        "decision_status": "EXISTING_240_QUOTA_PLUS_NEW_DESCRIPTOR_AND_R08_BINDING",
        "family_scale_cells": 12,
        "footprint_counts_per_ten_rows": [
            ["P2_PAIR", 3],
            ["P3_CHAIN", 3],
            ["P4_STAR", 2],
            ["PSET_AGGREGATE", 2],
        ],
        "high_overlap_structural_negative": {
            "descriptor_tokens": "ALL_SIX_TARGET_FAMILY_TOKENS",
            "expected_decision": "ABSTAIN",
            "exact_overlap": "6/6",
            "margin_contract": "CLEAR_INTERIOR_M_REJECT_AT_LEAST_3",
            "per_family_scale_cell": 10,
            "structural_oracle": "ALL_CANDIDATES_FAIL",
        },
        "included_in_main_accuracy_denominator": False,
        "low_overlap_structural_positive": {
            "descriptor_tokens": "FIXED_FIRST_SIX_NEUTRAL_TOKENS",
            "expected_decision": "ANSWER",
            "exact_overlap": "0/6",
            "margin_contract": "CLEAR_INTERIOR_M_SELECT_AT_LEAST_3",
            "per_family_scale_cell": 10,
            "structural_oracle": "EXACT_ONE_PASSING_FAMILY_BINDING_SCALE_GROUP",
        },
        "same_freeze_and_reveal_as_main": True,
        "r08_evaluation_binding": {
            "evaluation_owner_node": "R08_STATISTICAL_VALIDATION_PASSED",
            "generator_profile_defines_thresholds": False,
            "generator_profile_evaluates_bootstrap": False,
            "generator_profile_evaluates_wilson": False,
            "required_content_addresses_in_order": [
                "r08_validation_version_receipt_root",
                "challenge_threshold_table_root",
                "one_sided_wilson_contract_root",
                "paired_cluster_bootstrap_contract_root",
            ],
            "required_exact_freeze_bootstrap_reference": (
                "PAIRED_CLUSTER_BOOTSTRAP_10000_LATENT_BASE_CASE_"
                "ONE_SIDED_95_PERCENTILE"
            ),
            "status": "FUTURE_R08_EXACT_BINDING_REQUIRED_NOT_PRESENT",
        },
        "status": "DESCRIPTOR_SCHEMA_ONLY_NOT_MATERIALIZED_NOT_SCORED",
        "threshold_tuning_allowed": False,
    }


def _expected_private_uuid_projection_v1() -> dict[str, object]:
    return {
        "batch_api": "project_private_source_labels_to_public_order_v3",
        "canonical_source_label_bytes_required": True,
        "cross_namespace_rejected": True,
        "cross_row_rejected": True,
        "decision_status": "WIRE_V3_IMPLEMENTED_MECHANIC_REQUIRED_BY_NEW_COMPANION",
        "exact_namespace_coverage": 10,
        "foreign_namespace_rejected": True,
        "fresh_public_envelope_crosscheck_required": True,
        "global_injectivity_required": True,
        "mapping_row_dto": "FormalPrivateUuidMappingV3",
        "mapping_rows_field": "source_to_public_uuid_rows",
        "namespaces_in_order": list(_UUID_NAMESPACES_V1),
        "origin_authenticated_by_projection": False,
        "single_row_api": "project_private_source_label_to_public_v3",
        "stale_mapping_rejected": True,
        "truth_authenticated_by_projection": False,
    }


def _expected_oracle_contracts_v1() -> dict[str, object]:
    return {
        "atomic_failure": {
            "actual": False,
            "partial_ids": False,
            "partial_receipt": False,
            "partial_roots": False,
            "partial_rows": False,
        },
        "binding_archetypes_in_order": [
            ["B0", [0, 1, 2, 3, 4]],
            ["B1", [1, 0, 2, 3, 4]],
            ["B2", [1, 2, 0, 3, 4]],
            ["B3", [3, 2, 1, 0, 4]],
        ],
        "binding_schedule": "(FLOOR(r/4)+(r MOD 4)+family_index+2*scale_index) MOD 4",
        "candidate_grid_maximum": 50000,
        "closest_negative_separation_minimum": "1/1024",
        "constructive_template_trials_per_slot_maximum": 4096,
        "control_oracles_in_order": [
            ["wrong_family_hard_negative", "ALL_FAIL_UNIQUE_CLOSEST_CYCLIC_NEXT_FAMILY"],
            ["binding_counterfactual", "ALL_FAIL_UNIQUE_CLOSEST_SUCCESSOR_BINDING"],
            ["scale_counterfactual", "ALL_FAIL_UNIQUE_CLOSEST_OPPOSITE_SCALE"],
            ["sign_or_invariant_break", "ALL_FAIL_EXACTLY_ONE_SIGN_OR_INVARIANT_CLAUSE_BROKEN"],
            ["insufficient_or_nonidentifiable", "EXACT_TWO_EQUAL_PASSING_GROUPS_DIFFERENT_FAMILY_OR_BINDING"],
        ],
        "error_or_inconclusive_accepted": False,
        "footprint_counts_per_main_cell": [
            ["P2_PAIR", 15],
            ["P3_CHAIN", 15],
            ["P4_STAR", 15],
            ["PSET_AGGREGATE", 15],
        ],
        "footprint_schedule": "FOOTPRINT[(r+family_index+scale_index) MOD 4]",
        "m_reject_formula": "MIN_ALL_ELIGIBLE_CANDIDATE_NORMALIZED_LOWER_MINUS_1",
        "m_select_formula": "MIN_NONSELECTED_NORMALIZED_LOWER_MINUS_SELECTED_NORMALIZED_UPPER",
        "margin_intervals": [
            ["near_boundary_identifiable", "1/4", "1", "LOWER_CLOSED_UPPER_OPEN"],
            ["moderate", "1", "3", "LOWER_CLOSED_UPPER_OPEN"],
            ["clear_interior", "3", None, "LOWER_CLOSED"],
            ["nonunique_or_insufficient", "0", "0", "EXACT_TIE"],
        ],
        "rational_denominator_maximum": 1024,
        "scale_oracle": {
            "controls_and_challenge_in_main_regret_denominator": False,
            "loss_kind": "EXACT_NONNEGATIVE_RATIONAL_PER_SCALE",
            "normalizer": "EXACT_STRICTLY_POSITIVE_RATIONAL",
            "normalized_regret_formula": "(CHOSEN_LOSS-ORACLE_MIN_LOSS)/NORMALIZER",
            "oracle_scale_set": "EXACT_ARGMIN_OF_S0_AND_S1_LOSS_ROWS",
            "point_estimate_maximum": "0.05",
            "set_valued_joint_exact_still_required": True,
            "one_sided_bootstrap_upper_bound_maximum": "0.08",
        },
        "scale_semantics": {
            "S0_FINE_LOCAL_is_root": True,
            "S1_COARSE_AGGREGATE_is_root": False,
            "legal_target_bijection": "S0_TO_T0_AND_S1_TO_T1",
            "scale_swap_allowed": False,
            "source_scale_record_count": 2,
        },
        "uncertainty": {
            "allowed_kinds": ["absolute_bound"],
            "standard_error_encounter_policy": "ABORT_WHOLE_PROFILE_BUILD_NO_PARTIAL_OUTPUT",
            "standard_error_status": "STANDARD_ERROR_UNSUPPORTED",
        },
    }


def _expected_nontrivial_scale_map_schedule_v1() -> dict[str, object]:
    return {
        "decision_status": "NEW_PROSPECTIVE_COMPANION_DECISION",
        "invalid_controls_per_family_scale": 1,
        "legal_pairs_per_family": 10,
        "legal_pairs_per_family_scale": 5,
        "operation_counts_per_family": {
            "coarse_graining": 2,
            "sampling_resolution": 2,
            "spatial_aggregation": 3,
            "temporal_aggregation": 3,
        },
        "ordinal_rows_in_order": [
            {
                "base_scale": "S0_FINE_LOCAL",
                "operation": "temporal_aggregation",
                "ordinal": 0,
                "parameter_profile": "SUM_EXACT_PARTITION_GROUP_2",
                "target_scale": "T0_FOR_S0",
            },
            {
                "base_scale": "S0_FINE_LOCAL",
                "operation": "spatial_aggregation",
                "ordinal": 1,
                "parameter_profile": "WEIGHTED_MEAN_EXACT_PARTITION_GROUP_2_COEFFICIENTS_1_OVER_2_1_OVER_2",
                "target_scale": "T0_FOR_S0",
            },
            {
                "base_scale": "S0_FINE_LOCAL",
                "operation": "sampling_resolution",
                "ordinal": 2,
                "parameter_profile": "TEMPORAL_GRID_STRIDE_2_OFFSET_0_VALID_ONLY",
                "target_scale": "T0_FOR_S0",
            },
            {
                "base_scale": "S0_FINE_LOCAL",
                "operation": "coarse_graining",
                "ordinal": 3,
                "parameter_profile": "SUM_EXACT_PARTITION_QUOTIENT_GROUP_2",
                "target_scale": "T0_FOR_S0",
            },
            {
                "base_scale": "S0_FINE_LOCAL",
                "operation": "temporal_aggregation",
                "ordinal": 4,
                "parameter_profile": "WEIGHTED_MEAN_EXACT_PARTITION_GROUP_3_COEFFICIENTS_1_OVER_4_1_OVER_2_1_OVER_4",
                "target_scale": "T0_FOR_S0",
            },
            {
                "base_scale": "S1_COARSE_AGGREGATE",
                "operation": "temporal_aggregation",
                "ordinal": 5,
                "parameter_profile": "SUM_EXACT_PARTITION_GROUP_3",
                "target_scale": "T1_FOR_S1",
            },
            {
                "base_scale": "S1_COARSE_AGGREGATE",
                "operation": "spatial_aggregation",
                "ordinal": 6,
                "parameter_profile": "SUM_EXACT_PARTITION_GROUP_2",
                "target_scale": "T1_FOR_S1",
            },
            {
                "base_scale": "S1_COARSE_AGGREGATE",
                "operation": "sampling_resolution",
                "ordinal": 7,
                "parameter_profile": "SPATIAL_GRID_STRIDE_2_OFFSET_1_VALID_ONLY",
                "target_scale": "T1_FOR_S1",
            },
            {
                "base_scale": "S1_COARSE_AGGREGATE",
                "operation": "coarse_graining",
                "ordinal": 8,
                "parameter_profile": "WEIGHTED_MEAN_EXACT_PARTITION_QUOTIENT_GROUP_2_COEFFICIENTS_1_OVER_3_2_OVER_3",
                "target_scale": "T1_FOR_S1",
            },
            {
                "base_scale": "S1_COARSE_AGGREGATE",
                "operation": "spatial_aggregation",
                "ordinal": 9,
                "parameter_profile": "WEIGHTED_MEAN_EXACT_PARTITION_GROUP_3_COEFFICIENTS_1_OVER_4_1_OVER_2_1_OVER_4",
                "target_scale": "T1_FOR_S1",
            },
        ],
        "parameter_profile_semantics": {
            "all_coefficients_exact_rationals": True,
            "all_group_memberships_nonempty_and_exact": True,
            "boundary_policy_for_aggregation_and_coarse": "EXACT_PARTITION",
            "missing_policy": "REJECT",
            "sampling_boundary_policy": "VALID_ONLY",
            "source_to_target_bijection": "S0_TO_T0_AND_S1_TO_T1",
        },
    }


def _expected_preservation_manifest_v1() -> dict[str, object]:
    return {
        "base_permutation": {
            "formula": "(17*k+11*family_index+7*scale_index) MOD 60",
            "legal_then_one_invalid_per_applicable_transform_and_cell": True,
            "maximum_consumed_slots_by_family_in_order": [45, 45, 52, 45, 54, 45],
            "no_base_reuse_within_family_scale_cell": True,
            "transform_order": [row[0] for row in _PRESERVATION_ROWS_V1],
        },
        "decision_status_applicability_and_totals": "DERIVED_FROM_EXISTING_FREEZE",
        "decision_status_mapping_base_and_mutations": "NEW_PROSPECTIVE_COMPANION_DECISION",
        "future_manifest_fields_in_order": [
            "base_source_row_reference",
            "variant_source_row_reference",
            "preservation_taxonomy",
            "wire_operation_or_private_witness",
            "legal_or_invalid",
            "expected_projected_answer",
            "exact_oracle_disposition",
        ],
        "invalid_mutations_in_transform_order": [list(row) for row in _INVALID_MUTATIONS_V1],
        "invalid_total": 76,
        "legal_total": 496,
        "manifest_count": 572,
        "manifest_status": "FUTURE_REQUIRED_NOT_ISSUED",
        "scale_operation_distribution_per_family": {
            "coarse_graining": 2,
            "sampling_resolution": 2,
            "spatial_aggregation": 3,
            "temporal_aggregation": 3,
        },
        "nontrivial_scale_map_schedule": _expected_nontrivial_scale_map_schedule_v1(),
        "transform_rows": _preservation_rows_json_v1(),
        "wire_operation_enum_equals_preservation_taxonomy": False,
    }


def _expected_authorization_and_fresh_secret_contract_v1() -> dict[str, object]:
    return {
        "authorization_binding": {
            "authorization_config_sha256": _OWNER_AUTHORIZATION_CONFIG_SHA256_V1,
            "authorization_id": _OWNER_AUTHORIZATION_ID_V1,
            "authorization_source_sha256": _OWNER_AUTHORIZATION_SOURCE_SHA256_V1,
            "binding_scope": "PROSPECTIVE_CONDITIONAL_CONSTRUCTION_ONLY",
            "live_active_and_not_revoked_verified_by_profile": False,
            "owner_identity_or_signature_authenticated_by_profile": False,
        },
        "custody_and_exposure": {
            "answer_manifest_content_allowed_for_baselines": False,
            "answer_manifest_content_allowed_for_recognizer": False,
            "answer_manifest_plaintext_allowed_for_baselines": False,
            "answer_manifest_plaintext_allowed_for_recognizer": False,
            "answer_manifest_plaintext_allowed_for_scorer_before_prediction_commitment": False,
            "baseline_allowed_inputs_in_order": [
                "AUDITED_PUBLIC_WIRE_ENVELOPES",
                "ITS_EXACT_FROZEN_REGISTRATION_ARTIFACTS",
            ],
            "custodian_plaintext_classes_in_order": [
                "GENERATOR_SOURCE_MANIFEST",
                "FIVE_FRESH_SECRET_MANIFEST",
                "PRIVATE_LATENT_CASES",
                "PRIVATE_ANSWER_ALIGNMENT_MANIFEST",
            ],
            "generator_source_archive_or_bytes_allowed_for_baselines": False,
            "generator_source_archive_or_bytes_allowed_for_recognizer": False,
            "generator_source_manifest_allowed_for_baselines": False,
            "generator_source_manifest_allowed_for_recognizer": False,
            "recognizer_allowed_input": "AUDITED_PUBLIC_WIRE_ENVELOPES_ONLY",
            "scorer_post_commit_reveal_allowed": "ANSWER_ALIGNMENT_OPENING_ONLY_NOT_SOURCE_OR_SEED_MANIFESTS",
            "seed_manifest_content_allowed_for_baselines": False,
            "seed_manifest_content_allowed_for_recognizer": False,
            "seed_manifest_allowed_for_baselines": False,
            "seed_manifest_allowed_for_recognizer": False,
        },
        "fresh_secret_contract": {
            "commitment_domain_ascii_nul_terminated": "HEGEL/HM/R11/FRESH_SECRET_COMMITMENT/V1\\0",
            "commitment_formula": "SHA256(DOMAIN||U16BE(LEN(PURPOSE_UTF8))||PURPOSE_UTF8||RAW_32_BYTES)",
            "commitment_output": "sha256:PLUS_64_LOWER_HEX",
            "commitment_rows_fields_in_order": [
                "purpose",
                "commitment",
                "independent_source_attestation_root",
                "custody_receipt_root",
            ],
            "commitments_pairwise_distinct_required": True,
            "cross_purpose_same_commitment_rejected_before_store": True,
            "cross_purpose_same_raw_value_rejected_before_commitment": True,
            "commitments_cross_bound_to_campaign_and_attempt_at_r14": True,
            "commitments_frozen_at": "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
            "count": 5,
            "deterministic_replay_inputs_in_order": [
                "FROZEN_GENERATOR_SOURCE_ARCHIVE_ROOT",
                "FROZEN_DEPENDENCY_LOCK_ROOT",
                "FROZEN_RUNTIME_IMAGE_DIGEST",
                "EXACT_PROFILE_ID",
                "EXACT_WIRE_BINDING",
                "VERIFIED_LAUNCH_OVERLAY_RECEIPT_ROOT",
                "FRESH_CAMPAIGN_ID",
                "FRESH_ATTEMPT_ID",
                "ALL_FIVE_RAW_SECRETS_FROM_CUSTODY",
            ],
            "deterministic_replay_required": True,
            "fresh_generation_occurs_only_in_future_live_custody": True,
            "independent_source_attestation_required_per_purpose": True,
            "custody_receipt_required_per_purpose": True,
            "custody_attestation_must_bind_pairwise_raw_value_distinctness": True,
            "pairwise_distinct_raw_values_required": True,
            "plaintext_retained_only_by_custodian": True,
            "purposes_in_order": list(_FRESH_SECRET_PURPOSES_V1),
            "raw_bytes_each": 32,
            "statistical_independence_inferred_from_distinctness": False,
        },
        "independence_boundary": {
            "independent_human_actors_required": False,
            "organizational_independence_required": False,
            "same_admin_controller_disclosed": True,
            "technical_role_independence_live_verified_by_profile": False,
            "technical_role_independence_required_for_future_execution": True,
        },
        "status": "CONTRACT_ONLY_NO_SECRET_NO_COMMITMENT_NO_CUSTODY_EVIDENCE",
    }


def _expected_uuid_domain_contract_v1() -> dict[str, object]:
    return {
        "complete_private_source_set_definition": (
            "UNION_OF_EVERY_UUID_LEAF_FROM_EVERY_PRIVATE_SOURCE_GRAPH_"
            "ACROSS_ALL_TEN_NAMESPACES"
        ),
        "complete_primary_public_target_set_definition": (
            "UNION_OF_EVERY_PUBLIC_UUID_TARGET_IN_PRIMARY_960_"
            "SOURCE_TO_PUBLIC_MAPPING_ROWS"
        ),
        "complete_variant_public_target_set_definition": (
            "UNION_OF_EVERY_FRESH_TARGET_UUID_IN_ALL_572_VARIANT_"
            "FULL_DOMAIN_RENAMING_SCHEDULES"
        ),
        "decision_status": "NEW_PROSPECTIVE_COMPANION_DECISION_AND_REQUIRED_WIRE_DEPENDENCY",
        "global_rules": {
            "one_uuid_has_exactly_one_namespace": True,
            "primary_public_targets_globally_injective": True,
            "private_sources_globally_injective_across_cases": True,
            "variant_public_targets_globally_injective_across_all_variants": True,
            "variant_schedule_must_cover_every_base_public_uuid": True,
        },
        "namespaces_in_order": list(_UUID_NAMESPACES_V1),
        "pairwise_empty_intersections_in_order": [
            "PRIVATE_SOURCE_INTERSECT_PRIMARY_PUBLIC_TARGET",
            "PRIVATE_SOURCE_INTERSECT_VARIANT_PUBLIC_TARGET",
            "PRIMARY_PUBLIC_TARGET_INTERSECT_VARIANT_PUBLIC_TARGET",
        ],
        "per_namespace_and_global_disjointness_both_required": True,
        "primary_or_variant_target_may_reuse_any_private_or_public_uuid": False,
        "status": "CONTRACT_ONLY_NOT_VERIFIED_ON_ACTUAL_MATERIAL",
    }


def _expected_disjointness_contract_v1() -> dict[str, object]:
    return {
        "canonical_latent_case": {
            "identifier_free": True,
            "labels_truth_filename_ordinal_provenance_absent": True,
            "semantic_order_preserved": True,
            "unordered_records_sorted_by_semantic_key": True,
        },
        "decision_status": "NEW_PROSPECTIVE_COMPANION_DECISION",
        "exact_fingerprint": "SHA256(HEGEL/HM/R06/EXACT/V1\\0||RFC8785(CANONICAL_LATENT_WITH_DESCRIPTOR))",
        "forbidden_source_classes_in_order": [
            "PHASE2A_FIXTURES",
            "HISTORICAL_HOLDOUTS",
            "ALL_VALIDATION_ATTEMPTS",
            "QUALIFICATION_AND_KAT_SYNTHETIC_CASES",
            "SYNTHETIC_ANSWER_KEY_LATENT_CASES",
            "PRIOR_ACTUAL_CORPORA",
            "PREDICTION_OR_SCORER_DERIVED_CANDIDATE_ARTIFACTS",
        ],
        "manifest_parse_unknown_disposition": "NO_GO",
        "main_challenge_disjoint": True,
        "maximum_algorithmic_class": "O((N+D)LOG(N+D))",
        "primary_exact_duplicates_allowed": False,
        "primary_forbidden_manifest_intersection_allowed": False,
        "primary_structural_duplicates_allowed": False,
        "quadratic_pairwise_corpus_comparison_allowed": False,
        "structural_fingerprint": "SHA256(HEGEL/HM/R06/STRUCTURAL/V1\\0||RFC8785(CANONICAL_LATENT_DESCRIPTOR_CLEARED))",
        "variant_duplicate_rule": "ONLY_EXPLICIT_ONE_TO_ONE_PRESERVATION_PAIR_LINK_MAY_JUSTIFY_EXPECTED_EQUIVALENCE",
    }


def _expected_r06_qualification_contract_v1() -> dict[str, object]:
    return {
        "actual_corpus_input_allowed": False,
        "actual_identifier_or_secret_reuse_allowed": False,
        "claim_boundary": (
            "SYNTHETIC_DEV_NON_EVIDENCE_CACHE_AND_FRESH_RERUN_CONTRACT_ONLY_"
            "NOT_RUN_NOT_R06"
        ),
        "cache_evidence_policy": {
            "all_pass_fields_self_authenticate": False,
            "cache_receipt_is_admission_evidence": False,
            "cache_receipt_may_satisfy_r06": False,
            "cache_values_may_skip_or_parameterize_fresh_rerun": False,
            "content_address_or_signature_changes_cache_evidence_class": False,
            "forged_all_pass_cache_disposition": "IGNORE_AND_FRESH_RERUN",
            "fresh_admission_rerun_required": True,
        },
        "deterministic_replay": {
            "independent_fresh_process_replays": 2,
            "replay_outputs_required_byte_identical": True,
            "replay_roots_required_equal": True,
            "synthetic_public_seed_manifest_only": True,
        },
        "fixture_contracts_in_order": [
            ["SYNTHETIC_QUOTA_SHAPED_PRIMARY", 960, "ACCEPT"],
            ["SYNTHETIC_PRESERVATION_MANIFEST", 572, "ACCEPT"],
            ["SYNTHETIC_WIRE_CAPACITY", 1024, "ACCEPT"],
            ["SYNTHETIC_WIRE_CAPACITY_PLUS_ONE", 1025, "ATOMIC_REJECT"],
        ],
        "input_class": "SYNTHETIC_DEV_ONLY",
        "issuance_status": "FUTURE_REQUIRED_NOT_ISSUED",
        "issued_by_profile_verifier": False,
        "fresh_admission_rerun_contract": {
            "admission_controller_directly_observes_process_exit_and_resources": True,
            "admission_controller_must_not_trust_cache_values": True,
            "cache_receipt_may_replace_execution": False,
            "fresh_process_replays": 2,
            "fresh_rerun_occurs_inside_admission_decision_not_from_prior_cache": True,
            "future_admission_evidence_required": True,
            "future_admission_evidence_status": "FUTURE_REQUIRED_NOT_ISSUED",
            "pinned_execution_bindings_in_order": [
                "qualifier_identity_id",
                "qualifier_source_commit_sha",
                "qualifier_source_archive_root",
                "qualifier_dependency_lock_root",
                "qualifier_runtime_image_digest",
                "qualifier_entrypoint_sha256",
                "trusted_qualifier_registry_root",
                "admission_controller_key_id",
                "generator_source_commit_sha",
                "generator_source_archive_root",
                "generator_dependency_lock_root",
                "generator_runtime_image_digest",
                "synthetic_fixture_root",
            ],
            "recompute_all_kat_groups": True,
            "rerun_input_class": "SYNTHETIC_DEV_ONLY",
            "trusted_qualifier_identity_must_be_pinned_before_rerun": True,
        },
        "kat_groups_in_order": [
            "CANONICAL_PROFILE_AND_PROFILE_ID",
            "QUOTA_MARGIN_AND_ELIGIBILITY_CLOSURE",
            "FIVE_CONTROL_ORACLES",
            "SEMANTIC_DESCRIPTOR_FIXED_WIDTH",
            "STANDARD_ERROR_WHOLE_BATCH_REJECT",
            "PRIVATE_LABEL_PROJECTION_EXACT_TEN_NAMESPACES",
            "GLOBAL_UUID_THREE_DOMAIN_DISJOINTNESS",
            "EIGHT_PRESERVATION_TRANSFORMS_LEGAL_AND_INVALID",
            "NONTRIVIAL_SCALE_MAP_5_5_AND_3_3_2_2",
            "FORBIDDEN_SOURCE_DISJOINTNESS",
            "DETERMINISTIC_TWO_PROCESS_REPLAY",
            "CLOSED_TWO_STAGE_SEVEN_KEY_RECEIPT_PARTIAL_REUSE_REGISTRY_AND_SPLICE_ATTACKS",
            "QUALIFICATION_CACHE_NON_EVIDENCE_AND_FRESH_RERUN",
            "RESOURCE_LIMIT_AND_ATOMIC_FAILURE",
            "PUBLIC_SURFACE_AND_EFFECT_CAPABILITY_GUARD",
        ],
        "qualification_cache_schema": {
            "body_fields_in_order": [
                "schema_version",
                "evidence_class",
                "generator_construction_profile_id",
                "owner_authorization_id",
                "r05_qualification_receipt_root",
                "source_commit_sha",
                "source_archive_root",
                "dependency_lock_sha256",
                "runtime_image_digest",
                "synthetic_fixture_root",
                "synthetic_seed_manifest_root",
                "first_replay_root",
                "second_replay_root",
                "kat_report_root",
                "resource_report",
                "tests_collected",
                "tests_passed",
                "tests_failed",
                "actual_input_count",
                "atomic_success",
            ],
            "body_field_types": {
                "actual_input_count": "EXACT_INTEGER_ZERO",
                "atomic_success": "EXACT_TRUE",
                "dependency_lock_sha256": "CONTENT_ADDRESS",
                "evidence_class": "EXACT_NON_EVIDENCE_CACHE_LITERAL",
                "first_replay_root": "CONTENT_ADDRESS",
                "generator_construction_profile_id": "EXACT_PROFILE_ID",
                "kat_report_root": "CONTENT_ADDRESS",
                "owner_authorization_id": "EXACT_OWNER_AUTHORIZATION_ID",
                "r05_qualification_receipt_root": "CONTENT_ADDRESS",
                "resource_report": "CLOSED_OBJECT",
                "runtime_image_digest": "CONTENT_ADDRESS",
                "schema_version": "EXACT_LITERAL_STRING",
                "second_replay_root": "CONTENT_ADDRESS",
                "source_archive_root": "CONTENT_ADDRESS",
                "source_commit_sha": "LOWER_HEX_40_NONZERO",
                "synthetic_fixture_root": "CONTENT_ADDRESS",
                "synthetic_seed_manifest_root": "CONTENT_ADDRESS",
                "tests_collected": "INTEGER_1_THROUGH_100000",
                "tests_failed": "EXACT_INTEGER_ZERO",
                "tests_passed": "INTEGER_1_THROUGH_100000_EQUAL_TESTS_COLLECTED",
            },
            "closed_nested_resource_report_fields": [
                "wall_seconds",
                "maximum_rss_bytes",
                "logical_cpu_count",
                "output_bytes",
            ],
            "closed_nested_resource_report_types": {
                "logical_cpu_count": "INTEGER_1_THROUGH_16",
                "maximum_rss_bytes": "INTEGER_1_THROUGH_12884901888",
                "output_bytes": "INTEGER_0_THROUGH_134217728",
                "wall_seconds": "INTEGER_0_THROUGH_1800",
            },
            "content_address_type": "sha256:PLUS_64_LOWER_HEX",
            "id_domain_ascii_nul_terminated": "HEGEL/HM/R06/GENERATOR_QUALIFICATION_CACHE/V1\\0",
            "id_formula": "PREFIX||SHA256(DOMAIN||RFC8785(CACHE_BODY))",
            "id_prefix": "phase2b_generator_r06_qualification_cache_v1_",
            "required_literals": {
                "actual_input_count": 0,
                "atomic_success": True,
                "evidence_class": "NON_EVIDENCE_CACHE_ONLY",
                "owner_authorization_id": _OWNER_AUTHORIZATION_ID_V1,
                "schema_version": "hegel-machine-phase2b-generator-r06-qualification-cache/1",
                "tests_failed": 0,
            },
            "self_asserted_all_pass_is_admission_evidence": False,
            "source_commit_type": "40_LOWER_HEX_NONZERO",
            "wrapper_fields_in_order": ["cache_body", "cache_id"],
        },
        "required_artifact_bindings_in_order": [
            "source_commit_sha",
            "source_archive_root",
            "dependency_lock_sha256",
            "runtime_image_digest",
            "synthetic_fixture_root",
        ],
        "required_predecessor": "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
        "r06_admission_can_consume_cache_without_fresh_rerun": False,
        "r06_satisfied_by_contract": False,
        "resource_limits": {
            "maximum_logical_cpus": 16,
            "maximum_output_bytes": 134217728,
            "maximum_rss_bytes": 12884901888,
            "maximum_wall_seconds": 1800,
        },
        "target_node": "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
        "transport": {
            "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
            "floats_or_nonfinite_allowed": False,
            "maximum_bytes": 131072,
            "maximum_depth": 32,
            "maximum_nodes": 20000,
            "maximum_string_utf8_bytes": 4096,
            "unknown_or_duplicate_fields_allowed": False,
        },
    }


def _expected_generation_receipt_schema_v1() -> dict[str, object]:
    return {
        "body_fields_in_order": [
            "schema_version",
            "generator_construction_profile_id",
            "owner_authorization_id",
            "readiness_dag_id",
            "launch_overlay_receipt_root",
            "campaign_id",
            "attempt_id",
            "artifact_bindings",
            "readiness_evidence_roots",
            "fresh_secret_commitments",
            "signature_key_registry",
            "private_partition",
            "public_partition",
            "preservation_manifest_root",
            "disjointness_manifest_root",
            "scale_regret_oracle_root",
            "deterministic_replay_root",
            "logical_time_chain",
            "standard_error_count",
            "atomic_success",
            "body_signed_root",
            "durable_store_transaction",
            "final_receipt_root",
            "purpose_signatures",
        ],
        "body_field_types": {
            "artifact_bindings": "CLOSED_OBJECT",
            "atomic_success": "EXACT_TRUE",
            "attempt_id": "phase2b_actual_unsealed_960_attempt_v2_PLUS_64_LOWER_HEX",
            "body_signed_root": "CONTENT_ADDRESS",
            "campaign_id": "phase2b_actual_unsealed_960_campaign_v2_PLUS_64_LOWER_HEX",
            "deterministic_replay_root": "CONTENT_ADDRESS",
            "disjointness_manifest_root": "CONTENT_ADDRESS",
            "durable_store_transaction": "CLOSED_OBJECT",
            "final_receipt_root": "CONTENT_ADDRESS",
            "fresh_secret_commitments": "EXACT_FIVE_ROW_ARRAY",
            "generator_construction_profile_id": "EXACT_PROFILE_ID",
            "launch_overlay_receipt_root": "CONTENT_ADDRESS",
            "logical_time_chain": "BOUNDED_CLOSED_ROW_ARRAY",
            "owner_authorization_id": "EXACT_OWNER_AUTHORIZATION_ID",
            "preservation_manifest_root": "CONTENT_ADDRESS",
            "private_partition": "CLOSED_OBJECT",
            "public_partition": "CLOSED_OBJECT",
            "purpose_signatures": "EXACT_FOUR_ROW_ARRAY",
            "readiness_dag_id": "EXACT_READINESS_DAG_ID",
            "readiness_evidence_roots": "EXACT_THIRTEEN_ROW_ARRAY",
            "scale_regret_oracle_root": "CONTENT_ADDRESS",
            "schema_version": "EXACT_LITERAL_STRING",
            "signature_key_registry": "CLOSED_OBJECT",
            "standard_error_count": "EXACT_INTEGER_ZERO",
        },
        "body_signed_root_covered_fields_in_order": [
            "schema_version",
            "generator_construction_profile_id",
            "owner_authorization_id",
            "readiness_dag_id",
            "launch_overlay_receipt_root",
            "campaign_id",
            "attempt_id",
            "artifact_bindings",
            "readiness_evidence_roots",
            "fresh_secret_commitments",
            "signature_key_registry",
            "private_partition",
            "public_partition",
            "preservation_manifest_root",
            "disjointness_manifest_root",
            "scale_regret_oracle_root",
            "deterministic_replay_root",
            "logical_time_chain",
            "standard_error_count",
            "atomic_success",
        ],
        "body_signed_root_formula": (
            "SHA256(HEGEL/HM/R14/BODY_SIGNED_ROOT/V1\\0||"
            "RFC8785(OBJECT_OF_EXACT_BODY_SIGNED_ROOT_COVERED_FIELDS))"
        ),
        "closed_nested_schemas": {
            "artifact_bindings": {
                "fields_in_order": [
                    "generator_source_commit_sha",
                    "generator_source_archive_root",
                    "generator_dependency_lock_root",
                    "generator_runtime_image_digest",
                    "generator_source_manifest_root",
                    "answer_manifest_root",
                    "seed_manifest_root",
                    "wire_policy_and_manifest_binding_root",
                ],
                "field_types": {
                    "answer_manifest_root": "CONTENT_ADDRESS",
                    "generator_dependency_lock_root": "CONTENT_ADDRESS",
                    "generator_runtime_image_digest": "CONTENT_ADDRESS",
                    "generator_source_archive_root": "CONTENT_ADDRESS",
                    "generator_source_commit_sha": "LOWER_HEX_40_NONZERO",
                    "generator_source_manifest_root": "CONTENT_ADDRESS",
                    "seed_manifest_root": "CONTENT_ADDRESS",
                    "wire_policy_and_manifest_binding_root": "CONTENT_ADDRESS",
                },
                "private_manifest_plaintexts_absent_from_public_receipt": True,
                "public_values_are_content_addresses_only": True,
            },
            "fresh_secret_commitments": {
                "commitments_pairwise_distinct_required": True,
                "cross_purpose_same_commitment_rejected_before_store": True,
                "cross_purpose_same_raw_value_rejected_before_commitment": True,
                "custody_attestation_must_bind_pairwise_raw_value_distinctness": True,
                "exact_count": 5,
                "fields_per_row_in_order": [
                    "purpose",
                    "commitment",
                    "independent_source_attestation_root",
                    "custody_receipt_root",
                ],
                "purposes_in_order": list(_FRESH_SECRET_PURPOSES_V1),
                "raw_secret_or_seed_allowed": False,
                "row_field_types": {
                    "commitment": "CONTENT_ADDRESS",
                    "custody_receipt_root": "CONTENT_ADDRESS",
                    "independent_source_attestation_root": "CONTENT_ADDRESS",
                    "purpose": "EXACT_PURPOSE_ENUM",
                },
            },
            "signature_key_registry": {
                "fields_in_order": [
                    "schema_version",
                    "epoch",
                    "revocation_snapshot_root",
                    "rows",
                    "registry_root",
                ],
                "field_types": {
                    "epoch": "INTEGER_1_THROUGH_2_POW_53_MINUS_1",
                    "registry_root": "CONTENT_ADDRESS",
                    "revocation_snapshot_root": "CONTENT_ADDRESS",
                    "rows": "EXACT_FOUR_ROW_ARRAY",
                    "schema_version": "EXACT_LITERAL_STRING",
                },
                "key_id_type": "phase2b_r14_purpose_signing_key_v1_PLUS_64_LOWER_HEX",
                "launch_overlay_must_pin_epoch_registry_and_revocation_snapshot_roots": True,
                "private_signing_key_allowed_in_receipt": False,
                "public_key_hex_bytes": 32,
                "registry_root_formula": (
                    "SHA256(HEGEL/HM/R14/SIGNATURE_KEY_REGISTRY/V1\\0||"
                    "RFC8785({schema_version,epoch,revocation_snapshot_root,rows}))"
                ),
                "registry_root_must_be_freshly_recomputed": True,
                "required_schema_version": (
                    "hegel-machine-phase2b-r14-purpose-signature-key-registry/1"
                ),
                "revocation_check": (
                    "PINNED_SNAPSHOT_PLUS_ONLINE_NOT_REVOKED_RECHECK_AFTER_R13_"
                    "BEFORE_FINAL_SIGNATURES"
                ),
                "status_change_after_overlay_invalidates_attempt": True,
                "row_fields_in_order": [
                    "purpose",
                    "key_id",
                    "public_key_hex",
                    "status",
                    "revocation_record_root",
                ],
                "row_field_types": {
                    "key_id": "PURPOSE_SEPARATED_REGISTRY_KEY_ID",
                    "public_key_hex": "LOWER_HEX_64",
                    "purpose": "EXACT_PURPOSE_ENUM",
                    "revocation_record_root": "CONTENT_ADDRESS",
                    "status": "EXACT_ACTIVE_NOT_REVOKED",
                },
                "row_purposes_in_order": [
                    "GOVERNANCE_ANCHOR_EXECUTION_FREEZE",
                    "CUSTODIAN_GENERATION",
                    "STORE_IF_ABSENT_CAS_RECEIPT",
                    "TIME_ATTESTOR_LOGICAL_CHAIN",
                ],
                "row_key_ids_pairwise_distinct": True,
                "row_public_keys_pairwise_distinct": True,
            },
            "logical_time_chain": {
                "fields_per_row_in_order": [
                    "sequence",
                    "event",
                    "event_root",
                    "previous_token_root",
                    "time_token_root",
                ],
                "row_field_types": {
                    "event": "EXACT_EVENT_ENUM",
                    "event_root": "CONTENT_ADDRESS",
                    "previous_token_root": "NULL_FOR_ZERO_ELSE_CONTENT_ADDRESS",
                    "sequence": "INTEGER_0_THROUGH_15",
                    "time_token_root": "CONTENT_ADDRESS",
                },
                "minimum_prefix_events_in_order": [
                    "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
                    "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
                    "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
                    "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",
                ],
                "semantics": "SIGNED_MONOTONIC_LOGICAL_ORDER_NOT_UTC_AUTHORITY",
                "strictly_contiguous_from_zero": True,
            },
            "private_partition": {
                "fields_in_order": [
                    "main_count",
                    "challenge_count",
                    "primary_count",
                    "private_main_latent_root",
                    "private_challenge_latent_root",
                    "private_primary_latent_root",
                    "private_answer_alignment_root",
                ],
                "field_types": {
                    "challenge_count": "EXACT_INTEGER_240",
                    "main_count": "EXACT_INTEGER_720",
                    "primary_count": "EXACT_INTEGER_960",
                    "private_answer_alignment_root": "CONTENT_ADDRESS",
                    "private_challenge_latent_root": "CONTENT_ADDRESS",
                    "private_main_latent_root": "CONTENT_ADDRESS",
                    "private_primary_latent_root": "CONTENT_ADDRESS",
                },
                "required_counts": [720, 240, 960],
                "primary_root_formula": (
                    "SHA256(HEGEL/HM/R14/PRIVATE_PRIMARY_PARTITION/V1\\0||"
                    "RFC8785([[MAIN,720,PRIVATE_MAIN_ROOT],[CHALLENGE,240,PRIVATE_CHALLENGE_ROOT]]))"
                ),
            },
            "public_partition": {
                "fields_in_order": [
                    "main_count",
                    "challenge_count",
                    "primary_count",
                    "preservation_count",
                    "public_main_wire_root",
                    "public_challenge_wire_root",
                    "public_primary_wire_root",
                    "public_preservation_variant_root",
                ],
                "field_types": {
                    "challenge_count": "EXACT_INTEGER_240",
                    "main_count": "EXACT_INTEGER_720",
                    "preservation_count": "EXACT_INTEGER_572",
                    "primary_count": "EXACT_INTEGER_960",
                    "public_challenge_wire_root": "CONTENT_ADDRESS",
                    "public_main_wire_root": "CONTENT_ADDRESS",
                    "public_preservation_variant_root": "CONTENT_ADDRESS",
                    "public_primary_wire_root": "CONTENT_ADDRESS",
                },
                "required_counts": [720, 240, 960, 572],
                "primary_root_formula": (
                    "SHA256(HEGEL/HM/R14/PUBLIC_PRIMARY_PARTITION/V1\\0||"
                    "RFC8785([[MAIN,720,PUBLIC_MAIN_ROOT],[CHALLENGE,240,PUBLIC_CHALLENGE_ROOT]]))"
                ),
            },
            "durable_store_transaction": {
                "atomic_store_request_root_formula": (
                    "SHA256(HEGEL/HM/R14/ATOMIC_STORE_REQUEST/V1\\0||"
                    "RFC8785({schema_version,attempt_state_transition,"
                    "NORMALIZE(per_purpose_commitment_consumptions,"
                    "full_commitment_set_consumption)}))"
                ),
                "atomic_store_request_root_normalization": (
                    "OMIT_ONLY_WINNING_FINAL_RECEIPT_ROOT_FROM_EXACT_FIVE_"
                    "PER_PURPOSE_ROWS_AND_ONE_FULL_SET_ROW"
                ),
                "atomic_store_request_normalized_object_fields_in_order": [
                    "schema_version",
                    "attempt_state_transition",
                    "per_purpose_commitment_consumptions_without_winning_final_receipt_root",
                    "full_commitment_set_consumption_without_winning_final_receipt_root",
                ],
                "atomic_store_receipt_must_bind_request_root_and_committed_all_seven": True,
                "atomic_store_receipt_root_formula": (
                    "SHA256(HEGEL/HM/R14/ATOMIC_STORE_RECEIPT/V1\\0||"
                    "ATOMIC_STORE_REQUEST_ROOT_BYTES||U16BE(TRANSACTION_ID_LEN)||"
                    "TRANSACTION_ID_UTF8||RFC8785({store_status,exact_key_count}))"
                ),
                "attempt_state_transition": {
                    "attempt_state_key_formula": (
                        "SHA256(HEGEL/HM/R14/ATTEMPT_STATE_KEY/V1\\0||"
                        "U16BE(AUTHORIZATION_ID_LEN)||AUTHORIZATION_ID_UTF8||"
                        "U16BE(PROFILE_ID_LEN)||PROFILE_ID_UTF8||"
                        "LAUNCH_OVERLAY_ROOT_BYTES||U16BE(CAMPAIGN_ID_LEN)||"
                        "CAMPAIGN_ID_UTF8||U16BE(ATTEMPT_ID_LEN)||ATTEMPT_ID_UTF8)"
                    ),
                    "field_types": {
                        "attempt_state_key": "CONTENT_ADDRESS",
                        "expected_generation": "EXACT_INTEGER_ZERO",
                        "new_generation": "EXACT_INTEGER_ONE",
                        "new_unsigned_receipt_root": (
                            "CONTENT_ADDRESS_EQUAL_TOP_LEVEL_BODY_SIGNED_ROOT"
                        ),
                        "previous_value": "EXACT_NULL",
                    },
                    "fields_in_order": [
                        "attempt_state_key",
                        "expected_generation",
                        "new_generation",
                        "previous_value",
                        "new_unsigned_receipt_root",
                    ],
                    "new_unsigned_receipt_root_exactly_equals": "BODY_SIGNED_ROOT",
                    "attempt_state_key_must_recompute_from_top_level_scope": True,
                    "required_expected_generation": 0,
                    "required_new_generation": 1,
                    "required_previous_value": None,
                },
                "per_purpose_commitment_consumptions": {
                    "exact_count": 5,
                    "purposes_in_order": list(_FRESH_SECRET_PURPOSES_V1),
                    "fields_per_row_in_order": [
                        "purpose",
                        "commitment",
                        "commitment_state_key",
                        "expected_state",
                        "new_state",
                        "previous_winner",
                        "winning_campaign_id",
                        "winning_attempt_id",
                        "winning_body_signed_root",
                        "winning_final_receipt_root",
                    ],
                    "row_field_types": {
                        "commitment": (
                            "CONTENT_ADDRESS_EQUAL_CORRESPONDING_TOP_LEVEL_COMMITMENT"
                        ),
                        "commitment_state_key": "CONTENT_ADDRESS",
                        "expected_state": "EXACT_UNUSED",
                        "new_state": "EXACT_CONSUMED",
                        "previous_winner": "EXACT_NULL",
                        "purpose": "EXACT_PURPOSE_ENUM_IN_FROZEN_ORDER",
                        "winning_attempt_id": "EXACT_TOP_LEVEL_ATTEMPT_ID",
                        "winning_body_signed_root": (
                            "CONTENT_ADDRESS_EQUAL_TOP_LEVEL_BODY_SIGNED_ROOT"
                        ),
                        "winning_campaign_id": "EXACT_TOP_LEVEL_CAMPAIGN_ID",
                        "winning_final_receipt_root": (
                            "CONTENT_ADDRESS_EQUAL_TOP_LEVEL_FINAL_RECEIPT_ROOT"
                        ),
                    },
                    "commitment_state_key_formula": (
                        "SHA256(HEGEL/HM/R14/PER_PURPOSE_COMMITMENT_STATE_KEY/V1\\0||"
                        "U16BE(PURPOSE_LEN)||PURPOSE_UTF8||COMMITMENT_DIGEST_32_BYTES)"
                    ),
                    "purpose_and_commitment_must_equal_corresponding_top_level_row": True,
                    "state_key_must_recompute_from_purpose_and_commitment": True,
                    "state_key_excludes_authorization_profile_overlay_campaign_attempt": True,
                    "same_purpose_commitment_across_attempts_same_state_key": True,
                    "any_used_or_consumed_key_rejects_entire_transaction": True,
                    "overlap_cardinalities_rejected_in_order": [1, 2, 3, 4, 5],
                    "two_fresh_attempts_with_any_same_purpose_commitment_exactly_one_may_win": True,
                },
                "full_commitment_set_consumption": {
                    "commitment_set_id_formula": (
                        "SHA256(HEGEL/HM/R14/FIVE_COMMITMENT_SET/V1\\0||"
                        "RFC8785(EXACT_FIVE_[PURPOSE,COMMITMENT]_ROWS_IN_FROZEN_ORDER))"
                    ),
                    "commitment_set_state_key_formula": (
                        "SHA256(HEGEL/HM/R14/FIVE_COMMITMENT_SET_STATE_KEY/V1\\0||"
                        "COMMITMENT_SET_ID_BYTES)"
                    ),
                    "field_types": {
                        "commitment_set_id": "CONTENT_ADDRESS",
                        "commitment_set_state_key": "CONTENT_ADDRESS",
                        "purpose": "EXACT_FULL_FIVE_COMMITMENT_SET",
                        "winning_body_signed_root": (
                            "CONTENT_ADDRESS_EQUAL_TOP_LEVEL_BODY_SIGNED_ROOT"
                        ),
                        "expected_state": "EXACT_UNUSED",
                        "new_state": "EXACT_CONSUMED",
                        "previous_winner": "EXACT_NULL",
                        "winning_attempt_id": "EXACT_TOP_LEVEL_ATTEMPT_ID",
                        "winning_campaign_id": "EXACT_TOP_LEVEL_CAMPAIGN_ID",
                        "winning_final_receipt_root": (
                            "CONTENT_ADDRESS_EQUAL_TOP_LEVEL_FINAL_RECEIPT_ROOT"
                        ),
                    },
                    "fields_in_order": [
                        "purpose",
                        "commitment_set_id",
                        "commitment_set_state_key",
                        "expected_state",
                        "new_state",
                        "previous_winner",
                        "winning_campaign_id",
                        "winning_attempt_id",
                        "winning_body_signed_root",
                        "winning_final_receipt_root",
                    ],
                    "required_purpose": "FULL_FIVE_COMMITMENT_SET",
                    "same_five_commitments_across_attempts_same_state_key": True,
                    "commitment_set_id_must_recompute_from_top_level_fresh_secret_commitments": True,
                    "commitment_set_state_key_must_recompute_from_commitment_set_id": True,
                    "state_key_excludes_authorization_profile_overlay_campaign_attempt": True,
                    "state_key_must_differ_from_attempt_state_key": True,
                    "two_fresh_attempts_same_commitments_exactly_one_may_win": True,
                },
                "failure_semantics": (
                    "IF_ANY_ONE_OF_SEVEN_COMPARE_OR_FRESHNESS_CHECKS_FAILS_APPLY_"
                    "NONE_OF_SEVEN_MUTATIONS_AND_ISSUE_NO_STORE_RECEIPT_OR_FINAL_ROOT"
                ),
                "field_types": {
                    "atomic_store_receipt_root": "CONTENT_ADDRESS",
                    "atomic_store_request_root": "CONTENT_ADDRESS",
                    "attempt_state_transition": "CLOSED_OBJECT",
                    "full_commitment_set_consumption": "CLOSED_OBJECT",
                    "per_purpose_commitment_consumptions": "EXACT_FIVE_ROW_ARRAY",
                    "schema_version": "EXACT_LITERAL_STRING",
                    "store_status": "EXACT_COMMITTED_ALL_SEVEN",
                    "transaction_id": "FROZEN_PREFIX_PLUS_64_LOWER_HEX",
                    "transaction_root": "CONTENT_ADDRESS",
                },
                "fields_in_order": [
                    "schema_version",
                    "transaction_id",
                    "attempt_state_transition",
                    "per_purpose_commitment_consumptions",
                    "full_commitment_set_consumption",
                    "atomic_store_request_root",
                    "atomic_store_receipt_root",
                    "store_status",
                    "transaction_root",
                ],
                "required_schema_version": (
                    "hegel-machine-phase2b-r14-durable-store-transaction/1"
                ),
                "store_operation": "ATOMIC_SEVEN_KEY_COMPARE_AND_SWAP_ALL_OR_NONE",
                "exact_key_count": 7,
                "seven_state_keys_pairwise_distinct_required": True,
                "all_seven_preconditions_checked_before_any_mutation": True,
                "any_used_key_rejects_entire_transaction": True,
                "winner_binding_rows_count": 6,
                "winner_rows_bind_purpose_campaign_attempt_body_and_final_root": True,
                "winning_final_root_is_derived_and_omitted_only_from_request_root": True,
                "store_must_recompute_and_verify_all_six_winner_final_roots_before_commit": True,
                "derived_final_root_inserted_before_atomic_seven_key_commit": True,
                "computation_and_commit_order_in_order": [
                    "BODY_SIGNED_ROOT",
                    "NORMALIZED_ATOMIC_STORE_REQUEST_ROOT",
                    "TRANSACTION_ID",
                    "ATOMIC_STORE_RECEIPT_ROOT",
                    "TRANSACTION_ROOT",
                    "FINAL_RECEIPT_ROOT",
                    "INSERT_FINAL_ROOT_IN_SIX_WINNER_ROWS",
                    "VERIFY_ALL_SEVEN_CAS_PRECONDITIONS",
                    "ATOMIC_COMMIT_ALL_SEVEN",
                    "FRESH_READBACK_ALL_SEVEN",
                    "PURPOSE_SIGNATURES",
                ],
                "fresh_seven_key_readback_required_before_purpose_signatures": True,
                "deterministic_store_receipt_is_not_execution_evidence_without_store_signature_and_fresh_readback": True,
                "all_roots_and_ids_must_be_freshly_recomputed_before_commit": True,
                "transaction_id_formula": (
                    "phase2b_r14_durable_store_transaction_v1_||"
                    "SHA256(HEGEL/HM/R14/STORE_TRANSACTION_ID/V1\\0||"
                    "ATOMIC_STORE_REQUEST_ROOT_BYTES)"
                ),
                "transaction_root_formula": (
                    "SHA256(HEGEL/HM/R14/STORE_TRANSACTION_ROOT/V1\\0||"
                    "ATOMIC_STORE_REQUEST_ROOT_BYTES||ATOMIC_STORE_RECEIPT_ROOT_BYTES||"
                    "RFC8785({schema_version,transaction_id,store_status}))"
                ),
            },
            "purpose_signatures": {
                "algorithm": "Ed25519",
                "fields_per_row_in_order": [
                    "purpose",
                    "key_id",
                    "registry_epoch",
                    "registry_root",
                    "signed_final_receipt_root",
                    "signature_hex",
                ],
                "key_id_and_public_key_must_match_same_purpose_registry_row": True,
                "purpose_relabel_or_key_substitution_rejected": True,
                "row_field_types": {
                    "key_id": "PURPOSE_SEPARATED_REGISTRY_KEY_ID",
                    "purpose": "EXACT_PURPOSE_ENUM",
                    "registry_epoch": "EXACT_KEY_REGISTRY_EPOCH",
                    "registry_root": "CONTENT_ADDRESS_EQUAL_KEY_REGISTRY_ROOT",
                    "signature_hex": "LOWER_HEX_128",
                    "signed_final_receipt_root": (
                        "CONTENT_ADDRESS_EQUAL_TOP_LEVEL_FINAL_RECEIPT_ROOT"
                    ),
                },
                "purposes_in_order": [
                    "GOVERNANCE_ANCHOR_EXECUTION_FREEZE",
                    "CUSTODIAN_GENERATION",
                    "STORE_IF_ABSENT_CAS_RECEIPT",
                    "TIME_ATTESTOR_LOGICAL_CHAIN",
                ],
                "raw_signature_bytes": 64,
                "all_four_signatures_required_and_must_verify": True,
                "signed_preimage_domain_ascii_nul_terminated": (
                    "HEGEL/HM/R14/PURPOSE_FINAL_ROOT_SIGNATURE/V1\\0"
                ),
                "signed_preimage_formula": (
                    "DOMAIN||U16BE(PURPOSE_LEN)||PURPOSE_UTF8||"
                    "U16BE(KEY_ID_LEN)||KEY_ID_UTF8||U64BE(REGISTRY_EPOCH)||"
                    "REGISTRY_ROOT_BYTES||FINAL_RECEIPT_ROOT_BYTES"
                ),
                "signatures_cover_store_transcript_via_final_receipt_root": True,
            },
            "readiness_evidence_roots": {
                "exact_count": 13,
                "fields_per_row_in_order": ["node_id", "evidence_root"],
                "row_field_types": {
                    "evidence_root": "CONTENT_ADDRESS",
                    "node_id": "EXACT_NODE_ENUM_IN_CANONICAL_ORDER",
                },
                "nodes_in_order": [
                    "R01_AUTHORIZATION_VERIFIED",
                    "R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",
                    "R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",
                    "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",
                    "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
                    "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
                    "R07_BASELINE_REGISTRATIONS_FROZEN",
                    "R08_STATISTICAL_VALIDATION_PASSED",
                    "R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED",
                    "R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",
                    "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
                    "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
                    "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
                ],
            },
        },
        "content_address_type": "sha256:PLUS_64_LOWER_HEX",
        "cross_attempt_and_replay_rejection": {
            "attempt_scoped_nested_roots_bind_authorization_profile_overlay_campaign_attempt": True,
            "attempt_id_mismatch_rejected": True,
            "body_signed_root_mismatch_rejected": True,
            "campaign_id_mismatch_rejected": True,
            "cas_generation_other_than_0_to_1_rejected": True,
            "cas_transcript_or_store_receipt_splice_rejected": True,
            "commitment_set_state_key_including_any_execution_scope_rejected": True,
            "commitment_set_used_or_consumed_rejected": True,
            "component_from_prior_or_foreign_attempt_rejected": True,
            "cross_purpose_same_commitment_rejected_before_store": True,
            "cross_purpose_same_raw_value_rejected_before_commitment": True,
            "final_receipt_root_mismatch_rejected": True,
            "fresh_secret_commitment_reuse_across_attempts_rejected": True,
            "fresh_secret_overlap_cardinalities_rejected_in_order": [1, 2, 3, 4, 5],
            "launch_overlay_mismatch_rejected": True,
            "one_used_key_of_seven_rejects_entire_transaction": True,
            "purpose_or_key_relabel_rejected": True,
            "readiness_evidence_splice_rejected": True,
            "rogue_or_revoked_registry_key_rejected": True,
            "six_winner_rows_bind_purpose_campaign_attempt_body_and_final_root": True,
            "store_transaction_requires_exact_seven_key_atomicity": True,
            "nested_root_formula": (
                "SHA256(HEGEL/HM/R14/NESTED_ROOT/V1\\0||U16BE(PURPOSE_LEN)||"
                "PURPOSE_UTF8||AUTHORIZATION_ID_UTF8||PROFILE_ID_UTF8||"
                "LAUNCH_OVERLAY_ROOT_BYTES||U16BE(CAMPAIGN_ID_LEN)||CAMPAIGN_ID_UTF8||"
                "U16BE(ATTEMPT_ID_LEN)||ATTEMPT_ID_UTF8||"
                "RFC8785(NESTED_PAYLOAD))"
            ),
        },
        "id_domain_ascii_nul_terminated": "HEGEL/HM/R14/GENERATION_RECEIPT/V1\\0",
        "id_formula": "PREFIX||SHA256(DOMAIN||RFC8785(RECEIPT_BODY))",
        "id_prefix": "phase2b_actual_unsealed_960_generation_receipt_v1_",
        "issuance_status": "FUTURE_REQUIRED_NOT_ISSUED",
        "issued_by_profile_verifier": False,
        "required_literals": {
            "atomic_success": True,
            "owner_authorization_id": _OWNER_AUTHORIZATION_ID_V1,
            "schema_version": "hegel-machine-phase2b-generator-generation-receipt/1",
            "standard_error_count": 0,
        },
        "root_or_commitment_materialized_by_profile_verifier": False,
        "transport": {
            "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
            "floats_or_nonfinite_allowed": False,
            "integers": "EXACT_0_THROUGH_2_POW_53_MINUS_1",
            "maximum_array_items": 1024,
            "maximum_bytes": 262144,
            "maximum_depth": 32,
            "maximum_nodes": 20000,
            "maximum_string_utf8_bytes": 4096,
            "unknown_or_duplicate_fields_allowed": False,
            "campaign_attempt_id_type": "FROZEN_PREFIX_PLUS_64_LOWER_HEX",
        },
        "final_receipt_root_formula": (
            "SHA256(HEGEL/HM/R14/FINAL_RECEIPT_ROOT/V1\\0||"
            "BODY_SIGNED_ROOT_BYTES||DURABLE_STORE_TRANSACTION_ROOT_BYTES)"
        ),
        "signature_or_store_side_effect_performed_by_profile_verifier": False,
        "wrapper_fields_in_order": ["receipt_body", "receipt_id"],
    }


def _expected_version_bindings_v1() -> dict[str, object]:
    return {
        "exact_freeze_id": "phase2b_exact_freeze_ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe",
        "exact_freeze_version": "hegel-freeze-p2b-p3-v1.0.1",
        "owner_authorization_config_sha256": _OWNER_AUTHORIZATION_CONFIG_SHA256_V1,
        "owner_authorization_id": _OWNER_AUTHORIZATION_ID_V1,
        "owner_authorization_source_sha256": _OWNER_AUTHORIZATION_SOURCE_SHA256_V1,
        "protocol_id": "phase2b_protocol_62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a",
        "readiness_dag_id": "phase2b_actual_unsealed_960_readiness_dag_v1_775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84",
    }


def _expected_wire_binding_v1() -> dict[str, object]:
    return {
        "binding_status": "FINAL_WIRE_DEPENDENCY_PINS_FRESHLY_VERIFIED",
        "dependency_disposition": "FINAL_STATIC_GO_REPORTED_P0_0_P1_0",
        "default_ignorable_codepoint_count": 4173,
        "default_ignorable_range_count": 17,
        "default_ignorable_range_table_sha256": "sha256:629f6c5f9f20e8322371ff59b61ce34246a9c4afb8fd299756540058dc8a553e",
        "envelope_bytes": 65536,
        "exact_transform_core_source_sha256": "e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f",
        "exact_transform_core_test_sha256": "15d0d6084743d21c2b7a79ec0d4f2f07c84ecfef641fb3d47f5253c95e5a3920",
        "exact_transform_core_version": "hegel-machine-phase2b-pure-exact-transform-core/1",
        "field_classification_manifest_id": "phase2b_formal_wire_field_manifest_v3_12916eefe10770f54c3d6c71f4b8b2c14ebbd671642e96ff4082dd298e5088dc",
        "formal_case_graph_schema_version": "hegel-machine-phase2b-formal-case-graph/3",
        "formal_public_case_schema_version": "hegel-machine-phase2b-formal-public-case/3",
        "formal_trusted_wire_payload_schema_version": "hegel-machine-phase2b-formal-trusted-wire-payload/3",
        "formal_trusted_wire_schema_version": "hegel-machine-phase2b-formal-trusted-wire/3",
        "graph_completeness_policy_id": "phase2b_formal_case_graph_completeness_policy_v3_7585a9f9e58823d766906296905efcd24b1e350330ca5c3f64d75361e9ad90e5",
        "maximum_batch_cases": 1024,
        "namespace_registry_in_order": list(_UUID_NAMESPACES_V1),
        "operation_semantics_version": "hegel-machine-phase2b-exact-transform-semantics/1-projection-v3",
        "rfc8785_source_sha256": "16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654",
        "ucd_derived_core_properties_sha256": "sha256:a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248",
        "ucd_prop_list_sha256": "sha256:485b5a3ed25dbf1f94dfa5a9b69d8b4550ffd0c33045ccc55ccfd7c80b2a40cf",
        "unicode_data_version": "13.0.0",
        "variant_apis_require": "FULL_FORMAL_TRUSTED_WIRE_BUILD_SUCCESS_V3",
        "wire_config_sha256": "4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784",
        "wire_doc_sha256": "704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a",
        "wire_source_sha256": "c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765",
        "wire_stop_claim_boundary": "WIRE_IMPLEMENTATION_EVIDENCE_ONLY_NOT_R03_R04_R05_R06_NOT_ACTUAL_CORPUS",
        "wire_stop_executed_by_profile_verifier": False,
        "wire_test_sha256": "253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0",
        "wire_test_stop": {
            "core_and_rfc_tests_passed": 45,
            "wire_tests_passed": 73,
            "wire_tests_include_rich_1024_and_1025_atomic_reject": True,
        },
    }


def _expected_profile_body_v1() -> dict[str, object]:
    return {
        "authorization_and_fresh_secret_contract": (
            _expected_authorization_and_fresh_secret_contract_v1()
        ),
        "canonicalization": PROFILE_CANONICALIZATION_V1,
        "challenge_descriptor": _expected_challenge_descriptor_v1(),
        "claim_boundary": _expected_claim_boundary_v1(),
        "corpus_contract": _expected_corpus_contract_v1(),
        "dag_contract": _expected_dag_contract_v1(),
        "decision_provenance": _expected_decision_provenance_v1(),
        "disjointness_contract": _expected_disjointness_contract_v1(),
        "document_id": "hegel-machine-phase2b-actual-unsealed-960-generator-construction-profile-v1",
        "generation_receipt_schema": _expected_generation_receipt_schema_v1(),
        "oracle_contracts": _expected_oracle_contracts_v1(),
        "preservation_manifest": _expected_preservation_manifest_v1(),
        "private_uuid_projection": _expected_private_uuid_projection_v1(),
        "r06_qualification_contract": _expected_r06_qualification_contract_v1(),
        "schema_version": PROFILE_SCHEMA_VERSION_V1,
        "semantic_descriptor": _expected_semantic_descriptor_v1(),
        "status": "PROFILE_ONLY_NOT_GENERATOR_NOT_GENERATED",
        "uuid_domain_contract": _expected_uuid_domain_contract_v1(),
        "version_bindings": _expected_version_bindings_v1(),
        "wire_binding": _expected_wire_binding_v1(),
    }


class GeneratorConstructionProfileDispositionV1(str, Enum):
    PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED = (
        "PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED"
    )


@dataclass(frozen=True, slots=True)
class GeneratorCaseMarginQuotaV1:
    case_type: str
    clear_interior: int
    moderate: int
    near_boundary_identifiable: int
    nonunique_or_insufficient: int
    total_per_family_scale_cell: int
    oracle_contract: str

    def __post_init__(self) -> None:
        if type(self.case_type) is not str or not self.case_type:
            raise TypeError("case_type must be an exact nonempty string")
        counts = (
            self.clear_interior,
            self.moderate,
            self.near_boundary_identifiable,
            self.nonunique_or_insufficient,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise TypeError("case-margin counts must be exact nonnegative integers")
        if (
            type(self.total_per_family_scale_cell) is not int
            or self.total_per_family_scale_cell != sum(counts)
            or type(self.oracle_contract) is not str
            or not self.oracle_contract
        ):
            raise ValueError("case-margin row is not internally closed")


@dataclass(frozen=True, slots=True)
class GeneratorPreservationQuotaV1:
    transform: str
    applicable_families: tuple[str, ...]
    legal_pairs_per_family_scale: int
    legal_pairs_per_family: int
    invalid_controls_per_applicable_family: int
    legal_total: int
    invalid_total: int
    operation_oracle_contract: str

    def __post_init__(self) -> None:
        if type(self.transform) is not str or not self.transform:
            raise TypeError("transform must be an exact nonempty string")
        if (
            type(self.applicable_families) is not tuple
            or not self.applicable_families
            or len(set(self.applicable_families)) != len(self.applicable_families)
            or any(family not in _FAMILIES_V1 for family in self.applicable_families)
        ):
            raise ValueError("preservation applicability is not a closed family subset")
        counts = (
            self.legal_pairs_per_family_scale,
            self.legal_pairs_per_family,
            self.invalid_controls_per_applicable_family,
            self.legal_total,
            self.invalid_total,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise TypeError("preservation counts must be exact nonnegative integers")
        if sum(value > 0 for value in counts[:2]) != 1:
            raise ValueError("preservation legal quota needs exactly one counting axis")
        family_count = len(self.applicable_families)
        if self.legal_total != family_count * (
            2 * self.legal_pairs_per_family_scale + self.legal_pairs_per_family
        ):
            raise ValueError("preservation legal total differs")
        if self.invalid_total != family_count * self.invalid_controls_per_applicable_family:
            raise ValueError("preservation invalid total differs")
        if type(self.operation_oracle_contract) is not str or not self.operation_oracle_contract:
            raise ValueError("preservation operation oracle is empty")


@dataclass(frozen=True, slots=True)
class GeneratorQuotaOracleContractV1:
    families: tuple[str, ...]
    private_scale_classes: tuple[str, ...]
    margin_order: tuple[str, ...]
    case_margin_rows: tuple[GeneratorCaseMarginQuotaV1, ...]
    preservation_rows: tuple[GeneratorPreservationQuotaV1, ...]
    private_uuid_namespaces: tuple[str, ...]
    main_case_count: int
    semantic_conflict_case_count: int
    primary_case_count: int
    legal_preservation_variant_count: int
    invalid_preservation_control_count: int
    preservation_manifest_count: int
    generation_receipt_status: str

    def __post_init__(self) -> None:
        if self.families != _FAMILIES_V1 or self.private_scale_classes != _SCALES_V1:
            raise ValueError("family or scale axes differ")
        if self.margin_order != _MARGINS_V1:
            raise ValueError("margin order differs")
        if self.private_uuid_namespaces != _UUID_NAMESPACES_V1:
            raise ValueError("private UUID namespace order differs")
        if tuple(row.case_type for row in self.case_margin_rows) != tuple(
            row[0] for row in _CASE_MARGIN_ROWS_V1
        ):
            raise ValueError("case-margin row order differs")
        if tuple(row.transform for row in self.preservation_rows) != tuple(
            row[0] for row in _PRESERVATION_ROWS_V1
        ):
            raise ValueError("preservation row order differs")
        if sum(row.total_per_family_scale_cell for row in self.case_margin_rows) != 60:
            raise ValueError("per-cell case quota does not sum to 60")
        if (
            sum(row.clear_interior for row in self.case_margin_rows),
            sum(row.moderate for row in self.case_margin_rows),
            sum(row.near_boundary_identifiable for row in self.case_margin_rows),
            sum(row.nonunique_or_insufficient for row in self.case_margin_rows),
        ) != (21, 18, 12, 9):
            raise ValueError("per-cell margin quota differs")
        if (
            self.main_case_count,
            self.semantic_conflict_case_count,
            self.primary_case_count,
            self.legal_preservation_variant_count,
            self.invalid_preservation_control_count,
            self.preservation_manifest_count,
        ) != (720, 240, 960, 496, 76, 572):
            raise ValueError("top-level corpus counts differ")
        if sum(row.legal_total for row in self.preservation_rows) != 496:
            raise ValueError("legal preservation matrix does not sum to 496")
        if sum(row.invalid_total for row in self.preservation_rows) != 76:
            raise ValueError("invalid preservation matrix does not sum to 76")
        if self.generation_receipt_status != "FUTURE_REQUIRED_NOT_ISSUED":
            raise ValueError("generation receipt status differs")


@dataclass(frozen=True, slots=True)
class GeneratorConstructionProfileVerificationV1:
    schema_version: str
    profile_id: str
    disposition: GeneratorConstructionProfileDispositionV1
    claim_level: str
    canonical_profile_sha256: str
    profile: Mapping[str, object]
    contract: GeneratorQuotaOracleContractV1
    result_object_unforgeable: bool = False
    is_execution_evidence: bool = False
    fresh_original_bytes_verification_required: bool = True
    generator_implemented: bool = False
    generator_qualified: bool = False
    actual_rows_generated: bool = False
    actual_bytes_materialized: bool = False
    actual_corpus_audited: bool = False
    authority_verified: bool = False
    origin_authenticated: bool = False
    r03_satisfied: bool = False
    r04_satisfied: bool = False
    r05_verified: bool = False
    r06_qualification_executed: bool = False
    r06_qualification_cache_issued: bool = False
    qualification_cache_admission_evidence: bool = False
    r06_qualification_receipt_issued: bool = False
    r06_satisfied: bool = False
    r07_satisfied: bool = False
    r08_satisfied: bool = False
    r14_satisfied: bool = False
    r15_satisfied: bool = False
    r16_satisfied: bool = False
    entropy_used: bool = False
    identifiers_allocated: bool = False
    filesystem_accessed: bool = False
    network_accessed: bool = False
    docker_invoked: bool = False
    keys_generated: bool = False
    signatures_generated: bool = False
    signature_key_registry_verified: bool = False
    commitment_issued: bool = False
    commitment_set_consumed: bool = False
    per_purpose_commitments_consumed: bool = False
    generator_seed_commitment_issued: bool = False
    answer_seed_commitment_issued: bool = False
    generation_receipt_issued: bool = False
    final_receipt_root_issued: bool = False
    durable_store_transaction_executed: bool = False
    custody_verified: bool = False
    technical_role_independence_verified: bool = False
    deterministic_replay_executed: bool = False
    public_variant_uuid_disjointness_verified: bool = False
    wire_replay_executed: bool = False
    scoring_performed: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema_version != PROFILE_SCHEMA_VERSION_V1
            or self.disposition.value != PROFILE_SUCCESS_DISPOSITION_V1
            or self.claim_level != PROFILE_CLAIM_LEVEL_V1
            or type(self.profile_id) is not str
            or type(self.canonical_profile_sha256) is not str
            or len(self.canonical_profile_sha256) != 64
            or not isinstance(self.profile, Mapping)
            or type(self.contract) is not GeneratorQuotaOracleContractV1
        ):
            raise ValueError("profile verification result identity differs")
        if type(self.result_object_unforgeable) is not bool or self.result_object_unforgeable:
            raise ValueError("result_object_unforgeable must be exact false")
        if type(self.is_execution_evidence) is not bool or self.is_execution_evidence:
            raise ValueError("is_execution_evidence must be exact false")
        if (
            type(self.fresh_original_bytes_verification_required) is not bool
            or not self.fresh_original_bytes_verification_required
        ):
            raise ValueError("fresh original bytes verification must remain required")
        false_claims = (
            self.generator_implemented,
            self.generator_qualified,
            self.actual_rows_generated,
            self.actual_bytes_materialized,
            self.actual_corpus_audited,
            self.authority_verified,
            self.origin_authenticated,
            self.r03_satisfied,
            self.r04_satisfied,
            self.r05_verified,
            self.r06_qualification_executed,
            self.r06_qualification_cache_issued,
            self.qualification_cache_admission_evidence,
            self.r06_qualification_receipt_issued,
            self.r06_satisfied,
            self.r07_satisfied,
            self.r08_satisfied,
            self.r14_satisfied,
            self.r15_satisfied,
            self.r16_satisfied,
            self.entropy_used,
            self.identifiers_allocated,
            self.filesystem_accessed,
            self.network_accessed,
            self.docker_invoked,
            self.keys_generated,
            self.signatures_generated,
            self.signature_key_registry_verified,
            self.commitment_issued,
            self.commitment_set_consumed,
            self.per_purpose_commitments_consumed,
            self.generator_seed_commitment_issued,
            self.answer_seed_commitment_issued,
            self.generation_receipt_issued,
            self.final_receipt_root_issued,
            self.durable_store_transaction_executed,
            self.custody_verified,
            self.technical_role_independence_verified,
            self.deterministic_replay_executed,
            self.public_variant_uuid_disjointness_verified,
            self.wire_replay_executed,
            self.scoring_performed,
        )
        if any(type(value) is not bool or value for value in false_claims):
            raise ValueError("a prohibited execution or readiness claim is not exact false")


class GeneratorConstructionProfileErrorV1(ValueError):
    """Closed canonical-profile verification failure."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise GeneratorConstructionProfileErrorV1(code, detail)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("duplicate or non-string JSON object key")
        result[key] = value
    return result


def _reject_float(token: str) -> NoReturn:
    raise ValueError(f"floating-point JSON number is forbidden: {token}")


def _reject_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _parse_integer(token: str) -> int:
    value = int(token, 10)
    if not -_MAXIMUM_SAFE_INTEGER_V1 <= value <= _MAXIMUM_SAFE_INTEGER_V1:
        raise ValueError("JSON integer is outside the exact binary64-safe range")
    return value


def _validate_json_shape(value: object) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAXIMUM_JSON_NODES_V1:
            _fail("PROFILE_JSON_INVALID", "profile JSON node count exceeds the bound")
        if depth > _MAXIMUM_JSON_DEPTH_V1:
            _fail("PROFILE_JSON_INVALID", "profile JSON nesting exceeds the bound")
        if current is None or type(current) in {bool, int, str}:
            continue
        if type(current) is list:
            stack.extend((item, depth + 1) for item in current)
            continue
        if type(current) is dict:
            if any(type(key) is not str for key in current):
                _fail("PROFILE_JSON_INVALID", "profile JSON key is not an exact string")
            stack.extend((item, depth + 1) for item in current.values())
            continue
        _fail("PROFILE_JSON_INVALID", "profile contains a non-JSON runtime type")


def _decode_canonical_profile(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        _fail("WRONG_INPUT_TYPE", "profile payload must be exact bytes")
    if not 1 <= len(payload) <= _MAXIMUM_PROFILE_BYTES_V1:
        _fail("PROFILE_SIZE_INVALID", "profile byte length is outside the frozen bound")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail("PROFILE_JSON_INVALID", f"profile is not UTF-8: {exc}")
    if text.startswith("\ufeff"):
        _fail("PROFILE_JSON_INVALID", "UTF-8 BOM is forbidden")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        _fail("PROFILE_JSON_INVALID", f"profile is not strict JSON: {exc}")
    if type(decoded) is not dict:
        _fail("PROFILE_SCHEMA_INVALID", "profile root must be an exact JSON object")
    _validate_json_shape(decoded)
    try:
        canonical = canonicalize_rfc8785_v1(decoded)
    except (RFC8785Error, RecursionError) as exc:
        _fail("PROFILE_JSON_INVALID", f"profile is outside RFC 8785: {exc}")
    if canonical != payload:
        _fail("PROFILE_NOT_CANONICAL", "profile bytes are not exact RFC 8785 bytes")
    return decoded


def _exact_json_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _exact_json_equal(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_equal(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _require_exact(value: object, expected: object, field: str) -> None:
    if not _exact_json_equal(value, expected):
        _fail("PROFILE_SCHEMA_INVALID", f"{field} differs from the frozen profile")


def _deep_freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    if value is None or type(value) in {bool, int, str}:
        return value
    _fail("PROFILE_SCHEMA_INVALID", "profile contains a non-JSON runtime type")


def _contract_v1() -> GeneratorQuotaOracleContractV1:
    return GeneratorQuotaOracleContractV1(
        families=_FAMILIES_V1,
        private_scale_classes=_SCALES_V1,
        margin_order=_MARGINS_V1,
        case_margin_rows=tuple(
            GeneratorCaseMarginQuotaV1(*row) for row in _CASE_MARGIN_ROWS_V1
        ),
        preservation_rows=tuple(
            GeneratorPreservationQuotaV1(*row) for row in _PRESERVATION_ROWS_V1
        ),
        private_uuid_namespaces=_UUID_NAMESPACES_V1,
        main_case_count=720,
        semantic_conflict_case_count=240,
        primary_case_count=960,
        legal_preservation_variant_count=496,
        invalid_preservation_control_count=76,
        preservation_manifest_count=572,
        generation_receipt_status="FUTURE_REQUIRED_NOT_ISSUED",
    )


def _validate_profile_wrapper_v1(value: dict[str, object]) -> dict[str, object]:
    if set(value) != {"profile_body", "profile_id"}:
        _fail("PROFILE_SCHEMA_INVALID", "profile wrapper fields differ")
    profile_body = value["profile_body"]
    profile_id = value["profile_id"]
    if type(profile_body) is not dict or type(profile_id) is not str:
        _fail("PROFILE_SCHEMA_INVALID", "profile wrapper field types differ")
    expected_body = _expected_profile_body_v1()
    _require_exact(profile_body, expected_body, "profile_body")
    canonical_body = canonicalize_rfc8785_v1(profile_body)
    computed_profile_id = _PROFILE_ID_PREFIX_V1 + hashlib.sha256(
        _PROFILE_ID_DOMAIN_V1 + canonical_body
    ).hexdigest()
    if computed_profile_id != PROFILE_ID_V1 or profile_id != PROFILE_ID_V1:
        _fail("PROFILE_ID_INVALID", "profile identity differs from the frozen KAT")
    return profile_body


def verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
    profile_bytes: bytes,
) -> GeneratorConstructionProfileVerificationV1:
    """Freshly verify the one frozen canonical profile byte string.

    The returned object is intentionally publicly constructible and forgeable;
    only a fresh call on the original bytes establishes profile-byte equality.
    """

    decoded = _decode_canonical_profile(profile_bytes)
    profile_body = _validate_profile_wrapper_v1(decoded)
    canonical_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    if canonical_sha256 != PROFILE_CANONICAL_SHA256_V1:
        _fail("PROFILE_HASH_INVALID", "profile byte hash differs from the frozen KAT")
    frozen_profile = _deep_freeze(profile_body)
    if not isinstance(frozen_profile, Mapping):
        _fail("PROFILE_SCHEMA_INVALID", "frozen profile is not a mapping")
    return GeneratorConstructionProfileVerificationV1(
        schema_version=PROFILE_SCHEMA_VERSION_V1,
        profile_id=PROFILE_ID_V1,
        disposition=GeneratorConstructionProfileDispositionV1(
            PROFILE_SUCCESS_DISPOSITION_V1
        ),
        claim_level=PROFILE_CLAIM_LEVEL_V1,
        canonical_profile_sha256=canonical_sha256,
        profile=frozen_profile,
        contract=_contract_v1(),
    )


__all__ = ["verify_phase2b_actual_unsealed_960_generator_construction_profile_v1"]
