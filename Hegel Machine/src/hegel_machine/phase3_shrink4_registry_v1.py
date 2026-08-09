"""Normative diagnostic registry for Phase-3 shrink step 4.

The child changes only ``max_top_level_clauses`` from three to two.  Registry
IDs and states are inherited byte-for-byte from shrink step 3.  These objects
are engineering commitments; they neither create formal roots nor start M3.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Final

from .hashing import stable_hash
from .phase3_dsl_v1 import (
    BOTTOM_AND_EQUIVALENCE,
    BOOLEAN_COMPOSITION,
    CLOSURE_BUDGET,
    FORBIDDEN_FORMS,
    LEAF_EXPRESSIONS,
    RATIONAL_VALUE_GRID,
    SCOPE_CATALOG,
    SHRINK_ORDER,
    STRUCTURAL_LIMITS,
    TERNARY_OPERATORS,
    TRANSFORM_CATALOG,
    UNARY_OPERATORS,
)
from .phase3_m3_shrink4_core_v1 import (
    ACTIVE_AGGREGATE_IDS,
    ACTIVE_FORMAL_BINARY_OPERATOR_IDS,
    ACTIVE_RATIONAL_PARAMETER_IDS,
    ACTIVE_SOURCE_BINARY_OPERATOR_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    LEGAL_AST_TOMBSTONE_PRIORITY,
    MAX_TOP_LEVEL_CLAUSES,
    PARENT_DIAGNOSTIC_CLAIM_LEVEL,
    PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID,
    PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS,
    PARENT_DIAGNOSTIC_RESULT_COMMIT,
    PARENT_DIAGNOSTIC_STATUS,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    PARENT_MAX_TOP_LEVEL_CLAUSES,
    RESERVED_BINARY_OPERATOR_IDS,
    RESERVED_RATIONAL_PARAMETER_IDS,
    SEALED_DUAL_STRICT_OUTCOME_REPLAY_STATUS,
    SHRINK_STEP_ID,
    SOURCE_ALIAS_BINARY_OPERATOR_IDS,
    STRUCTURAL_LIMIT_ERROR,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_BINARY_OPERATOR_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
    TOP_LEVEL_AND_NORMALIZATION_ORDER,
)
from .phase3_shrink1_registry_v1 import (
    AST_HASH_DOMAIN,
    AST_SCHEMA_ID,
    CBOR_PROFILE_ID,
    MDL_CODE_TABLE_ID,
)
from .phase3_shrink3_registry_v1 import (
    BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID,
    OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID,
    REJECTION_PRIORITY_DIAGNOSTIC_ID,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID as PARENT_DSL_SURFACE_DIAGNOSTIC_ID,
)


SHRINK4_STRUCTURAL_LIMITS: Final = replace(
    STRUCTURAL_LIMITS,
    max_top_level_clauses=MAX_TOP_LEVEL_CLAUSES,
)


def _external_limits(value: object) -> dict[str, object]:
    result = asdict(value)  # type: ignore[arg-type]
    result["maximum_top_level_clauses"] = result.pop("max_top_level_clauses")
    return result


def structural_limit_semantics_object() -> dict[str, object]:
    """Bind the only shrink-4 language delta and its evaluation order."""

    parent_limits = _external_limits(STRUCTURAL_LIMITS)
    child_limits = _external_limits(SHRINK4_STRUCTURAL_LIMITS)
    changed = {
        key: {"parent": parent_limits[key], "child": child_limits[key]}
        for key in parent_limits
        if parent_limits[key] != child_limits[key]
    }
    return {
        "schema_version": "hegel-structural-limit-semantics/1.4.0",
        "dsl_version": DSL_VERSION,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "shrink_step_id": SHRINK_STEP_ID,
        "parent_limits": parent_limits,
        "child_limits": child_limits,
        "changed_fields": changed,
        "sole_changed_field": "maximum_top_level_clauses",
        "from_maximum_top_level_clauses": PARENT_MAX_TOP_LEVEL_CLAUSES,
        "to_maximum_top_level_clauses": MAX_TOP_LEVEL_CLAUSES,
        "normalization_before_limit": True,
        "normalization_order": list(TOP_LEVEL_AND_NORMALIZATION_ORDER),
        "and1_eliminated_before_limit": True,
        "nested_and_flattened_before_limit": True,
        "clauses_sorted_before_limit": True,
        "clauses_deduplicated_before_limit": True,
        "canonical_and3_disposition": STRUCTURAL_LIMIT_ERROR,
        "parent_validation_and_tombstone_priority_inherited": True,
        "registry_changed": False,
        "typing_changed": False,
        "bottom_semantics_changed": False,
        "surviving_rewrite_rules_changed": False,
        "closure_budget_changed": False,
        "scope_catalog_changed": False,
        "mdl_code_table_changed": False,
    }


STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID: Final = stable_hash(
    structural_limit_semantics_object(), prefix="structural_limit_semantics_"
)


def shrunk_dsl_surface_object() -> dict[str, object]:
    inherited_surface_content_ids = {
        "rational_value_grid": stable_hash(
            RATIONAL_VALUE_GRID, prefix="rational_grid_"
        ),
        "scope_catalog": stable_hash(SCOPE_CATALOG, prefix="scope_catalog_"),
        "transform_catalog": stable_hash(
            TRANSFORM_CATALOG, prefix="transform_catalog_"
        ),
        "leaf_expressions": stable_hash(
            LEAF_EXPRESSIONS, prefix="leaf_expressions_"
        ),
        "unary_operators": stable_hash(
            UNARY_OPERATORS, prefix="unary_operators_"
        ),
        "ternary_operators": stable_hash(
            TERNARY_OPERATORS, prefix="ternary_operators_"
        ),
        "boolean_composition_parent": stable_hash(
            BOOLEAN_COMPOSITION, prefix="boolean_composition_"
        ),
        "forbidden_forms": stable_hash(
            FORBIDDEN_FORMS, prefix="forbidden_forms_"
        ),
        "bottom_and_equivalence": stable_hash(
            BOTTOM_AND_EQUIVALENCE, prefix="bottom_equivalence_"
        ),
        "closure_budget": stable_hash(CLOSURE_BUDGET, prefix="closure_budget_"),
    }
    return {
        "schema_version": "hegel-old-dsl-freeze/1.4.0",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "parent_dsl_surface_diagnostic_id": PARENT_DSL_SURFACE_DIAGNOSTIC_ID,
        "engineering_trigger": {
            "result_commit": PARENT_DIAGNOSTIC_RESULT_COMMIT,
            "implementation_basis": PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS,
            "evidence_record_id": PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID,
            "status": PARENT_DIAGNOSTIC_STATUS,
            "claim_level": PARENT_DIAGNOSTIC_CLAIM_LEVEL,
            "execution_state": "NOT_RUN",
            "formal_roots": None,
            "formal_closure_execution_performed": False,
            "authority": "PREREGISTERED_ENGINEERING_ROUTING_ONLY",
        },
        "shrink_step_id": SHRINK_STEP_ID,
        "pre_registered_delta_only": (
            "reduce normalized max_top_level_clauses from 3 to 2"
        ),
        "structural_limit_semantics_diagnostic_id": (
            STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID
        ),
        "parent_binary_operator_registry_diagnostic_id": (
            BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID
        ),
        "parent_operator_admission_semantics_diagnostic_id": (
            OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID
        ),
        "parent_rejection_priority_diagnostic_id": (
            REJECTION_PRIORITY_DIAGNOSTIC_ID
        ),
        "canonical_ast_schema_id": AST_SCHEMA_ID,
        "canonical_cbor_profile_id": CBOR_PROFILE_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
        "mdl_code_table_id": MDL_CODE_TABLE_ID,
        "inherited_surface_content_ids": inherited_surface_content_ids,
        "parent_structural_limits_content_id": stable_hash(
            STRUCTURAL_LIMITS, prefix="structural_limits_"
        ),
        "child_structural_limits_content_id": stable_hash(
            SHRINK4_STRUCTURAL_LIMITS, prefix="structural_limits_"
        ),
        "remaining_shrink_order": [step.operation for step in SHRINK_ORDER[4:]],
        "canonical_program_budget": CLOSURE_BUDGET.max_canonical_program_count,
        "raw_application_cap": CLOSURE_BUDGET.max_raw_operator_applications,
        "surviving_ast_bytes_stable": True,
        "surviving_ast_hash_stable": True,
        "surviving_mdl_length_stable": True,
        "surviving_program_semantic_identity_must_rebind": True,
        "removed_ast_rewrite_or_migration_allowed": False,
        "cross_version_archive_root_reuse_allowed": False,
        "engineering_only": True,
        "sealed_dual_strict_outcome_replay_status": (
            SEALED_DUAL_STRICT_OUTCOME_REPLAY_STATUS
        ),
        "dual_complete_enumerator_qualified": False,
        "execution_state": "NOT_RUN",
        "complete_closure_enumerated": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "formal_state_transition_allowed": False,
        "target_roles_evaluated": False,
        "certificate_issued": False,
    }


SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID: Final = stable_hash(
    shrunk_dsl_surface_object(), prefix="dsl_spec_"
)


def validate_registry_freeze() -> None:
    if STRUCTURAL_LIMITS.max_top_level_clauses != PARENT_MAX_TOP_LEVEL_CLAUSES:
        raise AssertionError("parent max_top_level_clauses drift")
    if SHRINK4_STRUCTURAL_LIMITS.max_top_level_clauses != MAX_TOP_LEVEL_CLAUSES:
        raise AssertionError("child max_top_level_clauses drift")
    parent = asdict(STRUCTURAL_LIMITS)
    child = asdict(SHRINK4_STRUCTURAL_LIMITS)
    if {
        key for key in parent if parent[key] != child[key]
    } != {"max_top_level_clauses"}:
        raise AssertionError("shrink-4 structural-limit delta is not singular")
    if ACTIVE_AGGREGATE_IDS != (0, 1, 5) or TOMBSTONED_AGGREGATE_IDS != (2, 3, 4):
        raise AssertionError("aggregate registry inheritance drift")
    if ACTIVE_RATIONAL_PARAMETER_IDS != (1, 3, 5) or TOMBSTONED_RATIONAL_PARAMETER_IDS != (0, 2, 4, 6):
        raise AssertionError("rational registry inheritance drift")
    if ACTIVE_SOURCE_BINARY_OPERATOR_IDS != (1, 2, 3, 4, 5, 6):
        raise AssertionError("source binary registry inheritance drift")
    if ACTIVE_FORMAL_BINARY_OPERATOR_IDS != (1, 2, 3, 5, 6):
        raise AssertionError("formal binary registry inheritance drift")
    if SOURCE_ALIAS_BINARY_OPERATOR_IDS != (4,) or TOMBSTONED_BINARY_OPERATOR_IDS != (0,):
        raise AssertionError("binary alias/tombstone inheritance drift")
    if RESERVED_BINARY_OPERATOR_IDS != (7,) or RESERVED_RATIONAL_PARAMETER_IDS != (7,):
        raise AssertionError("reserved registry inheritance drift")
    if LEGAL_AST_TOMBSTONE_PRIORITY != (
        "AggregateMapId/v1",
        "RationalParameterId/v1",
        "BinaryOperatorId/v1",
    ):
        raise AssertionError("inherited tombstone priority drift")


validate_registry_freeze()


__all__ = [
    "DSL_VERSION",
    "FREEZE_VERSION",
    "HUMAN_AMENDMENT_ID",
    "MAX_TOP_LEVEL_CLAUSES",
    "PARENT_DSL_VERSION",
    "PARENT_FREEZE_VERSION",
    "SHRINK4_STRUCTURAL_LIMITS",
    "SHRINK_STEP_ID",
    "SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID",
    "STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID",
    "shrunk_dsl_surface_object",
    "structural_limit_semantics_object",
    "validate_registry_freeze",
]
