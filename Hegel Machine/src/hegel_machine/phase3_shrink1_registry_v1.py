"""Normative diagnostic freeze for Phase-3 shrink step 1.

This module publishes the child DSL surface selected after the parent
``hegel-old-dsl-v1.0.0`` capacity replay reached the bounded
``DSL_TOO_LARGE`` state.  It deliberately does not manufacture formal CBOR /
RFC6962 roots: publication-time formal roots remain null until the independent
formal bridge is implemented and replayed by Python and Rust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .hashing import stable_hash
from .phase3_dsl_v1 import (
    BOTTOM_AND_EQUIVALENCE,
    BOOLEAN_COMPOSITION,
    BINARY_OPERATORS,
    CLOSURE_BUDGET,
    FORBIDDEN_FORMS,
    LEAF_EXPRESSIONS,
    OLD_DSL_V1,
    PRIMITIVE_DOMAINS,
    RATIONAL_VALUE_GRID,
    SCOPE_CATALOG,
    SHRINK_ORDER,
    STRUCTURAL_LIMITS,
    TERNARY_OPERATORS,
    TRANSFORM_CATALOG,
    UNARY_OPERATORS,
)


PARENT_DSL_VERSION: Final = "hegel-old-dsl-v1.0.0"
PARENT_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.2"
DSL_VERSION: Final = "hegel-old-dsl-v1.1.0"
FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.1.0"
HUMAN_AMENDMENT_ID: Final = "hegel-freeze-p2b-p3-v1.1.0-shrink-step1"
SHRINK_STEP_ID: Final = "SHRINK_STEP_1_REMOVE_MEAN_MIN_MAX"

AST_SCHEMA_ID: Final = "hegel-canonical-ast-v1"
CBOR_PROFILE_ID: Final = "hegel-cbor-det-v1"
MDL_CODE_TABLE_ID: Final = "hegel-mdl-prefix-v1.0.0"
AST_HASH_DOMAIN: Final = "HEGEL/AST/V1"
AGGREGATE_REGISTRY_NAMESPACE: Final = "AggregateMapId/v1"
AGGREGATE_REGISTRY_POLICY: Final = "SPARSE_PRESERVING"
REMOVED_AGGREGATE_ERROR: Final = "REJECT_REMOVED_AGGREGATE_MAP"
UNKNOWN_AGGREGATE_ERROR: Final = "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"

FORMAL_ROOT_NAMES: Final = (
    "dsl_spec_root",
    "operator_semantics_root",
    "identifier_registry_root",
    "canonical_ast_schema_root",
    "canonical_cbor_profile_root",
    "bounded_universe_root",
    "target_truth_table_root",
    "canonical_program_archive_root",
    "program_output_archive_root",
    "chunk_manifest_root",
    "diagnostic_formal_bridge_root",
)


@dataclass(frozen=True, order=True, slots=True)
class AggregateRegistryEntryV1:
    numeric_id: int
    name: str
    state: str
    output_sort: str

    def __post_init__(self) -> None:
        if type(self.numeric_id) is not int or not 0 <= self.numeric_id < 6:
            raise ValueError("AggregateMapId/v1 entries must use allocated IDs 0..5")
        if self.state not in {"ACTIVE", "TOMBSTONE"}:
            raise ValueError("aggregate registry state must be ACTIVE or TOMBSTONE")
        if self.output_sort not in {"BoundedInt", "RationalValue"}:
            raise ValueError("aggregate output sort drift")


AGGREGATE_REGISTRY: Final = (
    AggregateRegistryEntryV1(0, "sum_v1", "ACTIVE", "RationalValue"),
    AggregateRegistryEntryV1(1, "count_nonzero_v1", "ACTIVE", "BoundedInt"),
    AggregateRegistryEntryV1(2, "mean_v1", "TOMBSTONE", "RationalValue"),
    AggregateRegistryEntryV1(3, "min_v1", "TOMBSTONE", "RationalValue"),
    AggregateRegistryEntryV1(4, "max_v1", "TOMBSTONE", "RationalValue"),
    AggregateRegistryEntryV1(5, "signed_balance_v1", "ACTIVE", "RationalValue"),
)

REGISTRY_WIDTH: Final = 6
ACTIVE_MAP_COUNT: Final = 3
TOMBSTONE_COUNT: Final = 3
ACTIVE_AGGREGATE_IDS: Final = (0, 1, 5)
TOMBSTONED_AGGREGATE_IDS: Final = (2, 3, 4)
ACTIVE_AGGREGATE_NAMES: Final = ("sum_v1", "count_nonzero_v1", "signed_balance_v1")
TOMBSTONED_AGGREGATE_NAMES: Final = ("mean_v1", "min_v1", "max_v1")
RATIONAL_ACTIVE_AGGREGATE_NAMES: Final = ("sum_v1", "signed_balance_v1")
NEXT_ALLOCATABLE_AGGREGATE_ID: Final = 6


def aggregate_registry_object() -> dict[str, object]:
    """Return the immutable active+tombstone diagnostic registry object."""

    return {
        "schema_version": "hegel-aggregate-registry/1",
        "registry_namespace": AGGREGATE_REGISTRY_NAMESPACE,
        "policy": AGGREGATE_REGISTRY_POLICY,
        "registry_width": REGISTRY_WIDTH,
        "active_map_count": ACTIVE_MAP_COUNT,
        "tombstone_count": TOMBSTONE_COUNT,
        "entries": [
            {
                "numeric_id": entry.numeric_id,
                "name": entry.name,
                "state": entry.state,
                "output_sort": entry.output_sort,
            }
            for entry in AGGREGATE_REGISTRY
        ],
        "next_allocatable_id": NEXT_ALLOCATABLE_AGGREGATE_ID,
        "id_reuse_allowed": False,
        "tombstones_permanent_in_registry_lineage": True,
        "removed_source_and_formal_error": REMOVED_AGGREGATE_ERROR,
        "future_unknown_id_error": UNKNOWN_AGGREGATE_ERROR,
    }


AGGREGATE_REGISTRY_DIAGNOSTIC_ID: Final = stable_hash(
    aggregate_registry_object(), prefix="aggregate_registry_"
)


def operator_admission_semantics_object() -> dict[str, object]:
    """Bind the only semantic admission delta authorized by shrink step 1."""

    return {
        "schema_version": "hegel-operator-admission-semantics/1",
        "dsl_version": DSL_VERSION,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "aggregate_registry_diagnostic_id": AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
        "active_aggregate_ids": list(ACTIVE_AGGREGATE_IDS),
        "tombstoned_aggregate_ids": list(TOMBSTONED_AGGREGATE_IDS),
        "removed_map_disposition": REMOVED_AGGREGATE_ERROR,
        "implicit_bit_to_rational_coercion": False,
        "typing_changed": False,
        "bottom_semantics_changed": False,
        "rewrite_rules_changed": False,
        "structural_limits_changed": False,
        "scope_catalog_changed": False,
        "equivalence_mode": "EXACT_EXTENSIONAL",
        "target_or_control_semantics_changed": False,
        "mdl_code_table_changed": False,
    }


OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID: Final = stable_hash(
    operator_admission_semantics_object(), prefix="operator_admission_semantics_"
)


def shrunk_dsl_surface_object() -> dict[str, object]:
    """Return the source/diagnostic child DSL freeze, never a formal root."""

    return {
        "schema_version": "hegel-old-dsl-freeze/1.1.0",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "parent_dsl_spec_id": OLD_DSL_V1.content_id,
        "shrink_step_id": SHRINK_STEP_ID,
        "aggregate_registry_diagnostic_id": AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
        "operator_admission_semantics_diagnostic_id": (
            OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID
        ),
        "canonical_ast_schema_id": AST_SCHEMA_ID,
        "canonical_cbor_profile_id": CBOR_PROFILE_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
        "mdl_code_table_id": MDL_CODE_TABLE_ID,
        "phase2b_contract_inherited_from": PARENT_FREEZE_VERSION,
        "phase2b_contract_changed": False,
        "inherited_surface_content_ids": {
            "primitive_domains": stable_hash(PRIMITIVE_DOMAINS, prefix="primitive_domains_"),
            "rational_value_grid": stable_hash(RATIONAL_VALUE_GRID, prefix="rational_grid_"),
            "scope_catalog": stable_hash(SCOPE_CATALOG, prefix="scope_catalog_"),
            "transform_catalog": stable_hash(TRANSFORM_CATALOG, prefix="transform_catalog_"),
            "leaf_expressions": stable_hash(LEAF_EXPRESSIONS, prefix="leaf_expressions_"),
            "unary_operators": stable_hash(UNARY_OPERATORS, prefix="unary_operators_"),
            "binary_operators": stable_hash(BINARY_OPERATORS, prefix="binary_operators_"),
            "ternary_operators": stable_hash(TERNARY_OPERATORS, prefix="ternary_operators_"),
            "boolean_composition": stable_hash(BOOLEAN_COMPOSITION, prefix="boolean_composition_"),
            "forbidden_forms": stable_hash(FORBIDDEN_FORMS, prefix="forbidden_forms_"),
            "bottom_and_equivalence": stable_hash(BOTTOM_AND_EQUIVALENCE, prefix="bottom_equivalence_"),
            "structural_limits": stable_hash(STRUCTURAL_LIMITS, prefix="structural_limits_"),
            "closure_budget": stable_hash(CLOSURE_BUDGET, prefix="closure_budget_"),
        },
        "remaining_shrink_order": [step.operation for step in SHRINK_ORDER[1:]],
        "canonical_program_budget": CLOSURE_BUDGET.max_canonical_program_count,
        "raw_application_cap": CLOSURE_BUDGET.max_raw_operator_applications,
        "surviving_ast_bytes_stable": True,
        "surviving_ast_hash_stable": True,
        "cross_version_archive_root_reuse_allowed": False,
        "execution_state": "NOT_RUN",
        "complete_closure_enumerated": False,
        "formal_roots": None,
    }


SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID: Final = stable_hash(
    shrunk_dsl_surface_object(), prefix="dsl_spec_"
)


def validate_registry_freeze() -> None:
    """Fail closed if the sparse registry or immutable IDs drift."""

    ids = tuple(entry.numeric_id for entry in AGGREGATE_REGISTRY)
    if ids != tuple(range(REGISTRY_WIDTH)):
        raise AssertionError("AggregateMapId/v1 allocation history is not contiguous")
    active = tuple(entry.numeric_id for entry in AGGREGATE_REGISTRY if entry.state == "ACTIVE")
    tombstones = tuple(
        entry.numeric_id for entry in AGGREGATE_REGISTRY if entry.state == "TOMBSTONE"
    )
    if active != ACTIVE_AGGREGATE_IDS or tombstones != TOMBSTONED_AGGREGATE_IDS:
        raise AssertionError("sparse aggregate registry disposition drift")
    if REGISTRY_WIDTH != 6 or ACTIVE_MAP_COUNT != 3 or TOMBSTONE_COUNT != 3:
        raise AssertionError("aggregate registry cardinality drift")
    if NEXT_ALLOCATABLE_AGGREGATE_ID != REGISTRY_WIDTH:
        raise AssertionError("tombstoned IDs must never be reused")


validate_registry_freeze()


__all__ = [
    "ACTIVE_AGGREGATE_IDS",
    "ACTIVE_AGGREGATE_NAMES",
    "AGGREGATE_REGISTRY",
    "AGGREGATE_REGISTRY_DIAGNOSTIC_ID",
    "AGGREGATE_REGISTRY_NAMESPACE",
    "AGGREGATE_REGISTRY_POLICY",
    "AST_HASH_DOMAIN",
    "AST_SCHEMA_ID",
    "CBOR_PROFILE_ID",
    "DSL_VERSION",
    "FORMAL_ROOT_NAMES",
    "FREEZE_VERSION",
    "HUMAN_AMENDMENT_ID",
    "MDL_CODE_TABLE_ID",
    "NEXT_ALLOCATABLE_AGGREGATE_ID",
    "OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID",
    "PARENT_DSL_VERSION",
    "PARENT_FREEZE_VERSION",
    "RATIONAL_ACTIVE_AGGREGATE_NAMES",
    "REGISTRY_WIDTH",
    "REMOVED_AGGREGATE_ERROR",
    "SHRINK_STEP_ID",
    "SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID",
    "TOMBSTONED_AGGREGATE_IDS",
    "TOMBSTONED_AGGREGATE_NAMES",
    "UNKNOWN_AGGREGATE_ERROR",
    "aggregate_registry_object",
    "operator_admission_semantics_object",
    "shrunk_dsl_surface_object",
    "validate_registry_freeze",
]
