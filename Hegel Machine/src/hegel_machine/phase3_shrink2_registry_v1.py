"""Normative diagnostic registry for Phase-3 shrink step 2.

This child freeze performs exactly one pre-registered change relative to
``hegel-old-dsl-v1.1.0``: RationalParameter admission is reduced to
``{-1, 0, 1}``.  Numeric IDs and the three-bit wire code remain those of
``RationalParameterId/v1``.  Removed values are permanent tombstones rather
than a compacted three-entry registry.

The objects in this module are diagnostic commitments.  They do not create
formal CBOR/RFC6962 roots or assert that the shrink-2 closure has executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from .hashing import stable_hash
from .phase3_dsl_v1 import (
    BOTTOM_AND_EQUIVALENCE,
    BOOLEAN_COMPOSITION,
    BINARY_OPERATORS,
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
from .phase3_m3_shrink2_core_v1 import (
    ACTIVE_AGGREGATE_IDS,
    ACTIVE_RATIONAL_PARAMETER_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    RATIONAL_PARAMETER_ALLOCATED_ID_COUNT,
    RATIONAL_PARAMETER_CODE_SPACE_SIZE,
    RATIONAL_PARAMETER_CODE_WIDTH_BITS,
    RATIONAL_PARAMETER_REGISTRY_NAMESPACE,
    REMOVED_AGGREGATE_ERROR,
    REMOVED_RATIONAL_PARAMETER_ERROR,
    RESERVED_RATIONAL_PARAMETER_IDS,
    SHRINK_STEP_ID,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
    UNKNOWN_RATIONAL_PARAMETER_ERROR,
)
from .phase3_shrink1_registry_v1 import (
    AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
    AST_HASH_DOMAIN,
    AST_SCHEMA_ID,
    CBOR_PROFILE_ID,
    MDL_CODE_TABLE_ID,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID as PARENT_DSL_SURFACE_DIAGNOSTIC_ID,
)


RATIONAL_PARAMETER_REGISTRY_POLICY: Final = "SPARSE_PRESERVING"

ACTIVE_RATIONAL_PARAMETER_VALUES: Final = ((-1, 1), (0, 1), (1, 1))
TOMBSTONED_RATIONAL_PARAMETER_VALUES: Final = (
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
)
NEXT_ALLOCATABLE_RATIONAL_PARAMETER_ID: Final = None


@dataclass(frozen=True, order=True, slots=True)
class RationalParameterRegistryEntryV1:
    numeric_id: int
    numerator: int
    denominator: int
    state: str

    def __post_init__(self) -> None:
        if type(self.numeric_id) is not int or not 0 <= self.numeric_id < 7:
            raise ValueError("RationalParameterId/v1 entries must use allocated IDs 0..6")
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("rational parameter components must be integers")
        if self.denominator <= 0:
            raise ValueError("rational parameter denominator must be positive")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("rational parameter registry values must be reduced")
        if self.state not in {"ACTIVE", "TOMBSTONE"}:
            raise ValueError("rational parameter state must be ACTIVE or TOMBSTONE")


RATIONAL_PARAMETER_REGISTRY: Final = (
    RationalParameterRegistryEntryV1(0, -2, 1, "TOMBSTONE"),
    RationalParameterRegistryEntryV1(1, -1, 1, "ACTIVE"),
    RationalParameterRegistryEntryV1(2, -1, 2, "TOMBSTONE"),
    RationalParameterRegistryEntryV1(3, 0, 1, "ACTIVE"),
    RationalParameterRegistryEntryV1(4, 1, 2, "TOMBSTONE"),
    RationalParameterRegistryEntryV1(5, 1, 1, "ACTIVE"),
    RationalParameterRegistryEntryV1(6, 2, 1, "TOMBSTONE"),
)


def rational_parameter_registry_object() -> dict[str, object]:
    """Return the immutable active+tombstone RationalParameterId/v1 registry."""

    return {
        "schema_version": "hegel-rational-parameter-registry/1",
        "registry_namespace": RATIONAL_PARAMETER_REGISTRY_NAMESPACE,
        "policy": RATIONAL_PARAMETER_REGISTRY_POLICY,
        "code_width_bits": RATIONAL_PARAMETER_CODE_WIDTH_BITS,
        "code_space_size": RATIONAL_PARAMETER_CODE_SPACE_SIZE,
        "allocated_id_count": RATIONAL_PARAMETER_ALLOCATED_ID_COUNT,
        "entries": [
            {
                "numeric_id": entry.numeric_id,
                "numerator": entry.numerator,
                "denominator": entry.denominator,
                "state": entry.state,
            }
            for entry in RATIONAL_PARAMETER_REGISTRY
        ],
        "active_ids": list(ACTIVE_RATIONAL_PARAMETER_IDS),
        "tombstoned_ids": list(TOMBSTONED_RATIONAL_PARAMETER_IDS),
        "reserved_out_of_range_ids": list(RESERVED_RATIONAL_PARAMETER_IDS),
        "next_allocatable_id": NEXT_ALLOCATABLE_RATIONAL_PARAMETER_ID,
        "id_reuse_allowed": False,
        "id_compaction_allowed": False,
        "tombstones_permanent_in_registry_lineage": True,
        "future_allocation_requires_new_registry_version_or_width": True,
        "removed_source_and_formal_error": REMOVED_RATIONAL_PARAMETER_ERROR,
        "reserved_or_unknown_id_error": UNKNOWN_RATIONAL_PARAMETER_ERROR,
    }


RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID: Final = stable_hash(
    rational_parameter_registry_object(), prefix="rational_parameter_registry_"
)


def operator_admission_semantics_object() -> dict[str, object]:
    """Bind the only semantic admission delta authorized by shrink step 2."""

    return {
        "schema_version": "hegel-operator-admission-semantics/1.2.0",
        "dsl_version": DSL_VERSION,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_aggregate_registry_diagnostic_id": AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
        "rational_parameter_registry_diagnostic_id": (
            RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID
        ),
        "active_rational_parameter_ids": list(ACTIVE_RATIONAL_PARAMETER_IDS),
        "tombstoned_rational_parameter_ids": list(
            TOMBSTONED_RATIONAL_PARAMETER_IDS
        ),
        "removed_parameter_disposition": REMOVED_RATIONAL_PARAMETER_ERROR,
        "constant_fold_result_must_be_active": True,
        "inactive_fold_result_disposition": "RETAIN_OPERATOR_AST",
        "inherited_active_aggregate_ids": list(ACTIVE_AGGREGATE_IDS),
        "inherited_tombstoned_aggregate_ids": list(TOMBSTONED_AGGREGATE_IDS),
        "inherited_removed_aggregate_disposition": REMOVED_AGGREGATE_ERROR,
        "implicit_bit_to_rational_coercion": False,
        "typing_changed": False,
        "bottom_semantics_changed": False,
        "rewrite_rules_changed": False,
        "closure_budget_changed": False,
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
    """Return the diagnostic child DSL freeze without manufacturing formal roots."""

    inherited_surface_content_ids = {
        "rational_value_grid": stable_hash(RATIONAL_VALUE_GRID, prefix="rational_grid_"),
        "scope_catalog": stable_hash(SCOPE_CATALOG, prefix="scope_catalog_"),
        "transform_catalog": stable_hash(TRANSFORM_CATALOG, prefix="transform_catalog_"),
        "leaf_expressions": stable_hash(LEAF_EXPRESSIONS, prefix="leaf_expressions_"),
        "unary_operators": stable_hash(UNARY_OPERATORS, prefix="unary_operators_"),
        "binary_operators": stable_hash(BINARY_OPERATORS, prefix="binary_operators_"),
        "ternary_operators": stable_hash(TERNARY_OPERATORS, prefix="ternary_operators_"),
        "boolean_composition": stable_hash(
            BOOLEAN_COMPOSITION, prefix="boolean_composition_"
        ),
        "forbidden_forms": stable_hash(FORBIDDEN_FORMS, prefix="forbidden_forms_"),
        "bottom_and_equivalence": stable_hash(
            BOTTOM_AND_EQUIVALENCE, prefix="bottom_equivalence_"
        ),
        "structural_limits": stable_hash(
            STRUCTURAL_LIMITS, prefix="structural_limits_"
        ),
        "closure_budget": stable_hash(CLOSURE_BUDGET, prefix="closure_budget_"),
    }
    return {
        "schema_version": "hegel-old-dsl-freeze/1.2.0",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "parent_dsl_surface_diagnostic_id": PARENT_DSL_SURFACE_DIAGNOSTIC_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "pre_registered_delta_only": "reduce RationalParameter to {-1,0,1}",
        "aggregate_registry_diagnostic_id": AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
        "rational_parameter_registry_diagnostic_id": (
            RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID
        ),
        "operator_admission_semantics_diagnostic_id": (
            OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID
        ),
        "canonical_ast_schema_id": AST_SCHEMA_ID,
        "canonical_cbor_profile_id": CBOR_PROFILE_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
        "mdl_code_table_id": MDL_CODE_TABLE_ID,
        "phase2b_contract_inherited_from": PARENT_FREEZE_VERSION,
        "phase2b_contract_changed": False,
        "primitive_domain_admission_changed": ["RationalParameter"],
        "inherited_surface_content_ids": inherited_surface_content_ids,
        "remaining_shrink_order": [step.operation for step in SHRINK_ORDER[2:]],
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
    """Fail closed on RationalParameter or inherited aggregate ID drift."""

    ids = tuple(entry.numeric_id for entry in RATIONAL_PARAMETER_REGISTRY)
    if ids != tuple(range(RATIONAL_PARAMETER_ALLOCATED_ID_COUNT)):
        raise AssertionError("RationalParameterId/v1 allocation history drift")
    active = tuple(
        entry.numeric_id for entry in RATIONAL_PARAMETER_REGISTRY if entry.state == "ACTIVE"
    )
    tombstones = tuple(
        entry.numeric_id
        for entry in RATIONAL_PARAMETER_REGISTRY
        if entry.state == "TOMBSTONE"
    )
    if active != ACTIVE_RATIONAL_PARAMETER_IDS:
        raise AssertionError("active RationalParameterId disposition drift")
    if tombstones != TOMBSTONED_RATIONAL_PARAMETER_IDS:
        raise AssertionError("tombstoned RationalParameterId disposition drift")
    values = tuple(
        (entry.numerator, entry.denominator) for entry in RATIONAL_PARAMETER_REGISTRY
    )
    if values != (
        (-2, 1),
        (-1, 1),
        (-1, 2),
        (0, 1),
        (1, 2),
        (1, 1),
        (2, 1),
    ):
        raise AssertionError("RationalParameterId/value identity drift")
    if RESERVED_RATIONAL_PARAMETER_IDS != (7,):
        raise AssertionError("three-bit code point 7 must remain reserved")
    if ACTIVE_AGGREGATE_IDS != (0, 1, 5) or TOMBSTONED_AGGREGATE_IDS != (2, 3, 4):
        raise AssertionError("shrink-1 aggregate tombstones were not inherited")


validate_registry_freeze()


__all__ = [
    "ACTIVE_AGGREGATE_IDS",
    "ACTIVE_RATIONAL_PARAMETER_IDS",
    "ACTIVE_RATIONAL_PARAMETER_VALUES",
    "DSL_VERSION",
    "FREEZE_VERSION",
    "HUMAN_AMENDMENT_ID",
    "NEXT_ALLOCATABLE_RATIONAL_PARAMETER_ID",
    "OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID",
    "PARENT_DSL_VERSION",
    "PARENT_FREEZE_VERSION",
    "RATIONAL_PARAMETER_ALLOCATED_ID_COUNT",
    "RATIONAL_PARAMETER_CODE_SPACE_SIZE",
    "RATIONAL_PARAMETER_CODE_WIDTH_BITS",
    "RATIONAL_PARAMETER_REGISTRY",
    "RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID",
    "RATIONAL_PARAMETER_REGISTRY_NAMESPACE",
    "RATIONAL_PARAMETER_REGISTRY_POLICY",
    "REMOVED_AGGREGATE_ERROR",
    "REMOVED_RATIONAL_PARAMETER_ERROR",
    "RESERVED_RATIONAL_PARAMETER_IDS",
    "SHRINK_STEP_ID",
    "SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID",
    "TOMBSTONED_AGGREGATE_IDS",
    "TOMBSTONED_RATIONAL_PARAMETER_IDS",
    "TOMBSTONED_RATIONAL_PARAMETER_VALUES",
    "UNKNOWN_RATIONAL_PARAMETER_ERROR",
    "operator_admission_semantics_object",
    "rational_parameter_registry_object",
    "shrunk_dsl_surface_object",
    "validate_registry_freeze",
]
