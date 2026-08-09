"""Normative diagnostic registry for Phase-3 shrink step 3.

This engineering child performs exactly one pre-registered language change
relative to ``hegel-old-dsl-v1.2.0``: binary operator ``add`` keeps numeric ID
0 but becomes a permanent tombstone.  ``difference`` keeps numeric ID 1 and
its inherited semantics.  The ``BinaryOperatorId/v1`` wire allocation is not
compacted, renumbered, reused, or migrated.

The objects in this module are diagnostic commitments only.  They do not
create formal CBOR/RFC6962 roots, start closure execution, or promote the
child beyond ``NOT_RUN``.
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
    RATIONAL_VALUE_GRID,
    SCOPE_CATALOG,
    SHRINK_ORDER,
    STRUCTURAL_LIMITS,
    TERNARY_OPERATORS,
    TRANSFORM_CATALOG,
    UNARY_OPERATORS,
)
from .phase3_m3_shrink3_core_v1 import (
    ACTIVE_AGGREGATE_IDS,
    ACTIVE_FORMAL_BINARY_OPERATOR_IDS,
    ACTIVE_RATIONAL_PARAMETER_IDS,
    ACTIVE_SOURCE_BINARY_OPERATOR_IDS,
    BINARY_OPERATOR_ALLOCATED_ID_COUNT,
    BINARY_OPERATOR_CODE_SPACE_SIZE,
    BINARY_OPERATOR_CODE_WIDTH_BITS,
    BINARY_OPERATOR_NAMES,
    BINARY_OPERATOR_REGISTRY_NAMESPACE,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    LEGAL_AST_TOMBSTONE_PRIORITY,
    PARENT_DSL_VERSION,
    PARENT_DIAGNOSTIC_CLAIM_LEVEL,
    PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID,
    PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS,
    PARENT_DIAGNOSTIC_RESULT_COMMIT,
    PARENT_DIAGNOSTIC_STATUS,
    PARENT_FREEZE_VERSION,
    POST_TOMBSTONE_VALIDATION_STAGES,
    PRE_TOMBSTONE_VALIDATION_STAGES,
    REMOVED_AGGREGATE_ERROR,
    REMOVED_BINARY_OPERATOR_ERROR,
    REMOVED_RATIONAL_PARAMETER_ERROR,
    RESERVED_BINARY_OPERATOR_IDS,
    RESERVED_FORMAL_BINARY_OPERATOR_ERROR,
    SEALED_DUAL_STRICT_OUTCOME_REPLAY_STATUS,
    SHRINK_STEP_ID,
    SOURCE_ALIAS_BINARY_OPERATOR_IDS,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_BINARY_OPERATOR_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
    UNALLOCATED_BINARY_OPERATOR_REGISTRY_ERROR,
    UNKNOWN_SOURCE_OPERATOR_NAME_ERROR,
)
from .phase3_shrink1_registry_v1 import (
    AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
    AST_HASH_DOMAIN,
    AST_SCHEMA_ID,
    CBOR_PROFILE_ID,
    MDL_CODE_TABLE_ID,
)
from .phase3_shrink2_registry_v1 import (
    RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID as PARENT_DSL_SURFACE_DIAGNOSTIC_ID,
)


BINARY_OPERATOR_REGISTRY_POLICY: Final = "SPARSE_PRESERVING"
NEXT_ALLOCATABLE_BINARY_OPERATOR_ID: Final = None


@dataclass(frozen=True, order=True, slots=True)
class BinaryOperatorRegistryEntryV1:
    numeric_id: int
    name: str
    state: str
    source_admission: str
    formal_canonical_admission: str

    def __post_init__(self) -> None:
        if (
            type(self.numeric_id) is not int
            or not 0 <= self.numeric_id < BINARY_OPERATOR_ALLOCATED_ID_COUNT
        ):
            raise ValueError(
                "BinaryOperatorId/v1 entries must use allocated IDs 0..6"
            )
        if self.name != BINARY_OPERATOR_NAMES[self.numeric_id]:
            raise ValueError("binary operator name/numeric identity drift")
        if self.state not in {"ACTIVE", "TOMBSTONE"}:
            raise ValueError("binary operator state must be ACTIVE or TOMBSTONE")
        if self.source_admission not in {"ACTIVE", "TOMBSTONE"}:
            raise ValueError("source admission must be ACTIVE or TOMBSTONE")
        if self.formal_canonical_admission not in {
            "ACTIVE",
            "SOURCE_ALIAS_ONLY",
            "TOMBSTONE",
        }:
            raise ValueError("unknown formal-canonical admission disposition")


BINARY_OPERATOR_REGISTRY: Final = (
    BinaryOperatorRegistryEntryV1(0, "add", "TOMBSTONE", "TOMBSTONE", "TOMBSTONE"),
    BinaryOperatorRegistryEntryV1(1, "difference", "ACTIVE", "ACTIVE", "ACTIVE"),
    BinaryOperatorRegistryEntryV1(2, "equal_exact", "ACTIVE", "ACTIVE", "ACTIVE"),
    BinaryOperatorRegistryEntryV1(3, "less_equal", "ACTIVE", "ACTIVE", "ACTIVE"),
    BinaryOperatorRegistryEntryV1(
        4,
        "greater_equal",
        "ACTIVE",
        "ACTIVE",
        "SOURCE_ALIAS_ONLY",
    ),
    BinaryOperatorRegistryEntryV1(5, "same_sign", "ACTIVE", "ACTIVE", "ACTIVE"),
    BinaryOperatorRegistryEntryV1(6, "opposite_sign", "ACTIVE", "ACTIVE", "ACTIVE"),
)


def binary_operator_registry_object() -> dict[str, object]:
    """Return the immutable sparse ``BinaryOperatorId/v1`` registry."""

    return {
        "schema_version": "hegel-binary-operator-registry/1",
        "registry_namespace": BINARY_OPERATOR_REGISTRY_NAMESPACE,
        "policy": BINARY_OPERATOR_REGISTRY_POLICY,
        "code_width_bits": BINARY_OPERATOR_CODE_WIDTH_BITS,
        "code_space_size": BINARY_OPERATOR_CODE_SPACE_SIZE,
        "allocated_id_count": BINARY_OPERATOR_ALLOCATED_ID_COUNT,
        "entries": [
            {
                "numeric_id": entry.numeric_id,
                "name": entry.name,
                "state": entry.state,
                "source_admission": entry.source_admission,
                "formal_canonical_admission": (
                    entry.formal_canonical_admission
                ),
            }
            for entry in BINARY_OPERATOR_REGISTRY
        ],
        "active_source_ids": list(ACTIVE_SOURCE_BINARY_OPERATOR_IDS),
        "active_formal_canonical_ids": list(
            ACTIVE_FORMAL_BINARY_OPERATOR_IDS
        ),
        "source_alias_ids": list(SOURCE_ALIAS_BINARY_OPERATOR_IDS),
        "source_alias_rewrite": {
            "source_id": 4,
            "source_name": "greater_equal",
            "canonical_id": 3,
            "canonical_name": "less_equal",
            "child_order": "SWAPPED",
        },
        "tombstoned_ids": list(TOMBSTONED_BINARY_OPERATOR_IDS),
        "reserved_ids": list(RESERVED_BINARY_OPERATOR_IDS),
        "next_allocatable_id": NEXT_ALLOCATABLE_BINARY_OPERATOR_ID,
        "id_reuse_allowed": False,
        "id_compaction_allowed": False,
        "automatic_operator_migration_allowed": False,
        "tombstones_permanent_in_registry_lineage": True,
        "future_allocation_requires_new_registry_version_or_width": True,
        "removed_source_and_formal_error": REMOVED_BINARY_OPERATOR_ERROR,
        "unallocated_registry_id_error": (
            UNALLOCATED_BINARY_OPERATOR_REGISTRY_ERROR
        ),
        "source_numeric_operator_id_accepted": False,
        "unknown_source_operator_name_error": UNKNOWN_SOURCE_OPERATOR_NAME_ERROR,
        "formal_reserved_id_error": RESERVED_FORMAL_BINARY_OPERATOR_ERROR,
    }


BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID: Final = stable_hash(
    binary_operator_registry_object(), prefix="binary_operator_registry_"
)


def rejection_priority_object() -> dict[str, object]:
    """Freeze source/formal rejection order without scanning malformed payloads."""

    tombstone_errors = {
        "AggregateMapId/v1": REMOVED_AGGREGATE_ERROR,
        "RationalParameterId/v1": REMOVED_RATIONAL_PARAMETER_ERROR,
        "BinaryOperatorId/v1": REMOVED_BINARY_OPERATOR_ERROR,
    }
    return {
        "schema_version": "hegel-shrink3-rejection-priority/1",
        "dsl_version": DSL_VERSION,
        "pre_tombstone_validation_stages": list(
            PRE_TOMBSTONE_VALIDATION_STAGES
        ),
        "pre_tombstone_validation_scope": (
            "generic CBOR/source syntax, AST structure and arity, parent "
            "typing, and parent registry/range validation"
        ),
        "arbitrary_payload_tombstone_scan_allowed": False,
        "complete_parent_structural_typing_registry_validation_before_tombstones": (
            True
        ),
        "relative_error_order_within_parent_validation": (
            "INHERIT_PARENT_LEFT_TO_RIGHT_RULES"
        ),
        "legal_ast_tombstone_priority": [
            {
                "priority": priority,
                "registry_namespace": namespace,
                "error": tombstone_errors[namespace],
            }
            for priority, namespace in enumerate(
                LEGAL_AST_TOMBSTONE_PRIORITY, start=1
            )
        ],
        "post_tombstone_validation_stages": list(
            POST_TOMBSTONE_VALIDATION_STAGES
        ),
        "formal_source_alias_noncanonical_check_after_tombstones": True,
        "formal_noncanonical_rewrite_check_after_binary_tombstone": True,
        "source_normalization_after_tombstones": True,
        "source_removed_add_checked_before_child_normalization": True,
        "left_to_right_child_validation_inherited": True,
    }


REJECTION_PRIORITY_DIAGNOSTIC_ID: Final = stable_hash(
    rejection_priority_object(), prefix="rejection_priority_"
)


def operator_admission_semantics_object() -> dict[str, object]:
    """Bind the sole shrink-3 delta and all inherited admission lineages."""

    return {
        "schema_version": "hegel-operator-admission-semantics/1.3.0",
        "dsl_version": DSL_VERSION,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "binary_operator_registry_diagnostic_id": (
            BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID
        ),
        "rejection_priority_diagnostic_id": REJECTION_PRIORITY_DIAGNOSTIC_ID,
        "removed_binary_operator": {"numeric_id": 0, "name": "add"},
        "removed_operator_disposition": REMOVED_BINARY_OPERATOR_ERROR,
        "automatic_operator_migration_allowed": False,
        "removed_operator_rewrite_or_fold_allowed": False,
        "retained_difference": {"numeric_id": 1, "name": "difference"},
        "difference_semantics_changed": False,
        "active_source_binary_operator_ids": list(
            ACTIVE_SOURCE_BINARY_OPERATOR_IDS
        ),
        "active_formal_canonical_binary_operator_ids": list(
            ACTIVE_FORMAL_BINARY_OPERATOR_IDS
        ),
        "source_alias_binary_operator_ids": list(
            SOURCE_ALIAS_BINARY_OPERATOR_IDS
        ),
        "parent_aggregate_registry_diagnostic_id": (
            AGGREGATE_REGISTRY_DIAGNOSTIC_ID
        ),
        "inherited_active_aggregate_ids": list(ACTIVE_AGGREGATE_IDS),
        "inherited_tombstoned_aggregate_ids": list(
            TOMBSTONED_AGGREGATE_IDS
        ),
        "inherited_removed_aggregate_disposition": REMOVED_AGGREGATE_ERROR,
        "parent_rational_parameter_registry_diagnostic_id": (
            RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID
        ),
        "inherited_active_rational_parameter_ids": list(
            ACTIVE_RATIONAL_PARAMETER_IDS
        ),
        "inherited_tombstoned_rational_parameter_ids": list(
            TOMBSTONED_RATIONAL_PARAMETER_IDS
        ),
        "inherited_removed_rational_parameter_disposition": (
            REMOVED_RATIONAL_PARAMETER_ERROR
        ),
        "implicit_bit_to_rational_coercion": False,
        "typing_changed": False,
        "bottom_semantics_changed": False,
        "surviving_operator_rewrite_rules_changed": False,
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
    """Return the engineering child surface without manufacturing formal roots."""

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
        "boolean_composition": stable_hash(
            BOOLEAN_COMPOSITION, prefix="boolean_composition_"
        ),
        "forbidden_forms": stable_hash(
            FORBIDDEN_FORMS, prefix="forbidden_forms_"
        ),
        "bottom_and_equivalence": stable_hash(
            BOTTOM_AND_EQUIVALENCE, prefix="bottom_equivalence_"
        ),
        "structural_limits": stable_hash(
            STRUCTURAL_LIMITS, prefix="structural_limits_"
        ),
        "closure_budget": stable_hash(CLOSURE_BUDGET, prefix="closure_budget_"),
    }
    surviving_binary_semantics = {
        str(numeric_id): stable_hash(
            BINARY_OPERATORS[numeric_id],
            prefix=f"binary_operator_{numeric_id}_",
        )
        for numeric_id in ACTIVE_SOURCE_BINARY_OPERATOR_IDS
    }
    return {
        "schema_version": "hegel-old-dsl-freeze/1.3.0",
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
            "remove add (BinaryOperatorId 0); retain difference "
            "(BinaryOperatorId 1)"
        ),
        "aggregate_registry_diagnostic_id": AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
        "rational_parameter_registry_diagnostic_id": (
            RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID
        ),
        "binary_operator_registry_diagnostic_id": (
            BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID
        ),
        "rejection_priority_diagnostic_id": REJECTION_PRIORITY_DIAGNOSTIC_ID,
        "operator_admission_semantics_diagnostic_id": (
            OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID
        ),
        "canonical_ast_schema_id": AST_SCHEMA_ID,
        "canonical_cbor_profile_id": CBOR_PROFILE_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
        "mdl_code_table_id": MDL_CODE_TABLE_ID,
        "phase2b_contract_inherited_from": PARENT_FREEZE_VERSION,
        "phase2b_contract_changed": False,
        "inherited_surface_content_ids": inherited_surface_content_ids,
        "parent_binary_operator_catalog_content_id": stable_hash(
            BINARY_OPERATORS, prefix="binary_operators_"
        ),
        "surviving_binary_operator_semantics_content_ids": (
            surviving_binary_semantics
        ),
        "remaining_shrink_order": [step.operation for step in SHRINK_ORDER[3:]],
        "canonical_program_budget": CLOSURE_BUDGET.max_canonical_program_count,
        "raw_application_cap": CLOSURE_BUDGET.max_raw_operator_applications,
        "surviving_ast_bytes_stable": True,
        "surviving_ast_hash_stable": True,
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
    """Fail closed on binary IDs, inherited registries, or priority drift."""

    ids = tuple(entry.numeric_id for entry in BINARY_OPERATOR_REGISTRY)
    if ids != tuple(range(BINARY_OPERATOR_ALLOCATED_ID_COUNT)):
        raise AssertionError("BinaryOperatorId/v1 allocation history drift")
    names = tuple(entry.name for entry in BINARY_OPERATOR_REGISTRY)
    if names != BINARY_OPERATOR_NAMES or names != tuple(
        spec.expression_id for spec in BINARY_OPERATORS
    ):
        raise AssertionError("BinaryOperatorId/name identity drift")
    active_source = tuple(
        entry.numeric_id
        for entry in BINARY_OPERATOR_REGISTRY
        if entry.source_admission == "ACTIVE"
    )
    active_formal = tuple(
        entry.numeric_id
        for entry in BINARY_OPERATOR_REGISTRY
        if entry.formal_canonical_admission == "ACTIVE"
    )
    aliases = tuple(
        entry.numeric_id
        for entry in BINARY_OPERATOR_REGISTRY
        if entry.formal_canonical_admission == "SOURCE_ALIAS_ONLY"
    )
    tombstones = tuple(
        entry.numeric_id
        for entry in BINARY_OPERATOR_REGISTRY
        if entry.state == "TOMBSTONE"
    )
    if active_source != ACTIVE_SOURCE_BINARY_OPERATOR_IDS:
        raise AssertionError("active source BinaryOperatorId disposition drift")
    if active_formal != ACTIVE_FORMAL_BINARY_OPERATOR_IDS:
        raise AssertionError("formal-canonical BinaryOperatorId disposition drift")
    if aliases != SOURCE_ALIAS_BINARY_OPERATOR_IDS:
        raise AssertionError("source-only BinaryOperatorId alias drift")
    if tombstones != TOMBSTONED_BINARY_OPERATOR_IDS:
        raise AssertionError("tombstoned BinaryOperatorId disposition drift")
    if BINARY_OPERATOR_REGISTRY[0].name != "add":
        raise AssertionError("add must remain permanent tombstone ID 0")
    if BINARY_OPERATOR_REGISTRY[1].name != "difference":
        raise AssertionError("difference must remain active ID 1")
    if RESERVED_BINARY_OPERATOR_IDS != (7,):
        raise AssertionError(
            "three-bit BinaryOperatorId code point 7 must stay reserved"
        )
    if ACTIVE_AGGREGATE_IDS != (0, 1, 5) or TOMBSTONED_AGGREGATE_IDS != (
        2,
        3,
        4,
    ):
        raise AssertionError("shrink-1 aggregate registry inheritance drift")
    if ACTIVE_RATIONAL_PARAMETER_IDS != (1, 3, 5) or (
        TOMBSTONED_RATIONAL_PARAMETER_IDS != (0, 2, 4, 6)
    ):
        raise AssertionError("shrink-2 rational registry inheritance drift")
    if LEGAL_AST_TOMBSTONE_PRIORITY != (
        "AggregateMapId/v1",
        "RationalParameterId/v1",
        "BinaryOperatorId/v1",
    ):
        raise AssertionError("legal-AST tombstone priority drift")


validate_registry_freeze()


__all__ = [
    "ACTIVE_FORMAL_BINARY_OPERATOR_IDS",
    "ACTIVE_SOURCE_BINARY_OPERATOR_IDS",
    "BINARY_OPERATOR_REGISTRY",
    "BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID",
    "BINARY_OPERATOR_REGISTRY_POLICY",
    "BinaryOperatorRegistryEntryV1",
    "DSL_VERSION",
    "FREEZE_VERSION",
    "HUMAN_AMENDMENT_ID",
    "NEXT_ALLOCATABLE_BINARY_OPERATOR_ID",
    "OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID",
    "PARENT_DSL_VERSION",
    "PARENT_DIAGNOSTIC_CLAIM_LEVEL",
    "PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID",
    "PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS",
    "PARENT_DIAGNOSTIC_RESULT_COMMIT",
    "PARENT_DIAGNOSTIC_STATUS",
    "PARENT_FREEZE_VERSION",
    "REJECTION_PRIORITY_DIAGNOSTIC_ID",
    "REMOVED_BINARY_OPERATOR_ERROR",
    "SHRINK_STEP_ID",
    "SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID",
    "SOURCE_ALIAS_BINARY_OPERATOR_IDS",
    "TOMBSTONED_BINARY_OPERATOR_IDS",
    "binary_operator_registry_object",
    "operator_admission_semantics_object",
    "rejection_priority_object",
    "shrunk_dsl_surface_object",
    "validate_registry_freeze",
]
