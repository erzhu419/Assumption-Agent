from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from hegel_machine.hashing import stable_hash
from hegel_machine.phase3_dsl_v1 import (
    AGGREGATE_CATALOG,
    AGGREGATE_MAP_IDS,
    ALL_EXPRESSIONS,
    BINARY_OPERATORS,
    BINARY_XOR_SANITY,
    BOOLEAN_COMPOSITION,
    BOTTOM,
    BOTTOM_AND_EQUIVALENCE,
    CLOSED_INTERVAL_GRID,
    CLOSURE_BUDGET,
    CONTEXT_IDS,
    DSL_EXECUTION_STATE,
    DSL_VERSION,
    ENTITY_SLOTS,
    FORBIDDEN_FORMS,
    HIDDEN_TARGET_REGISTRY,
    IDENTIFIER_REGISTRIES,
    LEAF_EXPRESSIONS,
    NO_FALSE_INVENTION,
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_SPLITS,
    ODD_REDUCTION_TARGET,
    ODD_REDUCTION_UNIVERSE,
    OLD_DSL_V1,
    OMITTED_SINK_UNIVERSE,
    PRIMITIVE_DOMAINS,
    PRIMITIVE_DOMAIN_BY_ID,
    PRIMITIVE_SORT_IDS,
    RATIONAL_PARAMETER_GRID,
    RATIONAL_VALUE_GRID,
    SCOPE_CATALOG,
    SCOPE_IDS,
    SHRINK_ORDER,
    STRUCTURAL_LIMITS,
    TERNARY_OPERATORS,
    TOLERANCE_GRID,
    TRANSFORM_CATALOG,
    UNARY_OPERATORS,
    BinaryXorStatus,
    ClosureStatus,
    RationalAtom,
    TargetPreflightStatus,
    TargetRegistryPredicate,
    evaluate_aggregate,
    evaluate_operator,
    odd_reduction,
    select_first_outside_target,
)


def _fractions(atoms):
    return tuple(atom.as_fraction() for atom in atoms)


def test_all_primitive_domains_have_exact_frozen_cardinalities():
    expected = {
        "Bool": 2,
        "Bit": 2,
        "Sign": 3,
        "BoundedInt": 17,
        "RationalValue": 663,
        "RationalParameter": 7,
        "Tolerance": 3,
        "IntervalEndpoint": 9,
        "ClosedInterval": 45,
        "EntitySlot": 8,
        "Index": 8,
        "QuantityId": 2,
        "ContextId": 4,
        "RoleId": 4,
        "ScaleId": 2,
        "TaskId": 2,
        "ScopeId": 4,
        "AggregateMapId": 6,
        "TransformId": 4,
    }
    assert {domain.sort_id: domain.cardinality for domain in PRIMITIVE_DOMAINS} == expected
    assert len(PRIMITIVE_DOMAINS) == len(expected) == 19
    assert len(PRIMITIVE_SORT_IDS) == len(set(PRIMITIVE_SORT_IDS)) == 19
    assert tuple(PRIMITIVE_DOMAIN_BY_ID) == PRIMITIVE_SORT_IDS
    assert PRIMITIVE_SORT_IDS.count("Bit") == 1
    assert all(len(domain.values) == domain.cardinality for domain in PRIMITIVE_DOMAINS)


def test_rational_interval_and_parameter_grids_are_exact():
    assert len(RATIONAL_VALUE_GRID) == 663
    assert RATIONAL_VALUE_GRID[0] == RationalAtom(-64, 1)
    assert RATIONAL_VALUE_GRID[-1] == RationalAtom(64, 1)
    assert all(
        abs(atom.numerator) <= 64 and 1 <= atom.denominator <= 8
        for atom in RATIONAL_VALUE_GRID
    )
    assert _fractions(RATIONAL_PARAMETER_GRID) == (
        Fraction(-2),
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(2),
    )
    assert _fractions(TOLERANCE_GRID) == (
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
    )
    assert len(CLOSED_INTERVAL_GRID) == 45
    assert CLOSED_INTERVAL_GRID[0].lower == -8
    assert CLOSED_INTERVAL_GRID[-1].upper == 8


def test_private_registries_and_catalog_cardinalities_are_frozen():
    assert IDENTIFIER_REGISTRIES.entity_slots == ENTITY_SLOTS == tuple(
        f"e{i}" for i in range(8)
    )
    assert IDENTIFIER_REGISTRIES.context_ids == CONTEXT_IDS == ("c0", "c1", "c2", "c3")
    assert len(SCOPE_CATALOG) == 4
    assert SCOPE_IDS == (
        "scope_all_observed_v1",
        "scope_primary_only_v1",
        "scope_boundary_only_v1",
        "control_volume_all_observed_v1",
    )
    assert AGGREGATE_MAP_IDS == (
        "sum_v1",
        "count_nonzero_v1",
        "mean_v1",
        "min_v1",
        "max_v1",
        "signed_balance_v1",
    )
    assert tuple(spec.transform_id for spec in TRANSFORM_CATALOG) == (
        "identity_v1",
        "negate_v1",
        "scale_by_2_v1",
        "scale_by_half_v1",
    )
    assert all(spec.adapter_only and not spec.old_dsl_composable for spec in TRANSFORM_CATALOG)


def test_typed_expression_surface_is_exact_and_has_no_xor_or_modulo():
    assert len(LEAF_EXPRESSIONS) == 6
    assert len(UNARY_OPERATORS) == 4
    assert len(BINARY_OPERATORS) == 7
    assert len(TERNARY_OPERATORS) == 1
    assert len(BOOLEAN_COMPOSITION) == 1
    assert len(ALL_EXPRESSIONS) == 19
    assert tuple(spec.expression_id for spec in UNARY_OPERATORS) == (
        "bit_to_scalar",
        "int_to_scalar",
        "absolute",
        "sign",
    )
    assert tuple(spec.expression_id for spec in BINARY_OPERATORS) == (
        "add",
        "difference",
        "equal_exact",
        "less_equal",
        "greater_equal",
        "same_sign",
        "opposite_sign",
    )
    assert TERNARY_OPERATORS[0].expression_id == "approx_equal"
    assert BOOLEAN_COMPOSITION[0].accepted_arities == (1, 2, 3)
    assert FORBIDDEN_FORMS == (
        "OR",
        "XOR",
        "NOT(compound)",
        "modulo",
        "parity",
        "arbitrary_lookup_table",
        "recursive_fold",
        "user_defined_reducer",
        "case_ID_branch",
    )


def test_scalar_reference_semantics_are_exact_and_bottom_strict():
    assert evaluate_operator("difference", (Fraction(1, 2), Fraction(1, 4))) == Fraction(1, 4)
    assert evaluate_operator("absolute", (Fraction(-2),)) == Fraction(2)
    assert evaluate_operator("sign", (Fraction(0),)) == 0
    assert evaluate_operator("approx_equal", (Fraction(1), Fraction(3, 4), Fraction(1, 4))) is True
    assert evaluate_operator("opposite_sign", (-1, 1)) is True
    assert evaluate_operator("same_sign", (0, 0)) is True
    assert evaluate_operator("top_level_AND", (True, True, False)) is False
    assert evaluate_operator("add", (Fraction(64), Fraction(1))) is BOTTOM
    assert evaluate_operator("difference", (BOTTOM, Fraction(0))) is BOTTOM
    assert not BOTTOM_AND_EQUIVALENCE.bottom_is_observable
    assert BOTTOM_AND_EQUIVALENCE.bottom_disqualifies_exact_match
    assert BOTTOM_AND_EQUIVALENCE.rational_equivalence == "exact_fraction_equality"
    with pytest.raises(TypeError):
        evaluate_operator("bit_to_scalar", (True,))


def test_aggregate_reference_semantics_are_exact_and_bounded():
    values = (Fraction(1), Fraction(2), Fraction(3))
    assert evaluate_aggregate("sum_v1", values) == Fraction(6)
    assert evaluate_aggregate("mean_v1", values) == Fraction(2)
    assert evaluate_aggregate("min_v1", values) == Fraction(1)
    assert evaluate_aggregate("max_v1", values) == Fraction(3)
    assert evaluate_aggregate("count_nonzero_v1", (Fraction(0), Fraction(1))) == 1
    assert evaluate_aggregate("mean_v1", ()) is BOTTOM
    assert evaluate_aggregate("signed_balance_v1", values) is BOTTOM
    assert evaluate_aggregate(
        "signed_balance_v1", values, orientations=(1, 1, -1)
    ) == Fraction(0)
    assert tuple(spec.map_id for spec in AGGREGATE_CATALOG) == AGGREGATE_MAP_IDS


def test_structural_limits_and_frozen_shrink_order_are_exact():
    assert STRUCTURAL_LIMITS.max_total_ast_depth == 4
    assert STRUCTURAL_LIMITS.max_total_node_count == 7
    assert STRUCTURAL_LIMITS.max_top_level_clauses == 3
    assert STRUCTURAL_LIMITS.max_distinct_bit_slots == 4
    assert STRUCTURAL_LIMITS.max_aggregate_leaves == 1
    assert STRUCTURAL_LIMITS.max_scope_clauses == 2
    assert STRUCTURAL_LIMITS.max_old_law_composition_depth == 2
    assert STRUCTURAL_LIMITS.max_fitted_scalar_parameters == 3
    assert STRUCTURAL_LIMITS.leaf_depth == 0
    assert tuple(step.order for step in SHRINK_ORDER) == (1, 2, 3, 4, 5, 6)
    assert SHRINK_ORDER[0].operation == "remove mean_v1, min_v1, max_v1"
    assert SHRINK_ORDER[-1].operation == "reduce max_total_ast_depth from 4 to 3"


def test_search_budget_counts_prequotient_programs_and_binds_all_roots():
    assert CLOSURE_BUDGET.max_canonical_program_count == 50_000
    assert CLOSURE_BUDGET.max_raw_operator_applications == 5_000_000
    assert "before extensional quotient" in CLOSURE_BUDGET.canonical_counted_object
    assert CLOSURE_BUDGET.traversal_sort_keys == (
        "total_ast_depth ascending",
        "total_node_count ascending",
        "output_sort_id ascending",
        "root_operator_id ascending",
        "canonical_ast_cbor bytes lexicographically ascending",
    )
    assert CLOSURE_BUDGET.dynamic_programming_bucket == (
        "output_sort",
        "depth",
        "node_count",
    )
    assert len(CLOSURE_BUDGET.replay_bound_roots) == 13
    assert "canonical_program_archive_root" in CLOSURE_BUDGET.replay_bound_roots
    assert "enumeration_exhaustion_receipt_root" in CLOSURE_BUDGET.replay_bound_roots


def test_freeze_is_content_addressed_but_explicitly_not_a_closure_result():
    assert OLD_DSL_V1.dsl_version == DSL_VERSION == "hegel-old-dsl-v1.0.0"
    assert OLD_DSL_V1.content_id.startswith("dsl_spec_")
    assert OLD_DSL_V1.operator_semantics_id.startswith("operator_semantics_")
    assert OLD_DSL_V1.rational_grid_id.startswith("rational_grid_")
    assert len(OLD_DSL_V1.content_id.rsplit("_", 1)[1]) == 64
    assert DSL_EXECUTION_STATE.surface_parameter_tables_frozen
    assert not DSL_EXECUTION_STATE.strict_canonical_ast_schema_frozen
    assert not DSL_EXECUTION_STATE.canonicalizer_implemented
    assert not DSL_EXECUTION_STATE.python_complete_enumerator_implemented
    assert not DSL_EXECUTION_STATE.rust_complete_enumerator_implemented
    assert DSL_EXECUTION_STATE.closure_status is ClosureStatus.NOT_RUN
    assert not DSL_EXECUTION_STATE.outside_frozen_closure_certificate_issued
    with pytest.raises(FrozenInstanceError):
        OLD_DSL_V1.dsl_version = "changed"  # type: ignore[misc]


def test_generic_odd_reduction_universe_is_complete_balanced_and_ordered():
    assert len(ODD_REDUCTION_UNIVERSE) == 480
    assert tuple(row.universe_index for row in ODD_REDUCTION_UNIVERSE) == tuple(range(480))
    by_size = {
        size: [row for row in ODD_REDUCTION_UNIVERSE if row.set_size == size]
        for size in range(5, 9)
    }
    assert {size: len(rows) for size, rows in by_size.items()} == {
        5: 32,
        6: 64,
        7: 128,
        8: 256,
    }
    assert {
        size: (sum(row.target_output == 0 for row in rows), sum(row.target_output == 1 for row in rows))
        for size, rows in by_size.items()
    } == {5: (16, 16), 6: (32, 32), 7: (64, 64), 8: (128, 128)}
    assert ODD_REDUCTION_UNIVERSE[0].bits == (0, 0, 0, 0, 0)
    assert ODD_REDUCTION_UNIVERSE[-1].bits == (1, 1, 1, 1, 1, 1, 1, 1)
    assert odd_reduction((1, 0, 1, 0, 1)) == 1
    with pytest.raises(ValueError):
        odd_reduction((0, 1))


def test_generic_odd_reduction_splits_are_exact_and_separate_from_full_table():
    assert tuple(
        (quota.set_size, quota.discovery_train, quota.validation, quota.sealed_prediction)
        for quota in ODD_REDUCTION_SPLITS
    ) == (
        (5, 12, 6, 14),
        (6, 26, 12, 26),
        (7, 52, 26, 50),
        (8, 102, 52, 102),
    )
    assert sum(quota.discovery_train for quota in ODD_REDUCTION_SPLITS) == 192
    assert sum(quota.validation for quota in ODD_REDUCTION_SPLITS) == 96
    assert sum(quota.sealed_prediction for quota in ODD_REDUCTION_SPLITS) == 192
    assert not ODD_REDUCTION_TARGET.full_truth_table_visible_to_synthesis_agent
    assert ODD_REDUCTION_TARGET.preflight_status is TargetPreflightStatus.AWAITING_COMPLETE_CLOSURE
    assert not ODD_REDUCTION_TARGET.outside_frozen_closure_certificate_issued
    assert ODD_REDUCTION_TARGET.diagnostic_universe_content_id.startswith(
        "bounded_universe_"
    )
    assert ODD_REDUCTION_TARGET.diagnostic_target_table_content_id.startswith(
        "target_truth_table_"
    )
    assert ODD_REDUCTION_TARGET.diagnostic_universe_content_id == stable_hash(
        tuple(
            (row.universe_index, row.set_size, row.bits)
            for row in ODD_REDUCTION_UNIVERSE
        ),
        prefix="bounded_universe_",
    )
    assert ODD_REDUCTION_TARGET.diagnostic_universe_content_id != stable_hash(
        ODD_REDUCTION_UNIVERSE,
        prefix="bounded_universe_",
    )
    assert ODD_REDUCTION_TARGET.diagnostic_target_table_content_id == stable_hash(
        tuple(
            (row.universe_index, row.bits, row.target_output)
            for row in ODD_REDUCTION_UNIVERSE
        ),
        prefix="target_truth_table_",
    )


def test_fallback_registry_order_prevalence_and_selection_are_frozen():
    assert tuple(entry.priority for entry in HIDDEN_TARGET_REGISTRY) == (1, 2, 3)
    assert tuple(entry.predicate for entry in HIDDEN_TARGET_REGISTRY) == (
        TargetRegistryPredicate.COUNT_MOD_2_EQ_1,
        TargetRegistryPredicate.COUNT_MOD_3_EQ_1,
        TargetRegistryPredicate.COUNT_IN_PRIME_SET,
    )
    assert tuple(entry.positive_count for entry in HIDDEN_TARGET_REGISTRY) == (240, 160, 288)
    assert tuple(entry.prevalence for entry in HIDDEN_TARGET_REGISTRY) == (
        Fraction(1, 2),
        Fraction(1, 3),
        Fraction(3, 5),
    )
    matches = {entry.target_id: 1 for entry in HIDDEN_TARGET_REGISTRY}
    matches[HIDDEN_TARGET_REGISTRY[1].target_id] = 0
    assert select_first_outside_target(matches) == HIDDEN_TARGET_REGISTRY[1]


def test_binary_xor_is_only_a_design_sanity_until_executable_closure():
    assert BINARY_XOR_SANITY.status is BinaryXorStatus.TARGET_DESIGN_SANITY_ONLY
    assert BINARY_XOR_SANITY.candidate_old_dsl_program == (
        "absolute(difference(bit_at(0), bit_at(1)))"
    )
    assert BINARY_XOR_SANITY.type_explicit_candidate_old_dsl_program == (
        "absolute(difference(bit_to_scalar(bit_at(0)), "
        "bit_to_scalar(bit_at(1))))"
    )
    assert not BINARY_XOR_SANITY.implicit_bit_to_rational_coercion_frozen
    assert not BINARY_XOR_SANITY.source_candidate_typechecks_under_frozen_typing
    assert BINARY_XOR_SANITY.truth_table == (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    assert len(BINARY_XOR_SANITY.required_machine_evidence) == 5
    assert not BINARY_XOR_SANITY.formal_language_verdict_issued


def test_observed_omitted_sink_universe_has_all_and_only_85_legal_rows():
    assert len(OMITTED_SINK_UNIVERSE) == 85
    assert tuple(row.universe_index for row in OMITTED_SINK_UNIVERSE) == tuple(range(85))
    tuples = {
        (row.inflow_a, row.inflow_b, row.primary_outflow, row.auxiliary_outflow)
        for row in OMITTED_SINK_UNIVERSE
    }
    expected = {
        (a, b, c, a + b - c)
        for a in range(5)
        for b in range(5)
        for c in range(5)
        if 0 <= a + b - c <= 4
    }
    assert tuples == expected
    assert all(row.full_balance_residual == 0 for row in OMITTED_SINK_UNIVERSE)
    assert all(row.baseline_residual == row.auxiliary_outflow for row in OMITTED_SINK_UNIVERSE)


def test_sink_is_observed_scope_refinement_with_exact_support_contract():
    control = OBSERVED_OMITTED_SINK_CONTROL
    assert control.control_id == "CONTROL_P3A_OBSERVED_OMITTED_SINK_V1"
    assert control.observed_channels == (
        "inflow_a",
        "inflow_b",
        "primary_outflow",
        "auxiliary_outflow",
    )
    assert control.all_channels_present_in_public_typed_evidence
    assert not control.latent_sink_allowed
    assert control.universe_rows == 85
    assert control.aggregation_map_id == "signed_balance_v1"
    assert control.correct_scope_id == "control_volume_all_observed_v1"
    assert control.baseline_scope_id == "scope_primary_only_v1"
    assert control.baseline_scope_id in SCOPE_IDS
    assert control.source_document_baseline_label == "control_volume_primary_only_v1"
    assert control.support.discovery_support_total == 16
    assert control.support.discovery_support_per_scale == 8
    assert control.support.validation_support_total == 8
    assert control.support.sealed_support_total == 8
    assert control.support.discovery_d_zero_per_scale == 4
    assert control.support.discovery_d_positive_per_scale == 4
    assert control.diagnostic_universe_content_id == stable_hash(
        tuple(
            (
                row.universe_index,
                row.inflow_a,
                row.inflow_b,
                row.primary_outflow,
                row.auxiliary_outflow,
            )
            for row in OMITTED_SINK_UNIVERSE
        ),
        prefix="bounded_universe_",
    )
    assert control.diagnostic_target_table_content_id == stable_hash(
        tuple(
            (row.universe_index, int(row.full_balance_residual == 0))
            for row in OMITTED_SINK_UNIVERSE
        ),
        prefix="target_truth_table_",
    )
    assert not control.outside_frozen_closure_certificate_issued


def test_no_false_invention_gate_requires_every_exact_condition():
    valid = dict(
        old_closure_exact_match_count=1,
        best_old_program_error=0,
        decision="IN_LANGUAGE_REFINEMENT",
        promoted_new_symbol_count=0,
        outside_frozen_closure_certificate_count=0,
        sealed_false_invention_rate=Fraction(0),
    )
    assert NO_FALSE_INVENTION.passes(**valid)
    for field, bad_value in (
        ("old_closure_exact_match_count", 0),
        ("best_old_program_error", 1),
        ("decision", "NEW_RELATION"),
        ("promoted_new_symbol_count", 1),
        ("outside_frozen_closure_certificate_count", 1),
        ("sealed_false_invention_rate", Fraction(1, 100)),
    ):
        invalid = dict(valid)
        invalid[field] = bad_value
        assert not NO_FALSE_INVENTION.passes(**invalid)
