from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import json
from pathlib import Path

import pytest

from hegel_machine.phase3_contract import (
    DEFAULT_PHASE3_PREREGISTRATION,
    FROZEN_DSL_LIMITS,
    FROZEN_FORBIDDEN_SYMBOLS,
    FROZEN_LEAVES,
    FROZEN_OPERATORS,
    FROZEN_SORTS,
    XOR2_ABSOLUTE_DIFFERENCE_WITNESS,
    AdequacyVerdict,
    ClosureAssessment,
    ClosureEnumerationReceipt,
    ClosureRunStatus,
    DslLimits,
    MdlGainReceipt,
    Phase3PrerequisiteContract,
    ReadinessBlocker,
    TargetRole,
    assess_closure,
    mdl_gain_gate,
    phase3_preregistration_report,
    xor2_via_absolute_difference,
)


def content_id(prefix: str, digit: str) -> str:
    return f"{prefix}_{digit * 64}"


def ready_contract() -> Phase3PrerequisiteContract:
    return replace(
        DEFAULT_PHASE3_PREREGISTRATION,
        canonicalizer_implementation_id=content_id("canonicalizer", "5"),
        enumerator_implementation_id=content_id("enumerator", "6"),
    )


def closure_receipt(
    contract: Phase3PrerequisiteContract,
    *,
    target_role: TargetRole = TargetRole.OUTSIDE_TARGET,
    closure_status: ClosureRunStatus = ClosureRunStatus.COMPLETE,
    semantics_total: bool = True,
    enumerated_count: int = 123,
    match_ids: tuple[str, ...] = (),
    first_out_of_budget_program_id: str | None = None,
) -> ClosureEnumerationReceipt:
    target_id = (
        contract.parity_target_id
        if target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_control_id
    )
    assert target_id is not None
    bounded_universe_diagnostic_id = (
        contract.bounded_universe_diagnostic_id
        if target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_universe_diagnostic_id
    )
    target_table_diagnostic_id = (
        contract.target_table_diagnostic_id
        if target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_target_table_diagnostic_id
    )
    assert bounded_universe_diagnostic_id is not None
    assert target_table_diagnostic_id is not None
    assert contract.dsl_spec_id is not None
    assert contract.operator_semantics_id is not None
    assert contract.equivalence_contract_id is not None
    assert contract.enumerator_implementation_id is not None
    return ClosureEnumerationReceipt(
        contract_id=contract.content_id,
        dsl_spec_id=contract.dsl_spec_id,
        target_id=target_id,
        target_role=target_role,
        bounded_universe_diagnostic_id=bounded_universe_diagnostic_id,
        operator_semantics_id=contract.operator_semantics_id,
        equivalence_contract_id=contract.equivalence_contract_id,
        enumerator_implementation_id=contract.enumerator_implementation_id,
        search_budget=50_000,
        enumerated_canonical_program_count=enumerated_count,
        raw_operator_application_count=(
            5_000_000
            if closure_status is ClosureRunStatus.INCONCLUSIVE_BUDGET
            else 1_000
        ),
        closure_cardinality=(
            enumerated_count
            if closure_status is ClosureRunStatus.COMPLETE
            else None
        ),
        closure_status=closure_status,
        frontier_exhausted=closure_status is ClosureRunStatus.COMPLETE,
        all_type_buckets_closed=closure_status is ClosureRunStatus.COMPLETE,
        raw_expansion_limit_hit=(
            closure_status is ClosureRunStatus.INCONCLUSIVE_BUDGET
        ),
        wall_clock_abort_hit=False,
        first_out_of_budget_program_id=first_out_of_budget_program_id,
        semantics_total=semantics_total,
        extensional_match_program_ids=match_ids,
        closure_root=content_id("closure_root", "b"),
        target_table_diagnostic_id=target_table_diagnostic_id,
    )


def mdl_receipt(
    *,
    code_table_id: str,
    new_data_bits: Fraction = Fraction(940, 1),
) -> MdlGainReceipt:
    return MdlGainReceipt(
        mdl_code_table_id=code_table_id,
        scoring_partition_id=content_id("scoring_partition", "d"),
        old_program_length_bits=Fraction(10, 1),
        old_data_length_bits=Fraction(1000, 1),
        new_program_length_bits=Fraction(20, 1),
        new_data_length_bits=new_data_bits,
    )


def test_decided_old_dsl_surface_and_limits_are_frozen():
    contract = DEFAULT_PHASE3_PREREGISTRATION
    assert contract.freeze_version == "hegel-freeze-p2b-p3-v1.0.1"
    assert contract.sorts == FROZEN_SORTS
    assert contract.leaves == FROZEN_LEAVES
    assert contract.operators == FROZEN_OPERATORS
    assert contract.forbidden_symbols == FROZEN_FORBIDDEN_SYMBOLS
    assert contract.limits == DslLimits(
        maximum_relation_arity=3,
        maximum_entity_set_size=8,
        maximum_ast_depth=4,
        maximum_ast_node_count=7,
        maximum_top_level_clauses=3,
        maximum_distinct_bit_slots=4,
        maximum_aggregate_leaves=1,
        maximum_composition_depth=2,
        maximum_fitted_parameters=3,
        maximum_scope_clauses=2,
        maximum_canonical_programs=50_000,
        maximum_raw_operator_applications=5_000_000,
    )
    assert contract.limits == FROZEN_DSL_LIMITS
    assert contract.shadow_only
    assert not contract.active_promotion_authorized


def test_contract_is_immutable_content_addressed_and_replay_stable():
    contract = DEFAULT_PHASE3_PREREGISTRATION
    with pytest.raises(FrozenInstanceError):
        contract.shadow_only = False  # type: ignore[misc]
    assert contract.content_id == Phase3PrerequisiteContract().content_id
    assert contract.content_id.startswith("phase3_preregistration_")
    assert len(contract.content_id.rsplit("_", 1)[1]) == 64


def test_frozen_decisions_cannot_be_changed_by_constructor_input():
    with pytest.raises(ValueError, match="operators differ"):
        replace(
            DEFAULT_PHASE3_PREREGISTRATION,
            operators=FROZEN_OPERATORS + ("xor",),
        )
    with pytest.raises(ValueError, match="DSL limits differ"):
        replace(
            DEFAULT_PHASE3_PREREGISTRATION,
            limits=replace(FROZEN_DSL_LIMITS, maximum_ast_depth=5),
        )
    with pytest.raises(ValueError, match="shadow-only"):
        replace(
            DEFAULT_PHASE3_PREREGISTRATION,
            active_promotion_authorized=True,
        )


def test_exact_bindings_are_resolved_and_implementations_remain_blocked():
    contract = DEFAULT_PHASE3_PREREGISTRATION
    for blocker in (
        ReadinessBlocker.RATIONAL_GRID,
        ReadinessBlocker.BOUNDED_UNIVERSE,
        ReadinessBlocker.OPERATOR_SEMANTICS,
        ReadinessBlocker.EQUIVALENCE_CONTRACT,
        ReadinessBlocker.MDL_CODE_TABLE,
        ReadinessBlocker.PARITY_TARGET,
        ReadinessBlocker.HIDDEN_SINK_CONTROL,
        ReadinessBlocker.HIDDEN_GENERATOR,
    ):
        assert blocker not in contract.readiness_blockers
    assert contract.readiness_blockers == (
        ReadinessBlocker.CANONICALIZER,
        ReadinessBlocker.ENUMERATOR,
        ReadinessBlocker.CANONICAL_AST_SCHEMA,
        ReadinessBlocker.PROGRAM_OUTPUT_ARCHIVE,
        ReadinessBlocker.PYTHON_CLOSURE_REPLAY,
        ReadinessBlocker.RUST_CLOSURE_REPLAY,
        ReadinessBlocker.LATEST_KEY_STATUS,
        ReadinessBlocker.FORMAL_MERKLE_ROOTS,
        ReadinessBlocker.CAPACITY_CLASSIFICATION,
        ReadinessBlocker.SEALED_CLOSURE_VERIFIER,
    )
    assert not contract.ready_for_outside_certificate
    assert ready_contract().readiness_blockers == (
        ReadinessBlocker.CANONICAL_AST_SCHEMA,
        ReadinessBlocker.PROGRAM_OUTPUT_ARCHIVE,
        ReadinessBlocker.PYTHON_CLOSURE_REPLAY,
        ReadinessBlocker.RUST_CLOSURE_REPLAY,
        ReadinessBlocker.LATEST_KEY_STATUS,
        ReadinessBlocker.FORMAL_MERKLE_ROOTS,
        ReadinessBlocker.CAPACITY_CLASSIFICATION,
        ReadinessBlocker.SEALED_CLOSURE_VERIFIER,
    )
    assert not ready_contract().ready_for_outside_certificate


def test_surface_freeze_bindings_cannot_be_replaced_by_caller_ids():
    with pytest.raises(ValueError, match="frozen Phase-3 surface"):
        replace(DEFAULT_PHASE3_PREREGISTRATION, rational_grid_id="named-grid-v1")


def test_banning_the_xor_or_parity_name_does_not_ban_xor_semantics():
    witness = XOR2_ABSOLUTE_DIFFERENCE_WITNESS
    assert "XOR" in FROZEN_FORBIDDEN_SYMBOLS
    assert "parity" in FROZEN_FORBIDDEN_SYMBOLS
    assert set(witness.operator_ids) == {
        "bit_to_scalar",
        "difference",
        "absolute",
    }
    assert not set(witness.operator_ids).intersection(FROZEN_FORBIDDEN_SYMBOLS)
    assert witness.truth_table == (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    assert witness.content_id.startswith("phase3_xor2_sanity_")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
)
def test_xor2_absolute_difference_is_exact(left: int, right: int, expected: int):
    assert xor2_via_absolute_difference(left, right) == expected


def test_xor2_sanity_rejects_nonbits_and_boolean_aliases():
    with pytest.raises(ValueError, match="belong"):
        xor2_via_absolute_difference(0, 2)
    with pytest.raises(TypeError, match="integer bits"):
        xor2_via_absolute_difference(True, 0)


def test_default_unready_contract_never_issues_outside_certificate():
    partially_bound = replace(
        DEFAULT_PHASE3_PREREGISTRATION,
        canonicalizer_implementation_id=content_id("canonicalizer", "5"),
        enumerator_implementation_id=content_id("enumerator", "6"),
    )
    receipt = closure_receipt(partially_bound)
    assessment = assess_closure(partially_bound, receipt)
    assert assessment.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert assessment.reason == "sealed_closure_verifier_not_implemented"
    assert assessment.outside_certificate_id is None
    assert assessment.shadow_only
    assert not assessment.active_promotion_authorized


def test_untrusted_closure_receipt_cannot_prove_outside_or_budget_state():
    contract = ready_contract()
    incomplete = closure_receipt(
        contract,
        closure_status=ClosureRunStatus.INCONCLUSIVE_BUDGET,
    )
    result = assess_closure(contract, incomplete)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert "sealed_closure_verifier_not_implemented" in result.reason
    assert result.outside_certificate_id is None

    exhausted = closure_receipt(
        contract,
        closure_status=ClosureRunStatus.DSL_TOO_LARGE,
        enumerated_count=50_000,
        first_out_of_budget_program_id=content_id("old_program", "7"),
    )
    result = assess_closure(contract, exhausted)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert "sealed_closure_verifier_not_implemented" in result.reason
    assert result.outside_certificate_id is None


def test_closure_receipt_rejects_false_completeness_and_budget_claims():
    contract = ready_contract()
    with pytest.raises(ValueError, match="full closure cardinality"):
        replace(
            closure_receipt(contract),
            closure_cardinality=122,
        )
    with pytest.raises(ValueError, match="50,000 accepted programs"):
        closure_receipt(
            contract,
            closure_status=ClosureRunStatus.DSL_TOO_LARGE,
            enumerated_count=49_999,
            first_out_of_budget_program_id=content_id("old_program", "7"),
        )
    with pytest.raises(ValueError, match="outside the frozen budget"):
        closure_receipt(contract, enumerated_count=50_001)
    budget_receipt = closure_receipt(
        contract,
        closure_status=ClosureRunStatus.INCONCLUSIVE_BUDGET,
    )
    with pytest.raises(ValueError, match="exactly 5,000,000"):
        replace(budget_receipt, raw_operator_application_count=4_999_999)
    semantic_receipt = closure_receipt(
        contract,
        closure_status=ClosureRunStatus.INCONCLUSIVE_SEMANTICS,
        semantics_total=False,
    )
    with pytest.raises(ValueError, match="cannot claim a closed frontier"):
        replace(semantic_receipt, frontier_exhausted=True)
    with pytest.raises(ValueError, match="cannot also claim a budget abort"):
        replace(semantic_receipt, wall_clock_abort_hit=True)
    with pytest.raises(ValueError, match="canonical order"):
        closure_receipt(
            contract,
            match_ids=(
                content_id("old_program", "f"),
                content_id("old_program", "e"),
            ),
        )


def test_self_reported_old_language_match_is_not_a_formal_verdict():
    contract = ready_contract()
    receipt = closure_receipt(
        contract,
        closure_status=ClosureRunStatus.INCONCLUSIVE_BUDGET,
        match_ids=(content_id("old_program", "e"),),
    )
    result = assess_closure(contract, receipt)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert "sealed_closure_verifier_not_implemented" in result.reason
    assert result.outside_certificate_id is None


def test_complete_self_reported_no_match_cannot_issue_outside_certificate():
    contract = ready_contract()
    receipt = closure_receipt(contract)
    result = assess_closure(contract, receipt)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert result.outside_certificate_id is None
    assert "sealed_closure_verifier_not_implemented" in result.reason
    assert not result.active_promotion_authorized


def test_outside_assessment_cannot_be_forged_by_direct_construction():
    with pytest.raises(RuntimeError, match="cannot be directly constructed"):
        ClosureAssessment(
            contract_id=content_id("phase3_contract", "1"),
            receipt_id=content_id("phase3_receipt", "2"),
            verdict=AdequacyVerdict.OUTSIDE_FROZEN_CLOSURE,
            reason="caller assertion",
            outside_certificate_id=content_id("outside_certificate", "3"),
        )


def test_dsl_too_large_receipt_binds_the_50001st_candidate_without_overcounting():
    contract = ready_contract()
    receipt = closure_receipt(
        contract,
        closure_status=ClosureRunStatus.DSL_TOO_LARGE,
        enumerated_count=50_000,
        first_out_of_budget_program_id=content_id("old_program", "7"),
    )
    assert receipt.closure_status is ClosureRunStatus.DSL_TOO_LARGE
    assert receipt.enumerated_canonical_program_count == 50_000
    assert receipt.first_out_of_budget_program_id is not None
    with pytest.raises(ValueError, match="50,001st program id"):
        closure_receipt(
            contract,
            closure_status=ClosureRunStatus.DSL_TOO_LARGE,
            enumerated_count=50_000,
        )


def test_receipt_binding_mismatch_fails_closed():
    contract = ready_contract()
    receipt = replace(
        closure_receipt(contract),
        bounded_universe_diagnostic_id=content_id("bounded_universe", "f"),
    )
    result = assess_closure(contract, receipt)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert result.outside_certificate_id is None

    dsl_mismatch = replace(
        closure_receipt(contract),
        dsl_spec_id=content_id("dsl_spec", "8"),
    )
    assert assess_closure(contract, dsl_mismatch).reason == (
        "dsl_spec_binding_missing_or_mismatched"
    )

    truth_mismatch = replace(
        closure_receipt(contract),
        target_table_diagnostic_id=content_id("target_truth_table", "9"),
    )
    assert assess_closure(contract, truth_mismatch).reason == (
        "target_truth_table_binding_missing_or_mismatched"
    )

    operator_mismatch = replace(
        closure_receipt(contract),
        operator_semantics_id=content_id("operator_semantics", "a"),
    )
    assert assess_closure(contract, operator_mismatch).reason == (
        "operator_semantics_binding_missing_or_mismatched"
    )


def test_hidden_sink_is_an_in_language_null_not_an_outside_target():
    contract = ready_contract()
    matched = closure_receipt(
        contract,
        target_role=TargetRole.IN_LANGUAGE_NULL,
        match_ids=(content_id("old_conservation_refinement", "e"),),
    )
    matched_result = assess_closure(contract, matched)
    assert matched_result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert "sealed_closure_verifier_not_implemented" in matched_result.reason

    broken_null = closure_receipt(
        contract,
        target_role=TargetRole.IN_LANGUAGE_NULL,
    )
    result = assess_closure(contract, broken_null)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert "sealed_closure_verifier_not_implemented" in result.reason
    assert result.outside_certificate_id is None
    assert (
        matched.bounded_universe_diagnostic_id
        == contract.hidden_sink_universe_diagnostic_id
    )
    assert (
        matched.target_table_diagnostic_id
        == contract.hidden_sink_target_table_diagnostic_id
    )
    assert (
        matched.bounded_universe_diagnostic_id
        != contract.bounded_universe_diagnostic_id
    )

    role_swapped = replace(matched, target_role=TargetRole.OUTSIDE_TARGET)
    swapped_result = assess_closure(contract, role_swapped)
    assert swapped_result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert swapped_result.reason == "bounded_universe_binding_missing_or_mismatched"


def test_partial_truth_table_fails_closed_even_for_a_complete_enumeration():
    contract = ready_contract()
    result = assess_closure(
        contract,
        closure_receipt(
            contract,
            closure_status=ClosureRunStatus.INCONCLUSIVE_SEMANTICS,
            semantics_total=False,
        ),
    )
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert result.outside_certificate_id is None


def test_fraction_mdl_gain_threshold_is_exact_at_the_boundary():
    contract = ready_contract()
    assert contract.mdl_code_table_id is not None
    receipt = mdl_receipt(code_table_id=contract.mdl_code_table_id)
    assert receipt.old_total_length_bits == Fraction(1010, 1)
    assert receipt.new_total_length_bits == Fraction(960, 1)
    assert receipt.compression_gain_bits == Fraction(50, 1)
    assert receipt.minimum_required_gain_bits == Fraction(50, 1)
    assert receipt.numeric_threshold_passed
    assert not mdl_gain_gate(contract, receipt)

    below = mdl_receipt(
        code_table_id=contract.mdl_code_table_id,
        new_data_bits=Fraction(1881, 2),
    )
    assert below.compression_gain_bits == Fraction(99, 2)
    assert not below.numeric_threshold_passed
    assert not mdl_gain_gate(contract, below)


def test_mdl_gate_requires_the_frozen_code_table_binding():
    contract = ready_contract()
    mismatched = mdl_receipt(code_table_id=content_id("mdl_code_table", "f"))
    assert mismatched.numeric_threshold_passed
    assert not mdl_gain_gate(contract, mismatched)

    unresolved = DEFAULT_PHASE3_PREREGISTRATION
    assert not mdl_gain_gate(unresolved, mismatched)


def test_mdl_lengths_reject_float_aliases_and_negative_values():
    contract = ready_contract()
    assert contract.mdl_code_table_id is not None
    receipt = mdl_receipt(code_table_id=contract.mdl_code_table_id)
    with pytest.raises(TypeError, match="exact Fraction"):
        replace(receipt, old_data_length_bits=1000.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(receipt, new_data_length_bits=Fraction(-1, 1))


def test_equivalent_fraction_encodings_have_the_same_mdl_content_id():
    contract = ready_contract()
    assert contract.mdl_code_table_id is not None
    first = mdl_receipt(code_table_id=contract.mdl_code_table_id)
    second = replace(first, new_program_length_bits=Fraction(40, 2))
    assert first.content_id == second.content_id


def test_four_adequacy_states_are_frozen():
    assert tuple(AdequacyVerdict) == (
        AdequacyVerdict.IN_LANGUAGE,
        AdequacyVerdict.OUTSIDE_FROZEN_CLOSURE,
        AdequacyVerdict.INCONCLUSIVE_BUDGET,
        AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
    )


def test_phase3_readiness_report_never_claims_an_outside_result():
    report = phase3_preregistration_report()
    assert report["artifact"] == "phase3_preregistration_readiness_v1"
    assert report["formal_phase3a_claim"] is False
    assert report["unbounded_outside_language_certificate_issued"] is False
    assert report["ready_for_outside_certificate"] is False
    assert report["sealed_closure_verifier_implemented"] is False
    assert report["closure_receipt_semantics_replayed"] is False
    assert report["sealed_mdl_scorer_implemented"] is False
    assert report["mdl_numeric_threshold_is_formal_gate"] is False
    assert (
        report["xor2_sanity"]["status"]
        == "TARGET_DESIGN_SANITY_ONLY"
    )
    assert report["xor2_sanity"]["formal_closure_verdict"] is None
    assert report["xor2_sanity"]["dsl_ast_executed"] is False
    assert (
        report["xor2_sanity"]["source_expression_typechecks_under_frozen_typing"]
        is False
    )
    assert report["hidden_sink_role"] == "in_language_null_control_only"
    assert report["surface_parameter_freeze_complete"] is True
    assert report["strict_acceptance_contract_complete"] is False
    assert report["normative_parameter_freeze_complete"] is False
    assert (
        report["closure_capacity_preflight"]["status"]
        == "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET"
    )
    assert report["closure_capacity_preflight"]["executed_closure_status"] == "NOT_RUN"
    assert report["target_freeze"]["universe_rows"] == 480
    assert report["target_freeze"]["formal_bounded_universe_root"] is None
    assert report["target_freeze"]["formal_target_truth_table_root"] is None
    assert report["hidden_sink_control"]["universe_rows"] == 85
    assert report["hidden_sink_control"]["formal_bounded_universe_root"] is None
    assert report["hidden_sink_control"]["formal_target_truth_table_root"] is None
    assert report["unbounded_outside_language_claim_prohibited"] is True
    assert report["shadow_only"] is True
    assert report["active_promotion_authorized"] is False


def test_checked_in_phase3_preregistration_artifact_matches_runtime():
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase3_preregistration_v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact == phase3_preregistration_report()
