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
    ClosureEnumerationReceipt,
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
    return Phase3PrerequisiteContract(
        rational_grid_id=content_id("rational_grid", "1"),
        bounded_universe_id=content_id("bounded_universe", "2"),
        operator_semantics_id=content_id("operator_semantics", "3"),
        equivalence_contract_id=content_id("equivalence_contract", "4"),
        canonicalizer_implementation_id=content_id("canonicalizer", "5"),
        enumerator_implementation_id=content_id("enumerator", "6"),
        mdl_code_table_id=content_id("mdl_code_table", "7"),
        parity_target_id=content_id("parity_target", "8"),
        hidden_sink_control_id=content_id("hidden_sink_control", "9"),
        hidden_generator_spec_id=content_id("hidden_generator", "a"),
    )


def closure_receipt(
    contract: Phase3PrerequisiteContract,
    *,
    target_role: TargetRole = TargetRole.OUTSIDE_TARGET,
    complete: bool = True,
    budget_exhausted: bool = False,
    semantics_total: bool = True,
    enumerated_count: int = 123,
    match_ids: tuple[str, ...] = (),
) -> ClosureEnumerationReceipt:
    target_id = (
        contract.parity_target_id
        if target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_control_id
    )
    assert target_id is not None
    assert contract.bounded_universe_id is not None
    assert contract.equivalence_contract_id is not None
    assert contract.enumerator_implementation_id is not None
    return ClosureEnumerationReceipt(
        contract_id=contract.content_id,
        target_id=target_id,
        target_role=target_role,
        bounded_universe_id=contract.bounded_universe_id,
        equivalence_contract_id=contract.equivalence_contract_id,
        enumerator_implementation_id=contract.enumerator_implementation_id,
        search_budget=50_000,
        enumerated_canonical_program_count=enumerated_count,
        closure_cardinality=enumerated_count if complete else None,
        complete=complete,
        budget_exhausted=budget_exhausted,
        semantics_total=semantics_total,
        extensional_match_program_ids=match_ids,
        closure_root=content_id("closure_root", "b"),
        target_truth_table_root=content_id("truth_table_root", "c"),
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
    assert contract.sorts == FROZEN_SORTS
    assert contract.leaves == FROZEN_LEAVES
    assert contract.operators == FROZEN_OPERATORS
    assert contract.forbidden_symbols == FROZEN_FORBIDDEN_SYMBOLS
    assert contract.limits == DslLimits(
        maximum_relation_arity=3,
        maximum_entity_set_size=8,
        maximum_ast_depth=4,
        maximum_top_level_clauses=3,
        maximum_composition_depth=2,
        maximum_fitted_parameters=3,
        maximum_scope_clauses=2,
        maximum_canonical_programs=50_000,
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


def test_unfrozen_bindings_are_machine_readable_blockers():
    contract = DEFAULT_PHASE3_PREREGISTRATION
    assert contract.readiness_blockers == tuple(ReadinessBlocker)
    assert ReadinessBlocker.RATIONAL_GRID in contract.readiness_blockers
    assert ReadinessBlocker.BOUNDED_UNIVERSE in contract.readiness_blockers
    assert ReadinessBlocker.MDL_CODE_TABLE in contract.readiness_blockers
    assert ReadinessBlocker.PARITY_TARGET in contract.readiness_blockers
    assert not contract.ready_for_outside_certificate
    assert ready_contract().readiness_blockers == (
        ReadinessBlocker.SEALED_CLOSURE_VERIFIER,
    )
    assert not ready_contract().ready_for_outside_certificate


def test_future_freeze_bindings_must_be_content_addressed():
    with pytest.raises(ValueError, match="content-addressed"):
        replace(DEFAULT_PHASE3_PREREGISTRATION, rational_grid_id="named-grid-v1")


def test_banning_the_xor_or_parity_name_does_not_ban_xor_semantics():
    witness = XOR2_ABSOLUTE_DIFFERENCE_WITNESS
    assert "xor" in FROZEN_FORBIDDEN_SYMBOLS
    assert "parity" in FROZEN_FORBIDDEN_SYMBOLS
    assert set(witness.operator_ids) == {
        "difference",
        "absolute",
        "approx_equal",
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
        bounded_universe_id=content_id("bounded_universe", "2"),
        equivalence_contract_id=content_id("equivalence_contract", "4"),
        enumerator_implementation_id=content_id("enumerator", "6"),
        parity_target_id=content_id("parity_target", "8"),
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
    incomplete = closure_receipt(contract, complete=False)
    result = assess_closure(contract, incomplete)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert "sealed_closure_verifier_not_implemented" in result.reason
    assert result.outside_certificate_id is None

    exhausted = closure_receipt(
        contract,
        complete=False,
        budget_exhausted=True,
        enumerated_count=50_000,
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
    with pytest.raises(ValueError, match="full budget"):
        closure_receipt(
            contract,
            complete=False,
            budget_exhausted=True,
            enumerated_count=49_999,
        )
    with pytest.raises(ValueError, match="outside the frozen budget"):
        closure_receipt(contract, enumerated_count=50_001)
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
        complete=False,
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


def test_receipt_binding_mismatch_fails_closed():
    contract = ready_contract()
    receipt = replace(
        closure_receipt(contract),
        bounded_universe_id=content_id("bounded_universe", "f"),
    )
    result = assess_closure(contract, receipt)
    assert result.verdict is AdequacyVerdict.INCONCLUSIVE_SEMANTICS
    assert result.outside_certificate_id is None


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


def test_partial_truth_table_fails_closed_even_for_a_complete_enumeration():
    contract = ready_contract()
    result = assess_closure(
        contract,
        closure_receipt(contract, semantics_total=False),
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
    assert report["outside_language_certificate_issued"] is False
    assert report["ready_for_outside_certificate"] is False
    assert report["sealed_closure_verifier_implemented"] is False
    assert report["closure_receipt_semantics_replayed"] is False
    assert report["sealed_mdl_scorer_implemented"] is False
    assert report["mdl_numeric_threshold_is_formal_gate"] is False
    assert (
        report["xor2_sanity"]["status"]
        == "intended_numeric_semantics_sanity_only"
    )
    assert report["xor2_sanity"]["formal_closure_verdict"] is None
    assert report["xor2_sanity"]["dsl_ast_executed"] is False
    assert report["hidden_sink_role"] == "in_language_null_control_only"
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
