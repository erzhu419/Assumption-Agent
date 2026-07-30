import pytest

from hegel_machine.benchmark import controlled_cases, recognize_laws
from hegel_machine.laws import LawEvaluation, evaluate_law
from hegel_machine.schema import LawKind


@pytest.mark.parametrize("case", controlled_cases(), ids=lambda case: case.case_id)
def test_controlled_cases_are_classified_structurally(case):
    recognized = {result.kind for result in recognize_laws(case.episode)}
    assert (case.kind in recognized) is case.relation_present


@pytest.mark.parametrize("case", controlled_cases(), ids=lambda case: case.case_id)
def test_entity_renaming_cannot_change_verdict(case):
    before = {result.kind for result in recognize_laws(case.episode)}
    renamed = dict(case.episode)
    renamed["entity_names"] = ("renamed_1", "renamed_2")
    after = {result.kind for result in recognize_laws(renamed)}
    assert before == after


def test_unobserved_conservation_boundary_abstains():
    result = evaluate_law(
        LawKind.CONSERVATION,
        {
            "storage_delta": 0.0,
            "inflows": (1.0,),
            "outflows": (1.0,),
            "sources": (),
            "sinks": (),
            "boundary_observed": False,
        },
        tolerance=0.01,
    )
    assert result.abstained
    assert not result.passed
    assert result.residual is None


def test_negative_feedback_requires_temporal_order():
    result = evaluate_law(
        LawKind.NEGATIVE_FEEDBACK,
        {
            "disturbance_delta": 1.0,
            "response_delta": -1.0,
            "deviation_before_response": 1.0,
            "deviation_after_response": 0.0,
            "controlled_quantity_observed": True,
            "same_controlled_quantity": True,
            "disturbance_precedes_response": False,
            "local_stability_window_observed": True,
            "system_induced_response": True,
            "response_margin": 0.1,
            "mitigation_margin": 0.1,
        },
    )
    assert result.abstained
    assert "temporal" in result.reason


def test_zero_or_externally_imposed_response_is_not_negative_feedback():
    base = {
        "disturbance_delta": 1.0,
        "response_delta": 0.0,
        "deviation_before_response": 1.0,
        "deviation_after_response": 1.0,
        "controlled_quantity_observed": True,
        "same_controlled_quantity": True,
        "disturbance_precedes_response": True,
        "local_stability_window_observed": True,
        "system_induced_response": True,
        "response_margin": 0.1,
        "mitigation_margin": 0.1,
    }
    assert not evaluate_law(LawKind.NEGATIVE_FEEDBACK, base).passed
    externally_imposed = {
        **base,
        "response_delta": -1.0,
        "deviation_after_response": 0.0,
        "system_induced_response": False,
    }
    assert not evaluate_law(
        LawKind.NEGATIVE_FEEDBACK, externally_imposed
    ).passed


def test_monotonicity_sign_flip_is_rejected():
    increasing = {
        "x_low": 0.0,
        "x_high": 1.0,
        "y_low": 1.0,
        "y_high": 2.0,
        "direction": 1.0,
    }
    sign_flipped = {**increasing, "direction": -1.0}
    assert evaluate_law(LawKind.MONOTONICITY, increasing).passed
    assert not evaluate_law(LawKind.MONOTONICITY, sign_flipped).passed


def test_law_evaluation_cannot_self_report_a_contradictory_pass():
    with pytest.raises(ValueError, match="pass flag"):
        LawEvaluation(
            LawKind.SYMMETRY,
            residual=999.0,
            tolerance=0.01,
            passed=True,
            abstained=False,
            reason="within_tolerance",
            components=(("residual", 999.0),),
        )
