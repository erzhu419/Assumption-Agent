"""Approximate reference laws compete before independent physical testing.

These fixtures use the existing public laboratory, controller and numerical
researcher. They test execution decisions, not autonomous scientific discovery.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from test_world_reference_hypothesis import PATH, prepared, square, submit


def _state(agent):
    return json.loads((agent.workspace / recipe.STATE).read_text())


def _questions(agent):
    return {question["id"]: question for question in agent.questions}


def test_successful_integer_setting_does_not_discard_approximate_next_prediction(tmp_path):
    agent, lab = prepared(tmp_path, intercept=100.3)
    result = submit(agent, lab, square("density", "density_square"))

    assert result["status"] == "completed" and lab.success
    assert [lab.frequencies[n] for n in (4, 5)] == [260, 350]
    fit = agent.relation_research["reference_fit_density_square_5"]
    observed = [row for row in fit["fit_rows"]
                if row.get("target_kind") == "observed_successful_setting"]
    assert len(observed) == 1 and observed[0]["target"] == 260
    assert observed[0]["target"] != lab.required[4]
    # The precise-law claim still fails its original tolerance. The empirical
    # approximation earns support only after testing the remaining crystal.
    assert fit["status"] == "refuted"
    assert fit["empirical_prediction"]["fit_mse"] > 1e-4
    question = _questions(agent)["reference_test_5_density_square"]
    assert question["status"] == "supported"
    assert question["answer"]["activated"] is True
    assert question["answer"]["actual_frequency"] == 350
    assert set(question["evidence"]).isdisjoint(fit["fit_evidence"])
    state = _state(agent)
    # Completion can end the agent immediately after answer(), before the
    # controller archives the final pending test on its next invocation.
    test = state["test"]
    assert test["number"] == 5
    assert test["fit_status"] == "approximate" and test["fit_mse"] > 1e-4
    assert state["reference_numbers"] == [1, 2, 3]
    assert agent.costs["model_calls"] == 0


def test_lower_leave_one_out_error_wins_over_first_declared_exact_interpolant(tmp_path):
    term = {"op": "feature", "name": "density"}
    quadratic = square("density", "density_square")
    squared = quadratic["basis"][0]
    overfit = {"id": "cubic_interpolant", "basis": [
        term, deepcopy(squared),
        {"op": "mul", "children": [deepcopy(squared), deepcopy(term)]}],
        "rationale": "Candidate interpolant with an extra degree of freedom."}
    source = _retain_declaration(Path(recipe.__file__).read_text(), overfit)
    source = _retain_declaration(source, quadratic)
    agent, lab = prepared(tmp_path, source=source, intercept=100.3)
    result = agent._run_procedure(lab, PATH, max_steps=256,
                                  query_arguments={"estimator_trial": True})

    assert result["status"] == "completed" and lab.success
    state = _state(agent)
    selection = state["prediction_selection"]
    assert selection["number"] == 5
    assert selection["criterion"] == "leave_one_crystal_out_mse"
    assert selection["selected_family"] == "density_square"
    candidates = {item["family"]: item for item in selection["candidates"]}
    approximate, interpolant = candidates["density_square"], candidates["cubic_interpolant"]
    assert interpolant["fit_mse"] < 1e-4 < approximate["fit_mse"]
    assert approximate["leave_one_out_mse"] < interpolant["leave_one_out_mse"]
    assert [(item["number"], item["family"]) for item in [*state["tests"], state["test"]]] == [
        (4, "density_square"), (5, "density_square")]
    # Both functional forms reached numerical research before the actuator
    # test; declaration order did not silently choose the first interpolant.
    assert {"reference_fit_cubic_interpolant_4", "reference_fit_density_square_4",
            "reference_fit_cubic_interpolant_5", "reference_fit_density_square_5"} <= set(agent.relation_research)
    assert agent.costs["model_calls"] == 0


def test_large_residual_candidate_remains_unconfirmed_until_physical_failure(tmp_path):
    agent, lab = prepared(tmp_path, intercept=100.3)
    lab.gap.update({1: 1., 2: 1., 3: 1.})
    result = submit(agent, lab, square("gap", "constant_signal"))

    assert not lab.success
    fit = agent.relation_research["reference_fit_constant_signal_4"]
    assert fit["status"] == "refuted"
    assert fit["empirical_prediction"]["fit_mse"] > 100
    tests = [item for item in result["tests"] if item["family"] == "constant_signal"]
    assert len(tests) == 1 and tests[0]["status"] == "refuted"
    answer = _questions(agent)["reference_test_4_constant_signal"]
    assert answer["status"] == "refuted" and answer["evidence"]
    assert answer["answer"]["activated"] is False
    assert set(answer["evidence"]).isdisjoint(fit["fit_evidence"])


def test_old_fit_only_rejection_reopens_without_erasing_its_record(tmp_path):
    agent, lab = prepared(tmp_path, intercept=100.3)
    old_refutation = {"number": 4, "family": "density_square",
        "fit_question": "reference_fit_density_square_4",
        "diagnosis": "representation_inadequate", "evidence": []}
    # A saved older controller classified this declaration only by numerical
    # tolerance. No actuator experiment has rejected it on this target.
    (agent.workspace / recipe.STATE).write_text(json.dumps({
        "completed": [], "rejected": {"4": ["density_square"]},
        "fit_refutations": [old_refutation]}))
    result = submit(agent, lab, square("density", "density_square"))

    assert result["status"] == "completed" and lab.success
    state = _state(agent)
    assert old_refutation in state["fit_refutations"]
    assert _questions(agent)["reference_test_4_density_square"]["status"] == "supported"
    assert [lab.frequencies[n] for n in (4, 5)] == [260, 350]


def test_physical_refutation_is_not_reopened_as_approximate_fit(tmp_path):
    agent, lab = prepared(tmp_path)
    hypothesis = square("gap", "gap_square")
    first = submit(agent, lab, hypothesis)
    test = next(item for item in first["tests"] if item["family"] == "gap_square")
    assert test["status"] == "refuted" and test["prediction"] == pytest.approx(350.)
    action_count = len(lab.actions)
    evidence = deepcopy(_questions(agent)["reference_test_4_gap_square"]["evidence"])
    second = submit(agent, lab, hypothesis)

    assert len(lab.actions) == action_count and second["tests"] == []
    assert _questions(agent)["reference_test_4_gap_square"]["evidence"] == evidence
    assert "gap_square" in _state(agent)["rejected"]["4"]
    assert not lab.success and agent.costs["model_calls"] == 0
