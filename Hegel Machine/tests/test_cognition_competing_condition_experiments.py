"""Competing explanations select an executable, paid condition experiment."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe


def feature(name):
    return Program("feature", name=name)


def conditional(response, threshold=0.):
    return Program("if", children=(Program("ge", children=(feature("meter"),
        Program("const", value=threshold))), response, Program("const", value=0.)))


def entry(program, name, intervals=()):
    return {"candidate": Candidate(name, program, "research"), "research_program": {
        "selection_steps": [{"operation": "discover_conditional_coordinates",
            "gate_coordinate": feature("meter").to_dict(), "threshold_interval": list(interval)}
            for interval in intervals]}}


def apparatus_rows():
    spec = TaskSpec("unlabelled", "Buy readings while applying legal commands.",
        ("drive", "bias", "meter"), (), (),
        feature_bounds={"drive": (-1., 1.), "bias": (-2., 2.)}, max_probe_length=32)
    # Four readings at each commanded bias identify its changing distribution.
    # The auxiliary reading itself remains outside the command interface.
    rows = [Observation({"drive": 0., "bias": bias, "meter": bias + noise}, 0.)
            for bias in (-1., 0., 1.) for noise in (-.8, -.3, .3, .8)]
    return spec, rows


def test_other_explanation_changes_the_chosen_legal_background():
    spec, rows = apparatus_rows()
    selected = entry(Program("const", value=0.), "current")
    alternative = entry(conditional(feature("drive")), "conditional", ((-.1, .1),))
    ordinary = plan_observed_condition_probe(spec, rows, selected, SearchBudget(10000))
    budget = SearchBudget(10000)
    joint = plan_observed_condition_probe(spec, rows, selected, budget,
        finalists=(alternative,), length=12)
    assert ordinary is None
    assert abs(joint["fixed_control_background"]["drive"]) == 1.
    assert joint["expected_disagreement"] > 0.
    assert {item["candidate_id"] for item in joint["competing_hypotheses"]} == {
        "current", "conditional"}
    assert joint["experiment_kind"] == "fixed_background"
    assert joint["probe"].cost == 12
    assert 0 < joint["compute_cost"] == budget.spent <= budget.limit
    assert all(set(point) == set(spec.feature_bounds) for point in joint["probe"].sequence)
    assert all(point == joint["probe"].sequence[0] for point in joint["probe"].sequence)
    assert all(lower <= point[name] <= upper
        for point in joint["probe"].sequence for name, (lower, upper) in spec.feature_bounds.items())


def test_full_history_separates_laws_that_agree_at_probe_startup():
    spec, rows = apparatus_rows()
    selected = entry(feature("drive"), "instantaneous")
    alternative = entry(Program("window_sum", lag=3, children=(feature("drive"),)), "window3")
    plan = plan_observed_condition_probe(spec, rows, selected, SearchBudget(10000),
        finalists=(alternative,), length=12)
    assert plan["startup_disagreement"] == 0.
    assert plan["duration_disagreement"] > 0.
    assert abs(plan["fixed_control_background"]["drive"]) == 1.
    assert plan["threshold_intervals"] == []
    assert plan["sampling_basis"] == "paid_reading_quantiles"


def test_current_and_historical_reading_gates_generate_different_forecasts():
    spec, rows = apparatus_rows()
    inside = entry(Program("window_sum", lag=3, children=(conditional(feature("drive")),)),
                   "historical_gates", ((-.1, .1),))
    outside = entry(conditional(Program("window_sum", lag=3, children=(feature("drive"),))),
                    "current_gate", ((-.1, .1),))
    plan = plan_observed_condition_probe(spec, rows, inside, SearchBudget(10000),
        finalists=(outside,), length=12)
    assert plan["startup_disagreement"] == 0.
    assert plan["duration_disagreement"] > 0.
    assert {item["candidate_id"] for item in plan["competing_hypotheses"]} == {
        "historical_gates", "current_gate"}
    assert len(plan["probe"].sequence) == 12


def test_narrow_bracket_without_an_old_interior_reading_remains_unresolved():
    spec, rows = apparatus_rows()
    interval = (.001, .002)
    selected = entry(conditional(feature("drive"), threshold=.0015), "current", (interval,))
    plan = plan_observed_condition_probe(spec, rows, selected, SearchBudget(10000), length=12)
    assert not any(interval[0] < row.features["meter"] < interval[1] for row in rows)
    assert plan["threshold_interval"] == list(interval)
    assert plan["interval_exposure"][0]["estimated_probability"] > 0.
    assert plan["interval_exposure"][0]["response_disagreement"] > 0.
    assert plan["boundary_disagreement"] > 0.
    assert {item["threshold_endpoint"] for item in plan["competing_hypotheses"]
            if item["kind"] == "threshold_endpoint"} == set(interval)


def test_new_evidence_replaces_the_older_bracket_for_the_same_gate():
    spec, rows = apparatus_rows()
    selected = entry(conditional(feature("drive"), threshold=.15), "current",
                     ((-.9, .9), (.1, .2)))
    plan = plan_observed_condition_probe(spec, rows, selected, SearchBudget(10000))
    assert [item["threshold_interval"] for item in plan["threshold_intervals"]] == [[.1, .2]]
    assert plan["interval_exposure"][0]["threshold_interval"] == [.1, .2]


def test_actual_reading_can_differ_from_the_predicted_sampling_mean():
    spec, rows = apparatus_rows()
    selected = entry(conditional(feature("drive")), "current", ((-.1, .1),))
    plan = plan_observed_condition_probe(spec, rows, selected, SearchBudget(10000), length=8)
    mean = Program.from_dict(plan["reading_mean_program"])
    purchased = [Observation({**point, "meter": point["bias"] + noise}, 0.)
                 for point, noise in zip(plan["probe"].sequence, (-.8, .8) * 4)]
    assert any(abs(row.features["meter"] - mean.predict(row)) > .5 for row in purchased)
    assert all("meter" not in point for point in plan["probe"].sequence)
    assert len(purchased) == plan["probe"].cost


def test_insufficient_search_money_does_not_launch_an_unassessed_plan():
    spec, rows = apparatus_rows()
    selected = entry(conditional(feature("drive")), "current", ((-.1, .1),))
    budget = SearchBudget(20)
    assert plan_observed_condition_probe(spec, rows, selected, budget) is None
    assert budget.spent <= budget.limit
