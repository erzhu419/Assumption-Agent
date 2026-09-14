"""Experiments should resolve consequential uncertainty within learning budget."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import (
    choose_probe, choose_valuable_probe, construct_probes, observed_classes,
)


@pytest.mark.parametrize("objective", ({}, {"risk_quantile": 0.8}, {"decision_costs": (3.0, 1.0)}))
def test_task_value_beats_arbitrarily_large_disagreement_on_a_small_region(objective):
    x, z = Program("feature", name="x"), Program("feature", name="z")
    double_x = Program("mul", children=(Program("const", value=2.0), x))
    # This third explanation agrees with x over the public reference design,
    # but can make raw variance arbitrarily large at an extreme query.
    spike = Program("if", children=(
        Program("ge", children=(z, Program("const", value=0.99))),
        Program("const", value=1000.0), x))
    candidates = [Candidate("x", x, "representation"),
                  Candidate("2x", double_x, "parameter"),
                  Candidate("spike", spike, "relation")]
    rows = [Observation({"x": 0.0, "z": z_value}, 0.0) for z_value in (0.0, 0.25, 0.5)]
    extreme = Probe("extreme", tuple({"x": 0.0, "z": 1.0} for _ in range(4)))
    relevant = Probe("relevant", tuple({"x": value, "z": 0.0} for value in (0.25, 0.5, 0.75, 1.0)))
    spec = TaskSpec("public", "Predict a scalar", ("x", "z"), (), (extreme, relevant),
                    feature_bounds={"x": (0.0, 1.0), "z": (0.0, 1.0)}, **objective)

    raw, _ = choose_probe([extreme, relevant], candidates, 0.0, SearchBudget(1000))
    assert raw == extreme
    budget = SearchBudget(6000)
    selected, details = choose_valuable_probe([extreme, relevant], candidates, 0.0, budget,
                                             observations=rows, spec=spec)
    assert selected == relevant
    by_id = {row["probe_id"]: row for row in details["evaluated_probes"]}
    assert by_id["relevant"]["value_per_observation"] > by_id["extreme"]["value_per_observation"]
    assert 0 < details["compute_cost"] == budget.spent <= 1200
    # Experiment selection leaves compute to fit and validate after observing.
    assert budget.spend(4800)


def test_measured_trajectory_aliases_are_excluded_and_control_decoupling_is_explored():
    x = Program("feature", name="x")
    candidate = Candidate("only", x, "representation")
    measured = Probe("initial", tuple({"x": 0.0, "z": 0.0} for _ in range(4)))
    rows, history = [], []
    for point in measured.sequence:
        rows.append(Observation(point, 0.0, tuple(history)))
        history.append({**point, "__target__": 0.0})
    alias = Probe("new_name_same_observations", measured.sequence)
    constant = Probe("constant", tuple({"x": 1.0, "z": 1.0} for _ in range(4)))
    decoupled = Probe("decoupled", tuple({"x": x_value, "z": -x_value}
                                        for x_value in (-1.0, 1.0, -1.0, 1.0)))
    spec = TaskSpec("public", "Predict a scalar", ("x", "z"), (measured,), (constant, decoupled),
                    feature_bounds={"x": (-1.0, 1.0), "z": (-1.0, 1.0)})
    selected, details = choose_valuable_probe([alias, constant, decoupled], [candidate], 0.0,
                                             SearchBudget(2000), observations=rows, spec=spec,
                                             queried_probes=[measured])
    assert selected == decoupled
    assert details["status"] == "coverage_only"
    assert details["duplicate_or_queried_trajectories"] == 1

    # An acquired sequence stays excluded when the next round gives it a new ID.
    repeated = Probe("renamed_decoupled", decoupled.sequence)
    selected, _ = choose_valuable_probe([repeated, constant], [candidate], 0.0,
                                        SearchBudget(2000), observations=rows, spec=spec,
                                        queried_probes=[measured, decoupled])
    assert selected == constant


def test_task_value_splits_explanations_equal_on_all_training_observations():
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    candidates = [Candidate("linear", x, "representation"), Candidate("square", square, "composition")]
    rows = [Observation({"x": value}, value ** 2) for value in (0.0, 1.0)]
    assert observed_classes(candidates, rows, 0.0, SearchBudget(100)) == [["linear", "square"]]
    spec = TaskSpec("public", "Predict a scalar", ("x",), (), (),
                    feature_bounds={"x": (0.0, 2.0)}, max_probe_length=4)
    probes = construct_probes(spec, candidates, round_number=0, max_history=1)
    selected, details = choose_valuable_probe(probes, candidates, 0.0, SearchBudget(6000),
                                             observations=rows, spec=spec)
    assert selected is not None and details["expected_risk_reduction"] > 0.0
    fresh = [Observation(point, point["x"] ** 2) for point in selected.sequence]
    assert observed_classes(candidates, rows + fresh, 0.0, SearchBudget(100)) == [["linear"], ["square"]]
