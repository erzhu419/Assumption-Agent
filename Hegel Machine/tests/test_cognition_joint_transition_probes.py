"""Joint interventions separate state laws hidden by correlated controls."""

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import transition_probe


def _coordinate(x_weight, z_weight):
    return Program("add", children=tuple(
        Program("mul", children=(Program("feature", name=name), Program("const", value=weight)))
        for name, weight in (("x", x_weight), ("z", z_weight))))


def _rule(coordinate, lower=-.4, upper=.4):
    def threshold(value):
        return Program("ge", children=(coordinate, Program("const", value=value)))

    return Program("if", children=(Program("available", lag=1),
        Program("if", children=(Program("ge", children=(Program("lag", name="__target__", lag=1),
                                                       Program("const", value=.5))),
                                threshold(lower), threshold(upper))), threshold(upper)))


def _observe(program, points):
    history, rows = [], []
    for point in points:
        target = program.predict(Observation(point, 0., tuple(history)))
        rows.append(Observation(point, target, tuple(history)))
        history.append({**point, "__target__": target})
    return rows


def _spec():
    return TaskSpec("opaque", "Predict the response", ("x", "z"), (), (),
                    feature_bounds={"x": (-1., 1.), "z": (-1., 1.)}, max_probe_length=24)


def test_joint_probe_exposes_missing_control_and_compares_oblique_candidates():
    truth = _rule(_coordinate(1., .5))
    axis = _rule(Program("feature", name="x"))
    rows = _observe(truth, [{"x": value, "z": 0.} for value in (-1., 0., 1., 0., -1., 1., -1.)])
    assert loss(axis, rows, 0., .5) == 0.
    budget = SearchBudget(500)
    probe, details = transition_probe(_spec(), [Candidate("axis", axis, "state"),
                                                Candidate("joint", truth, "state")], rows,
                                      budget=budget, remaining=24, round_number=0)
    fresh = _observe(truth, probe.sequence)
    assert details["policy"] == "joint_conditional_transition_interval"
    assert details["candidate_disagreement"] > 0.
    assert loss(axis, fresh, 0., .5) > 0.
    assert loss(truth, fresh, 0., .5) == 0.
    assert {point["z"] for point in probe.sequence} >= {-1., 1.}
    assert 0 < budget.spent <= 500
    assert details["mature_relative_width"] > 0.
    assert any(a.features == b.features and a.target != b.target for a in fresh for b in fresh)


@pytest.mark.parametrize("weights", [(1., .5), (.35, -1.)])
def test_affine_slices_stay_feasible_and_react_to_acquired_boundary_rows(weights):
    truth = _rule(_coordinate(*weights), lower=-.25, upper=.45)
    points = [{"x": x, "z": z} for x, z in
              ((-1., 1.), (1., -1.), (-.5, .5), (.5, -.5), (0., 0.), (1., 1.), (-1., -1.))]
    rows = _observe(truth, points)
    candidates = [Candidate("joint", truth, "state")]
    probe, details = transition_probe(_spec(), candidates, rows, budget=SearchBudget(500),
                                      remaining=17, round_number=0)
    assert probe is not None and 0 < probe.cost <= 17
    assert all(-1. <= value <= 1. for point in probe.sequence for value in point.values())
    assert any(len(coordinate) == 2 for coordinate in details["coordinates"])
    fresh = _observe(truth, probe.sequence)
    following, _ = transition_probe(_spec(), candidates, [*rows, *fresh], budget=SearchBudget(500),
                                     remaining=17, round_number=1, used=(probe,))
    assert following is None or following.sequence != probe.sequence


def test_axis_candidate_still_excites_unrepresented_controls():
    truth = _rule(_coordinate(1., .5))
    axis = _rule(Program("feature", name="x"))
    rows = _observe(truth, [{"x": value, "z": 0.} for value in (-1., 0., 1., 0., -1., 1., -1.)])
    probe, details = transition_probe(_spec(), [Candidate("axis", axis, "state")], rows,
                                      budget=SearchBudget(100), remaining=12, round_number=0)
    assert probe is not None and probe.cost <= 12
    assert details["candidate_disagreement"] == 0.
    assert loss(axis, _observe(truth, probe.sequence), 0., .5) > 0.
    old_probe, old_details = transition_probe(_spec(), [Candidate("axis", axis, "state")], rows,
        budget=SearchBudget(100), remaining=12, round_number=0, joint_controls=False)
    assert old_details["policy"] == "conditional_transition_interval"
    assert all(point["z"] == 0. for point in old_probe.sequence)


def test_startup_uncertainty_is_measured_before_any_state_preparation():
    axis = Program("feature", name="x")
    truth = _rule(axis, lower=-.4, upper=.4)
    startup = Program("ge", children=(axis, Program("const", value=.8)))
    alternative = Program("if", children=(*truth.children[:2], startup))
    # Both candidates explain every acquired row. Their only disagreement is
    # at startup; all mature-state thresholds are identical.
    rows = [*_observe(truth, [{"x": 0., "z": 0.}, {"x": 1., "z": 0.},
                             {"x": 0., "z": 0.}, {"x": -1., "z": 0.}]),
            *_observe(truth, [{"x": 1., "z": .5}, {"x": -.35, "z": .5},
                              {"x": -.45, "z": .5}, {"x": .35, "z": .5}, {"x": .45, "z": .5}])]
    assert loss(alternative, rows, 0., .5) == 0.
    probe, details = transition_probe(_spec(), [Candidate("late_start", alternative, "state"),
                                                Candidate("shared_start", truth, "state")], rows,
                                      budget=SearchBudget(500), remaining=12, round_number=0)
    assert probe.sequence[0] == details["startup_control"]
    assert .4 <= probe.sequence[0]["x"] < .8
    assert probe.cost == 1
    fresh = _observe(truth, probe.sequence)
    assert alternative.predict(fresh[0]) != fresh[0].target
    assert loss(alternative, fresh, 0., .5) > 0.
