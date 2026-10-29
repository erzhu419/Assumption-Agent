"""A discovered historical coordinate participates in joint response research."""

import json
from random import Random

import pytest

from hegel_machine.cognition.conditional_coordinates import (
    _coordinates, _masked, propose_conditional_coordinates,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import (
    quote_research_step_revisions, replay_research_program,
)


def guarded_lag(name="u"):
    return Program("if", children=(Program("available", lag=1),
        Program("lag", name=name, lag=1), Program("const", value=0.)))


def measured_rows(seed, placement, *, name="u", width=3, threshold=.15):
    rng = Random(seed)
    rows = []
    for _ in range(2):
        points, responses = [], []
        for _ in range(48):
            value = rng.choice((-.9, -.6, -.3, 0., .3, .6, .9))
            previous = points[-1][name] if points else 0.
            points.append({name: value})
            active = previous >= threshold
            responses.append((2.3 if active else -1.4) * value
                             + (.5 if active else .7) * previous)
            if placement == "inside_window":
                target = sum(responses[-width:])
            else:
                start = max(0, len(points) - width)
                target = sum((2.3 if active else -1.4) * points[index][name]
                             + (.5 if active else .7) *
                               (points[index - 1][name] if index else 0.)
                             for index in range(start, len(points)))
            rows.append(Observation(points[-1], .17 + target, tuple(points[:-1])))
    return rows


def spec(name="u"):
    return TaskSpec("history-coordinate", "Predict the purchased response", (name,), (), ())


def test_window_condition_uses_each_samples_prefix_without_double_shift():
    history = ({"u": 2.}, {"u": -1.}, {"u": 3.})
    row = Observation({"u": .5}, 0., history)
    lag = guarded_lag()
    condition = Program("ge", children=(lag, Program("const", value=.5)))
    current = Program("feature", name="u")
    assert _masked(current, condition, True, 3, "inside_window").predict(row) == -.5
    assert _masked(current, condition, True, 3, "outside_window").predict(row) == 2.5
    assert _masked(lag, condition, True, 3, "inside_window").predict(row) == 5.
    assert _masked(lag, condition, True, 3, "outside_window").predict(row) == 4.
    startup = Observation({"u": 9.}, 0.)
    assert _masked(lag, condition, True, 3, "inside_window").predict(startup) == 0.


def test_history_basis_survives_public_coordinate_binding():
    rows = measured_rows(11, "inside_window")
    lag = guarded_lag()
    target_lag = guarded_lag("__target__")
    missing_lag = guarded_lag("unmeasured")
    available, basis = _coordinates(spec(), rows, [lag, target_lag, missing_lag])
    assert available == [Program("feature", name="u")]
    assert lag.required_features() == set()
    assert lag in basis
    assert target_lag not in basis and missing_lag not in basis


@pytest.mark.parametrize("placement", ["inside_window", "outside_window"])
def test_history_gate_and_response_are_jointly_fitted_and_replayable(placement):
    training = measured_rows(71, placement)
    coordinate = guarded_lag()
    work = SearchBudget(180000)
    claims = propose_conditional_coordinates(spec(), training, work,
        basis=[Program("feature", name="u"), coordinate],
        gate_coordinates=[coordinate], width_candidates=[1, 3],
        placement_candidates=["inside_window", "outside_window"], max_candidates=4)
    assert claims
    best = min(claims, key=lambda item: item["training_loss"])
    assert best["training_loss"] < 1e-10
    step = next(step for step in best["research_program"]["selection_steps"]
                if step["operation"] == "discover_conditional_coordinates")
    assert step["gate_coordinate"] == coordinate.to_dict()
    assert coordinate.to_dict() in step["branch_basis"]
    assert step["placement"] == placement and step["width"] == 3
    assert loss(best["candidate"].program, measured_rows(1901, placement), 0., .5) < 1e-10
    assert 0 < work.spent <= work.limit

    # Saved history gates must survive normal procedure recovery and feature
    # rebinding; new coefficients are fitted on the new measured trajectories.
    saved = json.loads(json.dumps(best["research_program"]))
    changed = measured_rows(341, placement, name="v", threshold=-.45)
    replay = replay_research_program(saved, changed, SearchBudget(180000),
                                    feature_mapping={"u": "v"})
    assert replay is not None
    fresh = measured_rows(1902, placement, name="v", threshold=-.45)
    assert loss(replay["candidate"].program, fresh, 0., .5) < 1e-10
    assert all(name == "v" for name, _ in replay["candidate"].program.observed_variables())


def test_unavailable_history_is_not_silently_filled_for_a_gate():
    rows = measured_rows(18, "inside_window")
    work = SearchBudget(10000)
    claims = propose_conditional_coordinates(spec(), rows, work,
        basis=[Program("feature", name="u")],
        gate_coordinates=[Program("lag", name="u", lag=1)])
    assert claims == []
    assert 0 < work.spent <= work.limit


def test_saved_history_gate_quote_counts_its_brackets_and_covers_replay():
    training = measured_rows(71, "inside_window", width=1)
    validation = measured_rows(1901, "inside_window", width=1)
    coordinate = guarded_lag()
    claims = propose_conditional_coordinates(spec(), training, SearchBudget(100000),
        basis=[Program("feature", name="u"), coordinate], gate_coordinates=[coordinate],
        width_candidates=[1], max_candidates=1)
    best = min(claims, key=lambda item: item["training_loss"])
    assert best["training_loss"] < 1e-10
    saved = json.loads(json.dumps(best["research_program"]))
    quote = quote_research_step_revisions(saved, training, validation, max_candidates=1)
    assert quote["status"] == "quoted"
    baseline = quote["methods"][0]
    gate = next(item for item in baseline["gate_brackets"]
                if item["gate_coordinate"] == coordinate.to_dict())
    # Seven actually sampled lag levels, each side with at least three rows,
    # yield six possible brackets. This coordinate cannot be silently omitted.
    assert gate["brackets"] == 6
    assert baseline["gate_preparation_upper"] == 4 * len(training)
    assert quote["coordinate_readout_compute"] >= 2 * len(training)
    assert quote["new_fits"] == quote["new_world_predictions"] == 0

    work = SearchBudget(baseline["compute_upper"])
    replay = replay_research_program(saved, training, work)
    assert replay is not None
    assert work.spend(len(validation))
    assert loss(replay["candidate"].program, validation, 0., .5) < 1e-10
    assert work.last_denied_request is None
    assert work.spent <= baseline["compute_upper"]
