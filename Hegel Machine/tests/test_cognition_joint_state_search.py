"""A state coordinate must explain interventions on every participating input."""

import random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.state_search import state_candidates


def _rows(weights, *, seed, count=96):
    rng = random.Random(seed)
    rows = []
    for _ in range(count // 16):
        history, state = [], 0.0
        for _ in range(16):
            controls = {name: rng.uniform(-2.0, 2.0) for name in weights}
            drive = sum(controls[name] * weight for name, weight in weights.items())
            state = float(drive >= 0.55 or (state and drive >= -0.65))
            rows.append(Observation(controls, state, tuple(history)))
            history.append({**controls, "__target__": state})
    return rows


def _winner(candidates, rows):
    return min(candidates, key=lambda candidate: (loss(candidate.program, rows, 0.0, 0.5),
                                                 candidate.program.size)).program


@pytest.mark.parametrize("weights", (
    {"a": 1.0, "b": 0.47},
    {"heat": -0.73, "pressure": 1.19},
    {"fuel": 0.32, "air": -1.31, "load": 0.67},
))
def test_joint_state_coordinates_generalize_beyond_every_single_input(weights):
    training = _rows(weights, seed=43)
    fresh = _rows(weights, seed=77, count=1024)
    budget = SearchBudget(2200)
    candidates = state_candidates(training, budget=budget)
    learned = _winner(candidates, training)
    raw = _winner(state_candidates(training, budget=SearchBudget(2200), joint_coordinates=False), training)

    assert loss(learned, training, 0.0, 0.5) == 0.0
    assert loss(learned, fresh, 0.0, 0.5) < 0.45 * loss(raw, fresh, 0.0, 0.5)
    assert learned.required_features() == set(weights)
    assert learned.history_depth == 1 and learned.minimum_history_depth == 0
    # These current controls have no driver signal. The output must therefore
    # depend on the recorded state, even though the coordinate uses many inputs.
    zero = {name: 0.0 for name in weights}
    for previous in (0.0, 1.0):
        observation = Observation(zero, previous, ({**zero, "__target__": previous},))
        assert learned.predict(observation) == previous
    assert 900 < budget.spent <= 2200


def test_renaming_and_reversing_a_joint_law_are_learned_from_current_observations():
    original = _rows({"a": 1.0, "b": 0.47}, seed=43)
    mapping = {"a": "temperature", "b": "pressure"}
    renamed = [Observation({mapping[name]: value for name, value in row.features.items()}, row.target,
                           tuple({mapping.get(name, name): value for name, value in step.items()}
                                 for step in row.history)) for row in original]
    original_program = _winner(state_candidates(original, budget=SearchBudget(2200)), original)
    renamed_program = _winner(state_candidates(renamed, budget=SearchBudget(2200)), renamed)
    fresh = _rows({"a": 1.0, "b": 0.47}, seed=79, count=256)
    for row in fresh:
        renamed_row = Observation({mapping[name]: value for name, value in row.features.items()}, row.target,
                                  tuple({mapping.get(name, name): value for name, value in step.items()}
                                        for step in row.history))
        assert renamed_program.predict(renamed_row) == original_program.predict(row)

    changed = _rows({"a": -0.73, "b": 1.19}, seed=43)
    changed_fresh = _rows({"a": -0.73, "b": 1.19}, seed=77, count=1024)
    changed_program = _winner(state_candidates(changed, budget=SearchBudget(2200)), changed)
    assert loss(changed_program, changed_fresh, 0.0, 0.5) < 0.03
    assert loss(original_program, changed_fresh, 0.0, 0.5) > 0.25


def test_a_small_budget_keeps_executable_raw_alternatives_without_overspending():
    rows = _rows({"a": 1.0, "b": 0.47}, seed=43)
    for limit in (0, 20, 96, 400, 900):
        budget = SearchBudget(limit)
        candidates = state_candidates(rows, budget=budget)
        assert budget.spent <= limit
        for candidate in candidates:
            assert all(candidate.program.predict(row) in (0.0, 1.0) for row in rows)
    raw = state_candidates(rows, budget=SearchBudget(900), joint_coordinates=False)
    fallback = state_candidates(rows, budget=SearchBudget(900))
    assert raw and [candidate.program for candidate in raw] == [candidate.program for candidate in fallback]


@pytest.mark.parametrize("count, allowance", ((96, 1200), (80, 940)))
def test_failed_raw_rules_do_not_consume_the_budget_needed_to_invent_a_joint_coordinate(count, allowance):
    training = _rows({"a": 1.0, "b": 0.47}, seed=43, count=count)
    fresh = _rows({"a": 1.0, "b": 0.47}, seed=77, count=1024)
    # This is the controller's reachable reduced allowance after transfer and
    # screening work. Raw reset variants and extra lags used to exhaust it.
    budget = SearchBudget(allowance)
    program = _winner(state_candidates(training, budget=budget, max_history=4), training)
    assert program.required_features() == {"a", "b"}
    assert loss(program, training, 0.0, 0.5) == 0.0
    assert loss(program, fresh, 0.0, 0.5) < 0.03
    assert budget.spent <= allowance
