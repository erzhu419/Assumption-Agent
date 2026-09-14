"""Retained evidence must fund complete state and initial-condition fits."""

from random import Random

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.state_search import state_candidates, state_search_work


def measured_trajectories(count=20):
    rng = Random(721)
    rows = []
    for _ in range(count):
        history, active = [], False
        for index in range(8):
            features = {"a_context": rng.uniform(-2., 2.), "z_signal": rng.uniform(-2., 2.)}
            active = (features["a_context"] >= .2 if index == 0 else
                      features["z_signal"] >= .7 or (active and features["z_signal"] >= -.6))
            target = float(active)
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
    return rows


def test_quote_funds_every_state_branch_and_the_distinct_startup_condition_on_retained_trajectories():
    rows = measured_trajectories()
    small = SearchBudget(650)
    incomplete = state_candidates(rows, budget=small, max_history=4)
    assert incomplete and all(candidate.program.history_depth == 0 for candidate in incomplete)
    assert min(loss(candidate.program, rows, 0., .5) for candidate in incomplete) > .1
    quote = state_search_work(rows)
    assert quote == 6*len(rows)+42
    budget = SearchBudget(quote)
    candidates = state_candidates(rows, budget=budget, max_history=4)
    best = min(candidates, key=lambda candidate: (loss(candidate.program, rows, 0., .5), candidate.program.size))
    assert loss(best.program, rows, 0., .5) == 0.
    assert best.program.minimum_history_depth == 0 and best.program.history_depth == 1
    assert best.program.children[2].required_features() == {"a_context"}
    assert ("z_signal", 0) in best.program.observed_variables()
    assert small.spent <= small.limit and budget.spent <= quote


def test_quote_uses_observed_coordinates_and_does_not_request_state_work_for_continuous_responses():
    rows = measured_trajectories(count=8)
    original = state_search_work(rows)
    richer = [Observation({**row.features, "measured_context": 3.}, row.target,
              tuple({**point, "measured_context": 3.} for point in row.history)) for row in rows]
    assert state_search_work(richer) == original + 2*len(rows)+2
    assert state_search_work(rows, max_history=0) == 3*len(rows)+6
    assert state_search_work(rows, max_history=4) == original
    assert state_search_work([]) == 0
    assert state_search_work([Observation({"x": float(index)}, index / 3.) for index in range(5)]) == 0
    budget = SearchBudget(state_search_work(richer))
    candidates = state_candidates(richer, budget=budget)
    assert min(loss(candidate.program, richer, 0., .5) for candidate in candidates) == 0.
    assert budget.spent <= budget.limit
