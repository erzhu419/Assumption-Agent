"""An affordable sparse fit must finish before extra columns consume its budget."""

from random import Random
from statistics import mean

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.sparse_search import sparse_candidates


def test_limited_search_finishes_a_useful_model_without_precomputing_every_history_column():
    rng = Random(211)
    rows = []
    for _ in range(100):
        x, z = rng.uniform(-2., 2.), rng.uniform(-2., 2.)
        history = tuple({"x": rng.uniform(-2., 2.), "z": rng.uniform(-2., 2.)} for _ in range(5))
        rows.append(Observation({"x": x, "z": z}, .3 + .7 * x - 1.2 * z, history))
    budget = SearchBudget(4 * len(rows))
    candidates = sparse_candidates(rows, budget=budget, max_history=5)
    assert candidates
    assert min(loss(candidate.program, rows, 0., .5) for candidate in candidates) < 1e-12
    assert 0 < budget.spent <= budget.limit


def test_small_fit_uses_all_paid_observations_and_charges_every_column_and_fit_row():
    rows = [Observation({"x": index / 10. - 5.},
                        .4 + .7 * (index / 10. - 5.) + (3. if index >= 80 else 0.))
            for index in range(100)]
    budget = SearchBudget(2 * len(rows))
    candidates = sparse_candidates(rows, budget=budget, max_history=0)
    assert candidates and budget.spent == budget.limit
    best = min(candidates, key=lambda candidate: loss(candidate.program, rows, 0., .5)).program
    xs, ys = [row.features["x"] for row in rows], [row.target for row in rows]
    x_center, y_center = mean(xs), mean(ys)
    slope = sum((x - x_center) * (y - y_center) for x, y in zip(xs, ys)) / sum((x - x_center) ** 2 for x in xs)
    intercept = y_center - slope * x_center
    for x in (-4., 0., 4.):
        assert best.predict(Observation({"x": x}, 0.)) == pytest.approx(intercept + slope * x, abs=1e-8)


@pytest.mark.parametrize("history_depth", [2, 5])
def test_each_history_depth_gets_a_complete_extension_before_shallow_extra_coordinates(history_depth):
    def measured(seed):
        rng, rows = Random(seed), []
        for _ in range(60):
            x, z = rng.uniform(-2., 2.), rng.uniform(-2., 2.)
            history = tuple({"x": rng.uniform(-2., 2.), "z": rng.uniform(-2., 2.)}
                            for _ in range(history_depth))
            target = .2 + .3 * x + .9 * history[-history_depth]["x"]
            rows.append(Observation({"x": x, "z": z}, target, history))
        return rows

    training, fresh = measured(313), measured(419)
    budget = SearchBudget((7 + 3 * history_depth) * len(training))
    candidates = sparse_candidates(training, budget=budget, max_history=history_depth)
    best = min(candidates, key=lambda candidate: loss(candidate.program, training, 0., .5)).program
    assert best.history_depth == history_depth
    assert loss(best, fresh, 0., .5) < 1e-12
    assert 0 < budget.spent <= budget.limit
