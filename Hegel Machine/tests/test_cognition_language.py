"""Discovery checks use observations alone, without environment mechanism IDs."""

import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.synthesis import synthesize


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_expression_roundtrip_preserves_observation_and_history_semantics():
    program = Program("add", children=(
        Program("feature", name="signal"),
        Program("mul", children=(Program("const", value=2), Program("lag", name="__target__", lag=2))),
    ))
    row = Observation({"signal": 4.0}, 99.0, ({"__target__": 3.0}, {"__target__": 8.0}))
    restored = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored == program
    assert restored.predict(row) == 10.0
    assert restored.required_features() == {"signal"}
    assert restored.history_depth == 2
    with pytest.raises(ValueError, match="historical"):
        restored.predict(Observation({"signal": 4.0}, 99.0))


def test_refit_learns_multivariable_coefficients_and_transfers_to_unseen_values():
    rows = [Observation({"u": u, "v": v}, 1.7 * u - 0.6 * v + 2.1)
            for u in (-2.0, 0.0, 1.0, 3.0) for v in (-1.0, 0.5, 2.0)]
    candidates = synthesize(rows, operations=("refit",), budget=SearchBudget(1000))
    best = min(candidates, key=lambda candidate: mse(candidate.program, rows))
    unseen = Observation({"u": 4.5, "v": -2.0}, 10.95)
    assert best.program.predict(unseen) == pytest.approx(unseen.target, abs=1e-6)


def test_history_coordinate_is_learned_from_past_raw_measurements():
    rows = [Observation({"u": u}, 0.4 * u + 1.3 * past, ({"u": past, "__target__": 0.0},))
            for u in (-1.0, 0.0, 2.0) for past in (-2.0, 1.0, 3.0)]
    candidates = synthesize(rows, operations=("refit", "representation"), budget=SearchBudget(1800))
    best = min(candidates, key=lambda candidate: mse(candidate.program, rows))
    assert best.program.history_depth == 1
    assert mse(best.program, rows) < 1e-12
    row = Observation({"u": 3.0}, -4.0, ({"u": -4.0, "__target__": 0.0},))
    assert best.program.predict(row) == pytest.approx(row.target, abs=1e-6)


def test_boolean_composition_discovers_state_dependent_thresholds():
    # There is no state-dependent-threshold primitive in the expression language.
    rows = [Observation(
        {"signal": signal}, float(signal >= 1.5 or (prior and signal >= -0.5)),
        ({"signal": 0.0, "__target__": float(prior)},),
    ) for signal in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0) for prior in (False, True)]
    candidates = synthesize(rows, operations=("relation", "composition"), budget=SearchBudget(18000))
    exact = [candidate for candidate in candidates if mse(candidate.program, rows) == 0]
    assert exact
    assert any(candidate.program.history_depth == 1 for candidate in exact)


def test_conditional_quantiles_improve_tail_prediction_from_measured_residuals():
    rows = [Observation({"signal": float(group)}, 2.0 * group + noise * scale)
            for group, scale in ((-1, 0.2), (1, 3.0))
            for noise in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    base = Program("mul", children=(Program("const", value=2.0), Program("feature", name="signal")))
    candidates = synthesize(rows, operations=("robustification",), budget=SearchBudget(2000),
                            incumbent=base, risk_quantile=0.9)
    def pinball(program):
        residuals = [row.target - program.predict(row) for row in rows]
        return sum(max(0.9 * value, -0.1 * value) for value in residuals)
    best = min(candidates, key=lambda candidate: pinball(candidate.program))
    assert pinball(best.program) < 0.5 * pinball(base)
    high_offset = best.program.predict(Observation({"signal": 1.0}, 0)) - 2.0
    low_offset = best.program.predict(Observation({"signal": -1.0}, 0)) + 2.0
    assert high_offset > low_offset


def test_idealization_refits_after_removal_and_search_respects_row_budget():
    rows = [Observation({"u": u, "v": v}, 1.2 * u + 0.7)
            for u in (-1.0, 1.0, 2.0) for v in (-2.0, 0.0, 3.0)]
    incumbent = Program("add", children=(Program("feature", name="u"), Program("feature", name="v")))
    budget = SearchBudget(2 * len(rows))
    candidates = synthesize(rows, operations=("idealization",), budget=budget, incumbent=incumbent)
    assert budget.spent == 2 * len(rows)
    assert len(candidates) == 2
    best = min(candidates, key=lambda candidate: mse(candidate.program, rows))
    assert best.program.required_features() == {"u"}
    assert mse(best.program, rows) < 1e-12
    too_small = SearchBudget(len(rows) - 1)
    assert synthesize(rows, operations=("refit",), budget=too_small) == []
    assert too_small.spent == 0
