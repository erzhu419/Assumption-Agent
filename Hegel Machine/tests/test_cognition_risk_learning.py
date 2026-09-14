"""Quantile learning and fixed-cost decisions on observed outcome losses."""

from random import Random
from statistics import mean

import pytest

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.risk_search import risk_candidates
from hegel_machine.cognition.selection import representative_probe
from hegel_machine.cognition.synthesis import _fit_linear, _quantile
from hegel_machine.cognition.uncertainty import action_and_interval, decision_loss, fit_decision_model


def mean_with_residual_quantile(rows, names, q):
    fitted = _fit_linear([Program("feature", name=name) for name in names], rows)
    offset = _quantile([row.target - fitted.predict(row) for row in rows], q)
    return Program("add", children=(fitted, Program("const", value=offset)))


def actual_decision_loss(program, rows, model, costs):
    return mean(decision_loss(action_and_interval(program, row, 0.0, model, costs)[0], row.target, costs)
                for row in rows)


def test_direct_pinball_fit_learns_conditional_scale_and_generalizes_to_new_draws():
    rng = Random(120)
    rows = []
    for _ in range(700):
        control = rng.uniform(-3, 3)
        target = .2 + 1.4 * control + (.1 + abs(control)) * rng.gauss(0, 1)
        rows.append(Observation({"control": control}, target))
    training, held_out = rows[:160], rows[160:]
    baseline = mean_with_residual_quantile(training, ("control",), .9)
    budget = SearchBudget(8000)
    candidates = risk_candidates(training, budget=budget, q=.9)
    chosen = min(candidates, key=lambda candidate: loss(candidate.program, training, 0.0, .9))
    assert loss(chosen.program, held_out, 0.0, .9) < .9 * loss(baseline, held_out, 0.0, .9)
    assert "abs(control)" in chosen.program.describe()
    assert Program.from_dict(chosen.program.to_dict()) == chosen.program
    assert 0 < budget.spent <= budget.limit


def test_quantile_coordinate_fit_handles_negative_controls_and_renamed_roles():
    training = [Observation({"a": x}, 1.2 - 2.5 * x + noise)
                for x in (-2., -1., 0., 1., 2.) for noise in (-.4, -.1, 0., .3, .8)]
    changed = [Observation({"u": -row.features["a"]}, row.target) for row in training]
    original = min(risk_candidates(training, budget=SearchBudget(2500), q=.8),
                   key=lambda candidate: loss(candidate.program, training, 0.0, .8)).program
    reversed_role = min(risk_candidates(changed, budget=SearchBudget(2500), q=.8),
                        key=lambda candidate: loss(candidate.program, changed, 0.0, .8)).program
    for x in (-2.7, -.7, .6, 2.3):
        assert original.predict(Observation({"a": x}, 0.0)) == pytest.approx(
            reversed_role.predict(Observation({"u": -x}, 0.0)), abs=1e-6)


def test_tiny_fit_budget_still_returns_the_actual_empirical_tail_action():
    rows = [Observation({"x": float(index)}, target)
            for index, target in enumerate((0., 0., 0., 0., 1., 9.))]
    budget = SearchBudget(len(rows))
    candidates = risk_candidates(rows, budget=budget, q=.9)
    assert len(candidates) == 1
    assert candidates[0].program.predict(rows[0]) == 9.0
    assert budget.spent == len(rows)
    assert risk_candidates(rows, budget=SearchBudget(len(rows) - 1), q=.9) == []


def cost_blocks(count=6):
    blocks = []
    for _ in range(count):
        history, block = [], []
        for target in (-10., 0., 0., 0., 0., 0., 0., 0., 0., 1.):
            block.append(Observation({"context": 0.0}, target, tuple(history)))
            history.append({"context": 0.0, "__target__": target})
        blocks.append(block)
    return blocks


def test_fixed_cost_quantile_action_improves_actual_loss_without_endpoint_surrogate():
    blocks = cost_blocks()
    training, calibration, held_out = sum(blocks[:4], []), blocks[4], blocks[5]
    predictor = Program("const", value=-.9)
    costs = (.9, .1)
    old = fit_decision_model(predictor, training, calibration, fallback=0.0, costs=costs,
                             budget=SearchBudget(3000), adaptive=False)
    current = fit_decision_model(predictor, training, calibration, fallback=0.0, costs=costs,
                                 budget=SearchBudget(3000))
    assert current["fixed_costs"] == list(costs)
    assert current["action_rule"] == "empirical_cost_quantile"
    assert current["kind"] == "global_interval"
    assert current["residual_interval"] == old["residual_interval"]
    assert actual_decision_loss(predictor, held_out, current, costs) == pytest.approx(.19)
    assert actual_decision_loss(predictor, held_out, current, costs) < actual_decision_loss(predictor, held_out, old, costs)
    # The action lies at the empirical fixed-cost quantile, even though the
    # same displayed interval would suggest a different endpoint minimax action.
    action, low, high = action_and_interval(predictor, held_out[0], 0.0, current, costs)
    assert action == pytest.approx(0.0)
    assert action != pytest.approx(costs[0] * high + costs[1] * low)


@pytest.mark.parametrize("tail_fit", [False, True])
def test_reserved_calibration_labels_do_not_select_residual_structure(tail_fit):
    blocks = []
    for trajectory in range(6):
        history, block = [], []
        for step in range(8):
            x = (-1.0, 1.0)[step % 2]
            target = (5.0 if x > 0 else 0.0) + ((step + trajectory) % 3 - 1) * .1
            block.append(Observation({"x": x}, target, tuple(history)))
            history.append({"x": x, "__target__": target})
        blocks.append(block)
    training, calibration = sum(blocks[:5], []), blocks[5]
    shifted = [Observation(row.features, row.target + 100.0, row.history) for row in calibration]
    predictor, costs = Program("const", value=0.0), (.9, .1)
    first = fit_decision_model(predictor, training, calibration, fallback=0.0, costs=costs,
                               budget=SearchBudget(3000), tail_fit=tail_fit)
    second = fit_decision_model(predictor, training, shifted, fallback=0.0, costs=costs,
                                budget=SearchBudget(3000), tail_fit=tail_fit)
    assert first["alternatives"] == second["alternatives"]
    assert first["kind"] == second["kind"]
    assert first.get("feature") == second.get("feature")
    assert all(len(item["trajectory_losses"]) == 5 for item in first["alternatives"])
    assert first["selection_source"] == "whole_trajectory_cross_validation"
    assert first["calibration_rows"] == len(calibration)
    assert 0.0 < first["calibration_weight"] < 1.0
    expected = 100.0 if tail_fit else 100.0 * first["calibration_weight"]
    assert second["calibration_offset"] - first["calibration_offset"] == pytest.approx(expected)


def test_pooled_calibration_resists_one_extreme_value_but_adapts_to_repeated_shift():
    training = sum(cost_blocks(5), [])
    predictor, costs = Program("const", value=0.0), (.9, .1)
    calibration = [Observation({"context": 0.0}, target)
                   for target in (0., 0., 0., 0., 0., 0., 0., 1_000_000.)]
    budget = SearchBudget(3000)
    pooled = fit_decision_model(predictor, training, calibration, fallback=0.0,
                                costs=costs, budget=budget)
    old = fit_decision_model(predictor, training, calibration, fallback=0.0,
                             costs=costs, budget=SearchBudget(3000), tail_fit=False)
    assert pooled["calibration_method"] == "pooled_centered_residual_distribution"
    assert abs(pooled["calibration_offset"]) <= 1.0
    assert old["calibration_offset"] > 400_000.0
    shifted = [Observation({"context": 0.0}, 8.0) for _ in range(16)]
    adapted = fit_decision_model(predictor, training, shifted, fallback=0.0,
                                 costs=costs, budget=SearchBudget(3000))
    assert adapted["calibration_offset"] == 8.0
    assert adapted["calibration_rows"] == 16
    assert pooled["alternatives"] == adapted["alternatives"]
    assert 0 < budget.spent <= budget.limit


def test_pooled_scale_uses_central_rows_to_estimate_scale_with_few_tail_rows():
    rng = Random(937)
    rows = []
    for index in range(480):
        x, nuisance = rng.uniform(-2, 2), rng.uniform(-2, 2)
        noise = rng.gauss(0, 1)
        target = .3 - .7 * x + (.15 + .9 * abs(x)) * noise
        rows.append(Observation({"control": x, "nuisance": nuisance}, target))
    training, testing = rows[:80], rows[80:]
    budget = SearchBudget(2600)
    candidates = risk_candidates(training, budget=budget, q=.9)
    pooled = next(candidate for candidate in candidates if candidate.candidate_id.startswith("risk_pooled_scale:"))
    global_tail = next(candidate for candidate in candidates if candidate.candidate_id.startswith("risk_pooled_global:"))
    assert loss(pooled.program, testing, 0.0, .9) < loss(global_tail.program, testing, 0.0, .9)
    assert len(candidates) == 4
    assert Program.from_dict(pooled.program.to_dict()) == pooled.program
    assert 12 * len(training) < budget.spent <= budget.limit
    old_candidates = risk_candidates(training, budget=SearchBudget(2600), q=.9, tail_fit=False)
    assert not any(candidate.candidate_id.startswith("risk_pooled_") for candidate in old_candidates)


@pytest.mark.parametrize("partition,seed,task", [("confirmation", 47, "c04"), ("forward", 101, "f04")])
def test_seen_risk_failures_improve_under_the_same_observation_design(partition, seed, task):
    # Both are already-read v3 failures, now development regressions. No new
    # prospective partition is accessed by these checks.
    environment = next(item for item in make_suite(seed, partition=partition) if item.spec.task_id == task)
    blocks, queried = [], []
    for index in range(8):
        probe = representative_probe(environment.spec, seed=seed, index=index, remaining=96, used=queried)
        queried.append(probe)
        blocks.append(environment.observe(probe))
    training, calibration = sum(blocks[:6], []), sum(blocks[6:], [])
    q, costs = environment.spec.risk_quantile, environment.spec.decision_costs
    baseline = mean_with_residual_quantile(training, environment.spec.feature_names, q)
    fit_budget = SearchBudget(2000)
    chosen = min(risk_candidates(training, budget=fit_budget, q=q),
                 key=lambda candidate: loss(candidate.program, training, 0.0, q)).program
    decision_budget = SearchBudget(3000)
    current = fit_decision_model(chosen, training, calibration, fallback=0.0, costs=costs, budget=decision_budget)
    previous = fit_decision_model(baseline, training, calibration, fallback=0.0, costs=costs,
                                  budget=SearchBudget(3000), adaptive=False)
    held_out = environment.evaluation_rows()
    assert loss(chosen, held_out, 0.0, q) < loss(baseline, held_out, 0.0, q)
    assert actual_decision_loss(chosen, held_out, current, costs) < actual_decision_loss(baseline, held_out, previous, costs)
    assert fit_budget.spent <= fit_budget.limit and decision_budget.spent <= decision_budget.limit
