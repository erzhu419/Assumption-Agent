"""Paid repeats separate a missing temporal relation from random responses."""

from dataclasses import replace
from random import Random
from statistics import mean, stdev

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.noise_replication import assess_noise_replication


def test_fresh_paid_repeat_exposes_missing_window_relation_without_private_labels():
    lab = AnonymousObjectLab(objects=3, apparatus_seed=9013,
                             readout="window_total_event", name_seed=42114, wiring_seed=2680)
    probe = lab.random_probe(60113, length=12, name="original")
    original = lab.acquire(probe, ("fine",)).observations
    before = dict(lab.costs)
    repeated = lab.acquire(Probe("independent_repeat", probe.sequence), ("fine",)).observations
    budget = SearchBudget(1000)
    result = assess_noise_replication(Program("const", value=0.), original, repeated, budget)
    assert lab.observation_cost - before["observation_cost"] == 12
    assert lab.measurement_cost - before["measurement_cost"] == 108
    assert lab.acquisitions[-1]["probe_id"] != lab.acquisitions[-2]["probe_id"]
    assert result["repeatable_error"] and result["matched_rows"] == 12
    assert not result['noise_supported']
    assert result["repeat_mse"] == 0. and result["original_model_loss"] > .1
    assert result["compute_cost"] == budget.spent > 0


def trajectories(seed):
    rng, history, rows = Random(seed), [], []
    for index in range(16):
        features = {"input": index / 16}
        target = features["input"] + rng.uniform(-.2, .2)
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return rows


def test_random_response_history_does_not_hide_matching_external_inputs():
    result = assess_noise_replication(Program("feature", name="input"),
        trajectories(71), trajectories(171), SearchBudget(1000))
    assert result["matched_rows"] == 16
    assert result["repeat_mse"] > 1e-4
    assert not result["repeatable_error"]
    assert result['noise_supported']
    assert result["reason"] == "response_varies_on_repeat"


def test_different_input_prefix_is_not_a_repeat_of_a_temporal_experiment():
    original = trajectories(71)
    changed = [replace(row, history=tuple({**past, "input": past["input"] + 1.}
                                         for past in row.history)) for row in original]
    result = assess_noise_replication(Program("const", value=0.), original,
                                     changed, SearchBudget(1000))
    assert result["matched_rows"] == 1
    assert not result["repeatable_error"]
    assert result["reason"] == "insufficient_matched_rows"
    assert not result['noise_supported']


def test_nested_target_history_predictor_requires_the_same_response_inputs():
    definition = Definition("previous_response", (), Program("lag", name="__target__", lag=1))
    program = Program("call", name=definition.definition_id, definition=definition)
    result = assess_noise_replication(program, trajectories(71), trajectories(171), SearchBudget(1000))
    assert result["target_history_required"]
    assert result["matched_rows"] == 1 and not result["repeatable_error"]
    assert not result['noise_supported']


def test_fixed_quantile_and_predictor_fallback_remain_the_scored_objective():
    rows = [Observation({"input": 2.}, 2.), Observation({"input": 2.}, 2.)]
    missing_history = Program("lag", name="input", lag=1)
    exact = assess_noise_replication(missing_history, rows, rows, SearchBudget(100),
                                    fallback=2., risk_quantile=.9)
    assert exact["original_model_loss"] == 0. and not exact["repeatable_error"]
    wrong = assess_noise_replication(missing_history, rows, rows, SearchBudget(100),
                                    fallback=0., risk_quantile=.9)
    assert wrong["original_model_loss"] == 1.8 and wrong["repeatable_error"]


@pytest.mark.parametrize('quantile', [.5, .9])
@pytest.mark.parametrize('multiple', [1.01, 2.405270544368031, 9.9])
def test_reproduced_error_between_one_and_ten_tolerances_cannot_be_excused(quantile, multiple):
    tolerance = 1e-7
    target = (multiple * tolerance) ** .5 if quantile == .5 else multiple * tolerance / quantile
    original = [Observation({'input': float(index)}, target,
        tuple({'input': float(previous), '__target__': target} for previous in range(index)))
        for index in range(16)]
    repeated = [replace(row, features=dict(row.features), history=tuple(dict(past) for past in row.history))
                for row in original]
    budget = SearchBudget(1000)
    result = assess_noise_replication(Program('const', value=0.), original, repeated, budget,
        minimum_improvement=tolerance, risk_quantile=quantile)
    assert result['repeatable_error'] and not result['noise_supported']
    assert result['original_model_loss'] == pytest.approx(multiple * tolerance)
    assert result['repeated_model_loss'] == pytest.approx(multiple * tolerance)
    assert result['repeat_loss'] == 0. and result['repeatable_loss_limit'] == tolerance
    assert result['compute_cost'] == budget.spent == 3 * len(original)


def test_low_repeat_variation_supplies_the_allowance_in_the_actual_loss_units():
    tolerance = 1e-7
    center, variation = (.8 * tolerance) ** .5, (.4 * tolerance) ** .5
    original = [Observation({'input': float(index)}, center + (variation if index % 2 else -variation))
                for index in range(16)]
    repeated = [replace(row, target=center + (variation if index % 4 < 2 else -variation))
                for index, row in enumerate(original)]
    result = assess_noise_replication(Program('const', value=0.), original, repeated,
                                     SearchBudget(1000), minimum_improvement=tolerance)
    assert result['repeat_loss'] == pytest.approx(.8 * tolerance)
    assert result['original_model_loss'] == pytest.approx(1.2 * tolerance)
    assert result['repeatable_loss_limit'] == pytest.approx(1.8 * tolerance)
    assert not result['repeatable_error'] and not result['noise_supported']
    assert result['reason'] == 'no_material_repeatable_error'


@pytest.mark.parametrize('change', ['length', 'later_input', 'budget'])
def test_unfinished_or_partially_matching_repeat_is_not_noise_evidence(change):
    original = trajectories(71)
    repeated = trajectories(171)
    budget = SearchBudget(1000)
    if change == 'length':
        repeated = repeated[:-1]
    elif change == 'later_input':
        repeated = [row if index < 4 else replace(row,
            features={**row.features, 'input': row.features['input'] + .1})
            for index, row in enumerate(repeated)]
    else:
        budget = SearchBudget(len(original))
    result = assess_noise_replication(Program('feature', name='input'), original, repeated, budget)
    assert not result['noise_supported'] and not result['repeatable_error']
    assert result['reason'] == {'length': 'different_trajectory_length',
        'later_input': 'unmatched_trajectory_inputs', 'budget': 'insufficient_budget'}[change]
    assert result['compute_cost'] == budget.spent <= budget.limit


def noisy_responses(seed, count=96, *, omitted_gain=0.):
    rng, history, rows = Random(seed), [], []
    for index in range(count):
        features = {'input': -1. + 2. * index / (count - 1)}
        target = features['input'] + omitted_gain * features['input'] ** 2 + rng.gauss(0., .01)
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, '__target__': target})
    return rows


def test_independent_residual_products_expose_missing_response_under_real_variation():
    original = noisy_responses(17, omitted_gain=.02)
    repeated = noisy_responses(217, omitted_gain=.02)
    program = Program('feature', name='input')
    budget = SearchBudget(1000)
    result = assess_noise_replication(program, original, repeated, budget)
    products = [(left.target - program.predict(left)) * (right.target - program.predict(right))
                for left, right in zip(original, repeated)]
    assert result['repeat_loss'] > 1e-7
    assert min(result['original_model_loss'], result['repeated_model_loss']) <= result['repeatable_loss_limit']
    assert result['reproducible_mse'] == pytest.approx(mean(products))
    assert result['reproducible_mse_margin'] == pytest.approx(3 * stdev(products) / len(products) ** .5)
    assert result['reproducible_mse_lower_bound'] > 1e-7
    assert result['repeat_noise_variance'] == pytest.approx(result['repeat_mse'] / 2)
    assert result['repeatable_error'] and not result['noise_supported']
    assert result['reason'] == 'reproducible_residual_despite_noise'
    assert result['compute_cost'] == budget.spent == 4 * len(original)


def test_input_heterogeneity_cannot_hide_error_far_larger_than_repeat_variation():
    # One region has a large missed response. Across-input uncertainty makes
    # the product's lower bound negative, but the two independent trajectories
    # agree closely at each input and both strongly contradict the predictor.
    original, repeated = [], []
    original_history, repeated_history = [], []
    for index in range(24):
        features = {'input': float(index)}
        response = 6.4 if index == 23 else 0.
        first = response + (.01 if index % 2 else -.01)
        second = response + (.01 if index % 4 < 2 else -.01)
        original.append(Observation(features, first, tuple(original_history)))
        repeated.append(Observation(features, second, tuple(repeated_history)))
        original_history.append({**features, '__target__': first})
        repeated_history.append({**features, '__target__': second})
    products = [left.target * right.target for left, right in zip(original, repeated)]
    assert mean(products) - 3 * stdev(products) / len(products) ** .5 < 0.

    budget = SearchBudget(3 * len(original))
    result = assess_noise_replication(Program('const', value=0.), original, repeated, budget)
    assert result['matched_rows'] == 24
    assert result['repeat_loss'] == pytest.approx(.0002)
    assert min(result['original_model_loss'], result['repeated_model_loss']) > 1.7
    assert result['repeatable_error'] and not result['noise_supported']
    assert result['reason'] == 'repeat_variation_does_not_explain_error'
    assert result['reproducible_mse'] is None
    assert result['compute_cost'] == budget.spent == 3 * len(original)


@pytest.mark.parametrize('seed', [13, 55, 71])
def test_unpredictable_variation_remains_supported_without_reproducible_residual(seed):
    original, repeated = noisy_responses(seed), noisy_responses(seed + 1000)
    result = assess_noise_replication(Program('feature', name='input'), original, repeated, SearchBudget(1000))
    assert result['reproducible_mse_lower_bound'] <= 1e-7
    assert not result['repeatable_error'] and result['noise_supported']


def test_positive_but_uncertain_residual_product_does_not_refute_noise():
    original = [Observation({'input': float(index)}, .1) for index in range(24)]
    repeated = [replace(row, target=.12 if index % 2 else -.08) for index, row in enumerate(original)]
    result = assess_noise_replication(Program('const', value=0.), original, repeated, SearchBudget(1000))
    assert 0. < result['reproducible_mse'] < result['reproducible_mse_margin']
    assert not result['repeatable_error'] and result['noise_supported']


def test_squared_residual_product_does_not_replace_the_fixed_quantile_objective():
    original, repeated = noisy_responses(17), noisy_responses(217)
    upper = Program('add', children=(Program('feature', name='input'), Program('const', value=.02)))
    budget = SearchBudget(1000)
    result = assess_noise_replication(upper, original, repeated, budget, risk_quantile=.9)
    assert result['fixed_quantile'] == .9 and result['noise_supported']
    assert result['reproducible_mse'] is None
    assert result['compute_cost'] == budget.spent == 3 * len(original)


def test_paired_structure_comparison_is_not_performed_without_its_compute_budget():
    original, repeated = noisy_responses(17), noisy_responses(217)
    budget = SearchBudget(3 * len(original))
    result = assess_noise_replication(Program('feature', name='input'), original, repeated, budget)
    assert result['reason'] == 'insufficient_budget'
    assert result['reproducible_mse'] is None and not result['noise_supported']
    assert result['compute_cost'] == budget.spent == budget.limit
