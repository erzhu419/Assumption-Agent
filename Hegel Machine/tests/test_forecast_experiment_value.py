"""Decision value uses task consequences rather than raw forecast amplitude."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.probing import value_forecast_experiments


def panel():
    # The first experiment distinguishes a nuisance factor with no bearing on
    # which predictor should be used on the task's public reference controls.
    return [[-1.], [1.], [-1.], [1.]], [
        {'id': 'large_nuisance', 'predictions': [[-100.], [-100.], [100.], [100.]],
         'observation_cost': 1},
        {'id': 'small_relevant', 'predictions': [[-.1], [.1], [-.1], [.1]],
         'observation_cost': 1},
    ]


def test_task_value_prefers_relevant_split_over_larger_nuisance_variance():
    reference, experiments = panel()
    before = deepcopy(experiments)
    budget = SearchBudget(10000)
    result = value_forecast_experiments(reference, experiments,
        weights=[.25] * 4, noise_scale=1e-4, budget=budget)
    assert result['status'] == 'valued'
    assert result['selected_experiment_id'] == 'small_relevant'
    assert result['reference_risk'] == pytest.approx(2.)
    relevant, nuisance = result['comparisons']
    assert relevant['expected_risk_reduction'] == pytest.approx(2.)
    assert nuisance['expected_risk_reduction'] == pytest.approx(0.)
    assert result['compute_cost'] == budget.spent > 0
    assert experiments == before  # No acquisition, refitting or authored-input mutation.


def test_splitting_existing_weight_across_duplicate_hypotheses_preserves_value():
    reference, experiments = panel()
    original = value_forecast_experiments(reference, experiments,
        weights=[.25] * 4, noise_scale=1e-4, budget=SearchBudget(10000))
    duplicate = deepcopy(experiments)
    for experiment in duplicate:
        experiment['predictions'].append(experiment['predictions'][0][:])
    repeated = value_forecast_experiments(reference + [reference[0][:]], duplicate,
        weights=[.125, .25, .25, .25, .125], noise_scale=1e-4, budget=SearchBudget(10000))
    assert repeated['selected_experiment_id'] == original['selected_experiment_id']
    assert repeated['reference_risk'] == pytest.approx(original['reference_risk'])
    for left, right in zip(repeated['comparisons'], original['comparisons']):
        assert left['id'] == right['id']
        assert left['value_per_observation'] == pytest.approx(right['value_per_observation'])


def test_forecast_disagreement_with_no_reference_loss_has_no_recommendation():
    result = value_forecast_experiments([[2.], [2.]], [
        {'id': 'distinguishable_but_irrelevant', 'predictions': [[-10.], [10.]],
         'observation_cost': 1}], weights=[.5, .5], noise_scale=1e-4,
        budget=SearchBudget(10000))
    assert result['status'] == 'no_positive_value'
    assert result['selected_experiment_id'] is None
    assert result['reference_risk'] == 0.
    assert result['comparisons'][0]['expected_risk_reduction'] == 0.


def test_incomplete_panel_budget_does_not_recommend_or_spend_on_partial_comparison():
    reference, experiments = panel()
    full = value_forecast_experiments(reference, experiments,
        weights=[.25] * 4, noise_scale=1e-4, budget=SearchBudget(10000))
    budget = SearchBudget(full['required_compute'] - 1)
    result = value_forecast_experiments(reference, experiments,
        weights=[.25] * 4, noise_scale=1e-4, budget=budget)
    assert result['status'] == 'budget_exhausted'
    assert result['selected_experiment_id'] is None
    assert result['comparisons'] == []
    assert result['compute_cost'] == budget.spent == 0
