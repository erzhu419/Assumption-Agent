"""Paid repeat pairs can select a useful refit, pending a new confirmation."""

from copy import deepcopy
from dataclasses import replace
from statistics import mean, stdev

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.noise_replication import compare_response_refits
from test_cognition_noise_replication import noisy_responses


def entry(name, program):
    return {'candidate': Candidate(name, program, 'observed_refit'), 'kind': 'response_refit'}


def response(gain):
    feature = Program('feature', name='input')
    return Program('add', children=(feature, Program('mul', children=(Program('const', value=gain),
        Program('mul', children=(feature, feature))))))


def test_selects_refit_using_one_gain_per_paired_input_and_charges_predictions():
    program = Program('feature', name='input')
    original, repeated = noisy_responses(17, omitted_gain=.08), noisy_responses(217, omitted_gain=.08)
    wrong, correct = entry('wrong_refit', response(-.08)), entry('observed_refit', response(.08))
    before = deepcopy([wrong, correct])
    budget = SearchBudget(2000)
    result = compare_response_refits(program, [wrong, correct], original, repeated, budget)
    assert result['selected'] is correct and result['reason'] == 'supported_refit'
    assert [item['candidate_id'] for item in result['comparisons']] == ['wrong_refit', 'observed_refit']
    assert [item['supported'] for item in result['comparisons']] == [False, True]
    event = result['comparisons'][1]
    gains = [sum((row.target - program.predict(row)) ** 2
                 - (row.target - correct['candidate'].program.predict(row)) ** 2
                 for row in (left, right)) / 2 for left, right in zip(original, repeated)]
    assert event['gain'] == pytest.approx(mean(gains))
    assert event['margin'] == pytest.approx(3 * stdev(gains) / len(gains) ** .5)
    assert event['lower_bound'] > 1e-7
    assert event['paired_rows'] == len(original)
    assert event['compute_cost'] == 3 * len(original)
    assert result['compute_cost'] == budget.spent == 9 * len(original)
    assert [wrong, correct] == before
    assert 'validation_loss' not in correct and 'confirmed' not in correct


def test_weak_positive_paired_gain_does_not_recommend_a_candidate():
    original = [Observation({'input': float(index)}, .1 if index % 2 else -.1) for index in range(24)]
    repeated = [replace(row, target=.1 if index % 4 < 2 else -.1) for index, row in enumerate(original)]
    result = compare_response_refits(Program('const', value=.02),
        [entry('tiny_adjustment', Program('const', value=.019))], original, repeated, SearchBudget(1000))
    comparison = result['comparisons'][0]
    assert 0. < comparison['gain'] < comparison['margin']
    assert not comparison['supported'] and result['selected'] is None
    assert result['reason'] == 'no_supported_refit'


def test_response_history_candidate_needs_matching_history_but_does_not_block_other_refits():
    original, repeated = noisy_responses(17, omitted_gain=.08), noisy_responses(217, omitted_gain=.08)
    history = entry('response_history', Program('lag', name='__target__', lag=1))
    current = entry('current_refit', response(.08))
    budget = SearchBudget(2000)
    result = compare_response_refits(Program('feature', name='input'), [history, current],
                                     original, repeated, budget)
    assert result['selected'] is current
    assert result['comparisons'][0]['reason'] == 'unmatched_target_history'
    assert result['comparisons'][0]['compute_cost'] == len(original)
    assert result['compute_cost'] == budget.spent == 7 * len(original)


def test_incomplete_observed_trajectory_is_not_a_paired_refit_comparison():
    original, repeated = noisy_responses(17), noisy_responses(217)
    repeated[-1] = replace(repeated[-1], features={'input': 17.})
    budget = SearchBudget(2000)
    result = compare_response_refits(Program('feature', name='input'),
        [entry('response', response(.08))], original, repeated, budget)
    assert result['selected'] is None and result['comparisons'] == []
    assert result['matched_rows'] == len(original) - 1
    assert result['reason'] == 'unmatched_trajectory_inputs'
    assert result['compute_cost'] == budget.spent == len(original)


def test_unfunded_candidate_predictions_are_not_scored_or_adopted():
    original, repeated = noisy_responses(17), noisy_responses(217)
    budget = SearchBudget(5 * len(original))
    result = compare_response_refits(Program('feature', name='input'),
        [entry('response', response(.08))], original, repeated, budget)
    assert result['selected'] is None and result['reason'] == 'insufficient_budget'
    assert result['comparisons'][0]['reason'] == 'insufficient_budget'
    assert result['comparisons'][0]['compute_cost'] == 0
    assert 'gain' not in result['comparisons'][0]
    assert result['compute_cost'] == budget.spent == 3 * len(original)
