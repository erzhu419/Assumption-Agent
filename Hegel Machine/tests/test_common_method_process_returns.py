"""Comparable research resources are not fabricated prediction-loss gains."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.research_methods import (
    common_method_process_return, observe_common_method_return,
)
from test_common_method_returns import action_record, feedback


def case000_comparison(baseline='fixed_pr2'):
    directory = Path(__file__).resolve().parents[1] / 'docs/evidence/v7_research_semantics_20261026'
    source = directory / 'case000_actual_method_effect.json'
    final = directory / 'method_successor_execution.json'
    actual = json.loads(source.read_text())
    receipts = json.loads(final.read_text())['latest_author_completion_observation']['completed_receipts']

    def arm(name):
        row = next(item for item in receipts if item['case'] == 'case_000' and item['arm'] == name)
        return {'execution_complete': row['status'] == 'completed',
            'completed': row['terminal']['passed'],
            'quality': {'loss': row['terminal']['mse'],
                'threshold': actual['arms'][name]['confirmation']['mse_threshold']},
            'coverage': row['terminal']['evaluation_rows'], 'costs': deepcopy(row['costs'])}

    result = {'scope': 'matched_target_policy_comparison_not_individual_object_causal',
        'same_task': True, 'same_evaluation_contract': True,
        'source_refs': [{'file': str(source), 'json_path': []}, {'file': str(final),
            'json_path': ['latest_author_completion_observation', 'completed_receipts']}],
        'before': arm(baseline), 'after': arm('pr3')}
    delta = actual['comparisons'][f'full_target_C_minus_{"B" if baseline == "fixed_pr2" else "A"}_compute'][
        'conservative_interval']
    total = result['after']['costs']['compute_cost']
    result['before']['cost_intervals'] = {'compute_cost': [total - delta[1], total - delta[0]]}
    return result


def test_actual_case000_retains_equal_quality_process_gain_and_unknown_cost_order():
    comparison = case000_comparison()
    result = common_method_process_return(comparison)
    assert result['status'] == 'measured' and result['helpful'] is True
    assert result['scope'] == 'matched_target_policy_comparison_not_individual_object_causal'
    assert result['quality']['after_loss'] > result['quality']['before_loss']
    assert result['quality']['before_passed'] and result['quality']['after_passed']
    assert result['coverage'] == {'before': 512, 'after': 512, 'change': 0}
    assert result['resource_changes']['observation_cost']['delta_interval'] == [-24, -24]
    assert result['resource_changes']['compute_cost']['delta_interval'] == [-214734, -64662]
    against_archive = common_method_process_return(case000_comparison('archive'))
    assert against_archive['status'] == 'UNKNOWN' and against_archive['helpful'] is None
    assert against_archive['resource_changes']['observation_cost']['direction'] == 'lower'
    assert against_archive['resource_changes']['compute_cost']['delta_interval'] == [-4173842, 22894]


@pytest.mark.parametrize('changed', ['unfinished', 'paused', 'different_contract', 'missing_cost'])
def test_unfinished_censored_or_unknown_processes_never_become_negative_examples(changed):
    comparison = case000_comparison()
    if changed == 'unfinished':
        comparison['after']['execution_complete'] = False
    elif changed == 'paused':
        comparison['after']['administrative_censoring'] = {'kind': 'capacity_pause'}
    elif changed == 'different_contract':
        comparison['after']['quality']['threshold'] *= 2
    else:
        comparison['after']['costs']['compute_cost'] = None
    result = common_method_process_return(comparison)
    assert result['status'] == 'UNKNOWN' and result['helpful'] is None
    record = feedback()
    record['process_comparison'] = comparison
    state = observe_common_method_return(record)
    origin, = state['origins']
    assert origin['process_return'] == result and 'process_continuation_outcome' not in origin
    assert action_record(state)['trials'] == 0 and action_record(state)['unscored_compute_cost'] == 80


@pytest.mark.parametrize('changed', ['quality', 'coverage'])
def test_complete_process_degradation_cannot_be_offset_by_lower_resource_cost(changed):
    comparison = case000_comparison()
    if changed == 'quality':
        comparison['after']['quality']['loss'] = 1.
    else:
        comparison['after']['coverage'] = 256
    result = common_method_process_return(comparison)
    assert result['status'] == 'measured' and result['helpful'] is False
    assert result['reason'] == 'completed_research_degrades_completion_quality_or_coverage'


def test_process_return_keeps_zero_prediction_gain_and_whole_process_cost_once():
    record = feedback()
    record['process_comparison'] = case000_comparison()
    state = observe_common_method_return(record)
    origin, = state['origins']
    assert origin['process_return']['helpful'] is True
    continuation = origin['process_continuation_outcome']
    assert continuation['gain'] == continuation['raw_gain'] == 0.
    assert continuation['reference_rows'] == 512
    assert continuation['observation_cost'] == 168 and continuation['compute_cost'] == 101218
    assert continuation['process_return'] == origin['process_return']
    assert action_record(state)['trials'] == 0 and action_record(state)['unscored_compute_cost'] == 80
    assert observe_common_method_return(record, json.loads(json.dumps(state))) == state
    assert origin['cost_refs'] == record['cost_refs']
    record['fit_compute_cost'] = None
    unknown = observe_common_method_return(record)
    assert unknown['origins'][0]['fit_compute_cost'] is None and unknown['action_experience'] == {}
    assert unknown['origins'][0]['process_return']['helpful'] is True


def test_retained_exact_cost_intervals_enter_existing_continuation_without_duplicate_bill():
    record = feedback()
    comparison = record['process_comparison'] = case000_comparison()
    comparison['after']['cost_intervals'] = {
        key: [comparison['after']['costs'][key]] * 2
        for key in ('observation_cost', 'compute_cost')}
    state = observe_common_method_return(record)
    origin, = state['origins']
    outcome = origin['process_continuation_outcome']
    assert outcome['observation_cost'] == 168 and outcome['compute_cost'] == 101218
    assert observe_common_method_return(record, json.loads(json.dumps(state))) == state
    comparison['after']['cost_intervals']['compute_cost'] = [101218, 101219]
    revised = observe_common_method_return(record, state)
    assert revised['origins'][0]['process_return']['helpful'] is True
    assert 'process_continuation_outcome' not in revised['origins'][0]
    assert action_record(revised)['unscored_compute_cost'] == 80
