"""Common method work scores never stand in for paired later readings."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Observation, TaskSpec
from hegel_machine.cognition.research_methods import observe_common_method_return


def feedback():
    spec = TaskSpec('fixture', 'Public response', ('u',), (), ())
    rows = [Observation({'u': float(index)}, float(index)) for index in range(8)]
    profile = method_profile(spec, rows, Program('const', value=0.), residual_loss=1.)
    return {'method_ref': {'id': 'learned-method', 'version': 2}, 'profile': profile,
        'feedback_ref': {'file': 'method_feedback/use_000000.json', 'json_path': []},
        'status': 'completed', 'execution_complete': True, 'fit_compute_cost': 80,
        'cost_refs': [{'file': 'shared/theory_library.json', 'json_path': ['executions', 'use']}],
        'working_comparison': {'before_loss': .4, 'after_loss': .1,
            'rows': 72, 'same_rows': True, 'complete': True, 'data_role': 'exposed_development'},
        'confirmation': {'mse': 1e-15, 'rows': 96, 'passed': True}}


def paired(before=1., after=.5):
    return {'before_loss': before, 'after_loss': after, 'reference_rows': 96,
        'same_rows': True, 'data_role': 'new_paid_readings',
        'source_refs': [{'file': 'later_comparison.json', 'json_path': ['both_frozen_models']}]}


def action_record(state):
    entry = state['origins'][0]
    return state['action_experience'][entry['context']]['replay_research_program']


def test_working_comparison_and_single_model_confirmation_remain_one_unscored_paid_use():
    record = feedback()
    original = deepcopy(record)
    state = observe_common_method_return(record)
    result = action_record(state)
    assert result['trials'] == 0 and result['unscored_attempts'] == 1
    assert result['unscored_compute_cost'] == 80
    assert 'expected_gain' not in result and 'continuation_outcome' not in state['origins'][0]
    assert state['origins'][0]['cost_refs'] == record['cost_refs']
    assert record == original
    assert observe_common_method_return(record, json.loads(json.dumps(state))) == state


@pytest.mark.parametrize('invalid', ['different_rows', 'working_rows', 'missing_refs',
                                    'unfinished', 'administrative_censoring'])
def test_unpaired_or_censored_results_never_upgrade_work_improvement_to_return(invalid):
    record = feedback()
    record['paired_future'] = paired()
    if invalid == 'different_rows':
        record['paired_future']['same_rows'] = False
    elif invalid == 'working_rows':
        record['paired_future']['data_role'] = 'exposed_development'
    elif invalid == 'missing_refs':
        record['paired_future']['source_refs'] = []
    elif invalid == 'unfinished':
        record.update(status='incomplete', execution_complete=False)
    else:
        record['administrative_censoring'] = {'kind': 'capacity_freeze'}
    state = observe_common_method_return(record)
    assert action_record(state)['trials'] == 0
    assert action_record(state)['unscored_compute_cost'] == 80
    assert 'paired_future' not in state['origins'][0]


def test_later_paired_feedback_promotes_original_trial_and_revisions_never_duplicate_costs():
    record = feedback()
    state = observe_common_method_return(record)
    record['paired_future'] = paired()
    promoted = observe_common_method_return(record, json.loads(json.dumps(state)))
    result = action_record(promoted)
    assert result['trials'] == 1 and result['unscored_attempts'] == 0
    assert result['unscored_compute_cost'] == 0 and result['mean_compute_cost'] == 80
    assert result['expected_gain'] == result['mean_gain'] == .5
    continuation = promoted['origins'][0]['continuation_outcome']
    assert continuation['reference_rows'] == 96 and continuation['compute_cost'] == 80
    assert continuation['gain'] == continuation['raw_gain'] == .5
    assert observe_common_method_return(record, promoted) == promoted
    assert action_record(state)['trials'] == 0  # Prior saved state was not mutated.

    another = deepcopy(record)
    another['feedback_ref']['file'] = 'method_feedback/use_000001.json'
    another['fit_compute_cost'] = 40
    another['paired_future'] = paired(after=.75)
    both = observe_common_method_return(another, promoted)
    assert action_record(both)['trials'] == 2
    assert action_record(both)['mean_compute_cost'] == 60
    record['paired_future'] = paired(after=1.5)
    revised = observe_common_method_return(record, json.loads(json.dumps(both)))
    result = action_record(revised)
    assert result['trials'] == 2 and result['mean_compute_cost'] == 60
    assert result['mean_gain'] == result['expected_gain'] == -.125
    assert result['trial_outcomes']['1']['status'] == 'refuted'
    assert revised['origins'][0]['trial_origin']['trial'] == 1
    assert revised['origins'][0]['continuation_outcome']['gain'] == -.5
    assert len(revised['origins']) == 2
    assert observe_common_method_return(record, revised) == revised


def test_missing_paid_fit_cost_stays_unknown_until_original_bill_is_available():
    record = feedback()
    record.update(fit_compute_cost=None, paired_future=paired())
    state = observe_common_method_return(record)
    assert state['action_experience'] == {}
    assert state['origins'][0]['status'] == 'UNKNOWN'
    assert state['origins'][0]['fit_compute_cost'] is None
    assert state['origins'][0]['cost_refs'] == record['cost_refs']
    assert observe_common_method_return(record, state) == state
    record['fit_compute_cost'] = 80
    measured = observe_common_method_return(record, state)
    assert action_record(measured)['trials'] == 1
    assert action_record(measured)['mean_compute_cost'] == 80
