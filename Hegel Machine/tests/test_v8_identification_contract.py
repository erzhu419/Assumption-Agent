"""Matched strategy labels keep their data and evaluation-time interpretation."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.method_conditions import paired_method_preference
from hegel_machine.cognition.method_feedback import research_paired_method_conditions
from test_method_use_conditions import METHOD
from test_v8_paired_method_conditions import pair, learn


def query(value):
    return [{'action': 'query', 'arguments': {'actions': [
        {'kind': 'trajectory', 'sequence': [{'u': value}]}]}}]


def test_retained_parent_contrast_cannot_become_a_method_preference():
    item = pair(0)
    item['comparison_kind'] = 'additional_research_vs_retained_parent'
    result = paired_method_preference(item)
    assert result['preference'] == 'UNKNOWN'
    assert result['reason'] == 'retained_parent_contrast_is_not_a_method_preference'
    assert result['identified_effect'] == 'local_incremental_research_value'
    learned = learn([item])
    assert not learned['local_program_supported'] and not learned['general_guard_supported']


def test_referenced_parent_baseline_needs_no_fictional_method_profile(tmp_path):
    for name in ('baseline', 'challenger'):
        (tmp_path / f'{name}.json').write_text(json.dumps({
            'execution_complete': True, 'research_execution_complete': name == 'challenger'}))
    source = {'comparison_kind': 'additional_research_vs_retained_parent',
        **{f'{name}_ref': {'file': f'{name}.json'} for name in ('baseline', 'challenger')}}
    (tmp_path / 'pair.json').write_text(json.dumps(source))
    agent = SimpleNamespace(workspace=tmp_path, costs={'compute_cost': 0})
    result = research_paired_method_conditions(agent, METHOD,
        [{'file': 'pair.json'}], 128)
    assert not result['local_program_supported'] and not result['general_guard_supported']
    assert result['preferences'][0]['reason'] == 'retained_parent_contrast_is_not_a_method_preference'


def test_different_acquisitions_identify_whole_strategy_and_only_local_support():
    item = pair(0)
    item['baseline_action_content'] = query(.4)
    item['challenger_action_content'] = query(-.4)
    item['challenger']['costs']['observation_cost'] = 12
    preference = paired_method_preference(item)
    assert preference['preference'] == 'challenger'
    assert preference['identified_effect'] == 'research_strategy_effect'
    assert preference['acquisition_actions_differ']
    assert not preference['data_and_method_separately_identified']
    result = learn([item])
    assert result['local_program_supported']
    assert result['identified_effects'] == ['research_strategy_effect']
    assert not result['general_guard_supported'] and result['condition'] is None
    assert not result['independent_confirmation']


def test_paid_answer_query_without_new_rows_keeps_fixed_material_interpretation():
    item = pair(0)
    item['challenger_action_content'] = query(-.4)
    for side in ('baseline', 'challenger'):
        item[side]['costs']['observation_cost'] = 0
        # Restored workspaces have distinct file addresses despite equal input
        # material. The zero acquisition ledger matters, not path equality.
        item[side]['continuation_data_ref'] = {'file': side + '/observations.json'}
    result = paired_method_preference(item)
    assert result['preference'] == 'challenger'
    assert result['identified_effect'] == 'fixed_data_method_effect'


@pytest.mark.parametrize('target', ['fixed_data_method_effect', 'data_effect', 'method_data_interaction'])
def test_own_acquisition_pair_cannot_claim_isolated_data_or_method_effect(target):
    item = pair(0)
    item['baseline_action_content'], item['challenger_action_content'] = query(.4), query(-.4)
    item['identification_target'] = target
    assert paired_method_preference(item)['preference'] == 'UNKNOWN'
    assert not learn([item])['local_program_supported']


def test_qeval_labels_are_development_labels_only_after_both_outputs_freeze():
    item = pair(0)
    item['evaluation_lifecycle'] = {'role': 'development_after_freeze',
        'all_outputs_frozen_before_labels': False, 'labels_revealed': True}
    frozen_outputs = deepcopy(item)
    for loss in (1e-30, 1e3):
        item['challenger']['quality']['loss'] = loss
        assert paired_method_preference(item)['preference'] == 'UNKNOWN'
        assert not learn([item])['local_program_supported']
    item = frozen_outputs
    item['evaluation_lifecycle']['all_outputs_frozen_before_labels'] = True
    item['evaluation_lifecycle']['labels_revealed'] = False
    assert paired_method_preference(item)['preference'] == 'UNKNOWN'
    item['evaluation_lifecycle']['labels_revealed'] = True
    result = learn([item])
    assert result['local_program_supported']
    assert result['evaluation_use'] == 'exposed_development_labels_only'
    assert not result['general_guard_supported']
    assert item['baseline'] == frozen_outputs['baseline']


def test_an_activity_overrun_never_becomes_a_paired_method_preference():
    item = pair(0)
    item['challenger']['registered_time_limits'] = {'active_wall_seconds': 3000}
    item['challenger']['timing'] = {'activity_wall_seconds': 6119.813277066103}
    result = paired_method_preference(item)
    assert result['preference'] == 'UNKNOWN'
    assert result['reason'] == 'registered_activity_limit_exceeded'
    assert not learn([item])['local_program_supported']


def test_reference_consumer_checks_actual_registered_time_and_fixed_readout(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    item = pair(0)
    public = save('profile.json', item['baseline']['profile'])
    basis = {name: save(name + '.json', value) for name, value in item['baseline']['basis'].items()}
    basis['resource_contract'] = save('resource_contract.json', {'time_limits': {'active_wall_seconds': 3000}})
    outcomes = {}
    for side in ('baseline', 'challenger'):
        outcome = deepcopy(item[side])
        outcome.pop('basis'); outcome.pop('profile')
        outcome.update(profile_ref=public, **{name + '_ref': ref for name, ref in basis.items()})
        outcome['timing'] = {'activity_wall_seconds': 10 if side == 'baseline' else 6119.813}
        outcomes[side + '_ref'] = save(side + '.json', outcome)
    reference = save('pair.json', {**{k: v for k, v in item.items()
        if k not in {'baseline', 'challenger', 'feedback_ref'}}, **outcomes})
    agent = SimpleNamespace(workspace=tmp_path, costs={'compute_cost': 0})
    result = research_paired_method_conditions(agent, METHOD, [reference], 0)
    assert result['preferences'][0]['reason'] == 'registered_activity_limit_exceeded'
    assert not result['local_program_supported'] and agent.costs['compute_cost'] == 0

    # The real fixed-readout format has no method profile/basis and preserves
    # the original failed pair. It should be rejected as such, not expanded.
    for side in ('baseline', 'challenger'):
        save(side + '.json', {'status': 'fixed_readout_returned', 'original_pair_completed': False,
                              'measured': {'loss': .1, 'coverage': 24}})
    result = research_paired_method_conditions(agent, METHOD, [reference], 0)
    assert result['preferences'][0]['reason'] == 'fixed_model_readout_is_not_an_executed_method_pair'
    assert not result['local_program_supported'] and agent.costs['compute_cost'] == 0
