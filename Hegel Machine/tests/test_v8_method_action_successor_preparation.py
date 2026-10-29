"""Eligibility and protocol construction do not create scientific targets."""

from copy import deepcopy
import json

from scripts.prepare_pr3_source_successor import prepare_method_action_successors


def registration(tmp_path):
    method = tmp_path / 'method.json'
    method.write_text(json.dumps({'id': 'P1', 'version': 1, 'kind': 'python_research_method',
        'execution': {'kind': 'python_select_actions', 'path': 'P1.py'}}))
    returned = tmp_path / 'ordinary_return.json'
    returned.write_text(json.dumps({'execution_complete': True,
        'execution_carrier': 'python_select_actions', 'method_ref': {'id': 'P1', 'version': 1}}))
    spec = {'frozen_method_ref': {'file': str(method), 'json_path': []},
        'baseline_method_ref': {'file': 'old_method.json', 'json_path': []},
        'target_sources': [{'physical_source_id': f'new-donor-{index}', 'kind': kind,
            'generation_rule_ref': {'file': 'registered_generation.json', 'json_path': [index]}}
            for index, kind in enumerate(('memoryless', 'delay', 'hysteresis'))],
        'used_physical_source_ids': ['source-donor', 'old-case-000', 'old-case-001', 'old-case-002'],
        'used_task_seeds': [1, 2, 3],
        'target': {'seed': 1000, 'limits': {'compute_budget': 1000,
            'max_model_calls': 0, 'token_budget': 0, 'observation_budget': 576,
            'confirmation_rows': 96, 'execution_seconds': 1800.}},
        'organization': {'exploration_policy': {'kind': 'seeded_legal', 'seed': 1000}},
        'author': {'kind': 'finite_native', 'path': 'ordinary.py', 'max_steps': 48, 'confirm': False},
        'materials': {'baseline_archive': 'strong_archive.json'},
        'exposed_material_refs': [{'file': f'original_{scope}.json', 'json_path': []}
            for scope in ('nine_trials', '512', '144', '49')],
        'common_material_refs': [{'file': 'full_strong_pool_and_cache.json', 'json_path': []}],
        'cost_refs': {name: [{'file': f'{name}_ledger.json', 'json_path': []}]
            for name in ('source', 'proposals', 'installation', 'target_budget')}}
    selection = {'local_program_supported': True,
        'local_support_refs': [{'file': 'actual_matched_pair.json', 'json_path': []}]}
    return selection, spec, [{'file': str(returned), 'json_path': []}]


def test_no_new_supported_body_does_not_prepare_or_make_up_targets(tmp_path):
    destination = tmp_path / 'not_created'
    result = prepare_method_action_successors({'local_program_supported': False}, {}, destination)
    assert result['status'] == 'not_eligible'
    assert not destination.exists()
    assert not result['execution_authorized'] and not result['target_labels_generated_or_read']


def test_three_new_sources_use_existing_abc_protocol_and_preserve_exposure_and_costs(tmp_path):
    selection, spec, uses = registration(tmp_path)
    result = prepare_method_action_successors(selection, spec, ordinary_use_feedback_refs=uses)
    assert result['trial_count'] == 9
    assert result['plan']['arms'] == ['archive', 'fixed_pr2', 'pr3']
    assert all(item['installation_ref'] is None for item in result['trials'] if item['treatment'] == 'A')
    assert len({item['physical_source_id'] for item in result['trials']}) == 3
    assert result['exposed_material_refs'] == spec['exposed_material_refs']
    assert result['cost_refs'] == spec['cost_refs']
    assert result['wrong_transfer']['status'] == result['over_rejection']['status'] == 'UNKNOWN'
    assert result['source_net_benefit'] == 'UNKNOWN'
    assert not result['execution_authorized'] and not result['target_labels_generated_or_read']


def test_factorial_requires_four_actual_installations_and_does_not_claim_running(tmp_path):
    selection, spec, uses = registration(tmp_path)
    spec['design'] = 'body_guard_factorial'
    assert prepare_method_action_successors(selection, spec,
        ordinary_use_feedback_refs=uses)['reason'] == 'all_four_frozen_body_guard_installations_required'
    spec['body_guard_bindings'] = {name: {'file': f'{name}.json', 'json_path': []}
        for name in ('M0G0', 'M1G0', 'M0G1', 'M1G1')}
    result = prepare_method_action_successors(selection, spec, ordinary_use_feedback_refs=uses)
    assert result['trial_count'] == 15
    assert not result['execution_authorized']
    assert {item['treatment'] for item in result['trials']} == {'A', 'M0G0', 'M1G0', 'M0G1', 'M1G1'}


def test_renamed_old_source_or_seed_and_unrelated_method_return_do_not_qualify(tmp_path):
    selection, spec, uses = registration(tmp_path)
    old = deepcopy(spec)
    old['target_sources'][0]['physical_source_id'] = 'old-case-000'
    assert prepare_method_action_successors(selection, old,
        ordinary_use_feedback_refs=uses)['status'] == 'not_eligible'
    old = deepcopy(spec)
    old['used_task_seeds'].append(spec['target']['seed'])
    assert prepare_method_action_successors(selection, old,
        ordinary_use_feedback_refs=uses)['reason'] == 'exposed_task_cannot_be_renamed_as_a_successor'
    path = tmp_path / 'unrelated.json'
    path.write_text(json.dumps({'execution_complete': True, 'execution_carrier': 'python_select_actions',
        'method_ref': {'id': 'another-method', 'version': 1}}))
    assert prepare_method_action_successors(selection, spec,
        ordinary_use_feedback_refs=[{'file': str(path), 'json_path': []}])['reason'] == \
        'complete_actual_ordinary_use_required'
