"""Paid P0 continuation keeps completed P1 and the original comparison account."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_theory_consolidation import agent_scene
from test_v8_method_action_pairing import registration, runner
from test_v8_retained_parent_comparator import current_parent


def paid_continuation(tmp_path, *, max_steps=2):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    old = registration()
    old.update(compute_cap=300, execution_authorized=True,
        approval_ref={'file': 'authorized_pair.json'}, preparation_costs={'compute_cost': 2})
    for cell in old['cells'].values():
        cell['scope']['max_exports'] = 1
        cell['max_steps'] = max_steps
    original = save(tmp_path / 'original/registration.json', old)
    zeros = {'compute_cost': 0, 'observation_cost': 0, 'measurement_cost': 0,
        'python_execution_seconds': 0., 'model_calls': 0, 'reported_tokens': 0}
    frozen = tmp_path / 'original/study/10/frozen'
    frozen.mkdir(parents=True)
    (frozen / 'parent.py').write_text('def predict(row): return 2.\n')
    outcomes, refs = {}, {}
    for name in ('00', '10'):
        paid = {**zeros, 'compute_cost': 10 if name == '00' else 20,
                'python_execution_seconds': 2.}
        outcome = {key: deepcopy(old['cells'][name][key]) for key in (
            'method_ref', 'start_state_ref', 'data_ref', 'split_ref',
            'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}
        action = {'file': str(tmp_path / f'original/study/{name}/research_state.json'),
                  'json_path': ['method_results', 5, 'proposal']}
        outcome.update(status='native_representation_incomplete' if name == '00' else 'completed',
            execution_complete=name == '10', native_research_complete=name == '10',
            registered_attempt_returned=True, costs=paid, costs_after={**paid, 'compute_cost': paid['compute_cost'] + 7},
            timing={'activity_wall_seconds': 4., 'native_calls': 1, 'direct_external_python_calls': 1},
            actual_action_content=[{'action': 'run', 'arguments': {'operator': 'research_route'}}],
            actual_action_refs=[action], continuation_scope={'paid_rows': 1, 'pool_models': 1},
            profile_ref={'file': f'original_{name}_profile.json'}, origin_ref={'file': f'original_{name}_origin.json'},
            continuation_state_ref={'file': f'original_{name}_training.json'},
            method_artifact_ref={'file': f'original_{name}_method.json'})
        if name == '10':
            outcome['frozen_model'] = {'workspace': str(frozen), 'model': 'parent.py'}
        outcomes[name] = outcome
        refs[name] = save(tmp_path / f'original/study/{name}/outcome.json', outcome)
    save(tmp_path / 'original/study/frozen_outputs.json', {
        'models': {'10': outcomes['10']['frozen_model']}, 'evaluation_labels_opened': False})
    save(tmp_path / 'original/study/summary.json', {'ordinary_use': {'new_compute_cost': 1}})
    new = deepcopy(old)
    new.update(compute_cap=220, continuation={'registration_ref': original,
        'outcome_refs': refs, 'reuse_frozen_cells': ['10'], 'remaining_steps': {'00': max_steps - 1},
        'execution_repair': {'reason': 'minimum_native_unit_was_split_below_its_cost'}})
    for cell in new['cells'].values():
        cell.update(compute_cap=80, time_limits={'billed_python_seconds': 28., 'active_wall_seconds': 56.})
    new['cells']['00']['components']['search']['compute'] = 20
    new['cells']['00']['components']['export']['compute'] = 0
    new['cells']['10']['components'] = {
        key: {'compute': 0, 'observations': 0} for key in runner.CELL_COMPONENTS}
    new['quote'] = runner.quote_method_action_pairing(new)
    assert new['quote']['status'] == 'quoted', new['quote']
    return new, outcomes


def stopped_continuation(tmp_path):
    """The real second stage: selector stopped, paid native work stayed intact."""
    first, original = paid_continuation(tmp_path, max_steps=8)
    first['preparation_costs'] = {'compute_cost': 0}
    first_dir = tmp_path / 'first_continuation'
    first_dir.mkdir()
    registration_path = first_dir / 'registration.json'
    registration_path.write_text(json.dumps(first))
    refs, outcomes = {}, {}
    for name, previous in original.items():
        outcome = deepcopy(previous)
        costs = {key: 0. if key == 'python_execution_seconds' else 0 for key in previous['costs']}
        if name == '00':
            costs['python_execution_seconds'] = .3
        outcome.update(costs=costs,
            cumulative_original_stage_costs={key: previous['costs'][key] + value for key, value in costs.items()},
            costs_after={key: previous['costs_after'][key] + value for key, value in costs.items()},
            timing={'native_calls': 0, 'direct_external_python_calls': 1 if name == '00' else 0,
                'activity_wall_seconds': .5 if name == '00' else 0.})
        if name == '00':
            outcome.update(status='method_stopped', native_research_complete=None,
                registered_attempt_returned=False)
        path = first_dir / name / 'outcome.json'
        path.parent.mkdir()
        path.write_text(json.dumps(outcome))
        refs[name] = {'file': str(path), 'json_path': []}
        outcomes[name] = outcome
    (first_dir / 'frozen_outputs.json').write_text(json.dumps({
        'models': {'10': outcomes['10']['frozen_model']}, 'evaluation_labels_opened': False}))
    (first_dir / 'summary.json').write_text(json.dumps({'ordinary_use': {'new_compute_cost': 2}}))
    second = deepcopy(first)
    second.update(compute_cap=218, continuation={**first['continuation'],
        'registration_ref': {'file': str(registration_path), 'json_path': []},
        'outcome_refs': refs, 'remaining_steps': {'00': 6}})
    second['components']['feedback_preservation']['compute'] -= 2
    for cell in second['cells'].values():
        cell['time_limits'] = {'billed_python_seconds': 27.7, 'active_wall_seconds': 55.5}
    second['quote'] = runner.quote_method_action_pairing(second)
    assert second['quote']['status'] == 'quoted', second['quote']
    return second, original, outcomes


def test_second_continuation_keeps_original_start_and_all_paid_stages(tmp_path):
    plan, original, stopped = stopped_continuation(tmp_path)
    continuation = runner._paid_method_continuation(plan)
    assert continuation['original_registration']['compute_cap'] == 300
    assert continuation['registration']['compute_cap'] == 220
    assert plan['continuation']['remaining_steps']['00'] == 6
    for name in ('00', '10'):
        next_outcome = {'costs': {key: 0 for key in original[name]['costs']}}
        next_outcome['costs'].update(compute_cost=5 if name == '00' else 0, python_execution_seconds=.1)
        runner._continuation_costs(next_outcome, stopped[name])
        assert next_outcome['cumulative_original_stage_costs']['compute_cost'] == (
            original[name]['costs']['compute_cost'] + next_outcome['costs']['compute_cost'])
        assert next_outcome['cumulative_original_stage_costs']['python_execution_seconds'] == pytest.approx(
            original[name]['costs']['python_execution_seconds']
            + stopped[name]['costs']['python_execution_seconds'] + .1)
        assert next_outcome['prior_stage_costs'] == stopped[name]['costs']


@pytest.mark.parametrize('change', ['step_reset', 'feedback_refund', 'time_reset'])
def test_second_continuation_cannot_refund_the_stopped_stage(tmp_path, change):
    plan, _, _ = stopped_continuation(tmp_path)
    if change == 'step_reset':
        plan['continuation']['remaining_steps']['00'] = 7
    elif change == 'feedback_refund':
        plan['compute_cap'] = 220
    else:
        for cell in plan['cells'].values():
            cell['time_limits'] = {'billed_python_seconds': 28., 'active_wall_seconds': 56.}
    quoted = runner.quote_method_action_pairing(plan)
    assert quoted['status'] == 'quote_incomplete'
    assert any(reason.startswith('continuation.') for reason in quoted['missing'])


@pytest.mark.parametrize('change', [
    'account_reset', 'total_reset', 'changed_method', 'changed_data', 'step_reset',
    'unpaid_p0', 'refit_p1', 'opened_evaluation'])
def test_paid_continuation_rejects_reset_or_changed_original_attempt(tmp_path, change):
    plan, outcomes = paid_continuation(tmp_path)
    if change == 'account_reset':
        for cell in plan['cells'].values():
            cell['compute_cap'] = 100
    elif change == 'total_reset':
        plan['compute_cap'] = 300
    elif change == 'changed_method':
        plan['cells']['00']['method_ref']['version'] += 1
    elif change == 'changed_data':
        for cell in plan['cells'].values():
            cell['data_ref'] = {'file': 'new_data.json'}
    elif change == 'step_reset':
        plan['continuation']['remaining_steps']['00'] = 2
    elif change == 'unpaid_p0':
        path = Path(plan['continuation']['outcome_refs']['00']['file'])
        outcomes['00']['costs']['compute_cost'] = 0
        path.write_text(json.dumps(outcomes['00']))
    elif change == 'refit_p1':
        plan['cells']['10']['components']['search']['compute'] = 1
    else:
        path = tmp_path / 'original/study/frozen_outputs.json'
        data = json.loads(path.read_text()); data['evaluation_labels_opened'] = True
        path.write_text(json.dumps(data))
    quoted = runner.quote_method_action_pairing(plan)
    assert quoted['status'] == 'quote_incomplete'
    assert any(reason.startswith('continuation.') for reason in quoted['missing'])


def test_paid_p0_resumes_once_p1_is_not_restored_or_refrozen_and_old_fees_remain(tmp_path, monkeypatch):
    plan, previous = paid_continuation(tmp_path)
    restored, evaluated, opened, steps = [], [], [], []
    original_freeze = runner._freeze

    def freeze(agent, model, output):
        assert '/00/' in str(output)
        return original_freeze(agent, model, output)
    monkeypatch.setattr(runner, '_freeze', freeze)

    def restore(name, contract, output):
        assert name == '00', 'completed P1 must not restore or run again'
        restored.append(name)
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 2, 'feature_bounds': {'u': [-1., 1.]}}
        agent.rows = [Observation({'u': 0.}, 2.)]
        agent.costs = deepcopy(previous[name]['costs_after'])
        agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 80,
            execution_seconds=agent.costs['python_execution_seconds'] + 28., observation_budget=0)
        current_parent(agent)
        agent._save('method.py', 'def select_actions(*args): return {"stop": True}\n')
        agent._open_question({'id': 'q', 'question': 'Continue actual paid method', 'parent_id': 'task'})
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'}, 'native_attempt_limit': 1,
            'profile_ref': previous[name]['profile_ref'], 'origin_ref': previous[name]['origin_ref'],
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def method(agent, *args, **kwargs):
        steps.append(kwargs['max_steps'])
        agent.costs['compute_cost'] += 5
        agent._save('new.py', export_temporal_program(Program('const', value=2.), feature_names=('u',)))
        agent.candidate_paths.append('new.py')
        return {'status': 'completed', 'model': 'new.py', 'registered_attempt_returned': True,
            'native_research_complete': True, 'actual_action_content': [{'action': 'run'}],
            'actual_action_refs': [{'file': 'new_research.json'}]}
    monkeypatch.setattr(runner, 'execute_registered_method_unit', method)

    def prepare():
        manifest = json.loads((tmp_path / 'continued/frozen_outputs.json').read_text())
        assert set(manifest['models']) == {'00', '10'}
        assert not manifest['evaluation_labels_opened']
        opened.append(True)
        return {'file': 'one_shared_evaluation.json'}

    def evaluate(name, frozen, reference, account):
        evaluated.append(name)
        if name == '10':
            assert frozen == previous['10']['frozen_model']
            assert account.rows == ()  # No archive loading is needed for a fixed prediction.
        account.costs['compute_cost'] += 1
        return {'loss': 0., 'quality': {'loss': 0., 'threshold': 1e-10}, 'coverage': 1, 'completed': True}

    def ordinary(pair_refs, outcomes, output):
        pair = json.loads(Path(pair_refs[0]['file']).read_text())
        assert pair['method_condition_label']['preference'] == 'UNKNOWN'
        assert not pair['method_condition_label']['eligible_for_condition_learning']
        return {'status': 'execution_repair_recorded_no_condition_label'}

    result = runner.run_method_action_pairing(plan, tmp_path / 'continued', restore_cell=restore,
        prepare_evaluation=prepare, evaluate_frozen=evaluate, ordinary_use=ordinary)
    assert restored == ['00'] and steps == [1] and opened == [True] and evaluated == ['00', '10']
    for name in ('00', '10'):
        outcome = json.loads(Path(result['outcome_refs'][name]['file']).read_text())
        for key, value in outcome['new_stage_costs'].items():
            assert outcome['cumulative_original_stage_costs'][key] == previous[name]['costs'][key] + value
            assert outcome['costs_after'][key] == previous[name]['costs_after'][key] + value
        assert outcome['profile_ref'] == previous[name]['profile_ref']
        if name == '10':
            assert outcome['new_stage_costs']['compute_cost'] == 1
            assert outcome['timing']['native_calls'] == 0
            assert outcome['frozen_model'] == previous[name]['frozen_model']
            assert outcome['continuation_costs_ref']['file'] == result['outcome_refs'][name]['file']
        else:
            assert outcome['actual_action_refs'][0] == previous[name]['actual_action_refs'][0]
    assert not (tmp_path / 'continued/10/frozen').exists()
