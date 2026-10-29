"""Resume only the failed selector, retaining the already frozen control."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_theory_consolidation import agent_scene
from test_v8_method_action_pairing import runner
from test_v8_retained_parent_comparator import current_parent, retained_registration


def continued_pair(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    old = retained_registration()
    data = save(tmp_path / 'original/training.json', {'paid_observations': [
        {'rows': [{'features': {'u': 0.}, 'target': 2., 'history': []}]}]})
    for cell in old['cells'].values():
        cell['start_state_ref'] = data
        cell['data_ref'] = {**data, 'json_path': ['paid_observations']}
    old['source_scene'] = {'plan_ref': save(tmp_path / 'original/plan.json', {'limits': {}})}
    original = save(tmp_path / 'old/registration.json', old)
    zeros = {'observation_cost': 0, 'measurement_cost': 0, 'compute_cost': 0,
        'python_execution_seconds': 0., 'model_calls': 0, 'reported_tokens': 0}
    prefix = {**zeros, 'observation_cost': 1, 'compute_cost': 7, 'python_execution_seconds': 3.}
    frozen = tmp_path / 'old/study/00/frozen'
    frozen.mkdir(parents=True)
    (frozen / 'parent.py').write_text('def predict(row): return 1.\n')
    outcomes, refs = {}, {}
    for name in ('00', '10'):
        paid = {**zeros, 'python_execution_seconds': 4. if name == '10' else 0.}
        outcome = {key: deepcopy(old['cells'][name][key]) for key in (
            'method_ref', 'start_state_ref', 'data_ref', 'split_ref', 'quality_contract_ref',
            'resource_contract_ref', 'evaluation_ref')}
        outcome.update(execution_complete=name == '00', status='method_no_progress' if name == '10'
            else 'retained_parent_baseline_frozen', costs=paid,
            costs_after={key: value + paid[key] for key, value in prefix.items()},
            timing={'activity_wall_seconds': 10. if name == '10' else 2.,
                'direct_external_python_calls': 1 if name == '10' else 0, 'native_calls': 0},
            actual_action_content=[], continuation_scope={'paid_rows': 1, 'pool_models': 1},
            continuation_state_ref=data, continuation_data_ref=old['cells'][name]['data_ref'],
            continuation_costs_ref={'file': 'old_account.json'},
            research_execution_complete=False, research_status='author_work_required')
        if name == '00':
            outcome.update(method_delivery='registered_retained_parent_baseline',
                retained_parent_declaration=deepcopy(old['cells'][name]['retained_parent']),
                frozen_model={'workspace': str(frozen), 'model': 'parent.py'})
        outcomes[name] = outcome
        refs[name] = save(tmp_path / f'old/study/{name}/outcome.json', outcome)
    save(tmp_path / 'old/study/frozen_outputs.json', {'models': {'00': outcomes['00']['frozen_model']},
        'evaluation_labels_opened': False, 'evaluation_lifecycle': {'labels_revealed': False}})
    new = deepcopy(old)
    new['continuation'] = {'registration_ref': original, 'outcome_refs': refs,
        'remaining_steps': {'10': 1}, 'reuse_frozen_cells': ['00']}
    for cell in new['cells'].values():
        cell['time_limits'] = {'billed_python_seconds': 26., 'active_wall_seconds': 50.}
    new['quote'] = runner.quote_method_action_pairing(new)
    assert new['quote']['status'] == 'quoted', new['quote']
    return new, outcomes


@pytest.mark.parametrize('change', ['python_reset', 'activity_reset', 'step_reset',
    'method_change', 'logical_source_change', 'exposed_labels', 'missing_parent'])
def test_continuation_rejects_reset_budgets_or_changed_scientific_contract(tmp_path, change):
    plan, outcomes = continued_pair(tmp_path)
    if change == 'python_reset':
        for cell in plan['cells'].values():
            cell['time_limits']['billed_python_seconds'] = 30.
    elif change == 'activity_reset':
        for cell in plan['cells'].values():
            cell['time_limits']['active_wall_seconds'] = 60.
    elif change == 'step_reset':
        plan['continuation']['remaining_steps']['10'] = 2
    elif change == 'method_change':
        plan['cells']['10']['method_ref']['version'] += 1
    elif change == 'logical_source_change':
        for cell in plan['cells'].values():
            cell['start_state_ref'] = {'file': 'another_scene.json'}
    elif change == 'exposed_labels':
        path = tmp_path / 'old/study/frozen_outputs.json'
        manifest = json.loads(path.read_text())
        manifest['evaluation_labels_opened'] = True
        path.write_text(json.dumps(manifest))
    else:
        (Path(outcomes['00']['frozen_model']['workspace']) / 'parent.py').unlink()
    quote = runner.quote_method_action_pairing(plan)
    assert quote['status'] == 'quote_incomplete'
    assert any(reason.startswith('continuation.') for reason in quote['missing'])


def test_resume_keeps_control_frozen_and_charges_only_new_work(tmp_path, monkeypatch):
    plan, previous = continued_pair(tmp_path)
    restored, steps, evaluation = [], [], []
    freeze = runner._freeze

    def only_new_freeze(agent, model, output):
        assert '/10/' in str(output), 'the already frozen baseline was frozen again'
        return freeze(agent, model, output)
    monkeypatch.setattr(runner, '_freeze', only_new_freeze)

    def restore(name, contract, output):
        assert name == '10', 'the baseline must not restore its large research state'
        restored.append(name)
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 2, 'feature_bounds': {'u': [-1., 1.]}}
        agent.rows = [Observation({'u': 0.}, 2.)]
        agent.costs = deepcopy(previous[name]['costs_after'])
        agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 100,
            observation_budget=agent.costs['observation_cost'],
            execution_seconds=agent.costs['python_execution_seconds'] + 26.)
        current_parent(agent)
        agent._save('method.py', 'def select_actions(*args): return {"stop": True}\n')
        agent._open_question({'id': 'q', 'question': 'synthetic continuation', 'parent_id': 'task'})
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'}, 'native_attempt_limit': 1,
            'profile_ref': None, 'origin_ref': None,
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def method(agent, *args, **kwargs):
        steps.append(kwargs['max_steps'])
        agent._save('new.py', export_temporal_program(Program('const', value=2.), feature_names=('u',)))
        agent.candidate_paths.append('new.py')
        agent.operator_results.append({'operator': 'native_condition_research',
            'status': 'frozen_predictions_compared', 'native_condition_research': {
                'status': 'frozen_predictions_compared', 'structural_discrimination': {
                    'forecast_verdict': {'complete': True, 'next_research': {'kind': 'compare'}}}}})
        return {'status': 'completed', 'model': 'new.py', 'registered_attempt_returned': True,
            'actual_action_content': [], 'native_boundary_refs': [
                {'file': str(agent.workspace / 'research_state.json'), 'json_path': ['operator_results', 0]}]}
    monkeypatch.setattr(runner, 'execute_registered_method_unit', method)

    def evaluate(name, frozen, reference, account):
        evaluation.append(name)
        assert len(account.rows) == 1
        account.costs['compute_cost'] += 1
        return {'loss': 1. if name == '00' else 0., 'completed': name == '10', 'coverage': 1}

    result = runner.run_method_action_pairing(plan, tmp_path / 'new/study', restore_cell=restore,
        evaluate_frozen=evaluate, ordinary_use=lambda *_: pytest.fail('ordinary use is outside this pair'))
    assert restored == ['10'] and steps == [1] and evaluation == ['00', '10']
    for name in ('00', '10'):
        outcome = json.loads(Path(result['outcome_refs'][name]['file']).read_text())
        assert outcome['new_stage_costs'] == outcome['costs']
        assert outcome['prior_stage_costs'] == previous[name]['costs']
        for key, value in outcome['costs'].items():
            assert outcome['cumulative_original_stage_costs'][key] == previous[name]['costs'][key] + value
            assert outcome['costs_after'][key] == previous[name]['costs_after'][key] + value
    baseline = json.loads(Path(result['outcome_refs']['00']['file']).read_text())
    assert baseline['frozen_model'] == previous['00']['frozen_model']
    assert baseline['costs']['compute_cost'] == 1
    assert not (tmp_path / 'new/study/00/case_002').exists()
