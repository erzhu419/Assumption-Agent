"""Representation research returns through the existing one-unit method boundary."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_native_representation_routing import PublicOnlyLab
from test_native_startup_revision import scene as startup_scene
from test_theory_consolidation import agent_scene
from test_v8_method_action_pairing import registration, runner


def receipt(*, complete=True, failed=False):
    return {'operator': 'research_route',
        'native_representation_ref': {'file': 'native_representation_research/record_0000.json',
                                     'json_path': []},
        'native_representation': {
            'status': ('native_representation_execution_failed' if failed else
                'native_representation_draft_selected' if complete else
                'native_representation_inconclusive_budget'),
            'comparison_completed': complete,
            'research_completion': 'bounded_search_completed' if complete else 'comparison_incomplete',
            **({'error': 'actual solver failure'} if failed else {})}}


def test_actual_startup_route_ends_one_unit_without_repeating_the_method(tmp_path):
    agent, original, arguments = startup_scene(tmp_path)
    agent._save('method.py', 'def select_actions(*args):\n    return ' + repr(
        {'action': 'run', 'arguments': arguments}) + '\n')
    result = runner.execute_registered_method_unit(agent, PublicOnlyLab(), 'method.py',
        max_steps=8, native_attempt_limit=1)
    assert result['status'] == 'registered_native_attempt_returned'
    assert result['registered_attempt_returned'] and result['native_research_complete']
    assert len(agent.method_results) == len(result['actual_action_content']) == 1
    assert result['actual_action_content'][0]['arguments']['operator'] == 'research_route'
    assert result['native_representation_returns'][0]['comparison_completed']
    assert len(agent.candidate_paths) > 0
    assert agent.model is None  # Shared pool comparison and adoption still belong to the caller.


@pytest.mark.parametrize('failed', [False, True])
def test_partial_or_failed_representation_stops_attempt_without_claiming_delivery(tmp_path, monkeypatch, failed):
    agent, lab = agent_scene(tmp_path)
    calls = []
    def execute(action, arguments, laboratory):
        calls.append(True)
        feedback = receipt(complete=False, failed=failed)
        proposal = {'action': 'run', 'arguments': {'operator': 'research_route'}}
        agent.method_results.append({'proposal': proposal, 'feedback': feedback})
        return feedback
    monkeypatch.setattr(agent, '_execute', execute)
    result = runner.execute_registered_method_unit(agent, lab, 'method.py',
        max_steps=8, native_attempt_limit=1)
    assert len(calls) == 1
    assert result['registered_attempt_returned'] and not result['native_research_complete']
    assert result['status'] == ('method_error' if failed else 'native_representation_incomplete')
    assert result['native_representation_returns'][0]['failed'] is failed


def test_public_hook_representation_is_not_a_method_attempt(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    def before(laboratory):
        agent.operator_results.append(receipt())
    def execute(*args):
        feedback = {'status': 'author_work_request', 'request': {'reason': 'fixture'}}
        agent.method_results.append({'proposal': {'stop': feedback}, 'feedback': feedback})
        return feedback
    monkeypatch.setattr(agent, '_execute', execute)
    result = runner.execute_registered_method_unit(agent, lab, 'method.py',
        before_step=before, native_attempt_limit=1)
    assert not result['registered_attempt_returned']
    assert 'native_representation_returns' not in result
    assert result['actual_action_content'] == []


def test_partial_representation_cannot_be_rescued_by_a_complete_pool_comparison(tmp_path, monkeypatch):
    plan = registration()
    plan.update(execution_authorized=True, approval_ref={'file': 'fixture_authorization'})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 8, 'feature_bounds': {'u': [-1., 1.]}}
        agent.rows = [Observation({'u': 0.}, 1.)]
        agent.config = replace(agent.config, compute_budget=10000,
            observation_budget=0, execution_seconds=30.)
        agent._save('old.py', export_temporal_program(Program('const', value=0.), feature_names=('u',)))
        agent._open_question({'id': 'q', 'question': 'Fixture paid parent', 'parent_id': 'task'})
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'}, 'native_attempt_limit': 1,
            'profile_ref': {'file': 'profile.json'}, 'origin_ref': {'file': 'origin.json'},
            **{key: deepcopy(contract[key]) for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}
    def unit(*args, **kwargs):
        return {'status': 'native_representation_incomplete', 'registered_attempt_returned': True,
            'native_research_complete': False, 'actual_action_content': [],
            'native_representation_returns': [{'complete': False, 'status': 'comparison_incomplete'}]}
    monkeypatch.setattr(runner, 'execute_registered_method_unit', unit)
    def forbidden(*args):
        pytest.fail('incomplete method may not open or predict the evaluation panel')
    result = runner.run_method_action_pairing(plan, tmp_path / 'run', restore_cell=restore,
        prepare_evaluation=forbidden, evaluate_frozen=forbidden,
        ordinary_use=lambda *args: {'status': 'incomplete_pair_not_used'})
    assert result['evaluation_lifecycle']['status'] == 'blocked_incomplete_outputs'
    for name in ('00', '10'):
        outcome = json.loads((tmp_path / 'run' / name / 'outcome.json').read_text())
        assert outcome['comparison']['models_requested'] == outcome['comparison']['models_completed'] == 1
        assert not outcome['execution_complete']
        assert outcome['method_delivery'] == 'research_return_incomplete'
        assert not outcome['native_research_complete']
