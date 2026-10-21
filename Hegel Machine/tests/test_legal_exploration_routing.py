"""Operational choices use the existing frozen query path; fixtures are not science."""

from copy import deepcopy
import json
import sys

import pytest

from hegel_machine.cognition.question_continuation import pending_query_arguments
from scripts import run_anonymous_revision_inquiry as runner
from test_formal_three_arm_study import automatic, registration, study
from test_question_routing import X, SQUARE, future_question, parent, probe, scene, theory


POLICY = {'kind': 'seeded_legal', 'seed': 19}


def test_tied_route_prepares_and_executes_original_frozen_revision(tmp_path):
    agent, lab = scene(tmp_path)
    for name, code in [('linear', "return row['features']['u']"),
                       ('square', "return row['features']['u'] ** 2")]:
        agent._execute('write', {'path': f'{name}.py', 'code': f'def predict(row): {code}\n'}, lab)
    panel = [probe(agent, 2., 'two'), probe(agent, 4., 'four')]
    for plan in panel:
        plan['revision'] = {'id': 'operational_fixture', 'affected_object': 'program',
            'models': ['linear.py', 'square.py'], 'parent_and_scope': {'parent': 'task', 'scope': 'fixture'}}
    arguments = {'operator': 'research_route', 'research_question': future_question(agent),
        'theories': [theory('linear', X), theory('square', SQUARE)], 'experiments': panel}
    legacy = agent._execute('run', deepcopy(arguments), lab)
    assert legacy['status'] == 'tied' and 'pending_query' not in parent(agent)
    before = (lab.paid, len(lab.actions))
    arguments['exploration_policy'] = deepcopy(POLICY)
    result = agent._execute('run', arguments, lab)
    selection = result['selection']
    assert selection['status'] == 'tied'
    assert selection['tied_experiments'] == ['two', 'four']
    chosen = selection['selected_experiment']
    assert chosen and result['parent_use']['status'] == 'parent_query_prepared'
    assert (lab.paid, len(lab.actions)) == before
    assert result['cache_hit'] is False
    repeated = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': 'future_response'}}, lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert repeated['selection']['selected_experiment']['id'] == chosen['id']
    arguments = pending_query_arguments(agent, 'task')
    feedback = agent._execute('query', {'pending_question': 'task'}, lab)
    revision = feedback['revision']
    level = arguments['actions'][0]['sequence'][0]['u']
    assert revision['evidence_cutoff'] == {'rows': 2, 'records': 1}
    assert revision['forecasts'][0]['predictions'] == [level]
    assert revision['forecasts'][1]['predictions'] == [level ** 2]
    assert revision['forecasts'][0]['prospective_mse'] == 0
    assert lab.paid == before[0] + 1 and agent.costs['model_calls'] == 0
    assert not selection['support_updated'] and not agent.confirmations


def test_policy_change_invalidates_pending_execution_without_changing_question_meaning(tmp_path):
    agent, lab = scene(tmp_path)
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': future_question(agent), 'exploration_policy': POLICY,
        'theories': [theory('linear', X), theory('square', SQUARE)],
        'experiments': [probe(agent, 2., 'two'), probe(agent, 4., 'four')]}, lab)
    question = next(q for q in agent.questions if q['id'] == 'future_response')
    meaning = question['meaning_version']
    question['exploration_policy'] = {**POLICY, 'seed': 20}
    with pytest.raises(ValueError, match='rebuilding'):
        pending_query_arguments(agent, 'task')
    updated = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}}, lab)
    assert not updated['cache_hit'] and question['meaning_version'] == meaning
    assert question['routing_execution']['version'] > 1
    assert result['selection']['status'] == updated['selection']['status'] == 'tied'


def test_registered_groups_share_operational_policy_on_actual_automatic_route(tmp_path):
    registered = registration()
    registered['exploration_policy'] = deepcopy(POLICY)

    def tied_contract(question):
        original = question['resolution_experiments'][0]
        other = deepcopy(original)
        other['id'] = 'same_physical_action_alias'
        question['resolution_experiments'].append(other)

    agents, _, result, plan, _ = study(tmp_path, registered=registered, author_contract=tied_contract)
    for group in ('fixed_pr2', 'pr3'):
        event, = automatic(agents[group], 'research_route')
        assert event['arguments']['exploration_policy'] == POLICY
        selection = event['feedback']['selection']
        assert selection['tied_experiments'] == ['distinguishing_probe', 'same_physical_action_alias']
        assert selection['selected_experiment']
        assert next(q for q in agents[group].questions if q['id'] == 'task')['pending_query']
    assert plan['research_organization']['exploration_policy'] == POLICY
    assert all(row['costs']['model_calls'] == 0 for row in result['trials'])


def test_cli_requires_explicit_registered_arms_before_any_backend_check(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError('invalid CLI must not consult model backend')

    monkeypatch.setattr(runner, 'verify_codex_subscription', forbidden)
    for extra in ([], ['--arms', 'archive', 'fixed_pr2', 'pr3'], ['--arms', 'pr3']):
        monkeypatch.setattr(sys, 'argv', ['run_anonymous_revision_inquiry.py', '--output', str(tmp_path), *extra])
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 2


def test_cli_dispatches_registered_plan_instead_of_legacy_defaults(tmp_path, monkeypatch):
    document = tmp_path / 'registration.json'
    document.write_text(json.dumps({**registration(), 'exploration_policy': POLICY}))
    captured = []
    monkeypatch.setattr(runner, 'verify_codex_subscription', lambda: 'fixture_no_backend')
    monkeypatch.setattr(runner, 'run_study', lambda plan, output, **kwargs: captured.append(plan) or {'trials': []})
    monkeypatch.setattr(sys, 'argv', ['run_anonymous_revision_inquiry.py', '--output', str(tmp_path / 'run'),
        '--arms', *runner.ORGANIZATION_ARMS, '--research-organization', str(document)])
    runner.main()
    assert captured[0]['arms'] == list(runner.ORGANIZATION_ARMS)
    assert captured[0]['research_organization']['exploration_policy'] == POLICY
    assert captured[0]['study'] == 'anonymous_registered_question_theory_research'
