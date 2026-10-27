"""Mainline production-path receipt controls; original isolated receipt is retained."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import theory_consolidation
from test_common_method_feedback import scene, transfer_arguments


@pytest.fixture
def isolated():
    # The accepted assertions now address the actual patched Common module.
    return theory_consolidation


def decision(method, admission):
    return {'method_ref': {'id': method['id'], 'version': method['version']},
        'admission': admission,
        'next_action': 'try_or_continue_method' if admission == 'admit' else 'distinguish_method_use',
        'status': 'receipt_control', 'feedback_refs': [{'file': 'paid_return.json', 'json_path': []}]}


@pytest.mark.parametrize('admission', ['admit', 'bounded_exploration'])
def test_ordinary_success_retains_this_calls_decision(tmp_path, monkeypatch, isolated, admission):
    agent, lab, method, parent = scene(tmp_path, useful=True)
    actual = decision(method, admission)
    monkeypatch.setattr('hegel_machine.cognition.method_feedback.procedure_use_decision',
                        lambda *args: actual)
    result = agent._execute('run', transfer_arguments(method, parent, 'receipt_fit'), lab)
    assert result['generated_theories'] and result['new_compute_cost'] > 0
    assert result['method_use_decision'] == actual
    assert agent._active_method_use['decision'] == actual
    actual['admission'] = 'changed_after_return'
    assert result['method_use_decision']['admission'] == admission
    assert lab.paid == 0 and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('admission', ['admit', 'bounded_exploration'])
def test_ordinary_recall_work_retains_this_calls_decision(tmp_path, monkeypatch, isolated, admission):
    agent, lab, method, parent = scene(tmp_path, useful=True)
    actual = decision(method, admission)
    monkeypatch.setattr('hegel_machine.cognition.method_feedback.procedure_use_decision',
                        lambda *args: actual)
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': parent['id']}, 'allocation': {'compute': 3000}}, lab)
    work = next(item for item in result['frozen_recall']['work']
                if item['candidate_id'] == method['id'])
    assert work['theory_ids'] and work['new_compute_cost'] > 0
    assert work['method_use_decision'] == actual
    actual['admission'] = 'changed_after_return'
    assert work['method_use_decision']['admission'] == admission


def test_paid_cache_uses_current_decision_and_none_never_inherits_prior(tmp_path, monkeypatch, isolated):
    agent, lab, method, parent = scene(tmp_path, useful=True)
    current = [decision(method, 'admit')]
    monkeypatch.setattr('hegel_machine.cognition.method_feedback.procedure_use_decision',
                        lambda *args: current[0])
    arguments = transfer_arguments(method, parent, 'same_paid_receipt')
    first = agent._execute('run', deepcopy(arguments), lab)
    assert first['method_use_decision']['admission'] == 'admit'
    costs = deepcopy(agent.costs)
    current[0] = None
    second = agent._execute('run', deepcopy(arguments), lab)
    assert second['cache_hit'] and second['new_compute_cost'] == 0
    assert second['method_use_decision'] is None
    assert agent._active_method_use['decision'] == {}
    current[0] = decision(method, 'bounded_exploration')
    third = agent._execute('run', deepcopy(arguments), lab)
    assert third['cache_hit'] and third['new_compute_cost'] == 0
    assert third['method_use_decision']['admission'] == 'bounded_exploration'
    assert agent.costs == costs
    # A rejected contract has no current decision; an earlier active call is
    # insufficient to label this non-executed request.
    rejected = isolated._transfer_research_program(agent,
        {key: value for key, value in arguments.items() if key != 'research_program_fitting'},
        method, 'shared/theory_library.json')
    assert rejected['status'] == 'author_work_request'
    assert 'method_use_decision' not in rejected
