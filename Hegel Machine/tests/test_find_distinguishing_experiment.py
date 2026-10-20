"""The query searches the existing finite selector without treating a plan as data."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.question_continuation import pending_query_arguments
from hegel_machine.cognition.question_routing import find_distinguishing_experiment, register_theories
from test_question_routing import (SQUARE, X, assert_no_new_measurements, future_question,
    parent, probe, route, scene, theory)


def search_question(agent):
    question = future_question(agent, 'distinguishing_query')
    target = question['query_semantics']['query_expression']
    question['kind'] = 'experiment_selection'
    question['answer_type'] = 'binding'
    question['query_semantics']['query_expression'] = {
        'op': 'FindDistinguishingExperiment', 'target_query': target,
        'claim_refs': [{'id': 'linear', 'version': 1}, {'id': 'quadratic', 'version': 1}],
        'legal_constructor_ref': {'kind': 'resolution_experiments'}}
    return question


def search(agent, lab, question=None, panel=None):
    return route(agent, lab, question=question or search_question(agent),
        theories=[theory('linear', X), theory('quadratic', SQUARE)],
        experiments=[probe(agent)] if panel is None else panel)


def test_query_projects_actual_target_answers_and_prepares_original_parent_without_purchase(tmp_path):
    agent, lab = scene(tmp_path)
    paid, actions = lab.paid, deepcopy(lab.actions)
    result = search(agent, lab)
    assert result['status'] == 'selected'
    assert [group['task_answer'] for group in result['selection']['groups']] == [3., 9.]
    answer = result['query_answer']
    assert answer['status'] == 'answered' and answer['value']['kind'] == 'finite_experiment_plan'
    assert answer['value']['actions'] == [probe(agent)['actions'][0]]
    assert answer['value']['query_refs'][0]['experiment_id'] == 'distinguishing_probe'
    assert answer['scope'] == 'finite_candidate_heuristic'
    assert answer['support_updated'] is False and answer['independent_confirmation'] is False
    assert answer['candidate_completeness'] == 'not_established'
    assert result['parent_use']['status'] == 'parent_query_prepared'
    assert pending_query_arguments(agent, 'task')['actions'] == answer['value']['actions']
    question = next(q for q in agent.questions if q['id'] == 'distinguishing_query')
    assert 'semantic_answer' not in question and 'executable_answer' not in question
    assert question['query_semantics']['query_expression']['op'] == 'FindDistinguishingExperiment'
    assert_no_new_measurements(agent, lab, paid, actions)


def test_public_entry_reuses_original_routing_cache_without_repeating_comparison(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    search(agent, lab)
    question = next(q for q in agent.questions if q['id'] == 'distinguishing_query')
    costs = deepcopy(agent.costs)

    def repeated_comparison(*args, **kwargs):
        raise AssertionError('the completed original route must be cached')

    monkeypatch.setattr('hegel_machine.cognition.question_routing.select_question_plan', repeated_comparison)
    result = find_distinguishing_experiment(agent, question, lab)
    assert result['cache_hit'] and result['new_compute_cost'] == 0
    assert result['query_answer']['status'] == 'answered' and agent.costs == costs
    assert 'routing_execution' in question


def test_common_research_step_dispatches_once_to_the_same_route_and_prepares_no_extra_parent_use(tmp_path):
    agent, lab = scene(tmp_path)
    register_theories(agent, [theory('linear', X), theory('quadratic', SQUARE)])
    question = search_question(agent)
    question['resolution_experiments'] = [probe(agent)]
    paid, actions = lab.paid, deepcopy(lab.actions)
    result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert result['operator'] == 'research_step' and result['query_answer']['status'] == 'answered'
    assert result['selection']['scores'][0]['G'] == 1.
    assert result['parent_use']['status'] == 'parent_query_prepared'
    retained = next(q for q in agent.questions if q['id'] == question['id'])
    assert 'parent_use_history' not in retained and 'executable_answer' not in retained
    assert 'semantic_answer' not in retained
    costs = deepcopy(agent.costs)
    again = agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': question['id']}}, lab)
    assert again['cache_hit'] and again['new_compute_cost'] == 0 and agent.costs == costs
    assert pending_query_arguments(agent, 'task')['actions'] == question['resolution_experiments'][0]['actions']
    assert_no_new_measurements(agent, lab, paid, actions)


def test_equal_legal_experiments_keep_ties_and_do_not_prepare_an_arbitrary_action(tmp_path):
    agent, lab = scene(tmp_path)
    result = search(agent, lab, panel=[probe(agent, 2., 'two'), probe(agent, 4., 'four')])
    assert result['status'] == result['query_answer']['status'] == 'tied'
    assert result['query_answer']['tied_experiments'] == ['two', 'four']
    assert 'value' not in result['query_answer'] and result['selection']['selected_experiment'] is None
    assert 'pending_query' not in parent(agent)


def test_one_current_candidate_does_not_prove_impossibility_of_distinguishing(tmp_path):
    agent, lab = scene(tmp_path)
    question = search_question(agent)
    question['query_semantics']['query_expression']['claim_refs'] = [{'id': 'linear', 'version': 1}]
    result = search(agent, lab, question)
    assert result['query_answer']['status'] == 'scoped_unknown'
    assert result['query_answer']['reason'] == 'no_task_answer_disagreement'
    assert result['query_answer']['candidate_completeness'] == 'not_established'
    assert 'value' not in result['query_answer'] and 'pending_query' not in parent(agent)


@pytest.mark.parametrize('missing', ['target_query', 'panel', 'constructor'])
def test_missing_target_or_public_constructor_returns_author_work_without_new_probe(tmp_path, missing):
    agent, lab = scene(tmp_path)
    question = search_question(agent)
    query = question['query_semantics']['query_expression']
    if missing == 'target_query':
        query.pop('target_query')
    elif missing == 'constructor':
        query['legal_constructor_ref'] = {'kind': 'invent_new_probes'}
    before = deepcopy(agent.costs)
    paid, actions = lab.paid, deepcopy(lab.actions)
    result = search(agent, lab, question, [] if missing == 'panel' else None)
    assert result['status'] == 'author_work_request'
    assert result['request']['object_type'] == 'distinguishing_experiment'
    assert 'value' not in result['query_answer'] and agent.costs == before
    assert_no_new_measurements(agent, lab, paid, actions)


def test_claim_version_update_rejects_old_query_and_invalidates_its_pending_plan(tmp_path):
    agent, lab = scene(tmp_path)
    search(agent, lab)
    result = route(agent, lab, question={'id': 'distinguishing_query'},
                   theories=[theory('linear', X, declared_scope_ref='revised_source_scope')])
    assert result['query_answer']['status'] == 'scoped_unknown'
    assert result['reason'] == 'requested_claim_version_unavailable'
    with pytest.raises(ValueError, match='current research selection'):
        pending_query_arguments(agent, 'task')


def test_old_public_row_validation_and_compute_exhaustion_leave_search_unknown(tmp_path):
    agent, lab = scene(tmp_path)
    invalid = probe(agent)
    invalid['query']['binding']['features']['u'] = 99.
    result = search(agent, lab, panel=[invalid])
    assert result['selection']['excluded'][0]['reason'] == 'public_row_query_mismatch'
    assert result['query_answer']['status'] == 'scoped_unknown'
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'])
    result = search(agent, lab)
    assert result['query_answer']['status'] == 'scoped_unknown'
    assert 'value' not in result['query_answer']
    assert 'pending_query' not in parent(agent)
