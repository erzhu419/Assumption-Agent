"""Parameter queries consume the existing checked region, not a fitted point."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.question_object_queries import evaluate_parameter_region
from test_evidence_procedure_execution import prepared
from test_executable_evidence_binding import return_region


def region_query(agent, *, question_id='slope_question'):
    source = next(item for item in agent.questions if item['id'] == question_id)
    return {'op': 'ParameterRegion', 'source_ref': {'question_id': question_id,
        'meaning_version': source.get('meaning_version', 1),
        'answer_version': source['executable_answer']['version']},
        'scope': deepcopy(source['executable_answer']['scope'])}


def request(agent):
    return {'id': 'queried_slope_region', 'question': 'What is the retained checked working region?',
        'parent_id': 'task', 'depends_on': ['slope_question'], 'kind': 'parameter_region',
        'answer_type': 'interval', 'answer_resolution_ref': {'kind': 'retained_checked_working_region'},
        'query_semantics': {'mode': 'model_query', 'query_expression': region_query(agent)},
        'use_sites': [{'kind': 'bind_parameter_region', 'model': 'parent.py', 'slot': 'slope',
            'refit': {'validation_records': [2], 'path': 'shared/parameter_query_refit'}}]}


def run(agent, lab, question):
    return agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)


def test_parameter_query_returns_original_checked_scope_and_open_region_without_new_support(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    query = region_query(agent)
    before = deepcopy((agent.questions, agent.costs))
    result = evaluate_parameter_region(query, agent.questions)
    assert result['status'] == 'answered'
    assert result['value'] == {'lower': .9, 'upper': 1.1, 'lower_closed': False, 'upper_closed': False}
    assert result['scope'] == query['scope']
    assert result['answer_record']['check_results'][0]['status'] == 'satisfied'
    assert result['evidence'] == [0, 1] and result['compute_cost'] == 0
    assert result['completeness'] == 'not_established' and result['independent_confirmation'] is False
    assert (agent.questions, agent.costs) == before


@pytest.mark.parametrize('changed', ['version', 'meaning', 'scope', 'family', 'missing_scope'])
def test_parameter_query_needs_its_actual_version_and_declared_scope(tmp_path, changed):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    query = region_query(agent)
    if changed == 'version':
        query['source_ref']['answer_version'] += 1
    elif changed == 'meaning':
        query['source_ref']['meaning_version'] += 1
    elif changed == 'scope':
        query['scope'] = {'controls': 'another apparatus'}
    elif changed == 'family':
        query['family_ref'] = 'undeclared_family'
    else:
        query.pop('scope')
    result = evaluate_parameter_region(query, agent.questions)
    assert result['status'] in {'scoped_unknown', 'invalid_experiment_binding'}
    assert 'value' not in result and result['compute_cost'] == 0


def test_parameter_query_is_not_answered_by_a_fitted_point_or_unchecked_prose(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1, checked=False)
    result = evaluate_parameter_region(region_query(agent), agent.questions)
    assert result['status'] == 'scoped_unknown'
    source = next(item for item in agent.questions if item['id'] == 'slope_question')
    source['executable_answer'] = {'status': 'supported', 'evidence_status': 'supported', 'version': 1,
        'parameters': {'slope': 1.}, 'scope': {'controls': 'sampled background'}, 'evidence': [0, 1],
        'check_results': [{'status': 'satisfied'}]}
    result = evaluate_parameter_region({'op': 'ParameterRegion',
        'source_ref': {'question_id': source['id'], 'meaning_version': 1, 'answer_version': 1},
        'scope': deepcopy(source['executable_answer']['scope'])}, agent.questions)
    assert result['status'] == 'scoped_unknown' and result['detail'] == 'source_has_no_checked_parameter_region'


def test_parameter_query_changes_real_common_parent_fit_and_reuses_the_same_answer(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, 0., .5)
    before = deepcopy(agent.costs)
    first = run(agent, lab, request(agent))
    assert first['parent_revision_completed']
    fitted = first['uses'][0]['feedback']
    assert fitted['validation']['mse'] > 0
    assert fitted['validation']['research_bindings'][0]['actual_parameters']['slope'] == .25
    after_first = deepcopy(agent.costs)
    assert run(agent, lab, {'id': 'queried_slope_region'})['status'] == 'reused_answer'
    assert agent.costs == after_first
    return_region(agent, lab, .9, 1.1)
    second = run(agent, lab, request(agent))
    fitted = second['uses'][0]['feedback']
    assert fitted['validation']['mse'] == 0
    assert fitted['validation']['research_bindings'][0]['actual_parameters']['slope'] == 1.
    query_question = next(item for item in agent.questions if item['id'] == 'queried_slope_region')
    assert query_question['meaning_version'] == 2
    assert query_question['meaning_history'][0]['semantic_answer']['value']['upper'] == .5
    assert agent.costs['observation_cost'] == before['observation_cost'] == lab.paid
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_refuted_region_source_invalidates_derived_binding_without_reusing_old_semantic_answer(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    assert run(agent, lab, request(agent))['parent_revision_completed']
    source = next(item for item in agent.questions if item['id'] == 'slope_question')
    # The retained semantic answer must not mask the current executed failure.
    source['semantic_answer'] = {'status': 'answered', 'version': 1, 'value': {'lower': .9, 'upper': 1.1}}
    failed = {'status': 'supported', 'answer': 'Check the retained region against the paid response',
        'parameter_region': {'lower': .9, 'upper': 1.1},
        'scope': deepcopy(source['executable_answer']['scope']), 'evidence': [1],
        'checks': [{'id': 'zero', 'terms': [{'row': 1, 'coefficient': 1}],
            'relation': 'eq', 'expected': 0., 'scope': {'input': 1}}]}
    agent._execute('write', {'path': 'refuted_region.py',
        'code': f'def readout(result,spec,memory): return {failed!r}\n'}, lab)
    refuted = agent._execute('run', {'operator': 'analyze_evidence', 'path': 'refuted_region.py',
        'research_question': {'id': 'slope_question'}}, lab)
    assert refuted['revision']['evidence_status'] == 'refuted'
    assert refuted['revision']['check_results'][0]['status'] == 'violated'
    assert source['executable_answer']['status'] == 'refuted'
    before = deepcopy(agent.costs)
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved'
    result = run(agent, lab, {'id': 'queried_slope_region'})
    assert result['status'] == 'scoped_unknown'
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved'
    assert agent.costs == before and not agent.confirmations
