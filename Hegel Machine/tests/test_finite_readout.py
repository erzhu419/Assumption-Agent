"""Finite trajectory answers use the scalar reader and its real budget."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.question_queries import evaluate_on_evidence, evaluate_on_model
from hegel_machine.cognition.question_semantics import validate_answer_type
from test_question_queries import readout, records
from test_question_routing import (CONTEXT, SQUARE, X, future_question, probe,
                                   readout as public_readout, route, scene, theory)


def finite(cases, kind='vector', **fields):
    return {'op': 'Readout', 'finite_readouts': cases, 'statistic_ref': {'kind': kind}, **fields}


@pytest.mark.parametrize('kind, value', [('vector', [5., 8.]), ('mean', 6.5),
                                        ('empirical_distribution', {'5.0': .5, '8.0': .5})])
def test_finite_statistics_read_only_exact_bound_rows(kind, value):
    observations = records()
    query = finite([readout(observations, index) for index in (0, 1)], kind, units='V')
    before = deepcopy((observations, query))
    answer = evaluate_on_evidence(query, observations)
    assert answer['value'] == value and answer['evidence'] == [40, 41]
    assert answer['compute_cost'] == 2 and answer['scope']['case_count'] == 2
    assert answer['units'] == (None if kind == 'empirical_distribution' else 'V')
    if kind == 'empirical_distribution':
        assert answer['support_units'] == 'V' and validate_answer_type(value, 'distribution')
    assert (observations, query) == before


def test_finite_prediction_never_reads_targets_and_truncation_has_no_partial_answer():
    observations = records()
    query = finite([readout(observations, index) for index in (0, 1)])
    received = []

    def predictor(public):
        assert set(public) == {'features', 'history'}
        received.append(public)
        return public['features']['u']

    answer = evaluate_on_model(query, predictor, compute_budget=1)
    assert answer['status'] == 'scoped_unknown' and 'value' not in answer
    assert answer['compute_cost'] == len(received) == 1 and answer['evidence'] == []
    assert answer['scope']['answered_case_count'] == 1 and answer['failed_case_index'] == 1
    assert evaluate_on_model(query, predictor)['value'] == [1., 2.]
    del observations[1]['rows'][0]['target']
    unknown = evaluate_on_evidence(query, observations)
    assert unknown['status'] == 'scoped_unknown' and unknown['evidence'] == [40]
    assert unknown['compute_cost'] == 1 and 'value' not in unknown


def test_finite_units_and_vector_shapes_are_executable_contracts():
    observations = records()
    cases = [readout(observations, index, units='V') for index in (0, 1)]
    left = finite(cases)
    right = finite(list(reversed(cases)))
    difference = {'op': 'Difference', 'left': left, 'right': right}
    assert evaluate_on_evidence(difference, observations)['value'] == [-3., 3.]
    relation = {'op': 'RelationResidual', 'query': difference, 'expected': [-3., 3.], 'atol': 0.}
    assert evaluate_on_evidence(relation, observations)['check_status'] == 'satisfied'
    relation['expected'] = [-3., 2.]
    assert evaluate_on_evidence(relation, observations)['check_status'] == 'violated'
    right['finite_readouts'].pop()
    assert evaluate_on_evidence(difference, observations)['reason'] == 'numeric_answers_with_matching_shape_required'
    cases[1]['units'] = 'mV'
    assert evaluate_on_evidence(left, observations)['reason'] == 'unit_mismatch'
    assert not validate_answer_type([1., True], 'vector')


def test_units_recorded_only_on_evidence_cannot_be_lost_when_averaging():
    observations = records()
    observations[0]['units'], observations[1]['units'] = 'V', 'A'
    query = finite([readout(observations, index) for index in (0, 1)], 'mean')
    result = evaluate_on_evidence(query, observations)
    assert result['status'] == 'invalid_experiment_binding' and result['reason'] == 'unit_mismatch'
    assert 'value' not in result
    observations[1]['units'] = 'V'
    result = evaluate_on_evidence(query, observations)
    assert result['value'] == 6.5 and result['units'] == 'V'


def test_vector_answers_reach_Common_selection_and_semantic_answer(tmp_path):
    agent, lab = scene(tmp_path)
    question = future_question(agent)
    question.update(answer_type='vector', use_sites=[])
    question['mission'] = {'purpose': 'theory_learning', 'mission_id': 'finite_readout',
                           'theory_learning_allocation': {'compute': 5000}}
    question['query_semantics']['query_expression'] = finite([public_readout(2., 1), public_readout(3., 2)])
    plan = probe(agent)
    plan['query'] = finite([public_readout(2., 1)])
    selection = route(agent, lab, question=question, theories=[theory('linear', X), theory('square', SQUARE)],
                      experiments=[plan])
    assert selection['status'] == 'selected'
    assert [group['task_answer'] for group in selection['selection']['groups']] == [[2., 3.], [4., 9.]]
    assert route(agent, lab, question={'id': question['id']})['cache_hit']
    # The finite paid question below reads the existing two-row observation,
    # without pretending the prospective rows above have already been paid.
    record = agent.observations[0]
    paid = finite([{'op': 'Readout', 'record_index': 0, 'step': index, 'units': 'V',
        'binding': {'action': deepcopy(record['action']), 'features': deepcopy(row['features']),
                    'history': list(deepcopy(row['history']))}}
        for index, row in enumerate(record['rows'])])
    incoming = {**deepcopy(question), 'id': 'finite_paid',
                'query_semantics': {'mode': 'model_query', 'query_expression': paid}}
    result = agent._execute('run', {'operator': 'research_step', 'research_question': incoming}, lab)
    assert result['status'] == 'answer_ready_no_parent_use'
    assert result['answer']['value'] == [row['target'] for row in record['rows']]
    assert result['answer']['units'] == 'V' and result['answer']['statistic_ref'] == {'kind': 'vector'}
    assert result['answer']['scope']['case_count'] == 2
    assert agent.costs['model_calls'] == 0 and lab.paid == 2


@pytest.mark.parametrize('distribution_location', ['panel', 'task'])
def test_distribution_requires_a_defined_test_and_does_not_force_selection(tmp_path, distribution_location):
    agent, lab = scene(tmp_path)
    plan = probe(agent)
    question = future_question(agent)
    if distribution_location == 'panel':
        plan['query'] = finite([deepcopy(plan['query'])], 'empirical_distribution')
    else:
        question['answer_type'] = 'distribution'
        question['query_semantics']['query_expression'] = finite(
            [public_readout(2., 1), public_readout(3., 2)], 'empirical_distribution')
    result = route(agent, lab, question=question,
                   theories=[theory('linear', X), theory('square', SQUARE)], experiments=[plan])
    assert result['status'] == 'scoped_unknown'
    assert result['selection']['unknown_reason'] == 'incomplete_candidate_predictions'
    assert all(group['unknown_reason'] == 'undefined_distribution_or_answer_test'
               for group in result['selection']['groups'])
    assert result['selection']['selected_experiment'] is None


def test_vector_panel_public_row_mismatch_is_checked_before_scoring(tmp_path):
    agent, lab = scene(tmp_path)
    plan = probe(agent)
    plan['query'] = finite([deepcopy(plan['query'])])
    plan['query']['finite_readouts'][0]['binding']['features']['u'] = 99.
    result = route(agent, lab, question=future_question(agent),
                   theories=[theory('linear', X), theory('square', SQUARE)], experiments=[plan])
    assert result['selection']['excluded'][0]['reason'] == 'public_row_query_mismatch'
    assert result['selection']['compute_cost'] == 0
