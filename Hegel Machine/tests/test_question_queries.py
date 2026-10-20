"""Question answers preserve physical bindings and separate predictions from evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.question_queries import evaluate_on_evidence, evaluate_on_model


def records():
    return [{'action': {'reset': True, 'inputs': [{'u': level}], 'history': []},
             'rows': [{'features': {'u': level}, 'history': [], 'target': target}],
             'row_indices': [index]}
            for level, target, index in ((1., 5., 40), (2., 8., 41))]


def readout(observations, index=0, **fields):
    record = observations[index]
    row = record['rows'][0]
    return {'op': 'Readout', 'record_index': index, 'step': 0,
            'target': 'response', 'binding': {'action': deepcopy(record['action']),
                'features': deepcopy(row['features']), 'history': deepcopy(row['history'])}, **fields}


def difference(observations):
    return {'op': 'Difference', 'left': readout(observations, 1), 'right': readout(observations)}


def test_measured_difference_has_actual_values_rows_and_cost_without_mutation():
    observations = records()
    query = difference(observations)
    before = deepcopy((observations, query))
    result = evaluate_on_evidence(query, observations)
    assert result == {'status': 'answered', 'value': 3., 'units': None,
                      'evidence': [40, 41], 'compute_cost': 2}
    assert (observations, query) == before


def test_model_receives_only_public_rows_and_never_current_targets():
    observations = records()
    received = []

    def predictor(row):
        assert set(row) == {'features', 'history'}
        received.append(deepcopy(row))
        row['features']['mutated_by_model'] = True
        return 10. * row['features']['u']

    query = difference(observations)
    result = evaluate_on_model(query, predictor)
    assert result['value'] == 10. and result['evidence'] == [] and result['compute_cost'] == 2
    assert received == [{'features': {'u': 2.}, 'history': []}, {'features': {'u': 1.}, 'history': []}]
    assert 'mutated_by_model' not in query['left']['binding']['features']
    assert evaluate_on_evidence(query, observations)['value'] == 3.


@pytest.mark.parametrize('field', ['action', 'reset', 'inputs', 'action_history', 'features', 'history', 'target'])
def test_T16_legal_rows_cannot_support_a_different_experiment_binding(field):
    observations = records()
    query = readout(observations)
    if field == 'action':
        query['binding']['action'] = None
    elif field == 'reset':
        query['binding']['action']['reset'] = False
    elif field == 'inputs':
        query['binding']['action']['inputs'][0]['u'] = 7.
    elif field == 'action_history':
        query['binding']['action']['history'] = [{'u': 1.}]
    elif field == 'features':
        query['binding']['features']['u'] = 7.
    elif field == 'history':
        query['binding']['history'] = [{'u': 0.}]
    else:
        query['target'] = 'another_measurement'
    result = evaluate_on_evidence(query, observations)
    assert result['status'] == 'invalid_experiment_binding'
    assert result['evidence'] == [] and result['compute_cost'] == 0


@pytest.mark.parametrize('missing', ['record', 'step', 'value', 'row_id'])
def test_missing_evidence_is_unknown_not_a_predicted_answer(missing):
    observations = records()
    query = readout(observations)
    if missing == 'record':
        query['record_index'] = 9
    elif missing == 'step':
        query['step'] = 1
    elif missing == 'value':
        del observations[0]['rows'][0]['target']
    else:
        observations[0]['row_indices'] = []
    result = evaluate_on_evidence(query, observations)
    assert result['status'] == 'scoped_unknown'
    assert result['reason'] == 'unknown_due_to_missing_evidence'
    assert 'value' not in result and result['compute_cost'] == 0


def test_T17_nested_budget_is_shared_and_counts_only_actual_readouts():
    observations = records()
    query = difference(observations)
    result = evaluate_on_evidence(query, observations, compute_budget=1)
    assert result == {'status': 'scoped_unknown', 'reason': 'unknown_due_to_incomplete_search',
                      'detail': 'compute_budget', 'evidence': [41], 'compute_cost': 1}
    calls = []
    result = evaluate_on_model(query, lambda row: calls.append(row) or 0., compute_budget=0)
    assert result['status'] == 'scoped_unknown' and result['compute_cost'] == 0 and calls == []
    assert evaluate_on_evidence(query, observations, compute_budget=2)['value'] == 3.


def test_relation_residual_preserves_measured_failure_without_refuting_a_family():
    observations = records()
    query = {'op': 'RelationResidual', 'query': difference(observations),
             'expected': 0., 'relation': 'eq', 'atol': .01, 'premises': ['reset_protocol']}
    unknown = evaluate_on_evidence(query, observations)
    assert unknown['status'] == 'scoped_unknown' and unknown['reason'] == 'premise_unknown'
    assert unknown['compute_cost'] == 0
    result = evaluate_on_evidence(query, observations, context={'premises': {'reset_protocol': True}})
    assert result['status'] == 'answered' and result['value'] == 3.
    assert result['check_status'] == 'violated' and result['evidence'] == [40, 41]
    query['expected'] = 3.
    assert evaluate_on_evidence(query, observations,
        context={'premises': {'reset_protocol': True}})['check_status'] == 'satisfied'


def test_application_checks_mapping_and_premises_before_reading_evidence():
    observations = records()
    query = {'op': 'CheckApplication', 'query': readout(observations),
             'context_ref': {'scene': 'C5'}, 'mapping_ref': 'declared_mapping',
             'premises': [{'ref': 'calibrated'}]}
    context = {'context_ref': {'scene': 'C5'}, 'premises': {'calibrated': True}}
    result = evaluate_on_evidence(query, observations, context=context)
    assert result['status'] == 'scoped_unknown' and result['reason'] == 'unknown_due_to_uncertain_mapping'
    assert result['compute_cost'] == 0
    context['mappings'] = {'declared_mapping': True}
    assert evaluate_on_evidence(query, observations, context=context)['value'] == 5.
    context['premises']['calibrated'] = False
    result = evaluate_on_evidence(query, observations, context=context)
    assert result['status'] == 'invalid_experiment_binding' and result['reason'] == 'premise_not_satisfied'
    assert result['compute_cost'] == 0


def test_T04_context_units_and_retained_record_identity_must_match():
    observations = records()
    query = readout(observations, units='V', context_ref={'scene': 'C5', 'output_unit': 'V'})
    context = deepcopy(query['context_ref'])
    assert evaluate_on_evidence(query, observations, context=context)['value'] == 5.
    assert evaluate_on_evidence(query, observations,
        context={'scene': 'C4', 'output_unit': 'V'})['reason'] == 'context_mismatch'
    observations[0]['context_ref'] = {'scene': 'C4', 'output_unit': 'V'}
    assert evaluate_on_evidence(query, observations, context=context)['reason'] == 'record_context_mismatch'
    del observations[0]['context_ref']
    observations[0]['units'] = 'mV'
    assert evaluate_on_evidence(query, observations, context=context)['reason'] == 'unit_mismatch'
    query = difference(observations)
    del observations[0]['units']
    query['left']['units'], query['right']['units'] = 'V', 'mV'
    result = evaluate_on_evidence(query, observations)
    assert result['status'] == 'invalid_experiment_binding' and result['reason'] == 'unit_mismatch'


def test_embedded_query_expression_uses_its_explicit_context():
    observations = records()
    context = {'scene': 'C5'}
    question = {'context_ref': context, 'query_semantics': {'mode': 'model_query',
        'query_expression': readout(observations, context_ref=context)}}
    assert evaluate_on_evidence(question, observations)['value'] == 5.


def test_unimplemented_constructor_is_unknown_instead_of_an_empty_proof():
    result = evaluate_on_evidence({'op': 'FindSharedStructure'}, records())
    assert result == {'status': 'scoped_unknown', 'reason': 'unknown_due_to_opaque_program_semantics',
                      'op': 'FindSharedStructure', 'evidence': [], 'compute_cost': 0}


def finite_checks(observations):
    return {'op': 'AllOnFiniteSet', 'finite_cases': [
        {'op': 'RelationResidual', 'query': readout(observations, index),
         'expected': target, 'atol': 0.}
        for index, target in enumerate((5., 8.))]}


def test_finite_predicates_separate_measured_truth_from_model_forecasts():
    observations = records()
    query = finite_checks(observations)
    measured = evaluate_on_evidence(query, observations)
    assert measured['value'] is True and measured['finite_status'] == 'observed_true'
    assert measured['scope'] == {'kind': 'finite_cases', 'case_count': 2,
                                 'answered_case_count': 2, 'unresolved_case_count': 0}
    assert measured['evidence'] == [40, 41] and measured['compute_cost'] == 2
    predicted = evaluate_on_model(query, lambda row: 3. * row['features']['u'] + 2.)
    assert predicted['value'] is True and predicted['finite_status'] == 'predicted_true'
    assert predicted['evidence'] == []
    observations[1]['rows'][0]['target'] = 9.
    measured = evaluate_on_evidence(query, observations)
    assert measured['value'] is False and measured['counterexample_case_indices'] == [1]
    assert evaluate_on_model(query, lambda row: 3. * row['features']['u'] + 2.)['value'] is True


def test_finite_unknowns_and_budget_truncation_cannot_certify_all_cases():
    observations = records()
    query = finite_checks(observations)
    limited = evaluate_on_evidence(query, observations, compute_budget=1)
    assert limited['status'] == 'scoped_unknown' and 'value' not in limited
    assert limited['reason'] == 'unknown_due_to_incomplete_search'
    assert limited['compute_cost'] == 1 and limited['scope']['unresolved_case_count'] == 1
    query['finite_cases'][1]['query']['record_index'] = 999
    missing = evaluate_on_evidence(query, observations)
    assert missing['reason'] == 'unknown_due_to_missing_evidence' and 'value' not in missing
    # A genuine counterexample already disproves the finite conjunction even
    # though the other case remains unavailable; that uncertainty stays saved.
    query['finite_cases'][0]['expected'] = -1.
    false = evaluate_on_evidence(query, observations)
    assert false['value'] is False and false['scope']['unresolved_case_count'] == 1
    assert false['evidence'] == [40]
    assert evaluate_on_evidence({'op': 'AllOnFiniteSet', 'finite_cases': []}, observations)['status'] == 'scoped_unknown'


def test_finite_checks_preserve_binding_errors_and_require_predicates():
    observations = records()
    query = finite_checks(observations)
    query['finite_cases'][1]['query']['binding']['action']['reset'] = False
    invalid = evaluate_on_evidence(query, observations)
    assert invalid['status'] == 'invalid_experiment_binding'
    assert invalid['reason'] == 'action_mismatch'
    invalid = evaluate_on_evidence({'op': 'AllOnFiniteSet',
                                    'finite_cases': [readout(observations)]}, observations)
    assert invalid['reason'] == 'finite_case_requires_a_predicate'
