"""Use already computed predictions to expose local empirical model errors."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from hegel_machine.cognition.executable_evidence import compare_paid_relations
from hegel_machine.cognition.models import Observation


def checked_relation(rows, identifier='difference', terms=((1, 1.), (0, -1.)), **extra):
    witnesses = [{'row': index, 'coefficient': coefficient, **asdict(rows[index])}
                 for index, coefficient in terms]
    return {'id': identifier, 'scope': {'protocol': 'authored matched trajectories'},
            'target': 'response', 'status': 'violated', 'expected': 0.,
            'value': sum(item['coefficient'] * item['target'] for item in witnesses),
            'witnesses': witnesses, **extra}


def revision(*checks):
    return {'analysis_attempts': [{'execution': {'returncode': 0},
                                   'check_results': list(checks)}]}


@pytest.fixture
def rows():
    return [Observation({'u': 0.}, 1.),
            Observation({'u': 1.}, 3., ({'u': 0.},)),
            Observation({'u': 2.}, 5., ({'u': 0.}, {'u': 1.}))]


def test_refuted_expectation_is_not_used_as_the_candidate_target(rows):
    records = [revision(checked_relation(rows))]
    result = compare_paid_relations(records, rows, [1., 3., 5.])
    assert result['scope'] == 'paid_local_relations'
    assert result['revision_count'] == result['check_count'] == 1
    assert result['term_count'] == 2
    example = result['revisions'][0]['examples'][0]
    assert example['evidence_status'] == 'violated'
    assert example['measured_value'] == example['predicted_value'] == 2.
    assert example['residual'] == 0.
    assert example['scope'] == records[0]['analysis_attempts'][0]['check_results'][0]['scope']
    assert example['terms'] == [
        {'row_index': 1, 'coefficient': 1., 'measured_value': 3., 'predicted_value': 3.},
        {'row_index': 0, 'coefficient': -1., 'measured_value': 1., 'predicted_value': 1.}]
    assert not {'passed', 'inputs', 'research_input_bindings'} & result.keys()


def test_prefix_reports_only_relations_whose_complete_witnesses_are_present(rows):
    records = [revision(checked_relation(rows),
                        checked_relation(rows, 'later', ((2, 1.), (0, -1.))))]
    result = compare_paid_relations(records, rows[:2], [1., 2.])
    assert result['evaluated_rows'] == 2
    assert result['check_count'] == 1 and result['term_count'] == 2
    assert result['revisions'][0]['examples'][0]['residual'] == -1.
    assert compare_paid_relations(records, rows[:1], [1.])['revisions'] == []


@pytest.mark.parametrize('status,expected,predictions,residual', [
    ('violated', 0., [1., 3., 5.], 0.),
    ('satisfied', 2., [1., 1., 5.], -2.),
])
def test_historical_assertion_and_current_prediction_error_are_separate(
        rows, status, expected, predictions, residual):
    records = [revision(checked_relation(rows, status=status, expected=expected,
                                        relation='eq', atol=1e-9))]
    before = deepcopy(records)
    result = compare_paid_relations(records, rows, predictions)
    example = result['revisions'][0]['examples'][0]
    assert example['historical_check'] == {
        'status': status, 'expected': expected, 'relation': 'eq', 'atol': 1e-9}
    assert example['residual'] == residual
    assert 'not the current predictor accuracy' in result['field_meanings']['evidence_status']
    assert example['measured_value'] == 2. and records == before


@pytest.mark.parametrize('field,value', [('features', {'u': 9.}),
                                        ('history', [{'u': 7.}]), ('target', 8.)])
def test_reused_row_numbers_do_not_match_different_public_evidence(rows, field, value):
    records = [revision(checked_relation(rows))]
    current = [asdict(row) for row in rows]
    current[1][field] = value
    assert compare_paid_relations(records, current, [1., 3., 5.])['check_count'] == 0


@pytest.mark.parametrize('latest', [
    {'execution': {'returncode': 1}},
    {'execution': {'returncode': 0}, 'error': 'readout unfinished'},
    {'execution': {'returncode': 0}, 'check_results': []},
])
def test_failed_or_unchecked_latest_reanalysis_does_not_resurrect_old_checks(rows, latest):
    records = [revision(checked_relation(rows))]
    # An old top-level answer may remain in an actual revision after reanalysis.
    records[0]['check_results'] = deepcopy(records[0]['analysis_attempts'][0]['check_results'])
    records[0]['analysis_attempts'].append(latest)
    result = compare_paid_relations(records, rows, [1., 3., 5.])
    assert result['revision_count'] == result['check_count'] == result['term_count'] == 0


def test_latest_response_checks_rank_actual_errors_without_mutating_evidence(rows):
    checks = [checked_relation(rows, str(scale), ((1, scale), (0, -scale)))
              for scale in (1., 3., 2., 4.)]
    checks += [checked_relation(rows, 'measurement', target='sensor'),
               checked_relation(rows, 'unchecked', status='inconclusive')]
    records = [revision(checked_relation(rows, 'old'))]
    records[0]['analysis_attempts'].append({'execution': {'returncode': 0}, 'check_results': checks})
    public_rows = [asdict(row) for row in rows]
    public_rows[1]['history'] = list(public_rows[1]['history'])  # JSON-restored history.
    predictions = [1., 2., 5.]
    before = deepcopy((records, public_rows, predictions))
    result = compare_paid_relations(records, public_rows, predictions)
    assert (records, public_rows, predictions) == before
    assert result['check_count'] == 4 and result['term_count'] == 8
    observed = result['revisions'][0]
    assert observed['version'] == 2 and observed['max_abs_residual'] == 4.
    assert [item['id'] for item in observed['examples']] == ['4.0', '3.0', '2.0']


def test_predictions_must_cover_the_same_paid_prefix(rows):
    with pytest.raises(ValueError, match='cover the scored rows'):
        compare_paid_relations([], rows, [1.])
