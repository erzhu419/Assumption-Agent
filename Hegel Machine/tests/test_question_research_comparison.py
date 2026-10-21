"""The paid research subpool changes PR2 votes, without rescoring evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate
from hegel_machine.cognition.question_selection import select_question_plan
from hegel_machine.cognition.selection import AssessedCandidate
from test_question_selection import experiment, question, theory


def comparison_fixture():
    programs = [Program('const', value=0.), Program('feature', name='u'),
                Program('const', value=1.)]
    identifiers = ['zero', 'linear', 'one']
    theories = [theory(identifier, program) for identifier, program in zip(identifiers, programs)]
    basis = {'file': 'paid_fit.json', 'json_path': ['assessment_basis']}
    assessed = [AssessedCandidate(Candidate(identifier, program, 'reuse'), error, None, error,
        assessment_scope={'kind': 'retained_full_fit', 'row_count': 4,
                          'assessment_basis_ref': deepcopy(basis)})
        for identifier, program, error in zip(identifiers, programs, (0., .1, 1.))]
    comparison = {'maximum': 2, 'assessed': assessed, 'mandatory_theory_ids': [],
        'assessment_refs': {identifier: {'file': 'paid_fit.json', 'json_path': ['fits', identifier]}
                            for identifier in identifiers},
        'unknown': [], 'status': 'ready', 'scope': 'retained_paid_fit_research_only',
        'assessment_basis_ref': basis, 'row_count': 4,
        'support_updated': False, 'task_adoption_evaluated': False}
    return question(2.), theories, [experiment('at_zero', 0.), experiment('at_one', 1.)], comparison


def group_ids(result):
    return [identifier for group in result['groups'] for identifier in group['theory_ids']]


@pytest.mark.parametrize('vectors', [False, True])
def test_paid_research_retention_changes_actual_groups_edges_G_and_action_without_extra_prediction(
        monkeypatch, vectors):
    q, theories, panel, comparison = comparison_fixture()
    if vectors:
        for owner, key in [(q['query_semantics'], 'query_expression'),
                           *((plan, 'query') for plan in panel)]:
            owner[key] = {'op': 'Readout', 'statistic_ref': {'kind': 'vector'},
                          'finite_readouts': [deepcopy(owner[key]), deepcopy(owner[key])]}
    original = deepcopy((q, theories, panel, comparison))
    calls, predict = [], Program.predict

    def counted(self, row, *args, **kwargs):
        calls.append((self, deepcopy(row.features)))
        return predict(self, row, *args, **kwargs)

    monkeypatch.setattr(Program, 'predict', counted)
    ordinary = select_question_plan(q, theories, panel)
    ordinary_calls = list(calls)
    calls.clear()
    selected = select_question_plan(q, theories, panel, research_comparison=comparison)
    assert ordinary['status'] == 'tied' and ordinary['tied_experiments'] == ['at_zero', 'at_one']
    assert [score['G'] for score in ordinary['scores']] == [pytest.approx(2 / 3)] * 2
    assert selected['status'] == 'selected' and selected['selected_experiment']['id'] == 'at_one'
    assert group_ids(selected) == ['zero', 'linear']
    assert [score['G'] for score in selected['scores']] == [0., 1.]
    assert selected['pairwise_edges'] == [{'left': 'group_000', 'right': 'group_001',
        'different_task_answer': True, 'experiments': ['at_one']}]
    assert selected['query_arguments'] == {'actions': panel[1]['actions']}
    assert calls == ordinary_calls and selected['compute_cost'] == ordinary['compute_cost']
    details = selected['research_comparison']
    assert details['scope'] == 'retained_paid_fit_research_only'
    assert details['assessment_basis_ref'] == comparison['assessment_basis_ref']
    assert details['assessment_refs'] == comparison['assessment_refs']
    assert details['prediction_compute_cost'] == 0 and details['support_updated'] is False
    assert details['all_pool_optimality_established'] is False
    assert details['task_adoption_evaluated'] is False and selected['support_updated'] is False
    assert details['retained'][1]['assessment_scope'] == comparison['assessed'][1].assessment_scope
    assert (q, theories, panel, comparison) == original


@pytest.mark.parametrize('source', ['explicit', 'unassessed', 'declared_mandatory'])
def test_original_explicit_or_unassessed_candidates_are_mandatory_even_beyond_research_limit(source):
    q, theories, panel, comparison = comparison_fixture()
    if source == 'explicit':
        q['candidate_theory_ids'] = ['zero', 'linear', 'one']
    elif source == 'unassessed':
        comparison['assessed'].pop()
        comparison['assessment_refs'].pop('one')
        comparison['unknown'] = [{'id': 'one', 'reason': 'missing_same_scope_assessment'}]
    else:
        comparison['mandatory_theory_ids'] = ['one']
    result = select_question_plan(q, theories, panel, research_comparison=comparison)
    assert group_ids(result) == ['zero', 'linear', 'one']
    assert result['status'] == 'tied' and result['selected_experiment'] is None
    details = result['research_comparison']
    assert 'one' in details['mandatory_theory_ids'] and details['selected_theory_ids'] == group_ids(result)
    assert details['selected_theory_count'] == 3 and details['maximum'] == 2


@pytest.mark.parametrize('status', ['refuted', 'revoked', 'withdrawn'])
def test_mandatory_or_low_scoring_entry_cannot_resurrect_scientifically_excluded_model(status):
    q, theories, panel, comparison = comparison_fixture()
    theories[2]['support']['status'] = status
    comparison['mandatory_theory_ids'] = ['one']
    comparison['assessed'][2].evidence_loss = -1.
    original = deepcopy(theories)
    result = select_question_plan(q, theories, panel, research_comparison=comparison)
    assert group_ids(result) == ['zero', 'linear']
    assert result['selected_experiment']['id'] == 'at_one'
    assert result['excluded'] == [{'id': 'one', 'reason': 'incompatible_candidate'}]
    assert all(item['candidate_id'] != 'one' for item in result['research_comparison']['retained'])
    assert theories == original


def test_opaque_retained_python_column_remains_mandatory_without_inventing_a_program():
    q, theories, panel, comparison = comparison_fixture()
    opaque = theories[2]
    opaque.pop('program')
    opaque['forecast_ref'] = {'file': 'saved_python.json', 'json_path': ['forecasts', 0],
                              'model': 'old.py'}
    opaque['_retained_forecast'] = {'prediction_inputs': [
        {'features': {'u': level}, 'history': []} for level in (0., 1., 2.)],
        'predictions': [1., 1., 1.]}
    comparison['assessed'].pop()
    result = select_question_plan(q, theories, panel, research_comparison=comparison)
    assert group_ids(result) == ['zero', 'linear', 'one']
    assert result['status'] == 'tied' and result['compute_cost'] == 9
    assert result['research_comparison']['mandatory_theory_ids'] == ['one']
    assert 'program' not in opaque


def test_missing_assessments_preserve_entire_original_pool_and_report_unknown():
    q, theories, panel, comparison = comparison_fixture()
    comparison.update(assessed=[], assessment_refs={}, status='UNKNOWN')
    ordinary = select_question_plan(q, theories, panel)
    result = select_question_plan(q, theories, panel, research_comparison=comparison)
    assert result['groups'] == ordinary['groups'] and result['scores'] == ordinary['scores']
    assert result['compute_cost'] == ordinary['compute_cost']
    assert result['research_comparison']['status'] == 'UNKNOWN'
    assert result['research_comparison']['mandatory_theory_ids'] == ['zero', 'linear', 'one']
    assert {item['id'] for item in result['research_comparison']['unknown']} == {'zero', 'linear', 'one'}


def test_incomplete_panel_is_unknown_behavior_and_keeps_partial_prediction_scope():
    q, theories, panel, comparison = comparison_fixture()
    result = select_question_plan(q, theories, panel, compute_budget=5,
        exploration_policy={'kind': 'seeded_legal', 'seed': 0}, research_comparison=comparison)
    details = result['research_comparison']
    assert result['compute_cost'] == 5 and group_ids(result) == ['zero', 'linear']
    assert details['unknown_prediction_ids'] == ['linear', 'one']
    assert all('different_public_behavior' not in item['reasons'] for item in details['retained'])
    assert details['all_pool_optimality_established'] is False
    assert result['prediction_coverage']['complete'] is False
    assert result['next_work']['missing_plan_predictions']['at_one'] == ['linear']
    assert result['selected_experiment'] is None
