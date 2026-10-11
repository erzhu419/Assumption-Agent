"""Final paid responses can revise a failed band while preserving its exact law."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_response_hypotheses as research
from hegel_machine.cognition.world_response_transfer import export_response_knowledge
from test_world_pending_band_revision import _fixed_program_state


PATH = 'shared/consolidated_method.py'


def _source():
    state = _fixed_program_state()
    # Together with the existing negative at residual 4, this supplies both
    # finite boundary intervals around the required positive residuals [0, 2].
    state['response_settings'].append(
        {'crystal': 1, 'frequency': 0., 'activated': False, 'evidence': [11]})
    return state


def _no_new_program(*args, **kwargs):
    pytest.fail('Revising source conditions must retain the exact coefficient program')


def test_failed_exact_band_rebuilds_conditions_without_refitting_or_relabeling_old_model(monkeypatch):
    state = _source()
    before = deepcopy(state)
    monkeypatch.setattr(research, '_fit_linear', _no_new_program)
    monkeypatch.setattr(research, '_propose_response_members', _no_new_program)

    knowledge = export_response_knowledge(state, [], 'paid_source')

    assert state == before
    assert knowledge['candidate_count'] == knowledge['rejected_count'] == 1
    assert knowledge['revised_model_count'] == 4
    assert {(model['lower'], model['upper']) for model in knowledge['models']} == {
        (-.5, 2.), (-.5, 3.), (0., 2.), (0., 3.)}
    revision, = knowledge['band_revisions']
    assert revision['status'] == 'rebuilt'
    assert revision['family'] == 'linear' and revision['source_question'] == 'old_response'
    assert revision['rejected_band_count'] == 1
    assert revision['positive_residual_span'] == [0., 2.]
    assert revision['lower_boundary_range'] == {'exclusive_min': -1., 'inclusive_max': 0.}
    assert revision['upper_boundary_range'] == {'inclusive_min': 2., 'exclusive_max': 4.}
    assert revision['conflicting_responses'] == [] and revision['model_count'] == 4
    response_evidence = {index for row in state['response_settings'] for index in row['evidence']}
    assert response_evidence <= set(revision['evidence'])
    assert 4 in revision['evidence']  # target measurement used for its residual
    old_program = state['response_research']['hypotheses'][0]['program']
    assert revision['program'] == old_program
    for model in knowledge['models']:
        assert model['program'] == old_program and model['source_question'] == 'old_response'
        assert model['band_revision']['origin'] == 'final_paid_response_band_revision'
        assert model['band_revision']['rejected_band_count'] == 1
        assert response_evidence <= set(model['band_revision']['evidence'])


def test_interior_negative_rejects_the_fixed_program_instead_of_exporting_a_repaired_band():
    state = _source()
    conflict = {'crystal': 1, 'frequency': 2., 'activated': False, 'evidence': [12]}
    state['response_settings'].append(conflict)

    knowledge = export_response_knowledge(state, [], 'paid_source')

    assert knowledge['models'] == [] and knowledge['revised_model_count'] == 0
    assert knowledge['rejected_count'] == 1
    revision, = knowledge['band_revisions']
    assert revision['status'] == 'refuted' and revision['model_count'] == 0
    assert revision['positive_residual_span'] == [0., 2.]
    assert any(all(row[key] == value for key, value in conflict.items())
               for row in revision['conflicting_responses'])
    assert 12 in revision['evidence']


def test_a_compatible_exact_band_prevents_unnecessary_revision_of_its_program():
    state = _source()
    exact = deepcopy(state['response_research']['hypotheses'][0])
    exact.update(id='already_compatible', lower=0., upper=2.)
    state['response_research']['hypotheses'].append(exact)
    before = deepcopy(state)

    knowledge = export_response_knowledge(state, [], 'paid_source')

    assert state == before
    model, = knowledge['models']
    assert (model['lower'], model['upper'], model['program']) == (0., 2., exact['program'])
    assert 'band_revision' not in model
    assert knowledge['candidate_count'] == 2 and knowledge['rejected_count'] == 1
    assert knowledge['revised_model_count'] == 0 and knowledge['band_revisions'] == []


@pytest.mark.parametrize('positive, expected_uppers', [(False, {2., 2.5}), (True, {3., 3.5})])
def test_last_response_changes_exported_conditions_even_without_another_research_round(positive, expected_uppers):
    state = _source()
    state['response_settings'].append(
        {'crystal': 4, 'frequency': 7., 'activated': positive, 'evidence': [20]})
    old_research = deepcopy(state['response_research'])

    knowledge = export_response_knowledge(state, [], 'paid_source')

    assert state['response_research'] == old_research
    assert {model['upper'] for model in knowledge['models']} == expected_uppers
    assert knowledge['revised_model_count'] == 4
    assert knowledge['source_response_count'] == len(state['response_settings'])
    assert 20 in knowledge['band_revisions'][0]['evidence']
    for model in knowledge['models']:
        for row in state['response_settings']:
            # This fixture's retained program is exactly the measured coordinate.
            residual = row['frequency'] - float(row['crystal'])
            assert (model['lower'] <= residual <= model['upper']) == row['activated']


def test_all_failed_history_bands_are_rebuilt_once_per_program():
    state = _source()
    second = deepcopy(state['response_research']['hypotheses'][0])
    second.update(id='earlier_band', upper=8.)
    state['response_research_history'] = [
        {'question_id': 'earlier_response', 'hypotheses': [second, deepcopy(second)]}]

    knowledge = export_response_knowledge(state, [], 'paid_source')

    assert knowledge['candidate_count'] == knowledge['rejected_count'] == 2
    assert len(knowledge['band_revisions']) == 1
    assert knowledge['band_revisions'][0]['rejected_band_count'] == 2
    assert knowledge['revised_model_count'] == len(knowledge['models']) == 4


def test_rebuilt_conditions_are_consumed_by_frozen_recall_on_new_measurements(monkeypatch):
    knowledge = export_response_knowledge(_source(), [], 'paid_source')
    rows = [{'crystal': n, 'measurements': {
        'measurement': {'value': float(10 * n), 'evidence': 100 + n}}}
        for n in (1, 2, 3)]
    rows[0]['frequency'] = {'value': 10., 'evidence': 111}
    rows[1]['frequency'] = {'value': 22., 'evidence': 112}
    state = {'research_table': {'references': rows[:2], 'targets': rows[2:]},
        'target_numbers': [3], 'completed': [],
        'control_grids': {'3': {'step': 1., 'origin': 0., 'lower': 0., 'upper': 10000.}},
        'response_settings': [
            {'crystal': 1, 'frequency': 10., 'activated': True, 'evidence': [111]},
            {'crystal': 2, 'frequency': 22., 'activated': True, 'evidence': [112]},
            # New-world evidence rejects the two revised bands extending to 3.
            {'crystal': 1, 'frequency': 12.5, 'activated': False, 'evidence': [113]}],
        'frozen_response_knowledge': knowledge}
    before = deepcopy(state)
    monkeypatch.setattr(research, '_fit_linear', _no_new_program)
    monkeypatch.setattr(research, '_propose_response_members', _no_new_program)

    result = research.propose_response_research(state, [], PATH)

    assert state == before
    transfer = result['frozen_transfer']
    assert transfer['status'] == 'selected'
    assert transfer['model_count'] == 4
    assert transfer['retained_count'] == transfer['rejected_count'] == 2
    assert {(h['lower'], h['upper']) for h in result['hypotheses']} == {(-.5, 2.), (0., 2.)}
    assert all(h['program'] == knowledge['models'][0]['program'] for h in result['hypotheses'])
    selected = result['selection']['selected_experiment']
    assert selected['number'] == 3 and 30. <= selected['prediction'] <= 32.
    assert selected['procedure']['path'] == PATH
    assert result['selection']['consequence_scope'] == 'frozen_numeric_response_members'
    assert state['completed'] == []
    assert set(result['evidence']) <= {101, 102, 103, 111, 112, 113}
