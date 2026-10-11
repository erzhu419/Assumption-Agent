"""Frozen response knowledge predicts new objects without refitting its law."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition import world_response_hypotheses as research
from hegel_machine.cognition.world_response_transfer import (
    combine_response_knowledge, export_response_knowledge,
    forecast_response_knowledge, reuse_response_knowledge,
)
from test_world_response_hypotheses import PATH, fixture


def _source():
    state, estimators = fixture()
    estimators[0]['measurements'] = {'measurement': {
        'instrument': 'signal meter', 'pattern': r'signal magnitude = ([\d.]+) units', 'group': 1}}
    x = Program('feature', name='measurement')
    program = Program('add', children=(Program('const', value=5.),
        Program('mul', children=(Program('const', value=5.),
            Program('mul', children=(x, x)))))).to_dict()
    hypotheses = [{'id': identifier, 'candidate_id': 'old_fit', 'family': 'quadratic',
        'program': deepcopy(program), 'lower': 0., 'upper': upper}
        for identifier, upper in (('narrow', 1.), ('wide', 1.5))]
    state['response_research'] = {'question_id': 'source_response', 'hypotheses': hypotheses}
    return state, estimators


def _knowledge():
    state, estimators = _source()
    return state, estimators, export_response_knowledge(state, estimators, 'paid_source')


def _negative(state, frequency):
    state['response_settings'].append({'crystal': 5, 'frequency': frequency,
        'activated': False, 'evidence': [101]})


def test_export_rechecks_final_counterexample_and_deduplicates_history():
    state, estimators = _source()
    narrow, wide = state['response_research']['hypotheses']
    state['response_research_history'] = [
        {'question_id': 'earlier_source', 'hypotheses': [deepcopy(narrow), deepcopy(narrow)]}]
    state['response_research']['hypotheses'] = [wide]
    # This was paid after the latest comparison: only the wide band is false.
    _negative(state, 131.25)
    before = deepcopy(state)

    knowledge = export_response_knowledge(state, estimators, 'paid_source')

    assert state == before
    assert knowledge['source_id'] == 'paid_source'
    assert len(knowledge['models']) == 1
    saved = knowledge['models'][0]
    assert saved['program'] == narrow['program']
    assert (saved['lower'], saved['upper']) == (0., 1.)
    assert saved['source_question'] == 'earlier_source'
    assert json.loads(json.dumps(knowledge)) == knowledge


def test_frozen_forecast_reports_disagreement_on_new_readout_without_fitting(monkeypatch):
    _, _, knowledge = _knowledge()
    before = deepcopy(knowledge)

    def no_fit(*args, **kwargs):
        pytest.fail('A frozen forecast must not fit coefficients or response bands')

    monkeypatch.setattr(research, '_fit_linear', no_fit)
    predictions = forecast_response_knowledge(knowledge, {'measurement': 6.},
                                               [185., 186.25, 187.])

    assert knowledge == before
    assert [row['setting'] for row in predictions] == [185., 186.25, 187.]
    assert [row['prediction'] for row in predictions] == [True, None, False]
    assert all(row['model_count'] == 2 for row in predictions)
    assert [row['positive_models'] for row in predictions] == [2, 1, 0]
    assert [row['negative_models'] for row in predictions] == [0, 1, 2]


def test_new_counterexample_withdraws_exact_band_and_preserves_source():
    state, estimators, knowledge = _knowledge()
    before = deepcopy(knowledge)
    state['frozen_response_knowledge'] = knowledge
    _negative(state, 131.25)

    result = reuse_response_knowledge(state, estimators, PATH)

    assert knowledge == before
    transfer = result['frozen_transfer']
    assert transfer['model_count'] == 2
    assert transfer['retained_count'] == transfer['rejected_count'] == 1
    assert transfer['status'] == 'selected'
    assert result['selected_number'] == 5
    assert result['selected_setting'] in (130., 131.)
    assert len(result['hypotheses']) == 1
    hypothesis = result['hypotheses'][0]
    assert hypothesis['program'] == before['models'][0]['program']
    assert (hypothesis['lower'], hypothesis['upper']) == (0., 1.)


def test_instrument_semantics_mismatch_does_not_reuse_same_named_coordinate():
    state, estimators, knowledge = _knowledge()
    state['frozen_response_knowledge'] = knowledge
    changed = deepcopy(estimators)
    changed[0]['measurements']['measurement']['instrument'] = 'different instrument'

    result = reuse_response_knowledge(state, changed, PATH)

    assert result['frozen_transfer']['status'] == 'unavailable'
    assert result['selection'] is None


def test_native_research_uses_frozen_selection_then_reopens_after_actual_refutation(monkeypatch):
    state, estimators, knowledge = _knowledge()
    state['frozen_response_knowledge'] = knowledge
    original = research._propose_response_members

    def no_new_search(*args, **kwargs):
        pytest.fail('A usable frozen prediction should execute before new coefficient search')

    monkeypatch.setattr(research, '_propose_response_members', no_new_search)
    reused = research.propose_response_research(state, estimators, PATH)
    assert reused['frozen_transfer']['status'] == 'selected'
    assert reused['selection']['selected_experiment']['procedure']['path'] == PATH
    assert {(h['lower'], h['upper']) for h in reused['hypotheses']} == {(0., 1.), (0., 1.5)}

    monkeypatch.setattr(research, '_propose_response_members', original)
    _negative(state, 130.)  # Both unchanged source hypotheses predicted activation.
    baseline_state = deepcopy(state)
    baseline_state.pop('frozen_response_knowledge')
    baseline = research.propose_response_research(baseline_state, estimators, PATH)
    reopened = research.propose_response_research(state, estimators, PATH)

    transfer = reopened.pop('frozen_transfer')
    assert transfer['status'] == 'refuted'
    assert transfer['retained_count'] == 0
    assert transfer['rejected_count'] == 2
    screening_cost = reuse_response_knowledge(state, estimators, PATH)['compute_cost']
    assert reopened.pop('compute_cost') == baseline.pop('compute_cost') + screening_cost
    assert screening_cost > 0
    assert reopened == baseline
    assert state['frozen_response_knowledge'] == knowledge


def test_frozen_band_does_not_expand_to_reach_out_of_range_target():
    state, estimators, knowledge = _knowledge()
    state['frozen_response_knowledge'] = knowledge
    state['control_grid']['upper'] = 100.

    result = reuse_response_knowledge(state, estimators, PATH)

    assert result['frozen_transfer']['status'] == 'exhausted'
    assert result['selection'] is None
    assert result['selected_setting'] is None
    assert {(h['lower'], h['upper']) for h in result['hypotheses']} == {(0., 1.), (0., 1.5)}


def test_missing_knowledge_leaves_native_entry_untouched():
    state, estimators = fixture()
    assert reuse_response_knowledge(state, estimators, PATH) is None


def test_native_loader_keeps_prior_archive_separate_from_current_paid_evidence(tmp_path):
    from test_world_bound_response import _ready, _state
    from hegel_machine.cognition.world_reference_response import _reference_state
    from hegel_machine.cognition.world_response_transfer import KNOWLEDGE_PATH

    agent, lab = _ready(tmp_path)
    current = _state(agent)
    _, _, knowledge = _knowledge()
    (agent.workspace / KNOWLEDGE_PATH).write_text(json.dumps(knowledge))
    observations = deepcopy(agent.observations)

    _, loaded, _ = _reference_state(agent, lab, PATH)

    assert loaded['frozen_response_knowledge'] == knowledge
    assert loaded['response_settings'] == current['response_settings']
    assert loaded['completed'] == current['completed']
    assert loaded['research_table'] == current['research_table']
    assert agent.observations == observations


def test_bound_coordinate_comparison_cannot_consume_raw_frozen_knowledge(tmp_path):
    from test_world_bound_response import _ready, _compare, _state
    from hegel_machine.cognition import world_reference_recalibration as recipe
    from hegel_machine.cognition.world_response_transfer import KNOWLEDGE_PATH

    agent, lab = _ready(tmp_path)
    state = _state(agent)
    # Raw density and log_density belong to different comparison spaces.
    state['frozen_response_knowledge'] = {'source_id': 'raw_coordinates',
        'measurements': {}, 'models': []}
    (agent.workspace / KNOWLEDGE_PATH).write_text(json.dumps(state['frozen_response_knowledge']))
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))

    compared = _compare(agent, lab)

    assert 'frozen_transfer' not in compared
    assert _state(agent)['frozen_response_knowledge'] == state['frozen_response_knowledge']


@pytest.mark.parametrize('inherited', [False, True])
def test_source_consolidation_exports_frozen_models_after_latest_paid_counterexample(tmp_path, inherited):
    from scripts import consolidate_response_measurement as study
    from hegel_machine.cognition import world_reference_recalibration as recipe
    from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
    from hegel_machine.cognition.world_response_transfer import KNOWLEDGE_PATH
    from test_world_reference_hypothesis import prepared, square

    source = tmp_path / 'source'
    agent, lab = prepared(source / 'agent')
    declaration = square('density', 'density_square')
    (agent.workspace / PATH).write_text(_retain_declaration(
        (agent.workspace / PATH).read_text(), declaration))
    for number in range(1, 6):
        agent._execute('query', {'actions': [
            {'action': 'TALK', 'arg1': 100 + number}, {'chosen_dialog_option_int': 1},
            {'action': 'USE', 'arg1': 91, 'arg2': 200 + number}]}, lab)
    _, _, _, readings, references = recipe._scan(agent.observations, lab.public_observation())
    table = {'references': [], 'targets': []}
    for number in range(1, 6):
        row = {'crystal': number, 'measurements': {'density': readings['density_' + str(number)]}}
        if number <= 3:
            row['frequency'] = references[str(number)]
            table['references'].append(row)
        else:
            table['targets'].append(row)
    x = Program('feature', name='density')
    program = Program('add', children=(Program('const', value=100.),
        Program('mul', children=(Program('const', value=10.),
            Program('mul', children=(x, x)))))).to_dict()
    state = {'completed': [], 'target_numbers': [4, 5], 'research_table': table,
        'response_research': {'question_id': 'source_response', 'hypotheses': [
            {'program': program, 'lower': -width, 'upper': width, 'family': 'density_square'}
            for width in (.5, 2.)]}}
    recipe.refresh_response_evidence(state, agent.observations, lab.public_observation(), [declaration])
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))
    # New evidence arrives after the saved comparison. The archive must consume
    # this actual failed 111 Hz trial before it publishes the old wide band.
    agent._execute('query', {'actions': [
        {'action': 'TALK', 'arg1': 101}, {'chosen_dialog_option_int': 5},
        {'chosen_dialog_option_int': 1}]}, lab)
    assert lab.frequencies[1] == 111. and not lab.active(1)
    (source / 'result.json').write_text(json.dumps({'agent': agent.result()}))
    paid = deepcopy(agent.observations)
    old = {'source_id': 'earlier_source', 'measurements': {'density': None},
           'models': [{'program': program, 'lower': -2., 'upper': 2.,
                       'family': 'density_square', 'source_question': 'earlier_response'}]}
    if inherited:
        (agent.workspace / KNOWLEDGE_PATH).write_text(json.dumps(old))

    package, _ = study.source_knowledge(source, agent, 0)

    knowledge = json.loads(package['files'][KNOWLEDGE_PATH])
    assert len(knowledge['models']) == (2 if inherited else 1)
    revised = knowledge['models'][-1]
    assert revised['program'] == program
    assert (revised['lower'], revised['upper']) == (-.5, .5)
    if inherited:
        assert knowledge['models'][0] == {**old['models'][0], 'source_id': 'earlier_source'}
        assert revised['source_id'] == str(source)
        assert json.loads((agent.workspace / KNOWLEDGE_PATH).read_text()) == old
    else:
        assert knowledge['rejected_count'] == 1
    assert agent.observations == paid
    assert str(recipe.STATE) not in package['files']


def _revision(knowledge):
    revised = deepcopy(knowledge)
    revised['source_id'] = 'revised_source'
    for model in revised['models']:
        model['program'] = Program('add', children=(Program('const', value=10.),
            Program.from_dict(model['program']))).to_dict()
        model['source_question'] = 'revised_response'
    return revised


@pytest.mark.parametrize('changed', [False, True])
def test_combined_archive_recalls_each_law_from_current_evidence_without_erasing_other(changed, monkeypatch):
    state, estimators, previous = _knowledge()
    revised = _revision(previous)
    combined = combine_response_knowledge(previous, revised)
    before = deepcopy((previous, revised, combined))
    if changed:
        for row in state['response_settings']:
            row['frequency'] += 10.
    state['frozen_response_knowledge'] = combined

    def no_fit(*args, **kwargs):
        pytest.fail('Recall of retained laws must not fit new coefficients')

    monkeypatch.setattr(research, '_fit_linear', no_fit)
    result = reuse_response_knowledge(state, estimators, PATH)

    selected_source = 'revised_source' if changed else 'paid_source'
    rejected_source = 'paid_source' if changed else 'revised_source'
    assert (previous, revised, combined) == before
    assert result['frozen_transfer']['status'] == 'selected'
    counts = result['frozen_transfer']['source_counts']
    assert counts[selected_source] == {'model_count': 2, 'retained_count': 2, 'rejected_count': 0}
    assert counts[rejected_source] == {'model_count': 2, 'retained_count': 0, 'rejected_count': 2}
    assert result['selected_setting'] in ((140., 141.) if changed else (130., 131.))
    origins = [source for candidate in result['candidates'] for source in candidate['fit_sources']]
    assert {source['source_id'] for source in origins} == {selected_source}
    assert {source['source_question'] for source in origins} == {
        'revised_response' if changed else 'source_response'}


def test_combined_archive_preserves_single_archive_and_deduplicates_exact_repeated_laws():
    _, _, previous = _knowledge()
    empty = {**deepcopy(previous), 'source_id': 'empty_revision', 'models': []}
    assert combine_response_knowledge(previous, empty) == previous
    assert combine_response_knowledge(None, previous) == previous
    combined = combine_response_knowledge(previous, _revision(previous))
    again = combine_response_knowledge(combined, _revision(previous))
    assert again == combined
    # Re-observing the same numerical law does not invent a new independent model.
    repeated = deepcopy(previous)
    repeated['source_id'] = 'another_episode'
    assert len(combine_response_knowledge(previous, repeated)['models']) == len(previous['models'])


@pytest.mark.parametrize('missing', [False, True])
def test_combining_archives_rejects_changed_or_missing_required_measurement_declaration(missing):
    _, _, previous = _knowledge()
    revised = _revision(previous)
    if missing:
        revised['measurements'].pop('measurement')
    else:
        revised['measurements']['measurement']['instrument'] = 'different instrument'
    with pytest.raises(ValueError, match='measurement declarations'):
        combine_response_knowledge(previous, revised)

