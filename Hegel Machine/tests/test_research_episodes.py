"""Episode extraction reads actual outcomes rather than promoting notes to facts."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.research_episodes import (
    extract_episodes, load_episode_sources, load_source_values, resolve_program_sources,
)


PROGRAM = {'op': 'feature', 'name': 'u', 'children': []}


def agent(tmp_path):
    return SimpleNamespace(workspace=tmp_path, research_source_id='retained_trial',
        questions=[{'id': 'q', 'question': 'Does the declared response relation hold?',
                    'meaning_version': 1, 'status': 'inconclusive'}],
        observations=[{'row_indices': [0], 'rows': [{'target': 7.}], 'observation_cost': 1}],
        revisions=[], events=[], confirmations=[], relation_research={},
        theory_index={'old': {'id': 'old', 'version': 1,
            'source_ref': {'kind': 'native_candidate', 'question_id': 'q',
                           'round_index': 0, 'candidate_index': 0, 'meaning_version': 1},
            'context_ref': {'episode': 'retained_trial'},
            'support': {'status': 'candidate'}}})


def question_episode(result, identifier='q@meaning1'):
    return next(item for item in result['episodes'] if item['episode_id'] == identifier)


def test_declaration_stays_candidate_without_adoption_or_program_copy(tmp_path):
    original = agent(tmp_path)
    original.theory_index['inline'] = {'id': 'inline', 'version': 1, 'program': PROGRAM,
        'context_ref': {'episode': 'retained_trial'}}
    before = deepcopy(vars(original))
    result = extract_episodes(original)
    assert all(not item['adoption']['occurred'] for item in result['episodes'])
    assert all(item['source_id'] == 'retained_trial' for item in result['episodes'])
    assert all(not item['independent_results'] for item in result['episodes'])
    assert '"op"' not in json.dumps(result)
    assert result['unique_support_sources'] == []
    assert vars(original) == before


def test_inline_program_preserves_original_archive_source_after_temporary_workspace_ends(tmp_path):
    original = agent(tmp_path)
    source = {'archive': '/retained/history.sqlite3', 'unit': 'D',
              'member': 'cognition_v3_sequence_20260908.json', 'field': 'episodes[0].program'}
    original.theory_index = {'history': {'id': 'history', 'version': 1,
        'program': PROGRAM, 'program_source_ref': source}}
    result = extract_episodes(original)
    program_ref = result['episodes'][0]['program_refs'][0]['source_ref']
    assert program_ref == source
    assert 'research_state.json' not in json.dumps(program_ref)
    assert '"op"' not in json.dumps(result)


def test_inline_episode_reference_keeps_its_declared_meaning_version(tmp_path):
    original = agent(tmp_path)
    original.theory_index['new_meaning'] = {'id': 'new_meaning', 'version': 1,
        'program': PROGRAM, 'episode_ref': {'question_id': 'q', 'meaning_version': 2}}
    result = extract_episodes(original)
    assert 'new_meaning' not in question_episode(result)['theory_ids']
    assert question_episode(result, 'q@meaning2')['theory_ids'] == ['new_meaning']


def test_T21_repeated_notes_of_same_record_keep_one_evidence_source(tmp_path):
    original = agent(tmp_path)
    result = extract_episodes(original, [
        {'id': 'note1', 'kind': 'measured', 'source_ref': {'kind': 'observation', 'record_index': 0},
         'claim': {'text': 'Response equals seven.'}},
        {'id': 'note2', 'kind': 'author_interpretation',
         'source_id': 'pretend_independent_source',
         'source_ref': {'kind': 'observation', 'record_index': 0},
         'claim': {'text': 'The same measured response is seven.'}}])
    assert [item['kind'] for item in result['materials']] == ['measured', 'duplicate']
    assert result['materials'][1]['duplicate_of'] == 'note1'
    assert result['materials'][1]['claim']['text'].startswith('The same')
    assert result['materials'][1]['source_id'] == 'retained_trial'
    assert len(result['unique_support_sources']) == 1
    assert all(not item['adds_independent_support'] for item in result['materials'])


def test_T22_actual_numeric_contradiction_preserves_claim_and_observed_value(tmp_path):
    original = agent(tmp_path)
    original.revisions = [{'index': 0, 'question_id': 'q', 'status': 'answered',
        'record_indices': [0], 'authored_answer': {'status': 'supported', 'answer': 'response is zero'},
        'evidence_status': 'refuted',
        'check_results': [{'status': 'violated', 'value': 7., 'expected': 0., 'residual': 7.}]}]
    before = deepcopy(original.revisions)
    result = extract_episodes(original, [{'id': 'claim', 'kind': 'author_interpretation',
        'source_ref': {'kind': 'revision', 'index': 0},
        'claim': {'status': 'supported', 'text': 'Response is zero'}}])
    assert len(result['conflicts']) == 2
    assert all(item['actual_value'] == 7. and item['expected'] == 0. for item in result['conflicts'])
    assert question_episode(result)['failure_refs'] == [
        {'file': 'research_state.json', 'json_path': ['revisions', 0]}]
    assert original.revisions == before
    assert result['materials'][0]['claim']['text'] == 'Response is zero'


def test_revised_native_program_is_retained_but_identical_round_is_not_change(tmp_path):
    original = agent(tmp_path)
    original.questions[0]['native_condition_research'] = [
        {'status': 'candidate', 'fit_record_indices': [0], 'candidates': [{'program': PROGRAM}]},
        {'status': 'candidate', 'fit_record_indices': [0], 'candidates': [{'program': deepcopy(PROGRAM)}]},
        {'status': 'refuted', 'fit_record_indices': [0],
         'candidates': [{'program': {'op': 'const', 'value': 0.}}]}]
    result = extract_episodes(original)
    episode = question_episode(result)
    assert len(episode['actual_changes']) == 1
    assert episode['actual_changes'][0]['source_ref']['json_path'][-4:] == [
        2, 'candidates', 0, 'program']
    assert len(episode['change_evidence']) == 1
    assert episode['failure_refs'][0]['json_path'][-1] == 2
    assert not episode['adoption']['occurred']
    assert '"op"' not in json.dumps(result)


@pytest.mark.parametrize('cutoff,indices,independent', [(1, [1], True), (2, [0], False)])
def test_frozen_scoring_distinguishes_new_results_from_reused_paid_fit(tmp_path, cutoff, indices, independent):
    original = agent(tmp_path)
    original.observations.append({'row_indices': [1], 'rows': [{'target': 4.}], 'observation_cost': 1})
    original.revisions = [{'question_id': 'q', 'evidence_cutoff': {'records': cutoff},
        'record_indices': indices, 'forecasts': [{'model': 'old.py', 'complete_experiment': True,
                                                'prospective_mse': 9.}]}]
    result = extract_episodes(original)
    check = question_episode(result)['independent_results'][0]
    assert check['independent_from_fit'] is independent
    assert check['independent_confirmation'] is False
    assert check['prospective_mse'] == 9.


def test_historical_meanings_and_failed_acquisition_are_not_discarded(tmp_path):
    original = agent(tmp_path)
    original.questions[0].update(meaning_version=2,
        meaning_history=[{'meaning_version': 1, 'question': 'Old actual question', 'status': 'refuted'}])
    original.revisions = [{'question_id': 'q', 'question_meaning_version': 1,
        'status': 'acquisition_failed', 'acquisition_error': 'apparatus unavailable', 'record_indices': []}]
    result = extract_episodes(original)
    old = question_episode(result)
    assert old['original_question_semantics']['question'] == 'Old actual question'
    assert old['question_ref']['json_path'] == ['questions', 0, 'meaning_history', 0]
    assert old['failure_refs'] and old['status'] == 'refuted'
    assert old['independent_results'] == []


def test_prepared_plan_and_note_do_not_claim_adoption_but_actual_refit_does(tmp_path):
    original = agent(tmp_path)
    original.questions[0]['parent_use_history'] = [
        {'results': [{'status': 'parent_query_prepared'}]}]
    result = extract_episodes(original, [{'id': 'adoption_claim', 'kind': 'author_interpretation',
        'source_ref': {'kind': 'question', 'question_id': 'q'},
        'claim': {'adopted': True}}])
    assert not question_episode(result)['adoption']['occurred']
    original.questions[0]['parent_use_history'].append({'results': [{'status': 'parent_refitted'}]})
    result = extract_episodes(original)
    assert question_episode(result)['adoption'] == {'occurred': True, 'execution_refs': [
        {'file': 'research_state.json', 'json_path': ['questions', 0, 'parent_use_history', 1]}]}


def test_actual_executed_evaluation_and_failed_confirmation_are_preserved(tmp_path):
    original = agent(tmp_path)
    original.events = [{'action': 'evaluate', 'arguments': {'model': 'old.py'},
        'feedback': {'research_question_id': 'q', 'returncode': 0, 'mse': 4.,
                     'fresh_confirmation': {'passed': False}}}]
    result = extract_episodes(original)
    episode = question_episode(result)
    assert episode['adoption']['occurred'] is True
    assert episode['independent_results'][0]['passed'] is False
    assert episode['independent_results'][0]['independent_confirmation'] is True


def test_refuted_relation_source_stays_reference_without_copying_program(tmp_path):
    original = agent(tmp_path)
    original.relation_research['q'] = {'status': 'refuted', 'candidates': [
        {'status': 'refuted', 'program': PROGRAM}]}
    result = extract_episodes(original)
    episode = question_episode(result)
    assert any(item['status'] == 'refuted' for item in episode['program_refs'])
    assert episode['failure_refs'] == [
        {'file': 'research_state.json', 'json_path': ['relation_research', 'q']}]
    assert '"op"' not in json.dumps(result)


def test_reordered_candidate_pool_is_not_recorded_as_a_program_revision(tmp_path):
    original = agent(tmp_path)
    other = {'op': 'const', 'value': 3.}
    original.questions[0]['native_condition_research'] = [
        {'candidates': [{'program': PROGRAM}, {'program': other}]},
        {'candidates': [{'program': other}, {'program': PROGRAM}]}]
    result = extract_episodes(original)
    episode = question_episode(result)
    assert len(episode['initial_candidates']) == 3  # Declaration plus two original candidates.
    assert episode['actual_changes'] == []


def test_different_question_context_names_do_not_duplicate_the_same_paid_source(tmp_path):
    original = agent(tmp_path)
    original.questions[0]['context_ref'] = {'episode': 'description_a'}
    original.questions.append({'id': 'other', 'context_ref': {'episode': 'description_b'}})
    original.revisions = [{'question_id': identifier, 'record_indices': [0]}
                          for identifier in ('q', 'other')]
    result = extract_episodes(original)
    assert len(result['unique_support_sources']) == 1
    assert result['unique_support_sources'][0]['source_id'] == 'retained_trial'


def test_text_material_claim_remains_text_when_revision_has_a_numeric_check(tmp_path):
    original = agent(tmp_path)
    original.revisions = [{'question_id': 'q', 'record_indices': [0],
        'check_results': [{'status': 'violated', 'value': 7., 'expected': 0.}]}]
    result = extract_episodes(original, [{'kind': 'author_interpretation',
        'source_ref': {'kind': 'revision', 'index': 0}, 'claim': 'A textual relation proposal.'}])
    assert result['materials'][0]['claim'] == 'A textual relation proposal.'
    assert result['conflicts'] == []


def test_external_document_cannot_declare_itself_measured_or_executed(tmp_path):
    original = agent(tmp_path)
    result = extract_episodes(original, [
        {'kind': 'measured', 'source_ref': {'file': 'external_paper.pdf'}},
        {'kind': 'execution', 'source_ref': {'file': 'external_note.md'}}])
    assert [entry['kind'] for entry in result['materials']] == ['external_claim', 'author_interpretation']
    assert result['unique_support_sources'] == []


def test_missing_source_stays_explicit_instead_of_manufacturing_record(tmp_path):
    original = agent(tmp_path)
    with pytest.raises(ValueError, match='unavailable paid record'):
        extract_episodes(original, [{'kind': 'measured',
            'source_ref': {'kind': 'observation', 'record_index': 3}}])


def retained_source():
    return {'archive': '/retained/history.sqlite3', 'unit': 'D',
            'member': 'cognition_v3_sequence_20260908.json', 'json_path': ['episodes', 0]}


def retained_episode():
    return {'task_id': 't01', 'program': PROGRAM, 'status': 'completed',
        'questions': [{'question': 'Which executable revision is worth its cost?'}],
        'observation_cost': 3, 'compute_cost': 8, 'stop_reason': 'no_further_gain',
        'actions': [
            {'action': 'observe', 'observation_cost': 3, 'details': {'probe_id': 'initial', 'rows': 3}},
            {'action': 'compare', 'compute_cost': 8,
             'details': {'adopt': True, 'best_loss': .01, 'validation_is_adaptive_development': True}},
            {'action': 'revise', 'details': {'operation': 'abstract_reuse',
                'before_program': {'op': 'const', 'value': 0.}, 'after_program': PROGRAM}},
            {'action': 'reuse_check', 'details': {'accepted': False, 'loss': 2.}},
            {'action': 'prediction_check', 'details': {'revision_loss': .01, 'parent_loss': 1.}}],
        'evaluation': {'loss': .02, 'rows': 192,
                       'prediction_examples': [{'features': {'u': 1.}, 'target': 1., 'prediction': 1.}]}}


def test_real_sequence_episode_uses_saved_actions_costs_and_failures_without_copying_payload(tmp_path):
    original = agent(tmp_path)
    original.theory_index = {'historic': {'id': 'historic', 'version': 1, 'program': PROGRAM,
        'episode_ref': {'question_id': 't01', 'meaning_version': 1},
        'context_ref': {'episode': 'cognition_v3_sequence_20260908/t01'}}}
    saved = retained_episode()
    before = deepcopy(saved)
    result = extract_episodes(original, retained_episodes=[{'source_ref': retained_source(), 'episode': saved}])
    episode = question_episode(result, 't01@meaning1')
    assert episode['source_id'] == 'cognition_v3_sequence_20260908/t01'
    assert episode['theory_ids'] == ['historic']
    assert episode['costs'] == {'observation_cost': 3, 'compute_cost': 8}
    assert episode['adoption']['occurred'] is True
    assert episode['actual_changes'][0]['operation'] == 'abstract_reuse'
    assert episode['actual_changes'][0]['before_ref']['json_path'][-4:] == ['actions', 2, 'details', 'before_program']
    assert episode['behavior_changes'][1]['accepted'] is False
    assert episode['failure_refs'][0]['json_path'] == ['episodes', 0, 'actions', 3]
    assert episode['independent_results'][0]['loss'] == .02
    assert episode['independent_results'][0]['independent_confirmation'] is False
    assert len(result['unique_support_sources']) == 1
    assert '"op"' not in json.dumps(result) and '"target"' not in json.dumps(result)
    assert saved == before and original.events == []


def test_retained_action_material_is_execution_only_after_actual_source_resolution(tmp_path):
    original = agent(tmp_path)
    reference = {**retained_source(), 'json_path': ['episodes', 0, 'actions', 2]}
    materials = [{'id': 'note1', 'kind': 'execution', 'source_ref': reference},
                 {'id': 'note2', 'kind': 'execution', 'source_ref': reference}]
    external = extract_episodes(original, materials)
    assert external['materials'][0]['kind'] == 'author_interpretation'
    actual = extract_episodes(original, materials, retained_episodes=[
        {'source_ref': retained_source(), 'episode': retained_episode()}])
    assert [item['kind'] for item in actual['materials']] == ['execution', 'duplicate']
    assert actual['materials'][1]['duplicate_of'] == 'note1'
    assert all(not item['adds_independent_support'] for item in actual['materials'])


def test_episode_loader_reads_shared_small_json_once_and_returns_requested_objects(tmp_path, monkeypatch):
    path = tmp_path / 'sequence.json'
    path.write_text(json.dumps({'episodes': [retained_episode(), retained_episode()]}))
    reads, read_text = [], type(path).read_text

    def counted(self, *args, **kwargs):
        reads.append(str(self))
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(path), 'read_text', counted)
    references = [{'file': 'sequence.json', 'json_path': ['episodes', index]} for index in (0, 1)]
    loaded = load_episode_sources(references, tmp_path)
    assert len(loaded) == 2 and reads == [str(path)]
    assert loaded[0]['episode']['task_id'] == 't01'
    assert loaded[0]['source_ref'] == references[0]


def repeated_t02_records():
    # Actual repeated task labels, positions and costs in the retained 17-Episode
    # cognition_v3 sequence; these tests exercise identity, not scientific fits.
    return [{'task_id': 't02', 'episode_index': index, 'status': 'completed',
        'questions': [], 'actions': [], 'observation_cost': observations, 'compute_cost': compute,
        'evaluation': {'task_metadata': {'phase': phase}, 'rows': 192}}
        for index, phase, observations, compute in (
            (1, 'acquire', 45, 23519), (16, 'regime_change', 37, 7140))]


def test_repeated_real_task_id_keeps_both_original_source_indices_and_costs(tmp_path):
    original = agent(tmp_path)
    original.theory_index.clear()
    saved = [{'source_ref': {**retained_source(), 'json_path': ['episodes', episode['episode_index']]},
              'episode': episode} for episode in repeated_t02_records()]
    result = extract_episodes(original, retained_episodes=saved)
    episodes = result['episodes']
    assert {episode['episode_id'] for episode in episodes} == {
        't02@meaning1@source:episodes[1]', 't02@meaning1@source:episodes[16]'}
    assert {episode['source_ref']['json_path'][-1] for episode in episodes} == {1, 16}
    assert {episode['costs']['observation_cost'] for episode in episodes} == {45, 37}
    assert len({episode['source_id'] for episode in episodes}) == 2


def test_loading_only_second_t02_retains_the_same_source_identity_as_full_member(tmp_path):
    first, second = repeated_t02_records()
    rows = [{'task_id': f'other{index}'} for index in range(17)]
    rows[1], rows[16] = first, second
    path = tmp_path / 'sequence.json'
    path.write_text(json.dumps({'episodes': rows}))
    refs = [{'file': 'sequence.json', 'json_path': ['episodes', index]} for index in (1, 16)]
    original = agent(tmp_path)
    original.theory_index.clear()
    both = extract_episodes(original, retained_episodes=load_episode_sources(refs, tmp_path))
    single = extract_episodes(original, retained_episodes=load_episode_sources(refs[1:], tmp_path))
    assert single['episodes'][0]['episode_id'] == 't02@meaning1@source:episodes[16]'
    assert single['episodes'][0] == next(episode for episode in both['episodes']
                                       if episode['source_ref']['json_path'][-1] == 16)


def test_ambiguous_t02_theory_stays_unbound_until_it_names_the_actual_source(tmp_path):
    original = agent(tmp_path)
    original.theory_index = {'unknown': {'id': 'unknown', 'version': 1, 'program': PROGRAM,
        'episode_ref': {'question_id': 't02', 'meaning_version': 1}}}
    saved = [{'source_ref': {**retained_source(), 'json_path': ['episodes', episode['episode_index']]},
              'episode': episode} for episode in repeated_t02_records()]
    result = extract_episodes(original, retained_episodes=saved)
    assert all(not episode['theory_ids'] for episode in result['episodes'])
    unbound = question_episode(result, 'unbound:unknown@meaning1')
    assert unbound['status'] == 'UNKNOWN'
    assert unbound['unresolved_scope']['reason'] == 'ambiguous_retained_episode'
    assert unbound['unbound_theory_refs'][0]['json_path'] == ['theory_index', 'unknown']
    original.theory_index['unknown']['program_source_ref'] = {
        **retained_source(), 'json_path': ['episodes', 16, 'program']}
    bound = extract_episodes(original, retained_episodes=saved)
    assert question_episode(bound, 't02@meaning1@source:episodes[16]')['theory_ids'] == ['unknown']
    assert question_episode(bound, 't02@meaning1@source:episodes[1]')['theory_ids'] == []
    assert not any(episode['episode_id'].startswith('unbound:') for episode in bound['episodes'])


def test_action_cutoff_retains_later_outcomes_without_attributing_their_loss_to_old_program(tmp_path):
    original, saved = agent(tmp_path), retained_episode()
    saved['actions'].append({'action': 'revise', 'details': {'operation': 'later_change',
        'before_program': PROGRAM, 'after_program': {'op': 'const', 'value': 99.}}})
    reference = {**retained_source(), 'action_cutoff': 2}
    result = extract_episodes(original, retained_episodes=[{'source_ref': reference, 'episode': saved}])
    current = question_episode(result, 't01@meaning1')
    assert len(current['actual_changes']) == 1
    assert current['actual_changes'][0]['action_ref']['json_path'] == ['episodes', 0, 'actions', 2]
    assert current['source_cutoff'] == {'action_index_inclusive': 2}
    assert current['costs'] == {'observation_cost': 3, 'compute_cost': 8}
    assert current['status'] == 'historical_prefix'
    assert current['independent_results'][0]['kind'] == 'later_final_evaluation'
    assert 'loss' not in current['independent_results'][0]
    assert current['retained_later_outcome_refs'][1]['json_path'] == ['episodes', 0, 'evaluation']


def test_resolve_actual_before_after_reuses_loaded_episode_without_a_second_source_read(tmp_path, monkeypatch):
    import hegel_machine.cognition.research_episodes as module
    original, saved = agent(tmp_path), retained_episode()
    reference = retained_source()
    refs = [{**reference, 'json_path': ['episodes', 0, 'actions', 2, 'details', name]}
            for name in ('before_program', 'after_program')]

    def no_second_read(references, workspace):
        assert references == []
        return []

    monkeypatch.setattr(module, 'load_source_values', no_second_read)
    values = resolve_program_sources(original, refs, [{'source_ref': reference, 'episode': saved}])
    assert values[0] == {'op': 'const', 'value': 0.}
    assert values[1] == PROGRAM


def test_small_source_reader_resolves_program_array_without_treating_it_as_an_episode(tmp_path):
    path = tmp_path / 'programs.json'
    path.write_text(json.dumps({'basis': [PROGRAM]}))
    result = load_source_values([{'file': 'programs.json', 'json_path': ['basis', 0]}], tmp_path)
    assert result == [{'source_ref': {'file': 'programs.json', 'json_path': ['basis', 0]},
                       'value': PROGRAM}]
