"""Native authoring exposes actual typed bank cells to the existing theory routes."""
from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.research_episodes import extract_episodes
from hegel_machine.cognition.research_episodes import resolve_program_sources
from hegel_machine.cognition.structural_inquiry import _register_native_theories
from hegel_machine.cognition.theory_consolidation import consolidate_research
from test_native_condition_inquiry import (
    COMPOSITION, PaidConditionalLab, buy_fixture, execute, prepared,
)


CONTEXT = {'source_id': 'native-registration-fixture', 'domain': 'synthetic-controls'}
TARGET = {'role_signature': {
    'drive': {'role': 'drive_control', 'unit': '1', 'time': 'ordered_reset_trajectory'},
    'switch': {'role': 'switch_control', 'unit': '1', 'time': 'ordered_reset_trajectory'}},
    'output_unit': '1', 'context_ref': CONTEXT,
    'support': {'status': 'supported', 'independent_confirmation': True}}


def test_common_native_generation_registers_real_bank_cells_without_new_science(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    agent.provider = None
    buy_fixture(agent, lab)
    before = deepcopy(agent.costs)

    def no_acquisition(*args, **kwargs):
        raise AssertionError('Native registration must not purchase observations')

    monkeypatch.setattr(lab, 'execute', no_acquisition)
    question = {**COMPOSITION, 'context_ref': CONTEXT, 'theory_target': TARGET}
    feedback = execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'research_question': question, 'max_compute': 500_000, 'max_candidates': 2,
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}]})
    assert 'error' not in feedback
    registration = feedback['native_theory_registration']
    assert registration['status'] == 'registered'
    assert registration['registered']
    assert registration['new_compute_cost'] == 0
    assert registration['support_updated'] is False
    current = next(q for q in agent.questions if q['id'] == COMPOSITION['id'])
    bank = current['native_condition_research'][0]
    assert len(registration['registered']) == len(bank['candidates'])
    resolved = {theory['id']: theory for theory in resolved_theories(agent)}
    for index, item in enumerate(registration['registered']):
        theory = agent.theory_index[item['theory_id']]
        assert item['source_ref'] == {'kind': 'native_candidate',
            'question_id': COMPOSITION['id'], 'meaning_version': 1,
            'round_index': 0, 'candidate_index': index}
        assert item['program_source_ref']['json_path'][-5:] == [
            'native_condition_research', 0, 'candidates', index, 'program']
        assert resolved[item['theory_id']]['program'] == bank['candidates'][index]['program']
        assert theory['support'] == {'status': 'candidate', 'independent_confirmation': False}
        assert 'program' not in theory and 'rows' not in theory
        assert 'fit_mse' not in theory and 'parameter_paths' not in theory
        assert (tmp_path / bank['candidates'][index]['model']).exists()
    intake = extract_episodes(agent)
    episode = next(e for e in intake['episodes'] if e['episode_id'] == COMPOSITION['id'] + '@meaning1')
    assert set(episode['theory_ids']) == set(agent.theory_index)
    assert episode['source_id'] == CONTEXT['source_id']
    assert agent.costs['compute_cost'] - before['compute_cost'] == bank['compute_cost']
    assert agent.costs['observation_cost'] == before['observation_cost']
    assert agent.costs['model_calls'] == 0 and not agent.confirmations
    saved = json.loads((tmp_path / 'research_state.json').read_text())
    assert set(saved['theory_index']) == set(agent.theory_index)


def _retained_scene(tmp_path, *, target=None):
    agent = prepared(tmp_path)
    question = agent._open_question({**COMPOSITION,
        **({'theory_target': deepcopy(target)} if target is not None else {})})
    candidate = {'program': Program('feature', name='drive').to_dict(),
        'fit_mse': 3., 'fit_evidence': [0], 'status': 'refuted',
        'counterexample_refs': [{'file': 'laboratory_observations.json', 'json_path': [0]}]}
    question['native_condition_research'] = [{'status': 'search_incomplete', 'compute_cost': 17,
        'candidates': [candidate]}]
    return agent, question, agent.questions.index(question)


def test_missing_native_contract_is_author_work_and_can_be_completed_without_refitting(tmp_path):
    agent, question, index = _retained_scene(tmp_path)
    before = deepcopy(agent.costs)
    first = _register_native_theories(agent, question, index, 0)
    assert first['status'] == 'author_work_request'
    request = first['author_work_request']
    assert request['object_type'] == 'native_theory_contract'
    required = request['required_contract']['candidates'][0]['required']
    assert set(required) == {'output_unit', 'context_ref', 'role_signature.drive.role',
                             'role_signature.drive.unit', 'role_signature.drive.time'}
    assert not agent.theory_index
    question['theory_target'] = deepcopy(TARGET)
    question['theory_target']['role_signature']['drive'].pop('time')
    partial = _register_native_theories(agent, question, index, 0)
    assert partial['author_work_request']['required_contract']['candidates'][0][
        'required'] == ['role_signature.drive.time']
    question['theory_target'] = deepcopy(TARGET)
    complete = _register_native_theories(agent, question, index, 0)
    assert complete['status'] == 'registered'
    assert agent.costs == before
    assert question['native_condition_research'][0]['candidates'][0]['fit_mse'] == 3.


def test_native_registration_retains_refutation_and_is_idempotent(tmp_path):
    agent, question, index = _retained_scene(tmp_path, target=TARGET)
    first = _register_native_theories(agent, question, index, 0)
    second = _register_native_theories(agent, question, index, 0)
    assert second == first
    item, = first['registered']
    theory = agent.theory_index[item['theory_id']]
    assert theory['version'] == 1 and 'history' not in theory
    assert theory['support']['status'] == 'refuted'
    assert theory['counterexample_refs'] == question['native_condition_research'][0][
        'candidates'][0]['counterexample_refs']
    episode = next(e for e in extract_episodes(agent)['episodes']
                   if e['episode_id'] == COMPOSITION['id'] + '@meaning1')
    assert episode['failure_refs'] == [{'file': 'research_state.json', 'json_path': [
        'questions', index, 'native_condition_research', 0]}]
    resolved, = resolved_theories(agent)
    assert resolved['compatible'] is False
    assert resolved['source_status'] == 'refuted'
    assert resolved['scope_refutation']['counterexample_refs']
    assert theory['role_signature'] == {'drive': TARGET['role_signature']['drive']}


def test_changed_question_meaning_keeps_original_native_program_and_frozen_inverse_reference(tmp_path):
    agent = prepared(tmp_path)
    identifiers, programs = [], []
    for index, slope in enumerate((2., 4.)):
        context = {**CONTEXT, 'source_id': f'original-native-source-{index}'}
        target = {**deepcopy(TARGET), 'context_ref': context,
                  'context_family_ref': 'declared_synthetic_family'}
        question = agent._open_question({**COMPOSITION, 'id': f'local-{index}',
            'context_ref': context, 'theory_target': target})
        program = Program('add', children=(Program('mul', children=(
            Program('const', value=slope), Program('feature', name='drive'))),
            Program('const', value=1.)))
        programs.append(program.to_dict())
        question['native_condition_research'] = [{'status': 'candidates_require_evaluation',
            'compute_cost': 17, 'fit_record_indices': [], 'candidates': [
                {'program': program.to_dict(), 'fit_mse': 3., 'fit_evidence': []}]}]
        registered = _register_native_theories(agent, question, agent.questions.index(question), 0)
        identifiers.append(registered['registered'][0]['theory_id'])
    before = deepcopy(agent.theory_index[identifiers[0]])
    new = agent._open_question({'id': 'local-0', 'context_ref': {'source_id': 'new-meaning'}})
    assert new['meaning_version'] == 2 and 'native_condition_research' not in new
    resolved = {item['id']: item for item in resolved_theories(agent)}
    original = resolved[identifiers[0]]
    assert original['program'] == programs[0]
    path = original['program_source_ref']['json_path']
    assert path[:4] == ['questions', agent.questions.index(new), 'meaning_history', 0]
    assert resolve_program_sources(agent, [original['program_source_ref']]) == [programs[0]]
    assert agent.theory_index[identifiers[0]] == before  # Persistent identity remains unchanged.
    result = consolidate_research(agent, {'theme': 'original-native-family',
        'source_theory_ids': identifiers,
        'mission': {'purpose': 'theory_learning',
                    'theory_learning_allocation': {'compute': 20_000}},
        'allocation': {'compute': 20_000, 'max_groups': 4}})
    assert result['frozen_candidates']
    library = json.loads((tmp_path / result['library_path']).read_text())
    frozen = library['candidates'][result['frozen_candidates'][0]['candidate_id']]
    assert all(item['exact'] for item in frozen['source_reconstruction'])
    assert any(item['source_ref'] == original['program_source_ref']
               for item in frozen['instances'])
