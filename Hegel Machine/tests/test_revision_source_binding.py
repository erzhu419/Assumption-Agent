"""An executed replacement belongs to its actual after Program, not its bank."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.theory_consolidation import _revision_entries


def episode(theory_id, before, after):
    return {'episode_id': 'source_episode:' + theory_id, 'source_id': 'actual_source',
        'theory_id': theory_id, 'role_signature': {'u': {'role': 'control', 'unit': 'V', 'time': 'current'}},
        'output_unit': 'V', 'context_ref': {'source_id': 'actual_source'},
        'actual_changes': [{'kind': 'retained_program_replaced', 'before_ref': before,
                            'source_ref': after, 'action_ref': {'file': 'training.json',
                                'json_path': ['method_results', 11, 'feedback', 'evaluations', 3,
                                              'retention_decision']}}]}


@pytest.mark.parametrize('absolute_declaration', [False, True])
def test_native_replacement_matches_only_its_exact_after_cell_across_anchored_refs(tmp_path, absolute_declaration):
    before = Program('feature', name='u').to_dict()
    after = Program('add', children=(Program('feature', name='u'), Program('const', value=1.))).to_dict()
    questions = [{'id': 'source', 'native_condition_research': [
        {'candidates': [{'program': before}, {'program': after}]}]}]
    (tmp_path / 'research_state.json').write_text(json.dumps({'questions': questions}))
    agent = SimpleNamespace(workspace=tmp_path, questions=questions, revisions=[], theory_index={})
    base = ['questions', 0, 'native_condition_research', 0, 'candidates']
    actual_file = 'research_state.json' if absolute_declaration else str(tmp_path / 'research_state.json')
    declared_file = str(tmp_path / 'research_state.json') if absolute_declaration else 'research_state.json'
    before_ref = {'file': actual_file, 'json_path': [*base, 0, 'program']}
    after_ref = {'file': actual_file, 'json_path': [*base, 1, 'program']}
    pool = {identifier: {'program_source_ref': {'file': declared_file,
        'field': f'questions[0].native_condition_research[0].candidates[{index}].program'}}
        for index, identifier in enumerate(('earlier_native', 'actual_after'))}
    sources = [episode(identifier, before_ref, after_ref) for identifier in pool]
    original = deepcopy(sources)
    entries = _revision_entries(agent, sources, pool, retained=[])
    revision, = entries
    assert revision['theory_id'] == 'actual_after'
    assert revision['before_program'] == before and revision['after_program'] == after
    assert revision['before_ref'] == before_ref and revision['after_ref'] == after_ref
    assert revision['action_ref'] == sources[1]['actual_changes'][0]['action_ref']
    assert sources == original


def test_old_action_program_sources_and_unspecified_sources_keep_their_actual_revisions(tmp_path):
    before = Program('feature', name='u').to_dict()
    after = Program('const', value=2.).to_dict()
    payload = {'episodes': [{'actions': [{'details': {'before_program': before, 'after_program': after}}]}]}
    (tmp_path / 'retained.json').write_text(json.dumps(payload))
    agent = SimpleNamespace(workspace=tmp_path, questions=[], revisions=[], theory_index={})
    base = ['episodes', 0, 'actions', 0, 'details']
    before_ref = {'file': 'retained.json', 'json_path': [*base, 'before_program']}
    after_ref = {'file': 'retained.json', 'json_path': [*base, 'after_program']}
    pool = {'historical': {'program_source_ref': {
        'file': str(tmp_path / 'retained.json'), 'json_path': [*base, 'after_program']}}, 'unspecified': {}}
    sources = [episode(identifier, before_ref, after_ref) for identifier in pool]
    entries = _revision_entries(agent, sources, pool, retained=[])
    assert [entry['theory_id'] for entry in entries] == ['historical', 'unspecified']
    assert all(entry['before_program'] == before and entry['after_program'] == after for entry in entries)
