"""Frozen index Program references retain complete snapshot semantics."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from hegel_machine.cognition.checkpoint_storage import _json_values_equal
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.theory_consolidation import (
    INDEX_PATH, install_theory_index, save_theory_index,
)


def program(offset=0.):
    value = Program('feature', name='u')
    for index in range(12):
        value = Program('add', children=(value, Program('const', value=index + offset)))
    return value.to_dict()


def scene(path, indexed):
    path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(workspace=path, theory_index=indexed, questions=[],
        relation_research={}, events=[], files={})


def declaration(identifier, value, reference, version=1):
    return {'id': identifier, 'version': version,
        'source_snapshot': {'source_ref': {'kind': 'native_candidate',
            'question_id': 'old_task', 'round_index': 0, 'candidate_index': 0},
            'version': version},
        'program_source_ref': deepcopy(reference), 'program': deepcopy(value),
        'support': {'status': 'refuted', 'evidence': [{'record_index': 7}]},
        'scope_refutation': {'counterexample_refs': [{'record_index': 7}]},
        'costs': {'compute_cost': 31, 'model_calls': 0},
        'failed': ['original failure']}


def test_saved_index_program_aliases_and_history_survive_source_change_and_move(tmp_path):
    source = tmp_path / 'old_source.json'
    source.write_text(json.dumps({'program': program()}))
    reference = {'file': str(source), 'json_path': ['program']}
    current = declaration('original', program(), {**reference, 'version': 2}, version=2)
    current['history'] = [declaration('original', program(), {**reference, 'version': 1})]
    unaddressed = declaration('unaddressed', program(99.), reference, version=2)
    unaddressed.pop('program_source_ref')
    unaddressed['history'] = [declaration('unaddressed', program(), reference)]
    unaddressed['history'][0].pop('program_source_ref')
    indexed = {'original': current,
        'alias': declaration('alias', program(), {**reference, 'version': 3}, version=3),
        'unaddressed': unaddressed}
    expected = deepcopy(indexed)
    agent = scene(tmp_path / 'writer', indexed)
    save_theory_index(agent)
    assert _json_values_equal(agent.theory_index, expected)
    saved = agent.workspace / INDEX_PATH
    raw = json.loads(saved.read_text())
    assert 'program' in raw['original']
    assert raw['original']['history'][0]['program_ref'] == {
        'json_path': ['original', 'program']}
    assert raw['alias']['program_ref'] == {'json_path': ['original', 'program']}
    assert raw['unaddressed']['history'][0]['program_ref'] == {
        'json_path': ['original', 'program']}
    assert 'program_source_ref' not in raw['unaddressed']['history'][0]
    assert [(item['version'], item['program_source_ref']['version'],
             item['source_snapshot']['version'])
        for item in (raw['original'], raw['original']['history'][0], raw['alias'])] == [
            (2, 2, 2), (1, 1, 1), (3, 3, 3)]
    assert list(raw['alias']) == [
        'program_ref' if key == 'program' else key for key in expected['alias']]
    # A frozen Program must not depend on a later edit at its provenance address.
    source.write_text(json.dumps({'program': program(100.)}))
    source.unlink()
    moved = tmp_path / 'moved/shared/theory_index.json'
    moved.parent.mkdir(parents=True)
    shutil.move(str(saved), str(moved))
    installed = install_theory_index(scene(tmp_path / 'recipient', {}),
        {'file': str(moved), 'json_path': []})
    assert _json_values_equal(installed, expected)
    references = [{'file': str(moved), 'json_path': ['alias', 'program']},
        {'file': str(moved), 'json_path': ['original', 'history', 0, 'program']}]
    assert all(item['value'] == program() for item in load_source_values(references, tmp_path))
    installed['alias']['program'].clear()
    assert installed['original']['program'] == program()
    assert installed['original']['history'][0]['program'] == program()


def test_saved_index_compares_full_ordered_program_values_across_sources(tmp_path):
    reference = {'file': str(tmp_path / 'original.json'), 'json_path': ['program']}
    original = program()
    reordered = dict(reversed(list(original.items())))
    indexed = {
        'original': declaration('original', original, reference),
        'changed': declaration('changed', program(10.), reference),
        'reordered': declaration('reordered', reordered, reference),
        'other_source': declaration('other_source', original,
            {**reference, 'json_path': ['another_program']}),
    }
    agent = scene(tmp_path / 'writer', deepcopy(indexed))
    save_theory_index(agent)
    raw = json.loads((agent.workspace / INDEX_PATH).read_text())
    assert all('program' in raw[key] and 'program_ref' not in raw[key]
               for key in ('original', 'changed', 'reordered'))
    assert raw['other_source']['program_ref'] == {'json_path': ['original', 'program']}
    installed = install_theory_index(scene(tmp_path / 'recipient', {}),
        {'file': str(agent.workspace / INDEX_PATH), 'json_path': []})
    assert _json_values_equal(installed, indexed)


def test_saved_delta_index_keeps_base_removals_failures_and_local_program_addresses(tmp_path):
    reference = {'file': str(tmp_path / 'source.json'), 'json_path': ['program']}
    base_agent = scene(tmp_path / 'base', {'removed': declaration('removed', program(), reference)})
    save_theory_index(base_agent)
    base_path = base_agent.workspace / INDEX_PATH
    base_bytes = base_path.read_bytes()
    target = scene(tmp_path / 'target', {})
    target.theory_index = install_theory_index(target, {'file': str(base_path), 'json_path': []})
    target.theory_index.pop('removed')
    target.theory_index.update({'local': declaration('local', program(), reference),
        'alias': declaration('alias', program(), reference)})
    expected = deepcopy(target.theory_index)
    save_theory_index(target)
    path = target.workspace / INDEX_PATH
    raw = json.loads(path.read_text())
    assert raw['removed_entries'] == ['removed']
    assert raw['entries']['alias']['program_ref'] == {
        'json_path': ['entries', 'local', 'program']}
    installed = install_theory_index(scene(tmp_path / 'recipient', {}),
        {'file': str(path), 'json_path': []})
    assert _json_values_equal(installed, expected)
    assert base_path.read_bytes() == base_bytes
    loaded, = load_source_values([{'file': str(path),
        'json_path': ['entries', 'alias', 'program']}], tmp_path)
    assert loaded['value'] == program()
