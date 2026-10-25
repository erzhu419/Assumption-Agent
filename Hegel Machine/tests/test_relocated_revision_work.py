"""Freezing a retained library must preserve only its unchanged search work."""
from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition import theory_consolidation as library
from test_theory_library_read_performance import metadata_recall_scene, save


@pytest.mark.parametrize('change', ['none', 'body', 'holes', 'constraints', 'trigger', 'version'])
def test_relocated_work_requires_the_same_complete_frozen_candidate(tmp_path, change):
    candidate = {'id': 'retained', 'version': 1, 'kind': 'program_revision_relation',
        'definition': {'name': 'retained', 'arguments': [], 'body': {'op': 'const', 'value': 1}},
        'holes': [], 'learned_constraints': [], 'trigger_checks': [{'before': {'failed': True}}]}
    original = tmp_path / 'original.json'
    frozen = tmp_path / 'frozen.json'
    changed = deepcopy(candidate)
    if change == 'body':
        changed['definition']['body']['value'] = 2
    elif change == 'holes':
        changed['holes'] = [{'name': 'different_parameter_slot'}]
    elif change == 'constraints':
        changed['learned_constraints'] = [{'op': 'different_constraint'}]
    elif change == 'trigger':
        changed['trigger_checks'][0]['before']['failed'] = False
    elif change == 'version':
        changed['version'] = 2
    save(original, {'candidates': {candidate['id']: candidate}})
    save(frozen, {'candidates': {candidate['id']: changed}})
    from types import SimpleNamespace
    agent = SimpleNamespace(workspace=tmp_path)
    old = {'library_path': str(original), 'candidate_id': candidate['id'], 'candidate_version': 1,
        'evidence_checks': [], 'allocation': {'compute': 80}, 'parent': {'id': 'parent', 'version': 1}}
    current = {**deepcopy(old), 'library_path': str(frozen), 'allocation': {'compute': 120}}
    before = deepcopy(old)
    assert library._same_relation_work(agent, old, current,
        source_libraries={}) is (change == 'none')
    assert old == before
    # A new matching name and version is insufficient without its actual source.
    assert not library._same_relation_work(agent, old, current)


def test_relocation_does_not_hide_a_changed_evidence_or_parent_contract(tmp_path):
    candidate = {'id': 'retained', 'version': 1, 'definition': {'body': 'unchanged'}}
    old = {'library_path': 'old.json', 'candidate_id': 'retained', 'candidate_version': 1,
        'evidence_checks': [], 'allocation': {'compute': 80}, 'parent': {'id': 'parent', 'version': 1}}
    cache = {path: {'candidates': {'retained': deepcopy(candidate)}} for path in ('old.json', 'new.json')}
    from types import SimpleNamespace
    agent = SimpleNamespace(workspace=tmp_path)
    for changed in ({'parent': {'id': 'parent', 'version': 2}},
                    {'fitting': {'operations': ['different_fit']}}):
        new = {**deepcopy(old), 'library_path': 'new.json', **changed}
        assert not library._same_relation_work(agent, old, new, source_libraries=cache)


def test_recall_continues_original_inner_and_outer_cursors_from_a_thin_frozen_view(
        tmp_path, monkeypatch):
    agent, question, original, counts = metadata_recall_scene(tmp_path, monkeypatch)
    calls = []

    def proposed(candidate, parent, components, **kwargs):
        cursor = kwargs['cursor']
        calls.append((candidate['id'], deepcopy(cursor)))
        return {'status': 'UNKNOWN', 'reason': 'bounded_fixture', 'proposals': [],
            'excluded': [], 'actual_compute_cost': 0, 'budget_exhausted': True,
            'search_completed': False, 'cursor': {
                'layer': 1, 'offset': (cursor or {}).get('offset', 0) + 1, 'pending_fits': []}}

    monkeypatch.setattr(revision_relations, 'propose_revision', proposed)
    arguments = {'library_path': original, 'allocation': {'compute': 100}}
    first = library.recall_frozen_theories(agent, question, arguments)
    assert [identifier for identifier, _ in calls] == ['first', 'second']
    first_basis = deepcopy(question['frozen_recall_execution']['basis'])
    original_basis = {key: deepcopy(value['request_basis'])
        for key, value in agent.theory_consolidation['executions'].items()}
    assert first['object_cursor']['next_candidate_id'] == 'second'
    calls.clear()
    frozen = tmp_path / 'new_frozen.json'
    save(frozen, {'source_ref': {'file': original, 'json_path': []}})
    reads = []
    original_read = library.load_source_values

    def read(references, workspace):
        reads.extend(row['file'] for row in references)
        return original_read(references, workspace)

    monkeypatch.setattr(library, 'load_source_values', read)
    repeated = library.recall_frozen_theories(agent, question,
        {**arguments, 'library_path': str(frozen)})
    assert [identifier for identifier, _ in calls] == ['second', 'first']
    assert all(cursor == {'layer': 1, 'offset': 1, 'pending_fits': []} for _, cursor in calls)
    assert repeated['object_cursor']['next_candidate_id'] == 'first'
    # The thin view already read this source. Both objects must share that read.
    assert reads.count(original) == 1
    assert reads.count(str(frozen)) == 1
    assert first_basis['library_path'] == original
    assert all(value['library_path'] == original for value in original_basis.values())
    assert json.loads((tmp_path / 'new_frozen.json').read_text()) == {
        'source_ref': {'file': original, 'json_path': []}}
    assert all('_source_libraries' not in execution['request_basis']
               for execution in agent.theory_consolidation['executions'].values())


def test_each_thin_layer_retains_its_own_constructor_for_relocation_checks(tmp_path):
    source, view = tmp_path / 'source.json', tmp_path / 'view.json'
    save(source, {'candidates': {'same': {'id': 'same', 'version': 1, 'definition': {'body': 'old'}}}})
    save(view, {'source_ref': {'file': str(source), 'json_path': []},
        'candidates': {'same': {'id': 'same', 'version': 1, 'definition': {'body': 'new'}}}})
    from types import SimpleNamespace
    agent = SimpleNamespace(workspace=tmp_path)
    cache = {}
    selected = library.read_theory_library(agent, str(view), source_libraries=cache)
    assert selected['candidates']['same']['definition']['body'] == 'new'
    assert cache[str(source)]['candidates']['same']['definition']['body'] == 'old'
    assert not library._same_library_candidate(agent, str(source), str(view), 'same', 1, cache)
