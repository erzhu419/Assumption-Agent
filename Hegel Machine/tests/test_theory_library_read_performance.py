"""Reuse work within one actual library read without stale or shared results."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import theory_consolidation as library


def test_anchor_resolves_repeated_paths_once_but_keeps_each_record_independent(tmp_path, monkeypatch):
    calls = []
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve)
    reference = {'file': 'evidence.json', 'field': 'records[0]', 'note': {'failure': ['kept']}}
    values = [deepcopy(reference) for _ in range(100)]
    result = library._anchor_file_references(values, tmp_path)
    assert calls == [str(tmp_path / 'evidence.json')]
    assert all(item['file'] == str(tmp_path / 'evidence.json')
               and item['json_path'] == ['records', 0] and 'field' not in item for item in result)
    result[0]['note']['failure'].append('local')
    assert result[1]['note']['failure'] == reference['note']['failure'] == ['kept']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_install_reads_each_selected_layer_once_and_never_reuses_stale_or_mutable_state(tmp_path, monkeypatch):
    base = tmp_path / 'source/shared/theory_library.json'
    frozen = tmp_path / 'frozen/shared/theory_library.json'
    source = {'candidates': {'old': {'version': 1}}, 'episodes': [],
              'transfers': [{'theory_id': 'failed', 'status': 'refuted', 'cost': 7,
                  'checks': [{'source_ref': {'file': 'paid.json', 'json_path': [0]}, 'status': 'failed'}]}]}
    save(base, source)
    save(frozen, {'source_ref': {'file': str(base), 'json_path': []},
                  'candidates': {'new': {'version': 1}}, 'conflicts': [{'reason': 'kept'}]})
    calls = []
    original = library.load_source_values

    def read(references, workspace):
        calls.extend(str(row['file']) for row in references)
        return original(references, workspace)

    monkeypatch.setattr(library, 'load_source_values', read)
    agent = SimpleNamespace(workspace=tmp_path / 'target')
    result = library.install_theory_library(agent, {'file': str(frozen), 'json_path': []})
    assert calls == [str(frozen), str(base)]
    assert list(result['candidates']) == ['old', 'new']
    assert result['transfers'][0]['checks'][0]['source_ref']['file'] == str(base.parent.parent / 'paid.json')
    result['transfers'][0]['cost'] = 99
    assert agent._theory_library_base['transfers'][0]['cost'] == 7
    source['candidates']['old']['version'] = 2
    save(base, source)
    again = library.install_theory_library(agent, {'file': str(frozen), 'json_path': []})
    assert again['candidates']['old']['version'] == 2
    assert again['transfers'][0]['cost'] == 7


def test_local_delta_and_direct_local_library_preserve_their_original_contract(tmp_path):
    base = tmp_path / 'base.json'
    save(base, {'candidates': {'old': {'version': 1}}, 'transfers': []})
    local = tmp_path / library.LIBRARY_PATH
    save(local, {'source_ref': {'file': str(base), 'json_path': []},
                 'removed_candidates': ['old'], 'candidates': {'after': {'version': 1}},
                 'conflicts': [{'source_ref': {'file': 'failed.json', 'json_path': [0]}}]})
    agent = SimpleNamespace(workspace=tmp_path)
    result = library.install_theory_library(agent)
    assert list(result['candidates']) == ['after']
    assert list(agent._theory_library_base['candidates']) == ['old']
    assert result['conflicts'][0]['source_ref']['file'] == str(tmp_path / 'failed.json')
    original = {'candidates': {}, 'source_record': {'file': 'ordinary.json', 'json_path': []}}
    save(local, original)
    assert library.install_theory_library(agent) == original
    assert agent._theory_library_base == {} and agent._theory_library_source_ref is None


def metadata_recall_scene(tmp_path, monkeypatch, *, registered_sibling=True):
    from hegel_machine.cognition import question_routing, revision_relations
    from hegel_machine.cognition.language import Program
    from test_question_routing import CONTEXT, ROLES, theory
    from test_theory_consolidation import agent_scene

    agent, _ = agent_scene(tmp_path / 'work')
    parent, sibling = Program('feature', name='u'), Program('const', value=1.)
    owner = agent._open_question({'id': 'native_bank', 'question': 'Retained metadata.',
        'context_ref': deepcopy(CONTEXT), 'native_condition_research': [{'candidates': [
            {'program': parent.to_dict()}, {'program': sibling.to_dict()}],
            'native_theory_registration': {'registered': [{'theory_id': 'sibling', 'version': 1}]}}]})
    reference = {'kind': 'native_candidate', 'question_id': owner['id'],
                 'meaning_version': 1, 'round_index': 0}
    declarations = [theory('parent', parent, source_ref={**reference, 'candidate_index': 0})]
    if registered_sibling:
        declarations.append(theory('sibling', sibling, source_ref={**reference, 'candidate_index': 1}))
    question_routing.register_theories(agent, declarations)
    candidates = {name: {'id': name, 'kind': 'program_revision_relation', 'version': 1,
                       'output_unit': 'V', 'fixed_role_signature': {}} for name in ('first', 'second')}
    agent.theory_consolidation = {'candidates': deepcopy(candidates)}
    library._save(agent)
    frozen = tmp_path / 'frozen.json'
    save(frozen, {'candidates': candidates})
    question = agent._open_question({'id': 'metadata_recall', 'question': 'Reuse metadata only.',
        'theory_target': {'role_signature': deepcopy(ROLES), 'output_unit': 'V',
            'context_ref': deepcopy(CONTEXT), 'parent_theory_id': 'parent', 'evidence_checks': []}})
    counts = {'resolved': 0, 'proposal': 0, 'library_reads': 0, 'candidate_refs': 0}
    original_resolved, original_read, original_ref = (
        question_routing.resolved_theories, library.read_theory_library, library.library_entry_reference)

    def resolved(*args, **kwargs):
        counts['resolved'] += 1
        return original_resolved(*args, **kwargs)

    def read(*args, **kwargs):
        counts['library_reads'] += 1
        return original_read(*args, **kwargs)

    def entry_ref(*args, **kwargs):
        counts['candidate_refs'] += int(args[1] == 'candidates')
        return original_ref(*args, **kwargs)

    def proposed(candidate, parent, components, **kwargs):
        counts['proposal'] += 1
        return {'status': 'UNKNOWN', 'reason': 'metadata_only_fixture', 'proposals': [],
            'excluded': [], 'actual_compute_cost': 0, 'budget_exhausted': False,
            'search_completed': True}

    def forbidden(*_args, **_kwargs):
        pytest.fail('metadata reuse must not predict or fit')

    monkeypatch.setattr(Program, 'predict', forbidden)
    monkeypatch.setattr(revision_relations, 'propose_revision', proposed)
    monkeypatch.setattr(question_routing, 'resolved_theories', resolved)
    monkeypatch.setattr(library, 'read_theory_library', read)
    monkeypatch.setattr(library, 'library_entry_reference', entry_ref)
    return agent, question, str(frozen), counts


def test_real_inner_cache_hits_reuse_current_recall_pool_and_frozen_library(tmp_path, monkeypatch):
    agent, question, frozen, counts = metadata_recall_scene(tmp_path, monkeypatch)
    inputs = {'library_path': frozen, 'allocation': {'compute': 100}}
    first = library.recall_frozen_theories(agent, question, inputs)
    assert first['search_completed'] and counts['resolved'] == 2
    assert counts['proposal'] == counts['candidate_refs'] == 2
    # A changed outer target annotation requires reconsidering the route, while
    # each transfer still receives the exact same executable problem.
    question['theory_target']['parent_binding_evidence'] = {'note': 'same_retained_parent'}
    counts.update({key: 0 for key in counts})
    before = deepcopy(agent.costs)
    # Both objects receive no more than their already completed allowance;
    # UNKNOWN with a larger allowance intentionally requests new work.
    again = library.recall_frozen_theories(agent, question,
        {**inputs, 'allocation': {'compute': 40}})
    assert not again['cache_hit'] and again['search_completed']
    assert counts == {'resolved': 1, 'proposal': 0, 'library_reads': 1, 'candidate_refs': 0}
    assert [entry['candidate_id'] for entry in again['work']] == ['second', 'first']
    assert agent.costs['compute_cost'] - before['compute_cost'] == again['new_compute_cost'] == 6
    assert agent.costs['python_execution_seconds'] == before['python_execution_seconds']
    assert all('_resolved_pool' not in execution['request_basis']
               and '_source_library' not in execution['request_basis']
               for execution in agent.theory_consolidation['executions'].values())


@pytest.mark.parametrize('registered_sibling', [True, False])
def test_native_sibling_reuse_avoids_second_resolution_but_new_sibling_stays_visible(
        tmp_path, monkeypatch, registered_sibling):
    from hegel_machine.cognition import revision_relations

    agent, question, frozen, counts = metadata_recall_scene(tmp_path, monkeypatch,
        registered_sibling=registered_sibling)
    original = revision_relations.propose_revision
    consumed = []

    def proposed(candidate, parent, components, **kwargs):
        consumed.extend(component['id'] for component in components)
        return original(candidate, parent, components, **kwargs)

    monkeypatch.setattr(revision_relations, 'propose_revision', proposed)
    first_candidate = next(iter(agent.theory_consolidation['candidates']))
    inputs = {**deepcopy(question['theory_target']), 'library_path': frozen,
        'candidate_id': first_candidate, 'theory_id': 'metadata_transfer',
        'allocation': {'compute': 100}, 'component_theory_ids': []}
    result = library.transfer_theory(agent, inputs)
    assert counts['resolved'] == (1 if registered_sibling else 2)
    identifier, = result['retained_native_component_ids']
    if registered_sibling:
        assert identifier == 'sibling'
    else:
        assert identifier not in {'parent', 'sibling'} and identifier in agent.theory_index
    assert consumed == [identifier]
    assert counts['proposal'] == 1
    assert result['new_compute_cost'] == 0 and agent.costs['python_execution_seconds'] == 0


def test_non_cached_transfer_invalidates_shared_pool_for_next_object(tmp_path, monkeypatch):
    from hegel_machine.cognition import question_routing, revision_relations
    from hegel_machine.cognition.language import Program
    from test_question_routing import theory

    agent, question, frozen, counts = metadata_recall_scene(tmp_path, monkeypatch)
    original = revision_relations.propose_revision
    seen = []

    def add_retained_component(candidate, parent, components, **kwargs):
        seen.append(candidate['id'])
        if len(seen) == 1:
            question_routing.register_theories(agent, [theory('new_retained', Program('const', value=2.))])
        else:
            assert 'new_retained' in agent.theory_index
        return original(candidate, parent, components, **kwargs)

    monkeypatch.setattr(revision_relations, 'propose_revision', add_retained_component)
    result = library.recall_frozen_theories(agent, question,
        {'library_path': frozen, 'allocation': {'compute': 100}})
    assert seen == ['first', 'second'] and counts['resolved'] == 2
    # Recall's component roster remains frozen; new proposals never recursively
    # enter the same pass's search. The fresh full pool still observes them.
    assert result['new_compute_cost'] == 6 and 'new_retained' in agent.theory_index
