"""Routing metadata reuses the installed source without hiding real changes."""

from copy import deepcopy
import json
import os
from types import SimpleNamespace

from hegel_machine.cognition import question_routing
from hegel_machine.cognition import theory_consolidation as library
from test_question_routing import future_question
from test_theory_consolidation import agent_scene


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def views(tmp_path):
    source = tmp_path / 'source/shared/theory_library.json'
    candidates = {identifier: {'id': identifier, 'version': 1}
                  for identifier in ('old_b', 'old_a', 'old_alias', 'new_b', 'new_a')}
    save(source, {'candidates': candidates, 'transfers': [{'status': 'refuted', 'cost': 19}]})
    full, old = tmp_path / 'full.json', tmp_path / 'old.json'
    save(full, {'source_ref': {'file': str(source), 'json_path': []}})
    save(old, {'source_ref': {'file': str(source), 'json_path': []},
               'removed_candidates': ['new_b', 'new_a']})
    return source, full, old, candidates


def install(agent, path):
    agent.theory_consolidation = library.install_theory_library(
        agent, {'file': str(path), 'json_path': []})


def test_completion_refresh_reads_only_installed_ordered_roster(tmp_path, monkeypatch):
    source, full, old, candidates = views(tmp_path)
    agent, _ = agent_scene(tmp_path / 'target')
    install(agent, old)
    # New local work and aliases do not alter the frozen source's availability.
    agent.theory_consolidation['candidates']['local'] = {'version': 9}
    expected = [{'id': key, 'version': 1} for key in ('old_b', 'old_a', 'old_alias')]

    def forbidden(*args, **kwargs):
        raise AssertionError('routing metadata must not reopen the complete installed library')

    monkeypatch.setattr(library, 'load_source_values', forbidden)
    for index in range(5):
        question = agent._open_question(future_question(agent, identifier=f'future_{index}'))
        question['frozen_library_path'] = str(old)
        basis = question_routing._basis(agent, question, {})
        assert basis['frozen_library'] == expected
        question['routing_execution'] = {
            'basis': basis, 'result': {'status': 'selected'}, 'allowance': 100}
    question_routing._refresh_completion(agent, {}, 100)
    question_routing._refresh_completion(agent, {}, 100)
    assert all(question.get('completed_agenda_basis') for question in agent.questions[1:])
    # A consumer's annotations cannot mutate the next question's roster.
    basis['frozen_library'][0]['version'] = 33
    assert library.theory_library_candidate_versions(agent, str(old)) == expected
    assert library.theory_library_candidate_versions(agent) == [
        *expected, {'id': 'local', 'version': 9}]


def test_different_view_and_reinstallation_preserve_full_order_and_versions(tmp_path, monkeypatch):
    source, full, old, candidates = views(tmp_path)
    agent = SimpleNamespace(workspace=tmp_path / 'target')
    install(agent, old)
    reads = []
    original = library.load_source_values

    def read(references, workspace):
        reads.extend(row['file'] for row in references)
        return original(references, workspace)

    monkeypatch.setattr(library, 'load_source_values', read)
    assert [item['id'] for item in library.theory_library_candidate_versions(agent, str(full))] == list(candidates)
    assert reads == [str(full), str(source)]
    install(agent, full)
    reads.clear()
    assert library.theory_library_candidate_versions(agent, str(full)) == [
        {'id': key, 'version': 1} for key in candidates]
    assert reads == []
    candidates['old_b']['version'] = 2
    save(source, {'candidates': candidates})
    install(agent, full)
    reads.clear()
    assert library.theory_library_candidate_versions(agent, str(full))[0] == {'id': 'old_b', 'version': 2}
    assert reads == []


def test_changed_source_or_thin_view_is_visible_without_reinstall(tmp_path, monkeypatch):
    source, full, old, candidates = views(tmp_path)
    agent = SimpleNamespace(workspace=tmp_path / 'target')
    install(agent, old)
    original_base = deepcopy(agent._theory_library_base)
    reads = []
    original = library.load_source_values

    def read(references, workspace):
        reads.extend(row['file'] for row in references)
        return original(references, workspace)

    monkeypatch.setattr(library, 'load_source_values', read)
    # Atomic replacement with an equal-size new version is an actual host-writer case.
    before = source.stat()
    candidates['old_b']['version'] = 2
    replacement = source.with_suffix('.writing')
    save(replacement, {'candidates': candidates, 'transfers': [{'status': 'refuted', 'cost': 19}]})
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
    replacement.replace(source)
    assert source.stat().st_size == before.st_size
    assert library.theory_library_candidate_versions(agent, str(old))[0] == {'id': 'old_b', 'version': 2}
    assert reads == [str(old), str(source)]
    reads.clear()
    assert library.theory_library_candidate_versions(agent, str(old))[0]['version'] == 2
    assert reads == []
    save(old, {'source_ref': {'file': str(source), 'json_path': []},
               'removed_candidates': ['old_b', 'new_b', 'new_a'],
               'candidates': {'view_local': {'version': 4}}})
    assert library.theory_library_candidate_versions(agent, str(old)) == [
        {'id': 'old_a', 'version': 1}, {'id': 'old_alias', 'version': 1},
        {'id': 'view_local', 'version': 4}]
    assert reads == [str(old), str(source)]
    assert agent._theory_library_base == original_base
    assert agent.theory_consolidation == original_base


def test_direct_working_library_changes_are_immediately_visible(tmp_path):
    agent = SimpleNamespace(workspace=tmp_path, theory_consolidation={
        'candidates': {'first': {'version': 1}, 'alias': {'version': 1}}})
    assert library.theory_library_candidate_versions(agent) == [
        {'id': 'first', 'version': 1}, {'id': 'alias', 'version': 1}]
    agent.theory_consolidation['candidates']['first']['version'] = 2
    agent.theory_consolidation['candidates'].pop('alias')
    agent.theory_consolidation['candidates']['new'] = {'version': 1}
    assert library.theory_library_candidate_versions(agent) == [
        {'id': 'first', 'version': 2}, {'id': 'new', 'version': 1}]


def test_legacy_installed_candidate_keeps_default_version_without_rewriting_source(tmp_path):
    source = tmp_path / 'legacy.json'
    document = {'candidates': {'old': {'id': 'old'}, 'new': {'id': 'new', 'version': 2}}}
    save(source, document)
    agent = SimpleNamespace(workspace=tmp_path / 'target')
    install(agent, source)
    expected = [{'id': 'old', 'version': 1}, {'id': 'new', 'version': 2}]
    assert library.theory_library_candidate_versions(agent, str(source)) == expected
    assert library.theory_library_candidate_versions(agent) == expected
    assert json.loads(source.read_text()) == agent._theory_library_base == document
