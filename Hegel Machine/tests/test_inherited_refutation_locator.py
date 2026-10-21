"""Inherited failed checks keep physical delta addresses with one lookup per transfer."""

from copy import deepcopy
import json

from hegel_machine.cognition import theory_consolidation as libraries
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.research_episodes import load_source_values
from test_question_routing import CONTEXT, theory
from test_theory_consolidation import agent_scene, assert_no_new_science


def test_inherited_failed_checks_locate_each_transfer_once_and_keep_delta_addresses(tmp_path, monkeypatch):
    original = tmp_path / 'source' / 'shared' / 'theory_library.json'
    original.parent.mkdir(parents=True)
    delta = tmp_path / 'delta' / 'shared' / 'theory_library.json'
    delta.parent.mkdir(parents=True)

    def failure(prefix, count):
        return {'theory_id': 'failed', 'theory_ref': {'id': 'failed', 'version': 1},
            'status': 'refuted', 'binding': {'context_ref': deepcopy(CONTEXT)},
            'checks': [{'status': 'failed', 'context_ref': deepcopy(CONTEXT),
                'source_ref': {'file': f'{prefix}.json', 'json_path': [index]}}
                for index in range(count)]}

    old_failure, delta_failure = failure('original_paid', 3), failure('delta_paid', 2)
    original.write_text(json.dumps({'transfers': [
        {'theory_id': 'unrelated', 'binding': {'context_ref': deepcopy(CONTEXT)}, 'checks': []},
        old_failure]}))
    delta.write_text(json.dumps({'source_ref': {'file': str(original), 'json_path': []},
                                 'transfers': [delta_failure]}))
    agent, lab = agent_scene(tmp_path / 'target')
    agent.theory_consolidation = libraries.install_theory_library(agent, {'file': str(delta), 'json_path': []})
    register_theories(agent, [theory('failed', compatible=False,
        support={'status': 'refuted', 'evidence': []}), theory('ordinary_alias')])

    def forbidden_prediction(*args, **kwargs):
        raise AssertionError('Resolving saved failures must not rerun a prediction')

    monkeypatch.setattr(Program, 'predict', forbidden_prediction)
    locator_calls, reads = [], []
    original_locator, original_load = libraries.library_entry_reference, libraries.load_source_values

    def locate(owner, field, key, *parts, **kwargs):
        locator_calls.append((field, key, parts))
        return original_locator(owner, field, key, *parts, **kwargs)

    def load(references, workspace):
        reads.extend(reference['file'] for reference in references)
        return original_load(references, workspace)

    monkeypatch.setattr(libraries, 'library_entry_reference', locate)
    monkeypatch.setattr(libraries, 'load_source_values', load)
    before = deepcopy(agent.costs)
    alias = next(item for item in resolved_theories(agent) if item['id'] == 'ordinary_alias')
    assert alias['compatible'] is False
    check_refs = [reference for reference in alias['scope_refutation']['counterexample_refs']
                  if reference.get('json_path', [])[:1] == ['transfers']]
    assert check_refs == [
        *({'file': str(original), 'json_path': ['transfers', 1, 'checks', index]} for index in range(3)),
        *({'file': str(delta), 'json_path': ['transfers', 0, 'checks', index]} for index in range(2))]
    assert all(item['value']['status'] == 'failed' for item in load_source_values(check_refs, agent.workspace))
    # The merged index of the delta transfer is 2; its physical index remains 0.
    assert locator_calls == [('transfers', 1, ()), ('transfers', 2, ())]
    assert reads == [str(delta), str(original)]
    again = next(item for item in resolved_theories(agent) if item['id'] == 'ordinary_alias')
    assert again['scope_refutation'] == alias['scope_refutation']
    assert reads == [str(delta), str(original), str(delta), str(original)]
    assert agent.costs == before
    assert_no_new_science(agent, lab)
