"""RAM fit recovery saves actual work without sharing object search cursors."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.question_routing import register_theories
from test_cross_object_fit_reference import objects
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, feature
from test_theory_consolidation import agent_scene


def completed_scene(tmp_path, monkeypatch):
    agent, _ = agent_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=30000)
    old, new = objects()
    agent.theory_consolidation = {'candidates': {old['id']: old, new['id']: new}}
    consolidation._save(agent)
    parent = theory('parent', binary('mul', constant(5.), feature()))
    register_theories(agent, [parent])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)}
              for item in examples(Program.from_dict(parent['program']))]
    (tmp_path / 'paid_context_fixture.json').write_text(json.dumps([
        item['observation'] for item in checks]))

    def one_binding(lengths, cursor):
        yield [0] * len(lengths), {'layer': 0, 'offset': 1}

    monkeypatch.setattr(revisions, '_diagonal_indices', one_binding)
    result = consolidation.transfer_theory(agent, {
        'candidate_id': old['id'], 'theory_id': 'original_fit',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'allocation': {'compute': 10000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}})
    assert len(result['generated_theories']) == 3
    return agent, old, new, parent, checks


def query(agent, candidate, parent, limit=10000):
    budget = SearchBudget(limit)
    records = consolidation._retained_revision_fits(agent, candidate, parent, budget)
    return records, budget


def test_equivalent_object_query_reuses_complete_recovery_without_reconstruction(
        tmp_path, monkeypatch):
    agent, old, new, parent, _ = completed_scene(tmp_path, monkeypatch)
    records, cold = query(agent, old, parent)
    assert len(records) == 1

    def forbidden(*args, **kwargs):
        pytest.fail('A completed equal query must not match, predict or fit again')

    monkeypatch.setattr(revisions, '_match_parent', forbidden)
    monkeypatch.setattr(revisions, '_fit_revision_binding', forbidden)
    monkeypatch.setattr(Program, 'predict', forbidden)
    reused, warm = query(agent, new, parent)
    assert reused is records
    assert 0 <= warm.spent < cold.spent
    assert len(agent._retained_revision_fit_cache['queries']) == 1


@pytest.mark.parametrize('change', ['append_transfer', 'replace_library', 'role', 'unit',
                                  'context', 'constructor'])
def test_fit_query_is_invalidated_by_changed_sources_or_exact_contract(
        tmp_path, monkeypatch, change):
    agent, old, _, parent, _ = completed_scene(tmp_path, monkeypatch)
    records, _ = query(agent, old, parent)
    assert records
    previous_cache = agent._retained_revision_fit_cache
    matching, original = [], revisions._match_parent

    def tracked(*args, **kwargs):
        matching.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(revisions, '_match_parent', tracked)
    candidate, current_parent = deepcopy(old), deepcopy(parent)
    if change == 'append_transfer':
        agent.theory_consolidation['transfers'].append(deepcopy(
            agent.theory_consolidation['transfers'][0]))
    elif change == 'replace_library':
        agent.theory_consolidation = deepcopy(agent.theory_consolidation)
    elif change == 'role':
        current_parent['role_signature']['u']['role'] = 'different_drive'
    elif change == 'unit':
        current_parent['output_unit'] = 'A'
    elif change == 'context':
        current_parent['context_ref']['scene'] = 'different_paid_scene'
    else:
        candidate['definition']['body'] = constant(3.).to_dict()
    actual, _ = query(agent, candidate, current_parent)
    assert actual is not records
    if change in {'append_transfer', 'replace_library'}:
        assert agent._retained_revision_fit_cache is not previous_cache
        assert actual and matching
    else:
        assert agent._retained_revision_fit_cache is previous_cache
        assert len(previous_cache['queries']) == 2
        assert actual == []


def test_incomplete_recovery_is_not_cached_as_an_empty_completed_query(tmp_path, monkeypatch):
    agent, old, _, parent, _ = completed_scene(tmp_path, monkeypatch)
    partial, insufficient = query(agent, old, parent, limit=1)
    assert partial == [] and insufficient.denied_requests
    assert agent._retained_revision_fit_cache['queries'] == []
    complete, recovered = query(agent, old, parent)
    assert len(complete) == 1 and recovered.spent > insufficient.spent
    assert len(agent._retained_revision_fit_cache['queries']) == 1


def test_two_object_proposals_share_only_fit_index_and_preserve_separate_cursors(
        tmp_path, monkeypatch):
    agent, old, new, parent, checks = completed_scene(tmp_path, monkeypatch)
    records, _ = query(agent, old, parent)
    state = agent._retained_revision_fit_cache['queries'][0]
    assert state['records'] is records
    builds, original = [], revisions._retained_fit_index

    def tracked(*args, **kwargs):
        builds.append(True)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('Equal retained batches must not solve again')

    monkeypatch.setattr(revisions, '_retained_fit_index', tracked)
    monkeypatch.setattr(revisions, '_fit_revision_binding', forbidden)
    old_cursor = {'layer': 3, 'offset': 8, 'pending_fits': []}
    saved_cursor, observed = deepcopy(old_cursor), []

    def one_binding(lengths, cursor):
        observed.append(deepcopy(cursor))
        yield [0] * len(lengths), {'layer': 0, 'offset': 1}

    monkeypatch.setattr(revisions, '_diagonal_indices', one_binding)
    arguments = {'allocation': {'compute': 10000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1},
        'retained_fits': records, 'retained_fit_state': state}
    first = revisions.propose_revision(old, parent, [], cursor=old_cursor, **arguments)
    second = revisions.propose_revision(new, parent, [], **arguments)
    assert builds == [True]
    assert first['retained_fit_reuse_count'] == second['retained_fit_reuse_count'] == 1
    assert [item['program'] for item in first['proposals']] == [
        item['program'] for item in second['proposals']]
    assert [item['fitting']['fit_reuse_ref'] for item in first['proposals']] == [
        item['fitting']['fit_reuse_ref'] for item in second['proposals']]
    assert observed == [saved_cursor, None]
    assert old_cursor == saved_cursor
    assert first['cursor'] is not second['cursor']
    first['cursor']['offset'] = 99
    assert second['cursor']['offset'] == 1


def test_local_recall_keeps_one_library_and_fit_index_across_actual_transfers(
        tmp_path, monkeypatch):
    agent, old, new, parent, checks = completed_scene(tmp_path, monkeypatch)
    local_library = agent.theory_consolidation
    question = {'id': 'recall_cache_fixture', 'meaning_version': 1,
        'theory_target': {'context_ref': deepcopy(CONTEXT),
            'role_signature': deepcopy(ROLES), 'output_unit': 'V',
            'parent_theory_id': 'parent', 'evidence_checks': checks,
            'revision_fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}}
    prefix = f'frozen_recall:{question["id"]}@1:'
    old_key = f'relation_transfer:{prefix}{old["id"]}'
    old_cursor = {'layer': 3, 'offset': 8, 'pending_fits': []}
    old_inputs = {'component_theory_ids': [], 'role_signature': deepcopy(ROLES),
        'output_unit': 'V', 'context_ref': deepcopy(CONTEXT), 'evidence_checks': checks,
        'allocation': {'compute': 10000}, 'parent_theory_id': 'parent',
        'fitting': question['theory_target']['revision_fitting']}
    local_library['executions'][old_key] = {
        'request_basis': consolidation._relation_request_basis(
            parent, [], old_inputs, old, consolidation.LIBRARY_PATH),
        'allowance': 10000, 'result': {'status': 'UNKNOWN'},
        'search_completed': False, 'cursor': deepcopy(old_cursor)}
    builds, calls, queries = [], [], []
    build, recover = revisions._retained_fit_index, consolidation._retained_revision_fits

    def tracked_index(*args, **kwargs):
        builds.append(True)
        return build(*args, **kwargs)

    def tracked_query(agent, candidate, parent, budget, **kwargs):
        start = budget.spent
        result = recover(agent, candidate, parent, budget, **kwargs)
        queries.append((candidate['id'], budget.spent - start))
        return result

    def index_only(candidate, parent, components, allocation, *, cursor=None,
                   retained_fits=None, retained_fit_state=None, **kwargs):
        # Run the real recovery/index path while deliberately producing no new
        # transfer. Appended scientific results must invalidate this RAM cache.
        assert retained_fits and retained_fit_state is not None
        budget = SearchBudget(allocation['compute'])
        if 'index' not in retained_fit_state:
            retained_fit_state['index'] = revisions._retained_fit_index(retained_fits, budget)
        calls.append((candidate['id'], deepcopy(cursor)))
        return {'status': 'UNKNOWN', 'proposals': [], 'excluded': [],
            'budget_exhausted': False, 'actual_compute_cost': budget.spent,
            'cursor': deepcopy(cursor or {'layer': 0, 'offset': 0}),
            'search_completed': False}

    def forbidden(*args, **kwargs):
        pytest.fail('A local recall must reuse its installed library without re-reading it')

    monkeypatch.setattr(consolidation, 'install_theory_library', forbidden)
    monkeypatch.setattr(consolidation, '_retained_revision_fits', tracked_query)
    monkeypatch.setattr(revisions, '_retained_fit_index', tracked_index)
    monkeypatch.setattr(revisions, 'propose_revision', index_only)
    before_transfers = len(local_library['transfers'])
    result = consolidation.recall_frozen_theories(agent, question, {
        'allocation': {'compute': 10000}, '_retained_components': [], '_retained_parent': parent})
    assert agent.theory_consolidation is local_library
    assert len(local_library['transfers']) == before_transfers
    assert [item['candidate_id'] for item in result['work']] == [old['id'], new['id']]
    assert queries[1][1] < queries[0][1]
    assert builds == [True]
    assert calls == [(old['id'], old_cursor), (new['id'], None)]
    assert local_library['executions'][old_key]['cursor'] == old_cursor
    new_key = f'relation_transfer:{prefix}{new["id"]}'
    assert local_library['executions'][new_key]['cursor'] == {'layer': 0, 'offset': 0}


def test_bare_transfer_still_installs_the_explicit_saved_library(tmp_path, monkeypatch):
    agent, old, _, _, _ = completed_scene(tmp_path, monkeypatch)
    path = agent.workspace / consolidation.LIBRARY_PATH
    document = json.loads(path.read_text())
    document['candidates'][old['id']]['authored_note'] = 'saved_file_changed'
    path.write_text(json.dumps(document))
    installs, original = [], consolidation.install_theory_library

    def tracked(*args, **kwargs):
        installs.append(True)
        return original(*args, **kwargs)

    def inspect(agent, arguments, candidate, path, **kwargs):
        return {'authored_note': candidate.get('authored_note')}

    monkeypatch.setattr(consolidation, 'install_theory_library', tracked)
    monkeypatch.setattr(consolidation, '_transfer_relation', inspect)
    result = consolidation.transfer_theory(agent, {'candidate_id': old['id']})
    assert installs == [True]
    assert result['authored_note'] == 'saved_file_changed'
