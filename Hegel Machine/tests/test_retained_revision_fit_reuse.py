"""A different parent identity must consume an identical paid complete fit."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import (
    _save, transfer_theory, install_theory_library, recall_frozen_theories,
)
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene


@pytest.mark.parametrize('thin_restore', [False, True, 'native_parent_reference'])
@pytest.mark.parametrize('changed_tolerance', [False, True])
def test_mainline_reuses_complete_fit_with_different_parent_identity(
        tmp_path, monkeypatch, thin_restore, changed_tolerance):
    agent, lab = agent_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=30000)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)]
    (tmp_path / 'paid_context_fixture.json').write_text(json.dumps([item['observation'] for item in checks]))
    arguments = {'candidate_id': candidate['id'], 'theory_id': 'first',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'allocation': {'compute': 10000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    first = transfer_theory(agent, deepcopy(arguments))
    assert len(first['generated_theories']) == 3
    assert all(item['binding']['fitting']['search_completed'] for item in first['generated_theories'])
    if thin_restore is True:
        source = agent.workspace
        agent.workspace = tmp_path / 'restored'
        (agent.workspace / 'shared').mkdir(parents=True)
        agent.theory_consolidation = install_theory_library(agent,
            {'file': str(source / 'shared/theory_library.json'), 'json_path': []})
        _save(agent)
    if thin_restore == 'native_parent_reference':
        # Restored native declarations retain a reference into the real bank.
        agent.questions.append({'id': 'native_owner', 'meaning_version': 1,
            'native_condition_research': [{'candidates': [{'program': parent.to_dict()}]}]})
        saved = agent.theory_index['parent']
        saved.pop('program')
        saved['source_ref'] = {'kind': 'native_candidate', 'question_id': 'native_owner',
            'meaning_version': 1, 'round_index': 0, 'candidate_index': 0}
    if changed_tolerance:
        arguments['evidence_checks'] = [{**item, 'tolerance': 1e-12} for item in checks]
    register_theories(agent, [theory('another_parent', parent)])

    def no_refit(*args, **kwargs):
        raise AssertionError('same paid complete fit must be reused before fitting')

    monkeypatch.setattr(revision_relations, '_fit_revision_binding', no_refit)
    repeated = transfer_theory(agent, {**arguments, 'theory_id': 'second',
        'parent_theory_id': 'another_parent'})
    assert repeated['retained_complete_fit_batches_available'] > 0
    assert repeated['retained_fit_reuse_count'] == 1
    assert len(repeated['generated_theories']) == 3
    if changed_tolerance:
        assert not any(item.get('reused_existing_theory') for item in repeated['generated_theories'])
        assert all(item['theory_id'].startswith('second:') for item in repeated['generated_theories'])
        return
    assert all(item.get('reused_existing_theory') for item in repeated['generated_theories'])
    assert [item['theory_id'] for item in repeated['generated_theories']] == [
        item['theory_id'] for item in first['generated_theories']]
    assert all(item['version'] == 1 for item in repeated['generated_theories'])
    assert all(item['binding']['fitting']['fit_reuse_ref'] for item in repeated['generated_theories'])
    assert not repeated['candidate_pool_changed']


def test_recall_rechecks_work_after_a_retained_cursor_has_advanced(tmp_path):
    agent, _ = agent_scene(tmp_path)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    owner = agent._open_question({'id': 'finite_revision', 'question': 'Revise the retained fixture.',
        'context_ref': deepcopy(CONTEXT), 'theory_target': {
            'context_ref': deepcopy(CONTEXT), 'role_signature': deepcopy(ROLES), 'output_unit': 'V',
            'parent_theory_id': 'parent',
            'evidence_checks': [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)],
            'revision_fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 2}}})
    arguments = {'allocation': {'compute': 150, 'comparison_rows': 4}}
    first = recall_frozen_theories(agent, owner, arguments)
    assert first['status'] == 'UNKNOWN'
    # An explicitly requested equal-size slice continues incomplete work. It
    # must not cache UNKNOWN forever or prevent outer-object coverage.
    unchanged = recall_frozen_theories(agent, owner, arguments)
    assert not first['search_completed'] and not unchanged['cache_hit']
    assert 0 < unchanged['new_compute_cost'] <= 150
    execution = agent.theory_consolidation['executions'][
        f'relation_transfer:frozen_recall:{owner["id"]}@1:{candidate["id"]}']
    execution['cursor']['offset'] += 1
    advanced = recall_frozen_theories(agent, owner, arguments)
    assert not advanced['cache_hit']


@pytest.mark.parametrize('missing_one_row', [False, True])
def test_paid_rows_and_criterion_are_read_as_one_batch(tmp_path, monkeypatch, missing_one_row):
    agent, _ = agent_scene(tmp_path)
    rows = [{'features': {'u': float(i)}, 'target': float(i), 'history': []} for i in range(4)]
    (tmp_path / 'retained_rows.json').write_text(json.dumps(rows))
    (tmp_path / 'retained_criterion.json').write_text(json.dumps({'tau': 1e-10}))
    checks = [{'source_ref': {'file': 'retained_rows.json', 'json_path': [i]},
        'context_ref': deepcopy(CONTEXT), 'criterion': {'kind': 'paid_mse',
            'evaluated_rows': 4, 'mse_threshold': 1e-10,
            'criterion_ref': {'file': 'retained_criterion.json', 'json_path': ['tau']}}}
        for i in range(4)]
    if missing_one_row:
        checks[-1]['source_ref']['json_path'] = [99]
    calls = []
    original = consolidation.resolve_program_sources

    def read(agent, references):
        calls.append(len(references))
        return original(agent, references)

    monkeypatch.setattr(consolidation, 'resolve_program_sources', read)
    resolved = consolidation._resolve_paid_checks(agent, checks)
    assert [item['observation'] for item in resolved[:3]] == rows[:3]
    if missing_one_row:
        assert 'observation' not in resolved[-1] and resolved[-1]['resolution_reason']
    else:
        assert calls == [8]
        assert resolved[-1]['observation'] == rows[-1]


@pytest.mark.parametrize('change', ['none', 'row', 'order', 'equal_row_addresses', 'threshold', 'missing'])
def test_restored_check_addresses_preserve_only_the_same_ordered_problem(tmp_path, change):
    agent, _ = agent_scene(tmp_path)
    rows = [{'features': {'u': float(i)}, 'target': float(i), 'history': []} for i in range(2)]
    if change == 'equal_row_addresses':
        rows[1] = deepcopy(rows[0])
    (tmp_path / 'old_rows.json').write_text(json.dumps(rows))
    changed = deepcopy(rows)
    if change == 'row':
        changed[0]['target'] += 1
    (tmp_path / 'new_rows.json').write_text(json.dumps(changed))
    (tmp_path / 'criterion.json').write_text(json.dumps({'old': 1e-10, 'new': 1e-9}))
    checks = [{'source_ref': {'file': str(tmp_path / 'old_rows.json'), 'json_path': [i]},
        'context_ref': deepcopy(CONTEXT), 'criterion': {'kind': 'paid_mse',
            'evaluated_rows': 2, 'mse_threshold': 1e-10,
            'criterion_ref': {'file': str(tmp_path / 'criterion.json'), 'json_path': ['old']}}}
        for i in range(2)]
    old = {'evidence_checks': checks, 'allocation': {'compute': 100, 'comparison_rows': 2}}
    new = deepcopy(old)
    new['allocation']['compute'] = 200
    for item in new['evidence_checks']:
        item['source_ref']['file'] = 'new_rows.json'
        item['criterion']['criterion_ref']['file'] = 'criterion.json'
        if change == 'threshold':
            item['criterion']['mse_threshold'] = 1e-9
            item['criterion']['criterion_ref']['json_path'] = ['new']
    if change in {'order', 'equal_row_addresses'}:
        new['evidence_checks'].reverse()
    if change == 'missing':
        new['evidence_checks'][0]['source_ref']['json_path'] = [99]
    original = deepcopy(old)
    assert consolidation._same_relation_work(agent, old, new) is (change == 'none')
    assert old == original
