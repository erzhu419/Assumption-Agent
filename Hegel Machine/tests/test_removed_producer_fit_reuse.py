"""Removing future proposals keeps all already-paid producer fits available."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.question_routing import register_theories
from test_cross_object_fit_reference import objects
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, feature
from test_theory_consolidation import agent_scene


@pytest.mark.parametrize('via_recall', [False, True])
def test_removed_constructor_is_only_used_to_reference_paid_fit(
        tmp_path, monkeypatch, via_recall):
    agent, _ = agent_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=30000)
    old, removed = objects()
    agent.theory_consolidation = {'candidates': {removed['id']: removed}}
    consolidation._save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)]
    (tmp_path / 'paid_context_fixture.json').write_text(json.dumps([
        item['observation'] for item in checks]))

    def first_binding(lengths, cursor):
        yield [0] * len(lengths), {'layer': 0, 'offset': 1}

    monkeypatch.setattr(revisions, '_diagonal_indices', first_binding)
    arguments = {'candidate_id': removed['id'], 'theory_id': 'original_paid_fit',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'allocation': {'compute': 10000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    original = consolidation.transfer_theory(agent, deepcopy(arguments))
    assert len(original['generated_theories']) == 3
    original_execution = deepcopy(agent.theory_consolidation['executions'][
        original['execution_key']])
    constructors = {old['id']: deepcopy(old), removed['id']: deepcopy(removed)}
    agent.theory_consolidation['candidates'] = {old['id']: old}
    consolidation._save(agent)
    absent = consolidation._retained_revision_fits(
        agent, old, agent.theory_index['parent'], SearchBudget(10000))
    assert absent == []
    before_transfers = deepcopy(agent.theory_consolidation['transfers'])

    def no_refit(*args, **kwargs):
        pytest.fail('removing a future proposal must not cause a repeated paid solve')

    monkeypatch.setattr(revisions, '_fit_revision_binding', no_refit)
    args = {**arguments, 'candidate_id': old['id'], 'theory_id': 'future_old_only',
            '_retained_fit_constructors': constructors,
            '_retained_parent': deepcopy(agent.theory_index['parent']),
            '_retained_components': []}
    if via_recall:
        owner = agent._open_question({'id': 'old_only_parent_use',
            'question': 'Revise this parent with the retained old library.',
            'context_ref': deepcopy(CONTEXT), 'theory_target': {
                'parent_theory_id': 'parent', 'role_signature': deepcopy(ROLES),
                'output_unit': 'V', 'context_ref': deepcopy(CONTEXT),
                'evidence_checks': deepcopy(checks), 'revision_fitting': args['fitting']}})
        recalled = consolidation.recall_frozen_theories(agent, owner, args)
        assert recalled['examined_candidate_ids'] == [old['id']]
        assert [group['candidate_ids'] for group in recalled['execution_work_groups']] == [[old['id']]]
        key = f'relation_transfer:frozen_recall:{owner["id"]}@1:{old["id"]}'
        consumed = agent.theory_consolidation['executions'][key]['result']
    else:
        consumed = consolidation.transfer_theory(agent, args)

    assert consumed['retained_complete_fit_batches_available'] == 1
    assert consumed['retained_fit_reuse_count'] == 1
    assert len(consumed['generated_theories']) == 3
    assert all(item['binding']['candidate_id'] == old['id']
               for item in consumed['generated_theories'])
    assert list(agent.theory_consolidation['candidates']) == [old['id']]
    assert agent.theory_consolidation['transfers'][:len(before_transfers)] == before_transfers
    assert all(item['binding']['candidate_id'] == old['id'] for item in
               agent.theory_consolidation['transfers'][len(before_transfers):])
    assert agent.theory_consolidation['executions'][original['execution_key']] == original_execution
    assert all('_retained_fit_constructors' not in entry['request_basis']
               for entry in agent.theory_consolidation['executions'].values())
