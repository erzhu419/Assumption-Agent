"""Different knowledge objects reuse equal solves, never another cursor."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.question_routing import register_theories
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene


def objects():
    old = {**edit_candidate(), 'version': 1}
    new = deepcopy(old)
    new['id'] = 'with_current_parent_reference'
    new['definition']['definition_id'] = new['id']
    contract = {'output_unit': 'V', 'time_contract': ['current']}
    definition = Definition('parent_abs', ('parent_program',),
        Program('abs', children=(Program('arg', name='parent_program'),)))
    new['parent_reference_relations'] = [{'id': 'parent_abs',
        'definition': definition.to_dict(), 'parent_argument': 'parent_program',
        'parent_output_unit': 'V', 'source_ids': ['source_a', 'source_b'], 'source_bindings': [],
        'holes': [], 'parent_contract': deepcopy(contract), **contract,
        'subtree_contracts': {'': deepcopy(contract), '0': deepcopy(contract)},
        'target_holes': ['hole_1']}]
    return old, new


@pytest.mark.parametrize('thin_restore', [False, True])
def test_mainline_new_object_reuses_generic_first_fit_and_starts_its_own_cursor(
        tmp_path, monkeypatch, thin_restore):
    agent, _ = agent_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=30000)
    old, new = objects()
    agent.theory_consolidation = {'candidates': {old['id']: old, new['id']: new}}
    consolidation._save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)]
    (tmp_path / 'paid_context_fixture.json').write_text(json.dumps([
        item['observation'] for item in checks]))
    arguments = {'candidate_id': old['id'], 'theory_id': 'old_work',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'allocation': {'compute': 10000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    # This fixture exercises only the ordinary first construction in both
    # objects; the extra parent-reference construction is a distinct solve.
    seen = []

    def first_binding(lengths, cursor):
        seen.append(deepcopy(cursor))
        yield [0] * len(lengths), {'layer': 0, 'offset': 0}

    monkeypatch.setattr(revisions, '_diagonal_indices', first_binding)
    first = consolidation.transfer_theory(agent, deepcopy(arguments))
    assert len(first['generated_theories']) == 3
    old_key = first['execution_key']
    original = agent.theory_consolidation['executions'][old_key]
    original['search_completed'] = False
    original['cursor'].update(layer=3, offset=8,
        pending_fits=[{'indices': [0, 0], 'position': {'layer': 3, 'offset': 5}}])
    retained = deepcopy(original)
    consolidation._save(agent)
    if thin_restore:
        previous = agent.workspace
        agent.workspace = tmp_path / 'restored'
        (agent.workspace / 'shared').mkdir(parents=True)
        agent.theory_consolidation = consolidation.install_theory_library(agent,
            {'file': str(previous / consolidation.LIBRARY_PATH), 'json_path': []})
        consolidation._save(agent)
        assert agent.theory_consolidation['executions'][old_key]['cursor'] == retained['cursor']
        retained = deepcopy(agent.theory_consolidation['executions'][old_key])

    def no_refit(*args, **kwargs):
        pytest.fail('the equal completed generic fit must be reused across object IDs')

    monkeypatch.setattr(revisions, '_fit_revision_binding', no_refit)
    second = consolidation.transfer_theory(agent, {**arguments,
        'candidate_id': new['id'], 'theory_id': 'new_work'})
    assert seen == [None, None]
    assert second['retained_complete_fit_batches_available'] == 1
    assert second['retained_fit_reuse_count'] == 1
    assert len(second['generated_theories']) == 3
    assert all(item['binding']['fitting']['fit_reuse_ref'] for item in second['generated_theories'])
    assert [item['theory_id'] for item in second['generated_theories']] == [
        item['theory_id'] for item in first['generated_theories']]
    assert agent.theory_consolidation['executions'][old_key] == retained
    fresh = agent.theory_consolidation['executions'][second['execution_key']]
    assert fresh['request_basis']['candidate_id'] == new['id']
    assert fresh['cursor']['layer'] == 0 and fresh['cursor']['offset'] == 0
    assert fresh['cursor']['pending_fits'] == []
    assert not second.get('retained_execution_refs')
    assert not second['support_updated'] and not second['independent_confirmation']


@pytest.mark.parametrize('change', ['before_pattern', 'definition', 'holes', 'scope',
                                  'constraint_hypotheses', 'parameter_relations'])
def test_fit_source_selection_does_not_expand_to_a_different_positional_contract(
        tmp_path, change):
    agent, _ = agent_scene(tmp_path)
    old, new = objects()
    other = deepcopy(old)
    other['id'] = 'different_constructor'
    if change == 'before_pattern':
        other['before_pattern'] = constant(3.).to_dict()
    elif change == 'definition':
        other['definition']['body'] = constant(3.).to_dict()
    elif change == 'holes':
        other['holes'][0]['output_unit'] = 'A'
    elif change == 'scope':
        other['scope']['context_family_ref'] = 'different_family'
    elif change == 'constraint_hypotheses':
        other['constraint_hypotheses'] = []
    else:
        other['parameter_relations'] = [{'purpose': 'different_parameter_proposal'}]
    agent.theory_consolidation = {'candidates': {
        old['id']: old, new['id']: new, other['id']: other}}
    versions = consolidation._revision_fit_source_versions(agent, new,
        {new['id']: 1}, SearchBudget(1000))
    assert versions == {new['id']: 1, old['id']: 1}
