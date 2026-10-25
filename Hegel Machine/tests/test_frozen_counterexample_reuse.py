"""Actual scoped counterexamples follow a numerical Program across aliases."""

from copy import copy, deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.question_routing import (
    register_theories, resolved_theories, scoped_numeric_refutation,
)
from hegel_machine.cognition.theory_matching import match_theories, prepare_theories_for_question


ROLES = {'x': {'role': 'input', 'unit': 'coordinate', 'time': 'current'}}
CONTEXT = {'episode': 'new_domain'}
PROGRAM = Program('add', children=(Program('const', value=.3), Program('feature', name='x')))


def declaration(identifier, program=PROGRAM, **fields):
    return {'id': identifier, 'program': program.to_dict(), 'version': 1,
        'role_signature': deepcopy(ROLES), 'output_unit': 'response', 'context_ref': deepcopy(CONTEXT),
        'parameter_paths': [], 'compatible': True,
        'support': {'status': 'candidate', 'evidence': []}, **fields}


def scene():
    binding = {'library_path': 'shared/theory_library.json', 'candidate_id': 'frozen0',
        'candidate_version': 1, 'mapping': {'role_0': 'x'}, 'parameters': [.3],
        'context_ref': deepcopy(CONTEXT)}
    failed = declaration('failed', compatible=False, support={'status': 'refuted', 'evidence': []},
                         consolidation_ref=binding)
    transfer = {'theory_id': 'failed', 'theory_ref': {'id': 'failed', 'version': 1},
        'binding': binding, 'status': 'refuted', 'checks': [
            {'status': 'failed', 'context_ref': deepcopy(CONTEXT), 'actual': 5., 'prediction': 1.3,
             'source_ref': {'archive': '/retained/history', 'unit': 'D',
                            'member': 'sequence.json', 'field': 'counterexamples[0]'}}]}
    agent = SimpleNamespace(theory_index={}, questions=[], events=[], relation_research={},
        theory_consolidation={'candidates': {}, 'transfers': [transfer]})
    register_theories(agent, [failed])
    return agent


def pool(agent):
    return {item['id']: item for item in resolved_theories(agent)}


@pytest.mark.parametrize('kind', ['renamed_failure', 'changed_parameters', 'changed_context'])
def test_incoming_numeric_refutation_does_not_expand_unrelated_forecast_columns(tmp_path, monkeypatch, kind):
    from hegel_machine.cognition import retained_forecasts

    agent = scene()
    agent.workspace = tmp_path
    (tmp_path / 'forecasts.json').write_text(json.dumps({'forecasts': [{
        'model': 'retained.py', 'prediction_inputs': [{'features': {'x': 1.}, 'history': []}],
        'predictions': [1.3]}]}))
    forecasts = []
    for identifier, filename in [('ready_forecast', 'forecasts.json'), ('missing_forecast', 'missing.json')]:
        item = declaration(identifier)
        item.pop('program')
        item['forecast_ref'] = {'file': filename, 'json_path': ['forecasts', 0], 'model': 'retained.py'}
        forecasts.append(item)
    register_theories(agent, forecasts)
    pending = declaration('incoming_alias')
    if kind == 'changed_parameters':
        pending['program'] = Program('add', children=(Program('const', value=.7),
            Program('feature', name='x'))).to_dict()
    elif kind == 'changed_context':
        pending['context_ref'] = {'episode': 'another_domain'}

    # The original full routing path supplies an independent scoped-evidence
    # comparison, including ready and unavailable opaque forecast columns.
    full = copy(agent)
    full.theory_index = deepcopy(agent.theory_index)
    register_theories(full, [pending])
    baseline = {item['id']: item for item in resolved_theories(full)}
    expected = baseline[pending['id']].get('scope_refutation')
    assert bool(expected) is (kind == 'renamed_failure')
    assert baseline['ready_forecast']['source_status'] == 'retained_forecast_ready'
    assert baseline['missing_forecast']['source_status'] == 'unavailable_retained_forecast'
    calls, resolve = [], retained_forecasts.resolve_forecast_references

    def counted(*args, **kwargs):
        calls.append(True)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(retained_forecasts, 'resolve_forecast_references', counted)
    before = deepcopy(agent.theory_index)
    assert scoped_numeric_refutation(agent, pending) == expected
    assert calls == []  # No unrelated column file read or whole-pool deepcopy.
    assert agent.theory_index == before
    register_theories(agent, [pending])
    assert {item['id']: item for item in resolved_theories(agent)} == baseline
    assert calls == [True]  # Ordinary prediction/selection still resolves every column.


def test_two_ordinary_aliases_cannot_revive_the_same_refuted_numerical_program():
    agent = scene()
    register_theories(agent, [declaration('alias_a'), declaration('alias_b')])
    original = deepcopy(agent.theory_index)
    resolved = pool(agent)
    for identifier in ('alias_a', 'alias_b'):
        assert resolved[identifier]['compatible'] is False
        refs = resolved[identifier]['scope_refutation']['counterexample_refs']
        assert refs[0]['json_path'] == ['transfers', 0, 'checks', 0]
        assert refs[1]['field'] == 'counterexamples[0]'
    # Resolution applies evidence without rewriting the retained declarations.
    assert agent.theory_index == original
    question = {'id': 'task', 'context_ref': CONTEXT, 'theory_target': declaration('target'),
                'public_role_signature': ROLES, 'output_unit': 'response'}
    matches = match_theories(question, list(resolved.values()), target=question['theory_target'])
    prepared = prepare_theories_for_question(question, list(resolved.values()), matches)
    assert prepared['theories'] == []
    assert {item['id'] for item in prepared['blocked']} == {'failed', 'alias_a', 'alias_b'}


def test_original_domain_and_truly_changed_parameters_remain_candidates():
    agent = scene()
    changed = Program('add', children=(Program('const', value=.7), Program('feature', name='x')))
    register_theories(agent, [declaration('old_domain', context_ref={'episode': 'source_domain'}),
                              declaration('new_parameters', changed)])
    resolved = pool(agent)
    assert resolved['old_domain']['compatible'] is True
    assert resolved['new_parameters']['compatible'] is True
    assert 'scope_refutation' not in resolved['old_domain']
    assert 'scope_refutation' not in resolved['new_parameters']


def test_overwriting_failed_id_with_new_parameters_keeps_old_version_counterexample():
    agent = scene()
    changed = Program('const', value=5.)
    register_theories(agent, [declaration('failed', changed), declaration('old_alias')])
    resolved = pool(agent)
    assert resolved['failed']['compatible'] is True
    assert resolved['failed']['version'] == 2
    alias = resolved['old_alias']
    assert alias['compatible'] is False
    assert alias['scope_refutation']['theory_refs'][0]['json_path'] == [
        'theory_index', 'failed', 'history', 0]
    assert agent.theory_index['failed']['history'][0]['program'] == PROGRAM.to_dict()


def test_removing_frozen_object_preserves_failure_of_retained_numerical_instance():
    agent = scene()
    # The new object can be absent while actual checks and original Programs stay.
    assert agent.theory_consolidation['candidates'] == {}
    register_theories(agent, [declaration('ordinary_retained_program')])
    assert pool(agent)['ordinary_retained_program']['compatible'] is False


def test_incoming_new_candidate_or_library_alias_is_scoped_before_registration():
    agent = scene()
    alias = declaration('new_library_alias', consolidation_ref={
        'library_path': 'shared/copy.json', 'candidate_id': 'another_object',
        'mapping': {'role_0': 'x'}, 'parameters': [.3], 'context_ref': CONTEXT})
    before = deepcopy(agent.theory_index)
    scope = scoped_numeric_refutation(agent, alias)
    assert scope['status'] == 'refuted' and scope['context_ref'] == CONTEXT
    assert 'new_library_alias' not in agent.theory_index and agent.theory_index == before
    changed = declaration('genuine_new_parameters', Program('const', value=5.), scope_refutation=scope)
    assert scoped_numeric_refutation(agent, changed) is None


@pytest.mark.parametrize('kind', ['not_executed', 'unknown', 'wrong_context'])
def test_only_actual_failed_checks_in_the_matching_context_propagate(kind):
    agent = scene()
    checks = agent.theory_consolidation['transfers'][0]['checks']
    if kind == 'not_executed':
        checks.clear()
    elif kind == 'unknown':
        checks[0]['status'] = 'UNKNOWN'
    else:
        checks[0]['context_ref'] = {'episode': 'another_domain'}
    register_theories(agent, [declaration('alias')])
    resolved = pool(agent)
    assert resolved['failed']['compatible'] is False  # Keep its original status.
    assert resolved['alias']['compatible'] is True


def test_existing_unversioned_transfer_uses_its_exact_original_binding():
    agent = scene()
    agent.theory_consolidation['transfers'][0].pop('theory_ref')
    register_theories(agent, [declaration('alias')])
    assert pool(agent)['alias']['compatible'] is False


def test_json_default_fields_do_not_make_an_identical_program_a_new_instance():
    agent = scene()
    alias = declaration('alias')
    alias['program'] = {'op': 'add', 'children': [
        {'op': 'const', 'value': .3}, {'op': 'feature', 'name': 'x'}]}
    register_theories(agent, [alias])
    assert pool(agent)['alias']['compatible'] is False


@pytest.mark.parametrize('failed_is_call', [False, True])
def test_learned_definition_context_and_inline_program_share_exact_scoped_failure(failed_is_call):
    agent = scene()
    definition = Definition('learned_context', ('body',), Program('add', children=(
        Program('const', value=.3), Program('arg', name='body'))))
    wrapped = Program('call', name=definition.definition_id, definition=definition,
                      children=(Program('feature', name='x'),))
    if failed_is_call:
        agent.theory_index['failed']['program'] = wrapped.to_dict()
    alias = declaration('context_alias', PROGRAM if failed_is_call else wrapped)
    register_theories(agent, [alias])
    assert pool(agent)['context_alias']['compatible'] is False
    assert scoped_numeric_refutation(agent, alias)['counterexample_refs']
    # A different ordered computation does not inherit this instance's failure.
    changed = Definition('different_context', ('body',), Program('add', children=(
        Program('const', value=.4), Program('arg', name='body'))))
    other = Program('call', name=changed.definition_id, definition=changed,
                    children=(Program('feature', name='x'),))
    assert scoped_numeric_refutation(agent, declaration('changed_context', other)) is None
    agent.theory_consolidation.pop('candidates')
    assert pool(agent)['context_alias']['compatible'] is False


def test_actual_refuted_relation_bank_also_blocks_same_typed_alias_in_its_scope():
    agent = scene()
    agent.theory_consolidation['transfers'].clear()
    agent.theory_index.clear()
    agent.questions = [{'id': 'source', 'meaning_version': 1}]
    agent.relation_research = {'source': {'round_index': 0, 'candidates': [
        {'program': PROGRAM.to_dict(), 'status': 'refuted'}]}}
    bank_theory = declaration('bank')
    bank_theory.pop('program')
    bank_theory['source_ref'] = {'kind': 'relation_candidate', 'question_id': 'source',
        'round_index': 0, 'candidate_index': 0, 'meaning_version': 1}
    register_theories(agent, [bank_theory, declaration('alias')])
    resolved = pool(agent)
    assert resolved['alias']['compatible'] is False
    assert resolved['alias']['scope_refutation']['counterexample_refs'][0]['kind'] == 'relation_candidate'


@pytest.mark.parametrize('alias_kind', ['library_path', 'candidate_id'])
def test_real_transfer_of_copied_frozen_object_returns_refuted_before_new_registration(tmp_path, alias_kind):
    from test_theory_consolidation import (
        agent_scene, assert_no_new_science, candidate, consolidate, library, transfer, CONTEXT as TARGET,
    )

    agent, lab = agent_scene(tmp_path)
    result = consolidate(agent, lab)
    identifier = result['frozen_candidates'][0]['candidate_id']
    check = {'source_ref': {'file': 'retained_target.json', 'json_path': [0]},
        'context_ref': deepcopy(TARGET), 'tolerance': 0.,
        'observation': {'features': {'u': 1.}, 'target': 30., 'history': []}}
    failed = transfer(agent, lab, identifier, evidence_checks=[check])
    assert failed['status'] == 'refuted'
    original_reconstruction = deepcopy(candidate(agent, identifier)['source_reconstruction'])
    stored = library(agent)
    extra = {'theory_id': 'alias_of_known_failed_instance'}
    if alias_kind == 'library_path':
        agent._save('shared/copied_frozen_library.json', json.dumps(stored))
        extra['library_path'] = 'shared/copied_frozen_library.json'
    else:
        copied = deepcopy(stored['candidates'][identifier])
        copied['id'] = 'copied_frozen_candidate'
        stored['candidates'][copied['id']] = copied
        agent._save('shared/theory_library.json', json.dumps(stored))
        identifier = copied['id']
    returned = transfer(agent, lab, identifier, **extra)
    assert returned['status'] == 'refuted'
    alias = agent.theory_index[extra['theory_id']]
    assert alias['compatible'] is False and alias['support']['status'] == 'refuted'
    assert alias['version'] == 1
    assert alias['scope_refutation']['counterexample_refs']
    assert candidate(agent, identifier if alias_kind == 'candidate_id' else None)[
        'source_reconstruction'] == original_reconstruction
    assert_no_new_science(agent, lab)
