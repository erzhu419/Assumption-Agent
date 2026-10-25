"""Outer object coverage is distinct from each object's bounded fit cursor."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.question_routing import register_theories
from test_question_routing import CONTEXT, ROLES, theory
from hegel_machine.cognition.revision_relations import consolidate_revision_relations
from test_revision_relations import guarded_sources
from test_theory_consolidation import agent_scene, assert_no_new_science


ORDER = ['registered_z', 'registered_a', 'registered_m']


def scene(tmp_path, *, real=False):
    agent, lab = agent_scene(tmp_path)
    if real:
        retained = consolidate_revision_relations(guarded_sources(), allocation={'compute': 10000})
        base = {**next(item for item in retained['candidates']
                      if item['kind'] == 'program_context_relation'), 'version': 1}
    else:
        base = {'kind': 'program_context_relation', 'version': 1, 'output_unit': 'V',
                'fixed_role_signature': {}}
    candidates = {}
    for identifier in ORDER:
        item = {**deepcopy(base), 'id': identifier}
        if real:
            item['definition']['definition_id'] = identifier
        candidates[identifier] = item
    agent.theory_consolidation = {'candidates': candidates}
    consolidation._save(agent)
    register_theories(agent, [theory('current', Program('feature', name='u'))])
    question = agent._open_question({'id': 'outer_objects', 'question': 'Use the retained contexts.',
        'context_ref': deepcopy(CONTEXT), 'theory_target': {
            'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'context_ref': deepcopy(CONTEXT),
            'evidence_checks': []}})
    return agent, lab, question


def consuming_transfer(calls, *, completed=False, reserve=0):
    def transfer(agent, inputs, laboratory=None):
        allowance = inputs['allocation']['compute']
        calls.append((inputs['candidate_id'], allowance))
        spent = min(1 if completed or reserve else allowance, allowance)
        held = reserve if allowance >= spent + reserve else 0
        if reserve and not held:
            spent = 0
        agent.costs['compute_cost'] += spent
        return {'status': 'completed' if completed else 'UNKNOWN',
            'generated_theories': [], 'new_compute_cost': spent,
            'reserved_comparison_compute': held,
            'search_completed': completed, 'budget_exhausted': not completed}
    return transfer


def test_single_call_covers_objects_and_rotates_across_new_checks_and_parents(tmp_path, monkeypatch):
    agent, lab, question = scene(tmp_path)
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls))
    original_candidates = deepcopy(agent.theory_consolidation['candidates'])
    for index in range(3):
        # This is the actual changing-input caller contract. Coverage must not
        # restart merely because numeric fitting correctly needs fresh work.
        question['theory_target']['evidence_checks'] = [{
            'source_ref': {'file': 'paid_fixture.json', 'json_path': [index]},
            'context_ref': deepcopy(CONTEXT),
            'observation': {'features': {'u': float(index)}, 'target': float(index)}}]
        register_theories(agent, [theory('current', Program('const', value=float(index)))])
        before = agent.costs['compute_cost']
        result = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
        assert result['status'] == 'UNKNOWN' and not result['cache_hit']
        assert result['new_compute_cost'] == agent.costs['compute_cost'] - before == 80
    assert [identifier for identifier, _ in calls] == [
        *ORDER, *ORDER[1:], ORDER[0], ORDER[2], *ORDER[:2]]
    assert all(24 <= allowance <= 25 for _, allowance in calls)
    assert agent.theory_consolidation['candidates'] == original_candidates
    assert_no_new_science(agent, lab)


def test_same_input_partial_work_and_cold_restore_keep_next_object(tmp_path, monkeypatch):
    agent, lab, question = scene(tmp_path / 'source')
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls))
    first = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert first['status'] == 'UNKNOWN'
    agent._prepare_program_state()
    previous = agent.result('researching')
    restored, restored_lab = agent_scene(tmp_path / 'restored')
    restored_lab.costs = {'observation_cost': 0}
    restored.restore_practice_research(previous, agent.workspace, restored_lab, confirmation_seed=17)
    owner = next(q for q in restored.questions if q['id'] == question['id'])
    second = consolidation.recall_frozen_theories(restored, owner, {'allocation': {'compute': 80}})
    third = consolidation.recall_frozen_theories(restored, owner, {'allocation': {'compute': 80}})
    assert [identifier for identifier, _ in calls] == [
        *ORDER, *ORDER[1:], ORDER[0], ORDER[2], *ORDER[:2]]
    assert not second['cache_hit'] and not third['cache_hit']
    assert_no_new_science(restored, restored_lab)


def test_atomic_shortfall_survives_restore_and_unspent_share_funds_later_objects(tmp_path, monkeypatch):
    agent, _, question = scene(tmp_path / 'source')
    calls = []

    def atomic(agent, inputs, laboratory=None):
        key = f'relation_transfer:{inputs["theory_id"]}'
        executions = agent.theory_consolidation.setdefault('executions', {})
        old = executions.get(key)
        if old and old['search_completed']:
            return {**deepcopy(old['result']), 'new_compute_cost': 0}
        local = SearchBudget(inputs['allocation']['compute'])
        funded = local.spend(60)
        calls.append((inputs['candidate_id'], funded))
        agent.costs['compute_cost'] += local.spent
        result = {'status': 'completed' if funded else 'UNKNOWN', 'generated_theories': [],
            'new_compute_cost': local.spent, 'search_completed': funded,
            **({} if funded else {'minimum_compute_for_progress':
                local.limit + local.last_denied_request['additional_compute']})}
        executions[key] = {'result': deepcopy(result), 'search_completed': funded,
                           'cursor': {'layer': 0, 'offset': int(funded)}}
        consolidation._save(agent)
        return result

    monkeypatch.setattr(consolidation, 'transfer_theory', atomic)
    first = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert calls == [(ORDER[0], False), (ORDER[1], False), (ORDER[2], True)]
    assert first['new_compute_cost'] == 66  # six preparation units + one funded atomic action
    agent._prepare_program_state()
    previous = agent.result('researching')
    restored, lab = agent_scene(tmp_path / 'restored')
    lab.costs = {'observation_cost': 0}
    restored.restore_practice_research(previous, agent.workspace, lab, confirmation_seed=17)
    current = next(q for q in restored.questions if q['id'] == question['id'])
    second = consolidation.recall_frozen_theories(restored, current, {'allocation': {'compute': 80}})
    assert second['unfunded_objects'][0]['candidate_id'] == ORDER[1]
    assert second['new_compute_cost'] == 65 and calls[-1] == (ORDER[0], True)
    third = consolidation.recall_frozen_theories(restored, current, {'allocation': {'compute': 80}})
    assert third['status'] == 'completed' and not third['pending_object_ids']
    assert third['new_compute_cost'] == 66 and calls[-1] == (ORDER[1], True)
    assert_no_new_science(restored, lab)


def test_multiple_role_mappings_share_one_object_slice(tmp_path, monkeypatch):
    from hegel_machine.cognition.abstractions import AbstractTemplate

    agent, lab, question = scene(tmp_path)
    template = AbstractTemplate('mapped', 'fixture', Program('feature', name='x'),
                                ('x',), ('u',), ())
    mapped = {'id': ORDER[0], 'version': 1, 'kind': 'typed_program_family', 'output_unit': 'V',
        'template': template.to_dict(), 'role_signature': {'x': deepcopy(ROLES['u'])}}
    later = agent.theory_consolidation['candidates'][ORDER[1]]
    agent.theory_consolidation['candidates'] = {ORDER[0]: mapped, ORDER[1]: later}
    consolidation._save(agent)
    question['theory_target']['role_signature']['v'] = deepcopy(ROLES['u'])
    calls = []

    def transfer(agent, inputs, laboratory=None):
        calls.append((inputs['candidate_id'], inputs.get('mapping'), inputs['allocation']['compute']))
        agent.costs['compute_cost'] += 3
        return {'status': 'completed', 'generated_theories': [], 'new_compute_cost': 3,
                'search_completed': True}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
    result = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 40}})
    assert calls == [(ORDER[0], {'x': 'u'}, 16), (ORDER[0], {'x': 'v'}, 10),
                     (ORDER[1], None, 25)]
    assert result['status'] == 'completed' and result['new_compute_cost'] == 18
    assert_no_new_science(agent, lab)


def test_completed_stable_pass_still_reuses_result_without_charging(tmp_path, monkeypatch):
    agent, _, question = scene(tmp_path)
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls, completed=True))
    first = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    before = deepcopy(agent.costs)
    again = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert first['status'] == 'completed' and len(calls) == 3
    assert again['cache_hit'] and again['new_compute_cost'] == 0 and agent.costs == before


def test_finished_unknown_parent_agreement_is_cached_without_repeated_work(tmp_path):
    from test_revision_relations import edit_candidate

    agent, lab = agent_scene(tmp_path)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    consolidation._save(agent)
    register_theories(agent, [theory('current_parent', Program('feature', name='u'))])
    owner = agent._open_question({'id': 'already_agrees', 'question': 'Use the actual retained parent.',
        'context_ref': deepcopy(CONTEXT), 'theory_target': {
            'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'context_ref': deepcopy(CONTEXT),
            'parent_theory_id': 'current_parent', 'evidence_checks': [{
                'source_ref': {'file': 'retained_fixture.json', 'json_path': [0]},
                'context_ref': deepcopy(CONTEXT), 'tolerance': 0.,
                'observation': {'features': {'u': 2.}, 'target': 2., 'history': []}}]}})
    inputs = {'allocation': {'compute': 500}}
    first = consolidation.recall_frozen_theories(agent, owner, inputs)
    work, = first['work']
    assert work['reason'] == 'parent_agrees_with_declared_evidence'
    assert first['status'] == 'UNKNOWN' and first['search_completed']
    assert first['pending_object_ids'] == [] and not first['generated_theory_ids']
    before = deepcopy(agent.costs)
    again = consolidation.recall_frozen_theories(agent, owner, inputs)
    assert again['cache_hit'] and again['status'] == 'UNKNOWN'
    assert again['new_compute_cost'] == 0 and agent.costs == before
    assert_no_new_science(agent, lab)


def test_reservation_is_held_once_and_is_not_an_actual_compute_debit(tmp_path, monkeypatch):
    agent, _, question = scene(tmp_path)
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls, reserve=5))
    before = agent.costs['compute_cost']
    result = consolidation.recall_frozen_theories(agent, question, {
        'allocation': {'compute': 12, 'comparison_rows': 5}})
    assert result['new_compute_cost'] == agent.costs['compute_cost'] - before
    assert result['reserved_comparison_compute'] == 5
    assert result['new_compute_cost'] + result['reserved_comparison_compute'] <= 12
    assert result['status'] == 'UNKNOWN'


def test_partial_full_sweeps_rotate_who_receives_the_largest_slice(tmp_path, monkeypatch):
    agent, _, question = scene(tmp_path)
    calls = []

    def partial(agent, inputs, laboratory=None):
        calls.append((inputs['candidate_id'], inputs['allocation']['compute']))
        agent.costs['compute_cost'] += 10
        return {'status': 'UNKNOWN', 'generated_theories': [], 'new_compute_cost': 10,
                'search_completed': False}

    monkeypatch.setattr(consolidation, 'transfer_theory', partial)
    for _ in range(3):
        result = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
        assert not result['cache_hit']
    assert [calls[i][0] for i in (0, 3, 6)] == ORDER
    assert calls[0][1] == calls[3][1] == calls[6][1] == 24
    assert calls[2][1] == calls[5][1] == calls[8][1] == 54


@pytest.mark.parametrize('changed', ['meaning', 'object_version', 'typed_scope'])
def test_changed_applicability_scope_restarts_coverage_with_fresh_work(tmp_path, monkeypatch, changed):
    agent, _, question = scene(tmp_path)
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls))
    consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    if changed == 'meaning':
        question['meaning_version'] = 2
    elif changed == 'object_version':
        agent.theory_consolidation['candidates'][ORDER[0]]['version'] = 2
    else:
        question['theory_target']['context_family_ref'] = 'new_declared_family'
    again = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert [item[0] for item in calls] == ORDER * 2
    assert not again['cache_hit']


def test_unfunded_first_object_yields_to_others_and_remains_unknown(tmp_path, monkeypatch):
    agent, _, question = scene(tmp_path)
    contracts = {f'control_{i}': deepcopy(ROLES['u']) for i in range(90)}
    question['theory_target']['role_signature'].update(contracts)
    agent.theory_consolidation['candidates'][ORDER[0]]['fixed_role_signature'] = contracts
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls))
    first = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert first['status'] == 'UNKNOWN'
    assert [item[0] for item in calls] == ORDER[1:]
    assert ORDER[0] in first['pending_object_ids']
    second = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert [item[0] for item in calls] == ORDER[1:] * 2
    assert second['status'] == 'UNKNOWN'


def test_incompatible_objects_stay_explicit_unknown_without_running_transfer(tmp_path, monkeypatch):
    agent, _, question = scene(tmp_path)
    question['theory_target']['output_unit'] = 'A'
    monkeypatch.setattr(consolidation, 'transfer_theory', lambda *a, **k: pytest.fail('unit mismatch'))
    result = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert result['status'] == 'UNKNOWN' and result['work'] == []
    assert [entry['candidate_id'] for entry in result['excluded']] == ORDER
    assert all(entry['reason'] == 'output_unit_mismatch' for entry in result['excluded'])


def test_real_equal_contexts_share_execution_and_keep_all_source_objects(tmp_path):
    agent, lab, question = scene(tmp_path, real=True)
    lag = Program('lag', name='u', lag=1)
    register_theories(agent, [theory('mature', lag),
        theory('mature_abs', Program('abs', children=(lag,))),
        theory('startup_abs', Program('abs', children=(Program('feature', name='u'),)))])
    applied = set()
    for _ in range(3):
        result = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 220}})
        applied.update(item['candidate_id'] for item in result['work'] if item['theory_ids'])
        assert result['new_compute_cost'] <= 220
    assert applied == {ORDER[0]}
    assert list(agent.theory_consolidation['candidates']) == ORDER
    assert result['execution_work_groups'][0]['candidate_ids'] == ORDER
    assert any(item.get('family_id') == ORDER[0] for item in agent.theory_index.values())
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('alias_count', [1, 5])
def test_exact_execution_aliases_share_one_slice_without_diluting_other_work(
        tmp_path, monkeypatch, alias_count):
    agent, lab, question = scene(tmp_path, real=True)
    original = deepcopy(agent.theory_consolidation['candidates'][ORDER[0]])
    candidates = {}
    for index in range(alias_count):
        candidate = deepcopy(original)
        candidate['id'] = f'source_alias_{index}'
        candidate['definition']['definition_id'] = candidate['id']
        candidate['source_ids'] = [f'original_source_{index}']
        candidates[candidate['id']] = candidate
    other = deepcopy(original)
    other['id'] = 'different_constraint'
    other['learned_constraints'] = [*other['learned_constraints'], {'op': 'fixed',
        'left': {'object': 'pattern', 'path': [], 'property': 'minimum_history_depth'}, 'value': 3}]
    candidates[other['id']] = other
    agent.theory_consolidation['candidates'] = deepcopy(candidates)
    consolidation._save(agent)
    calls = []
    monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls))
    result = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert calls == [('source_alias_0', 38), ('different_constraint', 38)]
    assert len(result['execution_work_groups']) == 2
    assert result['work'][0].get('shared_candidate_ids', ['source_alias_0']) == list(candidates)[:-1]
    assert agent.theory_consolidation['candidates'] == candidates
    added = deepcopy(candidates['source_alias_0'])
    added['id'] = 'new_alias_of_same_work'
    agent.theory_consolidation['candidates'][added['id']] = added
    consolidation._save(agent)
    repeated = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert calls[2:] == [('different_constraint', 38), ('source_alias_0', 38)]
    assert repeated['object_cursor']['next_candidate_id'] == 'source_alias_0'
    # Removing the next name leaves its identical research work next in line.
    del agent.theory_consolidation['candidates']['source_alias_0']
    consolidation._save(agent)
    again = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 80}})
    assert calls[4][0] != 'different_constraint' and calls[5][0] == 'different_constraint'
    retained = next(group['execution_candidate_refs'] for group in again['execution_work_groups']
                    if 'new_alias_of_same_work' in group['candidate_ids'])
    assert {'id': 'source_alias_0', 'version': 1} in retained
    retained_key = 'relation_transfer:retained_owner:source_alias_0'
    for group in question['frozen_recall_execution']['result']['execution_work_groups']:
        if 'new_alias_of_same_work' in group['candidate_ids']:
            group['execution_key'] = retained_key
    unfunded = consolidation.recall_frozen_theories(agent, question, {'allocation': {'compute': 0}})
    assert next(group['execution_key'] for group in unfunded['execution_work_groups']
                if 'new_alias_of_same_work' in group['candidate_ids']) == retained_key
    assert_no_new_science(agent, lab)


def test_a_shared_failed_fit_does_not_split_untouched_aliases_into_extra_work(tmp_path):
    agent, _, question = scene(tmp_path, real=True)
    prefix = f'frozen_recall:{question["id"]}@1:'
    failed = {'theory_id': prefix + ORDER[0] + ':binding:0:0', 'status': 'refuted',
              'binding': {'candidate_id': ORDER[0], 'candidate_version': 1},
              'checks': [{'status': 'failed', 'actual': 3., 'prediction': 2.}]}
    agent.theory_consolidation['transfers'] = [failed]
    candidates = agent.theory_consolidation['candidates']
    assert consolidation._recall_work_groups(agent, candidates, agent.theory_consolidation,
        {}, prefix=prefix) == [ORDER]
    # A separate retained object-specific failure still changes its contract.
    failed['theory_id'] = 'different_original_transfer'
    assert consolidation._recall_work_groups(agent, candidates, agent.theory_consolidation,
        {}, prefix=prefix) == [[ORDER[0]], ORDER[1:]]


@pytest.mark.parametrize('change', ['hole', 'parameter_relation', 'source_trigger', 'blocked_binding',
                                  'failed_transfer', 'original_roster'])
def test_execution_sharing_does_not_merge_different_consumed_constraints(tmp_path, change):
    from test_revision_relations import edit_candidate

    agent, _ = agent_scene(tmp_path)
    first = {**edit_candidate(), 'version': 1}
    second = deepcopy(first)
    second['id'] = 'other_source'
    second['definition']['definition_id'] = second['id']
    inputs = {}
    if change == 'hole':
        second['holes'][0]['output_unit'] = 'A'
    elif change == 'parameter_relation':
        second['parameter_relations'] = [{'output_hole': 'unique_slot', 'parameters': [2.]}]
    elif change == 'source_trigger':
        second['trigger_checks'] = []
    elif change == 'blocked_binding':
        second['blocked_transfer_bindings'] = [{'mapping': {}, 'context_ref': deepcopy(CONTEXT)}]
    elif change == 'failed_transfer':
        agent.theory_consolidation['transfers'] = [{'status': 'refuted', 'binding': {
            'candidate_id': second['id'], 'candidate_version': 1, 'context_ref': deepcopy(CONTEXT)},
            'checks': [{'status': 'failed', 'actual': 3., 'prediction': 2.}]}]
    else:
        inputs[second['id']] = {'_retained_components': [theory('different_parent', Program('const', value=2.))]}
    candidates = {first['id']: first, second['id']: second}
    assert consolidation._recall_work_groups(agent, candidates, agent.theory_consolidation, inputs) == [
        [first['id']], [second['id']]]


def test_equal_aliases_reuse_furthest_cursor_and_preserve_other_pending_work(tmp_path):
    agent, _ = agent_scene(tmp_path)
    candidate = {'id': 'first', 'version': 1}
    basis = {'candidate_id': 'first', 'candidate_version': 1, 'evidence_checks': [],
             'allocation': {'compute': 80}, 'parent': {'id': 'actual_parent', 'version': 1}}
    executions = {}
    for identifier, layer in [('first', 1), ('alias', 3)]:
        executions[f'relation_transfer:recall:{identifier}'] = {
            'request_basis': {**deepcopy(basis), 'candidate_id': identifier},
            'cursor': {'layer': layer, 'offset': 2, 'pending_fits': [
                {'indices': [layer], 'position': {'layer': layer, 'offset': 1}}]},
            'result': {'status': 'UNKNOWN'}, 'search_completed': False}
    original = deepcopy(executions)
    previous, refs = consolidation._shared_relation_progress(agent,
        {'_execution_candidate_ids': ['first', 'alias']}, candidate, basis,
        'relation_transfer:recall:first', executions)
    assert previous['cursor']['layer'] == 3
    assert previous['execution_key'] == 'relation_transfer:recall:alias'
    assert len(previous['cursor']['pending_fits']) == 2
    assert refs == list(executions) and executions == original
    changed = {**basis, 'parent': {'id': 'actual_parent', 'version': 2}}
    assert consolidation._shared_relation_progress(agent,
        {'_execution_candidate_ids': ['first', 'alias']}, candidate, changed,
        'relation_transfer:recall:first', executions) == (None, [])
    assert consolidation._shared_relation_progress(agent,
        {'_execution_candidate_ids': ['first', 'alias']}, {**candidate, 'version': 2}, basis,
        'relation_transfer:recall:first', executions) == (None, [])


def test_same_program_refutation_metadata_preserves_numerical_work_identity(tmp_path):
    agent, _ = agent_scene(tmp_path)
    original = {'parent': {'id': 'parent', 'version': 1, 'source_status': 'candidate'},
                'components': [{'id': 'part', 'version': 1, 'compatible': True}],
                'evidence_checks': [], 'allocation': {'compute': 80}}
    failed = deepcopy(original)
    failed['parent']['source_status'] = 'refuted'
    failed['components'][0]['compatible'] = False
    assert consolidation._same_relation_work(agent, original, failed)
    failed['components'][0]['version'] = 2
    assert not consolidation._same_relation_work(agent, original, failed)


def test_retained_parent_and_component_versions_continue_without_current_pool_replacement(
        tmp_path, monkeypatch):
    from hegel_machine.cognition import revision_relations
    from test_revision_relations import edit_candidate

    agent, lab = agent_scene(tmp_path)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    consolidation._save(agent)
    register_theories(agent, [theory('old_parent', Program('feature', name='u')),
        theory('component_b', Program('const', value=2.)),
        theory('component_a', Program('const', value=1.))])
    old_parent = deepcopy(agent.theory_index['old_parent'])
    old_components = [deepcopy(agent.theory_index[name]) for name in ('component_b', 'component_a')]
    calls = []

    def propose(candidate, parent, components, **kwargs):
        calls.append((deepcopy(parent), deepcopy(components), deepcopy(kwargs['cursor'])))
        return {'status': 'UNKNOWN', 'proposals': [], 'excluded': [], 'actual_compute_cost': 0,
            'budget_exhausted': True, 'search_completed': False,
            'cursor': {'layer': 4, 'offset': 3, 'pending_fits': []}}

    monkeypatch.setattr(revision_relations, 'propose_revision', propose)
    monkeypatch.setattr(consolidation, '_native_parent_components',
        lambda *a, **k: pytest.fail('retained search cannot add current native components'))
    arguments = {'candidate_id': candidate['id'], 'theory_id': 'retained',
        'parent_theory_id': old_parent['id'], 'component_theory_ids': ['component_b', 'component_a'],
        'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'context_ref': deepcopy(CONTEXT),
        'evidence_checks': [], 'allocation': {'compute': 80},
        '_retained_parent': old_parent, '_retained_components': old_components}
    consolidation.transfer_theory(agent, deepcopy(arguments))
    register_theories(agent, [theory('old_parent', Program('const', value=99.)),
        theory('component_a', Program('const', value=11.)),
        theory('component_b', Program('const', value=22.))])
    consolidation.transfer_theory(agent, deepcopy(arguments))
    assert calls[1][0]['version'] == old_parent['version'] == 1
    assert calls[1][0]['program'] == old_parent['program']
    assert calls[1][1] == old_components
    assert calls[1][2] == {'layer': 4, 'offset': 3, 'pending_fits': []}
    assert [item['id'] for item in calls[1][1]] == ['component_b', 'component_a']
    assert_no_new_science(agent, lab)


def test_registered_fixed_pr2_and_pr3_use_the_same_persistent_route_schedule(tmp_path, monkeypatch):
    from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent

    source, lab, original = scene(tmp_path / 'frozen')
    library_path = str(source.workspace / consolidation.LIBRARY_PATH)
    traces = {}
    for group in ('fixed_pr2', 'pr3'):
        organization = {'phase': 'target', 'initial_library_path': library_path,
                        'pr3_library_path': library_path}
        agent = RegisteredOrganizationAgent(None, tmp_path / group, arm=group,
            research_organization=organization, source_task=False, config=source.config,
            knowledge={'theory_consolidation': {'file': library_path, 'json_path': []}})
        register_theories(agent, [theory('current', Program('feature', name='u'))])
        owner = agent._open_question({key: deepcopy(original[key]) for key in
                                      ('id', 'question', 'context_ref', 'theory_target')})
        calls = []
        monkeypatch.setattr(consolidation, 'transfer_theory', consuming_transfer(calls))
        for _ in range(3):
            agent._automatic_work({'operator': 'research_route',
                'research_question': {'id': owner['id']}, 'allocation': {'compute': 80}},
                lab, 'research_route')
            result = agent.events[-1]['feedback']
            assert result['frozen_recall']['status'] == 'UNKNOWN'
        traces[group] = [identifier for identifier, _ in calls]
        assert agent.arm == 'code_react' and agent.costs['model_calls'] == 0
    expected = [*ORDER, *ORDER[1:], ORDER[0], ORDER[2], *ORDER[:2]]
    assert traces == {'fixed_pr2': expected, 'pr3': expected}


def test_parameter_proposal_missing_whole_fit_reaches_existing_author_contract(tmp_path, monkeypatch):
    from hegel_machine.cognition import revision_relations

    agent, _, _ = scene(tmp_path)
    candidate = {'id': 'parameter_fixture', 'version': 1,
                 'kind': 'program_revision_relation', 'output_unit': 'V'}
    agent.theory_consolidation['candidates'][candidate['id']] = candidate
    consolidation._save(agent)
    reason = 'whole_after_fitting_required_for_parameter_proposal'
    monkeypatch.setattr(revision_relations, 'propose_revision', lambda *a, **k: {
        'status': 'UNKNOWN', 'reason': reason, 'proposals': [], 'excluded': [],
        'actual_compute_cost': 1, 'budget_exhausted': False, 'search_completed': True})
    result = consolidation.transfer_theory(agent, {'candidate_id': candidate['id'],
        'theory_id': 'awaiting_fit', 'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'context_ref': deepcopy(CONTEXT), 'allocation': {'compute': 100}})
    request, = result['author_work']
    assert request['reason'] == reason
    assert request['required'] == ['operations', 'risk_quantile', 'max_history', 'max_structures']


@pytest.mark.parametrize(('field', 'old_value', 'new_value'), [
    ('purpose', 'exact_edit_map', 'parameter_proposal'),
    ('source_change_kind', 'structural', 'parameter_only'),
    ('structural_change', True, False),
])
def test_frozen_candidate_identity_includes_consumed_change_semantics(field, old_value, new_value):
    old = {'kind': 'program_revision_relation', 'definition': {
        'arguments': ['x'], 'body': {'op': 'arg', 'name': 'x'}}}
    old[field] = old_value
    changed = {**deepcopy(old), field: new_value}
    assert not consolidation._same_frozen_candidate(old, changed)


def test_reconsolidation_preserves_old_semantics_and_reuses_only_equal_semantics(tmp_path, monkeypatch):
    from test_theory_consolidation import consolidate, library

    agent, lab = agent_scene(tmp_path)
    original_constructor = consolidation.consolidate_theories
    emitted = {'purpose': 'exact_edit_map', 'source_change_kind': 'structural',
               'structural_change': True, 'source_change_diagnosis': {'receipt': 'first'}}

    def construct(*args, **kwargs):
        result = original_constructor(*args, **kwargs)
        for item in result['candidates']:
            item.update(deepcopy(emitted))
        return result

    monkeypatch.setattr(consolidation, 'consolidate_theories', construct)
    first = consolidate(agent, lab, theme='first_semantics')
    first_id = first['frozen_candidates'][0]['candidate_id']
    old = deepcopy(library(agent)['candidates'][first_id])
    emitted.update(purpose='parameter_proposal', source_change_kind='parameter_only',
                   structural_change=False)
    second = consolidate(agent, lab, theme='changed_semantics')
    second_id = second['frozen_candidates'][0]['candidate_id']
    assert second_id != first_id
    assert library(agent)['candidates'][first_id] == old
    emitted['source_change_diagnosis'] = {'receipt': 'same_semantics_new_diagnostic'}
    third = consolidate(agent, lab, theme='same_semantics')
    assert third['frozen_candidates'][0]['candidate_id'] == second_id
    assert len(library(agent)['candidates']) == 2
    assert library(agent)['candidates'][first_id] == old
    assert_no_new_science(agent, lab)
