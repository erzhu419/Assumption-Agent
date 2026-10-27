"""Retained object work must pay for the inputs actually consumed by transfer."""
from copy import deepcopy

import pytest

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories
from test_question_routing import CONTEXT, ROLES, theory
from test_theory_consolidation import agent_scene, assert_no_new_science


def retained_scene(tmp_path, identifiers=('old_object',)):
    agent, lab = agent_scene(tmp_path)
    register_theories(agent, [theory(f'component_{index}',
        Program('const', value=float(index))) for index in range(8)])
    parent = deepcopy(agent.theory_index['component_0'])
    retained = [deepcopy(agent.theory_index[f'component_{index}']) for index in (5, 2, 7)]
    # The live pool has newer versions; this research retains its original roster.
    register_theories(agent, [theory('component_5', Program('const', value=55.)),
                             theory('component_2', Program('const', value=22.))])
    agent.candidate_paths = ['shared_old_python.py', 'shared_native_v1.py']
    agent.theory_consolidation = {'candidates': {identifier: {
        'id': identifier, 'version': 1, 'kind': 'program_revision_relation',
        'output_unit': 'V', 'fixed_role_signature': deepcopy(ROLES)}
        for identifier in identifiers}}
    consolidation._save(agent)
    question = agent._open_question({'id': 'retained_parent_research',
        'question': 'Continue the retained parent research.',
        'context_ref': deepcopy(CONTEXT), 'theory_target': {
            'parent_theory_id': parent['id'], 'role_signature': deepcopy(ROLES),
            'output_unit': 'V', 'context_ref': deepcopy(CONTEXT), 'evidence_checks': []}})
    args = {'allocation': {'compute': 80}, 'fitting': {'risk_quantile': .5},
        '_retained_execution_inputs': {identifier: {
            '_retained_parent': deepcopy(parent), '_retained_components': deepcopy(retained),
            'component_theory_ids': [item['id'] for item in retained]}
            for identifier in identifiers}}
    return agent, lab, question, args


def install_unfunded_proposal(monkeypatch, queries):
    """Use the real transfer identity/persistence, with no fit or prediction."""
    def old_fits(agent, candidate, parent, budget, **kwargs):
        queries.append(candidate['id'])
        return []

    def unfunded(candidate, parent, components, **kwargs):
        return {'status': 'UNKNOWN', 'proposals': [], 'excluded': [],
            'actual_compute_cost': 0, 'budget_exhausted': True,
            'search_completed': False, 'cursor': {'layer': 0, 'offset': 0},
            'minimum_compute_for_progress': 100000}

    monkeypatch.setattr(consolidation, '_retained_revision_fits', old_fits)
    monkeypatch.setattr(revision_relations, 'propose_revision', unfunded)
    monkeypatch.setattr(revision_relations, 'prepare_revision_visit',
        lambda candidate, parent, components, **kwargs: {
            'prepared_visit': None, 'result': unfunded(candidate, parent, components, **kwargs),
            'quote': None, 'actual_compute_cost': 0, 'cursor': {'layer': 0, 'offset': 0}})


@pytest.mark.parametrize('with_opaque', [False, True])
def test_outer_preparation_uses_saved_roster_and_preserves_full_pool(tmp_path, monkeypatch, with_opaque):
    agent, lab, question, args = retained_scene(tmp_path)
    if with_opaque:
        retained = args['_retained_execution_inputs']['old_object']
        retained['_retained_components'].insert(1, {
            'id': 'old_python_forecast', 'version': 8,
            'forecast_ref': {'file': 'retained_forecasts.json', 'json_path': ['forecasts', 0]}})
        retained['component_theory_ids'] = [item['id'] for item in retained['_retained_components']]
    roster = deepcopy(args['_retained_execution_inputs']['old_object']['_retained_components'])
    index, models = deepcopy(agent.theory_index), list(agent.candidate_paths)
    calls = []

    def transfer(agent, inputs, laboratory=None):
        calls.append(deepcopy(inputs))
        return {'status': 'UNKNOWN', 'generated_theories': [],
                'new_compute_cost': 0, 'search_completed': False}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert result['new_compute_cost'] == 1 + len(ROLES) + len(roster)
    assert calls[0]['allocation']['compute'] == 80 - result['new_compute_cost']
    assert calls[0]['_retained_components'] == roster
    assert calls[0]['component_theory_ids'] == [item['id'] for item in roster]
    assert [item['version'] for item in roster if item.get('program')] == [1, 1, 1]
    assert agent.theory_index['component_5']['version'] == 2
    assert agent.theory_index == index and agent.candidate_paths == models
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('unrelated_change', ['other_cursor', 'new_candidate'])
def test_unfinished_preparation_survives_other_object_changes_without_becoming_a_fit(
        tmp_path, monkeypatch, unrelated_change):
    agent, lab, question, args = retained_scene(tmp_path, ('target', 'other'))
    queries = []
    install_unfunded_proposal(monkeypatch, queries)
    consolidation.recall_frozen_theories(agent, question, args)
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:target'
    target_before = deepcopy(agent.theory_consolidation['executions'][key])
    if unrelated_change == 'other_cursor':
        other_key = key.removesuffix('target') + 'other'
        agent.theory_consolidation['executions'][other_key]['cursor']['offset'] = 3
    else:
        candidate = deepcopy(agent.theory_consolidation['candidates']['other'])
        candidate['id'] = 'third'
        agent.theory_consolidation['candidates']['third'] = candidate
        args['_retained_execution_inputs']['third'] = deepcopy(
            args['_retained_execution_inputs']['other'])
    consolidation._save(agent)
    queries.clear()
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert 'target' in queries
    assert any(item['candidate_id'] == 'target' and item['status'] == 'preparation_incomplete'
               for item in result['complete_research_visits'])
    assert agent.theory_consolidation['executions'][key]['cursor'] == target_before['cursor']
    assert 'target' in result['pending_object_ids'] and result['status'] == 'UNKNOWN'
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('target_change', ['parent', 'evidence', 'component'])
def test_changed_object_problem_does_not_apply_old_minimum(tmp_path, monkeypatch, target_change):
    agent, lab, question, args = retained_scene(tmp_path)
    queries = []
    install_unfunded_proposal(monkeypatch, queries)
    consolidation.recall_frozen_theories(agent, question, args)
    retained = args['_retained_execution_inputs']['old_object']
    if target_change == 'parent':
        retained['_retained_parent']['version'] += 1
    elif target_change == 'component':
        retained['_retained_components'][0]['version'] += 1
    else:
        question['theory_target']['evidence_checks'] = [{
            'source_ref': {'file': 'fixture_paid_row.json', 'json_path': ['rows', 1]},
            'context_ref': deepcopy(CONTEXT), 'observation': {
                'features': {'u': 1.}, 'target': 2., 'history': []}}]
    queries.clear()
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert queries == ['old_object']
    assert result['unfunded_objects'] == []
    assert [entry['candidate_id'] for entry in result['work']] == ['old_object']
    assert_no_new_science(agent, lab)


def test_renaming_old_and_new_objects_does_not_change_dispatch_or_coverage(tmp_path, monkeypatch):
    traces = []
    for number, identifiers in enumerate((('old19', 'new20', 'peer'),
                                         ('new20', 'peer', 'old19'))):
        agent, lab, question, args = retained_scene(tmp_path / str(number), identifiers)
        calls, cursors = [], []

        def transfer(agent, inputs, laboratory=None):
            if inputs.get('_prepare_complete_visit'):
                return {'prepared_visit': {}, 'quote': {'compute': 75, 'maximum_outputs': 1},
                    'new_compute_cost': 0, '_prepared_relation_visit': {}}
            spent = inputs['allocation']['compute']
            calls.append((identifiers.index(inputs['candidate_id']), spent))
            agent.costs['compute_cost'] += spent
            return {'status': 'UNKNOWN', 'generated_theories': [],
                    'new_compute_cost': spent, 'search_completed': False,
                    'completed_research_unit': True}

        monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
        for _ in range(3):
            result = consolidation.recall_frozen_theories(agent, question, args)
            cursors.append(identifiers.index(result['object_cursor']['next_candidate_id']))
            assert result['new_compute_cost'] == 80
        traces.append((calls, cursors))
        assert_no_new_science(agent, lab)
    assert traces[0] == traces[1]
    assert [index for index, _ in traces[0][0]] == [0, 1, 2]
    assert traces[0][1] == [1, 2, 0]


@pytest.mark.parametrize('index_bill', ['recorded', 'legacy_reused_fit'])
def test_valid_warm_fit_state_lowers_cold_minimum_and_reaches_transfer(
        tmp_path, monkeypatch, index_bill):
    agent, lab, question, args = retained_scene(tmp_path)
    queries = []
    install_unfunded_proposal(monkeypatch, queries)
    consolidation.recall_frozen_theories(agent, question, args)
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:old_object'
    previous = agent.theory_consolidation['executions'][key]['result']
    # This is the saved historical cold-prefix bill, not a new numerical fit.
    previous.update(minimum_compute_for_progress=18398,
        retained_fit_recovery_compute=6198, retained_fit_reuse_count=1)
    if index_bill == 'recorded':
        previous['retained_fit_index_compute'] = 8806
    else:
        previous.pop('retained_fit_index_compute', None)
    warm = {'selection_compute': 7, 'recovery_compute': 6198,
            'index_compute': 8806, 'index': {}, 'records': []}
    monkeypatch.setattr(consolidation, '_available_retained_fit_state',
                        lambda *a, **k: warm)
    queries.clear()
    args['allocation']['compute'] = 3500
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert queries == ['old_object']
    assert result['unfunded_objects'] == []
    assert [entry['candidate_id'] for entry in result['work']] == ['old_object']
    assert result['new_compute_cost'] <= 3500
    assert_no_new_science(agent, lab)


def test_old_warm_atomic_minimum_does_not_replace_current_complete_preparation(
        tmp_path, monkeypatch):
    agent, lab, question, args = retained_scene(tmp_path)
    queries = []
    install_unfunded_proposal(monkeypatch, queries)
    consolidation.recall_frozen_theories(agent, question, args)
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:old_object'
    previous = agent.theory_consolidation['executions'][key]['result']
    previous.update(minimum_compute_for_progress=3401,
        retained_fit_recovery_compute=7, retained_fit_index_compute=0,
        retained_fit_reuse_count=1)
    warm = {'selection_compute': 7, 'recovery_compute': 6198,
            'index_compute': 8806, 'index': {}, 'records': []}
    monkeypatch.setattr(consolidation, '_available_retained_fit_state',
                        lambda *a, **k: warm)
    before = deepcopy(agent.theory_consolidation['executions'][key])
    queries.clear()
    # An interruption-derived warm3401 is not a complete-work quote. Prepare
    # the actual current unit instead of rejecting it by that old minimum.
    args['allocation']['compute'] = 3400
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert queries == ['old_object']
    assert result['complete_research_visits'][0]['status'] == 'preparation_incomplete'
    assert agent.theory_consolidation['executions'][key]['cursor'] == before['cursor']
    assert_no_new_science(agent, lab)
