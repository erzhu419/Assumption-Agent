"""Unfinished old meanings remain research opportunities, never task winners."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.question_continuation import (
    parent_revision_work, select_parent_revision_work,
)
from hegel_machine.cognition.question_routing import _retained_revision_inputs, register_theories
from test_parent_revision_continuation import revision_scene
from test_question_routing import route, theory
from test_revision_relations import binary, constant, feature


def unfinished_history(tmp_path):
    agent, laboratory, child, _ = revision_scene(tmp_path)
    first = route(agent, laboratory, question={'id': child['id']})
    owner = next(q for q in agent.questions if q['id'] == first['question_id'])
    key, execution = next((key, value) for key, value in agent.theory_consolidation['executions'].items()
                          if key.startswith('relation_transfer:'))
    execution.update(search_completed=False, cursor={'layer': 1, 'offset': 2, 'pending_fits': []})
    execution['result'].update(search_completed=False, status='UNKNOWN', budget_exhausted=True,
                               minimum_compute_for_progress=3)
    owner['frozen_recall_execution']['result']['search_completed'] = False
    register_theories(agent, [theory('next_parent', binary('mul', constant(.7), feature()))])
    incoming = {'id': owner['id'],
                'subject_refs': [{'kind': 'theory', 'id': 'next_parent', 'version': 1}],
                'theory_target': {**deepcopy(owner['theory_target']), 'parent_theory_id': 'next_parent'}}
    agent._open_question(incoming)
    return agent, laboratory, child, owner, key


def test_parent_change_keeps_unfinished_old_meaning_in_the_bounded_agenda(tmp_path):
    agent, _, _, owner, key = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    winner, costs, cursor = deepcopy(agent.evaluated_model), deepcopy(agent.costs), deepcopy(
        agent.theory_consolidation['executions'][key]['cursor'])
    original = deepcopy(old['theory_target'])
    # The before's full predictor failure motivates revision; it does not
    # refute the unfinished space of structural edits.
    parent = agent.theory_index[original['parent_theory_id']]
    parent.update(compatible=False, support={'status': 'refuted', 'evidence': []})
    selected, first_agenda = select_parent_revision_work(agent, owner, 10000)
    assert selected is old and selected['meaning_version'] == 1
    current, second_agenda = select_parent_revision_work(agent, owner, 10000)
    assert current is owner
    entry = next(item for item in first_agenda['task_agenda'] if item['meaning_version'] == 1)
    assert entry['state'] == 'budget_incomplete'
    assert entry['source_ref']['json_path'][-2:] == ['meaning_history', 0]
    assert first_agenda['selected_question_ref']['meaning_version'] == 1
    assert second_agenda['selected_question_ref']['meaning_version'] == 2
    assert old['theory_target'] == original
    assert agent.theory_consolidation['executions'][key]['cursor'] == cursor
    assert agent.evaluated_model == winner and agent.costs == costs


def test_completed_shared_work_does_not_reopen_an_obsolete_alias_cursor(tmp_path):
    agent, _, _, owner, key = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    stale = deepcopy(agent.theory_consolidation['executions'][key])
    agent.theory_consolidation['executions'][key + ':old_alias'] = stale
    agent.theory_consolidation['executions'][key].update(search_completed=True)
    old['frozen_recall_execution']['result'].update(search_completed=True,
        execution_work_groups=[{'execution_key': key, 'candidate_ids': ['fixture_aliases']}])
    entry = next(item for item in parent_revision_work(agent, owner) if item['meaning_version'] == 1)
    assert entry['state'] == 'completed_no_gain'
    assert not stale['search_completed']


def test_old_theory_mission_cannot_spend_again_after_its_suballocation_is_exhausted(tmp_path):
    agent, _, _, owner, _ = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    old['mission'] = {'purpose': 'theory_learning', 'mission_id': 'spent_old_scope',
                      'allocation': {'compute': 3}, 'resource_usage': {'compute': 3}}
    selected, agenda = select_parent_revision_work(agent, owner, 10000)
    assert selected is owner
    entry = next(item for item in agenda['task_agenda'] if item['meaning_version'] == 1)
    assert entry['available_compute'] == 0
    assert entry['reason'] == 'minimum_progress_compute_unfunded'
    assert old['mission']['resource_usage']['compute'] == 3


@pytest.mark.parametrize('condition, expected', [
    ('completed', 'completed_no_gain'), ('refuted', 'refuted'),
    ('premise', 'premise_unknown'), ('unfunded', 'minimum_progress_compute_unfunded'),
])
def test_completed_refuted_unsupported_or_unfunded_old_research_is_not_selected(tmp_path, condition, expected):
    agent, _, _, owner, key = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    execution = agent.theory_consolidation['executions'][key]
    if condition in {'completed', 'refuted'}:
        execution['search_completed'] = True
        execution['result']['status'] = 'refuted' if condition == 'refuted' else 'candidates_registered'
        old['frozen_recall_execution']['result']['search_completed'] = True
    elif condition == 'premise':
        old['support_paths'] = [[{'question_id': 'unavailable_scientific_premise'}]]
    select_parent_revision_work(agent, owner, 10000)
    selected, agenda = select_parent_revision_work(agent, owner, 1 if condition == 'unfunded' else 10000)
    assert selected is owner
    old_entry = next(item for item in agenda['task_agenda'] if item['meaning_version'] == 1)
    assert old_entry['reason'] == expected


def test_unchanged_parent_use_is_required_and_added_rows_require_full_current_comparison(tmp_path):
    agent, _, _, owner, _ = unfinished_history(tmp_path)
    assert len(parent_revision_work(agent, owner)) == 2
    owner['use_sites'] = [{'kind': 'choose_parent_experiment', 'parent_id': 'different_task'}]
    assert [item['meaning_version'] for item in parent_revision_work(agent, owner)] == [2]
    owner['use_sites'] = deepcopy(owner['meaning_history'][0]['use_sites'])
    agent.rows.append(agent.rows[-1])
    old = next(item for item in parent_revision_work(agent, owner) if item['meaning_version'] == 1)
    assert old['requires_current_evidence_comparison']


def test_original_component_versions_and_native_siblings_are_resolved_in_original_order(tmp_path):
    agent, _, _, owner, key = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    execution = agent.theory_consolidation['executions'][key]
    expected = deepcopy(execution['request_basis'])
    register_theories(agent, [theory('current_parent', binary('mul', constant(.9), feature()))])
    resolved = _retained_revision_inputs(agent, old)
    actual = resolved['_retained_execution_inputs'][key.split('@1:', 1)[1]]
    assert actual['component_theory_ids'] == expected['component_theory_ids']
    assert [(item['id'], item['version']) for item in actual['_retained_components']] == [
        (item['id'], item['version']) for item in expected['components']]
    assert actual['_retained_parent']['version'] == expected['parent']['version'] == 1
    assert agent.theory_index['current_parent']['version'] == 2


def test_historical_export_uses_original_nested_meaning_without_fitting(tmp_path, monkeypatch):
    from hegel_machine.cognition.finite_native_author import export_revision_candidates
    from hegel_machine.cognition.language import Program

    agent, _, _, owner, _ = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    old['revision_candidate_exports'] = []
    winner = deepcopy(agent.evaluated_model)
    monkeypatch.setattr(Program, 'predict', lambda *args, **kwargs: pytest.fail('Export must not predict or fit'))
    result = export_revision_candidates(agent, old)
    assert result['new_exports'] > 0
    assert all(f'revision_{agent.questions.index(owner):04d}_0001/' in item['model']
               for item in result['exports'])
    assert owner['meaning_version'] == 2 and agent.evaluated_model == winner
    assert export_revision_candidates(agent, old)['new_exports'] == 0


def test_public_route_consumes_the_selected_historical_meaning_not_the_current_parent(tmp_path, monkeypatch):
    from hegel_machine.cognition import theory_consolidation
    from hegel_machine.cognition.language import Program

    agent, laboratory, _, owner, key = unfinished_history(tmp_path)
    old = owner['meaning_history'][0]
    old['candidate_theory_ids'] = []
    winner, target = deepcopy(agent.evaluated_model), deepcopy(old['theory_target'])
    original_cursor = deepcopy(agent.theory_consolidation['executions'][key]['cursor'])
    calls = []

    def recall(current_agent, selected, arguments):
        calls.append((selected, arguments))
        return {'status': 'UNKNOWN', 'generated_theory_ids': [], 'new_compute_cost': 0,
                'search_completed': False, 'work': []}

    monkeypatch.setattr(theory_consolidation, 'recall_frozen_theories', recall)
    monkeypatch.setattr(theory_consolidation, 'transfer_theory',
                        lambda *a, **k: pytest.fail('This route regression does not fit'))
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('No new predictions'))
    result = route(agent, laboratory, question={'id': owner['id']})
    assert calls and calls[0][0] is owner['meaning_history'][0]
    assert calls[0][0]['meaning_version'] == 1
    assert calls[0][1]['_retained_parent']['id'] == target['parent_theory_id']
    assert calls[0][1]['_retained_execution_inputs']
    assert result['research_track_question_ref'] == {'question_id': owner['id'], 'meaning_version': 1}
    assert old['theory_target'] == target and owner['theory_target']['parent_theory_id'] == 'next_parent'
    assert agent.evaluated_model == winner
    assert agent.theory_consolidation['executions'][key]['cursor'] == original_cursor
