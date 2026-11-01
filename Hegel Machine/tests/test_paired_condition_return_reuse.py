"""A non-paired return does not repeat completed paired condition research."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.common_object_research import record_native_parent_research_return
from hegel_machine.cognition.method_action_research import review_native_method_returns
from hegel_machine.cognition.method_feedback import ensure_python_method_topic
from hegel_machine.cognition.method_conditions import research_paired_method_conditions
from hegel_machine.cognition.models import SearchBudget
from test_native_paired_method_feedback import common, deliver, route, save
from test_v8_paired_method_conditions import METHOD, pair


def prohibit_learning(monkeypatch):
    from hegel_machine.cognition import condition_learning
    def unexpected(*args, **kwargs):
        raise AssertionError('The unchanged paid finite condition search ran again')
    monkeypatch.setattr(condition_learning, 'learn_condition', unexpected)
    monkeypatch.setattr(condition_learning, 'learn_affine_condition', unexpected)


def development():
    return [pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)]


def test_restored_topic_reuses_paid_condition_search_after_a_nonpaired_return(tmp_path, monkeypatch):
    original = common(tmp_path / 'original')
    for name, stage, cheaper in [('a', 'unresolved', True), ('b', 'unresolved', True),
                                 ('c', 'already_explained', False)]:
        deliver(original, tmp_path, name, stage=stage, cheaper=cheaper)
    result = route(original)['method_action_research']
    learned, = result['paired_condition_research']
    assert learned['new_compute_cost'] > 0 and learned['candidates']
    paid_ref = save(tmp_path / 'paid_review.json', result)
    owner = deepcopy(next(q for q in original.questions if q['id'] == 'task:method_choice'))
    for key in ('routing_execution', 'method_use_result', 'completed_agenda_basis'):
        owner.pop(key, None)
    owner['method_use_plan']['paired_condition_research_refs'] = [{
        **paid_ref, 'json_path': ['paired_condition_research', 0]}]
    topic_ref = save(tmp_path / 'previous_topic.json', {
        'question': owner, 'questions': [owner], 'methods': []})
    restored = common(tmp_path / 'restored')
    continued = ensure_python_method_topic(restored,
        parent_question_ref={'question_id': 'task'}, topic_ref=topic_ref)
    record_native_parent_research_return(restored, source_task_ref={'source': 'actual_continuation'},
        result_ref=save(tmp_path / 'new_return.json', {'status': 'improved',
            'execution_complete': True, 'method_refs': [{'file': 'actual_rebound_method.json'}]}))
    prohibit_learning(monkeypatch)
    reviewed = review_native_method_returns(restored, continued, compute_limit=5)
    reused, = reviewed['paired_condition_research']
    assert reviewed['status'] != 'scoped_unknown'
    assert reviewed['new_compute_cost'] == 5
    assert reused['reused'] and reused['new_compute_cost'] == 0
    assert reused['candidates'] == learned['candidates']
    assert reused['preferences'] == learned['preferences']
    assert reused['paired_invocations'] == 3 and len(reviewed['ordinary_returns']) == 4
    assert not reused['general_guard_supported']


@pytest.mark.parametrize('change', ['profile', 'preference', 'search_contract'])
def test_changed_condition_basis_does_not_reuse_old_search(change):
    pairs = development()
    previous = research_paired_method_conditions(METHOD, pairs, SearchBudget(100000))
    if change == 'profile':
        pairs[0]['baseline']['profile']['observation_count'] += 1
        pairs[0]['challenger']['profile'] = deepcopy(pairs[0]['baseline']['profile'])
    elif change == 'preference':
        pairs[0]['challenger']['costs']['compute_cost'] = 200
    else:
        previous['basis']['condition_search'] = 'earlier_single_coordinate_search'
    result = research_paired_method_conditions(METHOD, pairs, SearchBudget(0), previous=previous)
    assert result['budget_incomplete'] and not result.get('reused')


def test_reused_search_rechecks_new_validation_instead_of_inheriting_support(monkeypatch):
    pairs = development()
    validation = pair(3, role='condition_validation')
    accepted = research_paired_method_conditions(METHOD, [*pairs, validation], SearchBudget(100000))
    assert accepted['general_guard_supported']
    validation['challenger']['costs']['compute_cost'] = 200
    prohibit_learning(monkeypatch)
    rejected = research_paired_method_conditions(METHOD, [*pairs, validation],
        SearchBudget(0), previous=accepted)
    assert rejected['reused'] and rejected['compute_cost'] == 0
    assert rejected['status'] == 'development_condition_without_group_support'
    assert not rejected['general_guard_supported'] and rejected['condition'] is None
    assert rejected['candidate_pool'] == accepted['candidate_pool']


def test_validation_filter_does_not_erase_paid_competing_candidates(monkeypatch):
    pairs = development()
    previous = research_paired_method_conditions(METHOD, pairs, SearchBudget(100000))
    all_candidates = deepcopy(previous['candidates'])
    assert len(all_candidates) > 1
    # A preceding validation may expose only its accepted subset. The saved
    # search pool still owns every candidate; no finite fit is needed again.
    previous.update(status='paired_condition_supported_on_declared_groups',
        candidates=previous['candidates'][:1], general_guard_supported=True)
    prohibit_learning(monkeypatch)
    result = research_paired_method_conditions(METHOD, pairs, SearchBudget(0), previous=previous)
    assert result['reused'] and result['candidates'] == all_candidates
    assert not result['general_guard_supported']
