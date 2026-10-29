"""Ordinary completion consumes its funded method agenda without a host route."""

from copy import deepcopy

from hegel_machine.cognition.method_feedback import _read
from test_automatic_method_content_proposal import setup_scene, append_source


def deliver(agent, lab, returned):
    record = _read(agent, returned['feedback_ref'])
    return agent._execute('run', {'operator': 'method_action_feedback',
        'candidate_id': 'P0', 'binding': {'candidate_id': 'P0', 'candidate_version': 1,
            'origin_ref': record['origin_ref']}, 'result_ref': record['actual_result_ref']}, lab)


def test_ordinary_feedback_consumes_published_method_question_without_manual_research_route(tmp_path):
    agent, lab, _, p0, _ = setup_scene(tmp_path)
    first = deliver(agent, lab, append_source(agent, p0, 'success'))
    assert first['method_research']['research_result']['method_action_research'][
        'reason'] == 'distinct_completed_sources_required'
    active = agent._active_question_id
    second = append_source(agent, p0, 'missed', missed=True)
    result = deliver(agent, lab, second)
    action = result['method_research']['research_result']['method_action_research']
    assert action['proposals']['candidates'][0]['research_selection_plan'][
        'when_false']['first_segment']['parent_binding'] == 'paid_startup_witnesses'
    assert agent._active_question_id == active
    cost = deepcopy(agent.costs)
    repeated = deliver(agent, lab, second)
    assert repeated['method_research']['new_compute_cost'] == 0
    assert agent.costs == cost and lab.paid == 0


def test_solved_world_does_not_block_separately_funded_method_learning(tmp_path):
    agent, lab, _, p0, _ = setup_scene(tmp_path)
    agent.research_mission = {'purpose': 'task', 'task_supported': True,
        'supported_action': {'action': 'finish', 'arguments': {'model': 'retained.py'}}}
    deliver(agent, lab, append_source(agent, p0, 'success'))
    result = deliver(agent, lab, append_source(agent, p0, 'missed', missed=True))
    assert result['method_research']['research_result']['method_action_research']['proposals']['candidates']
    assert agent.research_mission['task_supported'] and lab.paid == 0


def test_unfunded_method_work_stays_in_agenda_without_spending_task_budget(tmp_path):
    agent, lab, _, p0, _ = setup_scene(tmp_path)
    returned = append_source(agent, p0, 'success')
    for question in agent.questions:
        question.pop('mission', None)
    agent.research_mission = {'purpose': 'task'}
    cost = deepcopy(agent.costs)
    result = deliver(agent, lab, returned)
    assert result['method_research']['reason'] == 'declared_method_mission_required'
    assert result['method_question_refs'] and agent.costs == cost
    assert lab.paid == 0


def test_imported_support_reopens_pending_agenda_without_reinducing_sources(tmp_path, monkeypatch):
    from test_method_proposal_resolution import supported_equivalent

    agent, lab, owner, p0, previous = setup_scene(tmp_path)
    deliver(agent, lab, append_source(agent, p0, 'success'))
    last = append_source(agent, p0, 'missed', missed=True)
    pending = deliver(agent, lab, last)['method_research']['research_result']['method_action_research']
    assert deliver(agent, lab, last)['method_research']['new_compute_cost'] == 0
    original_sources = list(owner['method_action_returns'])
    supported_equivalent(agent, owner, previous, pending)
    # This is the topic import contract: old body support is retained in its
    # original feedback group, without pretending it is a new source return.
    owner['method_action_returns'] = original_sources

    def no_reinduction(*args, **kwargs):
        raise AssertionError('new method support must not repeat retained source induction')

    monkeypatch.setattr('hegel_machine.cognition.research_programs.derive_paid_witness_binding_revision',
                        no_reinduction)
    result = deliver(agent, lab, last)['method_research']
    assert result.get('research_result') is not None
    assert result['research_result']['method_action_research']['proposal_analysis_reused']
    assert result['status'] == 'complete_supported_method_retained'
    assert lab.paid == 0
