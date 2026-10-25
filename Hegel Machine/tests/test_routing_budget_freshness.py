"""Temporary ordinary-completion reserves must not invalidate a saved plan."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.question_routing import pending_selection_current
from hegel_machine.cognition.finite_native_author import ordinary_completion_reserve
from test_inquiry_revision import action
from test_registered_finite_policy import registered_scene
from test_revision_context_fitting import FITTING


def reserved_selection(tmp_path, *, exhausted_route=False):
    agent, lab, declared = registered_scene(tmp_path)
    owner = next(q for q in agent.questions if q['id'] == declared['id'])
    # Exercise the real finite-author reserve without requiring a new relation.
    owner['theory_target']['revision_fitting'] = deepcopy(FITTING)
    agent.knowledge['finite_native_author']['question']['theory_target'] = deepcopy(
        owner['theory_target'])
    if exhausted_route:
        reserve = ordinary_completion_reserve(agent, owner)
        agent.config = replace(agent.config,
            compute_budget=agent.costs['compute_cost'] + reserve['compute_cost'])
    original_config = agent.config
    before = deepcopy(agent.costs)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    assert agent.config == original_config
    assert owner['ordinary_completion_reserve']['compute_cost'] > 0
    assert owner['ordinary_completion_reserve']['route_compute_cost'] <= (
        original_config.compute_budget - before['compute_cost']
        - owner['ordinary_completion_reserve']['compute_cost'])
    return agent, lab, owner


def test_registered_reserve_restores_nominal_budget_without_staling_real_pending_query(tmp_path):
    agent, lab, owner = reserved_selection(tmp_path)
    parent = next(q for q in agent.questions if q['id'] == 'task')
    result = owner['routing_execution']['result']
    assert result['parent_use']['status'] == 'parent_query_prepared'
    assert pending_selection_current(agent, parent)
    assert owner['routing_execution']['basis']['compute_limit'] == agent.config.compute_budget
    state = json.loads((agent.workspace / 'research_state.json').read_text())
    saved = next(q for q in state['questions'] if q['id'] == owner['id'])
    assert saved['routing_execution']['basis']['compute_limit'] == state['limits']['compute_budget']
    pending = deepcopy(parent['pending_query'])
    before = deepcopy(agent.costs)
    feedback = agent._execute('query', {'pending_question': 'task'}, lab)
    assert lab.actions[-1] == pending['actions'][0]
    assert len(feedback['observations']) == len(pending['actions'])
    assert agent.costs['observation_cost'] - before['observation_cost'] == 1
    assert agent.costs['model_calls'] == 0
    assert agent.costs['compute_cost'] <= agent.config.compute_budget


@pytest.mark.parametrize('changed', ['nominal_budget', 'paid_evidence'])
def test_real_budget_or_evidence_change_still_rejects_pending_before_purchase(tmp_path, changed):
    agent, lab, owner = reserved_selection(tmp_path)
    parent = next(q for q in agent.questions if q['id'] == 'task')
    if changed == 'nominal_budget':
        agent.config = replace(agent.config, compute_budget=agent.config.compute_budget + 37)
    else:
        agent._execute('query', {'actions': [action(7.)]}, lab)
    before = deepcopy(agent.costs)
    paid = lab.paid
    assert not pending_selection_current(agent, parent)
    with pytest.raises(ValueError, match='rebuilding'):
        agent._execute('query', {'pending_question': 'task'}, lab)
    assert lab.paid == paid and agent.costs == before
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    assert pending_selection_current(agent, parent)
    assert owner['routing_execution']['basis']['compute_limit'] == agent.config.compute_budget


def test_zero_execution_cap_does_not_gain_compute_from_nominal_budget(tmp_path):
    agent, lab, owner = reserved_selection(tmp_path, exhausted_route=True)
    assert owner['ordinary_completion_reserve']['route_compute_cost'] == 0
    assert owner['routing_execution']['basis']['compute_limit'] == agent.config.compute_budget
    assert not owner['routing_execution']['result'].get('parent_use', {}).get('changed')
    assert lab.paid == agent.costs['observation_cost'] == 1
    assert agent.costs['model_calls'] == 0


def test_ordinary_route_without_reserve_keeps_actual_compute_limit_in_basis(tmp_path):
    from test_question_routing import scene, future_question, route, theory, probe, SQUARE

    agent, lab = scene(tmp_path)
    owner = future_question(agent)
    route(agent, lab, question=owner, theories=[theory('one'), theory('two', SQUARE)],
          experiments=[probe(agent)])
    actual = next(q for q in agent.questions if q['id'] == owner['id'])
    parent = next(q for q in agent.questions if q['id'] == 'task')
    assert actual['routing_execution']['basis']['compute_limit'] == agent.config.compute_budget
    assert pending_selection_current(agent, parent)
    agent.config = replace(agent.config, compute_budget=agent.config.compute_budget - 1)
    assert not pending_selection_current(agent, parent)
