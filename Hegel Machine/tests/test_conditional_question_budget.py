"""Conditional plans consume actual finite answers under their owning mission."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.question_continuation import pending_query_arguments
from test_finite_readout import finite
from test_inquiry_revision import action
from test_question_routing import (SQUARE, future_question, parent, probe, readout,
                                   route, scene, theory)


def conditional(agent, vector):
    plan = probe(agent, 0., 'conditional_probe')
    plan['branches'] = [{'when': {'value': value, 'atol': 0.},
        'actions': [action(level)], 'query': readout(level, len(agent.observations) + 1)}
        for value, level in ((0., 2.), (10., 5.))]
    plan['observation_cost'] = 2
    if vector:
        plan['query'] = finite([plan['query']])
        for branch in plan['branches']:
            branch['when']['value'] = [branch['when']['value']]
            branch['query'] = finite([branch['query']])
    return plan


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('allowance', [8, 9])
def test_actual_branch_readout_respects_the_local_mission_after_first_purchase(
        tmp_path, monkeypatch, vector, allowance):
    agent, lab = scene(tmp_path)
    question = future_question(agent)
    question['mission'] = {'mission_id': 'conditional_theory', 'purpose': 'theory_learning',
                           'theory_learning_allocation': {'compute': allowance}}
    result = route(agent, lab, question=question,
        theories=[theory('linear'), theory('quadratic', SQUARE)],
        experiments=[conditional(agent, vector)], mission={'purpose': 'task_prediction'})
    assert result['status'] == 'selected'
    retained = next(q for q in agent.questions if q['id'] == question['id'])
    assert retained['mission']['resource_usage']['compute'] == 8
    original = lab.execute

    def surprising_first(action):
        measured = original(action)
        measured['rows'][0]['target'] = 10.
        return measured

    monkeypatch.setattr(lab, 'execute', surprising_first)
    before, paid = agent.costs['compute_cost'], lab.paid
    acquired = agent._execute('query', {'pending_question': 'task'}, lab)
    continuation = acquired['conditional_continuation']
    assert lab.paid == paid + 1 and lab.actions[-1] == action(0.)
    assert agent.costs['compute_cost'] - before == allowance - 8
    assert retained['mission']['resource_usage']['compute'] == allowance
    assert agent.research_mission == {'purpose': 'task_prediction'}
    if allowance == 8:
        assert continuation['status'] == 'scoped_unknown'
        assert continuation['readout']['reason'] == 'unknown_due_to_incomplete_search'
        assert 'value' not in continuation['readout'] and 'pending_query' not in parent(agent)
    else:
        assert continuation['status'] == 'second_step_prepared'
        assert continuation['first_outcome']['value'] == ([10.] if vector else 10.)
        assert continuation['first_outcome']['evidence'] == [paid]
        assert parent(agent)['pending_query_source']['stage'] == 2
        assert pending_query_arguments(agent, 'task')['actions'] == [action(5.)]
        assert continuation['purchased_second_step'] is False
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_vector_branches_keep_ambiguous_actual_outcome_unresolved(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    plan = conditional(agent, True)
    # Candidate forecasts still take the unique first branch at0, so this plan
    # can be prepared. The measured10 lies in both later authored intervals.
    plan['branches'].append({'when': {'value': [10.], 'atol': 1.},
        'actions': [action(7.)], 'query': finite([readout(7., len(agent.observations) + 1)])})
    result = route(agent, lab, question=future_question(agent),
        theories=[theory('linear'), theory('quadratic', SQUARE)], experiments=[plan])
    assert result['status'] == 'selected'
    original = lab.execute

    def surprising_first(selected):
        measured = original(selected)
        measured['rows'][0]['target'] = 10.
        return measured

    monkeypatch.setattr(lab, 'execute', surprising_first)
    acquired = agent._execute('query', {'pending_question': 'task'}, lab)
    assert acquired['conditional_continuation']['reason'] == 'first_step_has_no_unique_declared_branch'
    assert 'pending_query' not in parent(agent)
