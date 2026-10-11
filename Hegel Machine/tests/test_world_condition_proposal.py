"""Finite condition proposals reach the author and face real controller outcomes."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.world_llm_inquiry import refresh_condition_proposal
from hegel_machine.cognition.world_reference_response import response_premise_design_action
from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BINDING, _ready
from test_world_premise_dispatch import stalled
from test_world_premise_proposal_inputs import offered_path
from test_world_reference_hypothesis import PATH


def prepare_proposal(tmp_path, *, calls=1):
    agent, lab = _ready(tmp_path)
    _, comparison = stalled(agent, lab)
    action = response_premise_design_action(comparison)
    action['arguments']['max_model_calls'] = 1
    provider = agent.provider = Provider([
        ('finish', {'lesson': 'Return the computed rules for the parent to inspect.'})
        for _ in range(calls)
    ])
    return agent, lab, provider, action, offered_path(comparison)


def proposal_view(request, path):
    proposals = json.loads(request['files']['method_decision.json'])['explanation_proposals']
    return next(row for row in proposals if row['session_id'] == 'proposal:' + path)


@pytest.mark.parametrize('empty_authored_draft', [False, True])
def test_condition_rules_and_computed_trial_are_visible_before_first_author_decision(
        tmp_path, empty_authored_draft):
    agent, lab, provider, action, path = prepare_proposal(tmp_path)
    if empty_authored_draft:
        agent._execute('write', {'path': path, 'code': '''
def design(spec, observations, memory):
    return {'hypotheses': [{'id': 'shared', 'claim': 'The response condition is shared'},
                           {'id': 'local', 'claim': 'The response condition is local'}],
            'experiments': []}

def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'The authored draft supplies no test.',
            'evidence': []}
'''}, lab)
    scientific = deepcopy((agent.questions, agent.observations, lab.actions))
    sources = {name: (agent.workspace / name).read_text() for name in (PATH, BINDING)}
    compute = agent.costs['research_compute_units']
    python = agent.costs['python_execution_seconds']

    result = agent._execute(action['action'], action['arguments'], lab)

    request, = provider.requests
    conditions = request['events'][-1]['feedback']['condition_research']
    assert request['events'][-1]['action'] == 'condition_synthesis'
    assert conditions['candidates']
    assert any(rule['status'] == 'compatible' for rule in conditions['candidates'])
    assert all(rule['status'] != 'supported' for rule in conditions['candidates'])
    assert conditions['selected_experiment'] is not None
    assert set(conditions['selected_experiment']['predictions'].values()) == {False, True}
    trial = conditions['experiment_action']
    assert trial['arguments']['condition_proposal'] == path
    assert trial['arguments']['path'] == PATH
    assert proposal_view(request, path)['condition_research'] == conditions
    assert result['proposal']['condition_research'] == conditions
    assert agent.llm_inquiries['proposal:' + path]['proposal']['condition_research'] == conditions
    assert 'inputs' not in conditions
    assert 'inputs' in agent.llm_inquiries['proposal:' + path]['condition_research']
    assert agent.costs['research_compute_units'] - compute == conditions['compute_cost'] > 0
    assert agent.costs['python_execution_seconds'] > python
    assert (agent.questions, agent.observations, lab.actions) == scientific
    assert {name: (agent.workspace / name).read_text() for name in sources} == sources
    assert (agent.workspace / path).exists() == empty_authored_draft
    authored_status = ('no_experiments_proposed' if empty_authored_draft else
                       'NO_EXECUTABLE_EXPLANATION_PROPOSED')
    for current in (proposal_view(request, path), result['proposal']):
        assert current['authored_deliverable_status'] == authored_status
        assert current['deliverable_status'] == 'condition_experiment_available'
        assert current['experiment_action'] == trial
        assert current['contrast_action'] is None
        if empty_authored_draft:
            assert current['design_comparison']['diagnosis'] == 'no_experiments_proposed'


def test_unchanged_paid_data_after_saved_session_resume_reuses_condition_search(tmp_path):
    agent, lab, provider, action, path = prepare_proposal(tmp_path, calls=2)
    initial = agent._execute(action['action'], action['arguments'], lab)
    first = deepcopy(initial['proposal']['condition_research'])
    # Resume the persisted inquiry data rather than retaining a private search cache.
    agent._save_state()
    agent.llm_inquiries = json.loads((agent.workspace / 'llm_inquiries.json').read_text())
    compute = agent.costs['research_compute_units']
    physical = deepcopy((agent.questions, agent.observations, lab.actions))

    returned = agent._execute(action['action'], action['arguments'], lab)

    cached = returned['proposal']['condition_research']
    assert cached['cached'] is True
    assert cached['compute_cost'] == 0
    assert agent.costs['research_compute_units'] == compute
    assert cached['candidates'] == first['candidates']
    assert cached['selected_experiment'] == first['selected_experiment']
    assert cached['experiment_action'] == first['experiment_action']
    assert proposal_view(provider.requests[-1], path)['condition_research'] == cached
    assert returned['proposal']['experiment_action'] == cached['experiment_action']
    assert returned['proposal']['deliverable_status'] == 'condition_experiment_available'
    assert agent.llm_inquiries['proposal:' + path]['invocations'] == 2
    assert (agent.questions, agent.observations, lab.actions) == physical


def test_parent_executes_returned_condition_trial_and_same_author_receives_real_refutations(tmp_path):
    agent, lab, provider, action, path = prepare_proposal(tmp_path, calls=2)
    initial = agent._execute(action['action'], action['arguments'], lab)
    before = deepcopy(initial['proposal']['condition_research'])
    trial = before['selected_experiment']
    programs = {row['id']: row['program'] for row in before['candidates']}
    paid = len(agent.observations)
    compute = agent.costs['research_compute_units']
    returned_action = initial['proposal']['experiment_action']
    assert returned_action == before['experiment_action']

    result = agent._execute(returned_action['action'], returned_action['arguments'], lab)

    assert result['requested_setting_reached']
    assert len(agent.observations) > paid
    assert result['new_observation_indices'] == list(range(paid, len(agent.observations)))
    actual = next(row for row in result['observed_responses']
                  if row['crystal'] == trial['number'] and row['frequency'] == trial['prediction'])
    assert actual['evidence'] and set(actual['evidence']) <= set(result['new_observation_indices'])
    contradicted = {identifier for identifier, predicted in trial['predictions'].items()
                   if predicted != actual['activated']}
    assert contradicted
    after = result['condition_research']
    indexed = {row['id']: row for row in after['candidates']}
    assert all(indexed[identifier]['status'] == 'refuted' for identifier in contradicted)
    assert all(indexed[identifier]['program'] == program for identifier, program in programs.items())
    assert all(row['status'] != 'supported' for row in after['candidates'])
    assert agent.costs['research_compute_units'] - compute == (
        after['compute_cost'] + result['response_comparison']['compute_cost']) > 0
    session = agent.llm_inquiries['proposal:' + path]
    assert session['proposal']['condition_research'] == after
    assert session['proposal']['authored_deliverable_status'] == 'NO_EXECUTABLE_EXPLANATION_PROPOSED'
    event = session['events'][-1]
    assert event['action'] == 'condition_return'
    assert event['feedback']['condition_research'] == after
    assert event['feedback']['physical_test']['observed_responses'] == result['observed_responses']
    assert event['feedback']['physical_test']['new_observation_indices'] == result['new_observation_indices']
    if after['experiment_action']:
        assert after['experiment_action']['arguments']['condition_proposal'] == path
        assert session['proposal']['experiment_action'] == after['experiment_action']
        assert session['proposal']['deliverable_status'] == 'condition_experiment_available'
    else:
        assert 'experiment_action' not in session['proposal']
        assert session['proposal']['deliverable_status'] == session['proposal']['authored_deliverable_status']

    compute = agent.costs['research_compute_units']
    agent._execute(action['action'], action['arguments'], lab)

    request = provider.requests[-1]
    assert any(row['action'] == 'condition_return' for row in request['events'])
    current = proposal_view(request, path)['condition_research']
    assert current['candidates'] == after['candidates']
    assert current['cached'] and current['compute_cost'] == 0
    assert agent.costs['research_compute_units'] == compute


def test_new_conflicting_paid_response_removes_the_stale_top_level_condition_action(tmp_path):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    returned = agent._execute(action['action'], action['arguments'], lab)
    assert returned['proposal']['experiment_action']
    assert lab.active(1)
    # An actual changed response at the same measured coordinates invalidates
    # deterministic rules. The proposal must stop offering its obsolete trial.
    lab.required[1] += 10.
    first = len(agent.observations)
    agent._execute('query', {'actions': [
        {'action': 'TALK', 'arg1': 101}, {'chosen_dialog_option_int': 1}]}, lab)

    conditions = refresh_condition_proposal(agent, path)

    assert not lab.active(1) and len(agent.observations) == first + 2
    assert conditions['experiment_action'] is None
    assert not any(rule['status'] == 'compatible' for rule in conditions['candidates'])
    proposal = agent.llm_inquiries['proposal:' + path]['proposal']
    assert proposal['condition_research'] == conditions
    assert 'experiment_action' not in proposal
    assert proposal['authored_deliverable_status'] == 'NO_EXECUTABLE_EXPLANATION_PROPOSED'
    assert proposal['deliverable_status'] == proposal['authored_deliverable_status']
