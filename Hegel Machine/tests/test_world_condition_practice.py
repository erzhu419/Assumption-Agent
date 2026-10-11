"""An author can choose an available condition trial and inspect its real outcome."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.world_response_conditions import predict_response_condition
from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BINDING
from test_world_condition_proposal import prepare_proposal, proposal_view
from test_world_reference_hypothesis import PATH


class ChoosingAuthor(Provider):
    """Select actions from the current request, without fabricating evidence."""

    def __init__(self, agent, choices):
        super().__init__([])
        self.agent = agent
        self.choices = iter(choices)
        self.frames = []

    def next_action(self, request):
        self.frames.append(self.agent._inquiry_stack[-1])
        chosen = next(self.choices)(request)
        self.steps = iter([(chosen['action'], chosen['arguments'])])
        return super().next_action(request)


def finish(request):
    return {'action': 'finish', 'arguments': {'lesson': 'Return the actual condition feedback.'}}


def plain_query(request):
    return {'action': 'query', 'arguments': {'actions': [{'action': 'TALK', 'arg1': 101}]}}


def current_trial(path, *, declaration=None, edit=None):
    def choose(request):
        selected = deepcopy(proposal_view(request, path)['experiment_action'])
        if declaration:
            selected['arguments'][declaration] = deepcopy(request['llm_inquiry']['assignment'])
        if edit:
            edit(selected['arguments'])
        return selected
    return choose


def assert_current_condition_advice(request, path):
    conditions = proposal_view(request, path)['condition_research']
    advice = request['research']['research_advice']
    assert advice == json.loads(request['files']['method_decision.json'])['research_advice']
    assert advice['question_id'] == request['llm_inquiry']['assignment']['id']
    assert advice['diagnosis'] == conditions['status']
    assert advice['status'] == conditions['confirmation_status'] == 'unconfirmed'
    assert advice['compatible_candidates'] == sum(
        row['status'] == 'compatible' for row in conditions['candidates'])
    assert advice['refuted_candidates'] == sum(
        row['status'] == 'refuted' for row in conditions['candidates'])
    assert advice.get('next_action') == conditions['experiment_action']
    assert advice.get('target_consequences') == conditions.get('target_consequences')
    if conditions['experiment_action'] is None:
        assert 'next_action' not in advice
    return advice


@pytest.mark.parametrize('declaration', [None, 'question', 'research_question'])
def test_author_chosen_condition_trial_returns_actual_refutations_before_its_next_decision(
        tmp_path, declaration):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    action['arguments']['max_model_calls'] = 2
    provider = agent.provider = ChoosingAuthor(agent, [
        current_trial(path, declaration=declaration), finish])
    prior = agent._question({'id': 'retained_reference', 'parent_id': 'task',
        'question': 'Was reference 1 active at its observed initial setting?'})
    agent._record_answer(prior, {'status': 'supported',
        'answer': {'initial_reference_active': True}, 'evidence': [0]})
    retained = deepcopy(prior)
    task = deepcopy(next(question for question in agent.questions if question['id'] == 'task'))
    sources = {name: (agent.workspace / name).read_text() for name in (PATH, BINDING)}
    paid = len(agent.observations)
    agent.research_advice = {'question_id': 'old_numeric_binding', 'status': 'refuted',
        'diagnosis': 'representation_inadequate',
        'next_action': {'action': 'run', 'arguments': {'operator': 'research_update',
            'path': BINDING, 'question': {'id': 'old_numeric_binding'}}}}

    result = agent._execute(action['action'], action['arguments'], lab)

    assert result['status'] == 'inquiry_finished'
    assert result['model_calls'] == 2 and len(provider.requests) == 2
    before = proposal_view(provider.requests[0], path)['condition_research']
    initial_advice = assert_current_condition_advice(provider.requests[0], path)
    assert initial_advice['compatible_candidates'] > 0
    assert initial_advice['next_action'] == before['experiment_action']
    assert initial_advice['target_consequences']['targets']
    request = provider.requests[1]
    assert request['llm_inquiry']['stage'] == 'proposal'
    assert request['events'][-1]['action'] == 'run'
    actual = request['events'][-1]['feedback']
    assert actual['requested_setting_reached']
    assert actual['status'] != 'proposal_action_deferred'
    assert actual['new_observation_indices'] == list(range(paid, len(agent.observations)))
    assert len(agent.observations) > paid
    selected = before['selected_experiment']
    outcome = next(row for row in actual['observed_responses']
                   if row['crystal'] == selected['number'] and row['frequency'] == selected['prediction'])
    refuted = {identifier for identifier, forecast in selected['predictions'].items()
               if forecast != outcome['activated']}
    assert refuted
    after = actual['condition_research']
    rules = {row['id']: row for row in after['candidates']}
    assert all(rules[identifier]['status'] == 'refuted' for identifier in refuted)
    assert all(rules[row['id']]['program'] == row['program'] for row in before['candidates'])
    assert all(rule['status'] != 'supported' for rule in after['candidates'])
    current = proposal_view(request, path)
    assert current['condition_research'] == after
    assert current.get('experiment_action') == after['experiment_action']
    returned_advice = assert_current_condition_advice(request, path)
    assert returned_advice['refuted_candidates'] >= len(refuted)
    assert agent.research_advice == returned_advice
    assert result['proposal']['condition_research'] == after
    assert result['proposal']['deferred_actions'] == []
    assignment_id = action['arguments']['question']['id']
    enclosing = next(question for question in agent.questions if question['id'] == assignment_id)
    physical = next(question for question in agent.questions if question['id'] == actual['question_id'])
    assert physical['id'] != enclosing['id'] and physical['parent_id'] == enclosing['id']
    assert physical['status'] in {'supported', 'refuted'}
    assert enclosing['status'] not in {'supported', 'refuted'} and 'answer' not in enclosing
    assert next(question for question in agent.questions if question['id'] == prior['id']) == retained
    parent = next(question for question in agent.questions if question['id'] == 'task')
    assert parent['question'] == task['question'] and parent['evidence'] == task['evidence']
    assert parent.get('answer') == task.get('answer')
    assert parent['status'] == 'reopened' and parent['reopened_by'] == physical['id']
    assert {name: (agent.workspace / name).read_text() for name in sources} == sources
    assert all(frame['stage'] == 'proposal' for frame in provider.frames)
    assert agent._inquiry_stack == [] and agent.status == 'running'


def test_failed_condition_action_restores_proposal_stage_and_other_actions_still_defer(tmp_path):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    action['arguments']['max_model_calls'] = 3

    def wrong_public_number(arguments):
        arguments['experiment']['number'] = 999

    provider = agent.provider = ChoosingAuthor(agent, [
        current_trial(path, edit=wrong_public_number), plain_query, finish])
    paid, actions = len(agent.observations), deepcopy(lab.actions)

    result = agent._execute(action['action'], action['arguments'], lab)

    assert result['status'] == 'inquiry_finished'
    assert result['model_calls'] == 3
    error = provider.requests[1]['events'][-1]['feedback']['error']
    assert 'Read the selected crystal and reactor number' in error
    deferred = provider.requests[2]['events'][-1]['feedback']
    assert deferred['status'] == 'proposal_action_deferred'
    assert len(result['proposal']['deferred_actions']) == 1
    assert result['proposal']['deferred_actions'][0]['action'] == 'query'
    assert len(agent.observations) == paid and lab.actions == actions
    assert all(request['llm_inquiry']['stage'] == 'proposal' for request in provider.requests)
    assert all(frame['stage'] == 'proposal' for frame in provider.frames)
    assert agent._inquiry_stack == []


def test_exhausted_world_budget_restores_proposal_stage_without_confirming_assignment(tmp_path):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    action['arguments']['max_model_calls'] = 2
    provider = agent.provider = ChoosingAuthor(agent, [current_trial(path)])
    paid = len(agent.observations)
    agent.limits['environment_steps'] = lab.environment_steps + 1

    result = agent._execute(action['action'], action['arguments'], lab)

    assert result['status'] == 'environment_budget_exhausted'
    assert result['model_calls'] == 1 and len(provider.requests) == 1
    assert len(agent.observations) == paid + 1
    assert provider.frames[0]['stage'] == 'proposal'
    assert agent._inquiry_stack == []
    assignment_id = action['arguments']['question']['id']
    enclosing = next(question for question in agent.questions if question['id'] == assignment_id)
    assert enclosing['status'] not in {'supported', 'refuted'} and 'answer' not in enclosing
    assert all(row['status'] != 'supported' for row in result['proposal']['condition_research']['candidates'])


def test_condition_trial_without_its_proposal_association_remains_deferred(tmp_path):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    action['arguments']['max_model_calls'] = 2
    provider = agent.provider = ChoosingAuthor(agent, [
        current_trial(path, edit=lambda arguments: arguments.pop('condition_proposal')), finish])
    paid, actions = len(agent.observations), deepcopy(lab.actions)

    result = agent._execute(action['action'], action['arguments'], lab)

    assert result['status'] == 'inquiry_finished'
    assert provider.requests[1]['events'][-1]['feedback']['status'] == 'proposal_action_deferred'
    assert len(result['proposal']['deferred_actions']) == 1
    assert len(agent.observations) == paid and lab.actions == actions
    assert all(frame['stage'] == 'proposal' for frame in provider.frames)


def test_resumed_proposal_uses_current_assignment_without_relabeling_old_answers_or_events(tmp_path):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    agent._execute(action['action'], action['arguments'], lab)
    old_id = action['arguments']['question']['id']
    old_question = agent._question(deepcopy(action['arguments']['question']))
    agent._record_answer(old_question, {'status': 'inconclusive',
        'answer': {'scope': 'The original paid responses do not establish the response law.'},
        'evidence': [0]})
    old_answer = deepcopy(old_question)
    session_id = 'proposal:' + path
    previous_events = deepcopy(agent.llm_inquiries[session_id]['events'])
    assert any(event.get('inquiry_question_id') == old_id for event in previous_events)
    current_id = 'revised_response_premise_assignment'
    action['arguments']['question']['id'] = current_id
    action['arguments']['max_model_calls'] = 2
    provider = agent.provider = ChoosingAuthor(agent, [
        current_trial(path, declaration='question'), finish])

    returned = agent._execute(action['action'], action['arguments'], lab)

    assert returned['status'] == 'inquiry_finished' and returned['model_calls'] == 2
    session = agent.llm_inquiries[session_id]
    assert session['question_id'] == current_id and session['invocations'] == 2
    assert session['events'][:len(previous_events)] == previous_events
    assert next(question for question in agent.questions if question['id'] == old_id) == old_answer
    for request in provider.requests:
        assert request['llm_inquiry']['assignment']['id'] == current_id
        assert proposal_view(request, path)['question_id'] == current_id
        assert assert_current_condition_advice(request, path)['question_id'] == current_id
    actual = provider.requests[1]['events'][-1]['feedback']
    assert actual['requested_setting_reached']
    assert actual['question_id'] not in {old_id, current_id}
    physical = next(question for question in agent.questions if question['id'] == actual['question_id'])
    assert physical['parent_id'] == current_id and physical['status'] in {'supported', 'refuted'}
    enclosing = next(question for question in agent.questions if question['id'] == current_id)
    assert enclosing['status'] not in {'supported', 'refuted'} and 'answer' not in enclosing


def test_parent_chosen_new_point_returns_to_current_conditions_without_repeated_association(tmp_path):
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    proposed = agent._execute(action['action'], action['arguments'], lab)['proposal']
    assert agent._inquiry_stack == []
    session = agent.llm_inquiries['proposal:' + path]
    before = deepcopy(session['condition_research'])
    chosen = deepcopy(proposed['experiment_action'])
    chosen['arguments'].pop('condition_proposal')
    point = {'number': 1, 'setting': 101.}
    if chosen['arguments']['experiment'] == point:
        point['setting'] = 102.
    assert chosen['arguments']['experiment'] != point
    chosen['arguments']['experiment'] = point
    inputs = before['inputs']
    predictions = {row['id']: predict_response_condition(row,
        inputs['measurements']['1'], inputs['bound_rows']['1'], point['setting'])
        for row in before['candidates'] if row['status'] == 'compatible'}
    previous_question = deepcopy(next(question for question in agent.questions if question['id'] == 'computed'))
    paid, calls = len(agent.observations), agent.costs['model_calls']

    result = agent._execute(chosen['action'], chosen['arguments'], lab)

    assert result['requested_setting_reached']
    assert result['condition_proposal_origin'] == 'current_research_advice'
    assert result['selected_number'] == 1 and result['selected_setting'] == point['setting']
    assert len(agent.observations) > paid and agent.costs['model_calls'] == calls
    outcome = next(row for row in result['observed_responses']
                   if row['crystal'] == 1 and row['frequency'] == point['setting'])
    assert outcome['activated'] is False
    contradicted = {identifier for identifier, prediction in predictions.items()
                   if prediction != outcome['activated']}
    assert contradicted
    conditions = result['condition_research']
    assert conditions['cached'] is False and conditions['compute_cost'] > 0
    rules = {row['id']: row for row in conditions['candidates']}
    assert all(rules[identifier]['status'] == 'refuted' for identifier in contradicted)
    assert all(rules[row['id']]['program'] == row['program'] for row in before['candidates'])
    assert all(row['status'] != 'supported' for row in conditions['candidates'])
    assert session['proposal']['condition_research'] == conditions
    assert session['events'][-1]['action'] == 'condition_return'
    assert session['events'][-1]['feedback']['physical_test']['observed_responses'] == result['observed_responses']
    advice = agent.research_advice
    assert advice['question_id'] == session['question_id']
    assert advice['diagnosis'] == conditions['status'] and advice['status'] == 'unconfirmed'
    assert advice.get('next_action') == conditions['experiment_action']
    assert json.loads((agent.workspace / 'method_decision.json').read_text())['research_advice'] == advice
    assert next(question for question in agent.questions if question['id'] == 'computed') == previous_question
