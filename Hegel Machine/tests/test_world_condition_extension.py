"""Unusable finite target support reopens the existing researcher, not old laws.

Only the finite projection is controlled below. Paid observations, synthesized
rules, question handling, child execution and controller state remain real.
"""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_reference_response as response
from hegel_machine.cognition import world_response_conditions as conditions
from hegel_machine.cognition.world_llm_inquiry import refresh_condition_proposal
from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BINDING, BIND_CODE
from test_world_condition_proposal import prepare_proposal
from test_world_premise_dispatch import DESIGN, DESIGN_PATH, stalled
from test_world_reference_hypothesis import PATH


def extension_fixture(tmp_path, monkeypatch, *, projection='empty'):
    original = conditions._target_consequences

    def project(result, inputs, budget, previous=None):
        # Exercise scheduling after the finite rules agree on tested settings.
        # This synthetic projection is not a scientific result of the fixture.
        if not any(rule['status'] == 'compatible' for rule in result['candidates']):
            return original(result, inputs, budget, previous=previous)
        result['selected_experiment'] = result['experiment_action'] = None
        result['status'] = 'compatible_unconfirmed_conditions'
        original(result, inputs, budget, previous=previous)
        target = next(item for item in result['target_consequences']['targets']
                      if item['number'] == 5)
        target.clear()
        target.update(number=5, status='complete',
                      support={'intervals': [], 'grid_count': 0})
        result['target_consequences']['status'] = 'complete'
        if projection == 'unresolved':
            target.update(status='unresolved', reason='projection_compute_limit')
            target.pop('support')
            result['target_consequences']['status'] = 'unresolved'
        elif projection == 'usable':
            target['support'] = {'intervals': [[180, 180]], 'grid_count': 1}

    monkeypatch.setattr(conditions, '_target_consequences', project)
    agent, lab, _, action, path = prepare_proposal(tmp_path)
    result = agent._execute(action['action'], action['arguments'], lab)
    # stalled() installs a comparison-only author; this suite now resumes it
    # through the normal proposal loop, with its usual invocation accounting.
    author = agent.llm_inquiries['proposal:' + BINDING]
    author.update(invocations=0, model_calls=0, status='open')
    agent.provider = Provider([])
    return agent, lab, path, result


def test_empty_complete_target_support_requests_extension_without_refuting_compatible_rules(
        tmp_path, monkeypatch):
    agent, lab, path, result = extension_fixture(tmp_path, monkeypatch)
    session = agent.llm_inquiries['proposal:' + path]
    packet = response.response_premise_proposal_inputs(agent, path)
    research = result['proposal']['condition_research']

    assert packet['unfinished_targets'] == [4, 5]
    comparison = agent.llm_inquiries['proposal:' + BINDING]['proposal']['comparison_result']
    assert packet['revision_action'] == comparison['revision_action']
    obligation = research['research_obligation']
    assert obligation['status'] == 'condition_language_extension_needed'
    assert 5 in obligation['targets']
    assert obligation['revision_action'] == comparison['revision_action']
    assert research['status'] == 'compatible_unconfirmed_conditions'
    assert research['confirmation_status'] == 'unconfirmed'
    assert any(rule['status'] == 'compatible' for rule in research['candidates'])
    assert not any(rule['status'] == 'supported' for rule in research['candidates'])
    advice = agent.research_advice
    assert advice['diagnosis'] == obligation['status']
    assert advice['next_action'] == research['research_action']
    assert advice['next_action']['action'] == 'run'
    assert advice['next_action']['arguments']['operator'] == 'propose_explanations'
    assert advice['next_action']['arguments']['path'] == BINDING
    assert advice['next_action']['arguments']['response_path'] == PATH
    assert advice['next_action']['arguments']['question']['id'] == session['question_id'] + '_target_extension'
    assert session['proposal']['condition_research'] == research
    assert not any(question['id'].endswith('_target_extension') for question in agent.questions)


@pytest.mark.parametrize('projection', ['unresolved', 'usable'])
def test_missing_or_nonempty_projection_does_not_request_target_extension(
        tmp_path, monkeypatch, projection):
    agent, _, path, result = extension_fixture(tmp_path, monkeypatch, projection=projection)

    assert not result['proposal']['condition_research'].get('research_obligation')
    assert not agent.research_advice.get('research_obligation')
    assert not agent.research_advice.get('next_action')
    assert agent.llm_inquiries['proposal:' + path]['condition_research']['confirmation_status'] == 'unconfirmed'


def test_completed_target_is_not_reopened_by_its_old_empty_projection(tmp_path, monkeypatch):
    agent, lab, path, _ = extension_fixture(tmp_path, monkeypatch)
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 5, 'setting': lab.required[5]}}, lab)
    assert trial['requested_setting_reached'] and lab.active(5)
    packet = response.response_premise_proposal_inputs(agent, path)

    updated = refresh_condition_proposal(agent, path, packet)

    assert 5 not in packet['unfinished_targets']
    assert 5 not in (updated.get('research_obligation') or {}).get('targets', [])
    assert not agent.research_advice.get('next_action')


def test_automatic_extension_runs_real_child_experiment_and_returns_to_parent(
        tmp_path, monkeypatch):
    agent, lab, path, _ = extension_fixture(tmp_path, monkeypatch)
    session = agent.llm_inquiries['proposal:' + path]
    qid = session['question_id'] + '_target_extension'
    retained = agent._question({'id': 'paid_reference_success', 'parent_id': 'task',
        'question': 'Was reference 1 active in the initial paid observation?'})
    agent._record_answer(retained, {'status': 'supported',
        'answer': {'initial_reference_active': True}, 'evidence': [0]})
    retained_before = deepcopy(retained)
    candidates = {rule['id']: deepcopy(rule['program'])
                  for rule in session['condition_research']['candidates']}
    responses = len(session['condition_research']['inputs']['responses'])
    sources = {name: (agent.workspace / name).read_text() for name in (PATH, BINDING)}
    notes = 'Parent: keep the old results and inspect the child outcome.'
    (agent.workspace / 'research_notes.md').write_text(notes)
    paid, calls = len(agent.observations), agent.costs['model_calls']
    provider = agent.provider = Provider([
        ('write', {'path': DESIGN_PATH, 'code': DESIGN}),
        ('run', {'operator': 'reference_response', 'path': PATH,
                 'experiment': {'number': 1, 'setting': 101.}}),
        ('finish', {'lesson': 'Return the actual response without claiming the full law.'}),
        ('finish', {'lesson': 'The parent has received the child outcome.'}),
    ])

    returned = agent.run(lab)

    assert returned['status'] == 'agent_finished'
    assert agent.costs['model_calls'] - calls == len(provider.requests) == 4
    assert all(request['llm_inquiry']['question_id'] == qid for request in provider.requests[:3])
    assert all(request['llm_inquiry']['stage'] == 'proposal' for request in provider.requests[:3])
    assert all(request['llm_inquiry']['proposal_path'] == BINDING for request in provider.requests[:3])
    assert 'llm_inquiry' not in provider.requests[-1]
    assert provider.requests[-1]['research_notes'] == notes
    assert provider.requests[2]['events'][-1]['feedback'].get('status') != 'proposal_action_deferred'
    assert (agent.workspace / DESIGN_PATH).read_text() == DESIGN
    # The retained controller measures the failed point, then restores the
    # already-supported reference setting before returning to the author.
    assert len(agent.observations) == paid + 4 and lab.active(1)
    assert next(question for question in agent.questions if question['id'] == retained['id']) == retained_before
    assert {name: (agent.workspace / name).read_text() for name in sources} == sources
    after = agent.llm_inquiries['proposal:' + path]['condition_research']
    current = {rule['id']: rule['program'] for rule in after['candidates']}
    assert all(current[identifier] == program for identifier, program in candidates.items())
    assert len(after['inputs']['responses']) > responses
    assert any(row['crystal'] == 1 and row['frequency'] == 101. and not row['activated']
               for row in after['inputs']['responses'])
    assert after['confirmation_status'] == 'unconfirmed'
    assert agent._inquiry_stack == []
    agent.status = 'running'
    assert response.dispatch_response_premise(agent, lab) is False
    assert len(provider.requests) == 4


def test_extension_deduplicates_unchanged_inputs_but_accepts_revised_computed_representation(
        tmp_path, monkeypatch):
    agent, lab, path, _ = extension_fixture(tmp_path, monkeypatch)
    qid = agent.llm_inquiries['proposal:' + path]['question_id'] + '_target_extension'
    provider = agent.provider = Provider([
        ('finish', {'lesson': 'The finite model class still has no target solution.'}),
        ('finish', {'lesson': 'Inspect the changed measured representation.'}),
    ])
    assert response.dispatch_response_premise(agent, lab) is True
    assert response.dispatch_response_premise(agent, lab) is False
    assert len(provider.requests) == 1
    assert provider.requests[0]['llm_inquiry']['question_id'] == qid
    before = deepcopy(agent.llm_inquiries['proposal:' + path]['condition_research']['inputs'])
    paid = len(agent.observations)
    revised = BIND_CODE.replace('math.log(value)', 'math.sqrt(value)')
    agent._execute('write', {'path': BINDING, 'code': revised}, lab)
    author, _ = stalled(agent, lab)
    # A revised binding need not retain the old shared-premise diagnosis for
    # its existing condition author to consume the new computed coordinates.
    author['proposal']['comparison_result']['family_research']['selected_role'] = 'family_discrimination'
    refresh_condition_proposal(agent, path)

    assert response.dispatch_response_premise(agent, lab) is True

    assert len(provider.requests) == 2 and len(agent.observations) == paid
    assert provider.requests[-1]['llm_inquiry']['question_id'] == qid
    assert provider.requests[-1]['llm_inquiry']['invocation'] == 2
    after = agent.llm_inquiries['proposal:' + path]['condition_research']['inputs']
    assert before['bound_rows'] != after['bound_rows']
    assert response.dispatch_response_premise(agent, lab) is False


def test_extension_child_controller_trial_returns_conditions_before_its_next_decision(
        tmp_path, monkeypatch):
    agent, lab, path, _ = extension_fixture(tmp_path, monkeypatch)
    before = len(agent.observations)
    provider = agent.provider = Provider([
        ('run', {'operator': 'reference_response', 'path': PATH,
                 'experiment': {'number': 1, 'setting': 101.}}),
        ('finish', {'lesson': 'The returned condition rules include this physical negative.'}),
        ('finish', {'lesson': 'The parent will inspect the updated pool.'}),
    ])

    result = agent.run(lab)

    assert result['status'] == 'agent_finished'
    assert len(provider.requests) == 3
    assert len(agent.observations) > before
    response_feedback = provider.requests[1]['events'][-1]['feedback']
    assert response_feedback['requested_setting_reached']
    assert response_feedback['condition_proposal_origin'] == 'condition_research_inquiry'
    assert response_feedback['condition_research']['confirmation_status'] == 'unconfirmed'
    returned_event = next(event for event in reversed(
        agent.llm_inquiries['proposal:' + path]['events']) if event['action'] == 'condition_return')
    assert returned_event['feedback']['condition_research'] == response_feedback['condition_research']
    returned = agent.llm_inquiries['proposal:' + path]['condition_research']
    assert any(row['crystal'] == 1 and row['frequency'] == 101. and not row['activated']
               for row in returned['inputs']['responses'])
    assert 'llm_inquiry' not in provider.requests[-1]


def test_condition_extension_reserves_parent_call_and_yields_to_usable_alternative(
        tmp_path, monkeypatch):
    agent, lab, path, _ = extension_fixture(tmp_path, monkeypatch)
    calls = agent.costs['model_calls']
    agent.limits['max_model_calls'] = calls + 1
    assert response.dispatch_response_premise(agent, lab) is False
    agent.limits['max_model_calls'] = calls + 2
    comparison = deepcopy(agent.llm_inquiries['proposal:' + BINDING]['proposal']['comparison_result'])
    comparison.update(binding_path='shared/alternative.py', question_id='alternative',
        experiment_action={'action': 'run', 'arguments': {'operator': 'reference_response',
            'path': PATH, 'experiment': {'number': 1, 'setting': 101.}}})
    agent.llm_inquiries['proposal:shared/alternative.py'] = {
        'question_id': 'alternative', 'proposal': {'comparison_result': comparison}}
    assert response.dispatch_response_premise(agent, lab) is False
    assert not agent.provider.requests
    agent.llm_inquiries.pop('proposal:shared/alternative.py')
    provider = agent.provider = Provider([
        ('finish', {'lesson': 'The child has one call; this remains unresolved.'}),
        ('finish', {'lesson': 'Parent retains its own last decision.'}),
    ])

    returned = agent.run(lab)

    assert returned['status'] == 'agent_finished'
    assert len(provider.requests) == 2
    assert provider.requests[0]['llm_inquiry']['remaining_model_calls'] == 1
    assert 'llm_inquiry' not in provider.requests[1]
