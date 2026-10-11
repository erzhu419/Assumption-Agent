"""Empty finite support resumes the executable source author with paid feedback.

The scheduling projection is controlled by extension_fixture. These tests prove
the connection and candidate retention, not discovery of a scientific law.
"""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_reference_response as response
from hegel_machine.cognition import world_response_conditions as conditions
from hegel_machine.cognition.world_llm_inquiry import refresh_condition_proposal
from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BINDING, BIND_CODE
from test_world_condition_extension import extension_fixture
from test_world_reference_hypothesis import PATH


def test_empty_support_reuses_source_paths_with_current_extension_assignment(
        tmp_path, monkeypatch):
    agent, lab, condition_path, _ = extension_fixture(tmp_path, monkeypatch)
    condition_author = agent.llm_inquiries['proposal:' + condition_path]
    source_author = agent.llm_inquiries['proposal:' + BINDING]
    old_question = deepcopy(next(question for question in agent.questions
                                if question['id'] == source_author['question_id']))
    old_revision = deepcopy(source_author['proposal']['comparison_result']['revision_action'])
    source_author['notes'] = 'Retain the earlier paid comparison while revising this draft.'
    response.return_response_to_author(agent, source_author['proposal']['comparison_result'])
    old_events = deepcopy(source_author['events'])
    old_invocations = source_author.get('invocations', 0)
    action = condition_author['condition_research']['research_action']
    params = action['arguments']
    qid = condition_author['question_id'] + '_target_extension'
    assert params['operator'] == 'propose_explanations'
    assert params['path'] == old_revision['arguments']['path'] == BINDING
    assert params['response_path'] == old_revision['arguments']['response_path'] == PATH
    assert params['condition_proposal'] == condition_path
    assert params['question']['id'] == qid != old_revision['arguments']['question']['id']
    assert params['question']['question'] != old_revision['arguments']['question']['question']
    provider = agent.provider = Provider([
        ('finish', {'lesson': 'The author returns the unresolved source revision.'}),
        ('finish', {'lesson': 'The parent retains the existing evidence.'}),
    ])

    agent.run(lab)

    first = provider.requests[0]
    assert first['llm_inquiry']['stage'] == 'proposal'
    assert first['llm_inquiry']['proposal_path'] == BINDING
    assert first['llm_inquiry']['response_path'] == PATH
    assert first['llm_inquiry']['assignment'] == params['question'] | {'status': 'open'}
    assert first['research_notes'] == source_author['notes']
    assert source_author['events'][:len(old_events)] == old_events
    assert source_author['invocations'] == old_invocations + 1
    assert agent.llm_inquiries['proposal:' + BINDING] is source_author
    assert source_author['question_id'] == qid
    assert qid not in agent.llm_inquiries
    assert next(question for question in agent.questions if question['id'] == old_question['id']) == old_question
    assert 'llm_inquiry' not in provider.requests[-1]
    assert len(provider.requests) == 2


def test_actual_binding_edit_returns_new_coordinates_and_preserves_old_conditions(
        tmp_path, monkeypatch):
    agent, lab, condition_path, _ = extension_fixture(tmp_path, monkeypatch)
    source_author = agent.llm_inquiries['proposal:' + BINDING]
    condition_author = agent.llm_inquiries['proposal:' + condition_path]
    before = deepcopy(condition_author['condition_research'])
    old_rules = {rule['id']: rule for rule in before['candidates']}
    paid = len(agent.observations)
    old_controller = (agent.workspace / PATH).read_text()
    parent_notes = 'Parent: inspect the computed alternative and retain earlier evidence.'
    (agent.workspace / 'research_notes.md').write_text(parent_notes)
    # Keep the same feature name deliberately: retained rules must continue to
    # use their original log coordinates after the executable binding changes.
    revised = BIND_CODE.replace('math.log(value)', 'math.sqrt(value)')
    provider = agent.provider = Provider([
        ('write', {'path': BINDING, 'code': revised}),
        ('finish', {'lesson': 'Return the actual computed comparison and candidate pool.'}),
        ('finish', {'lesson': 'The parent received the new representation.'}),
    ])

    result = agent.run(lab)

    assert result['status'] == 'agent_finished'
    assert len(provider.requests) == 3
    assert (agent.workspace / BINDING).read_text() == revised
    assert (agent.workspace / PATH).read_text() == old_controller
    assert len(agent.observations) == paid
    first, after_write, parent = provider.requests
    assert first['llm_inquiry']['stage'] == after_write['llm_inquiry']['stage'] == 'proposal'
    event = after_write['events'][-1]
    assert event['action'] == 'write' and 'error' not in event['feedback']
    compared = event['feedback']['response_comparison']
    features = {row['crystal']: row['features']['log_density'] for row in compared['bound_rows']}
    assert features[1] == pytest.approx(1.)
    assert features[3] == pytest.approx(2.718281828459045)
    returned = event['feedback']['condition_research']
    assert returned['regenerated'] and returned['generation_compute_cost'] > 0
    after = condition_author['condition_research']
    assert before['inputs']['bound_rows'] != after['inputs']['bound_rows']
    retained = {rule['id']: rule for rule in returned['candidates']}
    assert old_rules.keys() < retained.keys()
    for identifier, rule in old_rules.items():
        for field in ('program', 'feature_derivations', 'bound_rows', 'status'):
            assert retained[identifier][field] == rule[field]
    assert all(rule['status'] != 'supported' for rule in returned['candidates'])
    assert any(event['action'] == 'response_return' and
               event['feedback']['response_comparison']['bound_rows'] == compared['bound_rows']
               for event in source_author['events'])
    assert 'llm_inquiry' not in parent and parent['research_notes'] == parent_notes
    returned_event = next(event for event in reversed(parent['events'])
                          if event.get('trigger_origin') == 'host_condition_language_extension')
    assert returned_event['feedback']['operator'] == 'propose_explanations'
    assert returned_event['feedback']['proposal']['comparison_result']['bound_rows'] == compared['bound_rows']
    assert returned_event['feedback']['condition_research']['candidates'] == after['candidates']
    assert source_author['condition_extension_context'] == after['inputs']
    assert agent._inquiry_stack == []
    assert response._dispatch_condition_extension(agent, lab, 8) is False
    assert len(provider.requests) == 3


def test_entry_recomparison_reaches_condition_pool_before_first_author_decision(
        tmp_path, monkeypatch):
    agent, lab, condition_path, _ = extension_fixture(tmp_path, monkeypatch)
    source_author = agent.llm_inquiries['proposal:' + BINDING]
    stale = deepcopy(source_author['proposal']['comparison_result'])
    source = (agent.workspace / BINDING).read_text()
    # Historical saved runs contain newer physical evidence alongside an old
    # source comparison. Acquire the response normally, then restore only that
    # old saved comparison to exercise reopening such a persisted state.
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'condition_proposal': condition_path,
        'experiment': {'number': 1, 'setting': 103.}}, lab)
    assert trial['requested_setting_reached']
    assert any(row['crystal'] == 1 and row['frequency'] == 103. and not row['activated']
               for row in trial['observed_responses'])
    source_author['proposal']['comparison_result'] = deepcopy(stale)
    refresh_condition_proposal(agent, condition_path)
    old_inputs = deepcopy(agent.llm_inquiries['proposal:' + condition_path]['condition_research']['inputs'])
    assert len(old_inputs['responses']) > stale['response_count']
    paid = len(agent.observations)

    class InspectProvider(Provider):
        def next_action(self, request):
            self.packet_at_decision = response.response_premise_proposal_inputs(agent, condition_path)
            self.conditions_at_decision = deepcopy(
                agent.llm_inquiries['proposal:' + condition_path]['condition_research'])
            return super().next_action(request)

    provider = agent.provider = InspectProvider([
        ('finish', {'lesson': 'Return the newly recomputed source comparison without editing.'}),
    ])

    assert response.dispatch_response_premise(agent, lab) is True

    request, = provider.requests
    compared = next(event['feedback']['response_comparison'] for event in reversed(request['events'])
                    if event['action'] == 'response_return')
    returned = next(event['feedback']['condition_research'] for event in reversed(request['events'])
                   if event['action'] == 'condition_return')
    new_evidence = set(trial['new_observation_indices'])
    assert compared['response_count'] > stale['response_count']
    assert new_evidence.intersection(compared['evidence'])
    assert new_evidence.intersection(provider.packet_at_decision['comparison_evidence'])
    assert provider.packet_at_decision['paid_family_checks'] == compared['paid_family_checks']
    assert provider.conditions_at_decision['inputs'] == conditions._inputs(provider.packet_at_decision)
    assert returned['candidates'] == provider.conditions_at_decision['candidates']
    assert old_inputs['residual_sources'] != provider.conditions_at_decision['inputs']['residual_sources']
    assert len(agent.observations) == paid
    assert (agent.workspace / BINDING).read_text() == source
    assert not any(event['action'] == 'write' for event in source_author['events'])


def test_author_physical_trial_refreshes_source_and_pool_before_next_decision(
        tmp_path, monkeypatch):
    agent, lab, condition_path, _ = extension_fixture(tmp_path, monkeypatch)
    source_author = agent.llm_inquiries['proposal:' + BINDING]
    initial = deepcopy(source_author['proposal']['comparison_result'])
    source = (agent.workspace / BINDING).read_text()
    paid = len(agent.observations)

    class InspectProvider(Provider):
        def next_action(self, request):
            if len(self.requests) == 1:
                self.packet_after_trial = response.response_premise_proposal_inputs(agent, condition_path)
                self.conditions_after_trial = deepcopy(
                    agent.llm_inquiries['proposal:' + condition_path]['condition_research'])
                self.source_after_trial = deepcopy(source_author['proposal']['comparison_result'])
            return super().next_action(request)

    provider = agent.provider = InspectProvider([
        ('run', {'operator': 'reference_response', 'path': PATH,
                 'experiment': {'number': 1, 'setting': 103.}}),
        ('finish', {'lesson': 'Return the recomputed comparison after the measured negative.'}),
        ('finish', {'lesson': 'The parent has the trial and revised source comparison.'}),
    ])

    result = agent.run(lab)

    assert result['status'] == 'agent_finished' and len(provider.requests) == 3
    after_trial = provider.requests[1]
    assert after_trial['llm_inquiry']['stage'] == 'proposal'
    event = after_trial['events'][-1]
    assert event['action'] == 'run'
    feedback = event['feedback']
    assert feedback['requested_setting_reached']
    assert feedback['condition_proposal_origin'] == 'condition_research_inquiry'
    assert any(row['crystal'] == 1 and row['frequency'] == 103. and not row['activated']
               for row in feedback['observed_responses'])
    new_evidence = set(feedback['new_observation_indices'])
    assert new_evidence and min(new_evidence) >= paid
    compared = feedback['response_comparison']
    assert compared['response_count'] > initial['response_count']
    assert new_evidence.intersection(compared['evidence'])
    assert provider.source_after_trial == compared
    assert provider.packet_after_trial['paid_family_checks'] == compared['paid_family_checks']
    assert new_evidence.intersection(provider.packet_after_trial['comparison_evidence'])
    assert provider.conditions_after_trial['inputs'] == conditions._inputs(provider.packet_after_trial)
    assert feedback['condition_research']['candidates'] == provider.conditions_after_trial['candidates']
    assert source_author['proposal']['comparison_result'] == compared
    returned = next(event for event in reversed(provider.requests[-1]['events'])
                    if event.get('trigger_origin') == 'host_condition_language_extension')
    assert returned['feedback']['proposal']['comparison_result'] == compared
    assert any(event['action'] == 'response_return' and
               event['feedback']['response_comparison'] == compared
               for event in source_author['events'])
    assert (agent.workspace / BINDING).read_text() == source


@pytest.mark.parametrize('explicit_origin', [False, True], ids=['condition-only', 'same-source-origin'])
def test_parent_condition_trial_refreshes_source_once_and_returns_current_pool(
        tmp_path, monkeypatch, explicit_origin):
    agent, lab, condition_path, _ = extension_fixture(tmp_path, monkeypatch)
    source_author = agent.llm_inquiries['proposal:' + BINDING]
    initial = deepcopy(source_author['proposal']['comparison_result'])
    arguments = {'operator': 'reference_response', 'path': PATH,
        'condition_proposal': condition_path, 'experiment': {'number': 1, 'setting': 103.}}
    if explicit_origin:
        arguments['response_comparison'] = {
            'binding_path': BINDING,
            'question': deepcopy(initial['revision_action']['arguments']['question'])}
    compare = response._compare_bound_response
    compared_bindings = []

    def record_comparison(agent, arguments, state, previous):
        compared_bindings.append(arguments['binding_path'])
        return compare(agent, arguments, state, previous)

    monkeypatch.setattr(response, '_compare_bound_response', record_comparison)
    paid, calls = len(agent.observations), agent.costs['model_calls']

    trial = agent._execute('run', arguments, lab)

    assert trial['requested_setting_reached']
    assert trial['condition_proposal_origin'] == 'explicit'
    assert compared_bindings == [BINDING]
    compared = trial['response_comparison']
    new_evidence = set(trial['new_observation_indices'])
    assert new_evidence and min(new_evidence) >= paid
    assert compared['response_count'] > initial['response_count']
    assert new_evidence.intersection(compared['evidence'])
    assert source_author['proposal']['comparison_result'] == compared
    packet = response.response_premise_proposal_inputs(agent, condition_path)
    current = agent.llm_inquiries['proposal:' + condition_path]['condition_research']
    assert new_evidence.intersection(packet['comparison_evidence'])
    assert packet['paid_family_checks'] == compared['paid_family_checks']
    assert current['inputs'] == conditions._inputs(packet)
    assert trial['condition_research']['candidates'] == current['candidates']
    returned = next(event for event in reversed(source_author['events'])
                    if event['action'] == 'response_return')
    assert returned['feedback']['response_comparison'] == compared
    assert returned['feedback']['physical_test']['new_observation_indices'] == trial['new_observation_indices']
    assert agent.costs['model_calls'] == calls
