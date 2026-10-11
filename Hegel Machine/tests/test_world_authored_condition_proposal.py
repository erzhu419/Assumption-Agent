"""Authored Boolean additions traverse the retained response-author path.

Scripted authors and the existing controller laboratory test this interface;
the supplied conditions are fixture inputs, not evidence of model discovery.
"""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_reference_response as response
from hegel_machine.cognition.language import Program
from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BINDING, BIND_CODE, _ready
from test_world_condition_proposal import prepare_proposal, proposal_view
from test_world_reference_hypothesis import PATH


def disconnected_rule(low_feature=.5, high_feature=1.9):
    frequency = Program('feature', name='frequency')
    bound = Program('feature', name='bound_log_density')

    def interval(lower, upper):
        return Program('and', children=(
            Program('ge', children=(frequency, Program('const', value=lower))),
            Program('ge', children=(Program('const', value=upper), frequency))))

    return Program('or', children=(
        Program('and', children=(
            Program('lt', children=(bound, Program('const', value=low_feature))),
            interval(100., 104.))),
        Program('and', children=(
            Program('ge', children=(bound, Program('const', value=high_feature))),
            interval(108., 112.))))).to_dict()


def authored_source(program, *, raw=False, root_coordinates=False):
    addition = program if raw else {
        'id': 'disconnected_response',
        'claim': 'Different measured regimes have disjoint response bands.',
        'program': program}
    source = BIND_CODE.replace("return {'rows':", "return {'candidate_additions': "
                               + repr([addition]) + ", 'rows':")
    if root_coordinates:
        source = source.replace('math.log(value)', 'math.sqrt(value)')
    return source, addition


def prepared_author(tmp_path):
    agent, lab, _, action, condition_path = prepare_proposal(tmp_path)
    agent._execute(action['action'], action['arguments'], lab)
    author = agent.llm_inquiries['proposal:' + BINDING]
    # stalled() supplies a saved comparison; initialize its normal loop counters.
    author.update(invocations=0, model_calls=0, status='open')
    revision = deepcopy(author['proposal']['comparison_result']['revision_action'])
    revision['arguments']['condition_proposal'] = condition_path
    return agent, lab, author, condition_path, revision


def resume_author(agent, lab, revision, steps):
    provider = agent.provider = Provider(steps)
    revision = deepcopy(revision)
    revision['arguments']['max_model_calls'] = len(steps)
    result = agent._execute(revision['action'], revision['arguments'], lab)
    return provider, result


def authored_candidate(conditions):
    return next(row for row in conditions['candidates'] if row.get('origin') == 'authored')


def test_fresh_source_author_owns_and_resumes_conditions_without_a_premise_session(tmp_path):
    agent, lab = _ready(tmp_path)
    assert not agent.llm_inquiries
    source, addition = authored_source(disconnected_rule())
    revision = {'action': 'run', 'arguments': {
        'operator': 'propose_explanations', 'path': BINDING, 'response_path': PATH,
        'question': {'id': 'authored_response', 'parent_id': 'task', 'depends_on': [],
                     'question': 'Compute conditional alternatives on the paid responses.'}}}
    physical_before = deepcopy((agent.observations, lab.actions))

    provider, first = resume_author(agent, lab, revision, [
        ('write', {'path': BINDING, 'code': source}),
        ('finish', {'lesson': 'Return this source and its computed condition bank.'}),
    ])

    assert set(agent.llm_inquiries) == {'proposal:' + BINDING}
    assert provider.requests[1]['llm_inquiry']['condition_proposal'] == BINDING
    author = agent.llm_inquiries['proposal:' + BINDING]
    before = deepcopy(author['condition_research']['candidates'])
    assert authored_candidate(first['proposal']['condition_research'])['status'] == 'compatible'
    assert first['proposal']['condition_research']['candidates'] == before
    assert response.response_premise_proposal_inputs(agent, BINDING)['candidate_additions'] == [addition]

    resumed_provider, resumed = resume_author(agent, lab, revision, [
        ('finish', {'lesson': 'The retained candidates need a selected physical experiment.'}),
    ])

    request, = resumed_provider.requests
    assert request['llm_inquiry']['condition_proposal'] == BINDING
    assert request['llm_inquiry']['invocation'] == 2
    returned = next(event['feedback']['condition_research'] for event in reversed(request['events'])
                    if event['action'] == 'condition_return')
    assert returned['cached'] and returned['compute_cost'] == 0
    assert returned['candidates'] == before
    assert resumed['proposal']['condition_research']['candidates'] == before
    assert author['condition_research']['candidates'] == before
    assert set(agent.llm_inquiries) == {'proposal:' + BINDING}
    assert (agent.observations, lab.actions) == physical_before


@pytest.mark.parametrize('raw', [False, True], ids=['named-addition', 'raw-ast'])
def test_author_binding_addition_reaches_comparison_and_condition_feedback_without_observations(
        tmp_path, raw):
    agent, lab, author, condition_path, revision = prepared_author(tmp_path)
    program = disconnected_rule()
    source, addition = authored_source(program, raw=raw)
    physical_before = deepcopy((agent.observations, lab.actions))
    controller = (agent.workspace / PATH).read_text()

    provider, result = resume_author(agent, lab, revision, [
        ('write', {'path': BINDING, 'code': source}),
        ('finish', {'lesson': 'Return the computed conditional alternative.'}),
    ])

    after_write = provider.requests[1]
    assert after_write['llm_inquiry']['proposal_path'] == BINDING
    event = after_write['events'][-1]
    assert event['action'] == 'write' and 'error' not in event['feedback']
    compared = event['feedback']['response_comparison']
    assert compared['candidate_additions'] == [addition]
    packet = response.response_premise_proposal_inputs(agent, condition_path)
    assert packet['candidate_additions'] == [addition]
    returned = event['feedback']['condition_research']
    candidate = authored_candidate(returned)
    assert Program.from_dict(candidate['program']) == Program.from_dict(program).simplified()
    assert candidate['status'] == 'compatible' and candidate['conflict_count'] == 0
    assert candidate['checked_response_count'] == len(packet['responses'])
    assert candidate['evidence'] and candidate['proposed_evidence']
    assert returned['confirmation_status'] == 'unconfirmed'
    if not raw:
        assert candidate['authored_id'] == addition['id']
        assert candidate['claim'] == addition['claim']
    assert author['proposal']['comparison_result']['candidate_additions'] == [addition]
    assert result['proposal']['comparison_result']['candidate_additions'] == [addition]
    assert agent.llm_inquiries['proposal:' + condition_path]['condition_research']['candidates'] == returned['candidates']
    assert (agent.observations, lab.actions) == physical_before
    assert (agent.workspace / PATH).read_text() == controller


def test_changed_addition_and_binding_keep_previous_rule_evidence_and_original_coordinates(tmp_path):
    agent, lab, _, condition_path, revision = prepared_author(tmp_path)
    original_source, original = authored_source(disconnected_rule())
    revised_source, revised = authored_source(disconnected_rule(1.5, 2.5), root_coordinates=True)
    physical_before = deepcopy((agent.observations, lab.actions))

    provider, _ = resume_author(agent, lab, revision, [
        ('write', {'path': BINDING, 'code': original_source}),
        ('write', {'path': BINDING, 'code': revised_source}),
        ('finish', {'lesson': 'Retain both alternatives under their own coordinates.'}),
    ])

    first = provider.requests[1]['events'][-1]['feedback']['condition_research']
    second_feedback = provider.requests[2]['events'][-1]['feedback']
    assert 'error' not in second_feedback
    second = second_feedback['condition_research']
    old = authored_candidate(first)
    retained = next(row for row in second['candidates'] if row['id'] == old['id'])
    for key in ('program', 'feature_derivations', 'bound_rows', 'status', 'evidence', 'proposed_evidence'):
        assert retained[key] == old[key]
    new = next(row for row in second['candidates']
               if row.get('origin') == 'authored' and row['id'] != old['id'])
    assert new['authored_id'] == old['authored_id'] == original['id'] == revised['id']
    assert new['program'] != old['program']
    assert new['bound_rows'] != old['bound_rows']
    assert new['bound_rows']['1']['log_density'] == pytest.approx(1.)
    assert old['bound_rows']['1']['log_density'] == pytest.approx(0.)
    assert new['status'] == 'compatible'
    assert second_feedback['response_comparison']['candidate_additions'] == [revised]
    assert response.response_premise_proposal_inputs(agent, condition_path)['candidate_additions'] == [revised]
    assert (agent.observations, lab.actions) == physical_before


def test_author_selected_controller_trial_refutes_frozen_addition_and_returns_its_counterexample(tmp_path):
    agent, lab, author, condition_path, revision = prepared_author(tmp_path)
    source, addition = authored_source(disconnected_rule())
    paid = len(agent.observations)

    provider, result = resume_author(agent, lab, revision, [
        ('write', {'path': BINDING, 'code': source}),
        ('run', {'operator': 'reference_response', 'path': PATH,
                 'experiment': {'number': 1, 'setting': 103.}}),
        ('finish', {'lesson': 'The measured negative refutes the proposed first band.'}),
    ])

    before = authored_candidate(provider.requests[1]['events'][-1]['feedback']['condition_research'])
    assert before['status'] == 'compatible'
    after_trial = provider.requests[2]
    assert after_trial['llm_inquiry']['proposal_path'] == BINDING
    feedback = after_trial['events'][-1]['feedback']
    assert 'error' not in feedback and feedback['requested_setting_reached']
    assert feedback['condition_proposal_origin'] == 'condition_research_inquiry'
    actual = next(row for row in feedback['observed_responses']
                  if row['crystal'] == 1 and row['frequency'] == 103.)
    assert actual['activated'] is False
    new_evidence = set(feedback['new_observation_indices'])
    assert new_evidence and min(new_evidence) >= paid
    after = next(row for row in feedback['condition_research']['candidates']
                 if row['id'] == before['id'])
    assert after['status'] == 'refuted' and after['conflict_count'] > 0
    for key in ('program', 'feature_derivations', 'bound_rows', 'proposed_evidence'):
        assert after[key] == before[key]
    conflict = next(row for row in after['conflicting_responses']
                    if row['crystal'] == 1 and row['frequency'] == 103.)
    assert conflict['predicted_activation'] is True and conflict['activated'] is False
    assert new_evidence.intersection(conflict['evidence'])
    assert feedback['response_comparison']['candidate_additions'] == [addition]
    assert result['proposal']['comparison_result']['candidate_additions'] == [addition]
    assert agent.llm_inquiries['proposal:' + condition_path]['condition_research']['candidates'] == feedback['condition_research']['candidates']
    returned = next(event for event in reversed(author['events'])
                    if event['action'] == 'response_return' and event['feedback'].get('physical_test'))
    assert returned['feedback']['physical_test']['new_observation_indices'] == feedback['new_observation_indices']
    assert (agent.workspace / BINDING).read_text() == source
    assert lab.active(1)  # Existing controller restores the already supported setting.


@pytest.mark.parametrize('association', ['condition-bank', 'response-binding'])
def test_parent_trial_after_author_return_refreshes_both_bank_and_source_proposal(
        tmp_path, association):
    agent, lab, author, condition_path, revision = prepared_author(tmp_path)
    source, _ = authored_source(disconnected_rule())
    _, returned = resume_author(agent, lab, revision, [
        ('write', {'path': BINDING, 'code': source}),
        ('finish', {'lesson': 'Return the conditional alternative for a parent-selected trial.'}),
    ])
    before = authored_candidate(returned['proposal']['condition_research'])
    assert before['status'] == 'compatible'
    assert author['proposal']['condition_proposal'] == condition_path
    assert not agent._inquiry_stack
    arguments = {'operator': 'reference_response', 'path': PATH,
                 'experiment': {'number': 1, 'setting': 103.}}
    if association == 'condition-bank':
        arguments['condition_proposal'] = condition_path
    else:
        # The source's recorded bank must survive after current advice changes;
        # this ordinary computed-source experiment has no bank/frame argument.
        agent.research_advice = None
        arguments['response_comparison'] = {
            'binding_path': BINDING,
            'question': deepcopy(author['proposal']['comparison_result']['revision_action']['arguments']['question'])}
    paid = len(agent.observations)

    trial = agent._execute('run', arguments, lab)

    assert trial['requested_setting_reached']
    assert trial['condition_proposal_origin'] == (
        'explicit' if association == 'condition-bank' else 'response_binding')
    assert any(row['crystal'] == 1 and row['frequency'] == 103. and not row['activated']
               for row in trial['observed_responses'])
    new_evidence = set(trial['new_observation_indices'])
    assert new_evidence and min(new_evidence) >= paid
    current = trial['condition_research']
    refuted = next(row for row in current['candidates'] if row['id'] == before['id'])
    assert refuted['status'] == 'refuted'
    assert refuted['program'] == before['program']
    assert refuted['bound_rows'] == before['bound_rows']
    assert any(new_evidence.intersection(row['evidence'])
               for row in refuted['conflicting_responses'])
    assert author['proposal']['condition_research'] == current
    assert agent.llm_inquiries['proposal:' + condition_path]['proposal']['condition_research'] == current
    exposed = proposal_view(agent._request(), BINDING)
    assert exposed['condition_research'] == current
    if current['experiment_action']:
        assert author['proposal']['experiment_action'] == current['experiment_action']
        assert exposed['experiment_action'] == current['experiment_action']
    else:
        assert not author['proposal'].get('experiment_action')
        assert not exposed.get('experiment_action')
    # The earlier returned value is historical; only the active source view is refreshed.
    assert authored_candidate(returned['proposal']['condition_research'])['status'] == 'compatible'

    resume = deepcopy(author['proposal']['comparison_result']['revision_action'])
    assert 'condition_proposal' not in resume['arguments']
    provider, reopened = resume_author(agent, lab, resume, [
        ('finish', {'lesson': 'Keep the retained premise bank and its physical refutation.'}),
    ])
    assert provider.requests[0]['llm_inquiry']['condition_proposal'] == condition_path
    assert 'condition_research' not in author  # No second source-local bank is created.
    assert reopened['proposal']['condition_proposal'] == condition_path
    assert next(row for row in reopened['proposal']['condition_research']['candidates']
                if row['id'] == before['id'])['status'] == 'refuted'


@pytest.mark.parametrize('edit_action', ['write', 'run'], ids=['child-write', 'child-python-edit'])
def test_response_author_receives_child_draft_revision_before_its_next_decision(tmp_path, edit_action):
    agent, lab, author, condition_path, revision = prepared_author(tmp_path)
    previous = deepcopy(author['proposal']['comparison_result'])
    old_rules = deepcopy(agent.llm_inquiries['proposal:' + condition_path]['condition_research']['candidates'])
    physical_before = deepcopy((agent.observations, lab.actions))
    source, addition = authored_source(disconnected_rule(1.5, 2.5), root_coordinates=True)
    child_id = 'computed_condition_prerequisite'
    edit = ('write', {'path': BINDING, 'code': source}) if edit_action == 'write' else (
        'run', {'code': 'from pathlib import Path\nPath(' + repr(BINDING)
                + ').write_text(' + repr(source) + ')\n'})

    provider, result = resume_author(agent, lab, revision, [
        ('run', {'operator': 'inquire', 'max_model_calls': 2,
                 'question': {'id': child_id, 'depends_on': [],
                              'question': 'Compute the candidate response condition in the retained draft.'}}),
        edit,
        ('finish', {'lesson': 'The revised executable draft is ready for the author to compare.'}),
        ('finish', {'lesson': 'Return the child revision and its actual paid comparison.'}),
    ])

    assert len(provider.requests) == result['model_calls'] == 4
    child = provider.requests[1]['llm_inquiry']
    assert child['question_id'] == child_id and child['depth'] == 2
    assert child['parent_id'] == 'proposal:' + BINDING
    assert child.get('stage') != 'proposal'
    child_execution = provider.requests[2]['events'][-1]['feedback']
    assert 'error' not in child_execution
    if edit_action == 'run':
        assert child_execution['returncode'] == 0
    returned = provider.requests[3]
    assert returned['llm_inquiry']['stage'] == 'proposal'
    assert returned['llm_inquiry']['proposal_path'] == BINDING
    assert returned['llm_inquiry']['remaining_model_calls'] == 1
    current = proposal_view(returned, BINDING)
    comparison = current['comparison_result']
    assert comparison['candidate_additions'] == [addition]
    assert comparison['bound_rows'] != previous['bound_rows']
    assert current['condition_proposal'] == condition_path
    conditions = current['condition_research']
    candidate = authored_candidate(conditions)
    assert candidate['authored_id'] == addition['id']
    assert candidate['status'] == 'compatible'
    assert candidate['bound_rows']['1']['log_density'] == pytest.approx(1.)
    retained = {row['id']: row for row in conditions['candidates']}
    for old in old_rules:
        for key in ('program', 'feature_derivations', 'bound_rows', 'proposed_evidence'):
            assert retained[old['id']][key] == old[key]
    assert result['proposal']['comparison_result'] == comparison
    assert result['proposal']['condition_research']['candidates'] == conditions['candidates']
    assert result['proposal']['draft_changed'] is True
    assert result['proposal']['author_work_attempted'] is True
    assert result['proposal']['deferred_actions'] == []
    assert (agent.workspace / BINDING).read_text() == source
    assert (agent.observations, lab.actions) == physical_before
    assert agent.llm_inquiries[child_id]['status'] == 'inquiry_finished'
    assert agent._inquiry_stack == [] and agent.status == 'running'
