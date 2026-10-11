"""Stalled response research allocates an existing, accountable child work loop."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_reference_response as response
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BINDING, BIND_CODE, _compare, _ready
from test_world_reference_hypothesis import PATH


AUTHOR = "proposal:" + BINDING
PREMISE = "computed_premise"
DESIGN_PATH = "shared/authored_premise_experiment.py"
DESIGN = """def design(spec, observations, memory):
    return {'hypotheses': [{'id': 'same', 'claim': 'A unit change retains activation'},
                          {'id': 'different', 'claim': 'A unit change loses activation'}],
            'experiments': [
                {'id': 'read_again', 'outcome': 'Reference is active',
                 'actions': [{'action': 'TALK', 'arg1': 101}],
                 'predictions': {'same': True, 'different': True}},
                {'id': 'unit_change', 'outcome': 'Reference is active after the change',
                 'actions': [{'action': 'TALK', 'arg1': 101},
                             {'chosen_dialog_option_int': 5},
                             {'chosen_dialog_option_int': 1}],
                 'predictions': {'same': True, 'different': False}}]}

def answer(spec, observations, memory):
    record = observations[-1]
    return {'status': 'inconclusive', 'answer': 'The acquired response needs interpretation.',
            'evidence': [record['index']]}
"""
ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    reactor = next(row for row in record['observation']['accessibleEnvironmentObjects']
                   if row['uuid'] == 101)
    active = reactor['name'] == 'crystal reactor (activated)'
    return {'status': 'supported' if active else 'refuted',
            'answer': {'unit_change_retains_activation': active},
            'evidence': [record['index']]}
"""


def stalled(agent, lab):
    """Exercise a diagnosed scheduling state on the real paid comparison schema."""
    compared = _compare(agent, lab)
    compared['experiment_action'] = None
    compared['selection'] = None
    compared['family_research'].update(selected_role='shared_premise_research',
        research_obligation={'reason': 'completed_tests_changed_members_without_task_or_nonlocal_progress',
            'question': 'Which shared response premise requires a distinguishing observation?'})
    author = agent.llm_inquiries.setdefault(AUTHOR, {
        'question_id': 'computed', 'notes': 'The same premise remains unresolved.',
        'events': [], 'proposal': {}})
    author['proposal']['comparison_result'] = deepcopy(compared)
    return author, compared


def finish_provider(agent, count=1):
    provider = agent.provider = Provider([
        ('finish', {'lesson': 'The premise remains unknown.'}) for _ in range(count)])
    return provider


def test_stagnation_dispatch_runs_authored_contrast_and_returns_evidence_to_parent(tmp_path):
    agent, lab = _ready(tmp_path)
    author, _ = stalled(agent, lab)
    before = len(agent.observations)
    binding = (tmp_path / BINDING).read_text()
    provider = agent.provider = Provider([
        ('write', {'path': DESIGN_PATH, 'code': DESIGN}),
        ('run', {'operator': 'contrast', 'path': DESIGN_PATH, 'question': {'id': PREMISE}}),
        ('evaluate', {'question_id': PREMISE, 'path': 'shared/premise_answer.py', 'code': ANALYSIS}),
        ('finish', {'lesson': 'Use the returned negative response in subsequent research.'}),
    ])

    result = agent.run(lab)

    assert result['status'] == 'agent_finished'
    assert author['premise_question_id'] == PREMISE
    assert len(provider.requests) == agent.costs['model_calls'] == 4
    assert all(request['llm_inquiry']['question_id'] == PREMISE for request in provider.requests[:3])
    assert all(request['llm_inquiry'].get('stage') != 'proposal' for request in provider.requests[:3])
    assert 'llm_inquiry' not in provider.requests[-1]
    assert provider.requests[0]['llm_inquiry']['remaining_model_calls'] == 8
    child = next(q for q in agent.questions if q['id'] == PREMISE)
    assert child['status'] == 'refuted'
    assert child['answer'] == {'unit_change_retains_activation': False}
    assert child['evidence'] == [before + 2]
    assert len(agent.observations) == before + 3 and not lab.active(1)
    assert (tmp_path / BINDING).read_text() == binding
    event, = [event for event in agent.events
              if event.get('trigger_origin') == 'host_shared_premise_research']
    assert event['arguments']['operator'] == 'inquire'
    assert event['arguments']['max_model_calls'] == 8
    assert event.get('usage', {}) == {}
    assert agent.llm_inquiries[PREMISE]['response_premise_context']
    # Evidence obtained inside this invocation is part of its terminal input
    # snapshot; returning to the parent cannot automatically rerun the child.
    agent.status = 'running'
    assert response.dispatch_response_premise(agent, lab) is False
    assert len(provider.requests) == 4


@pytest.mark.parametrize('condition', [
    'vanilla', 'frozen', 'nested', 'one_call', 'no_premise_role',
    'selected_experiment', 'executable_experiment',
])
def test_dispatch_keeps_existing_experiments_and_reserved_parent_call(tmp_path, condition):
    agent, lab = _ready(tmp_path)
    author, _ = stalled(agent, lab)
    provider = finish_provider(agent)
    compared = author['proposal']['comparison_result']
    if condition == 'vanilla':
        agent.agent_arm = 'code_react'
    elif condition == 'frozen':
        agent.provider = None
    elif condition == 'nested':
        agent._inquiry_stack = [{'question_id': 'enclosing'}]
    elif condition == 'one_call':
        agent.limits['max_model_calls'] = agent.costs['model_calls'] + 1
    elif condition == 'no_premise_role':
        compared['family_research']['selected_role'] = 'family_discrimination'
    elif condition == 'selected_experiment':
        compared['selection'] = {'selected_experiment': {'number': 1, 'prediction': 101.}}
    else:
        compared['experiment_action'] = {'action': 'run', 'arguments': {
            'operator': 'reference_response', 'path': PATH,
            'experiment': {'number': 1, 'setting': 101.}}}
    before = deepcopy(agent.costs), len(agent.observations)

    assert response.dispatch_response_premise(agent, lab) is False

    assert (agent.costs, len(agent.observations)) == before
    assert not provider.requests and PREMISE not in agent.llm_inquiries


def test_two_remaining_calls_allow_child_to_return_unknown_and_leave_parent_one(tmp_path):
    agent, lab = _ready(tmp_path)
    stalled(agent, lab)
    agent.limits['max_model_calls'] = agent.costs['model_calls'] + 2
    provider = agent.provider = Provider([
        ('finish', {'lesson': 'The available evidence does not distinguish the premise.'}),
        ('finish', {'lesson': 'The child returned without a scientific answer.'}),
    ])

    result = agent.run(lab)

    assert result['status'] == 'agent_finished' and agent.costs['model_calls'] == 2
    assert provider.requests[0]['llm_inquiry']['remaining_model_calls'] == 1
    assert 'llm_inquiry' not in provider.requests[1]
    child = next(q for q in agent.questions if q['id'] == PREMISE)
    assert child['status'] == 'open' and 'answer' not in child
    assert agent.llm_inquiries[PREMISE]['status'] == 'inquiry_finished'


def test_same_input_renaming_and_repeated_menu_reads_do_not_reopen_but_new_response_does(tmp_path):
    agent, lab = _ready(tmp_path)
    author, _ = stalled(agent, lab)
    provider = finish_provider(agent, 2)
    assert response.dispatch_response_premise(agent, lab) is True
    snapshot = deepcopy(agent.llm_inquiries[PREMISE]['response_premise_context'])
    assert response.dispatch_response_premise(agent, lab) is False
    compared = author['proposal']['comparison_result']
    compared['question_id'] = 'renamed_comparison'
    compared['revision_action']['arguments']['question'].update(
        id='renamed_comparison', question='The same physical problem with a new title')
    compared['family_research']['research_obligation']['question'] = 'A reworded premise question'
    assert response.dispatch_response_premise(agent, lab) is False

    agent._execute('query', {'actions': [
        {'action': 'TALK', 'arg1': 101}, {'chosen_dialog_option_int': 1}]}, lab)
    stalled(agent, lab)
    assert response.dispatch_response_premise(agent, lab) is False
    assert len(provider.requests) == 1
    assert agent.llm_inquiries[PREMISE]['response_premise_context'] == snapshot

    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 1, 'setting': 102.}}, lab)
    assert trial['requested_setting_reached'] and trial['prediction_status'] == 'refuted'
    # A physical action without response_comparison has not refreshed this
    # author's stalled-family verdict. Let the parent compare the new response
    # before that old verdict can allocate another premise inquiry.
    assert response.dispatch_response_premise(agent, lab) is False
    assert len(provider.requests) == 1
    assert agent.llm_inquiries[PREMISE]['response_premise_context'] == snapshot
    stalled(agent, lab)

    assert response.dispatch_response_premise(agent, lab) is True
    assert author['premise_question_id'] == PREMISE
    assert len(provider.requests) == 2
    assert provider.requests[-1]['llm_inquiry']['invocation'] == 2
    assert agent.llm_inquiries[PREMISE]['response_premise_context'] != snapshot
    assert response.dispatch_response_premise(agent, lab) is False


def test_restored_session_retains_the_completed_input_without_buying_another_call(tmp_path):
    agent, lab = _ready(tmp_path / 'source')
    stalled(agent, lab)
    finish_provider(agent)
    assert response.dispatch_response_premise(agent, lab) is True
    # The public saved session is the persistence boundary. No scheduling-only
    # transient set is copied into the restored agent.
    agent._save_state()
    restored = DiscoveryWorldAgent(Provider([]), tmp_path / 'restored', arm='hegel',
                                  limits=agent.limits)
    restored.lab = lab
    restored.observations = json.loads(json.dumps(agent.observations))
    restored.questions = json.loads(json.dumps(agent.questions))
    restored.llm_inquiries = json.loads((agent.workspace / 'llm_inquiries.json').read_text())
    restored.costs = deepcopy(agent.costs)
    for path in (agent.workspace / 'shared').rglob('*'):
        if path.is_file():
            target = restored.workspace / path.relative_to(agent.workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text())
    before = deepcopy(restored.costs), len(restored.observations)

    assert response.dispatch_response_premise(restored, lab) is False

    assert (restored.costs, len(restored.observations)) == before
    assert not restored.provider.requests
    assert restored.llm_inquiries[AUTHOR]['premise_question_id'] == PREMISE


@pytest.mark.parametrize('revision', ['basis', 'features'])
def test_revised_executable_binding_reopens_same_premise_session(tmp_path, revision):
    agent, lab = _ready(tmp_path)
    stalled(agent, lab)
    provider = finish_provider(agent, 2)
    assert response.dispatch_response_premise(agent, lab) is True
    before = len(agent.observations)
    if revision == 'basis':
        source = BIND_CODE.replace(
            "[{'op': 'feature', 'name': 'log_density'}]",
            "[{'op': 'feature', 'name': 'log_density'}, "
            "{'op': 'mul', 'children': [{'op': 'feature', 'name': 'log_density'}, "
            "{'op': 'feature', 'name': 'log_density'}]}]")
    else:
        source = BIND_CODE.replace('math.log(value)', 'math.sqrt(value)')
    agent._execute('write', {'path': BINDING, 'code': source}, lab)
    stalled(agent, lab)

    assert response.dispatch_response_premise(agent, lab) is True

    assert len(provider.requests) == 2
    assert provider.requests[-1]['llm_inquiry']['question_id'] == PREMISE
    assert provider.requests[-1]['llm_inquiry']['invocation'] == 2
    assert len(agent.observations) == before


def test_automatic_deduplication_does_not_block_explicit_parent_redelegation(tmp_path):
    agent, lab = _ready(tmp_path)
    stalled(agent, lab)
    provider = finish_provider(agent, 2)
    assert response.dispatch_response_premise(agent, lab) is True
    assert response.dispatch_response_premise(agent, lab) is False

    returned = agent._execute('run', {'operator': 'inquire',
        'question': {'id': PREMISE}, 'max_model_calls': 1}, lab)

    assert returned['status'] == 'inquiry_finished'
    assert provider.requests[-1]['llm_inquiry']['invocation'] == 2
    assert agent.costs['model_calls'] == len(provider.requests) == 2
    assert len([event for event in agent.events
                if event.get('trigger_origin') == 'host_shared_premise_research']) == 1


@pytest.mark.parametrize('revision', ['reordered', 'duplicate', 'simplified_equivalent'])
def test_equivalent_executable_bases_do_not_reopen_the_premise(tmp_path, revision):
    agent, lab = _ready(tmp_path)
    feature = {'op': 'feature', 'name': 'log_density'}
    square = {'op': 'mul', 'children': [deepcopy(feature), deepcopy(feature)]}
    original = [feature, square]

    def source(basis):
        return BIND_CODE.replace("[{'op': 'feature', 'name': 'log_density'}]", repr(basis))

    agent._execute('write', {'path': BINDING, 'code': source(original)}, lab)
    stalled(agent, lab)
    provider = finish_provider(agent)
    assert response.dispatch_response_premise(agent, lab) is True
    snapshot = deepcopy(agent.llm_inquiries[PREMISE]['response_premise_context'])
    if revision == 'reordered':
        equivalent = list(reversed(original))
    elif revision == 'duplicate':
        equivalent = original + [deepcopy(feature)]
    else:
        equivalent = [{'op': 'mul', 'children': [deepcopy(feature), {'op': 'const', 'value': 1.}]},
                      deepcopy(square)]
    agent._execute('write', {'path': BINDING, 'code': source(equivalent)}, lab)
    stalled(agent, lab)

    assert response.dispatch_response_premise(agent, lab) is False

    assert len(provider.requests) == 1
    assert agent.llm_inquiries[PREMISE]['response_premise_context'] == snapshot


@pytest.mark.parametrize('available', ['selection', 'experiment_action'])
def test_new_usable_proposal_keeps_old_stalled_author_from_taking_parent_calls(tmp_path, available):
    agent, lab = _ready(tmp_path)
    author, compared = stalled(agent, lab)
    newer = deepcopy(compared)
    newer.update(binding_path='shared/newer_response.py', question_id='newer_response')
    newer['family_research']['selected_role'] = 'family_discrimination'
    if available == 'selection':
        newer['selection'] = {'selected_experiment': {'number': 1, 'prediction': 101.}}
    else:
        newer['experiment_action'] = {'action': 'run', 'arguments': {
            'operator': 'reference_response', 'path': PATH,
            'experiment': {'number': 1, 'setting': 101.}}}
    agent.llm_inquiries['proposal:' + newer['binding_path']] = {
        'question_id': newer['question_id'], 'events': [], 'notes': '',
        'proposal': {'comparison_result': newer}}
    provider = finish_provider(agent)
    before = deepcopy(agent.costs), len(agent.observations)

    assert response.dispatch_response_premise(agent, lab) is False

    assert not provider.requests
    assert (agent.costs, len(agent.observations)) == before
    assert 'premise_question_id' not in author
