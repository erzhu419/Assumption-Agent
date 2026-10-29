"""Retained supported state models close before another representation search."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.finite_native_author import (
    export_finite_native_author, research_segment_options, select_actions,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_programs import (
    export_research_selection_program, propose_research_action_revisions,
)
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_native_representation_method_selection import current_memory, prepared


def retained_scene(tmp_path, monkeypatch, *, confirmation_available=True):
    agent, lab, contract, original = prepared(tmp_path, monkeypatch)
    # The retained bank already has the correct recurrence. The old diagnostic
    # parent and its outstanding representation plan remain available as well.
    mature = original.children[0].children[1]
    supported = Program('state_replay', children=(Program('if', children=(
        original.children[0].children[0], mature, mature.children[2])),))
    agent._save('supported.py', export_temporal_program(supported))
    contract['predictor_paths'].append('supported.py')
    if not confirmation_available:
        agent.config = replace(agent.config, observation_budget=0)
    comparison = agent._execute('run', {'operator': 'research_choice',
        'models': list(contract['predictor_paths']),
        'research_question': deepcopy(contract['question'])}, lab)
    assert comparison['models_completed'] == 2
    assert comparison['evaluated_model']['mse'] == 0.
    if confirmation_available:
        assert comparison['next_action']['action'] == 'finish'
    else:
        assert not comparison.get('next_action')
    assert not any(item['operator'] == 'native_representation_research'
                   for item in agent.operator_results)
    return agent, lab, comparison


def test_complete_supported_pool_closes_without_another_state_search(tmp_path, monkeypatch):
    agent, lab, comparison = retained_scene(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    before = dict(agent.costs)
    proposal = select_actions(lab.public_spec(), agent.observations, memory)
    assert proposal['action'] == 'finish'
    assert proposal['arguments'] == comparison['next_action']['arguments']
    assert agent.costs == before


def test_compiled_alternative_keeps_the_existing_supported_closure(tmp_path, monkeypatch):
    agent, lab, comparison = retained_scene(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    view = research_segment_options(lab.public_spec(), agent.observations, memory)
    proposed = propose_research_action_revisions(view['options'][0]['proposal'], view)
    assert len(proposed['candidates']) == 1
    compiled = {}
    exec(export_research_selection_program(
        proposed['candidates'][0]['research_selection_plan']), compiled)
    agent._save('ordinary.py', export_finite_native_author())
    before = dict(agent.costs)
    selected = compiled['select_actions'](lab.public_spec(), agent.observations,
        {**memory, 'ordinary_method_path': str(agent.workspace / 'ordinary.py')})
    assert selected['action'] == 'finish'
    assert selected['arguments'] == comparison['next_action']['arguments']
    assert 'research_segment_index' not in selected
    assert agent.costs == before


def test_bad_retained_pool_still_researches_its_actual_counterexample(tmp_path, monkeypatch):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    proposal = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert proposal['action'] == 'run'
    assert proposal['arguments']['operator'] == 'research_route'
    assert proposal['arguments']['native_representation_plan']['parent_theory_ids'] == ['old', 'constant']


def test_zero_mse_does_not_replace_the_original_confirmation_capacity_gate(tmp_path, monkeypatch):
    agent, lab, _ = retained_scene(tmp_path, monkeypatch, confirmation_available=False)
    selected = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert selected.get('action') != 'finish'


def test_genuine_pending_method_feedback_keeps_its_existing_priority(tmp_path, monkeypatch):
    agent, lab, _ = retained_scene(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    question = memory['finite_native_author']['question']
    basis = {'meaning_version': 1, 'answer_version': None}
    continuation = {'action': 'run', 'arguments': {
        'operator': 'method_action_feedback', 'question_id': 'method:retained'}}
    memory['research_state']['questions'].append({
        'id': 'method:retained', 'kind': 'method_use', 'parent_id': question['id'],
        'agenda_basis': basis, 'completed_agenda_basis': basis,
        'routing_execution': {'result': {
            'status': 'method_use_decided', 'next_action': continuation}}})
    agent._save('research_state.json', json.dumps(memory['research_state']))
    selected = select_actions(lab.public_spec(), agent.observations, memory)
    assert selected['action'] == continuation['action']
    assert selected['arguments'] == continuation['arguments']
