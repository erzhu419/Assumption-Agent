"""A measured proposal survives its own child analysis, not unrelated changes."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_llm_inquiry as inquiry
from hegel_machine.cognition import world_reference_response as response
from test_world_reference_extension import PATH, QUESTION, prepared
from test_world_reference_refutation_return import RECORDS


@pytest.mark.parametrize('case,expected_calls', [
    ('point_refuted', 2), ('excluded_family', 3), ('no_new_observation', 3),
    ('different_parent', 3), ('explicit_dependency', 3), ('changed_method', 3),
    ('new_unreviewed_evidence', 3), ('refreshed_comparison', 2),
    ('refreshed_family_exclusion', 3),
])
def test_measured_parent_handoff_is_reconsidered_after_its_child(tmp_path, monkeypatch,
                                                               case, expected_calls):
    returned = deepcopy(RECORDS[1]['feedback'])
    child_id = QUESTION + '_point'
    answer_code = """def answer(spec, observations, memory):
    return {'status': 'refuted', 'answer': 'This measured point failed; the family remains unresolved.',
            'evidence': [0]}
"""
    agent, provider, _ = prepared(tmp_path, [
        ('run', {'operator': 'reference_hypothesis', 'path': PATH,
                 'hypothesis': returned['hypothesis']}),
        ('evaluate', {'question_id': child_id, 'path': 'shared/point_answer.py', 'code': answer_code}),
        ('finish', {'lesson': 'Continue deciding from the current evidence.'}),
    ], calls=24, readings=(3, 4, 5))
    original_execute = agent._execute
    state_path = tmp_path / 'shared/reference_recalibration.json'

    def execute(action, params, laboratory):
        if action == 'run' and params.get('operator') == 'reference_hypothesis':
            agent._question({'id': child_id, 'question': 'Did the predicted point activate?',
                             'parent_id': QUESTION, 'depends_on': []})
            if case == 'explicit_dependency':
                next(q for q in agent.questions if q['id'] == QUESTION)['depends_on'] = [child_id]
            if case != 'no_new_observation':
                original_execute('query', {'actions': [{'action': 'MEASURE'}]}, laboratory)
            else:
                returned['new_observation_count'] = 0
            if case == 'excluded_family':
                next(f for f in returned['response_research']['family_research']['families']
                     if returned['hypothesis']['id'] in f['source_ids'])['status'] = 'UNSAT_VERIFIED'
            state = {'response_research': returned['response_research'], 'completed': [],
                     'response_settings': [], 'control_grids': {}, 'target_numbers': [3]}
            state['response_research_inputs'] = response._reference_response_inputs(state, [])
            state_path.write_text(json.dumps(state))
            return deepcopy(returned)
        if action == 'evaluate':
            if case == 'changed_method':
                original_execute('write', {'path': PATH,
                    'code': (tmp_path / PATH).read_text() + '\n# Author revised the method.\n'}, laboratory)
            if case in ('new_unreviewed_evidence', 'refreshed_comparison', 'refreshed_family_exclusion'):
                original_execute('query', {'actions': [{'action': 'MEASURE'}]}, laboratory)
                state = json.loads(state_path.read_text())
                state['response_settings'].append({'crystal': 3, 'frequency': 9., 'activated': False})
                if case != 'new_unreviewed_evidence':
                    state['response_research']['question_id'] = 'child_refreshed_comparison'
                    state['response_research_inputs'] = response._reference_response_inputs(state, [])
                if case == 'refreshed_family_exclusion':
                    next(f for f in state['response_research']['family_research']['families']
                         if returned['hypothesis']['id'] in f['source_ids'])['status'] = 'UNSAT_VERIFIED'
                state_path.write_text(json.dumps(state))
                # The instrument laboratory above charges the observation. Its
                # calibrated response projection is supplied here; freshness is
                # checked by the actual production input comparison.
                monkeypatch.setattr(response, '_reference_state',
                    lambda *args: ([], deepcopy(state), state_path))
        return original_execute(action, params, laboratory)

    monkeypatch.setattr(agent, '_execute', execute)
    result = inquiry.execute_llm_inquiry(agent, agent.lab, {'question': {
        'id': QUESTION, 'question': 'Extend the response family', 'parent_id': 'task',
        'depends_on': [], 'parent_program_path': 'shared/other.py' if case == 'different_parent' else PATH},
        'max_model_calls': 24})

    assert result['status'] == 'inquiry_finished'
    assert agent.costs['model_calls'] == len(provider.requests) == expected_calls
    assert [request['llm_inquiry']['question_id'] for request in provider.requests[:2]] == [QUESTION, child_id]
    child = next(q for q in agent.questions if q['id'] == child_id)
    assert child['status'] == 'refuted' and child['answer'].startswith('This measured point')
    assert result['question']['status'] == 'reopened'
    assert 'answer' not in result['question']
    assert agent.status == 'running' and not agent.lab.completed
    assert any(event.get('trigger_origin') == 'host_pending_child_research' for event in result['steps'])
    assert any(event['action'] == 'finish' for event in result['steps']) == (expected_calls == 3)
