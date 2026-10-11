"""Local handoff contracts; the recorded native trace is replayed separately."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_llm_inquiry as inquiry
from hegel_machine.cognition import world_reference_recalibration as recipe
from test_world_reference_extension import FAILED, PATH, QUESTION, prepared, wrapper


def response(prediction=7.):
    return {'status': 'family_research_needed', 'question_id': 'response_review',
        'family_research': {'research_obligation': {'question': 'Which response premise needs revision?'}},
        'operational_fallback': {'selected_experiment': {'family': 'retained', 'number': 3,
            'prediction': prediction, 'predictions': {'member': True}}}}


def memory():
    return {'model_available': True, 'limits': {'max_model_calls': 24}, 'costs': {'model_calls': 2},
        'questions': [{'id': QUESTION, 'status': 'open', 'parent_program_path': PATH,
                       'opened_before_observation': 1}],
        'llm_inquiries': {QUESTION: {'invocations': 1, 'status': 'inquiry_finished', 'notes': ''}}}


def test_returned_child_leaves_live_parent_free_to_acquire_successive_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state, account = deepcopy(FAILED), memory()
    for frequency in (7., 8.):
        state['response_research'] = response(frequency)
        chosen = recipe._extend_hypotheses(state, account, 3, PATH)
        assert chosen['response_trial']['prediction'] == frequency
        assert state['response_research']['family_research']['selected_role'] == 'operational_fallback'
        # The next comparison changes its paid responses and selected experiment;
        # a previous child return must not freeze the first response context.
        state.setdefault('response_settings', []).append(
            {'crystal': 3, 'frequency': frequency, 'activated': False})
    assert account['costs']['model_calls'] == 2 < account['limits']['max_model_calls']
    assert account['questions'][0]['status'] == 'open'
    state['response_research'].pop('operational_fallback')
    for _ in range(2):
        assert 'without a usable extension' in recipe._extend_hypotheses(state, account, 3, PATH)['stop']
    assert list(account['llm_inquiries']) == [QUESTION]


def test_new_paid_refutation_reopens_research_instead_of_old_return_authorizing_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'shared').mkdir()
    (tmp_path / PATH).write_text(wrapper())
    state, account = deepcopy(FAILED), memory()
    state['response_research'] = response()
    account['questions'][0].update(status='refuted', evidence=[2])
    result = recipe._extend_hypotheses(state, account, 3, PATH)
    assert result['arguments']['operator'] == 'inquire'
    assert result['arguments']['question']['id'] == QUESTION + '_2'
    assert result['arguments']['max_model_calls'] == 22
    assert 'response_trial' not in result
    retained = json.loads((tmp_path / 'shared/reference_extension_3_2.json').read_text())
    assert retained['returned_refutations'][0]['evidence'] == [2]


@pytest.mark.parametrize('condition,expected_calls', [
    ('acquired_experiment', 1), ('no_new_observation', 2),
    ('no_executable_fallback', 2), ('different_parent', 2),
])
def test_local_trial_yields_to_its_bound_parent_without_a_finish_decision(tmp_path, monkeypatch,
                                                                        condition, expected_calls):
    steps = [('run', {'operator': 'reference_hypothesis', 'path': PATH}),
             ('finish', {'lesson': 'No executable parent continuation returned'})]
    agent, provider, _ = prepared(tmp_path, steps, calls=24, readings=(3, 4))
    original_execute = agent._execute

    def execute(action, params, lab):
        if action != 'run' or params.get('operator') != 'reference_hypothesis':
            return original_execute(action, params, lab)
        # Only the domain-specific trial is stubbed in this local scheduling test.
        # Its observation is acquired and charged through the real agent query.
        start = len(agent.observations)
        if condition != 'no_new_observation':
            original_execute('query', {'actions': [{'action': 'MEASURE'}]}, lab)
        returned = response()
        if condition == 'no_executable_fallback':
            returned.pop('operational_fallback')
        return {'operator': 'reference_hypothesis', 'status': 'method_stopped', 'path': PATH,
                'new_observation_count': len(agent.observations) - start,
                'response_research': returned}

    monkeypatch.setattr(agent, '_execute', execute)
    bound = 'shared/other.py' if condition == 'different_parent' else PATH
    result = inquiry.execute_llm_inquiry(agent, agent.lab, {'question': {
        'id': QUESTION, 'parent_id': 'task', 'question': 'Study the current response',
        'parent_program_path': bound}, 'max_model_calls': 24})
    assert result['status'] == 'inquiry_finished'
    assert len(provider.requests) == agent.costs['model_calls'] == expected_calls
    assert result['question']['status'] == 'open' and 'answer' not in result['question']
    assert agent.status == 'running' and not agent.lab.completed
    assert agent.costs['environment_steps'] == len(agent.observations) == agent.lab.environment_steps
    if expected_calls == 1:
        assert all(step['action'] != 'finish' for step in result['steps'])
        assert agent.costs['model_calls'] < agent.limits['max_model_calls']


@pytest.mark.parametrize('count', [9, 16])
def test_new_package_carries_handoff_without_replacing_retained_scientific_declarations(count, tmp_path, monkeypatch):
    import ast
    from scripts import run_response_matched_adoption as matched

    monkeypatch.chdir(tmp_path)
    source = Path(recipe.__file__).read_text()
    declarations = [{'id': 'retained_' + str(i), 'basis': [{'op': 'feature', 'name': 'signal'}]}
                    for i in range(count)]
    source = source.replace('ESTIMATORS = ()', 'ESTIMATORS = ' + repr(declarations))
    old = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == '_extend_hypotheses')
    lines = source.splitlines(keepends=True)
    source = (''.join(lines[:old.lineno - 1])
              + "def _extend_hypotheses(state, memory, number, path):\n    return {'stop': 'Old handoff'}\n"
              + ''.join(lines[old.end_lineno:]))
    bundle = {'files': {matched.study.METHOD_PATH: source, 'shared/retained_note.txt': 'old evidence'},
              'methods': [{'id': 'retained', 'path': matched.study.METHOD_PATH}]}
    original = deepcopy(bundle)
    revised = matched.prepare_bundle(bundle)
    assert bundle == original
    assert revised['methods'] == bundle['methods']
    assert revised['files']['shared/retained_note.txt'] == 'old evidence'
    exported = revised['files'][matched.study.METHOD_PATH]
    namespace = {}
    exec(compile(exported, '<prepared method>', 'exec'), namespace)
    assert namespace['ESTIMATORS'] == declarations
    state = deepcopy(FAILED)
    state.update(response_research=response(), response_context={'number': 3,
        'estimators': declarations, 'eligible_targets': [], 'control_grids': {},
        'responses': [], 'measurements': []})
    result = namespace['_extend_hypotheses'](state, memory(), 3, PATH)
    assert result['response_trial']['prediction'] == 7.
