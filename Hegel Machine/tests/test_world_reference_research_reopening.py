"""Resume the existing author after its paid parent trial fails to make progress."""

from copy import deepcopy
import ast
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_llm_inquiry as inquiry
from hegel_machine.cognition import world_reference_recalibration as recipe
from test_world_reference_extension import FAILED, PATH, QUESTION, prepared
from test_world_reference_parent_return import memory, response


def stalled(response_count):
    result = response()
    result['family_research'].update(response_count=response_count,
        selected_role='shared_premise_research', no_progress_cycles=3,
        research_obligation={
            'reason': 'completed_tests_changed_members_without_task_or_nonlocal_progress',
            'question': 'Which shared response premise or measurement needs revision?'})
    return result


def returned_step(response_count=127):
    return {'model_call': 7, 'action': 'run', 'arguments': {
        'operator': 'reference_hypothesis', 'path': PATH}, 'feedback': {
        'status': 'method_stopped', 'new_observation_count': 25,
        'response_research': stalled(response_count)}}


def scene():
    state, account = deepcopy(FAILED), memory()
    state['response_research'] = stalled(131)
    account['questions'][0].update(question='Does the revised response explain these outcomes?',
                                   parent_id='task', depends_on=[], status='reopened')
    account['program_results'] = [
        {'path': PATH, 'feedback': {'operator': 'inquire', 'status': 'inquiry_finished',
            'question': deepcopy(account['questions'][0]), 'steps': [returned_step()]}},
        {'path': PATH, 'feedback': {'observation_indices': [217, 218, 219, 220]}},
    ]
    return state, account


def test_paid_parent_failure_returns_to_same_author_without_rewriting_prior_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state, account = scene()
    before = deepcopy((state, account))

    selected = recipe._extend_hypotheses(state, account, 3, PATH)

    assert selected['arguments']['operator'] == 'inquire'
    assert selected['arguments']['question'] == {
        key: account['questions'][0][key] for key in (
            'id', 'question', 'parent_id', 'depends_on', 'parent_program_path')}
    assert selected['arguments']['max_model_calls'] == 22
    assert selected['arguments']['assignment'] == recipe.reference_research_question(
        3, 'shared/reference_extension_3.json', PATH)
    assert (state, account) == before
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('case', [
    'first_parent_trial', 'no_parent_acquisition', 'no_new_response',
    'no_formal_obligation', 'explicit_finish', 'active_invocation',
    'resumed_same_evidence', 'no_provider', 'no_calls',
])
def test_parent_trials_and_existing_stop_boundaries_are_preserved(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    state, account = scene()
    current = state['response_research']['family_research']
    previous = account['program_results'][0]['feedback']
    if case == 'first_parent_trial':
        account['program_results'].pop()
        current['response_count'] = 127
    elif case == 'no_parent_acquisition':
        account['program_results'].pop()
    elif case == 'no_new_response':
        current['response_count'] = 127
    elif case == 'no_formal_obligation':
        current.pop('selected_role')
    elif case == 'explicit_finish':
        previous['steps'].append({'model_call': 8, 'action': 'finish',
                                  'arguments': {'lesson': 'Return unresolved.'}})
    elif case == 'active_invocation':
        account['llm_inquiries'][QUESTION]['invocations'] = 2
    elif case == 'resumed_same_evidence':
        account['llm_inquiries'][QUESTION]['invocations'] = 2
        account['program_results'].append({'path': PATH, 'feedback': {
            **deepcopy(previous), 'steps': [returned_step(131)]}})
        account['program_results'].append({'path': PATH, 'feedback': {
            'observation_indices': [221]}})
    elif case == 'no_provider':
        account['model_available'] = False
    elif case == 'no_calls':
        account['costs']['model_calls'] = 24

    selected = recipe._extend_hypotheses(state, account, 3, PATH)

    assert selected['response_trial']['prediction'] == 7.
    assert list(account['llm_inquiries']) == [QUESTION]


def test_automatic_blocking_child_return_does_not_hide_the_author_decision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state, account = scene()
    account['program_results'][0]['feedback']['steps'].append({
        'action': 'run', 'arguments': {'operator': 'inquire'},
        'trigger_origin': 'host_pending_child_research',
        'feedback': {'status': 'answered'}})
    assert recipe._extend_hypotheses(state, account, 3, PATH)['arguments']['question']['id'] == QUESTION


def test_informative_reference_experiment_still_runs_and_declared_trial_returns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state, account = scene()
    current = state['response_research']
    current['status'] = 'family_consequence_experiment'
    current['selection'] = current.pop('operational_fallback')
    current['selection']['selected_experiment']['number'] = 4
    selected = recipe._extend_hypotheses(state, account, 3, PATH)
    assert selected['response_trial']['number'] == 4

    state, account = scene()
    account['program_arguments'] = {'estimator_trial': True}
    selected = recipe._extend_hypotheses(state, account, 3, PATH)
    assert 'return the acquired outcomes to the proposing researcher' in selected['stop']


@pytest.mark.parametrize('verdict', [None, 'refuted', 'supported'])
def test_real_same_id_inquiry_preserves_scientific_question_and_resumes_assignment(
        tmp_path, monkeypatch, verdict):
    steps = []
    if verdict:
        steps.append(('evaluate', {'question_id': QUESTION,
            'path': 'shared/narrow_response_answer.py',
            'code': "def answer(spec, observations, memory):\n    return " + repr({
                'status': verdict, 'answer': 'This measured candidate only.', 'evidence': [1]})}))
    steps.append(('finish', {'lesson': 'The new parent outcome leaves the premise unresolved.'}))
    agent, provider, _ = prepared(tmp_path, steps, calls=24, readings=(3, 4))
    state, account = scene()
    question = agent._question(account['questions'][0])
    agent.llm_inquiries[QUESTION] = {
        'question_id': QUESTION, 'status': 'inquiry_finished', 'invocations': 1,
        'model_calls': 1, 'notes': 'Keep the previous curvature-and-gap hypothesis.',
        'events': [returned_step()]}
    # Acquire the new paid outcome through the ordinary executor. The archived
    # family verdict is a scheduling fixture, not a claim about this small lab.
    paid = agent._execute('query', {'actions': [{'action': 'MEASURE'}]}, agent.lab)
    agent.method_results.extend([
        {'step': 0, 'path': PATH, 'proposal': {},
         'feedback': account['program_results'][0]['feedback']},
        {'step': 1, 'path': PATH, 'proposal': {}, 'feedback': paid}])
    (tmp_path / recipe.STATE).write_text(json.dumps(state))
    prior_source = tmp_path / 'shared/reference_before_extension_3.py'
    prior_evidence = tmp_path / 'shared/reference_extension_3.json'
    prior_source.write_text('# Original research procedure\n')
    prior_evidence.write_text('{"historical_response_count":127}\n')
    agent._save_state()
    account = agent._state()
    account['program_results'] = deepcopy(agent.method_results)
    before = deepcopy(question)
    monkeypatch.chdir(tmp_path)

    selected = recipe._extend_hypotheses(state, account, 3, PATH)
    result = inquiry.execute_llm_inquiry(agent, agent.lab, selected['arguments'])

    assert result['status'] == 'inquiry_finished'
    assert agent.llm_inquiries[QUESTION]['invocations'] == 2
    assert len(provider.requests) == (2 if verdict else 1)
    request = provider.requests[0]
    assert request['llm_inquiry']['question_id'] == QUESTION
    assert request['llm_inquiry']['invocation'] == 2
    assignment = recipe.reference_research_question(3, 'shared/reference_extension_3.json', PATH)
    for request in provider.requests:
        assert request['llm_inquiry']['assignment']['question'] == assignment
        assert request['research_root']['question'] == assignment
        assert request['llm_inquiry']['question']['question'] == before['question']
        assert request['llm_inquiry']['assignment']['parent_program_path'] == PATH
    assert 'Keep the previous curvature-and-gap hypothesis.' in request['research_notes']
    assert request['evidence']['observation_count'] == 2
    assert request['current_observation']['world_steps'] == 2
    assert request['research']['method_results'][-1]['feedback'] == paid
    assert request['evidence']['recent_action_results'][-1]['feedback']['reading'] == 4
    if verdict:
        assert question['status'] == verdict and question['evidence'] == [1]
        assert provider.requests[-1]['llm_inquiry']['question']['answer'] == 'This measured candidate only.'
    else:
        assert question == before
    assert prior_source.read_text() == '# Original research procedure\n'
    assert prior_evidence.read_text() == '{"historical_response_count":127}\n'
    assert not agent.lab.completed


def test_exported_package_preserves_declarations_and_resumes_the_same_assignment(tmp_path, monkeypatch):
    from scripts import run_response_matched_adoption as matched

    monkeypatch.chdir(tmp_path)
    declarations = ({'id': 'retained_signal', 'basis': [{'op': 'feature', 'name': 'signal'}]},)
    source = Path(recipe.__file__).read_text().replace('ESTIMATORS = ()',
                                                      'ESTIMATORS = ' + repr(declarations))
    extension = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == '_extend_hypotheses')
    lines = source.splitlines(keepends=True)
    source = (''.join(lines[:extension.lineno - 1])
              + "def _extend_hypotheses(*args):\n    return {'stop': 'Old embedded continuation'}\n"
              + ''.join(lines[extension.end_lineno:]))
    bundle = {'files': {matched.study.METHOD_PATH: source}, 'methods': []}
    exported = matched.prepare_bundle(bundle)['files'][matched.study.METHOD_PATH]
    namespace = {}
    exec(compile(exported, '<exported reference method>', 'exec'), namespace)
    state, account = scene()
    state['response_context'] = {'number': 3, 'estimators': list(declarations),
        'eligible_targets': [], 'control_grids': {}, 'responses': [], 'measurements': []}

    action = namespace['_extend_hypotheses'](state, account, 3, PATH)

    assert namespace['ESTIMATORS'] == declarations
    assert action['arguments']['assignment'] == recipe.reference_research_question(
        3, 'shared/reference_extension_3.json', PATH)
    assert action['arguments']['question']['question'] == account['questions'][0]['question']
