"""A runtime interruption retains paid edits and partial evidence without replay."""

from copy import deepcopy
import json
import shutil

import pytest

from hegel_machine.cognition.interrupted_method_author import recover_interrupted_method_author
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_python_method_learning_study import usage_summary


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def stopped_edit(tmp_path):
    old, copied = tmp_path / 'stopped/author', tmp_path / 'native/author'
    source = 'def select_actions(spec, records, memory):\n    return {"stop": "authored"}\n'
    costs = {'observation_cost': 0, 'measurement_cost': 0, 'compute_cost': 0,
             'python_execution_seconds': 0., 'model_calls': 0, 'reported_tokens': 0}
    previous = {'status': 'researching', 'workspace': str(old), 'costs': costs, 'events': [],
        'model': None, 'method_source': None, 'outgoing_knowledge': {'files': {METHOD_PATH: 'old source'}},
        'practice_trials': [{'case': 'case_1', 'directory': 'trials/trial_000',
            'method_source': 'old source', 'status': 'method_stopped', 'costs': {},
            'fresh_confirmation': None, 'selected_prior_predictor': False}]}
    write_json(old / 'active.json', previous)
    write_json(old.parent / 'protocol.json', {'model': 'gpt-6-luna',
                                             'practice_cases': {'case_0': {}, 'case_1': {}}})
    write_json(old / 'code_action_001.json', {'action': 'write',
        'arguments_json': json.dumps({'path': METHOD_PATH}), 'source_code': source})
    (old / 'code_call_001.jsonl').write_text(json.dumps({'type': 'turn.completed',
        'usage': {'input_tokens': 31349, 'output_tokens': 3031, 'cached_input_tokens': 0}}) + '\n')
    trial = old / 'trials/trial_001'
    for path in (old / METHOD_PATH, trial / METHOD_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    state = {'costs': {**costs, 'observation_cost': 72, 'compute_cost': 20736,
                      'python_execution_seconds': 111.727654},
        'method_results': [{'step': i, 'path': METHOD_PATH,
                            'feedback': {'recorded_step': i}} for i in range(31)],
        'experiment_comparisons': [{'index': i} for i in range(32)],
        'evaluated_model': {'model': 'candidate.py', 'support': 'paid_fit'},
        'questions': [{'id': 'task'}], 'revisions': [], 'limits': {'observation_budget': 360}}
    write_json(trial / 'research_state.json', state)
    write_json(trial / 'laboratory_observations.json', [{'row_indices': list(range(72))}])
    shutil.copytree(old.parent, copied.parent)
    return old, copied, previous, state, source


def test_preserves_paid_reply_partial_steps_and_actual_costs_without_replay(stopped_edit):
    old, copied, previous, state, source = stopped_edit
    result = recover_interrupted_method_author(old, copied)
    assert json.loads((old / 'active.json').read_text()) == previous
    assert not (old / 'trials/trial_001/result.json').exists()
    assert (old / METHOD_PATH).read_text() == (copied / METHOD_PATH).read_text() == source
    assert result['status'] == 'runtime_interrupted' and result['model'] is None
    assert result['outgoing_knowledge']['files'][METHOD_PATH] == source
    assert result['practice_trials'][:-1] == previous['practice_trials']
    assert result['practice_trials'][-1]['case'] == 'case_1'
    assert result['practice_trials'][-1]['fresh_confirmation'] is None
    assert result['costs'] == {**state['costs'], 'model_calls': 1, 'reported_tokens': 34380}
    assert usage_summary(result)['known_tokens'] == 34380
    assert usage_summary(result)['missing_usage_calls'] == 0
    practice = json.loads((copied / 'trials/trial_001/result.json').read_text())
    assert len(practice['method_results']) == 31 and len(practice['experiment_comparisons']) == 32
    assert practice['method_results'] == state['method_results']
    assert practice['fresh_confirmation'] is None and practice['status'] == 'runtime_interrupted'
    assert result['events'][-1]['feedback']['edit_practice']['steps'] == state['method_results']
    assert json.loads((copied / 'runtime_interrupted.json').read_text()) == result


def test_does_not_attribute_a_changed_method_to_the_paid_reply(stopped_edit):
    old, copied, *_ = stopped_edit
    (copied / METHOD_PATH).write_text('a different method')
    with pytest.raises(ValueError, match='sources differ'):
        recover_interrupted_method_author(old, copied)
    assert not (copied / 'runtime_interrupted.json').exists()


def test_completed_practice_uses_its_real_result_instead_of_partial_recovery(stopped_edit):
    old, copied, *_ = stopped_edit
    write_json(old / 'trials/trial_001/result.json', {'status': 'method_step_limit'})
    with pytest.raises(ValueError, match='already has a result'):
        recover_interrupted_method_author(old, copied)
    assert not (copied / 'runtime_interrupted.json').exists()


def test_known_prior_phase_costs_are_retained_and_new_reply_charged_once(stopped_edit):
    old, copied, previous, state, _ = stopped_edit
    previous = deepcopy(previous)
    previous['costs'].update(observation_cost=10, compute_cost=20, reported_tokens=30,
                             python_execution_seconds=2.)
    write_json(old / 'active.json', previous)
    result = recover_interrupted_method_author(old, copied)
    assert result['costs']['observation_cost'] == 82
    assert result['costs']['compute_cost'] == 20756
    assert result['costs']['reported_tokens'] == 34410
    assert result['costs']['python_execution_seconds'] == state['costs']['python_execution_seconds'] + 2.


def test_stopped_repetition_returns_paid_evidence_and_debits_unfinished_worker(stopped_edit):
    old, copied, previous, state, source = stopped_edit
    no_progress = {'completed_comparisons': 3, 'paid_rows': 72,
                   'new_rows_between_comparisons': 0,
                   'diagnoses': ['single_supported_candidate'] * 3}
    stop = {'reason': 'operator_stopped_repeated_no_progress',
            'interrupted_python_seconds': 37.25, 'worker_pid': 1234,
            'no_progress': no_progress}
    write_json(old / 'runtime_stop.json', stop)

    result = recover_interrupted_method_author(old, copied)

    assert result['status'] == 'runtime_interrupted'
    practice = json.loads((copied / 'trials/trial_001/result.json').read_text())
    assert practice['status'] == 'method_no_progress'
    assert practice['no_progress'] == no_progress
    assert practice['runtime_interruption']['operator_stop'] == stop
    assert practice['fresh_confirmation'] is None and practice['model'] is None
    assert practice['paid_observations'] == [{'row_indices': list(range(72))}]
    assert practice['method_results'] == state['method_results']
    assert practice['experiment_comparisons'] == state['experiment_comparisons']
    assert practice['costs']['python_execution_seconds'] == (
        state['costs']['python_execution_seconds'] + stop['interrupted_python_seconds'])
    assert result['costs'] == {**practice['costs'], 'model_calls': 1, 'reported_tokens': 34380}
    feedback = result['events'][-1]['feedback']['edit_practice']
    assert feedback['status'] == 'method_no_progress'
    assert feedback['no_progress'] == no_progress
    assert feedback['steps'] == state['method_results']
    assert result['practice_trials'][-1]['no_progress'] == no_progress
    assert result['outgoing_knowledge']['files'][METHOD_PATH] == source
    assert json.loads((old / 'active.json').read_text()) == previous
    assert json.loads((old / 'trials/trial_001/research_state.json').read_text()) == state
    assert not (old / 'trials/trial_001/result.json').exists()

    # Re-exporting the same stopped input cannot charge the worker twice.
    assert recover_interrupted_method_author(old, copied) == result


@pytest.mark.parametrize('with_stop', [False, True])
def test_explicit_practice_recovers_its_actual_case_and_run_feedback(stopped_edit, with_stop):
    old, copied, previous, state, source = stopped_edit
    # The explicit case differs from the earlier failed case chosen after edits.
    arguments = {'operator': 'practice', 'case': 'case_0'}
    payload = {'action': 'run', 'arguments_json': json.dumps(arguments)}
    write_json(old / 'code_action_001.json', payload)
    write_json(copied / 'code_action_001.json', payload)
    if with_stop:
        write_json(old / 'runtime_stop.json', {
            'reason': 'operator_stopped_to_restore_missing_author_feedback',
            'interrupted_python_seconds': 12.5})

    result = recover_interrupted_method_author(old, copied)

    trial = result['practice_trials'][-1]
    assert trial['case'] == 'case_0' and trial['method_source'] == source
    assert trial['status'] == 'runtime_interrupted' and 'no_progress' not in trial
    event = result['events'][-1]
    assert event['action'] == 'run' and event['arguments'] == arguments
    assert event['feedback']['case'] == 'case_0'
    assert event['feedback']['status'] == 'runtime_interrupted'
    assert event['feedback']['steps'] == state['method_results']
    assert 'edit_practice' not in event['feedback'] and 'written' not in event['feedback']
    assert result['costs']['reported_tokens'] == 34380 and result['costs']['model_calls'] == 1
    assert result['costs']['python_execution_seconds'] == (
        state['costs']['python_execution_seconds'] + (12.5 if with_stop else 0.))
    practice = json.loads((copied / 'trials/trial_001/result.json').read_text())
    assert practice['status'] == 'runtime_interrupted' and 'no_progress' not in practice
    assert practice['fresh_confirmation'] is practice['model'] is None
    assert practice['experiment_comparisons'] == state['experiment_comparisons']
    assert (old / METHOD_PATH).read_text() == (copied / METHOD_PATH).read_text() == source
    assert json.loads((old / 'active.json').read_text()) == previous
    assert not (old / 'trials/trial_001/result.json').exists()


@pytest.mark.parametrize('action,arguments', [
    ('run', {'code': 'print(1)'}),
    ('run', {'operator': 'restore_method', 'trial': 'trials/trial_000'}),
    ('write', {'path': 'another.py', 'code': 'pass'}),
    ('finish', {'path': METHOD_PATH}),
])
def test_other_saved_actions_are_not_reinterpreted_as_paid_practice(stopped_edit, action, arguments):
    old, copied, *_ = stopped_edit
    write_json(old / 'code_action_001.json', {
        'action': action, 'arguments_json': json.dumps(arguments)})
    with pytest.raises(ValueError, match='saved method write or explicit practice run'):
        recover_interrupted_method_author(old, copied)
    assert not (copied / 'runtime_interrupted.json').exists()
    assert not (copied / 'trials/trial_001/result.json').exists()
