"""A short remaining budget preserves scores and stops before the next action."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from multiprocessing import get_context
import os
import signal
import sys
from time import monotonic, sleep

import pytest

from hegel_machine.cognition.experiment_runtime import (
    ActivityExecutionDeadline, PythonExecutionDeadline, python_execution_deadline,
)
from hegel_machine.cognition.proposers import _run
from test_python_paid_score_reuse import MODELS, choose, no_execution, scene
from test_cognition_temporal_operator_agent import make_agent


def test_operator_timing_binds_real_process_pause_without_refunding_its_python_charge(
        tmp_path, monkeypatch):
    agent, lab = make_agent(tmp_path)
    context = get_context('fork')
    receiver, sender = context.Pipe(duplex=False)

    def paused_history(arguments):
        sender.send({'entered': monotonic()})
        os.kill(os.getpid(), signal.SIGSTOP)
        return {'status': 'fixture_only', 'external_python_wall_seconds': 0.}

    monkeypatch.setattr(agent, '_history', paused_history)
    monkeypatch.setattr(agent, '_attach_research_advice',
        lambda action, arguments, result, laboratory: result)

    def execute():
        result = agent._execute('run', {'operator': 'history'}, lab)
        sender.send({'result': result, 'record': agent.operator_results[-1], 'costs': agent.costs})

    worker = context.Process(target=execute)
    worker.start()
    try:
        assert receiver.poll(5.)
        entered = receiver.recv()['entered']
        stopped_pid, status = os.waitpid(worker.pid, os.WUNTRACED)
        assert stopped_pid == worker.pid and os.WIFSTOPPED(status)
        pause_started = monotonic()
        sleep(.15)
        pause_ended = monotonic()
        os.kill(worker.pid, signal.SIGCONT)
        assert receiver.poll(5.)
        completed = receiver.recv()
        worker.join(5.)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            os.kill(worker.pid, signal.SIGCONT)
            worker.terminate()
            worker.join(5.)
        receiver.close()
        sender.close()
    result, record = completed['result'], completed['record']
    timing = result['operator_execution_timing']
    assert timing == record['operator_execution_timing']
    assert timing['operator'] == 'history' and timing['process_id'] == worker.pid
    assert (timing['monotonic_start_seconds'] <= entered <= pause_started
            <= pause_ended <= timing['monotonic_end_seconds'])
    # WSL's realtime clock stepped backwards during this control once. Retain
    # its raw UTC anchors; the monotonic interval establishes elapsed ordering.
    assert all(datetime.fromisoformat(timing[key]).utcoffset().total_seconds() == 0.
               for key in ('utc_start', 'utc_end'))
    assert timing['wall_seconds'] >= pause_ended-pause_started
    assert timing['wall_seconds']-timing['host_cpu_seconds'] >= .1
    assert timing['wall_clock_scope'] == 'monotonic_elapsed_including_wait_and_process_suspension'
    assert result['external_python_wall_seconds'] == 0.
    assert result['python_execution_seconds'] >= pause_ended-pause_started
    assert completed['costs']['python_execution_seconds'] == result['python_execution_seconds']
    assert completed['costs']['compute_cost'] == 0 and not result.get('truncated')


@pytest.mark.parametrize('failure', ['budget', 'operator_error'])
def test_operator_timing_keeps_existing_budget_refusal_and_failure_accounts(
        tmp_path, monkeypatch, failure):
    agent, lab = make_agent(tmp_path)
    if failure == 'budget':
        agent.config = replace(agent.config, execution_seconds=0.)
        expected = 'Python execution time budget exhausted'
    else:
        def fail(arguments):
            raise OSError('retained operator failure')
        monkeypatch.setattr(agent, '_history', fail)
        expected = 'retained operator failure'
    before = deepcopy(agent.costs)
    limits = agent.config
    with pytest.raises((ValueError, OSError), match=expected):
        agent._execute('run', {'operator': 'history'}, lab)
    record = agent.operator_results[-1]
    timing = record['operator_execution_timing']
    assert record['error'] == expected and timing['operator'] == 'history'
    assert timing['process_id'] == os.getpid()
    assert timing['monotonic_end_seconds'] >= timing['monotonic_start_seconds']
    assert timing['interval_scope'] == 'shared_operator_dispatch_and_budget_admission'
    assert timing['host_cpu_scope'] == 'current_process_only_excludes_child_processes'
    assert record['python_execution_seconds'] == agent.costs['python_execution_seconds']-before[
        'python_execution_seconds']
    if failure == 'budget':
        assert record['python_execution_seconds'] == 0. and agent.costs == before
    else:
        assert record['python_execution_seconds'] >= 0.
    assert record['compute_cost'] == 0 and agent.config is limits
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_slow_comparison_scope_stops_before_loading_the_whole_paid_pool(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    choose(agent, lab)
    winner = deepcopy(agent.evaluated_model)
    before = agent.costs['compute_cost']
    scores = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    paid = scores.read_bytes()
    prepare = agent._paid_python_score_scope
    started_scopes = []

    def delayed(spec):
        started_scopes.append(True)
        prepared = prepare(spec)
        sleep(2.)
        return prepared

    monkeypatch.setattr(agent, '_paid_python_score_scope', delayed)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    spent = agent.costs['python_execution_seconds']
    agent.config = replace(agent.config, execution_seconds=spent + .15)
    started = monotonic()
    result = choose(agent, lab)
    assert monotonic() - started < 1. and started_scopes == [True]
    assert result['status'] == 'comparison_incomplete_python_budget' and result['truncated']
    assert result['models_requested'] == 2 and result['models_completed'] == 0
    assert result['reused_score_count'] == result['new_scored_count'] == 0
    assert result['recommendation'] is None and 'next_action' not in result
    assert agent.evaluated_model == winner and agent.costs['compute_cost'] == before
    assert agent.costs['python_execution_seconds'] - spent >= .15
    assert scores.read_bytes() == paid and agent._paid_python_comparison is None
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_slow_paid_reference_returns_partial_and_preserves_old_winner(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    choose(agent, lab)
    winner = deepcopy(agent.evaluated_model)
    before = agent.costs['compute_cost']
    paid = (agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json').read_bytes()
    from hegel_machine.cognition import temporal_operator_agent as operators
    original = operators.python_model_context_matches
    visited = []

    def delayed(workspace, model, *args, **kwargs):
        visited.append(model)
        if model == MODELS[1]:
            sleep(2.)
        return original(workspace, model, *args, **kwargs)

    monkeypatch.setattr(operators, 'python_model_context_matches', delayed)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    spent = agent.costs['python_execution_seconds']
    agent.config = replace(agent.config, execution_seconds=spent + .2)
    started = monotonic()
    result = choose(agent, lab)
    assert monotonic() - started < 1.
    assert result['status'] == 'comparison_incomplete_python_budget' and result['truncated']
    assert result['reason'] == 'python_execution_budget_exhausted'
    assert result['models_requested'] == 2 and result['models_completed'] == 1
    assert result['reused_score_count'] == 1 and result['new_scored_count'] == 0
    assert result['recommendation'] is None and 'next_action' not in result
    assert agent.evaluated_model == winner and agent.costs['compute_cost'] == before
    assert agent.costs['python_execution_seconds'] - spent >= .2
    assert (agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json').read_bytes() == paid
    assert agent.operator_results[-1]['truncated']
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_deadline_kills_and_reaps_the_existing_model_process(tmp_path):
    child_pid = tmp_path / 'child.pid'
    code = ('import os, pathlib, time; '
            f'pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(10)')
    with pytest.raises(PythonExecutionDeadline):
        with python_execution_deadline(.2):
            _run([sys.executable, '-c', code], '', 5., str(tmp_path))
    pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_interrupted_new_score_keeps_its_real_fee_and_old_paid_results(tmp_path):
    agent, lab = scene(tmp_path)
    choose(agent, lab)
    winner = deepcopy(agent.evaluated_model)
    before = agent.costs['compute_cost']
    scores = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    paid = scores.read_bytes()
    agent._save(MODELS[1], 'import time\ndef predict(row):\n    time.sleep(10)\n    return 1.\n')
    spent = agent.costs['python_execution_seconds']
    agent.config = replace(agent.config, execution_seconds=spent + .3)
    result = choose(agent, lab)
    assert result['truncated'] and result['models_completed'] == 1
    assert result['reused_score_count'] == 1 and result['new_scored_count'] == 0
    assert agent.costs['compute_cost'] - before == len(agent.rows)
    assert agent.costs['python_execution_seconds'] - spent >= .3
    assert agent.evaluated_model == winner and scores.read_bytes() == paid
    assert result['recommendation'] is None and not agent.confirmations


def test_deadline_preserves_original_exception_and_clears_timer():
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(OSError, match='original failure'):
        with python_execution_deadline(1.):
            raise OSError('original failure')
    assert signal.getsignal(signal.SIGALRM) is handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_nested_score_cannot_consume_the_earlier_activity_deadline():
    handler = signal.getsignal(signal.SIGALRM)
    continued = []
    with pytest.raises(ActivityExecutionDeadline, match='activity time budget'):
        with python_execution_deadline(.05, activity=True):
            try:
                with python_execution_deadline(.5):
                    sleep(.2)
            except PythonExecutionDeadline:
                continued.append('converted_to_local_python_budget')
            continued.append('continued_past_activity_cutoff')
    assert continued == []
    assert signal.getsignal(signal.SIGALRM) is handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_shorter_local_python_deadline_restores_the_activity_timer():
    local_returns = []
    with pytest.raises(ActivityExecutionDeadline):
        with python_execution_deadline(.15, activity=True):
            try:
                with python_execution_deadline(.03):
                    sleep(.2)
            except PythonExecutionDeadline:
                local_returns.append('partial_python_result')
            assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
            sleep(.3)
    assert local_returns == ['partial_python_result']
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_activity_cutoff_preserves_paid_scores_without_becoming_a_local_truncation(
        tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    choose(agent, lab)
    incumbent, before = deepcopy(agent.evaluated_model), deepcopy(agent.costs)
    scores = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    paid = scores.read_bytes()
    prepare = agent._paid_python_score_scope

    def delayed(spec):
        result = prepare(spec)
        sleep(.2)
        return result

    monkeypatch.setattr(agent, '_paid_python_score_scope', delayed)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    with pytest.raises(ActivityExecutionDeadline):
        with python_execution_deadline(.05, activity=True):
            choose(agent, lab)
    assert agent.costs['compute_cost'] == before['compute_cost']
    assert agent.costs['python_execution_seconds'] > before['python_execution_seconds']
    assert agent.evaluated_model == incumbent and scores.read_bytes() == paid
    assert agent._paid_python_comparison is None
    assert agent.operator_results[-1]['error'] == 'Research activity time budget exhausted'
    assert not agent.operator_results[-1].get('truncated')
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
