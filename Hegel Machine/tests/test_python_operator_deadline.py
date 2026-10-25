"""A short remaining budget preserves scores and stops before the next action."""

from copy import deepcopy
from dataclasses import replace
import os
import signal
import sys
from time import monotonic, sleep

import pytest

from hegel_machine.cognition.experiment_runtime import (
    PythonExecutionDeadline, python_execution_deadline,
)
from hegel_machine.cognition.proposers import _run
from test_python_paid_score_reuse import MODELS, choose, no_execution, scene


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
