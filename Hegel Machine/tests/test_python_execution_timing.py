"""Observe external execution without changing the existing budget clock."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import _reused_python_score


@pytest.mark.parametrize('returncode,timed_out', [(0, False), (1, False), (-9, True)])
def test_external_call_timer_covers_success_failure_and_timeout(tmp_path, monkeypatch,
                                                              returncode, timed_out):
    (tmp_path / 'model.py').write_text('def predict(row): return 0.\n')
    clock = iter((100., 102.5))
    monkeypatch.setattr(code_agent, 'monotonic', lambda: next(clock))
    calls = []

    def run(*args):
        calls.append(args)
        return returncode, json.dumps({'predictions': [0.], 'model_dependencies': []}), '', timed_out

    monkeypatch.setattr(code_agent, '_run', run)
    result = code_agent.evaluate_python_model(tmp_path, 'model.py', [Observation({'u': 1.}, 0.)])
    assert result['external_python_wall_seconds'] == 2.5
    assert result['returncode'] == returncode and result['timeout'] is timed_out
    assert len(calls) == 1
    assert 'python_execution_seconds' not in result


def test_failed_process_start_keeps_original_exception_without_inventing_measurement(tmp_path, monkeypatch):
    (tmp_path / 'model.py').write_text('def predict(row): return 0.\n')
    monkeypatch.setattr(code_agent, 'monotonic', lambda: 100.)
    original = FileNotFoundError('synthetic interpreter startup failure')

    def failed_start(*args):
        raise original

    monkeypatch.setattr(code_agent, '_run', failed_start)
    with pytest.raises(FileNotFoundError) as error:
        code_agent.evaluate_python_model(tmp_path, 'model.py', [Observation({'u': 1.}, 0.)])
    assert error.value is original
    assert not hasattr(error.value, 'external_python_wall_seconds')


@pytest.mark.parametrize('previous_seconds', [None, 3.5])
def test_reused_score_records_no_new_execution_and_preserves_original_unknown(previous_seconds):
    original = {'returncode': 0, 'mse': 0.}
    if previous_seconds is not None:
        original['external_python_wall_seconds'] = previous_seconds
    assessment = {'result': original, 'model': 'source.py',
        'scope_ref': {'file': 'scope.json', 'json_path': []},
        'execution_costs': {'python_execution_seconds': 7.}}
    saved = deepcopy(assessment)
    reference = {'file': 'assessment.json', 'json_path': []}
    result = _reused_python_score('source.py', reference, assessment)
    assert result['external_python_wall_seconds'] == 0.
    assert result['paid_score_reuse']['execution_basis_ref'] == reference
    assert assessment == saved
    if previous_seconds is None:
        assert 'external_python_wall_seconds' not in assessment['result']


def test_metering_does_not_change_closed_prediction_contract():
    source = inspect.getsource(code_agent.evaluate_python_model)
    before = source.replace('        execution_input = json.dumps(payload)\n', '').replace(
        '                                             execution_input, timeout_seconds, str(workspace))',
        '                                             json.dumps(payload), timeout_seconds, str(workspace))').replace(
        '        external_started = monotonic()\n', '').replace(
        '        external_seconds = monotonic() - external_started\n', '').replace(
        '              "external_python_wall_seconds": external_seconds,\n', '')
    base = {'runner_source': code_agent._MODEL_RUNNER, 'evaluator_source': before}
    metered = {**base, 'evaluator_source': source}
    assert code_agent.closed_temporal_prediction_contract(base) == code_agent.closed_temporal_prediction_contract(metered)
    assert code_agent.closed_temporal_prediction_contract(metered) != code_agent.closed_temporal_prediction_contract(
        {**metered, 'evaluator_source': source.replace('"risk_quantile": risk_quantile', '"risk_quantile": .1')})


def test_saved_unmetered_closed_score_is_reused_without_execution(tmp_path, monkeypatch):
    from test_literal_program_score_reuse import MODEL, scene
    from test_python_paid_score_reuse import choose, no_execution

    agent, lab, calls = scene(tmp_path, monkeypatch)
    index = json.loads((agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json').read_text())
    scope_file = Path(index['scopes'][0]['file'])
    scope = json.loads(scope_file.read_text())
    scope['evaluation_contract']['evaluator_source'] = scope['evaluation_contract']['evaluator_source'].replace(
        '        execution_input = json.dumps(payload)\n', '').replace(
        '                                             execution_input, timeout_seconds, str(workspace))',
        '                                             json.dumps(payload), timeout_seconds, str(workspace))').replace(
        '        external_started = monotonic()\n', '').replace(
        '        external_seconds = monotonic() - external_started\n', '').replace(
        '              "external_python_wall_seconds": external_seconds,\n', '')
    scope_file.write_text(json.dumps(scope))
    paid = Path(index['assessments'][0]['file']).read_bytes()
    before = agent.costs['compute_cost']
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    result = choose(agent, lab, [MODEL])
    assert result['new_scored_count'] == 0 and result['reused_score_count'] == 1
    assert result['evaluations'][0]['external_python_wall_seconds'] == 0.
    assert agent.costs['compute_cost'] == before and calls == [MODEL]
    assert Path(index['assessments'][0]['file']).read_bytes() == paid
