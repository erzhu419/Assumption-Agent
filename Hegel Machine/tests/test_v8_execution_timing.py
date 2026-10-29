"""New-stage executor meters retain failures without changing paid contracts."""

import subprocess

import pytest

from hegel_machine.cognition import proposers
from scripts.run_method_action_pairing import measured_execution_timing, execution_timing_fields


def test_executor_success_reports_once_and_existing_return_is_unchanged(monkeypatch):
    class Process:
        returncode = 0
        def communicate(self, prompt, timeout):
            return 'answer', ''
    monkeypatch.setattr(proposers.subprocess, 'Popen', lambda *args, **kwargs: Process())
    clock = iter((10., 12.))
    monkeypatch.setattr(proposers, 'monotonic', lambda: next(clock))
    with measured_execution_timing() as meter:
        result = proposers._run(['fixture'], 'input', 3., '.')
    assert result == (0, 'answer', '', False)
    assert meter['direct_external_python_seconds'] == 2.
    assert meter['direct_external_python_calls'] == 1
    assert proposers._EXECUTION_TIMING.get() is None


@pytest.mark.parametrize('failure', [subprocess.TimeoutExpired('fixture', 1), RuntimeError('interrupted')])
def test_executor_timeout_and_host_interruption_keep_actual_interval(monkeypatch, failure):
    class Process:
        pid, returncode = 12345, -9
        def __init__(self):
            self.first = True
        def communicate(self, prompt=None, timeout=None):
            if self.first:
                self.first = False
                raise failure
            return 'retained partial output', ''
    killed = []
    monkeypatch.setattr(proposers.subprocess, 'Popen', lambda *args, **kwargs: Process())
    monkeypatch.setattr(proposers.os, 'killpg', lambda *args: killed.append(args))
    clock = iter((20., 24.))
    monkeypatch.setattr(proposers, 'monotonic', lambda: next(clock))
    with measured_execution_timing() as meter:
        if isinstance(failure, subprocess.TimeoutExpired):
            result = proposers._run(['fixture'], '', 1., '.')
            assert result == (-9, 'retained partial output', '', True)
        else:
            with pytest.raises(RuntimeError, match='interrupted'):
                proposers._run(['fixture'], '', 1., '.')
    assert killed and meter['direct_external_python_seconds'] == 4.
    assert meter['direct_external_python_calls'] == 1
    assert proposers._EXECUTION_TIMING.get() is None


def test_same_small_meter_accumulates_frozen_evaluation_without_parallel_wall_double_count():
    with measured_execution_timing() as meter:
        meter.update(native_dispatch_seconds=7., native_host_seconds=5., native_calls=1,
            direct_external_python_seconds=2., direct_external_python_calls=1)
    # Qeval reopens the same scalar account after the full agent was released.
    with measured_execution_timing(meter):
        meter['direct_external_python_seconds'] += 3.
        meter['direct_external_python_calls'] += 1
    recorded = execution_timing_fields(meter, 12.)
    assert recorded['native_host_seconds'] + recorded['external_python_seconds'] + recorded[
        'other_host_activity_seconds'] == 12.
    assert recorded['native_dispatch_seconds'] == 7.
    assert recorded['other_host_activity_seconds'] == 2.
    assert recorded['direct_external_python_calls'] == 2
    assert recorded['historical_clocks_reconstructed'] is False


@pytest.mark.parametrize('arguments', [
    {'operator': 'native_condition_research'},
    {'operator': 'research_route', 'native_representation_plan': {'compute_limit': 30}},
])
def test_real_common_native_boundary_excludes_nested_executor_wall(monkeypatch, arguments):
    from hegel_machine.cognition import common_archive_inquiry as common
    agent = object.__new__(common.CommonArchiveInquiryAgent)
    class Process:
        returncode = 0
        def communicate(self, prompt, timeout):
            return '', ''
    monkeypatch.setattr(proposers.subprocess, 'Popen', lambda *args, **kwargs: Process())
    external_clock, dispatch_clock = iter((2., 4.)), iter((0., 7.))
    monkeypatch.setattr(proposers, 'monotonic', lambda: next(external_clock))
    monkeypatch.setattr(common, 'monotonic', lambda: next(dispatch_clock))
    def native_primitive(action, arguments, laboratory):
        proposers._run(['fixture'], '', 3., '.')
        return {'status': 'fixture_return'}
    agent._execute_budgeted_action = native_primitive
    with measured_execution_timing() as meter:
        returned = agent._execute('run', arguments, None)
    assert returned['status'] == 'fixture_return'
    assert meter['native_dispatch_seconds'] == 7.
    assert meter['native_host_seconds'] == 5.
    assert meter['direct_external_python_seconds'] == 2.
    assert meter['native_calls'] == 1
