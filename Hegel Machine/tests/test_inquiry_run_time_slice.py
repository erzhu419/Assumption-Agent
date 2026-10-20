"""A timed-out computation must leave the online researcher able to revise it."""

import pytest

from hegel_machine.cognition import online_inquiry_llm as runtime
from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab


def decision(action, arguments):
    return {"action": action, "arguments": arguments,
            "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}


@pytest.mark.parametrize("arm", runtime.ARMS)
def test_expired_computation_returns_to_provider_with_time_for_its_next_action(tmp_path, arm):
    class RevisingProvider:
        calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 1:
                return decision("run", {
                    "code": "import time\nwhile True:\n    time.sleep(.01)\n"})
            feedback = request["events"][-1]["feedback"]
            assert feedback["timeout"] is True
            assert feedback["returncode"] != 0
            assert feedback["execution_seconds"] > 0
            assert feedback["remaining_execution_seconds"] > 0
            assert "not a scientific evaluation" in feedback["execution_notice"]
            return decision("run", {"code": "print('revised computation completed')"})

    provider = RevisingProvider()
    agent = runtime.InquiryLLMAgent(provider, tmp_path, arm=arm,
        config=runtime.InquiryLLMConfig(execution_seconds=5., max_model_calls=2,
                                      confirmation_rows=0, run_timeout_seconds=.2))

    result = agent.run(AnonymousHistoryLab())

    assert provider.calls == 2
    assert result["costs"]["model_calls"] == 2
    assert 0 < result["costs"]["python_execution_seconds"] < 5.
    assert result["events"][0]["feedback"]["timeout"] is True
    resumed = result["events"][1]["feedback"]
    assert resumed["returncode"] == 0
    assert resumed["timeout"] is False
    assert resumed["stdout"].strip() == "revised computation completed"
    assert resumed["remaining_execution_seconds"] == pytest.approx(
        5. - result["costs"]["python_execution_seconds"])


@pytest.mark.parametrize("arguments, expected_timeout", [
    ({}, 300.),
    ({"timeout_seconds": 900.}, 900.),
    ({"timeout_seconds": 10000.}, 3600.),
])
def test_default_long_task_slice_and_deliberate_longer_computation(
        tmp_path, monkeypatch, arguments, expected_timeout):
    timeouts = []

    def execute(command, payload, timeout, cwd):
        timeouts.append(timeout)
        return 0, "computed\n", "", False

    monkeypatch.setattr(runtime, "_run", execute)
    agent = runtime.InquiryLLMAgent(None, tmp_path, arm="hegel",
        config=runtime.InquiryLLMConfig(execution_seconds=3600.))

    agent._execute("run", {"code": "print('computed')", **arguments}, None)

    assert timeouts == [expected_timeout]
    assert agent.config.execution_seconds == 3600.


def test_default_run_slice_does_not_shorten_research_program_execution(tmp_path, monkeypatch):
    timeouts = []

    def execute(command, payload, timeout, cwd):
        timeouts.append(timeout)
        return 0, '[{"kind":"trajectory","sequence":[{"u":0.0}]}]', "", False

    monkeypatch.setattr(runtime, "_run", execute)
    agent = runtime.InquiryLLMAgent(None, tmp_path, arm="hegel",
        config=runtime.InquiryLLMConfig(execution_seconds=3600.))

    feedback = agent._execute("query", {"path": "shared/plan.py", "question": {
        "id": "q", "question": "What response occurs after reset?"}}, AnonymousHistoryLab())

    assert timeouts == [3600.]
    assert len(feedback["observations"]) == 1


def test_failed_process_still_charges_elapsed_time(tmp_path, monkeypatch):
    def execute(*args):
        raise OSError("filesystem became unavailable")

    clock = iter([10., 13.])
    monkeypatch.setattr(runtime, "_run", execute)
    monkeypatch.setattr(runtime, "monotonic", lambda: next(clock))
    agent = runtime.InquiryLLMAgent(None, tmp_path, arm="code_react",
        config=runtime.InquiryLLMConfig(execution_seconds=60.))
    agent.costs["python_execution_seconds"] = 7.

    with pytest.raises(OSError, match="filesystem became unavailable"):
        agent._execute("run", {"code": "print('computed')"}, None)

    assert agent.costs["python_execution_seconds"] == 10.


@pytest.mark.parametrize("frozen, requested, expected", [
    (False, None, 300.), (True, None, 3500.),
    (False, 900., 900.), (True, 900., 900.),
])
def test_frozen_procedure_keeps_task_allowance_while_live_run_returns_to_author(
        tmp_path, monkeypatch, frozen, requested, expected):
    import json
    from hegel_machine.cognition.temporal_paired_method import (
        ProcedureConsumerAgent, execute_procedure)

    arguments = {"code": "print('fitting')"}
    if requested is not None:
        arguments['timeout_seconds'] = requested
    proposal = json.dumps({"action": "run", "arguments": arguments})
    timeouts = []

    def execute(command, payload, timeout, cwd):
        timeouts.append(timeout)
        return (0, proposal, "", False) if len(timeouts) == 1 else (0, "fitted", "", False)

    monkeypatch.setattr(runtime, "_run", execute)
    monkeypatch.setattr(runtime, "monotonic", lambda: 0.)
    config = runtime.InquiryLLMConfig(execution_seconds=3600.)
    agent = ProcedureConsumerAgent(None, tmp_path, arm="hegel", config=config)
    agent.costs["python_execution_seconds"] = 100.
    (tmp_path / "procedure.py").write_text("# Authored method fixture\n")

    result = execute_procedure(agent, AnonymousHistoryLab(), "procedure.py", 1,
                               record_events=frozen)

    assert result["status"] == "method_step_limit"
    assert timeouts == [3500., expected]
    assert agent.config is config


@pytest.mark.parametrize("arguments, expected_timeout", [
    ({}, 8.),
    ({"timeout_seconds": 2.}, 2.),
    ({"timeout_seconds": 100.}, 8.),
])
def test_run_uses_requested_time_but_never_exceeds_remaining_account(
        tmp_path, monkeypatch, arguments, expected_timeout):
    timeouts = []

    def execute(command, payload, timeout, cwd):
        timeouts.append(timeout)
        return 0, "computed\n", "", False

    monkeypatch.setattr(runtime, "_run", execute)
    agent = runtime.InquiryLLMAgent(None, tmp_path, arm="code_react",
        config=runtime.InquiryLLMConfig(execution_seconds=20.))
    agent.costs["python_execution_seconds"] = 12.

    feedback = agent._execute("run", {"code": "print('computed')", **arguments}, None)

    assert timeouts == [expected_timeout]
    assert feedback["returncode"] == 0
    assert feedback["execution_seconds"] >= 0
    assert feedback["remaining_execution_seconds"] == pytest.approx(
        20. - agent.costs["python_execution_seconds"])


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), "soon"])
def test_invalid_time_returns_error_without_executing_script(tmp_path, monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("An invalid time slice must not launch Python.")

    monkeypatch.setattr(runtime, "_run", forbidden)

    class CorrectingProvider:
        calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 1:
                return decision("run", {"timeout_seconds": invalid,
                    "code": "from pathlib import Path\nPath('unexpected').touch()"})
            assert "error" in request["events"][-1]["feedback"]
            return decision("write", {"path": "revision.py", "code": "print('revised')"})

    provider = CorrectingProvider()
    agent = runtime.InquiryLLMAgent(provider, tmp_path, arm="code_react",
        config=runtime.InquiryLLMConfig(max_model_calls=2, confirmation_rows=0))

    result = agent.run(AnonymousHistoryLab())

    assert provider.calls == 2
    assert result["costs"]["python_execution_seconds"] == 0
    assert not (tmp_path / "unexpected").exists()
    assert (tmp_path / "revision.py").read_text() == "print('revised')"
    assert result["events"][1]["feedback"]["written"] == "revision.py"
