"""Explicit subscription routing must not fall through to an external API."""

import subprocess
import sys

import pytest

from scripts import run_online_inquiry_llm as runner


@pytest.mark.parametrize("stdout,stderr", [("Logged in using ChatGPT\n", ""),
                                          ("", "Logged in using ChatGPT\n")])
def test_verify_subscription_accepts_cli_login_status_without_model_call(monkeypatch, stdout, stderr):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.verify_codex_subscription() == "ChatGPT_subscription"
    assert calls == [(["codex", "login", "status"], {
        "capture_output": True, "text": True, "timeout": 30})]


@pytest.mark.parametrize("returncode,status", [(0, "Logged in using an API key"),
                                              (1, "Logged in using ChatGPT")])
def test_verify_subscription_refuses_api_key_or_failed_status(monkeypatch, returncode, status):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout=status, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(ValueError):
        runner.verify_codex_subscription()
    assert calls == [["codex", "login", "status"]]


def test_codex_cli_main_uses_subscription_without_loading_any_api_environment(tmp_path, monkeypatch):
    seen = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("subscription mode must not load API credentials or construct an API provider")

    def verify():
        seen["auth_checks"] = seen.get("auth_checks", 0) + 1
        return "ChatGPT_subscription"

    class Provider:
        def __init__(self, workspace, *, model, timeout_seconds, reasoning_effort):
            seen["provider"] = {"workspace": workspace, "model": model,
                "timeout_seconds": timeout_seconds, "reasoning_effort": reasoning_effort}

    def run_study(plan, output, *, provider_factory, workers):
        seen.update(plan=plan, output=output, workers=workers)
        assert isinstance(provider_factory(output / "probe"), Provider)
        return {"status": "interface_test"}

    monkeypatch.setattr(runner, "load_provider_environment", forbidden)
    monkeypatch.setattr(runner, "ResponsesCodeProvider", forbidden)
    monkeypatch.setattr(runner, "ChatCompletionsCodeProvider", forbidden)
    monkeypatch.setattr(runner, "verify_codex_subscription", verify)
    monkeypatch.setattr(runner, "CodexCodeProvider", Provider)
    monkeypatch.setattr(runner, "run_study", run_study)
    monkeypatch.setattr(sys, "argv", ["run_online_inquiry_llm.py", "--backend", "codex_cli",
        "--model", "gpt-6-luna", "--reasoning-effort", "low", "--timeout-seconds", "45",
        "--workers", "2", "--output", str(tmp_path)])
    runner.main()

    assert seen["auth_checks"] == 1
    assert seen["provider"] == {"workspace": tmp_path / "probe", "model": "gpt-6-luna",
                                 "timeout_seconds": 45, "reasoning_effort": "low"}
    plan = seen["plan"]
    assert plan["model"] == "gpt-6-luna"
    assert plan["base_url"] is None
    assert plan["provider"]["backend"] == "codex_cli"
    assert plan["provider"]["runtime"] == "codex_cli"
    assert plan["provider"]["auth_mode"] == "ChatGPT_subscription"
    assert plan["provider"]["reasoning_effort"] == "low"
    assert plan["provider"]["env_key"] is None
    assert plan["provider"]["max_output_tokens"] is None
    assert "gemini_cache" not in plan["provider"]
    assert seen["workers"] == 2


def test_codex_cli_rejects_gemini_cache_before_auth_or_study(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible cache must fail before auth or study")

    monkeypatch.setattr(runner, "verify_codex_subscription", forbidden)
    monkeypatch.setattr(runner, "load_provider_environment", forbidden)
    monkeypatch.setattr(runner, "run_study", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_online_inquiry_llm.py", "--backend", "codex_cli",
        "--model", "gpt-6-luna", "--gemini-cache-file", str(tmp_path / "unused.json"),
        "--output", str(tmp_path / "study")])
    with pytest.raises(ValueError, match="chat_completions"):
        runner.main()
