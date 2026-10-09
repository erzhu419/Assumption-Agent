"""Runner connection routing only; no API requests or model performance claims."""

import json
import os

import pytest

from scripts import run_online_inquiry_llm as runner
from hegel_machine.cognition.online_inquiry_llm import INTERFACE


@pytest.mark.parametrize("use_env_file", [False, True])
def test_default_ruoli_connection_remains_supported(tmp_path, monkeypatch, use_env_file):
    monkeypatch.setenv("RUOLI_GPT_KEY", "process-secret")
    monkeypatch.setenv("RUOLI_BASE_URL", "https://process.example.test")
    env_file = None
    if use_env_file:
        env_file = tmp_path / "provider.env"
        env_file.write_text('RUOLI_GPT_KEY="file-secret"\n'
                            "RUOLI_BASE_URL='https://file.example.test/'\n")

    base = runner.load_provider_environment(env_file)

    assert base == ("https://file.example.test/v1" if use_env_file
                    else "https://process.example.test/v1")
    assert os.environ["RUOLI_GPT_KEY"] == ("file-secret" if use_env_file else "process-secret")
    assert runner.protocol(model="chosen-model")["provider"]["env_key"] == "RUOLI_GPT_KEY"


@pytest.mark.parametrize("use_env_file", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_cli_routes_selected_connection_without_serializing_secret(
        tmp_path, monkeypatch, capsys, use_env_file, use_cache):
    secret = "gemini-test-secret"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("RUOLI_GPT_KEY", "unselected-secret")
    monkeypatch.setenv("RUOLI_BASE_URL", "https://old.example.test/v1")
    output = tmp_path / "output"
    argv = ["run_online_inquiry_llm.py", "--model", "gemini-3.5-flash-lite",
            "--backend", "chat_completions", "--base-url",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "--env-key", "GEMINI_API_KEY", "--output", str(output)]
    if use_env_file:
        monkeypatch.setenv("GEMINI_API_KEY", "stale-process-secret")
        env_file = tmp_path / "provider.env"
        env_file.write_text(f'GEMINI_API_KEY="{secret}"\n'
                            "RUOLI_BASE_URL=https://file.example.test/v1\n")
        argv.extend(["--env-file", str(env_file)])
    if use_cache:
        cache = {"name": "cachedContents/test-cache", "model": "models/gemini-3.5-flash-lite",
                 "task_interface": INTERFACE, "expireTime": "2026-10-06T00:00:00Z",
                 "usageMetadata": {"totalTokenCount": 1948}}
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        argv.extend(["--gemini-cache-file", str(cache_file)])
    monkeypatch.setattr("sys.argv", argv)
    captured = {}

    def capture_study(plan, path, *, provider_factory, workers):
        captured["provider"] = provider_factory(path / "provider")
        runner.write_json(path / "protocol.json", plan)
        return {"status": "configuration_test_only"}

    monkeypatch.setattr(runner, "run_study", capture_study)
    runner.main()

    provider = captured["provider"]
    assert isinstance(provider, runner.ChatCompletionsCodeProvider)
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert provider.env_key == "GEMINI_API_KEY"
    assert os.environ[provider.env_key] == secret
    serialized = (output / "protocol.json").read_text()
    plan = json.loads(serialized)
    assert plan["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert plan["provider"]["env_key"] == "GEMINI_API_KEY"
    assert plan["provider"]["backend"] == "chat_completions"
    assert plan["model"] == provider.model == "gemini-3.5-flash-lite"
    if use_cache:
        assert plan["provider"]["gemini_cache"] == cache
        assert provider.cached_content == cache["name"]
        assert provider.cached_task_interface == INTERFACE
    else:
        assert "gemini_cache" not in plan["provider"]
        assert provider.cached_content is None
    rendered = serialized + capsys.readouterr().out
    assert secret not in rendered
    assert "unselected-secret" not in rendered


@pytest.mark.parametrize("model,backend,message", [
    ("gemini-3.5-flash-lite", "responses", "requires --backend chat_completions"),
    ("another-model", "chat_completions", "cache model differs"),
])
def test_cache_rejects_wrong_model_or_backend(tmp_path, model, backend, message):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"name": "cachedContents/test-cache",
        "model": "models/gemini-3.5-flash-lite", "task_interface": INTERFACE}))
    with pytest.raises(ValueError, match=message):
        runner.load_gemini_cache(path, model=model, backend=backend)


def test_summary_distinguishes_cache_hits_zero_and_unknown_usage():
    usage = [{"input_tokens": 20, "cached_input_tokens": 12},
             {"input_tokens": 8, "cached_input_tokens": 0},
             {"input_tokens": 10}, {}]
    task = {"status": "completed", "events": [{"usage": item} for item in usage],
            "costs": dict.fromkeys(("observation_cost", "compute_cost", "python_execution_seconds",
                                   "model_calls", "reported_tokens"), 0)}
    results = [{"arm": "code_react", "tasks": [task], "evaluation": [{"mse": 0.0}]}]
    summary = runner.summarize(runner.protocol(model="chosen-model"), results)["arms"]["code_react"]

    assert summary["cache_hit_calls"] == 1
    assert summary["cached_input_tokens_known_lower_bound"] == 12
    assert summary["input_tokens_known_lower_bound"] == 38
    assert summary["calls_without_cache_usage"] == 2
