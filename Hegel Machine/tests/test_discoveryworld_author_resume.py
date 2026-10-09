"""Interrupted world authoring retains paid source support and one total account."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_method_learning
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import prepare_discoveryworld_transfer_method as author
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_world_method_learning import PracticeLab, SOLVE, prepared


def test_resume_retains_trials_source_notes_and_costs_then_practices_only_missing_case(
        tmp_path, monkeypatch):
    old_root = tmp_path / "interrupted"
    old, _, _ = prepared(old_root / "author", calls=64)
    old._practice("source_initial")  # The earlier draft failed.
    (old.workspace / METHOD_PATH).write_text(SOLVE)
    old.files[METHOD_PATH] = SOLVE
    old._practice("source_continuation")  # Only this exact draft/case has support.
    archive_text = json.dumps({"public_spec": {}, "origin": "FROZEN_PUBLIC_SOURCE",
        "measurement_history": [], "reference_history": [],
        "continuation_observations": [], "events": []})
    (old.workspace / "shared/source_episode.json").write_text(archive_text)
    (old.workspace / "shared/calculation.py").write_text("saved_value = 7\n")
    old.events = [
        {"action": "run", "arguments": {"operator": "practice", "case": "source_continuation",
            "notes": "Current draft solved continuation; initial remains untested."},
         "feedback": {"directory": "trials/trial_001", "status": "completed"}},
        {"action": "provider_error", "arguments": {}, "feedback": {"error": "Input too long"}},
    ]
    old.costs.update(model_calls=34, reported_tokens=1000, input_tokens=900, output_tokens=100,
                     cached_input_tokens=500, usage_missing_calls=1, wall_time_seconds=6.25)
    source_root = tmp_path / "source"
    source = source_root / "dw_04_normal_s0/hegel"
    (source / "agent").mkdir(parents=True)
    task = {"id": "dw_04_normal_s0", "split": "development", "difficulty": "Normal", "random_seed": 0}
    source_costs = {"model_calls": 4, "wall_time_seconds": 5.0}
    source_prior = {"model_calls": 279, "wall_time_seconds": 4119.0}
    model = {"requested_alias": "test", "reasoning_effort": "medium"}
    (source_root / "protocol.json").write_text(json.dumps({"tasks": [task], "model": model,
        "limits": LIMITS, "runtime": "unused"}))
    (source / "result.json").write_text(json.dumps({"task_id": task["id"],
        "prefix_environment_steps": 1, "prior_costs": source_prior, "costs": source_costs}))
    (source / "agent/laboratory_observations.json").write_text(json.dumps(old.source_records))
    prior = old.result("provider_failed")
    prior["source"] = str(source)
    prior_result = old_root / "result.json"
    prior_result.write_text(json.dumps(prior))
    protocol = {"source": str(source), "model": model, "max_model_calls": 64,
        "practice_cases": deepcopy(old.cases), "initial_bundle": "earlier/method_bundle.json",
        "initial_consolidation_costs": {"model_calls": 20, "wall_time_seconds": 123.0},
        "initial_accounting": "earlier/accounting.json", "source_prior_costs": source_prior,
        "source_last_costs": source_costs, "validation": "Both source cases must pass"}
    (old_root / "protocol.json").write_text(json.dumps(protocol))
    saved_result_text = prior_result.read_text()
    provider = ScriptedProvider([("finish", {"path": METHOD_PATH, "lesson": "Measured both cases"})])
    monkeypatch.setattr(author, "verify_codex_subscription", lambda: "test")
    monkeypatch.setattr(author, "CodexCodeProvider", lambda *args, **kwargs: provider)
    labs = []

    def laboratory(*args):
        labs.append(PracticeLab())
        return labs[-1]

    monkeypatch.setattr(world_method_learning, "DiscoveryWorldLab", laboratory)
    output = tmp_path / "resumed"
    assert author.prepare(source, output, resume_author=prior_result) == "published_source_practiced"
    result = json.loads((output / "result.json").read_text())
    resumed_plan = json.loads((output / "protocol.json").read_text())
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request["limits"]["max_model_calls"] - request["costs"]["model_calls"] == 30
    assert request["costs"] == prior["costs"] | {"wall_time_seconds": request["costs"]["wall_time_seconds"]}
    assert request["research_notes"] == old.events[0]["arguments"]["notes"]
    decision = json.loads(request["files"]["method_decision.json"])
    assert decision["current_source_coverage"]["confirmed_cases"] == ["source_continuation"]
    assert decision["current_source_coverage"]["missing_cases"] == ["source_initial"]
    assert len(labs) == 1  # No initial auto-practice or replay of the supported case.
    assert result["practice_trials"][:2] == prior["practice_trials"]
    assert [trial["case"] for trial in result["practice_trials"]] == [
        "source_initial", "source_continuation", "source_initial"]
    for trial in prior["practice_trials"]:
        path = trial["directory"] + "/result.json"
        assert (output / "author" / path).read_text() == (old.workspace / path).read_text()
    assert result["events"][:2] == prior["events"]
    assert result["method_source"] == SOLVE
    assert result["costs"]["model_calls"] == 35
    assert result["costs"]["reported_tokens"] == 1001
    assert result["costs"]["usage_missing_calls"] == 1
    assert result["costs"]["wall_time_seconds"] >= 6.25
    assert result["costs"]["observation_cost"] == prior["costs"]["observation_cost"] + 2
    assert resumed_plan["resume_author"]["previous_costs"] == prior["costs"]
    for key in ("initial_bundle", "initial_consolidation_costs", "initial_accounting",
                "source_prior_costs", "source_last_costs", "max_model_calls"):
        assert resumed_plan[key] == protocol[key]
    bundle = json.loads((output / "method_bundle.json").read_text())
    assert bundle["files"]["shared/source_episode.json"] == archive_text
    assert bundle["files"][METHOD_PATH] == SOLVE
    assert bundle["consolidation_costs"]["model_calls"] == 35
    assert bundle["prior_consolidation_costs"] == protocol["initial_consolidation_costs"]
    assert (output / "author/shared/calculation.py").read_text() == "saved_value = 7\n"
    assert prior_result.read_text() == saved_result_text
    with pytest.raises(ValueError, match="original total"):
        author.prepare(source, tmp_path / "wrong_allowance", calls=65, resume_author=prior_result)
