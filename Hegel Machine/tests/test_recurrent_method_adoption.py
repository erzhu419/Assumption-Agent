"""Actual laboratory/prefix wiring with a scripted stop, not scientific evidence."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_anonymous_method_adoption as study


STOP_METHOD = "def select_actions(spec, records, memory):\n    return {'stop': 'Scripted unresolved research'}\n"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_study(path, kind):
    cases = [{"id": f"target_{index:03d}", "kind": kind, "seed": 73321042 + index * 1009,
              "prefix_seed": 73421034 + index * 1009,
              "confirmation_seed": 73521058 + index * 1009,
              "evaluation_seed": 73621038 + index * 1009}
             for index in range(2)]
    if kind == "recurrent":
        for case in cases:
            case["apparatus_seed"] = case["seed"] + 37
    plan = {"transfer_cases": cases, "live_control_cases": [case["id"] for case in cases],
            "model": "scripted-interface-only", "provider": {"timeout_seconds": 30.},
            "limits": asdict(InquiryLLMConfig(max_model_calls=1, token_budget=100,
                observation_budget=576, confirmation_rows=96, execution_seconds=30.)),
            "prefix_rows": 72, "prefix_protocol": "24 independent three-step trajectories",
            "evaluation": {"rows": 512, "trajectory_length": 24, "mse_threshold": 1e-10},
            "execution_workers": 2,
            "runtime_snapshot": ["src/hegel_machine/cognition/temporal_paired_method.py"]}
    author = {"status": "completed", "method_path": METHOD_PATH, "method_source": STOP_METHOD,
              "method_entry": {"id": "scripted_method", "path": METHOD_PATH},
              "lesson": "Interface fixture only.", "practice_trials": []}
    save(path / "protocol.json", plan)
    save(path / "author/training.json", author)
    manifest = {"hypotheses": [], "research_programs": [], "support_questions": []}
    save(path / "initial_archive.json", {
        "files": {METHOD_PATH: "def select_actions(*args): return {'stop': 'Placeholder'}\n",
                  "shared/knowledge.json": json.dumps(manifest), "source_catalog.json": "[]"},
        "manifest": manifest, "lessons": [], "source_catalog": [],
        "research_roles": {"predictors": []}})
    save(path / "summary.json", {"private_terminal_marker": "not actor input"})
    save(path / "frozen/target_000/training.json", {"private_target_fit": "not actor input"})
    return plan


@pytest.mark.parametrize("kind", ["recurrent", "memoryless"])
def test_adoption_reuses_actual_laboratory_and_prefix_without_target_outcomes(tmp_path, monkeypatch, kind):
    source, output = tmp_path / "source", tmp_path / "adoption"
    original = source_study(source, kind)
    requests, dispatched, workers = [], [], []
    original_laboratory, original_run = study.laboratory, study.run_study

    def laboratory(case):
        lab = original_laboratory(case)
        assert isinstance(lab, TemporalInquiryLab if kind == "recurrent" else AnonymousHistoryLab)
        dispatched.append(deepcopy(case))
        return lab

    def run_study(*args, **kwargs):
        workers.append(kwargs["workers"])
        return original_run(*args, **kwargs)

    class StopProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {"action": "run", "arguments": {"operator": "procedure", "path": METHOD_PATH,
                    "max_steps": 12, "confirm": True},
                    "usage": {"backend": "scripted_test", "total_tokens": 1}}

    monkeypatch.setattr(study, "laboratory", laboratory)
    monkeypatch.setattr(study, "run_study", run_study)
    summary = study.run_adoption(source, output, provider_factory=lambda workspace: StopProvider())
    assert len(summary["trials"]) == len(requests) == len(dispatched) == 4
    assert workers == [2]
    assert all(dispatched.count(case) == 2 for case in original["transfer_cases"])
    for request in requests:
        assert request["costs"]["observation_cost"] == 72
        assert request["costs"]["measurement_cost"] == 0
        records = request["laboratory_observations"]
        assert len(records) == (24 if kind == "recurrent" else 3)
        assert all(len(record["rows"]) == (3 if kind == "recurrent" else 24) for record in records)
        assert all(not record["measurements"] for record in records)
        assert request["files"][METHOD_PATH] == STOP_METHOD
        assert "private_terminal_marker" not in json.dumps(request)
        assert "private_target_fit" not in json.dumps(request)
        assert bool(request["capabilities"]["channels"]) == (kind == "recurrent")
    for trial in summary["trials"]:
        assert trial["entry_attempts"] == 1
        # A stop-only response records the entry attempt, not supported method work.
        assert trial["unchanged_published_steps"] == 0
        assert trial["module_only_runs"] == 0
        assert not trial["published_method_confirmed"]
        assert trial["costs"]["model_calls"] == 1
        assert trial["costs"]["observation_cost"] == 72
        assert trial["terminal"]["status"] == "no_submitted_model"
    plan = json.loads((output / "protocol.json").read_text())
    assert plan["limits"] == original["limits"]
    assert plan["cases"] == original["transfer_cases"]
    assert plan["execution_workers"] == 2
    if kind == "recurrent":
        assert plan["study"] == "recurrent_published_method_live_continuation_development"
        assert "4 Code/Hegel episodes" in plan["scope"]
    else:
        assert plan["study"] == "anonymous_published_method_live_binding"
    for name in plan["runtime_snapshot"]:
        assert (output / "source_at_run" / name).read_text() == (study.ROOT / name).read_text()
