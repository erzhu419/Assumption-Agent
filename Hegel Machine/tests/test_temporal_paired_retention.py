"""Scripted retention-runner interface checks, not LLM performance evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_operator_agent import OPERATOR_INTERFACE


LESSON = "Assess whether an inherited law still applies before deciding to revise it."
METHOD = """def select_actions(spec, observations, memory):
    return {'model': memory['research_roles']['predictor']}
"""
OLD_PLAN = """def select_actions(spec, observations, memory):
    return {'actions':[{'kind':'trajectory','sequence':[{'u':.25}], 'channels':[]}]}
"""


def original_memory(label, instrument):
    manifest = {"hypotheses": [{"id": "prior", "path": "shared/prior.py"}],
                "research_programs": [{"id": "old_plan", "path": "shared/old_plan.py"}],
                "support_questions": []}
    return {
        "files": {"shared/prior.py": f"def predict(row):\n    return {label!r}\n",
                  instrument: json.dumps({"calibration": label}),
                  "shared/old_plan.py": OLD_PLAN,
                  "shared/knowledge.json": json.dumps(manifest)},
        "manifest": manifest,
        "research_roles": {"instrument": instrument, "predictor": "shared/prior.py"},
        "lessons": [f"Prior calibration {label}"],
    }


@pytest.fixture
def scripted_study(tmp_path, monkeypatch, request):
    from scripts import run_temporal_paired_retention as runner

    original = {
        "first": original_memory(1.25, "shared/meter.json"),
        "second": original_memory(2.5, "shared/current_instrument.json"),
    }
    donor_calls = []

    def donor(path, position, instrument):
        label = "first" if position == 1 else "second"
        assert instrument == original[label]["research_roles"]["instrument"]
        donor_calls.append(label)
        return deepcopy(original[label])

    learned = {"status": "completed", "method_path": METHOD_PATH,
               "method_source": METHOD,
               "method_entry": {"id": "paired_method", "path": METHOD_PATH},
               "outgoing_knowledge": {"lessons": ["Earlier supplied lesson", LESSON]},
               "costs": {"observation_cost": 96, "measurement_cost": 72,
                         "compute_cost": 120, "python_execution_seconds": .25,
                         "model_calls": 3, "reported_tokens": 7},
               "events": [{"action": "finish", "arguments": {}, "feedback": {},
                           "usage": {"backend": "scripted_interface_test", "total_tokens": 7}}]}
    learned_path = tmp_path / "published.json"
    learned_path.write_text(json.dumps(learned))
    prior_results = []
    failed = None
    if getattr(request, "param", False):
        failed = {"status": "token_budget_exhausted",
                  "costs": {"observation_cost": 144, "measurement_cost": 144,
                            "compute_cost": 2048, "python_execution_seconds": .5,
                            "model_calls": 17, "reported_tokens": 1022484}}
        failed_path = tmp_path / "failed_learning.json"
        failed_path.write_text(json.dumps(failed))
        prior_results.append(failed_path)
    output = tmp_path / "retention"
    providers, evaluation_order = [], []

    class AcquireThenFail:
        def __init__(self):
            self.requests = []

        def next_action(self, request):
            self.requests.append(deepcopy(request))
            if len(self.requests) == 1:
                return {"action": "query", "arguments": {"actions": [
                    {"kind": "trajectory", "sequence": [{"u": .2}, {"u": -.3}],
                     "channels": ["meter"]}]},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 2}}
            return {"action": "provider_error", "arguments": {},
                    "error": {"status": 503, "message": "scripted interruption after paid acquisition"},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 3}}

    def provider_factory(path):
        provider = AcquireThenFail()
        providers.append(provider)
        return provider

    actual_evaluate = runner.evaluate_chain

    def evaluate_after_all_training(chain, tasks, **kwargs):
        assert len(providers) == 8
        assert all(len(provider.requests) == 2 for provider in providers)
        paths = list(output.glob("*/*/training.json"))
        assert len(paths) == 8
        assert all(json.loads(path.read_text())["evaluation"] is None for path in paths)
        evaluation_order.append(deepcopy(tasks))
        return actual_evaluate(chain, tasks, **kwargs)

    monkeypatch.setattr(runner, "donor_memory", donor)
    monkeypatch.setattr(runner, "evaluate_chain", evaluate_after_all_training)
    extra = {"prior_learning_results": prior_results} if prior_results else {}
    summary = runner.run_study(learned_path, output, provider_factory=provider_factory, workers=2, **extra)
    return {"summary": summary, "providers": providers, "original": original,
            "donor_calls": donor_calls, "evaluation_order": evaluation_order,
            "output": output, "learned": learned, "learned_path": learned_path,
            "prior_results": prior_results, "failed": failed}


def test_retention_changes_only_method_and_entry_with_equal_interfaces_and_budgets(scripted_study):
    study = scripted_study
    assert study["donor_calls"] == ["first", "second"]
    requests = [provider.requests[0] for provider in study["providers"]]
    assert len(requests) == 8
    assert all(request["knowledge_condition"] == "hegel" for request in requests)
    assert all(request["limits"] == requests[0]["limits"] for request in requests)
    assert all(request["task_interface"] == requests[0]["task_interface"] for request in requests)
    assert OPERATOR_INTERFACE in requests[0]["task_interface"]
    assert requests[0]["limits"]["observation_budget"] == 240
    assert requests[0]["limits"]["confirmation_rows"] == 48
    assert requests[0]["limits"]["compute_budget"] == 100000
    for original in study["original"].values():
        matching = [request["session_knowledge"] for request in requests
                    if request["session_knowledge"]["research_roles"] == original["research_roles"]]
        assert len(matching) == 4
        retained = [memory for memory in matching if METHOD_PATH in memory["files"]]
        deleted = [memory for memory in matching if METHOD_PATH not in memory["files"]]
        assert len(retained) == len(deleted) == 2
        assert retained[0] == retained[1] and deleted[0] == deleted[1]
        kept, erased = retained[0], deepcopy(retained[0])
        assert kept["files"][METHOD_PATH] == METHOD
        assert kept["lessons"] == deleted[0]["lessons"] == original["lessons"] + [LESSON]
        for path, source in original["files"].items():
            if path != "shared/knowledge.json":
                assert kept["files"][path] == deleted[0]["files"][path] == source
        erased["files"].pop(METHOD_PATH)
        erased["manifest"]["research_programs"] = [entry for entry in erased["manifest"]["research_programs"]
                                                     if entry["path"] != METHOD_PATH]
        manifest = json.loads(erased["files"]["shared/knowledge.json"])
        manifest["research_programs"] = [entry for entry in manifest["research_programs"]
                                        if entry["path"] != METHOD_PATH]
        assert manifest == json.loads(deleted[0]["files"]["shared/knowledge.json"])
        erased["files"]["shared/knowledge.json"] = deleted[0]["files"]["shared/knowledge.json"]
        assert erased == deleted[0]


def test_all_paid_failures_survive_to_summary_without_seed_refit_and_evaluation_is_last(scripted_study):
    study = scripted_study
    summary = study["summary"]
    assert summary["status"] == "completed"
    assert summary["evidence_kind"] == "interface_test_not_llm_performance"
    assert len(summary["cases"]) == 8
    assert all(case["status"] == "provider_failed" for case in summary["cases"])
    assert all(not case["confirmed_and_correct"] for case in summary["cases"])
    for condition in ("method_retained", "method_deleted"):
        report = summary["conditions"][condition]
        assert report["registered_tasks"] == report["attempted_tasks"] == 4
        assert report["statuses"] == ["provider_failed"] * 4
        assert report["confirmed_and_correct"] == report["false_confirmations"] == 0
        assert report["transfer_costs"]["observation_cost"] == 8
        assert report["transfer_costs"]["measurement_cost"] == 24
        assert report["transfer_costs"]["compute_cost"] == 0
        assert report["transfer_costs"]["reported_tokens"] == 20
        assert report["transfer_plus_paired_learning_costs"]["reported_tokens"] == 27
    assert len(study["evaluation_order"]) == 8
    for provider in study["providers"]:
        first, second = provider.requests
        assert first["observations"] == first["laboratory_observations"] == []
        assert first["events"] == []
        assert first["costs"]["observation_cost"] == first["costs"]["compute_cost"] == 0
        assert len(second["observations"]) == 2
        assert len(second["laboratory_observations"]) == 1
        assert second["costs"]["observation_cost"] == 2
        assert second["costs"]["measurement_cost"] == 6
        assert second["costs"]["compute_cost"] == 0
        assert [event["action"] for event in second["events"]] == ["query"]
        assert not second["events"][0]["feedback"].get("after_observation_results")
    results = [json.loads(path.read_text()) for path in study["output"].glob("*/*/result.json")]
    assert len(results) == 8
    for chain in results:
        task = chain["tasks"][0]
        assert task["status"] == "provider_failed"
        assert task["model"] is None and task["fresh_confirmation"] is None
        assert task["costs"]["observation_cost"] == 2
        assert task["costs"]["measurement_cost"] == 6
        assert task["costs"]["compute_cost"] == 0
        assert task["costs"]["reported_tokens"] == 5
        assert task["inherited_files_changed"] == []
        assert len(task["paid_observations"]) == 1
        assert chain["evaluation"][0]["status"] == "missing_model"
        assert [event["action"] for event in task["events"]] == ["query", "provider_error"]


@pytest.mark.parametrize("scripted_study", [True], indirect=True)
def test_failed_learning_and_successful_continuation_are_charged_once_per_arm(scripted_study):
    study = scripted_study
    summary = study["summary"]
    components = summary["learning_components"]
    assert [item["path"] for item in components] == [
        str(study["prior_results"][0]), str(study["learned_path"])]
    assert [item["status"] for item in components] == ["token_budget_exhausted", "completed"]
    assert components[0]["costs"] == study["failed"]["costs"]
    assert components[1]["costs"] == study["learned"]["costs"]
    common = summary["common_learning_costs"]
    assert common["model_calls"] == 20
    assert common["reported_tokens"] == 1022491
    for key, value in common.items():
        assert value == study["failed"]["costs"][key] + study["learned"]["costs"][key]
    for condition in ("method_retained", "method_deleted"):
        report = summary["conditions"][condition]
        for key, value in report["transfer_plus_paired_learning_costs"].items():
            assert value == report["transfer_costs"][key] + common[key]
        assert report["transfer_plus_paired_learning_costs"]["model_calls"] == 28
        assert report["transfer_plus_paired_learning_costs"]["reported_tokens"] == 1022511
