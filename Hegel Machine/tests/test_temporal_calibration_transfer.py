"""Calibration-plan and actual-runtime checks; scripted providers are not LLM evidence."""

from copy import deepcopy
import json

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_temporal_calibration_transfer as transfer
from scripts import run_temporal_paired_retention as retention
from test_temporal_paired_retention import LESSON, METHOD, original_memory


def test_live_cases_are_preselected_frozen_cases_with_independent_changes():
    plan = transfer.protocol()
    cases, live = plan["frozen_cases"], plan["live_protocol"]["cases"]
    assert len(cases) == 28
    assert live == [cases[index] for index in (0, 1, 2, 3, 4, 7)]
    assert {case["change"] for case in live} == {
        "stable", "response_only", "instrument_only", "both"}
    known = {value[0] for value in transfer.CALIBRATIONS.values()}
    for case in cases:
        old_seed = transfer.CALIBRATIONS[case["prior"]][0]
        task = case["task"]
        if case["change"] in ("stable", "response_only"):
            assert task["apparatus_seed"] == old_seed
        else:
            assert task["apparatus_seed"] not in known
        if case["change"] == "instrument_only":
            assert task["response_seed"] == old_seed
        elif case["change"] == "response_only":
            assert task["response_seed"] not in known
        else:
            assert "response_seed" not in task
    assert len({case["task"]["apparatus_seed"] for case in cases
                if case["instrument_changed"]}) == 12
    # The two protocols are independent snapshots, not aliases that later runs can mutate.
    live[0]["task"]["apparatus_seed"] = -1
    assert cases[0]["task"]["apparatus_seed"] in known


def test_custom_plan_reaches_paid_lab_and_keeps_comparison_paired(tmp_path, monkeypatch):
    plan = transfer.protocol()["live_protocol"]
    # Exercise every preselected task: a dropped response seed would silently
    # convert the response-only cases into stable cases.
    original_plan = deepcopy(plan)
    originals = {"first": original_memory(1.25, "shared/meter.json"),
                 "second": original_memory(2.5, "shared/current_instrument.json")}
    monkeypatch.setattr(retention, "donor_memory", lambda path, position, instrument:
                        deepcopy(originals["first" if position == 1 else "second"]))
    learned = {"status": "completed", "method_path": METHOD_PATH,
               "method_source": METHOD, "method_entry": {"id": "method", "path": METHOD_PATH},
               "outgoing_knowledge": {"lessons": [LESSON]}, "costs": {}}
    learned_path = tmp_path / "learned.json"
    learned_path.write_text(json.dumps(learned))
    output = tmp_path / "live"
    providers = {}
    action = {"kind": "trajectory", "sequence": [{"u": .2}, {"u": -.3}],
              "channels": ["meter"]}

    class AcquireAndStop:
        def __init__(self, path):
            self.requests = []
            providers[(path.parent.name, path.name)] = self

        def next_action(self, request):
            self.requests.append(deepcopy(request))
            if len(self.requests) == 1:
                return {"action": "query", "arguments": {"actions": [action]},
                        "usage": {"backend": "scripted_interface_test", "total_tokens": 2}}
            return {"action": "provider_error", "arguments": {},
                    "error": {"status": 503, "message": "end scripted check"},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 3}}

    actual_evaluate = retention.evaluate_chain
    evaluated = []

    def evaluate_after_training(chain, tasks, **kwargs):
        assert len(providers) == 12
        assert all(len(provider.requests) == 2 for provider in providers.values())
        paths = list(output.glob("*/*/training.json"))
        assert len(paths) == 12
        assert all(json.loads(path.read_text())["evaluation"] is None for path in paths)
        evaluated.append((chain["case_id"], chain["condition"], deepcopy(tasks)))
        return actual_evaluate(chain, tasks, **kwargs)

    monkeypatch.setattr(retention, "evaluate_chain", evaluate_after_training)
    summary = retention.run_study(learned_path, output, provider_factory=AcquireAndStop,
                                  workers=2, study_protocol=plan)
    assert plan == original_plan
    assert summary["study"] == plan["study"]
    assert summary["evidence_kind"] == "interface_test_not_llm_performance"
    assert len(summary["cases"]) == len(evaluated) == 12
    for case in plan["cases"]:
        kept, deleted = [providers[(case["case_id"], condition)].requests
                         for condition in ("method_retained", "method_deleted")]
        assert kept[0]["task_interface"] == deleted[0]["task_interface"]
        assert kept[0]["limits"] == deleted[0]["limits"] == plan["transfer_limits"]
        assert kept[0]["observations"] == deleted[0]["observations"] == []
        assert kept[0]["events"] == deleted[0]["events"] == []
        kept_memory, deleted_memory = kept[0]["session_knowledge"], deleted[0]["session_knowledge"]
        assert kept_memory["files"][METHOD_PATH] == METHOD
        assert METHOD_PATH not in deleted_memory["files"]
        original = originals[case["prior"]]
        for path, source in original["files"].items():
            if path != "shared/knowledge.json":
                assert kept_memory["files"][path] == deleted_memory["files"][path] == source
        assert kept_memory["research_roles"] == deleted_memory["research_roles"] == original["research_roles"]
        assert kept_memory["lessons"] == deleted_memory["lessons"]
        intended = transfer.laboratory(case["task"]).execute(action)
        assert kept[1]["laboratory_observations"] == deleted[1]["laboratory_observations"] == [intended]
        assert kept[1]["costs"]["observation_cost"] == deleted[1]["costs"]["observation_cost"] == 2
        pair = [tasks for case_id, _, tasks in evaluated if case_id == case["case_id"]]
        assert pair == [[case["task"]], [case["task"]]]
        if case["change"] == "response_only":
            coupled = deepcopy(case["task"])
            coupled.pop("response_seed")
            old = transfer.laboratory(coupled).execute(action)
            assert old["measurements"] == intended["measurements"]
            assert old["rows"] != intended["rows"]
