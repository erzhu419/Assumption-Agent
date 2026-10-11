"""Keep actual method declarations when supplying an anonymous failed trial."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_temporal_counterexample_learning import supply_counterexample


def failed_trial():
    """Trial 016's recorded event layout and errors, with synthetic row values.

    The source trial had one write followed by five failed submissions of that
    same file. The test does not import an untracked artifact or its simulator.
    """
    model = "revised_model.py"
    source = "def predict(row):\n    return float(row['features']['u'])\n"
    write = {"action": "write", "arguments": {"path": model, "code": source}}
    records = [{"row_indices": list(range(start, start + 24)), "measurements": [],
                "rows": [{"features": {"u": 0.0}, "history": [], "target": 0.0}
                         for _ in range(24)]}
               for start in range(0, 552, 24)]
    actions = [{"kind": "trajectory", "channels": [],
                "sequence": [{"u": 0.0} for _ in range(24)]} for _ in range(3)]
    events = [{"step": -1, "action": "query", "arguments": {"actions": actions},
               "feedback": {"record_indices": [0, 1, 2], "observation_cost": 72}},
              {"step": 0, "action": "query", "arguments": {"path": METHOD_PATH},
               "feedback": {"written": model, "program_action": write}}]
    methods = [{"step": 0, "path": METHOD_PATH, "proposal": write,
                "feedback": deepcopy(events[1]["feedback"])}]
    errors = (2/96, 1/96, 2/96, 4/96, 1/96)
    for index, mse in enumerate(errors, start=1):
        start = 72 + (index - 1) * 96
        feedback = {"method_submission": model, "returncode": 0,
            "error": "fresh confirmation failed; new counterexamples are now paid evidence",
            "fresh_confirmation": {"passed": False, "mse": mse, "rows": 96,
                                   "observation_indices": list(range(start, start + 96))}}
        events.append({"step": index, "action": "query", "arguments": {"path": METHOD_PATH},
                       "feedback": feedback})
        methods.append({"step": index, "path": METHOD_PATH,
                        "proposal": {"model": model, "lesson": "The recorded candidate is ready for confirmation."},
                        "feedback": deepcopy(feedback)})
    return {"status": "confirmation_budget_exhausted",
            "costs": {"observation_cost": 552, "measurement_cost": 0, "compute_cost": 1800,
                      "python_execution_seconds": 1.785474717617035, "model_calls": 0, "reported_tokens": 0},
            "paid_observations": records, "events": events, "method_results": methods,
            "method_path": METHOD_PATH, "method_source": "# the actual executed method snapshot\n",
            "outgoing_knowledge": {"files": {METHOD_PATH: "# an outdated inherited copy\n"}},
            "hidden_settings": {"mechanism": "PRIVATE_MECHANISM_MARKER"},
            "terminal": {"mse": "PRIVATE_TERMINAL_MARKER"}}


@pytest.mark.parametrize("input_form", ["dict", "single_result_file", "legacy_tasks_file"])
def test_single_task_and_legacy_counterexamples_preserve_actual_proposals(tmp_path, input_form):
    prior = failed_trial()
    if input_form == "dict":
        # The current authored method is 4,169 characters; source must remain
        # readable without hiding the paid failure chronology behind a preview.
        prior["method_source"] = "# the actual executed method snapshot\n" * 160
    before = deepcopy(prior)
    supplied = prior
    if input_form != "dict":
        supplied = tmp_path / "source.json"
        payload = {"tasks": [prior]} if input_form == "legacy_tasks_file" else prior
        supplied.write_text(json.dumps(payload))
    workspace = tmp_path / "author"
    workspace.mkdir()
    learner = SimpleNamespace(workspace=workspace, files={}, common_interface="")

    assert supply_counterexample(learner, supplied, case_label="case_005") == prior["costs"]
    assert prior == before
    ledger = json.loads(learner.files["counterexample.json"])
    assert ledger["practice_case"] == "case_005"
    executed = ledger["executed_method"]
    if input_form == "dict":
        assert "source" not in executed
        assert (workspace / executed["source_file"]).read_text() == prior["method_source"]
    else:
        assert executed["source"] == prior["method_source"]
    assert len(learner.files["counterexample.json"]) < 12000
    assert len(ledger["steps"]) == 7
    assert "proposal" not in ledger["steps"][0]  # The common prefix is not authored by the method.
    for step, recorded in zip(ledger["steps"][1:], prior["method_results"]):
        assert step["action"] == "query"
        assert step["arguments"] == {"path": METHOD_PATH}
        assert step["proposal"] == recorded["proposal"]
    assert ledger["steps"][1]["feedback"]["written"] == "revised_model.py"
    assert [step["proposal"]["model"] for step in ledger["steps"][2:]] == ["revised_model.py"] * 5
    confirmations = [step["feedback"]["fresh_confirmation"] for step in ledger["steps"][2:]]
    assert [item["row_interval"] for item in confirmations] == [
        [72, 167], [168, 263], [264, 359], [360, 455], [456, 551]]
    assert all(item["passed"] is False and item["rows"] == 96 for item in confirmations)
    assert ledger["paid_records"][-1]["row_interval"] == [528, 551]

    full_trace = json.loads((workspace / "counterexample_trace.json").read_text())
    reference = ledger["steps"][0]["arguments"]["actions"]
    assert reference == {"full_content_file": "counterexample_trace.json",
                         "json_path": [0, "arguments", "actions"], "count": 3}
    acquired_actions = full_trace
    for key in reference["json_path"]:
        acquired_actions = acquired_actions[key]
    assert acquired_actions == prior["events"][0]["arguments"]["actions"]
    assert [event["proposal"] for event in full_trace[1:]] == [
        entry["proposal"] for entry in prior["method_results"]]
    assert json.loads((workspace / "counterexample_observations.json").read_text()) == prior["paid_observations"]
    all_supplied = "".join(path.read_text() for path in workspace.glob("counterexample*.json"))
    assert "PRIVATE_MECHANISM_MARKER" not in all_supplied
    assert "PRIVATE_TERMINAL_MARKER" not in all_supplied


def test_query_declaration_is_joined_only_to_the_same_recorded_invocation(tmp_path):
    prior = failed_trial()
    # A same-index call of another file is not evidence about this invocation.
    prior["method_results"][1]["path"] = "another_method.py"
    workspace = tmp_path / "author"
    workspace.mkdir()
    learner = SimpleNamespace(workspace=workspace, files={}, common_interface="")
    supply_counterexample(learner, prior)
    steps = json.loads(learner.files["counterexample.json"])["steps"]
    assert "proposal" not in steps[2]
    assert steps[2]["feedback"]["method_submission"] == "revised_model.py"
    assert steps[3]["proposal"] == prior["method_results"][2]["proposal"]
