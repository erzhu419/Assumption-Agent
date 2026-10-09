"""Isolate offer rendering from tool access and count adoption without inferred calls."""

from copy import deepcopy
import json

from scripts.run_python_procedure_offer_study import (
    offer_evidence, prepare_offer_target, protocol, summarize,
)
from scripts.run_python_procedure_study import PROGRAM_PATH, target_jobs


def test_control_has_same_program_tool_and_atomic_advice_but_no_rendered_offer(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "model.py").write_text("def predict(row):\n    return 0.0\n")
    candidate = {"candidate_id": "candidate_000", "training_rows": 72,
        "snapshot_workspace": str(archive), "inner_model": "model.py",
        "model": "shared/confirmed_models/candidate_000/_evaluated_model/model.py"}
    source = {"case": {"apparatus_seed": 971107010, "changed_response_seed": 981107010,
                       "seed": 991107010}, "candidate": candidate, "costs": {}}
    plan = protocol([source])
    payloads = {job["condition"]: prepare_offer_target(job, plan) for job in target_jobs([source], tmp_path, plan)
                if job["kind"] == "response_parameter_change"}
    results = {}
    for condition, payload in payloads.items():
        agent, lab = payload["agent"], payload["laboratory"]
        feedback = agent._execute("evaluate", {"model": candidate["model"]}, lab)
        agent.events.append({"action": "evaluate", "arguments": {"model": candidate["model"]},
                             "feedback": deepcopy(feedback)})
        agent._prepare_program_state()
        results[condition] = feedback
    visible, hidden = (payloads[name]["agent"] for name in ("feedback_offer", "interface_only"))
    assert visible.common_interface == hidden.common_interface
    assert visible.files[PROGRAM_PATH] == hidden.files[PROGRAM_PATH]
    assert visible.observations == hidden.observations
    assert all(visible.costs[key] == hidden.costs[key] for key in
               ("observation_cost", "measurement_cost", "compute_cost", "model_calls", "reported_tokens"))
    shown = results["feedback_offer"]["research_advice"]
    withheld = results["interface_only"]["research_advice"]
    assert shown["next_action"] == withheld["next_action"]
    assert shown["procedure_action"]["arguments"]["operator"] == "procedure"
    assert "procedure_action" not in withheld
    hidden_state = json.loads((hidden.workspace / "research_state.json").read_text())
    rendered = json.dumps({"events": hidden.events, "operators": hidden.operator_results, "state": hidden_state})
    assert '"procedure_action"' not in rendered and '"available_procedure_action"' not in rendered
    # Withholding the recommendation must not disable either agent's real tool.
    for payload in payloads.values():
        agent = payload["agent"]
        agent._save("stop.py", "def select_actions(spec, observations, memory):\n    return {'stop': 'done'}\n")
        feedback = agent._execute("run", {"operator": "procedure", "path": "stop.py", "max_steps": 1},
                                  payload["laboratory"])
        assert feedback["status"] == "method_stopped" and len(feedback["steps"]) == 1
        assert agent.costs["model_calls"] == 0


def test_offer_matching_internal_success_and_usage_remain_separate():
    option = {"action": "run", "arguments": {"operator": "procedure", "path": PROGRAM_PATH,
                                               "max_steps": 12, "confirm": True}}
    events = [
        {"action": "evaluate", "feedback": {"research_advice": {"procedure_action": option}}},
        {**deepcopy(option), "feedback": {"status": "confirmation_failed", "fresh_confirmation": {"passed": False},
                                          "steps": [{"research_advice": {"procedure_action": option}}, {}]}},
        {"action": "run", "arguments": {"operator": "research_choice"},
         "feedback": {"available_procedure_action": option}},
        {"action": "run", "arguments": {**option["arguments"], "confirm": False},
         "feedback": {"status": "confirmation_not_authorized", "steps": [{}]}},
        {**deepcopy(option), "feedback": {"status": "completed", "fresh_confirmation": {"passed": True},
                                          "steps": [{}, {}, {}]}},
        {"action": "evaluate", "feedback": {"available_procedure_action": option}},
    ]
    evidence = offer_evidence(events)
    assert evidence["visible_offer_events"] == 3 and evidence["offers_with_next_event"] == 2
    assert evidence["next_action_matches"] == 1
    assert evidence["procedure_invocations"] == 3 and evidence["procedure_internal_steps"] == 6
    assert evidence["procedures_completed_with_fresh_confirmation"] == 1
    assert evidence["offers"][-1]["next_action_matches"] is False
    sources = [{"case": {"apparatus_seed": 1, "seed": 2, "changed_response_seed": 3},
                "candidate": {"candidate_id": "actual_source"}, "costs": {"model_calls": 8}}]
    result = {"condition": "feedback_offer", "kind": "recurrent_instrument_change", "status": "provider_failed",
        "confirmed_and_correct": False, "false_confirmation": False,
        "costs": {"model_calls": 7, "reported_tokens": None, "observation_cost": 96},
        "known_tokens": 120, "input_tokens": 110, "cached_input_tokens": 60, "missing_usage_calls": 1,
        "procedure_requests": 3, "method_steps": 6, "operator_calls": {"refit": 1}, "procedure_offer": evidence}
    summary = summarize(sources, [result], protocol(sources))
    actual = summary["actual_new_targets"]
    assert actual["costs"]["model_calls"] == 7 and actual["costs"]["reported_tokens"] is None
    assert actual["missing_usage_calls"] == 1 and actual["confirmed_and_correct"] == 0
    assert actual["procedure_offer"]["procedures_completed_with_fresh_confirmation"] == 1
    assert summary["conditions"]["interface_only"]["procedure_offer"]["procedure_invocations"] == 0
