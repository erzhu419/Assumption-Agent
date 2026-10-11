#!/usr/bin/env python3
"""Matched finite-member, flat-family and online child-family response controls.

Reuse the three published ResponseLab diagnostics. All controllers receive the
same bases, paid prefix, public actions and eight additional response actions.
The child condition uses DiscoveryWorldAgent's real query/evaluate/return path;
the flat condition executes exactly the same analysis on its root question.
There is no model, new candidate, hidden child solver or learned-method claim.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.run_response_family_pilot import ResponseLab, ESTIMATORS, consequence_snapshot
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_response_constraints import query_response_family
from hegel_machine.cognition.world_response_consequences import assess_response_consequences
from hegel_machine.cognition.world_response_hypotheses import (
    _propose_fitted_responses, _append_joint_responses,
)


ARMS = ("representative", "flat_family", "recursive_family")
CASES = ("parameter_sufficient", "shared_feature_needed", "shared_assumption_false")
LIMITS = {"environment_steps": 11, "score_checkpoints_environment_steps": [],
          "max_model_calls": 0, "task_wall_seconds": 600,
          "provider_request_timeout_seconds": 5, "python_seconds_per_call": 60,
          "python_seconds_per_task": 480}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def member_snapshot(state, proposal):
    panel = state["consequence_panel"]
    signatures = sorted({tuple(h["lower"] <= p["frequency"] -
        h["predictions_by_crystal"][str(p["crystal"])] <= h["upper"]
        for p in panel) for h in proposal["hypotheses"]})
    return {"panel": deepcopy(panel), "possible_signatures": [list(s) for s in signatures],
            "families": {}, "family_scope": "finite_members_only_no_family_exclusion",
            "compute_cost": len(panel) * len(proposal["hypotheses"])}


def analyze(state, arm):
    """The same paid-state entry point is used online and in updater replay."""
    # This study holds the two declared candidate supports fixed. The current
    # researcher's additional support-deletion search is a different treatment.
    estimators = deepcopy(ESTIMATORS)
    path = "shared/response_analysis.py"
    proposal = _propose_fitted_responses(state, estimators, path)
    if proposal["status"] in {"no_compatible_response_hypotheses", "no_untried_supported_setting"}:
        proposal = _append_joint_responses(proposal, state, estimators, path)
    if arm != "representative":
        proposal = assess_response_consequences(state, estimators, proposal)
    snapshot = (member_snapshot(state, proposal) if arm == "representative"
                else consequence_snapshot(state))
    state = deepcopy(state)
    state["response_research"] = deepcopy(proposal)
    return {"state": state, "proposal": proposal, "snapshot": snapshot,
            "compute_cost": proposal["compute_cost"] + snapshot["compute_cost"]}


ANSWER_SOURCE = '''import json, sys
sys.path[:0] = ROOT_PATHS
from scripts.run_response_family_controls import analyze

def answer(public_spec, observations, memory):
    payload = json.load(open('response_input.json'))
    response = observations[-1]['observation']['response']
    state = payload['state']
    state['response_settings'].append(response)
    result = analyze(state, payload['arm'])
    # Supported here means the cited readout and its computed conditional
    # consequences were returned. It does not certify a family or target task.
    return {'status': 'supported', 'answer': result,
            'evidence': response['evidence']}
'''.replace("ROOT_PATHS", repr([str(ROOT / "src"), str(ROOT)]))


class ResponseWorldLab(ResponseLab):
    """Expose the already-published one-action instrument to the shared executor."""

    def __init__(self, case):
        super().__init__(case)
        self.environment_steps = 0

    def initial_state(self):
        state = super().initial_state()
        self.environment_steps = len(self.records)
        return state

    def public_observation(self):
        return {"response": deepcopy(self.records[-1]) if self.records else None,
                "taskProgress": [{"description": "Predict and test the reserved target panel",
                                  "completed": False}]}

    def capabilities(self):
        return {"actions": ["READ_RESPONSE"], "control_grid": {
            "lower": 0., "upper": 14., "step": 1., "origin": 0.}}

    def act(self, action):
        if action["action"] != "READ_RESPONSE":
            raise ValueError("Only the published response instrument is available")
        number, setting = action["arg1"], float(action["setting"])
        if number not in self.features or not 0 <= setting <= 14 or not setting.is_integer():
            raise ValueError("Action must be on the public control grid")
        self.measure(number, setting)
        self.environment_steps += 1
        return {"action": deepcopy(action), "feedback": {"success": True},
                "observation": self.public_observation()}

    def score(self):
        # The evaluator computes prediction completion separately. No target
        # truth or completion oracle enters the researcher's public record.
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


def prepare(case, arm, workspace):
    lab = ResponseWorldLab(case)
    state = lab.initial_state()
    agent = DiscoveryWorldAgent(None, workspace, arm="hegel", limits=LIMITS)
    agent.lab = lab
    # Preserve the original evidence numbering: features 0..4; initial responses
    # 5..7. The five feature rows are common prepaid input, itemized separately.
    agent.observations = [{"index": n - 1, "action": {"action": "FEATURES", "arg1": n},
        "feedback": {"success": True}, "observation": {"features": deepcopy(values)}}
        for n, values in lab.features.items()]
    for response in lab.records:
        agent.observations.append({"index": response["evidence"][0],
            "action": {"action": "READ_RESPONSE", "arg1": response["crystal"],
                       "setting": response["frequency"]},
            "feedback": {"success": True}, "observation": {"response": deepcopy(response)}})
    agent.costs["environment_steps"] = 3
    agent._question({"id": "task", "question": "Predict and test the fixed target panel"})
    (agent.workspace / "shared").mkdir(exist_ok=True)
    (agent.workspace / "shared/response_analysis.py").write_text(ANSWER_SOURCE)
    agent._save_state()
    return lab, agent, state


def consume_return(agent, question_id):
    """Consume the actual executor's answer, after its child dependency closes."""
    parent = next(q for q in agent.questions if q["id"] == "task")
    if agent._pending_dependencies(parent):
        raise RuntimeError("Parent cannot resume before the actual child answer returns")
    question = next(q for q in agent.questions if q["id"] == question_id)
    if question.get("status") != "supported" or "answer" not in question:
        raise RuntimeError("The paid-response analysis did not return a usable answer")
    return deepcopy(question["answer"])


def definite_predictions(snapshot, responses):
    seen = {(r["crystal"], r["frequency"]) for r in responses}
    possibilities = snapshot["possible_signatures"]
    return [{**point, "predicted_activated": next(iter(values))}
            for i, point in enumerate(snapshot["panel"])
            if (point["crystal"], point["frequency"]) not in seen and possibilities
            if len(values := {signature[i] for signature in possibilities}) == 1]


def checkpoint(step, result, confirmations, compute):
    snapshot, state = result["snapshot"], result["state"]
    confirmed = {(r["response"]["crystal"], r["response"]["frequency"])
                 for r in confirmations if r["correct"]}
    panel = {(p["crystal"], p["frequency"]) for p in snapshot["panel"]}
    return {"new_response_actions": step, "total_response_actions": step + 3,
            "research_compute_units": compute,
            "prediction_task_completed": confirmed == panel,
            "correct_confirmations": len(confirmed),
            "wrong_confirmations": sum(not r["correct"] for r in confirmations),
            "unmeasured_predictions": definite_predictions(snapshot, state["response_settings"]),
            "possible_signatures": deepcopy(snapshot["possible_signatures"]),
            "family_statuses": deepcopy(snapshot["families"]),
            "selected_number": result["proposal"].get("selected_number"),
            "selected_setting": result["proposal"].get("selected_setting")}


def add_observed_endpoints(row):
    """Separate finding an activation from explaining it before its test.

    These endpoints use only actual paid responses. Scoring previously frozen
    panel forecasts happens after the online run and never affects its policy.
    This function also supports rescoring saved runs without repeating research.
    """
    panel = {(p["crystal"], p["frequency"]) for p in row["final_snapshot"]["panel"]}
    evaluator = ResponseLab(row["case"])
    for point in row["common_spend"]:
        paid = row["final_responses"][:3 + point["actual_new_responses"]]
        positives = {r["crystal"] for r in paid if r["crystal"] in (4, 5) and r["activated"]}
        panel_positives = {(r["crystal"], r["frequency"]) for r in paid
                           if r["activated"] and (r["crystal"], r["frequency"]) in panel}
        point.update(targets_with_observed_positive=sorted(positives),
                     both_targets_observed_positive=positives == {4, 5},
                     reserved_panel_observed_positive=len(panel_positives))
        point["frozen_forecast_evaluation"] = []
        for forecast in point["unmeasured_predictions"]:
            actual = evaluator.measure(forecast["crystal"], forecast["frequency"])["activated"]
            point["frozen_forecast_evaluation"].append({**deepcopy(forecast), "actual": actual,
                "correct": forecast["predicted_activated"] == actual})
    row["observed_endpoints"] = {
        "both_targets_ever_activated": row["common_spend"][-1]["both_targets_observed_positive"],
        "first_both_targets_new_response_count": next((p["actual_new_responses"] for p in row["common_spend"]
            if p["both_targets_observed_positive"]), None),
        "reserved_panel_observed_positive": row["common_spend"][-1]["reserved_panel_observed_positive"],
        "scope": "Observed successful settings, independent of prior certainty; not simultaneous final physical activation"}
    return row


def heldout_predictions(case, state, arm, proposal):
    """Freeze predictions on the whole public target grid before scoring truth.

    Already purchased points are excluded. Evaluation computation is separate,
    does not purchase observations, and is never returned to the controller.
    """
    observed = {(r["crystal"], r["frequency"]) for r in state["response_settings"]}
    features = {r["crystal"]: {k: v["value"] for k, v in r["measurements"].items()}
                for kind in ("references", "targets") for r in state["research_table"][kind]}
    rows, cost = [], 0
    for number in state["target_numbers"]:
        for setting in range(15):
            if (number, float(setting)) in observed:
                continue
            if arm == "representative":
                possible = {h["lower"] <= setting - h["predictions_by_crystal"][str(number)]
                            <= h["upper"] for h in proposal["hypotheses"]}
                cost += len(proposal["hypotheses"])
            else:
                possible = set()
                for value in (False, True):
                    answers = [query_response_family(features, state["response_settings"], estimator,
                        queries=[{"crystal": number, "frequency": float(setting), "activated": value}])
                        for estimator in ESTIMATORS]
                    cost += sum(a["compute_cost"] for a in answers)
                    if any(a["status"] != "UNSAT_VERIFIED" for a in answers):
                        possible.add(value)
            rows.append({"crystal": number, "frequency": float(setting),
                         "prediction": next(iter(possible)) if len(possible) == 1 else None})
    # Fresh evaluator object is intentionally not passed to analyze/selectors.
    evaluator = ResponseLab(case)
    for row in rows:
        row["actual"] = evaluator.measure(row["crystal"], row["frequency"])["activated"]
    groups = {}
    for value in (False, True):
        selected = [r for r in rows if r["actual"] == value]
        groups["positive" if value else "negative"] = {
            "conditions": len(selected), "predicted": sum(r["prediction"] is not None for r in selected),
            "correct": sum(r["prediction"] == value for r in selected),
            "incorrect": sum(r["prediction"] is not None and r["prediction"] != value for r in selected)}
    return {"rows": rows, "groups": groups, "evaluator_compute_units": cost,
            "scope": "Unpurchased target settings from the full public grid; no selection feedback"}


def run_case(case, arm, output, *, max_new_responses=8):
    lab, agent, state = prepare(case, arm, output / case / arm / "agent")
    result = analyze(state, arm)
    compute = result["compute_cost"]
    steps, confirmations = [], []
    checkpoints = [checkpoint(0, result, confirmations, compute)]
    status = "response_budget_exhausted"
    for step in range(max_new_responses):
        proposal, state = result["proposal"], result["state"]
        if checkpoints[-1]["prediction_task_completed"]:
            status = "prediction_task_completed"
            break
        if proposal.get("selected_number") is None:
            status = proposal["status"]
            break
        number, setting = proposal["selected_number"], proposal["selected_setting"]
        predictions = definite_predictions(result["snapshot"], state["response_settings"])
        forecast = next((p for p in predictions if (p["crystal"], p["frequency"]) == (number, setting)), None)
        child = arm == "recursive_family" and number <= 3
        question_id = f"reference_{step + 1}" if child else "task"
        question = {"id": question_id, "question":
                    "Update the declared response families from this acquired reference outcome"
                    if child else "Predict and test the fixed target panel"}
        if child:
            question["parent_id"] = "task"
        write_json(agent.workspace / "response_input.json", {"state": state, "arm": arm})
        feedback = agent._execute("query", {"question": question,
            "actions": [{"action": "READ_RESPONSE", "arg1": number, "setting": setting}],
            "after_observation": [{"action": "evaluate", "arguments": {
                "path": "shared/response_analysis.py", "question_id": question_id}}]}, lab)
        if feedback.get("error") or feedback.get("after_observation_error"):
            raise RuntimeError(json.dumps(feedback))
        before_snapshot = result["snapshot"]
        result = consume_return(agent, question_id)
        compute += result["compute_cost"]
        response = result["state"]["response_settings"][-1]
        if forecast:
            confirmations.append({"forecast": forecast, "response": deepcopy(response),
                                  "correct": forecast["predicted_activated"] == response["activated"]})
        steps.append({"step": step + 1, "selected_role": proposal.get("family_research", {}).get("selected_role"),
            "response": deepcopy(response), "question_id": question_id,
            "child_call": child, "before_possible_signatures": before_snapshot["possible_signatures"],
            "returned_possible_signatures": result["snapshot"]["possible_signatures"],
            "parent_next_action": {"number": result["proposal"].get("selected_number"),
                                   "setting": result["proposal"].get("selected_setting")},
            "analysis_evidence": deepcopy(next(q for q in agent.questions if q["id"] == question_id)["evidence"]),
            "response_count_seen_by_parent": len(result["state"]["response_settings"])})
        checkpoints.append(checkpoint(step + 1, result, confirmations, compute))
    if checkpoints[-1]["prediction_task_completed"]:
        status = "prediction_task_completed"
    # A stopped arm contributes its last frozen state at all later common caps;
    # do not grant it new measurements or call carried-forward values new data.
    common = [{"cap_new_responses": cap, "actual_new_responses": min(cap, len(steps)),
               "state_carried_after_stop": cap > len(steps),
               **deepcopy(checkpoints[min(cap, len(steps))])}
              for cap in range(max_new_responses + 1)]
    row = {"case": case, "arm": arm, "status": status,
        "prediction_task_completed": checkpoints[-1]["prediction_task_completed"],
        "trace": steps, "confirmations": confirmations, "common_spend": common,
        "questions": deepcopy(agent.questions), "child_answers_returned": agent.child_answers_returned,
        "final_snapshot": result["snapshot"], "final_responses": result["state"]["response_settings"],
        "heldout": heldout_predictions(case, result["state"], arm, result["proposal"]),
        "costs": {"initial_feature_rows": 5, "initial_response_actions": 3,
                  "new_response_actions": len(steps), "world_actions": lab.environment_steps,
                  "model_calls": 0, "research_compute_units": compute,
                  "python_execution_seconds": agent.costs["python_execution_seconds"]},
        "final_research_obligation": result["proposal"].get("family_research", {}).get("research_obligation")}
    add_observed_endpoints(row)
    write_json(output / case / arm / "result.json", row)
    return row


def replay_log(case, source):
    """Update all three representations on B's fixed actual log; no acquisition."""
    all_arms = {}
    for arm in ARMS:
        state = ResponseLab(case).initial_state()
        snapshots, compute = [], 0
        for count in range(len(source["trace"]) + 1):
            result = analyze(state, arm)
            compute += result["compute_cost"]
            snapshots.append({"new_responses": count, "snapshot": result["snapshot"],
                              "compute_units_cumulative": compute})
            state = result["state"]
            if count < len(source["trace"]):
                state["response_settings"].append(deepcopy(source["trace"][count]["response"]))
        all_arms[arm] = snapshots
    return {"case": case, "source_arm": "flat_family", "arms": all_arms,
            "scope": "Same-log update comparison only; selectors do not acquire or affect the log"}


def summarize(rows):
    return {arm: {"conditions": len(selected := [r for r in rows if r["arm"] == arm]),
        "prediction_tasks_completed": sum(r["prediction_task_completed"] for r in selected),
        "both_targets_ever_activated": sum(r["observed_endpoints"]["both_targets_ever_activated"] for r in selected),
        "both_reserved_panel_points_observed_positive": sum(
            r["observed_endpoints"]["reserved_panel_observed_positive"] == 2 for r in selected),
        "new_response_actions": sum(r["costs"]["new_response_actions"] for r in selected),
        "research_compute_units": sum(r["costs"]["research_compute_units"] for r in selected),
        "wrong_confirmations": sum(not c["correct"] for r in selected for c in r["confirmations"]),
        "child_answers_returned": sum(r["child_answers_returned"] for r in selected),
        "heldout": {kind: {metric: sum(r["heldout"]["groups"][kind][metric] for r in selected)
                            for metric in ("conditions", "predicted", "correct", "incorrect")}
                    for kind in ("positive", "negative")}}
        for arm in ARMS}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"kind": "fixed_candidate_response_three_arm_mechanism_control",
        "cases": CASES, "arms": ARMS, "estimators": ESTIMATORS, "max_new_responses": 8,
        "shared_limits": LIMITS, "shared_paid_input": "5 feature rows and 3 actual response actions",
        "candidate_generation": "Frozen original two declarations, identical in all arms",
        "analysis": "A finite response members; B/C exact same family analysis and action selection",
        "recursive_intervention": "Reference query opens a child before acquisition; existing evaluate returns analysis to parent",
        "checkpoints": list(range(9)), "replay_source": "All B logs, one per existing condition",
        "endpoints": {"primary_prediction": "Both original reserved-panel predictions definite before their actual tests and correct",
            "observed_activation": "Both target objects had an actually positive paid setting, with no certainty requirement",
            "stopping": "Same rule: stop on primary prediction confirmation, absent action, or eight-new-response cap; common-spend carries stopped state without extra data"},
        "unit": "Three previously published diagnostic conditions, not independent random trials",
        "model_calls": 0, "scope": "Frozen numerical controllers; no autonomous LLM or method-learning claim"})
    rows = []
    for case in CASES:
        for arm in ARMS:
            row = run_case(case, arm, output)
            rows.append(row)
            write_json(output / "summary.json", summarize(rows))
            print(json.dumps({"case": case, "arm": arm, "status": row["status"], "costs": row["costs"]}), flush=True)
    replays = [replay_log(case, next(r for r in rows if r["case"] == case and r["arm"] == "flat_family"))
               for case in CASES]
    write_json(output / "same_log_replay.json", replays)
    equality = {case: {"actual_actions": (b := next(r for r in rows if r["case"] == case and r["arm"] == "flat_family"))["final_responses"] ==
                      (c := next(r for r in rows if r["case"] == case and r["arm"] == "recursive_family"))["final_responses"],
        "common_spend": b["common_spend"] == c["common_spend"],
        "final_predictions": b["heldout"] == c["heldout"],
        "updater_replay": next(r for r in replays if r["case"] == case)["arms"]["flat_family"] ==
                          next(r for r in replays if r["case"] == case)["arms"]["recursive_family"]}
        for case in CASES}
    write_json(output / "results.json", {"summary": summarize(rows), "family_arm_equality": equality,
                                         "cases": rows})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
