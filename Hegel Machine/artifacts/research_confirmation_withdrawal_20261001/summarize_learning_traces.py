"""Read final learning checkpoints once; no terminal observations or learning."""

from collections import Counter
import gzip
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def collect(cohort, arm):
    protocol = json.loads((BASE / cohort / "protocol.json").read_text())
    result = {"cohort": cohort, "arm": arm, "roots": 0, "tasks": 0,
        "learning_costs": dict.fromkeys(COSTS, 0), "actions": Counter(),
        "paid_refits": [], "zero_forecast_measured_gain_visits": [],
        "condition_outcomes": [], "method_visits": 0}
    tolerance = protocol["config"]["minimum_improvement"]
    for root in protocol["roots"]:
        path = BASE / cohort / "runs" / arm / root / "checkpoint.json.gz"
        with gzip.open(path, "rt") as stream:
            state = json.load(stream)
        if not state["study"]["learning_complete"]:
            raise ValueError(f"Learning is unfinished: {cohort}/{arm}/{root}")
        result["roots"] += 1
        for key in COSTS:
            result["learning_costs"][key] += state["costs"][key]
        for item in state["results"]:
            position, report = item["task_position"], item["research"]
            task, actions = state["tasks"][position], report["actions"]
            case = {"root": root, "position": position,
                    "stage": protocol["stages"][position], "family": task.get("family", "condition")}
            result["tasks"] += 1
            result["actions"].update(event["action"] for event in actions)
            for index, event in enumerate(actions):
                if event["action"] == "refit_paid_response_pending_confirmation":
                    result["paid_refits"].append({**case, "event_index": index, **event,
                        "task_final_status": report["status"],
                        "task_final_confirmation_loss": report["confirmation_loss"]})
            if case["family"] == "condition":
                result["condition_outcomes"].append({**case, "status": report["status"],
                    "selected_kind": report["selected_kind"],
                    "selection_loss": report.get("selection_loss"),
                    "confirmation_loss": report["confirmation_loss"],
                    "condition_status": (report.get("condition_uncertainty") or {}).get("status"),
                    "costs": report["costs"],
                    "last_actions": [event["action"] for event in actions[-8:]]})
        for index, visit in enumerate(state["revisits"]):
            candidate = visit.get("candidate") or {}
            opportunity = candidate.get("method_opportunity") or {}
            if not opportunity or not candidate.get("method_experiment"):
                continue
            result["method_visits"] += 1
            gain = opportunity["paid_counterexample_loss"] - opportunity["candidate_paid_loss"]
            if opportunity["predicted_separation"] != 0 or gain <= tolerance:
                continue
            following = next((item for item in state["revisits"][index + 1:]
                if item["task_position"] == visit["task_position"]), None)
            after = following["previous"] if following else state["results"][visit["task_position"]]["research"]
            actions = after["actions"][len(visit["previous"]["actions"]):]
            execution = [event for event in actions
                         if event["action"] == "execute_successor_method_experiment"]
            paired = [event for event in actions
                      if event["action"] == "compare_successor_method_on_fresh_response"]
            validations = [event for event in actions if event["action"] == "validate_continued_program"]
            result["zero_forecast_measured_gain_visits"].append({"root": root,
                "position": visit["task_position"], "source_position": visit["source_task_position"],
                "paid_gain": gain, "executed": any(event["observation_rows"] > 0 for event in execution),
                "experiment_rows": sum(event["observation_rows"] for event in execution),
                "fresh_response_comparisons": [{key: event[key] for key in
                    ("probe_id", "observation_rows", "before_loss", "after_loss")} for event in paired],
                "fresh_improvements": sum(event["after_loss"] + tolerance < event["before_loss"] for event in paired),
                "confirmed_point_fits": sum(bool(event.get("current_fit_confirmed")) for event in validations),
                "accepted_complete_programs": sum(bool(event["accepted"]) for event in validations),
                "visit_costs": visit["visit_costs"], "screening_compute_cost": visit.get("screening_compute_cost", 0),
                "status_after_visit": after["status"]})
    refits = result["paid_refits"]
    zero = result["zero_forecast_measured_gain_visits"]
    result["counts"] = {"paid_refit_events": len(refits),
        "working_prediction_adoptions": sum(event["accepted_as_working_prediction"] for event in refits),
        "recorded_refit_direct_costs": {key: sum(event.get(key, 0) for event in refits) for key in COSTS},
        "zero_forecast_measured_gain_visits": len(zero),
        "zero_forecast_executions": sum(event["executed"] for event in zero),
        "zero_forecast_fresh_comparisons": sum(len(event["fresh_response_comparisons"]) for event in zero),
        "zero_forecast_fresh_improvements": sum(event["fresh_improvements"] for event in zero),
        "zero_forecast_confirmed_point_fits": sum(event["confirmed_point_fits"] for event in zero),
        "zero_forecast_accepted_complete_programs": sum(event["accepted_complete_programs"] for event in zero)}
    result["interpretation"] = (
        "Paid-refit before/after losses reuse acquired data and are not independent confirmations. "
        "Task final status may follow later research and is not attributed to that local refit. "
        "Fresh method response comparisons, point-fit confirmation and complete-program acceptance are separate. "
        "Direct refit event costs are not an arm-level causal cost difference. No terminal or long-run benefit is inferred here.")
    output = BASE / cohort / f"learning_traces_{arm}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {"cohort": cohort, "arm": arm, "roots": result["roots"], "tasks": result["tasks"],
            "method_visits": result["method_visits"], **result["counts"], "output": str(output)}


if __name__ == "__main__":
    print(json.dumps(collect(*sys.argv[1:]), indent=2))
