"""Read completed short root01 paid trajectories; no model execution."""

import gzip
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def describe(base, version):
    source = base / "seen_regression/runs/new_cumulative/root01/checkpoint.json.gz"
    state = json.loads(gzip.decompress(source.read_bytes()))
    assert state["status"] == "completed"
    final = state["results"][1]["research"]
    visits = [visit for visit in state["revisits"] if visit["task_position"] == 1]
    entries = []
    for index, visit in enumerate(visits):
        before = visit["previous"]
        after = visits[index + 1]["previous"] if index + 1 < len(visits) else final
        actions = after["actions"][len(before["actions"]):]
        experiment = visit["candidate"].get("instrument_condition_experiment", {})
        decision = experiment.get("decision", {})
        pairs = [action for action in actions
                 if action["action"] == "compare_method_followup_on_fresh_response"]
        first_costs = ({**pairs[0]["stage_costs"], "compute_cost":
            pairs[0]["stage_costs"]["compute_cost"] - experiment.get("screening_compute_cost", 0)}
            if pairs else None)
        outcomes = []
        for action in actions:
            if action.get("research_return"):
                outcomes.append(action["research_return"])
            elif action["action"] == "allocate_condition_confirmation":
                outcomes.extend(action["outcomes"])
        entries.append({"visit": index + 1, "basis": visit["basis"],
            "paid_screening_compute_cost": experiment.get("screening_compute_cost"),
            "whole_visit_costs": visit["visit_costs"],
            "first_fresh_pair_execution_costs": first_costs,
            "after_first_pair_costs": ({key: visit["visit_costs"][key] - first_costs[key]
                for key in RESOURCES} if first_costs else None),
            "first_pair_expected_costs": decision.get("expected_cycle_costs"),
            "legacy_revision_compute_estimate": decision.get("estimated_remaining_compute_cost")
                if version == "0.164" else None,
            "cycle_forecast": experiment.get("cycle_cost_forecast"),
            "investment_value": decision.get("expected_marginal_value"),
            "investment_resource_share": decision.get("resource_share"),
            "expected_information_fraction": decision.get("expected_information_fraction"),
            "program_changed": before["program"] != after["program"],
            "fresh_loss_pairs": [[pair["before_loss"], pair["after_loss"]] for pair in pairs],
            "information_batches": len(outcomes),
            "information_events": sum(bool(outcome.get("information_event")) for outcome in outcomes),
            "measured_information_gain": sum(outcome.get("marginal_information_gain", 0.) for outcome in outcomes),
            "response_counterexamples": sum(bool(outcome.get("observed_response_counterexample")) for outcome in outcomes),
            "confirmation_allocations": [{key: action.get(key) for key in
                ("planned_rows", "acquired_rows", "stop_reason")}
                for action in actions if action["action"] == "allocate_condition_confirmation"]})
    screens = [event for event in state["scheduler_events"]
               if event.get("task_position") == 1 and event["action"] == "screen_instrument_condition_revisit"]
    uncertainty = final.get("condition_uncertainty") or {}
    return {"version": version, "source": str(source.relative_to(BASE.parent.parent)),
        "first_visit_costs": visits[0]["previous"]["costs"] if visits else final["costs"],
        "final_task_costs": final["costs"], "status": final["status"],
        "revisits": len(visits), "revisit_costs": {
            key: sum(visit["visit_costs"][key] for visit in visits) for key in RESOURCES},
        "screening_compute_cost": sum(event["compute_cost"] for event in screens),
        "instrument_screens": len(screens),
        "final_point_loss": uncertainty.get("original_target_loss"),
        "final_prediction_ambiguity": uncertainty.get("expected_prediction_ambiguity"),
        "final_intervals": [condition.get("threshold_intervals") for condition in uncertainty.get("conditions", [])],
        "visits": entries}


def main():
    reports = [describe(BASE.parent / name, version) for version, name in (
        ("0.164", "shared_condition_research_20261003"),
        ("0.165", "joint_mismatch_investment_20261003"),
        ("0.166", "cold_cycle_investment_20261003"))]
    result = {"scope": "Completed paid short root01 position1 trajectories only; no new training, repricing or terminal evaluation.",
        "versions": reports,
        "cost_semantics": "First fresh-pair stage costs exclude the candidate's already paid screening fee. Remaining whole-visit costs are shown separately. The legacy 0.164 compute estimate prices revision, not the complete paired cycle.",
        "limitations": "Whole-chain budgets and purchased stochastic responses differ. These traces do not by themselves establish each module's causal contribution or terminal accuracy."}
    (BASE / "repeated_investment_paid_diagnosis.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
