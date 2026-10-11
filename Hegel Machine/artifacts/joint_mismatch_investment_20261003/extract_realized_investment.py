"""Read completed paid checkpoints only; no acquisition or terminal evaluation."""

from collections import Counter
import gzip
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
OLD = BASE.parent / "shared_condition_research_20261003"
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
DECISION_FIELDS = ("buy", "reason", "reopened", "expected_marginal_value", "resource_share",
    "expected_information_fraction", "probability_of_new_information", "information_yield",
    "sampling_reliability", "predicted_response_risk_reduction", "current_response_risk_total",
    "expected_cycle_costs", "estimated_remaining_compute_cost", "quoted_compute_cost",
    "compute_spent_before_decision", "empirical_marginal_prediction_gain")


def question(state):
    if not state:
        return None
    return {"status": state.get("status"), "point_loss": state.get("original_target_loss"),
        "response_ambiguity": state.get("expected_prediction_ambiguity"),
        "intervals": [item.get("threshold_intervals", []) for item in state.get("conditions", [])]}


def information(actions):
    outcomes = []
    for action in actions:
        if action["action"] == "revise_competing_observed_conditions" and action.get("research_return"):
            outcomes.append(action["research_return"])
        elif action["action"] == "allocate_condition_confirmation":
            outcomes.extend(action["outcomes"])
    return {"batches": len(outcomes),
        "information_events": sum(bool(item.get("information_event")) for item in outcomes),
        "measured_information_gain_sum": sum(item.get("marginal_information_gain", 0.) for item in outcomes),
        "predicted_information_event_sum": sum(item.get("predicted_information_probability", 0.) for item in outcomes),
        "measured_counterexamples": sum(bool(item.get("observed_response_counterexample")) for item in outcomes)}


def describe(base, arm, root, position):
    source = base / "successor_regression" / "runs" / arm / root / "checkpoint.json.gz"
    state = json.loads(gzip.decompress(source.read_bytes()))
    assert state["status"] == "completed", (source, state["status"])
    final = state["results"][position]["research"]
    visits = [visit for visit in state["revisits"] if visit["task_position"] == position]
    summaries = []
    for index, visit in enumerate(visits):
        before = visit["previous"]
        after = visits[index + 1]["previous"] if index + 1 < len(visits) else final
        actions = after["actions"][len(before["actions"]):]
        experiment = visit["candidate"].get("instrument_condition_experiment", {})
        decision = experiment.get("decision", {})
        paired = [action for action in actions
                  if action["action"] == "compare_method_followup_on_fresh_response"]
        summaries.append({"visit": index + 1, "basis": visit["basis"],
            "costs": visit["visit_costs"], "screening_compute": experiment.get("screening_compute_cost"),
            "allocated": {resource: visit["resource_limits"][resource] - before["costs"][resource]
                          for resource in RESOURCES},
            "decision": {key: decision[key] for key in DECISION_FIELDS if key in decision},
            "hard_compute_reserve": (decision.get("quoted_compute_cost", 0)
                + experiment["confirmation_compute_quote"]) if experiment else None,
            "before": question(before.get("condition_uncertainty")),
            "after": question(after.get("condition_uncertainty")),
            "program_changed": before["program"] != after["program"],
            "information": information(actions),
            "fresh_pairs": [{key: pair[key] for key in ("before_loss", "after_loss", "stage_costs")}
                            for pair in paired],
            "reconstruction_stop_reasons": [action.get("reason") for action in actions
                if action["action"].startswith("stop_")]})
    events = [event for event in state["scheduler_events"]
              if event.get("task_position") == position]
    screens = [event for event in events if event["action"] == "screen_instrument_condition_revisit"]
    return {"source": str(source.relative_to(BASE.parent.parent)),
        "status": final["status"], "learning_costs": final["costs"],
        "final_condition": question(final.get("condition_uncertainty")),
        "last_recorded_question": question((final.get("condition_research_returns") or {}).get("last_question")),
        "revisits": len(visits), "revisit_bases": dict(Counter(v["basis"] for v in visits)),
        "revisit_costs": {resource: sum(v["visit_costs"][resource] for v in visits) for resource in RESOURCES},
        "instrument_screens": len(screens),
        "instrument_screening_compute": sum(event.get("compute_cost", 0) for event in screens),
        "all_paid_information": information(final["actions"]),
        "visits": summaries}


def main():
    cases = []
    for root, position in (("root08", 9), ("root04", 8), ("root07", 5)):
        arms = {}
        for arm in ("new_cumulative", "no_marginal_history"):
            arms[arm] = {version: describe(base, arm, root, position)
                         for version, base in (("0.164", OLD), ("0.165", BASE))}
        cases.append({"root": root, "position": position, "arms": arms})
    result = {"scope": "Read-only completed paid trajectories. No new training or terminal scores.",
        "cases": cases,
        "limitations": "Whole-chain paths differ. Paid information and fresh comparisons do not independently identify each module's causal contribution or terminal accuracy."}
    (BASE / "realized_investment_paid_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
