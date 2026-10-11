#!/usr/bin/env python3
"""Count paid experiment returns and actual research decisions, without terminal inference."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
protocol = json.loads((output / "protocol.json").read_text())
result = {"basis": "Final recorded learning actions. Paid information returns, followup "
    "decisions and fitting costs are separate. A stop count does not establish net benefit.", "arms": {}}
fields = ("action", "continue", "buy", "reason", "fresh_gain", "previous_max_point_loss",
    "new_max_point_loss", "condition_evidence_change", "supported_structure",
    "planning_compute_cost", "compute_cost", "measured_rows", "changed_program", "research_return",
    "condition_field", "threshold_intervals", "fixed_control_background", "experiment_revision_basis",
    "previous_experiment_count", "source_operation")
for arm in protocol["arms"]:
    decisions, purchases, plans, followups, condition_stops, rejections = [], [], [], [], [], []
    for root in protocol["roots"]:
        with gzip.open(output / "runs" / arm / root / "checkpoint.json.gz", "rt") as stream:
            state = json.load(stream)
        for record in state["results"]:
            for index, action in enumerate(record["research"]["actions"]):
                item = {"root": root, "position": record["task_position"], "action_index": index,
                    **{key: action[key] for key in fields if key in action}}
                if action["action"] == "choose_condition_research_investment":
                    decisions.append(item)
                elif action["action"] == "plan_competing_observed_conditions":
                    plans.append(item)
                elif action["action"] == "revise_competing_observed_conditions":
                    purchases.append(item)
                elif action["action"] == "choose_method_followup_research":
                    followups.append(item)
                elif action["action"] == "stop_condition_followup_research":
                    condition_stops.append(item)
                elif action["action"] == "assess_research_draft" and action["status"] == "current_counterexample":
                    rejections.append({"root": root, "position": record["task_position"],
                        "action_index": index, "draft_id": action["draft_id"],
                        "validation_loss": action.get("validation_loss")})
    returns = [item["research_return"] for item in purchases if item.get("research_return")]
    result["arms"][arm] = {
        "paid_competing_experiments": len(purchases),
        "paid_competing_rows": sum(item["measured_rows"] for item in purchases),
        "changed_programs": sum(item["changed_program"] for item in purchases),
        "recorded_paid_returns": len(returns),
        "informative_returns": sum(item["informative"] for item in returns),
        "interval_narrowing_returns": sum(item["interval_narrowed"] for item in returns),
        "counterexample_returns": sum(item["new_counterexample"] for item in returns),
        "uninformative_returns": sum(not item["informative"] for item in returns),
        "investment_decisions": len(decisions),
        "investment_stop_reasons": dict(Counter(item["reason"] for item in decisions if not item["buy"])),
        "investment_revision_bases": dict(Counter(item.get("experiment_revision_basis", "legacy")
            for item in decisions)),
        "investment_decisions_using_prior_experiments": sum(item.get("previous_experiment_count", 0) > 0
            for item in decisions),
        "plans_with_current_paid_condition_support": sum(any(
            interval.get("source_operation") == "current_paid_condition_support"
            for interval in item.get("threshold_intervals", []) if isinstance(interval, dict)) for item in plans),
        "plan_source_operations": dict(Counter(interval.get("source_operation", "legacy")
            for item in plans for interval in item.get("threshold_intervals", []) if isinstance(interval, dict))),
        "method_followup_decisions": len(followups),
        "method_followup_continues": sum(item["continue"] for item in followups),
        "method_followup_stops": sum(not item["continue"] for item in followups),
        "method_followup_reasons": dict(Counter(item["reason"] for item in followups)),
        "condition_followup_stops": len(condition_stops),
        "condition_followup_stop_reasons": dict(Counter(item["reason"] for item in condition_stops)),
        "counterexample_assessments_on_paid_data": len(rejections),
        "investment_decision_records": decisions, "paid_experiment_records": purchases, "plan_records": plans,
        "method_followup_records": followups, "condition_followup_stop_records": condition_stops,
        "paid_rejection_records": rejections}
(output / "experiment_investment_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in data.items() if not isinstance(value, list)}
       for arm, data in result["arms"].items()})
