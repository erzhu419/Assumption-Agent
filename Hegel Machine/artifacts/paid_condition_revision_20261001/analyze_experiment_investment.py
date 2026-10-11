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
    "previous_experiment_count", "source_operation", "current_question_trials",
    "current_question_information_batches", "current_question_predicted_information_batches",
    "sampling_basis", "interval_exposure", "current_question_returns", "evidence_rows",
    "before_loss", "after_loss", "before_validation_loss", "after_validation_loss",
    "requires_fresh_confirmation")
for arm in protocol["arms"]:
    decisions, purchases, plans, followups, condition_stops, rejections = [], [], [], [], [], []
    paid_revisions, paid_revision_validations = [], {}
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
                elif action["action"] == "revise_condition_on_paid_response":
                    later = record["research"]["actions"][index + 1:]
                    decision = next(((index + 1 + offset, candidate) for offset, candidate in enumerate(later)
                        if candidate["action"] == "choose_condition_research_investment"), None)
                    purchase = next((index + 1 + offset for offset, candidate in enumerate(later)
                        if candidate["action"] == "revise_competing_observed_conditions"), None)
                    confirmation = next(((index + 1 + offset, candidate) for offset, candidate in enumerate(later)
                        if candidate["action"] in ("validate_frozen_program", "validate_continued_program")
                        and candidate.get("observation_rows", 0) > 0), None)
                    if item["changed_program"] and confirmation:
                        confirmation_index, checked = confirmation
                        paid_revision_validations[(root, record["task_position"], confirmation_index)] = {
                            "root": root, "position": record["task_position"], "action_index": confirmation_index,
                            "action": checked["action"], "observation_rows": checked["observation_rows"],
                            "loss": checked["loss"], "accepted": checked["accepted"],
                            "same_as_final_program": checked["program"] == record["research"]["program"],
                            "probe_id": checked.get("probe_id")}
                    paid_revisions.append({**item,
                        "next_condition_decision_index": decision[0] if decision else None,
                        "next_condition_decision_buy": decision[1]["buy"] if decision else None,
                        "next_condition_purchase_index": purchase,
                        "subsequent_frozen_validation_index": confirmation[0]
                            if item["changed_program"] and confirmation else None})
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
        "decisions_with_current_question_calibration": sum("current_question_trials" in item for item in decisions),
        "decisions_with_current_question_misses_only": sum(item.get("current_question_trials", 0) > 0
            and item.get("current_question_information_batches", 0) == 0 for item in decisions),
        "stops_after_current_question_misses_only": sum(item.get("current_question_trials", 0) > 0
            and item.get("current_question_information_batches", 0) == 0 and not item["buy"] for item in decisions),
        "plan_sampling_bases": dict(Counter(item.get("sampling_basis", "legacy") for item in plans)),
        "plan_probability_bases": dict(Counter(exposure.get("probability_basis", "legacy")
            for item in plans for exposure in item.get("interval_exposure", []))),
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
        "paid_response_revisions": len(paid_revisions),
        "paid_response_program_changes": sum(item["changed_program"] for item in paid_revisions),
        "paid_response_revision_compute": sum(item["compute_cost"] for item in paid_revisions),
        "paid_response_revisions_without_new_rows": sum(item["measured_rows"] == 0 for item in paid_revisions),
        "paid_response_revisions_requiring_fresh_confirmation": sum(item["requires_fresh_confirmation"]
            for item in paid_revisions),
        "paid_response_revisions_before_investment": sum(item["next_condition_decision_index"] is not None
            for item in paid_revisions),
        "paid_response_next_investment_buys": sum(item["next_condition_decision_buy"] is True
            for item in paid_revisions),
        "paid_response_next_investment_stops": sum(item["next_condition_decision_buy"] is False
            for item in paid_revisions),
        "paid_response_changes_with_subsequent_frozen_validation": sum(item["changed_program"]
            and item["subsequent_frozen_validation_index"] is not None for item in paid_revisions),
        "paid_response_unique_subsequent_validation_events": len(paid_revision_validations),
        "paid_response_subsequent_validation_rows": sum(item["observation_rows"]
            for item in paid_revision_validations.values()),
        "paid_response_subsequent_validation_point_fits": sum(item["loss"] <= protocol["config"]["minimum_improvement"]
            for item in paid_revision_validations.values()),
        "paid_response_subsequent_validation_acceptances": sum(item["accepted"]
            for item in paid_revision_validations.values()),
        "paid_response_subsequent_validations_matching_final_program": sum(item["same_as_final_program"]
            for item in paid_revision_validations.values()),
        "investment_decision_records": decisions, "paid_experiment_records": purchases, "plan_records": plans,
        "method_followup_records": followups, "condition_followup_stop_records": condition_stops,
        "paid_rejection_records": rejections, "paid_response_revision_records": paid_revisions,
        "paid_response_subsequent_validation_records": list(paid_revision_validations.values())}
(output / "experiment_investment_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in data.items() if not isinstance(value, list)}
       for arm, data in result["arms"].items()})
