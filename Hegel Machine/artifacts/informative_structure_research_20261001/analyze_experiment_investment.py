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
    "decisions and fitting costs are separate. Initial and continued visits are distinguished by "
    "the actual resume_measured_research marker; subsequent validation is restricted to the same visit. "
    "A stop count does not establish net benefit.", "arms": {}}
fields = ("action", "continue", "buy", "reason", "fresh_gain", "previous_max_point_loss",
    "new_max_point_loss", "condition_evidence_change", "supported_structure",
    "planning_compute_cost", "compute_cost", "measured_rows", "changed_program", "research_return",
    "condition_field", "threshold_intervals", "fixed_control_background", "experiment_revision_basis",
    "previous_experiment_count", "source_operation", "current_question_trials",
    "current_question_information_batches", "current_question_predicted_information_batches",
    "sampling_basis", "interval_exposure", "current_question_returns", "evidence_rows",
    "before_loss", "after_loss", "before_validation_loss", "after_validation_loss",
    "requires_fresh_confirmation", "research_action", "evidence_basis", "paid_rows",
    "paid_response_loss", "current_response_loss", "observation_remaining",
    "measurement_remaining", "compute_remaining", "fitting_priority", "fitting_rows",
    "candidate_count", "evidence", "structure_nominations", "competing_structures", "response_supported",
    "fitted_structures", "comparison_basis", "alternatives", "selected_draft_ids",
    "fit_rows", "fit_features", "partial_structure_disagreement", "forecast_length",
    "boundary_disagreement", "duration_disagreement", "probability_of_new_information",
    "calibrated_information_probability", "sampling_reliability", "resource_share")

def response_geometry(program):
    widths, placements = set(), set()
    def walk(node, ancestors=()):
        if node["op"] == "window_sum":
            widths.add(node["lag"])
        if node["op"] == "if":
            placements.add("inside_window" if "window_sum" in ancestors else "outside_window")
        for child in node.get("children", []):
            walk(child, (*ancestors, node["op"]))
        if node["op"] == "call":
            walk(program["definitions"][node["name"]]["body"], (*ancestors, "call"))
    walk(program)
    return {"window_widths": sorted(widths), "condition_placements": sorted(placements)}

for arm in protocol["arms"]:
    decisions, purchases, plans, followups, condition_stops, rejections = [], [], [], [], [], []
    paid_revisions, paid_revision_validations = [], {}
    paid_reconstructions, paid_reconstruction_validations, agenda = [], {}, []
    structure_fits, duration_experiments, structure_validations = [], [], {}
    draft_selections, retained_quotes, deferred_funding, method_screens = [], [], [], []
    partial_fits, partial_plans, partial_confirmations = [], [], {}
    boundary_plans = []
    for root in protocol["roots"]:
        with gzip.open(output / "runs" / arm / root / "checkpoint.json.gz", "rt") as stream:
            state = json.load(stream)
        for record in state["results"]:
            visit_index = 0
            for index, action in enumerate(record["research"]["actions"]):
                if action["action"] == "resume_measured_research":
                    visit_index += 1
                item = {"root": root, "position": record["task_position"], "action_index": index,
                    "phase": "continuation" if visit_index else "initial", "visit_index": visit_index,
                    **{key: action[key] for key in fields if key in action}}
                if (action["action"] in ("plan_observed_condition_experiment", "plan_competing_observed_conditions")
                        and action.get("experiment_revision_basis") == "unmeasured_temporal_boundary"):
                    later = record["research"]["actions"][index + 1:]
                    boundary = next((offset for offset, candidate in enumerate(later)
                        if candidate["action"] in ("resume_measured_research", "plan_observed_condition_experiment",
                            "plan_competing_observed_conditions")), len(later))
                    purchase_action = ("distinguish_paid_condition_duration"
                        if action["action"] == "plan_observed_condition_experiment" else "revise_competing_observed_conditions")
                    executed = next((candidate for candidate in later[:boundary]
                        if candidate["action"] == purchase_action), None)
                    boundary_plans.append({**item, "executed_rows": executed["measured_rows"] if executed else 0})
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
                elif action["action"] == "choose_observed_response_followup":
                    agenda.append(item)
                elif action["action"] == "distinguish_paid_condition_duration":
                    duration_experiments.append(item)
                elif action["action"] == "plan_observed_condition_experiment":
                    if action.get("experiment_revision_basis") == "cross_background_partial_support":
                        later = record["research"]["actions"][index + 1:]
                        boundary = next((offset for offset, candidate in enumerate(later)
                            if candidate["action"] in ("resume_measured_research", "plan_observed_condition_experiment")), len(later))
                        executed = next((candidate for candidate in later[:boundary]
                            if candidate["action"] == "distinguish_paid_condition_duration"), None)
                        partial_plans.append({**item, "executed_rows": executed["measured_rows"] if executed else 0})
                elif action["action"] == "fit_partial_structure_competitors":
                    candidates = []
                    later = record["research"]["actions"][index + 1:]
                    boundary = next((offset for offset, candidate in enumerate(later)
                        if candidate["action"] == "resume_measured_research"), len(later))
                    for claim in action["competitors"]:
                        program = claim["candidate"]["program"]
                        support = claim["partial_support"]
                        candidates.append({"training_loss": claim["training_loss"],
                            "actual_response_geometry": response_geometry(program),
                            "support_experiments": sorted({entry["observed_experiment_index"] for entry in support}),
                            "support_background_count": len({tuple(sorted(entry["background"].items())) for entry in support}),
                            "zero_gain_conditional_support_rows": sum(entry["fit_rows"] for entry in support
                                if entry["proposed_structure"] == "conditional_duration"
                                and entry["response_supported"] and not entry["condition_identified"]),
                            "nonexact_informative_support_rows": sum(entry["fit_rows"] for entry in support
                                if not entry["response_supported"]
                                and entry.get("gain", 0.) > protocol["config"]["minimum_improvement"]),
                            "actual_local_losses": [entry["local_loss"] for entry in support],
                            "actual_recorded_local_gains": [entry.get("gain") for entry in support],
                            "matches_final_program": program == record["research"]["program"]})
                        for offset, checked in enumerate(later[:boundary]):
                            if (checked["action"] in ("validate_frozen_program", "validate_continued_program")
                                    and checked.get("observation_rows", 0) > 0 and checked["program"] == program):
                                key = root, record["task_position"], index + 1 + offset
                                partial_confirmations[key] = {"root": root, "position": record["task_position"],
                                    "action_index": index + 1 + offset, "probe_id": checked.get("probe_id"),
                                    "observation_rows": checked["observation_rows"], "loss": checked["loss"],
                                    "accepted": checked["accepted"], "same_as_final_program": program == record["research"]["program"],
                                    "final_report_confirmation_loss": record["research"].get("confirmation_loss"),
                                    "final_research_status": record["research"]["status"]}
                    partial_fits.append({**item, "actual_candidates": candidates})
                elif action["action"] == "select_research_drafts_on_paid_response":
                    draft_selections.append({**item, "location": "report"})
                elif action["action"] == "refit_observed_condition_response":
                    later = record["research"]["actions"][index + 1:]
                    next_visit = next((offset for offset, candidate in enumerate(later)
                        if candidate["action"] == "resume_measured_research"), len(later))
                    confirmation = next(((index + 1 + offset, candidate)
                        for offset, candidate in enumerate(later[:next_visit])
                        if candidate["action"] in ("validate_frozen_program", "validate_continued_program")
                        and candidate.get("observation_rows", 0) > 0), None)
                    combined = [*item.get("evidence", []), *item.get("structure_nominations", [])]
                    item["input_window_widths"] = sorted({width for entry in combined
                        for width in entry["width_candidates"]})
                    item["input_condition_placements"] = sorted({placement for entry in combined
                        for placement in entry["placement_candidates"]})
                    if "fitted_structures" in item:
                        item["actual_fitted_window_widths"] = sorted({entry["width"]
                            for entry in item["fitted_structures"]})
                        item["actual_fitted_condition_placements"] = sorted({entry["placement"]
                            for entry in item["fitted_structures"]})
                    item["subsequent_frozen_validation_index"] = confirmation[0] if confirmation else None
                    structure_fits.append(item)
                    if item["candidate_count"] > 0 and confirmation:
                        confirmation_index, checked = confirmation
                        structure_validations[(root, record["task_position"], confirmation_index)] = {
                            "root": root, "position": record["task_position"], "action_index": confirmation_index,
                            "phase": item["phase"], "visit_index": visit_index,
                            "action": checked["action"], "observation_rows": checked["observation_rows"],
                            "loss": checked["loss"], "accepted": checked["accepted"],
                            "same_as_final_program": checked["program"] == record["research"]["program"],
                            "probe_id": checked.get("probe_id"),
                            "actual_response_geometry": response_geometry(checked["program"])}
                elif action["action"] == "assess_research_draft" and action["status"] == "current_counterexample":
                    rejections.append({"root": root, "position": record["task_position"],
                        "action_index": index, "draft_id": action["draft_id"],
                        "validation_loss": action.get("validation_loss")})
                elif action["action"] in ("revise_condition_on_paid_response", "reconstruct_paid_observed_response"):
                    revision_events, revision_validations = ((paid_revisions, paid_revision_validations)
                        if action["action"] == "revise_condition_on_paid_response"
                        else (paid_reconstructions, paid_reconstruction_validations))
                    later = record["research"]["actions"][index + 1:]
                    next_visit = next((offset for offset, candidate in enumerate(later)
                        if candidate["action"] == "resume_measured_research"), len(later))
                    later = later[:next_visit]
                    decision = next(((index + 1 + offset, candidate) for offset, candidate in enumerate(later)
                        if candidate["action"] == "choose_condition_research_investment"), None)
                    purchase = next((index + 1 + offset for offset, candidate in enumerate(later)
                        if candidate["action"] == "revise_competing_observed_conditions"), None)
                    confirmation = next(((index + 1 + offset, candidate) for offset, candidate in enumerate(later)
                        if candidate["action"] in ("validate_frozen_program", "validate_continued_program")
                        and candidate.get("observation_rows", 0) > 0), None)
                    if item["changed_program"] and confirmation:
                        confirmation_index, checked = confirmation
                        revision_validations[(root, record["task_position"], confirmation_index)] = {
                            "root": root, "position": record["task_position"], "action_index": confirmation_index,
                            "phase": item["phase"], "visit_index": visit_index,
                            "action": checked["action"], "observation_rows": checked["observation_rows"],
                            "loss": checked["loss"], "accepted": checked["accepted"],
                            "same_as_final_program": checked["program"] == record["research"]["program"],
                            "probe_id": checked.get("probe_id")}
                    revision_events.append({**item,
                        "next_condition_decision_index": decision[0] if decision else None,
                        "next_condition_decision_buy": decision[1]["buy"] if decision else None,
                        "next_condition_purchase_index": purchase,
                        "subsequent_frozen_validation_index": confirmation[0]
                            if item["changed_program"] and confirmation else None})
        for event_index, event in enumerate(state.get("scheduler_events", [])):
            item = {"root": root, "event_index": event_index,
                **{key: event[key] for key in ("action", "task_position", "source_task_position",
                    "compute_cost", "scheduled", "reason", "basis", "old_evidence_gain", "minimum_resources")
                    if key in event}}
            if event["action"] == "try_retained_method":
                retained_quotes.append(item)
            elif event["action"] == "fund_deferred_method_experiment":
                deferred_funding.append(item)
            for field in ("method_screening", "screening"):
                for screening_index, action in enumerate(event.get(field, [])):
                    if action["action"] == "select_research_drafts_on_paid_response":
                        draft_selections.append({"root": root, "location": "scheduler",
                            "position": event["task_position"], "source_position": event.get("source_task_position"),
                            "event_index": event_index, "screening_index": screening_index,
                            **{key: action[key] for key in fields if key in action}})
        method_screens.extend({"root": root, "position": item["task_position"],
            "draft_id": item["draft_id"], "paid_batches": item.get("paid_batches"),
            "deferred_offer_remaining": "deferred_offer" in item}
            for item in state.get("method_revisit_attempts", []))
    returns = [item["research_return"] for item in purchases if item.get("research_return")]
    result["arms"][arm] = {
        "partial_complete_fit_events": len(partial_fits),
        "partial_complete_fit_compute": sum(item["compute_cost"] for item in partial_fits),
        "partial_actual_complete_candidates": sum(len(item["actual_candidates"]) for item in partial_fits),
        "partial_candidates_with_multiple_support_experiments": sum(len(candidate["support_experiments"]) > 1
            for item in partial_fits for candidate in item["actual_candidates"]),
        "partial_actual_candidates_using_zero_gain_conditional_support": sum(candidate["zero_gain_conditional_support_rows"] > 0
            for item in partial_fits for candidate in item["actual_candidates"]),
        "partial_actual_candidates_using_nonexact_informative_support": sum(candidate["nonexact_informative_support_rows"] > 0
            for item in partial_fits for candidate in item["actual_candidates"]),
        "partial_final_program_tasks": len({(item["root"], item["position"])
            for item in partial_fits if any(candidate["matches_final_program"] for candidate in item["actual_candidates"])}),
        "partial_cross_background_plans": len(partial_plans),
        "partial_cross_background_plans_executed": sum(item["executed_rows"] > 0 for item in partial_plans),
        "partial_cross_background_rows": sum(item["executed_rows"] for item in partial_plans),
        "partial_matching_new_confirmations": len(partial_confirmations),
        "partial_matching_new_confirmations_accepted": sum(item["accepted"] for item in partial_confirmations.values()),
        "partial_final_program_with_current_confirmation_tasks": len({(item["root"], item["position"])
            for item in partial_confirmations.values() if item["same_as_final_program"]
            and item["accepted"] and item["final_report_confirmation_loss"] is not None
            and item["final_report_confirmation_loss"] <= protocol["config"]["minimum_improvement"]}),
        "partial_final_program_current_confirmation_statuses": dict(Counter(
            status for _, _, status in {(item["root"], item["position"], item["final_research_status"])
                for item in partial_confirmations.values() if item["same_as_final_program"]
                and item["accepted"] and item["final_report_confirmation_loss"] is not None
                and item["final_report_confirmation_loss"] <= protocol["config"]["minimum_improvement"]})),
        "partial_fit_records": partial_fits,
        "partial_cross_background_plan_records": partial_plans,
        "partial_matching_confirmation_records": list(partial_confirmations.values()),
        "unmeasured_boundary_investment_decisions": sum(item.get("experiment_revision_basis") == "unmeasured_temporal_boundary"
            for item in decisions),
        "unmeasured_boundary_investment_stops": sum(item.get("experiment_revision_basis") == "unmeasured_temporal_boundary"
            and not item["buy"] for item in decisions),
        "unmeasured_boundary_actual_plans": len(boundary_plans),
        "unmeasured_boundary_actual_plan_purchases": sum(item["executed_rows"] > 0 for item in boundary_plans),
        "unmeasured_boundary_actual_rows": sum(item["executed_rows"] for item in boundary_plans),
        "unmeasured_boundary_plan_records": boundary_plans,
        "paid_draft_selection_actions_by_location": dict(Counter(item["location"] for item in draft_selections)),
        "paid_draft_selection_compute_by_location": {location: sum(item["compute_cost"]
            for item in draft_selections if item["location"] == location)
            for location in ("report", "scheduler")},
        "paid_draft_selection_actually_fitted_alternatives_by_location": {location: sum(
            alternative["compute_cost"] > 0 or alternative["training_loss"] is not None
            for item in draft_selections if item["location"] == location
            for alternative in item["alternatives"]) for location in ("report", "scheduler")},
        "retained_method_quotes": len(retained_quotes),
        "retained_method_quotes_scheduled": sum(item["scheduled"] for item in retained_quotes),
        "retained_method_quote_compute": sum(item["compute_cost"] for item in retained_quotes),
        "retained_method_quote_reasons": dict(Counter(item.get("reason", "no_reason_recorded") for item in retained_quotes)),
        "deferred_method_funding_events": len(deferred_funding),
        "deferred_method_funding_scheduled": sum(item["scheduled"] for item in deferred_funding),
        "registered_method_attempts": len(method_screens),
        "registered_method_attempts_with_paid_batch_state": sum(item["paid_batches"] is not None for item in method_screens),
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
        "paid_structure_fit_priorities": dict(Counter(item["fitting_priority"] for item in structure_fits)),
        "paid_structure_fit_priorities_producing_candidates": dict(Counter(item["fitting_priority"]
            for item in structure_fits if item["candidate_count"] > 0)),
        "paid_structure_fit_compute_by_priority": {priority: sum(item["compute_cost"]
            for item in structure_fits if item["fitting_priority"] == priority)
            for priority in sorted({item["fitting_priority"] for item in structure_fits})},
        "paid_structure_fitting_rows_by_priority": {priority: {
            "minimum": min(item["fitting_rows"] for item in structure_fits if item["fitting_priority"] == priority),
            "maximum": max(item["fitting_rows"] for item in structure_fits if item["fitting_priority"] == priority)}
            for priority in sorted({item["fitting_priority"] for item in structure_fits})},
        "paid_structure_fits_with_both_temporal_placements": sum(
            item["input_condition_placements"] == ["inside_window", "outside_window"] for item in structure_fits),
        "paid_structure_refits_recording_actual_fitted_geometry": sum("fitted_structures" in item for item in structure_fits),
        "paid_structure_fits_with_both_actual_temporal_placements": sum(
            item.get("actual_fitted_condition_placements") == ["inside_window", "outside_window"] for item in structure_fits),
        "paid_structure_actual_fitted_placements_by_priority": {priority: dict(Counter(entry["placement"]
            for item in structure_fits if item["fitting_priority"] == priority
            for entry in item.get("fitted_structures", [])))
            for priority in sorted({item["fitting_priority"] for item in structure_fits})},
        "paid_structure_nominated_identification_states": dict(Counter(str(entry.get("condition_identified"))
            for item in structure_fits for entry in item.get("structure_nominations", []))),
        "paid_structure_nominations_with_paid_source_index": sum("observed_experiment_index" in entry
            for item in structure_fits for entry in item.get("structure_nominations", [])),
        "paid_duration_experiment_records": len(duration_experiments),
        "paid_structure_unique_subsequent_validation_events": len(structure_validations),
        "paid_structure_subsequent_validation_rows": sum(item["observation_rows"] for item in structure_validations.values()),
        "paid_structure_subsequent_validation_point_fits": sum(item["loss"] <= protocol["config"]["minimum_improvement"]
            for item in structure_validations.values()),
        "paid_structure_subsequent_validations_matching_final_program": sum(item["same_as_final_program"]
            for item in structure_validations.values()),
        "observed_response_agenda_actions": len(agenda),
        "observed_response_agenda_directions": dict(Counter(item["research_action"] for item in agenda)),
        "observed_response_agenda_directions_by_phase": {phase: dict(Counter(item["research_action"]
            for item in agenda if item["phase"] == phase)) for phase in ("initial", "continuation")},
        "observed_response_agenda_evidence_bases": dict(Counter(item["evidence_basis"] for item in agenda)),
        "observed_response_agenda_with_paid_misfit": sum(item["paid_response_loss"] is not None
            and item["paid_response_loss"] > protocol["config"]["minimum_improvement"] for item in agenda),
        "observed_response_agenda_with_current_fit_and_paid_misfit": sum(item["current_response_loss"] is not None
            and item["current_response_loss"] <= protocol["config"]["minimum_improvement"]
            and item["paid_response_loss"] is not None
            and item["paid_response_loss"] > protocol["config"]["minimum_improvement"] for item in agenda),
        "paid_response_revisions": len(paid_revisions),
        "paid_response_program_changes": sum(item["changed_program"] for item in paid_revisions),
        "paid_response_revision_phases": dict(Counter(item["phase"] for item in paid_revisions)),
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
        "paid_response_reconstructions": len(paid_reconstructions),
        "paid_response_reconstruction_program_changes": sum(item["changed_program"] for item in paid_reconstructions),
        "paid_response_reconstruction_phases": dict(Counter(item["phase"] for item in paid_reconstructions)),
        "paid_response_reconstruction_compute": sum(item["compute_cost"] for item in paid_reconstructions),
        "paid_response_reconstructions_without_new_rows": sum(item["measured_rows"] == 0 for item in paid_reconstructions),
        "paid_response_reconstruction_unique_subsequent_validation_events": len(paid_reconstruction_validations),
        "paid_response_reconstruction_subsequent_validation_rows": sum(item["observation_rows"]
            for item in paid_reconstruction_validations.values()),
        "paid_response_reconstruction_subsequent_validation_point_fits": sum(item["loss"] <= protocol["config"]["minimum_improvement"]
            for item in paid_reconstruction_validations.values()),
        "paid_response_reconstruction_subsequent_validations_matching_final_program": sum(item["same_as_final_program"]
            for item in paid_reconstruction_validations.values()),
        "investment_decision_records": decisions, "paid_experiment_records": purchases, "plan_records": plans,
        "method_followup_records": followups, "condition_followup_stop_records": condition_stops,
        "paid_rejection_records": rejections, "paid_response_revision_records": paid_revisions,
        "paid_response_subsequent_validation_records": list(paid_revision_validations.values()),
        "observed_response_agenda_records": agenda, "paid_response_reconstruction_records": paid_reconstructions,
        "paid_response_reconstruction_subsequent_validation_records": list(paid_reconstruction_validations.values()),
        "paid_structure_fit_records": structure_fits, "paid_duration_experiment_action_records": duration_experiments,
        "paid_structure_subsequent_validation_records": list(structure_validations.values()),
        "paid_draft_selection_records": draft_selections, "retained_method_quote_records": retained_quotes,
        "deferred_method_funding_records": deferred_funding, "registered_method_attempt_records": method_screens}
(output / "experiment_investment_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in data.items() if not isinstance(value, list)}
       for arm, data in result["arms"].items()})
