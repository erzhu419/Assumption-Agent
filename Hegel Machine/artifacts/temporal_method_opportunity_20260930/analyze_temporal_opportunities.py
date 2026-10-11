#!/usr/bin/env python3
"""Report measured temporal steps, successor offers and actual revisit outcomes."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys


output = Path(sys.argv[1])
protocol = json.loads((output / "protocol.json").read_text())
result = {
    "basis": "Learning checkpoints and recorded acquisitions only; screening separation is not adoption or terminal improvement. Conditional support uses the recorded new response_supported flag; old events lack that flag and are derived from original paid loss and gain at the unchanged tolerance.",
    "arms": {},
}


def response_supported(event):
    if "response_supported" in event:
        return bool(event["response_supported"])
    tolerance = protocol["config"]["minimum_improvement"]
    return any(item["status"] == "conditional_duration" and item["loss"] <= tolerance
        and item["gain"] > tolerance for item in event.get("evidence", []))


for arm in protocol["arms"]:
    steps, offers, visits, final, background_revisions = [], [], [], [], []
    all_screening = 0
    for root in protocol["roots"]:
        with gzip.open(output / "runs" / arm / root / "checkpoint.json.gz", "rt") as handle:
            state = json.load(handle)
        all_screening += state.get("screening_compute_cost", 0)
        for record in state["results"]:
            report = record["research"]
            duration = [action for action in report["actions"]
                        if action["action"] == "distinguish_paid_condition_duration"]
            for previous, current in zip(duration, duration[1:]):
                old_background = previous.get("fixed_control_background", previous.get("fixed_controls"))
                new_background = current.get("fixed_control_background", current.get("fixed_controls"))
                if (not response_supported(previous)
                        and previous.get("condition_field") is not None
                        and previous["condition_field"] == current.get("condition_field")
                        and old_background != new_background):
                    background_revisions.append({"root": root, "position": record["task_position"],
                        "condition_field": current["condition_field"],
                        "previous_background": old_background, "revised_background": new_background,
                        "previous_rows": previous["measured_rows"], "revised_rows": current["measured_rows"],
                        "revised_supported": response_supported(current),
                        "duration_compute": previous["compute_cost"] + current["compute_cost"]})
            for action in report["actions"]:
                if (any(word in action["action"] for word in
                        ("duration", "temporal", "successor_partial_method", "successor_method"))
                        or action.get("purpose") == "successor_partial_method_experiment"):
                    steps.append({"root": root, "position": record["task_position"], **action})
            final.append({"root": root, "position": record["task_position"],
                "status": report["status"], "validation_loss": report.get("validation_loss"),
                "selection_loss": report.get("selection_loss"),
                "fresh_confirmation_loss": report.get("confirmation_loss"),
                "selected_kind": report.get("selected_kind"),
                "research_draft_id": report.get("research_draft_id")})
        for event in state.get("scheduler_events", []):
            if event.get("method_opportunity"):
                offers.append({"root": root, **event})
        for index, visit in enumerate(state["revisits"]):
            if visit["candidate"].get("kind") != "successor_partial_method":
                continue
            following = next((item for item in state["revisits"][index + 1:]
                if item["task_position"] == visit["task_position"]), None)
            after = following["previous"] if following else state["results"][visit["task_position"]]["research"]
            before = visit["previous"]
            added = after["actions"][len(before["actions"]):]
            priced_offer = next(event for event in state["scheduler_events"]
                if event.get("method_opportunity") and event["scheduled"]
                and event["task_position"] == visit["task_position"]
                and event["source_task_position"] == visit["source_task_position"]
                and event["method_opportunity"]["draft_ids"] == visit["candidate"]["method_opportunity"]["draft_ids"])
            visits.append({"root": root, "position": visit["task_position"],
                "source_position": visit["source_task_position"],
                "basis": visit.get("basis"), "costs": visit["visit_costs"],
                "prior_kind": before.get("selected_kind"), "after_kind": after.get("selected_kind"),
                "prior_validation_loss": before.get("validation_loss"),
                "after_validation_loss": after.get("validation_loss"),
                "prior_selection_loss": before.get("selection_loss"),
                "after_selection_loss": after.get("selection_loss"),
                "prior_fresh_loss": before.get("confirmation_loss"),
                "after_fresh_loss": after.get("confirmation_loss"),
                "method_opportunity": visit["candidate"].get("method_opportunity"),
                "quoted_minimum_resources": priced_offer["minimum_resources"],
                "action_counts": dict(Counter(action["action"] for action in added))})
    purchases = [step for step in steps if step.get("purpose") == "successor_partial_method_experiment"]
    executions = [step for step in steps if step["action"] == "execute_successor_method_experiment"
                  and step["observation_rows"] > 0]
    measured = [step for step in steps if step["action"] == "distinguish_paid_condition_duration"]
    comparison = [step for step in executions if step.get("incumbent_loss") is not None
                  and step.get("proposed_loss") is not None]
    result["arms"][arm] = {
        "temporal_and_method_action_counts": dict(Counter(step["action"] for step in steps)),
        "method_offers": len(offers), "scheduled_method_offers": sum(offer["scheduled"] for offer in offers),
        "method_screening_compute": sum(offer["compute_cost"] for offer in offers),
        "all_successor_screening_compute": all_screening,
        "method_revisits": len(visits), "method_experiments": len(purchases),
        "method_experiment_observations": sum(step["observation_cost"] for step in purchases),
        "executed_method_opportunities": len(executions),
        "frozen_method_comparisons": len(comparison),
        "frozen_method_comparisons_improved": sum(step["proposed_loss"] < step["incumbent_loss"]
            - protocol["config"]["minimum_improvement"] for step in comparison),
        "method_revisits_fresh_improved": sum(visit["after_fresh_loss"] is not None
            and visit["prior_fresh_loss"] is not None
            and visit["after_fresh_loss"] < visit["prior_fresh_loss"]
            - protocol["config"]["minimum_improvement"] for visit in visits),
        "measured_duration_batches": len(measured),
        "measured_duration_supported": sum(response_supported(step)
            for step in measured),
        "duration_background_revisions": len(background_revisions),
        "duration_background_revisions_supported": sum(item["revised_supported"] for item in background_revisions),
        "method_revisit_costs": {key: sum(visit["costs"][key] for visit in visits)
            for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "method_quoted_minimum_resources": {key: sum(visit["quoted_minimum_resources"][key] for visit in visits)
            for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "method_offer_records": offers, "method_revisit_records": visits,
        "measured_temporal_and_method_steps": steps, "final_selections": final,
        "duration_background_revision_records": background_revisions,
    }
(output / "temporal_method_opportunities.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in records.items()
    if key.endswith("counts") or not isinstance(value, list)}
    for arm, records in result["arms"].items()})
