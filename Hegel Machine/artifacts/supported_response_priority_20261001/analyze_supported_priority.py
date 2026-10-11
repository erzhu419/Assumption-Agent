#!/usr/bin/env python3
"""Report executed supported fits, actual method structure and new confirmation."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
protocol = json.loads((output / "protocol.json").read_text())
result = {"basis": "Recorded learning actions and actual whole-method visits. "
    "Supported replay, original paid validation and new confirmation are separate. "
    "Current method geometry comes from response_basis; old-method narration is reported separately.",
    "arms": {}}

def geometry(applicability):
    widths, placements = set(), set()
    def walk(node, ancestors=()):
        if node.get("op") == "window_sum":
            widths.add(node["lag"])
        if node.get("op") == "if":
            placements.add("inside_window" if "window_sum" in ancestors else "outside_window")
        for child in node.get("children", []):
            walk(child, (*ancestors, node.get("op")))
    basis = applicability.get("response_basis", [])
    for term in basis:
        walk(term)
    return {"basis_terms": len(basis), "window_widths": sorted(widths),
            "condition_placements": sorted(placements), "response_basis": basis,
            "method_steps": applicability.get("method_steps", [])}

for arm in protocol["arms"]:
    replays, fits, assessments, visits, offers = [], [], [], [], []
    for root in protocol["roots"]:
        with gzip.open(output / "runs" / arm / root / "checkpoint.json.gz", "rt") as stream:
            state = json.load(stream)
        for record in state["results"]:
            latest_replay = {}
            for index, action in enumerate(record["research"]["actions"]):
                item = {"root": root, "position": record["task_position"], "action_index": index, **action}
                if action["action"] == "replay_research_draft":
                    replays.append(item)
                    latest_replay[action["draft_id"]] = action.get("fitting_priority", "legacy")
                elif action["action"] == "refit_observed_condition_response":
                    fits.append(item)
                elif action["action"] == "assess_research_draft":
                    assessments.append({**item, "replay_fitting_priority":
                        latest_replay.get(action["draft_id"], "legacy")})
        for event in state.get("scheduler_events", []):
            if event.get("method_opportunity"):
                opportunity = event["method_opportunity"]
                offers.append({"root": root, **event,
                    "actual_method_geometry": geometry(opportunity.get("method_applicability", {}))})
        for visit in state["revisits"]:
            if visit["candidate"].get("kind") != "successor_partial_method":
                continue
            outcome = visit.get("method_visit_outcome")
            visits.append({"root": root, "position": visit["task_position"],
                "source_position": visit["source_task_position"], "outcome": outcome,
                "costs": visit["visit_costs"], "screening_compute_cost": visit.get("screening_compute_cost"),
                "resource_allocation": {key: visit["resource_limits"][key] - visit["previous"]["costs"][key]
                    for key in ("observation_cost", "measurement_cost", "compute_cost")},
                "actual_method_geometry":
                    geometry(visit["candidate"]["method_opportunity"].get("method_applicability", {}))})
    supported = [item for item in replays if item.get("fitting_priority") == "current_supported_structure"]
    supported_assessments = [item for item in assessments
        if item["replay_fitting_priority"] == "current_supported_structure"]
    confirmed = [item for item in visits if item["outcome"]
                 and item["outcome"]["fresh_comparison"] is not None]
    with_geometry = [item for item in visits if item["actual_method_geometry"]["response_basis"]]
    result["arms"][arm] = {
        "replay_priorities": dict(Counter(item.get("fitting_priority", "legacy") for item in replays)),
        "supported_replays": len(supported),
        "supported_replays_producing_candidates": sum(item.get("training_loss") is not None for item in supported),
        "supported_replay_compute": sum(item["compute_cost"] for item in supported),
        "supported_replay_assessments": len(supported_assessments),
        "supported_replay_assessment_statuses": dict(Counter(item["status"] for item in supported_assessments)),
        "response_refit_priorities": dict(Counter(item.get("fitting_priority", "legacy") for item in fits)),
        "method_visits": len(visits), "method_visits_with_new_confirmation": len(confirmed),
        "method_visits_without_new_confirmation": len(visits) - len(confirmed),
        "method_visit_confirmation_rate": len(confirmed) / len(visits) if visits else None,
        "method_visits_with_current_response_basis": len(with_geometry),
        "method_visit_allocations": {key: sum(item["resource_allocation"][key] for item in visits)
            for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "method_visit_costs_excluding_screening": {key: sum(item["costs"][key] for item in visits)
            for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "executed_method_screening_compute": sum(item["screening_compute_cost"] or 0 for item in visits),
        "method_visits_with_window_geometry": sum(bool(item["actual_method_geometry"]["window_widths"]) for item in with_geometry),
        "method_visits_with_condition_placement": sum(bool(item["actual_method_geometry"]["condition_placements"]) for item in with_geometry),
        "supported_replay_records": supported, "supported_replay_assessment_records": supported_assessments,
        "response_refit_records": fits, "method_visit_records": visits, "method_offer_records": offers}
(output / "supported_priority_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in data.items() if not isinstance(value, list)}
       for arm, data in result["arms"].items()})
