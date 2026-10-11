#!/usr/bin/env python3
"""Summarize actual unresolved fits and whole-visit successor research returns."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
protocol = json.loads((output / "protocol.json").read_text())
tolerance = protocol["config"]["minimum_improvement"]
result = {"basis": "Learning checkpoints and real recorded response comparisons. "
    "A temporal nomination is not a supported law; a method offer is not an executed visit. "
    "Whole-visit returns include screening and use newly acquired frozen-model comparisons.",
    "arms": {}}
for arm in protocol["arms"]:
    nominations, fits, paired, returns, offers, experience = [], [], [], [], [], []
    action_counts = Counter()
    for root in protocol["roots"]:
        with gzip.open(output / "runs" / arm / root / "checkpoint.json.gz", "rt") as handle:
            state = json.load(handle)
        for record in state["results"]:
            position = record["task_position"]
            report = record["research"]
            action_counts.update(action["action"] for action in report["actions"])
            for action in report["actions"]:
                if action["action"] == "distinguish_paid_condition_duration":
                    for item in action.get("competing_structures", []):
                        if (item.get("proposed_structure") == "conditional_duration"
                                and not item.get("condition_identified")
                                and item.get("gain", 0.) > tolerance):
                            nominations.append({"root": root, "position": position,
                                **{key: value for key, value in item.items()
                                   if key not in {"parameter_trials", "duration_trials"}}})
                elif (action["action"] == "refit_observed_condition_response"
                        and action.get("structure_nominations")):
                    fits.append({"root": root, "position": position, **action})
                elif action["action"] == "compare_successor_method_on_fresh_response":
                    paired.append({"root": root, "position": position, **action})
        for visit in state["revisits"]:
            if visit.get("method_visit_outcome") is not None:
                returns.append({"root": root, "position": visit["task_position"],
                    "source_position": visit["source_task_position"],
                    **visit["method_visit_outcome"]})
        offers.extend({"root": root, **event} for event in state.get("scheduler_events", [])
                      if event.get("method_opportunity") is not None)
        for context, actions in state.get("knowledge", {}).get("action_experience", {}).items():
            if "successor_partial_method_visit" in actions:
                experience.append({"root": root, "context": context,
                                   **actions["successor_partial_method_visit"]})
    scored = [item for item in returns if item["fresh_comparison"] is not None]
    learned_offers = [item for item in offers
                     if item["method_opportunity"].get("observed_visit_trials", 0) > 0]
    result["arms"][arm] = {
        "actual_action_counts": dict(action_counts),
        "unresolved_temporal_nominations": len(nominations),
        "nominated_structure_response_refits": len(fits),
        "nominated_refits_producing_candidates": sum(item["candidate_count"] > 0 for item in fits),
        "nominated_refit_compute": sum(item["compute_cost"] for item in fits),
        "fresh_method_comparisons": len(paired),
        "whole_visit_returns": len(returns), "scored_whole_visit_returns": len(scored),
        "unscored_whole_visit_returns": len(returns) - len(scored),
        "whole_visit_fresh_improved": sum(item["fresh_comparison"]["after_loss"] <
            item["fresh_comparison"]["before_loss"] - tolerance for item in scored),
        "whole_visit_costs": {key: sum(item["whole_visit_costs"][key] for item in returns)
            for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "offers_using_measured_method_experience": len(learned_offers),
        "scheduled_offers_using_measured_method_experience": sum(item["scheduled"] for item in learned_offers),
        "unresolved_nomination_records": nominations, "nominated_refit_records": fits,
        "fresh_method_comparison_records": paired, "whole_visit_return_records": returns,
        "measured_experience_offer_records": learned_offers, "method_experience_records": experience}
(output / "unresolved_research_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in data.items() if not isinstance(value, list)}
       for arm, data in result["arms"].items()})
