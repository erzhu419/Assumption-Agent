#!/usr/bin/env python3
"""Collect frozen cohort results without conflating learning and terminal fees."""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
result = {"baseline_version": "0.149.0", "candidate_version": "0.150.0",
          "external_model_calls": 0, "cohorts": {}}
for cohort in ("development", "fresh_replication"):
    directory = output / cohort
    summary = json.loads((directory / "summary.json").read_text())
    chains = json.loads((directory / "chain_comparisons.json").read_text())
    knowledge = json.loads((directory / "knowledge_consumption.json").read_text())
    methods = json.loads((directory / "unresolved_research_analysis.json").read_text())
    supported = json.loads((directory / "supported_priority_analysis.json").read_text())
    totals = summary["totals"]
    data = {"development_only": cohort == "development", "arms": {}, "comparisons": {}}
    for arm, total in totals.items():
        data["arms"][arm] = {key: total[key] for key in (
            "tasks", "completed", "solved", "false_completed", "mean_response_mse",
            "learning_cpu_seconds", "terminal_cpu_seconds", "learning_costs",
            "terminal_costs", "all_costs", "revisits")}
        data["arms"][arm]["all_cpu_seconds"] = total["learning_cpu_seconds"] + total["terminal_cpu_seconds"]
        data["arms"][arm]["competition"] = knowledge["arms"][arm]["competition"]
        data["arms"][arm]["method_returns"] = {key: value for key, value in
            methods["arms"][arm].items() if not isinstance(value, list)}
        data["arms"][arm]["supported_fits"] = {key: value for key, value in
            supported["arms"][arm].items() if not isinstance(value, list)}
        executed = []
        for index, visit in enumerate(supported["arms"][arm]["method_visit_records"]):
            fresh = (visit.get("outcome") or {}).get("fresh_comparison")
            source_position = visit["source_position"]
            direction = ("unknown_source" if source_position is None else
                "predecessor_return" if source_position > visit["position"] else
                "successor_transfer" if source_position < visit["position"] else "same_task")
            executed.append({"visit_index": index, "root": visit["root"],
                "source_position": source_position, "destination_position": visit["position"],
                "direction": direction,
                "fresh_rows": fresh["observation_rows"] if fresh else 0,
                "before_loss": fresh["before_loss"] if fresh else None,
                "after_loss": fresh["after_loss"] if fresh else None,
                "visit_costs": visit["costs"], "screening_compute_cost": visit["screening_compute_cost"],
                "allocation": visit["resource_allocation"]})
        data["arms"][arm]["executed_method_chain"] = executed
        data["arms"][arm]["executed_method_directions"] = {
            direction: {"visits": sum(v["direction"] == direction for v in executed),
                "fresh_confirmed": sum(v["direction"] == direction and v["fresh_rows"] > 0 for v in executed),
                "fresh_improved": sum(v["direction"] == direction and v["fresh_rows"] > 0 and
                    v["after_loss"] + 1e-7 < v["before_loss"] for v in executed)}
            for direction in ("predecessor_return", "successor_transfer", "unknown_source", "same_task")}
    for arm, comparison in summary["comparisons"].items():
        current, reference = totals["new_cumulative"], totals[arm]
        data["comparisons"][arm] = {
            **comparison,
            "chain_pairs": chains[arm],
            "solved_difference": current["solved"] - reference["solved"],
            "learning_cost_ratios": {key: current["learning_costs"][key] /
                reference["learning_costs"][key] for key in current["learning_costs"]},
            "all_cost_ratios": {key: current["all_costs"][key] /
                reference["all_costs"][key] for key in current["all_costs"]}}
    investment = directory / "experiment_investment_analysis.json"
    if investment.exists():
        data["experiment_investment"] = {arm: {key: value for key, value in analysis.items()
            if not isinstance(value, list)} for arm, analysis in
            json.loads(investment.read_text())["arms"].items()}
    retention = directory / "paid_retention_analysis.json"
    if retention.exists():
        data["paid_counterexample_retention"] = {arm: {key: value for key, value in analysis.items()
            if not isinstance(value, list)} for arm, analysis in
            json.loads(retention.read_text())["arms"].items()}
    result["cohorts"][cohort] = data
(output / "compact_results.json").write_text(json.dumps(result, indent=2) + "\n")
for cohort, data in result["cohorts"].items():
    print({"cohort": cohort,
        "arms": {arm: {key: total[key] for key in ("solved", "tasks", "false_completed",
            "mean_response_mse", "learning_cpu_seconds", "learning_costs")}
            for arm, total in data["arms"].items()},
        "comparisons": {arm: {key: comparison[key] for key in ("wins", "ties", "losses",
            "mean_mse_change", "research_cpu_ratio", "learning_cost_ratios")}
            for arm, comparison in data["comparisons"].items()}})
