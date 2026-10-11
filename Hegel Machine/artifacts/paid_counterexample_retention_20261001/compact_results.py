#!/usr/bin/env python3
"""Collect frozen cohort results without conflating learning and terminal fees."""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
result = {"baseline_version": "0.143.0", "candidate_version": "0.145.0",
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
        data["arms"][arm]["competition"] = knowledge["arms"][arm]["competition"]
        data["arms"][arm]["method_returns"] = {key: value for key, value in
            methods["arms"][arm].items() if not isinstance(value, list)}
        data["arms"][arm]["supported_fits"] = {key: value for key, value in
            supported["arms"][arm].items() if not isinstance(value, list)}
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
