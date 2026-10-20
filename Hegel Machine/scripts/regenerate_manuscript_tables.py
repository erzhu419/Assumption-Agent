#!/usr/bin/env python3
"""Regenerate quantitative main-paper tables from saved results, without model calls."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_FILES = {
    "agents": "online_inquiry_luna_subscription_20261005_main_30/summary.json",
    "agent_protocol": "online_inquiry_luna_subscription_20261005_main_30/protocol.json",
    "parent_revision": "temporal_counterexample_learning_luna_20261006/summary.json",
    "scope_revision": "temporal_evidence_scope_learning_luna_20261006/summary.json",
    "initial_selective": "temporal_selective_revision_luna_20261006/summary.json",
    "preservation": "temporal_selective_preservation_luna_20261006/summary.json",
    "retention": "paper_scoped_retention_20261005/summary.json",
    "sc_kg": "sc_kg_structural_prior_comparison_20261007/summary.json",
    "recurrent_method": "recurrent_method_learning_luna_20261012/summary.json",
    "recurrent_continuation": "recurrent_method_continuation_luna_20261012/summary.json",
    "boundary_transfer": "luna_accepted_boundary_transfer_20261016/summary.json",
    "boundary_protocol": "luna_accepted_boundary_transfer_20261016/protocol.json",
    "adaptive_boundary_transfer": "luna_adaptive_boundary_transfer_20261017/summary.json",
    "adaptive_boundary_protocol": "luna_adaptive_boundary_transfer_20261017/protocol.json",
    "window_transfer": "luna_window_parameter_transfer_20261017/summary.json",
    "window_protocol": "luna_window_parameter_transfer_20261017/protocol.json",
}


def saved_results(root):
    return {key: json.loads((Path(root) / "artifacts" / name).read_text())
            for key, name in RESULT_FILES.items()}


def markdown_table(headers, rows, *, text_columns=1):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" if i < text_columns else "---:"
                               for i in range(len(headers))) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def fraction(numerator, denominator):
    return f"{numerator}/{denominator}"


def method_fraction(method):
    return fraction(method["confirmed_and_correct"], len(method["cases"]))


def recurrent_execution_row(execution, agent, trials):
    """Include failed executions in costs; require both confirmations for success."""
    passed = sum(bool(trial.get("fresh_confirmation_passed"))
                 and bool((trial.get("terminal") or {}).get("passed"))
                 for trial in trials)
    totals = {key: sum(trial["costs"][key] for trial in trials)
              for key in ("model_calls", "observation_cost", "python_execution_seconds")}
    return (execution, agent, fraction(passed, len(trials)), totals["model_calls"],
            f"{totals['observation_cost']:,}", f"{totals['python_execution_seconds']:,.1f}")


def complete_parameter_groups(cases, protocol, *, require_confirmation=True):
    """Count apparatus groups, rather than treating their two stages as independent."""
    cases_by_id = {case["case"]: case for case in cases}
    groups = {}
    for case in protocol["transfer_cases"]:
        if "parameter_pair" in case and case["id"] not in protocol.get("old_target_ids", []):
            groups.setdefault(case["apparatus_seed"], []).append(case["id"])
    passed = sum(all(case_id in cases_by_id
                     and bool((cases_by_id[case_id].get("terminal") or {}).get("passed"))
                     and (not require_confirmation
                          or bool(cases_by_id[case_id].get("fresh_confirmation_passed")))
                     for case_id in members) for members in groups.values())
    return fraction(passed, len(groups))


def boundary_transfer_rows(result, protocol):
    rows = []
    for key, label in (("retention", "Earlier cohort / accepted 11/12 method / old 10"),
                       ("new_parameters", "Earlier cohort / accepted 11/12 method / new 4")):
        cases = result[key]
        rows.append((label,
            fraction(sum(bool(case["fresh_confirmation_passed"]) for case in cases), len(cases)),
            fraction(sum(bool(case["terminal"]["passed"]) for case in cases), len(cases)),
            "—" if key == "retention" else complete_parameter_groups(cases, protocol)))
    controls = result["frozen_source_answer"]
    rows.append(("Earlier cohort / frozen source predictor / new 4", "Not run",
        fraction(sum(bool(case["terminal"]["passed"]) for case in controls), len(controls)),
        complete_parameter_groups(controls, protocol, require_confirmation=False)))
    return rows


def adaptive_boundary_transfer_rows(result, protocol):
    rows = []
    for cases, label, groups in (
        (result["retention"], "Fresh cohort / published 12/12 method / old 10", False),
        (result["new_parameters"], "Fresh cohort / published 12/12 method / new 8", True),
        (result["paired_baseline"]["new_parameters"],
         "Fresh cohort / accepted 11/12 method / same 8", True),
    ):
        rows.append((label,
            fraction(sum(bool(case["fresh_confirmation_passed"]) for case in cases), len(cases)),
            fraction(sum(bool(case["terminal"]["passed"]) for case in cases), len(cases)),
            complete_parameter_groups(cases, protocol) if groups else "—"))
    controls = result["frozen_source_answer"]
    rows.append(("Fresh cohort / frozen source predictor / same 8", "Not run",
        fraction(sum(bool(case["terminal"]["passed"]) for case in controls), len(controls)),
        complete_parameter_groups(controls, protocol, require_confirmation=False)))
    return rows


def window_transfer_rows(result, protocol):
    """Retain all prior task groups and distinguish no model from a measured error."""
    rows = []
    retained = {case["case"]: case for case in result["retention"]}
    for key, label in (("original_ten", "14/14 method / original ten"),
                       ("previously_successful_eight", "14/14 method / earlier boundary eight"),
                       ("previously_successful_nested_eight", "14/14 method / earlier composition eight")):
        cases = [retained[name] for name in protocol["regression_groups"][key]]
        rows.append((label,
            fraction(sum(bool(c["fresh_confirmation_passed"]) for c in cases), len(cases)),
            fraction(sum(bool(c["terminal"]["passed"]) for c in cases), len(cases)), "—",
            f"{sum(c['costs']['observation_cost'] for c in cases):,}"))
    for cases, label in ((result["new_parameters"], "14/14 method / new window eight"),
                         (result["paired_baseline"]["new_parameters"], "13/13 method / same window eight")):
        rows.append((label,
            fraction(sum(bool(c["fresh_confirmation_passed"]) for c in cases), len(cases)),
            fraction(sum(bool(c["terminal"]["passed"]) for c in cases), len(cases)),
            complete_parameter_groups(cases, protocol),
            f"{sum(c['costs']['observation_cost'] for c in cases):,}"))
    controls = result["frozen_source_answer"]
    rows.append(("Fixed source predictor / same window eight", "Not run",
        fraction(sum(bool(c["terminal"]["passed"]) for c in controls), len(controls)),
        complete_parameter_groups(controls, protocol, require_confirmation=False), "—"))
    return rows


def generate_tables(root=ROOT):
    """Return complete Markdown table bodies keyed by main-paper table number.

    Labels describe the published interventions; every numerical cell comes from
    the saved protocol or result JSON. No manuscript text is read as data.
    """
    data = saved_results(root)
    agents, n = data["agents"]["arms"], len(data["agent_protocol"]["cases"])
    tables = {"4": markdown_table(
        ["Knowledge condition", "Source success", "Target success"],
        [(label, fraction(agents[arm]["source_correct"], n),
          fraction(agents[arm]["confirmed_and_correct"] - agents[arm]["source_correct"], n))
         for arm, label in (("code_react", "Ordinary Code/ReAct"),
                            ("textual_memory", "Textual memory"),
                            ("hegel", "Hegel executable research knowledge"))])}

    rows = []
    for key, assumption, revision in (
        ("parent_revision", "A supported instrument suffices for the response",
         "Preserve the instrument; refit and test the parent response"),
        ("scope_revision", "All response records share the current mechanism",
         "Fit on current instrument-supported records"),
    ):
        result = data[key]
        rows.append((assumption, revision,
            f"Old {method_fraction(result['original_method'])}; "
            f"revised {method_fraction(result['revised_method'])}"))
    corrected = method_fraction(data["preservation"]["revised_method"])
    refit_both = method_fraction(data["preservation"]["original_method"])
    final = (f"corrected and refit-both each {corrected}" if corrected == refit_both
             else f"corrected {corrected}; refit-both {refit_both}")
    rows.append(("New-case repair preserves prior abilities",
                 "Test old prediction, then retain or reopen its support",
                 f"Initial {method_fraction(data['initial_selective']['revised_method'])}; {final}"))
    tables["5"] = markdown_table(["Failed procedural assumption", "Executed revision",
                                   "Frozen target result"], rows, text_columns=3)

    retention = data["retention"]
    arms = ("conditional_retention", "unmarked_retention", "global_discard")
    totals, count = retention["totals"], retention["planned_cases"]
    phases = ("First changed task", "Original-condition return (primary)", "Changed-condition return")
    rows = [[label] + [fraction(totals[arm]["successors"][index]["solved"], count)
                        for arm in arms] for index, label in enumerate(phases)]
    rows.append(["All continuation tasks"] + [fraction(
        sum(row["solved"] for row in totals[arm]["successors"]), count * len(phases)) for arm in arms])
    rows.append(["Full sequence, including shared prefix"] + [
        fraction(totals[arm]["solved"], totals[arm]["tasks"]) for arm in arms])
    rows.extend([label] + [f"{totals[arm]['costs'][key]:,}" for arm in arms]
                for key, label in (("observation_cost", "Research observations"),
                                   ("measurement_cost", "Measurement units"),
                                   ("compute_cost", "Charged computation")))
    tables["6"] = markdown_table(["Outcome", "Conditional retention", "Unmarked retention",
                                   "Global discard"], rows)

    kg, rows = data["sc_kg"], []
    for arm, label in (("native_quadratic", "Native SC-KG"),
                       ("historical_full_structural_prior", "Historical full four-property configuration"),
                       ("hegel_goal_completion", "Frozen Hegel v0.53")):
        values = kg["profile_results"]["overall"][arm]
        wins = kg["hegel_paired_feasibility_first"].get(arm)
        rows.append((label, f"{values['mean_regret']:.6f}",
                     fraction(values["feasible"], values["cases"]),
                     "—" if wins is None else fraction(wins.get("win", 0), values["cases"])))
    tables["7"] = markdown_table(["Optimizer prior system", "Mean terminal regret",
                                   "Feasible recommendations", "Hegel paired wins"], rows)

    recurrent = data["recurrent_method"]
    rows = [recurrent_execution_row("Frozen targets", "Published procedure", recurrent["frozen"])]
    for execution, trials in (("Original live", recurrent["live_controls"]["trials"]),
                              ("Development rerun", data["recurrent_continuation"]["trials"])):
        rows.extend(recurrent_execution_row(execution, label,
                    [trial for trial in trials if trial["arm"] == arm])
                    for arm, label in (("code_react", "Code"), ("hegel", "Hegel")))
    tables["8"] = markdown_table(["Execution", "Agent", "Passed", "LLM calls",
                                   "Response rows", "Python seconds"], rows, text_columns=2)
    tables["9"] = markdown_table(
        ["Cohort / program", "Confirmation", "Terminal", "Complete parameter groups"],
        boundary_transfer_rows(data["boundary_transfer"], data["boundary_protocol"])
        + adaptive_boundary_transfer_rows(data["adaptive_boundary_transfer"],
                                          data["adaptive_boundary_protocol"]))
    tables["10"] = markdown_table(
        ["Program / cohort", "Confirmation", "Terminal", "Complete parameter groups", "Response rows"],
        window_transfer_rows(data["window_transfer"], data["window_protocol"]))
    return tables


def generate_differences(root=ROOT):
    """Recompute adjacent reported cost differences from the same saved summaries."""
    data = saved_results(root)
    retention = data["retention"]["totals"]
    retained, removed = (retention[key]["costs"]
                         for key in ("conditional_retention", "global_discard"))
    old, revised = (data["preservation"][key]["costs"]
                    for key in ("original_method", "revised_method"))
    adaptive = data["adaptive_boundary_transfer"]
    new_costs = {key: sum(case["costs"][key] for case in adaptive["new_parameters"])
                 for key in ("observation_cost", "compute_cost", "model_calls")}
    baseline_costs = {key: sum(case["costs"][key]
                              for case in adaptive["paired_baseline"]["new_parameters"])
                      for key in new_costs}
    return {
        "retention_sequence_cost_reduction_percent": {
            key: 100 * (removed[key] - retained[key]) / removed[key] for key in retained},
        "selective_vs_refit_both_compute_reduction_percent":
            100 * (old["compute_cost"] - revised["compute_cost"]) / old["compute_cost"],
        "selective_vs_refit_both_additional_observations":
            revised["observation_cost"] - old["observation_cost"],
        "selective_vs_refit_both_measurement_difference":
            revised["measurement_cost"] - old["measurement_cost"],
        "adaptive_boundary_new_parameter_costs": new_costs,
        "accepted_boundary_same_parameter_costs": baseline_costs,
        "adaptive_boundary_observation_increase_percent":
            100 * (new_costs["observation_cost"] - baseline_costs["observation_cost"])
            / baseline_costs["observation_cost"],
        "adaptive_boundary_compute_ratio":
            new_costs["compute_cost"] / baseline_costs["compute_cost"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Write Markdown; otherwise print it")
    parser.add_argument("--json-output", type=Path, help="Also save table bodies, source paths and differences")
    args = parser.parse_args()
    tables = generate_tables(args.root)
    text = "\n\n".join(f"**Table {number}.**\n\n{body}" for number, body in tables.items()) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.json_output:
        args.json_output.write_text(json.dumps({
            "tables": tables, "differences": generate_differences(args.root),
            "saved_result_sources": {key: f"artifacts/{name}" for key, name in RESULT_FILES.items()},
            "new_model_calls": 0, "new_experiments": 0}, indent=2) + "\n")


if __name__ == "__main__":
    main()
