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
    return tables


def generate_differences(root=ROOT):
    """Recompute adjacent reported cost differences from the same saved summaries."""
    data = saved_results(root)
    retention = data["retention"]["totals"]
    retained, removed = (retention[key]["costs"]
                         for key in ("conditional_retention", "global_discard"))
    old, revised = (data["preservation"][key]["costs"]
                    for key in ("original_method", "revised_method"))
    return {
        "retention_sequence_cost_reduction_percent": {
            key: 100 * (removed[key] - retained[key]) / removed[key] for key in retained},
        "selective_vs_refit_both_compute_reduction_percent":
            100 * (old["compute_cost"] - revised["compute_cost"]) / old["compute_cost"],
        "selective_vs_refit_both_additional_observations":
            revised["observation_cost"] - old["observation_cost"],
        "selective_vs_refit_both_measurement_difference":
            revised["measurement_cost"] - old["measurement_cost"],
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
