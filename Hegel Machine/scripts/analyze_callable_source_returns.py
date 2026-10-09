#!/usr/bin/env python3
"""Extract source bindings and measured returns from a fixed-session rerun."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.language import Program


def read(path):
    return json.loads(path.read_text())


def calls(node):
    found = [node["name"]] if node.get("op") == "call" else []
    return found + [name for child in node.get("children", []) for name in calls(child)]


def same_program(left, right):
    return (left is not None and right is not None and
        Program.from_dict(left).expanded().simplified() ==
        Program.from_dict(right).expanded().simplified())


def compact_receipt(receipt, program):
    fields = ("source_id", "template_id", "mapping", "mapping_roles", "episode_id",
              "accepted", "retained", "final_status", "value_status", "confirmation_status",
              "confirmation_replicates", "paired_gain", "retained_normalized_gain",
              "adaptation_compute", "comparison_compute", "joint_source_ids")
    return {**{key: receipt.get(key) for key in fields},
        "matches_final_program": same_program(receipt.get("retained_program"), program),
        "comparisons": [{key: item.get(key) for key in ("comparison_id", "loss", "baseline_loss",
            "accepted", "evidence_rows", "compute_cost")} for item in receipt.get("comparisons", [])]}


def analyze(output, previous):
    rows = []
    for case in read(output / "protocol.json")["cases"]:
        case_name = f"case_{case['apparatus_seed']}"
        result = read(output / "cohort" / case_name / "results.json")
        old_result = read(previous / "cohort" / case_name / "results.json")
        for arm in ("retained", "removed"):
            state = read(output / "cohort" / case_name / arm / "confirmed_session.json")
            training = read(output / "cohort" / case_name / arm / "training.json")
            old = read(previous / "cohort" / case_name / arm / "confirmed_session.json")
            for task, (entry, old_entry) in enumerate(zip(state["results"], old["results"])):
                research = entry["research"]
                program = research.get("program")
                returned = research.get("knowledge_return") or {}
                measured = [compact_receipt(item, program) for item in returned.get("returns", [])]
                native = [compact_receipt(item, program) for item in
                    (training["results"][task]["research"].get("knowledge_return") or {}).get("returns", [])]
                history = [compact_receipt(item, program)
                    for concept in state["knowledge"]["concepts"]
                    for item in concept.get("reuse_history", [])
                    if item.get("kind") == "research_return" and item.get("task_id") == research["task_id"]]
                workflow = [{"method_id": method["method_id"],
                    **{key: item.get(key) for key in ("source_concept_id", "episode_id",
                        "retained_gain", "measured_retained_gain", "exclusive_compute", "shared_compute")},
                    "source_return_statuses": dict(Counter(source.get("confirmation_status", "unspecified")
                        for source in item.get("source_returns", [])))}
                    for method in state["knowledge"]["research_workflows"]["methods"]
                    for item in method.get("knowledge_outcomes", [])
                    if item.get("downstream_task") == research["task_id"]]
                scored = result["results"][arm]["tasks"][task]
                rows.append({"apparatus_seed": case["apparatus_seed"], "arm": arm,
                    "task_position": task, "role": scored["role"],
                    "selected_kind": research.get("selected_kind"),
                    "final_call_definitions": sorted(set(calls(program))),
                    "expression": research.get("expression"),
                    "identical_final_program_as_previous": program == old_entry["research"].get("program"),
                    "same_final_program_as_previous": same_program(program, old_entry["research"].get("program")),
                    "confirmed_and_correct": scored["confirmed_and_correct"],
                    "fresh_confirmation": {key: scored["confirmation"].get("fresh_confirmation", {}).get(key)
                                           for key in ("passed", "mse", "rows", "observation_indices")},
                    "costs": scored["costs"],
                    "cost_delta_from_previous": {key: scored["costs"][key] -
                        old_result["results"][arm]["tasks"][task]["costs"][key]
                        for key in ("observation_cost", "measurement_cost", "compute_cost")},
                    "native_source_returns": native, "source_returns": measured,
                    "source_return_history": history, "workflow_knowledge_outcomes": workflow})
    receipts = [receipt for row in rows for receipt in row["source_returns"]]
    native_receipts = [receipt for row in rows for receipt in row["native_source_returns"]]
    history = [receipt for row in rows for receipt in row["source_return_history"]]
    coverage = {}
    for arm in ("retained", "removed"):
        consumers = [row for row in rows if row["arm"] == arm and row["role"] == "consumer"]
        coverage[arm] = {"apparatuses": len(consumers),
            "confirmed_positive_source_return_cases": sum(any(
                item["confirmation_status"] == "confirmed" and item["retained_normalized_gain"] > 0
                for item in row["source_returns"]) for row in consumers),
            "positive_workflow_return_cases": sum(any(
                (item.get("measured_retained_gain") or 0) > 0
                for item in row["workflow_knowledge_outcomes"]) for row in consumers)}
    return {"scope": "Same 12 apparatuses and unchanged protocol; attribution repair rerun, not new independent cases.",
        "case_coverage": coverage,
        "programs_identical": sum(row["identical_final_program_as_previous"] for row in rows),
        "programs_unchanged": sum(row["same_final_program_as_previous"] for row in rows),
        "programs_total": len(rows), "source_return_statuses": dict(Counter(
            item["confirmation_status"] for item in receipts)),
        "native_return_statuses": dict(Counter(item["confirmation_status"] for item in native_receipts)),
        "final_history_return_statuses": dict(Counter(item["confirmation_status"] for item in history)),
        "final_history_value_statuses": dict(Counter(item["value_status"] for item in history)),
        "returns_matching_final_program": sum(item["matches_final_program"] for item in receipts),
        "returns_total": len(receipts), "tasks": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze(args.output, args.previous)
    (args.output / "source_return_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({key: value for key, value in analysis.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
