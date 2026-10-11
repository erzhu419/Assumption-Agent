#!/usr/bin/env python3
"""Read frozen negative cases on the already fixed terminal trajectories."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_knowledge_lineage_study import read, load_runtime
from run_observed_condition_decoding_study import laboratory

output = Path(sys.argv[1]).resolve()
protocol = read(output / "protocol.json")
load_runtime(protocol, "new_cumulative")
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program


def score(task, report):
    settings = protocol["terminal"]
    program = Program.from_dict(report["program"])
    errors = []
    with laboratory({**task, "process_seed": task["seed"] + settings["seed_offset"]}) as lab:
        for index in range(settings["trajectories"]):
            probe = lab.random_probe(task["seed"] + settings["seed_offset"] + 10007 * index,
                length=settings["length"], name=f"terminal_{index}")
            rows = lab.acquire(probe, report["selected_channels"]).observations
            errors.append(loss(program, rows, report["fallback"], .5))
    return sum(errors) / len(errors)


def selected(report):
    return {key: report.get(key) for key in ("selected_kind", "selected_channels", "selection_loss",
        "confirmation_loss", "costs", "used_sources", "method_source_ids")}


def recorded_steps(report):
    result = []
    for action in report["actions"]:
        if action["action"] == "distinguish_paid_condition_duration":
            nominees = action.get("competing_structures", [])
            result.append({key: action.get(key) for key in ("action", "measured_rows", "condition_field",
                "fixed_controls", "response_supported", "compute_cost")})
            result[-1]["nominees"] = [{key: item.get(key) for key in
                ("status", "proposed_structure", "loss", "width", "placement", "threshold_interval")}
                for item in nominees]
        elif action["action"] in ("choose_condition_response_research", "validate_continued_program",
                "execute_successor_method_experiment", "preserve_better_prior_explanation"):
            result.append({key: action.get(key) for key in ("action", "selected_action", "reason",
                "observation_remaining", "measurement_remaining", "compute_remaining", "loss", "accepted",
                "observation_rows", "incumbent_loss", "proposed_loss", "compute_cost") if key in action})
    return result


cases = []
for root, position in (("root09", 4), ("root08", 3)):
    directory = output / "runs" / "new_cumulative" / root
    current = read(directory / "checkpoint.json.gz")
    baseline = read(output / "runs" / "old_cumulative" / root / "checkpoint.json.gz")
    task = current["tasks"][position]
    final = current["results"][position]["research"]
    old = baseline["results"][position]["research"]
    revisits = [visit for visit in current["revisits"] if visit["task_position"] == position]
    first = revisits[0]["previous"] if revisits else final
    case = {"root": root, "position": position, "stage": protocol["stages"][position],
        "old": {**selected(old), "same_terminal_mse": score(task, old)},
        "candidate_before_revisits": {**selected(first), "same_terminal_mse": score(task, first)},
        "candidate_final": {**selected(final), "same_terminal_mse": score(task, final)},
        "initial_steps": recorded_steps(first), "revisit_steps": recorded_steps(final)[len(recorded_steps(first)):],
        "revisits": [{"source_position": visit["source_task_position"],
            "kind": visit["candidate"]["kind"], "basis": visit.get("basis"),
            "costs": visit["visit_costs"], "limits": visit["resource_limits"]} for visit in revisits],
        "successor_events": [{key: event.get(key) for key in ("action", "source_task_position", "scheduled",
            "basis", "reason", "old_evidence_gain", "compute_cost", "minimum_resources") if key in event}
            for event in current.get("scheduler_events", []) if event.get("task_position") == position],
        "method_attempts": [{"draft_id": attempt["draft_id"], "deferred": "deferred_offer" in attempt}
            for attempt in current.get("method_revisit_attempts", []) if attempt["task_position"] == position],
        "chain_costs": current["costs"],
        "unused_chain_resources": {key: protocol["limits"][f"chain_{name}"] - current["costs"][key]
            for name, key in (("observation", "observation_cost"), ("measurement", "measurement_cost"),
                             ("compute", "compute_cost"))}}
    cases.append(case)
result = {"basis": "Post-terminal read-only diagnosis. Frozen first-visit, final and old models use the exact already fixed terminal raw probes and process draws. No learning, refit or protocol/source changes.", "cases": cases}
(output / "read_only_regression_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
print([{key: case[key] for key in ("root", "position", "old", "candidate_before_revisits", "candidate_final", "revisits", "unused_chain_resources")} for case in cases])
