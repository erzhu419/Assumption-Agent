#!/usr/bin/env python3
"""Summarize saved v0.15 comparisons without running research or evaluation."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.language import Program


def outcome(gain, threshold=.002):
    return "improved" if gain > threshold else "worsened" if gain < -threshold else "unchanged"


def read_run(directory, label, seed, date):
    prefix = f"hegel_machine_{label}_seed{seed}"
    paired = json.loads((directory / f"{prefix}_paired_{date}.json").read_text())
    sessions = {arm: json.loads((directory / f"{prefix}_{arm}_{date}.json").read_text())
                for arm in ("methods", "ordinary")}
    return paired, sessions


def queries(episode):
    # Probe IDs are labels. The ordered sequences define the actual commands,
    # and each list item retains the independent-trajectory reset boundary.
    return [action["details"]["sequence"] for action in episode["actions"]
            if action["action"] == "observe"]


def program(episode):
    # A different concept name alone is not a different executable expression.
    return Program.from_dict(episode["program"]).expanded().to_dict()


def compare_behavior(left, right):
    def visits(session):
        return {(item["task_index"], item["visit"]): item for item in session["episodes"]}

    a, b = visits(left), visits(right)
    changed = []
    for key in sorted(a.keys() | b.keys()):
        old, new = a.get(key), b.get(key)
        differences = {}
        if old is None or new is None:
            differences["visit"] = "added" if old is None else "removed"
        else:
            for field, first, second in (
                    ("actual_query_sequences", queries(old), queries(new)),
                    ("expanded_program", program(old), program(new)),
                    ("fallback", old["fallback"], new["fallback"]),
                    ("uncertainty_model", old["uncertainty_model"], new["uncertainty_model"]),
                    ("observation_cost", old["observation_cost"], new["observation_cost"]),
                    ("compute_cost", old["compute_cost"], new["compute_cost"])):
                if first != second:
                    differences[field] = True
        if differences:
            changed.append({"task_index": key[0], "task_id": (new or old)["task_id"],
                            "visit": key[1], "differences": differences})
    counts = Counter(field for item in changed for field in item["differences"])
    return {"compared_visits": len(a.keys() & b.keys()),
            "visit_order_changed": list(a) != list(b),
            "changed_visits": len(changed), "change_counts": dict(counts), "changes": changed}


def execution_returns(session):
    continuations, executions = [], []
    for episode in session["episodes"]:
        identity = {"task_index": episode["task_index"], "task_id": episode["task_id"],
                    "visit": episode["visit"], "scheduler_reason": episode["scheduler"]["reason"]}
        for action in episode["actions"]:
            details = action["details"]
            if action["action"] == "execute_method_step":
                executions.append({**identity, "round": action["round"], "details": details})
            if action["action"] == "continuation_feedback":
                continuations.append({**identity, **details,
                    "material_outcome": outcome(details["gain"], session["config"]["minimum_improvement"])})
    return {"executed_method_steps": len(executions), "executions": executions,
            "continuation_outcomes": dict(Counter(row["material_outcome"] for row in continuations)),
            "continuations": continuations,
            "interpretation": "Continuation gains are measured on adaptive learning trajectories. Independent revisit outcomes remain in the saved paired-study summary."}


def loss_changes(old, new, arm):
    original = {row["task_index"]: row for row in old["summary"]["tasks"]}
    changes = []
    threshold = new["summary"]["material_improvement_threshold"]
    for row in new["summary"]["tasks"]:
        prior = original[row["task_index"]]
        if (prior["task_id"], prior["partition"]) != (row["task_id"], row["partition"]):
            raise ValueError("comparison needs identical public task ordering")
        before, after = prior[f"{arm}_loss"], row[f"{arm}_loss"]
        record = {"task_index": row["task_index"], "task_id": row["task_id"],
                  "partition": row["partition"], "loss_kind": row["loss_kind"],
                  "before": before, "after": after, "gain": before-after,
                  "material_outcome": outcome(before-after, threshold)}
        if "decision" in row:
            record["decision_gain"] = prior["decision"][f"{arm}_loss"] - row["decision"][f"{arm}_loss"]
        changes.append(record)
    groups = {}
    for kind in sorted({row["loss_kind"] for row in changes}):
        selected = [row for row in changes if row["loss_kind"] == kind]
        before, after = mean(row["before"] for row in selected), mean(row["after"] for row in selected)
        groups[kind] = {"tasks": len(selected), "before": before, "after": after,
                        "relative_loss_change": (after-before)/before if before else None,
                        "material_outcomes": dict(Counter(row["material_outcome"] for row in selected))}
    return {"by_prediction_loss": groups, "tasks": changes}


def summarize(directory, label, seed, date):
    baseline = "v014_mainline" if seed == 919 else "v015_baseline"
    old, old_sessions = read_run(directory, baseline, seed, date)
    new, sessions = read_run(directory, label, seed, date)
    return {"seed": seed, "baseline_label": baseline, "new_label": label,
            "instance_status": "development regression" if seed == 919 else "first use in this development round; existing public generator",
            "before": {key: value for key, value in old["summary"].items() if key != "tasks"},
            "after": {key: value for key, value in new["summary"].items() if key != "tasks"},
            "cross_version": {arm: {"loss": loss_changes(old, new, arm),
                "behavior": compare_behavior(old_sessions[arm], sessions[arm])} for arm in sessions},
            "methods_vs_ordinary": compare_behavior(sessions["ordinary"], sessions["methods"]),
            "execution_returns": {arm: execution_returns(session) for arm, session in sessions.items()},
            "interpretation": "The policy comparison includes direct and downstream shared-memory effects. Method labels alone do not count as query or executable-program differences. All metrics are read from completed saved studies; this helper acquires no observations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(919, 1237), required=True)
    parser.add_argument("--label", default="v015_mainline")
    parser.add_argument("--date", default="20260908")
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    report = summarize(args.directory, args.label, args.seed, args.date)
    destination = args.directory / f"hegel_machine_{args.label}_seed{args.seed}_comparison_{args.date}.json"
    with destination.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(destination), "seed": args.seed,
        "cross_version": {arm: values["loss"]["by_prediction_loss"] for arm, values in report["cross_version"].items()},
        "methods_vs_ordinary": {key: value for key, value in report["methods_vs_ordinary"].items() if key != "changes"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
