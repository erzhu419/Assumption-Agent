#!/usr/bin/env python3
"""Check constraint mechanism rewrites on three new equal-budget tasks."""
import argparse
from copy import deepcopy
from pathlib import Path
import json

import run_sc_kg_constraint_knowledge_study as parent

study, previous, prior = parent.study, parent.previous, parent.prior
SEEDS = (2546016007, 2546116031, 2546216053)


def prepare(output, source):
    selected = study.read(source / "protocol.json")
    group = selected["groups"]["historical_retention"]
    original = study.read(group["source_protocol"])
    template = next(case for case in group["cases"]
        if case["case_id"] == "active_noise_2_mechanism_change_noise_0.03")
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    cases = []
    for index, seed in enumerate(SEEDS, 1):
        task = {**template["task"], "seed": seed}
        cases.append({"case_id": f"constraint_rewrite_{index}_mechanism_change",
            "split": "confirmation", "profile": "mechanism_change", "task": task,
            "next_problem": template["next_problem"],
            "research_id": f"constraint_rewrite_{index}_initial"})
    spec.update(experiment="independent_constraint_mechanism_rewrite", cases=cases,
        seeds=list(SEEDS), arms=[parent.BASELINE, parent.METHOD],
        source_v056=selected["source_v056"], method_source=selected["method_source"],
        source_protocol=str((source / "protocol.json").resolve()),
        method=selected["method"],
        accounting={"optimizer_runs": 6, "initial_calls": 96, "online_calls": 192,
            "evaluation_calls": 384, "physical_calls": 672},
        trigger="Run only if the preceding nine new goal tasks do not adequately exercise accepted constraint rebases.",
        limitations="Three new seeds from one fixed mechanism-change task family. Both arms share new paid initialization, with no old labels reused. No native arm is added to this focused representation comparison.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("saved constraint-rewrite confirmation differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def execute(spec, output):
    parent.configure_arms()
    previous.METHOD = parent.BASELINE
    baseline, _ = previous.load_method({**spec, "method_source": spec["source_v056"]},
        "historical_constraint_rewrite_baseline")
    initial = study.Program, study.MetaPriorResearch, study.SCKGEnvironment
    previous.METHOD = parent.METHOD
    current, evaluator = previous.load_method(spec, "historical_constraint_rewrite_method")
    study.Program, study.MetaPriorResearch, study.SCKGEnvironment = initial
    study.ARMS = tuple(spec["arms"])
    research = study.acquire_research(spec, output, spec["cases"])
    rows = study.optimize(spec, output, spec["cases"], research,
        arm_modules={**baseline, **current})
    evaluations = study.evaluate(spec, output, spec["cases"], rows, environment_factory=evaluator)
    for row in rows:
        study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    comparisons = [{"case_id": case["case_id"], "profile": case["profile"],
        "arms": {row["arm"]: previous.values(row) for row in rows
            if row["case_id"] == case["case_id"]}} for case in spec["cases"]]
    matched = {arm: [row for row in rows if row["arm"] == arm] for arm in spec["arms"]}
    initial_calls = sum(row["actual_new_initial_calls"] for row in research.values())
    online_calls = sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows)
    evaluation_calls = sum(row["simulator_calls"] for row in evaluations.values())
    mechanisms = []
    for row in rows:
        retention = row["costs"]["recovery_diagnostics"].get("constraint_retention", {})
        mechanisms.append({"case_id": row["case_id"], "arm": row["arm"],
            "constraint_scope_revisions": len(row.get("scope_revisions", [])),
            "retention": retention})
    brief = {"protocol": str((output / "protocol.json").resolve()),
        "completed_cases": len(comparisons), "optimizer_runs": len(rows),
        "comparisons": comparisons, "metrics": prior.comparison_metrics(comparisons, matched),
        "costs": {"initial_calls": initial_calls, "online_calls": online_calls,
            "evaluation_calls": evaluation_calls,
            "physical_calls": initial_calls + online_calls + evaluation_calls,
            "online_by_arm": {arm: prior.work(values) for arm, values in matched.items()}},
        "mechanisms": mechanisms, "limitations": spec["limitations"]}
    study.write(output / "result_brief.json", brief)
    print(json.dumps({"metrics": brief["metrics"], "costs": brief["costs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    spec = prepare(args.output, args.source)
    if args.phase == "run":
        execute(spec, args.output)
    else:
        print(json.dumps({"status": "prepared_not_run", "accounting": spec["accounting"],
            "seeds": spec["seeds"]}), flush=True)


if __name__ == "__main__":
    main()
