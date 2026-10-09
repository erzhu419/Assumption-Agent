#!/usr/bin/env python3
"""Actual source learning and paired retained/removed experiment-method transfer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Full learning traces repeat executable programs; keep them complete without
    # expanding every node onto separate lines in the larger paired studies.
    formatting = {"separators": (",", ":")} if path.name == "report.json" else {"indent": 2}
    path.write_text(json.dumps(payload, ensure_ascii=False, **formatting) + "\n")


def run_learning(task, limits, config, knowledge, runtime):
    sys.path.insert(0, str(ROOT / runtime))
    from hegel_machine import __version__
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_session import laboratory_for

    config = replace(ResearchConfig(**config), seed=task["seed"],
        observation_budget=limits["observation_budget"], compute_budget=limits["compute_budget"])
    library = KnowledgeLibrary.from_dict(knowledge) if knowledge else KnowledgeLibrary()
    wall, cpu = time.monotonic(), time.process_time()
    with laboratory_for(task) as laboratory:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=limits["measurement_budget"]).run(laboratory)
    return {"version": __version__, "runtime": runtime, "task": task, "limits": limits,
        "research": report, "knowledge": library.to_dict(),
        "elapsed_seconds": time.monotonic() - wall, "cpu_seconds": time.process_time() - cpu}


def source_report(output, protocol, root):
    existing = protocol["roots"][root].get("source_report")
    return ROOT / existing if existing else output / "sources" / root / "report.json"


def learn_source(output, protocol, root):
    destination = source_report(output, protocol, root)
    if destination.exists():
        return
    entry = protocol["roots"][root]
    result = run_learning(entry["source_task"], protocol["source_limits"], protocol["config"],
        None, protocol["source_runtime"])
    save(destination, result)
    print(json.dumps({"source": root, "status": result["research"]["status"],
        "methods": len(result["knowledge"]["program_experiment_methods"]),
        "costs": result["research"]["costs"]}), flush=True)


def learn(output, protocol, arm, case):
    destination = output / "runs" / arm / case / "report.json"
    if destination.exists():
        return
    declared = protocol["cases"][case]
    source = json.loads(source_report(output, protocol, declared["root"]).read_text())
    knowledge = deepcopy(source["knowledge"])
    original_methods = len(knowledge["program_experiment_methods"])
    if protocol["arms"][arm]["remove_program_experiment_methods"]:
        knowledge["program_experiment_methods"] = []
    result = run_learning(declared["task"], declared["limits"], protocol["config"],
        knowledge, protocol["arms"][arm]["runtime"])
    result["program_experiment_methods"] = result.pop("knowledge")["program_experiment_methods"]
    result.update(arm=arm, case=case, root=declared["root"],
        source_report=str(source_report(output, protocol, declared["root"]).relative_to(ROOT)),
        source_method_count=original_methods,
        retained_source_method_count=len(knowledge["program_experiment_methods"]))
    save(destination, result)
    print(json.dumps({"arm": arm, "case": case, "status": result["research"]["status"],
        "costs": result["research"]["costs"],
        **method_metrics(result["research"], result["program_experiment_methods"])}), flush=True)


def method_metrics(report, methods=()):
    actions = report["actions"]
    records = {method["method_id"]: method for method in methods}
    rewrites = [a for a in actions if a["action"] == "learn_program_experiment_method"
        and a.get("response_context_revision") and a.get("parent_method_id") in records
        and any(s["task_id"] == a["task_id"] and s["status"] in ("distinguished", "all_refuted")
            and s.get("calibration_status") != "counterexample" and s.get("observation_rows", 0) > 0
            for s in records.get(a["method_id"], {}).get("support", []))]
    plans = [a for a in actions if a["action"].startswith("plan_") and "experiment" in a["action"]]
    return {"method_uses": sum(a["action"] == "use_program_experiment_method" for a in actions),
        "cross_task_method_uses": sum(a["action"] == "use_program_experiment_method"
            and a.get("cross_task", False) for a in actions),
        "method_plan_attempts": sum(a.get("proposal_source") == "learned_method" for a in actions),
        "experiment_planning_compute": sum(a.get("compute_cost", 0) for a in plans),
        "method_generation_compute": sum(a.get("generation_cost", 0) for a in plans),
        "method_search_compute": sum(a.get("method_search_cost", 0) for a in plans),
        "response_background_rewrites": len(rewrites),
        "cross_task_background_rewrites": sum(records[a["parent_method_id"]]["source_task_id"]
            != a["task_id"] for a in rewrites),
        "methods_learned": sum(a["action"] == "learn_program_experiment_method" for a in actions),
        "method_revisions": sum(a["action"] == "revise_program_experiment_method" for a in actions)}


def evaluate(output, protocol, arm, case):
    missing = [(a, c) for a in protocol["arms"] for c in protocol["cases"]
        if not (output / "runs" / a / c / "report.json").exists()]
    if missing:
        raise ValueError(f"all learning must finish before terminal evaluation: {len(missing)} missing")
    destination = output / "runs" / arm / case / "evaluation.json"
    if destination.exists():
        return
    sys.path.insert(0, str(ROOT / protocol["arms"][arm]["runtime"]))
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.unified_session import laboratory_for

    saved = json.loads((destination.parent / "report.json").read_text())
    report, terminal = saved["research"], protocol["terminal"]
    with laboratory_for(saved["task"]) as laboratory:
        rows = laboratory.evaluation_rows(report["selected_channels"],
            seed=saved["task"]["seed"] + terminal["seed_offset"],
            length=terminal["length"], trajectories=terminal["trajectories"])
        rate = sum(c.cost_per_row for c in laboratory.channels if c.name in report["selected_channels"])
        mse = loss(Program.from_dict(report["program"]), rows, report["fallback"], .5)
    save(destination, {"mse": mse, "rows": len(rows), "observation_cost": len(rows),
        "measurement_cost": len(rows) * rate, "compute_cost": len(rows)})


def summarize(output, protocol):
    rows = []
    for arm in protocol["arms"]:
        for case, declared in protocol["cases"].items():
            folder = output / "runs" / arm / case
            if not (folder / "report.json").exists():
                continue
            saved = json.loads((folder / "report.json").read_text())
            report = saved["research"]
            methods = saved.get("program_experiment_methods",
                saved.get("knowledge", {}).get("program_experiment_methods", []))
            evaluation = json.loads((folder / "evaluation.json").read_text()) if (folder / "evaluation.json").exists() else None
            rows.append({"arm": arm, "case": case, "root": declared["root"],
                "condition": declared["condition"], "budget": declared["limits"]["observation_budget"],
                "status": report["status"], "costs": report["costs"], "evaluation": evaluation,
                "retained_source_method_count": saved["retained_source_method_count"],
                "cpu_seconds": saved["cpu_seconds"], **method_metrics(report, methods)})
    groups, comparisons = {}, []
    for condition, budget in sorted({(row["condition"], row["budget"]) for row in rows}):
        arms = {}
        for arm in protocol["arms"]:
            selected = [row for row in rows if row["arm"] == arm and row["condition"] == condition
                and row["budget"] == budget and row["evaluation"] is not None]
            if not selected:
                continue
            arms[arm] = {"tasks": len(selected),
                "solved": sum(r["evaluation"]["mse"] <= 1e-7 for r in selected),
                "completed": sum(r["status"] == "completed" for r in selected),
                "completed_and_solved": sum(r["status"] == "completed" and r["evaluation"]["mse"] <= 1e-7 for r in selected),
                "false_completed": sum(r["status"] == "completed" and r["evaluation"]["mse"] > 1e-7 for r in selected),
                "mean_mse": sum(r["evaluation"]["mse"] for r in selected) / len(selected),
                "costs": {k: sum(r["costs"][k] for r in selected) for k in COSTS},
                "cpu_seconds": sum(r["cpu_seconds"] for r in selected),
                **{key: sum(r[key] for r in selected) for key in method_metrics({"actions": []})}}
        groups[f"{condition}_{budget}"] = arms
        for reference, treatment in (("baseline", "revised"), ("without_methods", "revised")):
            selected = {arm: {row["case"]: row for row in rows if row["arm"] == arm
                and row["condition"] == condition and row["budget"] == budget
                and row["evaluation"] is not None} for arm in (reference, treatment)}
            common = sorted(set(selected[reference]) & set(selected[treatment]))
            if not common:
                continue
            differences = [selected[treatment][case]["evaluation"]["mse"] -
                selected[reference][case]["evaluation"]["mse"] for case in common]
            comparisons.append({"condition": condition, "budget": budget,
                "reference": reference, "treatment": treatment, "pairs": len(common),
                "better": sum(value < -1e-7 for value in differences),
                "worse": sum(value > 1e-7 for value in differences),
                "within_1e_minus_7": sum(abs(value) <= 1e-7 for value in differences),
                "cost_difference": {key: sum(selected[treatment][case]["costs"][key] -
                    selected[reference][case]["costs"][key] for case in common) for key in COSTS}})
    sources = [json.loads(source_report(output, protocol, root).read_text()) for root in protocol["roots"]
        if source_report(output, protocol, root).exists()]
    source_costs = {k: sum(s["research"]["costs"][k] for s in sources) for k in COSTS}
    arm_totals = {}
    for arm in protocol["arms"]:
        selected = [r for r in rows if r["arm"] == arm]
        learning = {k: sum(r["costs"][k] for r in selected) for k in COSTS}
        terminal = {k: sum(r["evaluation"][k] for r in selected if r["evaluation"]) for k in COSTS}
        arm_totals[arm] = {"successor_learning_costs": learning, "terminal_costs": terminal,
            "source_and_successor_learning_costs": {k: learning[k] + source_costs[k] for k in COSTS},
            "all_costs_including_terminal": {k: learning[k] + source_costs[k] + terminal[k] for k in COSTS},
            "successor_cpu_seconds": sum(r["cpu_seconds"] for r in selected),
            "source_and_successor_cpu_seconds": sum(r["cpu_seconds"] for r in selected)
                + sum(s["cpu_seconds"] for s in sources)}
    result = {"learning_jobs": len(rows), "planned_jobs": len(protocol["arms"]) * len(protocol["cases"]),
        "evaluated_jobs": sum(r["evaluation"] is not None for r in rows), "groups": groups,
        "source_costs": source_costs, "arm_totals": arm_totals,
        "source_cpu_seconds": sum(s["cpu_seconds"] for s in sources),
        "sources_with_methods": sum(bool(s["knowledge"]["program_experiment_methods"]) for s in sources),
        "paired_comparisons": comparisons, "results": rows}
    save(output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sources", "learn", "evaluate", "summary"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--case", default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    if args.phase == "summary":
        summarize(output, protocol)
        return
    cases = protocol["roots"] if args.phase == "sources" else protocol["cases"]
    if args.phase != "sources" and args.arm not in protocol["arms"]:
        parser.error("learning/evaluation needs an arm from the protocol")
    if args.case == "all":
        def dispatch(case):
            command = [sys.executable, str(Path(__file__).resolve()), "--phase", args.phase,
                "--output", str(output), "--case", case]
            if args.arm:
                command.extend(("--arm", args.arm))
            subprocess.run(command, check=True)
        with ThreadPoolExecutor(max_workers=protocol.get("workers", 3)) as pool:
            list(pool.map(dispatch, cases))
    elif args.case not in cases:
        parser.error("case is absent from the frozen protocol")
    elif args.phase == "sources":
        learn_source(output, protocol, args.case)
    elif args.phase == "learn":
        learn(output, protocol, args.arm, args.case)
    else:
        evaluate(output, protocol, args.arm, args.case)


if __name__ == "__main__":
    main()
