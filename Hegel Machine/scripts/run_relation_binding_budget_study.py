#!/usr/bin/env python3
"""Paired anonymous relation research at fixed budgets and declared source knowledge."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/relation_binding_budget_20260920"
BASELINE = ROOT / "artifacts/risk_knowledge_adoption_20260920/runtime/src"
SOURCE = ROOT / "artifacts/relation_binding_session_smoke_20260920/source_relation/checkpoint_final.json"
TASK = ROOT / "artifacts/relation_binding_session_smoke_20260920/source_relation_followup/config.json"
CASES = {"known96": (96, 30000, 1152), "medium192": (192, 120000, 2304),
         "large384": (384, 240000, 4608)}
ARMS = ("baseline", "revised")


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def prepare(output):
    path = output / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    task = json.loads(TASK.read_text())["appended_task"]
    protocol = {"experiment": "warmed_relation_binding_budget_development",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)), "task": task,
        "cases": {name: dict(zip(("observation_budget", "compute_budget", "measurement_budget"), values))
                  for name, values in CASES.items()},
        "arms": {"baseline": str(BASELINE.relative_to(ROOT)),
                 "revised": str((output / "runtime/src").relative_to(ROOT))},
        "terminal": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "scope": "Known development task, identical actual learned source library in both arms. "
            "Each budget restarts a warmed one-task continuation through UnifiedResearchAgent.run; "
            "these are not six new cold sequences or independent task roots. No helper-injected roles or calibration.",
        "evaluation_contract": "All six learning jobs end before any terminal scoring. Every unresolved "
            "output is scored on the same four independent 64-row trajectories. No terminal rows enter learning.",
        "maximum_concurrent_processes": 3}
    save(path, protocol)
    return protocol


def trace(report):
    names = {"acquire", "test_object_dependency_question", "retain_object_ambiguity",
        "adopt_discovered_objects", "choose_research_action", "compose_relation_kernels",
        "compare_research_directions", "compare_relation_kernel_bindings"}
    return [{key: action[key] for key in ("action", "purpose", "observation_cost", "compute_cost",
        "selected_action", "reason", "object_model_id", "rows", "question", "alternatives",
        "remaining_rows", "supported_bindings", "binding_source") if key in action}
        for action in report["actions"] if action["action"] in names]


def learn(output, protocol, arm, case, runtime):
    path = output / "runs" / arm / case / "report.json"
    if path.exists():
        return
    sys.path.insert(0, str(runtime))
    from hegel_machine import __version__
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_session import laboratory_for

    source = json.loads((ROOT / protocol["source_checkpoint"]).read_text())
    declared = protocol["cases"][case]
    task = {key: value for key, value in {**protocol["task"], **declared.get("task", {})}.items()
            if key != "research_limits"}
    limits = {key: declared[key] for key in ("observation_budget", "compute_budget", "measurement_budget")}
    config = replace(ResearchConfig(**source["config"]), seed=task["seed"],
        observation_budget=limits["observation_budget"], compute_budget=limits["compute_budget"])
    knowledge_mode = declared.get("knowledge_mode", "warm")
    if knowledge_mode not in ("warm", "empty"):
        raise ValueError(f"unknown knowledge mode: {knowledge_mode}")
    library = (KnowledgeLibrary.from_dict(source["knowledge"])
               if knowledge_mode == "warm" else KnowledgeLibrary())
    wall, cpu = time.monotonic(), time.process_time()
    with laboratory_for(task) as laboratory:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=limits["measurement_budget"]).run(laboratory)
    save(path, {"arm": arm, "case": case, "runtime": str(runtime), "version": __version__,
        "task": task, "limits": limits, "knowledge_mode": knowledge_mode,
        "source_checkpoint": protocol["source_checkpoint"] if knowledge_mode == "warm" else None,
        "research": report, "knowledge": library.to_dict(),
        "elapsed_seconds": time.monotonic() - wall, "cpu_seconds": time.process_time() - cpu})
    print(json.dumps({"arm": arm, "case": case, "status": report["status"],
        "costs": report["costs"], "used_sources": report["used_sources"],
        "method_source_ids": report.get("method_source_ids", []),
        "compose_events": sum(action["action"] == "compose_relation_kernels" for action in report["actions"])},
        ensure_ascii=False), flush=True)


def evaluate(output, protocol, arm, case, runtime):
    missing = [(a, c) for a in protocol["arms"] for c in protocol["cases"]
               if not (output / "runs" / a / c / "report.json").exists()]
    if missing:
        raise ValueError(f"all learning must finish before evaluation: {missing}")
    destination = output / "runs" / arm / case / "evaluation.json"
    if destination.exists():
        return
    sys.path.insert(0, str(runtime))
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.unified_session import laboratory_for

    saved = json.loads((destination.parent / "report.json").read_text())
    report, terminal = saved["research"], protocol["terminal"]
    with laboratory_for(saved["task"]) as laboratory:
        rows = laboratory.evaluation_rows(report["selected_channels"],
            seed=saved["task"]["seed"] + terminal["seed_offset"],
            length=terminal["length"], trajectories=terminal["trajectories"])
        rate = sum(channel.cost_per_row for channel in laboratory.channels
                   if channel.name in report["selected_channels"])
        mse = loss(Program.from_dict(report["program"]), rows, report["fallback"], .5)
    save(destination, {"mse": mse, "rows": len(rows), "observation_cost": len(rows),
        "measurement_cost": len(rows) * rate, "compute_cost": len(rows)})


def summarize(output, protocol):
    rows = []
    for arm in protocol["arms"]:
        for case in protocol["cases"]:
            folder = output / "runs" / arm / case
            if not (folder / "report.json").exists():
                continue
            saved = json.loads((folder / "report.json").read_text())
            report = saved["research"]
            evaluation = json.loads((folder / "evaluation.json").read_text()) if (folder / "evaluation.json").exists() else None
            row = {"arm": arm, "case": case, "status": report["status"], "limits": saved["limits"],
                "knowledge_mode": saved.get("knowledge_mode", "warm"),
                "costs": report["costs"], "selected_channels": report["selected_channels"],
                "used_sources": report["used_sources"], "method_source_ids": report.get("method_source_ids", []),
                "object_complete": bool((report.get("object_research") or {}).get("complete")),
                "object_acquisition_rows": (report.get("object_research") or {}).get("acquisition_rows", 0),
                "dependency_probes": sum(action.get("purpose") == "distinguish_object_dependencies"
                    for action in report["actions"] if action["action"] == "acquire"),
                "compose_events": sum(action["action"] == "compose_relation_kernels" for action in report["actions"]),
                "elapsed_seconds": saved["elapsed_seconds"], "cpu_seconds": saved["cpu_seconds"],
                "evaluation": evaluation}
            rows.append(row)
            save(folder / "trace.json", trace(report))
    result = {"learning_jobs": len(rows), "planned_jobs": len(protocol["arms"]) * len(protocol["cases"]),
        "evaluated_jobs": sum(row["evaluation"] is not None for row in rows), "results": rows}
    comparisons = []
    groups = sorted({(row["knowledge_mode"], row["limits"]["observation_budget"]) for row in rows})
    for knowledge_mode, limit in groups:
        selected = {arm: {row["case"]: row for row in rows
            if row["arm"] == arm and row["evaluation"] is not None
            and row["knowledge_mode"] == knowledge_mode
            and row["limits"]["observation_budget"] == limit} for arm in ARMS}
        common = sorted(set(selected["baseline"]) & set(selected["revised"]))
        if not common:
            continue
        aggregate = {}
        for arm in ARMS:
            paired = [selected[arm][case] for case in common]
            aggregate[arm] = {"mean_mse": sum(row["evaluation"]["mse"] for row in paired) / len(paired),
                "mse_at_most_1e_minus_7": sum(row["evaluation"]["mse"] <= 1e-7 for row in paired),
                "completed": sum(row["status"] == "completed" for row in paired),
                "object_complete": sum(row["object_complete"] for row in paired),
                "composed_tasks": sum(row["compose_events"] > 0 for row in paired),
                "selected_source_tasks": sum(bool(row["used_sources"] or row["method_source_ids"]) for row in paired),
                "learning_costs": {key: sum(row["costs"][key] for row in paired)
                    for key in ("observation_cost", "measurement_cost", "compute_cost")},
                "terminal_measurement": sum(row["evaluation"]["measurement_cost"] for row in paired)}
        differences = [selected["revised"][case]["evaluation"]["mse"] -
                       selected["baseline"][case]["evaluation"]["mse"] for case in common]
        change = {key: (aggregate["revised"]["learning_costs"][key] /
                        aggregate["baseline"]["learning_costs"][key] - 1) * 100
                  for key in aggregate["baseline"]["learning_costs"]
                  if aggregate["baseline"]["learning_costs"][key]}
        change["mean_mse"] = ((aggregate["revised"]["mean_mse"] / aggregate["baseline"]["mean_mse"] - 1) * 100
                              if aggregate["baseline"]["mean_mse"] else None)
        comparisons.append({"knowledge_mode": knowledge_mode,
            "observation_budget": limit, "paired_successors": len(common),
            "better": sum(value < -1e-7 for value in differences),
            "worse": sum(value > 1e-7 for value in differences),
            "within_1e_minus_7": sum(abs(value) <= 1e-7 for value in differences),
            "relative_change_percent": change, **aggregate})
    result["paired_by_budget"] = comparisons
    save(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "learn", "evaluate", "summary"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--case", default="all", help="case from the frozen protocol, or all")
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    protocol = prepare(args.output)
    if args.phase == "prepare":
        return
    if args.phase == "summary":
        summarize(args.output, protocol)
        return
    if args.arm is None:
        parser.error("learning and evaluation require --arm")
    runtime = (args.runtime or ROOT / protocol["arms"][args.arm]).resolve()
    if args.case == "all":
        def run(case):
            command = [sys.executable, str(Path(__file__).resolve()), "--phase", args.phase,
                "--output", str(args.output), "--arm", args.arm, "--case", case, "--runtime", str(runtime)]
            subprocess.run(command, check=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run, protocol["cases"]))
    elif args.case not in protocol["cases"]:
        parser.error("case is absent from the frozen protocol")
    elif args.phase == "learn":
        learn(args.output, protocol, args.arm, args.case, runtime)
    else:
        evaluate(args.output, protocol, args.arm, args.case, runtime)


if __name__ == "__main__":
    main()
