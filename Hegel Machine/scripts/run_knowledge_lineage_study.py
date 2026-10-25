#!/usr/bin/env python3
"""Carry complete learned libraries through fixed task chains, with restart controls."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_text_incrementally
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def checkpoint_path(path):
    """Prefer the current plain snapshot, retaining actual old gzip recovery."""
    path = Path(path)
    plain = path.with_suffix("") if path.suffix == ".gz" else path
    historical = plain.with_name(plain.name + ".gz")
    return plain if plain.exists() or not historical.exists() else historical


def read(path):
    path = checkpoint_path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def save(path, value, *, base=None):
    # Existing gzip snapshots stay untouched. New checkpoints use the ordinary
    # JSON reader and can share unchanged blocks with the actual prior state.
    path = Path(path)
    if path.suffix == ".gz":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    if base is not None:
        base = checkpoint_path(base)
        if base.suffix == ".gz":
            base = None
    write_text_incrementally(path, json.dumps(value, indent=2) + "\n", base=base)


def checkpoint(output, arm, root, position):
    return checkpoint_path(output / "runs" / arm / root / f"stage{position:02d}" / "checkpoint.json")


def source(protocol, root):
    return read(ROOT / protocol["roots"][root]["source_report"])


def load_runtime(protocol, arm):
    runtime = (ROOT / protocol["arms"][arm]["runtime"]).resolve()
    sys.path.insert(0, str(runtime))
    from hegel_machine.cognition import unified_session
    if not Path(unified_session.__file__).is_relative_to(runtime):
        raise RuntimeError("each arm requires its own runtime process")
    return unified_session


def learn(output, protocol, arm, root):
    module = load_runtime(protocol, arm)
    from hegel_machine import __version__
    from hegel_machine.cognition.memory import KnowledgeLibrary
    mode = protocol["arms"][arm]["knowledge"]
    original = source(protocol, root)["knowledge"] if mode != "empty" else None
    for position, task in enumerate(protocol["roots"][root]["tasks"]):
        path = checkpoint(output, arm, root, position)
        if path.exists():
            continue
        incoming = (read(checkpoint(output, arm, root, position - 1))["knowledge"]
            if mode == "cumulative" and position else original)
        library = KnowledgeLibrary.from_dict(deepcopy(incoming)) if incoming else KnowledgeLibrary()
        limits = {**protocol["limits"], **task.get("research_limits", {})}
        config = replace(module.ResearchConfig(**protocol["config"]), seed=task["seed"],
            observation_budget=limits["observation_budget"], compute_budget=limits["compute_budget"])
        laboratories = []
        def factory(arguments):
            laboratory = module.laboratory_for(arguments)
            laboratories.append(laboratory)
            return laboratory
        session = module.UnifiedResearchSession(config, tasks=[task], library=library,
            measurement_budget=limits["measurement_budget"], lab_factory=factory)
        wall, cpu = time.monotonic(), time.process_time()
        try:
            state = session.run()
        except Exception as error:
            save(path.parent / "failure.json", {"error": repr(error), "task": task,
                "cpu_seconds": time.process_time() - cpu, "elapsed_seconds": time.monotonic() - wall,
                "physical_costs": {key: sum(lab.costs[key] for lab in laboratories)
                    for key in ("observation_cost", "measurement_cost", "queries")},
                "compute_cost": None})
            raise
        state["study"] = {"arm": arm, "root": root, "position": position,
            "stage": protocol["stages"][position], "version": __version__,
            "cpu_seconds": time.process_time() - cpu, "elapsed_seconds": time.monotonic() - wall,
            "knowledge_input": (str(checkpoint(output, arm, root, position - 1).relative_to(ROOT))
                if mode == "cumulative" and position else
                protocol["roots"][root]["source_report"] if mode != "empty" else None),
            "input_counts": {key: len(value) for key, value in (incoming or {}).items()
                if isinstance(value, list)}}
        save(path, state, base=checkpoint(output, arm, root, position - 1)
            if mode == "cumulative" and position else None)
        print(json.dumps({**state["study"], "status": state["results"][0]["research"]["status"],
            "costs": state["costs"]}), flush=True)


def evaluate(output, protocol, arm, root):
    for other_arm in protocol["arms"]:
        for other_root in protocol["roots"]:
            for position in range(len(protocol["stages"])):
                if not checkpoint(output, other_arm, other_root, position).exists():
                    raise ValueError("finish all chains in every arm before terminal evaluation")
    module = load_runtime(protocol, arm)
    for position in range(len(protocol["stages"])):
        path = checkpoint(output, arm, root, position)
        destination = path.parent / "evaluation.json"
        if destination.exists():
            continue
        session = module.UnifiedResearchSession(state=read(path))
        state = session.evaluate(**protocol["terminal"])
        save(destination, state["evaluation"][0])


def summarize(output, protocol):
    rows = []
    for arm in protocol["arms"]:
        for root in protocol["roots"]:
            for position, stage in enumerate(protocol["stages"]):
                path = checkpoint(output, arm, root, position)
                if not path.exists():
                    continue
                state = read(path)
                report = state["results"][0]["research"]
                evaluation = read(path.parent / "evaluation.json") if (path.parent / "evaluation.json").exists() else None
                actions = Counter(item["action"] for item in report["actions"])
                rows.append({"arm": arm, "root": root, "position": position, "stage": stage,
                    "status": report["status"], "costs": state["costs"],
                    "cpu_seconds": state["study"]["cpu_seconds"], "evaluation": evaluation,
                    "input_counts": state["study"]["input_counts"],
                    "knowledge_counts": {key: len(value) for key, value in state["knowledge"].items()
                        if isinstance(value, list)}, "actions": dict(actions)})
    def aggregate(selected):
        scored = [row for row in selected if row["evaluation"] is not None]
        passed = lambda row: row["evaluation"]["mse"] <= protocol["pass_mse"]
        return {"learning_jobs": len(selected), "evaluated_jobs": len(scored),
            "solved": sum(passed(row) for row in scored),
            "completed": sum(row["status"] == "completed" for row in selected),
            "completed_and_solved": sum(row["status"] == "completed" and passed(row) for row in scored),
            "false_completed": sum(row["status"] == "completed" and not passed(row) for row in scored),
            "mean_mse": sum(row["evaluation"]["mse"] for row in scored) / len(scored) if scored else None,
            "learning_costs": {key: sum(row["costs"][key] for row in selected) for key in COSTS},
            "terminal_costs": {key: sum(row["evaluation"][key] for row in scored) for key in COSTS},
            "cpu_seconds": sum(row["cpu_seconds"] for row in selected),
            "actions": dict(sum((Counter(row["actions"]) for row in selected), Counter()))}
    totals = {arm: aggregate([row for row in rows if row["arm"] == arm]) for arm in protocol["arms"]}
    stages = {stage: {arm: aggregate([row for row in rows if row["arm"] == arm and row["stage"] == stage])
        for arm in protocol["arms"]} for stage in protocol["stages"]}
    comparisons = []
    for reference in protocol["arms"]:
        if reference == "cumulative":
            continue
        for stage in (None, *protocol["stages"]):
            selected = {arm: {(row["root"], row["position"]): row for row in rows
                if row["arm"] == arm and row["evaluation"] is not None
                and (stage is None or row["stage"] == stage)} for arm in (reference, "cumulative")}
            common = set(selected[reference]) & set(selected["cumulative"])
            differences = [selected["cumulative"][key]["evaluation"]["mse"] -
                selected[reference][key]["evaluation"]["mse"] for key in common]
            comparisons.append({"reference": reference, "treatment": "cumulative", "stage": stage,
                "pairs": len(common), "better": sum(value < -1e-7 for value in differences),
                "worse": sum(value > 1e-7 for value in differences),
                "tied": sum(abs(value) <= 1e-7 for value in differences),
                "learning_cost_difference": {cost: sum(selected["cumulative"][key]["costs"][cost] -
                    selected[reference][key]["costs"][cost] for key in common) for cost in COSTS}})
    sources = [source(protocol, root) for root in protocol["roots"]]
    reused_costs = {key: sum(item["research"]["costs"][key] for item in sources) for key in COSTS}
    reused_cpu = sum(item["cpu_seconds"] for item in sources)
    for arm, total in totals.items():
        seeded = protocol["arms"][arm]["knowledge"] != "empty"
        total["source_and_learning_costs"] = {key: total["learning_costs"][key] +
            (reused_costs[key] if seeded else 0) for key in COSTS}
        total["source_and_learning_cpu_seconds"] = total["cpu_seconds"] + (reused_cpu if seeded else 0)
    streams = {root: {arm: aggregate([row for row in rows if row["arm"] == arm and row["root"] == root])
        for arm in protocol["arms"]} for root in protocol["roots"]}
    result = {"learning_jobs": len(rows), "evaluated_jobs": sum(row["evaluation"] is not None for row in rows),
        "planned_jobs": len(protocol["arms"]) * len(protocol["roots"]) * len(protocol["stages"]),
        "reused_source_costs": reused_costs,
        "reused_source_cpu_seconds": reused_cpu,
        "arm_totals": totals, "stages": stages, "streams": streams, "comparisons": comparisons, "results": rows}
    save(output / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items()
        if key not in ("results", "stages", "streams", "comparisons")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("learn", "evaluate", "summary"), required=True)
    parser.add_argument("--arm", default="all")
    parser.add_argument("--root", default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    if args.phase == "summary":
        summarize(output, protocol)
        return
    arms = list(protocol["arms"]) if args.arm == "all" else [args.arm]
    roots = list(protocol["roots"]) if args.root == "all" else [args.root]
    if any(arm not in protocol["arms"] for arm in arms) or any(root not in protocol["roots"] for root in roots):
        parser.error("arm or root is absent from the frozen protocol")
    if len(arms) * len(roots) > 1:
        def dispatch(pair):
            arm, root = pair
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                "--phase", args.phase, "--arm", arm, "--root", root], check=True)
        with ThreadPoolExecutor(max_workers=protocol.get("workers", 3)) as pool:
            list(pool.map(dispatch, [(arm, root) for root in roots for arm in arms]))
    else:
        (learn if args.phase == "learn" else evaluate)(output, protocol, arms[0], roots[0])


if __name__ == "__main__":
    main()
