#!/usr/bin/env python3
"""Complete paid task banks, successor-triggered revisits, and matched deletion."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

from run_knowledge_lineage_study import ROOT, COSTS, read, save, load_runtime


def directory(output, arm, root):
    return output / "runs" / arm / root


def current(output, arm, root):
    return directory(output, arm, root) / "checkpoint.json.gz"


def persist(output, arm, root, state):
    path = directory(output, arm, root)
    save(path / f"visit{state['study']['steps']:02d}.json.gz", state)
    save(current(output, arm, root), state)


def initial_snapshot(output, root, position, arm="knowledge_revisit"):
    return read(directory(output, arm, root) / f"visit{position + 1:02d}.json.gz")


def learn(output, protocol, arm, root):
    module = load_runtime(protocol, arm)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    path = current(output, arm, root)
    if path.exists():
        state = read(path)
        if state["study"]["learning_complete"]:
            return
        session = module.UnifiedResearchSession(state=state)
    else:
        source = protocol["roots"][root]["source_report"]
        knowledge = read(ROOT / source)["knowledge"] if protocol["arms"][arm]["knowledge"] != "empty_start" else {}
        session = module.UnifiedResearchSession(
            replace(module.ResearchConfig(**protocol["config"]),
                seed=protocol["roots"][root]["tasks"][0]["seed"]),
            tasks=protocol["roots"][root]["tasks"],
            measurement_budget=protocol["limits"]["measurement_budget"],
            library=KnowledgeLibrary.from_dict(knowledge))
        session.state["study"] = {"arm": arm, "root": root, "steps": 0,
            "cpu_seconds": 0., "initial_cpu_seconds": 0., "learning_complete": False}
    while (len(session.state["results"]) < len(session.state["tasks"])
           or session.state["pending_revisits"]):
        first_visit = len(session.state["results"]) < len(session.state["tasks"])
        started = time.process_time()
        try:
            state = session.run(max_tasks=1)
        except Exception as error:
            save(path.parent / "failure.json", {"error": repr(error),
                "completed_steps": session.state["study"]["steps"],
                "interrupted_cpu_seconds": time.process_time() - started,
                "interrupted_resource_costs": None})
            raise
        elapsed = time.process_time() - started
        state["study"]["steps"] += 1
        state["study"]["cpu_seconds"] += elapsed
        if first_visit:
            state["study"]["initial_cpu_seconds"] += elapsed
            if len(state["results"]) == len(state["tasks"]):
                save(path.parent / "first_visits.json.gz", state)
        state["study"]["learning_complete"] = (
            len(state["results"]) == len(state["tasks"]) and not state["pending_revisits"])
        persist(output, arm, root, state)
        print({"arm": arm, "root": root, "steps": state["study"]["steps"],
            "first_visits": len(state["results"]), "revisits": len(state["revisits"]),
            "complete": state["study"]["learning_complete"], "costs": state["costs"]}, flush=True)
        session = module.UnifiedResearchSession(state=read(path))


def ablate(output, protocol, root):
    arm = "no_successor_knowledge"
    module = load_runtime(protocol, arm)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    paired = read(current(output, "knowledge_revisit", root))
    if not paired["study"]["learning_complete"]:
        raise ValueError("finish the actual new-arm schedule before its matched deletion")
    path = current(output, arm, root)
    state = read(path) if path.exists() else read(
        directory(output, "knowledge_revisit", root) / "first_visits.json.gz")
    if not path.exists():
        state["pending_revisits"], state["revisits"] = [], []
        state["knowledge"] = {}
        state["study"].update(arm=arm, learning_complete=False, matched=[], local_knowledge={},
            reused_initial_from=str(directory(output, "knowledge_revisit", root) / "first_visits.json.gz"))
    if state["study"]["learning_complete"]:
        return
    for index in range(len(state["study"]["matched"]), len(paired["revisits"])):
        visit = paired["revisits"][index]
        position = visit["task_position"]
        previous = state["results"][position]["research"]
        session = module.UnifiedResearchSession(state=state)
        requested = {key: visit["resource_limits"][key] - visit["previous"]["costs"][key]
                     for key in COSTS}
        allocation = {key: min(requested[key], session._remaining()[key]) for key in COSTS}
        receipt = {"matched_index": index, "task_position": position,
            "requested": requested, "allocation": allocation}
        if not allocation["observation_cost"] or not allocation["compute_cost"]:
            state["study"]["matched"].append({**receipt, "executed": False})
            save(path, state)
            continue
        local = state["study"]["local_knowledge"]
        own = local.get(str(position))
        if own is None:
            own = initial_snapshot(output, root, position)["knowledge"]
        library = KnowledgeLibrary.from_dict(deepcopy(own))
        limits = {key: previous["costs"][key] + allocation[key] for key in COSTS}
        task = state["tasks"][position]
        started = time.process_time()
        with module.laboratory_for(task) as laboratory:
            report = UnifiedResearchAgent(replace(session.config, seed=task["seed"],
                observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
                measurement_budget=limits["measurement_cost"], library=library).run(
                    laboratory, previous=previous, continuation_candidate=None)
        state["study"]["cpu_seconds"] += time.process_time() - started
        local[str(position)] = library.to_dict()
        state["results"][position]["research"] = report
        state["revisits"].append({"task_position": position, "source_task_position": None,
            "candidate": None, "previous": previous, "resource_limits": limits,
            "visit_costs": report["visit_costs"], "status": report["status"]})
        state["study"]["matched"].append({**receipt, "executed": True})
        state["study"]["steps"] += 1
        state["knowledge"] = library.to_dict()
        state["costs"] = module.UnifiedResearchSession(state=state).to_dict()["costs"]
        persist(output, arm, root, state)
    state["study"]["learning_complete"] = True
    state["status"] = "completed"
    save(path, state)
    print({"arm": arm, "root": root, "revisits": len(state["revisits"]),
           "costs": state["costs"]}, flush=True)


def evaluate(output, protocol, arm, root):
    for name in protocol["arms"]:
        for key in protocol["roots"]:
            if not read(current(output, name, key))["study"]["learning_complete"]:
                raise ValueError("finish all learning and matched ablations before terminal scoring")
    module = load_runtime(protocol, arm)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    state = read(current(output, arm, root))
    path = directory(output, arm, root) / "evaluation.json"
    scored = read(path) if path.exists() else []
    settings = protocol["terminal"]
    for position in range(len(scored), len(state["tasks"])):
        task = state["tasks"][position]
        final = state["results"][position]["research"]
        origin = "knowledge_revisit" if arm == "no_successor_knowledge" else arm
        initial = initial_snapshot(output, root, position, origin)["results"][position]["research"]
        views = {tuple(report["selected_channels"]) for report in (initial, final)}
        values = {"initial": [], "final": []}
        with module.laboratory_for(task) as laboratory:
            for trajectory in range(settings["trajectories"]):
                probe = laboratory.random_probe(task["seed"] + settings["seed_offset"] + trajectory * 10007,
                    length=settings["length"], name=f"terminal_{trajectory}")
                batches = {view: laboratory.acquire(probe, view).observations for view in sorted(views)}
                for name, report in (("initial", initial), ("final", final)):
                    values[name].append(loss(Program.from_dict(report["program"]),
                        batches[tuple(report["selected_channels"])], report["fallback"], .5))
            costs = {**laboratory.costs,
                "compute_cost": 2 * settings["length"] * settings["trajectories"]}
            rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        scored.append({"task_position": position,
            "initial_mse": sum(values["initial"]) / len(values["initial"]),
            "mse": sum(values["final"]) / len(values["final"]), "trajectories": values,
            "initial_status": initial["status"], "status": final["status"], "costs": costs,
            "initial_channels": initial["selected_channels"], "selected_channels": final["selected_channels"],
            "deployment_measurement_costs": {name: settings["length"] * settings["trajectories"] *
                sum(rates[channel] for channel in report["selected_channels"])
                for name, report in (("initial", initial), ("final", final))}})
        save(path, scored)


def summarize(output, protocol):
    rows, streams = [], []
    for arm in protocol["arms"]:
        for root in protocol["roots"]:
            path = current(output, arm, root)
            if not path.exists():
                continue
            state = read(path)
            terminal = directory(output, arm, root) / "evaluation.json"
            scored = read(terminal) if terminal.exists() else []
            source = read(ROOT / protocol["roots"][root]["source_report"])
            seeded = protocol["arms"][arm]["knowledge"] != "empty_start"
            streams.append({"arm": arm, "root": root, "complete": state["study"]["learning_complete"],
                "learning_costs": state["costs"], "cpu_seconds": state["study"]["cpu_seconds"],
                "source_costs": {key: source["research"]["costs"][key] if seeded else 0 for key in COSTS},
                "source_cpu_seconds": source["cpu_seconds"] if seeded else 0,
                "screening_compute": state["screening_compute_cost"], "revisits": len(state["revisits"]),
                "revisit_costs": {key: sum(v["visit_costs"][key] for v in state["revisits"]) for key in COSTS},
                "scheduler_actions": dict(Counter(x["action"] for x in state["scheduler_events"]))})
            for score in scored:
                position = score["task_position"]
                report = state["results"][position]["research"]
                rows.append({"arm": arm, "root": root, "stage": protocol["stages"][position], **score,
                    "revisits": sum(v["task_position"] == position for v in state["revisits"]),
                    "actions": dict(Counter(a["action"] for a in report["actions"]))})
    def scores(selected):
        passed = lambda value: value <= protocol["pass_mse"]
        return {"tasks": len(selected), "initial_pass": sum(passed(r["initial_mse"]) for r in selected),
            "final_pass": sum(passed(r["mse"]) for r in selected),
            "completed_and_pass": sum(r["status"] == "completed" and passed(r["mse"]) for r in selected),
            "false_completed": sum(r["status"] == "completed" and not passed(r["mse"]) for r in selected),
            "gained_pass": sum(not passed(r["initial_mse"]) and passed(r["mse"]) for r in selected),
            "lost_pass": sum(passed(r["initial_mse"]) and not passed(r["mse"]) for r in selected),
            "mean_mse": sum(r["mse"] for r in selected) / len(selected) if selected else None,
            "initial_mean_mse": sum(r["initial_mse"] for r in selected) / len(selected) if selected else None,
            "terminal_costs": {key: sum(r["costs"][key] for r in selected) for key in COSTS}}
    totals = {}
    for arm in protocol["arms"]:
        selected = [s for s in streams if s["arm"] == arm]
        totals[arm] = {**scores([r for r in rows if r["arm"] == arm]),
            **{name: sum(s[name] for s in selected) for name in
               ("cpu_seconds", "source_cpu_seconds", "screening_compute", "revisits")},
            **{name: {key: sum(s[name][key] for s in selected) for key in COSTS}
               for name in ("learning_costs", "source_costs", "revisit_costs")}}
    comparisons = []
    comparison_pairs = [("knowledge_revisit", arm) for arm in protocol["arms"]
                        if arm != "knowledge_revisit"]
    if "baseline_empty" in protocol["arms"] and "empty_start" in protocol["arms"]:
        comparison_pairs.append(("empty_start", "baseline_empty"))
    for treatment, control in comparison_pairs:
        paired = {a: {(r["root"], r["task_position"]): r for r in rows if r["arm"] == a}
                  for a in (control, treatment)}
        changes = [paired[treatment][k]["mse"] - paired[control][k]["mse"]
                   for k in paired[control].keys() & paired[treatment].keys()]
        comparisons.append({"treatment": treatment, "control": control, "pairs": len(changes),
            "better": sum(d < -protocol["pass_mse"] for d in changes),
            "worse": sum(d > protocol["pass_mse"] for d in changes),
            "tied": sum(abs(d) <= protocol["pass_mse"] for d in changes)})
    save(output / "summary.json", {"streams": streams, "results": rows, "arm_totals": totals,
        "comparisons": comparisons, "stages": {stage: {arm: scores([r for r in rows
            if r["stage"] == stage and r["arm"] == arm]) for arm in protocol["arms"]}
            for stage in protocol["stages"]}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("learn", "evaluate", "summary"))
    parser.add_argument("--arm", default="all")
    parser.add_argument("--root", default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    if args.phase == "summary":
        summarize(output, protocol)
    elif args.arm != "all":
        if args.phase == "evaluate":
            evaluate(output, protocol, args.arm, args.root)
        elif args.arm == "no_successor_knowledge":
            ablate(output, protocol, args.root)
        else:
            learn(output, protocol, args.arm, args.root)
    else:
        roots = protocol["roots"] if args.root == "all" else [args.root]
        def dispatch(job):
            root, arm = job
            subprocess.run([sys.executable, __file__, "--output", str(output), "--phase", args.phase,
                "--root", root, "--arm", arm], check=True)
        arms = [arm for arm in protocol["arms"] if args.phase != "learn" or arm != "no_successor_knowledge"]
        with ThreadPoolExecutor(max_workers=protocol["workers"]) as executor:
            list(executor.map(dispatch, [(root, arm) for root in roots for arm in arms]))
            if args.phase == "learn" and "no_successor_knowledge" in protocol["arms"]:
                list(executor.map(dispatch, [(root, "no_successor_knowledge") for root in roots]))


if __name__ == "__main__":
    main()
