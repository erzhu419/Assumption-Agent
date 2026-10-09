#!/usr/bin/env python3
"""Discover and revise private branches and response windows from paid trajectories."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from random import Random
import subprocess
import sys
import time

from run_knowledge_lineage_study import ROOT, COSTS, read, save, load_runtime


def laboratory(task):
    from hegel_machine.cognition.measurement_lab import MeasurementBatch
    from hegel_machine.cognition.models import Observation, Probe, TaskSpec

    class ConditionalScalarLab:
        channels = ()

        def __init__(self):
            apparatus = Random(task["apparatus_seed"])
            self._threshold = apparatus.uniform(-.35, .35)
            self._first = (apparatus.uniform(-.3, .3), apparatus.uniform(.65, 1.25),
                           apparatus.uniform(.35, .8))
            self._second = (apparatus.uniform(-.3, .3), apparatus.uniform(-1.25, -.65),
                            apparatus.uniform(-.8, -.35))
            self._stage = task["stage"]
            self._width = (3, 3, 5, 7, 7, 3, 5)[self._stage]
            self._threshold += .35 if self._stage in (3, 4) else 0.
            self._names = ("p", "q", "r") if self._stage == 4 else ("x", "z", "c")
            self._control_scale = 2.3 if self._stage == 4 else 1.
            self._response_scale = 1.25 if self._stage == 4 else 1.
            self._noise = Random(task.get("noise_seed", task["seed"]) + 49979687)
            bounds = (-1.5 * self._control_scale, 1.5 * self._control_scale)
            self.spec = TaskSpec(f"response_{task['seed']}",
                "Apply controllable input sequences and predict the measured response.",
                self._names, (), (), feature_bounds=dict.fromkeys(self._names, bounds),
                max_probe_length=64)
            self.observation_cost = 0
            self.acquisitions = []

        def spec_for(self, channels=()):
            assert not channels
            return self.spec

        @property
        def costs(self):
            return {"observation_cost": self.observation_cost, "measurement_cost": 0,
                    "queries": len(self.acquisitions)}

        def acquire(self, probe, channels=()):
            assert not channels
            rows, history, first, second, gated = [], [], [], [], []
            for point in probe.sequence:
                x, z, c = (point[name] / self._control_scale for name in self._names)
                first.append(self._first[0] + self._first[1] * x + self._first[2] * z * z)
                second.append(self._second[0] + self._second[1] * x + self._second[2] * z * z)
                gated.append(first[-1] if c < self._threshold else second[-1])
                if self._stage in (0, 6):
                    target = sum(first[-self._width:])
                elif self._stage in (1, 5):
                    target = sum((first if c < self._threshold else second)[-self._width:])
                else:
                    target = sum(gated[-self._width:])
                target *= self._response_scale
                target += self._noise.gauss(0., task["noise_sigma"])
                rows.append(Observation(dict(point), target, tuple(history)))
                history.append({**point, "__target__": target})
            self.observation_cost += len(rows)
            self.acquisitions.append({"probe_id": probe.probe_id,
                "sequence": list(probe.sequence), "channels": []})
            return MeasurementBatch(rows, len(rows), 0, ())

        def random_probe(self, seed, *, length=24, name="sample"):
            rng = Random(seed)
            return Probe(name, tuple({key: rng.uniform(*bounds)
                for key, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

        def restore(self, acquisitions):
            self.__init__()
            for item in acquisitions:
                self.acquire(Probe(item["probe_id"], tuple(item["sequence"])))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    return ConditionalScalarLab()


def run_dir(output, arm, root):
    return output / "runs" / arm / root


def executable_structure(program):
    """Describe executed expressions, including bodies of the actual saved calls."""
    windows, conditions = set(), []
    def walk(node, ancestors=()):
        if node["op"] == "window_sum":
            windows.add(node["lag"])
        if node["op"] == "if" and node["children"][0]["op"] in ("lt", "le", "gt", "ge"):
            conditions.append({"gate": node["children"][0],
                "placement": "inside_window" if "window_sum" in ancestors else "outside_window"})
        for child in node.get("children", []):
            walk(child, (*ancestors, node["op"]))
        if node["op"] == "call":
            walk(program["definitions"][node["name"]]["body"], (*ancestors, "call"))
    walk(program)
    unique = []
    for condition in conditions:
        if condition not in unique:
            unique.append(condition)
    return {"window_widths": sorted(windows), "conditions": unique}


def learn_independent(output, protocol, arm, root):
    """Reset the entire knowledge library and unused budget after each task."""
    module = load_runtime(protocol, arm)
    directory = run_dir(output, arm, root)
    path = directory / "checkpoint.json.gz"
    if path.exists() and read(path)["study"]["learning_complete"]:
        return
    tasks = protocol["roots"][root]["tasks"]
    state = {"tasks": tasks, "results": [], "revisits": [], "pending_revisits": [],
        "costs": dict.fromkeys(COSTS, 0), "knowledge": {},
        "study": {"steps": 0, "cpu_seconds": 0., "learning_complete": False}}
    for position, task in enumerate(tasks):
        task_path = directory / f"task{position:02d}.json.gz"
        if task_path.exists():
            learned = read(task_path)
        else:
            session = module.UnifiedResearchSession(module.ResearchConfig(**protocol["config"]),
                tasks=[task], measurement_budget=0, lab_factory=laboratory)
            started = time.process_time()
            try:
                learned = session.run()
            except Exception as error:
                save(directory / "failure.json", {"error": repr(error), "task_position": position,
                    "interrupted_cpu_seconds": time.process_time() - started})
                raise
            learned["cpu_seconds"] = time.process_time() - started
            save(task_path, learned)
        state["results"].append({**learned["results"][0], "task_position": position})
        state["revisits"].extend({**visit, "task_position": position, "source_task_position": position}
            for visit in learned["revisits"])
        for key in COSTS:
            state["costs"][key] += learned["costs"][key]
        state["study"]["cpu_seconds"] += learned["cpu_seconds"]
        state["study"]["steps"] = position + 1
        state["study"]["learning_complete"] = position + 1 == len(tasks)
        save(path, state)
        print({"arm": arm, "root": root, "steps": position + 1,
            "complete": state["study"]["learning_complete"], "costs": state["costs"]}, flush=True)


def learn(output, protocol, arm, root):
    if protocol["arms"][arm]["knowledge"] == "independent":
        return learn_independent(output, protocol, arm, root)
    module = load_runtime(protocol, arm)
    path = run_dir(output, arm, root) / "checkpoint.json.gz"
    if path.exists():
        state = read(path)
        if state["study"]["learning_complete"]:
            return
        session = module.UnifiedResearchSession(state=state, lab_factory=laboratory)
    else:
        session = module.UnifiedResearchSession(module.ResearchConfig(**protocol["config"]),
            tasks=protocol["roots"][root]["tasks"], measurement_budget=0, lab_factory=laboratory)
        session.state["study"] = {"steps": 0, "cpu_seconds": 0., "learning_complete": False}
    while len(session.state["results"]) < len(session.state["tasks"]) or session.state["pending_revisits"]:
        started = time.process_time()
        try:
            state = session.run(max_tasks=1)
        except Exception as error:
            save(path.parent / "failure.json", {"error": repr(error),
                "steps": session.state["study"]["steps"],
                "interrupted_cpu_seconds": time.process_time() - started})
            raise
        state["study"]["steps"] += 1
        state["study"]["cpu_seconds"] += time.process_time() - started
        state["study"]["learning_complete"] = (len(state["results"]) == len(state["tasks"])
            and not state["pending_revisits"])
        save(path.parent / f"visit{state['study']['steps']:02d}.json.gz", state)
        save(path, state)
        print({"arm": arm, "root": root, "steps": state["study"]["steps"],
            "complete": state["study"]["learning_complete"], "costs": state["costs"]}, flush=True)
        session = module.UnifiedResearchSession(state=state, lab_factory=laboratory)


def evaluate(output, protocol, arm, root):
    for name in protocol["arms"]:
        for key in protocol["roots"]:
            if not read(run_dir(output, name, key) / "checkpoint.json.gz")["study"]["learning_complete"]:
                raise ValueError("all arms must finish learning before terminal scoring")
    load_runtime(protocol, arm)
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    path = run_dir(output, arm, root) / "evaluation.json"
    if path.exists():
        return
    state = read(path.parent / "checkpoint.json.gz")
    scored, settings = [], protocol["terminal"]
    for position, task in enumerate(state["tasks"]):
        report = state["results"][position]["research"]
        program = Program.from_dict(report["program"])
        started = time.process_time()
        with laboratory({**task, "noise_seed": task["seed"] + settings["seed_offset"]}) as lab:
            errors, mean_errors = [], []
            for index in range(settings["trajectories"]):
                probe = lab.random_probe(task["seed"] + settings["seed_offset"] + 10007 * index,
                    length=settings["length"], name=f"terminal_{index}")
                observations = lab.acquire(probe).observations
                errors.append(loss(program, observations, report["fallback"], .5))
                with laboratory({**task, "noise_sigma": 0.}) as oracle:
                    expected = oracle.acquire(probe).observations
                mean_rows = [replace(row, target=truth.target)
                    for row, truth in zip(observations, expected)]
                mean_errors.append(loss(program, mean_rows, report["fallback"], .5))
            scored.append({"position": position, "stage": protocol["stages"][position],
                "mse": sum(errors) / len(errors), "trajectories": errors,
                "mean_response_mse": sum(mean_errors) / len(mean_errors),
                "mean_response_trajectories": mean_errors, "status": report["status"],
                "costs": {**lab.costs,
                    "compute_cost": 2 * settings["length"] * settings["trajectories"]},
                "cpu_seconds": time.process_time() - started})
    save(path, scored)


def summarize(output, protocol):
    rows, totals = [], {}
    for arm in protocol["arms"]:
        total = {"tasks": 0, "completed": 0, "solved": 0, "false_completed": 0, "revisits": 0,
            "conditional_program_revisits": 0, "reused_conditional_program_revisits": 0,
            "revisit_actions": {},
            "learning_costs": dict.fromkeys(COSTS, 0), "terminal_costs": dict.fromkeys(COSTS, 0),
            "learning_cpu_seconds": 0., "terminal_cpu_seconds": 0.}
        for root in protocol["roots"]:
            directory = run_dir(output, arm, root)
            state = read(directory / "checkpoint.json.gz")
            for key in COSTS:
                total["learning_costs"][key] += state["costs"][key]
            total["learning_cpu_seconds"] += state["study"]["cpu_seconds"]
            total["revisits"] += len(state["revisits"])
            for index, visit in enumerate(state["revisits"]):
                following = next((item for item in state["revisits"][index + 1:]
                    if item["task_position"] == visit["task_position"]), None)
                revised = (following["previous"] if following else
                    state["results"][visit["task_position"]]["research"])
                procedure = revised.get("research_program") or {}
                conditional = any(step["operation"] == "discover_conditional_coordinates"
                    for step in procedure.get("selection_steps", []))
                total["conditional_program_revisits"] += conditional
                total["reused_conditional_program_revisits"] += conditional and bool(
                    procedure.get("reused_procedure"))
                added_actions = revised["actions"][len(visit["previous"]["actions"]):]
                total["revisit_actions"] = dict(Counter(total["revisit_actions"]) +
                    Counter(action["action"] for action in added_actions))
            for score in read(directory / "evaluation.json"):
                report = state["results"][score["position"]]["research"]
                total["tasks"] += 1
                total["completed"] += score["status"] == "completed"
                solved = score["mean_response_mse"] <= protocol["pass_mse"]
                total["solved"] += solved
                total["false_completed"] += score["status"] == "completed" and not solved
                for key in COSTS:
                    total["terminal_costs"][key] += score["costs"][key]
                total["terminal_cpu_seconds"] += score["cpu_seconds"]
                procedure = report.get("research_program") or {}
                rows.append({"arm": arm, "root": root, **score,
                    "kind": report["selected_kind"], "learning_costs": report["costs"],
                    "used_sources": report["used_sources"],
                    "actions": dict(Counter(a["action"] for a in report["actions"])),
                    "research_operations": [step["operation"]
                        for step in procedure.get("selection_steps", [])],
                    "reused_procedure": bool(procedure.get("reused_procedure")),
                    "conditional_program": any(step["operation"] == "discover_conditional_coordinates"
                        for step in procedure.get("selection_steps", [])),
                    "conditions": [{key: step.get(key) for key in
                        ("gate_coordinate", "threshold", "threshold_interval", "width", "placement")}
                        for step in procedure.get("selection_steps", [])
                        if step["operation"] == "discover_conditional_coordinates"],
                    "executable_structure": executable_structure(report["program"]),
                    "response_questions": Counter(question["status"]
                        for question in report.get("response_questions", []))})
        selected = [row for row in rows if row["arm"] == arm]
        total["mean_mse"] = sum(row["mse"] for row in selected) / len(selected)
        total["mean_response_mse"] = sum(row["mean_response_mse"] for row in selected) / len(selected)
        total["all_costs"] = {key: total["learning_costs"][key] + total["terminal_costs"][key]
            for key in COSTS}
        total["actions"] = dict(sum((Counter(row["actions"]) for row in selected), Counter()))
        totals[arm] = total
    comparisons = {}
    for reference in ("old_cumulative", "new_independent"):
        matched = {(row["root"], row["position"]): row for row in rows if row["arm"] == reference}
        paired = [{"root": row["root"], "position": row["position"], "stage": row["stage"],
            "candidate_mse": row["mean_response_mse"],
            "reference_mse": matched[(row["root"], row["position"])]["mean_response_mse"],
            "difference": row["mean_response_mse"] -
                matched[(row["root"], row["position"])]["mean_response_mse"]}
            for row in rows if row["arm"] == "new_cumulative"]
        comparisons[reference] = {"paired_tasks": len(paired),
            "wins": sum(row["difference"] < -1e-12 for row in paired),
            "ties": sum(abs(row["difference"]) <= 1e-12 for row in paired),
            "losses": sum(row["difference"] > 1e-12 for row in paired),
            "mean_mse_change": totals["new_cumulative"]["mean_response_mse"] /
                totals[reference]["mean_response_mse"] - 1,
            "research_cpu_ratio": totals["new_cumulative"]["learning_cpu_seconds"] /
                totals[reference]["learning_cpu_seconds"],
            "worst_deteriorations": sorted(paired, key=lambda row: row["difference"], reverse=True)[:5]}
    by_stage = {}
    for position, stage in enumerate(protocol["stages"]):
        by_stage[stage] = {}
        for arm in protocol["arms"]:
            selected = [row for row in rows if row["arm"] == arm and row["position"] == position]
            by_stage[stage][arm] = {"tasks": len(selected),
                "mean_response_mse": sum(row["mean_response_mse"] for row in selected) / len(selected),
                "completed": sum(row["status"] == "completed" for row in selected),
                "solved": sum(row["mean_response_mse"] <= protocol["pass_mse"] for row in selected),
                "selected_kinds": dict(Counter(row["kind"] for row in selected)),
                "conditional_programs": sum(row["conditional_program"] for row in selected),
                "placements": dict(Counter(condition["placement"] for row in selected
                    for condition in row["conditions"])),
                "discovered_widths": dict(Counter(str(condition["width"]) for row in selected
                    for condition in row["conditions"])),
                "executable_gate_placements": dict(Counter(condition["placement"]
                    for row in selected for condition in row["executable_structure"]["conditions"])),
                "executable_window_widths": dict(Counter(str(width) for row in selected
                    for width in row["executable_structure"]["window_widths"])),
                "reused_procedures": sum(row["reused_procedure"] for row in selected)}
    summary = {"protocol": str(output / "protocol.json"), "totals": totals,
        "comparisons": comparisons, "by_stage": by_stage, "rows": rows}
    save(output / "summary.json", summary)
    print({"totals": totals, "comparisons": comparisons}, flush=True)


def dispatch(output, protocol, phase, arms, workers):
    jobs = [(arm, root) for arm in arms or protocol["arms"] for root in protocol["roots"]]
    def run(job):
        arm, root = job
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
            "--phase", phase, "--arm", arm, "--root", root], cwd=ROOT, check=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run, jobs))
    print({"phase": phase, "finished_jobs": len(jobs)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("learn", "evaluate", "summarize"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--arm")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    if args.phase == "summarize":
        summarize(output, protocol)
    elif args.root:
        {"learn": learn, "evaluate": evaluate}[args.phase](output, protocol, args.arm, args.root)
    else:
        dispatch(output, protocol, args.phase, args.arms, args.workers)


if __name__ == "__main__":
    main()
