#!/usr/bin/env python3
"""Compare paid discovery of multiple conditions across response-window layers."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
import subprocess
import sys
import time

from run_knowledge_lineage_study import ROOT, COSTS, read, save, load_runtime, checkpoint_path


def laboratory(task):
    from hegel_machine.cognition.measurement_lab import MeasurementBatch
    from hegel_machine.cognition.models import Observation, Probe, TaskSpec

    class ComposedConditionLab:
        channels = ()

        def __init__(self):
            apparatus = Random(task["apparatus_seed"])
            self._roles = list(range(4))
            apparatus.shuffle(self._roles)
            self._thresholds = (apparatus.uniform(-.35, .35), apparatus.uniform(-.35, .35))
            self._first = (apparatus.uniform(-.3, .3), apparatus.uniform(.65, 1.25),
                           apparatus.uniform(.35, .8))
            self._second = (apparatus.uniform(-.3, .3), apparatus.uniform(-1.25, -.65),
                            apparatus.uniform(-.8, -.35))
            self._stage = task["stage"]
            self._width = (3, 3, 5, 7, 7, 3, 5)[self._stage]
            if self._stage in (3, 4):
                self._thresholds = (self._thresholds[0] + .25, self._thresholds[1] - .2)
            self._names = ("p", "q", "r", "s") if self._stage == 4 else ("a", "b", "c", "d")
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

        def _logic(self, first, second):
            if self._stage == 2:
                return first or second
            if self._stage == 4:
                return first != second
            return first and second

        def acquire(self, probe, channels=()):
            assert not channels
            rows, history, first, second, gate1, gate2 = [], [], [], [], [], []
            for point in probe.sequence:
                values = [point[name] / self._control_scale for name in self._names]
                x, z, c, d = (values[index] for index in self._roles)
                first.append(self._first[0] + self._first[1] * x + self._first[2] * z * z)
                second.append(self._second[0] + self._second[1] * x + self._second[2] * z * z)
                gate1.append(c >= self._thresholds[0])
                gate2.append(d >= self._thresholds[1])
                start = max(0, len(first) - self._width)
                if self._stage in (0, 6):
                    target = sum(first[start:])
                elif self._stage in (1, 5):
                    target = sum((first if self._logic(gate1[-1], gate2[-1]) else second)[start:])
                elif self._stage == 2:
                    target = sum(first[index] if self._logic(gate1[index], gate2[index])
                                 else second[index] for index in range(start, len(first)))
                elif self._stage == 3:
                    target = sum(first[index] if self._logic(gate1[index], gate2[-1])
                                 else second[index] for index in range(start, len(first)))
                else:
                    target = sum(first[index] if self._logic(gate1[-1], gate2[index])
                                 else second[index] for index in range(start, len(first)))
                target *= self._response_scale
                target += self._noise.gauss(0., task.get("noise_sigma", 0.))
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

    return ComposedConditionLab()


def run_dir(output, arm, root):
    return output / "runs" / arm / root


def learn(output, protocol, arm, root):
    module = load_runtime(protocol, arm)
    directory = run_dir(output, arm, root)
    path = checkpoint_path(directory / "checkpoint.json")
    if path.exists() and read(path)["study"]["learning_complete"]:
        return
    tasks = protocol["roots"][root]["tasks"]
    if protocol["arms"][arm]["knowledge"] == "independent":
        state = {"tasks": tasks, "results": [], "revisits": [], "pending_revisits": [],
            "costs": dict.fromkeys(COSTS, 0), "knowledge": {},
            "study": {"steps": 0, "cpu_seconds": 0., "elapsed_seconds": 0., "learning_complete": False}}
        for position, task in enumerate(tasks):
            task_path = checkpoint_path(directory / f"task{position:02d}.json")
            if task_path.exists():
                learned = read(task_path)
            else:
                session = module.UnifiedResearchSession(module.ResearchConfig(**protocol["config"]),
                    tasks=[task], measurement_budget=0, lab_factory=laboratory)
                cpu, wall = time.process_time(), time.monotonic()
                try:
                    learned = session.run()
                except Exception as error:
                    save(directory / "failure.json", {"error": repr(error), "task_position": position,
                        "interrupted_cpu_seconds": time.process_time() - cpu})
                    raise
                learned["cpu_seconds"] = time.process_time() - cpu
                learned["elapsed_seconds"] = time.monotonic() - wall
                save(task_path, learned)
            state["results"].append({**learned["results"][0], "task_position": position})
            state["revisits"].extend({**visit, "task_position": position, "source_task_position": position}
                for visit in learned["revisits"])
            for key in COSTS:
                state["costs"][key] += learned["costs"][key]
            state["study"]["cpu_seconds"] += learned["cpu_seconds"]
            state["study"]["elapsed_seconds"] += learned["elapsed_seconds"]
            state["study"]["steps"] = position + 1
            state["study"]["learning_complete"] = position + 1 == len(tasks)
            save(path, state)
            print({"arm": arm, "root": root, "steps": position + 1,
                "complete": state["study"]["learning_complete"], "costs": state["costs"]}, flush=True)
        return
    if path.exists():
        session = module.UnifiedResearchSession(state=read(path), lab_factory=laboratory)
    else:
        session = module.UnifiedResearchSession(module.ResearchConfig(**protocol["config"]),
            tasks=tasks, measurement_budget=0, lab_factory=laboratory)
        session.state["study"] = {"steps": 0, "cpu_seconds": 0., "elapsed_seconds": 0., "learning_complete": False}
    while len(session.state["results"]) < len(tasks) or session.state["pending_revisits"]:
        cpu, wall = time.process_time(), time.monotonic()
        try:
            state = session.run(max_tasks=1)
        except Exception as error:
            save(directory / "failure.json", {"error": repr(error),
                "steps": session.state["study"]["steps"], "interrupted_cpu_seconds": time.process_time() - cpu})
            raise
        state["study"]["steps"] += 1
        state["study"]["cpu_seconds"] += time.process_time() - cpu
        state["study"]["elapsed_seconds"] += time.monotonic() - wall
        state["study"]["learning_complete"] = (len(state["results"]) == len(tasks)
            and not state["pending_revisits"])
        save(path, state)
        save(checkpoint_path(directory / f"visit{state['study']['steps']:02d}.json"), state, base=path)
        print({"arm": arm, "root": root, "steps": state["study"]["steps"],
            "complete": state["study"]["learning_complete"], "costs": state["costs"]}, flush=True)
        session = module.UnifiedResearchSession(state=state, lab_factory=laboratory)


def executable_structure(program):
    widths, conditions = set(), []

    def walk(node, ancestors=()):
        if node["op"] == "window_sum":
            widths.add(node["lag"])
        if node["op"] == "if":
            condition = {"expression": node["children"][0],
                "placement": "inside_window" if "window_sum" in ancestors else "outside_window"}
            if condition not in conditions:
                conditions.append(condition)
        for child in node.get("children", []):
            walk(child, (*ancestors, node["op"]))
        if node["op"] == "call":
            walk(program["definitions"][node["name"]]["body"], (*ancestors, "call"))

    walk(program)
    return {"window_widths": sorted(widths), "conditions": conditions}


def evaluate(output, protocol, arm, root):
    for name in protocol["arms"]:
        for key in protocol["roots"]:
            if not read(checkpoint_path(run_dir(output, name, key) / "checkpoint.json"))["study"]["learning_complete"]:
                raise ValueError("all arms must finish learning before terminal scoring")
    load_runtime(protocol, arm)
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    path = run_dir(output, arm, root) / "evaluation.json"
    if path.exists():
        return
    state = read(checkpoint_path(path.parent / "checkpoint.json"))
    scored, settings = [], protocol["terminal"]
    for position, task in enumerate(state["tasks"]):
        report = state["results"][position]["research"]
        program = Program.from_dict(report["program"])
        cpu = time.process_time()
        with laboratory({**task, "noise_seed": task["seed"] + settings["seed_offset"]}) as lab:
            errors = []
            for index in range(settings["trajectories"]):
                probe = lab.random_probe(task["seed"] + settings["seed_offset"] + 10007 * index,
                    length=settings["length"], name=f"terminal_{index}")
                errors.append(loss(program, lab.acquire(probe).observations, report["fallback"], .5))
            scored.append({"position": position, "stage": protocol["stages"][position],
                "mse": sum(errors) / len(errors), "trajectories": errors,
                "mean_response_mse": sum(errors) / len(errors), "status": report["status"],
                "costs": {**lab.costs, "compute_cost": settings["length"] * settings["trajectories"]},
                "cpu_seconds": time.process_time() - cpu})
    save(path, scored)


def summarize(output, protocol):
    rows, totals = [], {}
    for arm in protocol["arms"]:
        total = {"tasks": 0, "completed": 0, "solved": 0, "false_completed": 0,
            "revisits": 0, "revisit_actions": {}, "learning_costs": dict.fromkeys(COSTS, 0),
            "terminal_costs": dict.fromkeys(COSTS, 0), "learning_cpu_seconds": 0., "terminal_cpu_seconds": 0.}
        for root in protocol["roots"]:
            directory = run_dir(output, arm, root)
            state = read(checkpoint_path(directory / "checkpoint.json"))
            for key in COSTS:
                total["learning_costs"][key] += state["costs"][key]
            total["learning_cpu_seconds"] += state["study"]["cpu_seconds"]
            total["revisits"] += len(state["revisits"])
            for index, visit in enumerate(state["revisits"]):
                following = next((item for item in state["revisits"][index + 1:]
                    if item["task_position"] == visit["task_position"]), None)
                revised = following["previous"] if following else state["results"][visit["task_position"]]["research"]
                total["revisit_actions"] = dict(Counter(total["revisit_actions"]) +
                    Counter(action["action"] for action in revised["actions"][len(visit["previous"]["actions"]):]))
            for score in read(directory / "evaluation.json"):
                report = state["results"][score["position"]]["research"]
                solved = score["mse"] <= protocol["pass_mse"]
                total["tasks"] += 1
                total["completed"] += score["status"] == "completed"
                total["solved"] += solved
                total["false_completed"] += score["status"] == "completed" and not solved
                for key in COSTS:
                    total["terminal_costs"][key] += score["costs"][key]
                total["terminal_cpu_seconds"] += score["cpu_seconds"]
                procedure = report.get("research_program") or {}
                rows.append({"arm": arm, "root": root, **score,
                    "kind": report["selected_kind"], "learning_costs": report["costs"],
                    "used_sources": report["used_sources"],
                    "actions": dict(Counter(action["action"] for action in report["actions"])),
                    "research_operations": [step["operation"] for step in procedure.get("selection_steps", [])],
                    "reused_procedure": bool(procedure.get("reused_procedure")),
                    "executable_structure": executable_structure(report["program"])})
        selected = [row for row in rows if row["arm"] == arm]
        total["mean_response_mse"] = sum(row["mse"] for row in selected) / len(selected)
        total["all_costs"] = {key: total["learning_costs"][key] + total["terminal_costs"][key] for key in COSTS}
        total["actions"] = dict(sum((Counter(row["actions"]) for row in selected), Counter()))
        totals[arm] = total
    comparisons = {}
    for reference in ("old_cumulative", "new_independent"):
        matched = {(row["root"], row["position"]): row for row in rows if row["arm"] == reference}
        paired = [{"root": row["root"], "position": row["position"], "stage": row["stage"],
            "candidate_mse": row["mse"], "reference_mse": matched[(row["root"], row["position"])]["mse"],
            "difference": row["mse"] - matched[(row["root"], row["position"])]["mse"]}
            for row in rows if row["arm"] == "new_cumulative"]
        reference_mse = totals[reference]["mean_response_mse"]
        comparisons[reference] = {"paired_tasks": len(paired),
            "wins": sum(row["difference"] < -1e-12 for row in paired),
            "ties": sum(abs(row["difference"]) <= 1e-12 for row in paired),
            "losses": sum(row["difference"] > 1e-12 for row in paired),
            "mean_mse_change": totals["new_cumulative"]["mean_response_mse"] / reference_mse - 1 if reference_mse else None,
            "research_cpu_ratio": totals["new_cumulative"]["learning_cpu_seconds"] /
                totals[reference]["learning_cpu_seconds"],
            "worst_deteriorations": sorted(paired, key=lambda row: row["difference"], reverse=True)[:5]}
    by_stage = {}
    for position, stage in enumerate(protocol["stages"]):
        by_stage[stage] = {}
        for arm in protocol["arms"]:
            selected = [row for row in rows if row["arm"] == arm and row["position"] == position]
            by_stage[stage][arm] = {"tasks": len(selected),
                "mean_response_mse": sum(row["mse"] for row in selected) / len(selected),
                "completed": sum(row["status"] == "completed" for row in selected),
                "solved": sum(row["mse"] <= protocol["pass_mse"] for row in selected),
                "false_completed": sum(row["status"] == "completed" and row["mse"] > protocol["pass_mse"] for row in selected),
                "selected_kinds": dict(Counter(row["kind"] for row in selected))}
    save(output / "summary.json", {"protocol": str(output / "protocol.json"), "totals": totals,
        "comparisons": comparisons, "by_stage": by_stage, "rows": rows})
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
    parser.add_argument("--workers", type=int, default=4)
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
