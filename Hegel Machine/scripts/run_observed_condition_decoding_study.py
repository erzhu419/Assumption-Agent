#!/usr/bin/env python3
"""Compare paid observed-condition decoding and active discrimination with uncontrollable process readings."""

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
    from dataclasses import replace
    from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
    from hegel_machine.cognition.models import Observation, Probe, TaskSpec

    class ObservedConditionLab:
        """A purchased process reading cannot be set as an experimental control."""

        def __init__(self):
            apparatus = Random(task["apparatus_seed"])
            self._roles = list(range(3))
            apparatus.shuffle(self._roles)
            self._threshold = apparatus.uniform(-.35, .35)
            self._drift = apparatus.uniform(.2, .55)
            self._first = (apparatus.uniform(-.3, .3), apparatus.uniform(.65, 1.25),
                           apparatus.uniform(.35, .8))
            self._second = (apparatus.uniform(-.3, .3), apparatus.uniform(-1.25, -.65),
                            apparatus.uniform(-.8, -.35))
            self._stage = task["stage"]
            self._width = (3, 3, 5, 7, 7, 3, 5)[self._stage]
            if self._stage in (3, 4):
                self._threshold += .25
            self._names = ("p", "q", "r") if self._stage == 4 else ("a", "b", "c")
            self._control_scale = 2.3 if self._stage == 4 else 1.
            self._meter_scale = 2.3 if self._stage == 4 else 1.
            self._meter_offset = -.4 if self._stage == 4 else 0.
            self._response_scale = 1.25 if self._stage == 4 else 1.
            self._channel = "reading" if self._stage == 4 else "meter"
            if task.get("known_fixture"):
                self._roles = [0, 1, 2]
                self._threshold, self._drift = .173, .35
                self._first, self._second = (.13, .8, .6), (-.11, -.9, -.45)
                self._names = ("alpha", "beta", "gamma")
            self._random = Random(task.get("process_seed", task["seed"]))
            self._experiments = {}
            bounds = (-1.5 * self._control_scale, 1.5 * self._control_scale)
            self.spec = TaskSpec(f"observed_response_{task['seed']}",
                "Apply controllable input sequences and predict the measured response.",
                self._names, (), (), feature_bounds=dict.fromkeys(self._names, bounds),
                max_probe_length=64)
            control = self._names[self._roles[2]]
            calibration = (f"Purchase the auxiliary process reading. Its physical calibration is "
                f"{self._channel} = {self._meter_scale!r} * "
                f"({self._drift!r} * {control} / {self._control_scale!r} + Uniform(-1, 1)) "
                f"+ {self._meter_offset!r}. A repeated experiment preserves its process reading; "
                "a new experiment samples a new independent perturbation. This reading cannot be set.")
            self.channels = (MeasurementChannel(self._channel, 1, calibration),)
            self.observation_cost, self.measurement_cost, self.acquisitions = 0, 0, []

        def spec_for(self, channels=()):
            assert set(channels) <= {self._channel}
            return replace(self.spec, feature_names=(*self._names, *channels))

        @property
        def costs(self):
            return {"observation_cost": self.observation_cost,
                    "measurement_cost": self.measurement_cost, "queries": len(self.acquisitions)}

        def acquire(self, probe, channels=()):
            assert set(channels) <= {self._channel}
            assert 0 < len(probe.sequence) <= self.spec.max_probe_length
            assert all(set(point) == set(self.spec.feature_bounds) and
                all(low <= point[name] <= high for name, (low, high) in self.spec.feature_bounds.items())
                for point in probe.sequence)
            key = probe.probe_id, tuple(tuple(point[name] for name in self._names) for point in probe.sequence)
            if key not in self._experiments:
                raw, first, second, conditions = [], [], [], []
                for point in probe.sequence:
                    values = [point[name] / self._control_scale for name in self._names]
                    x, z, driver = (values[index] for index in self._roles)
                    latent = self._drift * driver + self._random.uniform(-1., 1.)
                    reading = self._meter_scale * latent + self._meter_offset
                    first.append(self._first[0] + self._first[1] * x + self._first[2] * z * z + .25 * latent)
                    second.append(self._second[0] + self._second[1] * x + self._second[2] * z * z + .25 * latent)
                    conditions.append(latent >= self._threshold)
                    start = max(0, len(first) - self._width)
                    if self._stage in (0, 6):
                        target = sum(first[start:])
                    elif self._stage in (2, 4):
                        target = sum((first if conditions[-1] else second)[start:])
                    else:
                        target = sum(first[index] if conditions[index] else second[index]
                                     for index in range(start, len(first)))
                    raw.append(({**point, self._channel: reading}, target * self._response_scale))
                self._experiments[key] = raw
            rows, history = [], []
            names = self.spec_for(channels).feature_names
            for readings, target in self._experiments[key]:
                features = {name: readings[name] for name in names}
                rows.append(Observation(features, target, tuple(history)))
                history.append({**features, "__target__": target})
            observation_cost = len(rows)
            measurement_cost = len(rows) if self._channel in channels else 0
            self.observation_cost += observation_cost
            self.measurement_cost += measurement_cost
            self.acquisitions.append({"probe_id": probe.probe_id,
                "sequence": list(probe.sequence), "channels": list(channels)})
            return MeasurementBatch(rows, observation_cost, measurement_cost, tuple(channels))

        def random_probe(self, seed, *, length=24, name="sample"):
            rng = Random(seed)
            return Probe(name, tuple({key: rng.uniform(*bounds)
                for key, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

        def restore(self, acquisitions):
            self.__init__()
            for item in acquisitions:
                self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item.get("channels", ()))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    return ObservedConditionLab()

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
                    tasks=[task], measurement_budget=protocol["limits"]["first_visit_measurement"], lab_factory=laboratory)
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
            tasks=tasks, measurement_budget=protocol["limits"]["first_visit_measurement"], lab_factory=laboratory)
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
    from run_research_investment_attribution_study import confirm_learning_complete
    confirm_learning_complete(output, protocol)
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
        with laboratory({**task, "process_seed": task["seed"] + settings["seed_offset"]}) as lab:
            errors = []
            for index in range(settings["trajectories"]):
                probe = lab.random_probe(task["seed"] + settings["seed_offset"] + 10007 * index,
                    length=settings["length"], name=f"terminal_{index}")
                errors.append(loss(program, lab.acquire(probe, report["selected_channels"]).observations, report["fallback"], .5))
            scored.append({"position": position, "stage": protocol["stages"][position],
                "mse": sum(errors) / len(errors), "trajectories": errors,
                "mean_response_mse": sum(errors) / len(errors), "status": report["status"],
                "selected_channels": report["selected_channels"],
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
    for reference in (name for name in protocol["arms"] if name != "new_cumulative"):
        if reference not in protocol["arms"]:
            continue
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
