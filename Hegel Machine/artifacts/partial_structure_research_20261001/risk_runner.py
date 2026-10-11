#!/usr/bin/env python3
"""Reuse the existing six-stage public risk study with frozen arms and true task reset."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_research_investment_study as shared

ARMS = ("old_cumulative", "new_cumulative", "new_independent")
STAGES = ("cold_A", "reuse_A", "changed_B", "changed_A_parameters", "return_B", "return_A")
shared.ARMS, shared.FAMILIES, shared.STAGES = ARMS, ("risk",), STAGES
original_load = shared.load


def load(protocol, arm, cohort):
    module, _, factory = original_load(protocol, arm, cohort)
    return module, module.UnifiedResearchSession, factory


shared.load = load


def learn(protocol, output, cohort, arm, seed):
    if arm != "new_independent":
        shared.learn(protocol, output, cohort, "risk", arm, seed)
        return
    path = shared.checkpoint(output, cohort, "risk", arm, seed)
    state = json.loads(path.read_text()) if path.exists() else {
        "tasks": protocol["tasks_by_family_seed"]["risk"][str(seed)], "results": [],
        "revisits": [], "pending_revisits": [], "scheduler_events": [], "evaluation": [],
        "knowledge": {}, "costs": dict.fromkeys(shared.COSTS, 0), "screening_compute_cost": 0,
        "study_visits": [], "study_initial_models": {}, "study_learning_complete": False,
        "independent_single_task_sessions": True}
    if state["study_learning_complete"]:
        return
    for position in range(len(state["results"]), len(STAGES)):
        one = deepcopy(protocol)
        one["tasks_by_family_seed"]["risk"][str(seed)] = [state["tasks"][position]]
        one["families"]["risk"]["stream_budget"] = {
            "observation_cost": one["families"]["risk"]["research_config"]["observation_budget"],
            "measurement_cost": one["families"]["risk"]["measurement_budget"],
            "compute_cost": one["families"]["risk"]["research_config"]["compute_budget"]}
        single_output = output / "independent_tasks" / f"task{position:02d}"
        shared.STAGES = (STAGES[position],)
        try:
            shared.learn(one, single_output, cohort, "risk", arm, seed)
        finally:
            shared.STAGES = STAGES
        learned = json.loads(shared.checkpoint(single_output, cohort, "risk", arm, seed).read_text())
        state["results"].append({**learned["results"][0], "task_position": position})
        state["revisits"].extend({**visit, "task_position": position, "source_task_position": position}
                                 for visit in learned["revisits"])
        for event in learned["scheduler_events"]:
            event = deepcopy(event)
            for key in ("task_position", "source_task_position"):
                if event.get(key) == 0:
                    event[key] = position
            state["scheduler_events"].append(event)
        for visit in learned["study_visits"]:
            state["study_visits"].append({**visit,
                "position": position if visit["position"] is not None else None})
        state["study_initial_models"][str(position)] = learned["study_initial_models"]["0"]
        state["knowledge"] = learned["knowledge"]
        state["screening_compute_cost"] += learned["screening_compute_cost"]
        for key in shared.COSTS:
            state["costs"][key] += learned["costs"][key]
        state["study_learning_complete"] = position == len(STAGES) - 1
        shared.save(path, state)
        print(json.dumps({"arm": arm, "seed": seed, "steps": position + 1,
            "finished": state["study_learning_complete"]}), flush=True)


def evaluate(protocol, output, cohort, arm, seed):
    started = time.process_time()
    shared.evaluate(protocol, output, cohort, "risk", arm, seed)
    path = shared.checkpoint(output, cohort, "risk", arm, seed)
    state = json.loads(path.read_text())
    state["study_terminal_cpu_seconds"] = state.get("study_terminal_cpu_seconds", 0.) + time.process_time() - started
    shared.save(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("learn", "evaluate", "summarize"), required=True)
    parser.add_argument("--cohort", choices=("development", "fresh_replication"), required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    if args.phase == "summarize":
        shared.summarize(protocol, output, args.cohort)
        return
    if args.arm is not None:
        (learn if args.phase == "learn" else evaluate)(protocol, output, args.cohort, args.arm, args.seed)
        return
    jobs = [(arm, seed) for arm in args.arms for seed in protocol["cohorts"][args.cohort]]
    def execute(job):
        arm, seed = job
        log = output / args.cohort / "logs" / f"{args.phase}_{arm}_{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                "--phase", args.phase, "--cohort", args.cohort, "--arm", arm, "--seed", str(seed)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"phase": args.phase, "cohort": args.cohort, "arm": arm,
            "seed": seed, "complete": True}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(execute, jobs))


if __name__ == "__main__":
    main()
