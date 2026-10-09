#!/usr/bin/env python3
"""Frozen whole-session comparison of literal and revisable dependency steps."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from random import Random
import subprocess
import sys

import run_anonymous_revisit_study as study

ROOT = study.ROOT
OUTPUT = ROOT / "artifacts/dependency_method_revision_20260920"
ARMS = ("fixed_methods", "rewritten_methods")
study.ARMS = (*ARMS, "unused_deletion")
# Both versions already implement pooled budgets and ordinary session.run.
# Neither uses the legacy 0.100 initial-visit adapter.
study.flow.ARMS = ("unused_legacy_adapter", ARMS[1], "unused_deletion")


def prepare(output):
    if (output / "protocol.json").exists():
        raise ValueError("preserve the declared protocol")
    prior = json.loads((ROOT / "artifacts/anonymous_revisit_study_20260920/protocol.json").read_text())
    version = (ROOT / "pyproject.toml").read_text().split('version = "', 1)[1].split('"', 1)[0]
    roots = list(range(96002001, 96002017))
    known = [95902002, 95902006]
    tasks = {str(seed): deepcopy(prior["tasks_by_seed"][str(seed)]) for seed in known}
    for index, seed in enumerate(roots):
        objects, readout = ((3, "focus"), (3, "event"), (4, "focus"), (4, "event"))[index % 4]
        rng = Random(seed + 17011)
        shared = {"objects": objects, "apparatus_seed": seed, "name_seed": seed + 63857,
            "wiring_seed": seed + 3709, "variant": "exchange", "profile": "confirmation",
            "actuator_curvature": rng.uniform(.08, .24), "relation_radius": rng.uniform(.70, 1.20),
            "environment": "anonymous_objects"}
        tasks[str(seed)] = [{**shared, "readout": readout, "seed": seed,
            "research_limits": {"observation_budget": 96, "compute_budget": 30000,
                                "measurement_budget": 1152}},
            {**shared, "readout": "focus", "seed": seed + 10007}]
    protocol = {key: deepcopy(prior[key]) for key in ("research_config", "measurement_budget",
        "total_research_budget", "terminal", "initial_contract", "schedule_contract",
        "evaluation_contract", "role_evaluation_contract", "shared_laboratory_source")}
    protocol.update(experiment="paid_contrasts_rewrite_dependency_experiments",
        fixed_at=datetime.now(timezone.utc).isoformat(), cohorts={"development": known, "main": roots},
        tasks_by_seed=tasks, development_runtime="src", maximum_concurrent_processes=4,
        arms={ARMS[0]: {"runtime": "artifacts/anonymous_revisit_study_20260920/repair_v01011/runtime/src",
                       "version": "0.101.1"},
              ARMS[1]: {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": version}},
        scope="Sixteen new apparatus/name/wiring roots, four per 3/4-object by focus/event cell. "
              "Both arms start empty, finish A+B and autonomously revisit within the same original total pool. "
              "Two previously observed event roots are development only. No main-root tuning.",
        primary_outcomes=["paired final random AND public-step prediction pass", "exact object roles",
                          "total observation, measurement and compute cost", "reset membership rows"],
        formal_evidence=False)
    study.save(output / "protocol.json", protocol)


def summarize(protocol, output, cohort):
    result = {"cohort": cohort, "scope": protocol["scope"], "arms": {}, "paired": []}
    for arm in ARMS:
        costs, terminal, actions, rows = Counter(), Counter(), Counter(), []
        for seed in protocol["cohorts"][cohort]:
            state = json.loads(study.flow.checkpoint(output, cohort, arm, seed).read_text())
            costs.update(state["costs"])
            for measured, record in zip(state["evaluation"], state["results"]):
                report = record["research"]
                terminal.update(measured["costs"])
                events = report["actions"]
                actions.update(event["action"] for event in events)
                membership = [event for event in events
                    if event["action"] == "test_object_dependency_question"
                    and event["question"].get("purpose") == "distinguish_membership_dependency"]
                rows.append({"seed": seed, "position": record["task_position"],
                    "objects": state["tasks"][record["task_position"]]["objects"],
                    "readout": state["tasks"][record["task_position"]]["readout"],
                    "mse": measured["mse"], "initial_mse": measured["initial_mse"],
                    "pass": measured["mse"] <= protocol["terminal"]["mse_tolerance"],
                    "exact_roles": measured["object_evaluation"]["exact"],
                    "costs": report["costs"], "selected_channels": report["selected_channels"],
                    "reset_membership_rows": sum(e["observation_rows"] for e in membership
                        if not e["question"].get("matched_prefix_rows")),
                    "history_membership_rows": sum(e["observation_rows"] for e in membership
                        if e["question"].get("matched_prefix_rows")),
                    "rewrites": sum(e["action"] == "rewrite_object_dependency_method" for e in events)})
        result["arms"][arm] = {"passes": sum(row["pass"] for row in rows), "tasks": rows,
            "exact_roles": sum(row["exact_roles"] for row in rows), "costs": dict(costs),
            "terminal_costs": dict(terminal), "actions": dict(actions),
            "reset_membership_rows": sum(row["reset_membership_rows"] for row in rows),
            "event_reset_rows": sum(row["reset_membership_rows"] for row in rows if row["readout"] == "event")}
    for before, after in zip(*(result["arms"][arm]["tasks"] for arm in ARMS)):
        result["paired"].append({"seed": before["seed"], "position": before["position"],
            "before_mse": before["mse"], "after_mse": after["mse"],
            "pass_change": int(after["pass"]) - int(before["pass"])})
    study.save(output / cohort / "summary.json", result)
    print(json.dumps({arm: {k: v for k, v in row.items() if k not in ("tasks", "actions")}
                      for arm, row in result["arms"].items()}), flush=True)


def queue(protocol, output, cohort):
    history = {"status": "running", "cohort": cohort, "jobs": []}
    def run(job):
        mode, arm, seed = job
        command = [sys.executable, "-B", str(Path(__file__).resolve()), mode, "--output", str(output),
                   "--cohort", cohort, "--arm", arm, "--seed", str(seed)]
        log = output / cohort / "logs" / f"{mode}_{arm}_{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        return {"job": job, "exit_code": code}
    for phase in ("initial", "revisit", "evaluate"):
        jobs = [(phase, arm, seed) for seed in protocol["cohorts"][cohort] for arm in ARMS]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for completed in pool.map(run, jobs):
                history["jobs"].append(completed)
                study.save(output / cohort / "execution.json", history)
                print(json.dumps(completed), flush=True)
        if any(item["exit_code"] for item in history["jobs"]):
            history["status"] = "failed"
            study.save(output / cohort / "execution.json", history)
            raise SystemExit(1)
    history["status"] = "completed"
    study.save(output / cohort / "execution.json", history)
    summarize(protocol, output, cohort)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "freeze", "initial", "revisit", "evaluate", "queue", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cohort", choices=("development", "main"), default="main")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "freeze":
        study.common.freeze(args.output)
    else:
        protocol = json.loads((args.output / "protocol.json").read_text())
        if args.mode in ("queue", "summarize"):
            globals()[args.mode](protocol, args.output, args.cohort)
        elif args.mode == "evaluate":
            study.evaluate(protocol, args.output, args.cohort, args.arm, args.seed, learning_arms=ARMS)
        else:
            getattr(study.flow, args.mode)(protocol, args.output, args.cohort, args.arm, args.seed)
