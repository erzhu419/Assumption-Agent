#!/usr/bin/env python3
"""Compare successor-guided challenges of short-history completed tasks."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import threading
import time

import run_observation_action_study as common
from run_unified_revisit_study import _physical_costs, _check_purchases

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/completed_challenge_study_20260920"
ARMS = ("baseline_v098", "knowledge_challenge", "no_successor_knowledge")
STAGES = ("short_initial_task", "long_history_successor")
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def save(path, payload):
    common.save(path, payload)


def checkpoint(output, cohort, arm, seed):
    return output / cohort / "runs" / arm / str(seed) / "checkpoint.json"


def prepare(output):
    if (output / "protocol.json").exists():
        return
    baseline = "artifacts/unified_revisit_study_20260920/runtime/src"
    roots = [11811, *range(95702001, 95702009)]
    save(output / "protocol.json", {
        "experiment": "later_knowledge_challenges_short_history_completion",
        "fixed_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {"development": roots[:1], "main": roots[1:]},
        "tasks_by_seed": {str(seed): [
            {"task_index": 8, "seed": seed, "profile": "development",
             "research_limits": {"observation_budget": 16, "compute_budget": 512,
                                 "measurement_budget": 768}},
            {"task_index": 8, "seed": 11812 if seed == 11811 else seed + 10007,
             "profile": "development"}] for seed in roots},
        "arms": {"baseline_v098": {"runtime": baseline, "version": "0.98.0"},
                 "knowledge_challenge": {"runtime": str((output / "runtime/src").relative_to(ROOT)),
                                         "version": "0.99.0"},
                 "no_successor_knowledge": {"runtime": str((output / "runtime/src").relative_to(ROOT)),
                                            "version": "0.99.0"}},
        "development_runtime": "src",
        "shared_laboratory_source": baseline + "/hegel_machine/cognition/unknown_task_lab.py",
        "research_config": {"observation_budget": 192, "compute_budget": 120000,
            "minimum_improvement": 1e-7, "max_rounds": 4, "active_probes": False},
        "measurement_budget": 768,
        "total_research_budget": {"observation_cost": 208, "measurement_cost": 1536,
                                  "compute_cost": 120512},
        "maximum_concurrent_processes": 4,
        "terminal": {"length": 24, "trajectories": 16, "seed_offset": 1000003,
                     "risk_quantile": .5, "mse_tolerance": 1e-7},
        "initial_contract": "Every arm starts empty. A gets only 16 observation rows and 512 compute; "
            "B gets 192 rows and 120000 compute. Both use the ordinary public horizon 96. "
            "All programs and source concepts are learned through paid observations; no private "
            "mechanism coefficients or programs enter research. Persist complete JSON after every visit. "
            "Keep every fixed root, including A unresolved or already correct on long trajectories.",
        "schedule_contract": "Complete all first visits before continuations. Frozen 0.98 can retain "
            "its existing unresolved-task continuation but has no completed-task challenge. Version "
            "0.99 schedules from compatible learned predictions, then acquires the proposed real "
            "distinguishing experiment and separate fresh confirmation. All screening, candidate "
            "prediction, failures and purchases consume the same shared resource pool.",
        "baseline_interface": "The old frozen session predates per-task research_limits. The runner "
            "executes its first visits explicitly with the same limits and computes its remaining "
            "pool from their sum. Its research, screening and continuation algorithms stay frozen.",
        "knowledge_ablation_contract": "After the full new arm ends, replay its actual task revisit "
            "positions and incremental allocations using only each old task's own first-visit library. "
            "Do not pass successor concepts, candidate programs, predictions, or proposed probes. "
            "candidate=None invokes ordinary continuation with the same new-resource allocation, "
            "clipped by its own remaining pool. Completed old tasks must still receive matched visits. "
            "This is a deletion-of-knowledge ablation under a given schedule, not an autonomous baseline.",
        "evaluation_contract": "After every learning and revisit phase has ended in the cohort, "
            "evaluate every latest model, including unresolved ones, on 16 fresh trajectories of "
            "24 rows. Score its own initial model on those same actual response rows so correction "
            "of mistaken completion is separated from an initially correct model. Include the union "
            "of initial/final required channels in actual evaluation costs; also report each model's "
            "deployment channel costs separately. No terminal outcomes enter learning or scheduling.",
        "comparisons": "Report paired stream and A-task MSE, all task statuses, initial completed "
            "but long-history-inaccurate count, final correction count, screening/challenge/actual "
            "revisit counts, provided versus actually used knowledge, and complete initial/revisit/" 
            "terminal resource totals. Do not drop zero-opportunity or unsuccessful roots.",
        "scope": "Eight prospectively fixed stochastic control roots of one existing continuous-state "
            "mechanism, not eight independently parameterized apparatus. Known 11811/11812 is development "
            "only. Forty-eight initial visits plus actual continuations; no post-result retuning.",
        "formal_evidence": False})


def load(protocol, arm, cohort):
    source = ROOT / (protocol["development_runtime"] if cohort == "development" and arm != ARMS[0]
                     else protocol["arms"][arm]["runtime"])
    sys.path.insert(0, str(source))
    canonical = "hegel_machine.cognition.unknown_task_lab"
    spec = importlib.util.spec_from_file_location(canonical, ROOT / protocol["shared_laboratory_source"])
    lab = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = lab
    spec.loader.exec_module(lab)
    module = importlib.import_module("hegel_machine.cognition.unified_session")
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("run distinct runtimes in separate processes")
    def factory(item):
        return lab.UnknownTaskLab(**{key: value for key, value in item.items() if key != "research_limits"})
    session_type = module.UnifiedResearchSession
    if arm == ARMS[0]:
        class OldBudgetSession(session_type):
            def _remaining(self):
                caps = Counter()
                for item in self.state["tasks"][:len(self.state["results"])]:
                    caps.update(task_limits(self, item))
                return {key: max(0, caps[key] - self._costs()[key]) for key in RESOURCES}
        session_type = OldBudgetSession
    return module, session_type, factory


def task_limits(session, item):
    limits = item.get("research_limits", {})
    return {"observation_cost": limits.get("observation_budget", session.config.observation_budget),
            "compute_cost": limits.get("compute_budget", session.config.compute_budget),
            "measurement_cost": limits.get("measurement_budget", session.state["measurement_budget"])}


def initial(protocol, output, cohort, arm, seed):
    module, session_type, lab_type = load(protocol, arm, cohort)
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    path, opened = checkpoint(output, cohort, arm, seed), []
    def factory(item):
        laboratory = lab_type(item)
        opened.append(laboratory)
        return laboratory
    state = json.loads(path.read_text()) if path.exists() else None
    session = session_type(state=state, lab_factory=factory) if state else session_type(
        module.ResearchConfig(**protocol["research_config"], seed=seed),
        tasks=protocol["tasks_by_seed"][str(seed)], measurement_budget=protocol["measurement_budget"],
        lab_factory=factory)
    started_cpu = time.process_time()
    while len(session.state["results"]) < len(STAGES):
        position = len(session.state["results"])
        opened.clear()
        start = time.perf_counter()
        if arm == ARMS[0]:
            item = session.state["tasks"][position]
            limits = task_limits(session, item)
            available = [concept.concept_id for concept in session.library.concepts]
            with factory(item) as laboratory:
                report = UnifiedResearchAgent(replace(session.config, seed=item["seed"],
                    observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
                    measurement_budget=limits["measurement_cost"], library=session.library).run(laboratory)
            session.state["results"].append({"task_position": position,
                "available_concepts": available, "research": report})
            new_ids = [concept.concept_id for concept in session.library.concepts if concept.concept_id not in available]
            session._offer_new_knowledge(position, new_ids)
            state = session.to_dict()
        else:
            state = session.run(max_tasks=1)
        record = state["results"][position]
        actual = _physical_costs(opened)
        _check_purchases(actual, record["research"]["costs"])
        record.update(learning_wall_seconds=time.perf_counter() - start, actual_costs=actual,
            acquisitions=[entry for laboratory in opened for entry in laboratory.acquisitions])
        state.setdefault("study_initial_snapshots", {})[str(position)] = {
            "research": deepcopy(record["research"]), "knowledge": deepcopy(state["knowledge"])}
        save(path, state)
        print(json.dumps({"phase": "initial", "arm": arm, "seed": seed, "position": position,
            "status": record["research"]["status"], "costs": record["research"]["costs"]}), flush=True)
        session = session_type(state=json.loads(path.read_text()), lab_factory=factory)
    state = session.to_dict()
    state.update(study_initial_complete=True, study_learning_complete=False)
    state.setdefault("study_cpu", []).append({"phase": "initial", "seconds": time.process_time() - started_cpu})
    save(path, state)


def revisit(protocol, output, cohort, arm, seed):
    _, session_type, lab_type = load(protocol, arm, cohort)
    path, opened = checkpoint(output, cohort, arm, seed), []
    def factory(item):
        laboratory = lab_type(item)
        opened.append(laboratory)
        return laboratory
    state = json.loads(path.read_text())
    session = session_type(state=state, lab_factory=factory)
    started_cpu = time.process_time()
    while session.state["pending_revisits"]:
        opened.clear()
        count = len(session.state["revisits"])
        start = time.perf_counter()
        state = session.run(max_tasks=1)
        if len(state["revisits"]) > count:
            visit = state["revisits"][-1]
            record = state["results"][visit["task_position"]]
            actual = _physical_costs(opened)
            _check_purchases(actual, visit["visit_costs"])
            elapsed = time.perf_counter() - start
            visit.update(actual_new_costs=actual, learning_wall_seconds=elapsed,
                         continuation=deepcopy(record["research"].get("continuation")),
                         confirmed_used_sources=record["research"].get("used_sources", []))
            record["learning_wall_seconds"] += elapsed
            for key, value in actual.items():
                record["actual_costs"][key] += value
            record["acquisitions"].extend(entry for laboratory in opened for entry in laboratory.acquisitions)
            print(json.dumps({"phase": "revisit", "arm": arm, "seed": seed,
                "position": visit["task_position"], "status": visit["status"], "costs": visit["visit_costs"]}), flush=True)
        save(path, state)
        session = session_type(state=json.loads(path.read_text()), lab_factory=factory)
    state = session.to_dict()
    state.update(study_learning_complete=True, status="completed")
    state.setdefault("study_cpu", []).append({"phase": "revisit", "seconds": time.process_time() - started_cpu})
    save(path, state)


def ablate(protocol, output, cohort, seed):
    arm = ARMS[2]
    _, session_type, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    path = checkpoint(output, cohort, arm, seed)
    state = json.loads(path.read_text())
    full = json.loads(checkpoint(output, cohort, ARMS[1], seed).read_text())
    if not full.get("study_learning_complete"):
        raise ValueError("the complete knowledge schedule is required")
    started_cpu = time.process_time()
    state["pending_revisits"] = []
    local = state.setdefault("study_revisit_knowledge", {})
    matched = state.setdefault("study_matched_revisits", [])
    for index in range(len(matched), len(full["revisits"])):
        paired = full["revisits"][index]
        position = paired["task_position"]
        record = state["results"][position]
        previous = record["research"]
        requested = {key: paired["resource_limits"][key] - paired["previous"]["costs"][key] for key in RESOURCES}
        session = session_type(state=state, lab_factory=factory)
        available = session._remaining()
        allocation = {key: min(requested[key], available[key]) for key in RESOURCES}
        receipt = {"matched_index": index, "task_position": position,
            "requested_allocation": requested, "available_pool": available, "allocation": allocation}
        if not allocation["observation_cost"] or not allocation["compute_cost"]:
            matched.append({**receipt, "executed": False, "reason": "remaining_resources"})
            save(path, state)
            continue
        knowledge = local.get(str(position), state["study_initial_snapshots"][str(position)]["knowledge"])
        library = KnowledgeLibrary.from_dict(knowledge)
        limits = {key: previous["costs"][key] + allocation[key] for key in RESOURCES}
        item = state["tasks"][position]
        start = time.perf_counter()
        with factory(item) as laboratory:
            report = UnifiedResearchAgent(replace(session.config, seed=item["seed"],
                observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
                measurement_budget=limits["measurement_cost"], library=library).run(laboratory,
                    previous=previous, continuation_candidate=None)
        actual, elapsed = dict(laboratory.costs), time.perf_counter() - start
        _check_purchases(actual, report["visit_costs"])
        local[str(position)] = library.to_dict()
        record["research"] = report
        record["learning_wall_seconds"] += elapsed
        for key, value in actual.items():
            record["actual_costs"][key] += value
        record["acquisitions"].extend(laboratory.acquisitions)
        state["revisits"].append({"task_position": position, "candidate": None,
            "previous": previous, "resource_limits": limits, "visit_costs": report["visit_costs"],
            "status": report["status"], "actual_new_costs": actual, "learning_wall_seconds": elapsed,
            "continuation": deepcopy(report.get("continuation")),
            "confirmed_used_sources": report.get("used_sources", []), "matched_index": index})
        matched.append({**receipt, "executed": True, "visit_costs": report["visit_costs"]})
        state["costs"] = session_type(state=state, lab_factory=factory).to_dict()["costs"]
        save(path, state)
        print(json.dumps({"phase": "ablate", "arm": arm, "seed": seed,
            "position": position, "status": report["status"], "costs": report["visit_costs"]}), flush=True)
    state.update(study_learning_complete=True, status="completed")
    state.setdefault("study_cpu", []).append({"phase": "ablate", "seconds": time.process_time() - started_cpu})
    save(path, state)


def evaluate(protocol, output, cohort, arm, seed):
    for name in ARMS:
        for root in protocol["cohorts"][cohort]:
            if not json.loads(checkpoint(output, cohort, name, root).read_text()).get("study_learning_complete"):
                raise ValueError("finish all learning before terminal evaluation")
    _, _, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    path = checkpoint(output, cohort, arm, seed)
    state = json.loads(path.read_text())
    settings = protocol["terminal"]
    for position in range(len(state["evaluation"]), len(STAGES)):
        report = state["results"][position]["research"]
        initial_report = state["study_initial_snapshots"][str(position)]["research"]
        channels = list(dict.fromkeys([*initial_report["selected_channels"], *report["selected_channels"]]))
        with factory(state["tasks"][position]) as laboratory:
            rows = laboratory.evaluation_rows(channels, seed=state["tasks"][position]["seed"] + settings["seed_offset"],
                length=settings["length"], trajectories=settings["trajectories"])
            rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        final_program, old_program = Program.from_dict(report["program"]), Program.from_dict(initial_report["program"])
        chunks = [rows[index:index + settings["length"]] for index in range(0, len(rows), settings["length"])]
        trajectories = [{"trajectory": index, "mse": loss(final_program, part, report["fallback"], .5),
                         "initial_mse": loss(old_program, part, initial_report["fallback"], .5)}
                        for index, part in enumerate(chunks)]
        state["evaluation"].append({"task_position": position, "rows": len(rows), "risk_quantile": .5,
            "mse": mean(item["mse"] for item in trajectories),
            "initial_mse": mean(item["initial_mse"] for item in trajectories), "trajectories": trajectories,
            "selected_channels": report["selected_channels"], "initial_channels": initial_report["selected_channels"],
            "final_deployment_measurement_cost": len(rows) * sum(rates[name] for name in report["selected_channels"]),
            "initial_deployment_measurement_cost": len(rows) * sum(rates[name] for name in initial_report["selected_channels"]),
            "costs": {"observation_cost": len(rows), "measurement_cost": len(rows) * sum(rates[name] for name in channels),
                      "compute_cost": 2 * len(rows), "queries": settings["trajectories"]}})
        save(path, state)


def summarize(protocol, output, cohort):
    summary = {"cohort": cohort, "scope": protocol["scope"], "arms": {}}
    for arm in ARMS:
        tasks, visits, events, challenge_events = [], [], [], []
        learning, terminal = Counter(), Counter()
        cpu = 0.
        for seed in protocol["cohorts"][cohort]:
            state = json.loads(checkpoint(output, cohort, arm, seed).read_text())
            if not state["study_learning_complete"] or len(state["evaluation"]) != len(STAGES):
                raise ValueError("all task outcomes must be present")
            for key, value in protocol["total_research_budget"].items():
                if state["costs"][key] > value:
                    raise ValueError("shared research budget exceeded")
            learning.update(state["costs"])
            learning["queries"] += sum(row["research"]["costs"]["queries"] for row in state["results"])
            cpu += sum(row["seconds"] for row in state["study_cpu"])
            for position, record in enumerate(state["results"]):
                report, score = record["research"], state["evaluation"][position]
                old = state["study_initial_snapshots"][str(position)]["research"]
                terminal.update(score["costs"])
                tasks.append({"root": seed, "position": position, "status": report["status"],
                    "initial_status": old["status"], "kind": report["selected_kind"], "costs": report["costs"],
                    "evaluation": score, "used_sources": report.get("used_sources", [])})
                challenge_events.extend({"root": seed, "position": position,
                    **{key: value for key, value in event.items() if key not in ("program", "baseline_program")}}
                    for event in report["actions"] if event["action"] in
                    {"compare_completed_hypotheses", "completed_task_challenge"})
            events.extend({"root": seed, **row} for row in state.get("scheduler_events", []))
            for visit in state["revisits"]:
                offered = (visit.get("continuation") or {}).get("source_concept_ids", [])
                used = visit.get("confirmed_used_sources", [])
                visits.append({"root": seed, "position": visit["task_position"], "status": visit["status"],
                    "previous_status": visit["previous"]["status"], "visit_costs": visit["visit_costs"],
                    "offered_sources": offered, "confirmed_offered_sources_used": sorted(set(offered) & set(used)),
                    "comparisons": (visit.get("continuation") or {}).get("comparisons", [])})
        tolerance = protocol["terminal"]["mse_tolerance"]
        wrong = [row for row in tasks if row["initial_status"] == "completed" and row["evaluation"]["initial_mse"] > tolerance]
        summary["arms"][arm] = {"tasks": tasks, "revisits": visits, "scheduler_events": events,
            "mean_mse": mean(row["evaluation"]["mse"] for row in tasks),
            "by_seed": {str(seed): mean(row["evaluation"]["mse"] for row in tasks if row["root"] == seed)
                        for seed in protocol["cohorts"][cohort]},
            "by_stage": {stage: mean(row["evaluation"]["mse"] for row in tasks if row["position"] == position)
                         for position, stage in enumerate(STAGES)},
            "learning_costs": dict(learning), "terminal_costs": dict(terminal), "cpu_seconds": cpu,
            "screening_compute_cost": sum(event.get("compute_cost", 0) for event in events
                                           if event["action"] == "try_successor_knowledge"),
            "screening_attempts": sum(event["action"] == "try_successor_knowledge" for event in events),
            "scheduled_offers": sum(event.get("scheduled", False) for event in events),
            "challenge_events": challenge_events,
            "challenge_outcomes": dict(Counter(event["outcome"] for event in challenge_events
                                               if event["action"] == "completed_task_challenge")),
            "paid_challenge_rows": sum(event["observation_rows"] for event in challenge_events
                                       if event["action"] == "compare_completed_hypotheses"),
            "final_deployment_measurement_cost": sum(row["evaluation"]["final_deployment_measurement_cost"]
                                                      for row in tasks),
            "status": dict(Counter(row["status"] for row in tasks)), "revisit_count": len(visits),
            "revisit_status": dict(Counter(row["status"] for row in visits)),
            "initial_completed_but_long_inaccurate": len(wrong),
            "corrected_initial_completions": sum(row["evaluation"]["mse"] <= tolerance for row in wrong),
            "visits_with_confirmed_offered_source_use": sum(bool(row["confirmed_offered_sources_used"]) for row in visits),
            "revisit_costs": {key: sum(row["visit_costs"][key] for row in visits) for key in common.COSTS}}
    learned = summary["arms"][ARMS[1]]
    summary["paired_changes"] = {}
    for arm in (ARMS[0], ARMS[2]):
        changes = {seed: value - summary["arms"][arm]["by_seed"][seed] for seed, value in learned["by_seed"].items()}
        summary["paired_changes"][arm] = {"new_minus_control_by_root": changes,
            "tolerance": tolerance, "lower": sum(value < -tolerance for value in changes.values()),
            "equal": sum(abs(value) <= tolerance for value in changes.values()),
            "higher": sum(value > tolerance for value in changes.values()), "mean_change": mean(changes.values()),
            "exact_lower": sum(value < 0 for value in changes.values()),
            "exact_equal": sum(value == 0 for value in changes.values()),
            "exact_higher": sum(value > 0 for value in changes.values())}
    save(output / cohort / "summary.json", summary)
    digest = {**summary, "arms": {arm: {key: value for key, value in data.items() if key not in
              ("tasks", "revisits", "scheduler_events")} for arm, data in summary["arms"].items()}}
    save(output / cohort / "final_digest.json", digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, cohort):
    roots = protocol["cohorts"][cohort]
    execution = {"status": "running", "active": [], "completed": [], "failed": []}
    lock = threading.Lock()
    def run(mode, arm, seed):
        job = [mode, arm, seed]
        log = output / cohort / "logs" / ("_".join(map(str, job)) + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            execution["active"].append(job)
            save(output / cohort / "execution.json", execution)
        with log.open("a") as stream:
            code = subprocess.call([sys.executable, "-B", str(Path(__file__).resolve()), mode,
                "--output", str(output), "--cohort", cohort, "--arm", arm, "--seed", str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            execution["active"].remove(job)
            execution["failed" if code else "completed"].append({"job": job, "exit_code": code})
            save(output / cohort / "execution.json", execution)
        print(json.dumps({"finished": job, "exit_code": code}), flush=True)
        return code
    phases = (("initial", [(arm, seed) for seed in roots for arm in ARMS]),
              ("revisit", [(arm, seed) for seed in roots for arm in ARMS[:2]]),
              ("ablate", [(ARMS[2], seed) for seed in roots]),
              ("evaluate", [(arm, seed) for seed in roots for arm in ARMS]))
    for mode, jobs in phases:
        execution["phase"] = mode
        with ThreadPoolExecutor(max_workers=protocol["maximum_concurrent_processes"]) as pool:
            futures = [pool.submit(run, mode, arm, seed) for arm, seed in jobs]
            for future in as_completed(futures):
                if not future.cancelled() and future.result():
                    for pending in futures:
                        pending.cancel()
        if execution["failed"]:
            save(output / cohort / "execution.json", {**execution, "status": "failed", "exit_code": 1})
            raise SystemExit(1)
    summarize(protocol, output, cohort)
    save(output / cohort / "execution.json", {**execution, "status": "completed", "exit_code": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "initial", "revisit", "ablate", "evaluate", "queue", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cohort", choices=("development", "main"), default="development")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "prepare":
        prepare(output)
        return
    protocol = json.loads((output / "protocol.json").read_text())
    if args.mode == "freeze":
        common.freeze(output)
    elif args.mode == "queue":
        queue(protocol, output, args.cohort)
    elif args.mode == "summarize":
        summarize(protocol, output, args.cohort)
    elif args.mode == "ablate":
        ablate(protocol, output, args.cohort, args.seed)
    else:
        globals()[args.mode](protocol, output, args.cohort, args.arm, args.seed)


if __name__ == "__main__":
    main()
