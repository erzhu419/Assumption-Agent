#!/usr/bin/env python3
"""Offline six-stage risk streams with explicit knowledge-source accounting."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import run_mixed_research_investment_study as shared


ROOT = shared.ROOT
OUTPUT = ROOT / "artifacts/risk_knowledge_adoption_20260920"
BASELINE = "artifacts/mixed_research_investment_20260920/runtime/src"
ARMS = ("baseline_v0107", "knowledge_adoption", "task_reset")
STAGES = ("cold_A", "reuse_A", "changed_B", "changed_A_parameters", "return_B", "return_A")

# The existing runner owns actual acquisitions, cumulative resource accounting,
# JSON continuation and terminal scoring. These are this study's finite axes.
shared.ARMS, shared.FAMILIES, shared.STAGES = ARMS, ("risk",), STAGES


def prepare(output):
    path = output / "protocol.json"
    if path.exists():
        raise ValueError("preserve the declared protocol; choose a new output for a different study")
    if not (ROOT / BASELINE / "hegel_machine/cognition/unified_session.py").exists():
        raise FileNotFoundError(BASELINE)
    cohorts = {"development": [96016003, 96016006], "main": list(range(96024001, 96024009))}
    tasks, scenarios = {}, {}
    for cohort, seeds in cohorts.items():
        for index, seed in enumerate(seeds):
            source, changed = ((12, 13) if cohort == "development" or index % 2 == 0 else (13, 12))
            conditions = ((source, "development"), (source, "development"),
                          (changed, "confirmation"), (source, "confirmation"),
                          (changed, "confirmation"), (source, "development"))
            tasks[str(seed)] = [{"environment": "scalar", "task_index": task,
                "profile": profile, "seed": seed + 200003 + 10007 * position}
                for position, (task, profile) in enumerate(conditions)]
            scenarios[str(seed)] = {"A_task_index": source, "B_task_index": changed,
                "conditions": ["A_development", "A_development", "B_confirmation",
                               "A_confirmation", "B_confirmation", "A_development"]}
    settings = {"research_config": {"observation_budget": 256, "compute_budget": 160000,
        "minimum_improvement": 1e-7, "max_rounds": 4, "max_history": 4, "active_probes": False},
        "measurement_budget": 1024, "loss_kind": "pinball", "risk_quantile": .9,
        "stream_budget": {"observation_cost": 1536, "measurement_cost": 6144, "compute_cost": 960000}}
    shared.save(path, {"experiment": "risk_knowledge_adoption_streams",
        "fixed_at": datetime.now(timezone.utc).isoformat(), "cohorts": cohorts,
        "tasks_by_family_seed": {"risk": tasks}, "scenarios": scenarios, "stages": STAGES,
        "development_runtime": "src", "families": {"risk": settings},
        "arms": {ARMS[0]: {"runtime": BASELINE, "version": "0.107.0"},
            ARMS[1]: {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.108.0"},
            ARMS[2]: {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.108.0"}},
        "shared_laboratory_source": BASELINE + "/hegel_machine/cognition",
        "maximum_concurrent_processes": 8,
        "terminal": {"risk": {"length": 64, "trajectories": 8}, "seed_offset": 1000003},
        "budget_basis": {"previous_study": "mixed_research_investment_20260920/main/summary.json",
            "previous_empty_library_mean_task_compute": 4426124 / 32,
            "previous_baseline_cpu_seconds_per_task": 4362.776089107 / 32,
            "previous_empty_library_cpu_seconds_per_task": 485.95708776699996 / 32,
            "reason": "Keep the previous 160000 compute ceiling: empty-library research already used about "
                "138316 per task, so a smaller ceiling would conflate adoption with unfunded confirmation. "
                "Six stages add a second source return and parameter-shift condition while avoiding eight-stage "
                "repetition. Eight roots give four paired streams per A/B order; they are not a power calculation. "
                "Development is inspected before main freeze. The previous baseline rate projects about 109 CPU "
                "minutes for the 48-task baseline arm; accumulated knowledge may make later tasks more expensive."},
        "budget_contract": "Each six-task stream has the same original observation, measurement and compute pool "
            "in every arm. All initial tasks, failed research, source screening and revisits count. Ordinary session "
            "scheduling may reuse unused visited-task resources, never an extra revisit allowance.",
        "reset_contract": "An empty KnowledgeLibrary starts every new task; ordinary within-task research, "
            "reports and paid observations remain. Cross-task offers are disabled. This removes both knowledge "
            "and the revisits it triggers; it does not identify their separate effects.",
        "learning_contract": "Start with no seeded knowledge. Restore complete session JSON after every visit. "
            "All learning in all arms and roots of a cohort must end before terminal scoring. Keep every failure "
            "and unresolved output. The two development roots were exposed in the previous study; only the eight "
            "new roots use the frozen main runtime.",
        "source_contract": "Preserve complete screen_risk_knowledge events with per-source comparison loss, gain, "
            "margin, status and trial cost. Paid screening is not independent confirmation. Record source birth "
            "task, parents, transfer counterexamples and final used_sources/method_source_ids separately. A "
            "multi-source confirmation supports their joint program, not an additive causal gain for each source. "
            "A non-selected source is not automatically a rejected source. Older runtime missing screening records "
            "is reported as unavailable. Condition-specific rejection is not global deletion.",
        "evaluation_contract": "After learning ends, acquire eight independent 64-row public random trajectories "
            "per task. Initial and returned programs face the same measured response. Keep the original q=.9 "
            "pinball loss, coverage and fixed (9,1) action loss; action loss is exactly ten times pinball. Terminal "
            "data never enter learning. Report losses, actual costs, deployment measurements, elapsed/CPU time "
            "and unresolved counts separately; no composite utility or completion-based exclusion.",
        "scope": "Risk only: six stages, eight new roots, four A/B orders each, with separate mechanism changes, "
            "parameter changes and two source-condition returns. The two previously exposed roots are development. "
            "This tests finite existing mechanisms and does not establish indefinitely stable long-run benefit.",
        "formal_evidence": False})


def _reports(state):
    """Include old visits for source births; final reports already retain actions."""
    for revisit in state["revisits"]:
        yield revisit["task_position"], revisit["previous"]
    for record in state["results"]:
        yield record["task_position"], record["research"]


def _evidence(value):
    """Keep evidence fields here; the complete program is in the checkpoint."""
    if isinstance(value, dict):
        if "op" in value and "children" in value:
            return {"program_in_checkpoint": True}
        return {key: _evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_evidence(item) for item in value]
    return value


def trace_sources(state, arm):
    origins = {}
    for position, report in _reports(state):
        for source in report.get("concepts_added", []):
            origins.setdefault(source, set()).add(position)
    concepts = {item["concept_id"]: item for item in state["knowledge"]["concepts"]}
    tasks, statuses, confirmed_sets = [], Counter(), Counter()
    for record in state["results"]:
        position, report = record["task_position"], record["research"]
        screens = [{"event_index": index, "event": _evidence(event)} for index, event in enumerate(report["actions"])
                   if event["action"] == "screen_risk_knowledge"]
        attempts = [{"event_index": index, "action": event["action"],
                     "compute_cost": event.get("compute_cost"),
                     "alternatives": event.get("alternatives", [])}
                    for index, event in enumerate(report["actions"])
                    if event["action"] in {"execute_research_programs", "compose_relation_kernels"}]
        transfers = [{key: value for key, value in event.items() if key != "program"}
                     for event in report.get("transfer", [])]
        for event in transfers:
            statuses[event.get("status", "unspecified")] += 1
        used = sorted(set(report.get("used_sources", [])))
        methods = sorted(set(report.get("method_source_ids", [])))
        confirmed = used if report["status"] == "completed" else []
        if confirmed:
            confirmed_sets[tuple(confirmed)] += 1
        tasks.append({"position": position, "stage": STAGES[position], "status": report["status"],
            "available_concepts": record["available_concepts"], "selected_sources": used,
            "selected_method_sources": methods, "confirmed_source_set": confirmed,
            "screening_available": bool(screens), "screening": screens,
            "program_research_attempts": attempts, "transfers": transfers,
            "confirmation": [{"event_index": index, "event": _evidence(event)}
                for index, event in enumerate(report["actions"])
                if event["action"] in {"research_action_feedback", "risk_research_feedback",
                    "validate_frozen_program", "validate_continued_program", "compare_continuation_candidates"}]})
    sources = []
    for source, positions in sorted(origins.items()):
        groups = [{position} for position in sorted(positions)] if arm == ARMS[2] else [positions]
        for group in groups:
            metadata = (concepts.get(source, {}) if arm != ARMS[2] or group == {len(state["results"]) - 1}
                        else {})
            sources.append({"source_id": source,
                "source_key": f"{next(iter(group))}:{source}" if arm == ARMS[2] else source,
                "birth_positions": sorted(group), "metadata_available": bool(metadata),
                **{key: metadata.get(key) for key in ("parents", "status", "excluded_tasks",
                                                     "counterexamples", "reuse_history")}})
    return {"arm": arm, "learning_complete": state.get("study_learning_complete", False),
        "source_identity": ("task-local IDs; origins can repeat after each library reset"
                            if arm == ARMS[2] else "session-wide concept IDs"),
        "sources": sources,
        "transfer_status_counts": dict(statuses),
        "confirmed_source_sets": [{"source_ids": list(key), "tasks": count}
                                  for key, count in sorted(confirmed_sets.items())],
        "tasks": tasks, "scheduler_events": state["scheduler_events"],
        "screening_is_fresh_confirmation": False}


def source_report(output, cohort, arm, seed):
    path = shared.checkpoint(output, cohort, "risk", arm, seed)
    state = json.loads(path.read_text())
    trace = trace_sources(state, arm)
    trace["screening_summary"] = summarize_source_trace(trace)
    shared.save(path.parent / "source_trace.json", trace)
    return trace


def summarize_source_trace(trace):
    """Count comparisons once; a shared source set does not split its gain."""
    names = ("supported_for_research", "weak_evidence", "no_marginal_gain", "comparison_unavailable")
    total, grouped, stages = Counter({name: 0 for name in names}), {}, []
    for task in trace["tasks"]:
        statuses, screen_cost = Counter({name: 0 for name in names}), 0
        for item in task["screening"]:
            event = item["event"]
            screen_cost += event["compute_cost"]
            for trial in event.get("comparisons", []):
                status = trial["status"]
                statuses[status] += 1
                sources = tuple(sorted(trial.get("source_ids", [])))
                group = grouped.setdefault(sources, {"source_ids": list(sources),
                    "comparison_status_counts": Counter(), "trial_compute_cost": 0})
                group["comparison_status_counts"][status] += 1
                group["trial_compute_cost"] += trial.get("compute_cost", 0)
        total.update(statuses)
        stages.append({"position": task["position"], "stage": task["stage"],
            "screen_events": len(task["screening"]), "comparison_status_counts": dict(statuses),
            "screen_compute_cost": screen_cost,
            "expensive_program_research_compute_cost": sum(event["compute_cost"] or 0
                for event in task["program_research_attempts"] if event["action"] == "execute_research_programs"),
            "selected_sources": task["selected_sources"], "confirmed_source_set": task["confirmed_source_set"]})
    return {"comparison_status_counts": dict(total),
        "screen_events": sum(item["screen_events"] for item in stages),
        "screen_compute_cost": sum(item["screen_compute_cost"] for item in stages),
        "expensive_program_research_compute_cost": sum(item["expensive_program_research_compute_cost"]
                                                       for item in stages),
        "source_sets": [{**group, "comparison_status_counts": dict(group["comparison_status_counts"])}
                        for _, group in sorted(grouped.items())], "by_stage": stages,
        "cost_definition": "Screen costs use enclosing action charges once, including baseline and partial work. "
            "Per-source-set trial costs exclude shared baseline cost. Expensive program research includes replay "
            "and subsequent combination search; the existing event does not isolate replay cost.",
        "evidence_definition": "Statuses count paid comparisons, not independent confirmations or unique sources. "
            "Confirmed sets describe the final jointly tested program, not a per-source causal gain."}


def learn(protocol, output, cohort, arm, seed, max_visits=None):
    shared.learn(protocol, output, cohort, "risk", arm, seed, max_visits)
    source_report(output, cohort, arm, seed)


def summarize(protocol, output, cohort):
    shared.summarize(protocol, output, cohort)
    sources = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in protocol["cohorts"][cohort]:
            trace = source_report(output, cohort, arm, seed)
            sources[arm][str(seed)] = {"known_sources": len(trace["sources"]),
                "transfer_status_counts": trace["transfer_status_counts"],
                "confirmed_source_sets": trace["confirmed_source_sets"],
                "screening_summary": trace["screening_summary"],
                "screening_tasks": sum(task["screening_available"] for task in trace["tasks"]),
                "by_stage": [{"stage": task["stage"], "status": task["status"],
                    "selected_sources": task["selected_sources"], "method_source_ids": task["selected_method_sources"],
                    "screening_events": len(task["screening"]),
                    "program_research_compute": sum(event["compute_cost"] or 0
                                                    for event in task["program_research_attempts"])}
                    for task in trace["tasks"]]}
    shared.save(output / cohort / "source_summary.json", sources)


def queue(protocol, output, cohort, *, learning_only=False):
    jobs = [(arm, seed) for arm in ARMS for seed in protocol["cohorts"][cohort]]
    history = {"cohort": cohort, "status": "running", "jobs": []}
    for phase in (("learn",) if learning_only else ("learn", "evaluate")):
        def execute(job):
            arm, seed = job
            path = output / cohort / "logs" / f"{phase}_{arm}_{seed}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                code = subprocess.call([sys.executable, "-B", str(Path(__file__).resolve()), phase,
                    "--output", str(output), "--cohort", cohort, "--arm", arm, "--seed", str(seed)],
                    stdout=stream, stderr=subprocess.STDOUT)
            return {"phase": phase, "arm": arm, "seed": seed, "exit_code": code}
        with ThreadPoolExecutor(max_workers=protocol["maximum_concurrent_processes"]) as pool:
            for future in as_completed([pool.submit(execute, job) for job in jobs]):
                record = future.result()
                history["jobs"].append(record)
                shared.save(output / cohort / "execution.json", history)
                print(json.dumps(record), flush=True)
        if any(item["exit_code"] for item in history["jobs"]):
            history["status"] = "failed"
            shared.save(output / cohort / "execution.json", history)
            raise SystemExit(1)
    if not learning_only:
        summarize(protocol, output, cohort)
    history["status"] = "learning_completed" if learning_only else "completed"
    shared.save(output / cohort / "execution.json", history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "learn", "evaluate", "queue",
                                         "learn-cohort", "summarize", "sources"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cohort", choices=("development", "main"), default="development")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-visits", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
        return
    if args.mode == "freeze":
        shared.freeze(args.output)
        return
    protocol = json.loads((args.output / "protocol.json").read_text())
    if args.mode in {"queue", "learn-cohort"}:
        queue(protocol, args.output, args.cohort, learning_only=args.mode == "learn-cohort")
    elif args.mode == "summarize":
        summarize(protocol, args.output, args.cohort)
    else:
        if args.arm is None or args.seed not in protocol["cohorts"][args.cohort]:
            parser.error("choose an arm and predeclared cohort seed")
        if args.mode == "learn":
            learn(protocol, args.output, args.cohort, args.arm, args.seed, args.max_visits)
        elif args.mode == "evaluate":
            shared.evaluate(protocol, args.output, args.cohort, "risk", args.arm, args.seed)
        else:
            source_report(args.output, args.cohort, args.arm, args.seed)


if __name__ == "__main__":
    main()
