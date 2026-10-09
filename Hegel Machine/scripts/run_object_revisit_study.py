#!/usr/bin/env python3
"""Object-task successor knowledge and paid unfinished research frontiers."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from random import Random
from statistics import mean
import subprocess
import threading
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import run_observation_action_study as common
import run_completed_challenge_study as flow

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/object_revisit_study_20260920"
BASELINE = "artifacts/completed_challenge_study_20260920/runtime/src"
ARMS = ("baseline_v099", "knowledge_revisit", "no_successor_knowledge")
STAGES = ("unfinished_object_task", "renamed_object_successor")
DEVELOPMENT_LIMITS = ((64, 12000), (96, 30000), (128, 60000))


def cold(protocol, output, index):
    source = ROOT / BASELINE
    sys.path.insert(0, str(source))
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
    from hegel_machine.cognition.memory import KnowledgeLibrary
    path = output / "development" / f"cold_{index}.json"
    if path.exists():
        raise ValueError("preserve the previous budget attempt")
    obs, compute = DEVELOPMENT_LIMITS[index]
    task = {"objects": 2, "readout": "event", "apparatus_seed": 9013,
            "actuator_curvature": .12, "seed": 221}
    config = ResearchConfig(observation_budget=obs, compute_budget=compute,
        minimum_improvement=1e-7, max_rounds=6, max_history=4, active_probes=True, seed=221)
    library = KnowledgeLibrary()
    cpu, wall = time.process_time(), time.perf_counter()
    with UnifiedObjectLab(**task) as laboratory:
        result = UnifiedResearchAgent(config, measurement_budget=768, library=library).run(laboratory)
    payload = {"runtime": BASELINE, "task": task, "config": asdict(config),
        "measurement_budget": 768, "research": result, "knowledge": library.to_dict(),
        "actual_costs": laboratory.costs, "acquisitions": laboratory.acquisitions,
        "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}
    common.save(path, payload)
    print(json.dumps({key: result[key] for key in ("status", "selected_kind", "selected_channels", "costs",
        "confirmation_loss", "concepts_added")}))


def successor(output, index):
    sys.path.insert(0, str(ROOT / BASELINE))
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
    from hegel_machine.cognition.memory import KnowledgeLibrary
    source = json.loads((output / "development" / f"cold_{index}.json").read_text())
    path = output / "development" / f"successor_{index}.json"
    if path.exists():
        raise ValueError("preserve the previous successor attempt")
    task = {**source["task"], "seed": 222, "rename_prefix": "successor_"}
    config = ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, max_history=4, active_probes=True, seed=222)
    library = KnowledgeLibrary.from_dict(source["knowledge"])
    cpu, wall = time.process_time(), time.perf_counter()
    with UnifiedObjectLab(**task) as laboratory:
        result = UnifiedResearchAgent(config, measurement_budget=4608, library=library).run(laboratory)
    payload = {"runtime": BASELINE, "task": task, "config": asdict(config),
        "measurement_budget": 4608, "research": result, "knowledge": library.to_dict(),
        "actual_costs": laboratory.costs, "acquisitions": laboratory.acquisitions,
        "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}
    common.save(path, payload)
    print(json.dumps({key: result[key] for key in ("status", "selected_kind", "selected_channels", "costs",
        "confirmation_loss", "concepts_added")}))



def prepare(output):
    if (output / "protocol.json").exists():
        return
    roots = [9013, *range(95802001, 95802009)]
    tasks, settings = {}, {}
    for seed in roots:
        if seed == 9013:
            objects, readout, curvature, radius, profile, first, second = 2, "event", .12, None, "development", 221, 222
        else:
            objects, readout = ((2, "focus"), (2, "event"), (3, "focus"), (3, "event"))[(seed - 95802001) % 4]
            rng = Random(seed + 17011)
            curvature, radius, profile, first, second = rng.uniform(.08, .24), rng.uniform(.70, 1.20), "confirmation", seed, seed + 10007
        common_task = {"environment": "objects", "objects": objects, "readout": readout,
            "apparatus_seed": seed, "actuator_curvature": curvature, "relation_radius": radius,
            "profile": profile, "variant": "exchange", "relation_direction": "symmetric", "relation_condition": "always"}
        tasks[str(seed)] = [{**common_task, "seed": first,
            "research_limits": {"observation_budget": 64, "compute_budget": 12000, "measurement_budget": 768}},
            {**common_task, "seed": second, "rename_prefix": "successor_"}]
        settings[str(seed)] = {"objects": objects, "readout": readout, "curvature": curvature,
                               "radius": radius, "profile": profile}
    common.save(output / "protocol.json", {
        "experiment": "successor_relations_resume_unfinished_object_research",
        "fixed_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {"development": roots[:1], "main": roots[1:]},
        "tasks_by_seed": tasks, "apparatus_parameters": settings,
        "arms": {"baseline_v099": {"runtime": BASELINE, "version": "0.99.0"},
            "knowledge_revisit": {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.100.0"},
            "no_successor_knowledge": {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.100.0"}},
        "development_runtime": "src", "shared_laboratory_source": BASELINE + "/hegel_machine/cognition",
        "research_config": {"observation_budget": 384, "compute_budget": 360000,
            "minimum_improvement": 1e-7, "max_rounds": 6, "max_history": 4, "active_probes": True},
        "measurement_budget": 4608,
        "total_research_budget": {"observation_cost": 448, "measurement_cost": 5376, "compute_cost": 372000},
        "maximum_concurrent_processes": 4,
        "terminal": {"length": 48, "trajectories": 4, "seed_offset": 1000003,
            "risk_quantile": .5, "mse_tolerance": 1e-6,
            "focus_offsets": [-1e-6, 0., 1e-6], "public_pair_distances": [.35, .65, .95, 1.25],
            "phases": ["off", "on", "on", "on", "on", "off", "off", "off", "off"]},
        "development_selection": "Only the known two-object event apparatus was used for budget selection. "
            "The first declared allowance, 64 observations/12000 compute, produced actual fine measurements, "
            "structural experiments and a refuted frozen confirmation. It remained unresolved. The two "
            "larger candidate allowances were not run. The renamed successor independently learned a "
            "relation program under its ordinary 384/360000 allowance. Preserve cold_0 and successor_0.",
        "initial_contract": "Start every stream with empty knowledge. A uses 64 rows/12000 compute/768 meter; "
            "B uses 384/360000/4608 and renamed measurement fields on the same physical apparatus. "
            "All programs, parameters, source definitions and relation methods are learned from paid "
            "observations through ordinary public interfaces; no private formula supplies an agent candidate.",
        "schedule_contract": "Complete all first visits before actual revisits. Keep the original paid "
            "observations, object transitions, structural and relation frontier when resuming A. Same stream "
            "resource cap is the sum of first-visit allowances, not an extra revisit allowance. All source "
            "learning, screening, candidate fitting, failed research and real measurements count.",
        "knowledge_ablation_contract": "After the knowledge arm finishes, replay only its task positions and "
            "incremental resource allocations with each old task's own first-visit library and candidate=None. "
            "Do not supply successor definitions, candidate programs, proposed probes or prediction scores. "
            "Clip allocated resources to that control stream's actual remaining pool. A matched schedule "
            "deletion of knowledge is not a separately autonomous scheduler baseline.",
        "persistence": "JSON restore complete state between visits. Initial task histories remain immutable "
            "snapshots; results contain each unique task's latest cumulative report, preventing double counting.",
        "evaluation_contract": "After every arm and root has ended learning, score both initial and latest "
            "models on the same 4x48 fresh random trajectories and public focus-cut/pair-distance probes. "
            "No test uses the true relation radius. Buy one legal fine/coarse/command view sufficient for "
            "both models, project paid observations and controls without repurchasing, and separately report "
            "each deployed model's channel cost. A task passes only when both random and boundary MSE <=1e-6. "
            "Unresolved and already-completed roots all remain in scoring; no outcome changes learning.",
        "scope": "Eight prospectively fixed independently parameterized apparatus roots, with two replicates "
            "of each 2/3-object by focus/event combination. Existing public object roles, measurement views "
            "and relation operators are supplied by the project. Not a new open-world benchmark.",
        "formal_evidence": False})


def load(protocol, arm, cohort):
    source = ROOT / (protocol["development_runtime"] if cohort == "development" and arm != ARMS[0]
                     else protocol["arms"][arm]["runtime"])
    sys.path.insert(0, str(source))
    for name in ("structure_lab", "unified_object_lab"):
        canonical = "hegel_machine.cognition." + name
        spec = importlib.util.spec_from_file_location(canonical,
            ROOT / protocol["shared_laboratory_source"] / (name + ".py"))
        laboratory = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = laboratory
        spec.loader.exec_module(laboratory)
    module = importlib.import_module("hegel_machine.cognition.unified_session")
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("separate runtimes require separate processes")
    def factory(item):
        return sys.modules["hegel_machine.cognition.unified_object_lab"].UnifiedObjectLab(
            **{key: value for key, value in item.items() if key not in ("environment", "research_limits")})
    return module, module.UnifiedResearchSession, factory


# The completed-task study already implements first visits, cost-checked resumptions,
# JSON persistence and deletion of successor knowledge; the laboratory is the only adapter.
flow.ARMS, flow.STAGES, flow.load = ARMS, STAGES, load


def evaluate(protocol, output, cohort, arm, seed):
    for name in ARMS:
        for root in protocol["cohorts"][cohort]:
            if not json.loads(flow.checkpoint(output, cohort, name, root).read_text())["study_learning_complete"]:
                raise ValueError("finish every cohort learning phase before terminal scoring")
    _, _, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.models import Observation, Probe
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    path = flow.checkpoint(output, cohort, arm, seed)
    state = json.loads(path.read_text())
    settings = protocol["terminal"]
    for position in range(len(state["evaluation"]), len(STAGES)):
        task = state["tasks"][position]
        report = state["results"][position]["research"]
        old = state["study_initial_snapshots"][str(position)]["research"]
        both_channels = set(report["selected_channels"]) | set(old["selected_channels"])
        source = ("fine",) if "fine" in both_channels else ("coarse",) if "coarse" in both_channels else ()
        with factory(task) as laboratory:
            def score(probe):
                batch = laboratory.acquire(probe, source)
                def viewed(channels):
                    if channels:
                        return laboratory.project_rows(batch.observations, source, channels)
                    history, rows = [], []
                    for point, row in zip(probe.sequence, batch.observations):
                        rows.append(Observation(point, row.target, tuple(history)))
                        history.append({**point, "__target__": row.target})
                    return rows
                return {"probe_id": probe.probe_id, "rows": len(batch.observations),
                    "mse": loss(Program.from_dict(report["program"]), viewed(report["selected_channels"]), report["fallback"], .5),
                    "initial_mse": loss(Program.from_dict(old["program"]), viewed(old["selected_channels"]), old["fallback"], .5)}
            random = [score(laboratory.random_probe(task["seed"] + settings["seed_offset"] + index * 10007,
                length=settings["length"], name=f"terminal_random_{index}")) for index in range(settings["trajectories"])]
            boundary = []
            count = task["objects"]
            for cut in range(count - 1):
                for offset in settings["focus_offsets"]:
                    for distance in settings["public_pair_distances"]:
                        points = []
                        for phase in settings["phases"]:
                            positions = [-1.95 + 3.9 * i / (count - 1) for i in range(count)]
                            if phase == "on":
                                positions[1] = positions[0] + distance
                            point = {f"{role}_o{i}": value for role, values in (("q", [.65, -.35, .45]),
                                ("x", positions), ("u", [.02, -.03, .04])) for i, value in enumerate(values[:count])}
                            point["focus"] = cut + .5 + offset
                            points.append(point)
                        boundary.append(score(Probe(f"public_boundary:{cut}:{offset}:{distance}", tuple(points))))
            costs = {**laboratory.costs, "compute_cost": 2 * laboratory.costs["observation_cost"]}
            rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        random_mse, boundary_mse = mean(row["mse"] for row in random), mean(row["mse"] for row in boundary)
        initial_random, initial_boundary = mean(row["initial_mse"] for row in random), mean(row["initial_mse"] for row in boundary)
        state["evaluation"].append({"task_position": position, "mse": max(random_mse, boundary_mse),
            "initial_mse": max(initial_random, initial_boundary), "random_mse": random_mse,
            "boundary_mse": boundary_mse, "initial_random_mse": initial_random,
            "initial_boundary_mse": initial_boundary, "random": random, "boundary": boundary,
            "costs": costs, "selected_channels": report["selected_channels"],
            "initial_channels": old["selected_channels"], "actual_channels": list(source),
            "final_deployment_measurement_cost": costs["observation_cost"] * sum(rates[name] for name in report["selected_channels"]),
            "initial_deployment_measurement_cost": costs["observation_cost"] * sum(rates[name] for name in old["selected_channels"])})
        common.save(path, state)


def summarize(protocol, output, cohort):
    flow.summarize(protocol, output, cohort)
    path = output / cohort / "summary.json"
    result = json.loads(path.read_text())
    for arm, data in result["arms"].items():
        data["paid_screening_attempts"] = sum(event.get("compute_cost", 0) > 0
            for event in data["scheduler_events"] if event["action"] == "try_successor_knowledge")
        restored = []
        for seed in protocol["cohorts"][cohort]:
            state = json.loads(flow.checkpoint(output, cohort, arm, seed).read_text())
            for index, visit in enumerate(state["revisits"]):
                position = visit["task_position"]
                report = state["results"][position]["research"]
                stop = next((len(later["previous"]["actions"]) for later in state["revisits"][index + 1:]
                             if later["task_position"] == position), len(report["actions"]))
                tail = report["actions"][len(visit["previous"]["actions"]):stop]
                purchases = [event for event in tail if event["action"] == "acquire"]
                # Acquisitions count new experiments; inherited/conditioned evidence does not.
                distinguishing = [event for event in purchases if event["purpose"].startswith(
                    ("distinguish_", "test_translation_", "test_permutation_", "test_locality_"))
                    or event["purpose"] == "resolve_structure_boundary"]
                restored.append({"root": seed, "position": position,
                    "resumed_candidates": sum(len(event["alternatives"]) for event in tail
                                              if event["action"] == "resume_object_research"),
                    "inherited_relation_checks": sum(event["action"] == "inherit_relation_intervals" for event in tail),
                    "conditioned_relation_checks": sum(event["action"] == "condition_inherited_relation_intervals" for event in tail),
                    "new_distinguishing_acquisitions": len(distinguishing),
                    "new_distinguishing_observations": sum(event["observation_cost"] for event in distinguishing),
                    "new_distinguishing_measurements": sum(event["measurement_cost"] for event in distinguishing),
                    "new_confirmation_acquisitions": sum(event["purpose"] == "confirm_frozen_program" for event in purchases),
                    "fresh_paired_comparisons": [{key: value for key, value in event.items()
                        if key not in ("program", "baseline_program")} for event in tail
                        if event["action"] == "compare_object_continuation_candidates"],
                    "new_purchases": purchases})
        data["restored_research"] = restored
        data["restored_research_counts"] = {key: sum(row[key] for row in restored) for key in (
            "resumed_candidates", "inherited_relation_checks", "conditioned_relation_checks",
            "new_distinguishing_acquisitions", "new_distinguishing_observations", "new_distinguishing_measurements",
            "new_confirmation_acquisitions")}
        data["solved"] = sum(row["evaluation"]["mse"] <= protocol["terminal"]["mse_tolerance"] for row in data["tasks"])
        data["initial_unresolved"] = sum(row["initial_status"] == "unresolved" for row in data["tasks"])
        data["unresolved_then_solved"] = sum(row["initial_status"] == "unresolved" and
            row["evaluation"]["mse"] <= protocol["terminal"]["mse_tolerance"] for row in data["tasks"])
        data["random_mse"] = mean(row["evaluation"]["random_mse"] for row in data["tasks"])
        data["boundary_mse"] = mean(row["evaluation"]["boundary_mse"] for row in data["tasks"])
    common.save(path, result)
    digest = {**result, "arms": {arm: {key: value for key, value in data.items()
        if key not in ("tasks", "revisits", "scheduler_events", "restored_research")} for arm, data in result["arms"].items()}}
    common.save(output / cohort / "final_digest.json", digest)


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
            common.save(output / cohort / "execution.json", execution)
        with log.open("a") as stream:
            code = subprocess.call([sys.executable, "-B", str(Path(__file__).resolve()), mode,
                "--output", str(output), "--cohort", cohort, "--arm", arm, "--seed", str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            execution["active"].remove(job)
            execution["failed" if code else "completed"].append({"job": job, "exit_code": code})
            common.save(output / cohort / "execution.json", execution)
        print(json.dumps({"finished": job, "exit_code": code}), flush=True)
        return code
    for mode, jobs in (("initial", [(a, s) for s in roots for a in ARMS]),
            ("revisit", [(a, s) for s in roots for a in ARMS[:2]]),
            ("ablate", [(ARMS[2], s) for s in roots]), ("evaluate", [(a, s) for s in roots for a in ARMS])):
        execution["phase"] = mode
        with ThreadPoolExecutor(max_workers=protocol["maximum_concurrent_processes"]) as pool:
            futures = [pool.submit(run, mode, arm, seed) for arm, seed in jobs]
            for future in as_completed(futures):
                if not future.cancelled() and future.result():
                    for pending in futures:
                        pending.cancel()
        if execution["failed"]:
            common.save(output / cohort / "execution.json", {**execution, "status": "failed", "exit_code": 1})
            raise SystemExit(1)
    summarize(protocol, output, cohort)
    common.save(output / cohort / "execution.json", {**execution, "status": "completed", "exit_code": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cold", "successor", "prepare", "freeze", "initial", "revisit", "ablate", "evaluate", "queue", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--index", type=int, choices=range(3))
    parser.add_argument("--cohort", choices=("development", "main"), default="development")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "cold":
        cold(None, output, args.index)
    elif args.mode == "successor":
        successor(output, args.index)
    elif args.mode == "prepare":
        prepare(output)
    elif args.mode == "freeze":
        common.freeze(output)
    else:
        protocol = json.loads((output / "protocol.json").read_text())
        if args.mode in {"queue", "summarize"}:
            globals()[args.mode](protocol, output, args.cohort)
        elif args.mode == "ablate":
            flow.ablate(protocol, output, args.cohort, args.seed)
        elif args.mode == "evaluate":
            evaluate(protocol, output, args.cohort, args.arm, args.seed)
        else:
            getattr(flow, args.mode)(protocol, output, args.cohort, args.arm, args.seed)


if __name__ == "__main__":
    main()
