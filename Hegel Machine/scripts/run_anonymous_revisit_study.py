#!/usr/bin/env python3
"""Successor knowledge resumes partial anonymous-object induction."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import importlib.util
from random import Random
from statistics import mean
import subprocess
import threading
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import run_observation_action_study as common
import run_completed_challenge_study as flow
import run_object_revisit_study as object_flow

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/anonymous_revisit_study_20260920"
BASELINE = "artifacts/object_revisit_study_20260920/runtime/src"
ARMS = ("baseline_v0100", "knowledge_revisit", "no_successor_knowledge")
STAGES = ("partial_anonymous_task", "successor_object_discovery")
LIMITS = ((96, 30000), (128, 60000), (192, 90000))
TASK = {"objects": 3, "apparatus_seed": 9013, "name_seed": 42114,
        "wiring_seed": 1780, "variant": "exchange", "readout": "focus"}


def develop(output, index, successor=False):
    sys.path.insert(0, str(ROOT / BASELINE))
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    from hegel_machine.cognition.memory import KnowledgeLibrary
    name = "successor" if successor else "cold"
    path = output / "development" / f"{name}_{index}.json"
    if path.exists():
        raise ValueError("preserve the previous development attempt")
    if successor:
        previous = json.loads((output / "development" / f"cold_{index}.json").read_text())
        library = KnowledgeLibrary.from_dict(previous["knowledge"])
        obs, compute, measurement, seed = 512, 480000, 6144, 10078
    else:
        library = KnowledgeLibrary()
        obs, compute = LIMITS[index]
        measurement, seed = 1152, 71
    task = {**TASK, "seed": seed}
    config = ResearchConfig(observation_budget=obs, compute_budget=compute, seed=seed,
        minimum_improvement=1e-7, max_rounds=6, max_history=4, active_probes=True)
    cpu, wall = time.process_time(), time.perf_counter()
    with AnonymousObjectLab(**task) as laboratory:
        report = UnifiedResearchAgent(config, measurement_budget=measurement, library=library).run(laboratory)
    payload = {"runtime": BASELINE, "task": task, "config": asdict(config),
        "measurement_budget": measurement, "research": report, "knowledge": library.to_dict(),
        "actual_costs": laboratory.costs, "acquisitions": laboratory.acquisitions,
        "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}
    from hegel_machine.cognition.language import Program
    program_paths = []
    def locate(value, location):
        if isinstance(value, Program):
            program_paths.append(location)
        elif isinstance(value, dict):
            for key, item in value.items():
                locate(item, location + "." + str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                locate(item, location + "[" + str(index) + "]")
    locate(payload, "payload")
    payload["program_objects_exported"] = program_paths
    common.save(path, json.loads(json.dumps(payload, default=lambda value: value.to_dict())))
    model = report.get("object_research") or {}
    print(json.dumps({"status": report["status"], "kind": report["selected_kind"],
        "costs": report["costs"], "object_status": model.get("status"),
        "object_complete": model.get("complete"), "groups": model.get("groups"),
        "unresolved": model.get("unresolved"), "concepts_added": report["concepts_added"]}))


def resume_development(output):
    sys.path.insert(0, str(ROOT / "src"))
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    source = output / "development" / "queued_current_offer.json"
    path = output / "development" / "resumed_current_offer.json"
    if path.exists():
        raise ValueError("preserve the previous continuation")
    session = UnifiedResearchSession(state=json.loads(source.read_text()))
    cpu, wall = time.process_time(), time.perf_counter()
    result = session.run(max_tasks=1)
    result["study_resume_cpu_seconds"] = time.process_time() - cpu
    result["study_resume_wall_seconds"] = time.perf_counter() - wall
    common.save(path, result)
    print(json.dumps({"status": result["results"][0]["research"]["status"],
        "costs": result["costs"], "visits": [{key: visit[key] for key in
            ("resource_limits", "visit_costs", "status")} for visit in result["revisits"]]}))


def ablate_development(output):
    sys.path.insert(0, str(ROOT / "src"))
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    a = json.loads((output / "development" / "cold_0.json").read_text())
    full = json.loads((output / "development" / "resumed_current_offer.json").read_text())
    visit = full["revisits"][0]
    path = output / "development" / "ablated_current_offer.json"
    if path.exists():
        raise ValueError("preserve the previous matched continuation")
    limits = visit["resource_limits"]
    library = KnowledgeLibrary.from_dict(a["knowledge"])
    cpu, wall = time.process_time(), time.perf_counter()
    with AnonymousObjectLab(**a["task"]) as laboratory:
        report = UnifiedResearchAgent(replace(ResearchConfig(**a["config"]),
            observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
            measurement_budget=limits["measurement_cost"], library=library).run(laboratory,
                previous=a["research"], continuation_candidate=None)
    result = {"research": report, "knowledge": library.to_dict(), "resource_limits": limits,
        "actual_new_costs": laboratory.costs, "acquisitions": laboratory.acquisitions,
        "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall,
        "source": "same allocated continuation resources, initial A knowledge only; no successor screening"}
    common.save(path, result)
    print(json.dumps({"status": report["status"], "visit_costs": report["visit_costs"]}))



def save(path, payload):
    # Frozen 0.100 leaves executable residual terms in report metadata. Export
    # their normal Program representation; the frozen researcher is unchanged.
    common.save(path, json.loads(json.dumps(payload, default=lambda value: value.to_dict())))


def prepare(output):
    if (output / "protocol.json").exists():
        return
    roots = [9013, *range(95902001, 95902009)]
    tasks, parameters = {}, {}
    for root in roots:
        if root == 9013:
            shared = {"objects": 3, "apparatus_seed": 9013, "name_seed": 42114,
                      "wiring_seed": 1780, "variant": "exchange", "profile": "development"}
            readout, first, second = "focus", 71, 10078
        else:
            count, readout = ((3, "focus"), (3, "event"), (4, "focus"), (4, "event"))[(root - 95902001) % 4]
            rng = Random(root + 17011)
            shared = {"objects": count, "apparatus_seed": root, "name_seed": root + 63857,
                "wiring_seed": root + 3709, "variant": "exchange", "profile": "confirmation",
                "actuator_curvature": rng.uniform(.08, .24), "relation_radius": rng.uniform(.70, 1.20)}
            first, second = root, root + 10007
        shared["environment"] = "anonymous_objects"
        tasks[str(root)] = [{**shared, "readout": readout, "seed": first,
            "research_limits": {"observation_budget": 96, "compute_budget": 30000, "measurement_budget": 1152}},
            {**shared, "readout": "focus", "seed": second}]
        parameters[str(root)] = shared
    save(output / "protocol.json", {
        "experiment": "successor_dependency_knowledge_resumes_anonymous_induction",
        "fixed_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {"development": roots[:1], "main": roots[1:]}, "tasks_by_seed": tasks,
        "apparatus_parameters": parameters,
        "arms": {"baseline_v0100": {"runtime": BASELINE, "version": "0.100.0"},
            "knowledge_revisit": {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.101.0"},
            "no_successor_knowledge": {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.101.0"}},
        "development_runtime": "src", "shared_laboratory_source": BASELINE + "/hegel_machine/cognition",
        "research_config": {"observation_budget": 512, "compute_budget": 480000,
            "minimum_improvement": 1e-7, "max_rounds": 6, "max_history": 4, "active_probes": True},
        "measurement_budget": 6144,
        "total_research_budget": {"observation_cost": 608, "measurement_cost": 7296, "compute_cost": 510000},
        "maximum_concurrent_processes": 4,
        "terminal": {"length": 48, "trajectories": 4, "seed_offset": 1000003,
            "risk_quantile": .5, "mse_tolerance": 1e-6, "random_backgrounds": 2,
            "background_seed_offset": 2000003, "sweep_fractions": [0., .25, .5, .75, 1.]},
        "development_selection": "First predeclared A budget96/30000 yielded three known quantity fields "
            "and unresolved controls/drive/position/membership. Larger128/60000 and192/90000 were not tried. "
            "B learned a complete model from actual interventions. An old raw Program metadata export "
            "failure was preserved and the same cold case rerun with to_dict serialization only.",
        "initial_contract": "All streams start with empty knowledge. Only anonymous controls, bounds and "
            "sensor banks are exposed. A uses96 observations/30000 compute/1152 meter; B uses512/480000/6144. "
            "A is focus or event, while B is focus on the same apparatus and wiring with a new trajectory seed. "
            "No object partition, hidden formula, source hypothesis or candidate score is supplied.",
        "schedule_contract": "Finish all first visits before scheduled continuations. The shared total "
            "pool is exactly the sum of initial allowances. Restore prior purchased transitions, partial "
            "role alternatives and recorded experiments from JSON. A later object model can propose "
            "additional research, never automatically certify A's roles or response. Every source visit, "
            "screen, failed hypothesis and new measurement is charged. Keep roots without an opportunity.",
        "knowledge_ablation_contract": "The deletion arm performs A and B but its external session adapter "
            "does not screen successors or autonomously queue visits. Once the knowledge arm finishes it "
            "receives only the actual task revisit positions and incremental resource allocations. Resume "
            "with A's initial library and candidate=None; no new source model, experiment recipe, candidate "
            "or prediction is passed. Do not charge nonexistent screening. Clip the matched allocation "
            "by its own remaining total pool, without enlarging a visit because it saved screen compute. "
            "This is a given-schedule deletion of knowledge, not a vanilla agent.",
        "evaluation_contract": "Only after all cohort learning ends, score initial and final programs "
            "on identical4x48 random trajectories plus per-anonymous-control low/quarter/mid/three-quarter/high "
            "steps at one center and two independently sampled nonzero public-bound backgrounds. Every "
            "step sequence includes reset and return history; no true relation radius or role label "
            "constructs probes. Anonymous banks are not projectable: purchase each needed legal bank "
            "separately, derive commands from actual applied controls, and count all terminal resources. "
            "Report deployed initial/final channel costs separately. Both random and sweep MSE must be <=1e-6.",
        "role_evaluation_contract": "After prediction scoring, use the existing private evaluation_schema "
            "only to count correctly inferred quantity/drive/position/membership and control roles. "
            "Missing roles remain unresolved; no evaluator result enters the learned library or scheduler.",
        "baseline_serialization": "Frozen0.100 cognition is unchanged. The runner converts raw Program "
            "terms inside saved finalist metadata through Program.to_dict so existing reports can persist.",
        "scope": "Eight predeclared independent apparatus/name/wiring roots;3/4 objects byfocus/event "
            "has two roots per cell. Object and relation operator families remain predefined. Forty-eight "
            "initial visits plus actual revisits; all roots retained and no tuning on main outcomes.",
        "formal_evidence": False})


def load(protocol, arm, cohort):
    source = ROOT / (protocol["development_runtime"] if cohort == "development" and arm != ARMS[0]
                     else protocol["arms"][arm]["runtime"])
    sys.path.insert(0, str(source))
    for name in ("structure_lab", "unified_object_lab", "anonymous_object_lab"):
        canonical = "hegel_machine.cognition." + name
        spec = importlib.util.spec_from_file_location(canonical,
            ROOT / protocol["shared_laboratory_source"] / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = module
        spec.loader.exec_module(module)
    module = importlib.import_module("hegel_machine.cognition.unified_session")
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("use a separate process for each runtime")
    def factory(item):
        return sys.modules["hegel_machine.cognition.anonymous_object_lab"].AnonymousObjectLab(
            **{key: value for key, value in item.items() if key not in ("environment", "research_limits")})
    session_type = module.UnifiedResearchSession
    if arm == ARMS[2]:
        class MatchedSession(session_type):
            def _offer_new_knowledge(self, *args, **kwargs):
                return None
        session_type = MatchedSession
    return module, session_type, factory


flow.ARMS, flow.STAGES, flow.load, flow.save = ARMS, STAGES, load, save
object_flow.ARMS, object_flow.STAGES = ARMS, STAGES


def import_development(protocol, output):
    p = output / "development"
    a, b = (json.loads((p / name).read_text()) for name in ("cold_0.json", "successor_0.json"))
    queued = json.loads((p / "queued_current_offer.json").read_text())
    full = json.loads((p / "resumed_current_offer.json").read_text())
    control = json.loads((p / "ablated_current_offer.json").read_text())
    for arm in ARMS:
        state = deepcopy(full if arm == ARMS[1] else queued)
        state["tasks"] = protocol["tasks_by_seed"]["9013"]
        state["evaluation"] = []
        state["pending_revisits"] = []
        if arm != ARMS[1]:
            state["scheduler_events"], state["revisits"], state["screening_compute_cost"] = [], [], 0
        if arm == ARMS[2]:
            state["results"][0]["research"] = control["research"]
            state["revisits"] = [{"task_position": 0, "candidate": None, "previous": a["research"],
                "resource_limits": control["resource_limits"], "visit_costs": control["research"]["visit_costs"],
                "status": control["research"]["status"], "confirmed_used_sources": control["research"].get("used_sources", []),
                "continuation": control["research"].get("continuation")}]
            state["study_revisit_knowledge"] = {"0": control["knowledge"]}
        for index, source in enumerate((a, b)):
            record = state["results"][index]
            record.update(learning_wall_seconds=source["wall_seconds"], initial_actual_costs=source["actual_costs"],
                          initial_acquisitions=source["acquisitions"])
        if arm == ARMS[1]:
            state["revisits"][0]["confirmed_used_sources"] = state["results"][0]["research"].get("used_sources", [])
            state["revisits"][0]["continuation"] = state["results"][0]["research"].get("continuation")
        state["study_initial_snapshots"] = {str(i): {"research": source["research"], "knowledge": source["knowledge"]}
                                            for i, source in enumerate((a, b))}
        cpu = a["cpu_seconds"] + b["cpu_seconds"]
        if arm == ARMS[1]:
            cpu += full.get("study_resume_cpu_seconds", 0.) + queued.get("study_development_screen_cpu_seconds", 0.)
        elif arm == ARMS[2]:
            cpu += control["cpu_seconds"]
        state["study_cpu"] = [{"phase": "imported_actual_development", "seconds": cpu}]
        state.update(study_initial_complete=True, study_learning_complete=True, status="completed")
        state["costs"] = {key: sum(row["research"]["costs"][key] for row in state["results"])
                          for key in ("observation_cost", "measurement_cost", "compute_cost")}
        state["costs"]["compute_cost"] += state["screening_compute_cost"]
        save(flow.checkpoint(output, "development", arm, 9013), state)


def role_score(report, truth):
    quantities = {fields["q"]: obj for obj, fields in truth.fields.items()}
    model = report.get("object_research") or {}
    groups = model.get("groups", [])
    correct, claimed, complete = 0, 0, 0
    for group in groups:
        actual = quantities.get(group.get("q"))
        role_fields = {key: group[key] for key in ("q", "u", "x", "f") if key in group}
        controls = group.get("controls", {})
        claimed += len(role_fields) + len(controls)
        if actual is None or sum(other.get("q") == group.get("q") for other in groups) != 1:
            continue
        matches = sum(value == truth.fields[actual].get(key) for key, value in role_fields.items())
        matches += sum(value == truth.controls[actual].get(key) for key, value in controls.items())
        correct += matches
        complete += matches == 7
    focus = model.get("focus_control") == truth.focus_control
    return {"declared_complete": bool(model.get("complete")), "correct_roles": correct,
            "claimed_roles": claimed, "complete_objects": complete, "focus_correct": focus,
            "exact": complete == len(truth.objects) == len(groups) and focus,
            "unresolved": model.get("unresolved", [])}


def evaluate(protocol, output, cohort, arm, seed, *, learning_arms=None):
    for name in learning_arms or ARMS:
        for root in protocol["cohorts"][cohort]:
            if not json.loads(flow.checkpoint(output, cohort, name, root).read_text())["study_learning_complete"]:
                raise ValueError("finish learning before evaluation")
    _, _, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.models import Observation, Probe
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    path = flow.checkpoint(output, cohort, arm, seed)
    state = json.loads(path.read_text())
    settings = protocol["terminal"]
    for position in range(len(state["evaluation"]), len(STAGES)):
        task = state["tasks"][position]
        report = state["results"][position]["research"]
        old = state["study_initial_snapshots"][str(position)]["research"]
        views = sorted({tuple(item["selected_channels"]) for item in (old, report)} - {()}) or [()]
        with factory(task) as laboratory:
            def score(probe):
                batches = {view: laboratory.acquire(probe, view) for view in views}
                sample = next(iter(batches.values())).observations
                def observed(channels):
                    if channels:
                        return batches[tuple(channels)].observations
                    history, rows = [], []
                    for point, row in zip(probe.sequence, sample):
                        rows.append(Observation(point, row.target, tuple(history)))
                        history.append({**point, "__target__": row.target})
                    return rows
                return {"probe_id": probe.probe_id, "rows": len(sample),
                    "mse": loss(Program.from_dict(report["program"]), observed(report["selected_channels"]), report["fallback"], .5),
                    "initial_mse": loss(Program.from_dict(old["program"]), observed(old["selected_channels"]), old["fallback"], .5)}
            random = [score(laboratory.random_probe(task["seed"] + settings["seed_offset"] + index * 10007,
                length=settings["length"], name=f"terminal_random_{index}")) for index in range(settings["trajectories"])]
            bounds = laboratory.spec.feature_bounds
            backgrounds = [{name: (low + high) / 2 for name, (low, high) in bounds.items()}]
            for index in range(settings["random_backgrounds"]):
                rng = Random(task["seed"] + settings["background_seed_offset"] + index * 10007)
                backgrounds.append({name: rng.uniform(low, high) for name, (low, high) in bounds.items()})
            sweeps = []
            for index, point in enumerate(backgrounds):
                for control, (low, high) in sorted(bounds.items()):
                    points = [dict(point) for _ in range(3)] + [{**point, control: low + (high - low) * fraction}
                        for fraction in settings["sweep_fractions"]] + [dict(point), dict(point)]
                    sweeps.append(score(Probe(f"anonymous_control_step:{index}:{control}", tuple(points))))
            costs = {**laboratory.costs, "compute_cost": 2 * sum(item["rows"] for item in [*random, *sweeps])}
            rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
            truth = laboratory._evaluation_schema()
            initial_roles, final_roles = role_score(old, truth), role_score(report, truth)
        random_mse, sweep_mse = mean(row["mse"] for row in random), mean(row["mse"] for row in sweeps)
        initial_random, initial_sweep = mean(row["initial_mse"] for row in random), mean(row["initial_mse"] for row in sweeps)
        unique_rows = sum(item["rows"] for item in [*random, *sweeps])
        state["evaluation"].append({"task_position": position, "mse": max(random_mse, sweep_mse),
            "initial_mse": max(initial_random, initial_sweep), "random_mse": random_mse,
            "boundary_mse": sweep_mse, "public_control_step_mse": sweep_mse,
            "initial_random_mse": initial_random, "initial_boundary_mse": initial_sweep,
            "random": random, "public_control_steps": sweeps, "costs": costs,
            "selected_channels": report["selected_channels"], "initial_channels": old["selected_channels"],
            "actual_views": [list(view) for view in views], "initial_object_evaluation": initial_roles,
            "object_evaluation": final_roles,
            "final_deployment_measurement_cost": unique_rows * sum(rates[name] for name in report["selected_channels"]),
            "initial_deployment_measurement_cost": unique_rows * sum(rates[name] for name in old["selected_channels"])})
        save(path, state)


def summarize(protocol, output, cohort):
    object_flow.summarize(protocol, output, cohort)
    path = output / cohort / "summary.json"
    result = json.loads(path.read_text())
    for arm, data in result["arms"].items():
        data["initial_exact_object_models"] = sum(row["evaluation"]["initial_object_evaluation"]["exact"] for row in data["tasks"])
        data["final_exact_object_models"] = sum(row["evaluation"]["object_evaluation"]["exact"] for row in data["tasks"])
        data["old_partial_then_exact_models"] = sum(not row["evaluation"]["initial_object_evaluation"]["exact"]
            and row["evaluation"]["object_evaluation"]["exact"] for row in data["tasks"])
        data["revisited_object_model_source_use"] = []
        data["research_errors"] = []
        for seed in protocol["cohorts"][cohort]:
            state = json.loads(flow.checkpoint(output, cohort, arm, seed).read_text())
            data["research_errors"].extend({"root": seed, **error}
                for error in state.get("study_research_errors", []))
            for visit in state["revisits"]:
                report = state["results"][visit["task_position"]]["research"]
                tail = report["actions"][len(visit["previous"]["actions"]):]
                model = report.get("object_research") or {}
                data["revisited_object_model_source_use"].append({"root": seed, "position": visit["task_position"],
                    "continuation": report.get("continuation"),
                    "used_source_object_model_ids": model.get("used_source_object_model_ids", []),
                    "new_dependency_experiments": sum(action["action"] == "test_object_dependency_question"
                        and action.get("resumed", False) for action in tail),
                    "new_dependency_rows": sum(action["observation_rows"] for action in tail
                        if action["action"] == "test_object_dependency_question" and action.get("resumed", False)),
                    "source_dependency_experiments": sum(action["action"] == "test_object_dependency_question"
                        and bool(action.get("question", {}).get("source_object_model_id")) for action in tail),
                    "source_dependency_rows": sum(action["observation_rows"] for action in tail
                        if action["action"] == "test_object_dependency_question"
                        and action.get("question", {}).get("source_object_model_id")),
                    "independent_role_confirmations": sum(action["action"] == "confirm_resumed_object_roles" for action in tail),
                    "object_events": [{key: value for key, value in action.items()
                        if key in ("action", "object_model_id", "source_object_model_ids", "complete", "rows", "reused_rows", "status", "reason")}
                        for action in tail if any(word in action["action"] for word in ("object", "induction", "anonymous"))]})
        receipts = data["revisited_object_model_source_use"]
        data["visits_using_source_object_experiments"] = sum(row["source_dependency_experiments"] > 0 for row in receipts)
        data["resumed_induction"] = {key: sum(row[key] for row in receipts) for key in
            ("new_dependency_experiments", "new_dependency_rows", "source_dependency_experiments",
             "source_dependency_rows", "independent_role_confirmations")}
        data["learning_costs_complete"] = not any(not row["costs_complete"] for row in data["research_errors"])
    save(path, result)
    digest = {**result, "arms": {arm: {key: value for key, value in data.items()
        if key not in ("tasks", "revisits", "scheduler_events", "restored_research", "revisited_object_model_source_use")}
        for arm, data in result["arms"].items()}}
    save(output / cohort / "final_digest.json", digest)


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
            code = subprocess.call([sys.executable, "-B", str(Path(__file__).resolve()), mode, "--output", str(output),
                "--cohort", cohort, "--arm", arm, "--seed", str(seed)], stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            execution["active"].remove(job)
            execution["failed" if code else "completed"].append({"job": job, "exit_code": code})
            save(output / cohort / "execution.json", execution)
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
            save(output / cohort / "execution.json", {**execution, "status": "failed", "exit_code": 1})
            raise SystemExit(1)
    summarize(protocol, output, cohort)
    save(output / cohort / "execution.json", {**execution, "status": "completed", "exit_code": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cold", "successor", "resume_development", "ablate_development",
        "prepare", "import_development", "freeze", "initial", "revisit", "ablate", "evaluate", "queue", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--index", type=int, choices=range(3), default=0)
    parser.add_argument("--cohort", choices=("development", "main"), default="development")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode in ("resume_development", "ablate_development"):
        globals()[args.mode](output)
    elif args.mode in ("cold", "successor"):
        develop(output, args.index, args.mode == "successor")
    elif args.mode == "prepare":
        prepare(output)
    elif args.mode == "freeze":
        common.freeze(output)
    else:
        protocol = json.loads((output / "protocol.json").read_text())
        if args.mode == "import_development":
            import_development(protocol, output)
        elif args.mode in ("queue", "summarize"):
            globals()[args.mode](protocol, output, args.cohort)
        elif args.mode == "ablate":
            flow.ablate(protocol, output, args.cohort, args.seed)
        elif args.mode == "evaluate":
            evaluate(protocol, output, args.cohort, args.arm, args.seed)
        else:
            getattr(flow, args.mode)(protocol, output, args.cohort, args.arm, args.seed)


if __name__ == "__main__":
    main()
